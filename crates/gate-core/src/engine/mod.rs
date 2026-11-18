//! The scheduler: events in, commands out.
//!
//! Every decision is a pure function of the scheduler's state and the event
//! it is handling. The caller performs the returned commands (forward a
//! request, save, restore or erase a slot, stop a request, answer a
//! refusal) and reports their outcomes back as further events.
//!
//! This module holds the state and the event handlers; [`place`] decides
//! where and when requests run, and [`status`] describes the state.
mod place;
mod status;

pub use status::{BackendStatus, SlotStatus, Status, WaitingStatus};

use crate::backend::{Backend, Shape};
use crate::evict::EvictionOrder;
use crate::pool::{Pool, Preparation, Resident, Reuse, SlotState, Waiting};
use crate::queue::QueueOrder;
use crate::{
    BackendId, Class, ClientId, Demand, Millis, OpId, Prefix, Rejection, SlotId, SnapshotId,
    SnapshotIndex, TicketId, Tokens,
};
use std::collections::{BTreeMap, BTreeSet};

/// A server's partial reuse is given up after this many misses without a
/// single hit.
const PARTIAL_TRIALS: u32 = 3;

#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    /// Indexed by [`crate::ClassId`]. Must not be empty.
    pub classes: Vec<Class>,
    pub backfill: Option<Backfill>,
    pub eviction: EvictionPolicy,
    /// How often one request may be paused for more urgent work; after that
    /// it runs undisturbed.
    pub max_preemptions: u32,
    /// Refuse a request that has waited this long while no server of its
    /// model was online. `None` waits indefinitely.
    pub unavailable_after: Option<Millis>,
}

/// Lets small requests run ahead of a large one that is waiting for running
/// work to finish, without letting them delay it indefinitely (a simplified
/// form of HPC "EASY" backfilling).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Backfill {
    /// Only requests needing at most this many tokens may run ahead.
    pub max_tokens: Tokens,
    /// Backfilling stops once the blocked request has waited this long.
    pub window: Millis,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvictionPolicy {
    /// Reuse contexts that share only the start of a prompt (a common system
    /// prompt, a rewound conversation), not just exact continuations. Turned
    /// off per backend when its server turns out unable to rewind to them.
    pub partial: bool,
    /// Contexts smaller than this are erased rather than saved.
    pub min_snapshot_tokens: Tokens,
    /// Restore a snapshot only if it covers at least this many more tokens
    /// than the slot would reuse anyway; also the least sharing worth a
    /// partial reuse.
    pub min_restore_gain: Tokens,
    /// Bytes of snapshots each backend keeps; the least recently used are
    /// forgotten.
    pub snapshot_budget: u64,
    /// How long a faulted slot rests before its erase is retried.
    pub retry_after: Millis,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        Self {
            partial: true,
            min_snapshot_tokens: 4096,
            min_restore_gain: 2048,
            snapshot_budget: 32 << 30,
            retry_after: 5_000,
        }
    }
}

/// Something that happened, reported by the caller.
#[derive(Clone, Debug)]
pub enum Event {
    /// A request arrived and was measured.
    Arrived {
        ticket: TicketId,
        demand: Demand,
        now: Millis,
    },
    /// The client went away (before or after it was dispatched).
    Withdrawn { ticket: TicketId, now: Millis },
    /// A dispatched request's output began reaching its client, so it can no
    /// longer be paused.
    Started { ticket: TicketId, now: Millis },
    /// A dispatched request completed. `resident` is the tokens the slot now
    /// holds (prompt plus generated), `generated` the new tokens, and
    /// `cached` the prompt tokens the server reused, when reported.
    Finished {
        ticket: TicketId,
        resident: Option<Tokens>,
        generated: Option<Tokens>,
        cached: Option<Tokens>,
        now: Millis,
    },
    /// A dispatched request never reached its server (the connection was
    /// refused). The backend is taken offline and the request placed again.
    Bounced { ticket: TicketId, now: Millis },
    /// A [`Command::Evict`] completed.
    Evicted {
        slot: SlotId,
        op: OpId,
        outcome: EvictOutcome,
        now: Millis,
    },
    /// A [`Command::Save`] completed.
    Saved {
        slot: SlotId,
        op: OpId,
        outcome: SaveOutcome,
        now: Millis,
    },
    /// A [`Command::Restore`] completed.
    Restored {
        slot: SlotId,
        op: OpId,
        outcome: RestoreOutcome,
        now: Millis,
    },
    /// A [`Command::Preempt`] completed.
    Preempted {
        slot: SlotId,
        op: OpId,
        outcome: PreemptOutcome,
        now: Millis,
    },
    /// What servers report their slots to hold. Used to adopt contexts the
    /// engine does not know about, such as those present at startup.
    Observed {
        slots: Vec<Observation>,
        now: Millis,
    },
    /// A backend is ready. The first report announces it; later ones bring
    /// it back online. With `restarted`, or a different shape, every context
    /// it held is gone. `snapshots`, when given, replaces its index of saved
    /// contexts (for example after loading it from disk).
    BackendUp {
        backend: BackendId,
        shape: Shape,
        restarted: bool,
        snapshots: Option<SnapshotIndex>,
        now: Millis,
    },
    /// A backend stopped answering. Nothing is placed on it until it is up.
    BackendDown { backend: BackendId, now: Millis },
    /// Time passed. Drives queue timeouts and retries.
    Tick { now: Millis },
    /// Stop admitting work and refuse everything waiting.
    Shutdown { now: Millis },
}

impl Event {
    pub fn now(&self) -> Millis {
        match *self {
            Event::Arrived { now, .. }
            | Event::Withdrawn { now, .. }
            | Event::Started { now, .. }
            | Event::Finished { now, .. }
            | Event::Bounced { now, .. }
            | Event::Evicted { now, .. }
            | Event::Saved { now, .. }
            | Event::Restored { now, .. }
            | Event::Preempted { now, .. }
            | Event::Observed { now, .. }
            | Event::BackendUp { now, .. }
            | Event::BackendDown { now, .. }
            | Event::Tick { now }
            | Event::Shutdown { now } => now,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvictOutcome {
    /// Saved to the requested snapshot (`tokens` long, 0 if unreported),
    /// then erased.
    Saved { bytes: u64, tokens: Tokens },
    /// Erased without saving (no save requested, or the save failed).
    Erased,
    /// The slot could not be erased; its content is unknown.
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaveOutcome {
    Saved { bytes: u64, tokens: Tokens },
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestoreOutcome {
    Restored { tokens: Tokens },
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreemptOutcome {
    /// The request was stopped and the slot saved (if asked) and erased.
    Stopped(EvictOutcome),
    /// Its output had already begun reaching the client; it runs on.
    Declined,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Observation {
    pub slot: SlotId,
    pub busy: bool,
    /// What the server says the slot holds, if it says: llama-server
    /// reports the prompt a slot last read, but not the answer after it, so
    /// this is a lower bound.
    pub tokens: Option<Tokens>,
}

/// Something the caller must do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    /// Forward the request to `slot`'s backend pinned to that slot, limiting
    /// its generation to `generation` tokens. An `exclusive` request is not
    /// pinned: the server spreads it over every slot. `reuse` is how much of
    /// the prompt the server is expected to find in its cache.
    Dispatch {
        ticket: TicketId,
        slot: SlotId,
        generation: Tokens,
        exclusive: bool,
        reuse: Tokens,
    },
    /// Save the slot to `save` (if set), then erase it. Report
    /// [`Event::Evicted`] with the same `op`.
    Evict {
        slot: SlotId,
        save: Option<SnapshotId>,
        op: OpId,
    },
    /// Save the slot to `snapshot` and leave it as it is. Report
    /// [`Event::Saved`] with the same `op`.
    Save {
        slot: SlotId,
        snapshot: SnapshotId,
        op: OpId,
    },
    /// Load `snapshot` into the empty slot. Report [`Event::Restored`] with
    /// the same `op`.
    Restore {
        slot: SlotId,
        snapshot: SnapshotId,
        op: OpId,
    },
    /// Stop the request without answering its client, which keeps waiting:
    /// the request is back in line. Then save the slot to `save` (if set) and
    /// erase it. If the request's output has already begun reaching its
    /// client, leave it running instead. Report [`Event::Preempted`] with the
    /// same `op`.
    Preempt {
        ticket: TicketId,
        slot: SlotId,
        save: Option<SnapshotId>,
        op: OpId,
    },
    /// Answer the request with this refusal.
    Reject {
        ticket: TicketId,
        rejection: Rejection,
    },
    /// Delete the snapshot's file; it is no longer indexed.
    Forget {
        backend: BackendId,
        snapshot: SnapshotId,
    },
    /// The backend's snapshot index changed; persist it if snapshots should
    /// outlive the process.
    IndexChanged { backend: BackendId },
    /// Something worth telling an operator.
    Notice(Notice),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Notice {
    /// The backend's server did not reuse contexts that shared only part of
    /// a prompt (its model cannot rewind to them), so they are no longer
    /// offered there.
    PartialReuseIneffective { backend: BackendId },
}

pub struct Scheduler {
    config: SchedulerConfig,
    backends: Vec<Backend>,
    waiting: Vec<Waiting>,
    order: Box<dyn QueueOrder>,
    victims: Box<dyn EvictionOrder>,
    sequence: u64,
    ops: u64,
    closing: bool,
    /// Requests started during the current scheduling pass, which are never
    /// paused in the same pass.
    fresh: BTreeSet<TicketId>,
}

impl std::fmt::Debug for Scheduler {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Scheduler")
            .field("backends", &self.backends.len())
            .field("usage", &self.usage())
            .field("waiting", &self.waiting.len())
            .finish_non_exhaustive()
    }
}

impl Scheduler {
    /// A scheduler with no backends yet: announce each with
    /// [`Event::BackendUp`]. Requests arriving before then wait.
    pub fn new(
        config: SchedulerConfig,
        order: Box<dyn QueueOrder>,
        victims: Box<dyn EvictionOrder>,
    ) -> Self {
        assert!(!config.classes.is_empty(), "at least one class is required");
        Self {
            config,
            backends: Vec::new(),
            waiting: Vec::new(),
            order,
            victims,
            sequence: 0,
            ops: 0,
            closing: false,
            fresh: BTreeSet::new(),
        }
    }

    /// Applies one event and returns the work it makes possible.
    pub fn handle(&mut self, event: Event) -> Vec<Command> {
        let now = event.now();
        let mut out = Vec::new();
        match event {
            Event::Arrived { ticket, demand, .. } => self.arrive(ticket, demand, now, &mut out),
            Event::Withdrawn { ticket, .. } => self.withdraw(ticket, now, &mut out),
            Event::Started { ticket, .. } => self.started(ticket),
            Event::Finished {
                ticket,
                resident,
                generated,
                cached,
                ..
            } => self.finish(ticket, resident, generated, cached, now, &mut out),
            Event::Bounced { ticket, .. } => self.bounced(ticket, now),
            Event::Evicted {
                slot, op, outcome, ..
            } => self.evicted(slot, op, outcome, now, &mut out),
            Event::Saved {
                slot, op, outcome, ..
            } => self.saved(slot, op, outcome, now, &mut out),
            Event::Restored {
                slot, op, outcome, ..
            } => self.restored(slot, op, outcome, now, &mut out),
            Event::Preempted {
                slot, op, outcome, ..
            } => self.preempted(slot, op, outcome, now, &mut out),
            Event::Observed { slots, .. } => self.observe(&slots, now, &mut out),
            Event::BackendUp {
                backend,
                shape,
                restarted,
                snapshots,
                ..
            } => self.backend_up(backend, shape, restarted, snapshots, now, &mut out),
            Event::BackendDown { backend, .. } => {
                if let Some(backend) = self.backend_mut(backend) {
                    backend.online = false;
                }
            }
            Event::Tick { .. } => self.tick(now, &mut out),
            Event::Shutdown { .. } => self.shutdown(&mut out),
        }
        self.schedule(now, &mut out);
        out
    }

    /// Applies a new configuration. `class_of` maps each class id of the
    /// old configuration to its id in the new one (classes may have been
    /// added, removed, or reordered); requests keep their place in line.
    pub fn reconfigure(
        &mut self,
        config: SchedulerConfig,
        class_of: impl Fn(crate::ClassId) -> crate::ClassId,
    ) {
        assert!(!config.classes.is_empty(), "at least one class is required");
        for waiting in &mut self.waiting {
            waiting.demand.class = class_of(waiting.demand.class);
        }
        for backend in &mut self.backends {
            for state in &mut backend.pool.slots {
                match state {
                    SlotState::Preparing(claim) => {
                        claim.waiting.demand.class = class_of(claim.waiting.demand.class);
                    }
                    SlotState::Running(running) | SlotState::Preempting { running, .. } => {
                        running.waiting.demand.class = class_of(running.waiting.demand.class);
                    }
                    _ => {}
                }
            }
        }
        self.config = config;
    }

    /// The backend's shape, once it has been announced.
    pub fn shape(&self, backend: BackendId) -> Option<Shape> {
        self.backend(backend)
            .filter(|backend| backend.known)
            .map(Backend::shape)
    }

    pub fn is_online(&self, backend: BackendId) -> bool {
        self.backend(backend).is_some_and(|backend| backend.online)
    }

    /// Stops saving contexts on a backend whose server cannot save slots.
    /// Its contexts are discarded from now on; its existing snapshots can
    /// still be restored.
    pub fn disable_swap(&mut self, backend: BackendId) {
        if let Some(backend) = self.backend_mut(backend) {
            backend.swap = false;
        }
    }

    pub fn swap_enabled(&self, backend: BackendId) -> bool {
        self.backend(backend).is_some_and(|backend| backend.swap)
    }

    pub fn snapshots(&self, backend: BackendId) -> Option<&SnapshotIndex> {
        self.backend(backend).map(|backend| &backend.snapshots)
    }

    pub fn classes(&self) -> &[Class] {
        &self.config.classes
    }

    /// The service the queue policy counts for `client`, for status pages.
    pub fn service(&self, client: ClientId) -> Option<f64> {
        self.order.service(client)
    }

    /// Reserved and resident tokens over every backend. Never exceeds the
    /// sum of the unified capacities.
    pub fn usage(&self) -> Tokens {
        self.backends.iter().map(Backend::usage).sum()
    }

    pub fn waiting(&self) -> usize {
        self.waiting.len()
    }

    // ---------------------------------------------------------------- events

    fn arrive(&mut self, ticket: TicketId, demand: Demand, now: Millis, out: &mut Vec<Command>) {
        let rejection = if self.closing {
            Some(Rejection::ShuttingDown)
        } else if let Some(class) = self.config.classes.get(usize::from(demand.class.0)) {
            let queued = self
                .waiting
                .iter()
                .filter(|waiting| waiting.demand.client == demand.client)
                .count();
            // A prompt no server of its model could ever hold is refused at
            // once rather than when it reaches the head of the line.
            let limit = self.limit(demand.model);
            if let Some(limit) =
                limit.filter(|limit| demand.prompt.saturating_add(demand.min_generation) > *limit)
            {
                Some(Rejection::TooLarge {
                    prompt: demand.prompt,
                    limit,
                })
            } else {
                class
                    .max_queued_per_client
                    .filter(|max| queued >= *max)
                    .map(|_| Rejection::QueueFull { queued })
            }
        } else {
            Some(Rejection::UnknownClass)
        };
        if let Some(rejection) = rejection {
            out.push(Command::Reject { ticket, rejection });
            return;
        }
        self.sequence += 1;
        self.enqueue(Waiting {
            ticket,
            demand,
            arrived: now,
            sequence: self.sequence,
            bumped: 0,
            paused: false,
            charged: false,
        });
    }

    fn withdraw(&mut self, ticket: TicketId, now: Millis, out: &mut Vec<Command>) {
        if let Some(index) = self.position(ticket) {
            // A paused request may still be stopping; its slot completes on
            // its own.
            self.take_waiting(index);
            return;
        }
        let Some(slot) = self.find_ticket(ticket) else {
            return;
        };
        match self.slot_mut(slot) {
            Some(SlotState::Preparing(claim)) => claim.abandoned = true,
            Some(SlotState::Running(_)) => self.finish(ticket, None, None, None, now, out),
            _ => {}
        }
    }

    fn started(&mut self, ticket: TicketId) {
        if let Some(slot) = self.find_ticket(ticket)
            && let Some(SlotState::Running(running)) = self.slot_mut(slot)
        {
            running.started = true;
        }
    }

    fn finish(
        &mut self,
        ticket: TicketId,
        resident: Option<Tokens>,
        generated: Option<Tokens>,
        cached: Option<Tokens>,
        now: Millis,
        out: &mut Vec<Command>,
    ) {
        let Some(slot) = self.find_ticket(ticket) else {
            // Paused just as it completed: it is done, not waiting.
            if let Some(index) = self
                .waiting
                .iter()
                .position(|waiting| waiting.ticket == ticket && waiting.paused)
            {
                let waiting = self.take_waiting(index);
                self.charge_generation(&waiting.demand, generated);
            }
            return;
        };
        let Some(state) = self.slot_mut(slot) else {
            return;
        };
        let running = match std::mem::replace(state, SlotState::Empty) {
            SlotState::Running(running) => running,
            SlotState::Preparing(mut claim) => {
                claim.abandoned = true;
                *state = SlotState::Preparing(claim);
                return;
            }
            other => {
                *state = other;
                return;
            }
        };
        // Without a report the slot may hold anything up to its reservation.
        let tokens = resident.unwrap_or(running.reserved);
        let demand = &running.waiting.demand;
        if demand.pinned {
            *state = SlotState::Idle(Resident {
                tokens,
                prefix: demand.prefix.clone(),
                last_used: now,
            });
        } else {
            self.strayed(slot, tokens, now, out);
        }
        self.charge_generation(&running.waiting.demand, generated);
        self.learn(slot.backend, running.reuse, cached, out);
    }

    /// An unpinned request in `slot` finished; its context is in whichever
    /// slot the server chose. In a split pool that slot's share bounds it.
    /// A unified pool ran it alone, so any empty slot may hold it: it is
    /// counted once, and every such slot is erased, after which it is gone.
    fn strayed(&mut self, slot: SlotId, tokens: Tokens, now: Millis, out: &mut Vec<Command>) {
        let unified = self
            .backend(slot.backend)
            .is_some_and(|backend| matches!(backend.pool.layout, crate::Layout::Unified { .. }));
        if !unified {
            self.set(
                slot,
                SlotState::Idle(Resident {
                    tokens,
                    prefix: Prefix::opaque(tokens),
                    last_used: now,
                }),
            );
            return;
        }
        self.set(slot, SlotState::Empty);
        let empty: Vec<usize> = self
            .backend(slot.backend)
            .map(|backend| {
                backend
                    .pool
                    .slots
                    .iter()
                    .enumerate()
                    .filter(|(_, state)| matches!(state, SlotState::Empty))
                    .map(|(index, _)| index)
                    .collect()
            })
            .unwrap_or_default();
        self.sweep(slot.backend, empty, tokens, now, out);
    }

    /// Up to `debt` cells the gate did not place may be in any of `slots`
    /// of a unified pool: they are counted once, as part of the pool, and
    /// every one of those slots is erased, after which they are gone.
    fn sweep(
        &mut self,
        backend: BackendId,
        slots: Vec<usize>,
        debt: Tokens,
        now: Millis,
        out: &mut Vec<Command>,
    ) {
        let ops: Vec<OpId> = slots.iter().map(|_| self.next_op()).collect();
        let Some(state) = self.backend_mut(backend) else {
            return;
        };
        state.pool.stray = state.pool.stray.max(debt);
        for (index, op) in slots.into_iter().zip(ops) {
            state.pool.strays.insert(index);
            state.pool.slots[index] = SlotState::Evicting {
                resident: Resident {
                    tokens: 0,
                    prefix: Prefix::opaque(0),
                    last_used: now,
                },
                save: None,
                op,
            };
            out.push(Command::Evict {
                slot: SlotId { backend, index },
                save: None,
                op,
            });
        }
    }

    fn bounced(&mut self, ticket: TicketId, now: Millis) {
        let Some(slot) = self.find_ticket(ticket) else {
            return;
        };
        let Some(backend) = self.backend_mut(slot.backend) else {
            return;
        };
        let Some(state) = backend.pool.slots.get_mut(slot.index) else {
            return;
        };
        let running = match std::mem::replace(state, SlotState::Empty) {
            SlotState::Running(running) => running,
            other => {
                *state = other;
                return;
            }
        };
        // Nothing reached the server, so the slot holds what it held before,
        // which the reservation covers; what exactly is no longer known.
        *state = SlotState::Faulted {
            holding: running.reserved,
            since: now,
        };
        backend.online = false;
        self.enqueue(running.waiting);
    }

    fn evicted(
        &mut self,
        slot: SlotId,
        op: OpId,
        outcome: EvictOutcome,
        now: Millis,
        out: &mut Vec<Command>,
    ) {
        let Some(state) = self.slot_mut(slot) else {
            return;
        };
        let current = match state {
            SlotState::Evicting { op, .. } => Some(*op),
            SlotState::Preparing(claim) => Some(claim.op),
            _ => None,
        };
        if current != Some(op) {
            return;
        }
        if outcome != EvictOutcome::Failed
            && let Some(backend) = self.backend_mut(slot.backend)
        {
            backend.pool.erased(slot.index);
        }
        let Some(state) = self.slot_mut(slot) else {
            return;
        };
        match std::mem::replace(state, SlotState::Empty) {
            SlotState::Evicting { resident, save, .. } => match outcome {
                EvictOutcome::Failed => {
                    *state = SlotState::Faulted {
                        holding: resident.tokens,
                        since: now,
                    };
                }
                EvictOutcome::Saved { bytes, tokens } => {
                    if let Some(id) = save {
                        let tokens = known_or(tokens, resident.tokens);
                        self.remember(slot.backend, id, resident.prefix, tokens, bytes, out);
                    }
                }
                EvictOutcome::Erased => {}
            },
            SlotState::Preparing(mut claim) => {
                let Preparation::Clearing {
                    previous,
                    save,
                    restore,
                } = claim.preparation.clone()
                else {
                    *state = SlotState::Preparing(claim);
                    return;
                };
                match outcome {
                    EvictOutcome::Failed => {
                        *state = SlotState::Faulted {
                            holding: claim.holding,
                            since: now,
                        };
                        if !claim.abandoned {
                            self.enqueue(claim.waiting);
                        }
                    }
                    EvictOutcome::Saved { .. } | EvictOutcome::Erased => {
                        if let (EvictOutcome::Saved { bytes, tokens }, Some(id)) = (outcome, save) {
                            let tokens = known_or(tokens, previous.tokens);
                            self.remember(slot.backend, id, previous.prefix, tokens, bytes, out);
                        }
                        claim.holding = 0;
                        if claim.abandoned {
                            return;
                        }
                        let restore = restore.filter(|snapshot| {
                            self.snapshots(slot.backend)
                                .is_some_and(|index| index.get(*snapshot).is_some())
                        });
                        match restore {
                            Some(snapshot) => {
                                let op = self.next_op();
                                claim.preparation = Preparation::Restoring { snapshot };
                                claim.op = op;
                                self.set(slot, SlotState::Preparing(claim));
                                out.push(Command::Restore { slot, snapshot, op });
                            }
                            None => {
                                let reserved = claim.reserved.max(claim.holding);
                                self.start(
                                    slot,
                                    claim.waiting,
                                    claim.generation,
                                    reserved,
                                    Reuse::default(),
                                    now,
                                    out,
                                );
                            }
                        }
                    }
                }
            }
            other => *state = other,
        }
    }

    fn saved(
        &mut self,
        slot: SlotId,
        op: OpId,
        outcome: SaveOutcome,
        now: Millis,
        out: &mut Vec<Command>,
    ) {
        let Some(state) = self.slot_mut(slot) else {
            return;
        };
        let matches = matches!(
            state,
            SlotState::Preparing(claim)
                if claim.op == op && matches!(claim.preparation, Preparation::Saving { .. })
        );
        if !matches {
            return;
        }
        let SlotState::Preparing(claim) = std::mem::replace(state, SlotState::Empty) else {
            return;
        };
        let Preparation::Saving { previous, save } = claim.preparation.clone() else {
            return;
        };
        if let SaveOutcome::Saved { bytes, tokens } = outcome {
            let tokens = known_or(tokens, previous.tokens);
            self.remember(
                slot.backend,
                save,
                previous.prefix.clone(),
                tokens,
                bytes,
                out,
            );
        }
        if claim.abandoned {
            // The context stays where it was.
            self.set(slot, SlotState::Idle(previous));
            return;
        }
        let reserved = claim.reserved.max(claim.holding);
        self.start(
            slot,
            claim.waiting,
            claim.generation,
            reserved,
            claim.reuse,
            now,
            out,
        );
    }

    fn restored(
        &mut self,
        slot: SlotId,
        op: OpId,
        outcome: RestoreOutcome,
        now: Millis,
        out: &mut Vec<Command>,
    ) {
        let Some(state) = self.slot_mut(slot) else {
            return;
        };
        if !matches!(state, SlotState::Preparing(claim) if claim.op == op) {
            return;
        }
        let SlotState::Preparing(mut claim) = std::mem::replace(state, SlotState::Empty) else {
            return;
        };
        let Preparation::Restoring { snapshot } = claim.preparation else {
            *state = SlotState::Preparing(claim);
            return;
        };
        match outcome {
            RestoreOutcome::Restored { tokens } => {
                let (prefix, tokens) = match self.backend_mut(slot.backend) {
                    Some(backend) => {
                        backend.snapshots.touch(snapshot);
                        backend.snapshots.get(snapshot).map_or_else(
                            || (Prefix::opaque(tokens), tokens),
                            |meta| (meta.prefix.clone(), known_or(tokens, meta.tokens)),
                        )
                    }
                    None => (Prefix::opaque(tokens), tokens),
                };
                out.push(Command::IndexChanged {
                    backend: slot.backend,
                });
                claim.holding = tokens;
                if claim.abandoned {
                    self.set(
                        slot,
                        SlotState::Idle(Resident {
                            tokens,
                            prefix,
                            last_used: now,
                        }),
                    );
                } else {
                    let reserved = claim.reserved.max(claim.holding);
                    self.start(
                        slot,
                        claim.waiting,
                        claim.generation,
                        reserved,
                        claim.reuse,
                        now,
                        out,
                    );
                }
            }
            RestoreOutcome::Failed => {
                // The file is unusable and the slot's content unknown: forget
                // the snapshot and erase the slot. The request chooses again,
                // keeping its place: another snapshot may hold its context.
                let tokens = self
                    .snapshots(slot.backend)
                    .and_then(|index| index.get(snapshot))
                    .map_or(0, |meta| meta.tokens);
                self.forget(slot.backend, snapshot, out);
                let holding = claim.holding.max(tokens);
                let op = self.next_op();
                self.set(
                    slot,
                    SlotState::Evicting {
                        resident: Resident {
                            tokens: holding,
                            prefix: Prefix::opaque(holding),
                            last_used: now,
                        },
                        save: None,
                        op,
                    },
                );
                out.push(Command::Evict {
                    slot,
                    save: None,
                    op,
                });
                if !claim.abandoned {
                    self.enqueue(claim.waiting);
                }
            }
        }
    }

    fn preempted(
        &mut self,
        slot: SlotId,
        op: OpId,
        outcome: PreemptOutcome,
        now: Millis,
        out: &mut Vec<Command>,
    ) {
        let Some(state) = self.slot_mut(slot) else {
            return;
        };
        if !matches!(state, SlotState::Preempting { op: current, .. } if *current == op) {
            return;
        }
        let SlotState::Preempting { running, save, .. } =
            std::mem::replace(state, SlotState::Empty)
        else {
            return;
        };
        let ticket = running.waiting.ticket;
        match outcome {
            PreemptOutcome::Declined => match self.position(ticket) {
                Some(index) => {
                    self.take_waiting(index);
                    let mut running = running;
                    running.started = true;
                    self.set(slot, SlotState::Running(running));
                }
                None => {
                    // Its client left meanwhile, so the request is over.
                    self.set(
                        slot,
                        SlotState::Idle(Resident {
                            tokens: running.reserved,
                            prefix: running.waiting.demand.prefix.clone(),
                            last_used: now,
                        }),
                    );
                }
            },
            PreemptOutcome::Stopped(outcome) => {
                if let Some(waiting) = self
                    .waiting
                    .iter_mut()
                    .find(|waiting| waiting.ticket == ticket)
                {
                    waiting.paused = false;
                }
                match outcome {
                    EvictOutcome::Failed => self.set(
                        slot,
                        SlotState::Faulted {
                            holding: running.reserved,
                            since: now,
                        },
                    ),
                    EvictOutcome::Saved { bytes, tokens } => {
                        if let Some(id) = save {
                            // It may have been stopped partway through its
                            // prompt: address only what was processed.
                            let tokens = known_or(tokens, running.reserved);
                            let prefix = running.waiting.demand.prefix.truncate(tokens);
                            self.remember(slot.backend, id, prefix, tokens, bytes, out);
                        }
                    }
                    EvictOutcome::Erased => {}
                }
            }
        }
    }

    /// Reconciles the server's view of its slots with the ledger. A report
    /// never lowers what the ledger holds: it leaves out answers.
    fn observe(&mut self, observations: &[Observation], now: Millis, out: &mut Vec<Command>) {
        let mut foreign: BTreeMap<BackendId, Vec<usize>> = BTreeMap::new();
        for observation in observations {
            let Some(backend) = self.backend_mut(observation.slot.backend) else {
                continue;
            };
            let limit = backend.pool.layout.request_limit();
            let Some(state) = backend.pool.slots.get_mut(observation.slot.index) else {
                continue;
            };
            match state {
                SlotState::Empty if observation.busy => {
                    // Work the engine did not start: assume the worst until
                    // the slot can be erased.
                    *state = SlotState::Faulted {
                        holding: limit,
                        since: now,
                    };
                }
                // Content the gate did not place, of a size the report
                // understates by its answer: erased.
                SlotState::Empty if observation.tokens.is_some_and(|tokens| tokens > 0) => {
                    foreign
                        .entry(observation.slot.backend)
                        .or_default()
                        .push(observation.slot.index);
                }
                _ => {}
            }
        }
        for (backend, slots) in foreign {
            // Until those slots are erased, the rest of the pool may be
            // theirs.
            let debt = self.backend(backend).map_or(0, |state| {
                let held: Tokens = state.pool.slots.iter().map(SlotState::usage).sum();
                state.pool.capacity().saturating_sub(held)
            });
            self.sweep(backend, slots, debt, now, out);
        }
    }

    fn backend_up(
        &mut self,
        id: BackendId,
        shape: Shape,
        restarted: bool,
        snapshots: Option<SnapshotIndex>,
        now: Millis,
        out: &mut Vec<Command>,
    ) {
        let index = usize::from(id.0);
        if self.backends.len() <= index {
            self.backends.resize_with(index + 1, Backend::unknown);
        }
        let backend = &mut self.backends[index];
        let first = !backend.known;
        let changed = first || backend.shape() != shape;
        let mut requeue = Vec::new();
        let mut resumed = Vec::new();
        if changed || restarted {
            let previous = std::mem::replace(
                &mut backend.pool,
                Pool::new(shape.layout, shape.slots.max(1)),
            );
            backend.known = true;
            backend.model = shape.model;
            backend.swap = shape.swap;
            if changed {
                // Another server may rewind differently: learn afresh.
                backend.partial = true;
                backend.partial_hits = 0;
                backend.partial_misses = 0;
            }
            for (slot, state) in previous.slots.into_iter().enumerate() {
                match state {
                    // Its finish (or failure) will still be reported, and if
                    // it had not been sent yet it runs on the new server.
                    SlotState::Running(running) if slot < backend.pool.slots.len() => {
                        backend.pool.slots[slot] = SlotState::Running(running);
                    }
                    SlotState::Preparing(claim) if !claim.abandoned => requeue.push(claim.waiting),
                    // The server stopping it is gone: it waits in line already.
                    SlotState::Preempting { running, .. } => resumed.push(running.waiting.ticket),
                    _ => {}
                }
            }
        }
        if let Some(snapshots) = snapshots {
            backend.snapshots = snapshots;
        }
        backend.online = true;
        for waiting in requeue {
            self.enqueue(waiting);
        }
        for ticket in resumed {
            if let Some(waiting) = self
                .waiting
                .iter_mut()
                .find(|waiting| waiting.ticket == ticket)
            {
                waiting.paused = false;
            }
        }
        // A server met for the first time may hold contexts from before the
        // gate, of a size it cannot report (it gives a slot's last prompt,
        // not the answer after it): every slot is erased before use.
        if first && !restarted {
            let pool = &self.backends[index].pool;
            let (slots, debt) = ((0..pool.slots.len()).collect(), pool.capacity());
            self.sweep(id, slots, debt, now, out);
        }
    }

    fn tick(&mut self, now: Millis, out: &mut Vec<Command>) {
        let mut index = 0;
        while index < self.waiting.len() {
            let waiting = &self.waiting[index];
            let waited = now.saturating_sub(waiting.arrived);
            let model = waiting.demand.model;
            // A request paused for more urgent work was admitted already.
            let expired = waiting.bumped == 0
                && self
                    .config
                    .classes
                    .get(usize::from(waiting.demand.class.0))
                    .and_then(|class| class.max_wait)
                    .is_some_and(|max| waited > max);
            let stranded = !waiting.paused
                && self
                    .config
                    .unavailable_after
                    .is_some_and(|max| waited > max)
                && !self.backends.iter().any(|backend| backend.serves(model));
            let rejection = if expired {
                Some(Rejection::Timeout { waited })
            } else if stranded {
                Some(Rejection::Unavailable { waited })
            } else {
                None
            };
            match rejection {
                Some(rejection) => {
                    let waiting = self.take_waiting(index);
                    out.push(Command::Reject {
                        ticket: waiting.ticket,
                        rejection,
                    });
                }
                None => index += 1,
            }
        }
        let retry_after = self.config.eviction.retry_after;
        let mut retries = Vec::new();
        for (backend_index, backend) in self.backends.iter().enumerate() {
            if !backend.online {
                continue;
            }
            for (slot, state) in backend.pool.slots.iter().enumerate() {
                if let SlotState::Faulted { holding, since } = *state
                    && now.saturating_sub(since) >= retry_after
                {
                    retries.push((SlotId::new(backend_index as u16, slot), holding, since));
                }
            }
        }
        for (slot, holding, since) in retries {
            let op = self.next_op();
            self.set(
                slot,
                SlotState::Evicting {
                    resident: Resident {
                        tokens: holding,
                        prefix: Prefix::opaque(holding),
                        last_used: since,
                    },
                    save: None,
                    op,
                },
            );
            out.push(Command::Evict {
                slot,
                save: None,
                op,
            });
        }
    }

    fn shutdown(&mut self, out: &mut Vec<Command>) {
        self.closing = true;
        for waiting in self.waiting.drain(..) {
            out.push(Command::Reject {
                ticket: waiting.ticket,
                rejection: Rejection::ShuttingDown,
            });
        }
    }

    // --------------------------------------------------------------- helpers

    /// The most one request of `model` may hold: its largest backend's limit,
    /// once any is announced.
    fn limit(&self, model: crate::ModelId) -> Option<Tokens> {
        self.backends
            .iter()
            .filter(|backend| backend.known && backend.model == model)
            .map(|backend| backend.pool.layout.request_limit())
            .max()
    }

    fn backend(&self, id: BackendId) -> Option<&Backend> {
        self.backends.get(usize::from(id.0))
    }

    fn backend_mut(&mut self, id: BackendId) -> Option<&mut Backend> {
        self.backends.get_mut(usize::from(id.0))
    }

    fn slot_mut(&mut self, slot: SlotId) -> Option<&mut SlotState> {
        self.backend_mut(slot.backend)?
            .pool
            .slots
            .get_mut(slot.index)
    }

    fn set(&mut self, slot: SlotId, state: SlotState) {
        if let Some(current) = self.slot_mut(slot) {
            *current = state;
        }
    }

    fn find_ticket(&self, ticket: TicketId) -> Option<SlotId> {
        self.backends
            .iter()
            .enumerate()
            .find_map(|(backend, state)| {
                state
                    .pool
                    .find_ticket(ticket)
                    .map(|index| SlotId::new(backend as u16, index))
            })
    }

    fn position(&self, ticket: TicketId) -> Option<usize> {
        self.waiting
            .iter()
            .position(|waiting| waiting.ticket == ticket)
    }

    /// Puts a request in line, telling the queue policy when its client
    /// starts waiting.
    fn enqueue(&mut self, waiting: Waiting) {
        let client = waiting.demand.client;
        if !self
            .waiting
            .iter()
            .any(|other| other.demand.client == client)
        {
            let backlogged: Vec<ClientId> = self
                .waiting
                .iter()
                .map(|other| other.demand.client)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            self.order.activated(client, &backlogged);
        }
        self.waiting.push(waiting);
    }

    /// Removes a waiting request, telling the queue policy when its client
    /// has nothing left waiting.
    fn take_waiting(&mut self, index: usize) -> Waiting {
        let waiting = self.waiting.remove(index);
        let client = waiting.demand.client;
        if !self
            .waiting
            .iter()
            .any(|other| other.demand.client == client)
        {
            self.order.deactivated(client);
        }
        waiting
    }

    fn charge_generation(&mut self, demand: &Demand, generated: Option<Tokens>) {
        if let Some(class) = self.config.classes.get(usize::from(demand.class.0)) {
            self.order
                .served_generation(demand.client, class, generated.unwrap_or(0));
        }
    }

    /// Learns from what the server actually reused whether partial reuse
    /// works on this backend.
    fn learn(
        &mut self,
        id: BackendId,
        reuse: Reuse,
        cached: Option<Tokens>,
        out: &mut Vec<Command>,
    ) {
        let Some(cached) = cached else {
            return;
        };
        if !reuse.partial || reuse.tokens == 0 {
            return;
        }
        let Some(backend) = self.backend_mut(id) else {
            return;
        };
        if !backend.partial {
            return;
        }
        if cached.saturating_mul(2) >= reuse.tokens {
            backend.partial_hits += 1;
        } else {
            backend.partial_misses += 1;
        }
        if backend.partial_hits == 0 && backend.partial_misses >= PARTIAL_TRIALS {
            backend.partial = false;
            out.push(Command::Notice(Notice::PartialReuseIneffective {
                backend: id,
            }));
        }
    }

    fn next_op(&mut self) -> OpId {
        self.ops += 1;
        OpId(self.ops)
    }

    fn remember(
        &mut self,
        backend: BackendId,
        id: SnapshotId,
        prefix: Prefix,
        tokens: Tokens,
        bytes: u64,
        out: &mut Vec<Command>,
    ) {
        let budget = self.config.eviction.snapshot_budget;
        let pinned = self.pinned(backend);
        let Some(state) = self.backend_mut(backend) else {
            return;
        };
        state.snapshots.insert(id, prefix, tokens, bytes);
        let doomed: Vec<SnapshotId> = state
            .snapshots
            .over_budget(budget)
            .into_iter()
            .filter(|doomed| !pinned.contains(doomed))
            .collect();
        for snapshot in doomed {
            state.snapshots.remove(snapshot);
            out.push(Command::Forget { backend, snapshot });
        }
        out.push(Command::IndexChanged { backend });
    }

    fn forget(&mut self, backend: BackendId, snapshot: SnapshotId, out: &mut Vec<Command>) {
        let removed = self
            .backend_mut(backend)
            .and_then(|state| state.snapshots.remove(snapshot));
        if removed.is_some() {
            out.push(Command::Forget { backend, snapshot });
            out.push(Command::IndexChanged { backend });
        }
    }

    /// Snapshots an in-flight preparation on `backend` is about to restore.
    fn pinned(&self, backend: BackendId) -> BTreeSet<SnapshotId> {
        let Some(backend) = self.backend(backend) else {
            return BTreeSet::new();
        };
        backend
            .pool
            .slots
            .iter()
            .filter_map(|state| match state {
                SlotState::Preparing(claim) => match claim.preparation {
                    Preparation::Clearing { restore, .. } => restore,
                    Preparation::Restoring { snapshot } => Some(snapshot),
                    Preparation::Saving { .. } => None,
                },
                _ => None,
            })
            .collect()
    }
}

/// A reported token count, or `fallback` when the server reported none.
fn known_or(reported: Tokens, fallback: Tokens) -> Tokens {
    if reported > 0 { reported } else { fallback }
}

/// Whether an idle context is worth saving before it is cleared.
fn worth_saving(policy: &EvictionPolicy, swap: bool, resident: &Resident) -> bool {
    swap && !resident.prefix.is_opaque() && resident.tokens >= policy.min_snapshot_tokens
}
