//! A test harness that drives the scheduler like the server runtime does,
//! while modelling what each llama.cpp server physically holds.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

use gate_core::{
    BackendId, Backfill, Class, ClassId, ClientId, Command, Demand, Event, EvictOutcome,
    EvictionPolicy, Fifo, Layout, LeastRecentlyUsed, Millis, ModelId, Notice, Observation,
    PreemptOutcome, Prefix, QueueOrder, RestoreOutcome, SaveOutcome, Scheduler, SchedulerConfig,
    Shape, SlotId, SnapshotId, TicketId, Tokens, Vtc,
};
use std::collections::{BTreeMap, VecDeque};

pub const BLOCK: usize = 64;

pub fn tokens(conversation: u32, length: usize) -> Vec<i32> {
    (0..length as i32)
        .map(|index| index.wrapping_mul(2_654_435) ^ conversation as i32)
        .collect()
}

pub fn prefix(conversation: u32, length: usize) -> Prefix {
    Prefix::from_tokens(&tokens(conversation, length), BLOCK)
}

/// `head` tokens of conversation `shared`, then `tail` of conversation `own`.
pub fn forked(shared: u32, head: usize, own: u32, tail: usize) -> Prefix {
    let mut sequence = tokens(shared, head);
    sequence.extend(tokens(own, tail));
    Prefix::from_tokens(&sequence, BLOCK)
}

pub fn class(name: &str, priority: i32) -> Class {
    let mut class = Class::new(name);
    class.priority = priority;
    class
}

pub fn slot(backend: u16, index: usize) -> SlotId {
    SlotId::new(backend, index)
}

#[derive(Clone, Copy, Debug)]
pub struct BackendSetup {
    pub model: u16,
    pub layout: Layout,
    pub slots: usize,
    pub swap: bool,
}

pub struct Setup {
    pub backends: Vec<BackendSetup>,
    pub classes: Vec<Class>,
    pub backfill: Option<Backfill>,
    pub eviction: EvictionPolicy,
    pub vtc: bool,
    pub max_preemptions: u32,
    pub unavailable_after: Option<Millis>,
    /// What each backend's slots already hold when the gate starts.
    pub holding: Vec<Vec<Tokens>>,
}

impl Setup {
    /// One backend with a unified pool, swapping enabled.
    pub fn unified(capacity: Tokens, slots: usize) -> Self {
        Self {
            backends: vec![BackendSetup {
                model: 0,
                layout: Layout::Unified { capacity },
                slots,
                swap: true,
            }],
            classes: vec![class("standard", 0)],
            backfill: None,
            eviction: EvictionPolicy {
                min_snapshot_tokens: 256,
                min_restore_gain: 256,
                ..EvictionPolicy::default()
            },
            vtc: true,
            max_preemptions: 3,
            unavailable_after: None,
            holding: Vec::new(),
        }
    }

    /// The first backend's slots hold contexts from before the gate started.
    pub fn holding(mut self, slots: &[Tokens]) -> Self {
        self.holding = vec![slots.to_vec()];
        self
    }

    /// Adds another backend like the first.
    pub fn replicate(mut self) -> Self {
        let first = self.backends[0];
        self.backends.push(first);
        self
    }

    pub fn swap(mut self, swap: bool) -> Self {
        for backend in &mut self.backends {
            backend.swap = swap;
        }
        self
    }

    pub fn build(self) -> Harness {
        let order: Box<dyn QueueOrder> = if self.vtc {
            Box::new(Vtc::default())
        } else {
            Box::new(Fifo)
        };
        let scheduler = Scheduler::new(
            SchedulerConfig {
                classes: self.classes,
                backfill: self.backfill,
                eviction: self.eviction,
                max_preemptions: self.max_preemptions,
                unavailable_after: self.unavailable_after,
            },
            order,
            Box::new(LeastRecentlyUsed),
        );
        let shapes: Vec<Shape> = self
            .backends
            .iter()
            .map(|backend| Shape {
                model: ModelId(backend.model),
                layout: backend.layout,
                slots: backend.slots,
                swap: backend.swap,
            })
            .collect();
        let mut harness = Harness {
            scheduler,
            now: 0,
            next_ticket: 0,
            physical: shapes
                .iter()
                .enumerate()
                .map(|(index, shape)| {
                    let mut held = vec![0; shape.slots];
                    for (slot, tokens) in self.holding.get(index).into_iter().flatten().enumerate()
                    {
                        held[slot] = *tokens;
                    }
                    held
                })
                .collect(),
            files: shapes.iter().map(|_| BTreeMap::new()).collect(),
            dispatched: Vec::new(),
            rejected: Vec::new(),
            pending: VecDeque::new(),
            auto: true,
            running: BTreeMap::new(),
            demands: BTreeMap::new(),
            shapes: shapes.clone(),
            preempted: Vec::new(),
            bounced: Vec::new(),
            notices: Vec::new(),
            bounce_next: false,
            declined: BTreeMap::new(),
            unpinned: Default::default(),
            relaxed: Default::default(),
            cached: None,
            trace: std::env::var_os("SIM_TRACE").is_some(),
        };
        for (index, shape) in shapes.into_iter().enumerate() {
            let now = harness.now;
            harness.feed(Event::BackendUp {
                backend: BackendId(index as u16),
                shape,
                restarted: false,
                snapshots: None,
                now,
            });
        }
        harness
    }
}

/// A dispatch the harness observed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dispatched {
    pub ticket: TicketId,
    pub slot: SlotId,
    pub generation: Tokens,
    pub reuse: Tokens,
}

#[derive(Clone, Copy, Debug)]
pub struct Run {
    pub slot: SlotId,
    pub prompt: Tokens,
    pub started: bool,
}

pub struct Harness {
    pub scheduler: Scheduler,
    pub now: Millis,
    next_ticket: u64,
    /// Cells each modelled server physically holds per slot.
    pub physical: Vec<Vec<Tokens>>,
    /// Saved snapshot files per backend: id -> tokens.
    pub files: Vec<BTreeMap<SnapshotId, Tokens>>,
    pub dispatched: Vec<Dispatched>,
    pub rejected: Vec<(TicketId, gate_core::Rejection)>,
    /// Commands not yet carried out (when `auto` is off).
    pub pending: VecDeque<Command>,
    /// Carry out operations immediately and successfully.
    pub auto: bool,
    /// Requests running on a server.
    pub running: BTreeMap<TicketId, Run>,
    demands: BTreeMap<TicketId, (Tokens, ModelId)>,
    pub shapes: Vec<Shape>,
    /// Requests the scheduler paused (each time).
    pub preempted: Vec<TicketId>,
    /// Requests whose delivery bounced, and the slot that refused them.
    pub bounced: Vec<(TicketId, SlotId)>,
    pub notices: Vec<Notice>,
    /// The next dispatch carried out finds its server refusing connections.
    pub bounce_next: bool,
    /// Pauses issued and not yet reported: whether each was declined.
    declined: BTreeMap<TicketId, bool>,
    /// Requests the server places itself.
    unpinned: std::collections::BTreeSet<TicketId>,
    /// Backends where an unpinned request ran away from its granted slot:
    /// slots there are checked as a pool, not one by one.
    relaxed: std::collections::BTreeSet<BackendId>,
    /// What finishing requests report the server reused (None: unreported).
    pub cached: Option<Tokens>,
    /// Print every event and command (set `SIM_TRACE`).
    trace: bool,
}

impl Harness {
    pub fn advance(&mut self, millis: Millis) {
        self.now += millis;
        self.feed(Event::Tick { now: self.now });
    }

    pub fn arrive(
        &mut self,
        client: u32,
        prompt: Tokens,
        generation: Tokens,
        prefix: Prefix,
    ) -> TicketId {
        self.arrive_with(client, ClassId(0), prompt, generation, prefix, false)
    }

    pub fn arrive_with(
        &mut self,
        client: u32,
        class: ClassId,
        prompt: Tokens,
        generation: Tokens,
        prefix: Prefix,
        exclusive: bool,
    ) -> TicketId {
        self.arrive_for(
            ModelId(0),
            client,
            class,
            prompt,
            generation,
            prefix,
            exclusive,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn arrive_for(
        &mut self,
        model: ModelId,
        client: u32,
        class: ClassId,
        prompt: Tokens,
        generation: Tokens,
        prefix: Prefix,
        exclusive: bool,
    ) -> TicketId {
        self.arrive_demand(
            model, client, class, prompt, generation, prefix, exclusive, true,
        )
    }

    /// A request the server places in a slot of its own choosing (an
    /// embedding, say), which the harness models as the busiest-looking
    /// idle slot rather than the one granted.
    #[allow(clippy::too_many_arguments)]
    pub fn arrive_unpinned(
        &mut self,
        model: ModelId,
        client: u32,
        prompt: Tokens,
        prefix: Prefix,
    ) -> TicketId {
        self.arrive_demand(model, client, ClassId(0), prompt, 0, prefix, false, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn arrive_demand(
        &mut self,
        model: ModelId,
        client: u32,
        class: ClassId,
        prompt: Tokens,
        generation: Tokens,
        prefix: Prefix,
        exclusive: bool,
        pinned: bool,
    ) -> TicketId {
        self.next_ticket += 1;
        let ticket = TicketId(self.next_ticket);
        self.demands.insert(ticket, (prompt, model));
        if !pinned {
            self.unpinned.insert(ticket);
        }
        self.feed(Event::Arrived {
            ticket,
            demand: Demand {
                client: ClientId(client),
                class,
                model,
                prompt,
                work: prompt,
                generation,
                min_generation: u64::from(pinned),
                prefix,
                exclusive,
                pinned,
            },
            now: self.now,
        });
        ticket
    }

    /// The request finished after generating `generated` tokens.
    pub fn finish(&mut self, ticket: TicketId, generated: Tokens) {
        let run = self.running.remove(&ticket).expect("ticket is running");
        let resident = run.prompt + generated;
        self.physical[usize::from(run.slot.backend.0)][run.slot.index] = resident;
        let cached = self.cached;
        self.feed(Event::Finished {
            ticket,
            resident: Some(resident),
            generated: Some(generated),
            cached,
            now: self.now,
        });
    }

    /// The request's output began reaching its client.
    pub fn start(&mut self, ticket: TicketId) {
        if let Some(run) = self.running.get_mut(&ticket) {
            run.started = true;
            let now = self.now;
            self.feed(Event::Started { ticket, now });
        }
    }

    /// The request's output began reaching its client, but the scheduler
    /// has not heard yet: a pause issued now loses the race and is declined.
    pub fn start_unannounced(&mut self, ticket: TicketId) {
        if let Some(run) = self.running.get_mut(&ticket) {
            run.started = true;
        }
    }

    pub fn withdraw(&mut self, ticket: TicketId) {
        if let Some(run) = self.running.remove(&ticket) {
            self.physical[usize::from(run.slot.backend.0)][run.slot.index] = run.prompt;
        }
        // A dispatch is delivered to the request's own handler; once its
        // client has gone there is nobody left to forward it.
        self.pending
            .retain(|command| !matches!(command, Command::Dispatch { ticket: pending, .. } if *pending == ticket));
        self.feed(Event::Withdrawn {
            ticket,
            now: self.now,
        });
    }

    pub fn feed(&mut self, event: Event) {
        let trace = self.trace;
        if trace {
            eprintln!("  event {event:?}");
        }
        let commands = self.scheduler.handle(event);
        if trace {
            for command in &commands {
                eprintln!("    -> {command:?}");
            }
            eprintln!(
                "    physical {:?} status {:?}",
                self.physical,
                self.scheduler
                    .status(self.now)
                    .backends
                    .iter()
                    .map(|backend| backend
                        .slots
                        .iter()
                        .map(|slot| (slot.state, slot.tokens))
                        .collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            );
        }
        self.check();
        for command in commands {
            // A pause takes effect as it is issued (the runtime stops the
            // request at once); only the save and erase take time.
            if let Command::Preempt { ticket, .. } = command {
                let declined = self.running.get(&ticket).is_some_and(|run| run.started);
                if !declined {
                    self.running.remove(&ticket);
                    self.pending.retain(|pending| {
                        !matches!(pending, Command::Dispatch { ticket: sent, .. } if *sent == ticket)
                    });
                    self.preempted.push(ticket);
                }
                self.declined.insert(ticket, declined);
            }
            self.pending.push_back(command);
        }
        if self.auto {
            self.drain();
        }
    }

    /// Carries out every pending command successfully.
    pub fn drain(&mut self) {
        while let Some(command) = self.pending.pop_front() {
            self.carry_out(command, true);
        }
    }

    /// Carries out the next pending slot operation or dispatch, succeeding
    /// or failing. Bookkeeping commands on the way are carried out silently.
    pub fn step(&mut self, succeed: bool) -> Option<Command> {
        loop {
            let command = self.pending.pop_front()?;
            let bookkeeping = matches!(
                command,
                Command::IndexChanged { .. }
                    | Command::Forget { .. }
                    | Command::Reject { .. }
                    | Command::Notice(_)
            );
            self.carry_out(command.clone(), succeed || bookkeeping);
            if !bookkeeping {
                return Some(command);
            }
        }
    }

    fn held(&mut self, slot: SlotId) -> &mut Tokens {
        &mut self.physical[usize::from(slot.backend.0)][slot.index]
    }

    fn carry_out(&mut self, command: Command, succeed: bool) {
        let now = self.now;
        match command {
            Command::Dispatch {
                ticket,
                slot,
                generation,
                reuse,
                ..
            } => {
                if std::mem::take(&mut self.bounce_next) {
                    self.bounced.push((ticket, slot));
                    self.feed(Event::Bounced { ticket, now });
                    return;
                }
                let (prompt, model) = self.demands[&ticket];
                assert_eq!(
                    self.shapes[usize::from(slot.backend.0)].model,
                    model,
                    "ticket {ticket} placed on a backend of another model"
                );
                assert!(
                    !self.running.contains_key(&ticket),
                    "ticket {ticket} dispatched while it was running"
                );
                // The server places an unpinned request in whichever idle
                // slot it likes: model the one holding the most.
                let runs_in = if self.unpinned.contains(&ticket) {
                    let backend = usize::from(slot.backend.0);
                    // Idle at the server: nothing running and nothing about
                    // to (a real server would make that work wait instead,
                    // which this model does not represent).
                    let mut busy: Vec<SlotId> = self.running.values().map(|run| run.slot).collect();
                    busy.extend(self.pending.iter().filter_map(|command| match command {
                        Command::Dispatch { slot, .. }
                        | Command::Evict { slot, .. }
                        | Command::Save { slot, .. }
                        | Command::Restore { slot, .. }
                        | Command::Preempt { slot, .. } => Some(*slot),
                        _ => None,
                    }));
                    let chosen = (0..self.physical[backend].len())
                        .map(|index| SlotId {
                            backend: slot.backend,
                            index,
                        })
                        .filter(|candidate| !busy.contains(candidate))
                        .max_by_key(|candidate| self.physical[backend][candidate.index])
                        .unwrap_or(slot);
                    if chosen != slot {
                        self.relaxed.insert(slot.backend);
                    }
                    chosen
                } else {
                    slot
                };
                *self.held(runs_in) = prompt;
                self.running.insert(
                    ticket,
                    Run {
                        slot: runs_in,
                        prompt,
                        started: false,
                    },
                );
                self.dispatched.push(Dispatched {
                    ticket,
                    slot,
                    generation,
                    reuse,
                });
            }
            Command::Evict { slot, save, op } => {
                let outcome = if !succeed {
                    EvictOutcome::Failed
                } else if let Some(id) = save {
                    let tokens = std::mem::take(self.held(slot));
                    self.files[usize::from(slot.backend.0)].insert(id, tokens);
                    EvictOutcome::Saved {
                        bytes: 1_000,
                        tokens,
                    }
                } else {
                    *self.held(slot) = 0;
                    EvictOutcome::Erased
                };
                self.feed(Event::Evicted {
                    slot,
                    op,
                    outcome,
                    now,
                });
            }
            Command::Save { slot, snapshot, op } => {
                let outcome = if succeed {
                    let tokens = *self.held(slot);
                    self.files[usize::from(slot.backend.0)].insert(snapshot, tokens);
                    SaveOutcome::Saved {
                        bytes: 1_000,
                        tokens,
                    }
                } else {
                    SaveOutcome::Failed
                };
                self.feed(Event::Saved {
                    slot,
                    op,
                    outcome,
                    now,
                });
            }
            Command::Restore { slot, snapshot, op } => {
                let file = self.files[usize::from(slot.backend.0)]
                    .get(&snapshot)
                    .copied();
                let outcome = match file {
                    Some(tokens) if succeed => {
                        *self.held(slot) = tokens;
                        RestoreOutcome::Restored { tokens }
                    }
                    _ => RestoreOutcome::Failed,
                };
                self.feed(Event::Restored {
                    slot,
                    op,
                    outcome,
                    now,
                });
            }
            Command::Preempt {
                ticket,
                slot,
                save,
                op,
            } => {
                let declined = self.declined.remove(&ticket).unwrap_or(false);
                let outcome = match declined {
                    true => PreemptOutcome::Declined,
                    false => PreemptOutcome::Stopped(if !succeed {
                        EvictOutcome::Failed
                    } else if let Some(id) = save {
                        let tokens = std::mem::take(self.held(slot));
                        self.files[usize::from(slot.backend.0)].insert(id, tokens);
                        EvictOutcome::Saved {
                            bytes: 1_000,
                            tokens,
                        }
                    } else {
                        *self.held(slot) = 0;
                        EvictOutcome::Erased
                    }),
                };
                self.feed(Event::Preempted {
                    slot,
                    op,
                    outcome,
                    now,
                });
            }
            Command::Reject { ticket, rejection } => self.rejected.push((ticket, rejection)),
            Command::Forget { backend, snapshot } => {
                self.files[usize::from(backend.0)].remove(&snapshot);
            }
            Command::IndexChanged { .. } => {}
            Command::Notice(notice) => self.notices.push(notice),
        }
    }

    /// No server's pool is ever overfilled, and the scheduler's accounting
    /// never under-counts what a server physically holds.
    pub fn check(&self) {
        let status = self.scheduler.status(self.now);
        for backend in &status.backends {
            let physical = &self.physical[usize::from(backend.backend.0)];
            if self.relaxed.contains(&backend.backend) || backend.stray > 0 {
                // A request ran outside its granted slot: the pool as a whole
                // still covers what it holds, and a split pool's slots still
                // cannot outgrow their share.
                match backend.layout {
                    Layout::Unified { .. } => {
                        let held: Tokens = physical.iter().sum();
                        assert!(
                            held <= backend.usage,
                            "backend {} physically holds {held} but is accounted at {}",
                            backend.backend,
                            backend.usage
                        );
                    }
                    Layout::Split { per_slot } => {
                        for (index, held) in physical.iter().enumerate() {
                            assert!(
                                *held <= per_slot,
                                "slot {}/{index} holds {held} > {per_slot}",
                                backend.backend
                            );
                        }
                    }
                }
            } else {
                for slot in &backend.slots {
                    assert!(
                        physical[slot.slot.index] <= slot.tokens,
                        "slot {} physically holds {} but is accounted at {} ({})",
                        slot.slot,
                        physical[slot.slot.index],
                        slot.tokens,
                        slot.state
                    );
                }
            }
            if let Layout::Unified { capacity } = backend.layout {
                assert!(
                    backend.usage <= capacity,
                    "backend {} usage {} exceeds capacity {capacity}",
                    backend.backend,
                    backend.usage
                );
                let held: Tokens = physical.iter().sum();
                assert!(
                    held <= capacity,
                    "backend {} holds {held} > {capacity}",
                    backend.backend
                );
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        for backend in &status.backends {
            for slot in &backend.slots {
                if let Some(ticket) = slot.ticket {
                    assert!(seen.insert(ticket), "ticket {ticket} occupies two slots");
                }
            }
        }
    }

    /// The server reports its slots: whether each is busy, and how many
    /// tokens it says each holds.
    pub fn observe(&mut self, backend: u16, reports: &[(bool, Option<Tokens>)]) {
        let now = self.now;
        let slots = reports
            .iter()
            .enumerate()
            .map(|(index, &(busy, tokens))| Observation {
                slot: SlotId {
                    backend: BackendId(backend),
                    index,
                },
                busy,
                tokens,
            })
            .collect();
        self.feed(Event::Observed { slots, now });
    }

    /// A client that bypasses the gate leaves `tokens` cells in slot `index`,
    /// which the gate believes empty, and the server reports only `reported`
    /// of them (the prompt, not the answer). It takes only room the gate has
    /// not promised: more would overfill the pool whatever the gate did.
    pub fn intrude(&mut self, backend: u16, index: usize, tokens: Tokens, reported: Tokens) {
        let status = self.scheduler.status(self.now);
        let Some(view) = status
            .backends
            .iter()
            .find(|view| view.backend == BackendId(backend))
        else {
            return;
        };
        let room = match view.layout {
            Layout::Unified { capacity } => capacity.saturating_sub(view.usage),
            Layout::Split { per_slot } => per_slot,
        };
        let empty = view
            .slots
            .get(index)
            .is_some_and(|slot| slot.state == "empty");
        if !view.online || !empty || tokens > room {
            return;
        }
        self.physical[usize::from(backend)][index] = tokens;
        let now = self.now;
        self.feed(Event::Observed {
            slots: vec![Observation {
                slot: SlotId {
                    backend: BackendId(backend),
                    index,
                },
                busy: false,
                tokens: Some(reported),
            }],
            now,
        });
    }

    /// A backend stops answering; it is announced again by [`Harness::up`].
    pub fn down(&mut self, backend: u16) {
        let now = self.now;
        self.feed(Event::BackendDown {
            backend: BackendId(backend),
            now,
        });
    }

    /// A backend answers again, as the same server.
    pub fn up(&mut self, backend: u16) {
        let now = self.now;
        let shape = self.shapes[usize::from(backend)];
        self.feed(Event::BackendUp {
            backend: BackendId(backend),
            shape,
            restarted: false,
            snapshots: None,
            now,
        });
    }

    /// A backend's server restarts: it loses every context, requests
    /// running on it fail, and operations in flight on it report failure
    /// late. Dispatches not yet carried out will run on the new server.
    pub fn restart(&mut self, backend: u16) {
        let on_backend = |slot: &SlotId| slot.backend == BackendId(backend);
        let stale: Vec<Command> = self
            .pending
            .iter()
            .filter(|command| match command {
                Command::Evict { slot, .. }
                | Command::Save { slot, .. }
                | Command::Restore { slot, .. }
                | Command::Preempt { slot, .. } => on_backend(slot),
                _ => false,
            })
            .cloned()
            .collect();
        self.pending.retain(|command| !stale.contains(command));
        self.physical[usize::from(backend)]
            .iter_mut()
            .for_each(|held| *held = 0);
        let now = self.now;
        let shape = self.shapes[usize::from(backend)];
        self.feed(Event::BackendUp {
            backend: BackendId(backend),
            shape,
            restarted: true,
            snapshots: None,
            now,
        });
        let failed: Vec<TicketId> = self
            .running
            .iter()
            .filter(|(_, run)| on_backend(&run.slot))
            .map(|(ticket, _)| *ticket)
            .collect();
        for ticket in failed {
            self.running.remove(&ticket);
            self.feed(Event::Finished {
                ticket,
                resident: None,
                generated: None,
                cached: None,
                now,
            });
        }
        for command in stale {
            let event = match command {
                Command::Evict { slot, op, .. } => Event::Evicted {
                    slot,
                    op,
                    outcome: EvictOutcome::Failed,
                    now,
                },
                Command::Save { slot, op, .. } => Event::Saved {
                    slot,
                    op,
                    outcome: SaveOutcome::Failed,
                    now,
                },
                Command::Restore { slot, op, .. } => Event::Restored {
                    slot,
                    op,
                    outcome: RestoreOutcome::Failed,
                    now,
                },
                Command::Preempt {
                    ticket, slot, op, ..
                } => {
                    self.declined.remove(&ticket);
                    Event::Preempted {
                        slot,
                        op,
                        outcome: PreemptOutcome::Stopped(EvictOutcome::Failed),
                        now,
                    }
                }
                _ => continue,
            };
            self.feed(event);
        }
    }

    pub fn was_dispatched(&self, ticket: TicketId) -> Option<&Dispatched> {
        self.dispatched
            .iter()
            .rev()
            .find(|dispatch| dispatch.ticket == ticket)
    }

    pub fn dispatches(&self, ticket: TicketId) -> usize {
        self.dispatched
            .iter()
            .filter(|dispatch| dispatch.ticket == ticket)
            .count()
    }

    /// Slots of `backend` physically holding something.
    pub fn occupied(&self, backend: u16) -> usize {
        self.physical[usize::from(backend)]
            .iter()
            .filter(|held| **held > 0)
            .count()
    }
}
