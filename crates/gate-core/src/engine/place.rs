//! Placement: when and where each waiting request runs, and what gives way.
//!
//! Requests are offered slots in service order. Each model's first request
//! in line is its *head*; the head is placed if any slot can take it now,
//! otherwise room is made for it (idle contexts evicted, or lower-priority
//! work that has produced nothing yet paused), and the rest of that model's
//! line may only backfill. Models are independent, since their backends are.
use super::{Command, Scheduler, worth_saving};
use crate::backend::Backend;
use crate::pool::{Claim, Layout, Preparation, Resident, Reuse, Running, SlotState, Waiting};
use crate::queue::Candidate;
use crate::{Class, Demand, Millis, ModelId, Rejection, SlotId, SnapshotId, SnapshotMeta, Tokens};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

pub(super) enum Plan {
    Place(Placement),
    MakeRoom(Vec<SlotId>),
    Preempt(Preemption),
    Wait,
    Refuse(Rejection),
}

pub(super) struct Placement {
    slot: SlotId,
    generation: Tokens,
    reserved: Tokens,
    clear: bool,
    save_first: bool,
    restore: Option<SnapshotId>,
    reuse: Reuse,
}

/// Making room by pausing requests.
pub(super) struct Preemption {
    victims: Vec<SlotId>,
    evict: Vec<SlotId>,
}

/// One way to place a request.
struct Choice {
    slot: SlotId,
    generation: Tokens,
    fits: bool,
    peak: Tokens,
    reserved: Tokens,
    clear: bool,
    save_first: bool,
    restore: Option<SnapshotId>,
    reuse: Reuse,
    /// The request needs the whole backend.
    exclusive: bool,
    /// Requests already on this backend.
    busy: usize,
    /// The backend's usage, in millionths of its capacity.
    load: u64,
    /// For a context to clear: the eviction policy's view (higher keeps it).
    victim: f64,
}

impl Choice {
    /// Orders choices from worst to best. In turn: fits now; reuses an
    /// appreciably larger part of the prompt; runs on a less busy backend;
    /// reuses more; destroys no context; destroys the policy's first
    /// victim; runs on a less full backend; the lowest slot.
    fn compare(left: &Choice, right: &Choice) -> Ordering {
        left.fits
            .cmp(&right.fits)
            .then((left.reuse.tokens >> 10).cmp(&(right.reuse.tokens >> 10)))
            .then(right.busy.cmp(&left.busy))
            .then(left.reuse.tokens.cmp(&right.reuse.tokens))
            .then((!left.clear).cmp(&!right.clear))
            .then(left.victim.total_cmp(&right.victim))
            .then(right.load.cmp(&left.load))
            .then(right.slot.cmp(&left.slot))
    }

    fn placement(&self) -> Placement {
        Placement {
            slot: self.slot,
            generation: self.generation,
            reserved: self.reserved,
            clear: self.clear,
            save_first: self.save_first,
            restore: self.restore,
            reuse: self.reuse,
        }
    }
}

/// A slot being considered, with what its backend offers.
#[derive(Clone, Copy)]
struct Site<'a> {
    slot: SlotId,
    backend: &'a Backend,
    state: &'a SlotState,
    /// The backend's best snapshot for the request, and what it covers.
    snapshot: Option<(&'a SnapshotMeta, Tokens)>,
}

enum Room {
    /// Operations already in flight will make enough room.
    Coming,
    /// Evicting these idle contexts makes enough room.
    Evict(Vec<SlotId>),
    /// Running work holds the room; only its completion helps.
    Blocked,
}

impl Scheduler {
    pub(super) fn schedule(&mut self, now: Millis, out: &mut Vec<Command>) {
        if self.closing {
            return;
        }
        self.fresh.clear();
        'pass: loop {
            let order = self.service_order(now);
            // Each model's head, and the models whose room is being made.
            let mut heads: BTreeMap<ModelId, usize> = BTreeMap::new();
            let mut held: BTreeSet<ModelId> = BTreeSet::new();
            for &index in &order {
                let model = self.waiting[index].demand.model;
                let Some(&head) = heads.get(&model) else {
                    heads.insert(model, index);
                    match self.plan(index, true) {
                        Plan::Place(placement) => {
                            self.commit(index, placement, now, out);
                            continue 'pass;
                        }
                        Plan::Refuse(rejection) => {
                            let waiting = self.take_waiting(index);
                            out.push(Command::Reject {
                                ticket: waiting.ticket,
                                rejection,
                            });
                            continue 'pass;
                        }
                        Plan::MakeRoom(victims) => {
                            self.make_room(&victims, out);
                            held.insert(model);
                        }
                        Plan::Preempt(preemption) => {
                            self.make_room(&preemption.evict, out);
                            for victim in preemption.victims {
                                self.preempt(victim, out);
                            }
                            held.insert(model);
                        }
                        Plan::Wait => {}
                    }
                    continue;
                };
                // The head is waiting for running work to finish. Requests
                // small enough to use what is free right now may run first,
                // until the head has waited for the backfill window.
                let Some(backfill) = self.config.backfill else {
                    continue;
                };
                if held.contains(&model)
                    || self.model_freeing(model)
                    || now.saturating_sub(self.waiting[head].arrived) >= backfill.window
                    || self.waiting[index].demand.need() > backfill.max_tokens
                {
                    continue;
                }
                if let Plan::Place(placement) = self.plan(index, false) {
                    self.commit(index, placement, now, out);
                    continue 'pass;
                }
            }
            return;
        }
    }

    /// Waiting requests in the order they are offered a slot: priority, then
    /// requests that were paused (they were admitted once already), then the
    /// queue policy's key, then arrival. Requests still being stopped are
    /// left out.
    pub(super) fn service_order(&self, now: Millis) -> Vec<usize> {
        let mut keyed: Vec<(i32, bool, f64, u64, usize)> = self
            .waiting
            .iter()
            .enumerate()
            .filter(|(_, waiting)| !waiting.paused)
            .map(|(index, waiting)| {
                let class = &self.config.classes[usize::from(waiting.demand.class.0)];
                let key = self.order.key(Candidate {
                    client: waiting.demand.client,
                    class,
                    need: waiting.demand.need(),
                    waited: now.saturating_sub(waiting.arrived),
                });
                (
                    -class.priority,
                    waiting.bumped == 0,
                    key,
                    waiting.sequence,
                    index,
                )
            })
            .collect();
        keyed.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then(left.1.cmp(&right.1))
                .then(left.2.total_cmp(&right.2))
                .then(left.3.cmp(&right.3))
        });
        keyed.into_iter().map(|(.., index)| index).collect()
    }

    /// Whether room is being made on any of the model's backends.
    fn model_freeing(&self, model: ModelId) -> bool {
        self.backends
            .iter()
            .any(|backend| backend.serves(model) && backend.pool.any_freeing())
    }

    fn plan(&self, index: usize, allow_eviction: bool) -> Plan {
        let demand = &self.waiting[index].demand;
        let floor = demand.prompt.saturating_add(demand.min_generation);
        // The model's largest backend decides whether the prompt can ever
        // fit. Until one is announced, the request waits.
        let Some(limit) = self.limit(demand.model) else {
            return Plan::Wait;
        };
        if floor > limit {
            return Plan::Refuse(Rejection::TooLarge {
                prompt: demand.prompt,
                limit,
            });
        }
        let mut options = Vec::new();
        for (backend_index, backend) in self.backends.iter().enumerate() {
            let limit = backend.pool.layout.request_limit();
            if !backend.serves(demand.model) || floor > limit {
                continue;
            }
            let generation = demand.generation.min(limit - demand.prompt);
            // The best snapshot does not depend on the slot.
            let partial = self.config.eviction.partial && backend.partial;
            let snapshot = backend
                .swap
                .then(|| backend.snapshots.best_for(&demand.prefix, partial))
                .flatten();
            for (slot_index, state) in backend.pool.slots.iter().enumerate() {
                if state.is_available() {
                    let slot = SlotId::new(backend_index as u16, slot_index);
                    let place = Site {
                        slot,
                        backend,
                        state,
                        snapshot,
                    };
                    options.push(self.option(place, demand, generation, limit));
                }
            }
        }
        options.sort_by(Choice::compare);
        if let Some(best) = options.last().filter(|option| option.fits) {
            return Plan::Place(best.placement());
        }
        if !allow_eviction {
            return Plan::Wait;
        }
        for option in options.iter().rev() {
            match self.room_for(option, option.exclusive) {
                Room::Coming => return Plan::Wait,
                Room::Evict(victims) => return Plan::MakeRoom(victims),
                Room::Blocked => {}
            }
        }
        match self.preemption(index) {
            Some(preemption) => Plan::Preempt(preemption),
            None => Plan::Wait,
        }
    }

    fn option(&self, site: Site<'_>, demand: &Demand, generation: Tokens, limit: Tokens) -> Choice {
        let Site {
            slot,
            backend,
            state,
            snapshot,
        } = site;
        let policy = &self.config.eviction;
        // The learned margin pads a request only up to what the pool holds: a
        // request that needs all of it runs alone either way.
        let need = (demand.prompt + generation + backend.overrun)
            .min(limit.max(demand.prompt + generation));
        let partial = policy.partial && backend.partial;
        // Where the server picks the slot itself, only an otherwise empty
        // unified pool can hold the request wherever it lands.
        let exclusive = demand.exclusive
            || (!demand.pinned && matches!(backend.pool.layout, Layout::Unified { .. }));
        let resident = match state {
            SlotState::Idle(resident) => Some(resident),
            _ => None,
        };
        // Run in place, the server keeps the longest common prefix of the
        // slot's context and the prompt.
        let in_place = resident.filter(|_| demand.pinned).and_then(|resident| {
            if resident.prefix.is_prefix_of(&demand.prefix) {
                return Some(Reuse {
                    tokens: resident.prefix.tokens(),
                    partial: false,
                });
            }
            let shared = resident.prefix.shared(&demand.prefix);
            (partial && shared >= policy.min_restore_gain.max(1)).then_some(Reuse {
                tokens: shared,
                partial: true,
            })
        });
        let kept_reuse = in_place.map_or(0, |reuse| reuse.tokens);
        let restore = snapshot.filter(|(_, covered)| {
            demand.pinned && *covered >= kept_reuse.saturating_add(policy.min_restore_gain)
        });
        let clear = resident.is_some() && (in_place.is_none() || restore.is_some());
        // Reusing part of a context in place truncates the rest; save it
        // first when the rest is worth keeping.
        let save_first = !clear
            && in_place.is_some_and(|reuse| reuse.partial)
            && resident.is_some_and(|resident| {
                worth_saving(policy, backend.swap, resident)
                    && resident.tokens.saturating_sub(kept_reuse) >= policy.min_snapshot_tokens
            });
        let holding = state.usage();
        let kept = if clear { 0 } else { holding };
        // A snapshot holds its conversation's prompt and generated answer, so
        // restoring it can briefly hold more than this request needs (for
        // example when the conversation was rewound).
        let restored = restore.map_or(0, |(meta, _)| meta.tokens);
        let reserved = if exclusive {
            limit
        } else {
            need.max(kept).max(restored)
        };
        let peak = reserved.max(holding);
        let fits = backend.pool.fits(slot.index, peak)
            && (!exclusive || backend.pool.others_usage_is_zero(slot.index));
        let reuse = match restore {
            Some((meta, covered)) => Reuse {
                tokens: covered,
                partial: !meta.prefix.is_prefix_of(&demand.prefix),
            },
            None if clear => Reuse::default(),
            None => in_place.unwrap_or_default(),
        };
        // Prefer keeping contexts; among contexts to clear, the policy's
        // first victim.
        let victim = resident
            .filter(|_| clear)
            .map_or(f64::MAX, |resident| -self.victims.rank(slot, resident));
        Choice {
            slot,
            generation,
            fits,
            peak,
            reserved,
            clear,
            save_first,
            restore: restore.map(|(meta, _)| meta.id),
            reuse,
            exclusive,
            busy: backend.pool.busy(),
            load: backend.load(),
            victim,
        }
    }

    /// Whether evicting idle contexts in other slots of the option's backend
    /// would let it fit.
    fn room_for(&self, option: &Choice, exclusive: bool) -> Room {
        let Some(backend) = self.backend(option.slot.backend) else {
            return Room::Blocked;
        };
        let mut idle: Vec<(SlotId, &Resident)> = Vec::new();
        let mut busy = false;
        let mut freeing = 0;
        for (index, state) in backend.pool.slots.iter().enumerate() {
            if index == option.slot.index {
                continue;
            }
            match state {
                SlotState::Idle(resident) => idle.push((
                    SlotId {
                        backend: option.slot.backend,
                        index,
                    },
                    resident,
                )),
                SlotState::Evicting { resident, .. } => freeing += resident.tokens,
                SlotState::Preempting { running, .. } => freeing += running.reserved,
                SlotState::Empty => {}
                _ => busy = true,
            }
        }
        // Stray cells go once their slots are erased.
        if !backend.pool.strays.is_empty() {
            freeing += backend.pool.stray;
        }
        if exclusive {
            return if busy {
                Room::Blocked
            } else if idle.is_empty() {
                if freeing > 0 {
                    Room::Coming
                } else {
                    Room::Blocked
                }
            } else {
                Room::Evict(idle.into_iter().map(|(slot, _)| slot).collect())
            };
        }
        let Layout::Unified { capacity } = backend.pool.layout else {
            return Room::Blocked;
        };
        let deficit =
            (option.peak + backend.pool.others(option.slot.index)).saturating_sub(capacity);
        if deficit <= freeing {
            return if freeing > 0 {
                Room::Coming
            } else {
                Room::Blocked
            };
        }
        let mut still = deficit - freeing;
        idle.sort_by(|left, right| {
            self.victims
                .rank(left.0, left.1)
                .total_cmp(&self.victims.rank(right.0, right.1))
        });
        let mut victims = Vec::new();
        for (slot, resident) in idle {
            if still == 0 {
                break;
            }
            victims.push(slot);
            still = still.saturating_sub(resident.tokens);
        }
        if still == 0 {
            Room::Evict(victims)
        } else {
            Room::Blocked
        }
    }

    /// The class of a running request that more urgent work may pause: its
    /// class allows it, it has produced no output, it has not been paused
    /// `max_preemptions` times, and it runs in the one slot the gate chose (an
    /// unpinned request's cells may be in any slot, and several answers at once
    /// fill several, so stopping it could not free a known one).
    pub(super) fn pausable(&self, running: &Running) -> Option<&Class> {
        let class = self
            .config
            .classes
            .get(usize::from(running.waiting.demand.class.0))?;
        (class.preemptible
            && running.waiting.demand.pinned
            && !running.waiting.demand.exclusive
            && !running.started
            && running.waiting.bumped < self.config.max_preemptions)
            .then_some(class)
    }

    /// Requests to pause (and idle contexts to evict) so the request at
    /// `index` can run on one of its backends, if pausing lower-priority
    /// work that has produced no output can make the room. Fewest paused
    /// requests first, then least paused work.
    fn preemption(&self, index: usize) -> Option<Preemption> {
        let demand = &self.waiting[index].demand;
        if demand.exclusive || !demand.pinned {
            return None;
        }
        let priority = self
            .config
            .classes
            .get(usize::from(demand.class.0))?
            .priority;
        let floor = demand.prompt.saturating_add(demand.min_generation);
        let mut best: Option<((usize, Tokens), Preemption)> = None;
        for (backend_index, backend) in self.backends.iter().enumerate() {
            let limit = backend.pool.layout.request_limit();
            if !backend.serves(demand.model) || floor > limit || backend.pool.any_freeing() {
                continue;
            }
            let need =
                (demand.prompt + demand.generation.min(limit - demand.prompt) + backend.overrun)
                    .min(limit);
            let slot_id = |index| SlotId::new(backend_index as u16, index);
            // Lowest priority first, then the most recently started.
            let mut victims: Vec<(i32, Millis, usize, Tokens)> = backend
                .pool
                .slots
                .iter()
                .enumerate()
                .filter_map(|(index, state)| {
                    let SlotState::Running(running) = state else {
                        return None;
                    };
                    let class = self.pausable(running)?;
                    (class.priority < priority && !self.fresh.contains(&running.waiting.ticket))
                        .then_some((class.priority, running.since, index, running.reserved))
                })
                .collect();
            if victims.is_empty() {
                continue;
            }
            victims.sort_by(|left, right| left.0.cmp(&right.0).then(right.1.cmp(&left.1)));
            let mut idle: Vec<(usize, &Resident)> = backend
                .pool
                .slots
                .iter()
                .enumerate()
                .filter_map(|(index, state)| match state {
                    SlotState::Idle(resident) => Some((index, resident)),
                    _ => None,
                })
                .collect();
            idle.sort_by(|left, right| {
                self.victims
                    .rank(slot_id(left.0), left.1)
                    .total_cmp(&self.victims.rank(slot_id(right.0), right.1))
            });
            // Where the request would run: an empty slot, else the idle slot
            // the policy gives up first, else the first victim's.
            let target = backend
                .pool
                .slots
                .iter()
                .position(|state| matches!(state, SlotState::Empty))
                .or_else(|| idle.first().map(|(index, _)| *index))
                .unwrap_or(victims[0].2);
            let fits = |freed: Tokens| match backend.pool.layout {
                Layout::Unified { capacity } => {
                    backend.usage().saturating_sub(freed).saturating_add(need) <= capacity
                }
                Layout::Split { .. } => true,
            };
            let mut freed = backend.pool.slots[target].usage();
            let mut chosen: Vec<(usize, Tokens)> = Vec::new();
            if let Some(position) = victims.iter().position(|victim| victim.2 == target) {
                let victim = victims.remove(position);
                chosen.push((victim.2, victim.3));
            }
            let mut evict = Vec::new();
            for (index, resident) in &idle {
                if fits(freed) {
                    break;
                }
                if *index != target {
                    freed += resident.tokens;
                    evict.push(slot_id(*index));
                }
            }
            for victim in victims {
                if fits(freed) {
                    break;
                }
                freed += victim.3;
                chosen.push((victim.2, victim.3));
            }
            if chosen.is_empty() || !fits(freed) {
                continue;
            }
            let cost = (
                chosen.len(),
                chosen.iter().map(|(_, reserved)| reserved).sum::<Tokens>(),
            );
            if best.as_ref().is_none_or(|(least, _)| cost < *least) {
                best = Some((
                    cost,
                    Preemption {
                        victims: chosen.iter().map(|(index, _)| slot_id(*index)).collect(),
                        evict,
                    },
                ));
            }
        }
        best.map(|(_, preemption)| preemption)
    }

    fn commit(&mut self, index: usize, placement: Placement, now: Millis, out: &mut Vec<Command>) {
        let waiting = self.take_waiting(index);
        let Placement {
            slot,
            generation,
            reserved,
            clear,
            save_first,
            restore,
            reuse,
        } = placement;
        let policy = self.config.eviction;
        let Some(backend) = self.backend_mut(slot.backend) else {
            return;
        };
        let swap = backend.swap;
        let state = std::mem::replace(&mut backend.pool.slots[slot.index], SlotState::Empty);
        let holding = state.usage();
        let preparation = match state {
            SlotState::Idle(previous) if clear => {
                let save =
                    worth_saving(&policy, swap, &previous).then(|| backend.snapshots.allocate());
                Some(Preparation::Clearing {
                    previous,
                    save,
                    restore,
                })
            }
            SlotState::Idle(previous) if save_first => Some(Preparation::Saving {
                save: backend.snapshots.allocate(),
                previous,
            }),
            _ => restore.map(|snapshot| Preparation::Restoring { snapshot }),
        };
        let Some(preparation) = preparation else {
            self.start(
                slot,
                waiting,
                generation,
                reserved.max(holding),
                reuse,
                now,
                out,
            );
            return;
        };
        let op = self.next_op();
        out.push(match &preparation {
            Preparation::Clearing { save, .. } => Command::Evict {
                slot,
                save: *save,
                op,
            },
            Preparation::Restoring { snapshot } => Command::Restore {
                slot,
                snapshot: *snapshot,
                op,
            },
            Preparation::Saving { save, .. } => Command::Save {
                slot,
                snapshot: *save,
                op,
            },
        });
        self.set(
            slot,
            SlotState::Preparing(Box::new(Claim {
                waiting,
                generation,
                reserved,
                holding,
                preparation,
                op,
                abandoned: false,
                reuse,
            })),
        );
    }

    /// Runs a request: charges its prompt to the queue policy (once, however
    /// often it is paused) and dispatches it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn start(
        &mut self,
        slot: SlotId,
        mut waiting: Waiting,
        generation: Tokens,
        reserved: Tokens,
        reuse: Reuse,
        now: Millis,
        out: &mut Vec<Command>,
    ) {
        if !waiting.charged {
            waiting.charged = true;
            let demand = &waiting.demand;
            if let Some(class) = self.config.classes.get(usize::from(demand.class.0)) {
                self.order.served_prompt(demand.client, class, demand.work);
            }
        }
        let ticket = waiting.ticket;
        let exclusive = waiting.demand.exclusive;
        self.fresh.insert(ticket);
        self.set(
            slot,
            SlotState::Running(Box::new(Running {
                waiting,
                reserved,
                since: now,
                reuse,
                started: false,
            })),
        );
        out.push(Command::Dispatch {
            ticket,
            slot,
            generation,
            exclusive,
            reuse: reuse.tokens,
        });
    }

    fn make_room(&mut self, victims: &[SlotId], out: &mut Vec<Command>) {
        let policy = self.config.eviction;
        for &slot in victims {
            let op = self.next_op();
            let Some(backend) = self.backend_mut(slot.backend) else {
                continue;
            };
            let swap = backend.swap;
            let state = std::mem::replace(&mut backend.pool.slots[slot.index], SlotState::Empty);
            match state {
                SlotState::Idle(resident) => {
                    let save = worth_saving(&policy, swap, &resident)
                        .then(|| backend.snapshots.allocate());
                    backend.pool.slots[slot.index] = SlotState::Evicting { resident, save, op };
                    out.push(Command::Evict { slot, save, op });
                }
                other => backend.pool.slots[slot.index] = other,
            }
        }
    }

    /// Pauses the request running in `slot`: it goes back in line, held
    /// until the server has stopped it, and its progress is saved when that
    /// is possible.
    fn preempt(&mut self, slot: SlotId, out: &mut Vec<Command>) {
        let min_snapshot = self.config.eviction.min_snapshot_tokens;
        let op = self.next_op();
        let Some(backend) = self.backend_mut(slot.backend) else {
            return;
        };
        let swap = backend.swap;
        let state = std::mem::replace(&mut backend.pool.slots[slot.index], SlotState::Empty);
        let running = match state {
            SlotState::Running(running) => running,
            other => {
                backend.pool.slots[slot.index] = other;
                return;
            }
        };
        let demand = &running.waiting.demand;
        let save = (swap && !demand.prefix.is_opaque() && demand.prompt >= min_snapshot)
            .then(|| backend.snapshots.allocate());
        let ticket = running.waiting.ticket;
        let mut waiting = running.waiting.clone();
        waiting.bumped += 1;
        waiting.paused = true;
        backend.pool.slots[slot.index] = SlotState::Preempting { running, save, op };
        self.enqueue(waiting);
        out.push(Command::Preempt {
            ticket,
            slot,
            save,
            op,
        });
    }
}
