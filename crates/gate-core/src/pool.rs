//! KV-pool accounting: what every slot holds or has been promised.
//!
//! Each backend's slots draw on that backend's own pool. Each slot is in
//! exactly one [`SlotState`]. Its *usage* is the most KV cells it can occupy
//! in that state, and the engine admits work only while a pool's total usage
//! stays within its layout's capacity. Because a running request is held to
//! its reserved generation, physical use never exceeds usage.
use crate::{Demand, Millis, OpId, Prefix, SnapshotId, TicketId, Tokens};
use serde::{Deserialize, Serialize};

/// How a server's KV cache is divided between its slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Layout {
    /// One pool shared by every slot (`--kv-unified`): any slot may use all
    /// of it, and the sum over slots is bounded. Overfilling it makes
    /// llama.cpp fail every running request, so it must never happen.
    Unified { capacity: Tokens },
    /// Each slot owns `per_slot` cells (`n_ctx / n_parallel`) and cannot
    /// affect the others.
    Split { per_slot: Tokens },
}

impl Layout {
    /// The most a single request may occupy.
    pub fn request_limit(&self) -> Tokens {
        match *self {
            Layout::Unified { capacity } => capacity,
            Layout::Split { per_slot } => per_slot,
        }
    }
}

/// A context left in a slot after its request finished.
#[derive(Clone, Debug)]
pub struct Resident {
    pub tokens: Tokens,
    /// The prompt it was built from; generated tokens are not addressed.
    pub prefix: Prefix,
    pub last_used: Millis,
}

/// A waiting request.
#[derive(Clone, Debug)]
pub(crate) struct Waiting {
    pub ticket: TicketId,
    pub demand: Demand,
    pub arrived: Millis,
    /// Arrival order; breaks every tie.
    pub sequence: u64,
    /// Times it was paused for more urgent work.
    pub bumped: u32,
    /// Being stopped on the server; not to be placed until that completes.
    pub paused: bool,
    /// Its prompt has been charged to the queue policy already.
    pub charged: bool,
}

/// What the server is expected to reuse from its cache for a request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Reuse {
    pub tokens: Tokens,
    /// The expectation rests on a context that shares only part of the
    /// prompt, which some models cannot rewind to.
    pub partial: bool,
}

/// What a claimed slot must do before its request can run.
#[derive(Clone, Debug)]
pub(crate) enum Preparation {
    /// Save (optionally) and erase the previous context, then restore a
    /// snapshot if `restore` is set.
    Clearing {
        previous: Resident,
        save: Option<SnapshotId>,
        restore: Option<SnapshotId>,
    },
    /// Load a snapshot into the empty slot.
    Restoring { snapshot: SnapshotId },
    /// Save the context the request will partly reuse in place, so the part
    /// it does not share survives on disk.
    Saving {
        previous: Resident,
        save: SnapshotId,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct Claim {
    pub waiting: Waiting,
    /// Generation granted (the demand's, clamped to the backend's limit).
    pub generation: Tokens,
    pub reserved: Tokens,
    /// Cells physically occupied right now.
    pub holding: Tokens,
    pub preparation: Preparation,
    /// The operation in flight for this preparation.
    pub op: OpId,
    /// The client left; finish the preparation, then keep the slot idle.
    pub abandoned: bool,
    pub reuse: Reuse,
}

#[derive(Clone, Debug)]
pub(crate) struct Running {
    /// Kept so the request can be put back in line.
    pub waiting: Waiting,
    pub reserved: Tokens,
    pub since: Millis,
    pub reuse: Reuse,
    /// Its output has begun reaching the client, so it can no longer be
    /// paused.
    pub started: bool,
}

#[derive(Clone, Debug)]
pub(crate) enum SlotState {
    Empty,
    Idle(Resident),
    /// Making room: saving (optionally) and erasing an idle context.
    Evicting {
        resident: Resident,
        save: Option<SnapshotId>,
        op: OpId,
    },
    /// Claimed by a request and being prepared for it.
    Preparing(Box<Claim>),
    Running(Box<Running>),
    /// Stopping a request that has produced no output, to make room for a
    /// more urgent one; its progress is saved to `save` when set.
    Preempting {
        running: Box<Running>,
        save: Option<SnapshotId>,
        op: OpId,
    },
    /// An operation failed and the content is unknown. Erased (again) from
    /// `retry_at` on, while its server is online.
    Faulted {
        holding: Tokens,
        retry_at: Millis,
    },
}

impl SlotState {
    /// The most cells this slot can occupy in its current state.
    pub fn usage(&self) -> Tokens {
        match self {
            SlotState::Empty => 0,
            SlotState::Idle(resident) => resident.tokens,
            SlotState::Evicting { resident, .. } => resident.tokens,
            SlotState::Preparing(claim) => claim.reserved.max(claim.holding),
            SlotState::Running(running) => running.reserved,
            SlotState::Preempting { running, .. } => running.reserved,
            SlotState::Faulted { holding, .. } => *holding,
        }
    }

    /// Whether a new request may be placed here.
    pub fn is_available(&self) -> bool {
        matches!(self, SlotState::Empty | SlotState::Idle(_))
    }

    /// Whether a request occupies the slot (being prepared, running, or
    /// being stopped).
    pub fn is_busy(&self) -> bool {
        matches!(
            self,
            SlotState::Preparing(_) | SlotState::Running(_) | SlotState::Preempting { .. }
        )
    }

    /// Whether the slot is being emptied, so its room is coming.
    pub fn is_freeing(&self) -> bool {
        matches!(
            self,
            SlotState::Evicting { .. } | SlotState::Preempting { .. }
        )
    }

    pub fn name(&self) -> &'static str {
        match self {
            SlotState::Empty => "empty",
            SlotState::Idle(_) => "idle",
            SlotState::Evicting { .. } => "evicting",
            SlotState::Preparing(_) => "preparing",
            SlotState::Running(_) => "running",
            SlotState::Preempting { .. } => "preempting",
            SlotState::Faulted { .. } => "faulted",
        }
    }

    /// The request occupying the slot, if any.
    pub fn ticket(&self) -> Option<TicketId> {
        match self {
            SlotState::Preparing(claim) => Some(claim.waiting.ticket),
            SlotState::Running(running) | SlotState::Preempting { running, .. } => {
                Some(running.waiting.ticket)
            }
            _ => None,
        }
    }
}

/// One backend's slots over its pool.
#[derive(Clone, Debug)]
pub(crate) struct Pool {
    pub layout: Layout,
    pub slots: Vec<SlotState>,
    /// Cells an unpinned request left in a slot the server chose, unknown
    /// to the engine: counted once, until every slot in `strays` (the slots
    /// that may hold them) has been erased.
    pub stray: Tokens,
    pub strays: std::collections::BTreeSet<usize>,
}

impl Pool {
    pub fn new(layout: Layout, slots: usize) -> Self {
        Self {
            layout,
            slots: (0..slots).map(|_| SlotState::Empty).collect(),
            stray: 0,
            strays: Default::default(),
        }
    }

    pub fn usage(&self) -> Tokens {
        self.slots.iter().map(SlotState::usage).sum::<Tokens>() + self.stray
    }

    /// Usage of everything except slot `index`.
    pub fn others(&self, index: usize) -> Tokens {
        self.slots
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .map(|(_, state)| state.usage())
            .sum::<Tokens>()
            + self.stray
    }

    /// Slot `index` was erased: it can no longer hold stray cells.
    pub fn erased(&mut self, index: usize) {
        if self.strays.remove(&index) && self.strays.is_empty() {
            self.stray = 0;
        }
    }

    /// Whether slot `index` may hold `usage` cells given every other slot.
    pub fn fits(&self, index: usize, usage: Tokens) -> bool {
        match self.layout {
            Layout::Unified { capacity } => usage.saturating_add(self.others(index)) <= capacity,
            Layout::Split { per_slot } => usage <= per_slot,
        }
    }

    /// Whether nothing but slot `index` holds anything: no other slot's
    /// ledger, and no slot still to be erased of content the gate did not
    /// place (a sweep can owe nothing yet still be erasing).
    pub fn others_usage_is_zero(&self, index: usize) -> bool {
        self.stray == 0
            && self.strays.is_empty()
            && self
                .slots
                .iter()
                .enumerate()
                .all(|(other, state)| other == index || state.usage() == 0)
    }

    /// The slot running or preparing `ticket` (not one stopping it).
    pub fn find_ticket(&self, ticket: TicketId) -> Option<usize> {
        self.slots.iter().position(|state| match state {
            SlotState::Running(running) => running.waiting.ticket == ticket,
            SlotState::Preparing(claim) => claim.waiting.ticket == ticket,
            _ => false,
        })
    }

    pub fn any_freeing(&self) -> bool {
        self.slots.iter().any(SlotState::is_freeing)
    }

    /// Requests occupying slots: a measure of how loaded the server is.
    pub fn busy(&self) -> usize {
        self.slots.iter().filter(|state| state.is_busy()).count()
    }

    /// The pool's capacity, as the sum of what its slots may hold.
    pub fn capacity(&self) -> Tokens {
        match self.layout {
            Layout::Unified { capacity } => capacity,
            Layout::Split { per_slot } => per_slot.saturating_mul(self.slots.len() as Tokens),
        }
    }
}
