//! A serializable view of the scheduler, for status pages and metrics.
use super::Scheduler;
use crate::pool::{Layout, SlotState};
use crate::{BackendId, ClientId, Millis, ModelId, SlotId, TicketId, Tokens};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct Status {
    pub backends: Vec<BackendStatus>,
    /// In service order; requests being paused come last.
    pub waiting: Vec<WaitingStatus>,
}

#[derive(Clone, Debug, Serialize)]
pub struct BackendStatus {
    pub backend: BackendId,
    pub online: bool,
    pub model: ModelId,
    pub layout: Layout,
    /// Cells held or promised.
    pub usage: Tokens,
    /// Of `usage`, cells an unpinned request left in a slot of the server's
    /// choosing, counted until the slots that may hold them are erased.
    pub stray: Tokens,
    /// Cells the pool holds in all.
    pub capacity: Tokens,
    pub swap: bool,
    /// Contexts sharing part of a prompt are reused here.
    pub partial_reuse: bool,
    pub slots: Vec<SlotStatus>,
    pub snapshots: usize,
    pub snapshot_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct SlotStatus {
    pub slot: SlotId,
    pub state: &'static str,
    pub tokens: Tokens,
    pub ticket: Option<TicketId>,
    pub client: Option<ClientId>,
    pub running_for: Option<Millis>,
    /// For a running request: whether its output has begun.
    pub started: Option<bool>,
    /// For a running request: whether more urgent work may pause it.
    pub pausable: Option<bool>,
    /// For a running request: prompt tokens expected from the cache.
    pub reuse: Option<Tokens>,
}

#[derive(Clone, Debug, Serialize)]
pub struct WaitingStatus {
    pub ticket: TicketId,
    pub client: ClientId,
    pub class: String,
    pub model: ModelId,
    pub need: Tokens,
    pub waited: Millis,
    /// Times it was paused for more urgent work.
    pub bumped: u32,
    /// Still being stopped on its server.
    pub paused: bool,
}

impl Scheduler {
    pub fn status(&self, now: Millis) -> Status {
        let backends = self
            .backends
            .iter()
            .enumerate()
            .filter(|(_, backend)| backend.known)
            .map(|(index, backend)| {
                let slots = backend
                    .pool
                    .slots
                    .iter()
                    .enumerate()
                    .map(|(slot, state)| {
                        let running = match state {
                            SlotState::Running(running) | SlotState::Preempting { running, .. } => {
                                Some(running)
                            }
                            _ => None,
                        };
                        let client = match state {
                            SlotState::Preparing(claim) => Some(claim.waiting.demand.client),
                            _ => running.map(|running| running.waiting.demand.client),
                        };
                        SlotStatus {
                            slot: SlotId::new(index as u16, slot),
                            state: state.name(),
                            tokens: state.usage(),
                            ticket: state.ticket(),
                            client,
                            running_for: running.map(|running| now.saturating_sub(running.since)),
                            started: running.map(|running| running.started),
                            pausable: running.map(|running| self.pausable(running).is_some()),
                            reuse: running.map(|running| running.reuse.tokens),
                        }
                    })
                    .collect();
                BackendStatus {
                    backend: BackendId(index as u16),
                    online: backend.online,
                    model: backend.model,
                    layout: backend.pool.layout,
                    usage: backend.usage(),
                    stray: backend.pool.stray,
                    capacity: backend.pool.capacity(),
                    swap: backend.swap,
                    partial_reuse: self.config.eviction.partial && backend.partial,
                    slots,
                    snapshots: backend.snapshots.len(),
                    snapshot_bytes: backend.snapshots.bytes(),
                }
            })
            .collect();
        let mut order = self.service_order(now);
        order.extend(
            self.waiting
                .iter()
                .enumerate()
                .filter(|(_, waiting)| waiting.paused)
                .map(|(index, _)| index),
        );
        let waiting = order
            .into_iter()
            .map(|index| {
                let waiting = &self.waiting[index];
                WaitingStatus {
                    ticket: waiting.ticket,
                    client: waiting.demand.client,
                    class: self.config.classes[usize::from(waiting.demand.class.0)]
                        .name
                        .clone(),
                    model: waiting.demand.model,
                    need: waiting.demand.need(),
                    waited: now.saturating_sub(waiting.arrived),
                    bumped: waiting.bumped,
                    paused: waiting.paused,
                }
            })
            .collect();
        Status { backends, waiting }
    }
}
