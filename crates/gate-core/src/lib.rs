//! The deterministic core of llama-gate.
//!
//! A llama.cpp server runs a fixed number of slots over a KV cache of fixed
//! size. When the cache is shared (`--kv-unified`) and the slots' contexts
//! together outgrow it, llama.cpp fails *every* running request. This crate
//! decides, for each request, when and on which backend and slot it may run
//! so that never happens, and which idle contexts to save, restore, or
//! discard (and which waiting work to pause) to make it possible:
//!
//! - [`Scheduler`] is a state machine. It consumes [`Event`]s stamped with a
//!   caller-supplied time and returns [`Command`]s for the caller to carry
//!   out. It performs no I/O and reads no clock, so every decision is
//!   reproducible.
//! - Backends are announced with their [`Shape`]; backends that run the same
//!   [`ModelId`] are interchangeable, and each keeps its own pool and
//!   [`SnapshotIndex`].
//! - [`QueueOrder`] decides who goes next among equal priorities
//!   ([`Vtc`] fair queueing or [`Fifo`]).
//! - [`EvictionOrder`] decides which idle context gives way
//!   ([`LeastRecentlyUsed`], [`Smallest`], [`Largest`], [`CostBenefit`]).
//! - [`Prefix`] content-addresses prompts so a request can be routed to the
//!   slot, or the saved snapshot, that already holds its prefix.
//!
//! Both policy traits are deliberately small; implement them to change how
//! the gate behaves without touching the engine.
#![forbid(unsafe_code)]

mod backend;
mod engine;
mod evict;
mod pool;
mod prefix;
mod queue;
mod snapshot;
mod types;

pub use backend::Shape;
pub use engine::{
    BackendStatus, Backfill, Command, Event, EvictOutcome, EvictionPolicy, Notice, Observation,
    PreemptOutcome, RestoreOutcome, SaveOutcome, Scheduler, SchedulerConfig, SlotStatus, Status,
    WaitingStatus,
};
pub use evict::{CostBenefit, EvictionOrder, Largest, LeastRecentlyUsed, Smallest};
pub use pool::{Layout, Resident};
pub use prefix::Prefix;
pub use queue::{Candidate, Fifo, QueueOrder, Vtc};
pub use snapshot::{SnapshotIndex, SnapshotMeta};
pub use types::{
    BackendId, Class, ClassId, ClientId, Demand, Millis, ModelId, OpId, Rejection, SlotId,
    SnapshotId, TicketId, Tokens,
};
