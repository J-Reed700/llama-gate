//! Which idle context gives way when room is needed.
use crate::{Resident, SlotId};

/// Ranks idle contexts for eviction; the lowest rank is evicted first.
/// Implement it to plug in another policy. Implementations must be
/// deterministic.
pub trait EvictionOrder: Send + std::fmt::Debug {
    fn rank(&self, slot: SlotId, resident: &Resident) -> f64;
}

/// The context unused for longest goes first.
#[derive(Clone, Copy, Debug, Default)]
pub struct LeastRecentlyUsed;

impl EvictionOrder for LeastRecentlyUsed {
    fn rank(&self, _slot: SlotId, resident: &Resident) -> f64 {
        resident.last_used as f64
    }
}

/// The smallest context goes first: cheapest to rebuild or reload.
#[derive(Clone, Copy, Debug, Default)]
pub struct Smallest;

impl EvictionOrder for Smallest {
    fn rank(&self, _slot: SlotId, resident: &Resident) -> f64 {
        resident.tokens as f64
    }
}

/// The largest context goes first: frees the most room per eviction.
#[derive(Clone, Copy, Debug, Default)]
pub struct Largest;

impl EvictionOrder for Largest {
    fn rank(&self, _slot: SlotId, resident: &Resident) -> f64 {
        -(resident.tokens as f64)
    }
}
