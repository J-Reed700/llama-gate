//! Which idle context gives way when room is needed.
use crate::{Millis, Resident, SlotId};

/// Ranks idle contexts for eviction; the lowest rank is evicted first.
/// Implement it to plug in another policy. Implementations must be
/// deterministic.
pub trait EvictionOrder: Send + std::fmt::Debug {
    fn rank(&self, slot: SlotId, resident: &Resident) -> f64;

    fn rank_at(&self, slot: SlotId, resident: &Resident, _now: Millis) -> f64 {
        self.rank(slot, resident)
    }
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

/// Ranks idle contexts by rebuild cost amortized by idle age, protecting
/// large warm prefixes while aging out stale ones (lowest rank evicted first).
#[derive(Clone, Copy, Debug, Default)]
pub struct CostBenefit;

impl EvictionOrder for CostBenefit {
    fn rank(&self, _slot: SlotId, resident: &Resident) -> f64 {
        resident.tokens as f64
    }

    fn rank_at(&self, _slot: SlotId, resident: &Resident, now: Millis) -> f64 {
        let age_minutes = now.saturating_sub(resident.last_used) as f64 / 60_000.0;
        (resident.tokens as f64) / (1.0 + age_minutes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Prefix;

    #[test]
    fn cost_benefit_protects_large_warm_and_ages_out_stale() {
        let policy = CostBenefit;
        let slot = SlotId::new(0, 0);
        let warm_large = Resident {
            tokens: 10_000,
            prefix: Prefix::opaque(10_000),
            last_used: 600_000,
        };
        let warm_small = Resident {
            tokens: 1_000,
            prefix: Prefix::opaque(1_000),
            last_used: 600_000,
        };
        let stale_large = Resident {
            tokens: 10_000,
            prefix: Prefix::opaque(10_000),
            last_used: 0, // 10 minutes old at now = 600_000 -> 10_000 / 11 ≈ 909
        };
        let now = 600_000;
        let r_warm_large = policy.rank_at(slot, &warm_large, now);
        let r_warm_small = policy.rank_at(slot, &warm_small, now);
        let r_stale_large = policy.rank_at(slot, &stale_large, now);
        assert!(r_stale_large < r_warm_small);
        assert!(r_warm_small < r_warm_large);

        // Edge cases: now < last_used, zero tokens, u64::MAX.
        let r_future = policy.rank_at(slot, &warm_large, 0);
        assert_eq!(r_future, 10_000.0);
        let zero = Resident {
            tokens: 0,
            prefix: Prefix::opaque(0),
            last_used: 0,
        };
        assert_eq!(policy.rank_at(slot, &zero, u64::MAX), 0.0);
        let max = Resident {
            tokens: u64::MAX,
            prefix: Prefix::opaque(0),
            last_used: 0,
        };
        assert!(policy.rank_at(slot, &max, u64::MAX).is_finite());

        // Legacy EvictionOrder without rank_at falls back to rank().
        assert_eq!(LeastRecentlyUsed.rank_at(slot, &warm_large, now), 600_000.0);
    }
}
