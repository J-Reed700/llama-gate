//! One backend as the scheduler sees it: its shape, its slots, its saved
//! contexts, and what it has learned about the server.
use crate::pool::{Layout, Pool};
use crate::{Millis, ModelId, SnapshotIndex, Tokens};

pub(crate) const DEFAULT_PREFILL_TPS: u64 = 2_000;
pub(crate) const DEFAULT_DECODE_TPS: u64 = 100;
pub(crate) const DEFAULT_RESTORE_MS: u64 = 500;
pub(crate) const DEFAULT_SAVE_MS: u64 = 500;
const MAX_TPS: u64 = 10_000_000;
const MAX_OP_MS: u64 = 3_600_000;

/// Integer/fixed-point EWMA with alpha = 1/4, overflow-safe via `u128`.
pub(crate) fn ewma_u64(current: u64, sample: u64) -> u64 {
    let blended = (u128::from(current) * 3 + u128::from(sample) + 2) / 4;
    u64::try_from(blended).unwrap_or(u64::MAX)
}

/// A backend's shape, as the caller discovers it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub model: ModelId,
    pub layout: Layout,
    pub slots: usize,
    /// The server can save and restore slots (`--slot-save-path`) and the
    /// caller manages the files.
    pub swap: bool,
}

#[derive(Debug)]
pub(crate) struct Backend {
    /// Announced at least once; unannounced backends hold nothing.
    pub known: bool,
    pub online: bool,
    pub model: ModelId,
    pub swap: bool,
    pub pool: Pool,
    pub snapshots: SnapshotIndex,
    /// Partial reuse has not (yet) proven useless on this server.
    pub partial: bool,
    pub partial_hits: u32,
    pub partial_misses: u32,
    /// Added to every reservation here: the largest recent excess of a
    /// finished context over what was reserved for it (prompts the gate
    /// measured short, tokens the server adds). It decays as contexts fit.
    pub overrun: Tokens,
    /// EWMA of uncached prompt prefill throughput (tokens/sec).
    pub prefill_tps: u64,
    /// EWMA of decode throughput (tokens/sec).
    pub decode_tps: u64,
    /// EWMA of snapshot restore duration (ms).
    pub restore_ms: u64,
    /// EWMA of snapshot save duration (ms).
    pub save_ms: u64,
    pub prefill_observed: bool,
    pub decode_observed: bool,
    pub restore_observed: bool,
    pub save_observed: bool,
}

impl Backend {
    pub fn unknown() -> Self {
        Self {
            known: false,
            online: false,
            model: ModelId(u16::MAX),
            swap: false,
            pool: Pool::new(Layout::Unified { capacity: 0 }, 0),
            snapshots: SnapshotIndex::default(),
            partial: true,
            partial_hits: 0,
            partial_misses: 0,
            overrun: 0,
            prefill_tps: DEFAULT_PREFILL_TPS,
            decode_tps: DEFAULT_DECODE_TPS,
            restore_ms: DEFAULT_RESTORE_MS,
            save_ms: DEFAULT_SAVE_MS,
            prefill_observed: false,
            decode_observed: false,
            restore_observed: false,
            save_observed: false,
        }
    }

    pub fn reset_telemetry(&mut self) {
        self.prefill_tps = DEFAULT_PREFILL_TPS;
        self.decode_tps = DEFAULT_DECODE_TPS;
        self.restore_ms = DEFAULT_RESTORE_MS;
        self.save_ms = DEFAULT_SAVE_MS;
        self.prefill_observed = false;
        self.decode_observed = false;
        self.restore_observed = false;
        self.save_observed = false;
    }

    pub fn has_speed_estimate(&self) -> bool {
        self.prefill_observed && self.decode_observed
    }

    pub fn observe_prefill(&mut self, uncached_tokens: Tokens, elapsed_ms: Millis) {
        if uncached_tokens == 0 || elapsed_ms == 0 {
            return;
        }
        let sample = ((u128::from(uncached_tokens) * 1000) / u128::from(elapsed_ms))
            .clamp(1, u128::from(MAX_TPS)) as u64;
        self.prefill_tps = if self.prefill_observed {
            ewma_u64(self.prefill_tps, sample).max(1)
        } else {
            self.prefill_observed = true;
            sample
        };
    }

    pub fn observe_decode(&mut self, generated: Tokens, elapsed_ms: Millis) {
        if generated < 2 || elapsed_ms == 0 {
            return;
        }
        let sample = ((u128::from(generated) * 1000) / u128::from(elapsed_ms))
            .clamp(1, u128::from(MAX_TPS)) as u64;
        self.decode_tps = if self.decode_observed {
            ewma_u64(self.decode_tps, sample).max(1)
        } else {
            self.decode_observed = true;
            sample
        };
    }

    pub fn observe_restore(&mut self, elapsed_ms: Millis) {
        if elapsed_ms == 0 {
            return;
        }
        let sample = elapsed_ms.min(MAX_OP_MS);
        self.restore_ms = if self.restore_observed {
            ewma_u64(self.restore_ms, sample)
        } else {
            self.restore_observed = true;
            sample
        };
    }

    pub fn observe_save(&mut self, elapsed_ms: Millis) {
        if elapsed_ms == 0 {
            return;
        }
        let sample = elapsed_ms.min(MAX_OP_MS);
        self.save_ms = if self.save_observed {
            ewma_u64(self.save_ms, sample)
        } else {
            self.save_observed = true;
            sample
        };
    }

    pub fn shape(&self) -> Shape {
        Shape {
            model: self.model,
            layout: self.pool.layout,
            slots: self.pool.slots.len(),
            swap: self.swap,
        }
    }

    /// Whether requests for `model` may be placed here now.
    pub fn serves(&self, model: ModelId) -> bool {
        self.known && self.online && self.model == model
    }

    pub fn usage(&self) -> Tokens {
        self.pool.usage()
    }

    /// Usage in millionths of capacity, for comparing backends.
    pub fn load(&self) -> u64 {
        let capacity = self.pool.capacity().max(1);
        self.usage().saturating_mul(1_000_000) / capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_guards_zero_and_overflow_and_resets() {
        let mut backend = Backend::unknown();
        assert!(!backend.has_speed_estimate());
        assert_eq!(backend.prefill_tps, DEFAULT_PREFILL_TPS);
        assert_eq!(backend.decode_tps, DEFAULT_DECODE_TPS);

        // Zero duration and zero/single token observations are ignored safely.
        backend.observe_prefill(0, 100);
        backend.observe_prefill(1_000, 0);
        backend.observe_decode(0, 100);
        backend.observe_decode(1, 100);
        backend.observe_decode(10, 0);
        backend.observe_restore(0);
        backend.observe_save(0);
        assert!(!backend.has_speed_estimate());
        assert_eq!(backend.prefill_tps, DEFAULT_PREFILL_TPS);
        assert_eq!(backend.decode_tps, DEFAULT_DECODE_TPS);
        assert_eq!(backend.restore_ms, DEFAULT_RESTORE_MS);
        assert_eq!(backend.save_ms, DEFAULT_SAVE_MS);

        // Huge u64 values never overflow or produce zero TPS.
        backend.observe_prefill(u64::MAX, 1);
        backend.observe_decode(u64::MAX, 1);
        backend.observe_restore(u64::MAX);
        backend.observe_save(u64::MAX);
        assert!(backend.has_speed_estimate());
        assert_eq!(backend.prefill_tps, MAX_TPS);
        assert_eq!(backend.decode_tps, MAX_TPS);
        assert_eq!(backend.restore_ms, MAX_OP_MS);
        assert_eq!(backend.save_ms, MAX_OP_MS);

        // Extremely slow rates clamp to >= 1 tok/s.
        backend.reset_telemetry();
        backend.observe_prefill(1, 10_000);
        backend.observe_decode(2, 10_000);
        assert_eq!(backend.prefill_tps, 1);
        assert_eq!(backend.decode_tps, 1);

        // EWMA blends subsequent observations cleanly.
        backend.reset_telemetry();
        backend.observe_prefill(4_000, 1_000); // 4_000
        backend.observe_prefill(8_000, 1_000); // (3*4000 + 8000 + 2)/4 = 5_000
        assert_eq!(backend.prefill_tps, 5_000);
    }
}
