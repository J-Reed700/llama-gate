//! One backend as the scheduler sees it: its shape, its slots, its saved
//! contexts, and what it has learned about the server.
use crate::pool::{Layout, Pool};
use crate::{ModelId, SnapshotIndex, Tokens};

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
        }
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
