//! Identifiers and value types shared by every part of the engine.
use crate::Prefix;
use serde::{Deserialize, Serialize};

/// A number of tokens, which is also a number of KV cells.
pub type Tokens = u64;

/// Milliseconds on a monotonic clock owned by the caller. The engine never
/// reads a clock itself; every event carries the time it happened.
pub type Millis = u64;

macro_rules! id {
    ($(#[$doc:meta])* $name:ident($inner:ty)) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub $inner);

        impl std::fmt::Display for $name {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.0.fmt(formatter)
            }
        }
    };
}

id!(
    /// One request, from arrival until it finishes or is refused.
    TicketId(u64)
);
id!(
    /// One server the gate schedules: a `llama-server` with its own slots and
    /// KV pool.
    BackendId(u16)
);
id!(
    /// A model the gate serves. Backends running the same model are
    /// interchangeable, so a request may go to any of them.
    ModelId(u16)
);
id!(
    /// A slot context saved to disk.
    SnapshotId(u64)
);
id!(
    /// A client as identified by the caller. The engine only compares them.
    ClientId(u32)
);
id!(
    /// One slot operation (an eviction, save, restore, or preemption). Its
    /// completion must name it, so a late report from an operation the engine
    /// has since abandoned (for example across a server restart) is ignored.
    OpId(u64)
);
id!(
    /// An index into [`crate::SchedulerConfig::classes`].
    ClassId(u16)
);

/// One slot of one backend; `index` is llama.cpp's `id_slot`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SlotId {
    pub backend: BackendId,
    pub index: usize,
}

impl SlotId {
    pub const fn new(backend: u16, index: usize) -> Self {
        Self {
            backend: BackendId(backend),
            index,
        }
    }
}

impl std::fmt::Display for SlotId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}/{}", self.backend, self.index)
    }
}

/// How one class of traffic is scheduled.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Class {
    pub name: String,
    /// Waiting requests of a higher priority are always offered a slot first.
    pub priority: i32,
    /// Share of service relative to other clients of the same priority.
    pub weight: f64,
    /// Longest a request may wait for a slot before it is refused.
    pub max_wait: Option<Millis>,
    /// Most requests one client of this class may have waiting at once.
    pub max_queued_per_client: Option<usize>,
    /// Its requests may be paused for a request of a higher priority while
    /// they have produced no output: stopped, their progress saved, and put
    /// back in line to resume where they stopped.
    pub preemptible: bool,
}

impl Class {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            priority: 0,
            weight: 1.0,
            max_wait: None,
            max_queued_per_client: None,
            preemptible: false,
        }
    }
}

/// What a request needs from the server.
#[derive(Clone, Debug)]
pub struct Demand {
    pub client: ClientId,
    pub class: ClassId,
    /// The backends that may serve it.
    pub model: ModelId,
    /// Prompt tokens one of its tasks holds at once: measured exactly, or a
    /// conservative estimate. For a request of several prompts, the largest.
    pub prompt: Tokens,
    /// Prompt tokens it processes in total, charged to its client for
    /// fairness. Equals `prompt` for a single prompt.
    pub work: Tokens,
    /// New tokens to reserve. The caller enforces the granted amount as the
    /// request's generation limit, so a running request can never outgrow
    /// its reservation.
    pub generation: Tokens,
    /// The least generation still worth running the request with. A request
    /// whose prompt leaves less room than this is refused instead.
    pub min_generation: Tokens,
    /// Content address of the prompt, for cache affinity and snapshots.
    pub prefix: Prefix,
    /// The request needs every slot of a backend: several prompts or choices
    /// at once, which the server spreads over its slots itself.
    pub exclusive: bool,
    /// The server runs it in the slot it is pinned to (`id_slot`). An
    /// unpinned request (an embedding, say) may land in any idle slot, so it
    /// is only placed in an empty one, whose room then holds wherever it runs,
    /// and never reuses a context.
    pub pinned: bool,
}

impl Demand {
    /// Prompt plus reserved generation.
    pub fn need(&self) -> Tokens {
        self.prompt.saturating_add(self.generation)
    }
}

/// Why a request was refused. Each maps to one HTTP answer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum Rejection {
    /// The prompt does not fit even in an empty server (HTTP 400).
    TooLarge { prompt: Tokens, limit: Tokens },
    /// The request waited longer than its class allows (HTTP 429).
    Timeout { waited: Millis },
    /// The client already has its class's maximum waiting (HTTP 429).
    QueueFull { queued: usize },
    /// The demand named a class the engine was not configured with (HTTP 500).
    UnknownClass,
    /// No server of its model came up within the configured time (HTTP 503).
    Unavailable { waited: Millis },
    /// The engine is shutting down (HTTP 503).
    ShuttingDown,
}
