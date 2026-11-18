//! Configuration: one TOML file, every key optional.
//!
//! Defaults suit a single llama.cpp server with a unified KV cache. Run
//! `llama-gate --print-config` for a complete, commented file.
use crate::units::{Size, Span};
use ipnet::IpNet;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub server: ServerConfig,
    /// The llama.cpp servers. Empty: one, at `server.upstream`.
    pub backends: Vec<BackendConfig>,
    pub kv: KvConfig,
    pub admission: AdmissionConfig,
    pub queue: QueueConfig,
    pub eviction: EvictionConfig,
    /// Scheduling classes by name. A `standard` class (priority 0) is added
    /// when none is configured.
    pub classes: BTreeMap<String, ClassConfig>,
    pub clients: ClientsConfig,
    pub auth: AuthConfig,
    pub observe: ObserveConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Schedule every generative request (the point of the gate).
    Enforce,
    /// Measure and log decisions, but forward everything at once. For
    /// trying the gate on live traffic before trusting it.
    Observe,
    /// A plain reverse proxy.
    Bypass,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    /// Address clients connect to.
    pub listen: SocketAddr,
    /// The llama.cpp server, when no `[[backends]]` are listed.
    pub upstream: String,
    pub mode: Mode,
    /// Largest request body accepted (images arrive base64-encoded).
    pub max_body: Size,
    /// How long running requests may finish after a shutdown signal.
    pub drain: Span,
    /// Request headers removed before forwarding. `x-conversation-id` makes
    /// llama.cpp keep generating after the client disconnects, which would
    /// hide work from the gate.
    pub strip_request_headers: Vec<String>,
    /// Let clients use the slot API (`/slots/{id}?action=...`) through the
    /// gate. Off by default: the gate owns the slots.
    pub allow_slot_actions: bool,
    /// How often each server's health is checked.
    pub health_interval: Span,
    /// How long a request waits while no server of its model is up (for
    /// example while one restarts) before it is refused. Nothing has reached
    /// a server yet, so waiting is safe. 0 waits indefinitely.
    pub upstream_wait: Span,
    /// The model requests that name none (or an unknown one) go to.
    /// Default: the first backend's.
    pub default_model: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen: SocketAddr::from(([0, 0, 0, 0], 8080)),
            upstream: "http://127.0.0.1:8081".into(),
            mode: Mode::Enforce,
            max_body: Size(64 << 20),
            drain: Span::secs(30),
            strip_request_headers: vec!["x-conversation-id".into()],
            allow_slot_actions: false,
            health_interval: Span::secs(2),
            upstream_wait: Span::secs(120),
            default_model: None,
        }
    }
}

/// One llama.cpp server.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct BackendConfig {
    /// For logs, metrics, and the status page. Default: `backend-N`.
    pub name: String,
    pub url: String,
    /// The model's name, as clients ask for it. Backends with the same model
    /// are interchangeable replicas. Default: the backend's name; whatever
    /// the server reports is also accepted.
    pub model: String,
    /// More names clients may use for the model.
    pub aliases: Vec<String>,
    /// The server runs in router mode (several models behind one URL): send
    /// `model` with every call so the router picks this one.
    pub router: bool,
    /// Bearer token the server expects (`llama-server --api-key`).
    pub api_key: Option<Secret>,
    /// Override `[kv]` for this server.
    pub layout: Option<LayoutKind>,
    pub capacity: Option<u64>,
    pub margin: Option<u64>,
    /// The server's `--slot-save-path`, as this process sees it. Each
    /// backend needs its own.
    pub snapshots: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LayoutKind {
    /// One pool shared by all slots (`--kv-unified`, llama.cpp's default when
    /// the slot count is automatic). The safe assumption when unsure.
    Unified,
    /// Each slot owns `n_ctx / n_parallel` cells.
    Split,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct KvConfig {
    /// llama.cpp does not report whether its cache is unified; say so here
    /// (or let `llama-gate run` read it from the server's flags).
    pub layout: LayoutKind,
    /// Total cells (unified) or cells per slot (split). 0 discovers the
    /// value from the server.
    pub capacity: u64,
    /// Cells never handed out, as a safety margin.
    pub margin: u64,
}

impl Default for KvConfig {
    fn default() -> Self {
        Self {
            layout: LayoutKind::Unified,
            capacity: 0,
            margin: 1024,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct AdmissionConfig {
    /// New tokens reserved for a request that sets no limit. Must cover a
    /// reasoning model's thinking budget plus its answer.
    pub default_generation: u64,
    /// Ceiling on any one request's reservation (0: only the pool limits it).
    pub max_generation: u64,
    /// Requests whose prompt leaves less room than this are refused.
    pub min_generation: u64,
    /// Write the granted generation limit into each request, so a request can
    /// never outgrow its reservation. Turn off only if every client already
    /// sets a limit you trust.
    pub enforce_generation: bool,
    /// Tokens assumed per image or audio item.
    pub media_tokens: u64,
    /// Bytes per token assumed when the server cannot tokenize.
    pub bytes_per_token: f64,
    /// Tokens per prefix block for cache affinity and snapshots.
    pub prefix_block: usize,
    /// Tokenize only what is new since a conversation's previous turn.
    pub incremental: bool,
    /// Tokens of conversation heads kept for that.
    pub token_cache: usize,
    /// Check one incremental result in this many against a full
    /// tokenization (the first few are always checked).
    pub verify_every: u64,
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        Self {
            default_generation: 32_768,
            max_generation: 0,
            min_generation: 256,
            enforce_generation: true,
            media_tokens: 4096,
            bytes_per_token: 2.0,
            prefix_block: 64,
            incremental: true,
            token_cache: 4 << 20,
            verify_every: 32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueOrderKind {
    /// Virtual Token Counter fair queueing between clients.
    Vtc,
    /// First come, first served.
    Fifo,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct QueueConfig {
    pub order: QueueOrderKind,
    /// VTC cost of a prompt token.
    pub input_weight: f64,
    /// VTC cost of a generated token.
    pub output_weight: f64,
    pub backfill: BackfillConfig,
    /// A streaming request queued longer than this starts receiving SSE
    /// comment pings so clients and proxies keep the connection open.
    pub keepalive_after: Span,
    /// Interval of those pings. 0 disables them.
    pub keepalive_interval: Span,
    /// `Retry-After` sent with 429 and 503 refusals.
    pub retry_after: Span,
    /// How often queue timeouts and retries are evaluated.
    pub tick: Span,
    /// How often one request may be paused for more urgent work.
    pub max_preemptions: u32,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            order: QueueOrderKind::Vtc,
            input_weight: 1.0,
            output_weight: 2.0,
            backfill: BackfillConfig::default(),
            keepalive_after: Span::secs(5),
            keepalive_interval: Span::secs(15),
            retry_after: Span::secs(5),
            tick: Span(std::time::Duration::from_millis(500)),
            max_preemptions: 3,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct BackfillConfig {
    pub enabled: bool,
    /// Only requests needing at most this many tokens may run ahead of a
    /// blocked one.
    pub max_tokens: u64,
    /// ...and only until the blocked request has waited this long.
    pub window: Span,
}

impl Default for BackfillConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_tokens: 16_384,
            window: Span::secs(60),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvictionMode {
    /// Save idle contexts to disk (`--slot-save-path`) before clearing them,
    /// and restore them when their conversation returns.
    Swap,
    /// Clear idle contexts; a returning conversation is recomputed.
    Discard,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VictimKind {
    LeastRecentlyUsed,
    Smallest,
    Largest,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct EvictionConfig {
    pub mode: EvictionMode,
    pub victim: VictimKind,
    /// The server's `--slot-save-path`, as this process sees it (the same
    /// directory, mounted into both), when no `[[backends]]` are listed.
    /// Swap needs it to manage the files; without it the gate discards.
    pub directory: Option<PathBuf>,
    /// Disk space each backend's snapshots may use; the least recently used
    /// go first.
    pub budget: Size,
    pub min_snapshot_tokens: u64,
    pub min_restore_gain: u64,
    /// Reuse contexts that share only the start of a prompt (a common system
    /// prompt), not just exact continuations. Turned off by itself on a
    /// server whose model cannot rewind to them.
    pub partial_reuse: bool,
    /// Wait before retrying a slot the server failed to clear.
    pub retry: Span,
}

impl Default for EvictionConfig {
    fn default() -> Self {
        Self {
            mode: EvictionMode::Swap,
            victim: VictimKind::LeastRecentlyUsed,
            directory: None,
            budget: Size(32 << 30),
            min_snapshot_tokens: 4096,
            min_restore_gain: 2048,
            partial_reuse: true,
            retry: Span::secs(5),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClassConfig {
    /// Higher priorities are served first.
    pub priority: i32,
    /// Share among clients of the same priority.
    pub weight: f64,
    /// Refuse requests that waited longer than this.
    pub max_wait: Option<Span>,
    pub max_queued_per_client: Option<usize>,
    /// Override [`AdmissionConfig::default_generation`] for this class.
    pub default_generation: Option<u64>,
    /// Override [`AdmissionConfig::max_generation`] for this class.
    pub max_generation: Option<u64>,
    /// Requests of this class may be paused for a request of a higher
    /// priority while they have produced no output (still reading their
    /// prompt, or not streaming). Their progress is saved, and they resume
    /// where they stopped.
    pub preemptible: bool,
}

impl Default for ClassConfig {
    fn default() -> Self {
        Self {
            priority: 0,
            weight: 1.0,
            max_wait: None,
            max_queued_per_client: None,
            default_generation: None,
            max_generation: None,
            preemptible: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClientsConfig {
    /// Class for requests no rule matches.
    pub default_class: String,
    /// Header naming the client (its fairness flow). Unset: clients are told
    /// apart by key, rule, the OpenAI `user` field, then address.
    pub identity_header: Option<String>,
    /// Header naming the class. Only honoured when `trust_class_header`.
    pub class_header: Option<String>,
    pub trust_class_header: bool,
    /// Tell clients apart by the OpenAI `user` field when no rule matches.
    pub use_openai_user: bool,
    /// Never place one client's request on another client's cached context
    /// or snapshot, so no client can learn from cache timings what another
    /// sent.
    pub isolate: bool,
    /// How often rule host names are re-resolved.
    pub resolve_interval: Span,
    pub rules: Vec<ClientRule>,
}

impl Default for ClientsConfig {
    fn default() -> Self {
        Self {
            default_class: "standard".into(),
            identity_header: Some("x-gate-client".into()),
            class_header: Some("x-gate-class".into()),
            trust_class_header: false,
            use_openai_user: true,
            isolate: false,
            resolve_interval: Span::secs(30),
            rules: Vec::new(),
        }
    }
}

/// Matches requests by where they come from. The first matching rule wins.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClientRule {
    /// Client name, used for fairness and in metrics.
    pub name: String,
    pub class: Option<String>,
    /// Source networks.
    pub cidrs: Vec<IpNet>,
    /// Host names whose addresses are sources (for example container names).
    pub hosts: Vec<String>,
    /// Header name and value that identify the client.
    pub header: Option<HeaderMatch>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HeaderMatch {
    pub name: String,
    pub value: String,
}

/// API keys: `Authorization: Bearer <key>` or `x-api-key: <key>`.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct AuthConfig {
    /// Refuse requests without a known key (401). Otherwise keys only name
    /// their clients, and requests without one are identified by rule.
    pub required: bool,
    pub keys: Vec<KeyConfig>,
    /// Keys for the gate's own endpoints (status, events, metrics, reload).
    /// Empty: those are open to anyone who can reach the gate.
    pub admin_keys: Vec<Secret>,
    /// Let `/gate/metrics` be scraped without an admin key.
    pub public_metrics: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KeyConfig {
    pub key: Secret,
    /// The client this key identifies.
    pub client: String,
    pub class: Option<String>,
}

/// A key, given as `"plain-text"`, `"sha256:<hex digest>"`, or `"env:VAR"`
/// (read from the environment). Only its SHA-256 digest is kept.
#[derive(Clone, PartialEq, Eq)]
pub struct Secret {
    digest: [u8; 32],
    /// The key itself when it is needed (to send upstream) and was given.
    plain: Option<String>,
}

impl Secret {
    pub fn new(key: &str) -> Self {
        Self {
            digest: Sha256::digest(key.as_bytes()).into(),
            plain: Some(key.to_owned()),
        }
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        if let Some(hex) = text.strip_prefix("sha256:") {
            let bytes: Vec<u8> = (0..hex.len())
                .step_by(2)
                .map(|index| u8::from_str_radix(hex.get(index..index + 2).unwrap_or("zz"), 16))
                .collect::<Result<_, _>>()
                .map_err(|_| format!("invalid sha256 digest {hex:?}"))?;
            let digest: [u8; 32] = bytes
                .try_into()
                .map_err(|_| "a sha256 digest has 64 hex digits".to_owned())?;
            return Ok(Self {
                digest,
                plain: None,
            });
        }
        if let Some(variable) = text.strip_prefix("env:") {
            let value = std::env::var(variable)
                .map_err(|_| format!("environment variable {variable} is not set"))?;
            return Ok(Self::new(&value));
        }
        if text.is_empty() {
            return Err("a key cannot be empty".into());
        }
        Ok(Self::new(text))
    }

    /// Whether `presented` is this key, compared in constant time.
    pub fn matches(&self, presented: &str) -> bool {
        let digest: [u8; 32] = Sha256::digest(presented.as_bytes()).into();
        digest
            .iter()
            .zip(self.digest.iter())
            .fold(0u8, |difference, (left, right)| difference | (left ^ right))
            == 0
    }

    /// The key itself, when it was given rather than its digest.
    pub fn reveal(&self) -> Option<&str> {
        self.plain.as_deref()
    }

    fn hex(&self) -> String {
        self.digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Secret(sha256:{}…)", &self.hex()[..8])
    }
}

impl<'de> Deserialize<'de> for Secret {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        Secret::parse(&text).map_err(serde::de::Error::custom)
    }
}

impl Serialize for Secret {
    /// Keys are never printed back, only their digests.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(&format_args!("sha256:{}", self.hex()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    Text,
    Json,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ObserveConfig {
    /// Prefix of the gate's own endpoints (`/status`, `/metrics`, `/health`,
    /// `/events`, `/reload`, and the dashboard at `/`).
    pub path_prefix: String,
    pub log_format: LogFormat,
    /// Default log filter (overridden by `RUST_LOG`).
    pub log_level: String,
    /// Serve the live dashboard.
    pub dashboard: bool,
    /// Log one line per finished request (timings, tokens, cache reuse).
    pub access_log: bool,
    /// Finished requests kept for the status page.
    pub recent: usize,
}

impl Default for ObserveConfig {
    fn default() -> Self {
        Self {
            path_prefix: "/gate".into(),
            log_format: LogFormat::Text,
            log_level: "info".into(),
            dashboard: true,
            access_log: true,
            recent: 200,
        }
    }
}

impl Config {
    pub fn from_toml(text: &str) -> Result<Self, String> {
        let mut config: Config = toml::from_str(text).map_err(|error| error.to_string())?;
        config.normalize();
        config.validate()?;
        Ok(config)
    }

    /// Adds the default class when none is configured.
    pub fn normalize(&mut self) {
        if self.classes.is_empty() {
            self.classes
                .insert("standard".into(), ClassConfig::default());
        }
    }

    /// The backends to run: those listed, or one at `server.upstream` (with
    /// `eviction.directory` for its snapshots). Names and models filled in.
    pub fn effective_backends(&self) -> Vec<BackendConfig> {
        let mut backends = if self.backends.is_empty() {
            vec![BackendConfig {
                name: "default".into(),
                url: self.server.upstream.clone(),
                snapshots: self.eviction.directory.clone(),
                ..BackendConfig::default()
            }]
        } else {
            self.backends.clone()
        };
        for (index, backend) in backends.iter_mut().enumerate() {
            if backend.name.trim().is_empty() {
                backend.name = format!("backend-{index}");
            }
            if backend.model.trim().is_empty() {
                backend.model = backend.name.clone();
            }
        }
        backends
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut problems = Vec::new();
        let class_exists = |name: &str| self.classes.contains_key(name);
        if !class_exists(&self.clients.default_class) {
            problems.push(format!(
                "clients.default_class {:?} is not a configured class",
                self.clients.default_class
            ));
        }
        for rule in &self.clients.rules {
            if rule.name.trim().is_empty() {
                problems.push("every client rule needs a name".into());
            }
            if let Some(class) = &rule.class
                && !class_exists(class)
            {
                problems.push(format!(
                    "client rule {:?} names unknown class {class:?}",
                    rule.name
                ));
            }
            if rule.cidrs.is_empty() && rule.hosts.is_empty() && rule.header.is_none() {
                problems.push(format!("client rule {:?} matches nothing", rule.name));
            }
        }
        for key in &self.auth.keys {
            if key.client.trim().is_empty() {
                problems.push("every API key needs a client name".into());
            }
            if let Some(class) = &key.class
                && !class_exists(class)
            {
                problems.push(format!(
                    "API key of {:?} names unknown class {class:?}",
                    key.client
                ));
            }
        }
        if self.auth.required && self.auth.keys.is_empty() {
            problems.push("auth.required needs at least one key in auth.keys".into());
        }
        for (name, class) in &self.classes {
            if !(class.weight > 0.0 && class.weight.is_finite()) {
                problems.push(format!("class {name:?} needs a positive weight"));
            }
        }
        let backends = self.effective_backends();
        let mut names = BTreeSet::new();
        let mut directories = BTreeSet::new();
        for backend in &backends {
            if !(backend.url.starts_with("http://") || backend.url.starts_with("https://")) {
                problems.push(format!(
                    "backend {:?}: url must be an http:// or https:// URL",
                    backend.name
                ));
            }
            if !names.insert(backend.name.clone()) {
                problems.push(format!("backend name {:?} is used twice", backend.name));
            }
            if backend
                .api_key
                .as_ref()
                .is_some_and(|key| key.reveal().is_none())
            {
                problems.push(format!(
                    "backend {:?}: api_key is sent to the server, so give the key itself or env:VAR, not a digest",
                    backend.name
                ));
            }
            // Compared as the filesystem resolves them: two spellings of one
            // directory are one directory.
            if let Some(directory) = &backend.snapshots
                && !directories
                    .insert(std::fs::canonicalize(directory).unwrap_or_else(|_| directory.clone()))
            {
                problems.push(format!(
                    "snapshot directory {} is shared by two backends; each server needs its own",
                    directory.display()
                ));
            }
        }
        if let Some(model) = &self.server.default_model
            && !backends
                .iter()
                .any(|backend| &backend.model == model || backend.aliases.contains(model))
        {
            problems.push(format!(
                "server.default_model {model:?} is not the model of any backend"
            ));
        }
        if self.admission.prefix_block == 0 {
            problems.push("admission.prefix_block must be at least 1".into());
        }
        if self.admission.bytes_per_token.is_nan() || self.admission.bytes_per_token <= 0.0 {
            problems.push("admission.bytes_per_token must be positive".into());
        }
        let prefix = self.observe.path_prefix.trim_end_matches('/');
        if !prefix.starts_with('/')
            || prefix.contains("//")
            || prefix
                .chars()
                .any(|c| c.is_whitespace() || matches!(c, '{' | '}' | '*' | '?'))
        {
            problems.push(format!(
                "observe.path_prefix {:?} must be a path like /gate",
                self.observe.path_prefix
            ));
        }
        for (name, weight) in [
            ("queue.input_weight", self.queue.input_weight),
            ("queue.output_weight", self.queue.output_weight),
        ] {
            if !(weight.is_finite() && weight >= 0.0) {
                problems.push(format!("{name} must be a number of at least 0"));
            }
        }
        if !(self.queue.input_weight > 0.0 || self.queue.output_weight > 0.0) {
            problems.push("queue.input_weight and queue.output_weight cannot both be 0".into());
        }
        // Zero would poll in a busy loop; anything this fast is a mistake.
        for (name, every, floor) in [
            ("server.health_interval", self.server.health_interval.0, 10),
            (
                "clients.resolve_interval",
                self.clients.resolve_interval.0,
                100,
            ),
            ("queue.tick", self.queue.tick.0, 10),
        ] {
            if every.as_millis() < floor {
                problems.push(format!("{name} must be at least {floor} ms"));
            }
        }
        let admission = &self.admission;
        if admission.max_generation > 0 && admission.default_generation > admission.max_generation {
            problems
                .push("admission.default_generation cannot exceed admission.max_generation".into());
        }
        if admission.min_generation > admission.default_generation {
            problems
                .push("admission.min_generation cannot exceed admission.default_generation".into());
        }
        for (name, class) in &self.classes {
            if let (Some(default), Some(max)) = (class.default_generation, class.max_generation)
                && default > max
            {
                problems.push(format!(
                    "class {name:?}: default_generation cannot exceed max_generation"
                ));
            }
        }
        if self.kv.capacity > 0 && self.kv.margin >= self.kv.capacity {
            problems.push("kv.margin must be smaller than kv.capacity".into());
        }
        for backend in &backends {
            if let (Some(capacity), margin) =
                (backend.capacity, backend.margin.unwrap_or(self.kv.margin))
                && capacity > 0
                && margin >= capacity
            {
                problems.push(format!(
                    "backend {:?}: margin must be smaller than capacity",
                    backend.name
                ));
            }
        }
        // Classes and backends are numbered in 16 bits.
        if self.classes.len() > usize::from(u16::MAX) {
            problems.push("too many classes".into());
        }
        if backends.len() > usize::from(u16::MAX) {
            problems.push("too many backends".into());
        }
        if problems.is_empty() {
            Ok(())
        } else {
            Err(problems.join("; "))
        }
    }

    /// Class names in a stable order; a class's index is its id.
    pub fn class_names(&self) -> Vec<String> {
        self.classes.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_file_is_a_valid_configuration() {
        let config = Config::from_toml("").unwrap();
        assert_eq!(config.class_names(), vec!["standard"]);
        assert_eq!(config.server.mode, Mode::Enforce);
        let backends = config.effective_backends();
        assert_eq!(backends.len(), 1);
        assert_eq!(backends[0].url, "http://127.0.0.1:8081");
        assert_eq!(backends[0].model, "default");
    }

    #[test]
    fn mistakes_are_reported_together() {
        let error = Config::from_toml(
            r#"
            [clients]
            default_class = "missing"
            [[clients.rules]]
            name = "nothing"
            [[backends]]
            url = "llama:8080"
            snapshots = "/slots"
            [[backends]]
            url = "http://other:8080"
            snapshots = "/slots"
            "#,
        )
        .unwrap_err();
        assert!(error.contains("default_class"), "{error}");
        assert!(error.contains("matches nothing"), "{error}");
        assert!(error.contains("http:// or https://"), "{error}");
        assert!(error.contains("shared by two backends"), "{error}");
    }

    #[test]
    fn the_shipped_configurations_parse() {
        Config::from_toml(crate::EXAMPLE_CONFIG).unwrap();
        Config::from_toml(include_str!("../../../deploy/llama-gate.toml")).unwrap();
    }

    #[test]
    fn unknown_keys_are_refused() {
        assert!(Config::from_toml("[server]\nlisten_port = 1").is_err());
    }

    #[test]
    fn the_printed_defaults_parse_back() {
        let printed = toml::to_string(&{
            let mut config = Config::default();
            config.normalize();
            config
        })
        .unwrap();
        Config::from_toml(&printed).unwrap();
    }

    #[test]
    fn keys_are_kept_as_digests_and_never_printed() {
        let config = Config::from_toml(
            r#"
            [auth]
            required = true
            admin_keys = ["sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"]
            [[auth.keys]]
            key = "sk-desktop"
            client = "desktop-app"
            "#,
        )
        .unwrap();
        assert!(config.auth.admin_keys[0].matches("test"));
        assert!(!config.auth.admin_keys[0].matches("tests"));
        assert!(config.auth.keys[0].key.matches("sk-desktop"));
        let printed = toml::to_string(&config).unwrap();
        assert!(!printed.contains("sk-desktop"), "{printed}");
        Config::from_toml(&printed).unwrap();
    }

    #[test]
    fn values_that_would_misbehave_at_runtime_are_refused() {
        // Parsing validates; validating again must agree.
        let refused = |toml: &str| {
            Config::from_toml(toml)
                .and_then(|config| config.validate())
                .expect_err(toml)
        };
        assert!(refused("[observe]\npath_prefix = \"//\"").contains("path_prefix"));
        assert!(refused("[server]\nhealth_interval = \"0s\"").contains("health_interval"));
        assert!(refused("[clients]\nresolve_interval = \"0s\"").contains("resolve_interval"));
        assert!(refused("[queue]\ninput_weight = -1.0").contains("input_weight"));
        assert!(refused("[kv]\ncapacity = 1000\nmargin = 1000").contains("margin"));
        assert!(
            refused("[admission]\ndefault_generation = 100\nmax_generation = 50")
                .contains("default_generation")
        );
        assert!(
            Config::from_toml("[observe]\npath_prefix = \"/gate/\"")
                .unwrap()
                .validate()
                .is_ok()
        );
    }
}
