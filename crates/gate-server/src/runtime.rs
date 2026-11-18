//! The engine actor: owns the scheduler, performs its commands, and turns
//! their outcomes back into events.
//!
//! Every change to scheduling state happens on one task, so the scheduler
//! needs no lock and its decisions are serialized exactly as in the core's
//! simulation tests. Slow work (saving, restoring, erasing slots, stopping a
//! paused request) runs in spawned tasks that report back through the same
//! queue. Each backend has a watcher task reporting its health and shape.
use crate::config::{BackendConfig, Config, EvictionMode, LayoutKind, QueueOrderKind, VictimKind};
use crate::control::Control;
use crate::metrics::{Metrics, Timing, labels};
use crate::models::{Models, reported_names};
use crate::snapshots::SnapshotStore;
use gate_core::{
    BackendId, Backfill, Class, ClassId, ClientId, Command, Demand, Event, EvictOutcome,
    EvictionOrder, EvictionPolicy, Fifo, Largest, Layout, LeastRecentlyUsed, Millis, ModelId,
    Notice, Observation, PreemptOutcome, QueueOrder, Rejection, RestoreOutcome, SaveOutcome,
    Scheduler, SchedulerConfig, Shape, SlotId, SnapshotId, SnapshotIndex, TicketId, Tokens, Vtc,
};
use gate_llama::usage::Usage;
use gate_llama::{
    Boundaries, Capabilities, Health, Measurer, Server, ServerError, ServerInfo, SlotView,
};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc, oneshot, watch};
use tracing::{debug, info, warn};

/// The answer to a waiting request.
#[derive(Debug)]
pub enum Grant {
    Run {
        slot: SlotId,
        generation: Tokens,
        exclusive: bool,
        /// Prompt tokens the server should find in its cache.
        reuse: Tokens,
    },
    Refused(Rejection),
}

/// A measured request asking for a slot.
#[derive(Debug)]
pub struct Arrival {
    pub ticket: TicketId,
    pub demand: Demand,
    pub client: Arc<str>,
    pub class: Arc<str>,
    pub dialect: &'static str,
    /// How its prompt was measured: exact, incremental, or estimated.
    pub measured: &'static str,
    pub control: Arc<Control>,
    pub reply: oneshot::Sender<Grant>,
}

/// How a request ended, as its handler saw it.
#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub ticket: TicketId,
    /// Milliseconds since the gate started; set by the engine.
    pub at_ms: Millis,
    pub client: Arc<str>,
    pub class: Arc<str>,
    pub model: String,
    pub dialect: &'static str,
    pub backend: Option<String>,
    pub slot: Option<usize>,
    /// `ok`, `error`, `cancelled`, or `refused`.
    pub outcome: &'static str,
    pub status: u16,
    pub prompt: Tokens,
    pub cached: Option<Tokens>,
    pub generated: Option<Tokens>,
    pub measure_ms: u64,
    pub queue_ms: u64,
    pub ttft_ms: Option<u64>,
    pub total_ms: u64,
    pub tokens_per_second: Option<f64>,
    pub preempted: u32,
    pub bounced: u32,
}

/// What a backend's watcher reports.
#[derive(Debug)]
pub enum Upstream {
    /// First seen ready, back after an outage, restarted, or reshaped.
    Ready {
        info: ServerInfo,
        capabilities: Capabilities,
        slots: Vec<SlotView>,
        restarted: bool,
    },
    /// Still ready and unchanged.
    Alive,
    Down,
}

enum Message {
    Arrive(Arrival),
    Withdraw {
        ticket: TicketId,
        report: Option<Report>,
    },
    Started(TicketId),
    Finish {
        ticket: TicketId,
        usage: Usage,
        report: Report,
    },
    Bounce {
        ticket: TicketId,
        reply: oneshot::Sender<Grant>,
    },
    Outcome {
        event: Event,
        detail: Option<Value>,
    },
    SwapUnsupported(BackendId, String),
    Upstream(BackendId, Upstream),
    Status(oneshot::Sender<Value>),
    Reconfigure(Arc<Config>, oneshot::Sender<Result<Vec<String>, String>>),
    Shutdown(oneshot::Sender<()>),
}

/// Milliseconds since the gate started: the engine's only clock.
#[derive(Clone, Copy, Debug)]
pub struct Clock(Instant);

impl Clock {
    pub fn now(&self) -> Millis {
        self.0.elapsed().as_millis() as Millis
    }
}

/// A cheap, cloneable way to talk to the engine.
#[derive(Clone, Debug)]
pub struct Handle {
    sender: mpsc::UnboundedSender<Message>,
    events: broadcast::Sender<Arc<str>>,
    pub metrics: Arc<Metrics>,
    pub clock: Clock,
}

impl Handle {
    pub fn arrive(&self, arrival: Arrival) {
        let _ = self.sender.send(Message::Arrive(arrival));
    }

    /// The request left (its client went away, or it failed upstream).
    pub fn withdraw(&self, ticket: TicketId, report: Option<Report>) {
        let _ = self.sender.send(Message::Withdraw { ticket, report });
    }

    /// The request's output began reaching its client.
    pub fn started(&self, ticket: TicketId) {
        let _ = self.sender.send(Message::Started(ticket));
    }

    pub fn finish(&self, ticket: TicketId, usage: Usage, report: Report) {
        let _ = self.sender.send(Message::Finish {
            ticket,
            usage,
            report,
        });
    }

    /// The request's server refused the connection before anything reached
    /// it. Returns where its next grant arrives.
    pub fn bounce(&self, ticket: TicketId) -> oneshot::Receiver<Grant> {
        let (reply, receiver) = oneshot::channel();
        let _ = self.sender.send(Message::Bounce { ticket, reply });
        receiver
    }

    pub async fn status(&self) -> Option<Value> {
        let (reply, answer) = oneshot::channel();
        self.sender.send(Message::Status(reply)).ok()?;
        tokio::time::timeout(Duration::from_secs(5), answer)
            .await
            .ok()?
            .ok()
    }

    /// Live events, one JSON object per message.
    pub fn subscribe(&self) -> broadcast::Receiver<Arc<str>> {
        self.events.subscribe()
    }

    /// Applies a new configuration; returns notes on what needs a restart.
    pub async fn reconfigure(&self, config: Arc<Config>) -> Result<Vec<String>, String> {
        let (reply, answer) = oneshot::channel();
        self.sender
            .send(Message::Reconfigure(config, reply))
            .map_err(|_| "the engine stopped".to_owned())?;
        answer.await.map_err(|_| "the engine stopped".to_owned())?
    }

    /// Refuses everything waiting and stops the engine.
    pub async fn shutdown(&self) {
        let (reply, answer) = oneshot::channel();
        if self.sender.send(Message::Shutdown(reply)).is_ok() {
            let _ = tokio::time::timeout(Duration::from_secs(5), answer).await;
        }
    }
}

/// What the data plane needs to know about one backend.
#[derive(Debug)]
pub struct BackendLink {
    pub id: BackendId,
    pub name: String,
    pub model: ModelId,
    pub server: Server,
    /// The model name to put in forwarded requests (router mode).
    pub router_model: Option<String>,
    /// Bearer token the server expects.
    pub api_key: Option<String>,
    pub health: watch::Receiver<Health>,
    pub measurer: watch::Receiver<Measurer>,
}

struct Waiter {
    reply: oneshot::Sender<Grant>,
    since: Instant,
    client: Arc<str>,
    client_id: ClientId,
    class: Arc<str>,
    dialect: &'static str,
    need: Tokens,
    control: Arc<Control>,
}

struct Runner {
    client: Arc<str>,
    client_id: ClientId,
    class: Arc<str>,
    dialect: &'static str,
    need: Tokens,
    control: Arc<Control>,
}

impl Runner {
    fn waiter(self, reply: oneshot::Sender<Grant>) -> Waiter {
        Waiter {
            reply,
            since: Instant::now(),
            client: self.client,
            client_id: self.client_id,
            class: self.class,
            dialect: self.dialect,
            need: self.need,
            control: self.control,
        }
    }
}

#[derive(Debug, Default, Serialize)]
struct ClientStats {
    #[serde(skip)]
    id: Option<ClientId>,
    requests: u64,
    prompt_tokens: u64,
    cached_tokens: u64,
    generated_tokens: u64,
}

struct BackendState {
    link: Arc<BackendLink>,
    config: BackendConfig,
    store: SnapshotStore,
    /// The model identity prefixes and snapshots are scoped to.
    namespace: Option<String>,
    info: Option<ServerInfo>,
    capabilities: Capabilities,
    known: bool,
    up: bool,
}

/// Policies a program may supply instead of the configured ones.
#[derive(Default)]
pub struct Policies {
    pub queue_order: Option<Box<dyn QueueOrder>>,
    pub eviction_order: Option<Box<dyn EvictionOrder>>,
}

impl std::fmt::Debug for Policies {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Policies")
            .field("queue_order", &self.queue_order)
            .field("eviction_order", &self.eviction_order)
            .finish()
    }
}

pub struct Engine {
    config: Arc<Config>,
    scheduler: Scheduler,
    backends: Vec<BackendState>,
    models: Arc<Models>,
    metrics: Arc<Metrics>,
    clock: Clock,
    sender: mpsc::UnboundedSender<Message>,
    receiver: mpsc::UnboundedReceiver<Message>,
    events: broadcast::Sender<Arc<str>>,
    waiting: HashMap<TicketId, Waiter>,
    running: HashMap<TicketId, Runner>,
    dirty: BTreeSet<BackendId>,
    recent: VecDeque<Report>,
    clients: BTreeMap<Arc<str>, ClientStats>,
    class_names: Vec<String>,
    started: Instant,
}

impl Engine {
    pub fn new(
        config: Arc<Config>,
        links: Vec<(Arc<BackendLink>, BackendConfig)>,
        models: Arc<Models>,
        policies: Policies,
    ) -> (Self, Handle) {
        let (sender, receiver) = mpsc::unbounded_channel();
        let (events, _) = broadcast::channel(1024);
        let metrics = Arc::new(Metrics::default());
        let clock = Clock(Instant::now());
        let handle = Handle {
            sender: sender.clone(),
            events: events.clone(),
            metrics: metrics.clone(),
            clock,
        };
        let order: Box<dyn QueueOrder> = match policies.queue_order {
            Some(order) => order,
            None => match config.queue.order {
                QueueOrderKind::Vtc => Box::new(Vtc::new(
                    config.queue.input_weight,
                    config.queue.output_weight,
                )),
                QueueOrderKind::Fifo => Box::new(Fifo),
            },
        };
        let victims: Box<dyn EvictionOrder> = match policies.eviction_order {
            Some(victims) => victims,
            None => match config.eviction.victim {
                VictimKind::LeastRecentlyUsed => Box::new(LeastRecentlyUsed),
                VictimKind::Smallest => Box::new(gate_core::Smallest),
                VictimKind::Largest => Box::new(Largest),
            },
        };
        let scheduler = Scheduler::new(scheduler_config(&config), order, victims);
        let backends = links
            .into_iter()
            .map(|(link, backend)| BackendState {
                store: SnapshotStore::new(backend.snapshots.clone()),
                link,
                config: backend,
                namespace: None,
                info: None,
                capabilities: Capabilities::default(),
                known: false,
                up: false,
            })
            .collect();
        let engine = Self {
            class_names: config.class_names(),
            config,
            scheduler,
            backends,
            models,
            metrics,
            clock,
            sender,
            receiver,
            events,
            waiting: HashMap::new(),
            running: HashMap::new(),
            dirty: BTreeSet::new(),
            recent: VecDeque::new(),
            clients: BTreeMap::new(),
            started: Instant::now(),
        };
        (engine, handle)
    }

    pub async fn run(mut self) {
        let mut tick =
            tokio::time::interval(self.config.queue.tick.0.max(Duration::from_millis(50)));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                message = self.receiver.recv() => {
                    let Some(message) = message else { break };
                    if !self.handle(message) {
                        break;
                    }
                }
                _ = tick.tick() => {
                    let now = self.clock.now();
                    self.feed(Event::Tick { now });
                }
            }
            self.persist();
            self.publish();
        }
    }

    /// Handles one message; false stops the engine.
    fn handle(&mut self, message: Message) -> bool {
        let now = self.clock.now();
        match message {
            Message::Arrive(arrival) => self.admit(arrival),
            Message::Withdraw { ticket, report } => {
                self.waiting.remove(&ticket);
                self.running.remove(&ticket);
                if let Some(report) = report {
                    self.record(report);
                }
                self.feed(Event::Withdrawn { ticket, now });
            }
            Message::Started(ticket) => self.feed(Event::Started { ticket, now }),
            Message::Finish {
                ticket,
                usage,
                report,
            } => {
                self.running.remove(&ticket);
                let resident = usage.resident(report.prompt);
                self.record(report);
                self.feed(Event::Finished {
                    ticket,
                    resident,
                    generated: usage.generated,
                    cached: usage.cached,
                    now,
                });
            }
            Message::Bounce { ticket, reply } => self.bounce(ticket, reply),
            Message::Outcome { event, detail } => {
                self.count_outcome(&event);
                if let Some(detail) = detail {
                    self.emit(detail);
                }
                self.feed(event);
            }
            Message::SwapUnsupported(backend, reason) => {
                if self.scheduler.swap_enabled(backend) {
                    warn!(backend = self.name(backend), %reason, "the server cannot save slots; its idle contexts will be discarded instead (start llama-server with --slot-save-path to enable swapping)");
                    self.scheduler.disable_swap(backend);
                }
            }
            Message::Upstream(backend, change) => self.upstream(backend, change),
            Message::Status(reply) => {
                let _ = reply.send(self.status());
            }
            Message::Reconfigure(config, reply) => {
                let _ = reply.send(self.reconfigure(config));
            }
            Message::Shutdown(reply) => {
                self.feed(Event::Shutdown { now });
                self.dirty
                    .extend(self.backends.iter().map(|backend| backend.link.id));
                self.persist();
                let _ = reply.send(());
                return false;
            }
        }
        true
    }

    fn admit(&mut self, arrival: Arrival) {
        let Arrival {
            ticket,
            demand,
            client,
            class,
            dialect,
            measured,
            control,
            reply,
        } = arrival;
        self.metrics.count(
            "llama_gate_prompt_tokens_total",
            labels(&[("measurement", measured)]),
            demand.work,
        );
        debug!(%ticket, %client, %class, dialect, prompt = demand.prompt, generation = demand.generation, measured, "request queued");
        self.clients.entry(client.clone()).or_default().id = Some(demand.client);
        self.waiting.insert(
            ticket,
            Waiter {
                reply,
                since: Instant::now(),
                client,
                client_id: demand.client,
                class,
                dialect,
                need: demand.need(),
                control,
            },
        );
        let now = self.clock.now();
        self.feed(Event::Arrived {
            ticket,
            demand,
            now,
        });
    }

    fn bounce(&mut self, ticket: TicketId, reply: oneshot::Sender<Grant>) {
        let Some(runner) = self.running.remove(&ticket) else {
            // Withdrawn meanwhile: nobody is waiting for the new grant.
            return;
        };
        self.metrics
            .count("llama_gate_bounces_total", String::new(), 1);
        if let Some(backend) = self.locate(ticket)
            && let Some(state) = self.backends.get_mut(usize::from(backend.0))
        {
            // The scheduler takes it offline; the watcher brings it back.
            state.up = false;
            let name = state.link.name.clone();
            warn!(%ticket, backend = name, "the server refused the connection; placing the request again");
            self.emit(json!({"type": "bounce", "ticket": ticket, "backend": name}));
        }
        self.waiting.insert(ticket, runner.waiter(reply));
        let now = self.clock.now();
        self.feed(Event::Bounced { ticket, now });
    }

    /// The backend a running ticket is on, as the status reports it.
    fn locate(&self, ticket: TicketId) -> Option<BackendId> {
        let status = self.scheduler.status(self.clock.now());
        status.backends.iter().find_map(|backend| {
            backend
                .slots
                .iter()
                .any(|slot| slot.ticket == Some(ticket))
                .then_some(backend.backend)
        })
    }

    fn upstream(&mut self, backend: BackendId, change: Upstream) {
        let index = usize::from(backend.0);
        let now = self.clock.now();
        let Some(name) = self
            .backends
            .get(index)
            .map(|state| state.link.name.clone())
        else {
            return;
        };
        match change {
            Upstream::Down => {
                if !self.backends[index].up {
                    return;
                }
                warn!(
                    backend = name,
                    url = self.backends[index].link.server.base(),
                    "server unavailable"
                );
                self.backends[index].up = false;
                self.emit(json!({"type": "backend", "backend": name, "state": "down"}));
                self.feed(Event::BackendDown { backend, now });
            }
            Upstream::Alive => {
                let state = &self.backends[index];
                if !state.known || state.up {
                    return;
                }
                let Some(shape) = self.scheduler.shape(backend) else {
                    return;
                };
                self.backends[index].up = true;
                info!(backend = name, "server back");
                self.emit(json!({"type": "backend", "backend": name, "state": "up"}));
                self.feed(Event::BackendUp {
                    backend,
                    shape,
                    restarted: false,
                    snapshots: None,
                    now,
                });
            }
            Upstream::Ready {
                info,
                capabilities,
                slots,
                restarted,
            } => {
                let config = self.config.clone();
                let state = &mut self.backends[index];
                let namespace = namespace(&info);
                let layout = layout(&config, &state.config, &info);
                let swap = config.eviction.mode == EvictionMode::Swap && state.store.enabled();
                let first = !state.known;
                if config.eviction.mode == EvictionMode::Swap && !swap && first {
                    warn!(
                        backend = name,
                        "eviction.mode is swap but this backend has no accessible snapshot directory; its idle contexts will be discarded"
                    );
                }
                let new_model = state.namespace.as_deref() != Some(namespace.as_str());
                let reshaped = state.info.as_ref() != Some(&info);
                let snapshots = (first || new_model).then(|| {
                    if swap {
                        state.store.load(&namespace)
                    } else {
                        SnapshotIndex::default()
                    }
                });
                let reset = !first && (restarted || reshaped);
                let was_up = state.up;
                let model = state.link.model;
                let mut aliases = Vec::new();
                for reported in [info.model.as_deref(), info.alias.as_deref()]
                    .into_iter()
                    .flatten()
                {
                    aliases.extend(reported_names(reported));
                }
                let shape = Shape {
                    model,
                    layout,
                    slots: info.slots,
                    swap,
                };
                if first || !was_up || reset {
                    info!(
                        backend = name,
                        url = state.link.server.base(),
                        slots = info.slots,
                        ?layout,
                        swap,
                        model = ?info.model,
                        "server ready"
                    );
                }
                state.namespace = Some(namespace);
                state.info = Some(info);
                state.capabilities = capabilities;
                state.known = true;
                state.up = true;
                self.models.register(&self.models.name(model), &aliases);
                if reset {
                    self.metrics.count(
                        "llama_gate_upstream_resets_total",
                        labels(&[("backend", &name)]),
                        1,
                    );
                    info!(
                        backend = name,
                        "server restarted or reconfigured; its slot state starts over"
                    );
                }
                let label = if reset { "restarted" } else { "up" };
                self.emit(json!({"type": "backend", "backend": name, "state": label}));
                self.feed(Event::BackendUp {
                    backend,
                    shape,
                    restarted: reset,
                    snapshots,
                    now,
                });
                if first || reset {
                    self.observe(backend, &slots);
                }
            }
        }
    }

    fn observe(&mut self, backend: BackendId, slots: &[SlotView]) {
        let observations = slots
            .iter()
            .map(|view| Observation {
                slot: SlotId {
                    backend,
                    index: view.id,
                },
                busy: view.busy,
                tokens: view.tokens,
            })
            .collect();
        let now = self.clock.now();
        self.feed(Event::Observed {
            slots: observations,
            now,
        });
    }

    /// Applies an event and carries out the resulting commands, including
    /// any follow-up events they produce.
    fn feed(&mut self, event: Event) {
        let mut events = VecDeque::from([event]);
        while let Some(event) = events.pop_front() {
            for command in self.scheduler.handle(event) {
                if let Some(follow) = self.execute(command) {
                    events.push_back(follow);
                }
            }
        }
    }

    fn execute(&mut self, command: Command) -> Option<Event> {
        let now = self.clock.now();
        match command {
            Command::Dispatch {
                ticket,
                slot,
                generation,
                exclusive,
                reuse,
            } => {
                let Some(waiter) = self.waiting.remove(&ticket) else {
                    return Some(Event::Withdrawn { ticket, now });
                };
                let waited = waiter.since.elapsed();
                self.metrics
                    .observe(Timing::QueueTime, waited.as_secs_f64());
                self.metrics.count(
                    "llama_gate_requests_total",
                    labels(&[("class", &waiter.class), ("outcome", "dispatched")]),
                    1,
                );
                let backend = self.name(slot.backend);
                info!(
                    %ticket,
                    client = %waiter.client,
                    class = %waiter.class,
                    dialect = waiter.dialect,
                    backend,
                    slot = slot.index,
                    need = waiter.need,
                    generation,
                    reuse,
                    waited_ms = waited.as_millis() as u64,
                    "dispatch"
                );
                self.emit(json!({
                    "type": "dispatch", "ticket": ticket, "client": waiter.client,
                    "class": waiter.class, "model": self.model_of(slot.backend),
                    "backend": backend, "slot": slot.index,
                    "need": waiter.need, "generation": generation, "reuse": reuse,
                    "waited_ms": waited.as_millis() as u64,
                }));
                waiter.control.dispatched();
                let grant = Grant::Run {
                    slot,
                    generation,
                    exclusive,
                    reuse,
                };
                if waiter.reply.send(grant).is_err() {
                    // The client left between being scheduled and being told.
                    return Some(Event::Withdrawn { ticket, now });
                }
                self.running.insert(
                    ticket,
                    Runner {
                        client: waiter.client,
                        client_id: waiter.client_id,
                        class: waiter.class,
                        dialect: waiter.dialect,
                        need: waiter.need,
                        control: waiter.control,
                    },
                );
            }
            Command::Reject { ticket, rejection } => {
                if let Some(waiter) = self.waiting.remove(&ticket) {
                    let reason = rejection_reason(&rejection);
                    self.metrics.count(
                        "llama_gate_requests_total",
                        labels(&[("class", &waiter.class), ("outcome", reason)]),
                        1,
                    );
                    warn!(%ticket, client = %waiter.client, class = %waiter.class, reason, "request refused");
                    self.emit(json!({
                        "type": "reject", "ticket": ticket, "client": waiter.client,
                        "class": waiter.class, "reason": reason,
                        "status": rejection_status(&rejection),
                    }));
                    let _ = waiter.reply.send(Grant::Refused(rejection));
                }
            }
            Command::Evict { slot, save, op } => {
                let (server, store, sender, clock) = self.tools(slot.backend)?;
                let backend = self.name(slot.backend);
                tokio::spawn(async move {
                    let (outcome, detail) =
                        evict(&server, &store, &sender, &backend, slot, save, "evict").await;
                    let _ = sender.send(Message::Outcome {
                        event: Event::Evicted {
                            slot,
                            op,
                            outcome,
                            now: clock.now(),
                        },
                        detail: Some(detail),
                    });
                });
            }
            Command::Save { slot, snapshot, op } => {
                let (server, store, sender, clock) = self.tools(slot.backend)?;
                let backend = self.name(slot.backend);
                tokio::spawn(async move {
                    let started = Instant::now();
                    let outcome = match server
                        .save(slot.index, &SnapshotStore::filename(snapshot))
                        .await
                    {
                        Ok(report) if report.tokens == Some(0) => {
                            cleared_behind(&backend);
                            tokio::task::spawn_blocking(move || store.delete(snapshot));
                            SaveOutcome::Failed
                        }
                        Ok(report) => SaveOutcome::Saved {
                            bytes: report.bytes.max(1),
                            tokens: report.tokens.unwrap_or(0),
                        },
                        Err(ServerError::Unsupported(reason)) => {
                            let _ = sender.send(Message::SwapUnsupported(slot.backend, reason));
                            SaveOutcome::Failed
                        }
                        Err(error) => {
                            warn!(backend, slot = slot.index, %error, "saving the shared context failed; its unshared part will be lost");
                            SaveOutcome::Failed
                        }
                    };
                    let millis = started.elapsed().as_millis() as u64;
                    let detail = match outcome {
                        SaveOutcome::Saved { bytes, tokens } => {
                            info!(
                                backend,
                                slot = slot.index,
                                snapshot = snapshot.0,
                                tokens,
                                bytes,
                                millis,
                                "saved a context before reusing its shared part"
                            );
                            json!({"type": "save", "backend": backend, "slot": slot.index, "outcome": "saved", "tokens": tokens, "bytes": bytes, "ms": millis})
                        }
                        SaveOutcome::Failed => {
                            json!({"type": "save", "backend": backend, "slot": slot.index, "outcome": "failed", "ms": millis})
                        }
                    };
                    let _ = sender.send(Message::Outcome {
                        event: Event::Saved {
                            slot,
                            op,
                            outcome,
                            now: clock.now(),
                        },
                        detail: Some(detail),
                    });
                });
            }
            Command::Restore { slot, snapshot, op } => {
                let (server, _, sender, clock) = self.tools(slot.backend)?;
                let backend = self.name(slot.backend);
                tokio::spawn(async move {
                    let started = Instant::now();
                    let result = server
                        .restore(slot.index, &SnapshotStore::filename(snapshot))
                        .await;
                    let millis = started.elapsed().as_millis() as u64;
                    let (outcome, detail) = match result {
                        Ok(report) if report.tokens == Some(0) => {
                            warn!(
                                backend,
                                slot = slot.index,
                                snapshot = snapshot.0,
                                "the snapshot held nothing; it is forgotten"
                            );
                            (
                                RestoreOutcome::Failed,
                                json!({"type": "restore", "backend": backend, "slot": slot.index, "outcome": "failed", "ms": millis}),
                            )
                        }
                        Ok(report) => {
                            let tokens = report.tokens.unwrap_or(0);
                            info!(
                                backend,
                                slot = slot.index,
                                snapshot = snapshot.0,
                                tokens,
                                millis,
                                "restored context"
                            );
                            (
                                RestoreOutcome::Restored { tokens },
                                json!({"type": "restore", "backend": backend, "slot": slot.index, "outcome": "restored", "tokens": tokens, "ms": millis}),
                            )
                        }
                        Err(error) => {
                            warn!(backend, slot = slot.index, snapshot = snapshot.0, %error, "restore failed; the slot will be cleared and the context recomputed");
                            (
                                RestoreOutcome::Failed,
                                json!({"type": "restore", "backend": backend, "slot": slot.index, "outcome": "failed", "ms": millis}),
                            )
                        }
                    };
                    let _ = sender.send(Message::Outcome {
                        event: Event::Restored {
                            slot,
                            op,
                            outcome,
                            now: clock.now(),
                        },
                        detail: Some(detail),
                    });
                });
            }
            Command::Preempt {
                ticket,
                slot,
                save,
                op,
            } => return self.preempt(ticket, slot, save, op),
            Command::Forget { backend, snapshot } => {
                if let Some(state) = self.backends.get(usize::from(backend.0)) {
                    let store = state.store.clone();
                    tokio::task::spawn_blocking(move || store.delete(snapshot));
                }
            }
            Command::IndexChanged { backend } => {
                self.dirty.insert(backend);
            }
            Command::Notice(Notice::PartialReuseIneffective { backend }) => {
                let message = format!(
                    "backend {}: its model did not reuse contexts that share only part of a prompt (it cannot rewind to them), so they are no longer offered there",
                    self.name(backend)
                );
                warn!("{message}");
                self.emit(json!({"type": "notice", "message": message}));
            }
        }
        None
    }

    /// Pauses a running request: it goes back in line with a fresh grant
    /// channel, and a task stops it on the server and saves its progress.
    fn preempt(
        &mut self,
        ticket: TicketId,
        slot: SlotId,
        save: Option<SnapshotId>,
        op: gate_core::OpId,
    ) -> Option<Event> {
        let now = self.clock.now();
        let backend = self.name(slot.backend);
        if let Some(runner) = self.running.get(&ticket) {
            let (reply, receiver) = oneshot::channel();
            if !runner.control.pause(receiver) {
                self.metrics.count(
                    "llama_gate_preemptions_total",
                    labels(&[("outcome", "declined")]),
                    1,
                );
                self.emit(json!({"type": "preempt", "ticket": ticket, "client": runner.client, "backend": backend, "slot": slot.index, "outcome": "declined"}));
                return Some(Event::Preempted {
                    slot,
                    op,
                    outcome: PreemptOutcome::Declined,
                    now,
                });
            }
            if let Some(runner) = self.running.remove(&ticket) {
                info!(%ticket, client = %runner.client, backend, slot = slot.index, "paused to make room for more urgent work; it resumes where it stopped");
                self.emit(json!({"type": "preempt", "ticket": ticket, "client": runner.client, "backend": backend, "slot": slot.index, "outcome": "paused"}));
                self.metrics.count(
                    "llama_gate_preemptions_total",
                    labels(&[("outcome", "paused")]),
                    1,
                );
                self.waiting.insert(ticket, runner.waiter(reply));
            }
        }
        let (server, store, sender, clock) = self.tools(slot.backend)?;
        tokio::spawn(async move {
            // The handler drops its connection; the server then stops the
            // request and runs the save, which it defers until it has.
            let (outcome, detail) =
                evict(&server, &store, &sender, &backend, slot, save, "preempt").await;
            let _ = sender.send(Message::Outcome {
                event: Event::Preempted {
                    slot,
                    op,
                    outcome: PreemptOutcome::Stopped(outcome),
                    now: clock.now(),
                },
                detail: Some(detail),
            });
        });
        None
    }

    fn tools(
        &self,
        backend: BackendId,
    ) -> Option<(Server, SnapshotStore, mpsc::UnboundedSender<Message>, Clock)> {
        let state = self.backends.get(usize::from(backend.0))?;
        Some((
            state.link.server.clone(),
            state.store.clone(),
            self.sender.clone(),
            self.clock,
        ))
    }

    fn model_of(&self, backend: BackendId) -> String {
        self.backends
            .get(usize::from(backend.0))
            .map_or_else(String::new, |state| self.models.name(state.link.model))
    }

    fn name(&self, backend: BackendId) -> String {
        self.backends
            .get(usize::from(backend.0))
            .map_or_else(|| backend.to_string(), |state| state.link.name.clone())
    }

    fn count_outcome(&self, event: &Event) {
        let (name, outcome) = match event {
            Event::Evicted { outcome, .. } => (
                "llama_gate_evictions_total",
                match outcome {
                    EvictOutcome::Saved { .. } => "saved",
                    EvictOutcome::Erased => "erased",
                    EvictOutcome::Failed => "failed",
                },
            ),
            Event::Restored { outcome, .. } => (
                "llama_gate_restores_total",
                match outcome {
                    RestoreOutcome::Restored { .. } => "restored",
                    RestoreOutcome::Failed => "failed",
                },
            ),
            _ => return,
        };
        self.metrics.count(name, labels(&[("outcome", outcome)]), 1);
    }

    /// Records a finished (or abandoned) request: metrics, client totals,
    /// the access log, the recent list, and the event stream.
    fn record(&mut self, mut report: Report) {
        report.at_ms = self.clock.now();
        let metrics = &self.metrics;
        metrics.count(
            "llama_gate_requests_total",
            labels(&[("class", &report.class), ("outcome", report.outcome)]),
            1,
        );
        if let Some(generated) = report.generated {
            metrics.count(
                "llama_gate_generated_tokens_total",
                labels(&[("class", &report.class)]),
                generated,
            );
        }
        if let Some(cached) = report.cached {
            metrics.count(
                "llama_gate_cached_prompt_tokens_total",
                String::new(),
                cached,
            );
        }
        if let Some(ttft) = report.ttft_ms {
            metrics.observe(Timing::TimeToFirstToken, ttft as f64 / 1000.0);
        }
        metrics.observe(Timing::Duration, report.total_ms as f64 / 1000.0);
        metrics.observe(Timing::Measure, report.measure_ms as f64 / 1000.0);
        let stats = self.clients.entry(report.client.clone()).or_default();
        stats.requests += 1;
        stats.prompt_tokens += report.prompt;
        stats.cached_tokens += report.cached.unwrap_or(0);
        stats.generated_tokens += report.generated.unwrap_or(0);
        if self.config.observe.access_log {
            info!(
                ticket = %report.ticket,
                client = %report.client,
                class = %report.class,
                model = report.model,
                dialect = report.dialect,
                backend = report.backend.as_deref().unwrap_or("-"),
                slot = report.slot,
                outcome = report.outcome,
                status = report.status,
                prompt = report.prompt,
                cached = report.cached,
                generated = report.generated,
                measure_ms = report.measure_ms,
                queue_ms = report.queue_ms,
                ttft_ms = report.ttft_ms,
                total_ms = report.total_ms,
                tokens_per_second = report.tokens_per_second.map(|rate| (rate * 10.0).round() / 10.0),
                preempted = report.preempted,
                bounced = report.bounced,
                "done"
            );
        }
        if let Ok(mut event) = serde_json::to_value(&report) {
            event["type"] = json!("finish");
            self.emit(event);
        }
        self.recent.push_front(report);
        self.recent.truncate(self.config.observe.recent);
    }

    fn emit(&self, mut event: Value) {
        if self.events.receiver_count() == 0 {
            return;
        }
        if event.get("at_ms").is_none() {
            event["at_ms"] = json!(self.clock.now());
        }
        let _ = self.events.send(Arc::from(event.to_string()));
    }

    fn persist(&mut self) {
        for backend in std::mem::take(&mut self.dirty) {
            let Some(state) = self.backends.get(usize::from(backend.0)) else {
                continue;
            };
            let (Some(index), Some(namespace)) =
                (self.scheduler.snapshots(backend), state.namespace.clone())
            else {
                continue;
            };
            if !state.store.enabled() {
                continue;
            }
            let index = index.clone();
            let store = state.store.clone();
            let name = state.link.name.clone();
            tokio::task::spawn_blocking(move || {
                if let Err(error) = store.persist(&namespace, &index) {
                    warn!(backend = name, %error, "could not persist the snapshot index");
                }
            });
        }
    }

    fn publish(&self) {
        let metrics = &self.metrics;
        metrics.set(
            "llama_gate_requests_waiting",
            String::new(),
            self.scheduler.waiting() as f64,
        );
        metrics.set(
            "llama_gate_requests_running",
            String::new(),
            self.running.len() as f64,
        );
        let status = self.scheduler.status(self.clock.now());
        for state in &self.backends {
            let name = labels(&[("backend", &state.link.name)]);
            metrics.set(
                "llama_gate_upstream_up",
                name.clone(),
                if state.up { 1.0 } else { 0.0 },
            );
            let Some(backend) = status
                .backends
                .iter()
                .find(|backend| backend.backend == state.link.id)
            else {
                continue;
            };
            metrics.set(
                "llama_gate_kv_capacity_tokens",
                name.clone(),
                backend.capacity as f64,
            );
            metrics.set(
                "llama_gate_kv_reserved_tokens",
                name.clone(),
                backend.usage as f64,
            );
            metrics.set(
                "llama_gate_kv_usage_ratio",
                name.clone(),
                if backend.capacity > 0 {
                    backend.usage as f64 / backend.capacity as f64
                } else {
                    0.0
                },
            );
            metrics.set(
                "llama_gate_snapshots",
                name.clone(),
                backend.snapshots as f64,
            );
            metrics.set(
                "llama_gate_snapshot_bytes",
                name,
                backend.snapshot_bytes as f64,
            );
        }
    }

    fn status(&self) -> Value {
        let now = self.clock.now();
        let status = self.scheduler.status(now);
        let names: HashMap<TicketId, (&str, &str)> = self
            .waiting
            .iter()
            .map(|(ticket, waiter)| (*ticket, (&*waiter.client, &*waiter.class)))
            .chain(
                self.running
                    .iter()
                    .map(|(ticket, runner)| (*ticket, (&*runner.client, &*runner.class))),
            )
            .collect();
        let models = self.models.list();
        let backends: Vec<Value> = self
            .backends
            .iter()
            .map(|state| {
                let core = status
                    .backends
                    .iter()
                    .find(|backend| backend.backend == state.link.id);
                let slots: Vec<Value> = core
                    .map(|backend| {
                        backend
                            .slots
                            .iter()
                            .map(|slot| {
                                let (client, class) = slot
                                    .ticket
                                    .and_then(|ticket| names.get(&ticket).copied())
                                    .unzip();
                                json!({
                                    "index": slot.slot.index,
                                    "state": slot.state,
                                    "tokens": slot.tokens,
                                    "ticket": slot.ticket,
                                    "client": client,
                                    "class": class,
                                    "running_ms": slot.running_for,
                                    "started": slot.started,
                                    "pausable": slot.pausable,
                                    "reuse": slot.reuse,
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                json!({
                    "id": state.link.id,
                    "name": state.link.name,
                    "url": state.link.server.base(),
                    "model": self.models.name(state.link.model),
                    "up": state.up,
                    "layout": core.map(|backend| backend.layout),
                    "capacity": core.map(|backend| backend.capacity),
                    "usage": core.map(|backend| backend.usage),
                    "stray": core.map(|backend| backend.stray),
                    "swap": core.is_some_and(|backend| backend.swap),
                    "partial_reuse": core.is_some_and(|backend| backend.partial_reuse),
                    "incremental": self.config.admission.incremental,
                    "build": state.info.as_ref().and_then(|info| info.build.clone()),
                    "reported_model": state.info.as_ref().and_then(|info| info.model.clone()),
                    "capabilities": state.capabilities,
                    "slots": slots,
                    "snapshots": {
                        "count": core.map_or(0, |backend| backend.snapshots),
                        "bytes": core.map_or(0, |backend| backend.snapshot_bytes),
                    },
                })
            })
            .collect();
        let waiting: Vec<Value> = status
            .waiting
            .iter()
            .map(|waiting| {
                json!({
                    "ticket": waiting.ticket,
                    "client": names.get(&waiting.ticket).map(|(client, _)| *client),
                    "class": waiting.class,
                    "model": self.models.name(waiting.model),
                    "need": waiting.need,
                    "waited_ms": waiting.waited,
                    "bumped": waiting.bumped,
                    "paused": waiting.paused,
                })
            })
            .collect();
        let clients: Vec<Value> = self
            .clients
            .iter()
            .map(|(name, stats)| {
                let waiting = self
                    .waiting
                    .values()
                    .filter(|waiter| &waiter.client == name)
                    .count();
                let running = self
                    .running
                    .values()
                    .filter(|runner| &runner.client == name)
                    .count();
                json!({
                    "name": name,
                    "service": stats.id.and_then(|id| self.scheduler.service(id)),
                    "requests": stats.requests,
                    "prompt_tokens": stats.prompt_tokens,
                    "cached_tokens": stats.cached_tokens,
                    "generated_tokens": stats.generated_tokens,
                    "waiting": waiting,
                    "running": running,
                })
            })
            .collect();
        json!({
            "version": env!("CARGO_PKG_VERSION"),
            "ready": self.backends.iter().any(|state| state.up),
            "mode": self.config.server.mode,
            "uptime_ms": self.started.elapsed().as_millis() as u64,
            "models": models.iter().map(|model| json!({
                "id": model.id,
                "name": model.name,
                "aliases": model.aliases,
                "backends": self.backends.iter().filter(|state| state.link.model == model.id).map(|state| state.link.id).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
            "backends": backends,
            "waiting": waiting,
            "clients": clients,
            "recent": self.recent.iter().take(50).collect::<Vec<_>>(),
        })
    }

    /// Applies a reloaded configuration: classes, queue and eviction
    /// policy, admission settings. Backends and the listen address need a
    /// restart, which the returned notes say.
    fn reconfigure(&mut self, config: Arc<Config>) -> Result<Vec<String>, String> {
        config.validate()?;
        let mut notes = Vec::new();
        let before: Vec<(String, String)> = self
            .config
            .effective_backends()
            .into_iter()
            .map(|backend| (backend.name, backend.url))
            .collect();
        let after: Vec<(String, String)> = config
            .effective_backends()
            .into_iter()
            .map(|backend| (backend.name, backend.url))
            .collect();
        if before != after {
            notes.push("backends changed: restart the gate to apply that".into());
        }
        if self.config.server.listen != config.server.listen {
            notes.push("server.listen changed: restart the gate to apply that".into());
        }
        let names = config.class_names();
        let default = names
            .iter()
            .position(|name| *name == config.clients.default_class)
            .unwrap_or(0);
        let old = self.class_names.clone();
        let class_of = |id: ClassId| {
            let index = old
                .get(usize::from(id.0))
                .and_then(|name| names.iter().position(|new| new == name))
                .unwrap_or(default);
            ClassId(index as u16)
        };
        self.scheduler
            .reconfigure(scheduler_config(&config), class_of);
        self.class_names = config.class_names();
        self.config = config;
        info!(notes = ?notes, "configuration reloaded");
        Ok(notes)
    }
}

/// Saves (when asked) and erases a slot, for an eviction or a pause.
async fn evict(
    server: &Server,
    store: &SnapshotStore,
    sender: &mpsc::UnboundedSender<Message>,
    backend: &str,
    slot: SlotId,
    save: Option<SnapshotId>,
    kind: &str,
) -> (EvictOutcome, Value) {
    let started = Instant::now();
    let mut saved = None;
    if let Some(snapshot) = save {
        match server
            .save(slot.index, &SnapshotStore::filename(snapshot))
            .await
        {
            Ok(report) if report.tokens == Some(0) => {
                cleared_behind(backend);
                let store = store.clone();
                tokio::task::spawn_blocking(move || store.delete(snapshot));
            }
            Ok(report) => saved = Some((snapshot, report)),
            Err(ServerError::Unsupported(reason)) => {
                let _ = sender.send(Message::SwapUnsupported(slot.backend, reason));
            }
            Err(error) => {
                warn!(backend, slot = slot.index, %error, "saving the context failed; it will be discarded")
            }
        }
    }
    let millis = || started.elapsed().as_millis() as u64;
    match server.erase(slot.index).await {
        Ok(_) => match saved {
            Some((snapshot, report)) => {
                info!(
                    backend,
                    slot = slot.index,
                    snapshot = snapshot.0,
                    tokens = report.tokens.unwrap_or(0),
                    bytes = report.bytes,
                    millis = millis(),
                    "saved and cleared context"
                );
                (
                    EvictOutcome::Saved {
                        bytes: report.bytes.max(1),
                        tokens: report.tokens.unwrap_or(0),
                    },
                    json!({"type": kind, "backend": backend, "slot": slot.index, "outcome": "saved", "tokens": report.tokens, "bytes": report.bytes, "ms": millis()}),
                )
            }
            None => {
                debug!(backend, slot = slot.index, "cleared context");
                (
                    EvictOutcome::Erased,
                    json!({"type": kind, "backend": backend, "slot": slot.index, "outcome": "erased", "ms": millis()}),
                )
            }
        },
        Err(error) => {
            warn!(backend, slot = slot.index, %error, "clearing the slot failed; it will be retried");
            if let Some((snapshot, _)) = saved {
                let store = store.clone();
                tokio::task::spawn_blocking(move || store.delete(snapshot));
            }
            (
                EvictOutcome::Failed,
                json!({"type": kind, "backend": backend, "slot": slot.index, "outcome": "failed", "ms": millis()}),
            )
        }
    }
}

/// A save found empty a slot the gate believed held a context: the server
/// clears idle slots itself (llama-server's `--cache-idle-slots`, on by
/// default with its prompt cache). Said once; it recurs with every save.
fn cleared_behind(backend: &str) {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
        warn!(
            backend,
            "the server clears idle slots itself, so the gate cannot save them; start llama-server with --no-cache-idle-slots"
        );
    }
}

/// Content addresses and snapshots are scoped to the model they belong to.
pub fn namespace(info: &ServerInfo) -> String {
    info.model.clone().unwrap_or_else(|| "unknown-model".into())
}

fn layout(config: &Config, backend: &BackendConfig, info: &ServerInfo) -> Layout {
    let capacity = backend.capacity.unwrap_or(config.kv.capacity);
    let capacity = if capacity > 0 {
        capacity
    } else {
        info.slot_context
    };
    let margin = backend.margin.unwrap_or(config.kv.margin);
    let usable = capacity.saturating_sub(margin).max(1);
    match backend.layout.unwrap_or(config.kv.layout) {
        LayoutKind::Unified => Layout::Unified { capacity: usable },
        LayoutKind::Split => Layout::Split { per_slot: usable },
    }
}

/// The HTTP status a refusal is answered with.
fn rejection_status(rejection: &Rejection) -> u16 {
    match rejection {
        Rejection::TooLarge { .. } => 400,
        Rejection::Timeout { .. } | Rejection::QueueFull { .. } => 429,
        Rejection::UnknownClass => 500,
        Rejection::Unavailable { .. } | Rejection::ShuttingDown => 503,
    }
}

fn rejection_reason(rejection: &Rejection) -> &'static str {
    match rejection {
        Rejection::TooLarge { .. } => "rejected_too_large",
        Rejection::Timeout { .. } => "rejected_timeout",
        Rejection::QueueFull { .. } => "rejected_queue_full",
        Rejection::UnknownClass => "rejected_unknown_class",
        Rejection::Unavailable { .. } => "rejected_unavailable",
        Rejection::ShuttingDown => "rejected_shutting_down",
    }
}

pub fn scheduler_config(config: &Config) -> SchedulerConfig {
    let classes = config
        .classes
        .iter()
        .map(|(name, class)| Class {
            name: name.clone(),
            priority: class.priority,
            weight: class.weight,
            max_wait: class.max_wait.map(|wait| wait.millis()),
            max_queued_per_client: class.max_queued_per_client,
            preemptible: class.preemptible,
        })
        .collect();
    SchedulerConfig {
        classes,
        backfill: config.queue.backfill.enabled.then(|| Backfill {
            max_tokens: config.queue.backfill.max_tokens,
            window: config.queue.backfill.window.millis(),
        }),
        eviction: EvictionPolicy {
            partial: config.eviction.partial_reuse,
            min_snapshot_tokens: config.eviction.min_snapshot_tokens,
            min_restore_gain: config.eviction.min_restore_gain,
            snapshot_budget: config.eviction.budget.0,
            retry_after: config.eviction.retry.millis(),
        },
        max_preemptions: config.queue.max_preemptions,
        unavailable_after: (config.server.upstream_wait.millis() > 0)
            .then(|| config.server.upstream_wait.millis()),
    }
}

/// Watches one backend's health and shape, reporting readiness, outages,
/// and restarts to the engine. A restart is recognized by the server's task
/// ids going backwards or its shape changing, not by a single missed probe.
pub async fn watch_upstream(
    backend: BackendId,
    server: Server,
    handle: Handle,
    interval: Duration,
    health: watch::Sender<Health>,
    measurer: watch::Sender<Measurer>,
    incremental: bool,
) {
    let mut ready_before = false;
    let mut last_info: Option<ServerInfo> = None;
    let mut last_task: Option<i64> = None;
    let mut down_polls = 0u32;
    loop {
        let mut state = server.health().await;
        if state == Health::Ready {
            match server.info().await {
                Ok(info) => {
                    let slots = server.slots().await.unwrap_or_default();
                    let task = slots.iter().filter_map(|view| view.task).max();
                    // Task ids only grow during a server's life; after a
                    // restart every slot starts over.
                    let went_backwards =
                        matches!((last_task, task), (Some(before), Some(now)) if now < before);
                    let blind_restart = slots.is_empty() && down_polls >= 2;
                    let changed = last_info.as_ref() != Some(&info);
                    if !ready_before || changed || went_backwards || blind_restart || down_polls > 0
                    {
                        let capabilities = server.capabilities().await;
                        let boundaries = if incremental
                            && capabilities.tokenize
                            && capabilities.apply_template
                        {
                            Boundaries::learn(&server)
                                .await
                                .ok()
                                .flatten()
                                .map(Arc::new)
                        } else {
                            None
                        };
                        let namespace = namespace(&info);
                        measurer.send_modify(|measurer| {
                            measurer.set_capabilities(capabilities);
                            measurer.set_namespace(namespace);
                            measurer.set_boundaries(boundaries);
                        });
                        handle.upstream(
                            backend,
                            Upstream::Ready {
                                info: info.clone(),
                                capabilities,
                                slots,
                                restarted: ready_before
                                    && (changed || went_backwards || blind_restart),
                            },
                        );
                    } else {
                        handle.upstream(backend, Upstream::Alive);
                    }
                    ready_before = true;
                    last_info = Some(info);
                    if task.is_some() {
                        last_task = task;
                    }
                    down_polls = 0;
                }
                Err(_) => state = Health::Down,
            }
        }
        if state != Health::Ready {
            down_polls = down_polls.saturating_add(1);
            if down_polls == 1 && ready_before {
                handle.upstream(backend, Upstream::Down);
            }
        }
        health.send_replace(state);
        tokio::time::sleep(interval).await;
    }
}

impl Handle {
    fn upstream(&self, backend: BackendId, change: Upstream) {
        let _ = self.sender.send(Message::Upstream(backend, change));
    }
}
