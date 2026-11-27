//! The engine actor: owns the scheduler, performs its commands, and turns
//! their outcomes back into events.
//!
//! Every change to scheduling state happens on one task, so the scheduler
//! needs no lock and its decisions are serialized exactly as in the core's
//! simulation tests. Slow work (saving, restoring, erasing slots, stopping a
//! paused request) runs in spawned tasks that report back through the same
//! queue. Each backend has a watcher task reporting its health and shape.
//!
//! - [`execute`]: commands become effects (slot operations, grants, refusals).
//! - [`watcher`]: one watcher per backend: health, shape, restarts, snapshot index.
//! - [`report`]: finished requests become metrics, totals, and events.
//! - [`view`]: the status document and gauges.
//! - [`reload`]: applying a reloaded configuration.
use crate::config::{BackendConfig, Config, LayoutKind, QueueOrderKind, VictimKind};
use crate::control::Control;
use crate::metrics::{Metrics, Timing, labels};
use crate::models::{Models, reported_names};
use crate::snapshots::{Loaded, Persister, SnapshotStore, StartOver};
use bytes::Bytes;
use gate_core::{
    BackendId, Backfill, Class, ClassId, ClientId, Command, Demand, Event, EvictOutcome,
    EvictionOrder, EvictionPolicy, Fifo, Largest, Layout, LeastRecentlyUsed, Millis, ModelId,
    Notice, Observation, PreemptOutcome, QueueOrder, Rejection, RestoreOutcome, SaveOutcome,
    Scheduler, SchedulerConfig, Shape, SlotId, SnapshotId, TicketId, Tokens, Vtc,
};
use gate_llama::usage::Usage;
use gate_llama::{
    Boundaries, Capabilities, Health, Measurer, Server, ServerError, ServerInfo, SlotView,
};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc, oneshot, watch};
use tracing::{debug, info, warn};
use watcher::describe_loaded;

mod execute;
mod reload;
mod report;
mod view;
mod watcher;

pub use reload::{restart_notes, scheduler_config};
pub use watcher::{namespace, watch_upstream};

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
    pub request_id: String,
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
    Ready(Box<Ready>),
    /// Still ready and unchanged.
    Alive,
    Down,
}

/// A backend found ready, with what the gate needs to plan for it.
#[derive(Debug)]
pub struct Ready {
    pub info: ServerInfo,
    pub capabilities: Capabilities,
    pub slots: Vec<SlotView>,
    pub restarted: bool,
    /// The snapshot index, loaded when the model identity changed.
    pub loaded: Option<Loaded>,
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
    pub vitals: Arc<Vitals>,
    status: watch::Receiver<Bytes>,
}

/// Whether the engine is alive: its loop beats at least every scheduling
/// tick, so a stopped or wedged engine shows within seconds.
#[derive(Debug, Default)]
pub struct Vitals {
    /// When the engine last went round its loop (gate milliseconds).
    beat: std::sync::atomic::AtomicU64,
    stopped: std::sync::atomic::AtomicBool,
    /// The gate began shutting down: it serves what it has, takes nothing new.
    draining: std::sync::atomic::AtomicBool,
    drain_started: tokio::sync::Notify,
}

impl Vitals {
    /// Why the gate cannot serve, if it cannot.
    pub fn trouble(&self, now: Millis) -> Option<&'static str> {
        use std::sync::atomic::Ordering::Acquire;
        if self.draining.load(Acquire) {
            Some("shutting down")
        } else if self.stopped.load(Acquire) {
            Some("the scheduler stopped")
        } else if now.saturating_sub(self.beat.load(Acquire)) > 10_000 {
            Some("the scheduler is not responding")
        } else {
            None
        }
    }

    pub fn drain(&self) {
        self.draining
            .store(true, std::sync::atomic::Ordering::Release);
        self.drain_started.notify_waiters();
    }

    /// Resolves once the gate starts shutting down.
    pub async fn drained(&self) {
        loop {
            let started = self.drain_started.notified();
            if self.draining() {
                return;
            }
            started.await;
        }
    }

    pub fn draining(&self) -> bool {
        self.draining.load(std::sync::atomic::Ordering::Acquire)
    }

    fn beat(&self, now: Millis) {
        self.beat.store(now, std::sync::atomic::Ordering::Release);
    }

    fn stop(&self) {
        self.stopped
            .store(true, std::sync::atomic::Ordering::Release);
    }
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

    /// The latest status document, serialized. The engine publishes it a
    /// few times a second; reading it never waits on the engine.
    pub fn status(&self) -> Bytes {
        self.status.borrow().clone()
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
    /// The snapshot directory, when this backend swaps.
    store: Option<SnapshotStore>,
    persister: Option<Persister>,
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
    vitals: Arc<Vitals>,
    /// The status document, republished at most every [`STATUS_EVERY`].
    status: watch::Sender<Bytes>,
    status_at: Option<Instant>,
}

/// How fresh the published status is.
const STATUS_EVERY: Duration = Duration::from_millis(100);

impl Engine {
    pub fn new(
        config: Arc<Config>,
        links: Vec<(Arc<BackendLink>, BackendConfig, Option<SnapshotStore>)>,
        models: Arc<Models>,
        policies: Policies,
    ) -> (Self, Handle) {
        let (sender, receiver) = mpsc::unbounded_channel();
        let (events, _) = broadcast::channel(1024);
        let metrics = Arc::new(Metrics::default());
        let clock = Clock(Instant::now());
        let vitals = Arc::new(Vitals::default());
        let (status, status_receiver) = watch::channel(Bytes::from_static(b"{}"));
        let handle = Handle {
            status: status_receiver,
            sender: sender.clone(),
            events: events.clone(),
            metrics: metrics.clone(),
            clock,
            vitals: vitals.clone(),
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
            .map(|(link, backend, store)| BackendState {
                persister: store
                    .clone()
                    .map(|store| Persister::spawn(store, link.name.clone())),
                store,
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
            vitals,
            status,
            status_at: None,
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
        // However the loop ends (a panic included), the gate learns of it.
        struct Stopped(Arc<Vitals>);
        impl Drop for Stopped {
            fn drop(&mut self) {
                self.0.stop();
            }
        }
        let _stopped = Stopped(self.vitals.clone());
        loop {
            self.vitals.beat(self.clock.now());
            tokio::select! {
                message = self.receiver.recv() => {
                    let Some(message) = message else { break };
                    if let ControlFlow::Break(reply) = self.handle(message) {
                        self.flush().await;
                        let _ = reply.send(());
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
            if self.status_at.is_none_or(|at| at.elapsed() >= STATUS_EVERY) {
                self.status_at = Some(Instant::now());
                let document = serde_json::to_vec(&self.status()).unwrap_or_default();
                self.status.send_replace(Bytes::from(document));
            }
        }
    }

    /// Handles one message; `Break` stops the engine, answering the
    /// shutdown once the snapshot indexes are written.
    fn handle(&mut self, message: Message) -> ControlFlow<oneshot::Sender<()>> {
        let now = self.clock.now();
        match message {
            Message::Arrive(arrival) => self.admit(arrival),
            Message::Withdraw { ticket, report } => {
                self.waiting.remove(&ticket);
                self.running.remove(&ticket);
                let relayed = report.as_ref().and_then(|report| report.generated);
                if let Some(report) = report {
                    self.record(report);
                }
                self.feed(Event::Withdrawn {
                    ticket,
                    relayed,
                    now,
                });
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
            Message::Reconfigure(config, reply) => {
                let _ = reply.send(self.reconfigure(config));
            }
            Message::Shutdown(reply) => {
                self.feed(Event::Shutdown { now });
                self.dirty
                    .extend(self.backends.iter().map(|backend| backend.link.id));
                self.persist();
                return ControlFlow::Break(reply);
            }
        }
        ControlFlow::Continue(())
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
        // Classes are matched by name: a reload between the request's
        // identification and now may have renumbered them.
        let mut demand = demand;
        demand.class = self
            .class_names
            .iter()
            .position(|name| **name == *class)
            .and_then(|index| u16::try_from(index).ok())
            .map_or(demand.class, ClassId);
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

    /// The backend a running ticket is on.
    fn locate(&self, ticket: TicketId) -> Option<BackendId> {
        self.scheduler.slot_of(ticket).map(|slot| slot.backend)
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
            Upstream::Ready(ready) => {
                let Ready {
                    info,
                    capabilities,
                    slots,
                    restarted,
                    loaded,
                } = *ready;
                let config = self.config.clone();
                let state = &mut self.backends[index];
                let namespace = namespace(&info);
                let layout = layout(&config, &state.config, &info);
                let swap = state.store.is_some();
                let first = !state.known;
                let new_model = state.namespace.as_deref() != Some(namespace.as_str());
                let reshaped = state.info.as_ref() != Some(&info);
                if let Some(loaded) = &loaded {
                    describe_loaded(&name, loaded);
                }
                let snapshots = (first || new_model)
                    .then(|| loaded.map(|loaded| loaded.index).unwrap_or_default());
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

    fn emit(&self, mut event: Value) {
        if self.events.receiver_count() == 0 {
            return;
        }
        if event.get("at_ms").is_none() {
            event["at_ms"] = json!(self.clock.now());
        }
        let _ = self.events.send(Arc::from(event.to_string()));
    }

    /// Hands each changed index to its backend's persister, which writes
    /// the latest version in the background.
    fn persist(&mut self) {
        for backend in std::mem::take(&mut self.dirty) {
            let Some(state) = self.backends.get_mut(usize::from(backend.0)) else {
                continue;
            };
            if let (Some(index), Some(namespace), Some(persister)) = (
                self.scheduler.snapshots(backend),
                state.namespace.clone(),
                state.persister.as_mut(),
            ) {
                persister.submit(namespace, index.clone());
            }
        }
    }

    /// Waits for every backend's latest index to be written.
    async fn flush(&mut self) {
        for state in &mut self.backends {
            if let Some(persister) = state.persister.as_mut()
                && !persister.flush(Duration::from_secs(10)).await
            {
                warn!(
                    backend = state.link.name,
                    "the snapshot index was not written before shutdown; snapshots since the last write will be removed at the next start"
                );
            }
        }
    }
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
