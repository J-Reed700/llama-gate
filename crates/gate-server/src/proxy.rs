//! The HTTP data plane.
//!
//! A generative request is a session: identified, measured, queued, then
//! sent to the slot the engine grants. A session may span several attempts:
//! its server may refuse the connection (the request is placed again,
//! perhaps on another server), or the engine may pause it for more urgent
//! work before its output begins (it waits, then resumes where it stopped).
//! The client sees one response throughout, and however the session ends it
//! is reported to the engine exactly once. Everything else passes through.
use crate::clients::{Caller, Identify, Identity, Names};
use crate::config::{Config, Mode};
use crate::control::Control;
use crate::models::Models;
use crate::runtime::{Arrival, BackendLink, Grant, Handle, Report};
use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::{ConnectInfo, Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use gate_core::{Demand, ModelId, Rejection, SlotId, TicketId, Tokens};
use gate_llama::usage::{Sniffer, Usage};
use gate_llama::{Dialect, Dialects, Health, Method as Tokenization};
use serde_json::{Value, json};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tracing::{info, warn};

pub struct AppState {
    pub config: RwLock<Arc<Config>>,
    pub handle: Handle,
    pub backends: Vec<Arc<BackendLink>>,
    pub models: Arc<Models>,
    pub dialects: Dialects,
    pub identify: RwLock<Arc<dyn Identify>>,
    /// Client names, kept across reloads.
    pub names: Arc<Mutex<Names>>,
    /// The program supplied its own identification: reloads keep it.
    pub custom_identify: bool,
    pub tickets: AtomicU64,
    /// The configuration file, for reloads.
    pub source: Option<PathBuf>,
}

impl AppState {
    pub fn config(&self) -> Arc<Config> {
        self.config
            .read()
            .map(|config| config.clone())
            .unwrap_or_else(|poisoned| poisoned.into_inner().clone())
    }

    pub fn identify(&self) -> Arc<dyn Identify> {
        self.identify
            .read()
            .map(|identify| identify.clone())
            .unwrap_or_else(|poisoned| poisoned.into_inner().clone())
    }

    /// A backend of `model` to measure with or forward to: an online one
    /// if there is one.
    fn backend_for(&self, model: Option<ModelId>) -> Option<Arc<BackendLink>> {
        let candidates: Vec<&Arc<BackendLink>> = self
            .backends
            .iter()
            .filter(|link| model.is_none_or(|model| link.model == model))
            .collect();
        candidates
            .iter()
            .find(|link| *link.health.borrow() == Health::Ready)
            .or_else(|| candidates.first())
            .map(|link| (*link).clone())
    }
}

/// The dashboard page, compiled into the binary.
const DASHBOARD: &str = include_str!("../assets/dashboard.html");

pub fn router(state: Arc<AppState>) -> Router {
    let config = state.config();
    let prefix = config.observe.path_prefix.trim_end_matches('/').to_owned();
    let mut router = Router::new()
        .route(&format!("{prefix}/status"), get(status))
        .route(&format!("{prefix}/metrics"), get(metrics))
        .route(&format!("{prefix}/health"), get(health))
        .route(&format!("{prefix}/events"), get(events))
        .route(&format!("{prefix}/reload"), post(reload))
        .route("/v1/models", get(models));
    if config.observe.dashboard {
        let to = format!("{prefix}/");
        router = router
            .route(&format!("{prefix}/"), get(dashboard))
            .route(&format!("{prefix}/dashboard"), get(dashboard))
            .route(
                &prefix,
                get(move || async move {
                    (StatusCode::PERMANENT_REDIRECT, [(header::LOCATION, to)]).into_response()
                }),
            );
    }
    router.fallback(proxy).with_state(state)
}

/// The refusal for a caller without an admin key, when admin keys are
/// configured.
fn not_admin(state: &AppState, headers: &HeaderMap) -> Option<Response> {
    (!state.identify().is_admin(headers)).then(|| {
        error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "an admin key is required".into(),
        )
    })
}

async fn dashboard() -> Response {
    (
        [
            (header::CONTENT_TYPE, "text/html; charset=utf-8"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        DASHBOARD,
    )
        .into_response()
}

async fn status(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Some(denied) = not_admin(&state, &headers) {
        return denied;
    }
    match state.handle.status().await {
        Some(status) => axum::Json(status).into_response(),
        None => error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "the engine is not responding".into(),
        ),
    }
}

async fn metrics(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !state.config().auth.public_metrics
        && let Some(denied) = not_admin(&state, &headers)
    {
        return denied;
    }
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        state.handle.metrics.render(),
    )
        .into_response()
}

async fn health(State(state): State<Arc<AppState>>) -> Response {
    let backends: Vec<Value> = state
        .backends
        .iter()
        .map(|link| json!({"name": link.name, "state": *link.health.borrow()}))
        .collect();
    axum::Json(json!({"status": "ok", "backends": backends})).into_response()
}

/// Live events as Server-Sent Events.
async fn events(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Some(denied) = not_admin(&state, &headers) {
        return denied;
    }
    let receiver = state.handle.subscribe();
    let stream = futures_util::stream::unfold(
        (
            receiver,
            true,
            tokio::time::interval(Duration::from_secs(15)),
        ),
        |(mut receiver, first, mut ping)| async move {
            if first {
                let hello = Bytes::from_static(b": connected\n\n");
                return Some((Ok::<_, std::io::Error>(hello), (receiver, false, ping)));
            }
            loop {
                tokio::select! {
                    event = receiver.recv() => match event {
                        Ok(event) => {
                            let chunk = Bytes::from(format!("data: {event}\n\n"));
                            return Some((Ok(chunk), (receiver, false, ping)));
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => return None,
                    },
                    _ = ping.tick() => {
                        return Some((Ok(Bytes::from_static(b": ping\n\n")), (receiver, false, ping)));
                    }
                }
            }
        },
    );
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(stream))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

/// Re-reads the configuration file and applies what can change live.
async fn reload(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Some(denied) = not_admin(&state, &headers) {
        return denied;
    }
    match apply_reload(&state).await {
        Ok(notes) => axum::Json(json!({"reloaded": true, "notes": notes})).into_response(),
        Err(problem) => error(StatusCode::BAD_REQUEST, "invalid_request_error", problem),
    }
}

/// Reloads the configuration from its file: the engine first, then the
/// data plane's settings and client rules.
pub async fn apply_reload(state: &AppState) -> Result<Vec<String>, String> {
    let Some(path) = &state.source else {
        return Err("the gate was not started from a configuration file".into());
    };
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let mut config = Config::from_toml(&text)?;
    let current = state.config();
    // Command-line overrides of these stay in force.
    config.server.listen = current.server.listen;
    config.server.upstream = current.server.upstream.clone();
    config.server.mode = current.server.mode;
    let config = Arc::new(config);
    let notes = state.handle.reconfigure(config.clone()).await?;
    if !state.custom_identify {
        let rules = crate::clients::Rules::new(
            &config.clients,
            &config.auth,
            &config.class_names(),
            state.names.clone(),
        );
        rules.resolve().await;
        if let Ok(mut identify) = state.identify.write() {
            *identify = Arc::new(rules);
        }
    }
    if let Ok(mut current) = state.config.write() {
        *current = config;
    }
    Ok(notes)
}

/// The models clients may ask for. With one backend, the server's own
/// answer; with several, the gate's model names.
async fn models(
    State(state): State<Arc<AppState>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    request: Request,
) -> Response {
    let (parts, _) = request.into_parts();
    let caller = Caller {
        peer: peer.ip(),
        headers: &parts.headers,
        body: &Value::Null,
    };
    if let Err(denied) = state.identify().identify(&caller) {
        return error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            denied.message().into(),
        );
    }
    if state.backends.len() == 1
        && let Some(link) = state.backend_for(None)
    {
        let owns_key = state.identify().owns_key(&parts.headers);
        return relay(
            &state,
            &link,
            &parts.method,
            &parts.uri,
            &parts.headers,
            peer,
            Bytes::new(),
            owns_key,
        )
        .await;
    }
    let data: Vec<Value> = state
        .models
        .list()
        .into_iter()
        .map(|model| {
            let backends: Vec<&str> = state
                .backends
                .iter()
                .filter(|link| link.model == model.id)
                .map(|link| link.name.as_str())
                .collect();
            json!({
                "id": model.name,
                "object": "model",
                "owned_by": "llama-gate",
                "aliases": model.aliases,
                "backends": backends,
            })
        })
        .collect();
    axum::Json(json!({"object": "list", "data": data})).into_response()
}

async fn proxy(
    State(state): State<Arc<AppState>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    request: Request,
) -> Response {
    let config = state.config();
    let (parts, body) = request.into_parts();
    let max_body = config.server.max_body.0 as usize;
    let Ok(bytes) = to_bytes(body, max_body).await else {
        return error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "invalid_request_error",
            format!("request body exceeds {}", config.server.max_body),
        );
    };
    let mode = config.server.mode;
    if mode != Mode::Bypass
        && is_slot_action(&parts.method, parts.uri.path())
        && !config.server.allow_slot_actions
    {
        return error(
            StatusCode::FORBIDDEN,
            "permission_error",
            "slot actions are managed by llama-gate (set server.allow_slot_actions to permit them)"
                .into(),
        );
    }
    let dialect = (mode != Mode::Bypass && parts.method == Method::POST)
        .then(|| state.dialects.find(parts.uri.path()))
        .flatten();
    let body: Value = if parts.method == Method::POST {
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    } else {
        Value::Null
    };
    let identify = state.identify();
    let identity = match identify.identify(&Caller {
        peer: peer.ip(),
        headers: &parts.headers,
        body: &body,
    }) {
        Ok(identity) => identity,
        Err(denied) => {
            return error(
                StatusCode::UNAUTHORIZED,
                "authentication_error",
                denied.message().into(),
            );
        }
    };
    let owns_key = identify.owns_key(&parts.headers);
    let requested_model = body
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| query_value(&parts.uri, "model"));
    let model = state.models.resolve(requested_model.as_deref());
    let (Some(dialect), true) = (dialect, body.is_object()) else {
        // Not scheduled: pass through to a server of the requested model.
        let Some(link) = state.backend_for(model) else {
            return error(StatusCode::BAD_GATEWAY, "server_error", "no backend".into());
        };
        return relay(
            &state,
            &link,
            &parts.method,
            &parts.uri,
            &parts.headers,
            peer,
            bytes,
            owns_key,
        )
        .await;
    };
    let Some(model) = model else {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "no models are configured".into(),
        );
    };
    generate(
        state,
        config,
        parts,
        peer,
        dialect,
        body,
        bytes.len(),
        identity,
        model,
        owns_key,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn generate(
    state: Arc<AppState>,
    config: Arc<Config>,
    parts: axum::http::request::Parts,
    peer: SocketAddr,
    dialect: Arc<dyn Dialect>,
    body: Value,
    size: usize,
    identity: Identity,
    model: ModelId,
    owns_key: bool,
) -> Response {
    let arrived = Instant::now();
    let Some(link) = state.backend_for(Some(model)) else {
        return error(
            StatusCode::BAD_GATEWAY,
            "server_error",
            "no backend serves this model".into(),
        );
    };
    let measurer = link.measurer.borrow().clone();
    let scope = config.clients.isolate.then(|| identity.name.to_string());
    let measurement = measurer
        .measure_in(dialect.as_ref(), &body, size, scope.as_deref())
        .await;
    let measured = arrived.elapsed();
    if measurement.mismatch {
        warn!(
            backend = link.name,
            "incremental tokenization disagreed with the server; prompts of this model are tokenized whole from now on"
        );
    }
    let class_name = config
        .class_names()
        .get(usize::from(identity.class.0))
        .cloned()
        .unwrap_or_default();
    let admission = &config.admission;
    let class = config.classes.get(&class_name);
    let (generation, min_generation) = if measurement.generates {
        let default = class
            .and_then(|class| class.default_generation)
            .unwrap_or(admission.default_generation);
        let ceiling = class
            .and_then(|class| class.max_generation)
            .or((admission.max_generation > 0).then_some(admission.max_generation));
        let mut generation = measurement.requested_generation.unwrap_or(default).max(1);
        if let Some(ceiling) = ceiling {
            generation = generation.min(ceiling.max(1));
        }
        (generation, admission.min_generation.min(generation))
    } else {
        (0, 0)
    };
    let model_name = state.models.name(model);
    let measured_label = match (measurement.exact, measurement.method) {
        (false, _) => "estimated",
        (true, Some(Tokenization::Incremental)) => "incremental",
        (true, _) => "exact",
    };

    if config.server.mode == Mode::Observe {
        info!(
            client = %identity.name,
            class = %class_name,
            model = model_name,
            dialect = dialect.name(),
            prompt = measurement.prompt,
            measured = measured_label,
            generation,
            "observe: forwarding without scheduling"
        );
        let bytes = Bytes::from(serde_json::to_vec(&body).unwrap_or_default());
        return relay(
            &state,
            &link,
            &parts.method,
            &parts.uri,
            &parts.headers,
            peer,
            bytes,
            owns_key,
        )
        .await;
    }

    let ticket = TicketId(state.tickets.fetch_add(1, Ordering::Relaxed) + 1);
    let control = Arc::new(Control::default());
    let (reply, grant) = oneshot::channel();
    let class_name: Arc<str> = Arc::from(class_name.as_str());
    state.handle.arrive(Arrival {
        ticket,
        demand: Demand {
            client: identity.client,
            class: identity.class,
            model,
            prompt: measurement.prompt,
            work: measurement.work,
            generation,
            min_generation,
            prefix: measurement.prefix.clone(),
            exclusive: measurement.exclusive,
            pinned: measurement.pinned,
        },
        client: identity.name.clone(),
        class: class_name.clone(),
        dialect: dialect.name(),
        measured: measured_label,
        control: control.clone(),
        reply,
    });
    let session = Session {
        state,
        config,
        ticket,
        control,
        method: parts.method,
        uri: parts.uri,
        headers: parts.headers,
        peer,
        streaming: measurement.streaming,
        requested: measurement.requested_generation,
        generates: measurement.generates,
        dialect,
        body,
        owns_key,
        arrived,
        waiting_since: Instant::now(),
        queued: Duration::ZERO,
        first_output: None,
        ended: false,
        report: Report {
            ticket,
            at_ms: 0,
            client: identity.name,
            class: class_name,
            model: model_name,
            dialect: "",
            backend: None,
            slot: None,
            outcome: "cancelled",
            status: 499,
            prompt: measurement.prompt,
            cached: None,
            generated: None,
            measure_ms: measured.as_millis() as u64,
            queue_ms: 0,
            ttft_ms: None,
            total_ms: 0,
            tokens_per_second: None,
            preempted: 0,
            bounced: 0,
        },
    };
    session.run(grant).await
}

/// One attempt at running the request on a granted slot.
enum Attempt {
    Response(reqwest::Response),
    /// Paused before its output began; the next grant arrives here.
    Paused(oneshot::Receiver<Grant>),
    /// The server refused the connection; the next grant arrives here.
    Bounced(oneshot::Receiver<Grant>),
    Failed(String),
}

/// A request from arrival to its end, across attempts.
struct Session {
    state: Arc<AppState>,
    config: Arc<Config>,
    ticket: TicketId,
    control: Arc<Control>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    peer: SocketAddr,
    dialect: Arc<dyn Dialect>,
    body: Value,
    streaming: bool,
    requested: Option<Tokens>,
    generates: bool,
    owns_key: bool,
    arrived: Instant,
    /// When the current wait for a grant began.
    waiting_since: Instant,
    /// Time spent waiting for grants so far.
    queued: Duration,
    first_output: Option<Instant>,
    ended: bool,
    report: Report,
}

impl Drop for Session {
    fn drop(&mut self) {
        if !self.ended {
            // The client left, or the handler stopped: withdraw once.
            self.close("cancelled", 499, None);
        }
    }
}

impl Session {
    async fn run(mut self, mut grant: oneshot::Receiver<Grant>) -> Response {
        let keepalive = self.streaming && !self.config.queue.keepalive_interval.0.is_zero();
        loop {
            let granted = if keepalive {
                match tokio::time::timeout(self.config.queue.keepalive_after.0, &mut grant).await {
                    Ok(granted) => granted,
                    Err(_) => return self.commit_waiting(grant),
                }
            } else {
                (&mut grant).await
            };
            let (slot, generation, exclusive) = match granted {
                Ok(Grant::Run {
                    slot,
                    generation,
                    exclusive,
                    ..
                }) => (slot, generation, exclusive),
                Ok(Grant::Refused(rejection)) => return self.refuse(&rejection),
                Err(_) => {
                    return self.fail(StatusCode::SERVICE_UNAVAILABLE, "the scheduler stopped");
                }
            };
            match self.attempt(slot, generation, exclusive).await {
                Attempt::Response(response) => {
                    if self.streaming && response.status().is_success() {
                        return self.commit_stream(response);
                    }
                    // A whole answer (or an error): it is all the output.
                    if !self.control.begin() {
                        match self.control.take_regrant() {
                            Some(next) => {
                                self.report.preempted += 1;
                                grant = next;
                                continue;
                            }
                            None => {
                                return self.fail(
                                    StatusCode::SERVICE_UNAVAILABLE,
                                    "paused without a way to resume",
                                );
                            }
                        }
                    }
                    return self.relay_whole(response);
                }
                Attempt::Paused(next) => {
                    self.report.preempted += 1;
                    grant = next;
                }
                Attempt::Bounced(next) => {
                    self.report.bounced += 1;
                    grant = next;
                }
                Attempt::Failed(message) => return self.fail(StatusCode::BAD_GATEWAY, &message),
            }
        }
    }

    /// Sends the request to its granted slot, unless it is paused first.
    async fn attempt(&mut self, slot: SlotId, generation: Tokens, exclusive: bool) -> Attempt {
        self.queued += self.waiting_since.elapsed();
        let Some(link) = self
            .state
            .backends
            .get(usize::from(slot.backend.0))
            .cloned()
        else {
            return Attempt::Failed(format!("granted unknown backend {}", slot.backend));
        };
        self.report.backend = Some(link.name.clone());
        self.report.slot = Some(slot.index);
        let mut body = self.body.clone();
        if !exclusive {
            self.dialect.pin(&mut body, slot.index);
        }
        if self.generates
            && self.config.admission.enforce_generation
            && self.requested != Some(generation)
        {
            self.dialect.limit_generation(&mut body, generation);
        }
        if let (Some(model), Some(object)) = (&link.router_model, body.as_object_mut()) {
            object.insert("model".into(), Value::String(model.clone()));
        }
        let bytes = Bytes::from(serde_json::to_vec(&body).unwrap_or_default());
        let request = upstream_request(
            &self.state,
            &link,
            &self.method,
            &self.uri,
            &self.headers,
            self.peer,
            bytes,
            self.owns_key,
        );
        let control = self.control.clone();
        let outcome = tokio::select! {
            biased;
            _ = control.paused() => None,
            result = request.send() => Some(result),
        };
        self.waiting_since = Instant::now();
        match outcome {
            None => match self.control.take_regrant() {
                Some(next) => Attempt::Paused(next),
                None => Attempt::Failed("paused without a way to resume".into()),
            },
            Some(Ok(response)) => Attempt::Response(response),
            Some(Err(error)) if error.is_connect() => {
                Attempt::Bounced(self.state.handle.bounce(self.ticket))
            }
            Some(Err(error)) => Attempt::Failed(format!("upstream request failed: {error}")),
        }
    }

    /// Relays a whole (non-streaming or error) response, reporting when its
    /// body is through.
    fn relay_whole(mut self, response: reqwest::Response) -> Response {
        self.first_output = Some(Instant::now());
        self.state.handle.started(self.ticket);
        let status = response.status();
        let mut builder = Response::builder().status(status.as_u16());
        for (name, value) in response.headers() {
            if !is_hop_by_hop(name.as_str()) {
                builder = builder.header(name.as_str(), value.as_bytes());
            }
        }
        let stream = Leased {
            inner: response.bytes_stream(),
            session: Some(self),
            sniffer: Sniffer::default(),
            status: status.as_u16(),
        };
        builder
            .body(Body::from_stream(stream))
            .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
    }

    /// Answers a streaming request whose server has answered: mirrors the
    /// server's status and headers, then streams through a channel that
    /// outlives this attempt, should the request be paused before it
    /// produces anything.
    fn commit_stream(self, response: reqwest::Response) -> Response {
        let mut builder = Response::builder().status(response.status().as_u16());
        for (name, value) in response.headers() {
            if !is_hop_by_hop(name.as_str()) {
                builder = builder.header(name.as_str(), value.as_bytes());
            }
        }
        let (sender, receiver) = mpsc::channel(64);
        tokio::spawn(self.pump(sender, Some(response), None));
        builder
            .body(channel_body(receiver))
            .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
    }

    /// Answers a streaming request that is still waiting: commits to a 200
    /// event stream now and pings it until the request runs. Later refusals
    /// and errors are reported in-band as `error` events, as llama.cpp
    /// reports mid-stream errors.
    fn commit_waiting(self, grant: oneshot::Receiver<Grant>) -> Response {
        let (sender, receiver) = mpsc::channel(64);
        tokio::spawn(self.pump(sender, None, Some(grant)));
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header("x-accel-buffering", "no")
            .body(channel_body(receiver))
            .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
    }

    /// Streams a committed response to its end: relays the server's events,
    /// and between attempts waits for the next grant with pings.
    async fn pump(
        mut self,
        sender: mpsc::Sender<Result<Bytes, std::io::Error>>,
        mut current: Option<reqwest::Response>,
        mut grant: Option<oneshot::Receiver<Grant>>,
    ) {
        let interval = self
            .config
            .queue
            .keepalive_interval
            .0
            .max(Duration::from_millis(100));
        loop {
            if let Some(response) = current.take() {
                if !response.status().is_success() {
                    let status = response.status().as_u16();
                    let text = response.text().await.unwrap_or_default();
                    let message = serde_json::from_str::<Value>(&text)
                        .ok()
                        .and_then(|value| {
                            value
                                .pointer("/error/message")
                                .and_then(Value::as_str)
                                .map(str::to_owned)
                        })
                        .unwrap_or(text);
                    let _ = sender
                        .send(Ok(sse_error(
                            "server_error",
                            &format!("HTTP {status}: {message}"),
                        )))
                        .await;
                    self.close("error", status, None);
                    return;
                }
                let mut stream = response.bytes_stream();
                let mut sniffer = Sniffer::default();
                loop {
                    let step = if self.control.started() {
                        Some(stream.next().await)
                    } else {
                        let control = self.control.clone();
                        tokio::select! {
                            biased;
                            _ = control.paused() => None,
                            chunk = stream.next() => Some(chunk),
                        }
                    };
                    match step {
                        None => {
                            // Paused before any output: drop this attempt.
                            self.report.preempted += 1;
                            grant = self.control.take_regrant();
                            break;
                        }
                        Some(Some(Ok(chunk))) => {
                            if self.first_output.is_none() && carries_data(&chunk) {
                                if !self.control.begin() {
                                    self.report.preempted += 1;
                                    grant = self.control.take_regrant();
                                    break;
                                }
                                self.first_output = Some(Instant::now());
                                self.state.handle.started(self.ticket);
                            }
                            sniffer.observe(&chunk);
                            if sender.send(Ok(chunk)).await.is_err() {
                                // The client left; dropping the stream
                                // cancels the request upstream.
                                self.close("cancelled", 499, None);
                                return;
                            }
                        }
                        Some(Some(Err(error))) => {
                            let _ = sender
                                .send(Ok(sse_error("server_error", &error.to_string())))
                                .await;
                            self.close("error", 502, None);
                            return;
                        }
                        Some(None) => {
                            self.close("ok", 200, Some(sniffer.usage()));
                            return;
                        }
                    }
                }
            }
            // Waiting for a grant (the first, or the next after a pause or
            // a bounce), pinging the client meanwhile.
            let Some(mut receiver) = grant.take() else {
                let _ = sender
                    .send(Ok(sse_error("server_error", "the request lost its place")))
                    .await;
                self.close("error", 500, None);
                return;
            };
            let mut ping = tokio::time::interval(interval);
            ping.tick().await;
            let granted = loop {
                tokio::select! {
                    granted = &mut receiver => break granted,
                    _ = ping.tick() => {
                        if sender.send(Ok(Bytes::from_static(b": queued\n\n"))).await.is_err() {
                            self.close("cancelled", 499, None);
                            return;
                        }
                    }
                }
            };
            match granted {
                Ok(Grant::Run {
                    slot,
                    generation,
                    exclusive,
                    ..
                }) => match self.attempt(slot, generation, exclusive).await {
                    Attempt::Response(response) => current = Some(response),
                    Attempt::Paused(next) => {
                        self.report.preempted += 1;
                        grant = Some(next);
                    }
                    Attempt::Bounced(next) => {
                        self.report.bounced += 1;
                        grant = Some(next);
                    }
                    Attempt::Failed(message) => {
                        let _ = sender.send(Ok(sse_error("server_error", &message))).await;
                        self.close("error", 502, None);
                        return;
                    }
                },
                Ok(Grant::Refused(rejection)) => {
                    let (status, kind, message) = describe(&rejection);
                    let _ = sender.send(Ok(sse_error(kind, &message))).await;
                    self.close("refused", status.as_u16(), None);
                    return;
                }
                Err(_) => {
                    let _ = sender
                        .send(Ok(sse_error("server_error", "the scheduler stopped")))
                        .await;
                    self.close("error", 503, None);
                    return;
                }
            }
        }
    }

    fn refuse(mut self, rejection: &Rejection) -> Response {
        let (status, ..) = describe(rejection);
        self.close("refused", status.as_u16(), None);
        refusal(rejection, self.config.queue.retry_after.0)
    }

    fn fail(mut self, status: StatusCode, message: &str) -> Response {
        self.close("error", status.as_u16(), None);
        error(status, "server_error", message.to_owned())
    }

    /// Ends the session: reports a completed request to the engine with
    /// its usage, anything else as withdrawn.
    fn close(&mut self, outcome: &'static str, status: u16, usage: Option<Usage>) {
        if self.ended {
            return;
        }
        self.ended = true;
        let now = Instant::now();
        let report = &mut self.report;
        report.dialect = self.dialect.name();
        report.outcome = outcome;
        report.status = status;
        report.queue_ms = self.queued.as_millis() as u64;
        report.total_ms = now.duration_since(self.arrived).as_millis() as u64;
        report.ttft_ms = self
            .first_output
            .map(|first| first.duration_since(self.arrived).as_millis() as u64);
        if let Some(usage) = usage {
            report.cached = usage.cached;
            report.generated = usage.generated;
            // The server's own measure; for a stream, failing that, the
            // rate the client saw.
            report.tokens_per_second = usage.tokens_per_second().or_else(|| {
                let generated = usage.generated.filter(|generated| *generated > 1)?;
                let seconds = now.duration_since(self.first_output?).as_secs_f64();
                (self.streaming && seconds > 0.0).then(|| generated as f64 / seconds)
            });
        }
        let report = report.clone();
        match usage {
            Some(usage) if outcome == "ok" => self.state.handle.finish(self.ticket, usage, report),
            _ => self.state.handle.withdraw(self.ticket, Some(report)),
        }
    }
}

fn channel_body(receiver: mpsc::Receiver<Result<Bytes, std::io::Error>>) -> Body {
    Body::from_stream(futures_util::stream::unfold(
        receiver,
        |mut receiver| async move { receiver.recv().await.map(|item| (item, receiver)) },
    ))
}

/// Whether a chunk of an event stream carries an event (as opposed to
/// comments and blank lines): output the client sees.
fn carries_data(chunk: &[u8]) -> bool {
    chunk
        .split(|byte| *byte == b'\n')
        .any(|line| line.starts_with(b"data:") || line.starts_with(b"event:"))
}

fn sse_error(kind: &str, message: &str) -> Bytes {
    let event = json!({"error": {"message": message, "type": kind}});
    Bytes::from(format!("data: {event}\n\n"))
}

/// A whole response's body, reporting its session's end when it is through
/// (or, if the client leaves first, its withdrawal).
struct Leased<S> {
    inner: S,
    session: Option<Session>,
    sniffer: Sniffer,
    status: u16,
}

impl<S> Stream for Leased<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    type Item = Result<Bytes, reqwest::Error>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let polled = self.inner.poll_next_unpin(context);
        match &polled {
            Poll::Ready(Some(Ok(chunk))) => {
                let chunk = chunk.clone();
                self.sniffer.observe(&chunk);
            }
            Poll::Ready(None) => {
                let usage = self.sniffer.usage();
                let status = self.status;
                if let Some(mut session) = self.session.take() {
                    let outcome = if (200..300).contains(&status) {
                        "ok"
                    } else {
                        "error"
                    };
                    session.close(outcome, status, Some(usage));
                }
            }
            Poll::Ready(Some(Err(_))) => {
                if let Some(mut session) = self.session.take() {
                    session.close("error", 502, None);
                }
            }
            Poll::Pending => {}
        }
        polled
    }
}

/// Builds the request to a backend: hop-by-hop and configured headers
/// removed, the gate's own credentials replaced by the server's, the
/// client's address added.
#[allow(clippy::too_many_arguments)]
fn upstream_request(
    state: &AppState,
    link: &BackendLink,
    method: &Method,
    uri: &Uri,
    headers: &HeaderMap,
    peer: SocketAddr,
    body: Bytes,
    owns_key: bool,
) -> reqwest::RequestBuilder {
    let url = format!(
        "{}{}",
        link.server.base(),
        uri.path_and_query().map_or("/", |path| path.as_str())
    );
    let config = state.config();
    let strip = &config.server.strip_request_headers;
    let mut request = link.server.http().request(method.clone(), &url).body(body);
    for (name, value) in headers {
        let lowered = name.as_str();
        let credential = matches!(lowered, "authorization" | "x-api-key");
        if is_hop_by_hop(lowered)
            || strip
                .iter()
                .any(|strip| strip.eq_ignore_ascii_case(lowered))
            || (credential && (owns_key || link.api_key.is_some()))
        {
            continue;
        }
        request = request.header(name, value);
    }
    if let Some(key) = &link.api_key {
        request = request.bearer_auth(key);
    }
    request.header(
        HeaderName::from_static("x-forwarded-for"),
        HeaderValue::from_str(&peer.ip().to_string())
            .unwrap_or(HeaderValue::from_static("unknown")),
    )
}

/// Forwards a request the gate does not schedule and streams the answer.
#[allow(clippy::too_many_arguments)]
async fn relay(
    state: &AppState,
    link: &BackendLink,
    method: &Method,
    uri: &Uri,
    headers: &HeaderMap,
    peer: SocketAddr,
    body: Bytes,
    owns_key: bool,
) -> Response {
    let request = upstream_request(state, link, method, uri, headers, peer, body, owns_key);
    let response = match request.send().await {
        Ok(response) => response,
        Err(problem) => {
            return error(
                StatusCode::BAD_GATEWAY,
                "server_error",
                format!("upstream request failed: {problem}"),
            );
        }
    };
    let status = response.status();
    let mut builder = Response::builder().status(status.as_u16());
    for (name, value) in response.headers() {
        if !is_hop_by_hop(name.as_str()) {
            builder = builder.header(name.as_str(), value.as_bytes());
        }
    }
    builder
        .body(Body::from_stream(response.bytes_stream()))
        .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
}

fn query_value(uri: &Uri, key: &str) -> Option<String> {
    uri.query()?.split('&').find_map(|pair| {
        let (name, value) = pair.split_once('=')?;
        (name == key).then(|| decode(value))
    })
}

/// Decodes a percent-encoded query value.
fn decode(value: &str) -> String {
    fn hex(byte: u8) -> Option<u8> {
        (byte as char).to_digit(16).map(|digit| digit as u8)
    }
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        let escaped = (byte == b'%')
            .then(|| Some(hex(*bytes.get(index + 1)?)? << 4 | hex(*bytes.get(index + 2)?)?))
            .flatten();
        match escaped {
            Some(decoded) => {
                out.push(decoded);
                index += 3;
            }
            None => {
                out.push(if byte == b'+' { b' ' } else { byte });
                index += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn is_hop_by_hop(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
            | "host"
            | "content-length"
    )
}

fn is_slot_action(method: &Method, path: &str) -> bool {
    *method == Method::POST && path.starts_with("/slots/")
}

/// Status, error type, and message for a refusal.
fn describe(rejection: &Rejection) -> (StatusCode, &'static str, String) {
    match rejection {
        Rejection::TooLarge { prompt, limit } => (
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "the request needs {prompt} prompt tokens, but the server's context holds {limit}"
            ),
        ),
        Rejection::Timeout { waited } => (
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            format!("no slot became free within {}s", waited / 1000),
        ),
        Rejection::QueueFull { queued } => (
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            format!("this client already has {queued} requests waiting"),
        ),
        Rejection::UnknownClass => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "the request was assigned to an unknown class".into(),
        ),
        Rejection::Unavailable { waited } => (
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            format!(
                "no server for this model was available within {}s",
                waited / 1000
            ),
        ),
        Rejection::ShuttingDown => (
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "llama-gate is shutting down".into(),
        ),
    }
}

fn refusal(rejection: &Rejection, retry_after: Duration) -> Response {
    let (status, kind, message) = describe(rejection);
    // A prompt that cannot fit carries OpenAI's `context_length_exceeded`
    // code, so clients can compact and retry.
    let code = match rejection {
        Rejection::TooLarge { .. } => json!("context_length_exceeded"),
        _ => json!(status.as_u16()),
    };
    let mut response = error_with_code(status, kind, message, code);
    let reason = serde_json::to_value(rejection)
        .ok()
        .and_then(|value| {
            value
                .get("reason")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_default();
    if let Ok(value) = HeaderValue::from_str(&reason) {
        response
            .headers_mut()
            .insert("x-gate-rejected-reason", value);
    }
    if matches!(
        status,
        StatusCode::TOO_MANY_REQUESTS | StatusCode::SERVICE_UNAVAILABLE
    ) {
        let seconds = retry_after.as_secs().max(1).to_string();
        if let Ok(value) = HeaderValue::from_str(&seconds) {
            response.headers_mut().insert(header::RETRY_AFTER, value);
        }
    }
    response
}

/// An OpenAI-style error body, its code the HTTP status.
fn error(status: StatusCode, kind: &str, message: String) -> Response {
    error_with_code(status, kind, message, json!(status.as_u16()))
}

fn error_with_code(status: StatusCode, kind: &str, message: String, code: Value) -> Response {
    (
        status,
        axum::Json(json!({"error": {"message": message, "type": kind, "code": code}})),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_values_are_found_and_decoded() {
        let uri: Uri = "/props?model=ggml-org%2Fgemma%3AQ4&x=1".parse().unwrap();
        assert_eq!(
            query_value(&uri, "model").as_deref(),
            Some("ggml-org/gemma:Q4")
        );
        assert_eq!(query_value(&uri, "missing"), None);
    }

    #[test]
    fn events_are_told_from_comments() {
        assert!(carries_data(b"data: {\"x\":1}\n\n"));
        assert!(carries_data(b"event: message_start\ndata: {}\n\n"));
        assert!(!carries_data(b": queued\n\n"));
        assert!(!carries_data(b"\n"));
    }
}
