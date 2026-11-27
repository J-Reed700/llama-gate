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
use crate::engine::{Arrival, BackendLink, Grant, Handle, Report};
use crate::models::Models;
use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::{ConnectInfo, Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use gate_core::{Demand, ModelId, Rejection, SlotId, TicketId, Tokens};
use gate_llama::stream::{Kind, Reader, WholeReader};
use gate_llama::usage::Usage;
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

mod admin;
mod errors;
mod relay;
mod session;

pub use admin::apply_reload;
use admin::*;
use errors::*;
use relay::*;
use session::*;

pub struct AppState {
    /// The configuration and the identification built from it, replaced
    /// together on reload so a request never sees one without the other.
    pub live: RwLock<Arc<Live>>,
    pub handle: Handle,
    pub backends: Vec<Arc<BackendLink>>,
    pub models: Arc<Models>,
    pub dialects: Dialects,
    /// Client names, kept across reloads.
    pub names: Arc<Mutex<Names>>,
    /// The program supplied its own identification: reloads keep it.
    pub custom_identify: bool,
    pub tickets: AtomicU64,
    pub request_ids: AtomicU64,
    /// Bytes of request bodies held by requests in the gate.
    pub held: Arc<AtomicU64>,
    /// The configuration file, for reloads.
    pub source: Option<PathBuf>,
    /// Reloads run one at a time.
    pub reloading: tokio::sync::Mutex<()>,
}

/// What a request is served under: one configuration and its rules.
pub struct Live {
    pub config: Arc<Config>,
    pub identify: Arc<dyn Identify>,
}

impl AppState {
    /// The configuration and identification in force, as one snapshot.
    pub fn live(&self) -> Arc<Live> {
        self.live
            .read()
            .map(|live| live.clone())
            .unwrap_or_else(|poisoned| poisoned.into_inner().clone())
    }

    pub fn config(&self) -> Arc<Config> {
        self.live().config.clone()
    }

    pub fn identify(&self) -> Arc<dyn Identify> {
        self.live().identify.clone()
    }

    /// Puts a reloaded configuration and its rules in force together.
    pub fn replace(&self, live: Live) {
        let mut current = self
            .live
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *current = Arc::new(live);
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

pub fn router(state: Arc<AppState>) -> Router {
    let config = state.config();
    let prefix = config.observe.path_prefix.trim_end_matches('/').to_owned();
    let mut router = Router::new()
        .route(&format!("{prefix}/status"), get(status))
        .route(&format!("{prefix}/metrics"), get(metrics))
        .route(&format!("{prefix}/health"), get(health))
        .route(&format!("{prefix}/events"), get(events))
        .route(&format!("{prefix}/reload"), post(reload))
        .route("/v1/models", get(models))
        .route("/v1/models/{*model}", get(model_info));
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

pub(super) const REQUEST_ID_HEADER: HeaderName = HeaderName::from_static("x-request-id");

pub(super) fn request_id_for(state: &AppState, headers: &HeaderMap) -> String {
    if let Some(value) = headers.get(&REQUEST_ID_HEADER)
        && let Some(valid) = valid_request_id(value)
    {
        return valid.to_owned();
    }
    let next = state.request_ids.fetch_add(1, Ordering::Relaxed) + 1;
    format!("req-{next}")
}

fn valid_request_id(value: &HeaderValue) -> Option<&str> {
    let bytes = value.as_bytes();
    ((1..=128).contains(&bytes.len()) && bytes.iter().all(|byte| (0x21..=0x7E).contains(byte)))
        .then(|| std::str::from_utf8(bytes).ok())
        .flatten()
}

async fn proxy(
    State(state): State<Arc<AppState>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    request: Request,
) -> Response {
    // One snapshot for the whole request: a reload meanwhile cannot pair
    // this configuration with other rules.
    let live = state.live();
    let config = live.config.clone();
    let (parts, body) = request.into_parts();
    let peer_ip = peer.ip();
    let mode = config.server.mode;
    let request_id = request_id_for(&state, &parts.headers);

    if mode != Mode::Bypass
        && is_slot_action(&parts.method, parts.uri.path())
        && !config.server.allow_slot_actions
    {
        return with_request_id(
            error(
                StatusCode::FORBIDDEN,
                "permission_error",
                "slot actions are managed by llama-gate (set server.allow_slot_actions to permit them)"
                    .into(),
            ),
            &request_id,
        );
    }

    if let Err(denied) = live.identify.precheck(peer_ip, &parts.headers) {
        return with_request_id(
            error(
                StatusCode::UNAUTHORIZED,
                "authentication_error",
                denied.message().into(),
            ),
            &request_id,
        );
    }

    let max_body = config.server.max_body.0 as usize;
    let bytes =
        match tokio::time::timeout(config.server.body_timeout.0, to_bytes(body, max_body)).await {
            Ok(Ok(bytes)) => bytes,
            Ok(Err(_)) => {
                return with_request_id(
                    error(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        "invalid_request_error",
                        format!("request body exceeds {}", config.server.max_body),
                    ),
                    &request_id,
                );
            }
            Err(_) => {
                return with_request_id(
                    error(
                        StatusCode::REQUEST_TIMEOUT,
                        "invalid_request_error",
                        "the request body did not arrive in time".into(),
                    ),
                    &request_id,
                );
            }
        };
    let body = if parts.method == Method::POST {
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    } else {
        Value::Null
    };
    let identity = match live.identify.identify(Caller {
        peer: peer_ip,
        headers: &parts.headers,
        body: &body,
    }) {
        Ok(identity) => identity,
        Err(denied) => {
            return with_request_id(
                error(
                    StatusCode::UNAUTHORIZED,
                    "authentication_error",
                    denied.message().into(),
                ),
                &request_id,
            );
        }
    };

    let dialect = (mode != Mode::Bypass && parts.method == Method::POST)
        .then(|| state.dialects.find(parts.uri.path()))
        .flatten();
    let owns_key = live.identify.owns_key(&parts.headers);
    let requested_model = body
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| query_value(&parts.uri, "model"));
    let model = state.models.resolve(requested_model.as_deref());
    let (Some(dialect), true) = (dialect, body.is_object()) else {
        // Not scheduled: pass through to a server of the requested model.
        let Some(link) = state.backend_for(model) else {
            return with_request_id(
                error(StatusCode::BAD_GATEWAY, "server_error", "no backend".into()),
                &request_id,
            );
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
            &request_id,
        )
        .await;
    };
    let Some(model) = model else {
        return with_request_id(
            error(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "no models are configured".into(),
            ),
            &request_id,
        );
    };
    // From here the parsed body is all the request needs.
    let size = bytes.len();
    drop(bytes);
    let Some(held) = Held::take(&state.held, size as u64, config.server.max_held_bytes.0) else {
        return with_request_id(busy(config.queue.retry_after.0), &request_id);
    };
    generate(
        state, config, parts, peer, dialect, body, size, held, identity, model, owns_key,
        request_id,
    )
    .await
}

/// Bytes of request bodies held by requests in the gate (each keeps its
/// body to resend after a pause), given back when the request ends.
pub(super) struct Held {
    counter: Arc<AtomicU64>,
    size: u64,
}

impl Held {
    /// Takes `size` from the budget, unless that would pass `limit` while
    /// something else is held: one request is always admitted.
    fn take(counter: &Arc<AtomicU64>, size: u64, limit: u64) -> Option<Self> {
        let mut current = counter.load(Ordering::Acquire);
        loop {
            if current > 0 && current.saturating_add(size) > limit {
                return None;
            }
            match counter.compare_exchange_weak(
                current,
                current + size,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Some(Self {
                        counter: counter.clone(),
                        size,
                    });
                }
                Err(actual) => current = actual,
            }
        }
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        self.counter.fetch_sub(self.size, Ordering::AcqRel);
    }
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

fn is_slot_action(method: &Method, path: &str) -> bool {
    *method == Method::POST && path.starts_with("/slots/")
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
    fn request_id_validation_accepts_printable_ascii_and_rejects_control_or_oversized() {
        assert_eq!(
            valid_request_id(&HeaderValue::from_static("req_abc-123.XYZ")),
            Some("req_abc-123.XYZ")
        );
        let max = "a".repeat(128);
        assert_eq!(
            valid_request_id(&HeaderValue::from_str(&max).unwrap()),
            Some(max.as_str())
        );
        assert_eq!(valid_request_id(&HeaderValue::from_static("")), None);
        let too_long = "a".repeat(129);
        assert_eq!(
            valid_request_id(&HeaderValue::from_str(&too_long).unwrap()),
            None
        );
        assert_eq!(valid_request_id(&HeaderValue::from_static("bad id")), None);
        assert_eq!(valid_request_id(&HeaderValue::from_static("bad\tid")), None);
        assert!(HeaderValue::from_bytes(b"bad\x7fid").is_err());
        assert_eq!(
            valid_request_id(&HeaderValue::from_bytes(b"bad\xc3\xa9").unwrap()),
            None
        );
    }
}
