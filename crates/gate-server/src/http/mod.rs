//! The HTTP data plane.
//!
//! A generative request is a session: identified, measured, queued, then
//! sent to the slot the engine grants. A session may span several attempts:
//! its server may refuse the connection (the request is placed again,
//! perhaps on another server), or the engine may pause it for more urgent
//! work before its output begins (it waits, then resumes where it stopped).
//! The client sees one response throughout, and however the session ends it
//! is reported to the engine exactly once. Everything else passes through.
use crate::clients::{Identify, Identity, Names, IdentifyResult};
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

async fn proxy(
    State(state): State<Arc<AppState>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    request: Request,
) -> Result<Response, Response> {
    // One snapshot for the whole request: a reload meanwhile cannot pair
    // this configuration with other rules.
    let live = state.live();
    let config = live.config.clone();
    let (parts, body) = request.into_parts();
    let peer_ip = peer.ip();

    let identity_result = match live.identify.identify_headers(&parts.headers, peer_ip) {
        Ok(res) => res,
        Err(denied) => {
            return Err(error(
                StatusCode::UNAUTHORIZED,
                "authentication_error",
                denied.message().into(),
            ));
        }
    };

    let max_body = config.server.max_body.0 as usize;
    let (identity, bytes, body) = match identity_result {
        IdentifyResult::Identified(identity) => (identity, Bytes::new(), Value::Null),
        IdentifyResult::NeedsBody(state) => {
            let bytes =
                match tokio::time::timeout(config.server.body_timeout.0, to_bytes(body, max_body)).await {
                    Ok(Ok(bytes)) => bytes,
                    Ok(Err(_)) => {
                        return Err(error(
                            StatusCode::PAYLOAD_TOO_LARGE,
                            "invalid_request_error",
                            format!("request body exceeds {}", config.server.max_body),
                        ));
                    }
                    Err(_) => {
                        return Err(error(
                            StatusCode::REQUEST_TIMEOUT,
                            "invalid_request_error",
                            "the request body did not arrive in time".into(),
                        ));
                    }
                };
            let body = if parts.method == Method::POST {
                serde_json::from_slice(&bytes).unwrap_or(Value::Null)
            } else {
                Value::Null
            };
            let identity = live.identify.identify_body(state, &body).map_err(|denied| error(
                StatusCode::UNAUTHORIZED,
                "authentication_error",
                denied.message().into(),
            ))?;
            (identity, bytes, body)
        }
    };

    let mode = config.server.mode;
    if mode != Mode::Bypass
        && is_slot_action(&parts.method, parts.uri.path())
        && !config.server.allow_slot_actions
    {
        return Err(error(
            StatusCode::FORBIDDEN,
            "permission_error",
            "slot actions are managed by llama-gate (set server.allow_slot_actions to permit them)"
                .into(),
        ));
    }
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
            return Err(error(StatusCode::BAD_GATEWAY, "server_error", "no backend".into()));
        };
        return Ok(relay(
            &state,
            &link,
            &parts.method,
            &parts.uri,
            &parts.headers,
            peer,
            bytes,
            owns_key,
        )
        .await);
    };
    let Some(model) = model else {
        return Err(error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "no models are configured".into(),
        ));
    };
    // From here the parsed body is all the request needs.
    let size = bytes.len();
    drop(bytes);
    let Some(held) = Held::take(&state.held, size as u64, config.server.max_held_bytes.0) else {
        return Ok(busy(config.queue.retry_after.0));
    };
    Ok(generate(
        state, config, parts, peer, dialect, body, size, held, identity, model, owns_key,
    )
    .await)
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
}
