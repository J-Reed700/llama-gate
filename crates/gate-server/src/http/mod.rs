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
