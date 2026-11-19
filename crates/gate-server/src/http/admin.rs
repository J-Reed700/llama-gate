//! The gate's own endpoints: dashboard, status, metrics, health, live
//! events, reload, and the model list.

use super::*;

/// The dashboard page, compiled into the binary.
pub(super) const DASHBOARD: &str = include_str!("../../assets/dashboard.html");

/// The refusal for a caller without an admin key, when admin keys are
/// configured.
pub(super) fn not_admin(state: &AppState, headers: &HeaderMap) -> Option<Response> {
    (!state.identify().is_admin(headers)).then(|| {
        error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "an admin key is required".into(),
        )
    })
}

pub(super) async fn dashboard() -> Response {
    (
        [
            (header::CONTENT_TYPE, "text/html; charset=utf-8"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        DASHBOARD,
    )
        .into_response()
}

pub(super) async fn status(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Some(denied) = not_admin(&state, &headers) {
        return denied;
    }
    (
        [
            (header::CONTENT_TYPE, "application/json"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        state.handle.status(),
    )
        .into_response()
}

pub(super) async fn metrics(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
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

pub(super) async fn health(State(state): State<Arc<AppState>>) -> Response {
    let backends: Vec<Value> = state
        .backends
        .iter()
        .map(|link| json!({"name": link.name, "state": *link.health.borrow()}))
        .collect();
    let now = state.handle.clock.now();
    match state.handle.vitals.trouble(now) {
        None => axum::Json(json!({"status": "ok", "backends": backends})).into_response(),
        Some(trouble) => (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(json!({"status": "unavailable", "reason": trouble, "backends": backends})),
        )
            .into_response(),
    }
}

/// Live events as Server-Sent Events.
pub(super) async fn events(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Some(denied) = not_admin(&state, &headers) {
        return denied;
    }
    let receiver = state.handle.subscribe();
    let vitals = state.handle.vitals.clone();
    // Ends when the gate starts shutting down, so an open dashboard does not
    // hold the shutdown for the whole drain period.
    let stream = futures_util::stream::unfold(
        (
            receiver,
            true,
            tokio::time::interval(Duration::from_secs(15)),
        ),
        move |(mut receiver, first, mut ping)| {
            let vitals = vitals.clone();
            async move {
                if first {
                    let hello = Bytes::from_static(b": connected\n\n");
                    return Some((Ok::<_, std::io::Error>(hello), (receiver, false, ping)));
                }
                loop {
                    tokio::select! {
                        () = vitals.drained() => return None,
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
pub(super) async fn reload(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
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
    // One reload at a time: a SIGHUP and a POST together would otherwise
    // leave the engine and the request path on different files.
    let _one_at_a_time = state.reloading.lock().await;
    let text = tokio::fs::read_to_string(path)
        .await
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let mut config = Config::from_toml(&text)?;
    let current = state.live();
    let mut notes = crate::engine::restart_notes(&current.config, &config);
    // These are fixed for the gate's life (some may come from the command
    // line): the notes say when the file asks for others.
    config.server.listen = current.config.server.listen;
    config.server.upstream = current.config.server.upstream.clone();
    config.server.mode = current.config.server.mode;
    let config = Arc::new(config);
    // The slow part first (resolving host names), so the engine and the
    // request path switch over at nearly the same moment.
    let identify: Arc<dyn Identify> = if state.custom_identify {
        current.identify.clone()
    } else {
        let rules = crate::clients::Rules::new(
            &config.clients,
            &config.auth,
            &config.class_names(),
            state.names.clone(),
        );
        rules.resolve().await;
        Arc::new(rules)
    };
    notes.extend(state.handle.reconfigure(config.clone()).await?);
    state.replace(Live { config, identify });
    Ok(notes)
}

/// The models clients may ask for. With one backend, the server's own
/// answer; with several, the gate's model names.
pub(super) async fn models(
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
