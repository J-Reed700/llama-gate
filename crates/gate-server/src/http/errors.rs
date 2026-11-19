//! Error bodies and refusals, as clients of each API expect them.

use super::*;

/// Status, error type, and message for a refusal.
pub(super) fn describe(rejection: &Rejection) -> (StatusCode, &'static str, String) {
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

pub(super) fn refusal(rejection: &Rejection, retry_after: Duration) -> Response {
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

/// The gate holds as many request bodies as it may: try again shortly.
pub(super) fn busy(retry_after: Duration) -> Response {
    let mut response = error(
        StatusCode::SERVICE_UNAVAILABLE,
        "server_error",
        "the gate is holding as many requests as it may; retry shortly".into(),
    );
    let headers = response.headers_mut();
    headers.insert("x-gate-rejected-reason", HeaderValue::from_static("busy"));
    if let Ok(value) = HeaderValue::from_str(&retry_after.as_secs().max(1).to_string()) {
        headers.insert(header::RETRY_AFTER, value);
    }
    response
}

/// An OpenAI-style error body, its code the HTTP status.
pub(super) fn error(status: StatusCode, kind: &str, message: String) -> Response {
    error_with_code(status, kind, message, json!(status.as_u16()))
}

pub(super) fn error_with_code(
    status: StatusCode,
    kind: &str,
    message: String,
    code: Value,
) -> Response {
    (
        status,
        axum::Json(json!({"error": {"message": message, "type": kind, "code": code}})),
    )
        .into_response()
}
