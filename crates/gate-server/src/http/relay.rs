//! Forwarding to a server: the upstream request, and unscheduled pass-through.

use super::*;

/// Builds the request to a backend: hop-by-hop and configured headers
/// removed, the gate's own credentials replaced by the server's, the
/// client's address added.
#[allow(clippy::too_many_arguments)]
pub(super) fn upstream_request(
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
pub(super) async fn relay(
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

pub(super) fn is_hop_by_hop(name: &str) -> bool {
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
