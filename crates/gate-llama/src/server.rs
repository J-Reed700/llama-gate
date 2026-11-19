//! Control-plane client for a llama.cpp `llama-server`.
//!
//! Only endpoints documented in upstream llama.cpp's server README are used,
//! and every response is parsed defensively: fields that moved between
//! versions are optional, and features a build lacks are reported as
//! [`ServerError::Unsupported`] so callers can degrade instead of failing.
use serde::Serialize;
use serde_json::{Value, json};
use std::time::Duration;

#[derive(Clone, Copy, Debug)]
pub struct Timeouts {
    /// Establishing a TCP connection.
    pub connect: Duration,
    /// Shape, health, and tokenization calls.
    pub control: Duration,
    /// Saving or restoring a slot, which moves gigabytes for long contexts.
    pub slot_io: Duration,
}

impl Default for Timeouts {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(5),
            control: Duration::from_secs(60),
            slot_io: Duration::from_secs(600),
        }
    }
}

#[derive(Debug)]
pub enum ServerError {
    /// No answer: the server is down, restarting, or unreachable.
    Unreachable(String),
    /// The server does not offer this feature (for example slot saving
    /// without `--slot-save-path`).
    Unsupported(String),
    /// The server answered with an error.
    Status { status: u16, message: String },
    /// The server answered with something this client cannot read.
    Invalid(String),
}

impl std::fmt::Display for ServerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServerError::Unreachable(message) => write!(formatter, "server unreachable: {message}"),
            ServerError::Unsupported(message) => write!(formatter, "not supported: {message}"),
            ServerError::Status { status, message } => {
                write!(formatter, "HTTP {status}: {message}")
            }
            ServerError::Invalid(message) => write!(formatter, "invalid response: {message}"),
        }
    }
}

impl std::error::Error for ServerError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Health {
    Ready,
    /// Up but still loading its model.
    Loading,
    Down,
}

/// The server's shape, from `/props` and `/slots`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ServerInfo {
    pub slots: usize,
    /// Context (KV cells) each slot may use, as the server reports it.
    pub slot_context: u64,
    /// The model file the server loaded.
    pub model: Option<String>,
    /// The name it serves the model under (`--alias`), if set.
    pub alias: Option<String>,
    pub build: Option<String>,
    /// The loaded weights' size and shape (`/v1/models` meta), which tell
    /// two files at the same path apart: a re-quantized or fine-tuned model
    /// dropped in place must not inherit the old one's saved contexts.
    pub weights: Option<String>,
}

/// Optional endpoints this server answers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Capabilities {
    pub slots: bool,
    pub tokenize: bool,
    pub apply_template: bool,
}

/// One slot as `/slots` reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotView {
    pub id: usize,
    pub busy: bool,
    pub context: Option<u64>,
    pub task: Option<i64>,
    /// Tokens the slot's last task covered, when the server reports them.
    pub tokens: Option<u64>,
}

/// Result of saving or restoring a slot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlotReport {
    /// Tokens saved or restored, when the server says.
    pub tokens: Option<u64>,
    pub bytes: u64,
    pub millis: f64,
}

/// One token as `/tokenize` reports it with `with_pieces`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Piece {
    pub id: i32,
    /// The token's text; empty when the server sent raw bytes that are not
    /// text on their own.
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct Server {
    base: String,
    http: reqwest::Client,
    timeouts: Timeouts,
    /// In router mode (one server, several models) the model every call is
    /// addressed to: the `model` field of POST bodies and the `model` query
    /// parameter of GET requests, as llama-server's router expects.
    model: Option<String>,
}

impl Server {
    pub fn new(base: &str, timeouts: Timeouts) -> Result<Self, ServerError> {
        let base = base.trim_end_matches('/').to_owned();
        if !(base.starts_with("http://") || base.starts_with("https://")) {
            return Err(ServerError::Invalid(format!(
                "upstream must be an http:// or https:// URL, got {base:?}"
            )));
        }
        let http = reqwest::Client::builder()
            .connect_timeout(timeouts.connect)
            .tcp_nodelay(true)
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .map_err(|error| ServerError::Invalid(error.to_string()))?;
        Ok(Self {
            base,
            http,
            timeouts,
            model: None,
        })
    }

    /// Sends `key` as a bearer token with every request, for a server
    /// started with `--api-key`.
    pub fn with_api_key(mut self, key: &str) -> Result<Self, ServerError> {
        let mut headers = reqwest::header::HeaderMap::new();
        let value = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
            .map_err(|_| ServerError::Invalid("the API key is not a valid header value".into()))?;
        headers.insert(reqwest::header::AUTHORIZATION, value);
        self.http = reqwest::Client::builder()
            .connect_timeout(self.timeouts.connect)
            .tcp_nodelay(true)
            .pool_idle_timeout(Duration::from_secs(90))
            .default_headers(headers)
            .build()
            .map_err(|error| ServerError::Invalid(error.to_string()))?;
        Ok(self)
    }

    /// Addresses every call to `model`, for a server in router mode.
    pub fn for_model(mut self, model: Option<String>) -> Self {
        self.model = model.filter(|model| !model.is_empty());
        self
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    /// The router-mode model this client addresses, if any.
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// The shared HTTP client, for forwarding requests on the data plane.
    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    pub async fn health(&self) -> Health {
        let response = self
            .http
            .get(self.url("/health"))
            .timeout(self.timeouts.control.min(Duration::from_secs(10)))
            .send()
            .await;
        match response {
            Ok(response) if response.status().is_success() => Health::Ready,
            // llama-server answers 503 while it loads the model.
            Ok(response) if response.status().as_u16() == 503 => Health::Loading,
            Ok(_) | Err(_) => Health::Down,
        }
    }

    pub async fn info(&self) -> Result<ServerInfo, ServerError> {
        let props = self.get_json("/props").await?;
        let settings = props.get("default_generation_settings");
        let slots_from_props = props
            .get("total_slots")
            .and_then(Value::as_u64)
            .map(|slots| slots as usize);
        let context_from_props = settings
            .and_then(|settings| settings.get("n_ctx"))
            .and_then(Value::as_u64);
        // `/slots` is authoritative for both when it is enabled.
        let views = self.slots().await.ok();
        let slots = views
            .as_ref()
            .map(Vec::len)
            .filter(|count| *count > 0)
            .or(slots_from_props)
            .unwrap_or(1);
        let slot_context = views
            .as_ref()
            .and_then(|views| views.iter().filter_map(|view| view.context).max())
            .or(context_from_props)
            .ok_or_else(|| ServerError::Invalid("the server reports no context size".into()))?;
        let weights = self.weights().await;
        Ok(ServerInfo {
            weights,
            slots,
            slot_context,
            model: props
                .get("model_path")
                .or_else(|| props.get("model_alias"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            alias: props
                .get("model_alias")
                .and_then(Value::as_str)
                .filter(|alias| !alias.is_empty())
                .map(str::to_owned),
            build: props
                .get("build_info")
                .and_then(Value::as_str)
                .map(str::to_owned),
        })
    }

    /// Probes the optional endpoints the gate can use.
    pub async fn capabilities(&self) -> Capabilities {
        let slots = self.slots().await.is_ok();
        let tokenize = self.tokenize("probe").await.is_ok();
        let apply_template = self
            .apply_template(&json!({"messages": [{"role": "user", "content": "probe"}]}))
            .await
            .is_ok();
        Capabilities {
            slots,
            tokenize,
            apply_template,
        }
    }

    pub async fn slots(&self) -> Result<Vec<SlotView>, ServerError> {
        let value = self.get_json("/slots").await?;
        let entries = value
            .as_array()
            .ok_or_else(|| ServerError::Invalid("/slots is not an array".into()))?;
        Ok(entries
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                let next = entry.get("next_token");
                let next = next
                    .and_then(Value::as_array)
                    .and_then(|items| items.first())
                    .or(next);
                let decoded = next
                    .and_then(|next| next.get("n_decoded"))
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let prompt = entry.get("n_prompt_tokens").and_then(Value::as_u64);
                SlotView {
                    id: entry
                        .get("id")
                        .and_then(Value::as_u64)
                        .map_or(index, |id| id as usize),
                    busy: entry
                        .get("is_processing")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    context: entry.get("n_ctx").and_then(Value::as_u64),
                    task: entry.get("id_task").and_then(Value::as_i64),
                    tokens: prompt.map(|prompt| prompt + decoded),
                }
            })
            .collect())
    }

    /// Tokenizes a whole prompt the way the server tokenizes one it is
    /// given: special tokens added and parsed.
    pub async fn tokenize(&self, text: &str) -> Result<Vec<i32>, ServerError> {
        self.tokenize_with(text, true, true).await
    }

    /// Tokenizes `text`, adding special tokens (BOS) only with
    /// `add_special` and recognizing special-token text only with
    /// `parse_special`.
    pub async fn tokenize_with(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<i32>, ServerError> {
        let value = self
            .post_json(
                "/tokenize",
                &json!({"content": text, "add_special": add_special, "parse_special": parse_special}),
                self.timeouts.control,
            )
            .await?;
        Ok(Self::pieces_of(&value)?
            .into_iter()
            .map(|piece| piece.id)
            .collect())
    }

    /// Tokenizes `text` (no special tokens added) with each token's text.
    pub async fn pieces(&self, text: &str, parse_special: bool) -> Result<Vec<Piece>, ServerError> {
        let value = self
            .post_json(
                "/tokenize",
                &json!({"content": text, "add_special": false, "parse_special": parse_special, "with_pieces": true}),
                self.timeouts.control,
            )
            .await?;
        Self::pieces_of(&value)
    }

    fn pieces_of(value: &Value) -> Result<Vec<Piece>, ServerError> {
        value
            .get("tokens")
            .and_then(Value::as_array)
            .ok_or_else(|| ServerError::Invalid("/tokenize returned no tokens".into()))?
            .iter()
            .map(|token| {
                let id = token
                    .as_i64()
                    .or_else(|| token.get("id").and_then(Value::as_i64))
                    .map(|token| token as i32)
                    .ok_or_else(|| ServerError::Invalid("non-numeric token".into()))?;
                let text = token
                    .get("piece")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                Ok(Piece { id, text })
            })
            .collect()
    }

    /// Renders a chat request through the model's own template.
    pub async fn apply_template(&self, body: &Value) -> Result<String, ServerError> {
        let value = self
            .post_json("/apply-template", body, self.timeouts.control)
            .await?;
        value
            .get("prompt")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| ServerError::Invalid("/apply-template returned no prompt".into()))
    }

    pub async fn save(&self, slot: usize, filename: &str) -> Result<SlotReport, ServerError> {
        let value = self
            .post_json(
                &format!("/slots/{slot}?action=save"),
                &json!({"filename": filename}),
                self.timeouts.slot_io,
            )
            .await?;
        Ok(SlotReport {
            tokens: value.get("n_saved").and_then(Value::as_u64),
            bytes: value.get("n_written").and_then(Value::as_u64).unwrap_or(0),
            millis: value
                .pointer("/timings/save_ms")
                .and_then(Value::as_f64)
                .unwrap_or(0.0),
        })
    }

    pub async fn restore(&self, slot: usize, filename: &str) -> Result<SlotReport, ServerError> {
        let value = self
            .post_json(
                &format!("/slots/{slot}?action=restore"),
                &json!({"filename": filename}),
                self.timeouts.slot_io,
            )
            .await?;
        Ok(SlotReport {
            tokens: value.get("n_restored").and_then(Value::as_u64),
            bytes: value.get("n_read").and_then(Value::as_u64).unwrap_or(0),
            millis: value
                .pointer("/timings/restore_ms")
                .and_then(Value::as_f64)
                .unwrap_or(0.0),
        })
    }

    /// Clears a slot's context. Returns the tokens erased.
    pub async fn erase(&self, slot: usize) -> Result<u64, ServerError> {
        let value = self
            .post_json(
                &format!("/slots/{slot}?action=erase"),
                &json!({}),
                self.timeouts.control,
            )
            .await?;
        Ok(value.get("n_erased").and_then(Value::as_u64).unwrap_or(0))
    }

    /// The URL of a GET of `path`, carrying the router-mode model as a query
    /// parameter.
    fn url(&self, path: &str) -> String {
        match &self.model {
            Some(model) => {
                let separator = if path.contains('?') { '&' } else { '?' };
                format!("{}{path}{separator}model={}", self.base, encode(model))
            }
            None => format!("{}{path}", self.base),
        }
    }

    /// The loaded model's size and shape, from `/v1/models` meta, when the
    /// server reports them.
    async fn weights(&self) -> Option<String> {
        let models = self.get_json("/v1/models").await.ok()?;
        let entries = models.get("data")?.as_array()?;
        let entry = match &self.model {
            Some(model) => entries
                .iter()
                .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model.as_str()))?,
            None => entries.first()?,
        };
        let meta = entry.get("meta")?;
        let field = |key: &str| meta.get(key).map_or_else(String::new, Value::to_string);
        Some(
            ["size", "n_params", "n_embd", "n_vocab", "n_ctx_train"]
                .iter()
                .map(|key| field(key))
                .collect::<Vec<_>>()
                .join("/"),
        )
    }

    async fn get_json(&self, path: &str) -> Result<Value, ServerError> {
        let request = self.http.get(self.url(path)).timeout(self.timeouts.control);
        Self::read(request.send().await).await
    }

    async fn post_json(
        &self,
        path: &str,
        body: &Value,
        timeout: Duration,
    ) -> Result<Value, ServerError> {
        let mut body = body.clone();
        if let (Some(model), Some(object)) = (&self.model, body.as_object_mut()) {
            object
                .entry("model")
                .or_insert_with(|| Value::String(model.clone()));
        }
        // The router routes a POST by its body's `model`, not the query.
        let request = self
            .http
            .post(format!("{}{path}", self.base))
            .timeout(timeout)
            .json(&body);
        Self::read(request.send().await).await
    }

    async fn read(
        response: Result<reqwest::Response, reqwest::Error>,
    ) -> Result<Value, ServerError> {
        let response = response.map_err(|error| ServerError::Unreachable(error.to_string()))?;
        let status = response.status().as_u16();
        let body = response
            .bytes()
            .await
            .map_err(|error| ServerError::Unreachable(error.to_string()))?;
        let value: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
        if (200..300).contains(&status) {
            return Ok(value);
        }
        let message = value
            .pointer("/error/message")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| String::from_utf8_lossy(&body).chars().take(300).collect());
        let kind = value.pointer("/error/type").and_then(Value::as_str);
        if status == 501 || status == 404 || kind == Some("not_supported_error") {
            return Err(ServerError::Unsupported(message));
        }
        Err(ServerError::Status { status, message })
    }
}

/// Percent-encodes a query value (RFC 3986 unreserved characters pass).
fn encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_mode_addresses_the_model_in_queries() {
        let server = Server::new("http://llama:8080/", Timeouts::default())
            .unwrap()
            .for_model(Some("ggml-org/gemma-3:Q4_K_M".into()));
        assert_eq!(
            server.url("/props"),
            "http://llama:8080/props?model=ggml-org%2Fgemma-3%3AQ4_K_M"
        );
        assert_eq!(
            server.url("/slots?fields=all"),
            "http://llama:8080/slots?fields=all&model=ggml-org%2Fgemma-3%3AQ4_K_M"
        );
        let plain = Server::new("http://llama:8080", Timeouts::default()).unwrap();
        assert_eq!(plain.url("/props"), "http://llama:8080/props");
    }
}
