//! Turning a request body into tokens.
use crate::tokens::{Boundaries, Method, TokenCache, Tokenized};
use crate::{Capabilities, Dialect, Prompt, Server};
use gate_core::{Prefix, Tokens};
use serde_json::Value;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct MeasurerConfig {
    /// Tokens per content-address block (llm-d uses 64).
    pub block: usize,
    /// Tokens assumed per image or audio item, which text tokenization
    /// cannot see. Overestimating only costs concurrency.
    pub media_tokens: Tokens,
    /// Bytes per token assumed when the server cannot tokenize. Real text
    /// runs at three to four bytes per token, so two overestimates safely.
    pub bytes_per_token: f64,
    /// Separates content addresses between models (see [`Prefix`]).
    pub namespace: String,
}

impl Default for MeasurerConfig {
    fn default() -> Self {
        Self {
            block: 64,
            media_tokens: 4096,
            bytes_per_token: 2.0,
            namespace: String::new(),
        }
    }
}

/// What a request needs, as far as it can be known before running it.
#[derive(Clone, Debug, PartialEq)]
pub struct Measurement {
    /// Prompt tokens one of its tasks holds at once: for several prompts,
    /// the largest.
    pub prompt: Tokens,
    /// Prompt tokens the request processes in total.
    pub work: Tokens,
    pub prefix: Prefix,
    /// The prompt was tokenized by the server rather than estimated.
    pub exact: bool,
    /// How an exact prompt was tokenized.
    pub method: Option<Method>,
    /// The token cache disagreed with the server and turned itself off for
    /// this model; the measurement itself is exact.
    pub mismatch: bool,
    pub requested_generation: Option<Tokens>,
    /// Several prompts or choices: the request needs every slot.
    pub exclusive: bool,
    pub streaming: bool,
    /// The request generates tokens (an embedding does not).
    pub generates: bool,
    /// The server honours `id_slot` for it.
    pub pinned: bool,
}

#[derive(Clone, Debug)]
pub struct Measurer {
    server: Server,
    capabilities: Capabilities,
    config: MeasurerConfig,
    cache: Arc<TokenCache>,
    boundaries: Option<Arc<Boundaries>>,
}

impl Measurer {
    pub fn new(server: Server, capabilities: Capabilities, config: MeasurerConfig) -> Self {
        Self {
            server,
            capabilities,
            config,
            cache: Arc::new(TokenCache::default()),
            boundaries: None,
        }
    }

    /// Shares a token cache with other measurers (other backends).
    pub fn with_cache(mut self, cache: Arc<TokenCache>) -> Self {
        self.cache = cache;
        self
    }

    pub fn set_capabilities(&mut self, capabilities: Capabilities) {
        self.capabilities = capabilities;
    }

    pub fn set_namespace(&mut self, namespace: String) {
        self.config.namespace = namespace;
    }

    /// Where the model's template may be cut for incremental tokenization;
    /// `None` tokenizes every prompt whole.
    pub fn set_boundaries(&mut self, boundaries: Option<Arc<Boundaries>>) {
        self.boundaries = boundaries;
    }

    pub fn server(&self) -> &Server {
        &self.server
    }

    pub fn namespace(&self) -> &str {
        &self.config.namespace
    }

    /// Measures `body` as `dialect` reads it. `size` is the body's length in
    /// bytes, used when the prompt cannot be tokenized.
    pub async fn measure(&self, dialect: &dyn Dialect, body: &Value, size: usize) -> Measurement {
        self.measure_in(dialect, body, size, None).await
    }

    /// Like [`Measurer::measure`], with the content address confined to
    /// `scope` (say, one client), so it matches nothing outside it.
    pub async fn measure_in(
        &self,
        dialect: &dyn Dialect,
        body: &Value,
        size: usize,
        scope: Option<&str>,
    ) -> Measurement {
        let media = dialect.media(body).saturating_mul(self.config.media_tokens);
        let generates = dialect.generates();
        let mut measurement = Measurement {
            prompt: 0,
            work: 0,
            prefix: Prefix::opaque(0),
            exact: false,
            method: None,
            mismatch: false,
            requested_generation: generates
                .then(|| dialect.requested_generation(body))
                .flatten(),
            exclusive: dialect.choices(body) > 1,
            streaming: dialect.streaming(body),
            generates,
            pinned: dialect.pinnable(),
        };
        match dialect.prompt(body) {
            Prompt::Many(items) if items.len() > 1 => {
                // Separate tasks: the largest bounds what one slot holds, all
                // of them are the work done. None is worth addressing.
                measurement.exclusive = true;
                let share = size / items.len();
                let mut exact = true;
                for item in items {
                    let size = Self::size_of(&item).unwrap_or(share);
                    let tokens = match self.tokens(item).await {
                        Some(tokenized) => tokenized.tokens.len() as Tokens,
                        None => {
                            exact = false;
                            self.estimate(size)
                        }
                    };
                    measurement.prompt = measurement.prompt.max(tokens);
                    measurement.work += tokens;
                }
                measurement.prompt += media;
                measurement.work += media;
                measurement.exact = exact;
                measurement.prefix = Prefix::opaque(measurement.prompt);
            }
            prompt => match self.tokens(prompt).await {
                Some(tokenized) => {
                    let namespace = match scope {
                        Some(scope) => format!("{}\u{1f}{scope}", self.config.namespace),
                        None => self.config.namespace.clone(),
                    };
                    measurement.prefix =
                        Prefix::from_tokens_in(&namespace, &tokenized.tokens, self.config.block);
                    measurement.prompt = tokenized.tokens.len() as Tokens + media;
                    measurement.exact = true;
                    measurement.method = Some(tokenized.method);
                    measurement.mismatch = tokenized.mismatch;
                }
                None => {
                    measurement.prompt = self.estimate(size) + media;
                    measurement.prefix = Prefix::opaque(measurement.prompt);
                }
            },
        }
        if measurement.work == 0 {
            measurement.work = measurement.prompt;
        }
        measurement
    }

    /// Exact tokens for one prompt, when the server can provide them.
    async fn tokens(&self, prompt: Prompt) -> Option<Tokenized> {
        match prompt {
            Prompt::Tokens(tokens) => Some(Tokenized {
                tokens,
                method: Method::Full,
                mismatch: false,
            }),
            Prompt::Text(text) if self.capabilities.tokenize => {
                let tokens = self.server.tokenize(&text).await.ok()?;
                Some(Tokenized {
                    tokens,
                    method: Method::Full,
                    mismatch: false,
                })
            }
            Prompt::Chat(chat)
                if self.capabilities.apply_template && self.capabilities.tokenize =>
            {
                let rendered = self.server.apply_template(&text_only(chat)).await.ok()?;
                self.cache
                    .tokenize(
                        &self.server,
                        self.boundaries.as_deref(),
                        &self.config.namespace,
                        &rendered,
                    )
                    .await
                    .ok()
            }
            Prompt::Many(prompts) => {
                // A single-element list; anything longer is split by the caller.
                let mut total = Vec::new();
                for prompt in prompts {
                    total.extend(Box::pin(self.tokens(prompt)).await?.tokens);
                }
                Some(Tokenized {
                    tokens: total,
                    method: Method::Full,
                    mismatch: false,
                })
            }
            _ => None,
        }
    }

    /// Bytes of a prompt's text, for estimating it.
    fn size_of(prompt: &Prompt) -> Option<usize> {
        match prompt {
            Prompt::Text(text) => Some(text.len()),
            Prompt::Tokens(tokens) => Some(tokens.len() * 2),
            Prompt::Chat(chat) => Some(chat.to_string().len()),
            _ => None,
        }
    }

    fn estimate(&self, size: usize) -> Tokens {
        let ratio = if self.config.bytes_per_token > 0.0 {
            self.config.bytes_per_token
        } else {
            1.0
        };
        (size as f64 / ratio).ceil() as Tokens
    }
}

/// The chat with media parts removed, so rendering the template never has
/// to fetch or decode an image; media are counted separately.
fn text_only(mut chat: Value) -> Value {
    if let Some(messages) = chat.get_mut("messages").and_then(Value::as_array_mut) {
        for message in messages {
            if let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) {
                parts.retain(|part| {
                    matches!(
                        part.get("type").and_then(Value::as_str),
                        None | Some("text")
                    )
                });
            }
        }
    }
    // Generation settings do not change the rendered prompt.
    if let Some(object) = chat.as_object_mut() {
        object.remove("stream");
    }
    chat
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Embeddings, OpenAiChat, Rerank, Timeouts};
    use serde_json::json;

    async fn fake_server() -> String {
        use axum::{Json, Router, routing::post};
        let app = Router::new()
            .route(
                "/apply-template",
                post(|Json(body): Json<Value>| async move {
                    let parts = body["messages"][0]["content"]
                        .as_array()
                        .map_or(0, Vec::len);
                    Json(json!({"prompt": format!("rendered with {parts} parts")}))
                }),
            )
            .route(
                "/tokenize",
                post(|Json(body): Json<Value>| async move {
                    let words = body["content"]
                        .as_str()
                        .unwrap_or("")
                        .split_whitespace()
                        .count();
                    Json(json!({"tokens": (0..words as i64).collect::<Vec<_>>()}))
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{address}")
    }

    fn capable() -> Capabilities {
        Capabilities {
            slots: false,
            tokenize: true,
            apply_template: true,
        }
    }

    #[tokio::test]
    async fn chats_are_rendered_without_media_and_media_are_counted_apart() {
        let base = fake_server().await;
        let server = Server::new(&base, Timeouts::default()).unwrap();
        let measurer = Measurer::new(server, capable(), MeasurerConfig::default());
        let body = json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        ]}], "max_tokens": 77, "stream": true});
        let measurement = measurer.measure(&OpenAiChat, &body, 200).await;
        // "rendered with 1 parts" is four words: the image part was removed
        // before rendering, then counted at the configured media size.
        assert_eq!(measurement.prompt, 4 + 4096);
        assert_eq!(measurement.work, measurement.prompt);
        assert!(measurement.exact);
        assert_eq!(measurement.method, Some(Method::Full));
        assert_eq!(measurement.requested_generation, Some(77));
        assert!(measurement.streaming && measurement.generates && measurement.pinned);
    }

    #[tokio::test]
    async fn several_inputs_hold_their_largest_and_charge_their_total() {
        let base = fake_server().await;
        let server = Server::new(&base, Timeouts::default()).unwrap();
        let measurer = Measurer::new(server, capable(), MeasurerConfig::default());
        let body = json!({"input": ["one two three", "four", "five six"]});
        let measurement = measurer.measure(&Embeddings, &body, 100).await;
        assert_eq!((measurement.prompt, measurement.work), (3, 6));
        assert!(measurement.exclusive && !measurement.pinned && !measurement.generates);
        assert!(measurement.prefix.is_opaque());
        assert_eq!(measurement.requested_generation, None);

        let ranked = measurer
            .measure(
                &Rerank,
                &json!({"query": "q", "documents": ["a b", "c"]}),
                50,
            )
            .await;
        assert_eq!((ranked.prompt, ranked.work), (3, 5));
    }

    #[tokio::test]
    async fn without_tokenization_the_estimate_is_conservative_and_opaque() {
        let server = Server::new("http://127.0.0.1:9", Timeouts::default()).unwrap();
        let measurer = Measurer::new(server, Capabilities::default(), MeasurerConfig::default());
        let body = json!({"messages": [{"role": "user", "content": "hello there"}]});
        let measurement = measurer.measure(&OpenAiChat, &body, 1_000).await;
        assert_eq!(measurement.prompt, 500);
        assert!(!measurement.exact);
        assert!(measurement.prefix.is_opaque());
    }
}
