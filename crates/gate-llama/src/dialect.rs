//! Request formats a llama.cpp server accepts.
//!
//! A [`Dialect`] knows where one API keeps its prompt and its generation
//! limit, and where the server's own options (`id_slot`, `cache_prompt`) go;
//! it never talks to the server. Register a new implementation in
//! [`Dialects`] to put another endpoint under the gate.
use crate::stream::Wire;
use serde_json::{Map, Value, json};
use std::sync::Arc;

/// Where a request's prompt is, as far as its dialect can tell.
#[derive(Clone, Debug, PartialEq)]
pub enum Prompt {
    /// A chat conversation, already in OpenAI chat-completions shape, to be
    /// rendered through the model's template.
    Chat(Value),
    /// Plain prompt text.
    Text(String),
    /// Pre-tokenized prompt.
    Tokens(Vec<i32>),
    /// Several independent prompts in one request, which the server runs as
    /// separate tasks.
    Many(Vec<Prompt>),
    /// Not recognized; the body size is used as a conservative estimate.
    Unknown,
}

pub trait Dialect: Send + Sync + std::fmt::Debug {
    /// Stable name, for logs and metrics.
    fn name(&self) -> &'static str;

    /// Request paths this dialect serves.
    fn paths(&self) -> &'static [&'static str];

    fn prompt(&self, body: &Value) -> Prompt;

    /// Media items (images, audio) whose tokens text tokenization cannot see.
    fn media(&self, _body: &Value) -> u64 {
        0
    }

    /// Whether requests generate tokens. Embedding and ranking requests only
    /// read their input.
    fn generates(&self) -> bool {
        true
    }

    /// The generation limit the server will apply, as it resolves the
    /// request's fields; `None` when there is none (absent, or negative,
    /// which llama.cpp reads as "until the context is full").
    fn requested_generation(&self, body: &Value) -> Option<u64>;

    /// Caps generation at `tokens`, in the field the server reads first, so
    /// no other field the client set can lift it.
    fn limit_generation(&self, body: &mut Value, tokens: u64);

    /// Completions requested at once. llama.cpp reads `n_cmpl` before
    /// OpenAI's `n` and runs the extra ones in other free slots.
    fn choices(&self, body: &Value) -> u64 {
        ["n_cmpl", "n"]
            .iter()
            .find_map(|key| body.get(*key).and_then(Value::as_u64))
            .unwrap_or(1)
            .max(1)
    }

    fn streaming(&self, body: &Value) -> bool {
        body.get("stream").and_then(Value::as_bool).unwrap_or(false)
    }

    /// Requests token usage on streamed responses when the client did not
    /// already ask for it; returns `true` if `body` was modified.
    fn ensure_stream_usage(&self, _body: &mut Value) -> bool {
        false
    }

    /// How its responses are shaped: what counts as output in its stream,
    /// and where its token counts are. A dialect that does not say is read
    /// conservatively (any event is output).
    fn wire(&self) -> Wire {
        Wire::Opaque
    }

    /// An error as this API reports one inside a stream that has begun
    /// (the status line is already sent), framed as a server-sent event.
    fn error_event(&self, kind: &str, message: &str) -> String {
        match self.wire() {
            // `event: error` with Anthropic's error envelope.
            Wire::Anthropic => format!(
                "event: error\ndata: {}\n\n",
                json!({"type": "error", "error": {"type": kind, "message": message}})
            ),
            // The Responses API's typed `error` event.
            Wire::Responses => format!(
                "event: error\ndata: {}\n\n",
                json!({"type": "error", "code": kind, "message": message, "param": null})
            ),
            // OpenAI and llama.cpp: an `error` object, as llama.cpp sends.
            _ => format!(
                "data: {}\n\n",
                json!({"error": {"type": kind, "message": message}})
            ),
        }
    }

    /// Whether the server runs the request in the slot named by `id_slot`.
    /// For endpoints that pick a slot themselves, the gate reserves room
    /// that holds wherever the server puts the request.
    fn pinnable(&self) -> bool {
        true
    }

    /// Asks the server to run the request in `slot` and to reuse that slot's
    /// cache.
    fn pin(&self, body: &mut Value, slot: usize) {
        pin_fields(body, slot);
    }
}

/// llama.cpp's slot options at the top level of the body.
fn pin_fields(body: &mut Value, slot: usize) {
    let Some(object) = body.as_object_mut() else {
        return;
    };
    object.insert("id_slot".into(), json!(slot));
    object.entry("cache_prompt").or_insert(Value::Bool(true));
}

/// The dialects the gate understands, looked up by request path.
#[derive(Clone, Debug, Default)]
pub struct Dialects {
    entries: Vec<Arc<dyn Dialect>>,
}

impl Dialects {
    /// Every dialect built into llama-gate.
    pub fn standard() -> Self {
        Self::default()
            .with(OpenAiChat)
            .with(OpenAiCompletions)
            .with(OpenAiResponses)
            .with(NativeCompletion)
            .with(Infill)
            .with(AnthropicMessages)
            .with(Embeddings)
            .with(Rerank)
    }

    /// Adds a dialect. A later dialect wins for a path both serve.
    pub fn with(mut self, dialect: impl Dialect + 'static) -> Self {
        self.entries.push(Arc::new(dialect));
        self
    }

    /// Keeps only the dialects for which `keep` holds.
    pub fn retain(mut self, keep: impl Fn(&dyn Dialect) -> bool) -> Self {
        self.entries.retain(|dialect| keep(dialect.as_ref()));
        self
    }

    pub fn find(&self, path: &str) -> Option<Arc<dyn Dialect>> {
        self.entries
            .iter()
            .rev()
            .find(|dialect| dialect.paths().contains(&path))
            .cloned()
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.entries.iter().map(|dialect| dialect.name()).collect()
    }
}

/// The fields llama.cpp reads a generation limit from, first present wins
/// (`server-schema.cpp`): every completion endpoint shares them.
const LIMIT_FIELDS: [&str; 3] = ["n_predict", "max_completion_tokens", "max_tokens"];

/// The limit set by the first of `fields` present in `object`: `Some(None)`
/// for a negative integer (no limit), `None` when none is present.
fn first_limit(object: &Value, fields: &[&str]) -> Option<Option<u64>> {
    fields.iter().find_map(|key| {
        let value = object.get(*key)?;
        if let Some(limit) = value.as_u64() {
            Some(Some(limit))
        } else if value.as_i64().is_some() {
            // Negative integers mean "until context is full" in llama.cpp.
            Some(None)
        } else {
            None
        }
    })
}

/// A completion endpoint's limit, as the server resolves it.
fn completion_limit(body: &Value) -> Option<u64> {
    first_limit(body, &LIMIT_FIELDS).flatten()
}

/// Sets the limit no other field can override: `n_predict` is read first.
fn set_completion_limit(body: &mut Value, tokens: u64) {
    if let Some(object) = body.as_object_mut() {
        object.insert("n_predict".into(), json!(tokens));
    }
}

/// The limit of an API llama.cpp converts to a chat completion (Responses,
/// Anthropic): its own field is converted first and `extra_params` are
/// merged over the result, so theirs win (`server-chat.cpp`).
fn converted_limit(body: &Value, own: &str) -> Option<u64> {
    body.get("extra_params")
        .and_then(|extra| first_limit(extra, &LIMIT_FIELDS))
        .or_else(|| first_limit(body, &[own]))
        .flatten()
}

fn set_converted_limit(body: &mut Value, tokens: u64) {
    let Some(object) = body.as_object_mut() else {
        return;
    };
    let extra = object.entry("extra_params").or_insert_with(|| json!({}));
    if !extra.is_object() {
        *extra = json!({});
    }
    set_completion_limit(extra, tokens);
}

fn tokens_of(value: &Value) -> Option<Vec<i32>> {
    value
        .as_array()?
        .iter()
        .map(|token| token.as_i64().map(|token| token as i32))
        .collect()
}

/// Reads llama.cpp's flexible `prompt` field: text, tokens, a mix, a
/// multimodal object, or a list of any of those.
fn prompt_field(value: Option<&Value>) -> Prompt {
    let Some(value) = value else {
        return Prompt::Unknown;
    };
    if let Some(text) = value.as_str() {
        return Prompt::Text(text.to_owned());
    }
    if let Some(tokens) = tokens_of(value) {
        return Prompt::Tokens(tokens);
    }
    if let Some(text) = value.get("prompt_string").and_then(Value::as_str) {
        return Prompt::Text(text.to_owned());
    }
    if let Some(parts) = value.get("content").and_then(Value::as_array) {
        return Prompt::Text(part_texts(parts));
    }
    if let Some(items) = value.as_array() {
        let prompts: Vec<Prompt> = items.iter().map(|item| prompt_field(Some(item))).collect();
        return match prompts.len() {
            1 => prompts.into_iter().next().unwrap_or(Prompt::Unknown),
            _ => Prompt::Many(prompts),
        };
    }
    Prompt::Unknown
}

/// The text of content parts, in order; other parts are counted as media.
fn part_texts(parts: &[Value]) -> String {
    parts
        .iter()
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Whether a content part is something other than text.
fn is_media(part: &Value) -> bool {
    !matches!(
        part.get("type").and_then(Value::as_str),
        None | Some("text" | "input_text" | "output_text" | "refusal")
    )
}

/// Counts non-text content parts in OpenAI-style messages.
fn media_parts(messages: Option<&Value>) -> u64 {
    messages.and_then(Value::as_array).map_or(0, |messages| {
        messages
            .iter()
            .filter_map(|message| message.get("content").and_then(Value::as_array))
            .flatten()
            .filter(|part| is_media(part))
            .count() as u64
    })
}

/// Ensures `stream_options.include_usage` is `true`, preserving any other keys
/// in an existing `stream_options` object. Returns `true` if `body` changed.
fn inject_stream_usage(body: &mut Value) -> bool {
    let Some(object) = body.as_object_mut() else {
        return false;
    };
    if object
        .get("stream_options")
        .and_then(|options| options.get("include_usage"))
        .and_then(Value::as_bool)
        == Some(true)
    {
        return false;
    }
    let options = object.entry("stream_options").or_insert_with(|| json!({}));
    if !options.is_object() {
        *options = json!({});
    }
    if let Some(map) = options.as_object_mut() {
        map.insert("include_usage".into(), Value::Bool(true));
    }
    true
}

/// OpenAI chat completions (`/v1/chat/completions`).
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAiChat;

impl Dialect for OpenAiChat {
    fn name(&self) -> &'static str {
        "openai_chat"
    }

    fn wire(&self) -> Wire {
        Wire::OpenAiChat
    }

    fn paths(&self) -> &'static [&'static str] {
        &["/v1/chat/completions", "/chat/completions"]
    }

    fn prompt(&self, body: &Value) -> Prompt {
        Prompt::Chat(body.clone())
    }

    fn media(&self, body: &Value) -> u64 {
        media_parts(body.get("messages"))
    }

    fn requested_generation(&self, body: &Value) -> Option<u64> {
        completion_limit(body)
    }

    fn limit_generation(&self, body: &mut Value, tokens: u64) {
        set_completion_limit(body, tokens);
    }

    fn ensure_stream_usage(&self, body: &mut Value) -> bool {
        self.streaming(body) && inject_stream_usage(body)
    }
}

/// OpenAI legacy completions (`/v1/completions`).
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAiCompletions;

impl Dialect for OpenAiCompletions {
    fn name(&self) -> &'static str {
        "openai_completions"
    }

    fn wire(&self) -> Wire {
        Wire::OpenAiCompletion
    }

    fn paths(&self) -> &'static [&'static str] {
        &["/v1/completions"]
    }

    fn prompt(&self, body: &Value) -> Prompt {
        prompt_field(body.get("prompt"))
    }

    fn requested_generation(&self, body: &Value) -> Option<u64> {
        completion_limit(body)
    }

    fn limit_generation(&self, body: &mut Value, tokens: u64) {
        set_completion_limit(body, tokens);
    }

    fn ensure_stream_usage(&self, body: &mut Value) -> bool {
        self.streaming(body) && inject_stream_usage(body)
    }
}

/// OpenAI responses (`/v1/responses`), which llama.cpp serves by turning the
/// request into a chat completion. Its llama.cpp options travel in
/// `extra_params`.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAiResponses;

impl OpenAiResponses {
    /// The same conversation in chat-completions shape, for the template.
    fn as_chat(body: &Value) -> Value {
        let mut messages = Vec::new();
        if let Some(instructions) = body.get("instructions").and_then(Value::as_str) {
            messages.push(json!({"role": "system", "content": instructions}));
        }
        match body.get("input") {
            Some(Value::String(text)) => messages.push(json!({"role": "user", "content": text})),
            Some(Value::Array(items)) => {
                for item in items {
                    match item.get("type").and_then(Value::as_str) {
                        Some("function_call") => messages.push(json!({
                            "role": "assistant",
                            "content": null,
                            "tool_calls": [{
                                "id": item.get("call_id").cloned().unwrap_or(Value::Null),
                                "type": "function",
                                "function": {
                                    "name": item.get("name").cloned().unwrap_or(Value::Null),
                                    "arguments": item.get("arguments").cloned().unwrap_or_default(),
                                },
                            }],
                        })),
                        Some("function_call_output") => messages.push(json!({
                            "role": "tool",
                            "tool_call_id": item.get("call_id").cloned().unwrap_or(Value::Null),
                            "content": match item.get("output") {
                                Some(Value::String(text)) => text.clone(),
                                Some(other) => other.to_string(),
                                None => String::new(),
                            },
                        })),
                        // Reasoning items are not rendered back into prompts.
                        Some("reasoning") => {}
                        _ => {
                            let role = match item.get("role").and_then(Value::as_str) {
                                Some("developer") => "system",
                                Some(role) => role,
                                None => "user",
                            };
                            let content = match item.get("content") {
                                Some(Value::String(text)) => text.clone(),
                                Some(Value::Array(parts)) => part_texts(parts),
                                _ => String::new(),
                            };
                            messages.push(json!({"role": role, "content": content}));
                        }
                    }
                }
            }
            _ => {}
        }
        let mut chat = Map::new();
        chat.insert("messages".into(), Value::Array(messages));
        if let Some(tools) = body.get("tools").and_then(Value::as_array) {
            let tools: Vec<Value> = tools
                .iter()
                .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("function"))
                .map(|tool| {
                    json!({"type": "function", "function": {
                        "name": tool.get("name").cloned().unwrap_or(Value::Null),
                        "description": tool.get("description").cloned().unwrap_or(Value::Null),
                        "parameters": tool.get("parameters").cloned().unwrap_or(json!({})),
                    }})
                })
                .collect();
            chat.insert("tools".into(), Value::Array(tools));
        }
        Value::Object(chat)
    }
}

impl Dialect for OpenAiResponses {
    fn name(&self) -> &'static str {
        "openai_responses"
    }

    fn wire(&self) -> Wire {
        Wire::Responses
    }

    fn paths(&self) -> &'static [&'static str] {
        &["/v1/responses", "/responses"]
    }

    fn prompt(&self, body: &Value) -> Prompt {
        Prompt::Chat(Self::as_chat(body))
    }

    fn media(&self, body: &Value) -> u64 {
        body.get("input")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|item| item.get("content").and_then(Value::as_array))
            .flatten()
            .filter(|part| is_media(part))
            .count() as u64
    }

    fn requested_generation(&self, body: &Value) -> Option<u64> {
        converted_limit(body, "max_output_tokens")
    }

    fn limit_generation(&self, body: &mut Value, tokens: u64) {
        set_converted_limit(body, tokens);
    }

    fn choices(&self, _body: &Value) -> u64 {
        1
    }

    fn pin(&self, body: &mut Value, slot: usize) {
        let Some(object) = body.as_object_mut() else {
            return;
        };
        let extra = object.entry("extra_params").or_insert_with(|| json!({}));
        if !extra.is_object() {
            *extra = json!({});
        }
        pin_fields(extra, slot);
    }
}

/// llama.cpp's native completion (`/completion`).
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeCompletion;

impl Dialect for NativeCompletion {
    fn name(&self) -> &'static str {
        "llama_completion"
    }

    fn wire(&self) -> Wire {
        Wire::Llama
    }

    fn paths(&self) -> &'static [&'static str] {
        &["/completion", "/completions"]
    }

    fn prompt(&self, body: &Value) -> Prompt {
        prompt_field(body.get("prompt"))
    }

    fn media(&self, body: &Value) -> u64 {
        body.get("prompt")
            .and_then(|prompt| prompt.get("multimodal_data"))
            .and_then(Value::as_array)
            .map_or(0, |items| items.len() as u64)
    }

    fn requested_generation(&self, body: &Value) -> Option<u64> {
        completion_limit(body)
    }

    fn limit_generation(&self, body: &mut Value, tokens: u64) {
        set_completion_limit(body, tokens);
    }
}

/// llama.cpp's fill-in-the-middle endpoint (`/infill`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Infill;

impl Dialect for Infill {
    fn name(&self) -> &'static str {
        "llama_infill"
    }

    fn wire(&self) -> Wire {
        Wire::Llama
    }

    fn paths(&self) -> &'static [&'static str] {
        &["/infill"]
    }

    fn prompt(&self, body: &Value) -> Prompt {
        let mut text = String::new();
        if let Some(extra) = body.get("input_extra").and_then(Value::as_array) {
            for chunk in extra {
                for key in ["filename", "text"] {
                    if let Some(part) = chunk.get(key).and_then(Value::as_str) {
                        text.push_str(part);
                        text.push('\n');
                    }
                }
            }
        }
        for key in ["input_prefix", "prompt", "input_suffix"] {
            if let Some(part) = body.get(key).and_then(Value::as_str) {
                text.push_str(part);
            }
        }
        Prompt::Text(text)
    }

    fn requested_generation(&self, body: &Value) -> Option<u64> {
        completion_limit(body)
    }

    fn limit_generation(&self, body: &mut Value, tokens: u64) {
        set_completion_limit(body, tokens);
    }
}

/// Anthropic messages (`/v1/messages`), which llama.cpp also serves.
#[derive(Clone, Copy, Debug, Default)]
pub struct AnthropicMessages;

impl AnthropicMessages {
    /// The same conversation in OpenAI chat shape, for the chat template.
    fn as_chat(body: &Value) -> Value {
        fn text(content: &Value) -> String {
            match content {
                Value::String(text) => text.clone(),
                Value::Array(blocks) => blocks
                    .iter()
                    .map(|block| match block.get("type").and_then(Value::as_str) {
                        Some("text") => block
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        Some("image" | "document") => String::new(),
                        _ => block.to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
                other => other.to_string(),
            }
        }
        let mut messages = Vec::new();
        if let Some(system) = body.get("system") {
            messages.push(json!({"role": "system", "content": text(system)}));
        }
        for message in body
            .get("messages")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            messages.push(json!({
                "role": message.get("role").and_then(Value::as_str).unwrap_or("user"),
                "content": message.get("content").map(text).unwrap_or_default(),
            }));
        }
        let mut chat = Map::new();
        chat.insert("messages".into(), Value::Array(messages));
        if let Some(tools) = body.get("tools").and_then(Value::as_array) {
            let tools: Vec<Value> = tools
                .iter()
                .map(|tool| {
                    json!({"type": "function", "function": {
                        "name": tool.get("name").cloned().unwrap_or(Value::Null),
                        "description": tool.get("description").cloned().unwrap_or(Value::Null),
                        "parameters": tool.get("input_schema").cloned().unwrap_or(json!({})),
                    }})
                })
                .collect();
            chat.insert("tools".into(), Value::Array(tools));
        }
        Value::Object(chat)
    }
}

impl Dialect for AnthropicMessages {
    fn name(&self) -> &'static str {
        "anthropic_messages"
    }

    fn wire(&self) -> Wire {
        Wire::Anthropic
    }

    fn paths(&self) -> &'static [&'static str] {
        &["/v1/messages"]
    }

    fn prompt(&self, body: &Value) -> Prompt {
        Prompt::Chat(Self::as_chat(body))
    }

    fn media(&self, body: &Value) -> u64 {
        body.get("messages")
            .and_then(Value::as_array)
            .map_or(0, |messages| {
                messages
                    .iter()
                    .filter_map(|message| message.get("content").and_then(Value::as_array))
                    .flatten()
                    .filter(|block| {
                        matches!(
                            block.get("type").and_then(Value::as_str),
                            Some("image" | "document")
                        )
                    })
                    .count() as u64
            })
    }

    fn requested_generation(&self, body: &Value) -> Option<u64> {
        converted_limit(body, "max_tokens")
    }

    fn limit_generation(&self, body: &mut Value, tokens: u64) {
        set_converted_limit(body, tokens);
    }

    fn choices(&self, _body: &Value) -> u64 {
        1
    }
}

/// Embeddings: OpenAI's `/v1/embeddings` (`input`) and llama.cpp's native
/// `/embedding` (`content`). They occupy slots but generate nothing, and the
/// server picks their slot itself.
#[derive(Clone, Copy, Debug, Default)]
pub struct Embeddings;

impl Dialect for Embeddings {
    fn name(&self) -> &'static str {
        "embeddings"
    }

    fn paths(&self) -> &'static [&'static str] {
        &["/v1/embeddings", "/embeddings", "/embedding"]
    }

    fn prompt(&self, body: &Value) -> Prompt {
        prompt_field(body.get("input").or_else(|| body.get("content")))
    }

    fn media(&self, body: &Value) -> u64 {
        body.get("input")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|item| item.get("content").and_then(Value::as_array))
            .flatten()
            .filter(|part| is_media(part))
            .count() as u64
    }

    fn generates(&self) -> bool {
        false
    }

    fn requested_generation(&self, _body: &Value) -> Option<u64> {
        None
    }

    fn limit_generation(&self, _body: &mut Value, _tokens: u64) {}

    fn choices(&self, _body: &Value) -> u64 {
        1
    }

    fn streaming(&self, _body: &Value) -> bool {
        false
    }

    fn pinnable(&self) -> bool {
        false
    }

    fn pin(&self, _body: &mut Value, _slot: usize) {}
}

/// Reranking (`/reranking`, `/rerank`, `/v1/rerank`, `/v1/reranking`): the
/// server scores each document against the query as its own task.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rerank;

impl Dialect for Rerank {
    fn name(&self) -> &'static str {
        "rerank"
    }

    fn paths(&self) -> &'static [&'static str] {
        &["/reranking", "/rerank", "/v1/rerank", "/v1/reranking"]
    }

    fn prompt(&self, body: &Value) -> Prompt {
        let query = body
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let documents = body
            .get("documents")
            .or_else(|| body.get("texts"))
            .and_then(Value::as_array);
        let Some(documents) = documents else {
            return Prompt::Unknown;
        };
        let pairs: Vec<Prompt> = documents
            .iter()
            .map(|document| {
                let text = match document {
                    Value::String(text) => text.clone(),
                    other => other
                        .get("text")
                        .and_then(Value::as_str)
                        .map_or_else(|| other.to_string(), str::to_owned),
                };
                Prompt::Text(format!("{query}\n{text}"))
            })
            .collect();
        match pairs.len() {
            1 => pairs.into_iter().next().unwrap_or(Prompt::Unknown),
            _ => Prompt::Many(pairs),
        }
    }

    fn generates(&self) -> bool {
        false
    }

    fn requested_generation(&self, _body: &Value) -> Option<u64> {
        None
    }

    fn limit_generation(&self, _body: &mut Value, _tokens: u64) {}

    fn choices(&self, _body: &Value) -> u64 {
        1
    }

    fn streaming(&self, _body: &Value) -> bool {
        false
    }

    fn pinnable(&self) -> bool {
        false
    }

    fn pin(&self, _body: &mut Value, _slot: usize) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_resolve_to_their_dialect_and_later_registrations_win() {
        let dialects = Dialects::standard();
        let name = |path| dialects.find(path).map(|dialect| dialect.name());
        assert_eq!(name("/v1/chat/completions"), Some("openai_chat"));
        assert_eq!(name("/completion"), Some("llama_completion"));
        assert_eq!(name("/v1/messages"), Some("anthropic_messages"));
        assert_eq!(name("/v1/responses"), Some("openai_responses"));
        assert_eq!(name("/v1/embeddings"), Some("embeddings"));
        assert_eq!(name("/rerank"), Some("rerank"));
        assert_eq!(
            name("/v1/messages/count_tokens"),
            None,
            "counting needs no slot"
        );

        #[derive(Debug)]
        struct Custom;
        impl Dialect for Custom {
            fn name(&self) -> &'static str {
                "custom"
            }
            fn paths(&self) -> &'static [&'static str] {
                &["/completion"]
            }
            fn prompt(&self, _body: &Value) -> Prompt {
                Prompt::Unknown
            }
            fn requested_generation(&self, _body: &Value) -> Option<u64> {
                None
            }
            fn limit_generation(&self, _body: &mut Value, _tokens: u64) {}
        }
        let dialects = dialects.with(Custom);
        assert_eq!(
            dialects.find("/completion").map(|d| d.name()),
            Some("custom")
        );
    }

    #[test]
    fn generation_limits_resolve_as_the_server_resolves_them() {
        // The first of n_predict, max_completion_tokens, max_tokens wins.
        let chat = json!({"messages": [], "max_completion_tokens": 900, "max_tokens": 50});
        assert_eq!(OpenAiChat.requested_generation(&chat), Some(900));
        let chat = json!({"messages": [], "n_predict": 70, "max_tokens": 900});
        assert_eq!(OpenAiChat.requested_generation(&chat), Some(70));
        // Negative is no limit at all, even beside a smaller field.
        let unbounded = json!({"messages": [], "n_predict": -1, "max_tokens": 900});
        assert_eq!(OpenAiChat.requested_generation(&unbounded), None);
        // Zero is a limit (evaluate the prompt only), not "unset".
        let prefill = json!({"messages": [], "max_tokens": 0});
        assert_eq!(OpenAiChat.requested_generation(&prefill), Some(0));
        // Infill shares the fields.
        let infill = json!({"input_prefix": "fn ", "max_tokens": 128});
        assert_eq!(Infill.requested_generation(&infill), Some(128));
        // Converted APIs: extra_params are merged last and win.
        let responses = json!({"input": "hi", "max_output_tokens": 300,
                               "extra_params": {"max_tokens": 4000}});
        assert_eq!(OpenAiResponses.requested_generation(&responses), Some(4000));
        let anthropic = json!({"messages": [], "max_tokens": 128,
                               "extra_params": {"n_predict": -1}});
        assert_eq!(AnthropicMessages.requested_generation(&anthropic), None);
    }

    #[test]
    fn a_written_limit_cannot_be_lifted_by_another_field() {
        let mut chat = json!({"messages": [], "n_predict": -1, "max_tokens": 900});
        OpenAiChat.limit_generation(&mut chat, 500);
        assert_eq!(OpenAiChat.requested_generation(&chat), Some(500));
        for dialect in [
            &OpenAiCompletions as &dyn Dialect,
            &NativeCompletion,
            &Infill,
        ] {
            let mut body = json!({"prompt": "hi", "max_tokens": 9000});
            dialect.limit_generation(&mut body, 64);
            assert_eq!(
                dialect.requested_generation(&body),
                Some(64),
                "{}",
                dialect.name()
            );
        }
        for dialect in [&OpenAiResponses as &dyn Dialect, &AnthropicMessages] {
            let mut body = json!({"messages": [], "input": "hi", "max_tokens": 9000,
                                  "max_output_tokens": 9000,
                                  "extra_params": {"max_tokens": 8000, "id_slot": 1}});
            dialect.limit_generation(&mut body, 64);
            assert_eq!(
                dialect.requested_generation(&body),
                Some(64),
                "{}",
                dialect.name()
            );
            assert_eq!(body["extra_params"]["id_slot"], 1, "other options survive");
        }
    }

    #[test]
    fn several_completions_are_counted_as_the_server_counts_them() {
        assert_eq!(OpenAiChat.choices(&json!({"n": 2})), 2);
        assert_eq!(OpenAiChat.choices(&json!({"n": 1, "n_cmpl": 3})), 3);
        assert_eq!(NativeCompletion.choices(&json!({})), 1);
    }

    #[test]
    fn bare_completions_is_llama_cpps_native_endpoint() {
        let dialects = Dialects::standard();
        let name = |path: &str| dialects.find(path).map(|dialect| dialect.name());
        assert_eq!(name("/completions"), Some("llama_completion"));
        assert_eq!(name("/v1/completions"), Some("openai_completions"));
    }

    #[test]
    fn slot_options_go_where_each_endpoint_reads_them() {
        let mut chat = json!({"messages": [], "cache_prompt": false});
        OpenAiChat.pin(&mut chat, 3);
        assert_eq!(chat["id_slot"], 3);
        assert_eq!(chat["cache_prompt"], false, "a client's choice is kept");

        let mut responses = json!({"input": "hi", "extra_params": {"seed": 7}});
        OpenAiResponses.pin(&mut responses, 1);
        assert_eq!(
            responses["extra_params"],
            json!({"seed": 7, "id_slot": 1, "cache_prompt": true})
        );
        assert!(responses.get("id_slot").is_none());

        let mut embeddings = json!({"input": "hi"});
        Embeddings.pin(&mut embeddings, 1);
        assert_eq!(embeddings, json!({"input": "hi"}));
        assert!(!Embeddings.pinnable() && !Embeddings.generates());
    }

    #[test]
    fn the_prompt_field_accepts_every_documented_shape() {
        assert_eq!(
            prompt_field(Some(&json!("text"))),
            Prompt::Text("text".into())
        );
        assert_eq!(
            prompt_field(Some(&json!([1, 2, 3]))),
            Prompt::Tokens(vec![1, 2, 3])
        );
        assert_eq!(
            prompt_field(Some(
                &json!({"prompt_string": "with image", "multimodal_data": ["x"]})
            )),
            Prompt::Text("with image".into())
        );
        assert!(
            matches!(prompt_field(Some(&json!(["a", "b"]))), Prompt::Many(items) if items.len() == 2)
        );
        assert_eq!(
            prompt_field(Some(&json!(["only"]))),
            Prompt::Text("only".into())
        );
    }

    #[test]
    fn media_parts_are_counted_separately() {
        let body = json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
            {"type": "input_audio", "input_audio": {}}
        ]}]});
        assert_eq!(OpenAiChat.media(&body), 2);
    }

    #[test]
    fn anthropic_requests_are_measured_through_the_chat_template() {
        let body = json!({
            "system": "be brief",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            "max_tokens": 128
        });
        let Prompt::Chat(chat) = AnthropicMessages.prompt(&body) else {
            panic!("expected a chat prompt");
        };
        assert_eq!(chat["messages"][0]["role"], "system");
        assert_eq!(chat["messages"][1]["content"], "hello");
        assert_eq!(AnthropicMessages.requested_generation(&body), Some(128));
    }

    #[test]
    fn responses_become_a_chat_with_tool_turns() {
        let body = json!({
            "instructions": "be brief",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "weather?"}, {"type": "input_image", "image_url": "x"}]},
                {"type": "function_call", "call_id": "c1", "name": "weather", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
                {"type": "reasoning", "summary": []}
            ],
            "tools": [{"type": "function", "name": "weather", "parameters": {}}]
        });
        let Prompt::Chat(chat) = OpenAiResponses.prompt(&body) else {
            panic!("expected a chat prompt");
        };
        let roles: Vec<&str> = chat["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|message| message["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, ["system", "user", "assistant", "tool"]);
        assert_eq!(chat["messages"][1]["content"], "weather?");
        assert_eq!(chat["tools"][0]["function"]["name"], "weather");
        assert_eq!(OpenAiResponses.media(&body), 1);
    }

    #[test]
    fn embedding_and_ranking_inputs_become_one_task_per_item() {
        assert_eq!(
            Embeddings.prompt(&json!({"input": "one"})),
            Prompt::Text("one".into())
        );
        assert!(matches!(
            Embeddings.prompt(&json!({"input": ["a", "b", [1, 2]]})),
            Prompt::Many(items) if items.len() == 3
        ));
        assert_eq!(
            Embeddings.prompt(&json!({"content": "native"})),
            Prompt::Text("native".into())
        );
        let ranked = Rerank.prompt(&json!({"query": "q", "documents": ["d1", {"text": "d2"}]}));
        assert_eq!(
            ranked,
            Prompt::Many(vec![
                Prompt::Text("q\nd1".into()),
                Prompt::Text("q\nd2".into())
            ])
        );
    }

    #[test]
    fn errors_inside_a_stream_are_framed_as_each_api_expects() {
        use crate::stream::{Kind, Reader};
        for dialect in [
            &OpenAiChat as &dyn Dialect,
            &AnthropicMessages,
            &OpenAiResponses,
            &NativeCompletion,
        ] {
            let event = dialect.error_event("server_error", "the scheduler stopped");
            let mut reader = Reader::new(dialect.wire());
            let pieces = reader.read(event.as_bytes());
            assert_eq!(pieces.len(), 1, "{}: one whole event", dialect.name());
            assert!(
                matches!(&pieces[0].kind, Kind::Error(message) if message == "the scheduler stopped"),
                "{}: read back as the failure it reports: {:?}",
                dialect.name(),
                pieces[0].kind
            );
        }
        assert!(
            AnthropicMessages
                .error_event("overloaded_error", "busy")
                .starts_with("event: error\n")
        );
    }

    #[test]
    fn ensure_stream_usage_injects_only_when_streaming_and_missing() {
        for dialect in [&OpenAiChat as &dyn Dialect, &OpenAiCompletions] {
            // Absent stream_options on streaming request -> injects and returns true.
            let mut absent = json!({"stream": true, "messages": []});
            assert!(dialect.ensure_stream_usage(&mut absent));
            assert_eq!(absent["stream_options"], json!({"include_usage": true}));

            // Existing object with extra keys -> preserves extra keys.
            let mut extra = json!({
                "stream": true,
                "stream_options": {"continuous_usage_stats": true}
            });
            assert!(dialect.ensure_stream_usage(&mut extra));
            assert_eq!(
                extra["stream_options"],
                json!({"continuous_usage_stats": true, "include_usage": true})
            );

            // Explicit include_usage: false -> flips to true, preserving extra keys.
            let mut disabled = json!({
                "stream": true,
                "stream_options": {"include_usage": false, "other": 7}
            });
            assert!(dialect.ensure_stream_usage(&mut disabled));
            assert_eq!(
                disabled["stream_options"],
                json!({"include_usage": true, "other": 7})
            );

            // Explicit include_usage: true -> untouched, returns false.
            let mut enabled = json!({
                "stream": true,
                "stream_options": {"include_usage": true, "other": 7}
            });
            let before = enabled.clone();
            assert!(!dialect.ensure_stream_usage(&mut enabled));
            assert_eq!(enabled, before);

            // Malformed stream_options (null/bool/array/string) -> replaced cleanly.
            for bad in [json!(null), json!(false), json!([]), json!("bad")] {
                let mut malformed = json!({"stream": true, "stream_options": bad});
                assert!(dialect.ensure_stream_usage(&mut malformed));
                assert_eq!(malformed["stream_options"], json!({"include_usage": true}));
            }

            // Non-streaming or non-bool stream -> untouched, returns false.
            for mut non_stream in [
                json!({"messages": []}),
                json!({"stream": false}),
                json!({"stream": "true"}),
                json!(null),
            ] {
                let before = non_stream.clone();
                assert!(!dialect.ensure_stream_usage(&mut non_stream));
                assert_eq!(non_stream, before);
            }
        }

        // Other dialects never modify the request.
        for dialect in [
            &OpenAiResponses as &dyn Dialect,
            &AnthropicMessages,
            &NativeCompletion,
            &Infill,
            &Embeddings,
            &Rerank,
        ] {
            let mut body = json!({"stream": true});
            assert!(!dialect.ensure_stream_usage(&mut body));
            assert_eq!(body, json!({"stream": true}));
        }
    }

    #[test]
    fn generation_limits_handle_large_integers_floats_and_non_numbers() {
        // Values above i64::MAX must not be misread as unlimited (None).
        let huge = json!({"n_predict": u64::MAX});
        assert_eq!(OpenAiChat.requested_generation(&huge), Some(u64::MAX));
        let huge_anthropic = json!({"max_tokens": u64::MAX});
        assert_eq!(
            AnthropicMessages.requested_generation(&huge_anthropic),
            Some(u64::MAX)
        );
        let huge_responses = json!({"max_output_tokens": u64::MAX});
        assert_eq!(
            OpenAiResponses.requested_generation(&huge_responses),
            Some(u64::MAX)
        );

        // Non-integers (float, string, null, bool) are skipped cleanly.
        let mixed = json!({
            "n_predict": null,
            "max_completion_tokens": 12.5,
            "max_tokens": 256
        });
        assert_eq!(OpenAiChat.requested_generation(&mixed), Some(256));

        let invalid = json!({
            "n_predict": "100",
            "max_completion_tokens": 1e30,
            "max_tokens": true
        });
        assert_eq!(OpenAiChat.requested_generation(&invalid), None);

        // Negative limits mean unlimited across dialects.
        assert_eq!(
            AnthropicMessages.requested_generation(&json!({"max_tokens": -1})),
            None
        );
        assert_eq!(
            OpenAiResponses.requested_generation(&json!({"max_output_tokens": -1})),
            None
        );
        assert_eq!(
            OpenAiChat
                .requested_generation(&json!({"max_completion_tokens": -1, "max_tokens": 10})),
            None
        );
    }
}
