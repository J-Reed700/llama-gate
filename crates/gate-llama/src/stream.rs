//! Reading a server's response as it passes through the gate.
//!
//! A streamed response is a sequence of server-sent events. The gate needs
//! three things from it, and gets each from the events' meaning rather than
//! their bytes:
//!
//! - **When output begins.** A request may be paused only until its client
//!   has seen output, and its time to first token ends there. llama.cpp
//!   sends events before any output: prompt-processing progress
//!   (`return_progress`), a role-only first chunk, the Responses API's
//!   `response.created` / `response.in_progress`, Anthropic's
//!   `message_start`. None of those is output.
//! - **Whether it failed.** llama.cpp reports a mid-stream error as an event
//!   (`data: {"error": ...}`, or `event: error`) and then ends the stream
//!   normally: the HTTP status says nothing.
//! - **What it cost.** Token counts arrive in different places per API:
//!   llama.cpp's `timings`, OpenAI's `usage`, the Responses API's
//!   `response.completed`, Anthropic's `message_start` (input) and
//!   `message_delta` (output).
//!
//! Events are framed as the standard defines them, across reads: an event
//! split between two chunks is held until it is complete, so a client is
//! only ever sent whole events.
use crate::usage::{Sniffer, Usage};
use bytes::Bytes;
use serde_json::Value;

/// How an API shapes its responses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wire {
    /// OpenAI chat completions: `choices[].delta`.
    OpenAiChat,
    /// OpenAI legacy completions: `choices[].text`.
    OpenAiCompletion,
    /// llama.cpp's native completion and infill: `content`, `stop`.
    Llama,
    /// OpenAI responses: typed events, usage in `response.completed`.
    Responses,
    /// Anthropic messages: named events, usage split between the first and
    /// the last.
    Anthropic,
    /// Not streamed, or not understood: every event counts as output.
    Opaque,
}

/// What an event means to the gate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Generated output the client sees.
    Output,
    /// Sent before any output (progress, a role-only chunk, the stream's
    /// opening event): the request may still be paused.
    Preamble,
    /// The request failed; the server ends the stream after this.
    Error(String),
    /// The stream's explicit end.
    Done,
    /// Comments, pings, usage-only chunks.
    Other,
}

/// One server-sent event.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Event {
    /// The `event:` field.
    pub name: Option<String>,
    /// The `data:` lines, joined with newlines; `None` without any.
    pub data: Option<String>,
}

/// A complete event as it arrived, with what it means.
#[derive(Clone, Debug)]
pub struct Piece {
    pub bytes: Bytes,
    pub kind: Kind,
}

/// Frames a byte stream into server-sent events, across reads.
#[derive(Clone, Debug, Default)]
pub struct Splitter {
    pending: Vec<u8>,
}

impl Splitter {
    /// Appends `chunk` and returns every event it completes, each with the
    /// bytes that carried it. An incomplete event stays pending.
    pub fn split(&mut self, chunk: &[u8]) -> Vec<(Bytes, Event)> {
        self.pending.extend_from_slice(chunk);
        let mut events = Vec::new();
        let mut start = 0;
        let mut cursor = 0;
        let mut event = Event::default();
        while let Some((line, next)) = next_line(&self.pending, cursor) {
            cursor = next;
            if line.is_empty() {
                let bytes = Bytes::copy_from_slice(&self.pending[start..cursor]);
                events.push((bytes, std::mem::take(&mut event)));
                start = cursor;
            } else {
                field(&mut event, line);
            }
        }
        self.pending.drain(..start);
        events
    }

    /// Bytes of an event not yet complete.
    pub fn pending(&self) -> &[u8] {
        &self.pending
    }

    /// Takes the incomplete remainder, at the end of a stream.
    pub fn take_pending(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.pending)
    }
}

/// The line starting at `from`, without its terminator, and where the next
/// begins. `None` until the line's terminator has arrived: a trailing `\r`
/// may be the first half of `\r\n`.
fn next_line(buffer: &[u8], from: usize) -> Option<(&[u8], usize)> {
    let rest = buffer.get(from..)?;
    let end = rest.iter().position(|byte| matches!(byte, b'\n' | b'\r'))?;
    let terminator = match (rest[end], rest.get(end + 1)) {
        (b'\r', Some(b'\n')) => 2,
        (b'\r', None) => return None,
        _ => 1,
    };
    Some((&rest[..end], from + end + terminator))
}

fn field(event: &mut Event, line: &[u8]) {
    let line = String::from_utf8_lossy(line);
    if line.starts_with(':') {
        return;
    }
    let (name, value) = match line.split_once(':') {
        Some((name, value)) => (name, value.strip_prefix(' ').unwrap_or(value)),
        None => (line.as_ref(), ""),
    };
    match name {
        "event" => event.name = Some(value.to_owned()),
        "data" => match &mut event.data {
            Some(data) => {
                data.push('\n');
                data.push_str(value);
            }
            None => event.data = Some(value.to_owned()),
        },
        _ => {}
    }
}

/// What `event` means for an API shaped like `wire`.
pub fn classify(wire: Wire, event: &Event) -> Kind {
    let Some(data) = event.data.as_deref() else {
        return Kind::Other;
    };
    if data.trim() == "[DONE]" {
        return Kind::Done;
    }
    let value: Option<Value> = serde_json::from_str(data).ok();
    if event.name.as_deref() == Some("error") {
        return Kind::Error(
            value
                .as_ref()
                .map_or_else(|| data.to_owned(), error_message),
        );
    }
    let Some(value) = value else {
        // Not JSON: nothing to tell it by, so it is treated as output.
        return Kind::Output;
    };
    if value.get("error").is_some_and(|error| !error.is_null()) {
        return Kind::Error(error_message(&value));
    }
    let kind = match wire {
        Wire::OpenAiChat => choices_kind(&value, |choice| {
            let delta = choice.get("delta").unwrap_or(&Value::Null);
            ["content", "reasoning_content", "refusal"]
                .iter()
                .any(|key| non_empty_text(delta.get(*key)))
                || non_empty_array(delta.get("tool_calls"))
        }),
        Wire::OpenAiCompletion => choices_kind(&value, |choice| non_empty_text(choice.get("text"))),
        Wire::Llama => {
            if non_empty_text(value.get("content"))
                || non_empty_array(value.get("tokens"))
                || value.get("stop").and_then(Value::as_bool) == Some(true)
            {
                Kind::Output
            } else {
                Kind::Preamble
            }
        }
        Wire::Responses => match value.get("type").and_then(Value::as_str) {
            Some("response.created" | "response.in_progress") => Kind::Preamble,
            Some("response.failed" | "error") => Kind::Error(error_message(&value)),
            _ => Kind::Output,
        },
        Wire::Anthropic => match event
            .name
            .as_deref()
            .or_else(|| value.get("type").and_then(Value::as_str))
        {
            Some("message_start") => Kind::Preamble,
            Some("ping") => Kind::Other,
            Some("message_stop") => Kind::Done,
            _ => Kind::Output,
        },
        Wire::Opaque => Kind::Output,
    };
    // Progress during prompt processing is never output, whatever else the
    // event carries.
    if kind == Kind::Preamble || value.get("prompt_progress").is_none() {
        kind
    } else {
        Kind::Preamble
    }
}

/// Output when any choice carries some, or finishes; a usage-only chunk
/// (no choices) is neither.
fn choices_kind(value: &Value, carries: impl Fn(&Value) -> bool) -> Kind {
    let Some(choices) = value.get("choices").and_then(Value::as_array) else {
        return Kind::Other;
    };
    if choices.is_empty() {
        return Kind::Other;
    }
    let output = choices.iter().any(|choice| {
        carries(choice)
            || choice
                .get("finish_reason")
                .is_some_and(|reason| !reason.is_null())
    });
    if output { Kind::Output } else { Kind::Preamble }
}

fn non_empty_text(value: Option<&Value>) -> bool {
    value
        .and_then(Value::as_str)
        .is_some_and(|text| !text.is_empty())
}

fn non_empty_array(value: Option<&Value>) -> bool {
    value
        .and_then(Value::as_array)
        .is_some_and(|items| !items.is_empty())
}

fn error_message(value: &Value) -> String {
    value
        .pointer("/error/message")
        .or_else(|| value.pointer("/response/error/message"))
        .or_else(|| value.get("message"))
        .or_else(|| value.get("error"))
        .map_or_else(
            || value.to_string(),
            |message| {
                message
                    .as_str()
                    .map_or_else(|| message.to_string(), str::to_owned)
            },
        )
}

/// Takes the token counts an event or body reports into `usage`. llama.cpp's
/// `timings` are the most exact and win; other fields fill what is missing.
fn absorb(wire: Wire, value: &Value, usage: &mut Usage) {
    if let Some(timings) = value.get("timings") {
        let field = |key: &str| timings.get(key).and_then(Value::as_u64);
        if let Some(evaluated) = field("prompt_n") {
            usage.prompt = Some(evaluated + field("cache_n").unwrap_or(0));
        }
        usage.cached = field("cache_n").or(usage.cached);
        usage.generated = field("predicted_n").or(usage.generated);
        usage.decode_ms = timings
            .get("predicted_ms")
            .and_then(Value::as_f64)
            .or(usage.decode_ms);
    }
    match wire {
        Wire::Responses => {
            // Streamed, in `response.completed`; whole, at the top level.
            // Its input count already includes cached tokens.
            let reported = value
                .pointer("/response/usage")
                .or_else(|| value.get("usage"));
            if let Some(reported) = reported {
                let field = |key: &str| reported.get(key).and_then(Value::as_u64);
                fill(&mut usage.prompt, field("input_tokens"));
                fill(&mut usage.generated, field("output_tokens"));
                fill(
                    &mut usage.cached,
                    reported
                        .pointer("/input_tokens_details/cached_tokens")
                        .and_then(Value::as_u64),
                );
            }
        }
        Wire::Anthropic => {
            // Input in `message_start`, output in the final `message_delta`
            // (or both at the top level of a whole message). Cache reads
            // are counted apart from the rest of the input.
            let reported = value
                .pointer("/message/usage")
                .or_else(|| value.get("usage"));
            if let Some(reported) = reported {
                let field = |key: &str| reported.get(key).and_then(Value::as_u64);
                let cached = field("cache_read_input_tokens");
                if let Some(input) = field("input_tokens") {
                    fill(&mut usage.prompt, Some(input + cached.unwrap_or(0)));
                }
                fill(&mut usage.cached, cached);
                // The opening event's count is a placeholder; the last wins.
                if let Some(output) = field("output_tokens")
                    && value.pointer("/message/usage").is_none()
                {
                    usage.generated = Some(output);
                }
            }
        }
        _ => {
            if let Some(reported) = value.get("usage") {
                let field = |key: &str| reported.get(key).and_then(Value::as_u64);
                fill(&mut usage.prompt, field("prompt_tokens"));
                fill(&mut usage.generated, field("completion_tokens"));
                fill(
                    &mut usage.cached,
                    reported
                        .pointer("/prompt_tokens_details/cached_tokens")
                        .and_then(Value::as_u64),
                );
            }
            fill(
                &mut usage.prompt,
                value.get("tokens_evaluated").and_then(Value::as_u64),
            );
            fill(
                &mut usage.generated,
                value.get("tokens_predicted").and_then(Value::as_u64),
            );
        }
    }
}

fn fill(slot: &mut Option<u64>, value: Option<u64>) {
    if slot.is_none() {
        *slot = value;
    }
}

/// Reads a streamed response: frames its events, says what each means, and
/// keeps the usage and any failure it reports.
#[derive(Clone, Debug)]
pub struct Reader {
    wire: Wire,
    splitter: Splitter,
    usage: Usage,
    failure: Option<String>,
}

impl Reader {
    pub fn new(wire: Wire) -> Self {
        Self {
            wire,
            splitter: Splitter::default(),
            usage: Usage::default(),
            failure: None,
        }
    }

    /// Reads a chunk; returns the complete events it finishes, in order.
    pub fn read(&mut self, chunk: &[u8]) -> Vec<Piece> {
        self.splitter
            .split(chunk)
            .into_iter()
            .map(|(bytes, event)| {
                let kind = classify(self.wire, &event);
                if let Some(value) = event
                    .data
                    .as_deref()
                    .and_then(|data| serde_json::from_str::<Value>(data).ok())
                {
                    absorb(self.wire, &value, &mut self.usage);
                }
                if let Kind::Error(message) = &kind
                    && self.failure.is_none()
                {
                    self.failure = Some(message.clone());
                }
                Piece { bytes, kind }
            })
            .collect()
    }

    /// The bytes of an incomplete event, at the end of a stream.
    pub fn finish(&mut self) -> Option<Bytes> {
        let rest = self.splitter.take_pending();
        (!rest.is_empty()).then(|| Bytes::from(rest))
    }

    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// The error the stream reported, if it did.
    pub fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }
}

/// Reads a whole (unstreamed) response body as it passes: kept whole up to a
/// bound and parsed at the end, or, past the bound, read from its tail.
#[derive(Clone, Debug)]
pub struct WholeReader {
    wire: Wire,
    body: Vec<u8>,
    tail: Option<Sniffer>,
}

impl WholeReader {
    /// Bodies up to this size are parsed whole.
    const LIMIT: usize = 4 << 20;

    pub fn new(wire: Wire) -> Self {
        Self {
            wire,
            body: Vec::new(),
            tail: None,
        }
    }

    pub fn read(&mut self, chunk: &[u8]) {
        if let Some(tail) = &mut self.tail {
            tail.observe(chunk);
        } else if self.body.len() + chunk.len() > Self::LIMIT {
            let mut tail = Sniffer::default();
            tail.observe(&self.body);
            tail.observe(chunk);
            self.body = Vec::new();
            self.tail = Some(tail);
        } else {
            self.body.extend_from_slice(chunk);
        }
    }

    pub fn usage(&self) -> Usage {
        if let Some(tail) = &self.tail {
            return tail.usage();
        }
        let mut usage = Usage::default();
        if let Ok(value) = serde_json::from_slice::<Value>(&self.body) {
            absorb(self.wire, &value, &mut usage);
        }
        usage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(wire: Wire, stream: &[&str]) -> Vec<Kind> {
        let mut reader = Reader::new(wire);
        stream
            .iter()
            .flat_map(|chunk| reader.read(chunk.as_bytes()))
            .map(|piece| piece.kind)
            .collect()
    }

    #[test]
    fn an_event_split_across_reads_is_held_until_complete() {
        let mut splitter = Splitter::default();
        assert!(splitter.split(b"da").is_empty());
        assert!(splitter.split(b"ta: {\"x\":1}\n").is_empty());
        let events = splitter.split(b"\ndata: [DONE]\r\n\r\n: ping\n\n");
        assert_eq!(events.len(), 3);
        assert_eq!(&events[0].0[..], b"data: {\"x\":1}\n\n");
        assert_eq!(events[0].1.data.as_deref(), Some("{\"x\":1}"));
        assert_eq!(events[1].1.data.as_deref(), Some("[DONE]"));
        assert_eq!(events[2].1, Event::default(), "a comment carries nothing");
        assert!(splitter.pending().is_empty());
    }

    #[test]
    fn a_trailing_carriage_return_waits_for_its_line_feed() {
        let mut splitter = Splitter::default();
        assert!(splitter.split(b"data: a\r\n\r").is_empty());
        let events = splitter.split(b"\n");
        assert_eq!(events.len(), 1);
        assert_eq!(&events[0].0[..], b"data: a\r\n\r\n");
    }

    #[test]
    fn multi_line_data_and_named_events_are_read() {
        let mut splitter = Splitter::default();
        let events = splitter.split(b"event: message_start\ndata: {\"a\":\ndata: 1}\n\n");
        assert_eq!(events[0].1.name.as_deref(), Some("message_start"));
        assert_eq!(events[0].1.data.as_deref(), Some("{\"a\":\n1}"));
    }

    /// What llama.cpp streams for a chat with `return_progress`: progress as
    /// role-only chunks from the moment prefill starts, then output.
    #[test]
    fn chat_progress_and_role_chunks_are_not_output() {
        let stream = [
            "data: {\"choices\":[{\"finish_reason\":null,\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null}}],\"prompt_progress\":{\"total\":1500,\"cache\":0,\"processed\":0,\"time_ms\":0}}\n\n",
            "data: {\"choices\":[{\"finish_reason\":null,\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null}}],\"prompt_progress\":{\"total\":1500,\"cache\":0,\"processed\":1024,\"time_ms\":900}}\n\n",
            "data: {\"choices\":[{\"finish_reason\":null,\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null}}]}\n\n",
            "data: {\"choices\":[{\"finish_reason\":null,\"index\":0,\"delta\":{\"reasoning_content\":\"Hm\"}}]}\n\n",
            "data: {\"choices\":[{\"finish_reason\":\"stop\",\"index\":0,\"delta\":{}}],\"timings\":{\"cache_n\":0,\"prompt_n\":1500,\"predicted_n\":2,\"predicted_ms\":40.0}}\n\n",
            "data: [DONE]\n\n",
        ];
        assert_eq!(
            kinds(Wire::OpenAiChat, &stream),
            vec![
                Kind::Preamble,
                Kind::Preamble,
                Kind::Preamble,
                Kind::Output,
                Kind::Output,
                Kind::Done
            ]
        );
    }

    #[test]
    fn tool_calls_and_usage_only_chunks_are_told_apart() {
        let stream = [
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"f\"}}]}}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":3}}\n\n",
        ];
        assert_eq!(
            kinds(Wire::OpenAiChat, &stream),
            vec![Kind::Output, Kind::Other]
        );
    }

    #[test]
    fn a_mid_stream_error_is_a_failure_not_output() {
        let mut reader = Reader::new(Wire::OpenAiChat);
        let pieces = reader.read(
            b"data: {\"error\":{\"code\":500,\"message\":\"Context size has been exceeded.\",\"type\":\"exceed_context_size_error\"}}\n\n",
        );
        assert!(matches!(pieces[0].kind, Kind::Error(_)));
        assert_eq!(reader.failure(), Some("Context size has been exceeded."));
        let named = kinds(
            Wire::Anthropic,
            &[
                "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}\n\n",
            ],
        );
        assert_eq!(named, vec![Kind::Error("busy".into())]);
    }

    #[test]
    fn native_and_completion_progress_is_not_output() {
        let native = [
            "data: {\"content\":\"\",\"tokens\":[],\"stop\":false,\"prompt_progress\":{\"total\":10,\"processed\":5}}\n\n",
            "data: {\"content\":\"Hi\",\"tokens\":[42],\"stop\":false}\n\n",
            "data: {\"content\":\"\",\"stop\":true,\"timings\":{\"prompt_n\":10,\"cache_n\":0,\"predicted_n\":1}}\n\n",
        ];
        assert_eq!(
            kinds(Wire::Llama, &native),
            vec![Kind::Preamble, Kind::Output, Kind::Output]
        );
        let completion = [
            "data: {\"choices\":[{\"text\":\"\",\"index\":0,\"finish_reason\":null}],\"prompt_progress\":{\"total\":10,\"processed\":5}}\n\n",
            "data: {\"choices\":[{\"text\":\"Hi\",\"index\":0,\"finish_reason\":null}]}\n\n",
        ];
        assert_eq!(
            kinds(Wire::OpenAiCompletion, &completion),
            vec![Kind::Preamble, Kind::Output]
        );
    }

    #[test]
    fn responses_events_and_usage_are_read_by_type() {
        let mut reader = Reader::new(Wire::Responses);
        let kinds: Vec<Kind> = [
            "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"r\"}}\n\n",
            "event: response.in_progress\ndata: {\"type\":\"response.in_progress\"}\n\n",
            "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n\n",
            "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":100000,\"input_tokens_details\":{\"cached_tokens\":99000},\"output_tokens\":12}}}\n\n",
        ]
        .iter()
        .flat_map(|chunk| reader.read(chunk.as_bytes()))
        .map(|piece| piece.kind)
        .collect();
        assert_eq!(
            kinds,
            vec![Kind::Preamble, Kind::Preamble, Kind::Output, Kind::Output]
        );
        let usage = reader.usage();
        assert_eq!(
            usage.prompt,
            Some(100_000),
            "input already includes the cached tokens"
        );
        assert_eq!(usage.cached, Some(99_000));
        assert_eq!(usage.generated, Some(12));
    }

    #[test]
    fn anthropic_input_comes_from_the_first_event_and_output_from_the_last() {
        let mut reader = Reader::new(Wire::Anthropic);
        let kinds: Vec<Kind> = [
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":900,\"cache_read_input_tokens\":100,\"output_tokens\":1}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: ping\ndata: {\"type\":\"ping\"}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":7}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        ]
        .iter()
        .flat_map(|chunk| reader.read(chunk.as_bytes()))
        .map(|piece| piece.kind)
        .collect();
        assert_eq!(
            kinds,
            vec![
                Kind::Preamble,
                Kind::Output,
                Kind::Other,
                Kind::Output,
                Kind::Output,
                Kind::Done
            ]
        );
        let usage = reader.usage();
        assert_eq!(usage.prompt, Some(1_000));
        assert_eq!(usage.cached, Some(100));
        assert_eq!(
            usage.generated,
            Some(7),
            "the final count, not the opening placeholder"
        );
    }

    #[test]
    fn whole_bodies_are_read_by_their_api() {
        let responses = br#"{"object":"response","usage":{"input_tokens":100000,"input_tokens_details":{"cached_tokens":99000},"output_tokens":500}}"#;
        let mut reader = WholeReader::new(Wire::Responses);
        reader.read(responses);
        assert_eq!(reader.usage().prompt, Some(100_000));

        let anthropic = br#"{"type":"message","usage":{"input_tokens":900,"cache_read_input_tokens":100,"output_tokens":5}}"#;
        let mut reader = WholeReader::new(Wire::Anthropic);
        reader.read(anthropic);
        let usage = reader.usage();
        assert_eq!(
            (usage.prompt, usage.cached, usage.generated),
            (Some(1_000), Some(100), Some(5))
        );

        let chat = br#"{"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":4},"timings":{"cache_n":15,"prompt_n":5,"predicted_n":4,"predicted_ms":30.0}}"#;
        let mut reader = WholeReader::new(Wire::OpenAiChat);
        reader.read(&chat[..10]);
        reader.read(&chat[10..]);
        let usage = reader.usage();
        assert_eq!(
            (usage.prompt, usage.cached, usage.generated),
            (Some(20), Some(15), Some(4))
        );
    }
}
