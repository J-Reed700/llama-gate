//! Reading final token counts from a finished response.
//!
//! llama.cpp reports what a request processed in several places depending
//! on the endpoint: a `timings` object (`cache_n`, `prompt_n`,
//! `predicted_n`), an OpenAI `usage` object, Anthropic `input_tokens` /
//! `output_tokens`, or native `tokens_evaluated` / `tokens_predicted`. They
//! all sit at the end of the response, streamed or not, so [`Sniffer`] keeps
//! only a bounded tail of the bytes passing through and reads it at the end.
use serde_json::Value;

/// Tokens a finished request processed.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Usage {
    /// Prompt tokens, including any reused from cache.
    pub prompt: Option<u64>,
    /// Of those, the tokens the server reused from its cache.
    pub cached: Option<u64>,
    pub generated: Option<u64>,
    /// Milliseconds the server spent generating, when it reports them.
    pub decode_ms: Option<f64>,
}

impl Usage {
    /// Generation speed as the server measured it. llama.cpp starts its
    /// generation clock at the first token, so its time covers the intervals
    /// between tokens: one fewer than the tokens (and none for one token).
    pub fn tokens_per_second(&self) -> Option<f64> {
        let (generated, millis) = (self.generated?, self.decode_ms?);
        (generated > 1 && millis > 0.0).then(|| (generated - 1) as f64 * 1000.0 / millis)
    }
}

impl Usage {
    /// Tokens the slot holds after the request: prompt plus generated.
    pub fn resident(&self, measured_prompt: u64) -> Option<u64> {
        let generated = self.generated?;
        Some(self.prompt.unwrap_or(measured_prompt) + generated)
    }
}

/// Keeps the last bytes of a response and reads the usage from them.
#[derive(Clone, Debug)]
pub struct Sniffer {
    tail: Vec<u8>,
    limit: usize,
}

impl Default for Sniffer {
    fn default() -> Self {
        Self::new(64 * 1024)
    }
}

impl Sniffer {
    pub fn new(limit: usize) -> Self {
        Self {
            tail: Vec::new(),
            limit: limit.max(1024),
        }
    }

    pub fn observe(&mut self, chunk: &[u8]) {
        if chunk.len() >= self.limit {
            self.tail.clear();
            self.tail
                .extend_from_slice(&chunk[chunk.len() - self.limit..]);
            return;
        }
        self.tail.extend_from_slice(chunk);
        if self.tail.len() > self.limit * 2 {
            let excess = self.tail.len() - self.limit;
            self.tail.drain(..excess);
        }
    }

    pub fn usage(&self) -> Usage {
        let text = String::from_utf8_lossy(&self.tail);
        let mut usage = Usage::default();
        if let Some(timings) = last_object(&text, "timings") {
            let field = |key: &str| timings.get(key).and_then(Value::as_u64);
            if let Some(evaluated) = field("prompt_n") {
                usage.prompt = Some(evaluated + field("cache_n").unwrap_or(0));
            }
            usage.cached = field("cache_n");
            usage.generated = field("predicted_n");
            usage.decode_ms = timings.get("predicted_ms").and_then(Value::as_f64);
        }
        if let Some(reported) = last_object(&text, "usage") {
            let field = |key: &str| reported.get(key).and_then(Value::as_u64);
            let cached = reported
                .pointer("/prompt_tokens_details/cached_tokens")
                .or_else(|| reported.pointer("/input_tokens_details/cached_tokens"))
                .or_else(|| reported.get("cache_read_input_tokens"))
                .and_then(Value::as_u64);
            usage.cached = usage.cached.or(cached);
            usage.prompt = usage.prompt.or_else(|| field("prompt_tokens")).or_else(|| {
                // Anthropic counts cached input apart from the rest.
                field("input_tokens").map(|input| input + cached.unwrap_or(0))
            });
            usage.generated = usage
                .generated
                .or_else(|| field("completion_tokens"))
                .or_else(|| field("output_tokens"));
        }
        if usage.generated.is_none() {
            usage.generated = last_number(&text, "tokens_predicted");
            usage.prompt = usage
                .prompt
                .or_else(|| last_number(&text, "tokens_evaluated"));
        }
        usage
    }
}

/// The JSON object that follows the last `"key":` in `text`.
fn last_object(text: &str, key: &str) -> Option<Value> {
    let needle = format!("\"{key}\"");
    let mut search = text.len();
    while let Some(found) = text[..search].rfind(&needle) {
        if let Some(value) = object_at(&text[found + needle.len()..]) {
            return Some(value);
        }
        search = found;
    }
    None
}

fn object_at(rest: &str) -> Option<Value> {
    let rest = rest.trim_start().strip_prefix(':')?.trim_start();
    if !rest.starts_with('{') {
        return None;
    }
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (index, byte) in rest.bytes().enumerate() {
        match byte {
            _ if escaped => escaped = false,
            b'\\' if in_string => escaped = true,
            b'"' => in_string = !in_string,
            b'{' if !in_string => depth += 1,
            b'}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return serde_json::from_str(&rest[..=index]).ok();
                }
            }
            _ => {}
        }
    }
    None
}

fn last_number(text: &str, key: &str) -> Option<u64> {
    let needle = format!("\"{key}\"");
    let found = text.rfind(&needle)?;
    let rest = text[found + needle.len()..]
        .trim_start()
        .strip_prefix(':')?
        .trim_start();
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sniff(parts: &[&str]) -> Usage {
        let mut sniffer = Sniffer::new(1024);
        for part in parts {
            sniffer.observe(part.as_bytes());
        }
        sniffer.usage()
    }

    #[test]
    fn llama_timings_count_cached_and_evaluated_prompt_tokens() {
        let usage = sniff(&[
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: {\"choices\":[],\"timings\":{\"cache_n\":900,\"prompt_n\":100,\"predicted_n\":42,\"predicted_ms\":600.0}}\n\n",
            "data: [DONE]\n\n",
        ]);
        assert_eq!(
            usage,
            Usage {
                prompt: Some(1000),
                cached: Some(900),
                generated: Some(42),
                decode_ms: Some(600.0),
            }
        );
        assert_eq!(usage.resident(0), Some(1042));
        assert_eq!(usage.tokens_per_second(), Some(41.0 * 1000.0 / 600.0));
    }

    #[test]
    fn openai_and_anthropic_usage_objects_are_read() {
        let openai = sniff(&[
            "{\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}",
        ]);
        assert_eq!(
            openai,
            Usage {
                prompt: Some(10),
                cached: None,
                decode_ms: None,
                generated: Some(5)
            }
        );
        let anthropic = sniff(&[
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":7}}\n\n",
        ]);
        assert_eq!(anthropic.generated, Some(7));
        assert_eq!(
            anthropic.resident(300),
            Some(307),
            "the measured prompt fills the gap"
        );
    }

    #[test]
    fn native_completion_counts_are_read() {
        let usage = sniff(&["{\"content\":\"x\",\"tokens_evaluated\":64,\"tokens_predicted\":8}"]);
        assert_eq!(
            usage,
            Usage {
                prompt: Some(64),
                cached: None,
                decode_ms: None,
                generated: Some(8)
            }
        );
    }

    #[test]
    fn braces_inside_strings_do_not_confuse_the_reader() {
        let usage = sniff(&[
            "{\"content\":\"a } { \\\" brace\",\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}",
        ]);
        assert_eq!(
            usage,
            Usage {
                prompt: Some(3),
                cached: None,
                decode_ms: None,
                generated: Some(2)
            }
        );
    }

    #[test]
    fn cache_reuse_is_read_from_every_format() {
        let openai = sniff(&[
            "{\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":5,\"prompt_tokens_details\":{\"cached_tokens\":80}}}",
        ]);
        assert_eq!(openai.cached, Some(80));
        let anthropic = sniff(&[
            "{\"usage\":{\"input_tokens\":20,\"cache_read_input_tokens\":80,\"output_tokens\":3}}",
        ]);
        assert_eq!((anthropic.prompt, anthropic.cached), (Some(100), Some(80)));
    }

    #[test]
    fn only_a_bounded_tail_is_kept() {
        let mut sniffer = Sniffer::new(1024);
        for _ in 0..10_000 {
            sniffer.observe(b"data: {\"choices\":[{\"delta\":{\"content\":\"token\"}}]}\n\n");
        }
        sniffer.observe(b"data: {\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":10000}}\n\n");
        assert!(sniffer.tail.len() <= 2048);
        assert_eq!(sniffer.usage().generated, Some(10_000));
    }
}
