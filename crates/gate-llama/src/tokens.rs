//! Incremental tokenization of rendered chat prompts.
//!
//! Tokenizing a long conversation from scratch on every turn costs the
//! server real time (about 0.2 s for 90k tokens). A conversation's next
//! prompt repeats its previous one, so the tokens of that shared head are
//! cached and only the new text is sent to `/tokenize`.
//!
//! Text can only be cut where tokenization composes. llama.cpp splits a
//! prompt at special tokens before tokenizing each piece of plain text on its
//! own, so the tokens of `a + b` are those of `a` followed by those of `b`
//! whenever `b` starts with a special token, such as a chat template's turn
//! marker. [`Boundaries`] learns those markers from the server (the tokens
//! that only appear when special tokens are parsed), and cuts are only made
//! right before one, never after whitespace a marker might absorb.
//!
//! The cache checks itself: the first few incremental results per model,
//! and then one in [`TokenCache::verify_every`], are compared with a full
//! tokenization, and a single mismatch turns the cache off for that model.
use crate::{Server, ServerError};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// Prompts shorter than this are tokenized whole: it is cheap already.
const MIN_TEXT: usize = 16 * 1024;
/// Incremental results always verified per model before sampling starts.
const VERIFY_FIRST: u64 = 3;

/// Where a chat template's text may be cut, and what `add_special` adds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Boundaries {
    /// Special-token texts, longest first.
    markers: Vec<String>,
    /// Tokens `add_special` puts before and after a prompt (BOS, EOS).
    head: Vec<i32>,
    tail: Vec<i32>,
}

impl Boundaries {
    /// Learns the template's special-token texts from the server, by
    /// tokenizing a rendered sample conversation with and without special
    /// tokens parsed: tokens that appear only in the first are special.
    pub async fn learn(server: &Server) -> Result<Option<Self>, ServerError> {
        let sample = server
            .apply_template(&json!({"messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "v"},
            ]}))
            .await?;
        let special = server.pieces(&sample, true).await?;
        let plain: HashSet<i32> = server
            .tokenize_with(&sample, false, false)
            .await?
            .into_iter()
            .collect();
        let mut markers: Vec<String> = special
            .into_iter()
            .filter(|piece| !plain.contains(&piece.id) && piece.text.len() > 1)
            .map(|piece| piece.text)
            .filter(|text| !text.trim().is_empty() && sample.contains(text.as_str()))
            .collect();
        markers.sort_by(|left, right| right.len().cmp(&left.len()).then(left.cmp(right)));
        markers.dedup();
        if markers.is_empty() {
            return Ok(None);
        }
        let probe = "x";
        let bare = server.tokenize_with(probe, false, true).await?;
        let whole = server.tokenize_with(probe, true, true).await?;
        let Some(at) = whole
            .windows(bare.len().max(1))
            .position(|window| window == bare.as_slice())
        else {
            return Ok(None);
        };
        Ok(Some(Self {
            markers,
            head: whole[..at].to_vec(),
            tail: whole[at + bare.len()..].to_vec(),
        }))
    }

    #[cfg(test)]
    fn new(markers: &[&str], head: Vec<i32>, tail: Vec<i32>) -> Self {
        Self {
            markers: markers.iter().map(|marker| (*marker).to_owned()).collect(),
            head,
            tail,
        }
    }

    /// Byte offsets where `text` may be cut: right before a marker, never at
    /// the start, and never after whitespace (a marker may absorb it).
    fn cuts(&self, text: &str) -> Vec<usize> {
        let mut cuts: Vec<usize> = self
            .markers
            .iter()
            .flat_map(|marker| text.match_indices(marker.as_str()).map(|(at, _)| at))
            .filter(|at| {
                *at > 0
                    && !text[..*at]
                        .chars()
                        .next_back()
                        .is_some_and(char::is_whitespace)
            })
            .collect();
        cuts.sort_unstable();
        cuts.dedup();
        cuts
    }
}

/// How a prompt was tokenized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// The whole prompt in one call.
    Full,
    /// Only the text after a cached head.
    Incremental,
}

/// Cached heads of rendered prompts, shared by every backend.
#[derive(Debug)]
pub struct TokenCache {
    inner: Mutex<Inner>,
    /// Most tokens kept; the least recently used heads go first.
    capacity: usize,
    /// After the first few, one incremental result in this many is checked.
    verify_every: u64,
}

#[derive(Debug, Default)]
struct Inner {
    entries: HashMap<[u8; 32], Entry>,
    tokens: usize,
    clock: u64,
    /// Per model: incremental results produced, and whether it is disabled.
    models: HashMap<String, (u64, bool)>,
}

#[derive(Debug)]
struct Entry {
    tokens: Arc<[i32]>,
    used: u64,
}

impl Default for TokenCache {
    fn default() -> Self {
        Self::new(4 << 20, 32)
    }
}

/// A tokenized prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tokenized {
    pub tokens: Vec<i32>,
    pub method: Method,
    /// An incremental result disagreed with a full tokenization; the cache
    /// is now off for this model and `tokens` is the full result.
    pub mismatch: bool,
}

impl TokenCache {
    pub fn new(capacity: usize, verify_every: u64) -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            capacity,
            verify_every: verify_every.max(1),
        }
    }

    /// Tokenizes a rendered prompt the way the server tokenizes one it is
    /// given (special tokens added and parsed), reusing a cached head.
    pub async fn tokenize(
        &self,
        server: &Server,
        boundaries: Option<&Boundaries>,
        namespace: &str,
        text: &str,
    ) -> Result<Tokenized, ServerError> {
        let full = || async {
            Ok(Tokenized {
                tokens: server.tokenize(text).await?,
                method: Method::Full,
                mismatch: false,
            })
        };
        let Some(boundaries) = boundaries.filter(|_| text.len() >= MIN_TEXT) else {
            return full().await;
        };
        if self.disabled(namespace) {
            return full().await;
        }
        let cuts = boundaries.cuts(text);
        let Some(&last) = cuts.last() else {
            return full().await;
        };
        // Content addresses of the text up to each cut.
        let mut hasher = Sha256::new();
        hasher.update(b"llama-gate/tokens/v1");
        hasher.update((namespace.len() as u64).to_le_bytes());
        hasher.update(namespace.as_bytes());
        let mut keys = Vec::with_capacity(cuts.len());
        let mut hashed = 0;
        for &cut in &cuts {
            hasher.update(&text.as_bytes()[hashed..cut]);
            hashed = cut;
            keys.push(<[u8; 32]>::from(hasher.clone().finalize()));
        }
        let (start, head) = self
            .longest(&keys)
            .map_or((0, Arc::from([])), |(index, tokens)| (cuts[index], tokens));
        // Tokenize what follows the cached head in two pieces, so the text up
        // to the last cut (the next turn's head) can be cached in turn.
        let (middle, rest) = if start < last {
            let (middle, rest) = tokio::join!(
                server.tokenize_with(&text[start..last], false, true),
                server.tokenize_with(&text[last..], false, true),
            );
            (middle?, rest?)
        } else {
            (
                Vec::new(),
                server.tokenize_with(&text[start..], false, true).await?,
            )
        };
        let mut content = Vec::with_capacity(head.len() + middle.len() + rest.len());
        content.extend_from_slice(&head);
        content.extend_from_slice(&middle);
        if start < last
            && let Some(key) = keys.last()
        {
            self.insert(*key, Arc::from(content.as_slice()));
        }
        content.extend_from_slice(&rest);
        let mut tokens = boundaries.head.clone();
        tokens.extend_from_slice(&content);
        tokens.extend_from_slice(&boundaries.tail);
        if self.due_for_check(namespace) {
            let whole = server.tokenize(text).await?;
            if whole != tokens {
                self.disable(namespace);
                return Ok(Tokenized {
                    tokens: whole,
                    method: Method::Full,
                    mismatch: true,
                });
            }
        }
        Ok(Tokenized {
            tokens,
            method: if start > 0 {
                Method::Incremental
            } else {
                Method::Full
            },
            mismatch: false,
        })
    }

    fn disabled(&self, namespace: &str) -> bool {
        self.inner
            .lock()
            .is_ok_and(|inner| inner.models.get(namespace).is_some_and(|(_, off)| *off))
    }

    fn disable(&self, namespace: &str) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.models.entry(namespace.to_owned()).or_default().1 = true;
            inner.entries.clear();
            inner.tokens = 0;
        }
    }

    /// Counts an incremental result and says whether to verify it.
    fn due_for_check(&self, namespace: &str) -> bool {
        let Ok(mut inner) = self.inner.lock() else {
            return true;
        };
        let count = &mut inner.models.entry(namespace.to_owned()).or_default().0;
        *count += 1;
        *count <= VERIFY_FIRST || (*count - VERIFY_FIRST).is_multiple_of(self.verify_every)
    }

    /// The cached head reaching furthest: its index among `keys`, and tokens.
    fn longest(&self, keys: &[[u8; 32]]) -> Option<(usize, Arc<[i32]>)> {
        let mut inner = self.inner.lock().ok()?;
        inner.clock += 1;
        let clock = inner.clock;
        keys.iter().enumerate().rev().find_map(|(index, key)| {
            let entry = inner.entries.get_mut(key)?;
            entry.used = clock;
            Some((index, entry.tokens.clone()))
        })
    }

    fn insert(&self, key: [u8; 32], tokens: Arc<[i32]>) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        if tokens.len() > self.capacity {
            return;
        }
        inner.clock += 1;
        let used = inner.clock;
        let length = tokens.len();
        if let Some(previous) = inner.entries.insert(key, Entry { tokens, used }) {
            inner.tokens -= previous.tokens.len();
        }
        inner.tokens += length;
        while inner.tokens > self.capacity {
            let Some(oldest) = inner
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.used)
                .map(|(key, _)| *key)
            else {
                break;
            };
            if let Some(removed) = inner.entries.remove(&oldest) {
                inner.tokens -= removed.tokens.len();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Timeouts;
    use axum::{Json, Router, routing::post};
    use serde_json::Value;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A tokenizer like llama.cpp's in the way that matters here: special
    /// tokens are split out first, plain text is tokenized per piece (one
    /// token per character pair), and BOS is added on request. It also
    /// counts the characters it tokenizes.
    async fn fake(work: Arc<AtomicUsize>) -> Server {
        fn tokenize(text: &str, add_special: bool) -> Vec<i64> {
            let mut out = Vec::new();
            if add_special {
                out.push(1);
            }
            let mut rest = text;
            while !rest.is_empty() {
                if let Some(after) = rest.strip_prefix("<|turn|>") {
                    out.push(2);
                    rest = after;
                    continue;
                }
                let next = rest[1..].find("<|turn|>").map_or(rest.len(), |at| at + 1);
                let chars: Vec<char> = rest[..next].chars().collect();
                for pair in chars.chunks(2) {
                    out.push(1000 + pair.iter().map(|c| *c as i64).sum::<i64>());
                }
                rest = &rest[next..];
            }
            out
        }
        let app = Router::new().route(
            "/tokenize",
            post(move |Json(body): Json<Value>| {
                let work = work.clone();
                async move {
                    let text = body["content"].as_str().unwrap_or("");
                    work.fetch_add(text.len(), Ordering::SeqCst);
                    let add = body["add_special"].as_bool().unwrap_or(false);
                    Json(json!({"tokens": tokenize(text, add)}))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Server::new(&format!("http://{address}"), Timeouts::default()).unwrap()
    }

    /// Turns end with their text, then the next marker, as chat templates
    /// render them ("...content<|im_end|>").
    fn conversation(turns: usize) -> String {
        (0..turns)
            .map(|turn| {
                format!(
                    "<|turn|>user {turn}: {}",
                    "lorem ipsum ".repeat(300).trim_end()
                )
            })
            .collect::<String>()
            + "<|turn|>assistant"
    }

    #[tokio::test]
    async fn a_continuing_conversation_only_tokenizes_its_new_text() {
        let work = Arc::new(AtomicUsize::new(0));
        let server = fake(work.clone()).await;
        let boundaries = Boundaries::new(&["<|turn|>"], vec![1], vec![]);
        let cache = TokenCache::new(1 << 20, 1_000);
        let first = conversation(20);
        let once = cache
            .tokenize(&server, Some(&boundaries), "m", &first)
            .await
            .unwrap();
        assert_eq!(once.method, Method::Full);
        assert_eq!(once.tokens, server.tokenize(&first).await.unwrap());

        let second = conversation(21);
        work.store(0, Ordering::SeqCst);
        let again = cache
            .tokenize(&server, Some(&boundaries), "m", &second)
            .await
            .unwrap();
        // The verification of the second result tokenized it whole once;
        // the incremental part itself only touched the new turn.
        let incremental = work.load(Ordering::SeqCst) - second.len();
        assert!(
            incremental < 4_000,
            "tokenized {incremental} bytes for one new turn"
        );
        assert_eq!(again.method, Method::Incremental);
        assert!(!again.mismatch);
        assert_eq!(again.tokens, server.tokenize(&second).await.unwrap());
    }

    #[tokio::test]
    async fn a_mismatch_turns_the_cache_off_for_that_model() {
        let server = fake(Arc::new(AtomicUsize::new(0))).await;
        // Wrong boundaries: the fake's BOS is 1, not 7.
        let boundaries = Boundaries::new(&["<|turn|>"], vec![7], vec![]);
        let cache = TokenCache::new(1 << 20, 1_000);
        let text = conversation(20);
        let result = cache
            .tokenize(&server, Some(&boundaries), "m", &text)
            .await
            .unwrap();
        assert!(result.mismatch);
        assert_eq!(result.tokens, server.tokenize(&text).await.unwrap());
        assert!(cache.disabled("m"));
        assert!(!cache.disabled("other"));
    }

    #[test]
    fn cuts_fall_before_markers_and_never_after_whitespace() {
        let boundaries = Boundaries::new(&["<|turn|>", "<|end|>"], vec![], vec![]);
        let text = "<|turn|>a<|end|> <|turn|>b<|end|><|turn|>";
        assert_eq!(boundaries.cuts(text), vec![9, 26, 33]);
    }

    #[test]
    fn the_cache_keeps_to_its_capacity() {
        let cache = TokenCache::new(10, 1);
        cache.insert([1; 32], Arc::from(vec![0; 6]));
        cache.insert([2; 32], Arc::from(vec![0; 6]));
        assert!(cache.longest(&[[1; 32]]).is_none(), "the older head went");
        assert!(cache.longest(&[[2; 32]]).is_some());
    }
}
