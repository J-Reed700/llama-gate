//! A fake llama-server with the behaviour that makes a gate necessary.
//!
//! - Every slot draws on one unified pool of `capacity` cells. A prompt is
//!   allocated when its request starts and grows by one cell per generated
//!   token; finished contexts stay resident until erased or replaced.
//! - Overfilling the pool fails every running request with "Context size has
//!   been exceeded", exactly like llama.cpp, and is recorded so tests can
//!   prove it never happened.
//! - Requests pinned to a busy slot (`id_slot`) wait for it; unpinned ones
//!   (embeddings) take any free slot, as llama.cpp does.
//! - A streaming answer's headers go out at once; its first token follows a
//!   prefill delay, during which the request can be cancelled.
//! - Slots can be saved, restored, and erased as files in a real directory.
//!
//! Tokenization is one token per whitespace-separated word, and
//! `/apply-template` joins the message texts, so the gate's measurement and
//! the fake's accounting agree exactly.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;

#[derive(Debug, Default, Clone)]
pub struct FakeSlot {
    pub tokens: u64,
    pub busy: bool,
    pub task: i64,
}

pub struct FakeState {
    pub capacity: u64,
    pub slots: Mutex<Vec<FakeSlot>>,
    pub overflowed: AtomicBool,
    pub actions: Mutex<Vec<String>>,
    pub directory: Option<PathBuf>,
    pub token_delay: Duration,
    /// Time before a streaming answer's first token.
    pub prefill: Mutex<Duration>,
    pub completions: AtomicU64,
    pub cancelled: AtomicU64,
    pub peak: AtomicU64,
    pub model: String,
    /// Answers this many more generation requests with 503 "Loading model",
    /// as llama.cpp does while it loads.
    pub loading: AtomicU64,
    next_task: AtomicI64,
    freed: Notify,
}

pub struct Fake {
    pub url: String,
    pub state: Arc<FakeState>,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for Fake {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl Fake {
    pub async fn start(
        capacity: u64,
        slots: usize,
        directory: Option<PathBuf>,
        token_delay: Duration,
    ) -> Self {
        Self::start_as("fake", capacity, slots, directory, token_delay).await
    }

    pub async fn start_as(
        model: &str,
        capacity: u64,
        slots: usize,
        directory: Option<PathBuf>,
        token_delay: Duration,
    ) -> Self {
        let state = Arc::new(FakeState {
            capacity,
            slots: Mutex::new(vec![
                FakeSlot {
                    task: -1,
                    ..FakeSlot::default()
                };
                slots
            ]),
            overflowed: AtomicBool::new(false),
            actions: Mutex::new(Vec::new()),
            directory,
            token_delay,
            prefill: Mutex::new(Duration::ZERO),
            completions: AtomicU64::new(0),
            cancelled: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            model: model.into(),
            loading: AtomicU64::new(0),
            next_task: AtomicI64::new(0),
            freed: Notify::new(),
        });
        let app = Router::new()
            .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
            .route("/props", get(props))
            .route("/slots", get(list_slots))
            .route("/slots/{id}", post(slot_action))
            .route("/tokenize", post(tokenize))
            .route("/apply-template", post(apply_template))
            .route("/v1/models", get(models))
            .route("/v1/chat/completions", post(chat))
            .route("/v1/embeddings", post(embeddings))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self { url, state, task }
    }

    pub fn actions(&self) -> Vec<String> {
        self.state.actions.lock().unwrap().clone()
    }

    pub fn set_prefill(&self, delay: Duration) {
        *self.state.prefill.lock().unwrap() = delay;
    }

    /// Restarts the server in place: same model, same flags, empty slots
    /// that have run no task yet.
    pub fn restart(&self) {
        for slot in self.state.slots.lock().unwrap().iter_mut() {
            *slot = FakeSlot {
                task: -1,
                ..FakeSlot::default()
            };
        }
    }
}

pub fn render(body: &Value) -> String {
    body.get("messages")
        .and_then(Value::as_array)
        .map(|messages| {
            messages
                .iter()
                .filter_map(|message| message.get("content").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default()
}

fn words(text: &str) -> u64 {
    text.split_whitespace().count() as u64
}

async fn props(State(state): State<Arc<FakeState>>) -> Json<Value> {
    let slots = state.slots.lock().unwrap().len();
    Json(json!({
        "default_generation_settings": {"n_ctx": state.capacity},
        "total_slots": slots,
        "model_path": format!("/models/{}.gguf", state.model),
        "build_info": "fake"
    }))
}

async fn models(State(state): State<Arc<FakeState>>) -> Json<Value> {
    Json(json!({"object": "list", "data": [{"id": state.model}]}))
}

async fn list_slots(State(state): State<Arc<FakeState>>) -> Json<Value> {
    let slots = state.slots.lock().unwrap();
    Json(Value::Array(
        slots
            .iter()
            .enumerate()
            .map(|(id, slot)| {
                json!({
                    "id": id,
                    "n_ctx": state.capacity,
                    "is_processing": slot.busy,
                    "id_task": slot.task,
                    "n_prompt_tokens": slot.tokens,
                    "next_token": [{"n_decoded": 0}]
                })
            })
            .collect(),
    ))
}

/// One token per word; the id is derived from the word, so equal texts
/// tokenize equally and different texts differ.
async fn tokenize(Json(body): Json<Value>) -> Json<Value> {
    let tokens: Vec<i64> = body["content"]
        .as_str()
        .unwrap_or("")
        .split_whitespace()
        .map(|word| {
            word.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
            }) as i64
                & 0x7fff_ffff
        })
        .collect();
    Json(json!({"tokens": tokens}))
}

async fn apply_template(Json(body): Json<Value>) -> Json<Value> {
    Json(json!({"prompt": render(&body)}))
}

async fn slot_action(
    State(state): State<Arc<FakeState>>,
    Path(id): Path<usize>,
    Query(query): Query<HashMap<String, String>>,
    body: Bytes,
) -> Response {
    let action = query.get("action").cloned().unwrap_or_default();
    let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let filename = body["filename"].as_str().unwrap_or("").to_owned();
    // Like llama.cpp, a slot action waits for the slot's task to stop.
    loop {
        let notified = state.freed.notified();
        if !state
            .slots
            .lock()
            .unwrap()
            .get(id)
            .is_some_and(|slot| slot.busy)
        {
            break;
        }
        notified.await;
    }
    state.actions.lock().unwrap().push(format!("{action}:{id}"));
    let mut slots = state.slots.lock().unwrap();
    let Some(slot) = slots.get_mut(id) else {
        return (StatusCode::BAD_REQUEST, "bad slot").into_response();
    };
    match action.as_str() {
        "erase" => {
            let erased = slot.tokens;
            slot.tokens = 0;
            drop(slots);
            state.freed.notify_waiters();
            Json(json!({"id_slot": id, "n_erased": erased})).into_response()
        }
        "save" | "restore" => {
            let Some(directory) = state.directory.clone() else {
                return (
                    StatusCode::NOT_IMPLEMENTED,
                    Json(json!({"error": {"code": 501, "message": "start with --slot-save-path", "type": "not_supported_error"}})),
                )
                    .into_response();
            };
            let path = directory.join(&filename);
            if action == "save" {
                std::fs::write(&path, slot.tokens.to_string()).unwrap();
                Json(json!({"id_slot": id, "filename": filename, "n_saved": slot.tokens, "n_written": slot.tokens * 100, "timings": {"save_ms": 1.0}})).into_response()
            } else {
                let Ok(text) = std::fs::read_to_string(&path) else {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({"error": {"message": "no file"}})),
                    )
                        .into_response();
                };
                let tokens: u64 = text.parse().unwrap_or(0);
                slot.tokens = tokens;
                let total: u64 = slots.iter().map(|slot| slot.tokens).sum();
                if total > state.capacity {
                    state.overflowed.store(true, Ordering::SeqCst);
                }
                Json(json!({"id_slot": id, "filename": filename, "n_restored": tokens, "n_read": tokens * 100, "timings": {"restore_ms": 1.0}})).into_response()
            }
        }
        _ => (StatusCode::BAD_REQUEST, "unknown action").into_response(),
    }
}

/// Clears a slot's busy flag however its request ends, counting requests
/// that end before completing.
struct Busy {
    state: Arc<FakeState>,
    slot: usize,
    done: bool,
}

impl Drop for Busy {
    fn drop(&mut self) {
        if !self.done {
            self.state.cancelled.fetch_add(1, Ordering::SeqCst);
        }
        if let Ok(mut slots) = self.state.slots.lock() {
            slots[self.slot].busy = false;
        }
        self.state.freed.notify_waiters();
    }
}

/// Allocates `cells` more in `slot`; false (and a recorded overflow) when
/// the pool cannot hold them, as llama.cpp would fail the batch.
fn grow(state: &FakeState, slot: usize, cells: u64, replace: bool) -> bool {
    let mut slots = state.slots.lock().unwrap();
    if replace {
        slots[slot].tokens = 0;
    }
    slots[slot].tokens += cells;
    let total: u64 = slots.iter().map(|slot| slot.tokens).sum();
    state.peak.fetch_max(total, Ordering::SeqCst);
    if total > state.capacity {
        state.overflowed.store(true, Ordering::SeqCst);
        return false;
    }
    true
}

/// Takes the pinned slot (waiting while it is busy) or any free one.
async fn take_slot(state: &Arc<FakeState>, pinned: Option<usize>) -> Busy {
    loop {
        let notified = state.freed.notified();
        {
            let mut slots = state.slots.lock().unwrap();
            let choice = match pinned {
                Some(slot) => (slot < slots.len() && !slots[slot].busy).then_some(slot),
                None => slots.iter().position(|slot| !slot.busy),
            };
            if let Some(slot) = choice {
                slots[slot].busy = true;
                slots[slot].task = state.next_task.fetch_add(1, Ordering::SeqCst);
                return Busy {
                    state: state.clone(),
                    slot,
                    done: false,
                };
            }
        }
        notified.await;
    }
}

async fn embeddings(State(state): State<Arc<FakeState>>, Json(body): Json<Value>) -> Response {
    let inputs: Vec<String> = match &body["input"] {
        Value::String(text) => vec![text.clone()],
        Value::Array(items) => items
            .iter()
            .filter_map(|item| item.as_str().map(str::to_owned))
            .collect(),
        _ => Vec::new(),
    };
    let mut data = Vec::new();
    for (index, input) in inputs.iter().enumerate() {
        let mut busy = take_slot(&state, None).await;
        if !grow(&state, busy.slot, words(input), true) {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Context size has been exceeded.",
            )
                .into_response();
        }
        tokio::time::sleep(state.token_delay).await;
        busy.done = true;
        data.push(json!({"object": "embedding", "index": index, "embedding": [0.1, 0.2]}));
    }
    state.completions.fetch_add(1, Ordering::SeqCst);
    Json(json!({"object": "list", "data": data, "usage": {"prompt_tokens": inputs.iter().map(|input| words(input)).sum::<u64>()}})).into_response()
}

async fn chat(State(state): State<Arc<FakeState>>, Json(body): Json<Value>) -> Response {
    if state
        .loading
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
            left.checked_sub(1)
        })
        .is_ok()
    {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}})),
        )
            .into_response();
    }
    let prompt = words(&render(&body));
    let hint = body["messages"]
        .as_array()
        .and_then(|messages| messages.last())
        .and_then(|message| message["content"].as_str())
        .and_then(|text| {
            text.split_whitespace()
                .find_map(|word| word.strip_prefix("gen:"))
        })
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(8);
    // `fail:N` in the last message: report an error after N tokens, inside
    // the 200 stream, as llama.cpp does.
    let fail_at = body["messages"]
        .as_array()
        .and_then(|messages| messages.last())
        .and_then(|message| message["content"].as_str())
        .and_then(|text| {
            text.split_whitespace()
                .find_map(|word| word.strip_prefix("fail:"))
        })
        .and_then(|value| value.parse::<u64>().ok());
    // As llama.cpp resolves it: the first of these present wins, and a
    // negative value means no limit.
    let limit = ["n_predict", "max_completion_tokens", "max_tokens"]
        .iter()
        .find_map(|key| body[*key].as_i64())
        .and_then(|limit| u64::try_from(limit).ok())
        .unwrap_or(u64::MAX);
    let generate = hint.min(limit);
    let streaming = body["stream"].as_bool().unwrap_or(false);
    let pinned = body["id_slot"]
        .as_i64()
        .filter(|slot| *slot >= 0)
        .map(|slot| slot as usize);

    let busy = take_slot(&state, pinned).await;
    let slot = busy.slot;
    if !grow(&state, slot, prompt, true) {
        drop(busy);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": {"code": 500, "message": "Context size has been exceeded.", "type": "server_error"}})),
        )
            .into_response();
    }
    let delay = state.token_delay;
    let prefill = *state.prefill.lock().unwrap();
    if !streaming {
        tokio::time::sleep(prefill).await;
        for _ in 0..generate {
            tokio::time::sleep(delay).await;
            if !grow(&state, slot, 1, false) {
                drop(busy);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Context size has been exceeded.",
                )
                    .into_response();
            }
        }
        let mut busy = busy;
        busy.done = true;
        drop(busy);
        state.completions.fetch_add(1, Ordering::SeqCst);
        return Json(json!({
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": generate},
            "timings": {"cache_n": 0, "prompt_n": prompt, "predicted_n": generate}
        }))
        .into_response();
    }
    // With `return_progress`, llama.cpp reports prompt processing as it
    // goes: role-only chunks carrying `prompt_progress`, before any output.
    let steps: u64 = if body["return_progress"].as_bool() == Some(true) && !prefill.is_zero() {
        4
    } else {
        0
    };
    let progress = futures_util::stream::unfold(0u64, move |step| async move {
        if step >= steps {
            return None;
        }
        tokio::time::sleep(prefill / steps as u32).await;
        let chunk = format!(
            "data: {}\n\n",
            json!({"choices": [{"delta": {"role": "assistant", "content": null}, "finish_reason": null, "index": 0}],
                   "prompt_progress": {"total": prompt, "cache": 0, "processed": prompt * (step + 1) / steps, "time_ms": 0}})
        );
        Some((Ok::<_, std::io::Error>(Bytes::from(chunk)), step + 1))
    });
    let stream = futures_util::stream::unfold((0u64, Some(busy)), move |(index, busy)| {
        let state = state.clone();
        async move {
            let mut busy = busy?;
            if index == 0 && !prefill.is_zero() && steps == 0 {
                tokio::time::sleep(prefill).await;
            }
            if fail_at == Some(index) {
                let chunk = "data: {\"error\":{\"code\":500,\"message\":\"the model crashed\",\"type\":\"server_error\"}}\n\n";
                drop(busy);
                return Some((Ok(Bytes::from(chunk)), (generate + 1, None)));
            }
            if index < generate {
                tokio::time::sleep(delay).await;
                if !grow(&state, slot, 1, false) {
                    let chunk =
                        "data: {\"error\":{\"message\":\"Context size has been exceeded.\"}}\n\n"
                            .to_owned();
                    drop(busy);
                    return Some((
                        Ok::<_, std::io::Error>(Bytes::from(chunk)),
                        (generate + 1, None),
                    ));
                }
                let chunk = format!(
                    "data: {}\n\n",
                    json!({"choices": [{"delta": {"content": "t "}}]})
                );
                Some((Ok(Bytes::from(chunk)), (index + 1, Some(busy))))
            } else if index == generate {
                state.completions.fetch_add(1, Ordering::SeqCst);
                let chunk = format!(
                    "data: {}\n\ndata: [DONE]\n\n",
                    json!({"choices": [{"delta": {}, "finish_reason": "stop"}], "timings": {"cache_n": 0, "prompt_n": prompt, "predicted_n": generate}})
                );
                busy.done = true;
                drop(busy);
                Some((Ok(Bytes::from(chunk)), (index + 1, None)))
            } else {
                None
            }
        }
    });
    Response::builder()
        .header("content-type", "text/event-stream")
        .body(axum::body::Body::from_stream(
            futures_util::StreamExt::chain(progress, stream),
        ))
        .unwrap()
}

/// A chat request of `prompt` words that asks to generate `generate` tokens.
pub fn chat_body(conversation: &str, prompt: usize, generate: u64, stream: bool) -> Value {
    let mut words: Vec<String> = (0..prompt.saturating_sub(1))
        .map(|index| format!("{conversation}{index}"))
        .collect();
    words.push(format!("gen:{generate}"));
    json!({
        "model": "fake",
        "stream": stream,
        "messages": [{"role": "user", "content": words.join(" ")}]
    })
}
