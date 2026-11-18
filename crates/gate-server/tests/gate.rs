//! End-to-end: the real gate in front of fake llama-servers that fail every
//! running request when their unified KV pool overflows.
#![allow(clippy::unwrap_used, clippy::expect_used)]
mod support;

use futures_util::StreamExt;
use gate_server::config::{BackendConfig, ClassConfig, Config, EvictionMode, KeyConfig, Secret};
use gate_server::units::Span;
use serde_json::{Value, json};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use support::{Fake, chat_body};

struct Gate {
    url: String,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for Gate {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        self.task.abort();
    }
}

/// Settings that make tests fast: quick health checks, ticks, and pings.
fn quick(url: &str) -> Config {
    let mut config = Config::from_toml("").unwrap();
    config.server.upstream = url.to_owned();
    config.server.health_interval = Span(Duration::from_millis(50));
    config.server.upstream_wait = Span(Duration::from_secs(5));
    config.kv.margin = 0;
    config.queue.tick = Span(Duration::from_millis(20));
    config.queue.keepalive_after = Span(Duration::from_millis(150));
    config.queue.keepalive_interval = Span(Duration::from_millis(50));
    config.admission.min_generation = 1;
    config.eviction.mode = EvictionMode::Discard;
    config.eviction.min_snapshot_tokens = 1;
    config.eviction.min_restore_gain = 1;
    config
}

async fn start(config: Config, source: Option<std::path::PathBuf>, admin: Option<&str>) -> Gate {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        let mut gate = gate_server::Gate::new(config);
        if let Some(source) = source {
            gate = gate.reload_from(source);
        }
        gate.serve(listener, async {
            let _ = stopped.await;
        })
        .await
        .unwrap();
    });
    let client = reqwest::Client::new();
    for _ in 0..200 {
        let mut request = client.get(format!("{url}/gate/status"));
        if let Some(key) = admin {
            request = request.bearer_auth(key);
        }
        if let Ok(response) = request.send().await
            && let Ok(status) = response.json::<Value>().await
            && status["ready"] == true
            && status["backends"]
                .as_array()
                .is_some_and(|backends| backends.iter().all(|backend| backend["up"] == true))
        {
            return Gate {
                url,
                stop: Some(stop),
                task,
            };
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("the gate never became ready");
}

async fn gate(fake: &Fake, tweak: impl FnOnce(&mut Config)) -> Gate {
    let mut config = quick(&fake.url);
    tweak(&mut config);
    start(config, None, None).await
}

async fn post(base: &str, body: Value) -> (u16, Value) {
    send(base, "/v1/chat/completions", body, &[]).await
}

async fn send(base: &str, path: &str, body: Value, headers: &[(&str, &str)]) -> (u16, Value) {
    let mut request = reqwest::Client::new()
        .post(format!("{base}{path}"))
        .json(&body);
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    let response = request.send().await.unwrap();
    let status = response.status().as_u16();
    let text = response.text().await.unwrap();
    (
        status,
        serde_json::from_str(&text).unwrap_or(Value::String(text)),
    )
}

async fn status_of(gate: &Gate) -> Value {
    reqwest::get(format!("{}/gate/status", gate.url))
        .await
        .unwrap()
        .json()
        .await
        .unwrap()
}

async fn metrics_of(gate: &Gate) -> String {
    reqwest::get(format!("{}/gate/metrics", gate.url))
        .await
        .unwrap()
        .text()
        .await
        .unwrap()
}

/// A burst of large requests sent straight to the server overfills its
/// pool. This is the failure the gate exists to prevent.
#[tokio::test]
async fn without_the_gate_a_burst_overfills_the_pool() {
    let fake = Fake::start(2_000, 2, None, Duration::from_millis(1)).await;
    let burst: Vec<_> = (0..4)
        .map(|index| post(&fake.url, chat_body(&format!("c{index}-"), 900, 300, false)))
        .collect();
    futures_util::future::join_all(burst).await;
    assert!(fake.state.overflowed.load(Ordering::SeqCst));
}

#[tokio::test]
async fn through_the_gate_the_same_burst_completes_without_overfilling() {
    let fake = Fake::start(2_000, 2, None, Duration::from_millis(1)).await;
    let gate = gate(&fake, |_| {}).await;
    let burst: Vec<_> = (0..6)
        .map(|index| {
            post(
                &gate.url,
                chat_body(&format!("c{index}-"), 900, 300, index % 2 == 0),
            )
        })
        .collect();
    for (status, body) in futures_util::future::join_all(burst).await {
        assert_eq!(status, 200, "{body}");
    }
    assert!(
        !fake.state.overflowed.load(Ordering::SeqCst),
        "peak {}",
        fake.state.peak.load(Ordering::SeqCst)
    );
    assert_eq!(fake.state.completions.load(Ordering::SeqCst), 6);
}

#[tokio::test]
async fn small_requests_share_the_server_concurrently() {
    let fake = Fake::start(20_000, 2, None, Duration::from_millis(20)).await;
    let gate = gate(&fake, |config| config.admission.default_generation = 100).await;
    let started = Instant::now();
    let pair: Vec<_> = (0..2)
        .map(|index| post(&gate.url, chat_body(&format!("s{index}-"), 50, 25, false)))
        .collect();
    for (status, _) in futures_util::future::join_all(pair).await {
        assert_eq!(status, 200);
    }
    // 25 tokens at 20 ms each: concurrent ≈ 0.5 s, serialized ≈ 1 s.
    assert!(
        started.elapsed() < Duration::from_millis(900),
        "{:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn a_queued_stream_receives_keepalives_then_its_answer() {
    let fake = Fake::start(2_000, 1, None, Duration::from_millis(10)).await;
    let gate = gate(&fake, |_| {}).await;
    let first = tokio::spawn(post(
        gate.url.clone().leak(),
        chat_body("long-", 500, 60, false),
    ));
    tokio::time::sleep(Duration::from_millis(50)).await;
    let response = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", gate.url))
        .json(&chat_body("queued-", 500, 5, true))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let text = response.text().await.unwrap();
    assert!(text.starts_with(": queued"), "{text}");
    assert!(text.contains("\"delta\""), "{text}");
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
    assert_eq!(first.await.unwrap().0, 200);
}

#[tokio::test]
async fn idle_contexts_are_swapped_to_disk_and_restored_when_their_conversation_returns() {
    let directory = tempfile::tempdir().unwrap();
    let fake = Fake::start(
        2_000,
        1,
        Some(directory.path().into()),
        Duration::from_millis(1),
    )
    .await;
    let path = directory.path().to_owned();
    let gate = gate(&fake, move |config| {
        config.eviction.mode = EvictionMode::Swap;
        config.eviction.directory = Some(path);
    })
    .await;
    let mut first = chat_body("alpha-", 900, 20, false);
    first["max_tokens"] = json!(100);
    assert_eq!(post(&gate.url, first.clone()).await.0, 200);
    let mut other = chat_body("beta-", 1000, 20, false);
    other["max_tokens"] = json!(100);
    assert_eq!(post(&gate.url, other).await.0, 200);

    // The first conversation continues: same prompt, one more message.
    let mut again = first.clone();
    again["messages"]
        .as_array_mut()
        .unwrap()
        .push(json!({"role": "user", "content": "and gen:5 more words here"}));
    assert_eq!(post(&gate.url, again).await.0, 200);

    let actions = fake.actions();
    assert!(
        actions.iter().any(|action| action.starts_with("save:")),
        "{actions:?}"
    );
    assert!(
        actions.iter().any(|action| action.starts_with("restore:")),
        "{actions:?}"
    );
    assert!(!fake.state.overflowed.load(Ordering::SeqCst));
    let index = std::fs::read_to_string(directory.path().join("llama-gate-index.json")).unwrap();
    assert!(
        index.contains("/models/fake.gguf"),
        "the index is scoped to the model"
    );
}

#[tokio::test]
async fn without_slot_save_support_the_gate_discards_instead() {
    let fake = Fake::start(2_000, 1, None, Duration::from_millis(1)).await;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().to_owned();
    let gate = gate(&fake, move |config| {
        config.eviction.mode = EvictionMode::Swap;
        config.eviction.directory = Some(path);
    })
    .await;
    for index in 0..3 {
        let mut body = chat_body(&format!("n{index}-"), 1200, 10, false);
        body["max_tokens"] = json!(100);
        assert_eq!(post(&gate.url, body).await.0, 200);
    }
    assert!(!fake.state.overflowed.load(Ordering::SeqCst));
    let status = status_of(&gate).await;
    assert_eq!(status["backends"][0]["swap"], false, "{status}");
}

#[tokio::test]
async fn a_client_that_disconnects_frees_its_slot() {
    let fake = Fake::start(4_000, 1, None, Duration::from_millis(20)).await;
    let gate = gate(&fake, |_| {}).await;
    let response = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", gate.url))
        .json(&chat_body("leaver-", 100, 10_000, true))
        .send()
        .await
        .unwrap();
    let mut stream = response.bytes_stream();
    let _ = stream.next().await;
    drop(stream);
    let started = Instant::now();
    let (status, _) = post(&gate.url, chat_body("next-", 100, 3, false)).await;
    assert_eq!(status, 200);
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "the slot was released promptly"
    );
}

#[tokio::test]
async fn a_prompt_too_large_for_the_server_is_refused_with_its_reason() {
    let fake = Fake::start(1_000, 1, None, Duration::from_millis(1)).await;
    let gate = gate(&fake, |_| {}).await;
    let (status, body) = post(&gate.url, chat_body("huge-", 1_500, 5, false)).await;
    assert_eq!(status, 400);
    assert_eq!(body["error"]["code"], "context_length_exceeded", "{body}");
    assert_eq!(
        fake.state.completions.load(Ordering::SeqCst),
        0,
        "it never reached the server"
    );
}

#[tokio::test]
async fn other_endpoints_pass_through_and_slot_actions_are_reserved() {
    let fake = Fake::start(1_000, 1, None, Duration::from_millis(1)).await;
    let gate = gate(&fake, |_| {}).await;
    let models: Value = reqwest::get(format!("{}/v1/models", gate.url))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(models["data"][0]["id"], "fake");
    let erase = reqwest::Client::new()
        .post(format!("{}/slots/0?action=erase", gate.url))
        .send()
        .await
        .unwrap();
    assert_eq!(erase.status(), 403);
}

#[tokio::test]
async fn metrics_status_and_the_dashboard_describe_the_gate() {
    let fake = Fake::start(4_000, 2, None, Duration::from_millis(1)).await;
    let gate = gate(&fake, |_| {}).await;
    assert_eq!(post(&gate.url, chat_body("m-", 20, 2, false)).await.0, 200);
    let metrics = metrics_of(&gate).await;
    assert!(
        metrics.contains("llama_gate_dispatches_total{class=\"standard\"} 1"),
        "{metrics}"
    );
    assert!(
        metrics.contains("llama_gate_kv_capacity_tokens{backend=\"default\"} 4000"),
        "{metrics}"
    );
    assert!(
        metrics.contains("llama_gate_time_to_first_token_seconds_count 1"),
        "{metrics}"
    );
    let status = status_of(&gate).await;
    assert_eq!(status["backends"][0]["layout"]["kind"], "unified");
    assert_eq!(
        status["backends"][0]["slots"].as_array().map(Vec::len),
        Some(2)
    );
    let recent = &status["recent"][0];
    assert_eq!(recent["outcome"], "ok", "{status}");
    assert_eq!(recent["prompt"], 20);
    assert_eq!(recent["generated"], 2);
    assert!(recent["ttft_ms"].is_u64());
    let page = reqwest::get(format!("{}/gate/", gate.url))
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(page.contains("llama-gate"));
}

// -------------------------------------------------------------- new ground

#[tokio::test]
async fn work_spreads_over_two_servers_and_neither_overflows() {
    let first = Fake::start(2_000, 1, None, Duration::from_millis(2)).await;
    let second = Fake::start(2_000, 1, None, Duration::from_millis(2)).await;
    let mut config = quick(&first.url);
    config.backends = vec![
        BackendConfig {
            name: "a".into(),
            url: first.url.clone(),
            model: "fake".into(),
            ..BackendConfig::default()
        },
        BackendConfig {
            name: "b".into(),
            url: second.url.clone(),
            model: "fake".into(),
            ..BackendConfig::default()
        },
    ];
    let gate = start(config, None, None).await;
    let burst: Vec<_> = (0..6)
        .map(|index| {
            post(
                &gate.url,
                chat_body(&format!("w{index}-"), 900, 100, index % 2 == 0),
            )
        })
        .collect();
    for (status, body) in futures_util::future::join_all(burst).await {
        assert_eq!(status, 200, "{body}");
    }
    for fake in [&first, &second] {
        assert!(!fake.state.overflowed.load(Ordering::SeqCst));
        assert!(
            fake.state.completions.load(Ordering::SeqCst) >= 2,
            "both servers worked"
        );
    }
}

#[tokio::test]
async fn a_server_that_dies_bounces_its_requests_to_a_live_one() {
    let doomed = Fake::start(2_000, 1, None, Duration::from_millis(1)).await;
    let survivor = Fake::start(2_000, 1, None, Duration::from_millis(1)).await;
    let mut config = quick(&doomed.url);
    // The watcher must not notice the death first.
    config.server.health_interval = Span(Duration::from_secs(30));
    config.backends = vec![
        BackendConfig {
            name: "doomed".into(),
            url: doomed.url.clone(),
            model: "fake".into(),
            ..BackendConfig::default()
        },
        BackendConfig {
            name: "survivor".into(),
            url: survivor.url.clone(),
            model: "fake".into(),
            ..BackendConfig::default()
        },
    ];
    let gate = start(config, None, None).await;
    drop(doomed);
    tokio::time::sleep(Duration::from_millis(50)).await;
    let requests: Vec<_> = (0..3)
        .map(|index| {
            post(
                &gate.url,
                chat_body(&format!("b{index}-"), 100, 5, index == 0),
            )
        })
        .collect();
    for (status, body) in futures_util::future::join_all(requests).await {
        assert_eq!(status, 200, "{body}");
    }
    assert_eq!(survivor.state.completions.load(Ordering::SeqCst), 3);
    let metrics = metrics_of(&gate).await;
    assert!(metrics.contains("llama_gate_bounces_total 1"), "{metrics}");
}

#[tokio::test]
async fn an_urgent_request_pauses_one_still_reading_its_prompt_and_both_complete() {
    urgent_request_pauses_a_prefill(false).await;
}

/// The same, with the paused request streaming llama.cpp's prompt-progress
/// events: progress is not output, so it stays pausable while it reads.
#[tokio::test]
async fn progress_events_leave_a_request_pausable() {
    urgent_request_pauses_a_prefill(true).await;
}

async fn urgent_request_pauses_a_prefill(progress: bool) {
    let fake = Fake::start(2_000, 2, None, Duration::from_millis(5)).await;
    fake.set_prefill(Duration::from_millis(800));
    let gate = gate(&fake, |config| {
        config.classes.insert(
            "agents".into(),
            ClassConfig {
                preemptible: true,
                ..ClassConfig::default()
            },
        );
        config.classes.insert(
            "interactive".into(),
            ClassConfig {
                priority: 10,
                ..ClassConfig::default()
            },
        );
        config.clients.default_class = "agents".into();
        config.clients.trust_class_header = true;
    })
    .await;
    let mut agent = chat_body("agent-", 1_200, 20, true);
    agent["max_tokens"] = json!(300);
    if progress {
        agent["return_progress"] = json!(true);
    }
    let agent = tokio::spawn({
        let url = gate.url.clone();
        async move {
            let response = reqwest::Client::new()
                .post(format!("{url}/v1/chat/completions"))
                .json(&agent)
                .send()
                .await
                .unwrap();
            let text = response.text().await.unwrap();
            (Instant::now(), text)
        }
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mut chat = chat_body("chat-", 900, 5, false);
    chat["max_tokens"] = json!(100);
    let (status, body) = send(
        &gate.url,
        "/v1/chat/completions",
        chat,
        &[("x-gate-class", "interactive")],
    )
    .await;
    let chat_done = Instant::now();
    assert_eq!(status, 200, "{body}");
    let (agent_done, text) = agent.await.unwrap();
    assert!(chat_done < agent_done, "the urgent request finished first");
    assert_eq!(
        text.matches("[DONE]").count(),
        1,
        "one seamless stream: {text}"
    );
    assert_eq!(
        text.matches("\"t \"").count(),
        20,
        "every token, once: {text}"
    );
    assert!(!fake.state.overflowed.load(Ordering::SeqCst));
    let metrics = metrics_of(&gate).await;
    assert!(
        metrics.contains("llama_gate_preemptions_total{outcome=\"paused\"} 1"),
        "{metrics}"
    );
    let status = status_of(&gate).await;
    let resumed = status["recent"]
        .as_array()
        .unwrap()
        .iter()
        .find(|report| report["preempted"] == 1)
        .expect("the agent's report");
    assert_eq!(resumed["outcome"], "ok");
}

#[tokio::test]
async fn keys_are_required_when_configured_and_the_gates_own_endpoints_need_an_admin_key() {
    let fake = Fake::start(4_000, 1, None, Duration::from_millis(1)).await;
    let mut config = quick(&fake.url);
    config.auth.required = true;
    config.auth.keys = vec![KeyConfig {
        key: Secret::new("sk-app"),
        client: "app".into(),
        class: None,
    }];
    config.auth.admin_keys = vec![Secret::new("sk-admin")];
    let gate = start(config, None, Some("sk-admin")).await;
    let (status, body) = post(&gate.url, chat_body("k-", 10, 2, false)).await;
    assert_eq!(status, 401, "{body}");
    let (status, _) = send(
        &gate.url,
        "/v1/chat/completions",
        chat_body("k-", 10, 2, false),
        &[("authorization", "Bearer sk-app")],
    )
    .await;
    assert_eq!(status, 200);
    let unauthorized = reqwest::get(format!("{}/gate/status", gate.url))
        .await
        .unwrap();
    assert_eq!(unauthorized.status(), 401);
    let status: Value = reqwest::Client::new()
        .get(format!("{}/gate/status", gate.url))
        .bearer_auth("sk-admin")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(status["clients"][0]["name"], "app");
}

#[tokio::test]
async fn embeddings_are_held_to_the_pool_like_any_request() {
    let fake = Fake::start(2_000, 2, None, Duration::from_millis(5)).await;
    fake.set_prefill(Duration::from_millis(300));
    let gate = gate(&fake, |_| {}).await;
    let mut chat = chat_body("e-", 1_200, 10, false);
    chat["max_tokens"] = json!(200);
    let chat = tokio::spawn(post(gate.url.clone().leak(), chat));
    tokio::time::sleep(Duration::from_millis(100)).await;
    let words: Vec<String> = (0..700).map(|index| format!("w{index}")).collect();
    let (status, body) = send(
        &gate.url,
        "/v1/embeddings",
        json!({"input": words.join(" ")}),
        &[],
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(chat.await.unwrap().0, 200);
    assert!(
        !fake.state.overflowed.load(Ordering::SeqCst),
        "peak {}",
        fake.state.peak.load(Ordering::SeqCst)
    );
}

#[tokio::test]
async fn a_reload_applies_new_classes_without_a_restart() {
    let fake = Fake::start(4_000, 1, None, Duration::from_millis(1)).await;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("gate.toml");
    let text = format!(
        "[server]\nupstream = \"{}\"\nhealth_interval = \"50ms\"\n[clients]\ntrust_class_header = true\n",
        fake.url
    );
    std::fs::write(&path, &text).unwrap();
    let config = Config::from_toml(&text).unwrap();
    let gate = start(config, Some(path.clone()), None).await;
    std::fs::write(
        &path,
        format!("{text}[classes.standard]\n[classes.vip]\npriority = 5\n"),
    )
    .unwrap();
    let response: Value = reqwest::Client::new()
        .post(format!("{}/gate/reload", gate.url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(response["reloaded"], true, "{response}");
    let (status, _) = send(
        &gate.url,
        "/v1/chat/completions",
        chat_body("v-", 10, 2, false),
        &[("x-gate-class", "vip")],
    )
    .await;
    assert_eq!(status, 200);
    let status = status_of(&gate).await;
    assert_eq!(status["recent"][0]["class"], "vip", "{status}");
}

#[tokio::test]
async fn requests_go_to_the_server_of_the_model_they_name() {
    let alpha = Fake::start_as("alpha", 4_000, 1, None, Duration::from_millis(1)).await;
    let beta = Fake::start_as("beta", 4_000, 1, None, Duration::from_millis(1)).await;
    let mut config = quick(&alpha.url);
    config.backends = vec![
        BackendConfig {
            name: "a".into(),
            url: alpha.url.clone(),
            model: "alpha".into(),
            ..BackendConfig::default()
        },
        BackendConfig {
            name: "b".into(),
            url: beta.url.clone(),
            model: "beta".into(),
            aliases: vec!["b-model".into()],
            ..BackendConfig::default()
        },
    ];
    let gate = start(config, None, None).await;
    let models: Value = reqwest::get(format!("{}/v1/models", gate.url))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ids: Vec<&str> = models["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|model| model["id"].as_str())
        .collect();
    assert_eq!(ids, ["alpha", "beta"]);
    let mut body = chat_body("r-", 10, 2, false);
    body["model"] = json!("b-model");
    assert_eq!(post(&gate.url, body).await.0, 200);
    body = chat_body("r-", 10, 2, false);
    body["model"] = json!("something-else");
    assert_eq!(post(&gate.url, body).await.0, 200);
    assert_eq!(beta.state.completions.load(Ordering::SeqCst), 1);
    assert_eq!(
        alpha.state.completions.load(Ordering::SeqCst),
        1,
        "unknown names go to the default (first) model"
    );
}

/// Time to first token ends at the first output, not at the first event:
/// llama.cpp's progress events arrive from the moment prefill starts.
#[tokio::test]
async fn time_to_first_token_waits_for_output_not_progress() {
    let fake = Fake::start(2_000, 2, None, Duration::from_millis(5)).await;
    fake.set_prefill(Duration::from_millis(600));
    let gate = gate(&fake, |_| {}).await;
    let mut body = chat_body("progress-", 300, 4, true);
    body["return_progress"] = json!(true);
    let text = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", gate.url))
        .json(&body)
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(
        text.matches("prompt_progress").count(),
        4,
        "progress is relayed: {text}"
    );
    assert_eq!(text.matches("\"t \"").count(), 4, "{text}");
    let status = status_of(&gate).await;
    let report = &status["recent"][0];
    let ttft = report["ttft_ms"].as_u64().expect("a time to first token");
    assert!(
        ttft >= 550,
        "measured to the first token, not the first progress event: {report}"
    );
    assert_eq!(report["outcome"], "ok");
}

/// A server that reports an error mid-stream (as llama.cpp does, inside a
/// 200 response) has not served the request: the client gets the error and
/// the gate records a failure, not a success.
#[tokio::test]
async fn an_error_reported_mid_stream_is_recorded_as_a_failure() {
    let fake = Fake::start(2_000, 1, None, Duration::from_millis(1)).await;
    let gate = gate(&fake, |_| {}).await;
    let mut body = chat_body("crash-", 100, 10, true);
    body["messages"][0]["content"] = json!("hello gen:10 fail:2");
    let text = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", gate.url))
        .json(&body)
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(text.contains("the model crashed"), "{text}");
    let status = status_of(&gate).await;
    assert_eq!(
        status["recent"][0]["outcome"], "error",
        "{}",
        status["recent"][0]
    );
}

/// A server still loading its model answers 503 before doing anything: the
/// request is placed again, not failed.
#[tokio::test]
async fn a_server_still_loading_its_model_is_waited_for() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("llama_gate=debug,gate_server=debug")
        .with_test_writer()
        .try_init();
    let fake = Fake::start(2_000, 2, None, Duration::from_millis(1)).await;
    let gate = gate(&fake, |_| {}).await;
    fake.state.loading.store(2, Ordering::SeqCst);
    let (status, body) = post(&gate.url, chat_body("loading-", 100, 4, false)).await;
    assert_eq!(status, 200, "{body}");
    let status = status_of(&gate).await;
    assert!(
        status["recent"][0]["bounced"].as_u64().unwrap() >= 1,
        "{}",
        status["recent"][0]
    );
}

/// A server restarted with the same model and flags looks the same from
/// outside, except that its slots have forgotten their tasks: the gate's
/// picture of those slots starts over.
#[tokio::test]
async fn a_plain_restart_is_noticed() {
    let fake = Fake::start(2_000, 2, None, Duration::from_millis(1)).await;
    let gate = gate(&fake, |_| {}).await;
    let (status, _) = post(&gate.url, chat_body("before-", 100, 4, false)).await;
    assert_eq!(status, 200);
    tokio::time::sleep(Duration::from_millis(300)).await;
    fake.restart();
    let mut noticed = false;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        if metrics_of(&gate)
            .await
            .contains("llama_gate_upstream_resets_total")
        {
            noticed = true;
            break;
        }
    }
    assert!(noticed, "{}", metrics_of(&gate).await);
}
