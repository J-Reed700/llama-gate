//! Finished requests and slot outcomes, as metrics, client totals, the
//! access log, and events.

use super::*;

impl Engine {
    pub(super) fn count_outcome(&self, event: &Event) {
        let (name, outcome) = match event {
            Event::Evicted { outcome, .. } => (
                "llama_gate_evictions_total",
                match outcome {
                    EvictOutcome::Saved { .. } => "saved",
                    EvictOutcome::Erased => "erased",
                    EvictOutcome::Failed => "failed",
                },
            ),
            Event::Restored { outcome, .. } => (
                "llama_gate_restores_total",
                match outcome {
                    RestoreOutcome::Restored { .. } => "restored",
                    RestoreOutcome::Failed => "failed",
                },
            ),
            _ => return,
        };
        self.metrics.count(name, labels(&[("outcome", outcome)]), 1);
    }

    /// Records a finished (or abandoned) request: metrics, client totals,
    /// the access log, the recent list, and the event stream.
    pub(super) fn record(&mut self, mut report: Report) {
        report.at_ms = self.clock.now();
        let metrics = &self.metrics;
        metrics.count(
            "llama_gate_requests_total",
            labels(&[("class", &report.class), ("outcome", report.outcome)]),
            1,
        );
        if let Some(generated) = report.generated {
            metrics.count(
                "llama_gate_generated_tokens_total",
                labels(&[("class", &report.class)]),
                generated,
            );
        }
        if let Some(cached) = report.cached {
            metrics.count(
                "llama_gate_cached_prompt_tokens_total",
                String::new(),
                cached,
            );
            if report.prompt > 0 {
                metrics.count("llama_gate_prefix_cache_queries_total", String::new(), 1);
                if cached > 0 {
                    metrics.count("llama_gate_prefix_cache_hits_total", String::new(), 1);
                }
            }
        }
        if let Some(ttft) = report.ttft_ms {
            metrics.observe(Timing::TimeToFirstToken, ttft as f64 / 1000.0);
            let prefill_ms = ttft.saturating_sub(report.measure_ms.saturating_add(report.queue_ms));
            metrics.observe(Timing::PrefillTime, prefill_ms as f64 / 1000.0);
            let decode_ms = report.total_ms.saturating_sub(ttft);
            metrics.observe(Timing::DecodeTime, decode_ms as f64 / 1000.0);
            if report.outcome == "ok"
                && let Some(generated) = report.generated
                && generated >= 2
            {
                let tpot = (decode_ms as f64 / 1000.0) / ((generated - 1) as f64);
                metrics.observe(Timing::TimePerOutputToken, tpot);
            }
        }
        metrics.observe(Timing::Duration, report.total_ms as f64 / 1000.0);
        // Once per request, all its waits together (a paused or bounced
        // request waits more than once).
        metrics.observe(Timing::QueueTime, report.queue_ms as f64 / 1000.0);
        metrics.observe(Timing::Measure, report.measure_ms as f64 / 1000.0);
        let stats = self.clients.entry(report.client.clone()).or_default();
        stats.requests += 1;
        stats.prompt_tokens += report.prompt;
        stats.cached_tokens += report.cached.unwrap_or(0);
        stats.generated_tokens += report.generated.unwrap_or(0);
        if self.config.observe.access_log {
            info!(
                ticket = %report.ticket,
                request_id = %report.request_id,
                client = %report.client,
                class = %report.class,
                model = report.model,
                dialect = report.dialect,
                backend = report.backend.as_deref().unwrap_or("-"),
                slot = report.slot,
                outcome = report.outcome,
                status = report.status,
                prompt = report.prompt,
                cached = report.cached,
                generated = report.generated,
                measure_ms = report.measure_ms,
                queue_ms = report.queue_ms,
                ttft_ms = report.ttft_ms,
                total_ms = report.total_ms,
                tokens_per_second = report.tokens_per_second.map(|rate| (rate * 10.0).round() / 10.0),
                preempted = report.preempted,
                bounced = report.bounced,
                "done"
            );
        }
        if let Ok(mut event) = serde_json::to_value(&report) {
            event["type"] = json!("finish");
            self.emit(event);
        }
        self.recent.push_front(report);
        self.recent.truncate(self.config.observe.recent);
    }
}

/// The HTTP status a refusal is answered with.
pub(super) fn rejection_status(rejection: &Rejection) -> u16 {
    match rejection {
        Rejection::TooLarge { .. } => 400,
        Rejection::Timeout { .. } | Rejection::QueueFull { .. } => 429,
        Rejection::UnknownClass => 500,
        Rejection::Unavailable { .. } | Rejection::ShuttingDown => 503,
    }
}

pub(super) fn rejection_reason(rejection: &Rejection) -> &'static str {
    match rejection {
        Rejection::TooLarge { .. } => "too_large",
        Rejection::Timeout { .. } => "timeout",
        Rejection::QueueFull { .. } => "queue_full",
        Rejection::UnknownClass => "unknown_class",
        Rejection::Unavailable { .. } => "unavailable",
        Rejection::ShuttingDown => "shutting_down",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::models::Models;

    fn base_report() -> Report {
        Report {
            ticket: TicketId(1),
            request_id: "req-1".into(),
            at_ms: 0,
            client: Arc::from("c"),
            class: Arc::from("standard"),
            model: "fake".into(),
            dialect: "openai-chat",
            backend: Some("default".into()),
            slot: Some(0),
            outcome: "ok",
            status: 200,
            prompt: 100,
            cached: Some(0),
            generated: Some(5),
            measure_ms: 10,
            queue_ms: 20,
            ttft_ms: Some(130),
            total_ms: 530,
            tokens_per_second: Some(10.0),
            preempted: 0,
            bounced: 0,
        }
    }

    #[test]
    fn golden_serving_metrics_record_tpot_prefill_decode_and_prefix_cache() {
        let config = Arc::new(Config::from_toml("").unwrap());
        let (mut engine, handle) = Engine::new(
            config,
            Vec::new(),
            Arc::new(Models::default()),
            Policies::default(),
        );

        // 1. Successful request with generated=5, ttft=130 (prefill=100ms), total=530 (decode=400ms => TPOT=100ms), cached=0.
        engine.record(base_report());

        // 2. Single-token request (generated=1): no TPOT sample, cache hit recorded.
        let mut one_tok = base_report();
        one_tok.generated = Some(1);
        one_tok.cached = Some(40);
        one_tok.ttft_ms = Some(5); // smaller than measure+queue -> saturates to 0ms prefill
        one_tok.total_ms = 25;
        engine.record(one_tok);

        // 3. Zero-token error request: no TPOT sample, cached=None -> no prefix cache query.
        let mut err = base_report();
        err.outcome = "error";
        err.generated = Some(0);
        err.cached = None;
        err.ttft_ms = None;
        engine.record(err);

        let rendered = handle.metrics.render();
        assert!(
            rendered.contains("llama_gate_time_per_output_token_seconds_count 1"),
            "{rendered}"
        );
        assert!(
            rendered.contains("llama_gate_time_per_output_token_seconds_sum 0.1"),
            "{rendered}"
        );
        assert!(
            rendered.contains("llama_gate_request_prefill_time_seconds_count 2"),
            "{rendered}"
        );
        assert!(
            rendered.contains("llama_gate_request_prefill_time_seconds_sum 0.1"),
            "{rendered}"
        );
        assert!(
            rendered.contains("llama_gate_request_decode_time_seconds_count 2"),
            "{rendered}"
        );
        assert!(
            rendered.contains("llama_gate_request_decode_time_seconds_sum 0.42"),
            "{rendered}"
        );
        assert!(
            rendered.contains("llama_gate_prefix_cache_queries_total 2"),
            "{rendered}"
        );
        assert!(
            rendered.contains("llama_gate_prefix_cache_hits_total 1"),
            "{rendered}"
        );
    }
}
