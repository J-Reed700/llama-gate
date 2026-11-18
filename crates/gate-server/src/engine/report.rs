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
        }
        if let Some(ttft) = report.ttft_ms {
            metrics.observe(Timing::TimeToFirstToken, ttft as f64 / 1000.0);
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
