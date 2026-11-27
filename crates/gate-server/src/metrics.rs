//! Prometheus metrics, named after the conventions of vLLM and llm-d so
//! existing dashboards translate directly.
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::Mutex;

/// Latency buckets in seconds, from interactive to batch scale.
const BUCKETS: [f64; 16] = [
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 900.0,
];

/// Help text and type for every metric the gate exports.
const CATALOG: &[(&str, &str, &str)] = &[
    (
        "llama_gate_requests_total",
        "counter",
        "Scheduled requests by class and outcome, counted once when they end.",
    ),
    (
        "llama_gate_dispatches_total",
        "counter",
        "Times requests were sent to a server, by class (a paused or bounced request is sent again).",
    ),
    (
        "llama_gate_rejections_total",
        "counter",
        "Requests refused, by class and reason.",
    ),
    (
        "llama_gate_overruns_total",
        "counter",
        "Contexts that finished larger than their reservation, by backend.",
    ),
    (
        "llama_gate_prompt_tokens_total",
        "counter",
        "Prompt tokens admitted, by measurement (exact, incremental, estimated).",
    ),
    (
        "llama_gate_cached_prompt_tokens_total",
        "counter",
        "Prompt tokens servers reused from their cache.",
    ),
    (
        "llama_gate_prefix_cache_queries_total",
        "counter",
        "Requests with prompt tokens checked for prefix cache reuse.",
    ),
    (
        "llama_gate_prefix_cache_hits_total",
        "counter",
        "Requests that reused at least one cached prompt token.",
    ),
    (
        "llama_gate_generated_tokens_total",
        "counter",
        "Tokens generated, by class.",
    ),
    (
        "llama_gate_evictions_total",
        "counter",
        "Idle contexts cleared to make room, by outcome.",
    ),
    (
        "llama_gate_restores_total",
        "counter",
        "Snapshot restores, by outcome.",
    ),
    (
        "llama_gate_preemptions_total",
        "counter",
        "Requests paused for more urgent work, by outcome (paused, declined).",
    ),
    (
        "llama_gate_bounces_total",
        "counter",
        "Requests placed again because their server refused the connection.",
    ),
    (
        "llama_gate_upstream_resets_total",
        "counter",
        "Times a server restarted or changed shape.",
    ),
    (
        "llama_gate_requests_waiting",
        "gauge",
        "Requests waiting for a slot.",
    ),
    (
        "llama_gate_requests_running",
        "gauge",
        "Requests running on a server.",
    ),
    (
        "llama_gate_kv_capacity_tokens",
        "gauge",
        "KV cells the gate schedules on each backend (after the margin).",
    ),
    (
        "llama_gate_kv_reserved_tokens",
        "gauge",
        "KV cells held or promised on each backend.",
    ),
    (
        "llama_gate_kv_usage_ratio",
        "gauge",
        "Reserved over capacity per backend, 0 to 1 (like vllm:kv_cache_usage_perc).",
    ),
    (
        "llama_gate_snapshots",
        "gauge",
        "Saved contexts on disk per backend.",
    ),
    (
        "llama_gate_snapshot_bytes",
        "gauge",
        "Bytes of saved contexts on disk per backend.",
    ),
    (
        "llama_gate_upstream_up",
        "gauge",
        "1 when the backend's server is ready.",
    ),
];

/// Histograms: name and help.
const HISTOGRAMS: &[(&str, &str)] = &[
    (
        "llama_gate_request_queue_time_seconds",
        "Time requests waited for a slot.",
    ),
    (
        "llama_gate_time_to_first_token_seconds",
        "Time from arrival to the first output reaching the client.",
    ),
    (
        "llama_gate_request_duration_seconds",
        "Time from arrival to the end of the response.",
    ),
    (
        "llama_gate_measure_seconds",
        "Time spent measuring prompts (template and tokenization).",
    ),
    (
        "llama_gate_time_per_output_token_seconds",
        "Time per output token after the first token.",
    ),
    (
        "llama_gate_request_prefill_time_seconds",
        "Time spent in prefill from dispatch to the first output.",
    ),
    (
        "llama_gate_request_decode_time_seconds",
        "Time spent decoding from the first output to completion.",
    ),
];

#[derive(Debug, Default)]
pub struct Metrics {
    counters: Mutex<BTreeMap<(&'static str, String), u64>>,
    gauges: Mutex<BTreeMap<(&'static str, String), f64>>,
    histograms: [Histogram; HISTOGRAMS.len()],
}

/// One histogram, updated and read whole, so a scrape never sees a bucket
/// above the total or a bucket above the next one.
#[derive(Debug, Default)]
struct Histogram {
    data: Mutex<HistogramData>,
}

#[derive(Clone, Copy, Debug, Default)]
struct HistogramData {
    counts: [u64; BUCKETS.len()],
    total: u64,
    /// Sum in microseconds.
    sum_micros: u64,
}

impl Histogram {
    fn observe(&self, seconds: f64) {
        let mut data = self
            .data
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for (bucket, count) in BUCKETS.iter().zip(&mut data.counts) {
            if seconds <= *bucket {
                *count += 1;
            }
        }
        data.total += 1;
        data.sum_micros += (seconds.max(0.0) * 1e6) as u64;
    }

    fn read(&self) -> HistogramData {
        *self
            .data
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// The histograms the gate records.
#[derive(Clone, Copy, Debug)]
pub enum Timing {
    QueueTime = 0,
    TimeToFirstToken = 1,
    Duration = 2,
    Measure = 3,
    TimePerOutputToken = 4,
    PrefillTime = 5,
    DecodeTime = 6,
}

impl Metrics {
    pub fn count(&self, name: &'static str, labels: impl Into<String>, by: u64) {
        if let Ok(mut counters) = self.counters.lock() {
            *counters.entry((name, labels.into())).or_default() += by;
        }
    }

    pub fn set(&self, name: &'static str, labels: impl Into<String>, value: f64) {
        if let Ok(mut gauges) = self.gauges.lock() {
            gauges.insert((name, labels.into()), value);
        }
    }

    pub fn observe(&self, timing: Timing, seconds: f64) {
        self.histograms[timing as usize].observe(seconds);
    }

    /// The Prometheus text exposition format.
    pub fn render(&self) -> String {
        let mut out = String::new();
        let counters = self.counters.lock().map(|c| c.clone()).unwrap_or_default();
        let gauges = self.gauges.lock().map(|g| g.clone()).unwrap_or_default();
        for (name, kind, help) in CATALOG {
            let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}");
            let series: Vec<(&String, String)> = if *kind == "counter" {
                counters
                    .iter()
                    .filter(|((counter, _), _)| counter == name)
                    .map(|((_, labels), value)| (labels, value.to_string()))
                    .collect()
            } else {
                gauges
                    .iter()
                    .filter(|((gauge, _), _)| gauge == name)
                    .map(|((_, labels), value)| (labels, value.to_string()))
                    .collect()
            };
            for (labels, value) in series {
                let labels = if labels.is_empty() {
                    String::new()
                } else {
                    format!("{{{labels}}}")
                };
                let _ = writeln!(out, "{name}{labels} {value}");
            }
        }
        for ((name, help), histogram) in HISTOGRAMS.iter().zip(&self.histograms) {
            let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} histogram");
            let data = histogram.read();
            for (bucket, count) in BUCKETS.iter().zip(&data.counts) {
                let _ = writeln!(out, "{name}_bucket{{le=\"{bucket}\"}} {count}");
            }
            let total = data.total;
            let _ = writeln!(out, "{name}_bucket{{le=\"+Inf\"}} {total}");
            let sum = data.sum_micros as f64 / 1e6;
            let _ = writeln!(out, "{name}_sum {sum}\n{name}_count {total}");
        }
        out
    }
}

/// Formats Prometheus labels, escaping values.
pub fn labels(pairs: &[(&str, &str)]) -> String {
    pairs
        .iter()
        .map(|(key, value)| {
            let value = value
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n");
            format!("{key}=\"{value}\"")
        })
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposition_includes_labels_buckets_and_help() {
        let metrics = Metrics::default();
        metrics.count(
            "llama_gate_requests_total",
            labels(&[("class", "a\"b"), ("outcome", "dispatched")]),
            2,
        );
        metrics.set(
            "llama_gate_kv_usage_ratio",
            labels(&[("backend", "gpu0")]),
            0.5,
        );
        metrics.observe(Timing::QueueTime, 0.2);
        metrics.observe(Timing::TimeToFirstToken, 1.5);
        metrics.observe(Timing::TimePerOutputToken, 0.04);
        metrics.observe(Timing::PrefillTime, 0.4);
        metrics.observe(Timing::DecodeTime, 0.8);
        metrics.count("llama_gate_prefix_cache_queries_total", String::new(), 3);
        metrics.count("llama_gate_prefix_cache_hits_total", String::new(), 2);
        let text = metrics.render();
        assert!(
            text.contains("llama_gate_requests_total{class=\"a\\\"b\",outcome=\"dispatched\"} 2"),
            "{text}"
        );
        assert!(text.contains("llama_gate_kv_usage_ratio{backend=\"gpu0\"} 0.5"));
        assert!(text.contains("llama_gate_request_queue_time_seconds_bucket{le=\"0.25\"} 1"));
        assert!(text.contains("llama_gate_request_queue_time_seconds_bucket{le=\"0.1\"} 0"));
        assert!(text.contains("llama_gate_time_to_first_token_seconds_bucket{le=\"2.5\"} 1"));
        assert!(text.contains("llama_gate_time_per_output_token_seconds_count 1"));
        assert!(text.contains("llama_gate_request_prefill_time_seconds_count 1"));
        assert!(text.contains("llama_gate_request_decode_time_seconds_count 1"));
        assert!(text.contains("llama_gate_prefix_cache_queries_total 3"));
        assert!(text.contains("llama_gate_prefix_cache_hits_total 2"));
        assert!(text.contains("# TYPE llama_gate_requests_waiting gauge"));
    }
}
