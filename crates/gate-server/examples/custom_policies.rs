//! Running llama-gate with your own policies.
//!
//! Every extension point is a small trait. This program swaps in four:
//!
//! - a queue order (highest response ratio next: short requests go first,
//!   and every request's priority grows the longer it waits);
//! - an eviction order (keep the longest contexts, which cost the most to
//!   rebuild);
//! - client identification by a tenant header;
//! - a request dialect for an endpoint the gate does not know.
//!
//! Run it with `cargo run --example custom_policies -- http://127.0.0.1:8081`.
use gate_core::{Candidate, ClassId, ClientId, EvictionOrder, QueueOrder, Resident, SlotId};
use gate_llama::{Dialect, Dialects, Prompt};
use gate_server::Gate;
use gate_server::clients::{Caller, Denied, Identify, Identity};
use gate_server::config::Config;
use serde_json::{Value, json};
use std::sync::Arc;

/// Highest Response Ratio Next: ratio = (waited + service) / service, where
/// service is the request's size. Small requests are favoured, and waiting
/// raises everyone's ratio, so nothing starves.
#[derive(Debug)]
struct HighestResponseRatio;

impl QueueOrder for HighestResponseRatio {
    fn key(&self, candidate: Candidate<'_>) -> f64 {
        let service = candidate.need.max(1) as f64;
        // Waiting time in token-equivalents: 1 ms ≈ 1 token.
        let ratio = (candidate.waited as f64 + service) / service;
        -ratio // lower keys go first
    }
}

/// Evicts the shortest idle context first.
#[derive(Debug)]
struct KeepLongContexts;

impl EvictionOrder for KeepLongContexts {
    fn rank(&self, _slot: SlotId, resident: &Resident) -> f64 {
        resident.tokens as f64
    }
}

/// Every tenant (named by an `x-tenant` header your auth proxy sets) is its
/// own client; tenants ending in "-batch" are class 1. Requests without a
/// tenant are turned away.
#[derive(Debug)]
struct ByTenant;

impl Identify for ByTenant {
    fn identify(&self, caller: &Caller<'_>) -> Result<Identity, Denied> {
        let tenant = caller
            .headers
            .get("x-tenant")
            .and_then(|value| value.to_str().ok())
            .ok_or(Denied::MissingKey)?;
        // A real implementation would intern names; a stable hash is enough
        // to show the shape.
        let client = ClientId(tenant.bytes().fold(0u32, |hash, byte| {
            hash.wrapping_mul(31).wrapping_add(u32::from(byte))
        }));
        let class = if tenant.ends_with("-batch") {
            ClassId(1)
        } else {
            ClassId(0)
        };
        Ok(Identity {
            client,
            name: Arc::from(tenant),
            class,
        })
    }
}

/// A made-up endpoint whose body is `{"text": ..., "limit": N}`.
#[derive(Debug)]
struct Summarize;

impl Dialect for Summarize {
    fn name(&self) -> &'static str {
        "summarize"
    }

    fn paths(&self) -> &'static [&'static str] {
        &["/v1/summarize"]
    }

    fn prompt(&self, body: &Value) -> Prompt {
        Prompt::Text(body["text"].as_str().unwrap_or_default().to_owned())
    }

    fn requested_generation(&self, body: &Value) -> Option<u64> {
        body["limit"].as_u64()
    }

    fn limit_generation(&self, body: &mut Value, tokens: u64) {
        body["limit"] = json!(tokens);
    }
}

#[tokio::main]
async fn main() {
    let upstream = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "http://127.0.0.1:8081".into());
    let mut config = Config::from_toml(
        r#"
        [classes.interactive]
        priority = 10
        [classes.batch]
        priority = 0
        [clients]
        default_class = "interactive"
        "#,
    )
    .expect("valid configuration");
    config.server.upstream = upstream;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080")
        .await
        .expect("port 8080 is free");
    Gate::new(config)
        .queue_order(HighestResponseRatio)
        .eviction_order(KeepLongContexts)
        .identify(ByTenant)
        .dialects(Dialects::standard().with(Summarize))
        .serve(listener, async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .expect("the gate ran");
}
