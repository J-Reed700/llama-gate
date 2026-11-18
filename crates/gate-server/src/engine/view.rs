//! What the engine shows: the status document and the gauges.

use super::*;

impl Engine {
    pub(super) fn publish(&self) {
        let metrics = &self.metrics;
        metrics.set(
            "llama_gate_requests_waiting",
            String::new(),
            self.scheduler.waiting() as f64,
        );
        metrics.set(
            "llama_gate_requests_running",
            String::new(),
            self.running.len() as f64,
        );
        let status = self.scheduler.status(self.clock.now());
        for state in &self.backends {
            let name = labels(&[("backend", &state.link.name)]);
            metrics.set(
                "llama_gate_upstream_up",
                name.clone(),
                if state.up { 1.0 } else { 0.0 },
            );
            let Some(backend) = status
                .backends
                .iter()
                .find(|backend| backend.backend == state.link.id)
            else {
                continue;
            };
            metrics.set(
                "llama_gate_kv_capacity_tokens",
                name.clone(),
                backend.capacity as f64,
            );
            metrics.set(
                "llama_gate_kv_reserved_tokens",
                name.clone(),
                backend.usage as f64,
            );
            metrics.set(
                "llama_gate_kv_usage_ratio",
                name.clone(),
                if backend.capacity > 0 {
                    backend.usage as f64 / backend.capacity as f64
                } else {
                    0.0
                },
            );
            metrics.set(
                "llama_gate_snapshots",
                name.clone(),
                backend.snapshots as f64,
            );
            metrics.set(
                "llama_gate_snapshot_bytes",
                name,
                backend.snapshot_bytes as f64,
            );
        }
    }

    pub(super) fn status(&self) -> Value {
        let now = self.clock.now();
        let status = self.scheduler.status(now);
        let names: HashMap<TicketId, (&str, &str)> = self
            .waiting
            .iter()
            .map(|(ticket, waiter)| (*ticket, (&*waiter.client, &*waiter.class)))
            .chain(
                self.running
                    .iter()
                    .map(|(ticket, runner)| (*ticket, (&*runner.client, &*runner.class))),
            )
            .collect();
        let models = self.models.list();
        let backends: Vec<Value> = self
            .backends
            .iter()
            .map(|state| {
                let core = status
                    .backends
                    .iter()
                    .find(|backend| backend.backend == state.link.id);
                let slots: Vec<Value> = core
                    .map(|backend| {
                        backend
                            .slots
                            .iter()
                            .map(|slot| {
                                let (client, class) = slot
                                    .ticket
                                    .and_then(|ticket| names.get(&ticket).copied())
                                    .unzip();
                                json!({
                                    "index": slot.slot.index,
                                    "state": slot.state,
                                    "tokens": slot.tokens,
                                    "ticket": slot.ticket,
                                    "client": client,
                                    "class": class,
                                    "running_ms": slot.running_for,
                                    "started": slot.started,
                                    "pausable": slot.pausable,
                                    "reuse": slot.reuse,
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                json!({
                    "id": state.link.id,
                    "name": state.link.name,
                    "url": state.link.server.base(),
                    "model": self.models.name(state.link.model),
                    "up": state.up,
                    "layout": core.map(|backend| backend.layout),
                    "capacity": core.map(|backend| backend.capacity),
                    "usage": core.map(|backend| backend.usage),
                    "stray": core.map(|backend| backend.stray),
                    "swap": core.is_some_and(|backend| backend.swap),
                    "partial_reuse": core.is_some_and(|backend| backend.partial_reuse),
                    "incremental": self.config.admission.incremental,
                    "build": state.info.as_ref().and_then(|info| info.build.clone()),
                    "reported_model": state.info.as_ref().and_then(|info| info.model.clone()),
                    "capabilities": state.capabilities,
                    "slots": slots,
                    "snapshots": {
                        "count": core.map_or(0, |backend| backend.snapshots),
                        "bytes": core.map_or(0, |backend| backend.snapshot_bytes),
                    },
                })
            })
            .collect();
        let waiting: Vec<Value> = status
            .waiting
            .iter()
            .map(|waiting| {
                json!({
                    "ticket": waiting.ticket,
                    "client": names.get(&waiting.ticket).map(|(client, _)| *client),
                    "class": waiting.class,
                    "model": self.models.name(waiting.model),
                    "need": waiting.need,
                    "waited_ms": waiting.waited,
                    "bumped": waiting.bumped,
                    "paused": waiting.paused,
                })
            })
            .collect();
        // Requests in flight per client, counted in one pass.
        let mut in_flight: HashMap<&str, (usize, usize)> = HashMap::new();
        for waiter in self.waiting.values() {
            in_flight.entry(&waiter.client).or_default().0 += 1;
        }
        for runner in self.running.values() {
            in_flight.entry(&runner.client).or_default().1 += 1;
        }
        let clients: Vec<Value> = self
            .clients
            .iter()
            .map(|(name, stats)| {
                let (waiting, running) = in_flight.get(&**name).copied().unwrap_or_default();
                json!({
                    "name": name,
                    "service": stats.id.and_then(|id| self.scheduler.service(id)),
                    "requests": stats.requests,
                    "prompt_tokens": stats.prompt_tokens,
                    "cached_tokens": stats.cached_tokens,
                    "generated_tokens": stats.generated_tokens,
                    "waiting": waiting,
                    "running": running,
                })
            })
            .collect();
        json!({
            "version": env!("CARGO_PKG_VERSION"),
            "ready": self.backends.iter().any(|state| state.up),
            "mode": self.config.server.mode,
            "uptime_ms": self.started.elapsed().as_millis() as u64,
            "models": models.iter().map(|model| json!({
                "id": model.id,
                "name": model.name,
                "aliases": model.aliases,
                "backends": self.backends.iter().filter(|state| state.link.model == model.id).map(|state| state.link.id).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
            "backends": backends,
            "waiting": waiting,
            "clients": clients,
            "recent": self.recent.iter().take(50).collect::<Vec<_>>(),
        })
    }
}
