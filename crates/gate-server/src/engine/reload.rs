//! Applying a reloaded configuration.

use super::*;

impl Engine {
    /// Applies a reloaded configuration: classes, queue and eviction
    /// policy, admission settings. Backends and the listen address need a
    /// restart, which the returned notes say.
    pub(super) fn reconfigure(&mut self, config: Arc<Config>) -> Result<Vec<String>, String> {
        config.validate()?;
        let mut notes = Vec::new();
        let before: Vec<(String, String)> = self
            .config
            .effective_backends()
            .into_iter()
            .map(|backend| (backend.name, backend.url))
            .collect();
        let after: Vec<(String, String)> = config
            .effective_backends()
            .into_iter()
            .map(|backend| (backend.name, backend.url))
            .collect();
        if before != after {
            notes.push("backends changed: restart the gate to apply that".into());
        }
        if self.config.server.listen != config.server.listen {
            notes.push("server.listen changed: restart the gate to apply that".into());
        }
        let names = config.class_names();
        let default = names
            .iter()
            .position(|name| *name == config.clients.default_class)
            .unwrap_or(0);
        let old = self.class_names.clone();
        let class_of = |id: ClassId| {
            let index = old
                .get(usize::from(id.0))
                .and_then(|name| names.iter().position(|new| new == name))
                .unwrap_or(default);
            ClassId(index as u16)
        };
        self.scheduler
            .reconfigure(scheduler_config(&config), class_of);
        self.class_names = config.class_names();
        self.config = config;
        info!(notes = ?notes, "configuration reloaded");
        Ok(notes)
    }
}

pub fn scheduler_config(config: &Config) -> SchedulerConfig {
    let classes = config
        .classes
        .iter()
        .map(|(name, class)| Class {
            name: name.clone(),
            priority: class.priority,
            weight: class.weight,
            max_wait: class.max_wait.map(|wait| wait.millis()),
            max_queued_per_client: class.max_queued_per_client,
            preemptible: class.preemptible,
        })
        .collect();
    SchedulerConfig {
        classes,
        backfill: config.queue.backfill.enabled.then(|| Backfill {
            max_tokens: config.queue.backfill.max_tokens,
            window: config.queue.backfill.window.millis(),
        }),
        eviction: EvictionPolicy {
            partial: config.eviction.partial_reuse,
            min_snapshot_tokens: config.eviction.min_snapshot_tokens,
            min_restore_gain: config.eviction.min_restore_gain,
            snapshot_budget: config.eviction.budget.0,
            retry_after: config.eviction.retry.millis(),
        },
        max_preemptions: config.queue.max_preemptions,
        unavailable_after: (config.server.upstream_wait.millis() > 0)
            .then(|| config.server.upstream_wait.millis()),
    }
}
