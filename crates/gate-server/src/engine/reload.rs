//! Applying a reloaded configuration.

use super::*;

impl Engine {
    /// Applies a reloaded configuration: classes, queue and eviction
    /// policy, admission settings. Backends and the listen address need a
    /// restart, which the returned notes say.
    pub(super) fn reconfigure(&mut self, config: Arc<Config>) -> Result<Vec<String>, String> {
        config.validate()?;
        let notes = Vec::new();
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

/// What a reloaded configuration changes that only takes effect at a
/// restart (or, for `[kv]`, when each server next reconnects). Everything
/// else applies at once: classes, client rules and keys, queue and eviction
/// policy, admission, observation.
pub fn restart_notes(current: &Config, next: &Config) -> Vec<String> {
    fn differs<T: Serialize>(a: &T, b: &T) -> bool {
        serde_json::to_value(a).ok() != serde_json::to_value(b).ok()
    }
    let mut notes = Vec::new();
    let mut restart =
        |what: &str| notes.push(format!("{what} changed: restart the gate to apply it"));
    if current.server.listen != next.server.listen {
        restart("server.listen");
    }
    if current.server.upstream != next.server.upstream
        || differs(&current.effective_backends(), &next.effective_backends())
    {
        restart("the backends");
    }
    if current.server.mode != next.server.mode {
        restart("server.mode");
    }
    for (name, changed) in [
        (
            "server.health_interval",
            current.server.health_interval.0 != next.server.health_interval.0,
        ),
        (
            "server.header_timeout",
            current.server.header_timeout.0 != next.server.header_timeout.0,
        ),
        (
            "server.max_connections",
            current.server.max_connections != next.server.max_connections,
        ),
        (
            "server.drain",
            current.server.drain.0 != next.server.drain.0,
        ),
        ("queue.tick", current.queue.tick.0 != next.queue.tick.0),
        ("queue.order", current.queue.order != next.queue.order),
        (
            "queue.input_weight",
            current.queue.input_weight != next.queue.input_weight,
        ),
        (
            "queue.cached_input_weight",
            current.queue.cached_input_weight != next.queue.cached_input_weight,
        ),
        (
            "queue.output_weight",
            current.queue.output_weight != next.queue.output_weight,
        ),
        ("eviction.mode", current.eviction.mode != next.eviction.mode),
        (
            "eviction.victim",
            current.eviction.victim != next.eviction.victim,
        ),
        (
            "eviction.directory",
            current.eviction.directory != next.eviction.directory,
        ),
        (
            "observe.path_prefix",
            current.observe.path_prefix != next.observe.path_prefix,
        ),
        (
            "observe.dashboard",
            current.observe.dashboard != next.observe.dashboard,
        ),
        (
            "admission.incremental",
            current.admission.incremental != next.admission.incremental,
        ),
        (
            "admission.token_cache",
            current.admission.token_cache != next.admission.token_cache,
        ),
    ] {
        if changed {
            restart(name);
        }
    }
    if differs(&current.kv, &next.kv) {
        notes.push("[kv] changed: it applies as each server next reconnects".into());
    }
    notes
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reload_says_what_waits_for_a_restart() {
        let current = Config::from_toml("").unwrap();
        let same = restart_notes(&current, &current);
        assert!(same.is_empty(), "{same:?}");

        let next = Config::from_toml(
            "[server]\nmode = \"observe\"\nhealth_interval = \"5s\"\n[kv]\nmargin = 4096\n[classes.standard]\n[classes.batch]\npriority = -1\n",
        )
        .unwrap();
        let notes = restart_notes(&current, &next).join("\n");
        assert!(notes.contains("server.mode"), "{notes}");
        assert!(notes.contains("server.health_interval"), "{notes}");
        assert!(notes.contains("[kv]"), "{notes}");
        assert!(!notes.contains("classes"), "classes apply at once: {notes}");
    }
}
