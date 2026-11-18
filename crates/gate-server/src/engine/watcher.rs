//! One watcher per backend: health, shape, and restarts, and the backend's
//! snapshot index, loaded off the engine when the model identity changes.

use super::*;

/// Watches one backend's health and shape, reporting readiness, outages,
/// and restarts to the engine. A restart is recognized by the server's task
/// ids going backwards or its shape changing, not by a single missed probe.
#[allow(clippy::too_many_arguments)]
pub async fn watch_upstream(
    backend: BackendId,
    server: Server,
    handle: Handle,
    interval: Duration,
    health: watch::Sender<Health>,
    measurer: watch::Sender<Measurer>,
    incremental: bool,
    store: Option<SnapshotStore>,
) {
    // The model identity whose snapshot index was last loaded. Loading reads
    // and reconciles the directory, so it happens here, off the engine.
    let mut indexed: Option<String> = None;
    let mut ready_before = false;
    let mut last_info: Option<ServerInfo> = None;
    let mut last_task: Option<i64> = None;
    let mut down_polls = 0u32;
    loop {
        let mut state = server.health().await;
        if state == Health::Ready {
            match server.info().await {
                Ok(info) => {
                    let slots = server.slots().await.unwrap_or_default();
                    let task = slots.iter().filter_map(|view| view.task).max();
                    // Task ids only grow during a server's life; after a
                    // restart every slot starts over.
                    let went_backwards =
                        matches!((last_task, task), (Some(before), Some(now)) if now < before);
                    let blind_restart = slots.is_empty() && down_polls >= 2;
                    let changed = last_info.as_ref() != Some(&info);
                    if !ready_before || changed || went_backwards || blind_restart || down_polls > 0
                    {
                        let capabilities = server.capabilities().await;
                        let boundaries = if incremental
                            && capabilities.tokenize
                            && capabilities.apply_template
                        {
                            Boundaries::learn(&server)
                                .await
                                .ok()
                                .flatten()
                                .map(Arc::new)
                        } else {
                            None
                        };
                        let namespace = namespace(&info);
                        let loaded = match &store {
                            Some(store) if indexed.as_deref() != Some(namespace.as_str()) => {
                                let (store, wanted) = (store.clone(), namespace.clone());
                                let loaded =
                                    tokio::task::spawn_blocking(move || store.load(&wanted))
                                        .await
                                        .ok();
                                if loaded.is_some() {
                                    indexed = Some(namespace.clone());
                                }
                                loaded
                            }
                            _ => None,
                        };
                        measurer.send_modify(|measurer| {
                            measurer.set_capabilities(capabilities);
                            measurer.set_namespace(namespace);
                            measurer.set_boundaries(boundaries);
                        });
                        handle.upstream(
                            backend,
                            Upstream::Ready(Box::new(Ready {
                                info: info.clone(),
                                capabilities,
                                slots,
                                restarted: ready_before
                                    && (changed || went_backwards || blind_restart),
                                loaded,
                            })),
                        );
                    } else {
                        handle.upstream(backend, Upstream::Alive);
                    }
                    ready_before = true;
                    last_info = Some(info);
                    if task.is_some() {
                        last_task = task;
                    }
                    down_polls = 0;
                }
                Err(_) => state = Health::Down,
            }
        }
        if state != Health::Ready {
            down_polls = down_polls.saturating_add(1);
            if down_polls == 1 && ready_before {
                handle.upstream(backend, Upstream::Down);
            }
        }
        health.send_replace(state);
        tokio::time::sleep(interval).await;
    }
}

/// Says what loading a backend's snapshot index found and repaired.
pub(super) fn describe_loaded(backend: &str, loaded: &Loaded) {
    let removed_gib = loaded.removed_bytes as f64 / f64::from(1u32 << 30);
    match &loaded.started_over {
        None => info!(
            backend,
            snapshots = loaded.index.len(),
            dropped = loaded.dropped,
            removed = loaded.removed,
            removed_gib,
            "loaded the snapshot index"
        ),
        Some(StartOver::Missing) => info!(
            backend,
            removed = loaded.removed,
            removed_gib,
            "no snapshot index yet; starting empty"
        ),
        Some(StartOver::OtherModel(model)) => info!(
            backend,
            previous_model = model,
            removed = loaded.removed,
            removed_gib,
            "the snapshots belong to another model; removed them"
        ),
        Some(StartOver::OtherVersion(version)) => warn!(
            backend,
            version,
            removed = loaded.removed,
            removed_gib,
            "the snapshot index is from an incompatible version; starting empty"
        ),
        Some(StartOver::Corrupt) => warn!(
            backend,
            removed = loaded.removed,
            removed_gib,
            "the snapshot index was unreadable; set it aside as llama-gate-index.json.corrupt and started empty"
        ),
        Some(StartOver::Unreadable(error)) => warn!(
            backend,
            %error,
            "could not read the snapshot index; starting empty and leaving the files until it can be read"
        ),
    }
}

/// Content addresses and snapshots are scoped to the model they belong to.
pub fn namespace(info: &ServerInfo) -> String {
    info.model.clone().unwrap_or_else(|| "unknown-model".into())
}

impl Handle {
    pub(super) fn upstream(&self, backend: BackendId, change: Upstream) {
        let _ = self.sender.send(Message::Upstream(backend, change));
    }
}
