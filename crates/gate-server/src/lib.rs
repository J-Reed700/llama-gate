//! llama-gate: a KV-aware admission and fair-scheduling gateway for
//! llama.cpp servers.
//!
//! The crate wires the deterministic engine (`gate-core`) and the llama.cpp
//! adapter (`gate-llama`) into a reverse proxy:
//!
//! ```text
//!  clients ──▶ proxy (identify · measure · session) ──▶ llama-server(s)
//!                 │  ▲ grant / pause                        ▲
//!                 ▼  │                                      │ save · restore · erase
//!               runtime (one actor) ── Scheduler (pure) ── commands
//!                 ▲            │
//!                 │            └──▶ events, metrics, status, dashboard
//!         one watcher per backend
//! ```
//!
//! [`Gate`] is the whole program; `main.rs` only parses arguments, installs
//! logging, and (with `-- llama-server ...`) starts the server itself.
#![forbid(unsafe_code)]

pub mod clients;
pub mod config;
pub mod control;
pub mod engine;
pub mod http;
pub mod launcher;
pub mod listen;
pub mod metrics;
pub mod models;
pub mod snapshots;
pub mod units;

use crate::clients::{Identify, Names, Rules};
use crate::config::{BackendConfig, Config, EvictionMode};
use crate::engine::{BackendLink, Engine, Policies, watch_upstream};
use crate::http::AppState;
use crate::models::Models;
use crate::snapshots::SnapshotStore;
use gate_core::{BackendId, EvictionOrder, QueueOrder};
use gate_llama::{
    Capabilities, Dialects, Health, Measurer, MeasurerConfig, Server, Timeouts, TokenCache,
};
use std::future::Future;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex, RwLock};
use tokio::net::TcpListener;
use tokio::sync::watch;
use tracing::{info, warn};

/// The example configuration shipped with the gate, fully commented.
pub const EXAMPLE_CONFIG: &str = include_str!("../../../examples/llama-gate.toml");

/// Serves with the configured policies until `shutdown` resolves, then
/// refuses everything still waiting and lets running requests finish for up
/// to `server.drain`.
pub async fn serve(
    config: Config,
    listener: TcpListener,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    Gate::new(config).serve(listener, shutdown).await
}

/// The gate with its extension points. Everything not supplied comes from
/// the configuration:
///
/// ```no_run
/// # async fn example(config: gate_server::config::Config, listener: tokio::net::TcpListener) {
/// use gate_core::Fifo;
/// gate_server::Gate::new(config)
///     .queue_order(Fifo)
///     .serve(listener, std::future::pending())
///     .await
///     .unwrap();
/// # }
/// ```
pub struct Gate {
    config: Config,
    dialects: Dialects,
    policies: Policies,
    identify: Option<Arc<dyn Identify>>,
    source: Option<PathBuf>,
}

impl Gate {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            dialects: Dialects::standard(),
            policies: Policies::default(),
            identify: None,
            source: None,
        }
    }

    /// The request formats to schedule (default: [`Dialects::standard`]).
    pub fn dialects(mut self, dialects: Dialects) -> Self {
        self.dialects = dialects;
        self
    }

    /// Who goes next among requests of equal priority.
    pub fn queue_order(mut self, order: impl QueueOrder + 'static) -> Self {
        self.policies.queue_order = Some(Box::new(order));
        self
    }

    /// Which idle context gives way first.
    pub fn eviction_order(mut self, order: impl EvictionOrder + 'static) -> Self {
        self.policies.eviction_order = Some(Box::new(order));
        self
    }

    /// How requests are attributed to clients and classes (and whether
    /// they may be served at all).
    pub fn identify(mut self, identify: impl Identify + 'static) -> Self {
        self.identify = Some(Arc::new(identify));
        self
    }

    /// The configuration file to reload on SIGHUP or `POST {prefix}/reload`.
    pub fn reload_from(mut self, path: impl Into<PathBuf>) -> Self {
        self.source = Some(path.into());
        self
    }

    pub async fn serve(
        self,
        listener: TcpListener,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> std::io::Result<()> {
        let Gate {
            config,
            dialects,
            policies,
            identify,
            source,
        } = self;
        let invalid =
            |message: String| std::io::Error::new(std::io::ErrorKind::InvalidInput, message);
        config.validate().map_err(invalid)?;
        let config = Arc::new(config);
        let models = Arc::new(Models::default());
        let cache = Arc::new(TokenCache::new(
            config.admission.token_cache,
            config.admission.verify_every,
        ));
        let mut links = Vec::new();
        let mut watchers = Vec::new();
        for (index, backend) in config.effective_backends().into_iter().enumerate() {
            let id = BackendId(index as u16);
            let mut server = Server::new(&backend.url, Timeouts::default())
                .map_err(|error| invalid(format!("backend {}: {error}", backend.name)))?
                .for_model(backend.router.then(|| backend.model.clone()));
            let api_key = backend
                .api_key
                .as_ref()
                .and_then(|key| key.reveal())
                .map(str::to_owned);
            if let Some(key) = &api_key {
                server = server
                    .with_api_key(key)
                    .map_err(|error| invalid(format!("backend {}: {error}", backend.name)))?;
            }
            let model = models.register(&backend.model, &backend.aliases);
            let measurer = Measurer::new(
                server.clone(),
                Capabilities::default(),
                MeasurerConfig {
                    block: config.admission.prefix_block,
                    media_tokens: config.admission.media_tokens,
                    bytes_per_token: config.admission.bytes_per_token,
                    namespace: String::new(),
                },
            )
            .with_cache(cache.clone());
            let (measurer_sender, measurer) = watch::channel(measurer);
            let (health_sender, health) = watch::channel(Health::Down);
            let link = Arc::new(BackendLink {
                id,
                name: backend.name.clone(),
                model,
                server: server.clone(),
                router_model: backend.router.then(|| backend.model.clone()),
                api_key,
                health,
                measurer,
            });
            let store = open_store(&config, &backend);
            watchers.push((id, server, health_sender, measurer_sender, store.clone()));
            links.push((link, backend, store));
        }
        if let Some(default) = &config.server.default_model {
            models.set_default(default);
        }
        let backends: Vec<Arc<BackendLink>> = links.iter().map(|(link, ..)| link.clone()).collect();
        let (engine, handle) = Engine::new(config.clone(), links, models.clone(), policies);
        let mut engine = tokio::spawn(engine.run());
        for (id, server, health, measurer, store) in watchers {
            tokio::spawn(watch_upstream(
                id,
                server,
                handle.clone(),
                config.server.health_interval.0,
                health,
                measurer,
                config.admission.incremental,
                store,
            ));
        }

        let names = Arc::new(Mutex::new(Names::default()));
        let custom_identify = identify.is_some();
        let identify: Arc<dyn Identify> = match identify {
            Some(identify) => identify,
            None => Arc::new(Rules::new(
                &config.clients,
                &config.auth,
                &config.class_names(),
                names.clone(),
            )),
        };
        let state = Arc::new(AppState {
            live: RwLock::new(Arc::new(http::Live {
                config: config.clone(),
                identify,
            })),
            handle: handle.clone(),
            backends,
            models,
            dialects,
            names,
            custom_identify,
            tickets: AtomicU64::new(0),
            held: Arc::default(),
            source,
            reloading: tokio::sync::Mutex::new(()),
        });
        if !custom_identify {
            // Rule host names are re-resolved so recreated containers keep
            // their identity.
            let state = state.clone();
            tokio::spawn(async move {
                loop {
                    let every = state.config().clients.resolve_interval.0;
                    let identify = state.identify();
                    if let Some(upkeep) = identify.refresh() {
                        upkeep.await;
                    }
                    tokio::time::sleep(every).await;
                }
            });
        }
        #[cfg(unix)]
        if state.source.is_some() {
            let state = state.clone();
            tokio::spawn(async move {
                let Ok(mut hangup) =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup())
                else {
                    return;
                };
                while hangup.recv().await.is_some() {
                    match http::apply_reload(&state).await {
                        Ok(notes) => info!(?notes, "reloaded the configuration (SIGHUP)"),
                        Err(problem) => {
                            tracing::warn!(%problem, "the configuration was not reloaded")
                        }
                    }
                }
            });
        }

        let address = listener.local_addr()?;
        info!(
            listen = %address,
            backends = state.backends.len(),
            mode = ?config.server.mode,
            "llama-gate listening"
        );
        let app = http::router(state);
        let limits = listen::Limits {
            header_timeout: config.server.header_timeout.0,
            max_connections: config.server.max_connections,
        };
        let drain = config.server.drain.0;
        let (stopping, stopped) = tokio::sync::oneshot::channel::<()>();
        let vitals = handle.vitals.clone();
        let graceful = async move {
            shutdown.await;
            info!("shutting down: refusing queued requests and draining running ones");
            handle.vitals.drain();
            handle.shutdown().await;
            let _ = stopping.send(());
        };
        let server = listen::serve(listener, app, limits, graceful);
        // Without its engine the gate can only refuse: better to stop and be
        // restarted than to answer 503 to everything while looking up. (An
        // engine that stops because the gate is shutting down is expected.)
        let engine_lost = async move {
            let ended = (&mut engine).await;
            if vitals.draining() {
                std::future::pending::<()>().await;
            }
            ended
        };
        tokio::select! {
            result = server => result,
            () = async move {
                let _ = stopped.await;
                tokio::time::sleep(drain).await;
            } => {
                info!("drain period over; closing remaining connections");
                Ok(())
            }
            ended = engine_lost => {
                let why = match ended {
                    Err(error) if error.is_panic() => "the scheduler panicked",
                    _ => "the scheduler stopped",
                };
                tracing::error!("{why}; exiting so the gate can be restarted");
                Err(std::io::Error::other(why))
            }
        }
    }
}

/// Takes ownership of a swapping backend's snapshot directory. Without one,
/// the backend discards idle contexts instead of saving them.
fn open_store(config: &Config, backend: &BackendConfig) -> Option<SnapshotStore> {
    if config.eviction.mode != EvictionMode::Swap {
        return None;
    }
    let Some(directory) = backend.snapshots.as_deref() else {
        warn!(
            backend = backend.name,
            "eviction.mode is swap but this backend has no snapshot directory; its idle contexts will be discarded"
        );
        return None;
    };
    match SnapshotStore::open(directory) {
        Ok(store) => Some(store),
        Err(error) => {
            warn!(
                backend = backend.name,
                directory = %directory.display(),
                %error,
                "cannot use the snapshot directory; this backend's idle contexts will be discarded"
            );
            None
        }
    }
}
