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
pub mod launcher;
pub mod metrics;
pub mod models;
pub mod proxy;
pub mod runtime;
pub mod snapshots;
pub mod units;

use crate::clients::{Identify, Names, Rules};
use crate::config::Config;
use crate::models::Models;
use crate::proxy::AppState;
use crate::runtime::{BackendLink, Engine, Policies, watch_upstream};
use gate_core::{BackendId, EvictionOrder, QueueOrder};
use gate_llama::{
    Capabilities, Dialects, Health, Measurer, MeasurerConfig, Server, Timeouts, TokenCache,
};
use std::future::Future;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex, RwLock};
use tokio::net::TcpListener;
use tokio::sync::watch;
use tracing::info;

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
            watchers.push((id, server, health_sender, measurer_sender));
            links.push((link, backend));
        }
        if let Some(default) = &config.server.default_model {
            models.set_default(default);
        }
        let backends: Vec<Arc<BackendLink>> = links.iter().map(|(link, _)| link.clone()).collect();
        let (engine, handle) = Engine::new(config.clone(), links, models.clone(), policies);
        tokio::spawn(engine.run());
        for (id, server, health, measurer) in watchers {
            tokio::spawn(watch_upstream(
                id,
                server,
                handle.clone(),
                config.server.health_interval.0,
                health,
                measurer,
                config.admission.incremental,
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
            config: RwLock::new(config.clone()),
            handle: handle.clone(),
            backends,
            models,
            dialects,
            identify: RwLock::new(identify),
            names,
            custom_identify,
            tickets: AtomicU64::new(0),
            source,
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
                    match proxy::apply_reload(&state).await {
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
        let app = proxy::router(state).into_make_service_with_connect_info::<SocketAddr>();
        let drain = config.server.drain.0;
        let (stopping, stopped) = tokio::sync::oneshot::channel::<()>();
        let graceful = async move {
            shutdown.await;
            info!("shutting down: refusing queued requests and draining running ones");
            handle.shutdown().await;
            let _ = stopping.send(());
        };
        let server = axum::serve(listener, app).with_graceful_shutdown(graceful);
        tokio::select! {
            result = server => result,
            () = async move {
                let _ = stopped.await;
                tokio::time::sleep(drain).await;
            } => {
                info!("drain period over; closing remaining connections");
                Ok(())
            }
        }
    }
}
