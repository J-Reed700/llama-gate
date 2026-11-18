//! Accepting connections.
//!
//! The gate runs its own loop over hyper rather than `axum::serve`, for
//! three limits a server facing many clients needs:
//!
//! - **A deadline on request headers.** `axum::serve` never gives hyper a
//!   timer, so hyper's header-read timeout silently never fires: a client
//!   could hold a connection open forever by sending headers one byte at a
//!   time.
//! - **A cap on open connections.** Past it, new connections wait to be
//!   accepted instead of exhausting file descriptors and memory.
//! - **Graceful shutdown.** On the signal the gate stops accepting, lets
//!   each connection finish the response it is sending (running requests
//!   drain), and closes idle keep-alive connections.
use axum::Router;
use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
use hyper_util::server::conn::auto::Builder;
use hyper_util::server::graceful::GracefulShutdown;
use hyper_util::service::TowerToHyperService;
use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tower::Service as _;
use tracing::{debug, warn};

/// What a client may take and hold.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub header_timeout: Duration,
    pub max_connections: usize,
}

/// Serves `router` on `listener` until `shutdown` resolves, then waits for
/// open connections to finish (the caller bounds that wait).
pub async fn serve(
    listener: TcpListener,
    router: Router,
    limits: Limits,
    shutdown: impl Future<Output = ()>,
) -> std::io::Result<()> {
    let mut make = router.into_make_service_with_connect_info::<SocketAddr>();
    let mut builder = Builder::new(TokioExecutor::new());
    builder
        .http1()
        .timer(TokioTimer::new())
        .header_read_timeout(limits.header_timeout);
    builder.http2().timer(TokioTimer::new());
    let open = Arc::new(Semaphore::new(limits.max_connections.max(1)));
    let graceful = GracefulShutdown::new();
    tokio::pin!(shutdown);
    loop {
        // Wait for room first, so a full gate leaves new connections in the
        // kernel's backlog rather than accepting what it cannot serve.
        let permit = tokio::select! {
            permit = open.clone().acquire_owned() => match permit {
                Ok(permit) => permit,
                Err(_) => break,
            },
            () = &mut shutdown => break,
        };
        let (stream, peer) = tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok(accepted) => accepted,
                Err(error) => {
                    // Out of descriptors or a connection reset before accept:
                    // pause briefly rather than spin.
                    warn!(%error, "accepting a connection failed");
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    continue;
                }
            },
            () = &mut shutdown => break,
        };
        let service = match make.call(peer).await {
            Ok(service) => service,
            Err(never) => match never {},
        };
        let connection = builder
            .serve_connection(TokioIo::new(stream), TowerToHyperService::new(service))
            .into_owned();
        let connection = graceful.watch(connection);
        tokio::spawn(async move {
            if let Err(error) = connection.await {
                debug!(%peer, %error, "connection ended with an error");
            }
            drop(permit);
        });
    }
    graceful.shutdown().await;
    Ok(())
}
