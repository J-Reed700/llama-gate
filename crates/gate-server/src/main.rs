//! `llama-gate`: parse arguments, install logging, serve — and, given a
//! server command after `--`, start and supervise that server too.
use clap::Parser;
use gate_server::config::{Config, LogFormat, Mode};
use gate_server::{Gate, launcher};
use std::net::SocketAddr;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "llama-gate",
    version,
    about = "KV-aware admission and fair scheduling in front of llama.cpp servers",
    after_help = "Start a server too:  llama-gate -- llama-server -m model.gguf -c 65536"
)]
struct Arguments {
    /// Configuration file (TOML). Every key is optional. SIGHUP reloads it.
    #[arg(short, long, env = "LLAMA_GATE_CONFIG")]
    config: Option<PathBuf>,
    /// The llama.cpp server, overriding `server.upstream`.
    #[arg(long, env = "LLAMA_GATE_UPSTREAM")]
    upstream: Option<String>,
    /// Listen address, overriding `server.listen`.
    #[arg(long, env = "LLAMA_GATE_LISTEN")]
    listen: Option<SocketAddr>,
    /// enforce, observe, or bypass, overriding `server.mode`.
    #[arg(long, env = "LLAMA_GATE_MODE", value_parser = parse_mode)]
    mode: Option<Mode>,
    /// Print the fully commented example configuration and exit.
    #[arg(long)]
    print_config: bool,
    /// Validate the configuration and exit.
    #[arg(long)]
    check: bool,
    /// Exit 0 if the gate configured here answers its health endpoint (for
    /// container health checks; reads the same configuration, so it always
    /// probes the right address).
    #[arg(long)]
    probe: bool,
    /// A llama-server command line to start and keep running behind the
    /// gate. Its flags tell the gate how the server's cache is shared.
    #[arg(last = true)]
    server: Vec<String>,
}

fn parse_mode(text: &str) -> Result<Mode, String> {
    match text {
        "enforce" => Ok(Mode::Enforce),
        "observe" => Ok(Mode::Observe),
        "bypass" => Ok(Mode::Bypass),
        other => Err(format!("unknown mode {other:?} (enforce, observe, bypass)")),
    }
}

fn load(arguments: &Arguments) -> Result<Config, String> {
    let mut config = match &arguments.config {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
            Config::from_toml(&text).map_err(|error| format!("{}: {error}", path.display()))?
        }
        None => Config::from_toml("")?,
    };
    if let Some(upstream) = &arguments.upstream {
        config.server.upstream = upstream.clone();
        config.backends.clear();
    }
    if let Some(listen) = arguments.listen {
        config.server.listen = listen;
    }
    if let Some(mode) = arguments.mode {
        config.server.mode = mode;
    }
    config.validate()?;
    Ok(config)
}

#[tokio::main]
async fn main() {
    let arguments = Arguments::parse();
    if arguments.print_config {
        print!("{}", gate_server::EXAMPLE_CONFIG);
        return;
    }
    let mut config = match load(&arguments) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("llama-gate: invalid configuration: {error}");
            std::process::exit(2);
        }
    };
    let launch = if arguments.server.is_empty() {
        None
    } else {
        match launcher::plan(&arguments.server, &mut config) {
            Ok(launch) => Some(launch),
            Err(error) => {
                eprintln!("llama-gate: {error}");
                std::process::exit(2);
            }
        }
    };
    if arguments.check {
        println!("configuration is valid");
        return;
    }
    if arguments.probe {
        std::process::exit(probe(&config));
    }
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(config.observe.log_level.clone()));
    // Colour only for a terminal, never for log files or collectors.
    let colour = std::io::IsTerminal::is_terminal(&std::io::stdout());
    match config.observe.log_format {
        LogFormat::Json => tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .init(),
        LogFormat::Text => tracing_subscriber::fmt()
            .with_ansi(colour)
            .with_env_filter(filter)
            .init(),
    }
    let listener = match tokio::net::TcpListener::bind(config.server.listen).await {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!(
                "llama-gate: cannot listen on {}: {error}",
                config.server.listen
            );
            std::process::exit(1);
        }
    };
    let (stop, stopped) = tokio::sync::watch::channel(false);
    let supervisor = launch.map(|launch| tokio::spawn(launcher::supervise(launch, stopped)));
    let mut gate = Gate::new(config);
    if let Some(path) = &arguments.config {
        gate = gate.reload_from(path);
    }
    let result = gate.serve(listener, shutdown_signal()).await;
    // The server goes after the gate has drained.
    let _ = stop.send(true);
    if let Some(supervisor) = supervisor {
        let _ = supervisor.await;
    }
    if let Err(error) = result {
        eprintln!("llama-gate: {error}");
        std::process::exit(1);
    }
}

async fn shutdown_signal() {
    let interrupt = tokio::signal::ctrl_c();
    #[cfg(unix)]
    {
        let mut terminate =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(signal) => signal,
                Err(_) => {
                    let _ = interrupt.await;
                    return;
                }
            };
        tokio::select! {
            _ = interrupt => {}
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = interrupt.await;
    }
}

/// Asks the gate's own health endpoint whether it is up.
fn probe(config: &Config) -> i32 {
    use std::io::{Read, Write};
    let mut address = config.server.listen;
    if address.ip().is_unspecified() {
        address.set_ip(if address.is_ipv4() {
            std::net::Ipv4Addr::LOCALHOST.into()
        } else {
            std::net::Ipv6Addr::LOCALHOST.into()
        });
    }
    let timeout = std::time::Duration::from_secs(3);
    let Ok(mut stream) = std::net::TcpStream::connect_timeout(&address, timeout) else {
        return 1;
    };
    let _ = stream.set_read_timeout(Some(timeout));
    let path = format!(
        "{}/health",
        config.observe.path_prefix.trim_end_matches('/')
    );
    let request = format!("GET {path} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() {
        return 1;
    }
    let mut head = [0u8; 12];
    match stream.read_exact(&mut head) {
        Ok(()) if head.starts_with(b"HTTP/1.1 200") || head.starts_with(b"HTTP/1.0 200") => 0,
        _ => 1,
    }
}
