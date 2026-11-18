//! `llama-gate -- llama-server ...`: the gate starts the server itself.
//!
//! The server's command line says what the gate would otherwise have to be
//! told: whether its KV cache is unified (llama.cpp does not report it), and
//! where it saves slots. The gate gives the server a private address (unless
//! one is given), adds a slot directory for swapping (unless one is given),
//! and restarts the server if it exits, backing off when it keeps failing.
use crate::config::{Config, EvictionMode, LayoutKind};
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::process::{Child, Command};
use tokio::sync::watch;
use tracing::{error, info, warn};

/// A server to start.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Launch {
    pub program: String,
    pub args: Vec<String>,
    /// Where the gate reaches it.
    pub url: String,
    /// Environment set for it, beside the gate's own.
    pub env: Vec<(String, String)>,
}

/// llama-server's setting for clearing idle slots into its own prompt cache.
const CACHE_IDLE_SLOTS: &str = "LLAMA_ARG_CACHE_IDLE_SLOTS";

/// Reads the server's command line, completes it, and points `config` at
/// the server it starts.
pub fn plan(command: &[String], config: &mut Config) -> Result<Launch, String> {
    let (program, args) = command
        .split_first()
        .ok_or("no server command given after --")?;
    let mut args = args.to_vec();
    let mut env = Vec::new();
    let port = match value_of(&args, &["--port"]) {
        Some(port) => port.to_owned(),
        None => {
            let port = free_port()?;
            args.extend(["--port".into(), port.to_string()]);
            if value_of(&args, &["--host"]).is_none() {
                args.extend(["--host".into(), "127.0.0.1".into()]);
            }
            port.to_string()
        }
    };
    let host = match value_of(&args, &["--host"]) {
        None | Some("0.0.0.0" | "::" | "[::]") => "127.0.0.1",
        Some(host) => host,
    };
    let url = format!("http://{host}:{port}");
    // llama-server unifies its cache when asked, and by default when it
    // picks the slot count itself.
    let unified = if has_flag(&args, &["-kvu", "--kv-unified"]) {
        true
    } else if has_flag(&args, &["-no-kvu", "--no-kv-unified"]) {
        false
    } else {
        value_of(&args, &["-np", "--parallel"]).is_none_or(|slots| slots == "-1")
    };
    config.kv.layout = if unified {
        LayoutKind::Unified
    } else {
        LayoutKind::Split
    };
    if config.eviction.mode == EvictionMode::Swap {
        let directory = match value_of(&args, &["--slot-save-path"]) {
            Some(directory) => PathBuf::from(directory),
            None => {
                let directory = config
                    .eviction
                    .directory
                    .clone()
                    .unwrap_or_else(default_slot_directory);
                std::fs::create_dir_all(&directory)
                    .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;
                args.extend([
                    "--slot-save-path".into(),
                    directory.to_string_lossy().into_owned(),
                ]);
                directory
            }
        };
        config.eviction.directory = Some(directory);
        // Its own prompt cache would clear idle slots as tasks arrive,
        // before the gate could save them. Set through the environment,
        // which builds without the setting ignore, unless chosen already.
        if !has_flag(&args, &["--cache-idle-slots", "--no-cache-idle-slots"])
            && std::env::var_os(CACHE_IDLE_SLOTS).is_none()
        {
            env.push((CACHE_IDLE_SLOTS.into(), "0".into()));
        }
    }
    config.backends.clear();
    config.server.upstream = url.clone();
    Ok(Launch {
        program: program.clone(),
        args,
        url,
        env,
    })
}

/// Runs the server until `stop` turns true, restarting it whenever it
/// exits: at once after a long run, with growing delays while it keeps
/// failing.
pub async fn supervise(launch: Launch, mut stop: watch::Receiver<bool>) {
    let mut delay = Duration::from_secs(1);
    loop {
        if *stop.borrow() {
            return;
        }
        let started = Instant::now();
        let mut child = match Command::new(&launch.program)
            .args(&launch.args)
            .envs(launch.env.iter().map(|(name, value)| (name, value)))
            .stdin(Stdio::null())
            .kill_on_drop(true)
            .spawn()
        {
            Ok(child) => child,
            Err(problem) => {
                error!(program = launch.program, %problem, "cannot start the server");
                if wait_or_stop(&mut stop, delay).await {
                    return;
                }
                delay = (delay * 2).min(Duration::from_secs(30));
                continue;
            }
        };
        info!(
            program = launch.program,
            pid = child.id(),
            url = launch.url,
            "server started"
        );
        tokio::select! {
            status = child.wait() => {
                if started.elapsed() > Duration::from_secs(60) {
                    delay = Duration::from_secs(1);
                }
                warn!(?status, restart_in = ?delay, "the server exited; restarting it");
                if wait_or_stop(&mut stop, delay).await {
                    return;
                }
                delay = (delay * 2).min(Duration::from_secs(30));
            }
            _ = stop.changed() => {
                terminate(&mut child).await;
                return;
            }
        }
    }
}

/// Sleeps for `delay`; true if `stop` turned true meanwhile.
async fn wait_or_stop(stop: &mut watch::Receiver<bool>, delay: Duration) -> bool {
    tokio::select! {
        _ = tokio::time::sleep(delay) => *stop.borrow(),
        _ = stop.changed() => true,
    }
}

/// Asks the server to exit, and makes it if it will not.
async fn terminate(child: &mut Child) {
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        use nix::sys::signal::{Signal, kill};
        use nix::unistd::Pid;
        let _ = kill(Pid::from_raw(pid as i32), Signal::SIGTERM);
        if tokio::time::timeout(Duration::from_secs(10), child.wait())
            .await
            .is_ok()
        {
            return;
        }
    }
    let _ = child.kill().await;
}

fn value_of<'a>(args: &'a [String], names: &[&str]) -> Option<&'a str> {
    args.iter().enumerate().find_map(|(index, arg)| {
        names.iter().find_map(|name| {
            if arg == name {
                args.get(index + 1).map(String::as_str)
            } else {
                arg.strip_prefix(name)
                    .and_then(|rest| rest.strip_prefix('='))
            }
        })
    })
}

fn has_flag(args: &[String], names: &[&str]) -> bool {
    args.iter().any(|arg| names.contains(&arg.as_str()))
}

fn free_port() -> Result<u16, String> {
    std::net::TcpListener::bind(("127.0.0.1", 0))
        .and_then(|listener| listener.local_addr())
        .map(|address| address.port())
        .map_err(|error| format!("cannot find a free port: {error}"))
}

fn default_slot_directory() -> PathBuf {
    let state = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/state")))
        .unwrap_or_else(std::env::temp_dir);
    state.join("llama-gate").join("slots")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn command(line: &str) -> Vec<String> {
        line.split_whitespace().map(str::to_owned).collect()
    }

    #[test]
    fn the_command_line_says_how_the_cache_is_shared() {
        let directory = tempfile::tempdir().unwrap();
        let mut config = Config::from_toml("").unwrap();
        config.eviction.directory = Some(directory.path().join("slots"));
        let launch = plan(&command("llama-server -m m.gguf -np 4"), &mut config).unwrap();
        assert_eq!(
            config.kv.layout,
            LayoutKind::Split,
            "an explicit -np splits it"
        );
        assert!(launch.url.starts_with("http://127.0.0.1:"));
        assert_eq!(config.server.upstream, launch.url);
        assert!(
            launch
                .args
                .windows(2)
                .any(|pair| pair[0] == "--slot-save-path")
        );
        assert!(directory.path().join("slots").is_dir());
        assert_eq!(
            launch.env,
            [(CACHE_IDLE_SLOTS.to_owned(), "0".to_owned())],
            "the server leaves idle slots to the gate"
        );

        let mut config = Config::from_toml("").unwrap();
        config.eviction.directory = Some(directory.path().join("slots"));
        let launch = plan(
            &command("llama-server -m m.gguf --cache-idle-slots"),
            &mut config,
        )
        .unwrap();
        assert!(launch.env.is_empty(), "an explicit choice stands");

        let mut config = Config::from_toml("[eviction]\nmode = \"discard\"").unwrap();
        let launch = plan(
            &command("llama-server -m m.gguf --port=9000 --host 0.0.0.0"),
            &mut config,
        )
        .unwrap();
        assert_eq!(
            config.kv.layout,
            LayoutKind::Unified,
            "automatic slots unify it"
        );
        assert_eq!(launch.url, "http://127.0.0.1:9000");
        assert!(!launch.args.iter().any(|arg| arg == "--slot-save-path"));

        let mut config = Config::from_toml("[eviction]\nmode = \"discard\"").unwrap();
        plan(&command("llama-server -np 2 -kvu"), &mut config).unwrap();
        assert_eq!(config.kv.layout, LayoutKind::Unified);
    }
}
