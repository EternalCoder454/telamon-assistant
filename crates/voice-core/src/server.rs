//! A llama-server of Telamon's own (telamon-llama's, Vulkan), started when
//! none answers at the configured address, and stopped with this value.

use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct Server {
    child: Child,
}

/// telamon-llama's llama-server, else one on $PATH.
pub fn find_binary() -> Option<PathBuf> {
    let packaged = PathBuf::from("/usr/libexec/telamon-llama/llama-server");
    if packaged.exists() {
        return Some(packaged);
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|p| p.join("llama-server"))
            .find(|p| p.exists())
    })
}

/// Telamon's own server, shared so Turn Off and the graphics memory cap can
/// stop it at once, even while it is still loading.
pub type ServerSlot = Arc<Mutex<Option<Server>>>;

/// Stops the server in `slot`, if any.
pub fn stop(slot: &ServerSlot) {
    let server = slot.lock().unwrap_or_else(|e| e.into_inner()).take();
    drop(server);
}

impl Server {
    /// Starts `binary` with `model` on 127.0.0.1:`port`; [`wait_ready`]
    /// waits until it answers.
    pub fn spawn(binary: &Path, model: &Path, port: u16, log: &Path) -> Result<Self> {
        if !model.exists() {
            return Err(anyhow!("there is no model at {}", model.display()));
        }
        let log = std::fs::File::create(log).with_context(|| format!("{}", log.display()))?;
        let child = Command::new(binary)
            .arg("-m")
            .arg(model)
            .args(["--jinja", "-ngl", "99", "-c", "4096", "--host", "127.0.0.1"])
            .args(["--port", &port.to_string()])
            .stdin(Stdio::null())
            .stdout(log.try_clone()?)
            .stderr(log)
            .spawn()
            .with_context(|| format!("cannot run {}", binary.display()))?;
        Ok(Self { child })
    }
}

/// Waits (up to two minutes) until the server in `slot` answers on `port`.
/// Gives up at once when `stop` is set or the server was taken (stopped).
pub fn wait_ready(slot: &ServerSlot, port: u16, stop: &AtomicBool) -> Result<()> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(2)))
        .build()
        .into();
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(120) {
        if stop.load(Ordering::Relaxed) {
            return Err(anyhow!("stopped while llama-server was loading"));
        }
        {
            let mut guard = slot.lock().unwrap_or_else(|e| e.into_inner());
            let Some(server) = guard.as_mut() else {
                return Err(anyhow!("llama-server was stopped while loading"));
            };
            if let Some(status) = server.child.try_wait()? {
                return Err(anyhow!("llama-server stopped ({status})"));
            }
        }
        if agent
            .get(format!("http://127.0.0.1:{port}/health"))
            .call()
            .is_ok()
        {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    Err(anyhow!("llama-server did not start in two minutes"))
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn a_loading_server_stops_at_once() {
        let dir = std::env::temp_dir().join(format!("telamon-llama-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // A server that never answers, as one loading a model.
        let fake = dir.join("llama-server");
        std::fs::write(&fake, "#!/bin/sh\nexec sleep 600\n").unwrap();
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
        let model = dir.join("model.gguf");
        std::fs::write(&model, "").unwrap();

        // Turn Off: the wait ends within a beat.
        let slot = ServerSlot::default();
        *slot.lock().unwrap() = Some(Server::spawn(&fake, &model, 9, &dir.join("log")).unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let s = stop.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            s.store(true, Ordering::Relaxed);
        });
        let t = Instant::now();
        assert!(wait_ready(&slot, 9, &stop).is_err());
        assert!(t.elapsed() < Duration::from_secs(3));

        // The cap takes the server out of the slot: the wait ends too.
        let other = slot.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            super::stop(&other);
        });
        let t = Instant::now();
        assert!(wait_ready(&slot, 9, &AtomicBool::new(false)).is_err());
        assert!(t.elapsed() < Duration::from_secs(3));
        assert!(slot.lock().unwrap().is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
