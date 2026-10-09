//! A llama-server of Telamon's own (telamon-llama's, Vulkan), started when
//! none answers at the configured address, and stopped with this value.

use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
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

impl Server {
    /// Starts `binary` with `model` on 127.0.0.1:`port` and waits (up to two
    /// minutes) until it answers.
    pub fn start(binary: &Path, model: &Path, port: u16, log: &Path) -> Result<Self> {
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
        let mut server = Self { child };
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(2)))
            .build()
            .into();
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(120) {
            if let Some(status) = server.child.try_wait()? {
                return Err(anyhow!("llama-server stopped ({status})"));
            }
            if agent.get(format!("http://127.0.0.1:{port}/health")).call().is_ok() {
                return Ok(server);
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        Err(anyhow!("llama-server did not start in two minutes"))
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
