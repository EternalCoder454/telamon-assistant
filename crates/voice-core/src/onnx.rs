//! ONNX Runtime, the system's (Fedora's `onnxruntime`), loaded once at run
//! time. `ORT_DYLIB_PATH` points at another copy.

use anyhow::{Context, Result, anyhow};
use ort::session::Session;
use std::path::Path;
use std::sync::OnceLock;

static INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();

fn init() -> Result<()> {
    INIT.get_or_init(|| {
        let path =
            std::env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| "libonnxruntime.so.1".to_string());
        if !path.contains('/') && !Path::new("/usr/lib64").join(&path).exists() {
            return Err(format!("ONNX Runtime ({path}) is not installed"));
        }
        // Loading happens with the first session, whose error names it.
        let _ = ort::init_from(&path).with_name("telamon").commit();
        Ok(())
    })
    .clone()
    .map_err(|e| anyhow!(e))
}

/// A model from `path`, run on the CPU with `threads` threads.
pub fn session(path: &Path, threads: usize) -> Result<Session> {
    init()?;
    Session::builder()
        .and_then(|b| b.with_intra_threads(threads))
        .and_then(|b| b.with_inter_threads(1))
        .and_then(|b| b.commit_from_file(path))
        .with_context(|| format!("cannot load the model {}", path.display()))
}
