//! Telamon's settings, in the Telamon framework's settings file
//! (`[Assistant]` group). Off unless the user turned it on.

use telamon_framework_ui::telamon_framework_core::settings::Settings;

const GROUP: &str = "Assistant";

fn file() -> Settings {
    Settings::for_app(telamon_framework_ui::app_info())
}

fn get(key: &str) -> String {
    file()
        .get(GROUP, key)
        .unwrap_or_default()
        .trim()
        .to_string()
}

pub fn enabled() -> bool {
    get("Enabled") == "true"
}

pub fn set_enabled(on: bool) {
    if let Err(e) = file().set(GROUP, "Enabled", Some(if on { "true" } else { "false" })) {
        log::warn!("cannot save whether Telamon is on: {e}");
    }
}

pub struct Options {
    /// llama-server's address; empty: Telamon starts its own.
    pub llama_url: String,
    /// The GGUF model for its own llama-server; empty: the default.
    pub model: String,
    pub voice: String,
    pub location: String,
}

pub fn options() -> Options {
    Options {
        llama_url: get("ServerUrl"),
        model: get("Model"),
        voice: get("Voice"),
        location: get("Location"),
    }
}
