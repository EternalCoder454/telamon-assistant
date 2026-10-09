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
    /// The graphics memory cap: "off", "85", "90", "95" or "98" (default 95).
    pub vram_cap: String,
}

pub fn set_vram_cap(percent: i32) {
    let value = if percent <= 0 {
        "off".to_string()
    } else {
        percent.to_string()
    };
    if let Err(e) = file().set(GROUP, "VramCap", Some(&value)) {
        log::warn!("cannot save the graphics memory cap: {e}");
    }
}

pub fn options() -> Options {
    Options {
        llama_url: get("ServerUrl"),
        model: get("Model"),
        voice: get("Voice"),
        location: get("Location"),
        vram_cap: get("VramCap"),
    }
}
