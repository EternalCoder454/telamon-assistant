//! Telamon (the assistant), Rust side. `cpp/main.cpp` only starts Qt and
//! loads the QML; the QObject QML talks to is here, and the voice pipeline
//! is the `voice-core` crate.

mod assistant;
mod settings;

use std::ffi::c_void;

// cxx-qt-build's generated initializer calls into cxx-qt-lib; keep the crate
// linked even while no bridge uses one of its types.
extern crate cxx_qt_lib;

// Who this app is, for the Telamon framework: `main.cpp`'s `telamon_app_init`
// and `telamon_app_ready` take the names, the logger and the crash hooks from
// it. The ID is the desktop file, the icon and the single-instance D-Bus name.
telamon_framework_ui::app! {
    name: "Telamon",
    id: "net.eterneon.telamon.assistant",
    repo: "telamon-assistant",
    ui: "2.0.10",
}

/// Whether the user turned Telamon on (for `--background`).
#[unsafe(no_mangle)]
pub extern "C" fn telamon_assistant_enabled() -> bool {
    settings::enabled()
}

/// Called once from `main.cpp`: the Assistant QObject, owned by the caller.
#[unsafe(no_mangle)]
pub extern "C" fn telamon_assistant_new() -> *mut c_void {
    let mut assistant = assistant::qobject::assistant_make_unique();
    assistant.pin_mut().start();
    assistant.into_raw().cast()
}
