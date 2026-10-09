//! The assistant as QML sees it: on or off, what it is doing, and how loud
//! the voice is for the glow. The pipeline (voice-core) runs on a worker
//! thread; its events come back through `qt_thread().queue`.

#[cxx_qt::bridge]
pub mod qobject {
    unsafe extern "C++" {
        include!("cxx-qt-lib/qstring.h");
        type QString = cxx_qt_lib::QString;
    }

    extern "RustQt" {
        #[qobject]
        /// The user allowed the microphone and Telamon is on.
        #[qproperty(bool, enabled)]
        /// "off", "loading", "listening", "awake", "thinking", "speaking"
        /// or "error".
        #[qproperty(QString, phase)]
        /// The loudness of whoever speaks, 0..1, for the glow.
        #[qproperty(f64, level)]
        /// The last question, as heard.
        #[qproperty(QString, heard)]
        /// The last answer.
        #[qproperty(QString, reply)]
        /// What went wrong last; empty when nothing did.
        #[qproperty(QString, error)]
        #[namespace = "telamon_assistant"]
        type Assistant = super::AssistantRust;
    }

    unsafe extern "RustQt" {
        /// The user allows the microphone: turns Telamon on and remembers it.
        #[qinvokable]
        fn enable(self: Pin<&mut Assistant>);
        /// Turns Telamon off, closing the microphone, and remembers it.
        #[qinvokable]
        fn disable(self: Pin<&mut Assistant>);
        /// Starts listening when the user turned Telamon on before.
        #[qinvokable]
        fn start(self: Pin<&mut Assistant>);
    }

    impl cxx_qt::Threading for Assistant {}

    #[namespace = "rust::cxxqtlib1"]
    unsafe extern "C++" {
        include!("cxx-qt-lib/common.h");

        #[cxx_name = "make_unique"]
        fn assistant_make_unique() -> UniquePtr<Assistant>;
    }
}

use crate::settings;
use core::pin::Pin;
use cxx_qt::{CxxQtType, Threading};
use cxx_qt_lib::QString;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use voice_core::audio::{self, Source};
use voice_core::{Event, Phase};

pub struct AssistantRust {
    enabled: bool,
    phase: QString,
    level: f64,
    heard: QString,
    reply: QString,
    error: QString,
    /// Set to stop the running worker.
    stop: Option<Arc<AtomicBool>>,
    /// The worker's pw-record, for closing the microphone at once.
    mic: audio::MicSlot,
}

/// Held by the worker for its whole life: a worker started right after
/// "Turn Off" waits until the previous one (and its llama-server) is gone.
static WORKER: Mutex<()> = Mutex::new(());

impl Default for AssistantRust {
    fn default() -> Self {
        Self {
            enabled: false,
            phase: QString::from("off"),
            level: 0.0,
            heard: QString::default(),
            reply: QString::default(),
            error: QString::default(),
            stop: None,
            mic: audio::MicSlot::default(),
        }
    }
}

impl qobject::Assistant {
    pub fn enable(mut self: Pin<&mut Self>) {
        settings::set_enabled(true);
        self.as_mut().set_enabled(true);
        self.start();
    }

    pub fn disable(mut self: Pin<&mut Self>) {
        settings::set_enabled(false);
        if let Some(stop) = self.as_mut().rust_mut().stop.take() {
            stop.store(true, Ordering::Relaxed);
        }
        // The microphone closes now, not when the worker next looks.
        audio::close_mic(&self.rust().mic);
        self.as_mut().set_enabled(false);
        self.as_mut().set_level(0.0);
        self.set_phase(QString::from("off"));
    }

    pub fn start(mut self: Pin<&mut Self>) {
        let enabled = settings::enabled();
        self.as_mut().set_enabled(enabled);
        if !enabled || self.rust().stop.is_some() {
            return;
        }
        let stop = Arc::new(AtomicBool::new(false));
        self.as_mut().rust_mut().stop = Some(stop.clone());
        self.as_mut().set_error(QString::default());
        self.as_mut().set_phase(QString::from("loading"));
        let qt = self.qt_thread();
        let mic = self.rust().mic.clone();
        std::thread::spawn(move || {
            let _one = WORKER.lock().unwrap_or_else(|e| e.into_inner());
            let result = if stop.load(Ordering::Relaxed) {
                Ok(())
            } else {
                run(&stop, &mic, &qt)
            };
            let _ = qt.queue(move |mut a| {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                a.as_mut().rust_mut().stop = None;
                a.as_mut().set_level(0.0);
                match result {
                    Ok(()) => a.set_phase(QString::from("off")),
                    Err(e) => {
                        log::warn!("Telamon stopped: {e:#}");
                        a.as_mut()
                            .set_error(QString::from(format!("{e:#}").as_str()));
                        a.set_phase(QString::from("error"));
                    }
                }
            });
        });
    }
}

/// Where the models are: $TELAMON_ASSISTANT_MODELS, else
/// ~/.local/share/telamon-assistant/models, else the package's.
fn models_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("TELAMON_ASSISTANT_MODELS") {
        return dir.into();
    }
    let user = data_dir().join("models");
    if user.join("telamon.onnx").exists() {
        return user;
    }
    PathBuf::from("/usr/share/telamon-assistant/models")
}

fn data_dir() -> PathBuf {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("telamon-assistant")
}

fn run(
    stop: &Arc<AtomicBool>,
    mic: &audio::MicSlot,
    qt: &cxx_qt::CxxQtThread<qobject::Assistant>,
) -> anyhow::Result<()> {
    let mut cfg = voice_core::Config::new(models_dir());
    let options = settings::options();
    if !options.llama_url.is_empty() {
        cfg.llama_url = options.llama_url.clone();
    }
    if !options.voice.is_empty() {
        cfg.voice = options.voice.clone();
    }
    cfg.location = options.location.clone();
    // list_files is confined to it: no home, no Telamon.
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .filter(|h| h.is_absolute() && h != std::path::Path::new("/"))
        .ok_or_else(|| anyhow::anyhow!("HOME is not set"))?;
    let mut assistant = voice_core::Assistant::load(&cfg, home)?;

    // Our own llama-server when none answers at the address.
    let _server = if assistant.llm().ready() || !options.llama_url.is_empty() {
        None
    } else {
        let binary = voice_core::server::find_binary()
            .ok_or_else(|| anyhow::anyhow!("telamon-llama is not installed"))?;
        let model = if options.model.is_empty() {
            data_dir().join("models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
        } else {
            PathBuf::from(&options.model)
        };
        std::fs::create_dir_all(data_dir())?;
        Some(voice_core::server::Server::start(
            &binary,
            &model,
            8091,
            &data_dir().join("llama-server.log"),
        )?)
    };

    // A test feeds a recording instead of the microphone.
    let mut source: Box<dyn Source> = match std::env::var_os("TELAMON_ASSISTANT_TEST_WAV") {
        Some(wav) => Box::new(audio::WavSource::open(std::path::Path::new(&wav), true)?),
        None => Box::new(audio::PipeWireSource::open(stop.clone(), mic.clone())?),
    };
    let mut sink = audio::PipeWireSink { stop: stop.clone() };
    let mut last_level = 0.0f32;
    assistant.run(source.as_mut(), &mut sink, stop, &mut |event| {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        // Small changes are not worth a trip to the GUI thread.
        if let Event::Level(l) = event {
            if (l - last_level).abs() < 0.02 {
                return;
            }
            last_level = l;
        }
        let _ = qt.queue(move |mut a| apply(a.as_mut(), event));
    })?;
    // A recording ends; the microphone only when stopped.
    if std::env::var_os("TELAMON_ASSISTANT_TEST_WAV").is_some() {
        while !stop.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
    }
    Ok(())
}

fn apply(mut a: Pin<&mut qobject::Assistant>, event: Event) {
    match event {
        Event::Phase(p) => {
            if p == Phase::Listening {
                a.as_mut().set_level(0.0);
            }
            a.set_phase(QString::from(p.name()));
        }
        Event::Level(l) => a.set_level(l as f64),
        Event::Woke { score } => log::info!("woke ({score:.2})"),
        Event::Heard(t) => a.set_heard(QString::from(t.as_str())),
        Event::Tool(t) => log::info!("tool {t}"),
        Event::Reply(t) => a.set_reply(QString::from(t.as_str())),
        Event::Error(t) => {
            log::warn!("{t}");
            a.set_error(QString::from(t.as_str()));
        }
        Event::Timing(t) => log::info!("{t:?}"),
    }
}
