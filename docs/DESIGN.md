# Telamon: design

Telamon is the optional voice assistant of Telamon OS. Say "Hey Telamon,
what's the weather like?": a gold-to-bronze glow rises around every screen's
edges, follows your voice while you ask, pulses while Telamon thinks, and
follows Telamon's voice while it answers out loud. Everything runs on this
computer. The only thing that goes online is the weather tool, which sends the
place name to Open-Meteo.

**Phase: Functionable** (F.S.R.P). Security hardening, reliability and
performance work come later, in that order. Their open items are listed at the
end.

## Pipeline

```
pw-record (16 kHz s16) ──80 ms chunks──▶ wake word ──"Telamon"──▶ record question
                                        (openWakeWord:          (Silero VAD: ends after
                                         melspec → embedding     700 ms of silence; gives up
                                         → telamon.onnx)         after 4 s without speech)
        ┌───────────────────────────────────────────────────────────┘
        ▼
 whisper.cpp base.en ──text──▶ llama-server (/v1/chat/completions, --jinja tools,
 (Vulkan, else CPU)            reasoning_effort low, enable_thinking false)
                                   │  ◀── read-only tool calls (≤ 4 rounds)
                                   ▼
                     reply ─▶ sentences ─▶ Kokoro-82M (espeak-ng IPA → tokens → ONNX)
                                              │  next sentence synthesized while one plays
                                              ▼
                                    pw-play (24 kHz f32), paced; level → glow
```

| Stage | Choice | Why |
|---|---|---|
| Wake word | openWakeWord feature models + our own `models/telamon.onnx` head | It's small and on the CPU. The pretrained heads are CC BY-NC-SA, so ours is trained from Kokoro-synthesized speech (`scripts/train-wakeword.sh`) |
| Voice detection | Silero VAD v5 (ONNX, CPU) | Ends the question reliably, even in noise |
| Speech to text | whisper.cpp `base.en`, through whisper-rs with Vulkan | Fast on the Radeon; falls back to the CPU |
| Model | telamon-llama's `llama-server` (the one Telamon Gates uses), Qwen3-4B-Instruct-2507 Q4_K_M | Tool calling with `--jinja`. A 4B model answers in well under a second |
| Speech | Kokoro-82M (ONNX fp32; int8 is only ~15 % faster), voice `bm_george` | Apache-2.0 and natural; espeak-ng is run as a program, not linked, because it's GPL |
| Audio I/O | `pw-record` and `pw-play` (PipeWire) | Nothing is opened until the user turns Telamon on; no audio library is linked |

Moshi and Ollama were looked at and rejected (2026-10-09 research):
llama-server is already on Telamon OS for Gates.

## Code layout

- `crates/voice-core`: the pipeline, with no Qt in it.
  - `wake.rs`: openWakeWord's streaming features, ported (`models/README.md`).
  - `vad.rs`, `stt.rs`, `llm.rs`, `tts.rs`: the four model stages.
  - `tools.rs`: the read-only tools.
  - `audio.rs`: PipeWire and WAV sources and sinks, and loudness.
  - `pipeline.rs`: `Assistant::run`, the state machine, emitting `Event`s.
  - `server.rs`: starts Telamon's own llama-server when none answers.
  - `bin/telamon-voice.rs`: the pipeline as a CLI, for tests and benchmarks.
- `apps/telamon-assistant`: the Qt app (Rust + Qt 6.11 + Kirigami via CXX-Qt,
  Telamon.Ui 2.0.10).
  - `Main.qml`: a small window with a centred hero (orb and status), one
    accent button ("Allow Microphone and Turn On", or "Turn Off"), and the
    last exchange in a grouped section.
  - `GlowWindow.qml`: one per screen. A layer-shell overlay on Wayland
    (LayerOverlay, anchored to all edges, no exclusive zone, no keyboard,
    input-transparent); a frameless input-transparent tool window on X11. It
    isn't shown, so not drawn, while Telamon sleeps.
  - `--background` (the XDG autostart entry) exits at once unless Telamon is
    on.

Telamon.Ui's `TelamonEdgeGlow` draws inside a window and only in violet, and
the framework's screen glow (roadmap item 43, `AtlasScreenGlow`) isn't
released yet. So the gold glow is Telamon's own small `GlowWindow`. When the
framework ships its screen glow with a colour and level input, Telamon moves
to it.

## QObject API (`Assistant`, `src/assistant.rs`)

| Member | Type | Meaning |
|---|---|---|
| `enabled` | bool | The user allowed the microphone (saved as `[Assistant] Enabled`) |
| `phase` | string | `off`, `loading`, `listening`, `awake`, `thinking`, `speaking`, `error` |
| `level` | real 0..1 | The loudness of whoever speaks; drives the glow |
| `heard`, `reply`, `error` | string | The last question, answer and problem |
| `enable()`, `disable()`, `start()` | invokable | Turn on (saved), turn off (saved, mic closed), start if on |

Settings (`[Assistant]` in `~/.config/telamon-assistantrc`, the framework's settings file): `Enabled`,
`ServerUrl` (an existing llama-server; empty starts our own on 127.0.0.1:8091),
`Model` (a GGUF), `Voice` (Kokoro voice), `Location` (the weather's place;
empty means the time zone's city). Models are read from
`$TELAMON_ASSISTANT_MODELS`, else `~/.local/share/telamon-assistant/models`,
else `/usr/share/telamon-assistant/models` (`scripts/fetch-models.sh`).

## Graphics memory cap

A full graphics card crashed the desktop before, so while Telamon's models run
(whisper on Vulkan from load, plus Telamon's own llama-server), a watcher
(`voice-core/src/vram.rs`) reads `/sys/class/drm/card*/device/mem_info_vram_{used,total}`
every 2 s. The fullest card counts. Two readings in a row at or over the cap
(`[Assistant] VramCap`: `off`, `85`, `90`, `95` (default) or `98`; the
Safety section's combo box) stop everything: the microphone closes,
llama-server is killed, and the worker ends, which frees whisper. The phase
becomes `error`, with the reason. Telamon doesn't start, and isn't restarted,
until usage is 5 points under the cap. Then it starts again by itself if
still on. No readable sysfs means the cap is off. Gates has the same cap
(`feat/vram-cap`).

## Threading

The GUI thread never blocks. One worker thread owns the pipeline: it loads
the models, reads the microphone, runs every model and plays the answer. A
second, scoped thread synthesizes the next sentence while one plays. Events
come back to the QObject through `qt_thread().queue`, and level changes
under 0.02 are dropped. `disable()` sets an `AtomicBool`, which the worker
checks at least every 80 ms.

## Privilege and privacy

- **Off by default.** No microphone, no models loaded, and no process
  running until the user presses "Allow Microphone and Turn On". The choice
  is saved. Turning off kills `pw-record` at once.
- No new privilege. It runs as the user, with no D-Bus service, no polkit
  action and no root.
- **Read-only tools.** They only look: time and calendar, system stats from
  `/proc` and `/sys`, processes, network interfaces, location from the
  setting or time zone, weather, and file *names* under `$HOME`.
  `list_files` canonicalizes the path, so `..` and symlinks can't leave
  home. No tool writes, runs a program the model names, or reads a file's
  contents.
- **Network.** llama-server listens on 127.0.0.1 only. The weather tool sends
  the place name and its coordinates to open-meteo.com over HTTPS; nothing
  else leaves the computer.

## Attack surface

- **Speech is untrusted input.** Anyone within earshot, a video or a song can
  say "Hey Telamon". Because every tool only reads, the worst case is that
  Telamon reads out system facts or file names aloud in the room. Speaker
  verification and confirmation for anything sensitive belong to the Secure
  phase.
- **Model output is untrusted.** It's shown as plain text (QML `Text`,
  `PlainText` formatting in Telamon.Ui labels) and spoken. Tool arguments
  are parsed as JSON and checked for each tool.
- **Model files** are pinned by sha256 in `scripts/fetch-models.sh`.

## Failure modes

| Failure | Behaviour |
|---|---|
| A model file is missing | `phase = error`, and the message names the file |
| llama-server is missing or won't start | `phase = error` ("telamon-llama is not installed" or its log) |
| llama-server fails mid-question | Telamon says "Sorry, I can't reach my language model right now." and goes back to listening |
| Nobody speaks after the wake word | After 4 s, it goes back to listening |
| espeak-ng or pw-play is missing | `error` is set, and the reply text is still shown |
| `pw-record` ends (PipeWire restarted) | The worker ends, and the phase becomes `off` (restart: Reliable phase) |

## Performance budget (to measure, then meet in the Performant phase)

| What | Budget | Note |
|---|---|---|
| End of question to first audio | ≤ 1.5 s | STT + LLM + first sentence's TTS |
| Wake word to first audio | ≤ question length + 2 s | Includes the 700 ms end-of-speech wait |
| Idle CPU while listening | ≤ 3 % of one core | wake word every 80 ms |
| Idle RSS (models loaded) | ≤ 700 MB | whisper base, Kokoro, ONNX sessions; llama-server separate |
| Glow | drawn only while shown | 30–60 fps while active |

Measured numbers go in `benchmarks.md` (gitignored).

## Testing

- `scripts/dev.sh cargo test --workspace`: unit tests (tools, text
  cleaning, sentence splitting, `.npy` parsing).
- `scripts/test-pipeline.sh` (in the container, GPU, real models): recorded
  clips go through wake, STT, llama-server with a tool, and Kokoro; the
  spoken reply is transcribed back and checked.
- `scripts/smoke.sh` (in the container, Xvfb plus a private bus): the app hears
  "Hey Telamon" from a clip, and a screenshot shows the glow.

## Later phases (open items)

- **Secure:** speaker verification or a confirmation word, a rate limit on
  wakes, a sandboxed tool process, and a review of the weather privacy note.
- **Reliable:** restart `pw-record` on PipeWire restarts, recover a
  crashed llama-server, measure false wakes per hour on real speech with more
  training data (openWakeWord's ACAV negatives, real recorded voices), and
  share Gates' llama-server instead of a second one.
- **Performant:** barge-in with echo cancellation (PipeWire's
  `module-echo-cancel`), streamed LLM output into TTS, a smaller glow
  surface (four strips instead of a full-screen surface), and Kokoro on the
  GPU.
- **Packaging:** an RPM spec, CI, a model download inside the app, and a
  calendar tool over the user's calendars (Akonadi or ICS files).
