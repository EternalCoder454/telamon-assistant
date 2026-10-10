# Telamon (assistant)

The optional voice assistant of Telamon OS (formerly AtlasOS; local folders
still say AtlasOS). It's Rust + Qt 6.11 + Kirigami (CXX-Qt) with Telamon.Ui
2.0.10, and the voice pipeline is in `crates/voice-core`. Read
`docs/DESIGN.md` first, and change it along with the code.

**Phase: Functionable** (F.S.R.P, see `~/.claude/CLAUDE.md`).

The stack, build and look are Telamon Gates' (`~/Documents/Projects/AtlasOS/Telamon Gates`).
When in doubt, do what it does.

## Hard rules

- **Build and test inside the fedora:44 dev container**, never on the host:
  `scripts/dev.sh <command>`. The repo is at `/src` and the build cache at
  `/work` (`~/.cache/claude-builds/telamon-assistant`). The image builds on
  `localhost/telamon-gates-dev:44`.
- **Never run the GUI, the microphone or the speaker on the user's machine.**
  `scripts/smoke.sh` uses Xvfb, a private bus and recorded clips.
- **Never start local AI models from our work** (Zach's standing rule,
  2026-10-09). He uses Telamon Gates himself and needs the GPU and RAM free.
  - That means no llama-server, whisper.cpp, Kokoro, ONNX wake word or VAD,
    and nothing on the GPU, not even in tests.
  - Test with fakes, the recorded fixtures in `tests/audio`, or stand-in
    programs (see the fake `pw-record` and `llama-server` tests).
  - Leave real-model checks to Zach, and say so in every report.
  - Model-free builds and unit tests are fine, one heavy job at a time,
    after checking `pgrep -af 'cargo |cmake|podman run|llama-server|whisper'`
    and `podman ps`.
- **One heavy job at a time**, capped (`--memory=12g --cpus=12`, 8 jobs;
  `dev.sh` sets this).
- **Tools only read.** A new tool must never write, run a program the model
  names, or read a file's contents.
- **Never ship openWakeWord's pretrained wake-word models**, which are
  CC BY-NC-SA. Ours is `models/telamon.onnx`.
- Telamon.Ui is the installed `telamon-ui`. Never copy its components; ask
  the framework's owner.
- Commits are authored as
  `EternalHell <77252745+EternalCoder454@users.noreply.github.com>`.
  Licence: MIT. App ID `net.eterneon.telamon.assistant`.

## Commands

| Task | Command (from the repo root on the host) |
|---|---|
| Models | `scripts/dev.sh scripts/fetch-models.sh /work/models` |
| Unit tests | `scripts/dev.sh cargo test --workspace` |
| Lint | `scripts/dev.sh cargo clippy --workspace --all-targets -- -D warnings` |
| App build | `scripts/dev.sh bash -c 'cmake -S apps/telamon-assistant -B /work/build/dev -G Ninja && cmake --build /work/build/dev'` |
| Glow smoke (Xvfb). **Zach only**: the app loads whisper, Kokoro and the ONNX models | `scripts/dev.sh scripts/smoke.sh` → `out/smoke/glow.png` |
| Pipeline test (GPU, real models). **Zach only** | `DEV_GPU=1 DEV_LLM="$HOME/Documents/Projects/AtlasOS/Telamon Gates/out/agent-models" scripts/dev.sh scripts/test-pipeline.sh` (`REALTIME=1` for the benchmark) |
| Wake word training. **Zach only**: Kokoro generates the data | `scripts/train-wakeword.sh` |
