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
- **Before any GPU or model work**, run
  `pgrep -af 'cargo |cmake|podman run|llama-server|ollama|whisper'` and
  `podman ps`, and read `/sys/class/drm/card*/device/mem_info_vram_used`.
  Wait while another job runs. Stop every model you start.
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
| Glow smoke (Xvfb) | `scripts/dev.sh scripts/smoke.sh` → `out/smoke/glow.png` |
| Pipeline test (GPU, real models) | `DEV_GPU=1 DEV_LLM="$HOME/Documents/Projects/AtlasOS/Telamon Gates/out/agent-models" scripts/dev.sh scripts/test-pipeline.sh` (`REALTIME=1` for the benchmark) |
| Wake word training | `scripts/train-wakeword.sh` |
