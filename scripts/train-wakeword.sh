#!/usr/bin/env bash
# Train the "Telamon" wake-word head in a fedora:44 podman container (CPU only).
#
#   scripts/train-wakeword.sh                  # all stages
#   scripts/train-wakeword.sh tts features     # chosen stages, in the order given
#   scripts/train-wakeword.sh score tests/audio/hey-telamon-time.wav
#
# Stages: image fetch tts features train eval clips
#   image     build localhost/telamon-wakeword:44 (Python 3.13, CPU torch, kokoro-onnx)
#   fetch     scripts/fetch-models.sh on the host (needs network)
#   tts       Kokoro speech for every English voice (held-out voices: see models/README.md)
#   features  augment, run melspectrogram + embedding model, store embeddings
#   train     train the head, export models/telamon.onnx
#   eval      held-out recall / false accepts per hour, writes models/telamon.json
#   clips     tests/audio/*.wav from the held-out voice am_michael
#   score F   per-chunk wake score of a WAV (reference streaming pipeline)
#   py M ...  run training.wakeword.M with the given arguments (e.g. py analyze --thr 0.9)
#
# Only one container job runs at a time, so the script waits while another
# heavy job (podman run, cargo, cmake, llama-server, whisper) is active.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image=localhost/telamon-wakeword:44
cd "$root"

wait_idle() {
    while true; do
        busy="$(pgrep -af 'cargo |cmake|podman run|podman build|llama-server|whisper' | grep -v -E 'pgrep|flatpak-spawn|bash -c' || true)"
        running="$(podman ps -q)"
        [[ -z "$busy" && -z "$running" ]] && return 0
        echo "another heavy job is running; waiting 60 s ..." >&2
        sleep 60
    done
}

run() { # run <cmd...> inside the container: no network, capped, no GPU
    wait_idle
    podman run --rm --init --network=none \
        --memory=12g --cpus=12 --security-opt label=disable \
        -v "$root:/src" -w /src \
        -e TELAMON_MODELS=/src/out/models -e OMP_NUM_THREADS=2 \
        "$image" "$@"
}

stage_image() {
    wait_idle
    podman build --security-opt label=disable --memory=12g -t "$image" \
        -f training/wakeword/Containerfile training/wakeword
}

stage_fetch() { scripts/fetch-models.sh; }
stage_tts() { run python -m training.wakeword.gen_tts --procs 6 --threads 2; }
stage_features() { run python -m training.wakeword.dataset --procs 10; }
stage_train() { run python -m training.wakeword.train; }
stage_eval() { run python -m training.wakeword.evaluate --write-meta; }
stage_clips() { run python -m training.wakeword.make_test_clips; }

if [[ $# -eq 0 ]]; then
    set -- fetch image tts features train eval clips
fi

while [[ $# -gt 0 ]]; do
    st="$1"
    shift
    case "$st" in
    score)
        run python training/wakeword/score_wav.py "$@"
        break
        ;;
    py) # py <module> [args]: run training.wakeword.<module> in the container
        mod="$1"
        shift
        run python -m "training.wakeword.$mod" "$@"
        break
        ;;
    image | fetch | tts | features | train | eval | clips)
        echo "=== $st ($(date +%T))"
        "stage_$st"
        ;;
    *)
        echo "unknown stage: $st" >&2
        exit 2
        ;;
    esac
done
