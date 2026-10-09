#!/bin/bash
# Run a command in the fedora:44 build container, with the repo at /src, the
# build cache at /work (~/.cache/claude-builds/telamon-assistant) and the
# cargo caches in the podman volumes the Telamon apps share.
#   scripts/dev.sh <command...>     e.g. scripts/dev.sh cargo test --workspace
#   scripts/dev.sh                  an interactive shell
#   DEV_GPU=1 scripts/dev.sh ...    with the GPU (/dev/dri), for llama-server
#                                   and whisper.cpp on Vulkan
#   DEV_LLM=<dir>                   mounted read-only at /llm (GGUF models)
# The image starts from Telamon Gates' dev image (Qt 6.11, Kirigami, CXX-Qt's
# needs and telamon-ui 2.0.10 installed) and adds the voice stack's packages
# listed below; a changed list rebuilds it.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
image=localhost/telamon-assistant-dev:44
base=${DEV_BASE:-localhost/telamon-gates-dev:44}
work=${DEV_WORK:-$HOME/.cache/claude-builds/telamon-assistant}
mkdir -p "$work"
export TMPDIR=${TMPDIR:-$work/tmp}
mkdir -p "$TMPDIR"

# onnxruntime: the wake word, VAD and Kokoro. espeak-ng: Kokoro's phonemes
# (run as a program). clang and Vulkan: whisper-rs builds whisper.cpp with
# Vulkan. layer-shell-qt: the glow above every window on Wayland.
# pipewire-utils: pw-record and pw-play. telamon-llama comes from
# $DEV_LLAMA_RPM when given (Telamon Gates' packaging).
deps="onnxruntime onnxruntime-devel espeak-ng clang-devel cmake ninja-build
vulkan-headers vulkan-loader-devel glslc mesa-vulkan-drivers layer-shell-qt
layer-shell-qt-devel pipewire-utils python3"
deps=$(echo $deps)
# By default the RPM staged in the build cache, if any.
DEV_LLAMA_RPM=${DEV_LLAMA_RPM:-$(ls "$work"/llama/telamon-llama-*.rpm 2>/dev/null | head -n1)}
deps_sum=$(printf '%s %s' "$deps" "${DEV_LLAMA_RPM:+llama}" | sha256sum | cut -c1-16)

if [ "$(podman image inspect --format '{{index .Labels "deps"}}' "$image" 2>/dev/null)" != "$deps_sum" ]; then
    podman image exists "$base" || { echo "dev.sh: needs $base (Telamon Gates' scripts/dev.sh true)" >&2; exit 1; }
    mounts=()
    [ -n "${DEV_LLAMA_RPM:-}" ] && mounts=(-v "$(realpath "$DEV_LLAMA_RPM")":/llama.rpm:ro)
    ctr=$(podman run -d --security-opt label=disable -v atlas-dnf:/var/cache/libdnf5 "${mounts[@]}" \
        "$base" sleep infinity)
    trap 'podman rm -f -t 0 "$ctr" >/dev/null' EXIT
    # shellcheck disable=SC2086
    podman exec "$ctr" bash -c "dnf -y install $deps && if [ -e /llama.rpm ]; then dnf -y install /llama.rpm; fi" >&2
    podman commit --change "LABEL deps=$deps_sum" "$ctr" "$image" >/dev/null
    podman rm -f -t 0 "$ctr" >/dev/null
    trap - EXIT
fi

extra=()
if [ "${DEV_GPU:-0}" = 1 ]; then
    extra+=(--device /dev/dri --group-add keep-groups)
fi
if [ -n "${DEV_LLM:-}" ]; then
    extra+=(-v "$(realpath "$DEV_LLM")":/llm:ro)
fi

tty=()
[ -t 0 ] && tty=(-it)
# No relabelling (label=disable rather than :z/:Z, which would relabel the
# user's files); --init reaps and forwards signals. Capped: this PC is also
# the user's desktop, and uncapped parallel builds ran it out of memory.
exec podman run --rm --init "${tty[@]}" --security-opt label=disable \
    --memory="${DEV_MEMORY:-12g}" --cpus="${DEV_CPUS:-12}" \
    -e CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}" \
    -e CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}" \
    -v "$repo":/src -w /src \
    -v "$work":/work \
    -v atlas-cargo:/root/.cargo/registry \
    -v atlas-cargo-git:/root/.cargo/git \
    -v telamon-assistant-ccache:/root/.cache/ccache \
    -e CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/work/target}" \
    -e QMAKE=/usr/bin/qmake6 \
    "${extra[@]}" \
    "$image" bash -c '
        if command -v ccache >/dev/null; then
            export CMAKE_CXX_COMPILER_LAUNCHER=ccache CMAKE_C_COMPILER_LAUNCHER=ccache
        fi
        exec "$@"' bash "${@:-bash}"
