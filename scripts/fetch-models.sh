#!/usr/bin/env bash
# Download the third-party model files Telamon Assistant needs at runtime.
#
# Usage: scripts/fetch-models.sh [--llm] [dest]  (dest defaults to out/models;
#        --llm adds the default language model, 2.4 GiB)
#        scripts/fetch-models.sh --print-hashes [dest]   (maintainers: hash what is downloaded)
#
# Every file is pinned by sha256 and verified after download. Files that are
# already present with the right hash are left alone, so re-running is cheap.
#
# NOT downloaded, on purpose: openWakeWord's pretrained wake-word models
# (alexa, hey_jarvis, hey_mycroft, hey_rhasspy, timer, weather). They are
# CC BY-NC-SA 4.0 and must not ship. Our own head is models/telamon.onnx.
#
# Licences and the exact feature-extraction algorithm: models/README.md
set -euo pipefail

print_hashes=0
llm=0
while [[ "${1:-}" == --* ]]; do
    case $1 in
        --print-hashes) print_hashes=1 ;;
        # The language model the app starts its own llama-server with by
        # default (2.4 GiB). Telamon Gates' copy is the same file.
        --llm) llm=1 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
    shift
done

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="${1:-$root/out/models}"

OWW=https://github.com/dscripka/openWakeWord/releases/download/v0.5.1
# Silero VAD v5.1.2, pinned by commit.
SILERO=https://raw.githubusercontent.com/snakers4/silero-vad/6478567951ae5c9979ad7b234185b5515f4be7a1/src/silero_vad/data
KOKORO=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
# whisper.cpp model repo, pinned by commit.
WHISPER=https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1

# Kokoro-82M HF repo, pinned by commit (config.json holds the phoneme vocab).
KOKORO_HF=https://huggingface.co/hexgrad/Kokoro-82M/resolve/f3ff3571791e39611d31c381e3a41a3af07b4987

# name|url|sha256
FILES=(
    "melspectrogram.onnx|$OWW/melspectrogram.onnx|ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f"
    "embedding_model.onnx|$OWW/embedding_model.onnx|70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f"
    "silero_vad.onnx|$SILERO/silero_vad.onnx|2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f"
    "kokoro-v1.0.onnx|$KOKORO/kokoro-v1.0.onnx|7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5"
    "kokoro-v1.0.int8.onnx|$KOKORO/kokoro-v1.0.int8.onnx|6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb"
    "voices-v1.0.bin|$KOKORO/voices-v1.0.bin|bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"
    "kokoro-config.json|$KOKORO_HF/config.json|5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f"
    "ggml-base.en.bin|$WHISPER/ggml-base.en.bin|a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002"
)
# Qwen3-4B-Instruct-2507 Q4_K_M (Apache-2.0), unsloth's GGUF pinned by commit.
QWEN=https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/a06e946bb6b655725eafa393f4a9745d460374c9
if [ "$llm" = 1 ]; then
    FILES+=("Qwen3-4B-Instruct-2507-Q4_K_M.gguf|$QWEN/Qwen3-4B-Instruct-2507-Q4_K_M.gguf|3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597")
fi

sha_of() { sha256sum "$1" | cut -d' ' -f1; }

mkdir -p "$dest"
fail=0
for entry in "${FILES[@]}"; do
    IFS='|' read -r name url want <<<"$entry"
    out="$dest/$name"

    if [[ $print_hashes -eq 1 ]]; then
        if [[ ! -f "$out" ]]; then
            curl --proto '=https' --tlsv1.2 -fL --retry 3 --retry-delay 2 -sS -o "$out.part" "$url"
            mv "$out.part" "$out"
        fi
        echo "$name  $(sha_of "$out")"
        continue
    fi

    if [[ -f "$out" && "$(sha_of "$out")" == "$want" ]]; then
        echo "ok      $name (already present)"
        continue
    fi
    [[ -f "$out" ]] && echo "stale   $name (hash mismatch, re-downloading)"

    echo "fetch   $name"
    rm -f "$out.part"
    curl --proto '=https' --tlsv1.2 -fL --retry 3 --retry-delay 2 -sS -o "$out.part" "$url"
    got="$(sha_of "$out.part")"
    if [[ "$got" != "$want" ]]; then
        echo "ERROR   $name: sha256 mismatch" >&2
        echo "          want $want" >&2
        echo "          got  $got" >&2
        rm -f "$out.part"
        fail=1
        continue
    fi
    mv "$out.part" "$out"
    echo "ok      $name"
done

[[ $print_hashes -eq 1 ]] && exit 0
[[ $fail -eq 0 ]] || { echo "fetch-models: verification failed" >&2; exit 1; }

cat <<EOF

Models are in: $dest

Kokoro: two builds were fetched.
  kokoro-v1.0.onnx       fp32, 326 MB  <- recommended at runtime (best quality)
  kokoro-v1.0.int8.onnx  int8,  92 MB     smaller, ~15% faster on CPU, slightly lower quality;
                                           the fallback if RAM or load time matters more
Own wake-word head: models/telamon.onnx in the repo (not downloaded).
EOF
