#!/bin/bash
# The whole pipeline on recorded clips, with real models: wake word, VAD,
# whisper.cpp, llama-server with a read-only tool, and Kokoro. The spoken
# reply is transcribed back and checked. Run in the dev container:
#   DEV_GPU=1 DEV_LLM=<dir with the GGUF> scripts/dev.sh scripts/test-pipeline.sh
# Models: /work/models (scripts/fetch-models.sh /work/models, plus
# models/telamon.onnx). Output: /work/out/pipeline.
#   REALTIME=1   feed the clips at the speed of speech (the benchmark)
#   LLM_MODEL    the GGUF (default /llm/Qwen3-4B-Instruct-2507-Q4_K_M.gguf)
set -euo pipefail
cd /src

models=${MODELS:-/work/models}
out=/work/out/pipeline
llm=${LLM_MODEL:-/llm/Qwen3-4B-Instruct-2507-Q4_K_M.gguf}
port=8091
mkdir -p "$out"
cp -f models/telamon.onnx "$models/telamon.onnx"

cargo build --release -p voice-core --bin telamon-voice
bin=${CARGO_TARGET_DIR:-target}/release/telamon-voice

/usr/libexec/telamon-llama/llama-server -m "$llm" --jinja -ngl 99 -c 4096 \
    --host 127.0.0.1 --port $port >"$out/llama-server.log" 2>&1 &
llama=$!
trap 'kill $llama 2>/dev/null; wait $llama 2>/dev/null || true' EXIT
for _ in $(seq 240); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null && break
    kill -0 $llama 2>/dev/null || { tail -20 "$out/llama-server.log"; exit 1; }
    sleep 0.5
done
curl -sf "http://127.0.0.1:$port/health" >/dev/null || { echo "llama-server did not start" >&2; exit 1; }

rt=()
[ "${REALTIME:-0}" = 1 ] && rt=(--realtime)
fail=0

# check <clip> <tool expected or -> <word expected in the spoken reply or ->
check() {
    local clip=$1 tool=$2 word=$3 name
    name=$(basename "$clip" .wav)
    echo "== $name"
    "$bin" --models "$models" --wav "$clip" --out "$out/$name.reply.wav" \
        --llama "http://127.0.0.1:$port" --location London "${rt[@]}" | tee "$out/$name.jsonl"
    if [ "$tool" = - ]; then
        if grep -q '"event":"woke"' "$out/$name.jsonl"; then
            echo "FAIL: $name woke Telamon"; fail=1
        else
            echo "ok: $name did not wake Telamon"
        fi
        return
    fi
    grep -q '"event":"woke"' "$out/$name.jsonl" || { echo "FAIL: $name did not wake"; fail=1; return; }
    grep -q "\"event\":\"tool\",\"name\":\"$tool\"" "$out/$name.jsonl" \
        || { echo "FAIL: $name did not call $tool"; fail=1; }
    local said
    said=$("$bin" --models "$models" --transcribe "$out/$name.reply.wav" | tee -a "$out/$name.jsonl")
    echo "spoken reply, transcribed: $said"
    if [ "$word" != - ] && ! grep -qiE "$word" <<<"$said"; then
        echo "FAIL: the spoken reply has no /$word/"; fail=1
    fi
}

check tests/audio/hey-telamon-time.wav get_time '[0-9]|o.clock|morning|afternoon|evening|night'
check tests/audio/hey-telamon-cpu.wav get_system_stats 'percent|cpu|processor'
check tests/audio/no-wake.wav - -
echo "llama-server rss_mb: $(( $(ps -o rss= -p $llama) / 1024 ))"
[ "$fail" = 0 ] && echo "PASS" || { echo "FAILED"; exit 1; }
