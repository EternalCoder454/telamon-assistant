#!/bin/bash
# A headless run of the built app, inside the dev container:
#   scripts/dev.sh scripts/smoke.sh [binary]   (default /work/build/dev/telamon-assistant)
# Xvfb and a private session bus, with every XDG dir under out/smoke. It
# never touches the user's display, microphone or files.
#  1. Telamon off: the window, and no glow.
#  2. Telamon on, hearing a recorded clip ("Hey Telamon … what's the
#     weather like?") instead of the microphone: screenshots until the gold
#     edge glow shows, saved as out/smoke/glow.png.
# Models: $MODELS (default /work/models, plus models/telamon.onnx). No
# llama-server is needed: the glow shows at the wake word.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bin=$(realpath "${1:-/work/build/dev/telamon-assistant}")
out=$repo/out/smoke
models=${MODELS:-/work/models}
clip=${CLIP:-$repo/tests/audio/hey-telamon-pause-weather.wav}
rm -rf "$out"
mkdir -p "$out"/{config,data,cache,runtime,home,state,frames}
chmod 700 "$out/runtime"
cp -f "$repo/models/telamon.onnx" "$models/telamon.onnx"
export HOME=$out/home XDG_CONFIG_HOME=$out/config XDG_DATA_HOME=$out/data \
    XDG_CACHE_HOME=$out/cache XDG_RUNTIME_DIR=$out/runtime XDG_STATE_HOME=$out/state
export QT_QPA_PLATFORM=xcb QT_SCALE_FACTOR=${QT_SCALE_FACTOR:-1.5}
export TELAMON_ASSISTANT_MODELS=$models

shot() {
    import -window root "$out/$1.png"
    echo "saved $out/$1.png"
}

# Gold at the left edge, halfway down: red over green over blue.
gold() {
    local p r g b
    p=$(convert "$1" -format '%[fx:int(255*p{2,540}.r)] %[fx:int(255*p{2,540}.g)] %[fx:int(255*p{2,540}.b)]' info:)
    read -r r g b <<<"$p"
    [ "$r" -gt 90 ] && [ "$r" -gt "$g" ] && [ "$g" -gt $((b + 20)) ]
}

run() {
    # 1. Off: a window, no glow, no microphone.
    "$bin" >"$out/app-off.log" 2>&1 &
    local app=$!
    sleep 4
    shot 1-off
    if gold "$out/1-off.png"; then
        echo "FAIL: the glow shows while Telamon is off" >&2
        kill $app
        exit 1
    fi
    kill $app
    wait $app 2>/dev/null || true

    # 2. On, hearing the clip.
    printf '[Assistant]\nEnabled=true\nServerUrl=http://127.0.0.1:9\nLocation=London\n' \
        >"$XDG_CONFIG_HOME/assistantrc"
    TELAMON_ASSISTANT_TEST_WAV=$clip "$bin" >"$out/app-on.log" 2>&1 &
    app=$!
    local found=""
    for i in $(seq -w 1 80); do
        sleep 0.25
        import -window root "$out/frames/$i.png"
        if gold "$out/frames/$i.png"; then
            found=$i
            sleep 0.5
            import -window root "$out/frames/$i-later.png"
            break
        fi
    done
    sleep 1
    shot 3-after
    kill $app
    wait $app 2>/dev/null || true
    if [ -z "$found" ]; then
        echo "FAIL: no glow in 20 s; the app's log:" >&2
        tail -20 "$out/app-on.log" >&2
        exit 1
    fi
    cp "$out/frames/$found.png" "$out/glow.png"
    cp "$out/frames/$found-later.png" "$out/2-glow.png"
    echo "PASS: the glow shows $((10#$found * 250)) ms after start ($out/glow.png)"
}

export -f run shot gold
export bin out clip
xvfb-run -a -s "-screen 0 1920x1080x24" dbus-run-session -- bash -c run
