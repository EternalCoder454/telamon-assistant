"""Write tests/audio/*.wav with a held-out Kokoro voice.

16 kHz mono s16, ~0.7 s of leading and ~1.5 s of trailing near-silence, light
pink noise throughout (about -52 dBFS RMS), speech peaking around -8 dBFS.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import soundfile as sf

from .common import CLIP_VOICE, MODELS, REPO, SR, lang_for, to_int16
from .compose import pink
from .gen_tts import synth_one

CLIPS = {
    "hey-telamon-time.wav": ["Hey Telamon, what time is it?"],
    "hey-telamon-cpu.wav": ["Hey Telamon, how busy is my CPU right now?"],
    "hey-telamon-pause-weather.wav": ["Hey Telamon", 0.6, "what's the weather like?"],
    "no-wake.wav": ["Tell a man to call the telephone company about the salmon."],
}


def main() -> None:
    from kokoro_onnx import Kokoro

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(MODELS / "kokoro-v1.0.onnx"), so, providers=["CPUExecutionProvider"])
    k = Kokoro.from_session(sess, str(MODELS / "voices-v1.0.bin"))
    out = REPO / "tests" / "audio"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2024)
    for name, parts in CLIPS.items():
        segs: list[np.ndarray] = [np.zeros(int(0.7 * SR))]
        for p in parts:
            if isinstance(p, float):
                segs.append(np.zeros(int(p * SR)))
            else:
                segs.append(synth_one(k, CLIP_VOICE, {"text": p, "ph": None, "speed": 1.0}).astype(np.float64))
        segs.append(np.zeros(int(1.5 * SR)))
        x = np.concatenate(segs)
        peak = np.abs(x).max()
        x = x * (32767 * 10 ** (-8 / 20) / peak)
        nz = pink(len(x), rng)
        x = x + nz * 32767 * 10 ** (-52 / 20)
        y = to_int16(x)
        sf.write(out / name, y, SR, subtype="PCM_16")
        print(f"{name}: {len(y) / SR:.2f} s  {parts}")


if __name__ == "__main__":
    main()
