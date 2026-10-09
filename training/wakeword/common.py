"""Shared paths, voice roles and small helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("TELAMON_REPO", Path(__file__).resolve().parents[2]))
MODELS = Path(os.environ.get("TELAMON_MODELS", REPO / "out" / "models"))
WORK = Path(os.environ.get("TELAMON_WORK", REPO / "out" / "wakeword"))
HEAD_PATH = REPO / "models" / "telamon.onnx"
HEAD_META = REPO / "models" / "telamon.json"

SR = 16000
KOKORO_SR = 24000

# Voices that never take part in training. TEST is used only for the final
# numbers and the clips in tests/audio; VAL picks the threshold.
TEST_VOICES = ["af_nova", "am_michael", "bf_emma", "bm_lewis"]
VAL_VOICES = ["af_sky", "bm_george"]
CLIP_VOICE = "am_michael"  # voice used for tests/audio/*.wav


def english_voices(all_names: list[str]) -> list[str]:
    return sorted(n for n in all_names if n[:2] in ("af", "am", "bf", "bm"))


def roles(all_names: list[str]) -> dict[str, list[str]]:
    eng = english_voices(all_names)
    for v in TEST_VOICES + VAL_VOICES:
        if v not in eng:
            raise SystemExit(f"voice {v} missing from voices file")
    held = set(TEST_VOICES) | set(VAL_VOICES)
    return {
        "train": [v for v in eng if v not in held],
        "val": list(VAL_VOICES),
        "test": list(TEST_VOICES),
    }


def lang_for(voice: str) -> str:
    return "en-gb" if voice.startswith("b") else "en-us"


def to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x), -32768, 32767).astype(np.int16)


def load_threshold(default: float = 0.5) -> float:
    try:
        return float(json.loads(HEAD_META.read_text())["threshold"])
    except Exception:
        return default
