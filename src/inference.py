"""Shared inference logic used by both the CLI (predict.py) and the FastAPI backend.

Kept separate from app.py so the backend stays a thin HTTP layer and this
module can be unit-tested / reused without spinning up a server.
"""

import base64
import io
import json
import os
import sys

import librosa
import librosa.display
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    SAMPLE_RATE, SEGMENT_SAMPLES, MODEL_PATH, META_PATH, HISTORY_PATH,
)
from src.preprocess import audio_to_melspectrogram

_model = None
_meta = None


def _load():
    global _model, _meta
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run preprocessing + "
                f"training first (see README.md), or the backend runs in demo mode."
            )
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH)
        with open(META_PATH) as f:
            _meta = json.load(f)
    return _model, _meta


def model_is_ready() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)


def load_training_history():
    if not os.path.exists(HISTORY_PATH):
        return None
    with open(HISTORY_PATH) as f:
        return json.load(f)


def _spectrogram_png_base64(signal: np.ndarray) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mel = librosa.feature.melspectrogram(y=signal, sr=SAMPLE_RATE, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor="none")
    ax.set_facecolor("none")
    img = librosa.display.specshow(
        log_mel, sr=SAMPLE_RATE, x_axis="time", y_axis="mel", ax=ax, cmap="magma",
    )
    ax.set_xlabel("Time (s)", color="#cbd5e1")
    ax.set_ylabel("Mel frequency", color="#cbd5e1")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _waveform_peaks(signal: np.ndarray, num_points: int = 200) -> list:
    chunk = max(1, len(signal) // num_points)
    peaks = [
        float(np.abs(signal[i:i + chunk]).max())
        for i in range(0, len(signal) - chunk + 1, chunk)
    ]
    max_peak = max(peaks) if peaks else 1.0
    return [p / max_peak if max_peak > 0 else 0.0 for p in peaks]


def predict_genre(audio_bytes: bytes):
    """Run the full pipeline on raw audio bytes and return a JSON-friendly dict."""
    model, meta = _load()
    genres = meta["genres"]
    mean, std = meta["mean"], meta["std"]

    signal, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
    duration = len(signal) / SAMPLE_RATE

    if len(signal) < SEGMENT_SAMPLES:
        signal = np.pad(signal, (0, SEGMENT_SAMPLES - len(signal)))

    num_segments = max(1, len(signal) // SEGMENT_SAMPLES)
    segment_probs = []
    for i in range(num_segments):
        segment = signal[i * SEGMENT_SAMPLES:(i + 1) * SEGMENT_SAMPLES]
        if len(segment) < SEGMENT_SAMPLES:
            continue
        feat = audio_to_melspectrogram(segment)
        feat = (feat - mean) / std
        probs = model.predict(feat[np.newaxis, ...], verbose=0)[0]
        segment_probs.append(probs)

    avg_probs = np.mean(segment_probs, axis=0)
    order = np.argsort(avg_probs)[::-1]

    predictions = [
        {"genre": genres[i], "confidence": round(float(avg_probs[i]) * 100, 2)}
        for i in order
    ]

    preview = signal[: SEGMENT_SAMPLES * min(num_segments, 3)]

    return {
        "predictions": predictions,
        "top_genre": predictions[0]["genre"],
        "duration": round(duration, 2),
        "segments_analyzed": len(segment_probs),
        "spectrogram_image": _spectrogram_png_base64(preview),
        "waveform_peaks": _waveform_peaks(signal),
    }
