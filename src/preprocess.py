"""Turn the raw GTZAN wav files into fixed-size log-mel-spectrograms.

Expected layout (standard GTZAN "genres_original" release):
    data/genres_original/<genre>/<genre>.NNNNN.wav

Each 30s track is split into SEGMENTS_PER_TRACK short segments before the
mel-spectrogram is computed. This does two things at once: it turns ~1000
tracks into ~10x that many training examples, and it teaches the CNN to
recognize a genre from a short clip -- which is what the demo needs, since
users will upload arbitrary-length audio, not exactly 30s tracks.
"""

import json
import os
import sys

import librosa
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DATA_DIR, GENRES, SAMPLE_RATE, SEGMENT_SAMPLES, SEGMENTS_PER_TRACK,
    N_MELS, N_FFT, HOP_LENGTH, FRAMES_PER_SEGMENT, FEATURES_PATH, MODELS_DIR,
)


def audio_to_melspectrogram(signal: np.ndarray) -> np.ndarray:
    """Convert a 1D audio segment into a (N_MELS, FRAMES_PER_SEGMENT, 1) log-mel image."""
    mel = librosa.feature.melspectrogram(
        y=signal, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Pad/crop so every example has an identical shape regardless of rounding.
    if log_mel.shape[1] < FRAMES_PER_SEGMENT:
        pad_width = FRAMES_PER_SEGMENT - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=log_mel.min())
    else:
        log_mel = log_mel[:, :FRAMES_PER_SEGMENT]

    return log_mel[..., np.newaxis].astype(np.float32)


def build_dataset():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"Expected GTZAN data at {DATA_DIR}. Download the dataset and place the "
            f"'genres_original' folder (one subfolder per genre of .wav files) there. "
            f"See README.md for the download link."
        )

    features, labels = [], []

    for genre_idx, genre in enumerate(GENRES):
        genre_dir = os.path.join(DATA_DIR, genre)
        if not os.path.isdir(genre_dir):
            print(f"[warn] missing genre folder: {genre_dir} -- skipping")
            continue

        files = [f for f in os.listdir(genre_dir) if f.lower().endswith(".wav")]
        for filename in tqdm(files, desc=f"{genre:10s}"):
            path = os.path.join(genre_dir, filename)
            try:
                signal, _ = librosa.load(path, sr=SAMPLE_RATE)
            except Exception as exc:
                print(f"[warn] could not read {path}: {exc}")
                continue

            for s in range(SEGMENTS_PER_TRACK):
                start = SEGMENT_SAMPLES * s
                end = start + SEGMENT_SAMPLES
                segment = signal[start:end]
                if len(segment) < SEGMENT_SAMPLES:
                    continue
                features.append(audio_to_melspectrogram(segment))
                labels.append(genre_idx)

    X = np.stack(features)
    y = np.array(labels, dtype=np.int64)

    os.makedirs(MODELS_DIR, exist_ok=True)
    np.savez_compressed(FEATURES_PATH, X=X, y=y)
    print(f"Saved {X.shape[0]} examples with shape {X.shape[1:]} to {FEATURES_PATH}")


if __name__ == "__main__":
    build_dataset()
