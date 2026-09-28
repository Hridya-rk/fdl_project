"""Shared constants for preprocessing, training and inference.

Keeping these in one place matters here specifically: the backend must
slice/normalize incoming audio in *exactly* the same way train.py did,
or the CNN sees out-of-distribution inputs and predictions degrade silently.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "genres_original")
MODELS_DIR = os.path.join(BASE_DIR, "models")

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

SAMPLE_RATE = 22050
TRACK_DURATION = 30          # seconds, matches GTZAN clip length
SEGMENTS_PER_TRACK = 10      # split each clip into shorter segments (data augmentation)
SEGMENT_SAMPLES = (SAMPLE_RATE * TRACK_DURATION) // SEGMENTS_PER_TRACK

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
# Frames per segment for a fixed-size CNN input, derived from segment length.
FRAMES_PER_SEGMENT = 1 + SEGMENT_SAMPLES // HOP_LENGTH

MODEL_PATH = os.path.join(MODELS_DIR, "genre_cnn.keras")
META_PATH = os.path.join(MODELS_DIR, "meta.json")
HISTORY_PATH = os.path.join(MODELS_DIR, "history.json")
CONFUSION_MATRIX_PATH = os.path.join(MODELS_DIR, "confusion_matrix.png")
FEATURES_PATH = os.path.join(MODELS_DIR, "features.npz")
