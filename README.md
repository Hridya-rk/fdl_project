# 🎵 AI Music Genre Classifier

A deep learning lab project that classifies the genre of a music clip using a
**CNN trained on mel-spectrograms**, served through a FastAPI backend with a
polished web UI.

```
Audio  →  Mel-Spectrogram  →  CNN  →  Genre (softmax over 10 classes)
```

## Project structure

```
fdl_project/
├── data/genres_original/   # put the GTZAN dataset here (one folder per genre)
├── models/                 # trained model, meta.json, training history (generated)
├── src/
│   ├── config.py           # shared constants (sample rate, mel params, genres...)
│   ├── preprocess.py       # wav files -> log-mel-spectrogram numpy arrays
│   ├── model.py            # CNN architecture (Keras)
│   ├── train.py            # trains + evaluates + saves the model
│   ├── inference.py        # shared inference pipeline (used by CLI + API)
│   └── predict.py          # CLI: predict a single file
├── backend/
│   └── app.py              # FastAPI server (serves API + the frontend)
├── frontend/
│   ├── index.html / style.css / script.js   # the UI
└── requirements.txt
```

## 1. Setup

> **Python version:** TensorFlow currently supports Python 3.9–3.12. This
> machine only has Python 3.14 installed, which is too new for TensorFlow —
> install Python 3.11 (https://www.python.org/downloads/) alongside it and
> use it to create the venv, e.g. `py -3.11 -m venv venv`.

```bash
py -3.11 -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## 2. Get the dataset

Download **GTZAN Genre Collection** (10 genres × 100 tracks × 30s, ~1.2 GB),
e.g. from Kaggle:
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

Extract it so you end up with:

```
data/genres_original/blues/blues.00000.wav
data/genres_original/classical/classical.00000.wav
... (10 genre folders total)
```

## 3. Preprocess → train

```bash
python -m src.preprocess   # builds models/features.npz (log-mel spectrograms)
python -m src.train        # trains the CNN, saves model + history to models/
```

Training produces:
- `models/genre_cnn.keras` — the trained model
- `models/meta.json` — genre labels + normalization stats (needed for inference)
- `models/history.json` — per-epoch accuracy/loss, test metrics, confusion matrix (powers the UI's charts)
- `models/confusion_matrix.png`

On a laptop CPU, expect ~20-40 min depending on epochs/early stopping.

## 4. Run the app

```bash
uvicorn backend.app:app --reload --port 8000
```

Open **http://localhost:8000/app** — upload an audio clip and see:
- the waveform and the actual mel-spectrogram the CNN sees
- predicted genre with a full confidence breakdown across all 10 genres
- training/validation accuracy & loss curves, model architecture, and short
  write-ups of the core concepts (convolution, pooling, spectrograms, softmax...)

You can also predict from the command line without the web UI:

```bash
python -m src.predict path/to/song.wav
```

## Concepts this project demonstrates

- **Spectrogram generation** — converting audio into a mel-scaled, dB-scaled
  time-frequency image (`librosa.feature.melspectrogram`).
- **CNNs for non-image data** — treating the spectrogram as an image lets a
  standard conv/pool architecture learn genre-discriminative patterns.
- **Convolution & pooling** — 4 conv blocks (32→64→128→256 filters) with
  batch normalization, max-pooling and dropout.
- **Feature extraction** — Global Average Pooling condenses learned feature
  maps instead of a huge Flatten+Dense, reducing overfitting on a small dataset.
- **Softmax classification** — a 10-way probability distribution over genres.
- **Train/val/test methodology** — stratified splits, normalization fit only
  on training data, early stopping on validation accuracy, and a confusion
  matrix / classification report for evaluation.

## Notes

- The model expects short segments (3s by default, see `SEGMENTS_PER_TRACK`
  in `src/config.py`) — this both multiplies the training data 10x and lets
  the demo classify audio clips of any length, not just exact 30s tracks.
- If you upload a track before training a model, the API returns a clear
  503 error explaining that `src/preprocess.py` + `src/train.py` need to run
  first — there's no fake/random prediction fallback, since the whole point
  of the lab is demonstrating a real trained model.
