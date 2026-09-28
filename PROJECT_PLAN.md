# Project Plan — AI Music Genre Classifier

**Course context:** Deep Learning Lab
**Author:** flowfrontierweb@gmail.com
**Status date:** 2026-09-17

---

## 1. Objective

Build a small but complete deep learning system that classifies the genre of a
music clip, and present it through a polished web interface so the underlying
concepts (spectrograms, CNNs, convolution, pooling, softmax) are visible and
explainable — not just a black-box accuracy number.

```
Audio  →  Mel-Spectrogram  →  CNN  →  Genre (softmax over 10 classes)
```

## 2. Problem statement

Genre is a perceptual, high-level property of audio that isn't linearly
separable from raw waveform samples. The classic approach — and the one this
project follows — is to first re-represent audio as a **time-frequency
image** (a spectrogram), which turns the problem into image classification, a
setting where CNNs are well understood and effective.

## 3. Dataset

**GTZAN Genre Collection**
- 1,000 tracks, 30 seconds each, 22.05 kHz mono WAV
- 10 genres × 100 tracks: blues, classical, country, disco, hiphop, jazz,
  metal, pop, reggae, rock
- Source used: the public `marsyas/gtzan` dataset on Hugging Face
  (no account/login required), ~1.23 GB
- Known issue handled: `jazz.00054.wav` is corrupted in the original release;
  preprocessing skips unreadable files instead of crashing

## 4. Pipeline / methodology

| Stage | What happens | Where |
|---|---|---|
| 1. Segment | Each 30s track is split into 10 × 3s segments | `src/preprocess.py` |
| 2. Spectrogram | Each segment → log-mel spectrogram (128 mel bins) via `librosa` | `src/preprocess.py` |
| 3. Dataset split | Stratified 70/15/15 train/val/test | `src/train.py` |
| 4. Normalize | Z-score using **training-set** mean/std only (no leakage) | `src/train.py` |
| 5. Train | CNN trained with early stopping on validation accuracy | `src/train.py`, `src/model.py` |
| 6. Evaluate | Test accuracy/loss, confusion matrix, classification report | `src/train.py` |
| 7. Serve | FastAPI loads the trained model and predicts on uploaded audio | `backend/app.py`, `src/inference.py` |
| 8. Present | Web UI shows waveform, spectrogram, prediction confidence, training curves, and concept explanations | `frontend/` |

Splitting each track into segments both **augments the dataset** (~1,000 →
~9,981 usable examples after removing the corrupt file) and lets the trained
model classify clips of arbitrary length at inference time, not just exact
30-second tracks.

## 5. Model architecture

A 4-block CNN, ~424K parameters:

```
Input (128 mel bins × 130 time frames × 1)
 → [Conv2D(32) → BatchNorm → MaxPool → Dropout]
 → [Conv2D(64) → BatchNorm → MaxPool → Dropout]
 → [Conv2D(128) → BatchNorm → MaxPool → Dropout]
 → [Conv2D(256) → BatchNorm → MaxPool → Dropout]
 → GlobalAveragePooling2D
 → Dense(128, ReLU) → Dropout
 → Dense(10, Softmax)
```

Design choices and why:
- **Increasing filter depth (32→256)** — early layers learn generic
  time-frequency textures, deeper layers combine them into more abstract,
  genre-relevant patterns.
- **BatchNorm + Dropout at every block** — GTZAN is small (~1,000 base
  tracks), so overfitting is the main risk; both are countermeasures.
- **GlobalAveragePooling2D instead of Flatten+Dense** — keeps parameter count
  low and adds a form of spatial invariance.
- **Softmax output** — a proper probability distribution over the 10 genres,
  which the UI displays as full confidence bars, not just the top guess.

## 6. Deep learning concepts this project demonstrates

- Spectrogram generation (mel scale, dB/log intensity scale)
- Treating non-image data (audio) as an image for CNN input
- Convolution and receptive fields
- Pooling (downsampling + translation tolerance)
- Batch normalization and dropout as regularization
- Feature extraction / hierarchical representation learning
- Global average pooling
- Softmax classification
- Train/validation/test methodology and data-leakage avoidance
- Early stopping
- Confusion matrix & per-class precision/recall/F1 evaluation

## 7. Tech stack

| Layer | Choice |
|---|---|
| Audio processing | `librosa` |
| Model | TensorFlow / Keras (CNN) |
| Backend API | FastAPI + Uvicorn |
| Frontend | HTML / CSS / vanilla JS + Chart.js |
| Dataset | GTZAN (via Hugging Face) |

## 8. Project structure

```
fdl_project/
├── data/genres_original/   # GTZAN wav files, one folder per genre
├── models/                 # generated: trained model, meta.json, history.json, confusion_matrix.png
├── src/
│   ├── config.py           # shared constants
│   ├── preprocess.py       # audio -> spectrogram dataset builder
│   ├── model.py             # CNN architecture
│   ├── train.py             # training + evaluation
│   ├── inference.py         # shared prediction pipeline
│   └── predict.py           # CLI predictor
├── backend/app.py           # FastAPI server (API + serves the frontend)
├── frontend/                # index.html / style.css / script.js
├── requirements.txt
└── README.md                 # setup & run instructions
```

## 9. UI/UX plan

- **Pipeline explainer** — a 4-step visual (Audio → Spectrogram → CNN →
  Genre) so the mechanism is visible before any interaction.
- **Upload** — drag-and-drop, with a live client-side waveform preview.
- **Prediction** — top genre badge + full confidence bars for all 10 genres
  (not just the winner), so the model's uncertainty is visible.
- **Spectrogram view** — shows the literal image the CNN classified.
- **Model insights** — tabs for accuracy/loss curves (from real training
  history), architecture summary, and an accordion explaining each concept
  in plain language.
- Dark, glassmorphism-style theme; fully responsive.

## 10. Milestones & current status

| Milestone | Status |
|---|---|
| Project scaffold (src/backend/frontend) | ✅ Done |
| Preprocessing pipeline | ✅ Done — verified on real data (9,981 examples generated) |
| CNN architecture | ✅ Done |
| Training script (splits, normalization, early stopping, evaluation) | ✅ Done |
| FastAPI backend (`/api/predict`, `/api/model-info`, `/api/health`) | ✅ Done, tested live on localhost |
| Frontend UI | ✅ Done, verified in-browser (layout, tabs, accordion, upload) |
| Python 3.11 + TensorFlow environment | ✅ Installed |
| GTZAN dataset downloaded & cleaned | ✅ Done (AppleDouble metadata files removed) |
| Model training run | ✅ Done — 34 epochs (early stopping), best val accuracy 91.0% |
| Training history / confusion matrix generated | ✅ Done (`models/history.json`, `models/confusion_matrix.png`) |
| End-to-end demo (real prediction through the UI) | ✅ Done — verified via API and browser (e.g. `rock.00000.wav` → rock at 90.7%) |

## 11. Evaluation results

Final CNN, trained 34 epochs (early stopping on validation accuracy, patience 10):

| Metric | Value |
|---|---|
| Test accuracy | **91.0%** |
| Test loss | 0.348 |
| Best validation accuracy | 91.0% (epoch 24) |
| Final training accuracy | 97.2% |

Per-genre F1 score (test set):

| Genre | Precision | Recall | F1 |
|---|---|---|---|
| metal | 0.94 | 0.98 | 0.96 |
| pop | 0.90 | 0.99 | 0.94 |
| hiphop | 0.91 | 0.96 | 0.94 |
| jazz | 0.92 | 0.95 | 0.94 |
| blues | 0.88 | 0.95 | 0.92 |
| classical | 0.86 | 0.99 | 0.92 |
| disco | 0.91 | 0.91 | 0.91 |
| reggae | 0.97 | 0.84 | 0.90 |
| country | 0.96 | 0.73 | 0.83 |
| rock | 0.87 | 0.79 | 0.83 |

The model is strongest on genres with distinctive timbre (metal, classical,
hiphop) and weakest distinguishing **rock vs. country**, which is the
classically hard pair for this dataset — both draw on overlapping
instrumentation (guitar, drums, similar tempos), so confusion between the two
is expected and visible in `models/confusion_matrix.png`.

Training accuracy (97.2%) is noticeably higher than validation/test (~91%),
indicating mild overfitting — expected given GTZAN's small size (~1,000 base
tracks). Dropout and batch normalization already limit this; more
aggressive data augmentation (see §12) would be the next lever to pull.

## 12. Possible extensions (stretch goals, not required)

- Data augmentation on the spectrogram (time/frequency masking — SpecAugment)
- Compare against a simpler baseline (e.g. MFCC features + SVM/random forest)
- Grad-CAM visualization to show which time-frequency regions drove a prediction
- Model export to TensorFlow Lite for on-device inference
