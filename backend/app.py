"""FastAPI backend: serves the trained CNN over HTTP for the frontend.

Run with:  uvicorn backend.app:app --reload --port 8000
"""

import os
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import GENRES
from src.inference import load_training_history, model_is_ready, predict_genre

app = FastAPI(title="AI Music Genre Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


@app.get("/api/health")
def health():
    return {"status": "ok", "model_ready": model_is_ready()}


@app.get("/api/model-info")
def model_info():
    history = load_training_history()
    return {
        "genres": GENRES,
        "model_ready": model_is_ready(),
        "architecture": [
            {"layer": "Input", "detail": "128 x 130 x 1 log-mel spectrogram"},
            {"layer": "Conv2D + BatchNorm + MaxPool", "detail": "32 filters, 3x3"},
            {"layer": "Conv2D + BatchNorm + MaxPool", "detail": "64 filters, 3x3"},
            {"layer": "Conv2D + BatchNorm + MaxPool", "detail": "128 filters, 3x3"},
            {"layer": "Conv2D + BatchNorm + MaxPool", "detail": "256 filters, 3x3"},
            {"layer": "GlobalAveragePooling2D", "detail": "spatial feature aggregation"},
            {"layer": "Dense (ReLU) + Dropout", "detail": "128 units, 0.4 dropout"},
            {"layer": "Dense (Softmax)", "detail": "10-way genre classification"},
        ],
        "history": history,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not model_is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run `python -m src.preprocess` then "
                   "`python -m src.train` after placing the GTZAN dataset in data/genres_original/.",
        )

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'.")

    audio_bytes = await file.read()
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 25 MB).")

    try:
        result = predict_genre(audio_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process audio: {exc}")

    return result
