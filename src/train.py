"""Train the CNN on preprocessed GTZAN features and save everything the
backend needs for inference: the model, normalization stats, class labels,
and training history (for the UI's accuracy/loss charts).

Run src/preprocess.py first to generate models/features.npz.
"""

import json
import os
import sys

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    GENRES, FEATURES_PATH, MODELS_DIR, MODEL_PATH, META_PATH,
    HISTORY_PATH, CONFUSION_MATRIX_PATH,
)
from src.model import build_cnn


def main(epochs=60, batch_size=32):
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found. Run `python -m src.preprocess` first."
        )

    data = np.load(FEATURES_PATH)
    X, y = data["X"], data["y"]

    # 70% train / 15% val / 15% test, stratified so every genre is represented in each split.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42,
    )

    # Normalize using training-set statistics only, to avoid leaking test info.
    mean, std = X_train.mean(), X_train.std()
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    model = build_cnn(input_shape=X_train.shape[1:], num_classes=len(GENRES))
    model.summary()

    os.makedirs(MODELS_DIR, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    report = classification_report(y_test, y_pred, target_names=GENRES, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(GENRES)), GENRES, rotation=45, ha="right")
        ax.set_yticks(range(len(GENRES)), GENRES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        for i in range(len(GENRES)):
            for j in range(len(GENRES)):
                ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=8)
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    except Exception as exc:
        print(f"[warn] could not save confusion matrix plot: {exc}")

    with open(HISTORY_PATH, "w") as f:
        json.dump({
            "accuracy": history.history["accuracy"],
            "val_accuracy": history.history["val_accuracy"],
            "loss": history.history["loss"],
            "val_loss": history.history["val_loss"],
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "classification_report": report,
            "confusion_matrix": cm,
        }, f, indent=2)

    with open(META_PATH, "w") as f:
        json.dump({
            "genres": GENRES,
            "input_shape": list(X_train.shape[1:]),
            "mean": float(mean),
            "std": float(std),
        }, f, indent=2)

    model.save(MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metadata to {META_PATH}")
    print(f"Saved training history to {HISTORY_PATH}")


if __name__ == "__main__":
    main()
