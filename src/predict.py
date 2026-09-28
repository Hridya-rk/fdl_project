"""CLI: python -m src.predict path/to/song.wav"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import predict_genre


def main():
    parser = argparse.ArgumentParser(description="Predict the genre of an audio file.")
    parser.add_argument("audio_path", help="Path to a .wav/.mp3 file")
    args = parser.parse_args()

    with open(args.audio_path, "rb") as f:
        audio_bytes = f.read()

    result = predict_genre(audio_bytes)
    print(f"\nTop genre: {result['top_genre']}  ({result['duration']}s analyzed)\n")
    for p in result["predictions"]:
        bar = "█" * int(p["confidence"] / 2)
        print(f"  {p['genre']:10s} {p['confidence']:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
