"""Transcribe videos from dataset/test/ or dataset/train/ using Groq Whisper V3 Turbo.

Extracts audio with ffmpeg, sends to Groq's Whisper API, outputs CSV with id + transcription.

Usage:
    python scripts/dataset_sample.py                    # default: dataset/test/
    python scripts/transcribe.py --split train          # dataset/train/
    python scripts/transcribe.py --split test --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from groq import Groq

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
WHISPER_MODEL = "whisper-large-v3-turbo"


def extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extract mono 16kHz WAV from video using ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(audio_path),
        ],
        capture_output=True,
        timeout=60,
    )


def transcribe_file(client: Groq, audio_path: Path) -> str:
    """Send audio to Groq Whisper and return the transcription text."""
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=f,
            language="id",
        )
    return result.text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--output", type=str, default=None, help="output CSV path")
    parser.add_argument("--limit", type=int, default=None, help="max files to process")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("error: set GROQ_API_KEY environment variable", file=sys.stderr)
        return 1

    client = Groq(api_key=api_key)

    split_dir = DATASET_DIR / args.split
    if not split_dir.exists():
        print(f"error: {split_dir} does not exist", file=sys.stderr)
        return 1

    videos = sorted(split_dir.glob("*.mp4"), key=lambda p: int(p.stem))
    if args.limit:
        videos = videos[:args.limit]

    if not videos:
        print(f"no .mp4 files in {split_dir}")
        return 0

    output_path = Path(args.output) if args.output else DATASET_DIR / f"{args.split}_transcriptions.csv"

    print(f"[transcribe] {len(videos)} videos from {split_dir}")
    print(f"[transcribe] output: {output_path}\n")

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "transcribe"])

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, video in enumerate(videos, 1):
                audio_path = Path(tmpdir) / "audio.wav"

                print(f"[{i}/{len(videos)}] {video.name} -> extracting audio...", end=" ", flush=True)
                extract_audio(video, audio_path)

                if not audio_path.exists():
                    print("FAILED (ffmpeg)")
                    writer.writerow([video.stem, ""])
                    continue

                print("transcribing...", end=" ", flush=True)
                try:
                    text = transcribe_file(client, audio_path)
                    print(f"OK ({len(text)} chars)")
                except Exception as exc:
                    print(f"FAILED ({exc})")
                    text = ""

                writer.writerow([video.stem, text])

    print(f"\nDone. {len(videos)} transcriptions written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
