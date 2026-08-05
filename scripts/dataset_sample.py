"""Randomly sample downloaded IG reels into dataset/test/ or dataset/train/.

Records which files were used so train doesn't leak from test.
"""

import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "script-download" / "data" / "ig"
DATASET_DIR = ROOT / "dataset"
TRACKING_FILE = DATASET_DIR / "copied.jsonl"
SAMPLE_SIZE = 100


def load_copied() -> set[str]:
    """Load shortcodes already used in test/train splits."""
    if not TRACKING_FILE.exists():
        return set()
    copied = set()
    for line in TRACKING_FILE.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            copied.add(json.loads(line)["shortcode"])
        except (json.JSONDecodeError, KeyError):
            continue
    return copied


def record_copied(shortcode: str, split: str) -> None:
    """Append a record to the tracking file."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with TRACKING_FILE.open("a") as fh:
        fh.write(json.dumps({"shortcode": shortcode, "split": split}) + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=["test", "train"],
        default="test",
        help="which split to sample into (default: test)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=SAMPLE_SIZE,
        help=f"number of videos to sample; 0 = all remaining (default: {SAMPLE_SIZE})",
    )
    args = parser.parse_args()

    out_dir = DATASET_DIR / args.split
    copied = load_copied()

    videos = list(DATA_DIR.rglob("*.mp4"))
    if not videos:
        print("no .mp4 files found in data/ig/")
        return

    # Exclude already-used files
    available = [v for v in videos if v.stem not in copied]
    if not available:
        print("all videos already used in other splits")
        return

    n = len(available) if args.size == 0 else min(args.size, len(available))
    sampled = random.sample(available, n)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(sampled, 1):
        dst = out_dir / f"{i}.mp4"
        shutil.copy2(src, dst)
        record_copied(src.stem, args.split)
        print(f"[{i}/{len(sampled)}] {src.parent.name}/{src.stem} -> {args.split}/{i}.mp4")

    print(f"\n{len(sampled)} videos copied to {out_dir}")
    print(f"tracking file: {TRACKING_FILE}")


if __name__ == "__main__":
    main()
