"""Shared filesystem layout and profile config loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

CONFIG_FILE = ROOT / "config" / "profiles.yml"
STATE_DIR = ROOT / "state"
DATA_DIR = ROOT / "data"

BROWSER_PROFILE_DIR = STATE_DIR / "browser_profile"
COOKIES_FILE = STATE_DIR / "cookies.txt"
COOKIES_IMPORT_FILE = STATE_DIR / "cookies_import.txt"


@dataclass(frozen=True)
class Profile:
    id: str
    label: str
    url: str

    @property
    def state_dir(self) -> Path:
        return STATE_DIR / self.id

    @property
    def data_dir(self) -> Path:
        return DATA_DIR / self.id

    @property
    def manifest_file(self) -> Path:
        return self.state_dir / "manifest.jsonl"

    @property
    def archive_file(self) -> Path:
        return self.state_dir / "archive.txt"


def load_profiles(only: str | None = None) -> list[Profile]:
    """Load enabled profiles from config, optionally narrowed to a single id or label."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"missing config: {CONFIG_FILE}")

    raw = yaml.safe_load(CONFIG_FILE.read_text()) or {}
    entries = raw.get("profiles") or []

    profiles: list[Profile] = []
    for entry in entries:
        pid = str(entry["id"])
        if not entry.get("enabled", True):
            continue
        profiles.append(
            Profile(
                id=pid,
                label=entry.get("label") or pid,
                # Numeric-id form: the /people/<slug>/ slug is cosmetic and can change.
                url=entry.get("url")
                or f"https://www.facebook.com/profile.php?id={pid}&sk=reels_tab",
            )
        )

    if only:
        profiles = [p for p in profiles if p.id == only or p.label == only]
        if not profiles:
            raise SystemExit(f"no enabled profile matching {only!r} in {CONFIG_FILE}")

    if not profiles:
        raise SystemExit(f"no enabled profiles in {CONFIG_FILE}")

    return profiles


def ensure_dirs(profile: Profile) -> None:
    profile.state_dir.mkdir(parents=True, exist_ok=True)
    profile.data_dir.mkdir(parents=True, exist_ok=True)
