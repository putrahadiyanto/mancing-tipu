"""Instagram-specific filesystem layout and profile config loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent

IG_CONFIG_FILE = ROOT / "config" / "ig-profiles.yml"
IG_STATE_DIR = ROOT / "state" / "ig"
IG_DATA_DIR = ROOT / "data" / "ig"

IG_BROWSER_PROFILE_DIR = IG_STATE_DIR / "browser_profile"
IG_COOKIES_FILE = IG_STATE_DIR / "cookies.txt"
IG_COOKIES_IMPORT_FILE = IG_STATE_DIR / "cookies_import.txt"


@dataclass(frozen=True)
class IGProfile:
    username: str
    label: str
    url: str

    @property
    def state_dir(self) -> Path:
        return IG_STATE_DIR / self.username

    @property
    def data_dir(self) -> Path:
        return IG_DATA_DIR / self.username

    @property
    def manifest_file(self) -> Path:
        return self.state_dir / "manifest.jsonl"

    @property
    def archive_file(self) -> Path:
        return self.state_dir / "archive.txt"


def reel_url(shortcode: str) -> str:
    """Build a direct reel URL from a shortcode."""
    return f"https://www.instagram.com/reel/{shortcode}/"


def load_ig_profiles(only: str | None = None) -> list[IGProfile]:
    """Load enabled profiles from ig-profiles.yml, optionally narrowed to one."""
    if not IG_CONFIG_FILE.exists():
        raise FileNotFoundError(f"missing config: {IG_CONFIG_FILE}")

    raw = yaml.safe_load(IG_CONFIG_FILE.read_text()) or {}
    entries = raw.get("profiles") or []

    profiles: list[IGProfile] = []
    for entry in entries:
        username = str(entry["username"])
        if not entry.get("enabled", True):
            continue
        profiles.append(
            IGProfile(
                username=username,
                label=entry.get("label") or username,
                url=entry.get("url")
                or f"https://www.instagram.com/{username}/reels/",
            )
        )

    if only:
        profiles = [
            p for p in profiles if p.username == only or p.label == only
        ]
        if not profiles:
            raise SystemExit(
                f"no enabled profile matching {only!r} in {IG_CONFIG_FILE}"
            )

    if not profiles:
        raise SystemExit(f"no enabled profiles in {IG_CONFIG_FILE}")

    return profiles


def ensure_ig_dirs(profile: IGProfile) -> None:
    profile.state_dir.mkdir(parents=True, exist_ok=True)
    profile.data_dir.mkdir(parents=True, exist_ok=True)
