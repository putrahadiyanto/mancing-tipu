"""Download discovered reels with yt-dlp, capped at 720p.

Reels are resolved fresh at download time — never from cached media URLs, which are
signature-scoped and expire within hours. Downloads run strictly sequentially;
parallelism is the fastest route to a checkpointed account.

Instagram has stricter rate limiting than Facebook, so sleep intervals are longer
(5-10s between downloads, 1-2s between HTTP requests within a download).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import yt_dlp

from .auth import SESSION_COOKIE
from .paths import IG_COOKIES_FILE, IGProfile, ensure_ig_dirs, reel_url
from .discover import load_known_shortcodes

# 720p ceiling with graceful fallback: best <=720 video + best audio, else a
# progressive <=720 stream, else whatever exists (some reels only ship one rendition).
FORMAT = "bv*[height<=720]+ba/b[height<=720]/b"


def _ydl_opts(profile: IGProfile) -> dict:
    return {
        "format": FORMAT,
        "merge_output_format": "mp4",
        "outtmpl": str(profile.data_dir / "%(id)s.%(ext)s"),
        "cookiefile": str(IG_COOKIES_FILE),
        "download_archive": str(profile.archive_file),
        # Longer pacing than FB — Instagram's rate limits are tighter.
        "sleep_interval": 5,
        "max_sleep_interval": 10,
        "sleep_interval_requests": 1,
        "retries": 5,
        "fragment_retries": 5,
        "ignoreerrors": True,
        "noprogress": False,
        "quiet": False,
        "no_warnings": False,
        "concurrent_fragment_downloads": 1,
    }


def _archive_ids(profile: IGProfile) -> set[str]:
    """Shortcodes recorded in the yt-dlp download archive."""
    if not profile.archive_file.exists():
        return set()
    ids = set()
    for line in profile.archive_file.read_text(errors="replace").split():
        if line.isalnum():
            ids.add(line)
    return ids


def _on_disk_ids(profile: IGProfile) -> set[str]:
    """Shortcodes already present as media files.

    InstagramReelIE passes the shortcode straight through as the video id, so the
    output filename stem is the shortcode.
    """
    return {
        p.stem
        for p in profile.data_dir.glob("*")
        if p.is_file() and not p.name.endswith(".part")
    }


def _record(profile: IGProfile, shortcode: str, status: str, detail: str = "") -> None:
    """Append a per-reel outcome to the manifest."""
    with profile.manifest_file.open("a") as fh:
        fh.write(
            json.dumps(
                {
                    "shortcode": shortcode,
                    "event": "download",
                    "status": status,
                    "detail": detail,
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            )
            + "\n"
        )


def download(profile: IGProfile, limit: int | None = None) -> dict:
    """Download every discovered-but-missing reel for a profile."""
    ensure_ig_dirs(profile)

    if not IG_COOKIES_FILE.exists():
        raise SystemExit(f"missing {IG_COOKIES_FILE} — run the `auth` step first")
    if SESSION_COOKIE not in IG_COOKIES_FILE.read_text(errors="replace"):
        raise SystemExit(
            f"{IG_COOKIES_FILE} has no `{SESSION_COOKIE}` cookie (not a logged-in session) "
            "— run the `auth` step first"
        )

    discovered = load_known_shortcodes(profile)
    if not discovered:
        return {"profile": profile.label, "pending": 0, "ok": 0, "failed": 0, "skipped": 0}

    # Pre-filter locally so already-downloaded reels cost zero network requests.
    have = _archive_ids(profile) | _on_disk_ids(profile)
    pending = sorted(discovered - have)
    skipped = len(discovered) - len(pending)

    if limit is not None:
        pending = pending[:limit]

    print(
        f"[ig-download] {profile.label}: {len(discovered)} discovered, "
        f"{skipped} already have, {len(pending)} to fetch"
    )

    ok = 0
    failed = 0
    with yt_dlp.YoutubeDL(_ydl_opts(profile)) as ydl:
        for i, sc in enumerate(pending, 1):
            print(f"[ig-download] ({i}/{len(pending)}) reel {sc}")
            try:
                info = ydl.extract_info(reel_url(sc), download=True)
            except Exception as exc:
                failed += 1
                _record(profile, sc, "error", f"{type(exc).__name__}: {exc}")
                print(f"[ig-download] reel {sc} failed: {exc}")
                continue

            if info is None:
                failed += 1
                _record(profile, sc, "error", "extraction returned no info")
            else:
                ok += 1
                _record(profile, sc, "ok")

    return {
        "profile": profile.label,
        "pending": len(pending),
        "ok": ok,
        "failed": failed,
        "skipped": skipped,
    }
