"""Download discovered reels with yt-dlp, capped at 720p.

Reels are resolved fresh at download time — never from cached media URLs, which are
signature-scoped and expire within hours. Downloads run strictly sequentially;
parallelism is the fastest route to a checkpointed account.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import yt_dlp

from .auth import SESSION_COOKIE
from .paths import COOKIES_FILE, Profile, ensure_dirs
from .discover import load_known_ids, reel_url

# 720p ceiling with graceful fallback: best <=720 video + best audio, else a
# progressive <=720 stream, else whatever exists (some reels only ship one rendition).
FORMAT = "bv*[height<=720]+ba/b[height<=720]/b"


def _ydl_opts(profile: Profile) -> dict:
    return {
        "format": FORMAT,
        "merge_output_format": "mp4",
        "outtmpl": str(profile.data_dir / "%(id)s.%(ext)s"),
        "cookiefile": str(COOKIES_FILE),
        "download_archive": str(profile.archive_file),
        # Randomized human-like pacing between items.
        "sleep_interval": 3,
        "max_sleep_interval": 8,
        "sleep_interval_requests": 1,
        "retries": 5,
        "fragment_retries": 5,
        "ignoreerrors": True,
        "noprogress": False,
        "quiet": False,
        "no_warnings": False,
        "concurrent_fragment_downloads": 1,
    }


def _archive_ids(profile: Profile) -> set[str]:
    """Reel IDs recorded in the yt-dlp download archive."""
    if not profile.archive_file.exists():
        return set()
    ids = set()
    for line in profile.archive_file.read_text(errors="replace").split():
        if line.isdigit():
            ids.add(line)
    return ids


def _on_disk_ids(profile: Profile) -> set[str]:
    """Reel IDs already present as media files.

    FacebookReelIE passes the reel id straight through as the video id, so the
    output filename stem is the reel id.
    """
    return {
        p.stem
        for p in profile.data_dir.glob("*")
        if p.is_file() and not p.name.endswith(".part")
    }


def _record(profile: Profile, reel_id: str, status: str, detail: str = "") -> None:
    """Append a per-reel outcome to the manifest."""
    with profile.manifest_file.open("a") as fh:
        fh.write(
            json.dumps(
                {
                    "reel_id": reel_id,
                    "event": "download",
                    "status": status,
                    "detail": detail,
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            )
            + "\n"
        )


def download(profile: Profile, limit: int | None = None) -> dict:
    """Download every discovered-but-missing reel for a profile."""
    ensure_dirs(profile)

    # An empty/sessionless cookies.txt passes an exists() check but then fails every
    # single reel with an opaque extractor error, so check for the session cookie.
    if not COOKIES_FILE.exists():
        raise SystemExit(f"missing {COOKIES_FILE} — run the `auth` step first")
    if SESSION_COOKIE not in COOKIES_FILE.read_text(errors="replace"):
        raise SystemExit(
            f"{COOKIES_FILE} has no `{SESSION_COOKIE}` cookie (not a logged-in session) "
            "— run the `auth` step first"
        )

    discovered = load_known_ids(profile)
    if not discovered:
        return {"profile": profile.label, "pending": 0, "ok": 0, "failed": 0, "skipped": 0}

    # Pre-filter locally so already-downloaded reels cost zero network requests.
    # yt-dlp's own archive check happens only after it has already extracted the page.
    have = _archive_ids(profile) | _on_disk_ids(profile)
    pending = sorted(discovered - have)
    skipped = len(discovered) - len(pending)

    if limit is not None:
        pending = pending[:limit]

    print(
        f"[download] {profile.label}: {len(discovered)} discovered, "
        f"{skipped} already have, {len(pending)} to fetch"
    )

    ok = 0
    failed = 0
    with yt_dlp.YoutubeDL(_ydl_opts(profile)) as ydl:
        for i, rid in enumerate(pending, 1):
            print(f"[download] ({i}/{len(pending)}) reel {rid}")
            try:
                # ignoreerrors=True makes extract_info return None instead of raising
                # on a dead/blocked reel, so check the result explicitly.
                info = ydl.extract_info(reel_url(rid), download=True)
            except Exception as exc:  # network death, auth loss, unexpected extractor error
                failed += 1
                _record(profile, rid, "error", f"{type(exc).__name__}: {exc}")
                print(f"[download] reel {rid} failed: {exc}")
                continue

            if info is None:
                failed += 1
                _record(profile, rid, "error", "extraction returned no info")
            else:
                ok += 1
                _record(profile, rid, "ok")

    return {
        "profile": profile.label,
        "pending": len(pending),
        "ok": ok,
        "failed": failed,
        "skipped": skipped,
    }
