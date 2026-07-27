"""Harvest reel IDs from a profile's reels tab.

yt-dlp has no extractor for a profile reels listing (only facebook.com/reel/<id>),
so enumeration has to happen here. We collect only reel IDs — never resolved media
URLs, which are signature-scoped and expire within hours.
"""

from __future__ import annotations

import json
import random
import re
import time
from datetime import datetime, timezone

from .paths import Profile, ensure_dirs

# Regex over the raw response body rather than walking the GraphQL schema: Facebook
# reshapes its response types without notice, but the /reel/<id> URL shape is stable.
REEL_URL_RE = re.compile(r'"/reel/(\d+)')
VIDEO_ID_RE = re.compile(r'"video_id":"(\d+)"')

SCROLL_PAUSE = (1.5, 4.0)
DEFAULT_STALL_ROUNDS = 5
DEFAULT_MAX_SCROLLS = 400


def reel_url(reel_id: str) -> str:
    return f"https://www.facebook.com/reel/{reel_id}"


def load_known_ids(profile: Profile) -> set[str]:
    """Read already-discovered reel IDs so re-runs are additive."""
    if not profile.manifest_file.exists():
        return set()

    known: set[str] = set()
    for line in profile.manifest_file.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            known.add(str(json.loads(line)["reel_id"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return known


def _append_manifest(profile: Profile, reel_ids: list[str]) -> None:
    """Append incrementally so a crash mid-crawl keeps what was already found."""
    stamp = datetime.now(timezone.utc).isoformat()
    with profile.manifest_file.open("a") as fh:
        for rid in reel_ids:
            fh.write(
                json.dumps(
                    {
                        "reel_id": rid,
                        "url": reel_url(rid),
                        "profile_id": profile.id,
                        "discovered_at": stamp,
                    }
                )
                + "\n"
            )


def _dom_reel_ids(page) -> set[str]:
    """Secondary harvester: reel links currently in the DOM.

    Needed because interception can miss the server-rendered first page, but
    insufficient alone — the reels grid is virtualized and recycles earlier tiles out
    of the DOM as you scroll.
    """
    try:
        hrefs = page.eval_on_selector_all(
            'a[href*="/reel/"]', "els => els.map(e => e.getAttribute('href'))"
        )
    except Exception:
        return set()

    found = set()
    for href in hrefs or []:
        m = re.search(r"/reel/(\d+)", href or "")
        if m:
            found.add(m.group(1))
    return found


def discover(
    context,
    profile: Profile,
    max_scrolls: int = DEFAULT_MAX_SCROLLS,
    stall_rounds: int = DEFAULT_STALL_ROUNDS,
) -> dict:
    """Scroll the reels tab until discovery goes dry; return a run summary."""
    ensure_dirs(profile)

    known = load_known_ids(profile)
    seen: set[str] = set(known)
    intercepted: set[str] = set()

    def on_response(response):
        # Primary harvester: reel IDs streamed in GraphQL pagination responses.
        if "/api/graphql" not in response.url:
            return
        try:
            body = response.text()
        except Exception:
            return  # body already consumed or connection dropped — nothing to salvage
        intercepted.update(REEL_URL_RE.findall(body))
        intercepted.update(VIDEO_ID_RE.findall(body))

    page = context.pages[0] if context.pages else context.new_page()
    page.on("response", on_response)

    print(f"[discover] {profile.label}: opening {profile.url}")
    page.goto(profile.url, wait_until="domcontentloaded")
    page.wait_for_timeout(4000)

    if "/login" in page.url:
        raise RuntimeError("redirected to login — session is dead, re-run the auth step")

    new_ids: list[str] = []
    stalls = 0
    scrolls = 0

    while scrolls < max_scrolls and stalls < stall_rounds:
        batch = (intercepted | _dom_reel_ids(page)) - seen
        if batch:
            ordered = sorted(batch)
            seen.update(ordered)
            new_ids.extend(ordered)
            _append_manifest(profile, ordered)
            stalls = 0
            print(f"[discover] +{len(ordered)} (total {len(seen)})")
        else:
            stalls += 1

        page.keyboard.press("End")
        page.mouse.wheel(0, 2500)
        time.sleep(random.uniform(*SCROLL_PAUSE))
        scrolls += 1

        if "/login" in page.url:
            raise RuntimeError("redirected to login mid-crawl — session lost")

    # A final sweep: the last scroll's responses land after the loop's last check.
    batch = (intercepted | _dom_reel_ids(page)) - seen
    if batch:
        ordered = sorted(batch)
        seen.update(ordered)
        new_ids.extend(ordered)
        _append_manifest(profile, ordered)

    exhausted = stalls >= stall_rounds
    if not exhausted:
        print(
            f"[discover] WARNING: stopped at the --max-scrolls backstop ({max_scrolls}). "
            "Coverage is likely truncated — raise it and re-run."
        )

    return {
        "profile": profile.label,
        "already_known": len(known),
        "new": len(new_ids),
        "total": len(seen),
        "scrolls": scrolls,
        "exhausted": exhausted,
    }
