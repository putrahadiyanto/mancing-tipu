"""Harvest reel shortcodes from a profile's reels tab.

yt-dlp has no extractor for an Instagram profile reels listing (only
instagram.com/reel/<shortcode>), so enumeration has to happen here. We collect
only shortcodes — never resolved media URLs, which are signature-scoped and
expire within hours.

Instagram's GraphQL responses are intercepted via page.on("response") and
shortcodes are extracted with regex rather than walking the schema: Instagram
rotates doc_id values every 2-4 weeks as an anti-scraping measure, but the
shortcode format ([A-Za-z0-9_-]{11}) is stable.
"""

from __future__ import annotations

import json
import random
import re
import time
from datetime import datetime, timezone

from .paths import IGProfile, ensure_ig_dirs

# Shortcodes are alphanumeric + hyphens/underscores, typically 11 characters.
# This regex is applied to the raw JSON body of intercepted GraphQL responses.
SHORTCODE_RE = re.compile(r'"shortcode":"([A-Za-z0-9_-]{5,15})"')

# Broader fallback: reel links in the DOM use /reel/<shortcode>/.
DOM_REEL_RE = re.compile(r"/reel/([A-Za-z0-9_-]{5,15})")

SCROLL_PAUSE = (2.0, 5.0)  # longer than FB — Instagram is stricter
DEFAULT_STALL_ROUNDS = 5
DEFAULT_MAX_SCROLLS = 400


def load_known_shortcodes(profile: IGProfile) -> set[str]:
    """Read already-discovered shortcodes so re-runs are additive."""
    if not profile.manifest_file.exists():
        return set()

    known: set[str] = set()
    for line in profile.manifest_file.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            if entry.get("event") == "download":
                continue  # skip download records
            known.add(str(entry["shortcode"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return known


def _append_manifest(profile: IGProfile, shortcodes: list[str]) -> None:
    """Append incrementally so a crash mid-crawl keeps what was already found."""
    stamp = datetime.now(timezone.utc).isoformat()
    with profile.manifest_file.open("a") as fh:
        for sc in shortcodes:
            fh.write(
                json.dumps(
                    {
                        "shortcode": sc,
                        "url": f"https://www.instagram.com/reel/{sc}/",
                        "username": profile.username,
                        "discovered_at": stamp,
                    }
                )
                + "\n"
            )


def _dom_reel_shortcodes(page) -> set[str]:
    """Secondary harvester: reel shortcodes from links currently in the DOM.

    The reels grid is virtualized and recycles tiles out of view as you scroll,
    so DOM scraping alone is insufficient — but it catches server-rendered
    content missed by interception.
    """
    try:
        hrefs = page.eval_on_selector_all(
            'a[href*="/reel/"]',
            "els => els.map(e => e.getAttribute('href'))",
        )
    except Exception:
        return set()

    found = set()
    for href in hrefs or []:
        m = DOM_REEL_RE.search(href or "")
        if m:
            found.add(m.group(1))
    return found


def discover(
    context,
    profile: IGProfile,
    max_scrolls: int = DEFAULT_MAX_SCROLLS,
    stall_rounds: int = DEFAULT_STALL_ROUNDS,
    video_limit: int | None = None,
) -> dict:
    """Scroll the reels tab until discovery goes dry; return a run summary.

    If video_limit is set, stop after discovering that many new shortcodes.
    """
    ensure_ig_dirs(profile)

    known = load_known_shortcodes(profile)
    seen: set[str] = set(known)
    intercepted: set[str] = set()

    def on_response(response):
        # Primary harvester: shortcodes streamed in GraphQL pagination responses.
        url = response.url
        if "/graphql/query" not in url and "/api/v1/" not in url:
            return
        try:
            body = response.text()
        except Exception:
            return
        intercepted.update(SHORTCODE_RE.findall(body))

    page = context.pages[0] if context.pages else context.new_page()
    page.on("response", on_response)

    print(f"[ig-discover] {profile.label}: opening {profile.url}")
    page.goto(profile.url, wait_until="domcontentloaded")
    page.wait_for_timeout(4000)

    if "/login" in page.url or "/accounts/login" in page.url:
        raise RuntimeError(
            "redirected to login — session is dead, re-run the auth step"
        )

    new_shortcodes: list[str] = []
    stalls = 0
    scrolls = 0

    while scrolls < max_scrolls and stalls < stall_rounds:
        batch = (intercepted | _dom_reel_shortcodes(page)) - seen
        if batch:
            ordered = sorted(batch)
            seen.update(ordered)
            new_shortcodes.extend(ordered)
            _append_manifest(profile, ordered)
            stalls = 0
            print(f"[ig-discover] +{len(ordered)} (total {len(seen)})")
        else:
            stalls += 1

        # Stop early if we've hit the video limit.
        if video_limit is not None and len(new_shortcodes) >= video_limit:
            print(f"[ig-discover] reached video limit of {video_limit}")
            break

        page.keyboard.press("End")
        page.mouse.wheel(0, 2500)
        time.sleep(random.uniform(*SCROLL_PAUSE))
        scrolls += 1

        if "/login" in page.url or "/accounts/login" in page.url:
            raise RuntimeError("redirected to login mid-crawl — session lost")

    # A final sweep: the last scroll's responses land after the loop's last check.
    batch = (intercepted | _dom_reel_shortcodes(page)) - seen
    if batch:
        ordered = sorted(batch)
        seen.update(ordered)
        new_shortcodes.extend(ordered)
        _append_manifest(profile, ordered)

    exhausted = stalls >= stall_rounds
    hit_limit = video_limit is not None and len(new_shortcodes) >= video_limit
    if not exhausted and not hit_limit:
        print(
            f"[ig-discover] WARNING: stopped at the --max-scrolls backstop ({max_scrolls}). "
            "Coverage is likely truncated — raise it and re-run."
        )

    return {
        "profile": profile.label,
        "already_known": len(known),
        "new": len(new_shortcodes),
        "total": len(seen),
        "scrolls": scrolls,
        "exhausted": exhausted,
    }
