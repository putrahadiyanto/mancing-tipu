"""Instagram session management.

Single source of session truth for both pipeline stages. The Playwright persistent
context holds the logged-in session; `export_cookies` mirrors it into a Netscape
cookies.txt so yt-dlp downloads under the same identity.

Uses playwright-stealth to reduce bot detection on Instagram (Meta's anti-bot
system is significantly more aggressive than for Facebook).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth

from .paths import (
    IG_BROWSER_PROFILE_DIR,
    IG_COOKIES_FILE,
    IG_COOKIES_IMPORT_FILE,
    IG_STATE_DIR,
)

# `sessionid` is the canonical Instagram logged-in indicator.
SESSION_COOKIE = "sessionid"

LOGIN_POLL_INTERVAL = 5
LOGIN_TIMEOUT = 300

# Netscape format treats expires=0 as a session cookie; some consumers drop those.
# Rewrite them with a far-future stamp so yt-dlp keeps them.
FAR_FUTURE = 2147483647


class NotLoggedIn(RuntimeError):
    """No usable Instagram session, and none could be established."""


_STEALTH = Stealth()


def _apply_stealth(context) -> None:
    """Apply playwright-stealth patches to all pages in the context.

    Patches navigator.webdriver, plugins, chrome.runtime, and other
    fingerprint vectors that Instagram's anti-bot system checks.
    """
    _STEALTH.apply_stealth_sync(context)


@contextmanager
def browser_context(headed: bool = True):
    """Yield a persistent browser context rooted at the IG-specific profile dir.

    Uses Playwright's bundled Chromium with its *default* user agent — a spoofed UA
    that contradicts the rest of the fingerprint is worse than no spoofing at all.
    """
    IG_BROWSER_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        context = pw.chromium.launch_persistent_context(
            user_data_dir=str(IG_BROWSER_PROFILE_DIR),
            headless=not headed,
            viewport={"width": 1440, "height": 900},
            locale="en-US",
            args=["--disable-blink-features=AutomationControlled"],
        )
        _apply_stealth(context)
        try:
            yield context
        finally:
            context.close()


def is_logged_in(context) -> bool:
    return any(
        c["name"] == SESSION_COOKIE and c["value"]
        for c in context.cookies("https://www.instagram.com")
    )


def _parse_netscape(path: Path) -> list[dict]:
    """Parse a Netscape cookies.txt into Playwright cookie dicts."""
    cookies: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("#HttpOnly_"):
            line = line[len("#HttpOnly_") :]
        elif not line or line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) != 7:
            continue
        domain, _include_sub, cpath, secure, expires, name, value = parts

        try:
            expiry = int(float(expires))
        except ValueError:
            expiry = FAR_FUTURE

        cookies.append(
            {
                "name": name,
                "value": value,
                "domain": domain,
                "path": cpath or "/",
                "expires": expiry if expiry > 0 else FAR_FUTURE,
                "secure": secure.upper() == "TRUE",
            }
        )
    return cookies


def _seed_from_import(context) -> bool:
    """Try to establish a session from a user-supplied cookies_import.txt."""
    if not IG_COOKIES_IMPORT_FILE.exists():
        return False

    cookies = _parse_netscape(IG_COOKIES_IMPORT_FILE)
    if not cookies:
        print(f"[ig-auth] {IG_COOKIES_IMPORT_FILE.name} contained no parsable cookies")
        return False

    print(f"[ig-auth] importing {len(cookies)} cookies from {IG_COOKIES_IMPORT_FILE.name}")
    context.add_cookies(cookies)

    page = context.pages[0] if context.pages else context.new_page()
    page.goto("https://www.instagram.com/", wait_until="domcontentloaded")
    return is_logged_in(context)


def _interactive_login(context) -> bool:
    """Hold the browser open and poll until the user logs in by hand."""
    page = context.pages[0] if context.pages else context.new_page()
    page.goto("https://www.instagram.com/accounts/login/", wait_until="domcontentloaded")

    print(
        "\n[ig-auth] Not logged in. Log into Instagram in the browser window.\n"
        f"[ig-auth] Waiting up to {LOGIN_TIMEOUT}s for the session to appear...\n"
    )

    deadline = time.monotonic() + LOGIN_TIMEOUT
    while time.monotonic() < deadline:
        if is_logged_in(context):
            print("[ig-auth] session detected")
            return True
        time.sleep(LOGIN_POLL_INTERVAL)

    return False


def ensure_session(context) -> None:
    """Guarantee a logged-in context, or raise NotLoggedIn.

    Order: existing persistent profile -> cookies_import.txt -> interactive login.
    """
    page = context.pages[0] if context.pages else context.new_page()
    page.goto("https://www.instagram.com/", wait_until="domcontentloaded")

    if is_logged_in(context):
        print("[ig-auth] using existing session from browser profile")
        return

    if _seed_from_import(context):
        print("[ig-auth] session established from imported cookies")
        return

    if _interactive_login(context):
        return

    raise NotLoggedIn(
        "could not establish an Instagram session. Either log in interactively "
        f"(run with --headed) or drop a Netscape cookies file at {IG_COOKIES_IMPORT_FILE}"
    )


def export_cookies(context, dest: Path = IG_COOKIES_FILE) -> Path:
    """Write the context's Instagram cookies as Netscape cookies.txt for yt-dlp."""
    IG_STATE_DIR.mkdir(parents=True, exist_ok=True)

    cookies = [
        c
        for c in context.cookies()
        if "instagram.com" in c.get("domain", "")
        or "cdninstagram.com" in c.get("domain", "")
    ]

    lines = ["# Netscape HTTP Cookie File", "# Generated by script-download/src/ig/auth.py"]
    for c in cookies:
        expires = int(c.get("expires") or 0)
        if expires <= 0:
            expires = FAR_FUTURE  # keep session cookies alive for yt-dlp

        domain = c["domain"]
        include_sub = "TRUE" if domain.startswith(".") else "FALSE"
        secure = "TRUE" if c.get("secure") else "FALSE"
        lines.append(
            "\t".join(
                [domain, include_sub, c.get("path", "/"), secure, str(expires), c["name"], c["value"]]
            )
        )

    dest.write_text("\n".join(lines) + "\n")
    dest.chmod(0o600)
    print(f"[ig-auth] exported {len(cookies)} cookies -> {dest}")
    return dest


def refresh_session(headed: bool = True) -> Path:
    """Standalone auth step: verify/establish a session and export cookies."""
    with browser_context(headed=headed) as context:
        ensure_session(context)
        return export_cookies(context)
