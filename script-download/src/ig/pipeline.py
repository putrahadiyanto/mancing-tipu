"""CLI orchestrator for the Instagram reels pipeline.

    python -m src.ig.pipeline auth [--headless]
    python -m src.ig.pipeline discover [--profile USERNAME] [--max-scrolls N] [--video-limit N] [--headless]
    python -m src.ig.pipeline download [--profile USERNAME] [--limit N]
    python -m src.ig.pipeline select [--profile USERNAME] [--limit N]
    python -m src.ig.pipeline run [--profile USERNAME] [--limit N] [--video-limit N] [--headless]
"""

from __future__ import annotations

import argparse
import sys

from . import auth, discover as discover_mod, download as download_mod, select as select_mod
from .paths import load_ig_profiles


def _print_summary(title: str, rows: list[dict]) -> None:
    print(f"\n=== {title} ===")
    for row in rows:
        parts = [f"{k}={v}" for k, v in row.items() if k != "profile"]
        print(f"  {row['profile']}: " + "  ".join(parts))
    print()


def cmd_auth(args) -> int:
    auth.refresh_session(headed=not args.headless)
    return 0


def cmd_discover(args) -> int:
    profiles = load_ig_profiles(args.profile)
    rows = []
    with auth.browser_context(headed=not args.headless) as context:
        auth.ensure_session(context)
        auth.export_cookies(context)
        for profile in profiles:
            rows.append(
                discover_mod.discover(
                    context,
                    profile,
                    max_scrolls=args.max_scrolls,
                    stall_rounds=args.stall_rounds,
                    video_limit=args.video_limit,
                )
            )
    _print_summary("discovery", rows)
    return 0


def cmd_download(args) -> int:
    profiles = load_ig_profiles(args.profile)
    rows = [download_mod.download(p, limit=args.limit) for p in profiles]
    _print_summary("downloads", rows)
    return 1 if any(r["failed"] for r in rows) else 0


def cmd_select(args) -> int:
    profiles = load_ig_profiles(args.profile)
    rows = [select_mod.select_and_download(p, limit=args.limit) for p in profiles]
    _print_summary("selection + downloads", rows)
    return 1 if any(r.get("download_failed", 0) for r in rows) else 0


def cmd_run(args) -> int:
    profiles = load_ig_profiles(args.profile)

    discovered = []
    with auth.browser_context(headed=not args.headless) as context:
        auth.ensure_session(context)
        auth.export_cookies(context)
        for profile in profiles:
            discovered.append(
                discover_mod.discover(
                    context,
                    profile,
                    max_scrolls=args.max_scrolls,
                    stall_rounds=args.stall_rounds,
                    video_limit=args.video_limit,
                )
            )
    _print_summary("discovery", discovered)

    # Browser is closed before downloading — yt-dlp works off the exported cookies.
    fetched = [download_mod.download(p, limit=args.limit) for p in profiles]
    _print_summary("downloads", fetched)

    return 1 if any(r["failed"] for r in fetched) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ig-pipeline", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_browser_flags(p):
        p.add_argument(
            "--headless",
            action="store_true",
            help="run Chromium headless (more detectable; prefer headed for early runs)",
        )

    def add_crawl_flags(p):
        p.add_argument("--profile", help="restrict to one profile username or label")
        p.add_argument(
            "--max-scrolls",
            type=int,
            default=discover_mod.DEFAULT_MAX_SCROLLS,
            help="runaway backstop only; discovery normally stops when it goes dry",
        )
        p.add_argument(
            "--stall-rounds",
            type=int,
            default=discover_mod.DEFAULT_STALL_ROUNDS,
            help="consecutive empty scroll rounds before declaring discovery done",
        )
        p.add_argument(
            "--video-limit",
            type=int,
            default=None,
            help="stop discovery after finding this many new shortcodes (e.g. 15)",
        )

    p_auth = sub.add_parser("auth", help="establish/verify session, export cookies")
    add_browser_flags(p_auth)
    p_auth.set_defaults(func=cmd_auth)

    p_disc = sub.add_parser("discover", help="harvest reel shortcodes from reels tabs")
    add_browser_flags(p_disc)
    add_crawl_flags(p_disc)
    p_disc.set_defaults(func=cmd_discover)

    p_dl = sub.add_parser("download", help="download discovered reels")
    p_dl.add_argument("--profile", help="restrict to one profile username or label")
    p_dl.add_argument("--limit", type=int, help="max reels to fetch this run")
    p_dl.set_defaults(func=cmd_download)

    p_sel = sub.add_parser(
        "select",
        help="interactive selection: preview thumbnails, choose which reels to download",
    )
    p_sel.add_argument("--profile", help="restrict to one profile username or label")
    p_sel.add_argument("--limit", type=int, help="max reels to download after selection")
    p_sel.set_defaults(func=cmd_select)

    p_run = sub.add_parser("run", help="auth -> discover -> download")
    add_browser_flags(p_run)
    add_crawl_flags(p_run)
    p_run.add_argument("--limit", type=int, help="max reels to fetch this run")
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except auth.NotLoggedIn as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[abort] interrupted — progress is saved, re-run to resume", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
