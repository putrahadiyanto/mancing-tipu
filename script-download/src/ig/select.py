"""Interactive reel selection UI.

Shows thumbnails of discovered-but-undownloaded reels in a tkinter window,
lets the user preview each one (opens in browser) and decide Download or Skip.
Selections are persisted so skipped reels aren't shown again.

Usage:
    python -m src.ig.pipeline select --profile <username>
"""

from __future__ import annotations

import json
import subprocess
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageTk

from .paths import (
    IG_COOKIES_FILE,
    IGProfile,
    ensure_ig_dirs,
    reel_url,
)
from .discover import load_known_shortcodes

# Try to import tkinter
try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError:
    tk = None  # will raise at runtime if GUI is used

THUMB_WIDTH = 480
THUMB_HEIGHT = 480


def _load_skipped(profile: IGProfile) -> set[str]:
    """Load shortcodes the user already skipped."""
    if not profile.selections_file.exists():
        return set()
    skipped: set[str] = set()
    for line in profile.selections_file.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            if entry.get("action") == "skip":
                skipped.add(entry["shortcode"])
        except (json.JSONDecodeError, KeyError):
            continue
    return skipped


def _load_downloaded_ids(profile: IGProfile) -> set[str]:
    """Shortcodes already on disk or in the archive."""
    archive_ids: set[str] = set()
    if profile.archive_file.exists():
        for line in profile.archive_file.read_text(errors="replace").split():
            if line.isalnum():
                archive_ids.add(line)

    disk_ids = {
        p.stem
        for p in profile.data_dir.glob("*")
        if p.is_file() and not p.name.endswith(".part")
    }
    return archive_ids | disk_ids


def _record_selection(profile: IGProfile, shortcode: str, action: str) -> None:
    """Append a selection record."""
    with profile.selections_file.open("a") as fh:
        fh.write(
            json.dumps(
                {
                    "shortcode": shortcode,
                    "action": action,
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            )
            + "\n"
        )


def _fetch_thumbnail(profile: IGProfile, shortcode: str) -> Path | None:
    """Extract a thumbnail frame using yt-dlp. Returns path to jpg or None."""
    thumb_dir = profile.thumbnails_dir
    thumb_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(thumb_dir / "thumb")

    result = subprocess.run(
        [
            "yt-dlp",
            "--skip-download",
            "--write-thumbnail",
            "--convert-thumbnails", "jpg",
            "--cookies", str(IG_COOKIES_FILE),
            "-o", out_template,
            reel_url(shortcode),
        ],
        capture_output=True,
        timeout=30,
    )

    thumb_path = Path(out_template + ".jpg")
    if thumb_path.exists():
        return thumb_path

    # Fallback: check for .webp
    webp_path = Path(out_template + ".webp")
    if webp_path.exists():
        return webp_path

    return None


class ReelSelector:
    """Tkinter GUI for interactive reel selection."""

    def __init__(self, profile: IGProfile):
        self.profile = profile
        self.pending: list[str] = []
        self.current_idx = 0
        self.selected: list[str] = []
        self.skipped: list[str] = []
        self._photo: ImageTk.PhotoImage | None = None

    def _build_queue(self) -> None:
        """Build the list of shortcodes needing user decision."""
        known = load_known_shortcodes(self.profile)
        downloaded = _load_downloaded_ids(self.profile)
        skipped = _load_skipped(self.profile)

        # Pending = discovered - already downloaded - already skipped
        self.pending = sorted(known - downloaded - skipped)

    def run(self) -> dict:
        """Run the selection UI. Returns a summary dict."""
        ensure_ig_dirs(self.profile)
        self._build_queue()

        if not self.pending:
            print(f"[ig-select] {self.profile.label}: nothing to review")
            return {"profile": self.profile.label, "reviewed": 0, "selected": 0, "skipped": 0}

        print(f"[ig-select] {self.profile.label}: {len(self.pending)} reels to review")

        root = tk.Tk()
        root.title(f"IG Reel Selector — {self.profile.label}")
        root.resizable(False, False)

        # --- layout ---
        self._img_label = tk.Label(root)
        self._img_label.pack(padx=10, pady=(10, 5))

        self._info_var = tk.StringVar()
        info_label = tk.Label(root, textvariable=self._info_var, justify=tk.LEFT, font=("monospace", 10))
        info_label.pack(padx=10, anchor=tk.W)

        btn_frame = ttk.Frame(root)
        btn_frame.pack(padx=10, pady=10, fill=tk.X)

        self._preview_btn = ttk.Button(btn_frame, text="Open in Browser", command=self._open_browser)
        self._preview_btn.pack(side=tk.LEFT, padx=(0, 10))

        self._skip_btn = ttk.Button(btn_frame, text="Skip", command=self._skip)
        self._skip_btn.pack(side=tk.LEFT, padx=(0, 10))

        self._download_btn = ttk.Button(btn_frame, text="Download", command=self._download)
        self._download_btn.pack(side=tk.LEFT)

        self._progress_var = tk.StringVar()
        prog_label = tk.Label(root, textvariable=self._progress_var, font=("monospace", 9))
        prog_label.pack(padx=10, pady=(0, 10))

        self._root = root
        self._show_current()

        # keyboard shortcuts
        root.bind("<d>", lambda e: self._download())
        root.bind("<s>", lambda e: self._skip())
        root.bind("<o>", lambda e: self._open_browser())
        root.bind("<Left>", lambda e: self._skip())
        root.bind("<Right>", lambda e: self._download())
        root.bind("<Escape>", lambda e: self._quit())

        root.protocol("WM_DELETE_WINDOW", self._quit)
        root.mainloop()

        return {
            "profile": self.profile.label,
            "reviewed": len(self.selected) + len(self.skipped),
            "selected": len(self.selected),
            "skipped": len(self.skipped),
        }

    def _show_current(self) -> None:
        """Display the current reel's thumbnail and info."""
        if self.current_idx >= len(self.pending):
            self._finish()
            return

        sc = self.pending[self.current_idx]
        remaining = len(self.pending) - self.current_idx

        self._info_var.set(f"Shortcode: {sc}\nProfile: {self.profile.username}")
        self._progress_var.set(f"{remaining} remaining  |  {len(self.selected)} selected  |  {len(self.skipped)} skipped")

        # Try to load thumbnail
        self._img_label.config(image="", text="Loading thumbnail...")
        self._root.update_idletasks()

        thumb_path = _fetch_thumbnail(self.profile, sc)
        if thumb_path and thumb_path.exists():
            try:
                img = Image.open(thumb_path)
                img.thumbnail((THUMB_WIDTH, THUMB_HEIGHT), Image.LANCZOS)
                self._photo = ImageTk.PhotoImage(img)
                self._img_label.config(image=self._photo, text="")
            except Exception:
                self._img_label.config(image="", text="[thumbnail load failed]")
        else:
            self._img_label.config(image="", text="[no thumbnail available]")

    def _open_browser(self) -> None:
        """Open the reel URL in the default browser for full preview."""
        if self.current_idx >= len(self.pending):
            return
        sc = self.pending[self.current_idx]
        webbrowser.open(reel_url(sc))

    def _download(self) -> None:
        """Mark current reel as selected for download."""
        if self.current_idx >= len(self.pending):
            return
        sc = self.pending[self.current_idx]
        self.selected.append(sc)
        _record_selection(self.profile, sc, "download")
        print(f"[ig-select] + {sc}")
        self.current_idx += 1
        self._show_current()

    def _skip(self) -> None:
        """Skip current reel."""
        if self.current_idx >= len(self.pending):
            return
        sc = self.pending[self.current_idx]
        self.skipped.append(sc)
        _record_selection(self.profile, sc, "skip")
        print(f"[ig-select] - {sc}")
        self.current_idx += 1
        self._show_current()

    def _quit(self) -> None:
        """Close the window early."""
        self._root.destroy()

    def _finish(self) -> None:
        """All reels reviewed — close the window."""
        messagebox.showinfo(
            "Done",
            f"Review complete!\n\nSelected: {len(self.selected)}\nSkipped: {len(self.skipped)}",
        )
        self._root.destroy()


def select_and_download(profile: IGProfile, limit: int | None = None) -> dict:
    """Run interactive selection, then download only the selected reels.

    Returns a summary dict with selection and download counts.
    """
    import yt_dlp

    from .auth import SESSION_COOKIE
    from .download import _ydl_opts, _record

    # --- selection phase ---
    selector = ReelSelector(profile)
    sel_result = selector.run()

    if not selector.selected:
        print(f"[ig-select] {profile.label}: no reels selected for download")
        return {**sel_result, "download_ok": 0, "download_failed": 0}

    to_download = selector.selected
    if limit is not None:
        to_download = to_download[:limit]

    # --- download phase ---
    print(f"\n[ig-select] {profile.label}: downloading {len(to_download)} selected reels")

    if not IG_COOKIES_FILE.exists():
        raise SystemExit(f"missing {IG_COOKIES_FILE} — run the `auth` step first")
    if SESSION_COOKIE not in IG_COOKIES_FILE.read_text(errors="replace"):
        raise SystemExit(
            f"{IG_COOKIES_FILE} has no `{SESSION_COOKIE}` cookie — run the `auth` step first"
        )

    ok = 0
    failed = 0
    with yt_dlp.YoutubeDL(_ydl_opts(profile)) as ydl:
        for i, sc in enumerate(to_download, 1):
            print(f"[ig-select] ({i}/{len(to_download)}) reel {sc}")
            try:
                info = ydl.extract_info(reel_url(sc), download=True)
            except Exception as exc:
                failed += 1
                _record(profile, sc, "error", f"{type(exc).__name__}: {exc}")
                print(f"[ig-select] reel {sc} failed: {exc}")
                continue

            if info is None:
                failed += 1
                _record(profile, sc, "error", "extraction returned no info")
            else:
                ok += 1
                _record(profile, sc, "ok")

    return {
        **sel_result,
        "download_ok": ok,
        "download_failed": failed,
    }
