# Facebook Reels downloader

Two-stage pipeline for bulk-collecting reel videos from Facebook profiles.

yt-dlp has an extractor for `facebook.com/reel/<id>` but **none** for a profile's reels
listing, so enumeration is done here with Playwright and only the download is delegated
to yt-dlp:

1. **discover** — drive a logged-in browser over `?sk=reels_tab`, scroll until it goes
   dry, harvest reel IDs into `state/<profile_id>/manifest.jsonl`
2. **download** — fetch each `facebook.com/reel/<id>` with yt-dlp, capped at 720p

Discovery stores **only reel IDs**, never resolved media URLs: `*.fbcdn.net` links are
signature-scoped and expire within hours, so a cached URL manifest would 403 by the next
day. yt-dlp re-resolves each reel at download time.

## Setup

```bash
conda env create -f environment.yml
conda activate mancing-scrape

playwright install chromium
playwright install chromium-headless-shell   # required for --headless
```

The second `playwright install` is separate and easy to miss — without it the `--headless`
flag fails with "Executable doesn't exist at .../chrome-headless-shell".

## Usage

Run from this directory (the `src` package is imported by module path):

```bash
python -m src.pipeline auth                    # log in once, export cookies for yt-dlp
python -m src.pipeline discover                # harvest reel IDs
python -m src.pipeline download --limit 3      # fetch a few to smoke-test
python -m src.pipeline download                # fetch the rest
python -m src.pipeline run                     # all three, every enabled profile
```

Useful flags: `--profile <id|label>` to narrow to one target, `--limit N` to cap a
download batch, `--headless` once a headed run is known to work, `--max-scrolls` /
`--stall-rounds` to tune discovery.

First `auth` run opens a real browser window — log into the dummy account by hand. The
session persists in `state/browser_profile/` and is reused. Alternatively drop a Netscape
cookies file at `state/cookies_import.txt` and it will be imported instead.

## Adding profiles

Append to `config/profiles.yml`. The numeric ID is the stable key — the slug in a
`/people/<slug>/<id>/` URL is cosmetic and can change, so the reels tab URL is built as
`profile.php?id=<id>&sk=reels_tab`. State and output partition per profile ID.

```yaml
profiles:
  - id: "61575602817209"
    label: berkah-hari-ini
    enabled: true
```

## Resuming

Everything is idempotent. Discovery appends to the manifest and dedupes on re-run;
downloads pre-filter against the yt-dlp archive **and** files on disk, so reels you
already have cost zero network requests. `.part` files don't count as done, so
interrupted downloads are retried. Interrupt with Ctrl-C any time — progress is on disk.

## Operational notes

Automated collection breaks Facebook's ToS and accounts doing it get checkpointed or
disabled. Use a throwaway account, never a personal one, and treat account loss as an
operating cost rather than a surprise.

What actually reduces the failure rate:

- **Warm the account first.** Browse normally as it for a day. A fresh account that
  immediately paginates a reels tab gets checkpointed fast.
- Run **headed** until it works; headless Chromium is more detectable.
- Run in bursts with gaps, not one continuous multi-hour crawl.
- Stay on a normal residential connection — datacenter/VPN IPs raise the checkpoint rate.
- Downloads are deliberately **sequential**; don't add concurrency.

If discovery prints the `--max-scrolls` backstop warning, coverage was truncated — raise
the value and re-run. Normal termination is the stall counter, not the backstop.

`state/` holds live credentials (`cookies.txt`, browser profile) and `data/` holds media;
both are gitignored.
