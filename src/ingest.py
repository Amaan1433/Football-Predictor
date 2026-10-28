"""Download and cache raw data under data/raw/.

Sources:
- DataHub La Liga CSVs (mirror of football-data.co.uk, odds columns removed).
  One file per season, 1993-94 to 2025-26.
- Highlightly API: current-season results and fixtures (free tier, 100 calls/day).

DataHub files are never re-fetched. The current Highlightly season is
re-fetched at most once per HIGHLIGHTLY_MAX_AGE_H hours (4 calls per refresh).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DATAHUB_DIR = RAW_DIR / "datahub_laliga"
DATAHUB_URL = "https://datahub.io/football/spanish-la-liga/_r/-/season-{code}.csv"

FIRST_SEASON_START = 1993  # 1993-94 is the earliest season on DataHub
LAST_SEASON_START = 2025   # 2025-26 is the latest complete season on DataHub
REQUEST_DELAY_S = 1.0      # polite pause between uncached requests
USER_AGENT = "football-predictor/0.1 (personal research)"

CURRENT_SEASON_START = 2026  # 2026-27, not on DataHub; comes from Highlightly
HIGHLIGHTLY_LALIGA_ID = 119924
HIGHLIGHTLY_UCL_ID = 2486  # "UEFA Champions League", found via /leagues?leagueName=...
# Current-season domestic leagues (openfootball lags by weeks); fetched only on request (~22 calls).
HIGHLIGHTLY_DOMESTIC_IDS = {"ENG": 33973, "GER": 67162, "ITA": 115669, "FRA": 52695, "POR": 80778, "NED": 75672}
HIGHLIGHTLY_MAX_AGE_H = 12.0
STATS_MAX_CALLS_PER_RUN = 80  # leaves headroom in the 100/day budget for fixture refreshes

# openfootball (CC0): Champions League results + domestic leagues for cross-league ratings
OPENFOOTBALL_DIR = RAW_DIR / "openfootball"
OPENFOOTBALL_UCL_URL = "https://raw.githubusercontent.com/openfootball/champions-league/master/{season}/{file}.txt"
OPENFOOTBALL_JSON_URL = "https://raw.githubusercontent.com/openfootball/football.json/master/{season}/{code}.json"
UCL_FILES = ("cl", "clq")              # main competition, qualifiers (qualifiers only in recent seasons)
UCL_FIRST_SEASON_START = 2011          # 2011-12 is the earliest season on openfootball
UCL_LAST_SEASON_START = 2025           # 2026-27 comes from Highlightly
DOMESTIC_FIRST_SEASON_START = 2010     # one warm-up season before the first UCL season
# Spain is not listed: it comes from our own La Liga tables (DataHub + Highlightly).
EURO_LEAGUES = {"ENG": "en.1", "GER": "de.1", "ITA": "it.1", "FRA": "fr.1", "POR": "pt.1", "NED": "nl.1"}


def season_code(start_year: int) -> str:
    """Return football-data style season code, e.g. 2025 -> '2526'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """Return a readable season label, e.g. 2025 -> '2025-26'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def fetch(url: str, dest: Path) -> bool:
    """Download url to dest unless already cached. Returns True if downloaded."""
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read()
    if body.lstrip()[:1] == b"<":
        raise RuntimeError(f"Expected CSV but got HTML from {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(body)
    tmp.replace(dest)
    return True


def download_datahub(first: int = FIRST_SEASON_START, last: int = LAST_SEASON_START) -> list[Path]:
    """Download every DataHub La Liga season in [first, last]. Fails loudly on errors."""
    paths = []
    for year in range(first, last + 1):
        dest = DATAHUB_DIR / f"season-{season_code(year)}.csv"
        url = DATAHUB_URL.format(code=season_code(year))
        try:
            downloaded = fetch(url, dest)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Failed to download {season_label(year)} from {url}: {e}") from e
        print(f"{season_label(year)}: {'downloaded' if downloaded else 'cached'} -> {dest.name}")
        if downloaded:
            time.sleep(REQUEST_DELAY_S)
        paths.append(dest)
    return paths


def download_highlightly_season(start_year: int, max_age_h: float = HIGHLIGHTLY_MAX_AGE_H,
                                league_id: int = HIGHLIGHTLY_LALIGA_ID) -> list[dict]:
    """All matches (finished and scheduled) of one league season, paginated 100 per call."""
    from src.api_client import get  # local import: api_client imports this module

    matches: list[dict] = []
    offset = 0
    while True:
        page = get(
            "highlightly", "/matches",
            {"leagueId": league_id, "season": start_year, "limit": 100, "offset": offset},
            max_age_s=max_age_h * 3600,
        )
        matches.extend(page["data"])
        total = page["pagination"]["totalCount"]
        offset += 100
        if offset >= total:
            break
    out = highlightly_season_path(start_year, league_id)
    out.write_text(json.dumps(matches, ensure_ascii=False), encoding="utf-8")
    print(f"{season_label(start_year)} (Highlightly league {league_id}): {len(matches)} matches -> {out.name}")
    return matches


def fetch_season_stats(start_year: int, max_calls: int = STATS_MAX_CALLS_PER_RUN,
                       teams: set[str] | None = None) -> int:
    """Fetch per-match statistics for finished matches, 1 call each, cached forever.

    teams: only matches involving these canonical team names (None = all).
    Resumable: already-cached matches cost nothing. Stops after max_calls network
    calls or when the daily budget is reached; rerun later to continue.
    Returns the number of network calls made.
    """
    from src.api_client import BudgetExceededError, cached, get
    from src.names import ALIASES

    season = json.loads(highlightly_season_path(start_year).read_text(encoding="utf-8"))
    finished = [
        m["id"] for m in season
        if m["state"]["description"] == "Finished"
        and (teams is None or {ALIASES.get(m["homeTeam"]["name"]), ALIASES.get(m["awayTeam"]["name"])} & teams)
    ]
    todo = [i for i in finished if cached("highlightly", f"/statistics/{i}") is None]
    print(f"{season_label(start_year)} stats: {len(finished) - len(todo)}/{len(finished)} cached, "
          f"{len(todo)} to fetch (cap {max_calls} this run)")
    done = 0
    for match_id in todo[:max_calls]:
        try:
            get("highlightly", f"/statistics/{match_id}")
        except BudgetExceededError as e:
            print(f"Stopped: {e}")
            break
        done += 1
    print(f"Fetched {done}; {len(todo) - done} still missing")
    return done


def fetch_extra_time_details(start_year: int, league_id: int = HIGHLIGHTLY_UCL_ID) -> int:
    """Match details (goal events) for matches that went to extra time, 1 call each, cached forever.

    The season listing only has the after-extra-time score; the events give the 90-minute one.
    Returns the number of network calls made.
    """
    from src.api_client import cached, get

    season = json.loads(highlightly_season_path(start_year, league_id).read_text(encoding="utf-8"))
    et = [m["id"] for m in season if m["state"]["description"] in ("Finished after extra time", "Finished after penalties")]
    todo = [i for i in et if cached("highlightly", f"/matches/{i}") is None]
    for match_id in todo:
        get("highlightly", f"/matches/{match_id}")
    print(f"{season_label(start_year)} extra-time details: {len(et) - len(todo)} cached, {len(todo)} fetched")
    return len(todo)


def fetch_optional(url: str, dest: Path) -> str:
    """Like fetch, but a 404 is cached as a '.missing' marker so it is never re-requested.

    Returns 'downloaded', 'cached', 'missing' (new 404) or 'missing (cached)'.
    """
    marker = dest.with_suffix(dest.suffix + ".missing")
    if marker.exists():
        return "missing (cached)"
    try:
        return "downloaded" if fetch(url, dest) else "cached"
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        return "missing"


def download_openfootball() -> None:
    """Champions League text files and domestic league JSON from openfootball (CC0), cached."""
    counts: dict[str, int] = {}
    jobs = [
        (OPENFOOTBALL_UCL_URL.format(season=season_label(y), file=f), OPENFOOTBALL_DIR / "ucl" / f"{season_label(y)}_{f}.txt")
        for y in range(UCL_FIRST_SEASON_START, UCL_LAST_SEASON_START + 1) for f in UCL_FILES
    ] + [
        (OPENFOOTBALL_JSON_URL.format(season=season_label(y), code=code), OPENFOOTBALL_DIR / "leagues" / f"{season_label(y)}_{code}.json")
        for y in range(DOMESTIC_FIRST_SEASON_START, CURRENT_SEASON_START + 1) for code in EURO_LEAGUES.values()
    ]
    for url, dest in jobs:
        status = fetch_optional(url, dest)
        counts[status] = counts.get(status, 0) + 1
        if status in ("downloaded", "missing"):  # a network request was made
            time.sleep(REQUEST_DELAY_S / 2)
    print(f"openfootball: {counts}")


def download_highlightly_domestic(max_age_h: float = HIGHLIGHTLY_MAX_AGE_H) -> None:
    """Current season of the six non-Spanish leagues from Highlightly (3-4 calls each if the cache is stale)."""
    for league_id in HIGHLIGHTLY_DOMESTIC_IDS.values():
        download_highlightly_season(CURRENT_SEASON_START, max_age_h, league_id=league_id)


def refresh_openfootball_current(max_age_h: float = HIGHLIGHTLY_MAX_AGE_H) -> int:
    """Re-download the in-progress domestic seasons (openfootball, not the API budget) if older than max_age_h.

    Returns the number of files re-downloaded. A failed download keeps the cached file.
    """
    done = 0
    for code in EURO_LEAGUES.values():
        dest = OPENFOOTBALL_DIR / "leagues" / f"{season_label(CURRENT_SEASON_START)}_{code}.json"
        if dest.exists() and time.time() - dest.stat().st_mtime < max_age_h * 3600:
            continue
        tmp = dest.with_name(dest.name + ".new")
        tmp.unlink(missing_ok=True)
        fetch(OPENFOOTBALL_JSON_URL.format(season=season_label(CURRENT_SEASON_START), code=code), tmp)
        tmp.replace(dest)
        done += 1
        time.sleep(REQUEST_DELAY_S / 2)
    return done


def highlightly_season_path(start_year: int, league_id: int = HIGHLIGHTLY_LALIGA_ID) -> Path:
    """Combined per-season file that the table builders read (no API calls there)."""
    if league_id == HIGHLIGHTLY_LALIGA_ID:
        return RAW_DIR / "highlightly" / f"season_{start_year}.json"
    return RAW_DIR / "highlightly" / f"league_{league_id}_season_{start_year}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and cache raw data.")
    parser.add_argument("--first", type=int, default=FIRST_SEASON_START)
    parser.add_argument("--last", type=int, default=LAST_SEASON_START)
    parser.add_argument("--no-current", action="store_true", help="skip the Highlightly current season")
    parser.add_argument("--europe", action="store_true", help="also download openfootball UCL + domestic leagues")
    parser.add_argument("--refresh-leagues", action="store_true",
                        help="with --europe: current domestic seasons from Highlightly (~18 calls)")
    parser.add_argument("--stats-max-calls", type=int, default=STATS_MAX_CALLS_PER_RUN,
                        help="max Highlightly calls for per-match stats this run (0 = skip)")
    args = parser.parse_args()
    download_datahub(args.first, args.last)
    if args.europe:
        download_openfootball()
        if not args.no_current:
            download_highlightly_season(CURRENT_SEASON_START, league_id=HIGHLIGHTLY_UCL_ID)
            fetch_extra_time_details(CURRENT_SEASON_START)
            if args.refresh_leagues:
                download_highlightly_domestic()
    if not args.no_current:
        download_highlightly_season(CURRENT_SEASON_START)
        if args.stats_max_calls > 0:
            fetch_season_stats(CURRENT_SEASON_START, args.stats_max_calls)


if __name__ == "__main__":
    main()
