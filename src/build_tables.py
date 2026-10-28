"""Build processed tables from cached raw data. Never calls an API.

matches:  one row per finished match (see MATCH_COLUMNS).
fixtures: scheduled current-season matches (see FIXTURE_COLUMNS).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.api_client import cached
from src.ingest import CURRENT_SEASON_START, DATAHUB_DIR, ROOT, highlightly_season_path, season_label
from src.names import check_mapped, ALIASES

PROCESSED_DIR = ROOT / "data" / "processed"
MATCHES_PATH = PROCESSED_DIR / "matches.csv"
FIXTURES_PATH = PROCESSED_DIR / "fixtures.csv"
TEAM_MATCH_PATH = PROCESSED_DIR / "team_match.csv"
LOCAL_TZ = "Europe/Madrid"  # DataHub dates are local match dates

FIXTURE_COLUMNS = [
    "match_id", "kickoff_utc", "date", "season", "competition", "matchweek",
    "home_team", "away_team", "status", "source_id",
]

MATCH_COLUMNS = [
    "match_id", "date", "season", "competition",
    "home_team", "away_team", "home_goals", "away_goals",
    "home_xg", "away_xg", "home_shots", "away_shots",
    "home_sot", "away_sot", "odds_h", "odds_d", "odds_a",
]


def _season_start_from_code(code: str) -> int:
    """'9394' -> 1993, '2526' -> 2025."""
    yy = int(code[:2])
    return 1900 + yy if yy >= 90 else 2000 + yy


def load_datahub_season(path: Path) -> pd.DataFrame:
    """Read one DataHub season CSV into the matches schema."""
    raw = pd.read_csv(path, encoding="latin-1")
    raw = raw.dropna(how="all")
    check_mapped(pd.concat([raw["HomeTeam"], raw["AwayTeam"]]))

    start = _season_start_from_code(path.stem.split("-")[1])
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"], format="%Y-%m-%d"),
        "season": season_label(start),
        "competition": "LaLiga",
        "home_team": raw["HomeTeam"].map(ALIASES),
        "away_team": raw["AwayTeam"].map(ALIASES),
        "home_goals": raw["FTHG"].astype(int),
        "away_goals": raw["FTAG"].astype(int),
        "home_xg": np.nan,  # no permitted xG source wired in yet
        "away_xg": np.nan,
        "home_shots": raw["HS"],
        "away_shots": raw["AS"],
        "home_sot": raw["HST"],
        "away_sot": raw["AST"],
        "odds_h": np.nan,  # odds columns are stripped from the DataHub mirror
        "odds_d": np.nan,
        "odds_a": np.nan,
    })

    # Sanity check: the recorded result letter must agree with the goals.
    implied = np.select(
        [df.home_goals > df.away_goals, df.home_goals < df.away_goals], ["H", "A"], "D"
    )
    bad = raw["FTR"].to_numpy() != implied
    if bad.any():
        raise ValueError(f"{path.name}: FTR disagrees with goals in {bad.sum()} rows")
    return df


def _cached_shots(source_id: int, home_id: int) -> tuple[float, float, float, float]:
    """(home_shots, away_shots, home_sot, away_sot) from cached Highlightly stats, NaN if absent.

    Total shots = on target + off target + blocked (matches DataHub's HS/AS definition).
    """
    stats = cached("highlightly", f"/statistics/{source_id}")
    if not stats or len(stats) != 2:
        return (np.nan,) * 4
    by_side = {}
    for team in stats:
        s = {x["displayName"]: x["value"] for x in team["statistics"]}
        total = sum(s.get(k, 0) for k in ("Shots on target", "Shots off target", "Blocked shots"))
        by_side["home" if team["team"]["id"] == home_id else "away"] = (total, s.get("Shots on target", np.nan))
    if set(by_side) != {"home", "away"}:
        raise ValueError(f"Highlightly stats {source_id}: could not identify home/away team")
    return by_side["home"][0], by_side["away"][0], by_side["home"][1], by_side["away"][1]


def load_highlightly_season(start_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the cached Highlightly season. Returns (finished matches, all non-finished rows)."""
    path = highlightly_season_path(start_year)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m src.ingest` first")
    raw = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame({
        "source_id": [m["id"] for m in raw],
        "kickoff_utc": pd.to_datetime([m["date"] for m in raw], utc=True),
        "matchweek": [int(m["round"].rsplit("-", 1)[1]) for m in raw],
        "status": [m["state"]["description"] for m in raw],
        "score": [m["state"]["score"]["current"] for m in raw],
        "home_id": [m["homeTeam"]["id"] for m in raw],
        "home_raw": [m["homeTeam"]["name"] for m in raw],
        "away_raw": [m["awayTeam"]["name"] for m in raw],
    })
    check_mapped(pd.concat([df.home_raw, df.away_raw]))
    df["home_team"] = df.home_raw.map(ALIASES)
    df["away_team"] = df.away_raw.map(ALIASES)
    df["date"] = df.kickoff_utc.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).dt.normalize()
    df["season"] = season_label(start_year)
    df["competition"] = "LaLiga"

    fin = df[df.status == "Finished"].copy()
    goals = fin.score.str.split(" - ", expand=True).astype(int)
    shots = pd.DataFrame(
        [_cached_shots(sid, hid) for sid, hid in zip(fin.source_id, fin.home_id)],
        index=fin.index, columns=["home_shots", "away_shots", "home_sot", "away_sot"],
    )
    matches = pd.DataFrame({
        "date": fin.date, "season": fin.season, "competition": fin.competition,
        "home_team": fin.home_team, "away_team": fin.away_team,
        "home_goals": goals[0], "away_goals": goals[1],
        "home_xg": np.nan, "away_xg": np.nan,
        **{c: shots[c] for c in shots.columns},  # NaN where stats not fetched yet
        "odds_h": np.nan, "odds_d": np.nan, "odds_a": np.nan,
    })
    others = df[df.status != "Finished"].copy()
    others["match_id"] = make_match_id(others)
    return matches, others[FIXTURE_COLUMNS].sort_values("kickoff_utc").reset_index(drop=True)


def make_match_id(df: pd.DataFrame) -> pd.Series:
    """Deterministic id: competition_date_home_away (spaces -> underscores)."""
    s = df.competition + "_" + df.date.dt.strftime("%Y%m%d") + "_" + df.home_team + "_" + df.away_team
    return s.str.replace(" ", "_", regex=False)


def validate_matches(df: pd.DataFrame, complete_seasons: set[str]) -> None:
    """Fail loudly on duplicate ids/pairings, or an unbalanced list in a complete season."""
    dup = df.match_id.duplicated()
    if dup.any():
        raise ValueError(f"Duplicate match_ids: {df.loc[dup, 'match_id'].tolist()[:5]}")
    for season, g in df.groupby("season"):
        pairs = g.groupby(["home_team", "away_team"]).size()
        if (pairs != 1).any():
            raise ValueError(f"{season}: some home/away pairing appears more than once")
        if season not in complete_seasons:
            continue
        teams = pd.unique(g[["home_team", "away_team"]].to_numpy().ravel())
        n = len(teams)
        if len(g) != n * (n - 1):
            raise ValueError(f"{season}: {len(g)} matches but {n} teams (expected {n * (n - 1)})")


def build_matches() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine DataHub seasons and the current Highlightly season. Returns (matches, fixtures)."""
    files = sorted(DATAHUB_DIR.glob("season-*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw files in {DATAHUB_DIR}; run `python -m src.ingest` first")
    hist = pd.concat([load_datahub_season(f) for f in files], ignore_index=True)
    current, fixtures = load_highlightly_season(CURRENT_SEASON_START)
    df = pd.concat([hist, current], ignore_index=True)
    df["match_id"] = make_match_id(df)
    df = df.sort_values(["date", "home_team"], kind="stable").reset_index(drop=True)[MATCH_COLUMNS]
    validate_matches(df, complete_seasons=set(hist.season))
    return df, fixtures


def build_team_match(matches: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
    """Two rows per match, one from each team's point of view.

    Unplayed fixtures (NaN goals) get NaN points; pass validate=False for those.
    """
    def side(is_home: bool) -> pd.DataFrame:
        us, them = ("home", "away") if is_home else ("away", "home")
        gf, ga = matches[f"{us}_goals"], matches[f"{them}_goals"]
        return pd.DataFrame({
            "match_id": matches.match_id, "date": matches.date, "season": matches.season,
            "competition": matches.competition,
            "team": matches[f"{us}_team"], "opp": matches[f"{them}_team"], "is_home": int(is_home),
            "gf": gf, "ga": ga,
            "xg_for": matches[f"{us}_xg"], "xg_against": matches[f"{them}_xg"],
            "shots_for": matches[f"{us}_shots"], "shots_against": matches[f"{them}_shots"],
            "sot_for": matches[f"{us}_sot"], "sot_against": matches[f"{them}_sot"],
            "points": np.where(gf.isna(), np.nan, np.select([gf > ga, gf == ga], [3, 1], 0)),
        })

    tm = pd.concat([side(True), side(False)], ignore_index=True)
    tm = tm.sort_values(["date", "match_id", "is_home"], ascending=[True, True, False], kind="stable")
    tm = tm.reset_index(drop=True)
    if validate:
        validate_team_match(tm, matches)
    return tm


def validate_team_match(tm: pd.DataFrame, matches: pd.DataFrame) -> None:
    """Each match has exactly one home and one away row that mirror each other."""
    if len(tm) != 2 * len(matches):
        raise ValueError(f"team_match has {len(tm)} rows, expected {2 * len(matches)}")
    home = tm[tm.is_home == 1].set_index("match_id")
    away = tm[tm.is_home == 0].set_index("match_id").loc[home.index]
    checks = {
        "team/opp": (home.team == away.opp) & (home.opp == away.team),
        "goals": (home.gf == away.ga) & (home.ga == away.gf),
        "shots": (home.shots_for.fillna(-1) == away.shots_against.fillna(-1)),
        "points": (home.points + away.points).isin([2, 3]),
    }
    for name, ok in checks.items():
        if not ok.all():
            raise ValueError(f"team_match {name} mismatch in {(~ok).sum()} matches")
    if tm.groupby(["team", "date"]).size().max() > 1:
        raise ValueError("A team has two matches on the same date")


def report_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Per-season row counts and missing-data counts."""
    g = df.groupby("season")
    rep = pd.DataFrame({
        "rows": g.size(),
        "teams": g.home_team.nunique(),
        "first": g.date.min().dt.date,
        "last": g.date.max().dt.date,
        "missing_shots": g.home_shots.apply(lambda s: s.isna().sum()),
        "missing_sot": g.home_sot.apply(lambda s: s.isna().sum()),
        "missing_xg": g.home_xg.apply(lambda s: s.isna().sum()),
        "missing_odds": g.odds_h.apply(lambda s: s.isna().sum()),
    })
    return rep


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed tables.")
    parser.parse_args()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matches, fixtures = build_matches()
    team_match = build_team_match(matches)
    matches.to_csv(MATCHES_PATH, index=False)
    fixtures.to_csv(FIXTURES_PATH, index=False)
    team_match.to_csv(TEAM_MATCH_PATH, index=False)
    print(f"Wrote {len(matches)} matches to {MATCHES_PATH.relative_to(ROOT)}")
    print(f"Wrote {len(team_match)} team-match rows to {TEAM_MATCH_PATH.relative_to(ROOT)}")
    print(f"Wrote {len(fixtures)} non-finished fixtures to {FIXTURES_PATH.relative_to(ROOT)} "
          f"(status counts: {fixtures.status.value_counts().to_dict()})\n")
    with pd.option_context("display.width", 200, "display.max_rows", 100, "display.max_columns", None):
        print(report_matches(matches))


if __name__ == "__main__":
    main()
