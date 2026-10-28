"""CLI: pick a team -> find its next La Liga fixture -> predict it.

    python -m src.predict --team "Barcelona"
    python -m src.predict --team "Real Madrid" --offline   # no API calls, cache only
    python -m src.predict --competition ucl --team "Arsenal"   # Champions League: src/predict_ucl.py

Uses the Poisson + Dixon-Coles model, refitted on every played match from
2006-07 up to today. Features use only matches played before the fixture date.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.build_tables import build_matches
from src.evaluate import OUTCOMES
from src.features import EloParams, build_features
from src.ingest import CURRENT_SEASON_START, download_highlightly_season, fetch_season_stats
from src.models import CONFIG_PATH, MAX_GOALS, TRAIN_SEASONS, PoissonModel
from src.names import ALIASES, CANONICAL

LOCAL_TZ = "Europe/Madrid"
PREDICT_STATS_MAX_CALLS = 10  # stats for the two teams' recent matches only


class TeamNotFoundError(ValueError):
    """Raised when the --team text matches no known team."""


def resolve_team(text: str) -> str:
    """Map user text to a canonical team name (case/accent-insensitive aliases); suggest on failure."""
    def norm(s: str) -> str:
        import unicodedata
        return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).casefold().strip()

    lookup = {norm(alias): canon for alias, canon in ALIASES.items()}
    if norm(text) in lookup:
        return lookup[norm(text)]
    close = difflib.get_close_matches(norm(text), lookup, n=3, cutoff=0.6)
    hint = f" Did you mean: {', '.join(sorted({lookup[c] for c in close}))}?" if close else ""
    raise TeamNotFoundError(f"Unknown team '{text}'.{hint}")


def next_fixture(fixtures: pd.DataFrame, team: str, now: pd.Timestamp) -> tuple[pd.Series | None, pd.DataFrame]:
    """Earliest not-started fixture for team with kickoff after now.

    Also returns not-started fixtures whose kickoff is already in the past
    (stale cache or postponed), so the caller can warn about them.
    """
    mine = fixtures[((fixtures.home_team == team) | (fixtures.away_team == team)) & (fixtures.status == "Not started")]
    upcoming = mine[mine.kickoff_utc > now].sort_values("kickoff_utc")
    overdue = mine[mine.kickoff_utc <= now]
    return (upcoming.iloc[0] if len(upcoming) else None), overdue


def refresh_data(teams: set[str]) -> int:
    """Refresh the current season (only if cache > 12 h old) and the two teams' missing stats."""
    from src.api_client import BudgetExceededError

    try:
        download_highlightly_season(CURRENT_SEASON_START)
        return fetch_season_stats(CURRENT_SEASON_START, PREDICT_STATS_MAX_CALLS, teams=teams)
    except (BudgetExceededError, OSError, RuntimeError) as e:
        print(f"WARNING: refresh failed ({e}); using cached data.")
        return 0


def last_results(matches: pd.DataFrame, team: str, n: int = 5) -> pd.DataFrame:
    """The team's last n results from its own point of view."""
    mine = matches[(matches.home_team == team) | (matches.away_team == team)].sort_values("date").tail(n)
    rows = []
    for r in mine.itertuples():
        home = r.home_team == team
        gf, ga = (r.home_goals, r.away_goals) if home else (r.away_goals, r.home_goals)
        rows.append({"date": r.date.date(), "venue": "H" if home else "A",
                     "opponent": r.away_team if home else r.home_team,
                     "score": f"{gf}-{ga}", "result": "W" if gf > ga else "D" if gf == ga else "L"})
    return pd.DataFrame(rows)


def head_to_head(matches: pd.DataFrame, team: str, opp: str, n: int = 6) -> tuple[pd.DataFrame, dict[str, int]]:
    """Last n league meetings (team's point of view) and the all-time W/D/L record in the data."""
    pair = matches[((matches.home_team == team) & (matches.away_team == opp))
                   | ((matches.home_team == opp) & (matches.away_team == team))].sort_values("date")
    full = last_results(pair, team, n=len(pair)) if len(pair) else pd.DataFrame(columns=["result"])
    record = {k: int((full.result == k).sum()) for k in "WDL"}
    return full.tail(n), record


def forecast_many(played: pd.DataFrame, fixtures: pd.DataFrame) -> dict:
    """Poisson + Dixon-Coles forecasts for fixtures, trained on `played` (all before every fixture date).

    Returns arrays in the order of `rows` (the fixtures' feature rows).
    """
    cfg = json.loads(CONFIG_PATH.read_text())
    f = build_features(played, fixtures, elo=EloParams(**cfg["elo"]))
    train = f[(f.season >= TRAIN_SEASONS[0]) & f.target_result.notna()]
    model = PoissonModel().fit(train)
    rows = f[f.match_id.isin(fixtures.match_id)]
    lam_h, lam_a = model.expected_goals(rows)
    return {"proba": model.predict_proba(rows), "xg_home": np.asarray(lam_h), "xg_away": np.asarray(lam_a),
            "grid": model.score_grid(rows), "rows": rows}


def forecast(played: pd.DataFrame, fixture: pd.Series) -> dict:
    """Poisson + Dixon-Coles forecast for one fixture, trained on `played` (all before the fixture date)."""
    fc = forecast_many(played, fixture.to_frame().T.infer_objects())
    return {"proba": fc["proba"][0], "xg_home": float(fc["xg_home"][0]), "xg_away": float(fc["xg_away"][0]),
            "grid": fc["grid"][0], "row": fc["rows"]}


def day_table(names: pd.DataFrame, proba: np.ndarray, xg_home: np.ndarray, xg_away: np.ndarray,
              grid: np.ndarray, extra: dict[str, list] | None = None) -> pd.DataFrame:
    """One row per match for a matchday printout. names: home, away (+ result if played) columns."""
    top = [np.unravel_index(g.argmax(), g.shape) for g in grid]
    t = names.reset_index(drop=True).assign(
        **{"home win": [f"{p:.0%}" for p in proba[:, 0]], "draw": [f"{p:.0%}" for p in proba[:, 1]],
           "away win": [f"{p:.0%}" for p in proba[:, 2]],
           "xG": [f"{h:.2f}-{a:.2f}" for h, a in zip(xg_home, xg_away)],
           "likeliest": [f"{i}-{j} ({g[i, j]:.0%})" for (i, j), g in zip(top, grid)]})
    for k, v in (extra or {}).items():
        t[k] = v
    return t


def predict_date_laliga(day: pd.Timestamp, offline: bool = False) -> pd.DataFrame | None:
    """All La Liga matches on a date (Madrid time): forecasts from results before that date; actual results if played."""
    if not offline:
        refresh_data(set())
    matches, fixtures = build_matches()
    local = fixtures.kickoff_utc.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).dt.normalize()
    todo = fixtures[(local == day) & (fixtures.status == "Not started")]
    done = matches[matches.date == day]
    if todo.empty and done.empty:
        return None
    hidden = done.assign(home_goals=np.nan, away_goals=np.nan)  # forecast played matches without their result
    cols = ["match_id", "date", "season", "competition", "home_team", "away_team"]
    games = pd.concat([hidden[cols + ["home_goals", "away_goals"]], todo[cols]], ignore_index=True)
    games["date"] = pd.to_datetime(games.date)
    fc = forecast_many(matches[matches.date < day], games)
    res = done.set_index("match_id")
    r = fc["rows"]
    names = pd.DataFrame({"home": r.home_team.to_numpy(), "away": r.away_team.to_numpy()})
    result = [f"{int(res.home_goals[i])}-{int(res.away_goals[i])}" if i in res.index else "" for i in r.match_id]
    extra = {"result": result} if any(result) else None
    return day_table(names, fc["proba"], fc["xg_home"], fc["xg_away"], fc["grid"], extra)


def predict(team_text: str, offline: bool = False, now: pd.Timestamp | None = None) -> int:
    """Run the full prediction for one team. Returns a process exit code."""
    now = now or pd.Timestamp(datetime.now(timezone.utc))
    try:
        team = resolve_team(team_text)
    except TeamNotFoundError as e:
        print(e)
        return 2

    _, fixtures = build_matches()  # cache only, to find the opponent before refreshing
    fixture, _ = next_fixture(fixtures, team, now)
    calls = 0
    if not offline:
        teams = {team} | ({fixture.home_team, fixture.away_team} if fixture is not None else set())
        calls = refresh_data(teams)
    matches, fixtures = build_matches()
    fixture, overdue = next_fixture(fixtures, team, now)

    current = matches[matches.season == matches.season.max()]
    if team not in set(current.home_team) | set(current.away_team):
        print(f"{team} is not playing in La Liga {current.season.iloc[0]}; no fixture to predict.")
        return 1
    if len(overdue):
        print(f"WARNING: {len(overdue)} fixture(s) for {team} are past kickoff but not marked finished "
              f"(postponed, or data not refreshed): "
              + "; ".join(f"{r.home_team} v {r.away_team} {r.kickoff_utc:%Y-%m-%d}" for r in overdue.itertuples()))
    if fixture is None:
        print(f"No upcoming La Liga fixture found for {team} in the published schedule "
              f"(latest published kickoff: {fixtures.kickoff_utc.max():%Y-%m-%d}). Not guessing.")
        return 1

    played = matches[matches.date < fixture.date]
    fc = forecast(played, fixture)
    p, lam_h, lam_a, grid, row = fc["proba"], fc["xg_home"], fc["xg_away"], fc["grid"], fc["row"]
    top = np.dstack(np.unravel_index(np.argsort(grid, axis=None)[::-1][:3], grid.shape))[0]

    home, away = fixture.home_team, fixture.away_team
    opp = away if team == home else home
    kickoff = fixture.kickoff_utc.tz_convert(LOCAL_TZ)
    r = row.iloc[0]

    print(f"\n{home} vs {away}")
    print(f"La Liga {fixture.season}, matchweek {fixture.matchweek}, kickoff {kickoff:%a %d %b %Y %H:%M} (Madrid)")
    print(f"Data: results up to {played.date.max():%Y-%m-%d}; API calls this run: {calls}\n")
    width = max(len(home), len(away), 4) + 6
    for label, prob in zip([f"{home} win", "Draw", f"{away} win"], p):
        print(f"  {label:<{width}} {prob:6.1%}")
    print(f"\n  Expected goals: {home} {lam_h:.2f} - {lam_a:.2f} {away}")
    print("  Most likely scores: " + ", ".join(f"{i}-{j} ({grid[i, j]:.0%})" for i, j in top))
    print(f"  Pre-match Elo: {home} {r.home_elo:.0f}, {away} {r.away_elo:.0f}")

    for t in (team, opp):
        lr = last_results(played, t)
        print(f"\nLast 5 results, {t} ({''.join(lr.result)}):")
        print(lr.to_string(index=False))

    meetings, rec = head_to_head(played, team, opp)
    print(f"\nHead-to-head, {team} v {opp} (La Liga since 1993-94): "
          f"{rec['W']}W {rec['D']}D {rec['L']}L for {team}")
    print(meetings.to_string(index=False) if len(meetings) else "  no previous league meetings")
    print("\nNote: based on past results and shots only - no lineups, injuries or odds. "
          "Test-season RPS 0.198; treat as rough probabilities, not certainties.")
    return 0


def predict_next(team_text: str, n: int, competition: str, offline: bool = False, refresh_leagues: bool = False,
                 now: pd.Timestamp | None = None) -> int:
    """A team's next n fixtures (La Liga and/or Champions League), all forecast from today's data."""
    from src.predict_ucl import (ClubNotFoundError, load_ucl_fixtures, next_ucl_forecasts, refresh_ucl, resolve_club,
                                 season_clubs)
    from src.euro_elo import load_euro_matches

    now = now or pd.Timestamp(datetime.now(timezone.utc))
    today = now.tz_convert(LOCAL_TZ).tz_localize(None).normalize()
    team = club = None
    notes = []
    if competition in ("laliga", "all"):
        try:
            team = resolve_team(team_text)
        except TeamNotFoundError as e:
            notes.append(f"La Liga: {e}")
    if not offline:
        refresh_data({team} if team else set())
        if competition in ("ucl", "all"):
            refresh_ucl(refresh_leagues)
    label = team
    if competition in ("ucl", "all"):
        clubs = season_clubs(load_euro_matches(), load_ucl_fixtures())
        try:
            club = resolve_club(team_text, clubs)
            label = label or clubs.set_index("club_id").name[club]
        except ClubNotFoundError as e:
            notes.append(f"Champions League: {e}")
    if team is None and club is None:
        print("\n".join(notes))
        return 2
    if team is None and competition == "all":  # found in the UCL only: its own league is not predicted
        notes = [f"{label} is not a La Liga club; only La Liga and Champions League matches are predicted, "
                 "so its domestic league matches are not listed."]

    games = []
    if team:
        matches, fixtures = build_matches()
        mine = fixtures[((fixtures.home_team == team) | (fixtures.away_team == team))
                        & (fixtures.status == "Not started") & (fixtures.kickoff_utc > now)]
        for fx in mine.sort_values("kickoff_utc").head(n).itertuples(index=False):
            fx = pd.Series(fx._asdict())
            fc = forecast(matches, fx)  # every played match is before every fixture
            games.append({"date": fx.kickoff_utc.tz_convert(LOCAL_TZ).tz_localize(None), "comp": "La Liga",
                          "stage": f"Matchweek {fx.matchweek}", "home": fx.home_team, "away": fx.away_team,
                          "proba": fc["proba"], "xg_home": fc["xg_home"], "xg_away": fc["xg_away"],
                          "grid": fc["grid"], "home_through": None})
    if club:
        ucl = next_ucl_forecasts(club, n, today)
        games += ucl.to_dict("records")
    if not games:
        print(f"No upcoming fixtures for '{team_text}' in the published schedule. Not guessing.")
        return 1
    g = pd.DataFrame(games).sort_values("date", kind="stable").head(n).reset_index(drop=True)

    names = pd.DataFrame({"date": [f"{d:%a %d %b}" for d in g.date], "comp": g.comp, "stage": g.stage,
                          "home": g.home, "away": g.away})
    t = day_table(names, np.stack(g.proba), g.xg_home.to_numpy(float), g.xg_away.to_numpy(float), np.stack(g.grid))
    if g.home_through.notna().any():
        t["home through"] = [f"{p:.0%}" if pd.notna(p) else "" for p in g.home_through]
    print(f"\nNext {len(t)} match{'es' if len(t) > 1 else ''}, {label} "
          f"({' + '.join(c for c, x in (('La Liga', team), ('Champions League', club)) if x)}), 90-minute probabilities:")
    print(t.to_string(index=False))
    for note in notes:
        print(f"Note - {note}")
    print("\nAll forecasts use results up to today: later matches do not know the outcome of earlier ones,\n"
          "and the further ahead, the more lineups, injuries and form can change. No lineups, injuries or odds used.")
    return 0


def predict_date(day: pd.Timestamp, competition: str, offline: bool = False, refresh_leagues: bool = False) -> int:
    """Print forecasts for every match on `day`. Returns a process exit code (1 if there are none)."""
    from src.predict_ucl import predict_date_ucl

    tables = {}
    if competition in ("laliga", "all"):
        tables["La Liga"] = predict_date_laliga(day, offline)
    if competition in ("ucl", "all"):
        tables["Champions League"] = predict_date_ucl(day, offline, refresh_leagues)
    found = {k: t for k, t in tables.items() if t is not None}
    if not found:
        print(f"No {' or '.join(tables)} matches found on {day:%a %d %b %Y} in the published schedule.")
        return 1
    for name, t in found.items():
        played = "result" in t
        print(f"\n{name}, {day:%a %d %b %Y}: {len(t)} match{'es' if len(t) > 1 else ''} "
              f"(90-minute probabilities{'; forecasts made without that day' + chr(39) + 's results' if played else ''})")
        print(t.to_string(index=False))
    if "Champions League" in found:
        print("\npos = domestic league position (context only, not a model input; n/a = league not covered).")
    print("Based on past results only - no lineups, injuries or odds.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a team's next La Liga or Champions League fixture, "
                                                 "or every match on a date.")
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--team", help='e.g. "Barcelona", "Atletico Madrid", "Sevilla FC"')
    who.add_argument("--date", help="YYYY-MM-DD: all matches that day (past dates: forecast vs actual result)")
    parser.add_argument("--next", type=int, metavar="N", help="with --team: the next N matches in one table")
    parser.add_argument("--competition", choices=["laliga", "ucl", "all"],
                        help="default: laliga with --team, all with --date or --next")
    parser.add_argument("--offline", action="store_true", help="no API calls; use cached data only")
    parser.add_argument("--refresh-leagues", action="store_true",
                        help="ucl only: fetch current English/German/Italian/French/Portuguese/Dutch league "
                             "results from Highlightly (~18 calls if older than 12 h)")
    args = parser.parse_args()
    if args.date:
        if args.next:
            parser.error("--next applies to --team, not --date")
        sys.exit(predict_date(pd.Timestamp(args.date), args.competition or "all", args.offline, args.refresh_leagues))
    if args.next is not None:
        if args.next < 1:
            parser.error("--next must be at least 1")
        sys.exit(predict_next(args.team, args.next, args.competition or "all", args.offline, args.refresh_leagues))
    args.competition = args.competition or "laliga"
    if args.competition == "all":
        parser.error("--competition all only applies with --date")
    if args.competition == "ucl":
        from src.predict_ucl import predict_ucl
        sys.exit(predict_ucl(args.team, offline=args.offline, refresh_leagues=args.refresh_leagues))
    if args.refresh_leagues:
        parser.error("--refresh-leagues only applies to --competition ucl")
    sys.exit(predict(args.team, offline=args.offline))


if __name__ == "__main__":
    main()
