"""CLI: pick a club -> find its next Champions League fixture -> predict it.

    python -m src.predict --competition ucl --team "Arsenal"
    python -m src.predict --competition ucl --team "Bayern" --offline   # no network, cache only
    python -m src.predict --competition ucl --team "Bayern" --refresh-leagues   # + current domestic results (~18 calls)

Model: Poisson + Dixon-Coles on the cross-league Elo (src/euro_elo.py, src/ucl.py), fitted
on every played Champions League match since 2011-12. Ratings use all results (domestic
and European) before the fixture. Knockout fixtures also get P(qualify): two-legged ties
combine both legs (or use the known first-leg score), then extra time and 50/50 penalties.
"""

from __future__ import annotations

import difflib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.euro_elo import EURO_CONFIG_PATH, LEAGUE_NAMES, EuroEloParams, load_euro_matches
from src.europe import CLUBS_PATH, EURO_MATCHES_PATH, UCL_FIXTURES_PATH, build_euro_matches, clubs_table
from src.ingest import CURRENT_SEASON_START, HIGHLIGHTLY_UCL_ID, season_label
from src.names import ALIASES
from src.names_europe import normalize
from src.ucl import (CHOSEN_MODEL, FIRST_UCL_SEASON, LAST_AWAY_GOALS_SEASON, build_ucl_frame, is_group_stage,
                     p_through_single, p_through_two_legs, ucl_models)

UCL_SEASON = season_label(CURRENT_SEASON_START)


class ClubNotFoundError(ValueError):
    """Raised when the --team text matches no club in the current Champions League season."""


# ------------------------------------------------------------------ data

def load_ucl_fixtures() -> pd.DataFrame:
    return pd.read_csv(UCL_FIXTURES_PATH, parse_dates=["date"])


def refresh_ucl(refresh_leagues: bool = False) -> int:
    """Refresh the cached sources (each at most every 12 h) and rebuild the tables. Returns Highlightly calls made.

    refresh_leagues: also fetch the six other domestic leagues from Highlightly (~18 calls when
    stale); otherwise they come from openfootball, which can lag by weeks.
    """
    from src.api_client import BudgetExceededError, calls_today
    from src.build_tables import FIXTURES_PATH, MATCHES_PATH, build_matches
    from src.ingest import (download_highlightly_domestic, download_highlightly_season, fetch_extra_time_details,
                            refresh_openfootball_current)

    before = calls_today("highlightly")
    try:
        download_highlightly_season(CURRENT_SEASON_START)                      # La Liga (Spanish ratings)
        download_highlightly_season(CURRENT_SEASON_START, league_id=HIGHLIGHTLY_UCL_ID)
        fetch_extra_time_details(CURRENT_SEASON_START)
        refresh_openfootball_current()                                         # other domestic leagues
        if refresh_leagues:
            download_highlightly_domestic()
    except (BudgetExceededError, OSError, RuntimeError) as e:
        print(f"WARNING: refresh failed ({e}); using cached data.")
    matches, fixtures = build_matches()
    matches.to_csv(MATCHES_PATH, index=False)
    fixtures.to_csv(FIXTURES_PATH, index=False)
    m, fx = build_euro_matches()
    m.to_csv(EURO_MATCHES_PATH, index=False)
    fx.to_csv(UCL_FIXTURES_PATH, index=False)
    clubs_table(m).to_csv(CLUBS_PATH, index=False)
    return calls_today("highlightly") - before


def season_clubs(played: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Clubs in this season's Champions League (qualifiers included): club_id, latest name, still in it."""
    cur = pd.concat([played[played.season == UCL_SEASON], fixtures])
    cur = cur[cur.competition.isin(["UCL", "UCLQ"])].sort_values("date")
    long = pd.concat([cur[["date", "competition", "home_id", "home_raw"]].set_axis(["date", "competition", "club_id", "name"], axis=1),
                      cur[["date", "competition", "away_id", "away_raw"]].set_axis(["date", "competition", "club_id", "name"], axis=1)])
    g = long.groupby("club_id")
    return pd.DataFrame({"name": g.name.last(), "in_main": g.competition.apply(lambda s: s.eq("UCL").any())}).reset_index()


def resolve_club(text: str, clubs: pd.DataFrame) -> str:
    """Map user text to a club id: spelling, id key or (Spanish clubs) any La Liga alias; suggest on failure."""
    lookup: dict[str, str] = {}
    for r in clubs.itertuples():
        lookup[normalize(r.name)] = r.club_id
        lookup[r.club_id.split(":", 1)[1]] = r.club_id
    for alias, canon in ALIASES.items():
        cid = f"ESP:{normalize(canon)}"
        if cid in set(clubs.club_id):
            lookup[normalize(alias)] = cid
    key = normalize(text)
    if key in lookup:
        return lookup[key]
    starts = {cid for k, cid in lookup.items() if k.startswith(key + " ") or k.endswith(" " + key)}
    if len(starts) == 1:  # "Bayern", "Leverkusen"
        return starts.pop()
    close = difflib.get_close_matches(key, lookup, n=4, cutoff=0.6)
    names = clubs.set_index("club_id").name
    hint = sorted({names[lookup[c]] for c in close} | {names[c] for c in starts})
    raise ClubNotFoundError(f"No {UCL_SEASON} Champions League club matches '{text}'."
                            + (f" Did you mean: {', '.join(hint)}?" if hint else ""))


# ------------------------------------------------------------------ forecast

def tie_role(fixture: pd.Series, played: pd.DataFrame, fixtures: pd.DataFrame) -> tuple[str, pd.Series | None]:
    """'league', 'leg1' (return leg scheduled), 'leg2' (first leg played; returned),
    'leg2_pending' (first leg scheduled earlier, not yet played) or 'single'."""
    if fixture.competition == "UCL" and is_group_stage(pd.Series([fixture.stage])).iloc[0]:
        return "league", None
    same = lambda df: df[(df.season == fixture.season) & (df.stage == fixture.stage)  # noqa: E731
                         & (df.home_id == fixture.away_id) & (df.away_id == fixture.home_id)]
    first = same(played)
    if len(first):
        return "leg2", first.iloc[0]
    other = same(fixtures)
    if len(other):
        return ("leg2_pending", None) if (other.date < fixture.date).any() else ("leg1", None)
    return "single", None


def p_home_side_through(model, row: pd.DataFrame, role: str, first_leg: pd.Series | None, season: str) -> float:
    """P(the fixture's home side goes through). row: the fixture with its features."""
    rule = season <= LAST_AWAY_GOALS_SEASON
    lh, la = (float(x[0]) for x in model.expected_goals(row))
    grid = model.score_grid(row)[0]
    if role == "single":
        return p_through_single(grid, lh, la)
    if role == "leg2":  # A = first-leg home side = this fixture's away side
        a_through = p_through_two_legs(grid, lh, la, rule, leg1=(int(first_leg.home_goals), int(first_leg.away_goals)))
        return 1 - a_through
    rev = row.assign(elo_diff=-row.elo_diff, neutral_f=0.0)  # the other leg, venue swapped, as seen today
    rev_grid = model.score_grid(rev)[0]
    if role == "leg2_pending":  # the other leg (A at home) comes first; this fixture is leg 2
        return 1 - p_through_two_legs(grid, lh, la, rule, grid1=rev_grid)
    lh2, la2 = (float(x[0]) for x in model.expected_goals(rev))
    return p_through_two_legs(rev_grid, lh2, la2, rule, grid1=grid)


def frame_and_model(played: pd.DataFrame, upcoming: pd.DataFrame):
    """UCL feature frame over played + upcoming matches (upcoming carry ratings only) and the fitted model."""
    p = EuroEloParams(**json.loads(EURO_CONFIG_PATH.read_text())["elo"])
    allm = pd.concat([played, upcoming], ignore_index=True)
    allm = allm.sort_values(["date", "competition", "home_id"], kind="stable").reset_index(drop=True)
    u = build_ucl_frame(allm, p)
    train = u[(u.competition == "UCL") & u.target_result.notna() & (u.season >= FIRST_UCL_SEASON)]
    return u, ucl_models()[CHOSEN_MODEL].fit(train), len(train)


def forecast(played: pd.DataFrame, fixtures: pd.DataFrame, fixture: pd.Series) -> dict:
    """Forecast one fixture from all results before it (other fixtures only carry ratings forward)."""
    u, model, n_train = frame_and_model(played, fixtures[fixtures.date <= fixture.date])
    row = u[u.match_id == fixture.match_id]
    lam_h, lam_a = model.expected_goals(row)
    role, first_leg = tie_role(fixture, played, fixtures)
    out = {"proba": model.predict_proba(row)[0], "xg_home": float(lam_h[0]), "xg_away": float(lam_a[0]),
           "grid": model.score_grid(row)[0], "row": row, "role": role, "first_leg": first_leg, "train_n": n_train}
    if role != "league":
        out["p_home_through"] = p_home_side_through(model, row, role, first_leg, fixture.season)
    return out


def league_context(played: pd.DataFrame, club: str, season: str) -> dict | None:
    """Club's current domestic standing and last 3 league results, or None if its league is not covered."""
    country = club.split(":", 1)[0]
    lg = played[(played.competition == country) & (played.season == season)]
    if lg.empty:
        return None
    long = pd.concat([lg[["home_id", "home_goals", "away_goals"]].set_axis(["club", "gf", "ga"], axis=1),
                      lg[["away_id", "away_goals", "home_goals"]].set_axis(["club", "gf", "ga"], axis=1)])
    long["pts"] = np.select([long.gf > long.ga, long.gf == long.ga], [3, 1], 0)
    t = long.groupby("club").agg(played=("pts", "size"), pts=("pts", "sum"), gf=("gf", "sum"), ga=("ga", "sum"))
    t["gd"] = t.gf - t.ga
    t = t.sort_values(["pts", "gd", "gf"], ascending=False)
    if club not in t.index:
        return None
    last3 = last_results(lg, club, 3)
    return {"country": country, "league": LEAGUE_NAMES.get(country, country), "pos": t.index.get_loc(club) + 1,
            "of": len(t), **t.loc[club].astype(int).to_dict(), "last3": last3, "latest": lg.date.max()}


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def league_pos_label(played: pd.DataFrame, club: str, season: str) -> str:
    c = league_context(played, club, season)
    return f"{_ordinal(c['pos'])} {c['country']}" if c else "n/a"


def predict_date_ucl(day: pd.Timestamp, offline: bool = False, refresh_leagues: bool = False) -> pd.DataFrame | None:
    """All Champions League matches (incl. qualifiers) on a date; actual results shown if already played."""
    from src.predict import day_table

    if not offline:
        refresh_ucl(refresh_leagues)
    played, fixtures = load_euro_matches(), load_ucl_fixtures()
    euro = ["UCL", "UCLQ"]
    done = played[(played.date == day) & played.competition.isin(euro)]
    todo = fixtures[fixtures.date == day]
    if done.empty and todo.empty:
        return None
    before = played[played.date < day]
    day_games = pd.concat([done.assign(home_goals=np.nan, away_goals=np.nan), todo], ignore_index=True)
    u, model, _ = frame_and_model(before, day_games)
    rows = u[u.match_id.isin(day_games.match_id)]
    lam_h, lam_a = model.expected_goals(rows)
    grids = model.score_grid(rows)
    t = day_table(pd.DataFrame({"home": rows.home_raw.to_numpy(), "away": rows.away_raw.to_numpy()}),
                  model.predict_proba(rows), np.asarray(lam_h), np.asarray(lam_a), grids)
    season = rows.season.iloc[0]
    t.insert(1, "pos", [league_pos_label(before, c, season) for c in rows.home_id])
    t.insert(3, "pos ", [league_pos_label(before, c, season) for c in rows.away_id])
    through = []
    for i, (_, g) in enumerate(rows.iterrows()):
        role, first = tie_role(g, before, pd.concat([todo, fixtures[fixtures.date > day]]))
        through.append("" if role == "league" else
                       f"{p_home_side_through(model, rows.iloc[[i]], role, first, g.season):.0%}")
    if any(through):
        t["home through"] = through
    res = done.set_index("match_id")
    result = [f"{int(res.home_goals[m])}-{int(res.away_goals[m])}" if m in res.index else "" for m in rows.match_id]
    if any(result):
        t["result"] = result
    return t


def next_ucl_forecasts(club: str, n: int, today: pd.Timestamp) -> pd.DataFrame:
    """The club's next n Champions League fixtures, all forecast from today's data (cache only)."""
    played, fixtures = load_euro_matches(), load_ucl_fixtures()
    mine = fixtures[((fixtures.home_id == club) | (fixtures.away_id == club)) & (fixtures.date >= today)]
    mine = mine.sort_values("date").head(n)
    if mine.empty:
        return pd.DataFrame()
    u, model, _ = frame_and_model(played, fixtures[fixtures.date <= mine.date.max()])
    out = []
    for fx in mine.itertuples(index=False):
        fx = pd.Series(fx._asdict())
        row = u[u.match_id == fx.match_id]
        lh, la = model.expected_goals(row)
        role, first = tie_role(fx, played, fixtures)
        out.append({"date": fx.date, "comp": "UCL", "stage": fx.stage, "home": fx.home_raw, "away": fx.away_raw,
                    "proba": model.predict_proba(row)[0], "xg_home": float(lh[0]), "xg_away": float(la[0]),
                    "grid": model.score_grid(row)[0],
                    "home_through": None if role == "league" else p_home_side_through(model, row, role, first, fx.season)})
    return pd.DataFrame(out)


def last_results(played: pd.DataFrame, club: str, n: int = 5) -> pd.DataFrame:
    """The club's last n results in any competition, from its own point of view."""
    mine = played[(played.home_id == club) | (played.away_id == club)].sort_values("date").tail(n)
    rows = []
    for r in mine.itertuples():
        home = r.home_id == club
        gf, ga = (r.home_goals, r.away_goals) if home else (r.away_goals, r.home_goals)
        rows.append({"date": r.date.date(), "comp": r.competition, "venue": "N" if r.neutral else "H" if home else "A",
                     "opponent": r.away_raw if home else r.home_raw, "score": f"{int(gf)}-{int(ga)}",
                     "result": "W" if gf > ga else "D" if gf == ga else "L"})
    return pd.DataFrame(rows)


def predict_ucl(team_text: str, offline: bool = False, now: pd.Timestamp | None = None,
                refresh_leagues: bool = False) -> int:
    """Run the full Champions League prediction for one club. Returns a process exit code."""
    now = now or pd.Timestamp(datetime.now(timezone.utc))
    today = now.tz_localize(None).normalize() if now.tzinfo else now.normalize()
    calls = 0 if offline else refresh_ucl(refresh_leagues)
    played, fixtures = load_euro_matches(), load_ucl_fixtures()
    clubs = season_clubs(played, fixtures)
    try:
        club = resolve_club(team_text, clubs)
    except ClubNotFoundError as e:
        print(e)
        return 2
    name = clubs.set_index("club_id").name[club]

    mine = fixtures[(fixtures.home_id == club) | (fixtures.away_id == club)].sort_values("date")
    overdue, upcoming = mine[mine.date < today], mine[mine.date >= today]
    if len(overdue):
        print(f"WARNING: {len(overdue)} fixture(s) for {name} are past their date but have no result "
              "(postponed, or data not refreshed): "
              + "; ".join(f"{r.home_raw} v {r.away_raw} {r.date:%Y-%m-%d}" for r in overdue.itertuples()))
    if upcoming.empty:
        status = ("was eliminated in qualifying" if not clubs.set_index("club_id").in_main[club]
                  else "has no upcoming fixture in the published schedule")
        print(f"{name} {status} ({UCL_SEASON} Champions League). Not guessing.")
        return 1
    fixture = upcoming.iloc[0]
    before = played[played.date < fixture.date]
    fc = forecast(before, fixtures, fixture)
    pr, grid, r = fc["proba"], fc["grid"], fc["row"].iloc[0]
    top = np.dstack(np.unravel_index(np.argsort(grid, axis=None)[::-1][:3], grid.shape))[0]
    home, away = fixture.home_raw, fixture.away_raw

    print(f"\n{home} vs {away}{' (neutral venue)' if fixture.neutral else ''}")
    print(f"Champions League {fixture.season}, {fixture.stage}, {fixture.date:%a %d %b %Y}")
    countries = {fixture.home_id.split(':')[0], fixture.away_id.split(':')[0]}
    latest = ", ".join(f"{c} {before[before.competition == c].date.max():%Y-%m-%d}" for c in sorted(countries)
                       if (before.competition == c).any())
    print(f"Data: latest league results {latest or 'n/a'}; latest UCL result "
          f"{before[before.competition.isin(['UCL', 'UCLQ'])].date.max():%Y-%m-%d}; Highlightly calls this run: {calls}\n")

    width = max(len(home), len(away), 4) + 6
    print("90 minutes:")
    for label, prob in zip([f"{home} win", "Draw", f"{away} win"], pr):
        print(f"  {label:<{width}} {prob:6.1%}")
    print(f"\n  Expected goals: {home} {fc['xg_home']:.2f} - {fc['xg_away']:.2f} {away}")
    print("  Most likely scores: " + ", ".join(f"{i}-{j} ({grid[i, j]:.0%})" for i, j in top))
    league = lambda cid: LEAGUE_NAMES.get(cid.split(":")[0], cid.split(":")[0])  # noqa: E731
    print(f"  Rating (cross-league Elo): {home} {r.home_elo:.0f} ({league(fixture.home_id)}), "
          f"{away} {r.away_elo:.0f} ({league(fixture.away_id)})")

    if fc["role"] == "league":
        print("\n  League-phase match: qualification depends on the whole 36-team table (not modelled here).")
    else:
        ph = fc["p_home_through"]
        what = {"single": "single match: extra time, then penalties if level",
                "leg1": "first leg of two; return leg at " + away,
                "leg2_pending": "second leg; first leg at " + away + " not played yet",
                "leg2": "second leg; first leg {0} {1}-{2} {3}".format(
                    fc["first_leg"].home_raw, int(fc["first_leg"].home_goals), int(fc["first_leg"].away_goals),
                    fc["first_leg"].away_raw)}[fc["role"]]
        print(f"\nTo qualify ({what}):")
        print(f"  {home:<{width}} {ph:6.1%}\n  {away:<{width}} {1 - ph:6.1%}")

    print("\nDomestic league (context only, not a model input):")
    for cid, label in ((fixture.home_id, home), (fixture.away_id, away)):
        c = league_context(before, cid, fixture.season)
        if c is None:
            print(f"  {label}: league not covered by our data")
            continue
        l3 = ", ".join(f"{x.result} {x.score} {'v' if x.venue == 'H' else 'at'} {x.opponent}" for x in c["last3"].itertuples())
        print(f"  {label}: {_ordinal(c['pos'])} of {c['of']} in the {c['league']} ({c['pts']} pts from {c['played']}, "
              f"GD {c['gd']:+d}); last 3: {l3}")

    for cid, label in ((fixture.home_id, home), (fixture.away_id, away)):
        lr = last_results(before, cid)
        print(f"\nLast 5 results, {label} ({''.join(lr.result)}):")
        print(lr.to_string(index=False))
    h2h = before[before.competition.isin(["UCL", "UCLQ"])
                 & (((before.home_id == fixture.home_id) & (before.away_id == fixture.away_id))
                    | ((before.home_id == fixture.away_id) & (before.away_id == fixture.home_id)))]
    print(f"\nChampions League meetings since {FIRST_UCL_SEASON}: "
          + ("; ".join(f"{m.date:%Y-%m-%d} {m.home_raw} {int(m.home_goals)}-{int(m.away_goals)} {m.away_raw}"
                       for m in h2h.tail(6).itertuples()) or "none"))
    print("\nNote: based on past results only - no lineups, injuries or odds. Backtest 2019-20..2025-26: "
          "RPS 0.190, log loss 0.908 per match; treat as rough probabilities, not certainties.")
    return 0
