"""Champions League + domestic-league tables for the cross-league model. Never calls an API.

euro_matches: one row per match (domestic leagues and UCL), 90-minute scores,
club ids from src/names_europe.py. Spain comes from our La Liga matches table; the
current UCL season comes from the cached Highlightly files (read with no network call).
"""

from __future__ import annotations

import argparse
import difflib
import itertools
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

from src.build_tables import MATCHES_PATH, PROCESSED_DIR
from src.ingest import CURRENT_SEASON_START, EURO_LEAGUES, OPENFOOTBALL_DIR, ROOT, season_label
from src.names_europe import COUNTRY_OVERRIDES, DISTINCT_CLUBS, SHARED_WORD_STOPLIST, ClubNameError, club_id

EURO_MATCHES_PATH = PROCESSED_DIR / "euro_matches.csv"
CLUBS_PATH = PROCESSED_DIR / "clubs.csv"
UCL_FIXTURES_PATH = PROCESSED_DIR / "ucl_fixtures.csv"
COVERED = ["ESP", *EURO_LEAGUES]  # countries with domestic league data
MAX_NEW_CLUBS_PER_SEASON = 4      # more new names than promotion explains => a spelling change

MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
DAY = r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)"
DATE_RE = re.compile(rf"^\s+{DAY} (\w{{3}}) (\d{{1,2}})(?: (\d{{4}}))?\s*$")
MATCH_RE = re.compile(r"^\s*(?:\d{1,2}:\d{2}\s+)?(?P<home>.+?) \((?P<hc>[A-Z]{3})\)\s+v\s+"
                      r"(?P<away>.+?) \((?P<ac>[A-Z]{3})\)\s*(?P<score>.*)$")
SCORE_RE = re.compile(r"^(?P<a>\d+)-(?P<b>\d+)(?P<pen>\s+pen\.\s+(?P<c>\d+)-(?P<d>\d+))?(?P<aet>\s+a\.e\.t\.)?"
                      r"(?:\s+\((?P<p1>\d+-\d+)(?:,\s*(?P<p2>\d+-\d+))?\))?\s*$")
LISBON_2020 = pd.Timestamp("2020-08-12")  # 2019-20 QF onward: single matches at neutral venues


def parse_score(text: str) -> dict | None:
    """90-minute score, extra-time flag and shoot-out winner from an openfootball score string."""
    text = text.strip()
    if not text:
        return None
    m = SCORE_RE.match(text)
    if not m:
        raise ValueError(f"Unrecognized score format: '{text}'")
    h_aet = a_aet = None
    if m["aet"]:
        if not m["p1"]:
            raise ValueError(f"Extra-time score without a 90-minute score: '{text}'")
        h90, a90 = map(int, m["p1"].split("-"))
        h_aet, a_aet = (int(m["c"]), int(m["d"])) if m["pen"] else (int(m["a"]), int(m["b"]))
    else:
        h90, a90 = int(m["a"]), int(m["b"])
    pens = ("home" if int(m["a"]) > int(m["b"]) else "away") if m["pen"] else None
    return {"home_goals": h90, "away_goals": a90, "aet": bool(m["aet"]), "pens_winner": pens,
            "home_goals_aet": h_aet, "away_goals_aet": a_aet}


def parse_ucl_file(path: Path, season: str, competition: str) -> pd.DataFrame:
    """Parse one openfootball Champions League text file."""
    rows, stage, cur = [], None, None
    start = int(season[:4])
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("▪"):
            stage = line.lstrip("▪ ").strip()
            continue
        d = DATE_RE.match(line)
        if d:
            month, day = MONTHS[d[1]], int(d[2])
            if d[3]:
                cur = date(int(d[3]), month, day)
            else:
                # Files list groups one after another, so dates can jump back (Dec -> Sep);
                # the year therefore comes from the month, not from the previous date.
                # Jun-Aug: qualifiers at the start of a file (first year) or late finals such
                # as Lisbon 2020 once the file has reached the second year.
                if month >= 9:
                    year = start
                elif month <= 5:
                    year = start + 1
                else:
                    year = start + 1 if cur is not None and cur.year == start + 1 else start
                cur = date(year, month, day)
            continue
        m = MATCH_RE.match(line)
        if not m:
            continue
        if cur is None:
            raise ValueError(f"{path.name}: match before any date: {line!r}")
        score = parse_score(m["score"])
        if score is None:
            continue  # not played (not expected in completed seasons)
        rows.append({"date": pd.Timestamp(cur), "season": season, "competition": competition, "stage": stage,
                     "home_raw": m["home"].strip(), "home_country": COUNTRY_OVERRIDES.get(m["hc"], m["hc"]),
                     "away_raw": m["away"].strip(), "away_country": COUNTRY_OVERRIDES.get(m["ac"], m["ac"]), **score})
    df = pd.DataFrame(rows)
    final = df.stage.str.split(",").str[-1].str.strip().eq("Final")
    df["neutral"] = final | ((df.season == "2019-20") & (df.date >= LISBON_2020))
    return df


def load_ucl() -> pd.DataFrame:
    """All cached Champions League files (main competition + qualifiers where available)."""
    frames = []
    for path in sorted((OPENFOOTBALL_DIR / "ucl").glob("*.txt")):
        season, kind = path.stem.split("_")
        frames.append(parse_ucl_file(path, season, "UCL" if kind == "cl" else "UCLQ"))
    return pd.concat(frames, ignore_index=True)


HIGHLIGHTLY_ET = ("Finished after extra time", "Finished after penalties")
HIGHLIGHTLY_GOALS = ("Goal", "Penalty", "Own Goal")  # an own goal is listed under the team it counts for


def _event_score(detail: dict, home: str, upto: int, shootout: bool) -> tuple[int, int]:
    """Goals from match events up to minute `upto` (stoppage time included), shoot-out kicks excluded."""
    h = a = 0
    for e in detail["events"]:
        base = int(e["time"].split("+")[0])
        if e["type"] not in HIGHLIGHTLY_GOALS or base > upto or (shootout and base >= 120 and "+" in e["time"]):
            continue
        if e["team"]["name"] == home:
            h += 1
        else:
            a += 1
    return h, a


def _highlightly_stage(rnd: str, when: pd.Timestamp) -> tuple[str, str]:
    """(competition, stage) in the openfootball style that src.ucl understands."""
    if rnd.startswith("League Stage"):
        return "UCL", f"League, Matchday {rnd.rsplit('-', 1)[1].strip()}"
    if "Qualifying" in rnd or (rnd == "Play-offs" and when.month in (7, 8)):
        return "UCLQ", rnd
    if rnd == "Play-offs":
        return "UCL", "Playoffs"  # knockout play-offs in February
    return "UCL", f"Finals, {rnd}"


def load_highlightly_ucl(start_year: int, known: set[str]) -> pd.DataFrame:
    """Champions League season from the cached Highlightly files: played matches and fixtures.

    The season listing has the after-extra-time score (the 90-minute one for shoot-outs), so
    extra-time matches take their 90-minute score from the cached match events, checked
    against the listed score and against the tie (level after 90 minutes of a second leg).
    """
    from src.api_client import cached
    from src.ingest import HIGHLIGHTLY_UCL_ID, highlightly_season_path
    from src.names_europe import highlightly_club_id

    rows = []
    for x in json.loads(highlightly_season_path(start_year, HIGHLIGHTLY_UCL_ID).read_text(encoding="utf-8")):
        when = pd.Timestamp(x["date"][:10])
        competition, stage = _highlightly_stage(x["round"], when)
        home, away = x["homeTeam"]["name"], x["awayTeam"]["name"]
        row = {"date": when, "season": season_label(start_year), "competition": competition, "stage": stage,
               "home_raw": home, "home_country": None, "away_raw": away, "away_country": None,
               "home_id": highlightly_club_id(home, known), "away_id": highlightly_club_id(away, known),
               "home_goals": None, "away_goals": None, "aet": False, "pens_winner": None,
               "home_goals_aet": None, "away_goals_aet": None, "neutral": stage == "Finals, Final"}
        row["home_country"], row["away_country"] = row["home_id"].split(":")[0], row["away_id"].split(":")[0]
        state = x["state"]
        if state["description"].startswith("Finished"):
            listed = tuple(int(v) for v in state["score"]["current"].split(" - "))
            row["home_goals"], row["away_goals"] = listed
            if state["description"] in HIGHLIGHTLY_ET:
                detail = cached("highlightly", f"/matches/{x['id']}")
                if detail is None:
                    raise FileNotFoundError(f"No cached details for extra-time match {x['id']}; run src.ingest --europe")
                detail = detail[0] if isinstance(detail, list) else detail
                pens = state["score"]["penalties"]
                s90 = _event_score(detail, home, 90, pens is not None)
                s120 = _event_score(detail, home, 120, pens is not None)
                if (s90 if pens else s120) != listed:
                    raise ValueError(f"{home} v {away}: events give {s90}/{s120} but the listing says {listed}")
                row.update(home_goals=s90[0], away_goals=s90[1], aet=True,
                           home_goals_aet=s120[0], away_goals_aet=s120[1])
                if pens:
                    ph, pa = (int(v) for v in pens.split(" - "))
                    row["pens_winner"] = "home" if ph > pa else "away"
        rows.append(row)
    df = pd.DataFrame(rows)
    _check_extra_time_ties(df)
    return df


def _check_extra_time_ties(df: pd.DataFrame) -> None:
    """Extra time only follows a level tie: aggregate level after 90 minutes of a second leg, or a drawn single match."""
    for r in df[df.aet].itertuples():
        leg1 = df[(df.stage == r.stage) & (df.home_id == r.away_id) & (df.away_id == r.home_id) & (df.date < r.date)]
        if len(leg1):
            level = leg1.home_goals.iloc[0] + r.away_goals == leg1.away_goals.iloc[0] + r.home_goals
        else:
            level = r.home_goals == r.away_goals
        if not level:
            raise ValueError(f"{r.home_raw} v {r.away_raw} ({r.date.date()}): went to extra time but the tie "
                             f"was not level after 90 minutes; check the 90-minute score")


def load_highlightly_domestic(country: str) -> pd.DataFrame | None:
    """Finished current-season league matches from the cached Highlightly file, or None if not fetched."""
    from src.ingest import HIGHLIGHTLY_DOMESTIC_IDS, highlightly_season_path

    path = highlightly_season_path(CURRENT_SEASON_START, HIGHLIGHTLY_DOMESTIC_IDS[country])
    if not path.exists():
        return None
    rows = []
    for x in json.loads(path.read_text(encoding="utf-8")):
        if x["state"]["description"] != "Finished":
            continue
        hg, ag = (int(v) for v in x["state"]["score"]["current"].split(" - "))
        rows.append({"date": pd.Timestamp(x["date"][:10]), "season": season_label(CURRENT_SEASON_START),
                     "competition": country, "stage": x["round"], "home_raw": x["homeTeam"]["name"],
                     "home_country": country, "away_raw": x["awayTeam"]["name"], "away_country": country,
                     "home_goals": hg, "away_goals": ag, "aet": False, "pens_winner": None,
                     "home_goals_aet": None, "away_goals_aet": None, "neutral": False})
    return pd.DataFrame(rows)


def load_domestic() -> pd.DataFrame:
    """Played matches from the cached openfootball league JSON files.

    For the current season, a fetched Highlightly file replaces openfootball when it has the
    later latest result (openfootball can lag by weeks; Highlightly is fetched only on request).
    """
    code_to_country = {v: k for k, v in EURO_LEAGUES.items()}
    current = season_label(CURRENT_SEASON_START)
    rows, frames = [], []
    for path in sorted((OPENFOOTBALL_DIR / "leagues").glob("*.json")):
        season, code = path.stem.split("_")
        if season == current:
            hl = load_highlightly_domestic(code_to_country[code])
            of_dates = [m["date"] for m in json.loads(path.read_text(encoding="utf-8"))["matches"] if m.get("score")]
            if hl is not None and len(hl) and (not of_dates or hl.date.max() > pd.Timestamp(max(of_dates))):
                frames.append(hl)
                continue
        for m in json.loads(path.read_text(encoding="utf-8"))["matches"]:
            score = m.get("score")
            ft = score.get("ft") if isinstance(score, dict) else score
            if not ft:
                continue
            cty = code_to_country[code]
            rows.append({"date": pd.Timestamp(m["date"]), "season": season, "competition": cty, "stage": m.get("round"),
                         "home_raw": m["team1"], "home_country": cty, "away_raw": m["team2"], "away_country": cty,
                         "home_goals": int(ft[0]), "away_goals": int(ft[1]), "aet": False, "pens_winner": None,
                         "home_goals_aet": None, "away_goals_aet": None, "neutral": False})
    return pd.concat([pd.DataFrame(rows), *frames], ignore_index=True)


def load_spain() -> pd.DataFrame:
    """Our La Liga matches (DataHub + Highlightly) in the same layout."""
    m = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
    return pd.DataFrame({"date": m.date, "season": m.season, "competition": "ESP", "stage": None,
                         "home_raw": m.home_team, "home_country": "ESP", "away_raw": m.away_team, "away_country": "ESP",
                         "home_goals": m.home_goals, "away_goals": m.away_goals, "aet": False, "pens_winner": None,
                         "home_goals_aet": None, "away_goals_aet": None, "neutral": False})


# ------------------------------------------------------------------ validation

def check_domestic_identities(dom: pd.DataFrame) -> None:
    """No two spellings collapse within a season; season-to-season churn looks like promotion only."""
    problems = []
    for (cty, season), g in dom.groupby(["competition", "season"]):
        names = pd.unique(g[["home_raw", "away_raw"]].to_numpy().ravel())
        ids = pd.Series({n: club_id(n, cty) for n in names})
        dup = ids[ids.duplicated(keep=False)]
        if len(dup):
            problems.append(f"{cty} {season}: different clubs share an id: {dup.to_dict()}")
    for cty, g in dom.groupby("competition"):
        by_season = {s: set(x.home_id) | set(x.away_id) for s, x in g.groupby("season")}
        seasons = sorted(by_season)
        for a, b in zip(seasons, seasons[1:]):
            new = by_season[b] - by_season[a]
            if len(new) > MAX_NEW_CLUBS_PER_SEASON:
                gone = by_season[a] - by_season[b]
                problems.append(f"{cty} {a}->{b}: {len(new)} new ids {sorted(new)} vs gone {sorted(gone)}")
        # A club split across two spellings within a season plays far fewer matches than its rivals.
        for s, x in g.groupby("season"):
            counts = pd.concat([x.home_id, x.away_id]).value_counts()
            short = counts[counts < counts.median() - 2]
            if len(short):
                problems.append(f"{cty} {s}: clubs with too few matches (split name?): {short.to_dict()}")
        # A rename at a season boundary: one id ends right where a similar id begins.
        first = {cid: min(s for s in seasons if cid in by_season[s]) for cid in set().union(*by_season.values())}
        last = {cid: max(s for s in seasons if cid in by_season[s]) for cid in first}
        for old, new_id in itertools.permutations(first, 2):
            if seasons.index(last[old]) + 1 != seasons.index(first[new_id]):
                continue
            shared = {t for t in set(old.split(":")[1].split()) & set(new_id.split(":")[1].split())
                      if len(t) >= 4 and t not in SHARED_WORD_STOPLIST}
            if shared and frozenset((old, new_id)) not in DISTINCT_CLUBS:
                problems.append(f"{cty}: '{old}' ends {last[old]} and '{new_id}' starts {first[new_id]} (rename?)")
    if problems:
        raise ClubNameError("Domestic name problems (add EU_ALIASES):\n  " + "\n  ".join(problems))


def check_ucl_identities(ucl: pd.DataFrame, dom: pd.DataFrame) -> None:
    """UCL clubs from covered countries must exist in that country's league data; flag near-duplicates elsewhere."""
    problems = []
    league_ids = dom.groupby("competition").apply(lambda g: set(g.home_id) | set(g.away_id), include_groups=False)
    for side in ("home", "away"):
        covered = ucl[ucl[f"{side}_country"].isin(COVERED)]
        for (cty, cid, raw), _ in covered.groupby([f"{side}_country", f"{side}_id", f"{side}_raw"]):
            if cid not in league_ids.get(cty, set()):
                close = difflib.get_close_matches(cid, sorted(league_ids.get(cty, set())), n=3, cutoff=0.5)
                problems.append(f"UCL '{raw}' ({cty}) -> {cid} not found in {cty} league data; close: {close}")
    ids = pd.unique(ucl[["home_id", "away_id"]].to_numpy().ravel())
    by_country: dict[str, list[str]] = {}
    for cid in ids:
        by_country.setdefault(cid.split(":")[0], []).append(cid.split(":", 1)[1])
    for cty, keys in by_country.items():
        if cty in COVERED:
            continue
        for i, a in enumerate(sorted(keys)):
            for b in sorted(keys)[i + 1:]:
                shared = {t for t in set(a.split()) & set(b.split()) if len(t) >= 4}
                similar = difflib.SequenceMatcher(None, a, b).ratio() > 0.8
                if (shared or similar) and frozenset((f"{cty}:{a}", f"{cty}:{b}")) not in DISTINCT_CLUBS:
                    problems.append(f"{cty}: possible duplicate club '{a}' vs '{b}'")
    if problems:
        raise ClubNameError("UCL name problems (add EU_ALIASES or src/names.py entries):\n  " + "\n  ".join(sorted(set(problems))))


def check_dates_in_season(m: pd.DataFrame) -> None:
    """Every match must fall between 1 June of the season's first year and 31 August of its second."""
    start = m.season.str[:4].astype(int)
    lo = pd.to_datetime(start.astype(str) + "-06-01")
    hi = pd.to_datetime((start + 1).astype(str) + "-08-31")
    bad = m[(m.date < lo) | (m.date > hi)]
    if len(bad):
        raise ValueError(f"{len(bad)} matches dated outside their season, e.g.:\n"
                         + bad[["season", "competition", "date", "home_raw", "away_raw"]].head(8).to_string(index=False))


# ------------------------------------------------------------------ build

def _with_match_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["date", "competition", "home_id"], kind="stable").reset_index(drop=True)
    out["match_id"] = (out.competition + "_" + out.date.dt.strftime("%Y%m%d") + "_" + out.home_id + "_" + out.away_id)
    if out.match_id.duplicated().any():
        raise ValueError(f"Duplicate match ids: {out.loc[out.match_id.duplicated(), 'match_id'].head().tolist()}")
    return out


def build_euro_matches() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine domestic leagues, Spain and the UCL into one validated table.

    Returns (played matches, unplayed current-season UCL fixtures).
    """
    dom = pd.concat([load_domestic(), load_spain()], ignore_index=True)
    ucl = load_ucl()
    for df in (dom, ucl):
        df["home_id"] = [club_id(n, c) for n, c in zip(df.home_raw, df.home_country)]
        df["away_id"] = [club_id(n, c) for n, c in zip(df.away_raw, df.away_country)]
    hl = load_highlightly_ucl(CURRENT_SEASON_START, set(dom.home_id) | set(dom.away_id) | set(ucl.home_id) | set(ucl.away_id))
    ucl = pd.concat([ucl, hl[hl.home_goals.notna()]], ignore_index=True)
    check_domestic_identities(dom)
    check_ucl_identities(ucl, dom)
    check_dates_in_season(pd.concat([dom, ucl, hl], ignore_index=True))
    return _with_match_id(pd.concat([dom, ucl], ignore_index=True)), _with_match_id(hl[hl.home_goals.isna()])


def clubs_table(m: pd.DataFrame) -> pd.DataFrame:
    """One row per club id: latest spelling, country, seasons seen, competitions."""
    long = pd.concat([
        m[["date", "season", "competition", "home_id", "home_raw", "home_country"]].set_axis(
            ["date", "season", "competition", "club_id", "name", "country"], axis=1),
        m[["date", "season", "competition", "away_id", "away_raw", "away_country"]].set_axis(
            ["date", "season", "competition", "club_id", "name", "country"], axis=1),
    ]).sort_values("date")
    g = long.groupby("club_id")
    return pd.DataFrame({"name": g.name.last(), "country": g.country.first(), "first_season": g.season.min(),
                         "last_season": g.season.max(), "ucl_matches": g.competition.apply(lambda s: s.isin(["UCL", "UCLQ"]).sum()),
                         "domestic_matches": g.competition.apply(lambda s: (~s.isin(["UCL", "UCLQ"])).sum())}).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build euro_matches and clubs tables.")
    parser.parse_args()
    m, fixtures = build_euro_matches()
    m.to_csv(EURO_MATCHES_PATH, index=False)
    fixtures.to_csv(UCL_FIXTURES_PATH, index=False)
    clubs = clubs_table(m)
    clubs.to_csv(CLUBS_PATH, index=False)
    print(f"Wrote {len(m)} matches to {EURO_MATCHES_PATH.relative_to(ROOT)}, {len(clubs)} clubs to {CLUBS_PATH.relative_to(ROOT)}\n")
    counts = m.pivot_table(index="season", columns="competition", values="match_id", aggfunc="count").fillna(0).astype(int)
    print(counts.to_string())
    ucl = m[m.competition == "UCL"]
    print(f"\nUCL: {len(ucl)} matches, {ucl.aet.sum()} went to extra time, {ucl.pens_winner.notna().sum()} to penalties, "
          f"{ucl.neutral.sum()} at neutral venues")
    print(f"Upcoming UCL fixtures: {len(fixtures)} -> {UCL_FIXTURES_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
