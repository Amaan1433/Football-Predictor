"""Tests for Champions League parsing and cross-league club identities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.europe import EURO_MATCHES_PATH, _check_extra_time_ties, _event_score, parse_score, parse_ucl_file
from src.ingest import CURRENT_SEASON_START, season_label
from src.names_europe import ClubNameError, club_id, highlightly_club_id, normalize


def _ev(time: str, kind: str, team: str) -> dict:
    return {"time": time, "type": kind, "team": {"name": team}}


def test_highlightly_event_score_counts_own_goals_and_skips_shootout():
    detail = {"events": [_ev("21", "Own Goal", "A"), _ev("34", "Goal", "B"), _ev("90+3", "Penalty", "A"),
                         _ev("97", "Penalty", "B"), _ev("100", "Missed Penalty", "A"),
                         _ev("120+1", "Penalty", "A"), _ev("120+2", "Penalty", "B")]}
    assert _event_score(detail, "A", 90, shootout=True) == (2, 1)
    assert _event_score(detail, "A", 120, shootout=True) == (2, 2)
    assert _event_score(detail, "A", 120, shootout=False) == (3, 3)  # 120+ goals count in plain extra time


def test_extra_time_needs_a_level_tie():
    base = {"stage": "Play-offs", "aet": False, "home_raw": "x", "away_raw": "y"}
    leg1 = {**base, "date": pd.Timestamp("2026-08-19"), "home_id": "B", "away_id": "A", "home_goals": 3, "away_goals": 0}
    leg2 = {**base, "date": pd.Timestamp("2026-08-26"), "home_id": "A", "away_id": "B", "home_goals": 4, "away_goals": 1,
            "aet": True}
    _check_extra_time_ties(pd.DataFrame([leg1, leg2]))  # 4-4 on aggregate: fine
    with pytest.raises(ValueError, match="not level"):
        _check_extra_time_ties(pd.DataFrame([leg1, {**leg2, "home_goals": 3}]))


def test_highlightly_names_must_resolve_to_one_club():
    known = {"GER:bayern munchen", "ENG:arsenal"}
    assert highlightly_club_id("Bayern Munich", known) == "GER:bayern munchen"
    assert highlightly_club_id("Arsenal", known) == "ENG:arsenal"
    with pytest.raises(ClubNameError):
        highlightly_club_id("Unknown Rovers", known)


@pytest.mark.parametrize("text, expected", [
    ("2-1 (1-1)", (2, 1, False, None)),
    ("0-0", (0, 0, False, None)),
    ("4-2 a.e.t. (2-2, 0-2)", (2, 2, True, None)),                 # 90-minute score is 2-2
    ("5-3 pen. 1-1 a.e.t. (1-1, 1-0)", (1, 1, True, "home")),
    ("3-4 pen. 0-0 a.e.t. (0-0)", (0, 0, True, "away")),
])
def test_parse_score_returns_90_minute_result(text, expected):
    s = parse_score(text)
    assert (s["home_goals"], s["away_goals"], s["aet"], s["pens_winner"]) == expected


def test_parse_score_keeps_extra_time_totals():
    s = parse_score("4-2 a.e.t. (2-2, 0-2)")
    assert (s["home_goals_aet"], s["away_goals_aet"]) == (4, 2)
    s = parse_score("5-3 pen. 1-1 a.e.t. (1-1, 1-0)")
    assert (s["home_goals_aet"], s["away_goals_aet"]) == (1, 1)
    assert parse_score("2-1 (1-1)")["home_goals_aet"] is None


def test_parse_score_rejects_unknown_formats():
    assert parse_score("") is None
    with pytest.raises(ValueError):
        parse_score("2-1 abandoned")
    with pytest.raises(ValueError):
        parse_score("3-2 a.e.t.")  # no 90-minute score available


def test_parse_ucl_file_dates_stages_and_neutral_venues(tmp_path: Path):
    text = """= UEFA Champions League 2019/20

▪ Group A
  Tue Dec 10 2019
    21:00  Real Madrid CF (ESP)     v Club Brugge KV (BEL)    3-1 (1-0)
  Wed Jan 15
    21:00  AS Monaco FC (MCO)       v FC Porto (POR)          0-0
▪ Quarterfinals
  Wed Aug 12
    21:00  Atalanta (ITA)          v Paris Saint-Germain (FRA)  1-2 (1-0)
▪ Final
  Sun Aug 23
    21:00  Paris Saint-Germain (FRA) v Bayern München (GER)  0-1 (0-0)
"""
    p = tmp_path / "cl.txt"
    p.write_text(text, encoding="utf-8")
    df = parse_ucl_file(p, "2019-20", "UCL")
    assert list(df.date.dt.strftime("%Y-%m-%d")) == ["2019-12-10", "2020-01-15", "2020-08-12", "2020-08-23"]
    assert list(df.neutral) == [False, False, True, True]     # Lisbon 2020 and the final
    assert df.home_country.iloc[1] == "FRA"                   # Monaco plays in Ligue 1
    assert df.stage.iloc[0] == "Group A"


def test_parse_ucl_file_group_blocks_restart_in_september(tmp_path: Path):
    """Files list groups one after another: Group B's dates jump back to September without a year."""
    text = """▪ Group A
  Wed Sep 14 2011
    20:45  Manchester City (ENG)   v SSC Napoli (ITA)         1-1 (0-0)
  Tue Dec 6
    20:45  Manchester City (ENG)   v Bayern München (GER)     2-0 (0-0)
▪ Group B
  Wed Sep 14
    20:45  Lille OSC (FRA)         v CSKA Moskva (RUS)        2-2 (1-0)
▪ Round of 16
  Tue Feb 14
    20:45  Olympique Lyonnais (FRA) v APOEL Nikosia (CYP)     1-0 (1-0)
"""
    p = tmp_path / "cl.txt"
    p.write_text(text, encoding="utf-8")
    df = parse_ucl_file(p, "2011-12", "UCL")
    assert list(df.date.dt.strftime("%Y-%m-%d")) == ["2011-09-14", "2011-12-06", "2011-09-14", "2012-02-14"]


@pytest.mark.parametrize("a, b, country", [
    ("Manchester City", "Manchester City FC", "ENG"),
    ("TSG 1899 Hoffenheim", "TSG Hoffenheim", "GER"),
    ("Inter", "FC Internazionale Milano", "ITA"),
    ("SL Benfica", "Sport Lisboa e Benfica", "POR"),
    ("Real Madrid", "Real Madrid CF", "ESP"),
    ("Atlético Madrid", "Club Atlético de Madrid", "ESP"),
])
def test_spellings_map_to_one_club(a, b, country):
    assert club_id(a, country) == club_id(b, country)


def test_different_clubs_stay_different():
    assert club_id("Slavia Praha", "CZE") != club_id("Sparta Praha", "CZE")
    assert club_id("Real Madrid", "ESP") != club_id("Atlético Madrid", "ESP")
    assert club_id("Celtic FC", "SCO") != club_id("Celtic FC", "IRL")  # country is part of the id


def test_unknown_spanish_club_fails_loudly():
    with pytest.raises(ClubNameError):
        club_id("Not A Real Club CF", "ESP")


def test_normalize():
    assert normalize("FC Bayern München") == "bayern munchen"
    assert normalize("FK Bodø/Glimt") == "bodo glimt"


# ------------------------------------------------------------------ built table

@pytest.fixture(scope="module")
def euro() -> pd.DataFrame:
    if not EURO_MATCHES_PATH.exists():
        pytest.skip("run `python -m src.europe` first")
    return pd.read_csv(EURO_MATCHES_PATH, parse_dates=["date"])


def test_shootouts_follow_extra_time_and_single_matches_were_level(euro):
    pens = euro[euro.pens_winner.notna()]
    assert len(pens) > 0 and pens.aet.all()
    # Second legs can be decided on aggregate (e.g. 2-1 after a 0-1 first leg), so only
    # single-match ties (finals, Lisbon 2020) must be level after 90 minutes.
    single = pens[pens.neutral]
    assert len(single) > 0 and (single.home_goals == single.away_goals).all()


def test_one_neutral_final_per_season(euro):
    ucl = euro[(euro.competition == "UCL") & (euro.season < season_label(CURRENT_SEASON_START))]  # completed seasons
    finals = ucl[ucl.stage.str.split(",").str[-1].str.strip().eq("Final")]
    assert finals.groupby("season").size().eq(1).all() and finals.neutral.all()
    assert ucl.season.nunique() == finals.season.nunique()


def test_all_matches_dated_within_their_season(euro):
    start = euro.season.str[:4].astype(int)
    assert (euro.date >= pd.to_datetime(start.astype(str) + "-06-01")).all()
    assert (euro.date <= pd.to_datetime((start + 1).astype(str) + "-08-31")).all()


def test_ucl_season_sizes(euro):
    sizes = euro[euro.competition == "UCL"].groupby("season").size()
    current = season_label(CURRENT_SEASON_START)
    assert sizes[current] % 18 == 0 and sizes[current] <= 144             # in progress: whole league-phase matchdays
    sizes = sizes.drop(current)
    assert sizes.drop(["2019-20", "2024-25", "2025-26"]).eq(125).all()   # 32-team format
    assert sizes["2019-20"] == 119                                        # single-leg Lisbon finals
    assert sizes[["2024-25", "2025-26"]].eq(189).all()                    # 36-team league phase
