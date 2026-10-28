"""Tests for the Champions League CLI helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.predict_ucl import ClubNotFoundError, p_home_side_through, resolve_club, tie_role

CLUBS = pd.DataFrame({"club_id": ["GER:bayern munchen", "GER:bayer leverkusen", "ESP:atletico madrid", "ENG:arsenal"],
                      "name": ["Bayern Munich", "Bayer Leverkusen", "Atlético Madrid", "Arsenal"],
                      "in_main": True})


@pytest.mark.parametrize("text, cid", [
    ("bayern munich", "GER:bayern munchen"), ("Bayern München", "GER:bayern munchen"), ("Bayern", "GER:bayern munchen"),
    ("Atletico Madrid", "ESP:atletico madrid"), ("ARSENAL", "ENG:arsenal"),
])
def test_resolve_club(text, cid):
    assert resolve_club(text, CLUBS) == cid


def test_resolve_club_suggests():
    with pytest.raises(ClubNotFoundError, match="Arsenal"):
        resolve_club("Arsenl", CLUBS)


def _m(date, stage, h, a, hg=np.nan, ag=np.nan, comp="UCL"):
    return {"date": pd.Timestamp(date), "season": "2026-27", "competition": comp, "stage": stage,
            "home_id": h, "away_id": a, "home_goals": hg, "away_goals": ag}


def test_tie_roles():
    played = pd.DataFrame([_m("2027-02-17", "Finals, Round of 16", "A", "B", 2, 0)])
    fixtures = pd.DataFrame([_m("2027-03-10", "Finals, Round of 16", "B", "A"),
                             _m("2027-04-07", "Finals, Quarter-finals", "C", "D"),
                             _m("2027-04-14", "Finals, Quarter-finals", "D", "C"),
                             _m("2027-05-29", "Finals, Final", "E", "F"),
                             _m("2026-11-03", "League, Matchday 4", "A", "C")])
    roles = [tie_role(f, played, fixtures)[0] for _, f in fixtures.iterrows()]
    assert roles == ["leg2", "leg1", "leg2_pending", "single", "league"]
    assert tie_role(fixtures.iloc[0], played, fixtures)[1].home_goals == 2


def test_league_context_table_and_last3():
    from src.predict_ucl import _ordinal, league_context

    rows = [("2026-08-20", "ENG:a", "ENG:b", 2, 0), ("2026-08-20", "ENG:c", "ENG:d", 1, 1),
            ("2026-08-27", "ENG:b", "ENG:c", 0, 3), ("2026-08-27", "ENG:d", "ENG:a", 0, 0),
            ("2026-09-03", "ENG:a", "ENG:c", 1, 2)]
    played = pd.DataFrame([{"date": pd.Timestamp(d), "season": "2026-27", "competition": "ENG", "home_id": h,
                            "away_id": a, "home_raw": h, "away_raw": a, "home_goals": hg, "away_goals": ag,
                            "neutral": False} for d, h, a, hg, ag in rows])
    c = league_context(played, "ENG:c", "2026-27")
    assert (c["pos"], c["pts"], c["played"], c["gd"]) == (1, 7, 3, 4)
    assert "".join(c["last3"].result) == "DWW"
    assert league_context(played, "ENG:a", "2026-27")["pos"] == 2  # 4 pts, GD +1
    assert league_context(played, "GRE:x", "2026-27") is None
    assert [_ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22)] == \
        ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd"]


def test_past_date_forecast_ignores_that_days_results(monkeypatch):
    import src.predict as pr
    from src.build_tables import build_matches
    matches, fixtures = build_matches()
    day = matches[matches.season == "2025-26"].date.value_counts().idxmax()
    base = pr.predict_date_laliga(day, offline=True)
    flipped = matches.copy()
    on_day = flipped.date == day
    flipped.loc[on_day, ["home_goals", "away_goals"]] = flipped.loc[on_day, ["away_goals", "home_goals"]].to_numpy() + [5, 0]
    monkeypatch.setattr(pr, "build_matches", lambda: (flipped, fixtures))
    after = pr.predict_date_laliga(day, offline=True)
    cols = ["home win", "draw", "away win", "xG"]
    pd.testing.assert_frame_equal(base[cols], after[cols])
    assert not base.result.equals(after.result)


class _FlatModel:
    """Stand-in model: equal teams, 1.3 goals each, no venue effect."""

    def expected_goals(self, df):
        return np.array([1.3]), np.array([1.3])

    def score_grid(self, df):
        from scipy.stats import poisson
        g = np.outer(poisson.pmf(np.arange(11), 1.3), poisson.pmf(np.arange(11), 1.3))
        return (g / g.sum())[None]


class _EloModel(_FlatModel):
    """Stand-in model whose scoring rates depend on elo_diff and home advantage."""

    def expected_goals(self, df):
        d = float(df.elo_diff.iloc[0]) / 400 + (0 if float(df.neutral_f.iloc[0]) else 0.1)
        return np.array([1.3 * np.exp(d)]), np.array([1.3 * np.exp(-d)])

    def score_grid(self, df):
        from scipy.stats import poisson
        lh, la = (x[0] for x in self.expected_goals(df))
        g = np.outer(poisson.pmf(np.arange(11), lh), poisson.pmf(np.arange(11), la))
        return (g / g.sum())[None]


def test_both_legs_of_an_unplayed_tie_agree():
    m = _EloModel()
    leg1 = pd.DataFrame({"elo_diff": [120.0], "neutral_f": [0.0]})   # A (stronger) at home first
    leg2 = pd.DataFrame({"elo_diff": [-120.0], "neutral_f": [0.0]})  # B at home second
    a_through = p_home_side_through(m, leg1, "leg1", None, "2026-27")
    b_through = p_home_side_through(m, leg2, "leg2_pending", None, "2026-27")
    assert a_through > 0.5
    assert a_through + b_through == pytest.approx(1.0, abs=1e-9)


def test_qualify_probability_by_role():
    row = pd.DataFrame({"elo_diff": [0.0], "neutral_f": [0.0]})
    m = _FlatModel()
    assert p_home_side_through(m, row, "single", None, "2026-27") == pytest.approx(0.5, abs=1e-6)
    assert p_home_side_through(m, row, "leg1", None, "2026-27") == pytest.approx(0.5, abs=1e-6)
    first = pd.Series({"home_goals": 2, "away_goals": 0})  # this fixture's away side won leg 1 2-0
    assert p_home_side_through(m, row, "leg2", first, "2026-27") < 0.2
