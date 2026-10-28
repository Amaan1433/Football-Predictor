"""Tests for the cross-league Elo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.euro_elo import EuroEloParams, load_euro_matches, run_elo
from src.europe import EURO_MATCHES_PATH


def _m(rows: list[tuple]) -> pd.DataFrame:
    """rows: (date, competition, season, home_id, away_id, hg, ag, neutral)."""
    df = pd.DataFrame(rows, columns=["date", "competition", "season", "home_id", "away_id",
                                     "home_goals", "away_goals", "neutral"])
    df["date"] = pd.to_datetime(df.date)
    df["home_goals"] = df.home_goals.astype(float)
    df["away_goals"] = df.away_goals.astype(float)
    df["match_id"] = [f"m{i}" for i in range(len(df))]
    return df


def test_domestic_matches_do_not_move_league_strength():
    p = EuroEloParams(k_league=10, season_regress=0)
    m = _m([("2020-08-01", "ENG", "2020-21", "ENG:a", "ENG:b", 3, 0, False),
            ("2020-08-08", "ENG", "2020-21", "ENG:b", "ENG:a", 0, 2, False)])
    _, final, _ = run_elo(m, p)
    assert np.mean([final["ENG:a"], final["ENG:b"]]) == pytest.approx(p.init)


def test_european_result_moves_both_leagues_and_is_zero_sum():
    p = EuroEloParams(k_league=10, season_regress=0, h_euro=0)
    m = _m([("2020-08-01", "ENG", "2020-21", "ENG:a", "ENG:b", 1, 1, False),
            ("2020-08-01", "ESP", "2020-21", "ESP:c", "ESP:d", 1, 1, False),
            ("2020-09-15", "UCL", "2020-21", "ENG:a", "ESP:c", 2, 0, False)])
    _, final, _ = run_elo(m, p)
    # ENG:b never played in Europe but its rating rises with the league offset
    assert final["ENG:b"] > p.init and final["ESP:d"] < p.init
    total = sum(final.values())
    assert total == pytest.approx(4 * p.init)


def test_neutral_venue_has_no_home_advantage():
    p = EuroEloParams(h_euro=100, season_regress=0)
    base = [("2020-08-01", "ENG", "2020-21", "ENG:a", "ENG:b", 1, 1, False),
            ("2020-08-01", "ESP", "2020-21", "ESP:c", "ESP:d", 1, 1, False)]
    home = run_elo(_m(base + [("2020-09-15", "UCL", "2020-21", "ENG:a", "ESP:c", 1, 1, False)]), p)[1]
    neutral = run_elo(_m(base + [("2020-09-15", "UCL", "2020-21", "ENG:a", "ESP:c", 1, 1, True)]), p)[1]
    assert home["ENG:a"] < neutral["ENG:a"]          # a home draw is a disappointment, a neutral one is not
    assert neutral["ENG:a"] == pytest.approx(neutral["ESP:c"])


def test_promoted_club_enters_at_relegated_rating():
    p = EuroEloParams(season_regress=0)
    m = _m([("2020-08-01", "ENG", "2020-21", "ENG:a", "ENG:b", 3, 0, False),
            ("2021-08-01", "ENG", "2021-22", "ENG:a", "ENG:c", 0, 0, False)])
    pre, _, _ = run_elo(m, p)
    pre = pre.set_index("match_id")
    b_final = 2 * p.init - pre.loc["m1", "home_elo"]  # zero-sum: b = 3000 - a
    assert pre.loc["m1", "away_elo"] == pytest.approx(b_final)


@pytest.mark.skipif(not EURO_MATCHES_PATH.exists(), reason="run `python -m src.europe` first")
def test_future_results_do_not_change_pre_match_ratings():
    m = load_euro_matches()
    m = m[(m.date >= "2016-07-01") & (m.date < "2018-07-01")].reset_index(drop=True)
    cutoff = pd.Timestamp("2017-03-07")  # a Champions League matchday
    assert ((m.date == cutoff) & (m.competition == "UCL")).any()
    base, _, _ = run_elo(m)
    scrambled = m.copy()
    rng = np.random.default_rng(0)
    mask = scrambled.date >= cutoff
    scrambled.loc[mask, "home_goals"] = rng.integers(0, 6, mask.sum()).astype(float)
    scrambled.loc[mask, "away_goals"] = rng.integers(0, 6, mask.sum()).astype(float)
    after, _, _ = run_elo(scrambled)
    upto = m.date <= cutoff
    pd.testing.assert_frame_equal(base[upto], after[upto])
    assert not base[m.date > cutoff + pd.Timedelta(days=7)].equals(after[m.date > cutoff + pd.Timedelta(days=7)])
