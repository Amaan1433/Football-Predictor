"""Tests for Champions League tie logic and features."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from src.euro_elo import EURO_CONFIG_PATH, EuroEloParams, load_euro_matches
from src.europe import EURO_MATCHES_PATH
from src.models import MAX_GOALS
from src.ucl import build_ucl_frame, identify_ties, p_through_single, p_through_two_legs

G = np.arange(MAX_GOALS + 1)


def _grid(lh: float, la: float) -> np.ndarray:
    g = np.outer(poisson.pmf(G, lh), poisson.pmf(G, la))
    return g / g.sum()


def test_even_tie_is_a_coin_flip():
    g = _grid(1.3, 1.3)
    assert p_through_two_legs(g, 1.3, 1.3, False, leg1=(0, 0)) == pytest.approx(0.5, abs=1e-6)
    assert p_through_two_legs(g, 1.3, 1.3, False, grid1=g) == pytest.approx(0.5, abs=1e-6)
    assert p_through_single(g, 1.3, 1.3) == pytest.approx(0.5, abs=1e-6)


def test_first_leg_lead_matters_and_is_monotonic():
    g = _grid(1.6, 1.0)
    probs = [p_through_two_legs(g, 1.6, 1.0, False, leg1=(a, 0)) for a in range(4)]
    assert all(x < y for x, y in zip(probs, probs[1:]))


@pytest.mark.parametrize("lead", [0, 1, 3])
def test_two_legs_match_independent_skellam_calculation(lead):
    """A leads by `lead` after leg 1; leg 2 at B: goal difference B - A ~ Skellam(1.6, 1.0)."""
    from scipy.stats import skellam
    lh, la = 1.6, 1.0
    d90, det = skellam(lh, la), skellam(lh / 3, la / 3)
    p_et_a = det.cdf(-1) + 0.5 * det.pmf(0)
    expected = d90.cdf(lead - 1) + d90.pmf(lead) * p_et_a
    got = p_through_two_legs(_grid(lh, la), lh, la, False, leg1=(lead, 0))
    assert got == pytest.approx(expected, abs=1e-6)


def test_away_goals_rule():
    # First leg 1-1 at A's home; second leg ends 0-0 almost surely.
    g = _grid(1e-6, 1e-6)
    assert p_through_two_legs(g, 1e-6, 1e-6, away_goals_rule=True, leg1=(1, 1)) == pytest.approx(0.0, abs=1e-4)
    assert p_through_two_legs(g, 1e-6, 1e-6, away_goals_rule=False, leg1=(1, 1)) == pytest.approx(0.5, abs=1e-4)


def test_probabilities_in_range():
    g = _grid(2.0, 0.7)
    for leg1 in [(0, 0), (2, 1), (0, 4)]:
        for rule in (True, False):
            assert 0 <= p_through_two_legs(g, 2.0, 0.7, rule, leg1=leg1) <= 1


@pytest.fixture(scope="module")
def ucl() -> pd.DataFrame:
    if not (EURO_MATCHES_PATH.exists() and EURO_CONFIG_PATH.exists()):
        pytest.skip("run src.europe and src.euro_elo first")
    p = EuroEloParams(**json.loads(EURO_CONFIG_PATH.read_text())["elo"])
    return build_ucl_frame(load_euro_matches(), p)


@pytest.fixture(scope="module")
def ties(ucl) -> pd.DataFrame:
    return identify_ties(ucl)


def test_tie_counts_match_the_format(ties):
    per_season = ties.groupby("season").size()
    assert per_season.drop(["2019-20", "2024-25", "2025-26"]).eq(15).all()  # 8 + 4 + 2 + final
    assert per_season["2019-20"] == 15 and ties[ties.season == "2019-20"].leg2.isna().sum() == 7  # Lisbon + final
    assert per_season[["2024-25", "2025-26"]].eq(23).all()                    # + 8 knockout play-offs
    assert (ties.winner.eq(ties.team_a) | ties.winner.eq(ties.team_b)).all()


@pytest.mark.parametrize("season, a, b, winner", [
    ("2016-17", "FRA:paris saint germain", "ESP:barcelona", "ESP:barcelona"),   # 4-0, 1-6
    ("2018-19", "ENG:tottenham hotspur", "NED:ajax", "ENG:tottenham hotspur"),  # 0-1, 3-2: away goals
    ("2015-16", "ESP:real madrid", "ESP:atletico madrid", "ESP:real madrid"),   # final on penalties
])
def test_known_tie_winners(ties, season, a, b, winner):
    t = ties[(ties.season == season) & (ties.team_a == a) & (ties.team_b == b)]
    assert len(t) == 1 and t.winner.iloc[0] == winner


def test_pre_tie_forecast_ignores_first_leg_score(ucl, ties):
    from src.ucl import tie_forecasts
    t = ties[ties.season == "2022-23"]
    base = tie_forecasts(ucl, ties, ["2022-23"])  # baselines are fitted on earlier seasons' ties
    u2 = ucl.copy()
    legs = u2.match_id.isin(t.leg1)
    u2.loc[legs, ["home_goals", "away_goals"]] = u2.loc[legs, ["away_goals", "home_goals"]].to_numpy() + [3, 0]
    after = tie_forecasts(u2, ties, ["2022-23"])
    np.testing.assert_allclose(base.model_pre, after.model_pre)
    assert not np.allclose(base.model_post.dropna(), after.model_post.dropna())


def test_league_form_uses_only_earlier_league_matches(ucl):
    from src.ucl import add_league_form
    m = load_euro_matches()
    m = m[(m.date >= "2016-07-01") & (m.date < "2017-07-01")].reset_index(drop=True)
    u = ucl[ucl.season == "2016-17"]
    cutoff = pd.Timestamp("2016-11-01")
    base = add_league_form(u, m)
    s = m.copy()
    mask = s.date >= cutoff
    s.loc[mask, ["home_goals", "away_goals"]] = np.random.default_rng(1).integers(0, 6, (mask.sum(), 2)).astype(float)
    after = add_league_form(u, s)
    cols = ["home_rank_frac", "away_rank_frac", "home_last3_ppg", "away_last3_ppg", "home_last3_gd"]
    keep = (base.date <= cutoff).to_numpy()
    pd.testing.assert_frame_equal(base.loc[keep, cols], after.loc[keep, cols])
    assert not base.loc[~keep, cols].equals(after.loc[~keep, cols])  # the scramble is detectable


def test_form_uses_only_earlier_matches(ucl):
    # Scrambling results on/after a date must not change form or Elo before/on that date.
    m = load_euro_matches()
    m = m[(m.date >= "2016-07-01") & (m.date < "2017-07-01")].reset_index(drop=True)
    p = EuroEloParams(**json.loads(EURO_CONFIG_PATH.read_text())["elo"])
    cutoff = pd.Timestamp("2017-02-14")
    base = build_ucl_frame(m, p)
    s = m.copy()
    mask = s.date >= cutoff
    s.loc[mask, ["home_goals", "away_goals"]] = np.random.default_rng(0).integers(0, 6, (mask.sum(), 2)).astype(float)
    after = build_ucl_frame(s, p)
    cols = ["home_elo", "away_elo", "home_form", "away_form"]
    keep = base.date <= cutoff
    pd.testing.assert_frame_equal(base.loc[keep, cols], after.loc[keep, cols])
