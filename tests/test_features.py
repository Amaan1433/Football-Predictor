"""Leakage and unit tests for src/features.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.build_tables import MATCHES_PATH
from src.features import EloParams, build_features, compute_elo, feature_columns, H2H_SHRINK_K

RESULT_COLS = ["home_goals", "away_goals", "home_shots", "away_shots", "home_sot", "away_sot"]


def _toy(rows: list[tuple[str, str, str, int, int]], season: str = "2000-01") -> pd.DataFrame:
    """Matches from (date, home, away, home_goals, away_goals); shots = 10 * goals + 1."""
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_goals", "away_goals"])
    df["date"] = pd.to_datetime(df.date)
    df["season"], df["competition"] = season, "LaLiga"
    df["match_id"] = [f"m{i}" for i in range(len(df))]
    for side in ("home", "away"):
        df[f"{side}_shots"] = df[f"{side}_goals"] * 10 + 1
        df[f"{side}_sot"] = df[f"{side}_goals"] + 1
        df[f"{side}_xg"] = np.nan
    for c in ("odds_h", "odds_d", "odds_a"):
        df[c] = np.nan
    return df


@pytest.fixture(scope="module")
def real_matches() -> pd.DataFrame:
    if not MATCHES_PATH.exists():
        pytest.skip("run src.ingest and src.build_tables first")
    m = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
    return m[(m.season >= "2012-13") & (m.season <= "2014-15")].reset_index(drop=True)


def _scramble(m: pd.DataFrame, mask: pd.Series, seed: int = 0) -> pd.DataFrame:
    """Replace every result/stat column on masked rows with random values."""
    rng = np.random.default_rng(seed)
    out = m.copy()
    for c in RESULT_COLS:
        out.loc[mask, c] = rng.integers(0, 8, mask.sum()).astype(float)
    return out


def _features_on(f: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    return f.set_index("match_id").loc[ids, feature_columns(f)]


# ------------------------------------------------------------------ leakage

@pytest.mark.parametrize("cutoff", ["2013-10-19", "2014-01-11", "2014-08-23", "2015-05-23"])
def test_future_and_same_day_results_do_not_change_features(real_matches, cutoff):
    """Scramble every result on/after the cutoff: features up to and on the cutoff date must not move."""
    d = pd.Timestamp(cutoff)
    assert (real_matches.date == d).sum() >= 2, "cutoff should be a date with several matches"
    base = build_features(real_matches)
    scrambled = build_features(_scramble(real_matches, real_matches.date >= d))
    ids = real_matches.loc[real_matches.date <= d, "match_id"]
    pd.testing.assert_frame_equal(_features_on(base, ids), _features_on(scrambled, ids))


def test_same_day_only_scramble(real_matches):
    """Changing only matches played on day D leaves every day-D feature unchanged."""
    d = pd.Timestamp("2014-01-11")
    base = build_features(real_matches)
    scrambled = build_features(_scramble(real_matches, real_matches.date == d, seed=1))
    ids = real_matches.loc[real_matches.date <= d, "match_id"]
    pd.testing.assert_frame_equal(_features_on(base, ids), _features_on(scrambled, ids))


def test_scramble_is_detectable(real_matches):
    """Sanity check on the test itself: scrambling earlier results DOES change later features."""
    d = pd.Timestamp("2014-01-11")
    base = build_features(real_matches)
    scrambled = build_features(_scramble(real_matches, real_matches.date < d))
    ids = real_matches.loc[real_matches.date == d, "match_id"]
    assert not _features_on(base, ids).equals(_features_on(scrambled, ids))


def test_fixture_path_matches_historical_features(real_matches):
    """A match fed as an unplayed fixture gets the same features it got in the full history."""
    d = pd.Timestamp("2014-01-11")
    target = real_matches[real_matches.date == d].iloc[[0]]
    history = real_matches[real_matches.date < d]
    fixture = target.copy()
    fixture[RESULT_COLS] = np.nan
    via_fixture = build_features(history, fixture)
    full = build_features(real_matches)
    pd.testing.assert_frame_equal(_features_on(full, target.match_id), _features_on(via_fixture, target.match_id))


def test_no_target_columns_in_features(real_matches):
    f = build_features(real_matches.head(50))
    cols = feature_columns(f)
    assert not [c for c in cols if "target" in c or c in RESULT_COLS]


# ------------------------------------------------------------------ units

def test_first_match_has_no_form_and_second_uses_first():
    m = _toy([("2000-09-01", "A", "B", 2, 0), ("2000-09-08", "B", "A", 1, 1)])
    f = build_features(m).set_index("match_id")
    assert np.isnan(f.loc["m0", "home_ewm10_points"])
    assert f.loc["m1", "away_ewm10_points"] == 3.0     # A won its only previous match
    assert f.loc["m1", "home_last5_gd"] == -2.0        # B lost 0-2
    assert f.loc["m1", "home_rest_days"] == 7
    assert f.loc["m1", "home_matches_14d"] == 1


def test_ewm_matches_hand_calculation():
    m = _toy([("2000-09-01", "A", "B", 1, 0), ("2000-09-08", "B", "A", 0, 0), ("2000-09-15", "A", "B", 0, 0)])
    f = build_features(m).set_index("match_id")
    # A's points before m2: [3, 1]; half-life 10 -> weights (0.5**0.1, 1)
    w = 0.5 ** (1 / 10)
    assert f.loc["m2", "home_ewm10_points"] == pytest.approx((3 * w + 1) / (w + 1))


def test_elo_single_match_hand_calculation():
    p = EloParams(k=20, home_adv=65, season_regress=0.0, init=1500)
    m = _toy([("2000-09-01", "A", "B", 3, 0), ("2000-09-08", "B", "A", 0, 0)])
    elo = compute_elo(m, p).set_index("match_id")
    exp_home = 1 / (1 + 10 ** (-65 / 400))
    delta = 20 * (11 + 3) / 8 * (1 - exp_home)
    assert elo.loc["m0", "home_elo"] == 1500 and elo.loc["m0", "away_elo"] == 1500
    assert elo.loc["m1", "away_elo"] == pytest.approx(1500 + delta)  # A, now away
    assert elo.loc["m1", "home_elo"] == pytest.approx(1500 - delta)


def test_promoted_team_enters_at_relegated_mean():
    p = EloParams(season_regress=0.0)
    s1 = _toy([("2000-09-01", "A", "B", 2, 0), ("2000-09-08", "B", "A", 0, 1)], season="2000-01")
    s2 = _toy([("2001-09-01", "A", "C", 1, 1)], season="2001-02")
    s2["match_id"] = ["n0"]
    elo = compute_elo(pd.concat([s1, s2], ignore_index=True), p).set_index("match_id")
    # Two teams, zero-sum updates: B's final rating = 3000 - A's final rating.
    a_final = elo.loc["n0", "home_elo"]  # no regression, so A carries its final rating
    assert elo.loc["n0", "away_elo"] == pytest.approx(3000 - a_final)  # C enters at relegated B's rating
    assert elo.loc["n0", "away_elo"] < elo.loc["m1", "home_elo"]       # B lost m1, so ended lower


def test_h2h_is_shrunk_toward_zero():
    m = _toy([("2000-09-01", "A", "B", 2, 0), ("2000-09-08", "A", "B", 0, 0)])
    f = build_features(m).set_index("match_id")
    assert f.loc["m0", "h2h_gd"] == 0 and f.loc["m0", "h2h_n"] == 0
    assert f.loc["m1", "h2h_gd"] == pytest.approx(2 / (1 + H2H_SHRINK_K))
    assert f.loc["m1", "h2h_pts_diff"] == pytest.approx(3 / (1 + H2H_SHRINK_K))


def test_fixture_checks_reject_two_fixtures_per_team():
    m = _toy([("2000-09-01", "A", "B", 2, 0)])
    fx = _toy([("2000-09-08", "A", "C", 0, 0), ("2000-09-15", "D", "A", 0, 0)])
    fx[RESULT_COLS] = np.nan
    with pytest.raises(ValueError, match="more than one fixture"):
        build_features(m, fx)
