"""Leakage-safe features, one row per match.

Rule: every feature for a match uses only matches dated strictly BEFORE that
match's date. Team-level stats are computed on each team's date-sorted history
with shift(1) before any rolling/ewm step. A team never plays twice on one date
(enforced in build_tables), so shift(1) excludes the current match, every
later match and every other match played on the same date.

Upcoming fixtures (NaN goals) can be appended so they get features through the
same code path, as long as each team has at most one unplayed fixture and it
comes after all of that team's played matches.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.build_tables import MATCHES_PATH, PROCESSED_DIR, build_team_match
from src.ingest import ROOT

FEATURES_PATH = PROCESSED_DIR / "features.csv"

FORM_METRICS = ["points", "gd", "gf", "ga", "shots_for", "shots_against", "sot_for", "sot_against"]
SPLIT_METRICS = ["gf", "ga", "sot_for", "sot_against"]
HALF_LIVES = (5, 10)
LAST_N = 5
SPLIT_HALF_LIFE = 10
REST_DAYS_CAP = 30      # rest_days = 30 means "30 or more" (season breaks, promotions)
H2H_LAST_N = 6          # most recent meetings used
H2H_SHRINK_K = 3.0      # h2h value = sum / (n + k): pulls toward 0 when n is small


@dataclass(frozen=True)
class EloParams:
    """Elo settings. Tuned later on training seasons only."""
    k: float = 20.0
    home_adv: float = 65.0
    season_regress: float = 0.2   # fraction pulled toward league mean between seasons
    init: float = 1500.0


# ---------------------------------------------------------------- team level

def _within(tm: pd.DataFrame, keys: list[str], col: str, fn) -> pd.Series:
    """Apply fn to each group's history shifted by one (previous matches only)."""
    prev = tm.groupby(keys, sort=False)[col].shift(1)
    return prev.groupby([tm[k] for k in keys], sort=False).transform(fn)


def add_form(tm: pd.DataFrame) -> pd.DataFrame:
    """EWM (half-lives 5, 10) and last-5 means of each form metric, plus match counts."""
    for col in FORM_METRICS:
        for h in HALF_LIVES:
            tm[f"ewm{h}_{col}"] = _within(tm, ["team"], col, lambda s, h=h: s.ewm(halflife=h, ignore_na=True).mean())
        tm[f"last{LAST_N}_{col}"] = _within(tm, ["team"], col, lambda s: s.rolling(LAST_N, min_periods=1).mean())
    tm["n_prev"] = tm.groupby("team").cumcount()
    tm["n_season"] = tm.groupby(["team", "season"]).cumcount()
    return tm


def add_venue_splits(tm: pd.DataFrame) -> pd.DataFrame:
    """Form at the same venue type only (home form for home rows, away form for away rows)."""
    for col in SPLIT_METRICS:
        tm[f"venue_ewm_{col}"] = _within(
            tm, ["team", "is_home"], col, lambda s: s.ewm(halflife=SPLIT_HALF_LIFE, ignore_na=True).mean()
        )
    return tm


def add_schedule(tm: pd.DataFrame) -> pd.DataFrame:
    """Rest days since previous match and number of matches in the previous 14 days."""
    prev_date = tm.groupby("team").date.shift(1)
    tm["rest_days"] = (tm.date - prev_date).dt.days.clip(upper=REST_DAYS_CAP)
    counts = np.zeros(len(tm))
    for idx in tm.groupby("team").indices.values():
        d = tm.date.to_numpy()[idx]
        lo = np.searchsorted(d, d - np.timedelta64(14, "D"), side="left")
        hi = np.searchsorted(d, d, side="left")  # strictly before this date
        counts[idx] = hi - lo
    tm["matches_14d"] = counts
    return tm


def add_h2h(tm: pd.DataFrame) -> pd.DataFrame:
    """Shrunk head-to-head points difference and goal difference vs this opponent."""
    tm["pts_diff"] = np.sign(tm.gd) * 3  # +3 win, 0 draw, -3 loss; NaN if unplayed
    keys = ["team", "opp"]
    n = _within(tm, keys, "gd", lambda s: s.notna().astype(float).rolling(H2H_LAST_N, min_periods=1).sum())
    n = n.fillna(0)  # first meeting: no history
    for col in ["pts_diff", "gd"]:
        total = _within(tm, keys, col, lambda s: s.rolling(H2H_LAST_N, min_periods=1).sum()).fillna(0)
        tm[f"h2h_{col}"] = total / (n + H2H_SHRINK_K)
    tm["h2h_n"] = n
    return tm


# ---------------------------------------------------------------- Elo

def _goal_mult(margin: float) -> float:
    """World Football Elo goal-difference multiplier."""
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11 + margin) / 8


def compute_elo(m: pd.DataFrame, p: EloParams = EloParams()) -> pd.DataFrame:
    """Pre-match Elo for every row of m (sorted by date). Unplayed rows don't update ratings.

    Between seasons: ratings regress toward the league mean; newly promoted teams
    enter at the mean final rating of the teams that went down (or of the 3 lowest
    teams if nobody went down).
    """
    ratings: dict[str, float] = {}
    prev_teams: set[str] = set()
    pre_h, pre_a = np.empty(len(m)), np.empty(len(m))
    pos = {mid: i for i, mid in enumerate(m.match_id)}

    for _, g in m.groupby("season", sort=True):
        teams = set(g.home_team) | set(g.away_team)
        if not prev_teams:
            ratings.update({t: p.init for t in teams})
        else:
            final = {t: ratings[t] for t in prev_teams}
            down = prev_teams - teams
            entry = np.mean([final[t] for t in down]) if down else np.mean(sorted(final.values())[:3])
            mean = np.mean(list(final.values()))
            for t in prev_teams & teams:
                ratings[t] = mean + (ratings[t] - mean) * (1 - p.season_regress)
            for t in teams - prev_teams:
                ratings[t] = entry
        prev_teams = teams

        for r in g.itertuples(index=False):
            i = pos[r.match_id]
            rh, ra = ratings[r.home_team], ratings[r.away_team]
            pre_h[i], pre_a[i] = rh, ra
            if np.isnan(r.home_goals):
                continue
            exp_h = 1 / (1 + 10 ** (-(rh + p.home_adv - ra) / 400))
            score = 1.0 if r.home_goals > r.away_goals else 0.5 if r.home_goals == r.away_goals else 0.0
            delta = p.k * _goal_mult(abs(r.home_goals - r.away_goals)) * (score - exp_h)
            ratings[r.home_team] = rh + delta
            ratings[r.away_team] = ra - delta

    out = pd.DataFrame({"match_id": m.match_id.to_numpy(), "home_elo": pre_h, "away_elo": pre_a})
    out["elo_diff"] = out.home_elo - out.away_elo
    out["elo_exp_home"] = 1 / (1 + 10 ** (-(out.elo_diff + p.home_adv) / 400))
    return out


# ---------------------------------------------------------------- match level

TEAM_FEATURES = (
    [f"ewm{h}_{c}" for c in FORM_METRICS for h in HALF_LIVES]
    + [f"last{LAST_N}_{c}" for c in FORM_METRICS]
    + [f"venue_ewm_{c}" for c in SPLIT_METRICS]
    + ["rest_days", "matches_14d", "n_prev", "n_season"]
)


def _check_fixtures(played: pd.DataFrame, fixtures: pd.DataFrame) -> None:
    """Each team: at most one fixture, dated after its last played match."""
    teams = pd.concat([fixtures.home_team, fixtures.away_team])
    if teams.duplicated().any():
        raise ValueError(f"Teams with more than one fixture: {sorted(set(teams[teams.duplicated()]))}")
    last = pd.concat([played[["home_team", "date"]].set_axis(["team", "date"], axis=1),
                      played[["away_team", "date"]].set_axis(["team", "date"], axis=1)]).groupby("team").date.max()
    for r in fixtures.itertuples():
        for t in (r.home_team, r.away_team):
            if t in last.index and r.date <= last[t]:
                raise ValueError(f"Fixture {r.match_id} is not after {t}'s last played match")


def build_features(matches: pd.DataFrame, fixtures: pd.DataFrame | None = None,
                   elo: EloParams = EloParams()) -> pd.DataFrame:
    """One row per match (and per fixture): home_*, away_*, diff_* features and target_* columns."""
    m = matches.copy()
    if fixtures is not None and len(fixtures):
        _check_fixtures(matches, fixtures)
        fx = fixtures.reindex(columns=matches.columns)
        m = pd.concat([m, fx], ignore_index=True)
    m["date"] = pd.to_datetime(m.date)
    for c in ["home_goals", "away_goals"]:
        m[c] = m[c].astype(float)
    m = m.sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)

    tm = build_team_match(m, validate=False)
    tm["gd"] = tm.gf - tm.ga
    tm = tm.sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)
    for step in (add_form, add_venue_splits, add_schedule, add_h2h):
        tm = step(tm)

    home = tm[tm.is_home == 1].set_index("match_id")
    away = tm[tm.is_home == 0].set_index("match_id")
    f = m[["match_id", "date", "season", "competition", "home_team", "away_team"]].set_index("match_id")
    f = f.join(home[TEAM_FEATURES].add_prefix("home_")).join(away[TEAM_FEATURES].add_prefix("away_"))
    diffs = pd.DataFrame({f"diff_{c}": f[f"home_{c}"] - f[f"away_{c}"] for c in TEAM_FEATURES})
    f = pd.concat([f, diffs], axis=1)
    f = f.join(home[["h2h_pts_diff", "h2h_gd", "h2h_n"]])  # from the home team's perspective
    f["matchweek"] = (f.home_n_season + f.away_n_season) / 2 + 1
    f = f.join(compute_elo(m, elo).set_index("match_id"))

    played = m.set_index("match_id")
    f["target_home_goals"] = played.home_goals
    f["target_away_goals"] = played.away_goals
    result = np.select([played.home_goals > played.away_goals, played.home_goals == played.away_goals,
                        played.home_goals < played.away_goals], ["H", "D", "A"], "")
    f["target_result"] = pd.Series(result, index=played.index).replace("", np.nan)
    return f.reset_index()


def feature_columns(f: pd.DataFrame) -> list[str]:
    """Model inputs: everything except ids, labels and targets."""
    meta = {"match_id", "date", "season", "competition", "home_team", "away_team"}
    return [c for c in f.columns if c not in meta and not c.startswith("target_")]


# ---------------------------------------------------------------- form comparison

TRAIN_SEASONS = ("2006-07", "2023-24")  # inclusive; shots exist from 2005-06, first season is warm-up
VALID_SEASON = "2024-25"                # the test season (2025-26) is never touched here


def compare_form_variants(f: pd.DataFrame) -> pd.DataFrame:
    """Validation log loss of a small multinomial LR on each form variant's diff features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    played = f[f.target_result.notna()]
    train = played[(played.season >= TRAIN_SEASONS[0]) & (played.season <= TRAIN_SEASONS[1])]
    valid = played[played.season == VALID_SEASON]
    base = ["points", "gd", "sot_for", "sot_against"]
    variants = {f"ewm half-life {h}": f"ewm{h}_" for h in HALF_LIVES} | {f"last {LAST_N}": f"last{LAST_N}_"}
    rows = []
    for name, prefix in variants.items():
        cols = [f"diff_{prefix}{c}" for c in base]
        tr, va = train.dropna(subset=cols), valid.dropna(subset=cols)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(tr[cols], tr.target_result)
        ll = log_loss(va.target_result, model.predict_proba(va[cols]), labels=model.classes_)
        rows.append({"variant": name, "train_rows": len(tr), "valid_rows": len(va), "valid_log_loss": ll})
    prior = train.target_result.value_counts(normalize=True)
    rows.append({"variant": "no features (class frequencies)", "train_rows": len(train), "valid_rows": len(valid),
                 "valid_log_loss": log_loss(valid.target_result, np.tile(prior[["A", "D", "H"]].to_numpy(), (len(valid), 1)),
                                            labels=["A", "D", "H"])})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe features.")
    parser.parse_args()
    matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
    f = build_features(matches)
    f.to_csv(FEATURES_PATH, index=False)
    cols = feature_columns(f)
    print(f"Wrote {len(f)} rows x {len(cols)} features to {FEATURES_PATH.relative_to(ROOT)}")
    recent = f[f.season >= TRAIN_SEASONS[0]]
    nan = recent[cols].isna().mean().sort_values(ascending=False)
    print(f"\nMissing values (seasons >= {TRAIN_SEASONS[0]}), worst 6:")
    print(nan.head(6).round(4).to_string())
    print(f"\nForm variant comparison (train {TRAIN_SEASONS[0]}..{TRAIN_SEASONS[1]}, validate {VALID_SEASON}):")
    print(compare_form_variants(f).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
