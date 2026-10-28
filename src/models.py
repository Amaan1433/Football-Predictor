"""Match outcome models. Every model: fit(train_df) -> self, predict_proba(df) -> (n, 3) in H/D/A order.

Development protocol (this module's main):
- train seasons 2006-07..2023-24, choose settings on 2024-25 (validation).
- 2025-26 is the test season and is never used here.
Chosen settings are written to models/config.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.build_tables import MATCHES_PATH
from src.evaluate import OUTCOMES, score_all
from src.features import EloParams, build_features, compute_elo, feature_columns
from src.ingest import ROOT

SEED = 42
CONFIG_PATH = ROOT / "models" / "config.json"
TRAIN_SEASONS = ("2006-07", "2023-24")
VALID_SEASON = "2024-25"
TEST_SEASON = "2025-26"

FORM = ["points", "gd", "gf", "ga", "shots_for", "shots_against", "sot_for", "sot_against"]
LOGIT_FEATURES = (
    ["elo_diff"] + [f"diff_ewm10_{c}" for c in FORM]
    + [f"diff_venue_ewm_{c}" for c in ["gf", "ga", "sot_for", "sot_against"]]
    + ["h2h_pts_diff", "h2h_gd", "diff_rest_days", "diff_matches_14d"]
)
POISSON_HOME = ["elo_diff", "home_ewm10_gf", "home_ewm10_sot_for", "away_ewm10_ga", "away_ewm10_sot_against",
                "home_venue_ewm_gf", "away_venue_ewm_ga"]
POISSON_AWAY = ["elo_diff", "away_ewm10_gf", "away_ewm10_sot_for", "home_ewm10_ga", "home_ewm10_sot_against",
                "away_venue_ewm_gf", "home_venue_ewm_ga"]
MAX_GOALS = 10


def _reorder(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """sklearn sorts classes alphabetically (A, D, H); return columns in H, D, A order."""
    return proba[:, [list(classes).index(o) for o in OUTCOMES]]


# ------------------------------------------------------------------ models

class ClassFrequencyModel:
    """Predicts the training-set H/D/A frequencies for every match."""

    def fit(self, train: pd.DataFrame) -> "ClassFrequencyModel":
        self.p = train.target_result.value_counts(normalize=True).reindex(OUTCOMES).to_numpy()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.tile(self.p, (len(df), 1))


class LogitModel:
    """Multinomial logistic regression on a fixed feature list (median-imputed, standardized)."""

    def __init__(self, features: list[str], C: float = 1.0):
        self.features, self.C = features, C

    def fit(self, train: pd.DataFrame) -> "LogitModel":
        self.pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(C=self.C, max_iter=2000))
        self.pipe.fit(train[self.features], train.target_result)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return _reorder(self.pipe.predict_proba(df[self.features]), self.pipe.classes_)


def elo_only_model() -> LogitModel:
    """Elo baseline: H/D/A from the pre-match Elo difference alone."""
    return LogitModel(["elo_diff"], C=1e6)


class PoissonModel:
    """Two Poisson GLMs (home goals, away goals) + Dixon-Coles low-score correction.

    Score grid 0..MAX_GOALS is summed into H/D/A. rho is fitted on the training set.
    Feature lists default to the La Liga ones; the Champions League model passes its own.
    """

    def __init__(self, home_features: list[str] | None = None, away_features: list[str] | None = None):
        self.home_features = home_features or POISSON_HOME
        self.away_features = away_features or POISSON_AWAY

    def fit(self, train: pd.DataFrame) -> "PoissonModel":
        self.imp_h = SimpleImputer(strategy="median").fit(train[self.home_features])
        self.imp_a = SimpleImputer(strategy="median").fit(train[self.away_features])
        self.glm_h = sm.GLM(train.target_home_goals, self._x(train, "h"), family=sm.families.Poisson()).fit()
        self.glm_a = sm.GLM(train.target_away_goals, self._x(train, "a"), family=sm.families.Poisson()).fit()
        lam_h, lam_a = self.expected_goals(train)
        hg = train.target_home_goals.clip(upper=MAX_GOALS).astype(int).to_numpy()
        ag = train.target_away_goals.clip(upper=MAX_GOALS).astype(int).to_numpy()
        rows = np.arange(len(train))

        def nll(rho: float) -> float:
            return -np.log(self._grid(lam_h, lam_a, rho)[rows, hg, ag] + 1e-15).sum()

        self.rho = float(minimize_scalar(nll, bounds=(-0.3, 0.3), method="bounded").x)
        return self

    def _x(self, df: pd.DataFrame, side: str) -> np.ndarray:
        imp, cols = (self.imp_h, self.home_features) if side == "h" else (self.imp_a, self.away_features)
        return sm.add_constant(imp.transform(df[cols]), has_constant="add")

    def expected_goals(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Expected home and away goals for each row."""
        return self.glm_h.predict(self._x(df, "h")), self.glm_a.predict(self._x(df, "a"))

    @staticmethod
    def _grid(lam_h: np.ndarray, lam_a: np.ndarray, rho: float) -> np.ndarray:
        """(n, G, G) scoreline probabilities with the Dixon-Coles adjustment, renormalized."""
        g = np.arange(MAX_GOALS + 1)
        ph, pa = poisson.pmf(g, lam_h[:, None]), poisson.pmf(g, lam_a[:, None])
        grid = ph[:, :, None] * pa[:, None, :]
        grid[:, 0, 0] *= 1 - lam_h * lam_a * rho
        grid[:, 0, 1] *= 1 + lam_h * rho
        grid[:, 1, 0] *= 1 + lam_a * rho
        grid[:, 1, 1] *= 1 - rho
        return grid / grid.sum(axis=(1, 2), keepdims=True)

    def score_grid(self, df: pd.DataFrame) -> np.ndarray:
        return self._grid(*self.expected_goals(df), self.rho)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        grid = self.score_grid(df)
        home = np.tril(np.ones((MAX_GOALS + 1,) * 2), -1).astype(bool)  # home goals > away goals
        return np.stack([grid[:, home].sum(1), np.einsum("nii->n", grid), grid[:, home.T].sum(1)], axis=1)


class XGBModel:
    """Gradient-boosted trees + temperature scaling.

    The last `calib_seasons` training seasons are held out from the trees and
    used only to fit the temperature T (one parameter, robust on small data).
    """

    def __init__(self, params: dict, calib_seasons: int = 2):
        self.params, self.calib_seasons = params, calib_seasons

    def fit(self, train: pd.DataFrame) -> "XGBModel":
        self.features = feature_columns(train)
        seasons = sorted(train.season.unique())
        calib_mask = train.season.isin(seasons[-self.calib_seasons:])
        fit_df, cal_df = train[~calib_mask], train[calib_mask]
        y = fit_df.target_result.map({o: i for i, o in enumerate(OUTCOMES)})
        self.clf = XGBClassifier(objective="multi:softprob", random_state=SEED, n_jobs=4,
                                 eval_metric="mlogloss", **self.params)
        self.clf.fit(fit_df[self.features], y)
        logits = np.log(np.clip(self.clf.predict_proba(cal_df[self.features]), 1e-15, 1))
        y_cal = cal_df.target_result.to_numpy()

        def nll(t: float) -> float:
            return score_all(_softmax(logits / t), y_cal)["log_loss"]

        self.temperature = float(minimize_scalar(nll, bounds=(0.3, 5.0), method="bounded").x)
        return self

    def predict_proba(self, df: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        p = self.clf.predict_proba(df[self.features])
        if not calibrated:
            return p
        return _softmax(np.log(np.clip(p, 1e-15, 1)) / self.temperature)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------ tuning

def split(f: pd.DataFrame, train_seasons: tuple[str, str], eval_season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Played matches in the inclusive training season range, and in the evaluation season."""
    played = f[f.target_result.notna()]
    train = played[(played.season >= train_seasons[0]) & (played.season <= train_seasons[1])]
    return train, played[played.season == eval_season]


def tune_elo(matches: pd.DataFrame) -> tuple[EloParams, pd.DataFrame]:
    """Grid-search Elo settings by validation log loss of the Elo-only model."""
    base = build_features(matches)  # non-Elo columns don't depend on Elo settings
    rows = []
    for k, h, r in itertools.product([8, 10, 15, 20, 25], [30, 50, 65, 80], [0.0, 0.1, 0.2, 0.33]):
        p = EloParams(k=k, home_adv=h, season_regress=r)
        f = base.drop(columns=["home_elo", "away_elo", "elo_diff", "elo_exp_home"]).merge(
            compute_elo(matches.sort_values(["date", "match_id"], kind="stable"), p), on="match_id")
        train, valid = split(f, TRAIN_SEASONS, VALID_SEASON)
        m = elo_only_model().fit(train)
        rows.append({**asdict(p), **score_all(m.predict_proba(valid), valid.target_result)})
    grid = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    best = grid.iloc[0]
    return EloParams(k=float(best.k), home_adv=float(best.home_adv), season_regress=float(best.season_regress)), grid


XGB_GRID = [
    {"max_depth": d, "n_estimators": n, "learning_rate": 0.03, "subsample": 0.8,
     "colsample_bytree": 0.7, "min_child_weight": 10, "reg_lambda": 5.0}
    for d, n in itertools.product([2, 3], [150, 300])
]


def develop(matches: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Tune every model on the validation season. Returns (config, validation results table)."""
    elo, elo_grid = tune_elo(matches)
    print("Elo grid, best 5 by validation log loss:")
    print(elo_grid.head(5).round(4).to_string(index=False), "\n")

    f = build_features(matches, elo=elo)
    train, valid = split(f, TRAIN_SEASONS, VALID_SEASON)
    results, config = [], {"elo": asdict(elo)}

    def record(name: str, model, extra: str = "") -> None:
        results.append({"model": name, "settings": extra, **score_all(model.predict_proba(valid), valid.target_result)})

    record("class frequencies", ClassFrequencyModel().fit(train))
    record("Elo only", elo_only_model().fit(train), f"k={elo.k:g}, home_adv={elo.home_adv:g}, regress={elo.season_regress:g}")

    logit_runs = [(C, LogitModel(LOGIT_FEATURES, C).fit(train)) for C in [0.01, 0.1, 1.0]]
    for C, m in logit_runs:
        record("logistic regression", m, f"C={C}")
    config["logit"] = {"C": min(logit_runs, key=lambda cm: score_all(cm[1].predict_proba(valid), valid.target_result)["log_loss"])[0]}

    pois = PoissonModel().fit(train)
    record("Poisson + Dixon-Coles", pois, f"rho={pois.rho:.3f}")
    config["poisson"] = {"rho_fitted_on_train": pois.rho}

    xgb_runs = []
    for params in XGB_GRID:
        m = XGBModel(params).fit(train)
        xgb_runs.append((params, m))
        label = f"depth={params['max_depth']}, trees={params['n_estimators']}, T={m.temperature:.2f}"
        results.append({"model": "XGBoost (uncalibrated)", "settings": label,
                        **score_all(m.predict_proba(valid, calibrated=False), valid.target_result)})
        record("XGBoost + temperature", m, label)
    best_params, _ = min(xgb_runs, key=lambda pm: score_all(pm[1].predict_proba(valid), valid.target_result)["log_loss"])
    config["xgb"] = best_params
    config["splits"] = {"train": TRAIN_SEASONS, "valid": VALID_SEASON, "test": TEST_SEASON,
                        "train_rows": len(train), "valid_rows": len(valid)}
    return config, pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune models on the validation season.")
    parser.parse_args()
    np.random.seed(SEED)
    matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
    config, results = develop(matches)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    print(f"Validation season {VALID_SEASON} ({config['splits']['valid_rows']} matches), "
          f"trained on {TRAIN_SEASONS[0]}..{TRAIN_SEASONS[1]} ({config['splits']['train_rows']} matches):")
    with pd.option_context("display.width", 200):
        print(results.round(4).to_string(index=False))
    print(f"\nChosen settings written to {CONFIG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
