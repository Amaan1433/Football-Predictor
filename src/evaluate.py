"""Probability metrics for H/D/A predictions.

All probability arrays are shape (n, 3) with columns in OUTCOMES order (H, D, A).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingest import ROOT

OUTCOMES = ["H", "D", "A"]
REPORTS_DIR = ROOT / "reports"
LIVE_SEASON = "2026-27"  # in progress; scored as an extra out-of-sample check


def one_hot(y: Sequence[str]) -> np.ndarray:
    """Outcome labels -> (n, 3) indicator matrix in OUTCOMES order."""
    y = np.asarray(y)
    return np.stack([(y == o).astype(float) for o in OUTCOMES], axis=1)


def rps_per_match(p: np.ndarray, y: Sequence[str]) -> np.ndarray:
    """Ranked probability score of each match; respects the H > D > A ordering."""
    cp, co = np.cumsum(p, axis=1)[:, :2], np.cumsum(one_hot(y), axis=1)[:, :2]
    return np.sum((cp - co) ** 2, axis=1) / 2


def rps(p: np.ndarray, y: Sequence[str]) -> float:
    """Mean ranked probability score (lower is better)."""
    return float(np.mean(rps_per_match(p, y)))


def brier(p: np.ndarray, y: Sequence[str]) -> float:
    """Multi-class Brier score, summed over the 3 outcomes (0 = perfect, 2 = worst)."""
    return float(np.mean(np.sum((p - one_hot(y)) ** 2, axis=1)))


def log_loss(p: np.ndarray, y: Sequence[str]) -> float:
    """Mean negative log probability of the observed outcome."""
    picked = np.sum(p * one_hot(y), axis=1)
    return float(-np.mean(np.log(np.clip(picked, 1e-15, 1))))


def accuracy(p: np.ndarray, y: Sequence[str]) -> float:
    """Share of matches where the most likely outcome happened."""
    return float(np.mean(np.asarray(OUTCOMES)[p.argmax(axis=1)] == np.asarray(y)))


def score_all(p: np.ndarray, y: Sequence[str]) -> dict[str, float]:
    """All metrics in one dict."""
    return {"rps": rps(p, y), "brier": brier(p, y), "log_loss": log_loss(p, y), "accuracy": accuracy(p, y)}


# ------------------------------------------------------------------ uncertainty

def bootstrap_rps_diff(p_model: np.ndarray, p_ref: np.ndarray, y: Sequence[str],
                       n_boot: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    """Paired bootstrap over matches: mean RPS(model) - RPS(ref) and its 95% interval.

    Negative = model better than the reference.
    """
    d = rps_per_match(p_model, y) - rps_per_match(p_ref, y)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


# ------------------------------------------------------------------ calibration

CAL_BINS = np.linspace(0, 1, 11)
MIN_BIN_N = 10  # bins with fewer predictions are reported but not plotted


def calibration_table(p: np.ndarray, y: Sequence[str]) -> pd.DataFrame:
    """Per outcome and probability bin: count, mean predicted probability, observed frequency."""
    obs = one_hot(y)
    rows = []
    for k, outcome in enumerate(OUTCOMES):
        bins = np.clip(np.digitize(p[:, k], CAL_BINS) - 1, 0, len(CAL_BINS) - 2)
        for b in np.unique(bins):
            m = bins == b
            rows.append({"outcome": outcome, "bin": f"{CAL_BINS[b]:.1f}-{CAL_BINS[b + 1]:.1f}", "n": int(m.sum()),
                         "mean_predicted": float(p[m, k].mean()), "observed": float(obs[m, k].mean())})
    return pd.DataFrame(rows)


# Reference palette (dataviz skill): categorical slots 1-3, validated all-pairs; chrome inks.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def plot_calibration(tables: dict[str, pd.DataFrame], title: str, path: Path) -> None:
    """Reliability diagram: one panel per outcome, one line per model, diagonal = perfect."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {"H": "Home win", "D": "Draw", "A": "Away win"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True, facecolor=SURFACE)
    for ax, outcome in zip(axes, OUTCOMES):
        ax.set_facecolor(SURFACE)
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1.5, ls="--", zorder=1)
        for (model, t), color in zip(tables.items(), SERIES_COLORS):
            s = t[(t.outcome == outcome) & (t.n >= MIN_BIN_N)]
            ax.plot(s.mean_predicted, s.observed, color=color, lw=2, marker="o", ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=model, zorder=3)
        ax.set_title(names[outcome], color=INK, fontsize=12, loc="left")
        ax.set_xlim(0, 0.9 if outcome != "D" else 0.5)
        ax.set_ylim(0, 1)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_xlabel("Predicted probability", color=INK_2, fontsize=10)
    axes[0].set_ylabel("Observed frequency", color=INK_2, fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")
    fig.text(0.01, 0.005, f"Dashed line = perfect calibration. Bins of width 0.1; bins with < {MIN_BIN_N} "
             "predictions omitted.", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ final evaluation

def evaluate_final(n_boot: int = 2000) -> None:
    """Refit every model on train+valid seasons with the tuned settings; score test and live seasons once."""
    from src.build_tables import MATCHES_PATH
    from src.features import EloParams, build_features
    from src.models import (CONFIG_PATH, LOGIT_FEATURES, ClassFrequencyModel, LogitModel, PoissonModel,
                            XGBModel, elo_only_model, split)

    cfg = json.loads(CONFIG_PATH.read_text())
    test_season, live_season = cfg["splits"]["test"], LIVE_SEASON
    f = build_features(pd.read_csv(MATCHES_PATH, parse_dates=["date"]), elo=EloParams(**cfg["elo"]))

    def fresh_models() -> dict:
        return {
            "Class frequencies": ClassFrequencyModel(),
            "Elo only": elo_only_model(),
            "Logistic regression": LogitModel(LOGIT_FEATURES, cfg["logit"]["C"]),
            "Poisson + Dixon-Coles": PoissonModel(),
            "XGBoost + temperature": XGBModel(cfg["xgb"]),
        }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    # Each evaluated season is predicted by models trained on every season before it.
    for season, last_train in ((test_season, cfg["splits"]["valid"]), (live_season, test_season)):
        train_seasons = (cfg["splits"]["train"][0], last_train)
        train, ev = split(f, train_seasons, season)
        models = {name: m.fit(train) for name, m in fresh_models().items()}
        y = ev.target_result.to_numpy()
        probs = {name: m.predict_proba(ev) for name, m in models.items()}
        rows = []
        for name, p in probs.items():
            diff, lo, hi = bootstrap_rps_diff(p, probs["Elo only"], y, n_boot=n_boot)
            rows.append({"season": season, "matches": len(ev), "model": name, **score_all(p, y),
                         "rps_vs_elo": diff, "rps_vs_elo_ci_low": lo, "rps_vs_elo_ci_high": hi})
        res = pd.DataFrame(rows)
        all_rows.append(res)
        print(f"\n=== {season}: {len(ev)} matches (trained on {train_seasons[0]}..{train_seasons[1]}, "
              f"{len(train)} matches) ===")
        with pd.option_context("display.width", 220):
            print(res.drop(columns=["season", "matches"]).round(4).to_string(index=False))

        if season == test_season:
            plotted = {k: calibration_table(probs[k], y) for k in
                       ("Elo only", "Poisson + Dixon-Coles", "XGBoost + temperature")}
            cal = pd.concat([calibration_table(p, y).assign(model=n) for n, p in probs.items()])
            cal.to_csv(REPORTS_DIR / f"calibration_{season}.csv", index=False)
            plot_calibration(plotted, f"Calibration on the {season} test season ({len(ev)} matches)",
                             REPORTS_DIR / f"calibration_{season}.png")
            obs_rate = pd.Series(y).value_counts(normalize=True).reindex(OUTCOMES)
            print("\nMean predicted vs observed H/D/A rate:")
            print(pd.DataFrame({n: p.mean(axis=0) for n, p in probs.items()}, index=OUTCOMES)
                  .assign(observed=obs_rate.to_numpy()).round(3).T.to_string())

    pd.concat(all_rows).to_csv(REPORTS_DIR / "results.csv", index=False)
    print(f"\nWrote {REPORTS_DIR.relative_to(ROOT)}/results.csv, calibration_{test_season}.csv/.png")


BACKTEST_SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def backtest(n_boot: int = 2000) -> pd.DataFrame:
    """Rolling-origin backtest: each season is predicted by models trained on all earlier seasons.

    Caveat: model settings (Elo K, XGBoost depth, ...) were chosen on 2024-25, so for
    seasons before that the settings carry a little information from later data.
    """
    from src.build_tables import MATCHES_PATH
    from src.features import EloParams, build_features
    from src.models import (CONFIG_PATH, LOGIT_FEATURES, LogitModel, PoissonModel, XGBModel,
                            elo_only_model, split)

    cfg = json.loads(CONFIG_PATH.read_text())
    f = build_features(pd.read_csv(MATCHES_PATH, parse_dates=["date"]), elo=EloParams(**cfg["elo"]))
    first = cfg["splits"]["train"][0]
    probs: dict[str, list[np.ndarray]] = {}
    ys, rows = [], []
    for season in BACKTEST_SEASONS:
        prev = BACKTEST_SEASONS[BACKTEST_SEASONS.index(season) - 1] if season != BACKTEST_SEASONS[0] else None
        last_train = prev or "2017-18"
        train, ev = split(f, (first, last_train), season)
        models = {"Elo only": elo_only_model(), "Logistic regression": LogitModel(LOGIT_FEATURES, cfg["logit"]["C"]),
                  "Poisson + Dixon-Coles": PoissonModel(), "XGBoost + temperature": XGBModel(cfg["xgb"])}
        y = ev.target_result.to_numpy()
        ys.append(y)
        for name, m in models.items():
            p = m.fit(train).predict_proba(ev)
            probs.setdefault(name, []).append(p)
            rows.append({"season": season, "model": name, "rps": rps(p, y)})
    per_season = pd.DataFrame(rows).pivot(index="season", columns="model", values="rps")

    y_all = np.concatenate(ys)
    pooled = []
    for name, ps in probs.items():
        p = np.concatenate(ps)
        diff, lo, hi = bootstrap_rps_diff(p, np.concatenate(probs["Elo only"]), y_all, n_boot=n_boot)
        pooled.append({"model": name, **score_all(p, y_all), "rps_vs_elo": diff,
                       "ci_low": lo, "ci_high": hi, "seasons_beating_elo": int((per_season[name] < per_season["Elo only"]).sum())})
    pooled = pd.DataFrame(pooled)

    print(f"Rolling backtest, RPS per season (train {first}..previous season):")
    print(per_season.round(4).to_string())
    print(f"\nPooled over {len(BACKTEST_SEASONS)} seasons ({len(y_all)} matches):")
    with pd.option_context("display.width", 220):
        print(pooled.round(4).to_string(index=False))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    per_season.to_csv(REPORTS_DIR / "backtest_per_season.csv")
    pooled.to_csv(REPORTS_DIR / "backtest_pooled.csv", index=False)
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser(description="Final evaluation on the test and live seasons.")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--backtest", action="store_true", help="rolling backtest over 2018-19..2025-26")
    args = parser.parse_args()
    np.random.seed(42)
    if args.backtest:
        backtest(args.n_boot)
    else:
        evaluate_final(args.n_boot)


if __name__ == "__main__":
    main()
