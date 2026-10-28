"""Champions League match and tie models on top of the cross-league Elo (src/euro_elo.py).

Features (all from matches strictly before the match date):
- Elo difference and a neutral-venue flag.
- Form relative to rating: exponentially weighted mean of (actual score - Elo-expected
  score) over each club's recent matches in all competitions. Unlike raw goals this is
  comparable across leagues of different strength.

Models: Elo-only multinomial LR, Poisson + Dixon-Coles on Elo, Poisson + Dixon-Coles on
Elo + form. Ties: probability of going through a two-legged or single-match tie, with
extra time = 30 minutes at the same scoring rates and penalties = 50/50. The away-goals
rule is applied up to 2020-21, as UEFA did.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.euro_elo import COVERED, EURO_CONFIG_PATH, EUROPEAN, EuroEloParams, TUNE_SEASONS, load_euro_matches, run_elo
from src.evaluate import (REPORTS_DIR, bootstrap_rps_diff, calibration_table, plot_calibration, rps,
                          score_all)
from src.ingest import ROOT
from src.models import MAX_GOALS, ClassFrequencyModel, LogitModel, PoissonModel

FORM_HALF_LIFE = 10
ET_FRACTION = 30 / 90
LAST_AWAY_GOALS_SEASON = "2020-21"  # UEFA abolished the away-goals rule from 2021-22
FIRST_UCL_SEASON = "2011-12"

ELO_FEATURES = ["elo_diff", "neutral_f"]
FORM_FEATURES = ["elo_diff", "neutral_f", "home_form", "away_form"]


# ------------------------------------------------------------------ features

def build_ucl_frame(m: pd.DataFrame, p: EuroEloParams) -> pd.DataFrame:
    """European matches with pre-match Elo, form and targets. m: all matches (domestic + European), date-sorted."""
    pre, _, _ = run_elo(m, p)
    d = m.merge(pre, on="match_id", how="left")
    euro = d.competition.isin(EUROPEAN)
    h = np.where(d.neutral, 0.0, np.where(euro, p.h_euro, p.h_dom))
    exp_home = 1 / (1 + 10 ** (-(d.home_elo + h - d.away_elo) / 400))
    score = np.where(d.home_goals.isna(), np.nan,
                     np.select([d.home_goals > d.away_goals, d.home_goals == d.away_goals], [1.0, 0.5], 0.0))
    resid = score - exp_home

    order = np.arange(len(d))
    long = pd.concat([
        pd.DataFrame({"order": order, "side": "home", "club": d.home_id, "resid": resid}),
        pd.DataFrame({"order": order, "side": "away", "club": d.away_id, "resid": -resid}),
    ]).sort_values(["order", "side"], kind="stable")
    prev = long.groupby("club", sort=False).resid.shift(1)  # strictly earlier matches (one match per club per date)
    long["form"] = prev.groupby(long.club, sort=False).transform(
        lambda s: s.ewm(halflife=FORM_HALF_LIFE, ignore_na=True).mean())
    form = long.pivot(index="order", columns="side", values="form")
    d["home_form"], d["away_form"] = form["home"].to_numpy(), form["away"].to_numpy()

    u = d[euro].copy()
    u["elo_diff"] = u.home_elo - u.away_elo
    u["neutral_f"] = u.neutral.astype(float)
    u["target_home_goals"], u["target_away_goals"] = u.home_goals, u.away_goals
    res = np.select([u.home_goals > u.away_goals, u.home_goals == u.away_goals], ["H", "D"], "A")
    u["target_result"] = np.where(u.home_goals.isna(), None, res)
    return u.reset_index(drop=True)


MIN_LEAGUE_GAMES = 3  # standings and last-3 form need at least this many league matches this season


def league_snapshots(m: pd.DataFrame) -> pd.DataFrame:
    """Domestic table after each matchday: one row per (league, season, date, club).

    Columns: rank_frac (0 = top, 1 = bottom; points, goal difference, goals), ppg, played,
    last3_ppg, last3_gd. A snapshot dated D includes matches on D, so it is valid for
    matches strictly after D.
    """
    dom = m[m.competition.isin(COVERED) & m.home_goals.notna()]
    rows = []
    for (comp, season), g in dom.groupby(["competition", "season"]):
        clubs = pd.unique(g[["home_id", "away_id"]].to_numpy().ravel())
        pts, gd, gf = (dict.fromkeys(clubs, 0) for _ in range(3))
        hist: dict[str, list[tuple[int, int]]] = {c: [] for c in clubs}
        for date, day in g.sort_values("date").groupby("date"):
            for r in day.itertuples():
                hg, ag = int(r.home_goals), int(r.away_goals)
                for club, f, a in ((r.home_id, hg, ag), (r.away_id, ag, hg)):
                    p = 3 if f > a else 1 if f == a else 0
                    pts[club] += p
                    gd[club] += f - a
                    gf[club] += f
                    hist[club].append((p, f - a))
            order = sorted(clubs, key=lambda c: (-pts[c], -gd[c], -gf[c]))
            for rank, c in enumerate(order):
                n = len(hist[c])
                last = hist[c][-3:]
                rows.append({"club_id": c, "season": season, "date": date, "played": n,
                             "rank_frac": rank / (len(clubs) - 1), "ppg": pts[c] / n if n else np.nan,
                             "last3_ppg": sum(p for p, _ in last) / 3 if n >= 3 else np.nan,
                             "last3_gd": sum(d for _, d in last) / 3 if n >= 3 else np.nan})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def add_league_form(u: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    """Each side's domestic standing and last-3 form from league matches strictly before the match.

    NaN for clubs outside the covered leagues or with fewer than MIN_LEAGUE_GAMES league
    matches this season (the models impute the training median, i.e. ~no information).
    """
    snaps = league_snapshots(m)
    cols = ["rank_frac", "ppg", "last3_ppg", "last3_gd"]
    out = u.copy()
    order = np.arange(len(out))
    for side in ("home", "away"):
        q = pd.DataFrame({"order": order, "club_id": out[f"{side}_id"], "season": out.season, "date": out.date})
        q = q.sort_values("date")
        got = pd.merge_asof(q, snaps, on="date", by=["club_id", "season"], allow_exact_matches=False)
        got.loc[got.played.fillna(0) < MIN_LEAGUE_GAMES, cols] = np.nan
        got = got.set_index("order").reindex(order)
        for c in cols:
            out[f"{side}_{c}"] = got[c].to_numpy()
    for c in cols:
        out[f"{c}_diff"] = out[f"home_{c}"] - out[f"away_{c}"]
    return out


LEAGUE_FORM_VARIANTS = {
    "Poisson, Elo": ELO_FEATURES,
    "+ standing": ELO_FEATURES + ["rank_frac_diff", "ppg_diff"],
    "+ last 3 league": ELO_FEATURES + ["last3_ppg_diff", "last3_gd_diff"],
    "+ standing + last 3": ELO_FEATURES + ["rank_frac_diff", "ppg_diff", "last3_ppg_diff", "last3_gd_diff"],
}


def compare_league_form(u: pd.DataFrame, seasons: list[str], n_boot: int = 2000) -> pd.DataFrame:
    """Rolling scores of the LEAGUE_FORM_VARIANTS, RPS difference vs plain Poisson-on-Elo with 95% CI."""
    played = u[(u.competition == "UCL") & u.target_result.notna()]
    probs: dict[str, list] = {}
    ys = []
    for s in seasons:
        train = played[(played.season >= FIRST_UCL_SEASON) & (played.season < s)]
        test = played[played.season == s]
        ys.append(test.target_result.to_numpy())
        for name, feats in LEAGUE_FORM_VARIANTS.items():
            probs.setdefault(name, []).append(PoissonModel(feats, feats).fit(train).predict_proba(test))
    y = np.concatenate(ys)
    ref = np.concatenate(probs["Poisson, Elo"])
    rows = []
    for name, ps in probs.items():
        p = np.concatenate(ps)
        diff, lo, hi = bootstrap_rps_diff(p, ref, y, n_boot=n_boot)
        rows.append({"model": name, **score_all(p, y), "rps_vs_elo": diff, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows).assign(matches=len(y))


def ucl_models() -> dict:
    """Fresh, unfitted Champions League models."""
    return {
        "Class frequencies": ClassFrequencyModel(),
        "Elo only (logistic)": LogitModel(ELO_FEATURES, C=1e6),
        "Poisson, Elo": PoissonModel(ELO_FEATURES, ELO_FEATURES),
        "Poisson, Elo + form": PoissonModel(FORM_FEATURES, FORM_FEATURES),
    }


# ------------------------------------------------------------------ ties

def is_group_stage(stage: pd.Series) -> pd.Series:
    """Group / league phase (the files also use the German 'Gruppe')."""
    return stage.str.match(r"^(Group|Gruppe|League)")


def round_of(stage: pd.Series) -> pd.Series:
    """Knockout round label: 'Finals, Round of 16' -> 'Round of 16'; both play-off legs -> 'Playoffs'."""
    return np.where(stage.str.startswith("Playoffs"), "Playoffs", stage.str.split(",").str[-1].str.strip())


def identify_ties(u: pd.DataFrame) -> pd.DataFrame:
    """Knockout ties in the main competition: team_a = leg-1 home side, winner from the actual results."""
    ko = u[(u.competition == "UCL") & ~is_group_stage(u.stage) & u.home_goals.notna()].copy()
    ko["round"] = round_of(ko.stage)
    rows = []
    for (season, stage), g in ko.groupby(["season", "round"]):
        pairs = g.groupby(g.apply(lambda r: frozenset((r.home_id, r.away_id)), axis=1))
        for _, legs in pairs:
            legs = legs.sort_values("date")
            if len(legs) == 2:
                l1, l2 = legs.iloc[0], legs.iloc[1]
                if l2.home_id != l1.away_id:
                    raise ValueError(f"{season} {stage}: second leg venue mismatch for {l1.home_id} v {l1.away_id}")
                rows.append({"season": season, "stage": stage, "team_a": l1.home_id, "team_b": l1.away_id,
                             "leg1": l1.match_id, "leg2": l2.match_id, "winner": _two_leg_winner(l1, l2, season)})
            elif len(legs) == 1:
                l1 = legs.iloc[0]
                rows.append({"season": season, "stage": stage, "team_a": l1.home_id, "team_b": l1.away_id,
                             "leg1": l1.match_id, "leg2": None, "winner": _single_winner(l1)})
            else:
                raise ValueError(f"{season} {stage}: {len(legs)} meetings between the same clubs")
    return pd.DataFrame(rows)


def _goals_incl_et(r: pd.Series) -> tuple[int, int]:
    return (int(r.home_goals_aet), int(r.away_goals_aet)) if r.aet else (int(r.home_goals), int(r.away_goals))


def _two_leg_winner(l1: pd.Series, l2: pd.Series, season: str) -> str:
    a = l1.home_id
    b_home2, a_away2 = _goals_incl_et(l2)
    agg_a, agg_b = l1.home_goals + a_away2, l1.away_goals + b_home2
    if isinstance(l2.pens_winner, str):
        return l2.away_id if l2.pens_winner == "away" else l2.home_id
    if agg_a != agg_b:
        return a if agg_a > agg_b else l1.away_id
    if season <= LAST_AWAY_GOALS_SEASON and a_away2 != l1.away_goals:
        return a if a_away2 > l1.away_goals else l1.away_id
    raise ValueError(f"Cannot decide tie {l1.match_id} / {l2.match_id}")


def _single_winner(r: pd.Series) -> str:
    if isinstance(r.pens_winner, str):
        return r.home_id if r.pens_winner == "home" else r.away_id
    h, a = _goals_incl_et(r)
    if h == a:
        raise ValueError(f"Cannot decide single match {r.match_id}")
    return r.home_id if h > a else r.away_id


def _et_grid(lam_home: float, lam_away: float) -> np.ndarray:
    g = np.arange(MAX_GOALS + 1)
    return np.outer(poisson.pmf(g, lam_home * ET_FRACTION), poisson.pmf(g, lam_away * ET_FRACTION))


def p_et_away_side_through(lam_home: float, lam_away: float, away_goals_rule: bool) -> float:
    """Leg-2 away side goes through, given the aggregate is level after 90 minutes of leg 2."""
    et = _et_grid(lam_home, lam_away)  # [home ET goals, away ET goals]
    away_more = np.triu(et, 1).sum()
    level = np.trace(et)
    if away_goals_rule:  # equal non-zero ET goals: the away side has scored more away goals
        return float(away_more + (level - et[0, 0]) + 0.5 * et[0, 0])
    return float(away_more + 0.5 * level)


def p_through_two_legs(grid2: np.ndarray, lam2_home: float, lam2_away: float, away_goals_rule: bool,
                       leg1: tuple[int, int] | None = None, grid1: np.ndarray | None = None) -> float:
    """P(team A goes through), A = leg-1 home side = leg-2 away side.

    grid2[i, j]: leg-2 90-minute probabilities, i = B (home) goals, j = A (away) goals.
    leg1 = (A goals, B goals) if played, else grid1[x, y] with x = A goals, y = B goals.
    """
    n = grid2.shape[0]
    i, j = np.arange(n)[:, None], np.arange(n)[None, :]
    p_et = p_et_away_side_through(lam2_home, lam2_away, away_goals_rule)

    def given(a1: int, b1: int) -> float:
        agg_a, agg_b = a1 + j, b1 + i
        win, level = agg_a > agg_b, agg_a == agg_b
        if away_goals_rule:
            return float((grid2 * win).sum() + (grid2 * (level & (j > b1))).sum()
                         + (grid2 * (level & (j == b1))).sum() * p_et)
        return float((grid2 * win).sum() + (grid2 * level).sum() * p_et)

    if leg1 is not None:
        return given(*leg1)
    return float(sum(grid1[x, y] * given(x, y) for x in range(n) for y in range(n) if grid1[x, y] > 1e-9))


def p_through_single(grid: np.ndarray, lam_home: float, lam_away: float) -> float:
    """P(designated home side goes through a single match): 90 minutes, then extra time, then 50/50 penalties."""
    win90 = np.tril(grid, -1).sum()
    et = _et_grid(lam_home, lam_away)
    p_et_home = np.tril(et, -1).sum() + 0.5 * np.trace(et)
    return float(win90 + np.trace(grid) * p_et_home)


# ------------------------------------------------------------------ development check

def rolling_probs(u: pd.DataFrame, seasons: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Match-level H/D/A probabilities, each season predicted by models fitted on earlier UCL seasons.

    Returns (probs per model, observed results, season of each match).
    """
    played = u[(u.competition == "UCL") & u.target_result.notna()]
    probs: dict[str, list] = {}
    ys, ss = [], []
    for s in seasons:
        train = played[(played.season >= FIRST_UCL_SEASON) & (played.season < s)]
        test = played[played.season == s]
        ys.append(test.target_result.to_numpy())
        ss.append(test.season.to_numpy())
        for name, model in ucl_models().items():
            probs.setdefault(name, []).append(model.fit(train).predict_proba(test))
    return {n: np.concatenate(p) for n, p in probs.items()}, np.concatenate(ys), np.concatenate(ss)


def rolling(u: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    """Pooled match-level scores of rolling_probs."""
    probs, y, _ = rolling_probs(u, seasons)
    return pd.DataFrame([{"model": n, **score_all(p, y)} for n, p in probs.items()]).assign(matches=len(y))


# ------------------------------------------------------------------ backtest

BACKTEST_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
CHOSEN_MODEL = "Poisson, Elo"  # best on the tuning seasons; form added nothing
TIE_BINS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


def tie_features(u: pd.DataFrame, ties: pd.DataFrame) -> pd.DataFrame:
    """Per tie: pre-tie Elo difference (A - B, from leg 1), leg-1 goal difference, outcome."""
    by_id = u.set_index("match_id")
    l1 = by_id.loc[ties.leg1]
    return ties.assign(elo_diff=l1.elo_diff.to_numpy(),
                       leg1_gd=(l1.home_goals - l1.away_goals).to_numpy(),
                       a_through=(ties.winner == ties.team_a).astype(int).to_numpy(),
                       two_legged=ties.leg2.notna().to_numpy())


def tie_forecasts(u: pd.DataFrame, ties: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    """P(team A through) before each tie and, for two-legged ties, after leg 1.

    Model: CHOSEN_MODEL fitted on earlier UCL seasons. Before the tie, leg 2 is forecast
    from the leg-1 ratings with the venue swapped, so no leg-1 information leaks in.
    Baselines: Elo expectation and a logistic regression fitted on earlier seasons' ties.
    """
    from sklearn.linear_model import LogisticRegression

    played = u[(u.competition == "UCL") & u.target_result.notna()]
    by_id = u.set_index("match_id")
    tf = tie_features(u, ties)
    out = []
    for s in seasons:
        model = ucl_models()[CHOSEN_MODEL].fit(played[(played.season >= FIRST_UCL_SEASON) & (played.season < s)])
        prior = tf[tf.season < s]
        lr_pre = LogisticRegression(C=1e6).fit(prior[["elo_diff"]], prior.a_through)
        two = prior[prior.two_legged]
        lr_post = LogisticRegression(C=1e6).fit(two[["elo_diff", "leg1_gd"]], two.a_through)
        rule = s <= LAST_AWAY_GOALS_SEASON
        cur = tf[tf.season == s]
        for t in cur.itertuples():
            leg1 = by_id.loc[[t.leg1]].reset_index()
            g1 = model.score_grid(leg1)[0]
            if not t.two_legged:
                lh, la = (float(x[0]) for x in model.expected_goals(leg1))
                pre, post = p_through_single(g1, lh, la), np.nan
            else:
                rev = leg1.assign(elo_diff=-leg1.elo_diff, neutral_f=0.0)  # leg 2 as seen before the tie
                lh, la = (float(x[0]) for x in model.expected_goals(rev))
                pre = p_through_two_legs(model.score_grid(rev)[0], lh, la, rule, grid1=g1)
                leg2 = by_id.loc[[t.leg2]].reset_index()  # ratings after leg 1: known before leg 2
                lh, la = (float(x[0]) for x in model.expected_goals(leg2))
                post = p_through_two_legs(model.score_grid(leg2)[0], lh, la, rule,
                                          leg1=(int(leg1.home_goals[0]), int(leg1.away_goals[0])))
            out.append({"tie": t.Index, "model_pre": pre, "model_post": post})
        idx = cur.index
        tf.loc[idx, "elo_pre"] = 1 / (1 + 10 ** (-cur.elo_diff / 400))
        tf.loc[idx, "logit_pre"] = lr_pre.predict_proba(cur[["elo_diff"]])[:, 1]
        c2 = cur[cur.two_legged]
        tf.loc[c2.index, "logit_post"] = lr_post.predict_proba(c2[["elo_diff", "leg1_gd"]])[:, 1]
    f = pd.DataFrame(out).set_index("tie")
    return tf.loc[f.index].join(f)


def _binary_scores(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "accuracy": float(np.mean((p > 0.5) == (y == 1)))}


def _bootstrap_brier_diff(p: np.ndarray, q: np.ndarray, y: np.ndarray, n_boot: int, seed: int = 42) -> tuple[float, float, float]:
    """Paired bootstrap over ties: mean Brier(p) - Brier(q) and its 95% interval."""
    d = (p - y) ** 2 - (q - y) ** 2
    boots = d[np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def tie_calibration(p: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Calibration from the favourite's side: bins of max(p, 1-p) vs how often the favourite went through."""
    fav_p = np.maximum(p, 1 - p)
    fav_won = np.where(p >= 0.5, y, 1 - y)
    b = np.clip(np.digitize(fav_p, TIE_BINS) - 1, 0, len(TIE_BINS) - 2)
    return pd.DataFrame([{"bin": f"{TIE_BINS[k]:.1f}-{TIE_BINS[k + 1]:.1f}", "ties": int((b == k).sum()),
                          "mean_predicted": float(fav_p[b == k].mean()), "observed": float(fav_won[b == k].mean())}
                         for k in np.unique(b)])


def backtest(u: pd.DataFrame, n_boot: int = 2000) -> None:
    """Out-of-sample check on BACKTEST_SEASONS: Elo settings and model choice were fixed on TUNE_SEASONS."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    probs, y, seasons = rolling_probs(u, BACKTEST_SEASONS)
    ref = probs["Elo only (logistic)"]
    rows = []
    for name, p in probs.items():
        diff, lo, hi = bootstrap_rps_diff(p, ref, y, n_boot=n_boot)
        rows.append({"model": name, **score_all(p, y), "rps_vs_elo": diff, "ci_low": lo, "ci_high": hi})
    pooled = pd.DataFrame(rows)
    per_season = pd.DataFrame({n: [rps(p[seasons == s], y[seasons == s]) for s in BACKTEST_SEASONS]
                               for n, p in probs.items()}, index=BACKTEST_SEASONS)
    per_season.insert(0, "matches", [int((seasons == s).sum()) for s in BACKTEST_SEASONS])
    print(f"=== Match level, UCL {BACKTEST_SEASONS[0]}..{BACKTEST_SEASONS[-1]}: {len(y)} matches ===")
    with pd.option_context("display.width", 220):
        print(pooled.round(4).to_string(index=False))
        print("\nRPS per season:")
        print(per_season.round(4).to_string())
    plot_calibration({k: calibration_table(probs[k], y) for k in ("Elo only (logistic)", CHOSEN_MODEL, "Poisson, Elo + form")},
                     f"Champions League calibration, {BACKTEST_SEASONS[0]}..{BACKTEST_SEASONS[-1]} ({len(y)} matches)",
                     REPORTS_DIR / "ucl_calibration_backtest.png")
    pooled.to_csv(REPORTS_DIR / "ucl_backtest_matches.csv", index=False)
    per_season.to_csv(REPORTS_DIR / "ucl_backtest_per_season.csv")

    tf = tie_forecasts(u, identify_ties(u), BACKTEST_SEASONS)
    tf.to_csv(REPORTS_DIR / "ucl_backtest_ties.csv", index=False)
    for when, cols, sub in (("before the tie", ["model_pre", "logit_pre", "elo_pre"], tf),
                            ("after leg 1 (two-legged ties)", ["model_post", "logit_post"], tf[tf.two_legged])):
        yt = sub.a_through.to_numpy()
        res = []
        for c in cols:
            diff, lo, hi = _bootstrap_brier_diff(sub[c].to_numpy(), sub[cols[1]].to_numpy(), yt, n_boot)
            res.append({"forecast": c, **_binary_scores(sub[c].to_numpy(), yt),
                        "brier_vs_logit": diff, "ci_low": lo, "ci_high": hi})
        res.append({"forecast": "coin flip", **_binary_scores(np.full(len(yt), 0.5), yt)})
        print(f"\n=== Ties, P(qualify) {when}: {len(sub)} ties ===")
        with pd.option_context("display.width", 220):
            print(pd.DataFrame(res).round(4).to_string(index=False))
        cal = tie_calibration(sub[cols[0]].to_numpy(), yt)
        print(f"Calibration of {cols[0]} (favourite's probability vs how often the favourite went through):")
        print(cal.round(3).to_string(index=False))
        cal.to_csv(REPORTS_DIR / f"ucl_tie_calibration_{cols[0]}.csv", index=False)
    print(f"\nWrote {REPORTS_DIR.relative_to(ROOT)}/ucl_backtest_*.csv, ucl_tie_calibration_*.csv, ucl_calibration_backtest.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Champions League models: development check, or --backtest.")
    parser.add_argument("--backtest", action="store_true", help=f"out-of-sample check on {BACKTEST_SEASONS[0]}..{BACKTEST_SEASONS[-1]}")
    parser.add_argument("--league-form", action="store_true",
                        help="test domestic standing / last-3 league form features against plain Elo")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    p = EuroEloParams(**json.loads(EURO_CONFIG_PATH.read_text())["elo"])
    u = build_ucl_frame(load_euro_matches(), p)
    if args.league_form:
        u = add_league_form(u, load_euro_matches())
        for label, seasons in (("Tuning", TUNE_SEASONS), ("Backtest", BACKTEST_SEASONS)):
            print(f"{label} {seasons[0]}..{seasons[-1]}:")
            print(compare_league_form(u, seasons, args.n_boot).round(4).to_string(index=False))
        return
    if args.backtest:
        backtest(u, args.n_boot)
        return
    print(f"Development check, UCL {TUNE_SEASONS[0]}..{TUNE_SEASONS[-1]} (each season fitted on earlier seasons only):")
    print(rolling(u, TUNE_SEASONS).round(4).to_string(index=False))

    ties = identify_ties(u)
    print(f"\nKnockout ties identified: {len(ties)} ({ties.leg2.notna().sum()} two-legged, {ties.leg2.isna().sum()} single)")

    # Illustration: 2016-17 Round of 16, Paris Saint-Germain v Barcelona (first leg 4-0)
    t = ties[(ties.season == "2016-17") & ties.team_a.str.contains("paris") & ties.team_b.str.contains("barcelona")]
    if len(t):
        t = t.iloc[0]
        train = u[(u.competition == "UCL") & u.target_result.notna() & (u.season < "2016-17")]
        model = PoissonModel(FORM_FEATURES, FORM_FEATURES).fit(train)
        l1, l2 = u.set_index("match_id").loc[t.leg1], u.set_index("match_id").loc[t.leg2]
        row2 = u[u.match_id == t.leg2]
        lam_h, lam_a = (float(x[0]) for x in model.expected_goals(row2))
        grid2 = model.score_grid(row2)[0]
        row1 = u[u.match_id == t.leg1]
        before = p_through_two_legs(grid2, lam_h, lam_a, True, grid1=model.score_grid(row1)[0])
        after = p_through_two_legs(grid2, lam_h, lam_a, True, leg1=(int(l1.home_goals), int(l1.away_goals)))
        print(f"\nExample, 2016-17 Round of 16 PSG v Barcelona: P(Barcelona through) before the tie "
              f"{1 - before:.0%}; after PSG won the first leg {int(l1.home_goals)}-{int(l1.away_goals)}: {1 - after:.1%}. "
              f"Actual second leg: Barcelona {int(l2.home_goals)}-{int(l2.away_goals)}, winner {t.winner}.")


if __name__ == "__main__":
    main()
