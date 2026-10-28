"""Cross-league Elo: one rating per club, updated by domestic league and Champions League matches.

- A club's rating = its own part + its country's league offset. Domestic matches move
  only the club parts (zero-sum within the league). European matches move both club
  parts and the two countries' offsets (rate k_league), so a whole league's strength
  is learned from all of its clubs' European results.
- Between league seasons: ratings regress toward the league mean; promoted clubs enter
  at the mean final rating of the relegated clubs.
- A league's first season of data: unrated clubs enter at the country's rated (UCL)
  clubs' mean minus the typical gap between UCL clubs and their league mean, measured
  on leagues we already have (past data only).
- Clubs from countries without league data: their country's current mean if known,
  otherwise `foreign_init`. No regression (they play few matches).

Tuning (main): burn-in 2011-12..2013-14, tune on UCL 2014-15..2018-19 with each season
predicted by a model fitted on earlier UCL seasons only. 2019-20 onward is not used here.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluate import OUTCOMES, score_all
from src.europe import COVERED, EURO_MATCHES_PATH
from src.features import _goal_mult
from src.ingest import ROOT
from src.models import _reorder

EURO_CONFIG_PATH = ROOT / "models" / "euro_config.json"
EUROPEAN = {"UCL", "UCLQ"}
TUNE_SEASONS = ["2014-15", "2015-16", "2016-17", "2017-18", "2018-19"]
FIRST_UCL_SEASON = "2011-12"
DEFAULT_UCL_GAP = 100.0  # used only until a UCL-vs-league gap can be measured
TIE_TOLERANCE = 0.001    # log-loss differences below this are treated as ties when tuning


@dataclass(frozen=True)
class EuroEloParams:
    """Cross-league Elo settings."""
    k_dom: float = 20.0
    k_euro: float = 30.0
    h_dom: float = 50.0
    h_euro: float = 60.0
    season_regress: float = 0.2
    foreign_init: float = 1400.0
    init: float = 1500.0
    k_league: float = 0.0  # learning rate of the per-country league offset (European matches only)


def run_elo(m: pd.DataFrame, p: EuroEloParams = EuroEloParams(),
            snapshot_dates: list[pd.Timestamp] | None = None) -> tuple[pd.DataFrame, dict, dict]:
    """Pre-match ratings for every row of m (sorted by date).

    Returns (pre-match ratings per match_id, final ratings, {snapshot date: ratings copy}).
    Rows with NaN goals (fixtures) get pre-match ratings but do not update.
    """
    ratings: dict[str, float] = {}          # club parts
    offset: dict[str, float] = {}           # per-country league offsets
    dom = m[m.competition.isin(COVERED)]
    league_teams = {k: set(g.home_id) | set(g.away_id) for k, g in dom.groupby(["competition", "season"])}
    league_seasons = {c: sorted(s for (cc, s) in league_teams if cc == c) for c in COVERED}
    current: dict[str, str] = {}
    ucl_clubs_by_season = {s: set(g.home_id) | set(g.away_id) for s, g in m[m.competition == "UCL"].groupby("season")}
    snaps, pending = {}, sorted(snapshot_dates or [])

    def country_of(cid: str) -> str:
        return cid.split(":", 1)[0]

    def eff(cid: str) -> float:
        return ratings[cid] + offset.get(country_of(cid), 0.0)

    def effective_all() -> dict[str, float]:
        return {t: eff(t) for t in ratings}

    def ucl_gap() -> float:
        """Mean (UCL club rating - its league mean) over leagues that are already running."""
        gaps = []
        for c, s in current.items():
            members = [ratings[t] for t in league_teams[(c, s)] if t in ratings]
            ucl = ucl_clubs_by_season.get(s, set())
            gaps += [ratings[t] - np.mean(members) for t in league_teams[(c, s)] if t in ucl and t in ratings]
        return float(np.mean(gaps)) if len(gaps) >= 5 else DEFAULT_UCL_GAP

    def start_league_season(c: str, s: str) -> None:
        teams = league_teams[(c, s)]
        idx = league_seasons[c].index(s)
        prev = league_seasons[c][idx - 1] if idx > 0 else None
        if prev is not None and c in current:
            prev_teams = league_teams[(c, prev)]
            final = {t: ratings[t] for t in prev_teams if t in ratings}
            down = prev_teams - teams
            entry = np.mean([final[t] for t in down if t in final]) if down else np.mean(sorted(final.values())[:3])
            mean = np.mean(list(final.values()))
            for t in prev_teams & teams:
                ratings[t] = mean + (ratings[t] - mean) * (1 - p.season_regress)
            for t in teams - prev_teams:
                ratings[t] = entry
        else:  # first season of data for this league
            rated = [ratings[t] for t in teams if t in ratings]
            base = (np.mean(rated) - ucl_gap()) if rated else p.init
            for t in teams:
                ratings.setdefault(t, base)
        current[c] = s

    def ensure(cid: str) -> None:
        if cid in ratings:
            return
        same = [r for t, r in ratings.items() if country_of(t) == country_of(cid)]
        ratings[cid] = np.mean(same) if same else (p.init if country_of(cid) in COVERED else p.foreign_init)

    pre_h, pre_a = np.empty(len(m)), np.empty(len(m))
    for i, r in enumerate(m.itertuples(index=False)):
        while pending and r.date > pending[0]:
            snaps[pending.pop(0)] = effective_all()
        if r.competition in COVERED and current.get(r.competition) != r.season:
            start_league_season(r.competition, r.season)
        ensure(r.home_id)
        ensure(r.away_id)
        rh, ra = eff(r.home_id), eff(r.away_id)
        pre_h[i], pre_a[i] = rh, ra
        if np.isnan(r.home_goals):
            continue
        euro = r.competition in EUROPEAN
        h = 0.0 if r.neutral else (p.h_euro if euro else p.h_dom)
        exp_h = 1 / (1 + 10 ** (-(rh + h - ra) / 400))
        score = 1.0 if r.home_goals > r.away_goals else 0.5 if r.home_goals == r.away_goals else 0.0
        surprise = _goal_mult(abs(r.home_goals - r.away_goals)) * (score - exp_h)
        delta = (p.k_euro if euro else p.k_dom) * surprise
        ratings[r.home_id] += delta
        ratings[r.away_id] -= delta
        ch, ca = country_of(r.home_id), country_of(r.away_id)
        if euro and ch != ca and p.k_league:
            offset[ch] = offset.get(ch, 0.0) + p.k_league * surprise
            offset[ca] = offset.get(ca, 0.0) - p.k_league * surprise
    for d in pending:
        snaps[d] = effective_all()

    pre = pd.DataFrame({"match_id": m.match_id.to_numpy(), "home_elo": pre_h, "away_elo": pre_a})
    return pre, effective_all(), snaps


def load_euro_matches() -> pd.DataFrame:
    m = pd.read_csv(EURO_MATCHES_PATH, parse_dates=["date"])
    return m.sort_values(["date", "competition", "home_id"], kind="stable").reset_index(drop=True)


# ------------------------------------------------------------------ UCL outcome model on Elo

def result_of(df: pd.DataFrame) -> np.ndarray:
    """90-minute H/D/A."""
    return np.select([df.home_goals > df.away_goals, df.home_goals == df.away_goals], ["H", "D"], "A")


class UclEloModel:
    """Multinomial LR on [Elo difference, neutral venue]; the intercept learns European home advantage."""

    def fit(self, df: pd.DataFrame) -> "UclEloModel":
        self.lr = LogisticRegression(C=1e6, max_iter=2000).fit(self._x(df), result_of(df))
        return self

    @staticmethod
    def _x(df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([(df.home_elo - df.away_elo) / 100, df.neutral.astype(float)])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return _reorder(self.lr.predict_proba(self._x(df)), self.lr.classes_)


def rolling_ucl_scores(ucl: pd.DataFrame, seasons: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Predict each season with a model fitted on UCL matches from all earlier seasons."""
    ps, ys = [], []
    for s in seasons:
        train = ucl[(ucl.season >= FIRST_UCL_SEASON) & (ucl.season < s)]
        test = ucl[ucl.season == s]
        ps.append(UclEloModel().fit(train).predict_proba(test))
        ys.append(result_of(test))
    return np.concatenate(ps), np.concatenate(ys)


def tune(m: pd.DataFrame) -> tuple[EuroEloParams, pd.DataFrame]:
    """Grid search on UCL 2014-15..2018-19 (rolling, past-only fits)."""
    rows = []
    grid = itertools.product([5, 10, 15], [5, 10, 20], [20, 50], [0.0, 0.05, 0.15], [1250, 1350], [0, 8, 16, 24])
    for k_dom, k_euro, h_euro, reg, finit, k_league in grid:
        p = EuroEloParams(k_dom=k_dom, k_euro=k_euro, h_euro=h_euro, season_regress=reg, foreign_init=finit,
                          k_league=k_league)
        pre, _, _ = run_elo(m, p)
        ucl = m[m.competition == "UCL"].merge(pre, on="match_id")
        prob, y = rolling_ucl_scores(ucl, TUNE_SEASONS)
        rows.append({**asdict(p), **score_all(prob, y), "n": len(y)})
    grid_df = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    # Tie-break: within TIE_TOLERANCE of the best log loss (indistinguishable on ~600 matches),
    # prefer settings that let league strength learn (k_league > 0); without it, league
    # strengths stay wherever the initialization put them.
    near = grid_df[grid_df.log_loss <= grid_df.log_loss.iloc[0] + TIE_TOLERANCE]
    learning = near[near.k_league > 0]
    b = (learning if len(learning) else near).iloc[0]
    best = EuroEloParams(**{k: float(b[k]) for k in asdict(EuroEloParams())})
    return best, grid_df


# ------------------------------------------------------------------ sanity reports

LEAGUE_NAMES = {"ENG": "Premier League", "ESP": "La Liga", "GER": "Bundesliga", "ITA": "Serie A",
                "FRA": "Ligue 1", "POR": "Primeira Liga", "NED": "Eredivisie"}


def league_strength(m: pd.DataFrame, snaps: dict) -> pd.DataFrame:
    """Mean rating of each league's members at the end of each season."""
    out = {}
    for d, ratings in snaps.items():
        season = f"{d.year - 1}-{d.year % 100:02d}"
        row = {}
        for c in COVERED:
            g = m[(m.competition == c) & (m.season == season)]
            members = set(g.home_id) | set(g.away_id)
            vals = [ratings[t] for t in members if t in ratings]
            row[c] = np.mean(vals) if vals else np.nan
        out[season] = row
    return pd.DataFrame(out).T


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune and sanity-check the cross-league Elo.")
    parser.parse_args()
    m = load_euro_matches()
    best, grid = tune(m)
    print(f"Tuning on UCL {TUNE_SEASONS[0]}..{TUNE_SEASONS[-1]} ({int(grid.n.iloc[0])} matches), best 5 of {len(grid)}:")
    print(grid.drop(columns=["h_dom", "init", "n"]).head(5).round(4).to_string(index=False))
    print(f"(worst grid log loss: {grid.log_loss.max():.4f})")
    print(f"Chosen (tie-break: within {TIE_TOLERANCE} of best, prefer k_league > 0): {asdict(best)}")
    print("Best log loss by league-offset rate k_league:",
          grid.groupby("k_league").log_loss.min().round(4).to_dict())

    ucl_all = m[m.competition == "UCL"]
    freq = pd.Series(result_of(ucl_all[ucl_all.season < TUNE_SEASONS[0]])).value_counts(normalize=True).reindex(OUTCOMES)
    y_tune = result_of(ucl_all[ucl_all.season.isin(TUNE_SEASONS)])
    base = score_all(np.tile(freq.to_numpy(), (len(y_tune), 1)), y_tune)
    print(f"No-information baseline on the same matches: log loss {base['log_loss']:.4f}, RPS {base['rps']:.4f}")

    snap_dates = [pd.Timestamp(f"{y}-06-30") for y in range(2012, 2027)]
    pre, final, snaps = run_elo(m, best, snapshot_dates=snap_dates)
    EURO_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    EURO_CONFIG_PATH.write_text(json.dumps({"elo": asdict(best), "tune_seasons": TUNE_SEASONS}, indent=2))

    strength = league_strength(m, snaps).rename(columns=LEAGUE_NAMES)
    print("\nAverage rating of each league's clubs at season end:")
    print(strength.round(0).to_string())

    clubs = pd.read_csv(ROOT / "data" / "processed" / "clubs.csv").set_index("club_id")
    end = snaps[pd.Timestamp("2026-06-30")]
    s2526 = m[m.season == "2025-26"]
    active = set(s2526.home_id) | set(s2526.away_id)
    top = pd.Series({k: v for k, v in end.items() if k in active}).sort_values(ascending=False).head(15)
    print("\nTop 15 clubs at the end of 2025-26 (clubs that played in 2025-26):")
    for cid, r in top.items():
        print(f"  {r:6.0f}  {clubs.loc[cid, 'name']} ({cid.split(':')[0]})")
    print(f"\nSettings written to {EURO_CONFIG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
