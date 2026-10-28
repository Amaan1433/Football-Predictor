# La Liga and Champions League match predictor (phase 1)

Predicts home win / draw / away win probabilities and expected goals for La Liga
matches from past results and shot counts, and (separate model, see
[Champions League](#champions-league)) for Champions League matches, including the
probability of going through a knockout tie. Phase 1 covers data, features, models,
evaluation and a CLI. No lineup, injury or news information is used.

## Setup

```
python -m pip install -r requirements.txt
copy .env.example .env        # then put your Highlightly key in .env
python -m src.ingest          # download + cache raw data (uses Highlightly calls, see below)
python -m src.build_tables    # matches, team_match, fixtures (no API calls)
python -m src.features        # leakage-safe features (no API calls)
python -m src.models          # tune settings on the validation season -> models/config.json
python -m src.evaluate        # final scores on the test + live seasons -> reports/
python -m pytest tests        # leakage, feature, model and secret-handling tests
python -m src.api_client usage   # API call usage report
```

## Predict a team's next match

```
python -m src.predict --team "Barcelona"
python -m src.predict --team "Atletico Madrid" --offline   # cached data only, zero API calls
python -m src.predict --team "Barcelona" --next 5           # next 5 matches, La Liga + Champions League combined
python -m src.predict --date 2026-11-04                     # every La Liga + Champions League match that day
python -m src.predict --date 2026-10-25 --competition laliga   # past date: forecast (without that day's results) vs result
```

`--next N` lists a team's next N fixtures across La Liga and the Champions League (`--competition`
narrows it), all forecast from today's data, so later matches do not account for the results of
earlier ones. `--date` prints one table per competition: H/D/A, expected goals, likeliest score, and for the
Champions League each club's domestic league position (context only) and, for knockout matches,
the home side's probability of going through. For a past date it shows the forecast the model would
have made beforehand next to the actual result.

The CLI finds the team's next published La Liga fixture, refits the Poisson + Dixon-Coles model
on every played match from 2006-07 up to today, builds features using only matches before the
fixture date, and prints H/D/A probabilities, expected goals, the three most likely scores, both
teams' last 5 results and the head-to-head record. It takes about 8 seconds.

- API calls: the current-season list is re-fetched only if the cache is older than 12 hours
  (3-4 calls); missing shot stats are fetched only for the two teams' matches (usually 0-2 calls).
  If a refresh fails or the daily budget is used up, it falls back to the cache with a warning.
- If there is no upcoming fixture in the published schedule, or the team is not in this season's
  La Liga, it says so and exits with code 1 instead of guessing. Fixtures past kickoff but not
  marked finished (postponed or stale data) are reported as a warning.
- Team names are matched case- and accent-insensitively against every known spelling; unknown
  names get suggestions.

Python 3.14, versions pinned in `requirements.txt`. Random seeds are fixed (42).

## Data sources

| Data | Source | Coverage | Notes |
|---|---|---|---|
| Results, shots, shots on target | [DataHub La Liga](https://datahub.io/football/spanish-la-liga) (mirror of football-data.co.uk) | 1993-94 to 2025-26; shots from 2005-06 | football-data.co.uk itself is blocked on our network (filtered as "gambling"). The mirror has **no betting odds**. |
| Current season results, fixtures, shots | [Highlightly](https://highlightly.net) football API, free tier | 2026-27 | 100 calls/day. Season list = 3-4 calls per refresh (max once per 12 h); per-match stats = 1 call per match, cached forever. |
| xG | **not used** | – | Understat disallows bots (robots.txt). Highlightly has team xG only from 2024-25, too short to train on. A shot-based xG proxy was tested and was no better than raw goals and shots on target. |

- Raw downloads are cached under `data/raw/` and never re-fetched (current-season list: at most every 12 h).
- Team names from every source map to one canonical name in `src/names.py`; an unknown name stops the build.
- The API key lives only in `.env` (gitignored). It is sent only in a request header, never written to URLs, caches, logs or errors. `tests/test_secrets.py` fails if the key appears in any other project file.
- Every API call is logged (time, endpoint, status, provider-reported remaining quota) in `data/usage/api_usage.csv`, and the client refuses to exceed the daily budget.

## Method

**Features** (`src/features.py`). Every feature for a match uses only matches dated strictly before it (shift(1) on each team's date-sorted history; a team never plays twice on one date).
- Form: exponentially weighted means (half-life 10 matches) of points, goal difference, goals, shots and shots on target, for and against. Half-life 10 beat half-life 5 and a plain last-5 mean on the validation season.
- Elo before each match (K = 10, home advantage 30, 20% regression to the mean between seasons, goal-margin multiplier; promoted teams start at the rating of the relegated teams). 1993-2005 is burn-in only.
- Home-only / away-only form, rest days, matches in the previous 14 days, head-to-head points and goal difference (last 6 meetings, shrunk toward 0), matchweek.
- Tests in `tests/test_features.py` scramble all results on and after a date and check that no feature on or before that date changes (and that scrambling earlier results does change them, so the test can fail). A deliberately injected leak was caught.

**Models** (`src/models.py`)
1. Elo only: multinomial logistic regression on the Elo difference.
2. Multinomial logistic regression on 17 form, Elo and head-to-head differences.
3. Poisson goals: two Poisson GLMs (home goals, away goals) with a Dixon-Coles low-score correction; the scoreline grid is summed into H/D/A. This model also gives expected goals.
4. XGBoost (depth 2, 300 trees) with temperature scaling, fitted on the two most recent training seasons, which the trees do not see.

**Splits** (time-based only, no shuffling)
- Tuning: train 2006-07..2023-24, choose settings on 2024-25.
- Test: refit on 2006-07..2024-25, score 2025-26 once.
- Live check: refit on 2006-07..2025-26, score the 99 finished 2026-27 matches (to 26 Oct 2026).

## Results

Lower is better for RPS (ranked probability score), Brier and log loss. "RPS vs Elo" is the
paired difference to the Elo-only model with a 95% bootstrap interval (2,000 resamples over matches);
negative means better than Elo.

**Test season 2025-26 (380 matches)**

| Model | RPS | Brier | Log loss | Accuracy | RPS vs Elo [95% CI] |
|---|---|---|---|---|---|
| Class frequencies (no information) | 0.2229 | 0.631 | 1.047 | 49.0% | +0.0221 [+0.0090, +0.0349] |
| Elo only | 0.2008 | 0.581 | 0.980 | 51.6% | – |
| Logistic regression | **0.1974** | **0.572** | **0.967** | 53.4% | −0.0034 [−0.0066, −0.0003] |
| Poisson + Dixon-Coles | 0.1979 | 0.575 | 0.969 | 52.9% | −0.0028 [−0.0058, +0.0001] |
| XGBoost + temperature | **0.1970** | **0.571** | 0.968 | 52.9% | −0.0037 [−0.0085, +0.0006] |
| Bookmaker (margin removed) | n/a | | | | no odds source available |

**Live check 2026-27 so far (99 matches)**

| Model | RPS | Log loss | Accuracy | RPS vs Elo [95% CI] |
|---|---|---|---|---|
| Elo only | 0.1862 | 0.955 | 53.5% | – |
| Logistic regression | 0.1857 | 0.953 | 53.5% | −0.0004 [−0.0063, +0.0065] |
| Poisson + Dixon-Coles | 0.1847 | 0.946 | 51.5% | −0.0015 [−0.0063, +0.0039] |
| XGBoost + temperature | 0.1847 | 0.926 | 51.5% | −0.0015 [−0.0091, +0.0067] |

**Rolling backtest, 2018-19 to 2025-26 (8 seasons, 3,040 matches)** — `python -m src.evaluate --backtest`.
Each season is predicted by models trained on 2006-07 up to the season before it. Caveat: the model
settings were chosen on 2024-25, so for earlier seasons they carry a little information from later data.

| Model | RPS (pooled) | Log loss | Accuracy | RPS vs Elo [95% CI] | Seasons better than Elo |
|---|---|---|---|---|---|
| Elo only | 0.1986 | 0.993 | 51.7% | – | – |
| Logistic regression | 0.1975 | 0.990 | 52.2% | −0.0011 [−0.0022, +0.0001] | 5 of 8 |
| Poisson + Dixon-Coles | 0.1978 | 0.989 | 52.1% | −0.0008 [−0.0019, +0.0003] | 7 of 8 |
| XGBoost + temperature | 0.1973 | 0.988 | 51.7% | −0.0013 [−0.0030, +0.0004] | 5 of 8 |

Per-season RPS ranges from 0.187 (2023-24) to 0.206 (2022-23) for every model: **season-to-season
variation is 10-20 times larger than the differences between models**. Over 3,040 matches, form,
shots and head-to-head features add about 0.001 RPS over Elo alone, which is not statistically
distinguishable from zero. The 0.003 gain on the single 2025-26 test season was partly a favourable season.

Calibration: `reports/calibration_2025-26.png` and `.csv` (predicted vs observed frequency per 0.1 bin).
On the test season the models predicted a 46-47% home-win rate against 48.9% observed; strong home
favourites won slightly more often than predicted and strong away favourites slightly less. Draw
probabilities were well calibrated but never exceeded ~35%.

### How to read this honestly
- All models clearly beat the no-information baseline. Beyond that, **the differences are small**:
  the feature models improve on Elo by about 0.003 RPS on the test season (only logistic
  regression's interval excludes zero, just barely), by about 0.001 over the 8-season backtest
  (no interval excludes zero), and not at all on 2026-27 so far. Most of the signal is in Elo.
- XGBoost looked best on the validation season (log loss 0.956 vs 0.973 for Elo) but its lead did
  not hold on the test season, where the three feature models are essentially tied. Its settings were
  chosen on the validation season, so the validation figure was optimistic.
- Accuracy around 52-53% is typical for football; draws are rarely the most likely outcome, so they
  are almost never "picked". Use the probabilities, not the top pick.
- One test season is 380 matches. Treat the second decimal of any metric as noise unless the
  interval says otherwise.
- There is no bookmaker comparison, because the odds data could not be obtained. Bookmakers use
  lineups, injuries and market information this model does not have, so do not assume this model
  is competitive with them.

## Known limitations
- **Small samples**: one 380-match test season, a 99-match live check, and calibration bins with a few dozen matches each.
- **No lineup, injury, suspension or transfer information** (planned for phase 2).
- **La Liga model uses La Liga matches only**: Champions League, Europa League and Copa del Rey matches are not features, so fixture congestion and European form are missing, and promoted teams have no second-division history (they start at the relegated teams' Elo).
- **No xG and no odds**; shots and shots on target stand in for chance quality, and only from 2005-06.
- Current-season shots come from a different provider (Highlightly) than history (DataHub). One match cross-checked exactly, but small definitional differences between providers are possible; 2026-27 shot averages run slightly higher than 2025-26.
- Head-to-head looks at the last 6 meetings regardless of how long ago they were.
- Highlightly has only published fixtures through matchweek 26 so far; kickoff times can change.

## Champions League

```
python -m src.ingest --europe     # + openfootball UCL/league files and the 2026-27 UCL season from Highlightly
python -m src.europe              # euro_matches, clubs, ucl_fixtures (no API calls)
python -m src.euro_elo            # tune the cross-league Elo on 2014-15..2018-19 -> models/euro_config.json
python -m src.ucl --backtest      # out-of-sample check on 2019-20..2025-26 -> reports/ucl_*
python -m src.predict --competition ucl --team "Arsenal"
python -m src.predict --competition ucl --team "Bayern" --offline
```

The CLI finds the club's next Champions League fixture and prints 90-minute H/D/A probabilities,
expected goals, the most likely scores and both clubs' ratings, plus each club's domestic league
position and last 3 league results (shown for context; tested as model inputs and rejected, see below). For knockout fixtures it adds the
probability of going through: before a two-legged tie both legs are simulated; before a second leg
the actual first-leg score is used; level ties go to extra time (30 minutes at the same scoring
rates) and then 50/50 penalties. League-phase matches show no qualification probability (that
depends on the whole 36-team table and is not modelled). A refresh costs 0-7 Highlightly calls
(La Liga and UCL season lists, each at most every 12 h).

**Data.** Clubs from different leagues rarely meet, so the ratings need their domestic results too.
| Data | Source | Coverage |
|---|---|---|
| UCL + qualifiers | [openfootball/champions-league](https://github.com/openfootball/champions-league) (CC0) | 2011-12 to 2025-26 (qualifiers from 2024-25) |
| UCL 2026-27 | Highlightly (league id 2486) | league phase so far + qualifiers; 90-minute scores of extra-time matches from match events, checked against the tie |
| Domestic leagues | [openfootball/football.json](https://github.com/openfootball/football.json) (CC0): England, Germany, Italy, France, Portugal, Netherlands | 2010-11 to now |
| Spain | our La Liga tables | as above |

ClubElo (the usual source of cross-league ratings) could not be used: its site and API are blocked
from our network. Club names are reconciled across sources in `src/names_europe.py`; the build stops
on unknown, duplicated or silently renamed clubs.

**Method.**
- Cross-league Elo (`src/euro_elo.py`): one rating per club = club part + a per-country league
  offset. Domestic and European matches update the club parts; European matches between countries
  also move the league offsets. Clubs from leagues we do not cover start at a fixed lower rating.
  Settings tuned on 2014-15..2018-19 (K 15 domestic / 10 European, home advantage 50 / 20).
  The league offsets did not improve log loss there (0.937 with or without); they are kept by a
  stated tie-break (within 0.001, prefer the variant that can learn league strength), which gives
  plausible league strengths over time.
- Match model (`src/ucl.py`): Poisson + Dixon-Coles on the Elo difference and a neutral-venue flag,
  fitted on all earlier Champions League matches. Form relative to rating was tried and added nothing.
- The away-goals rule is applied to ties up to 2020-21, as UEFA did.
- Tested and rejected (`python -m src.ucl --league-form`): domestic league position and points per
  game, and points / goal difference over the last 3 league matches, both taken from league matches
  before the UCL match. Neither improved on Elo alone: on 2019-20..2025-26, last-3 form made
  predictions slightly worse (RPS +0.0022 [95% CI +0.0004, +0.0040] on the 571 matches where both
  clubs play in covered leagues). League results already move the Elo ratings, and a 3-match
  window is mostly noise.

**Results, 2019-20 to 2025-26 (out of sample: settings and model chosen on earlier seasons)**

| Model, per match (997 matches) | RPS | Log loss | Accuracy | RPS vs Elo [95% CI] |
|---|---|---|---|---|
| Class frequencies | 0.2377 | 1.038 | 47.2% | +0.0468 [+0.0374, +0.0556] |
| Elo only (logistic) | 0.1909 | 0.910 | 59.6% | – |
| **Poisson, Elo** (used by the CLI) | **0.1904** | **0.908** | 59.4% | −0.0005 [−0.0014, +0.0004] |
| Poisson, Elo + form | 0.1907 | 0.910 | 59.0% | −0.0002 [−0.0012, +0.0009] |

| P(qualify), 121 knockout ties | Brier | Log loss | Accuracy |
|---|---|---|---|
| Before the tie: model | 0.218 | 0.625 | 65% |
| Before the tie: logistic on Elo, fitted on earlier ties | 0.216 | 0.623 | 64% |
| Before the tie: coin flip | 0.250 | 0.693 | – |
| After leg 1 (108 two-legged ties): model | **0.127** | **0.382** | 81% |
| After leg 1: logistic on Elo + leg-1 margin | 0.139 | 0.424 | 78% |

- Champions League matches are more predictable than La Liga matches (log loss 0.91 vs 0.98)
  because mismatches are common. Ratings carry almost all the signal.
- Before a tie the model is no better than a simple logistic fit, and slightly overconfident at the
  top: favourites given > 90% went through 6 of 7 times, 80-90%: 81% (26 ties). After the first leg
  it is better than the logistic baseline (Brier −0.012, 95% CI −0.029 to +0.005); favourites given
  > 90% went through 98% of the time (47 ties).
- 2026-27 so far (54 league-phase matches): log loss 0.84, 74% of most-likely outcomes correct.
- Calibration plot: `reports/ucl_calibration_backtest.png`; per-tie forecasts: `reports/ucl_backtest_ties.csv`.

**Limitations.**
- **Domestic results lag unless refreshed**: openfootball's 2026-27 files stopped at 20 Sep 2026 for
  all six leagues. Add `--refresh-leagues` (CLI, or `src.ingest --europe --refresh-leagues`) to fetch
  the current English, German, Italian, French, Portuguese and Dutch seasons from Highlightly
  (league ids in `src/ingest.py`; 18 calls, at most every 12 h). When fetched, they replace
  openfootball for the current season. On 194 overlapping matches the two sources agreed on every
  score. The CLI prints the latest league result date per club's country.
- Only 7 domestic leagues are covered; clubs from other leagues are rated from European matches only.
- Small samples for ties (121); qualification probabilities ignore squad changes between legs.
- No qualification probability for the league phase yet.

## Repository layout
```
data/raw/          cached downloads (gitignored)
data/processed/    matches, team_match, fixtures, features (gitignored)
data/usage/        API call log (gitignored)
models/config.json tuned settings
reports/           results.csv, calibration table and plot
src/ingest.py      download + cache (DataHub, Highlightly)
src/api_client.py  rate-limited, cached, usage-tracked API client; key handling
src/names.py       team-name normalization
src/build_tables.py matches, team_match, fixtures
src/features.py    leakage-safe features
src/models.py      models + tuning on the validation season
src/evaluate.py    metrics, bootstrap intervals, calibration, final evaluation
src/predict.py     CLI: next fixture -> prediction (--competition ucl -> src/predict_ucl.py)
src/names_europe.py club identities across leagues and sources
src/europe.py      UCL + domestic tables, validation
src/euro_elo.py    cross-league Elo + tuning
src/ucl.py         UCL match models, tie probabilities, backtest
tests/             leakage, feature, model, tie-logic and secret tests
```
