"""Tests for CLI helpers: team lookup, next-fixture selection, result summaries. No API calls."""

from __future__ import annotations

import pandas as pd
import pytest

from src.predict import TeamNotFoundError, head_to_head, last_results, next_fixture, resolve_team

NOW = pd.Timestamp("2026-10-27 12:00", tz="UTC")


def _fixtures(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["kickoff_utc", "home_team", "away_team", "status"])
    df["kickoff_utc"] = pd.to_datetime(df.kickoff_utc, utc=True)
    return df


@pytest.mark.parametrize("text, expected", [
    ("Barcelona", "Barcelona"), ("barcelona", "Barcelona"), ("Atlético Madrid", "Atletico Madrid"),
    ("atletico madrid", "Atletico Madrid"), ("Ath Bilbao", "Athletic Bilbao"), ("Sevilla FC", "Sevilla"),
])
def test_resolve_team(text, expected):
    assert resolve_team(text) == expected


def test_resolve_team_unknown_suggests():
    with pytest.raises(TeamNotFoundError, match="Did you mean.*Real Madrid"):
        resolve_team("Reall Madrid")


def test_next_fixture_picks_earliest_future_and_flags_overdue():
    fx = _fixtures([
        ("2026-10-20 18:00", "Barcelona", "Getafe", "Not started"),   # past kickoff, not finished
        ("2026-11-08 20:00", "Real Madrid", "Barcelona", "Not started"),
        ("2026-10-31 20:00", "Barcelona", "Alaves", "Not started"),
        ("2026-10-29 20:00", "Sevilla", "Valencia", "Not started"),
        ("2026-10-28 20:00", "Barcelona", "Elche", "Abandoned"),
    ])
    nxt, overdue = next_fixture(fx, "Barcelona", NOW)
    assert (nxt.home_team, nxt.away_team) == ("Barcelona", "Alaves")
    assert list(overdue.away_team) == ["Getafe"]


def test_next_fixture_none_when_schedule_exhausted():
    fx = _fixtures([("2026-10-20 18:00", "Sevilla", "Valencia", "Not started")])
    nxt, _ = next_fixture(fx, "Barcelona", NOW)
    assert nxt is None


def test_last_results_and_h2h_use_the_teams_point_of_view():
    m = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"]),
        "home_team": ["A", "B", "A"], "away_team": ["B", "A", "C"],
        "home_goals": [2, 3, 0], "away_goals": [0, 1, 0],
    })
    lr = last_results(m, "A")
    assert list(lr.result) == ["W", "L", "D"] and list(lr.venue) == ["H", "A", "H"]
    assert list(lr.score) == ["2-0", "1-3", "0-0"]
    meetings, rec = head_to_head(m, "A", "B")
    assert rec == {"W": 1, "D": 0, "L": 1} and len(meetings) == 2
