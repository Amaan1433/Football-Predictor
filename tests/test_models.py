"""Unit tests for metrics and the Poisson score-grid conversion."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import poisson

from src.evaluate import brier, log_loss, rps
from src.models import MAX_GOALS, PoissonModel


def test_rps_hand_calculation():
    p = np.array([[0.5, 0.3, 0.2]])
    # observed H: cumulative obs (1, 1); cumulative p (0.5, 0.8)
    assert rps(p, ["H"]) == pytest.approx(((0.5 - 1) ** 2 + (0.8 - 1) ** 2) / 2)
    # observed A: cumulative obs (0, 0)
    assert rps(p, ["A"]) == pytest.approx((0.5 ** 2 + 0.8 ** 2) / 2)


def test_perfect_forecast_scores():
    p = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert rps(p, ["H", "A"]) == 0 and brier(p, ["H", "A"]) == 0 and log_loss(p, ["H", "A"]) == pytest.approx(0)


def test_poisson_grid_without_correction_is_independent_poisson():
    lam_h, lam_a = np.array([1.6]), np.array([0.9])
    grid = PoissonModel._grid(lam_h, lam_a, rho=0.0)[0]
    g = np.arange(MAX_GOALS + 1)
    expected = np.outer(poisson.pmf(g, 1.6), poisson.pmf(g, 0.9))
    assert grid == pytest.approx(expected / expected.sum())


def test_dixon_coles_negative_rho_raises_draws():
    lam = np.array([1.3])
    draws = [np.trace(PoissonModel._grid(lam, lam, rho)[0]) for rho in (0.0, -0.1)]
    assert draws[1] > draws[0]
