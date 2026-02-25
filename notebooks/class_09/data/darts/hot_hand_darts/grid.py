from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class StateGrid:
    """Discretized state space for Kitagawa-style likelihood approximation.

    Parameters
    ----------
    m:
        Number of intervals / discrete states.
    bm:
        Half-width of the state range; states cover [-bm, bm].
    b:
        Interval boundaries, shape (m+1,).
    bstar:
        Interval midpoints, shape (m,).
    h:
        Interval length (constant).
    """
    m: int
    bm: float
    b: np.ndarray
    bstar: np.ndarray
    h: float


def make_state_grid(m: int, bm: float) -> StateGrid:
    if m <= 1:
        raise ValueError("m must be > 1")
    b = np.linspace(-bm, bm, m + 1)
    h = float(b[1] - b[0])
    bstar = (b[:-1] + b[1:]) * 0.5
    return StateGrid(m=m, bm=float(bm), b=b, bstar=bstar, h=h)


def transition_matrix_ar1(grid: StateGrid, phi: float, sigma: float) -> np.ndarray:
    """Compute the m×m transition matrix for st = phi * s_{t-1} + sigma * eps, eps~N(0,1).

    This matches the R code:
        Gamma[i,] <- diff(pnorm(b, phi*bstar[i], sigma))
        Gamma[i,] <- Gamma[i,] / sum(Gamma[i,])

    Returns
    -------
    Gamma : np.ndarray
        Row-stochastic transition matrix, shape (m, m).
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    means = phi * grid.bstar  # shape (m,)
    # CDF at boundaries for each previous state i.
    cdf = norm.cdf(grid.b[None, :], loc=means[:, None], scale=sigma)  # (m, m+1)
    probs = np.diff(cdf, axis=1)  # (m, m)
    row_sums = probs.sum(axis=1, keepdims=True)
    # Renormalize due to truncation of state space to [-bm, bm].
    probs = probs / row_sums
    return probs


def initial_mass(grid: StateGrid, mu: float, sigma: float) -> np.ndarray:
    """Approximate initial distribution mass over grid intervals.

    R code uses midpoint rule:
        delta <- dnorm(bstar, mu, sigma) * h

    We return the same (not additionally normalized).
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    delta = norm.pdf(grid.bstar, loc=mu, scale=sigma) * grid.h
    return delta
