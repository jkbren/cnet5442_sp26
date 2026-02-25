from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
import statsmodels.api as sm
import patsy

from .grid import StateGrid, make_state_grid, transition_matrix_ar1, initial_mass
from .preprocess import LegSequence


@dataclass(frozen=True)
class Model1Result:
    """Result object for Model 1 (logistic regression)."""
    glm_result: sm.GLM
    coef: pd.Series


def fit_model1(throws: pd.DataFrame) -> Model1Result:
    """Fit Model 1: triple ~ name - 1 + ind (R) using statsmodels GLM.

    This yields:
      - 73 player fixed effects (baseline for ind==t1pt)
      - 2 within-turn dummy effects (t2pt, t3pt relative to t1pt)
    """
    # Ensure string columns for patsy.
    df = throws.copy()
    df["name"] = df["name"].astype(str)
    df["ind"] = df["ind"].astype(str)

    y, X = patsy.dmatrices(
        "triple ~ 0 + C(name) + C(ind, Treatment(reference='t1pt'))",
        data=df,
        return_type="dataframe",
    )
    model = sm.GLM(y, X, family=sm.families.Binomial())
    res = model.fit()
    coef = res.params.copy()
    coef.index = X.columns
    return Model1Result(glm_result=res, coef=coef)


def _emission_vector(
    bstar: np.ndarray,
    beta_p: float,
    throw2: float,
    throw3: float,
    dart_in_turn: int,
    y: int,
) -> np.ndarray:
    """Emission probabilities for a single observation at all grid points."""
    if dart_in_turn == 1:
        p = expit(bstar + beta_p)
    elif dart_in_turn == 2:
        p = expit(bstar + beta_p + throw2)
    elif dart_in_turn == 3:
        p = expit(bstar + beta_p + throw3)
    else:
        raise ValueError(f"dart_in_turn must be 1/2/3, got {dart_in_turn}")
    return p if y == 1 else (1.0 - p)


def negloglik_model2(
    theta_star: np.ndarray,
    legs_by_player: Sequence[Sequence[LegSequence]],
    grid: StateGrid,
) -> float:
    """Negative log-likelihood for Model 2 (AR(1) latent state).

    Parameterization matches the R code:
      theta_star = [
        phi,
        log_sigma,
        mu_delta_win,
        mu_delta_lose,
        log_sigma_delta,
        throw2,
        throw3,
        beta_player_1, ..., beta_player_73
      ]
    """
    theta_star = np.asarray(theta_star, dtype=float)
    P = len(legs_by_player)
    if theta_star.size != 7 + P:
        raise ValueError(f"Expected {7+P} parameters (7 + {P} player effects), got {theta_star.size}")

    phi = float(theta_star[0])
    sigma = float(np.exp(theta_star[1]))
    mu_win = float(theta_star[2])
    mu_lose = float(theta_star[3])
    sigma_delta = float(np.exp(theta_star[4]))
    throw2 = float(theta_star[5])
    throw3 = float(theta_star[6])
    betas = theta_star[7:]

    Gamma = transition_matrix_ar1(grid, phi=phi, sigma=sigma)
    # Precompute both possible initial distributions.
    delta_win = initial_mass(grid, mu=mu_win, sigma=sigma_delta)
    delta_lose = initial_mass(grid, mu=mu_lose, sigma=sigma_delta)

    ll = 0.0

    bstar = grid.bstar

    for p in range(P):
        beta_p = float(betas[p])
        # Precompute success probabilities per dart position.
        p1 = expit(bstar + beta_p)
        p2 = expit(bstar + beta_p + throw2)
        p3 = expit(bstar + beta_p + throw3)
        for leg in legs_by_player[p]:
            y = leg.y
            darts = leg.dart_in_turn
            # initial distribution
            alpha = (delta_win if leg.wonprevious == 1 else delta_lose) * (p1 if y[0] == 1 else (1.0 - p1))
            s = alpha.sum()
            if not np.isfinite(s) or s <= 0:
                return np.inf
            ll += np.log(s)
            alpha = alpha / s
            # forward recursion with scaling
            for t in range(1, y.size):
                d = int(darts[t])
                if d == 1:
                    emiss = p1 if y[t] == 1 else (1.0 - p1)
                elif d == 2:
                    emiss = p2 if y[t] == 1 else (1.0 - p2)
                else:
                    emiss = p3 if y[t] == 1 else (1.0 - p3)
                alpha = alpha @ Gamma
                alpha = alpha * emiss
                s = alpha.sum()
                if not np.isfinite(s) or s <= 0:
                    return np.inf
                ll += np.log(s)
                alpha = alpha / s

    return -ll


def negloglik_model3(
    theta_star: np.ndarray,
    legs_by_player: Sequence[Sequence[LegSequence]],
    grid: StateGrid,
) -> float:
    """Negative log-likelihood for Model 3 (periodic AR(1) latent state).

    Parameterization matches the R code:
      theta_star = [
        phi_within,
        phi_across,
        log_sigma_within,
        log_sigma_across,
        mu_delta_win,
        mu_delta_lose,
        log_sigma_delta,
        throw2,
        throw3,
        beta_player_1, ..., beta_player_73
      ]

    The transition used between t-1 -> t depends on whether the *current* observation is the
    first dart of a new turn (dart_in_turn==1). This mirrors the R likelihood code.
    """
    theta_star = np.asarray(theta_star, dtype=float)
    P = len(legs_by_player)
    if theta_star.size != 9 + P:
        raise ValueError(f"Expected {9+P} parameters (9 + {P} player effects), got {theta_star.size}")

    phi_w = float(theta_star[0])
    phi_a = float(theta_star[1])
    sigma_w = float(np.exp(theta_star[2]))
    sigma_a = float(np.exp(theta_star[3]))
    mu_win = float(theta_star[4])
    mu_lose = float(theta_star[5])
    sigma_delta = float(np.exp(theta_star[6]))
    throw2 = float(theta_star[7])
    throw3 = float(theta_star[8])
    betas = theta_star[9:]

    Gamma_w = transition_matrix_ar1(grid, phi=phi_w, sigma=sigma_w)
    Gamma_a = transition_matrix_ar1(grid, phi=phi_a, sigma=sigma_a)

    delta_win = initial_mass(grid, mu=mu_win, sigma=sigma_delta)
    delta_lose = initial_mass(grid, mu=mu_lose, sigma=sigma_delta)

    ll = 0.0
    bstar = grid.bstar

    for p in range(P):
        beta_p = float(betas[p])
        p1 = expit(bstar + beta_p)
        p2 = expit(bstar + beta_p + throw2)
        p3 = expit(bstar + beta_p + throw3)
        for leg in legs_by_player[p]:
            y = leg.y
            darts = leg.dart_in_turn
            alpha = (delta_win if leg.wonprevious == 1 else delta_lose) * (p1 if y[0] == 1 else (1.0 - p1))
            s = alpha.sum()
            if not np.isfinite(s) or s <= 0:
                return np.inf
            ll += np.log(s)
            alpha = alpha / s

            for t in range(1, y.size):
                d = int(darts[t])
                # choose transition matrix based on whether current throw is the first of a turn
                Gamma = Gamma_a if d == 1 else Gamma_w

                if d == 1:
                    emiss = p1 if y[t] == 1 else (1.0 - p1)
                elif d == 2:
                    emiss = p2 if y[t] == 1 else (1.0 - p2)
                else:
                    emiss = p3 if y[t] == 1 else (1.0 - p3)

                alpha = alpha @ Gamma
                alpha = alpha * emiss
                s = alpha.sum()
                if not np.isfinite(s) or s <= 0:
                    return np.inf
                ll += np.log(s)
                alpha = alpha / s

    return -ll


def fit_model2(
    legs_by_player: Sequence[Sequence[LegSequence]],
    m: int = 150,
    bm: float = 2.5,
    theta_init: Optional[np.ndarray] = None,
    method: str = "L-BFGS-B",
    options: Optional[dict] = None,
) -> dict:
    """Fit Model 2 by numerical maximization (slow).

    Warning: fitting is computationally expensive (the authors report hours/days in R).

    Returns a dict with the optimizer result and convenient unpacked parameters.
    """
    P = len(legs_by_player)
    grid = make_state_grid(m=m, bm=bm)

    if theta_init is None:
        # Rough translation of R starting values:
        # theta <- c(0.6, 0.3, 0, 0, 0.4, qlogis(0.15), qlogis(0.15), qlogis(0.35))
        # theta.star <- c(phi, log(sigma), mu_win, mu_lose, log(sigma_delta), throw2, throw3, rep(beta0,73))
        from scipy.special import logit
        theta_init = np.concatenate(
            [
                np.array([0.6, np.log(0.3), 0.0, 0.0, np.log(0.4), logit(0.15), logit(0.15)]),
                np.full(P, logit(0.35)),
            ]
        )

    def obj(th: np.ndarray) -> float:
        return negloglik_model2(th, legs_by_player=legs_by_player, grid=grid)

    res = minimize(obj, np.asarray(theta_init, dtype=float), method=method, options=options or {})
    return {"grid": grid, "opt": res}


def fit_model3(
    legs_by_player: Sequence[Sequence[LegSequence]],
    m: int = 150,
    bm: float = 2.5,
    theta_init: Optional[np.ndarray] = None,
    method: str = "L-BFGS-B",
    options: Optional[dict] = None,
) -> dict:
    """Fit Model 3 by numerical maximization (slow).

    Warning: fitting is computationally expensive.

    Returns a dict with the optimizer result and the grid.
    """
    P = len(legs_by_player)
    grid = make_state_grid(m=m, bm=bm)

    if theta_init is None:
        from scipy.special import logit
        # R starting values:
        # theta <- c(0.3, 0.2, 0.2, 0.5, 0, 0, 0.5, qlogis(0.25), qlogis(0.25), qlogis(0.35))
        # theta.star <- c(phi1, phi2, log(sigma1), log(sigma2), mu_win, mu_lose, log(sigma_delta), throw2, throw3, rep(beta0,73))
        theta_init = np.concatenate(
            [
                np.array([0.3, 0.2, np.log(0.2), np.log(0.5), 0.0, 0.0, np.log(0.5), logit(0.25), logit(0.25)]),
                np.full(P, logit(0.35)),
            ]
        )

    def obj(th: np.ndarray) -> float:
        return negloglik_model3(th, legs_by_player=legs_by_player, grid=grid)

    res = minimize(obj, np.asarray(theta_init, dtype=float), method=method, options=options or {})
    return {"grid": grid, "opt": res}
