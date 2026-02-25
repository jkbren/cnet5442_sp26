from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import expit

from .grid import StateGrid, make_state_grid, transition_matrix_ar1, initial_mass
from .preprocess import LegSequence


@dataclass(frozen=True)
class ViterbiResult:
    state_index: np.ndarray  # shape (T,), values 0..m-1
    decoded_success_prob: np.ndarray  # shape (T,), values in (0,1)
    grid: StateGrid


def viterbi_decode_model3(
    leg: LegSequence,
    *,
    player_beta: float,
    params: Dict[str, float],
    m: int = 1000,
    bm: Optional[float] = None,
) -> ViterbiResult:
    """Decode the most likely latent-state path for Model 3 via Viterbi.

    This mirrors the R function `cont.viterbi()` in Code_Simulation.R, but implemented in
    log-probability space and with explicit backpointers.

    Parameters
    ----------
    leg:
        A single leg sequence (y, dart_in_turn, wonprevious).
    player_beta:
        Player-specific fixed effect (baseline, for dart_in_turn==1).
    params:
        Dictionary with keys:
          - phi_within, phi_across
          - sigma_within, sigma_across
          - mu_delta_win, mu_delta_lose, sigma_delta
          - throw2, throw3
          - bm (optional if `bm` argument is not given)
    m:
        Grid size for decoding (R code uses 1000).
    bm:
        Half-width of state range; default uses params['bm'] if present.

    Returns
    -------
    ViterbiResult
    """
    T = int(leg.y.size)
    if T == 0:
        raise ValueError("Empty leg sequence")

    bm_eff = float(bm if bm is not None else params.get("bm", 2.5))
    grid = make_state_grid(m=m, bm=bm_eff)

    phi_w = float(params["phi_within"])
    phi_a = float(params["phi_across"])
    sigma_w = float(params["sigma_within"])
    sigma_a = float(params["sigma_across"])
    mu_win = float(params["mu_delta_win"])
    mu_lose = float(params["mu_delta_lose"])
    sigma_delta = float(params["sigma_delta"])
    throw2 = float(params["throw2"])
    throw3 = float(params["throw3"])

    Gamma_w = transition_matrix_ar1(grid, phi=phi_w, sigma=sigma_w)
    Gamma_a = transition_matrix_ar1(grid, phi=phi_a, sigma=sigma_a)
    # log transition matrices (avoid log(0) by clipping very small values)
    eps = 1e-300
    logGamma_w = np.log(np.clip(Gamma_w, eps, 1.0))
    logGamma_a = np.log(np.clip(Gamma_a, eps, 1.0))

    # initial distribution
    delta = initial_mass(
        grid,
        mu=(mu_win if leg.wonprevious == 1 else mu_lose),
        sigma=sigma_delta,
    )
    logdelta = np.log(np.clip(delta, eps, None))

    bstar = grid.bstar

    # Precompute per-dart success probabilities across the grid.
    p1 = expit(bstar + player_beta)
    p2 = expit(bstar + player_beta + throw2)
    p3 = expit(bstar + player_beta + throw3)

    def log_emission(dart: int, y: int) -> np.ndarray:
        if dart == 1:
            p = p1
        elif dart == 2:
            p = p2
        else:
            p = p3
        return np.log(np.clip(p, eps, 1.0)) if y == 1 else np.log(np.clip(1.0 - p, eps, 1.0))

    # Viterbi DP
    log_delta_t = logdelta + log_emission(int(leg.dart_in_turn[0]), int(leg.y[0]))
    psi = np.empty((T, m), dtype=np.int32)
    psi[0, :] = -1

    for t in range(1, T):
        dart = int(leg.dart_in_turn[t])
        logGamma = logGamma_a if dart == 1 else logGamma_w

        # scores[i,j] = log_delta_{t-1}(i) + logGamma[i,j]
        scores = log_delta_t[:, None] + logGamma  # (m, m)
        psi[t, :] = np.argmax(scores, axis=0)
        log_delta_t = scores[psi[t, :], np.arange(m)] + log_emission(dart, int(leg.y[t]))

    state = np.empty(T, dtype=np.int32)
    state[T - 1] = int(np.argmax(log_delta_t))
    for t in range(T - 2, -1, -1):
        state[t] = psi[t + 1, state[t + 1]]

    # Convert decoded states to decoded success probabilities for each throw.
    decoded_prob = np.empty(T, dtype=float)
    for t in range(T):
        dart = int(leg.dart_in_turn[t])
        s_val = bstar[state[t]]
        if dart == 1:
            decoded_prob[t] = float(expit(s_val + player_beta))
        elif dart == 2:
            decoded_prob[t] = float(expit(s_val + player_beta + throw2))
        else:
            decoded_prob[t] = float(expit(s_val + player_beta + throw3))

    return ViterbiResult(state_index=state, decoded_success_prob=decoded_prob, grid=grid)
