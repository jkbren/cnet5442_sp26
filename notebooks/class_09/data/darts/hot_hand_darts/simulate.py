from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit

from .models import Model1Result


def first_n_turns(throws: pd.DataFrame, n_turns: int = 2) -> pd.DataFrame:
    """Return throws restricted to the first `n_turns` within each (player, leg)."""
    req = {"turn_in_leg"}
    if not req.issubset(throws.columns):
        raise ValueError("throws must contain 'turn_in_leg' (use throws_tidy.csv)")
    out = throws[throws["turn_in_leg"] <= n_turns].copy()
    return out


def table5_empirical(throws: pd.DataFrame) -> pd.DataFrame:
    """Empirical relative frequencies of the 8 within-turn sequences (Table 5)."""
    df = first_n_turns(throws, n_turns=2)

    # Each (name, TSID, turn_in_leg) should have 3 throws in these early turns.
    seq_df = (
        df.groupby(["name", "TSID", "turn_in_leg"], sort=False)["triple"]
        .apply(lambda x: " ".join(str(int(v)) for v in x.to_list()))
        .reset_index(name="sequence")
    )
    freq = seq_df["sequence"].value_counts().sort_index().reset_index()
    freq.columns = ["sequence", "count"]
    freq["proportion"] = freq["count"] / freq["count"].sum()
    return freq


def table6_empirical(throws: pd.DataFrame) -> pd.DataFrame:
    """Empirical relative frequencies of pairs of turn-success counts (Table 6)."""
    df = first_n_turns(throws, n_turns=2)

    sums = (
        df.groupby(["name", "TSID", "turn_in_leg"], sort=False)["triple"]
        .sum()
        .unstack("turn_in_leg")
        .reset_index()
        .rename(columns={1: "turn1", 2: "turn2"})
    )
    # Safety: ensure both columns exist (they should).
    if "turn1" not in sums or "turn2" not in sums:
        raise RuntimeError("Expected to find turn 1 and turn 2 for every leg in the data")

    sums["pair"] = sums["turn1"].astype(int).astype(str) + "," + sums["turn2"].astype(int).astype(str)

    freq = sums["pair"].value_counts().sort_index().reset_index()
    freq.columns = ["pair", "count"]
    freq["proportion"] = freq["count"] / freq["count"].sum()
    return freq


def simulate_model1(
    throws: pd.DataFrame,
    model1: Model1Result,
    n_sims: int = 1000,
    seed: int = 2305,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate Table 5 and Table 6 expectations under Model 1.

    Returns (table5_mean, table6_mean) as DataFrames with proportions.
    """
    rng = np.random.default_rng(seed)

    df = first_n_turns(throws, n_turns=2).copy()

    # Predict success probabilities for the subset.
    # Rebuild design matrix with the same formula used in fit_model1.
    import patsy
    y, X = patsy.dmatrices(
        "triple ~ 0 + C(name) + C(ind, Treatment(reference='t1pt'))",
        data=df,
        return_type="dataframe",
    )
    p_hat = model1.glm_result.predict(X).to_numpy()

    # Storage
    table5_acc = []
    table6_acc = []

    for _ in range(n_sims):
        y_sim = rng.binomial(1, p_hat).astype(int)
        df["triple_sim"] = y_sim

        # Table 5 sequences
        seq_df = (
            df.groupby(["name", "TSID", "turn_in_leg"], sort=False)["triple_sim"]
            .apply(lambda x: " ".join(str(int(v)) for v in x.to_list()))
            .reset_index(name="sequence")
        )
        freq5 = seq_df["sequence"].value_counts()
        table5_acc.append(freq5)

        # Table 6 pairs
        sums = (
            df.groupby(["name", "TSID", "turn_in_leg"], sort=False)["triple_sim"]
            .sum()
            .unstack("turn_in_leg")
        )
        pairs = sums[1].astype(int).astype(str) + "," + sums[2].astype(int).astype(str)
        freq6 = pairs.value_counts()
        table6_acc.append(freq6)

    # Convert accumulators to mean proportions
    def mean_prop(series_list):
        dfc = pd.concat(series_list, axis=1).fillna(0.0)
        mean_counts = dfc.mean(axis=1)
        prop = mean_counts / mean_counts.sum()
        out = prop.sort_index().reset_index()
        out.columns = ["key", "proportion"]
        return out

    t5 = mean_prop(table5_acc)
    t5 = t5.rename(columns={"key": "sequence"})
    t6 = mean_prop(table6_acc)
    t6 = t6.rename(columns={"key": "pair"})
    return t5, t6


def _simulate_leg_model2(
    darts: np.ndarray,
    wonprevious: int,
    beta_p: float,
    *,
    phi: float,
    sigma: float,
    mu_win: float,
    mu_lose: float,
    sigma_delta: float,
    throw2: float,
    throw3: float,
    rng: np.random.Generator,
) -> np.ndarray:
    T = darts.size
    y = np.empty(T, dtype=np.int8)

    st = rng.normal(mu_win if wonprevious == 1 else mu_lose, sigma_delta)
    for t in range(T):
        if t > 0:
            st = phi * st + sigma * rng.normal()
        d = int(darts[t])
        pred = beta_p + st + (throw2 if d == 2 else 0.0) + (throw3 if d == 3 else 0.0)
        y[t] = rng.binomial(1, expit(pred))
    return y


def _simulate_leg_model3(
    darts: np.ndarray,
    wonprevious: int,
    beta_p: float,
    *,
    phi_w: float,
    phi_a: float,
    sigma_w: float,
    sigma_a: float,
    mu_win: float,
    mu_lose: float,
    sigma_delta: float,
    throw2: float,
    throw3: float,
    rng: np.random.Generator,
) -> np.ndarray:
    T = darts.size
    y = np.empty(T, dtype=np.int8)

    st = rng.normal(mu_win if wonprevious == 1 else mu_lose, sigma_delta)
    for t in range(T):
        if t > 0:
            # Across-turn update when entering a new turn: dart sequence is 1,2,3,1,2,3,...
            if int(darts[t]) == 1:
                st = phi_a * st + sigma_a * rng.normal()
            else:
                st = phi_w * st + sigma_w * rng.normal()

        d = int(darts[t])
        pred = beta_p + st + (throw2 if d == 2 else 0.0) + (throw3 if d == 3 else 0.0)
        y[t] = rng.binomial(1, expit(pred))
    return y


def simulate_model2(
    throws: pd.DataFrame,
    params: Dict,
    n_sims: int = 1000,
    seed: int = 2305,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate Table 5 and Table 6 expectations under Model 2 using fitted params JSON."""
    rng = np.random.default_rng(seed)
    df = first_n_turns(throws, n_turns=2).copy()

    # Prepare per-leg structure
    legs = df.groupby(["name", "TSID"], sort=False)
    leg_keys = list(legs.groups.keys())

    # For first 2 turns we always have 6 throws: 1,2,3,1,2,3
    darts_per_leg = legs["dart_in_turn"].apply(lambda s: s.to_numpy(dtype=np.int8))
    wonprev_per_leg = legs["wonprevious"].first().astype(int)

    # Unpack params
    phi = float(params["phi"])
    sigma = float(params["sigma"])
    mu_win = float(params["mu_delta_win"])
    mu_lose = float(params["mu_delta_lose"])
    sigma_delta = float(params["sigma_delta"])
    throw2 = float(params["throw2"])
    throw3 = float(params["throw3"])
    beta_by_player = params["beta_by_player"]

    table5_acc = []
    table6_acc = []

    for _ in range(n_sims):
        # simulate per leg, then compute tables
        seqs = []
        pairs = []

        for (player, tsid) in leg_keys:
            darts = darts_per_leg.loc[(player, tsid)]
            wonprev = int(wonprev_per_leg.loc[(player, tsid)])
            beta_p = float(beta_by_player[player])

            y = _simulate_leg_model2(
                darts,
                wonprev,
                beta_p,
                phi=phi,
                sigma=sigma,
                mu_win=mu_win,
                mu_lose=mu_lose,
                sigma_delta=sigma_delta,
                throw2=throw2,
                throw3=throw3,
                rng=rng,
            )

            # Turn 1 sequence (first 3)
            s1 = " ".join(str(int(v)) for v in y[:3])
            s2 = " ".join(str(int(v)) for v in y[3:6])
            seqs.extend([s1, s2])

            pairs.append(f"{int(y[:3].sum())},{int(y[3:6].sum())}")

        table5_acc.append(pd.Series(seqs).value_counts())
        table6_acc.append(pd.Series(pairs).value_counts())

    def mean_prop(series_list):
        dfc = pd.concat(series_list, axis=1).fillna(0.0)
        mean_counts = dfc.mean(axis=1)
        prop = mean_counts / mean_counts.sum()
        out = prop.sort_index().reset_index()
        out.columns = ["key", "proportion"]
        return out

    t5 = mean_prop(table5_acc).rename(columns={"key": "sequence"})
    t6 = mean_prop(table6_acc).rename(columns={"key": "pair"})
    return t5, t6


def simulate_model3(
    throws: pd.DataFrame,
    params: Dict,
    n_sims: int = 1000,
    seed: int = 2305,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate Table 5 and Table 6 expectations under Model 3 using fitted params JSON."""
    rng = np.random.default_rng(seed)
    df = first_n_turns(throws, n_turns=2).copy()

    legs = df.groupby(["name", "TSID"], sort=False)
    leg_keys = list(legs.groups.keys())
    darts_per_leg = legs["dart_in_turn"].apply(lambda s: s.to_numpy(dtype=np.int8))
    wonprev_per_leg = legs["wonprevious"].first().astype(int)

    phi_w = float(params["phi_within"])
    phi_a = float(params["phi_across"])
    sigma_w = float(params["sigma_within"])
    sigma_a = float(params["sigma_across"])
    mu_win = float(params["mu_delta_win"])
    mu_lose = float(params["mu_delta_lose"])
    sigma_delta = float(params["sigma_delta"])
    throw2 = float(params["throw2"])
    throw3 = float(params["throw3"])
    beta_by_player = params["beta_by_player"]

    table5_acc = []
    table6_acc = []

    for _ in range(n_sims):
        seqs = []
        pairs = []
        for (player, tsid) in leg_keys:
            darts = darts_per_leg.loc[(player, tsid)]
            wonprev = int(wonprev_per_leg.loc[(player, tsid)])
            beta_p = float(beta_by_player[player])

            y = _simulate_leg_model3(
                darts,
                wonprev,
                beta_p,
                phi_w=phi_w,
                phi_a=phi_a,
                sigma_w=sigma_w,
                sigma_a=sigma_a,
                mu_win=mu_win,
                mu_lose=mu_lose,
                sigma_delta=sigma_delta,
                throw2=throw2,
                throw3=throw3,
                rng=rng,
            )
            seqs.extend([" ".join(str(int(v)) for v in y[:3]), " ".join(str(int(v)) for v in y[3:6])])
            pairs.append(f"{int(y[:3].sum())},{int(y[3:6].sum())}")

        table5_acc.append(pd.Series(seqs).value_counts())
        table6_acc.append(pd.Series(pairs).value_counts())

    def mean_prop(series_list):
        dfc = pd.concat(series_list, axis=1).fillna(0.0)
        mean_counts = dfc.mean(axis=1)
        prop = mean_counts / mean_counts.sum()
        out = prop.sort_index().reset_index()
        out.columns = ["key", "proportion"]
        return out

    t5 = mean_prop(table5_acc).rename(columns={"key": "sequence"})
    t6 = mean_prop(table6_acc).rename(columns={"key": "pair"})
    return t5, t6
