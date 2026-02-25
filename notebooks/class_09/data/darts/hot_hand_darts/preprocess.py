from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LegSequence:
    """A single player's throw sequence for one leg (truncated at score>=180).

    Attributes
    ----------
    y:
        1 if throw hits H={T11..T20, bull's-eye}, else 0.
    dart_in_turn:
        1,2,3 indicating the throw within the player's turn.
    wonprevious:
        1 if the player won the previous leg in that match, else 0.
    """
    y: np.ndarray  # shape (T,)
    dart_in_turn: np.ndarray  # shape (T,), values 1/2/3
    wonprevious: int


def build_leg_sequences(
    throws: pd.DataFrame,
    players: Sequence[str] | None = None,
) -> Tuple[List[str], List[List[LegSequence]]]:
    """Convert the tidy throw table into the nested list structure used by the R code.

    In the original R supplement, `data.list` is a list of length 73 (players). Each element
    is itself a list over legs, where each leg contains a data.frame with columns including
    `triple`, `ind`, and `wonprevious`.

    Here we return:
      - players_out: sorted player names (unless `players` is provided)
      - legs_by_player: list of lists of LegSequence in TSID order

    Notes
    -----
    - The TSID in the provided data is a per-player leg index (starts at 1 for every player),
      so we group by (name, TSID).
    - Within each (name, TSID) group we rely on `throw_in_leg` to be sequential.
    """
    required = {"name", "TSID", "throw_in_leg", "dart_in_turn", "triple", "wonprevious"}
    missing = required - set(throws.columns)
    if missing:
        raise ValueError(f"throws is missing required columns: {sorted(missing)}")

    if players is None:
        players_out = sorted(throws["name"].unique().tolist())
    else:
        players_out = list(players)

    # Ensure stable ordering.
    throws_sorted = throws.sort_values(["name", "TSID", "throw_in_leg"], kind="mergesort")

    legs_by_player: List[List[LegSequence]] = []

    for player in players_out:
        df_p = throws_sorted[throws_sorted["name"] == player]
        player_legs: List[LegSequence] = []
        for tsid, df_leg in df_p.groupby("TSID", sort=True):
            y = df_leg["triple"].to_numpy(dtype=np.int8, copy=True)
            dart_in_turn = df_leg["dart_in_turn"].to_numpy(dtype=np.int8, copy=True)
            wonprev = int(df_leg["wonprevious"].iloc[0])
            player_legs.append(LegSequence(y=y, dart_in_turn=dart_in_turn, wonprevious=wonprev))
        legs_by_player.append(player_legs)

    return players_out, legs_by_player
