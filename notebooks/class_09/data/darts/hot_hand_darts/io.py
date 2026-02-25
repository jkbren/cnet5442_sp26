from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def load_throws(path: str | Path) -> pd.DataFrame:
    """Load the tidy per-throw dataset.

    Expected columns (see data/throws_tidy.csv):
      - name, TSID, throw_in_leg, turn_in_leg, dart_in_turn, ind, pbt,
        points_before_turn, points, segment_hit, triple, wonprevious,
        is_throw2, is_throw3
    """
    return pd.read_csv(path)


def load_players(path: str | Path) -> pd.DataFrame:
    """Load players table (player_id, player)."""
    return pd.read_csv(path)


def load_params(path: str | Path) -> Dict[str, Any]:
    """Load fitted model parameters exported to JSON.

    The repository provides:
      - model2_params.json
      - model3_params.json
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
