"""Quick start example for the hot_hand_darts Python port.

Run from the repo root (hot_hand_darts_python/):

    python -m examples.quickstart

This script:
- Loads the tidy throw data (first rows printed)
- Computes empirical Table 5 and Table 6 summaries (very fast)
- Loads fitted Model 3 parameters (from the authors' R run, exported to JSON)
- Builds the leg-sequence structure and evaluates the Model 3 negative log-likelihood
- Runs a Viterbi decode for one example leg

Note: Refitting Model 2/3 is computationally expensive; use the JSON params unless you
really need to refit.
"""

from pathlib import Path
import numpy as np

from hot_hand_darts.io import load_throws, load_params
from hot_hand_darts.preprocess import build_leg_sequences
from hot_hand_darts.grid import make_state_grid
from hot_hand_darts.models import negloglik_model3
from hot_hand_darts.decode import viterbi_decode_model3
from hot_hand_darts.simulate import table5_empirical, table6_empirical


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "throws_tidy.csv"

def main() -> None:
    throws = load_throws(DATA)
    print("Loaded throws:", throws.shape)
    print(throws.head())

    # Empirical summaries (Tables 5 & 6 of the paper)
    t5 = table5_empirical(throws)
    t6 = table6_empirical(throws)
    print("\nTable 5 (empirical) head:\n", t5.head(8))
    print("\nTable 6 (empirical) head:\n", t6.head(10))

    # Load Model 3 fitted params (exported from the provided Model3.RData)
    params3 = load_params(ROOT / "model3_params.json")
    players, legs_by_player = build_leg_sequences(throws)

    # Construct theta_star vector in the same parameterization used by negloglik_model3
    P = len(players)
    theta_star = np.concatenate([
        np.array([
            params3["phi_within"],
            params3["phi_across"],
            np.log(params3["sigma_within"]),
            np.log(params3["sigma_across"]),
            params3["mu_delta_win"],
            params3["mu_delta_lose"],
            np.log(params3["sigma_delta"]),
            params3["throw2"],
            params3["throw3"],
        ], dtype=float),
        np.array([params3["beta_by_player"][p] for p in players], dtype=float),
    ])

    grid = make_state_grid(m=int(params3["m"]), bm=float(params3["bm"]))
    nll = negloglik_model3(theta_star, legs_by_player=legs_by_player, grid=grid)
    k = 9 + P  # 9 global + player effects
    aic = 2 * (nll + k)
    print(f"\nModel 3 check: nll={nll:.3f}, AIC={aic:.3f} (should match JSON-derived AIC close)")
    print(f"JSON AIC: {params3.get('AIC', 'not stored in json')}")

    # Viterbi decode: example leg
    # pick Gary Anderson, leg TSID=1
    player_name = "Gary Anderson"
    player_idx = players.index(player_name)
    leg0 = legs_by_player[player_idx][0]
    beta_p = params3["beta_by_player"][player_name]

    vit = viterbi_decode_model3(
        leg0,
        player_beta=beta_p,
        params=params3,
        m=1000,
    )
    print(f"\nViterbi decoded success probs for {player_name}, leg 1 (length {leg0.y.size}):")
    print(np.round(vit.decoded_success_prob, 3))

if __name__ == "__main__":
    main()
