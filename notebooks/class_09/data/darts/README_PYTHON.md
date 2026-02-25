# Hot hand in professional darts — Python-ready version

This folder is a Python-friendly port of the *supplemental* code + data shipped with:

**M. Otting, R. Langrock, C. Deutscher & V. Leos-Barajas (2020)**  
*The hot hand in professional darts* (JRSS-A 183(2), 565–580). fileciteturn0file0

The original supplement (inside `A1527Otting.zip`) provides:
- `DataDF.RData` (a data.frame with one row per throw),
- `DataList.RData` (an R list-of-lists used for likelihood computation),
- `Code_Models.R` and `Code_Simulation.R` (R scripts),
- `Model2.RData`, `Model3.RData` (fitted model objects).

This Python version aims to make it easy to **analyze the data and run the same models** without R.

---

## What you get in this Python port

### 1) CSVs you can load directly with pandas

All are in `data/`:

- `data_df.csv`  
  A direct CSV export of `data.df` from `DataDF.RData` (same columns as the R data.frame).

- `throws_tidy.csv`  
  A “tidy” per-throw table derived from `data_df.csv`. This is the most convenient table
  for Python analysis and reproducing the paper’s tables.

- `legs.csv`  
  One row per *(player, leg)* with number of throws/turns (within the truncated segment).

- `turns.csv`  
  One row per *(player, leg, turn)* with the within-turn success sequence and count.

- `players.csv`  
  The 73 players (sorted like the R code does: `sort(unique(data.df$name))`).

### 2) Fitted parameters as JSON (so you do **not** need to re-fit)

At the repo root:

- `model2_params.json`
- `model3_params.json`

These were extracted from `Model2.RData` / `Model3.RData` shipped in the supplement and
include player fixed effects mapped to player names.

> Refitting Model 2/3 from scratch is *computationally expensive* (the authors note that it
> can take hours/days in R). This port includes fitting code, but the JSON parameters let you
> run simulations/decoding immediately.

### 3) A small Python package implementing the models

Package: `hot_hand_darts/`

Main modules:
- `io.py` — load CSVs / load JSON params
- `preprocess.py` — build the nested list-of-legs structure (Python replacement for `data.list`)
- `grid.py` — state-space discretization and AR(1) transition matrices
- `models.py` — Model 1 (GLM), Model 2 / Model 3 likelihood + optional (slow) fitting
- `simulate.py` — empirical and simulated versions of Tables 5 & 6
- `decode.py` — Viterbi decoding for Model 3 (Figure 2-style decoded probabilities)

Examples:
- `examples/quickstart.py`

---

## Data dictionary (from the original supplement README)

The original R supplement described these `DataDF.RData` fields:

- `name`: player name  
- `pbt1`, `pbt2`, `pbt3`: points before 1st/2nd/3rd throw of the *current* turn  
- `t1`, `t2`, `t3`: segment hit with 1st/2nd/3rd throw of the *current* turn  
- `points`: points scored with the **current** throw  
- `ind`: which throw in the turn (`t1pt`, `t2pt`, `t3pt`)  
- `pbt`: points before the **current** throw  
- `wonprevious`: 1 if the player won the previous leg in the match, else 0  
- `triple`: 1 if the throw landed in **H = {T11,…,T20,bull}**, else 0  
- `TSID`: per-player leg index (time series id)

The `throws_tidy.csv` file keeps the essentials and adds:
- `throw_in_leg` (1…T), `turn_in_leg` (1…)
- `dart_in_turn` (1/2/3)
- `segment_hit` (the segment of *this* throw)

---

## Quick start

```bash
# from this folder
python -m examples.quickstart
```

Inside your own code:

```python
from hot_hand_darts.io import load_throws, load_params
from hot_hand_darts.preprocess import build_leg_sequences
from hot_hand_darts.grid import make_state_grid
from hot_hand_darts.models import negloglik_model3

throws = load_throws("data/throws_tidy.csv")
players, legs_by_player = build_leg_sequences(throws)

params3 = load_params("model3_params.json")

# Build theta_star vector in the parameterization used by the likelihood
import numpy as np
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
print("Negative log-likelihood:", nll)
```

---

## Notes on model definitions (matching the paper)

- **Outcome**: `y_t` is binary success (hit set H) in the early part of a leg (score ≥ 180).  
- **Model 1**: logistic regression with player fixed effects and within-turn dummies.  
- **Model 2**: Model 1 + latent AR(1) state `s_t` on the logit scale.  
- **Model 3**: Model 1 + *periodic* AR(1), separating within-turn vs across-turn transitions.

For Models 2/3, likelihood evaluation follows the discretization approach described in the paper
(Section 3.3) and in `Code_Models.R`:
- choose a grid `[-bm, bm]` split into `m` bins
- compute a large-state HMM transition matrix via Gaussian CDF differences
- run a scaled forward algorithm to get the (approximate) log-likelihood.

---

## Requirements

See `requirements.txt`.
