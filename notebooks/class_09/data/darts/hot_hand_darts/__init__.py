"""Hot hand in professional darts (Otting et al.) — Python port.

This package is a lightweight, dependency-minimal port of the R code shipped with the
JRSS-A paper:

    M. Otting, R. Langrock, C. Deutscher, V. Leos-Barajas (2020)
    "The hot hand in professional darts", JRSS-A 183(2), 565–580.

Core ideas:
- Binary success outcome: whether a throw hits H={T11..T20, bull's-eye} while score>=180.
- Model 1: logistic regression with player fixed effects + within-turn dummies.
- Model 2: Model 1 + latent AR(1) state (continuous) on logit scale; likelihood via grid
  discretization (Kitagawa-style) -> HMM forward algorithm.
- Model 3: Model 1 + periodic AR(1) separating within-turn vs across-turn transitions.

See README_PYTHON.md in the repository root for a quick start.
"""

from .io import load_throws, load_players, load_params
from .preprocess import build_leg_sequences
from .models import (
    fit_model1,
    negloglik_model2,
    negloglik_model3,
)
from .decode import viterbi_decode_model3
