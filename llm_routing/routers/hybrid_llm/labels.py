"""Training labels for the three Hybrid LLM routers (Ding et al., ICLR 2024, Sec. 3).

Notation (paper):
    S(x), L(x)       responses of the small / large model to query x
    q(.)             response quality (BARTScore)
    H(x) = q(S(x)) - q(L(x))       the *quality gap* (random: decoding is stochastic)

The router learns p_w(x) ~ Pr[small model is "good enough" on x]. The three
variants differ only in how "good enough" becomes a training label:

r_det   (deterministic)  y = 1[ q(S(x)) >= q(L(x)) ]     from ONE sample each
r_prob  (probabilistic)  y = Pr[ H(x) >= 0 ]             estimated from k samples each
r_trans (transformed)    y = Pr[ H(x) >= -t* ]           relax by t*, chosen automatically

Why r_trans? When the small model is much weaker, Pr[H >= 0] is ~0 for almost
every query -> nearly all labels are 0 -> the classifier learns nothing useful
(severe class imbalance). Allowing the small model to be "up to t worse" lifts
the labels. t* is picked to maximise how spread out the labels are:

    t* = argmax_t  (1/N^2) sum_{i,j} | y_i(t) - y_j(t) |

At t = 0 almost all labels are 0, at t = +inf all are 1; both have zero spread.
In between, the spread peaks when labels are most informative.
"""

from __future__ import annotations

import numpy as np


def deterministic_labels(small_scores: np.ndarray, large_scores: np.ndarray) -> np.ndarray:
    """y_i = 1[ q(S(x_i)) >= q(L(x_i)) ] using the first sample of each model."""
    s, l = np.atleast_2d(small_scores), np.atleast_2d(large_scores)
    return (s[:, 0] >= l[:, 0]).astype(float)


def gap_probability(small_scores: np.ndarray, large_scores: np.ndarray, t: float = 0.0,
                    estimator: str = "all_pairs") -> np.ndarray:
    """Estimate Pr[ H(x) >= -t ] per query from k samples of each model.

    estimator
        "all_pairs": compare every small sample with every large sample (k*k
                     comparisons). Unbiased for independent samples, lower variance.
        "paired":    compare sample j of the small model with sample j of the
                     large model only (k comparisons).
    """
    s, l = np.atleast_2d(small_scores), np.atleast_2d(large_scores)
    if estimator == "all_pairs":
        diff = s[:, :, None] - l[:, None, :]  # (N, k_s, k_l)
        return (diff >= -t).mean(axis=(1, 2))
    if estimator == "paired":
        k = min(s.shape[1], l.shape[1])
        return (s[:, :k] - l[:, :k] >= -t).mean(axis=1)
    raise ValueError(estimator)


def probabilistic_labels(small_scores, large_scores, estimator: str = "all_pairs") -> np.ndarray:
    return gap_probability(small_scores, large_scores, 0.0, estimator)


def label_spread(y: np.ndarray) -> float:
    """(1/N^2) * sum_{i,j} |y_i - y_j|, computed in O(N log N) via sorting.

    For sorted y_(0) <= ... <= y_(N-1):  sum_{i<j} (y_(j) - y_(i)) = sum_k y_(k) * (2k - N + 1).
    The ordered-pair sum is twice that.
    """
    y = np.sort(np.asarray(y, dtype=float))
    n = len(y)
    if n == 0:
        return 0.0
    k = np.arange(n)
    return float(2.0 * np.sum(y * (2 * k - n + 1)) / n**2)


def find_relaxation(small_scores, large_scores, num_grid: int = 200, estimator: str = "all_pairs"):
    """Grid-search t* >= 0 maximising label spread. Returns (t_star, ts, spreads)."""
    s, l = np.atleast_2d(small_scores), np.atleast_2d(large_scores)
    max_gap = float(np.max(l[:, :, None] - s[:, None, :]))
    hi = max(max_gap, 1e-6)
    ts = np.linspace(0.0, hi, num_grid)
    spreads = np.array([label_spread(gap_probability(s, l, t, estimator)) for t in ts])
    return float(ts[int(np.argmax(spreads))]), ts, spreads


def transformed_labels(small_scores, large_scores, t: float | None = None, estimator: str = "all_pairs"):
    """Returns (labels, t). If t is None, it is chosen with ``find_relaxation``."""
    if t is None:
        t, _, _ = find_relaxation(small_scores, large_scores, estimator=estimator)
    return gap_probability(small_scores, large_scores, t, estimator), t
