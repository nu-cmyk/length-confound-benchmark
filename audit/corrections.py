"""Length-confound corrections: Strategy I (feature residualization) and
Strategy II (data-level caliper matching).

Strategy I: cross-fitted polynomial residualization.
  We fit a Ridge regression mapping a polynomial basis of length
  Phi(L) = [L, L^2, log(L+1), sqrt(L)] to each feature dimension, then
  subtract the predicted feature from the observed feature. Cross-fitting
  ensures every training residual is evaluated out-of-sample, following
  the double machine learning protocol (Chernozhukov et al. 2018).

Strategy II: greedy nearest-neighbor caliper matching.
  Each sample in the minority class is matched to a sample in the majority
  class with the closest length, subject to a caliper |L_i - L_j| <= c
  (default c = 2 tokens, without replacement). Unmatched minority samples
  are discarded. The matched subset is balanced in length distribution
  (Length-only AUROC ~= 0.5 by construction).

A stricter variant, bin-exact matching, rounds lengths to integer-token
bins and downsamples within each bin so length-only AUROC is exactly 0.5.
This is the strongest possible length neutralization and is used for the
HaluEval sensitivity sweep in Section 7.5 of the paper.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_clean(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)


def poly_basis(L: np.ndarray) -> np.ndarray:
    """Phi(L) = [L, L^2, log(L+1), sqrt(L)] for each row."""
    L = np.asarray(L, dtype=np.float64).ravel()
    return np.column_stack([L, L ** 2, np.log(L + 1.0), np.sqrt(L)])


# ---------------------------------------------------------------------------
# Strategy I: feature-level residualization
# ---------------------------------------------------------------------------
def residualize_poly(Xtr: np.ndarray, Xte: np.ndarray,
                     Ltr: np.ndarray, Lte: np.ndarray,
                     seed: int = 42, n_splits: int = 5):
    """Cross-fitted polynomial residualization (the headline correction).

    Train fold is split into n_splits sub-folds. For each sub-fold k,
    we fit Ridge on the complement and predict on sub-fold k, ensuring
    every training residual is out-of-sample. The final Ridge for the
    test fold is trained on the entire training fold.

    Returns (Xtr_residual, Xte_residual) cleaned of NaN/inf.
    """
    kf = KFold(n_splits, shuffle=True, random_state=seed)
    Xtr_res = Xtr.copy().astype(np.float64)
    for itr, ite in kf.split(np.arange(len(Ltr))):
        ridge = Ridge(alpha=1.0).fit(poly_basis(Ltr[itr]), Xtr[itr])
        Xtr_res[ite] = Xtr[ite] - ridge.predict(poly_basis(Ltr[ite]))
    final = Ridge(alpha=1.0).fit(poly_basis(Ltr), Xtr)
    Xte_res = Xte - final.predict(poly_basis(Lte))
    return _safe_clean(Xtr_res), _safe_clean(Xte_res)


def residualize_linear(Xtr: np.ndarray, Xte: np.ndarray,
                       Ltr: np.ndarray, Lte: np.ndarray,
                       seed: int = 42, n_splits: int = 5):
    """Cross-fitted linear residualization (diagnostic, not headline).

    Used to test whether the length dependence is linear vs more complex.
    If the linear correction collapses the detector to chance but the
    polynomial correction preserves signal, the dependence is non-linear.
    """
    kf = KFold(n_splits, shuffle=True, random_state=seed)
    Xtr_res = Xtr.copy().astype(np.float64)
    Ltr_col = np.asarray(Ltr).reshape(-1, 1)
    Lte_col = np.asarray(Lte).reshape(-1, 1)
    for itr, ite in kf.split(np.arange(len(Ltr))):
        r = Ridge(alpha=1.0).fit(Ltr_col[itr], Xtr[itr])
        Xtr_res[ite] = Xtr[ite] - r.predict(Ltr_col[ite])
    final = Ridge(alpha=1.0).fit(Ltr_col, Xtr)
    Xte_res = Xte - final.predict(Lte_col)
    return _safe_clean(Xtr_res), _safe_clean(Xte_res)


# ---------------------------------------------------------------------------
# Strategy II: data-level matching
# ---------------------------------------------------------------------------
def caliper_match(lengths: np.ndarray, labels: np.ndarray,
                  caliper: float = 2.0, seed: int = 42) -> np.ndarray:
    """Greedy nearest-neighbor caliper matching by length.

    For each sample in the minority class, find the closest unmatched
    sample in the majority class with |L_i - L_j| <= caliper. Without
    replacement. Returns the deterministic matched index set M (sorted),
    or an empty array if too few matches are found.

    Args:
        lengths: per-sample length (whitespace tokens), shape (N,).
        labels:  binary labels in {0, 1}, shape (N,).
        caliper: maximum allowed |L_i - L_j| in tokens (default 2.0).
        seed:    random seed for tie-breaking ordering only.

    Returns:
        np.ndarray of integer indices forming the matched subset.
    """
    rng = np.random.RandomState(seed)
    lengths = np.asarray(lengths, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return np.array([], dtype=int)

    # Pick the smaller class as the "minority to be matched"
    if len(pos_idx) <= len(neg_idx):
        minority, majority = pos_idx, neg_idx
    else:
        minority, majority = neg_idx, pos_idx

    # Process minority in random order; majority can be matched only once
    minority = rng.permutation(minority)
    majority_avail = set(majority.tolist())
    majority_sorted = np.sort(majority)
    majority_lengths_sorted = lengths[majority_sorted]

    matched_min, matched_maj = [], []
    for m in minority:
        Lm = lengths[m]
        # Find candidate window in sorted majority lengths via searchsorted
        lo = np.searchsorted(majority_lengths_sorted, Lm - caliper, side='left')
        hi = np.searchsorted(majority_lengths_sorted, Lm + caliper, side='right')
        best_j = -1
        best_diff = np.inf
        for k in range(lo, hi):
            cand = int(majority_sorted[k])
            if cand not in majority_avail:
                continue
            diff = abs(lengths[cand] - Lm)
            if diff < best_diff:
                best_diff = diff
                best_j = cand
        if best_j >= 0:
            matched_min.append(m)
            matched_maj.append(best_j)
            majority_avail.discard(best_j)

    if not matched_min:
        return np.array([], dtype=int)

    out = np.unique(np.concatenate([matched_min, matched_maj]))
    return out.astype(int)


def bin_exact_match(lengths: np.ndarray, labels: np.ndarray,
                    seed: int = 42) -> np.ndarray:
    """Integer-token bin matching with within-bin downsampling.

    Round each length to the nearest integer, group by bin, and within
    each bin downsample both classes to the minimum count. Length-only
    AUROC on the resulting subset is exactly 0.500 by construction
    (each bin contains the same number of class-0 and class-1 samples).

    Used for the HaluEval sensitivity sweep where caliper matching
    is not strict enough to fully neutralize length.
    """
    rng = np.random.RandomState(seed)
    lengths = np.asarray(lengths, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    bins = np.floor(lengths).astype(int)

    kept = []
    for b in np.unique(bins):
        idx_b = np.where(bins == b)[0]
        idx0 = idx_b[labels[idx_b] == 0]
        idx1 = idx_b[labels[idx_b] == 1]
        n = min(len(idx0), len(idx1))
        if n == 0:
            continue
        keep0 = rng.choice(idx0, size=n, replace=False)
        keep1 = rng.choice(idx1, size=n, replace=False)
        kept.extend(keep0.tolist())
        kept.extend(keep1.tolist())

    if not kept:
        return np.array([], dtype=int)
    return np.sort(np.array(kept, dtype=int))


# ---------------------------------------------------------------------------
# Length-share statistic (Stage 3)
# ---------------------------------------------------------------------------
def length_share(auroc_raw: float, auroc_op: float,
                 min_raw: float = 0.55) -> float:
    """Length-share statistic LS = (raw - op) / (raw - 0.5).

    Reports the fraction of above-chance AUROC attributable to length.
    Returns NaN if raw AUROC is below `min_raw` to avoid denominator
    instability near chance.
    """
    if auroc_raw < min_raw:
        return float('nan')
    denom = auroc_raw - 0.5
    if denom <= 0:
        return float('nan')
    return (auroc_raw - auroc_op) / denom
