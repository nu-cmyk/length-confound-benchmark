"""Significance tests for the length-confound audit.

Two levels of testing are reported:

1. Per-condition Nadeau-Bengio corrected paired t-test.
   Standard k-fold cross-validation produces fold-wise AUROC estimates that
   are NOT independent (training folds overlap). The Nadeau-Bengio correction
   inflates the variance estimate by (1/n + n_test/n_train) to account for
   this overlap, preventing inflated false-positive rates.

   Nadeau, C. and Bengio, Y. (2003). "Inference for the generalization error."
   Machine Learning 52(3): 239-281.

2. Across-condition meta-analytic tests.
   We aggregate per-condition deltas (corrected_AUROC - length_only_AUROC)
   across the 13 (model, dataset) conditions via:
     - one-sample t-test against the null Delta = 0 (parametric)
     - Wilcoxon signed-rank test (non-parametric robustness check)
   A detector passes the audit only if BOTH yield p < 0.05.

   With Bonferroni correction across k = 8 audited detectors (excluding the
   length-only reference floor), the threshold tightens to alpha/8 = 0.00625.
"""
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Per-condition Nadeau-Bengio
# ---------------------------------------------------------------------------
def nadeau_bengio_test(folds_a: list, folds_b: list,
                       n_train: int, n_test: int,
                       alternative: str = 'greater'):
    """Nadeau-Bengio corrected paired t-test for k-fold CV AUROC differences.

    Args:
        folds_a: per-fold AUROCs for the method being tested.
        folds_b: per-fold AUROCs for the reference baseline (length-only).
        n_train: number of training samples per fold.
        n_test:  number of test samples per fold.
        alternative: 'greater', 'less', or 'two-sided'.

    Returns:
        (t_statistic, p_value), or (nan, nan) if the test cannot be computed.
    """
    a = np.asarray(folds_a, dtype=np.float64)
    b = np.asarray(folds_b, dtype=np.float64)
    if len(a) != len(b) or len(a) < 2:
        return float('nan'), float('nan')

    diffs = a - b
    n = len(diffs)
    var_d = diffs.var(ddof=1)
    if var_d <= 0:
        return float('nan'), float('nan')

    # Nadeau-Bengio variance correction: (1/n + n_test/n_train) * Var(Delta)
    se = np.sqrt(((1.0 / n) + (n_test / max(n_train, 1))) * var_d)
    if se <= 0:
        return float('nan'), float('nan')

    t = diffs.mean() / se
    df = n - 1
    if alternative == 'greater':
        p = stats.t.sf(t, df)
    elif alternative == 'less':
        p = stats.t.cdf(t, df)
    else:
        p = 2.0 * stats.t.sf(abs(t), df)
    return float(t), float(p)


# ---------------------------------------------------------------------------
# Across-condition meta-analytic tests
# ---------------------------------------------------------------------------
def meta_analytic_test(deltas: list,
                       alternative: str = 'greater',
                       method: str = 't'):
    """Across-condition aggregation against the null Delta = 0.

    Args:
        deltas: per-condition Delta values (corrected - reference AUROC).
        alternative: 'greater', 'less', or 'two-sided'.
        method:  't' for one-sample t-test, 'wilcoxon' for signed-rank.

    Returns:
        (statistic, p_value, effect_size):
          For 't': effect_size is Cohen's d (mean / std).
          For 'wilcoxon': effect_size is the median delta.
        Returns (nan, nan, nan) if fewer than 3 valid conditions.
    """
    d = np.asarray(deltas, dtype=np.float64)
    d = d[~np.isnan(d)]
    if len(d) < 3:
        return float('nan'), float('nan'), float('nan')

    if method == 'wilcoxon':
        try:
            stat, p = stats.wilcoxon(d, alternative=alternative)
        except ValueError:
            stat, p = float('nan'), float('nan')
        eff = float(np.median(d))
    else:
        stat, p = stats.ttest_1samp(d, 0.0, alternative=alternative)
        eff = float(d.mean() / max(d.std(ddof=1), 1e-9))

    return float(stat), float(p), eff


# ---------------------------------------------------------------------------
# Verdict helper
# ---------------------------------------------------------------------------
def dual_correction_verdict(orth_deltas: list, matched_deltas: list,
                            alpha: float = 0.05):
    """Combine the two corrections into a single PASS/FAIL verdict.

    A detector PASSES if BOTH corrections yield Delta > 0 and p < alpha
    under both the t-test and the Wilcoxon signed-rank test.

    Args:
        orth_deltas:    per-condition deltas under polynomial orthogonalization.
        matched_deltas: per-condition deltas under caliper matching.
        alpha:          significance threshold (default 0.05).

    Returns:
        dict with keys 'orth_pass', 'matched_pass', 'overall_pass',
        plus the underlying p-values and effect sizes for transparency.
    """
    _t_o, p_t_o, d_o = meta_analytic_test(orth_deltas, 'greater', 't')
    _w_o, p_w_o, _med_o = meta_analytic_test(orth_deltas, 'greater', 'wilcoxon')
    _t_m, p_t_m, d_m = meta_analytic_test(matched_deltas, 'greater', 't')
    _w_m, p_w_m, _med_m = meta_analytic_test(matched_deltas, 'greater', 'wilcoxon')

    orth_pass = (np.mean(orth_deltas) > 0
                 and p_t_o < alpha and p_w_o < alpha)
    matched_pass = (np.mean(matched_deltas) > 0
                    and p_t_m < alpha and p_w_m < alpha)
    return {
        'orth_pass':       bool(orth_pass),
        'matched_pass':    bool(matched_pass),
        'overall_pass':    bool(orth_pass and matched_pass),
        'orth_mean_delta':  float(np.mean(orth_deltas))
                            if len(orth_deltas) else float('nan'),
        'orth_cohens_d':    d_o,
        'orth_p_t':         p_t_o,
        'orth_p_wilcox':    p_w_o,
        'matched_mean_delta': float(np.mean(matched_deltas))
                              if len(matched_deltas) else float('nan'),
        'matched_cohens_d': d_m,
        'matched_p_t':      p_t_m,
        'matched_p_wilcox': p_w_m,
    }
