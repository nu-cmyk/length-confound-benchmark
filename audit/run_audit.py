"""Length-confound audit driver (Phase 3a + Phase 3b).

Phase 3a: per-(method, condition) AUROC under four evaluation modes:
  - raw:         no correction applied
  - stratified:  per-length-quartile AUROC then averaged
  - orth_linear: residualized against length only (linear)
  - orth_poly:   residualized against polynomial basis Phi(L) [HEADLINE]

Phase 3b: significance testing of the headline (orth_poly) metric against
the Length-only baseline. Per-condition Nadeau-Bengio corrected paired
t-tests, plus across-condition meta-analytic one-sample t-test and
Wilcoxon signed-rank test.

Cross-validation: stratified 5-fold across 3 seeds = K = 15 folds.
"""
import os
import json
import time
import sys
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from methods import METHODS, fit_and_predict
from corrections import (residualize_poly, residualize_linear,
                         poly_basis)
from significance import nadeau_bengio_test, meta_analytic_test


CACHE_DIR = './data/cache'
SELFCHECK_DIR = './data/selfcheck'
SEMANTIC_ENTROPY_DIR = './data/semantic_entropy'
OUT_DIR = './data/audit_results'
os.makedirs(OUT_DIR, exist_ok=True)

P3_OUT = f"{OUT_DIR}/phase3_main_results.json"

SEEDS = [42, 123, 456]
N_FOLDS = 5

LENGTH_ONLY_NAME = 'Length-only'

CONDITIONS = [
    ('Qwen/Qwen2.5-7B-Instruct',            'Qwen',    'triviaqa'),
    ('Qwen/Qwen2.5-7B-Instruct',            'Qwen',    'truthfulqa'),
    ('Qwen/Qwen2.5-7B-Instruct',            'Qwen',    'coqa'),
    ('Qwen/Qwen2.5-7B-Instruct',            'Qwen',    'tydiqa'),
    ('mistralai/Mistral-7B-Instruct-v0.2',  'Mistral', 'triviaqa'),
    ('mistralai/Mistral-7B-Instruct-v0.2',  'Mistral', 'truthfulqa'),
    ('mistralai/Mistral-7B-Instruct-v0.2',  'Mistral', 'coqa'),
    ('mistralai/Mistral-7B-Instruct-v0.2',  'Mistral', 'tydiqa'),
    ('meta-llama/Meta-Llama-3-8B-Instruct', 'Llama3',  'triviaqa'),
    ('meta-llama/Meta-Llama-3-8B-Instruct', 'Llama3',  'nq_open'),
    ('meta-llama/Meta-Llama-3-8B-Instruct', 'Llama3',  'halueval'),
    ('meta-llama/Meta-Llama-3-8B-Instruct', 'Llama3',  'coqa'),
    ('meta-llama/Meta-Llama-3-8B-Instruct', 'Llama3',  'tydiqa'),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_condition(model: str, dataset: str) -> Optional[dict]:
    prefix = model.replace('/', '_')
    hs_p = f"{CACHE_DIR}/{prefix}_{dataset}_rtraj_hidden.npz"
    ft_p = f"{CACHE_DIR}/{prefix}_{dataset}_rtraj_features.npz"
    if not (Path(hs_p).exists() and Path(ft_p).exists()):
        return None
    ft = np.load(ft_p, allow_pickle=True)
    lengths = np.array([len(str(r).split()) for r in ft['responses']],
                       dtype=np.float64)
    return {
        'hidden':    np.load(hs_p)['hidden_states'],
        'labels':    ft['labels'].astype(int),
        'lengths':   lengths,
        'proj_h':    ft['proj_h_reasoning'],
        'responses': ft['responses'],
    }


def load_cached_baselines(model: str, dataset: str) -> dict:
    """Load pre-computed scalar baselines (SelfCheckGPT, SemanticEntropy)."""
    prefix = model.replace('/', '_')
    out = {}
    sc_p = f"{SELFCHECK_DIR}/{prefix}_{dataset}_selfcheck.npz"
    if Path(sc_p).exists():
        d = np.load(sc_p, allow_pickle=True)
        out['SelfCheckGPT'] = {
            'scores': np.asarray(d['scores_mean'], dtype=np.float64),
            'labels': np.asarray(d['labels'], dtype=int),
        }
    se_p = f"{SEMANTIC_ENTROPY_DIR}/{prefix}_{dataset}_semantic_entropy.npz"
    if Path(se_p).exists():
        d = np.load(se_p)
        out['SemanticEntropy'] = {
            'scores': np.asarray(d['sem_entropy'], dtype=np.float64),
            'labels': np.asarray(d['labels'], dtype=int),
        }
    return out


# ---------------------------------------------------------------------------
# Phase 3a: per-method, per-condition evaluation
# ---------------------------------------------------------------------------
def evaluate_method_on_condition(d: dict, build_fn, clf_kind: str,
                                 method_name: str) -> dict:
    """Run all 4 evaluation modes (raw, stratified, orth_lin, orth_poly).

    Length-only is special-cased: only raw is computed. Residualizing length
    against itself is mathematically undefined; within-quartile evaluation
    on a feature that IS length is 0.5 by construction.
    """
    labels = d['labels']
    lengths = d['lengths']
    is_length_only = (method_name == LENGTH_ONLY_NAME)

    raw_f, strat_f, orth_lin_f, orth_poly_f = [], [], [], []
    n_attempted = 0

    for seed in SEEDS:
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
        for tri, tei in skf.split(labels, labels):
            n_attempted += 1
            if labels[tei].sum() < 2 or (1 - labels[tei]).sum() < 2:
                continue

            try:
                Xtr, Xte = build_fn(d, tri, tei, seed)
            except Exception as e:
                print(f"  [BUILD FAIL] {method_name} seed={seed}: "
                      f"{type(e).__name__}: {e}")
                continue

            # raw
            try:
                p_raw = fit_and_predict(Xtr, Xte, labels[tri], clf_kind, seed)
                raw_f.append(roc_auc_score(labels[tei], p_raw))
            except Exception as e:
                print(f"  [RAW FAIL] {method_name}: {type(e).__name__}: {e}")
                continue

            if is_length_only:
                continue

            # stratified (train-quartile edges, then evaluate on test bins)
            edges = np.quantile(lengths[tri], [0.25, 0.5, 0.75])
            q_te = np.digitize(lengths[tei], edges)
            y_te = labels[tei]
            strat = []
            for q in range(4):
                mask = q_te == q
                if mask.sum() < 10:
                    continue
                y_q = y_te[mask]
                if y_q.sum() < 2 or (1 - y_q).sum() < 2:
                    continue
                try:
                    strat.append(roc_auc_score(y_q, p_raw[mask]))
                except Exception:
                    pass
            if strat:
                strat_f.append(float(np.mean(strat)))

            # orth linear (diagnostic)
            try:
                Xtr_l, Xte_l = residualize_linear(
                    Xtr, Xte, lengths[tri], lengths[tei], seed=seed)
                p = fit_and_predict(Xtr_l, Xte_l, labels[tri], clf_kind, seed)
                orth_lin_f.append(roc_auc_score(labels[tei], p))
            except Exception as e:
                print(f"  [ORTH-LIN FAIL] {method_name}: {type(e).__name__}")

            # orth poly (HEADLINE)
            try:
                Xtr_p, Xte_p = residualize_poly(
                    Xtr, Xte, lengths[tri], lengths[tei], seed=seed)
                p = fit_and_predict(Xtr_p, Xte_p, labels[tri], clf_kind, seed)
                orth_poly_f.append(roc_auc_score(labels[tei], p))
            except Exception as e:
                print(f"  [ORTH-POLY FAIL] {method_name}: {type(e).__name__}")

    def _mean(v): return float(np.mean(v)) if v else None
    def _std(v):  return float(np.std(v))  if v else None

    return {
        'raw':         _mean(raw_f),       'raw_std':         _std(raw_f),
        'strat':       _mean(strat_f),     'strat_std':       _std(strat_f),
        'orth_linear': _mean(orth_lin_f),  'orth_linear_std': _std(orth_lin_f),
        'orth_poly':   _mean(orth_poly_f), 'orth_poly_std':   _std(orth_poly_f),
        'raw_folds':         raw_f,
        'strat_folds':       strat_f,
        'orth_linear_folds': orth_lin_f,
        'orth_poly_folds':   orth_poly_f,
        'n_folds_attempted':      n_attempted,
        'n_folds_used_raw':       len(raw_f),
        'n_folds_used_strat':     len(strat_f),
        'n_folds_used_orth_lin':  len(orth_lin_f),
        'n_folds_used_orth_poly': len(orth_poly_f),
        'n_train_per_fold': int((N_FOLDS - 1) / N_FOLDS * len(labels)),
        'n_test_per_fold':  int(len(labels) / N_FOLDS),
        'is_length_only':   bool(is_length_only),
    }


def eval_scalar_baseline_cv(scores: np.ndarray, labels: np.ndarray,
                            lengths: np.ndarray, method_name: str) -> dict:
    """CV evaluation for pre-computed scalar baselines (SelfCheck, SemEntropy).

    Residualization is applied directly to the scalar score (Strategy I).
    Sign of the score is corrected per training fold so the metric is always
    'higher = more likely hallucinated'.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    lengths = np.asarray(lengths, dtype=np.float64)

    raw_f, strat_f, orth_lin_f, orth_poly_f = [], [], [], []

    for seed in SEEDS:
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
        for tri, tei in skf.split(labels, labels):
            if labels[tei].sum() < 2 or (1 - labels[tei]).sum() < 2:
                continue
            s_tr, s_te = scores[tri], scores[tei]
            L_tr, L_te = lengths[tri], lengths[tei]
            y_tr, y_te = labels[tri], labels[tei]

            try:
                tr_auc = roc_auc_score(y_tr, s_tr)
            except Exception:
                continue
            sign = 1.0 if tr_auc >= 0.5 else -1.0
            s_tr = sign * s_tr
            s_te = sign * s_te

            try:
                raw_f.append(roc_auc_score(y_te, s_te))
            except Exception:
                continue

            edges = np.quantile(L_tr, [0.25, 0.5, 0.75])
            q_te = np.digitize(L_te, edges)
            strat = []
            for q in range(4):
                mask = q_te == q
                if mask.sum() < 10:
                    continue
                y_q = y_te[mask]
                if y_q.sum() < 2 or (1 - y_q).sum() < 2:
                    continue
                try:
                    strat.append(roc_auc_score(y_q, s_te[mask]))
                except Exception:
                    pass
            if strat:
                strat_f.append(float(np.mean(strat)))

            try:
                r_lin = Ridge(alpha=1.0).fit(L_tr.reshape(-1, 1), s_tr)
                s_te_lin = s_te - r_lin.predict(L_te.reshape(-1, 1))
                orth_lin_f.append(roc_auc_score(y_te, s_te_lin))
            except Exception:
                pass

            try:
                r_poly = Ridge(alpha=1.0).fit(poly_basis(L_tr), s_tr)
                s_te_poly = s_te - r_poly.predict(poly_basis(L_te))
                orth_poly_f.append(roc_auc_score(y_te, s_te_poly))
            except Exception:
                pass

    def _mean(v): return float(np.mean(v)) if v else None
    def _std(v):  return float(np.std(v))  if v else None

    return {
        'raw':         _mean(raw_f),       'raw_std':         _std(raw_f),
        'strat':       _mean(strat_f),     'strat_std':       _std(strat_f),
        'orth_linear': _mean(orth_lin_f),  'orth_linear_std': _std(orth_lin_f),
        'orth_poly':   _mean(orth_poly_f), 'orth_poly_std':   _std(orth_poly_f),
        'raw_folds':         raw_f,
        'strat_folds':       strat_f,
        'orth_linear_folds': orth_lin_f,
        'orth_poly_folds':   orth_poly_f,
        'n_folds_used_raw':       len(raw_f),
        'n_folds_used_orth_poly': len(orth_poly_f),
        'n_train_per_fold': int((N_FOLDS - 1) / N_FOLDS * len(labels)),
        'n_test_per_fold':  int(len(labels) / N_FOLDS),
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def save_results(results: dict, path: str):
    clean = {}
    for cond, mdict in results.items():
        clean[cond] = {}
        for method, r in mdict.items():
            if r is None:
                continue
            clean[cond][method] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in r.items()
            }
    with open(path, 'w') as f:
        json.dump(clean, f, indent=2)


# ---------------------------------------------------------------------------
# Phase 3a driver
# ---------------------------------------------------------------------------
def run_phase3a(skip_existing: bool = True) -> dict:
    if Path(P3_OUT).exists() and skip_existing:
        with open(P3_OUT) as f:
            res = json.load(f)
        print(f"Loaded existing: {P3_OUT}")
    else:
        res = {}

    t0_all = time.time()

    for i, (model, short, dataset) in enumerate(CONDITIONS):
        key = f"{short}_{dataset}"
        print(f"\n[{i+1}/{len(CONDITIONS)}] {key}")
        d = load_condition(model, dataset)
        if d is None:
            print("  MISSING (skipping)")
            continue
        if key not in res:
            res[key] = {}

        for m_name, (build_fn, clf_kind) in METHODS.items():
            done_key = 'raw' if m_name == LENGTH_ONLY_NAME else 'orth_poly'
            if (skip_existing and m_name in res[key]
                    and res[key][m_name].get(done_key) is not None):
                print(f"    {m_name:<18s}  (already done)")
                continue
            t0 = time.time()
            r = evaluate_method_on_condition(d, build_fn, clf_kind, m_name)
            res[key][m_name] = r
            fmt = lambda v: f"{v:.4f}" if v is not None else "  -   "
            print(f"    {m_name:<18s}  raw={fmt(r['raw'])}  "
                  f"strat={fmt(r['strat'])}  "
                  f"orth_lin={fmt(r['orth_linear'])}  "
                  f"orth_poly={fmt(r['orth_poly'])}  "
                  f"({r['n_folds_used_raw']}/{r['n_folds_attempted']} folds, "
                  f"{time.time()-t0:.0f}s)")
            save_results(res, P3_OUT)

        # Scalar baselines
        cached = load_cached_baselines(model, dataset)
        for name, obj in cached.items():
            if (skip_existing and name in res[key]
                    and res[key][name].get('orth_poly') is not None
                    and len(res[key][name].get('orth_poly_folds', [])) >= 10):
                continue
            r = eval_scalar_baseline_cv(obj['scores'], obj['labels'],
                                        d['lengths'], method_name=name)
            res[key][name] = r
            fmt = lambda v: f"{v:.4f}" if v is not None else "  -   "
            print(f"    {name:<18s}  raw={fmt(r['raw'])}  "
                  f"orth_poly={fmt(r['orth_poly'])}  (15-fold CV)")
            save_results(res, P3_OUT)

    print(f"\nTotal Phase 3a: {(time.time() - t0_all) / 60:.1f} min")
    save_results(res, P3_OUT)
    return res


# ---------------------------------------------------------------------------
# Phase 3b: significance testing against length-only floor
# ---------------------------------------------------------------------------
def run_phase3b(results_path: str = P3_OUT,
                reference_name: str = LENGTH_ONLY_NAME,
                headline_mode: str = 'orth_poly'):
    with open(results_path) as f:
        res = json.load(f)

    compared = [m for m in METHODS if m != reference_name] + [
        'SelfCheckGPT', 'SemanticEntropy']

    print("=" * 130)
    print(f"Phase 3b: each method's {headline_mode} AUROC vs "
          f"{reference_name}'s raw AUROC")
    print("=" * 130)
    print(f"  {'Condition':<22s}  "
          + "  ".join(f"{m[:14]:>14s}" for m in compared))

    agg = {m: {'deltas': [], 'wins': 0, 'losses': 0, 'sig_wins': 0}
           for m in compared}

    for cond, mdict in res.items():
        if reference_name not in mdict:
            continue
        ref = mdict[reference_name]
        ref_folds = ref.get('raw_folds', [])
        if len(ref_folds) < 3:
            continue
        n_tr = ref.get('n_train_per_fold', 0)
        n_te = ref.get('n_test_per_fold', 1)

        cells = []
        for m in compared:
            if m not in mdict:
                cells.append("     n/a    ")
                continue
            m_folds = mdict[m].get(f'{headline_mode}_folds', [])
            if len(m_folds) == len(ref_folds) and len(m_folds) >= 3:
                t, p = nadeau_bengio_test(m_folds, ref_folds, n_tr, n_te,
                                          alternative='greater')
                delta = float(np.mean(m_folds) - np.mean(ref_folds))
                stars = ('***' if p < 0.001 else '**' if p < 0.01
                         else '*' if p < 0.05 else '')
                cells.append(f"{delta:+.3f}{stars:<3s}  ")
                agg[m]['deltas'].append(delta)
                if delta > 0.0005:
                    agg[m]['wins'] += 1
                    if p < 0.05:
                        agg[m]['sig_wins'] += 1
                elif delta < -0.0005:
                    agg[m]['losses'] += 1
            elif len(m_folds) >= 1:
                delta = float(np.mean(m_folds) - np.mean(ref_folds))
                cells.append(f"{delta:+.3f}    ")
                agg[m]['deltas'].append(delta)
            else:
                cells.append("     n/a    ")

        print(f"  {cond:<22s}  " + "  ".join(f"{c:>14s}" for c in cells))

    print("\n" + "=" * 110)
    print(f"ACROSS-CONDITION TEST: {headline_mode} vs {reference_name} raw")
    print("=" * 110)
    print(f"  {'Method':<18s}  {'n':>3s}  {'mean Δ':>9s}  {'t':>7s}  "
          f"{'p (t)':>10s}  {'Cohen d':>9s}  {'p (Wilcox)':>11s}  "
          f"{'wins':>5s}  {'sig w':>6s}  {'losses':>7s}")
    for m, a in agg.items():
        deltas = a['deltas']
        if len(deltas) < 3:
            continue
        t_stat, p_t, eff = meta_analytic_test(deltas, 'greater', 't')
        _, p_w, _ = meta_analytic_test(deltas, 'greater', 'wilcoxon')
        print(f"  {m:<18s}  {len(deltas):>3d}  {np.mean(deltas):>+9.4f}  "
              f"{t_stat:>7.2f}  {p_t:>10.4g}  {eff:>+9.2f}  {p_w:>11.4g}  "
              f"{a['wins']:>5d}  {a['sig_wins']:>6d}  {a['losses']:>7d}")

    print("\n  *** p<0.001  ** p<0.01  * p<0.05")
    print("  Per-condition test: Nadeau-Bengio corrected paired t.")
    print("  Across-condition: one-sample t-test + Wilcoxon signed-rank.")


if __name__ == '__main__':
    results = run_phase3a(skip_existing=True)
    run_phase3b()
