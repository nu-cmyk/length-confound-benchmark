"""HaluEval length-matching sensitivity sweep (Section 7.5, Table 6).

HaluEval has the most extreme length confound in the benchmark (length-only
AUROC = 0.969 on Llama-3-8B). We progressively tighten the length-matching
constraint from c = 2 tokens down to c = 0 (bin-exact), which forces the
length-only AUROC on the matched subset to exactly 0.500.

Detectors whose AUROC survives bin-exact matching cannot be relying on
length. This is the strongest single-condition proof in the paper that
probe-based detectors encode genuine non-length signal while spectral
detectors substantially exploit the length artifact.

Output:
  Table 6-style summary printed to stdout and saved as markdown.

Run:
  python analysis/halueval_sensitivity.py
"""
import os
import sys
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'audit'))
from methods import METHODS, fit_and_predict
from corrections import caliper_match, bin_exact_match


CACHE_DIR = './data/cache'
OUT_MD = './data/audit_results/halueval_sensitivity.md'

MODEL = 'meta-llama/Meta-Llama-3-8B-Instruct'
DATASET = 'halueval'
SEEDS = [42, 123, 456]
N_FOLDS = 5

# Calipers to sweep (in whitespace tokens). c=None means no matching (RAW).
# Bin-exact is a special last entry that forces length AUROC to exactly 0.5.
SWEEP = [
    ('RAW (FULL)',     None,       False),
    ('Caliper c=2.0',  2.0,        False),
    ('Caliper c=1.0',  1.0,        False),
    ('Caliper c=0.5',  0.5,        False),
    ('BIN-EXACT',      None,       True),
]


def load_halueval():
    prefix = MODEL.replace('/', '_')
    hs = f"{CACHE_DIR}/{prefix}_{DATASET}_rtraj_hidden.npz"
    ft = f"{CACHE_DIR}/{prefix}_{DATASET}_rtraj_features.npz"
    if not (Path(hs).exists() and Path(ft).exists()):
        print(f"ERROR: missing HaluEval cache. Run extract_halueval.py first.")
        sys.exit(1)
    ftd = np.load(ft, allow_pickle=True)
    lengths = np.array(
        [len(str(r).split()) for r in ftd['responses']], dtype=np.float64)
    return {
        'hidden':    np.load(hs)['hidden_states'],
        'labels':    ftd['labels'].astype(int),
        'lengths':   lengths,
        'proj_h':    ftd['proj_h_reasoning'],
        'responses': ftd['responses'],
    }


def cv_auroc_on_subset(d_full: dict, subset_idx: np.ndarray,
                       build_fn, clf_kind: str, method_name: str):
    """Standard 15-fold CV AUROC restricted to a matched subset."""
    if len(subset_idx) < 100:
        return None, None

    # Build a sub-data dict viewed via fancy indexing
    sub = {
        'hidden':    d_full['hidden'][subset_idx],
        'labels':    d_full['labels'][subset_idx],
        'lengths':   d_full['lengths'][subset_idx],
        'proj_h':    d_full['proj_h'][subset_idx],
        'responses': d_full['responses'][subset_idx],
    }
    labels = sub['labels']
    if labels.sum() < 10 or (1 - labels).sum() < 10:
        return None, None

    aurocs = []
    for seed in SEEDS:
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
        for tri, tei in skf.split(labels, labels):
            if labels[tei].sum() < 2 or (1 - labels[tei]).sum() < 2:
                continue
            try:
                Xtr, Xte = build_fn(sub, tri, tei, seed)
                p = fit_and_predict(Xtr, Xte, labels[tri], clf_kind, seed)
                aurocs.append(roc_auc_score(labels[tei], p))
            except Exception as e:
                print(f"    [{method_name} seed={seed}] FAIL: {e}")
    if not aurocs:
        return None, None
    return float(np.mean(aurocs)), float(np.std(aurocs))


def length_auroc(d: dict, subset_idx: np.ndarray) -> float:
    """Logistic regression on length alone, restricted to the matched subset."""
    from sklearn.linear_model import LogisticRegression
    if len(subset_idx) < 100:
        return float('nan')
    L = d['lengths'][subset_idx].reshape(-1, 1)
    y = d['labels'][subset_idx]
    if y.sum() < 5 or (1 - y).sum() < 5:
        return float('nan')
    aurocs = []
    for seed in SEEDS:
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
        for tri, tei in skf.split(L, y):
            if y[tei].sum() < 2 or (1 - y[tei]).sum() < 2:
                continue
            m = LogisticRegression(class_weight='balanced',
                                   random_state=seed,
                                   max_iter=1000).fit(L[tri], y[tri])
            aurocs.append(roc_auc_score(y[tei], m.predict_proba(L[tei])[:, 1]))
    return float(np.mean(aurocs)) if aurocs else float('nan')


def main():
    d = load_halueval()
    print(f"HaluEval ({MODEL.split('/')[-1]}): N={len(d['labels'])}, "
          f"halluc={int(d['labels'].sum())}\n")

    audit_methods = ['SAPLMA', 'HaloScope', 'HARP', 'XGB-AllLayers',
                     'MiddleLayerProbe', 'LayerCovariance']

    rows_md = ["## HaluEval matching sensitivity (Table 6 of the paper)\n"]
    header = "| Variant | n | len-AUC |"
    sep = "|---|---:|---:|"
    for m in audit_methods:
        header += f" {m} |"
        sep += "---:|"
    rows_md.append(header)
    rows_md.append(sep)

    raw_aurocs = {}

    for label, caliper, bin_exact in SWEEP:
        if caliper is None and not bin_exact:
            subset = np.arange(len(d['labels']))
        elif bin_exact:
            subset = bin_exact_match(d['lengths'], d['labels'], seed=42)
        else:
            subset = caliper_match(d['lengths'], d['labels'],
                                   caliper=caliper, seed=42)

        n = len(subset)
        len_auc = length_auroc(d, subset)
        print(f"--- {label} (n_matched={n}, length-AUC={len_auc:.4f}) ---")

        row = f"| {label} | {n} | {len_auc:.3f} |"
        for m in audit_methods:
            build_fn, clf_kind = METHODS[m]
            mean_auc, std_auc = cv_auroc_on_subset(
                d, subset, build_fn, clf_kind, m)
            if mean_auc is None:
                row += " n/a |"
                print(f"  {m:<20s} : n/a")
            else:
                row += f" {mean_auc:.3f} |"
                print(f"  {m:<20s} : {mean_auc:.4f}+/-{std_auc:.4f}")
                if label == 'RAW (FULL)':
                    raw_aurocs[m] = mean_auc
        rows_md.append(row)

    # Final delta row vs RAW
    if 'BIN-EXACT' in [s[0] for s in SWEEP] and raw_aurocs:
        delta_row = "| dlt vs Raw (Bin-exact) | | |"
        bin_label = 'BIN-EXACT'
        # Re-derive bin-exact aurocs by replaying
        subset = bin_exact_match(d['lengths'], d['labels'], seed=42)
        for m in audit_methods:
            build_fn, clf_kind = METHODS[m]
            mean_auc, _ = cv_auroc_on_subset(
                d, subset, build_fn, clf_kind, m)
            if mean_auc is None or m not in raw_aurocs:
                delta_row += " n/a |"
            else:
                delta_row += f" {mean_auc - raw_aurocs[m]:+.3f} |"
        rows_md.append(delta_row)

    os.makedirs(Path(OUT_MD).parent, exist_ok=True)
    out = "\n".join(rows_md)
    with open(OUT_MD, 'w') as f:
        f.write(out)
    print(f"\nSaved: {OUT_MD}")
    print(out)


if __name__ == '__main__':
    main()
