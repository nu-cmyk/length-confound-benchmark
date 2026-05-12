"""Regenerate Tables 1-5 from the cached audit results.

Inputs:
  ./data/audit_results/phase3_main_results.json (produced by audit/run_audit.py)
  ./data/cache/*_rtraj_features.npz             (for Table 1 length statistics)

Outputs (printed to stdout, also saved as markdown):
  Table 1: per-condition length statistics + length-only AUROC
  Table 2: per-method length-share statistics
  Table 3: per-family length-share statistics
  Table 4: across-condition orthogonalization-stage verdict
  Table 5: dual-correction AUROC convergence

Run:
  python analysis/build_tables.py

Tables are printed to stdout and also saved to ./data/audit_results/tables.md
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'audit'))
from corrections import length_share
from significance import meta_analytic_test


CACHE_DIR = './data/cache'
AUDIT_RESULTS = './data/audit_results/phase3_main_results.json'
OUT_MD = './data/audit_results/tables.md'


# Order of conditions for table display (matches paper Table 1).
CONDITION_ORDER = [
    ('Qwen',    'triviaqa',    'Qwen2.5-7B',  'TriviaQA',
     'Qwen/Qwen2.5-7B-Instruct'),
    ('Qwen',    'truthfulqa',  'Qwen2.5-7B',  'TruthfulQA',
     'Qwen/Qwen2.5-7B-Instruct'),
    ('Qwen',    'coqa',        'Qwen2.5-7B',  'CoQA',
     'Qwen/Qwen2.5-7B-Instruct'),
    ('Qwen',    'tydiqa',      'Qwen2.5-7B',  'TyDiQA-GP',
     'Qwen/Qwen2.5-7B-Instruct'),
    ('Mistral', 'triviaqa',    'Mistral-7B',  'TriviaQA',
     'mistralai/Mistral-7B-Instruct-v0.2'),
    ('Mistral', 'truthfulqa',  'Mistral-7B',  'TruthfulQA',
     'mistralai/Mistral-7B-Instruct-v0.2'),
    ('Mistral', 'coqa',        'Mistral-7B',  'CoQA',
     'mistralai/Mistral-7B-Instruct-v0.2'),
    ('Mistral', 'tydiqa',      'Mistral-7B',  'TyDiQA-GP',
     'mistralai/Mistral-7B-Instruct-v0.2'),
    ('Llama3',  'triviaqa',    'Llama-3-8B',  'TriviaQA',
     'meta-llama/Meta-Llama-3-8B-Instruct'),
    ('Llama3',  'nq_open',     'Llama-3-8B',  'NQ-Open',
     'meta-llama/Meta-Llama-3-8B-Instruct'),
    ('Llama3',  'halueval',    'Llama-3-8B',  'HaluEval',
     'meta-llama/Meta-Llama-3-8B-Instruct'),
    ('Llama3',  'coqa',        'Llama-3-8B',  'CoQA',
     'meta-llama/Meta-Llama-3-8B-Instruct'),
    ('Llama3',  'tydiqa',      'Llama-3-8B',  'TyDiQA-GP',
     'meta-llama/Meta-Llama-3-8B-Instruct'),
]

METHOD_FAMILY = {
    'XGB-AllLayers':    'Multi-layer probe',
    'HARP':             'Multi-layer probe',
    'SAPLMA':           'Single-layer probe',
    'MiddleLayerProbe': 'Single-layer probe',
    'HaloScope':        'Spectral',
    'LayerCovariance':  'Spectral',
    'SelfCheckGPT':     'Generation-based',
    'SemanticEntropy':  'Generation-based',
}


# ---------------------------------------------------------------------------
# Table 1: length-only baseline statistics
# ---------------------------------------------------------------------------
def build_table_1() -> str:
    rows = []
    rows.append("## Table 1: Per-condition length statistics + length-only AUROC\n")
    rows.append("| Model | Dataset | N | P(y=1) | L0 | L1 | dL | AUROC(L) |")
    rows.append("|---|---|---:|---:|---:|---:|---:|---:|")

    with open(AUDIT_RESULTS) as f:
        res = json.load(f)

    for short, ds, model_disp, ds_disp, model_full in CONDITION_ORDER:
        key = f"{short}_{ds}"
        prefix = model_full.replace('/', '_')
        ft_path = f"{CACHE_DIR}/{prefix}_{ds}_rtraj_features.npz"
        if not Path(ft_path).exists():
            continue
        ft = np.load(ft_path, allow_pickle=True)
        labels = ft['labels'].astype(int)
        responses = ft['responses']
        lengths = np.array([len(str(r).split()) for r in responses],
                           dtype=np.float64)
        N = len(labels)
        p_pos = float(labels.mean())
        L0 = float(lengths[labels == 0].mean()) if (labels == 0).any() else 0.0
        L1 = float(lengths[labels == 1].mean()) if (labels == 1).any() else 0.0
        dL = L1 - L0
        len_auc = None
        if key in res and 'Length-only' in res[key]:
            len_auc = res[key]['Length-only'].get('raw')
        len_auc_str = f"{len_auc:.3f}" if len_auc is not None else "n/a"
        if len_auc is not None and len_auc >= 0.65:
            len_auc_str = f"**{len_auc_str}**"
        rows.append(f"| {model_disp} | {ds_disp} | {N} | {p_pos:.2f} | "
                    f"{L0:.1f} | {L1:.1f} | {dL:+.1f} | {len_auc_str} |")
    rows.append("\nBold marks AUROC(L) >= 0.65 (where length contamination "
                "materially distorts standard metrics).\n")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Table 2: per-method length-share
# ---------------------------------------------------------------------------
def build_table_2() -> str:
    rows = []
    rows.append("## Table 2: Per-method length-share statistics\n")
    rows.append("| Method | n | Mean LS | Median LS | Max LS |")
    rows.append("|---|---:|---:|---:|---:|")

    with open(AUDIT_RESULTS) as f:
        res = json.load(f)

    by_method: Dict[str, list] = {}
    for cond, mdict in res.items():
        for method, r in mdict.items():
            if method == 'Length-only':
                continue
            raw = r.get('raw')
            op = r.get('orth_poly')
            if raw is None or op is None:
                continue
            ls = length_share(raw, op)
            if np.isnan(ls):
                continue
            by_method.setdefault(method, []).append(ls)

    order = ['XGB-AllLayers', 'MiddleLayerProbe', 'HARP', 'SAPLMA',
             'SelfCheckGPT', 'SemanticEntropy',
             'LayerCovariance', 'HaloScope']
    for m in order:
        if m not in by_method:
            continue
        vals = by_method[m]
        rows.append(f"| {m} | {len(vals)} | "
                    f"{100 * np.mean(vals):.1f}% | "
                    f"{100 * np.median(vals):.1f}% | "
                    f"{100 * np.max(vals):.1f}% |")
    rows.append("")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Table 3: per-family length-share
# ---------------------------------------------------------------------------
def build_table_3() -> str:
    rows = []
    rows.append("## Table 3: Per-family length-share statistics\n")
    rows.append("| Family | n | Mean LS | Median LS | Max LS |")
    rows.append("|---|---:|---:|---:|---:|")

    with open(AUDIT_RESULTS) as f:
        res = json.load(f)

    by_family: Dict[str, list] = {}
    for cond, mdict in res.items():
        for method, r in mdict.items():
            fam = METHOD_FAMILY.get(method)
            if fam is None:
                continue
            raw = r.get('raw')
            op = r.get('orth_poly')
            if raw is None or op is None:
                continue
            ls = length_share(raw, op)
            if np.isnan(ls):
                continue
            by_family.setdefault(fam, []).append(ls)

    order = ['Multi-layer probe', 'Single-layer probe',
             'Generation-based', 'Spectral']
    for fam in order:
        if fam not in by_family:
            continue
        vals = by_family[fam]
        rows.append(f"| {fam} | {len(vals)} | "
                    f"{100 * np.mean(vals):.1f}% | "
                    f"{100 * np.median(vals):.1f}% | "
                    f"{100 * np.max(vals):.1f}% |")
    rows.append("")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Table 4: orthogonalization-stage verdict
# ---------------------------------------------------------------------------
def build_table_4() -> str:
    rows = []
    rows.append("## Table 4: Across-condition orthogonalization verdict\n")
    rows.append("| Method | n | Mean dlt | t | p | Cohen d | p(Wilcox) |")
    rows.append("|---|---:|---:|---:|---:|---:|---:|")

    with open(AUDIT_RESULTS) as f:
        res = json.load(f)

    deltas_by_method: Dict[str, list] = {}
    for cond, mdict in res.items():
        if 'Length-only' not in mdict:
            continue
        ref_folds = mdict['Length-only'].get('raw_folds', [])
        if len(ref_folds) < 3:
            continue
        ref_mean = float(np.mean(ref_folds))
        for method, r in mdict.items():
            if method == 'Length-only':
                continue
            op_folds = r.get('orth_poly_folds', [])
            if len(op_folds) < 3:
                continue
            delta = float(np.mean(op_folds)) - ref_mean
            deltas_by_method.setdefault(method, []).append(delta)

    order = ['XGB-AllLayers', 'HARP', 'MiddleLayerProbe', 'SAPLMA',
             'SelfCheckGPT', 'SemanticEntropy',
             'LayerCovariance', 'HaloScope']
    for m in order:
        if m not in deltas_by_method:
            continue
        deltas = deltas_by_method[m]
        t, p_t, eff = meta_analytic_test(deltas, 'greater', 't')
        _, p_w, _ = meta_analytic_test(deltas, 'greater', 'wilcoxon')
        rows.append(f"| {m} | {len(deltas)} | {np.mean(deltas):+.4f} | "
                    f"{t:.2f} | {p_t:.4g} | {eff:+.2f} | {p_w:.4g} |")
    rows.append("")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Table 5: dual-correction AUROC convergence
# ---------------------------------------------------------------------------
def build_table_5() -> str:
    rows = []
    rows.append("## Table 5: AUROC under raw vs orth_poly\n")
    rows.append("| Method | RAW | ORTH_POLY | dlt R->O |")
    rows.append("|---|---:|---:|---:|")

    with open(AUDIT_RESULTS) as f:
        res = json.load(f)

    by_method: Dict[str, dict] = {}
    for cond, mdict in res.items():
        for method, r in mdict.items():
            if method == 'Length-only':
                continue
            raw = r.get('raw')
            op = r.get('orth_poly')
            if raw is None or op is None:
                continue
            by_method.setdefault(method, {'raw': [], 'op': []})
            by_method[method]['raw'].append(raw)
            by_method[method]['op'].append(op)

    order = ['XGB-AllLayers', 'HARP', 'MiddleLayerProbe', 'SAPLMA',
             'LayerCovariance', 'HaloScope',
             'SelfCheckGPT', 'SemanticEntropy']
    for m in order:
        if m not in by_method:
            continue
        raw_mean = float(np.mean(by_method[m]['raw']))
        op_mean = float(np.mean(by_method[m]['op']))
        rows.append(f"| {m} | {raw_mean:.3f} | {op_mean:.3f} | "
                    f"{op_mean - raw_mean:+.3f} |")
    rows.append("")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    if not Path(AUDIT_RESULTS).exists():
        print(f"ERROR: {AUDIT_RESULTS} not found.")
        print("Run `python audit/run_audit.py` first to produce the JSON.")
        sys.exit(1)

    sections = [
        build_table_1(),
        build_table_2(),
        build_table_3(),
        build_table_4(),
        build_table_5(),
    ]
    out = "\n\n".join(sections)

    print(out)

    os.makedirs(Path(OUT_MD).parent, exist_ok=True)
    with open(OUT_MD, 'w') as f:
        f.write(out)
    print(f"\n\nSaved markdown to {OUT_MD}")


if __name__ == '__main__':
    main()
