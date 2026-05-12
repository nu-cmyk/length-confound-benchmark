"""Detector feature builders and classifier registry.

Each detector is defined by:
  - a feature builder: (data, train_idx, test_idx, seed) -> (X_train, X_test)
  - a classifier kind: 'lr' (LogisticRegression), 'mlp' (PyTorch MLP), or
    'xgb' (XGBoost on GPU)

The audit operates on the FEATURE representation X built by each detector
(substrate-level evaluation). The classifier is retrained from scratch after
any correction so the original classifier weights are never reused on
residualized features.

Detectors:
  Length-only       : balanced LR on the length scalar L (the cheating floor)
  SAPLMA            : MLP on PCA-128(last-layer hidden state)
  HaloScope         : LR on TruncatedSVD top-3 of last-3-layers concatenated
  HARP              : XGB on PCA-200(V_R-projected per-layer features)
  XGB-AllLayers     : XGB on PCA-200(all-layer flattened hidden states)
  LayerCovariance   : LR on per-sample log-eigenspectrum of HiH^T/D
  MiddleLayerProbe  : MLP on PCA-128(middle-layer hidden state)

Note: HaloScope-simplified and LayerCovariance are single-response adaptations
of the original INSIDE/HaloScope methods, which use K=10 sampled generations.
See Section 4 and Appendix H of the paper.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _safe_clean(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)


def _balanced_resample(X: np.ndarray, y: np.ndarray, seed: int):
    """Oversample minority class to match majority."""
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    n_max = int(counts.max())
    parts = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        if len(idx_c) < n_max:
            idx_c = rng.choice(idx_c, size=n_max, replace=True)
        parts.append(idx_c)
    indices = np.concatenate(parts)
    rng.shuffle(indices)
    return X[indices], y[indices]


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------
def build_length_only(d, tri, tei, seed):
    """Length-only baseline. The cheating floor."""
    L = d['lengths'].reshape(-1, 1).astype(np.float64)
    return L[tri], L[tei]


def build_saplma(d, tri, tei, seed):
    """SAPLMA (Azaria & Mitchell 2023): probe on last-layer hidden state."""
    X = d['hidden'][:, -1, :].astype(np.float64)
    sc = StandardScaler().fit(X[tri])
    n_comp = min(128, len(tri) - 1, X.shape[1])
    pca = PCA(n_comp, random_state=seed).fit(sc.transform(X[tri]))
    return (_safe_clean(pca.transform(sc.transform(X[tri]))),
            _safe_clean(pca.transform(sc.transform(X[tei]))))


def build_haloscope(d, tri, tei, seed):
    """HaloScope-simplified (Du et al. 2024).

    TruncatedSVD top-3 components on the concatenated last 3 layers.
    This is a supervised single-response adaptation of the original
    unsupervised HaloScope (which uses K=10 samples). See Appendix H.
    """
    N = d['hidden'].shape[0]
    X = d['hidden'][:, -3:, :].reshape(N, -1).astype(np.float64)
    sc = StandardScaler().fit(X[tri])
    Xtr_s = sc.transform(X[tri])
    Xte_s = sc.transform(X[tei])
    svd = TruncatedSVD(n_components=3, random_state=seed,
                       algorithm='randomized')
    Xtr = svd.fit_transform(Xtr_s)
    Xte = svd.transform(Xte_s)
    return _safe_clean(Xtr), _safe_clean(Xte)


def build_harp(d, tri, tei, seed):
    """HARP (Hu et al. 2025): all-layer V_R-projected hidden states + PCA-200."""
    N = d['labels'].shape[0]
    X = d['proj_h'].reshape(N, -1).astype(np.float64)
    sc = StandardScaler().fit(X[tri])
    n_comp = min(200, len(tri) - 1, X.shape[1])
    pca = PCA(n_comp, random_state=seed).fit(sc.transform(X[tri]))
    return (_safe_clean(pca.transform(sc.transform(X[tri]))),
            _safe_clean(pca.transform(sc.transform(X[tei]))))


def build_xgb_all(d, tri, tei, seed):
    """All-layer flattened hidden states + PCA-200.

    Doubles as the V_R ablation for HARP.
    """
    N = d['hidden'].shape[0]
    X = d['hidden'].reshape(N, -1).astype(np.float64)
    sc = StandardScaler().fit(X[tri])
    n_comp = min(200, len(tri) - 1, X.shape[1])
    pca = PCA(n_comp, random_state=seed).fit(sc.transform(X[tri]))
    return (_safe_clean(pca.transform(sc.transform(X[tri]))),
            _safe_clean(pca.transform(sc.transform(X[tei]))))


def build_layercov(d, tri, tei, seed):
    """LayerCovariance (single-response adaptation of INSIDE).

    Per-sample (L+1)x(L+1) hidden-state covariance, log-eigenspectrum.
    """
    N, n_layers_plus_1, D = d['hidden'].shape
    feats = np.zeros((N, n_layers_plus_1), dtype=np.float64)
    for i in range(N):
        H = d['hidden'][i].astype(np.float64)
        H_c = H - H.mean(axis=0, keepdims=True)
        C = (H_c @ H_c.T) / max(D, 1)
        eigvals = np.maximum(np.linalg.eigvalsh(C), 1e-10)
        feats[i] = np.log(eigvals)
    sc = StandardScaler().fit(feats[tri])
    return (_safe_clean(sc.transform(feats[tri])),
            _safe_clean(sc.transform(feats[tei])))


def build_middle_probe(d, tri, tei, seed):
    """MiddleLayerProbe: same as SAPLMA but on the middle layer.

    Inspired by MIND, but applied post-hoc to a single supervised probe rather
    than during training (the original MIND is a training-time intervention).
    """
    n_layers_plus_1 = d['hidden'].shape[1]
    mid = n_layers_plus_1 // 2
    X = d['hidden'][:, mid, :].astype(np.float64)
    sc = StandardScaler().fit(X[tri])
    n_comp = min(128, len(tri) - 1, X.shape[1])
    pca = PCA(n_comp, random_state=seed).fit(sc.transform(X[tri]))
    return (_safe_clean(pca.transform(sc.transform(X[tri]))),
            _safe_clean(pca.transform(sc.transform(X[tei]))))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
METHODS = {
    'Length-only':      (build_length_only,  'lr'),
    'SAPLMA':           (build_saplma,       'mlp'),
    'HaloScope':        (build_haloscope,    'lr'),
    'HARP':             (build_harp,         'xgb'),
    'XGB-AllLayers':    (build_xgb_all,      'xgb'),
    'LayerCovariance':  (build_layercov,     'lr'),
    'MiddleLayerProbe': (build_middle_probe, 'mlp'),
}


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------
class _TorchMLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128, 64), out_dim=2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _train_torch_mlp(Xtr, ytr, seed, max_epochs=200, batch_size=128,
                     lr=1e-3, val_frac=0.1, patience=10):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    X = np.asarray(Xtr, dtype=np.float32)
    y = np.asarray(ytr, dtype=np.int64)
    n = len(y)
    n_val = max(1, int(val_frac * n))
    perm = rng.permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    Xt_tr = torch.from_numpy(X[tr_idx]).to(DEVICE)
    yt_tr = torch.from_numpy(y[tr_idx]).to(DEVICE)
    Xt_va = torch.from_numpy(X[val_idx]).to(DEVICE)
    yt_va = torch.from_numpy(y[val_idx]).to(DEVICE)

    model = _TorchMLP(in_dim=X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val = float('inf')
    best_state = None
    bad = 0
    n_tr = len(tr_idx)
    for _epoch in range(max_epochs):
        model.train()
        idx = torch.randperm(n_tr, device=DEVICE)
        for s in range(0, n_tr, batch_size):
            sel = idx[s:s + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt_tr[sel]), yt_tr[sel])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xt_va), yt_va).item()
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict_torch_mlp(model, Xte):
    model.eval()
    Xt = torch.from_numpy(np.asarray(Xte, dtype=np.float32)).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(Xt), dim=1)[:, 1].cpu().numpy()
    return probs


def fit_and_predict(Xtr, Xte, ytr, clf_kind: str, seed: int):
    """Train a classifier of the requested kind and return predicted probs."""
    if clf_kind == 'mlp':
        Xtr_b, ytr_b = _balanced_resample(Xtr, ytr, seed)
        mu = Xtr_b.mean(axis=0, keepdims=True)
        sd = Xtr_b.std(axis=0, keepdims=True) + 1e-8
        Xtr_b = (Xtr_b - mu) / sd
        Xte_n = (Xte - mu) / sd
        model = _train_torch_mlp(Xtr_b, ytr_b, seed)
        return _predict_torch_mlp(model, Xte_n)

    if clf_kind == 'lr':
        m = LogisticRegression(max_iter=1000, class_weight='balanced',
                               random_state=seed).fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]

    # xgb
    pw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                      subsample=0.8, colsample_bytree=0.8,
                      scale_pos_weight=pw, verbosity=0,
                      random_state=seed, tree_method='hist',
                      device='cuda:0' if torch.cuda.is_available() else 'cpu',
                      n_jobs=1, eval_metric='logloss').fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]
