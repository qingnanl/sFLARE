# src/yourpkg/propagate.py
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def step2_aggregate_knn_to_all(
    Z_c_all: np.ndarray,           # (Nc, d) all-cell embedding
    sub_idx_global: np.ndarray,    # (Ns,) indices of subset cells in Z_c_all row space
    pred_sub_all: np.ndarray,      # (Ns, Ng) subset predictions
    k: int = 10,
):
    """
    For cells not in Step 1, kNN aggregate from subset in Z_c_all (Euclidean).
    Returns:
        pred_all : (Nc, Ng) full imputed matrix
    """
    Z_c_all = np.asarray(Z_c_all, dtype=np.float32)
    sub_idx_global = np.asarray(sub_idx_global, dtype=int)

    Nc = Z_c_all.shape[0]
    Ns = len(sub_idx_global)
    Ng = pred_sub_all.shape[1]
    assert pred_sub_all.shape[0] == Ns, "pred_sub_all rows must match sub_idx_global order/length"

    all_idx = np.arange(Nc)
    non_idx = np.setdiff1d(all_idx, sub_idx_global, assume_unique=False)

    pred_all = np.zeros((Nc, Ng), dtype=np.float32)
    pred_all[sub_idx_global] = pred_sub_all

    if len(non_idx) == 0:
        return pred_all

    k_eff = min(k, Ns)
    nbrs = NearestNeighbors(n_neighbors=k_eff, metric="euclidean").fit(Z_c_all[sub_idx_global])
    dists, neigh = nbrs.kneighbors(Z_c_all[non_idx])  # neigh indexes into subset order [0..Ns-1]

    w = 1.0 / (dists + 1e-8)
    w = w / w.sum(axis=1, keepdims=True)

    agg = np.einsum("nk,nkg->ng", w, pred_sub_all[neigh])
    pred_all[non_idx] = agg.astype(np.float32)
    return pred_all
