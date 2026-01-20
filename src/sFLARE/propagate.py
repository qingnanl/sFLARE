# src/sFLARE/propagate.py
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def step2_aggregate_knn_to_all(
    Z_c_all: np.ndarray,          # (Nc, d) all-cell embedding, order must match all_cell_ids
    sub_cell_ids,                 # (Ns,) subset cell IDs (strings with letters)
    pred_sub_all: np.ndarray,     # (Ns, Ng) subset predictions (same order as sub_cell_ids)
    all_cell_ids,                # (Nc,) all cell IDs (strings), must match Z_c_all row order
    k: int = 10,
):
    """
    For cells not in Step 1, kNN aggregate from subset in Z_c_all (Euclidean).

    IMPORTANT:
    - sub_cell_ids: list/array of string IDs for the subset cells
    - all_cell_ids: list/array of string IDs for ALL cells, in the SAME ORDER as Z_c_all
    - pred_sub_all rows must match sub_cell_ids order

    Returns:
        pred_all : (Nc, Ng) full imputed matrix
    """
    Z_c_all = np.asarray(Z_c_all, dtype=np.float32)
    all_cell_ids = np.asarray(all_cell_ids).astype(str)
    sub_cell_ids = np.asarray(sub_cell_ids).astype(str)

    Nc = Z_c_all.shape[0]
    Ng = pred_sub_all.shape[1]

    assert len(all_cell_ids) == Nc, (
        "all_cell_ids length must match Z_c_all rows.\n"
        f"len(all_cell_ids)={len(all_cell_ids)} vs Nc={Nc}"
    )

    Ns = len(sub_cell_ids)
    assert pred_sub_all.shape[0] == Ns, (
        "pred_sub_all rows must match sub_cell_ids length.\n"
        f"pred_sub_all.shape[0]={pred_sub_all.shape[0]} vs Ns={Ns}"
    )

    # ✅ map global string IDs -> row indices
    id_to_idx = {cid: i for i, cid in enumerate(all_cell_ids)}

    # ✅ convert subset IDs -> integer row positions in global matrix
    try:
        sub_idx_global = np.array([id_to_idx[cid] for cid in sub_cell_ids], dtype=int)
    except KeyError as e:
        missing = str(e).strip("'")
        raise ValueError(
            f"Subset cell id not found in all_cell_ids: {missing}\n"
            "Make sure subset_adata.obs_names is a subset of codex_adata.obs_names, "
            "and that Z_c_all is aligned with codex_adata.obs_names order."
        )

    # Full prediction matrix
    pred_all = np.zeros((Nc, Ng), dtype=np.float32)
    pred_all[sub_idx_global] = pred_sub_all

    # find non-subset cells
    all_idx = np.arange(Nc)
    non_idx = np.setdiff1d(all_idx, sub_idx_global, assume_unique=False)
    if len(non_idx) == 0:
        return pred_all

    # KNN on subset only
    k_eff = min(int(k), Ns)
    nbrs = NearestNeighbors(n_neighbors=k_eff, metric="euclidean").fit(Z_c_all[sub_idx_global])
    dists, neigh = nbrs.kneighbors(Z_c_all[non_idx])  # neigh in [0..Ns-1] subset order

    # inverse-distance weights
    w = 1.0 / (dists + 1e-8)
    w = w / w.sum(axis=1, keepdims=True)

    # aggregate predictions from subset neighbors
    agg = np.einsum("nk,nkg->ng", w, pred_sub_all[neigh])
    pred_all[non_idx] = agg.astype(np.float32)

    return pred_all
