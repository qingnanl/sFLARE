# src/yourpkg/embeddings.py
from __future__ import annotations
from typing import Sequence, Tuple, Optional
import numpy as np

def compute_codex_pca_all_and_mnn_subset(
    codex_adata,
    *,
    protein_cells_with_neighbors: Sequence[int],
    n_comps: int = 15,
    scale_max_value: float = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs PCA on:
      1) all CODEX cells (optionally with scaling)
      2) the MNN-linked subset of CODEX cells

    Returns:
      codex_pca_all: (n_cells_all, n_comps)
      codex_pca_mnn_subset: (n_subset, n_comps)
    """
    import scanpy as sc

    # --- all cells ---
    ad_all = codex_adata.copy()
    ad_all.raw = ad_all.copy()
    sc.pp.scale(ad_all, max_value=scale_max_value)
    sc.tl.pca(ad_all, n_comps=n_comps)
    codex_pca_all = ad_all.obsm["X_pca"].copy()

    # --- subset (MNN-linked proteins) ---
    ad_sub = codex_adata[protein_cells_with_neighbors].copy()
    ad_sub.raw = ad_sub.copy()
    sc.tl.pca(ad_sub, n_comps=n_comps)
    codex_pca_mnn_subset = ad_sub.obsm["X_pca"].copy()

    return codex_pca_all, codex_pca_mnn_subset
