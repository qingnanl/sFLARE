# src/yourpkg/edges.py
from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple, Optional

from scipy.spatial import cKDTree


def build_protein_gene_edges(
    rna_X_norm: np.ndarray,
    protein_X_norm: np.ndarray,
    X_rna_hvg: np.ndarray,
    mnn_pairs: Sequence[Tuple[int, int]],
    n_neighbors: int = 10,
    p_min: int = 3,
    *,
    rna_neighbor_X: Optional[np.ndarray] = None,
    n_jobs: int = 8,
):
    """
    Build edges from protein-cells (CODEX) to genes (HVG indices), based on:
      1) MNN pairs between RNA latent and protein latent (mnn_pairs)
      2) For each protein cell, choose its nearest MNN RNA neighbor (in rna_X_norm space)
      3) Find k nearest RNA neighbors of that RNA cell in a *separate RNA neighbor latent space*
         (rna_neighbor_X; default = rna_X_norm)
      4) Add edges to genes expressed ( >0 ) in at least p_min of those RNA neighbors

    Parameters
    ----------
    rna_X_norm:
        RNA latent used for selecting the nearest MNN RNA cell to a protein cell.
        Shape: (n_rna_cells, d1)
    protein_X_norm:
        Protein latent used for selecting nearest MNN RNA cell.
        Shape: (n_protein_cells, d1)
    X_rna_hvg:
        RNA expression matrix restricted to HVGs (and possibly further filtered).
        Must be row-aligned with rna_X_norm.
        Shape: (n_rna_cells, n_hvgs)
    mnn_pairs:
        Sequence of (rna_idx, protein_idx) pairs (indices into rna_X_norm and protein_X_norm).
    n_neighbors:
        Number of RNA neighbors (k) to query around the chosen nearest MNN RNA cell.
    p_min:
        Minimum number of those neighbors that must express a gene (>0) to create an edge.
    rna_neighbor_X:
        RNA latent used ONLY for RNA-RNA neighbor queries (KDTree built on this).
        If None, defaults to rna_X_norm.
        Shape: (n_rna_cells, d2)
    n_jobs:
        KDTree query workers (scipy cKDTree 'workers').

    Returns
    -------
    dict with:
      protein_cells_with_neighbors : list[int]
      protein_id_map               : dict[old_protein_idx -> new_contiguous_idx]
      edges_protein                : (E,) int64 array (new protein idx)
      edges_gene                   : (E,) int64 array (gene idx in [0..n_hvgs-1])
      edges_weight                 : (E,) float32 array
      num_proteins                 : int
      num_genes                    : int
    """
    if rna_neighbor_X is None:
        rna_neighbor_X = rna_X_norm

    # ---- sanity checks: prevent silent index mismatches ----
    rna_X_norm = np.asarray(rna_X_norm)
    protein_X_norm = np.asarray(protein_X_norm)
    rna_neighbor_X = np.asarray(rna_neighbor_X)

    if rna_neighbor_X.shape[0] != rna_X_norm.shape[0]:
        raise ValueError(
            f"rna_neighbor_X must have same number of rows as rna_X_norm. "
            f"Got {rna_neighbor_X.shape[0]} vs {rna_X_norm.shape[0]}."
        )
    if X_rna_hvg.shape[0] != rna_X_norm.shape[0]:
        raise ValueError(
            f"X_rna_hvg must be row-aligned with rna_X_norm. "
            f"Got {X_rna_hvg.shape[0]} vs {rna_X_norm.shape[0]}."
        )
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1")
    if p_min < 1:
        raise ValueError("p_min must be >= 1")

    # protein -> list of RNA neighbors (from MNN)
    protein_to_rna_neighbors: Dict[int, List[int]] = defaultdict(list)
    for rna_idx, protein_idx in mnn_pairs:
        protein_to_rna_neighbors[int(protein_idx)].append(int(rna_idx))

    protein_cells_with_neighbors = sorted(protein_to_rna_neighbors.keys())
    protein_id_map = {old_id: new_id for new_id, old_id in enumerate(protein_cells_with_neighbors)}

    # KDTree on RNA *neighbor* latent (e.g., rna_adata.obsm["X_pca_harmony"])
    rna_tree = cKDTree(rna_neighbor_X)

    edges_protein: List[int] = []
    edges_gene: List[int] = []
    edges_weight: List[float] = []

    # precompute for speed
    n_hvgs = X_rna_hvg.shape[1]
    k_eff = min(int(n_neighbors), rna_neighbor_X.shape[0])

    for old_protein_idx, mnn_rna_indices in protein_to_rna_neighbors.items():
        if len(mnn_rna_indices) == 0:
            continue

        new_protein_idx = protein_id_map[old_protein_idx]

        # 1) choose nearest MNN RNA cell (in rna_X_norm space)
        protein_vec = protein_X_norm[old_protein_idx].reshape(1, -1)     # (1,d1)
        mnn_rna_vecs = rna_X_norm[mnn_rna_indices]                       # (m,d1)
        dists = ((mnn_rna_vecs - protein_vec) ** 2).sum(axis=1)          # (m,)
        nearest_rna_idx = mnn_rna_indices[int(np.argmin(dists))]

        # 2) find RNA neighbors of that RNA cell (in rna_neighbor_X space)
        #    neighbor_indices are RNA indices into X_rna_hvg
        _, neighbor_indices = rna_tree.query(
            rna_neighbor_X[nearest_rna_idx].reshape(1, -1),
            k=k_eff,
            workers=n_jobs,
        )
        neighbor_indices = np.asarray(neighbor_indices[0], dtype=int)

        # 3) expression among those neighbors -> candidate genes
        neighbors_expr = X_rna_hvg[neighbor_indices, :]                  # (k, n_hvgs)
        gene_presence_counts = (neighbors_expr > 0).sum(axis=0)          # (n_hvgs,)

        candidate_genes = np.where(gene_presence_counts >= p_min)[0]
        if candidate_genes.size == 0:
            continue

        # 4) add edges
        edges_protein.extend([new_protein_idx] * int(candidate_genes.size))
        edges_gene.extend(candidate_genes.astype(int).tolist())
        edges_weight.extend([1.0] * int(candidate_genes.size))

    return {
        "protein_cells_with_neighbors": protein_cells_with_neighbors,
        "protein_id_map": protein_id_map,
        "edges_protein": np.asarray(edges_protein, dtype=np.int64),
        "edges_gene": np.asarray(edges_gene, dtype=np.int64),
        "edges_weight": np.asarray(edges_weight, dtype=np.float32),
        "num_proteins": len(protein_cells_with_neighbors),
        "num_genes": n_hvgs,
    }


def edges_to_torch(edges_dict):
    import torch
    return (
        torch.LongTensor(edges_dict["edges_protein"]),
        torch.LongTensor(edges_dict["edges_gene"]),
        torch.FloatTensor(edges_dict["edges_weight"]),
    )
