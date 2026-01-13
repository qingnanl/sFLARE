
# src/yourpkg/edges.py
from __future__ import annotations
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
from typing import Dict, List, Sequence, Tuple

def build_protein_gene_edges(
    rna_X_norm: np.ndarray,
    protein_X_norm: np.ndarray,
    X_rna_hvg: np.ndarray,
    mnn_pairs: Sequence[Tuple[int, int]],
    n_neighbors: int = 10,
    p_min: int = 3,
):
    # protein -> list of RNA neighbors (from MNN)
    protein_to_rna_neighbors: Dict[int, List[int]] = defaultdict(list)
    for rna_idx, protein_idx in mnn_pairs:
        protein_to_rna_neighbors[protein_idx].append(rna_idx)

    protein_cells_with_neighbors = sorted(protein_to_rna_neighbors.keys())
    protein_id_map = {old_id: new_id for new_id, old_id in enumerate(protein_cells_with_neighbors)}

    rna_tree = cKDTree(rna_X_norm)

    edges_protein = []
    edges_gene = []
    edges_weight = []

    for old_protein_idx, mnn_rna_indices in protein_to_rna_neighbors.items():
        new_protein_idx = protein_id_map[old_protein_idx]
        if len(mnn_rna_indices) == 0:
            continue

        protein_vec = protein_X_norm[old_protein_idx].reshape(1, -1)
        mnn_rna_vecs = rna_X_norm[mnn_rna_indices]
        dists = ((mnn_rna_vecs - protein_vec) ** 2).sum(axis=1)
        nearest_rna_idx = mnn_rna_indices[int(np.argmin(dists))]

        _, neighbor_indices = rna_tree.query(rna_X_norm[nearest_rna_idx].reshape(1, -1), k=n_neighbors)
        neighbor_indices = neighbor_indices[0]

        neighbors_expr = X_rna_hvg[neighbor_indices, :]
        gene_presence_counts = (neighbors_expr > 0).sum(axis=0)
        candidate_genes = np.where(gene_presence_counts >= p_min)[0]

        for gene_idx in candidate_genes:
            edges_protein.append(new_protein_idx)
            edges_gene.append(int(gene_idx))
            edges_weight.append(1.0)

    return {
        "protein_cells_with_neighbors": protein_cells_with_neighbors,
        "protein_id_map": protein_id_map,
        "edges_protein": np.asarray(edges_protein, dtype=np.int64),
        "edges_gene": np.asarray(edges_gene, dtype=np.int64),
        "edges_weight": np.asarray(edges_weight, dtype=np.float32),
        "num_proteins": len(protein_cells_with_neighbors),
        "num_genes": X_rna_hvg.shape[1],
    }

def edges_to_torch(edges_dict):
    import torch
    return (
        torch.LongTensor(edges_dict["edges_protein"]),
        torch.LongTensor(edges_dict["edges_gene"]),
        torch.FloatTensor(edges_dict["edges_weight"]),
    )