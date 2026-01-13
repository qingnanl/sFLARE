# src/yourpkg/main.py
from __future__ import annotations
from typing import Any, Dict

from .data import MultiModalData
from .mnn import find_mutual_nn_fast
from .hvg import select_hvgs_scanpy, filter_hvgs_by_mnn_expression
from .edges import build_protein_gene_edges, edges_to_torch

def build_edges_from_data(
    data: MultiModalData,
    *,
    k1=30, k2=30, n_jobs=8, progress=True,
    n_top_genes=5000, flavor="seurat_v3",
    n_neighbors=10, p_min=3,
    return_torch=True,
) -> Dict[str, Any]:
    mnns = find_mutual_nn_fast(data.rna_X_norm, data.protein_X_norm, k1=k1, k2=k2, n_jobs=n_jobs, progress=progress)
    rna_adata_hvg, X_rna_hvg, hvg_names = select_hvgs_scanpy(data.rna_adata, n_top_genes=n_top_genes, flavor=flavor)
    X_rna_hvg_f, hvg_names_f, gene_nonzero_mask, rna_mnn_indices = filter_hvgs_by_mnn_expression(X_rna_hvg, hvg_names, mnns)
    edges = build_protein_gene_edges(data.rna_X_norm, data.protein_X_norm, X_rna_hvg_f, mnns, n_neighbors=n_neighbors, p_min=p_min)

    out = {
        "mnns": mnns,
        "rna_adata_hvg": rna_adata_hvg,
        "X_rna_hvg": X_rna_hvg_f,
        "hvg_names": hvg_names_f,
        "gene_nonzero_mask": gene_nonzero_mask,
        "rna_mnn_indices": rna_mnn_indices,
        "edges": edges,
    }
    if return_torch:
        out["edges_torch"] = edges_to_torch(edges)
    return out




