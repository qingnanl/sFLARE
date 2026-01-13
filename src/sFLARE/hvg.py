# src/yourpkg/hvg.py
from __future__ import annotations
import numpy as np
from typing import List, Sequence, Tuple

def select_hvgs_scanpy(rna_adata, n_top_genes=5000, flavor="seurat_v3"):
    import scanpy as sc  # local import so yourpkg can be imported without scanpy if needed
    rna_adata = rna_adata.copy()
    sc.pp.highly_variable_genes(rna_adata, n_top_genes=n_top_genes, flavor=flavor)
    hvg_mask = rna_adata.var["highly_variable"].values
    hvg_names = rna_adata.var_names[hvg_mask].tolist()
    X_rna_hvg = rna_adata.X[:, hvg_mask]
    return rna_adata, X_rna_hvg, hvg_names

def filter_hvgs_by_mnn_expression(
    X_rna_hvg,
    hvg_names: List[str],
    mnn_pairs: Sequence[Tuple[int, int]],
):
    # RNA indices that appear in MNN pairs
    rna_cells_in_mnns = sorted({rna_idx for rna_idx, _ in mnn_pairs})
    rna_mnn_indices = np.array(rna_cells_in_mnns, dtype=int)

    X_rna_mnn = X_rna_hvg[rna_mnn_indices, :]
    gene_nonzero_mask = (X_rna_mnn.sum(axis=0) > 0)

    X_rna_hvg_f = X_rna_hvg[:, gene_nonzero_mask]
    hvg_names_f = [name for i, name in enumerate(hvg_names) if gene_nonzero_mask[i]]

    return X_rna_hvg_f, hvg_names_f, gene_nonzero_mask, rna_mnn_indices

