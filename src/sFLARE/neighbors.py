# src/yourpkg/neighbors.py
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_gene_knn_indices(prior_gene_embedding, k_gene: int = 20) -> torch.LongTensor:
    """
    prior_gene_embedding: torch.Tensor or np.ndarray, shape (n_genes, d)
    returns: torch.LongTensor, shape (n_genes, k_gene)
    """
    if isinstance(prior_gene_embedding, torch.Tensor):
        X = prior_gene_embedding.detach().cpu().numpy()
    else:
        X = np.asarray(prior_gene_embedding)

    nbrs = NearestNeighbors(n_neighbors=k_gene + 1).fit(X)
    idx = nbrs.kneighbors(return_distance=False)[:, 1:]
    return torch.tensor(idx, dtype=torch.long)
