# src/yourpkg/priors.py
from __future__ import annotations
import numpy as np
from scipy.sparse import isspmatrix
from tqdm import tqdm

def calculate_P_dist_v2(density, weights, pseudo=1e-300):
  from anndata import AnnData
  from numpy import ndarray
  if (isspmatrix(weights)):
    index = weights.nonzero()[0]
    P = density[index, :] * weights.data.reshape(-1,1)
  if (isinstance(weights, ndarray)):
    P = density * weights.reshape(-1,1)
  P = np.sum(P, axis=0)
  P = P + pseudo
 # P = P / np.sum(P)
  return P

def calculate_P_matrix_v2(density, weights, pseudo=1e-300, verbose=False):
  if (isspmatrix(weights)):
    weights = weights.tocsc()
  ngenes = weights.shape[1]
  ngrid_points = density.shape[1]
  if (verbose):
    print("> calculating P for " + str(ngenes) + " features ...")
    pbar = tqdm(total=ngenes)
  res = np.zeros([ngenes, ngrid_points])
  for k in range(ngenes):
    res[k, :] = calculate_P_dist_v2(density, weights[:, k], pseudo=pseudo)
    if (verbose):
      pbar.update(n=1)
  return res


# src/yourpkg/priors.py (continued)
from typing import List, Sequence, Tuple, Optional

def compute_P_for_adata(
    rna_adata,
    *,
    coord_key: str = "X_pca_harmony",
    n_grid_points: int = 100,
    random_state: int = 1,
    pseudo: float = 1e-300,
    verbose: bool = True,
):
    """
    Computes grid density from rna_adata.obsm[coord_key] and then computes P matrix
    from rna_adata.X (expression).
    Returns: (P, gene_names, grid_density, grid_points)
    """
    from .haystack_grid import compute_grid_density_from_coord

    coord = rna_adata.obsm[coord_key]
    grid_points, _, grid_density = compute_grid_density_from_coord(
        coord, n_grid_points=n_grid_points, random_state=random_state, verbose=verbose
    )

    expr = rna_adata.X  # keep sparse if it is sparse
    P = calculate_P_matrix_v2(grid_density, expr, pseudo=pseudo, verbose=verbose)

    gene_names = rna_adata.var_names.tolist()
    return P, gene_names, grid_density, grid_points


def subset_P_to_genes(P: np.ndarray, gene_names: Sequence[str], subset_names: Sequence[str]) -> np.ndarray:
    """
    Subset P (genes x gridpoints) to subset_names in the same order as subset_names.
    """
    name_to_idx = {g: i for i, g in enumerate(gene_names)}
    idx = [name_to_idx[g] for g in subset_names]
    return P[idx, :]

# src/yourpkg/priors.py (continued)
import numpy as np

def make_prior_gene_embedding(
    P_subset: np.ndarray,
    *,
    embedding_dim: int = 15,
    pca_components: int = 15,
    random_state: int = 42,
):
    """
    Standardize -> PCA -> pad/trim -> return (prior_embedding_np, prior_embedding_torch)
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import torch

    scaler = StandardScaler()
    P_scaled = scaler.fit_transform(P_subset)

    pca = PCA(n_components=pca_components, random_state=random_state)
    P_pca = pca.fit_transform(P_scaled)

    if P_pca.shape[1] < embedding_dim:
        padding = np.zeros((P_pca.shape[0], embedding_dim - P_pca.shape[1]))
        prior_np = np.hstack([P_pca, padding])
    else:
        prior_np = P_pca[:, :embedding_dim]

    prior_torch = torch.tensor(prior_np, dtype=torch.float32)
    return prior_np, prior_torch
