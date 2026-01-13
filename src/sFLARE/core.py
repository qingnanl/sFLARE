

from scipy.spatial import cKDTree
import numpy as np
from tqdm import tqdm

def find_mutual_nn_fast(data1, data2, k1=30, k2=30, n_jobs=8, progress=True):
    n1, n2 = data1.shape[0], data2.shape[0]
    tree1 = cKDTree(data1)
    tree2 = cKDTree(data2)
    k_index_1 = tree1.query(x=data2, k=k1, workers=n_jobs)[1]  # shape: (n2, k1) -> indices into data1
    k_index_2 = tree2.query(x=data1, k=k2, workers=n_jobs)[1]  # shape: (n1, k2) -> indices into data2
    inv = [[] for _ in range(n2)]
    for i1 in range(n1):
        for i2 in k_index_2[i1]:
            inv[i2].append(i1)
    for i2 in range(n2):
        if inv[i2]:
            inv[i2] = np.array(sorted(inv[i2]), dtype=np.int32)
        else:
            inv[i2] = np.empty(0, dtype=np.int32)
    mutual_pairs = []
    rng = range(n2)
    if progress:
        rng = tqdm(rng, desc="Mutual NN (scan)", unit="cell")
    for i2 in rng:
        # candidates: i1 among k1-NN of i2
        a = k_index_1[i2]
        # i1 that have i2 among their k2-NN
        b = inv[i2]
        if b.size == 0:
            continue
        common = np.intersect1d(a, b, assume_unique=False)
        if common.size:
            mutual_pairs.extend(zip(common.tolist(), [i2]*common.size))
    return mutual_pairs

res = find_mutual_nn_fast(rna_X_norm, protein_X_norm)

    
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import pickle
rna_adata = adata.copy()
sc.pp.highly_variable_genes(
    rna_adata,
    n_top_genes=5000,
    flavor="seurat_v3",
)
hvg_mask = rna_adata.var['highly_variable'].values
#hvg_mask = np.ones(adata.n_vars, dtype=bool)

hvg_names = rna_adata.var_names[hvg_mask].tolist()
# RNA expression for HVGs
X_rna_hvg = rna_adata.X[:, hvg_mask]
# X_rna_hvg = X_rna_hvg.toarray().astype(np.float32, copy=False) 

# Filter HVGs with non-zero expression in any MNN RNA cell
rna_cells_in_mnns = set(rna_idx for rna_idx, _ in res)
rna_mnn_indices = np.array(sorted(rna_cells_in_mnns))
X_rna_mnn = X_rna_hvg[rna_mnn_indices, :]

gene_nonzero_mask = (X_rna_mnn.sum(axis=0) > 0)
print(f"Genes expressed in MNN RNA cells: {gene_nonzero_mask.sum()}/{len(gene_nonzero_mask)}")

# Apply filter
X_rna_hvg = X_rna_hvg[:, gene_nonzero_mask]
hvg_names = [name for i, name in enumerate(hvg_names) if gene_nonzero_mask[i]]
n_hvgs = X_rna_hvg.shape[1]

print(f"Final HVG count after filtering: {n_hvgs}")

protein_to_rna_neighbors = defaultdict(list)
for rna_idx, protein_idx in res:
    protein_to_rna_neighbors[protein_idx].append(rna_idx)

protein_cells_with_neighbors = sorted(protein_to_rna_neighbors.keys())
num_proteins_with_neighbors = len(protein_cells_with_neighbors)
print(f"Protein cells with RNA neighbors: {num_proteins_with_neighbors}")

# Mapping for contiguous IDs
protein_id_map = {old_id: new_id for new_id, old_id in enumerate(protein_cells_with_neighbors)}

edges_protein = []
edges_gene = []
edges_weight = []

from scipy.spatial import cKDTree

# Build RNA latent space KDTree for neighbor search
rna_tree = cKDTree(rna_X_norm)

# Use these hyperparameters for the test
n_neighbors = 10
p_min = 3

for old_protein_idx, mnn_rna_indices in protein_to_rna_neighbors.items():
    new_protein_idx = protein_id_map[old_protein_idx]
    if len(mnn_rna_indices) == 0:
        continue
    # 1️⃣ Find *nearest* MNN RNA neighbor in normalized space
    protein_vec = protein_X_norm[old_protein_idx].reshape(1, -1)
    mnn_rna_vecs = rna_X_norm[mnn_rna_indices]
    dists = ((mnn_rna_vecs - protein_vec)**2).sum(axis=1)
    nearest_rna_idx = mnn_rna_indices[np.argmin(dists)]
    # 2️⃣ Find n (5) nearest neighbors of that RNA cell in RNA space
    _, neighbor_indices = rna_tree.query(
        rna_X_norm[nearest_rna_idx].reshape(1, -1),
        k=n_neighbors
    )
    neighbor_indices = neighbor_indices[0]
    # 3️⃣ Get HVG expression for those RNA neighbors
    neighbors_expr = X_rna_hvg[neighbor_indices, :]  # shape: (5, n_hvgs)
    # 4️⃣ For each gene, count number of expressing neighbors
    gene_presence_counts = (neighbors_expr > 0).sum(axis=0)
    # 5️⃣ Make edges for genes meeting threshold
    candidate_genes = np.where(gene_presence_counts >= p_min)[0]
    for gene_idx in candidate_genes:
        edges_protein.append(new_protein_idx)
        edges_gene.append(gene_idx)
        edges_weight.append(1.0)  # just appearance

print(f"Total edges after new strategy: {len(edges_protein)}")

# Convert to tensors
edges_protein = torch.LongTensor(edges_protein)
edges_gene = torch.LongTensor(edges_gene)
edges_weight = torch.FloatTensor(edges_weight)


num_proteins = len(protein_cells_with_neighbors)
num_genes = n_hvgs