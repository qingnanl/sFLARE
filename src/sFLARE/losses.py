# src/yourpkg/losses.py
import torch
import torch.nn.functional as F

def compute_similarity_matrix(X, beta=1.0, eps=1e-12):
    dists_sq = torch.cdist(X, X, p=2).pow(2)
    sim = torch.exp(-beta * dists_sq)
    sim = sim - torch.diag_embed(torch.diagonal(sim))
    sim_sum = sim.sum(dim=1, keepdim=True) + eps
    return sim / sim_sum

def kl_divergence(P, Q, eps=1e-12):
    P = P.clamp(min=eps)
    Q = Q.clamp(min=eps)
    return (P * (P / Q).log()).sum()

def kl_divergence_rowwise(P, Q, eps=1e-12, reduction="mean", use_double=True):
    if use_double:
        P = P.double()
        Q = Q.double()
    I = torch.eye(P.shape[0], dtype=torch.bool, device=P.device)
    Pm = P.masked_fill(I, 0)
    Qm = Q.masked_fill(I, 0)
    Pm = Pm / (Pm.sum(dim=1, keepdim=True) + eps)
    Qm = Qm / (Qm.sum(dim=1, keepdim=True) + eps)
    kl_rows = (Pm * (torch.log(Pm + eps) - torch.log(Qm + eps))).sum(dim=1)
    if reduction == "mean":
        out = kl_rows.mean()
    elif reduction == "sum":
        out = kl_rows.sum()
    else:
        out = kl_rows
    return out.float()

def local_structure_loss(embedding, knn_indices):
    anchor = embedding
    neighbors = embedding[knn_indices]
    anchor_exp = anchor.unsqueeze(1)
    dists = F.pairwise_distance(anchor_exp, neighbors, p=2).pow(2)
    return dists.mean()

def bipartite_laplacian_loss(protein_vec, gene_vec):
    return F.mse_loss(protein_vec, gene_vec)
