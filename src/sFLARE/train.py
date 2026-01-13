# src/yourpkg/train.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .models import BipartiteEmbeddingModel
from .losses import (
    compute_similarity_matrix,
    kl_divergence_rowwise,
    local_structure_loss,
    bipartite_laplacian_loss,
)

@dataclass
class TrainConfig:
    embedding_dim: int = 15
    num_epochs: int = 40
    lr: float = 1e-3
    lr_cells_edge: float = 1e-3
    lr_cells_kl: float = 2e-5
    cell_beta: float = 1.0
    eps: float = 1e-12
    edge_steps: int = 1
    print_every: int = 5

def get_embeddings_snapshot(model: BipartiteEmbeddingModel):
    with torch.no_grad():
        P = model.protein_emb.weight.detach().cpu().numpy().astype(np.float32)
        G = model.gene_emb.weight.detach().cpu().numpy().astype(np.float32)
    return P, G

def fit_bipartite_embeddings(
    *,
    edges_protein: torch.LongTensor,
    edges_gene: torch.LongTensor,
    codex_pca_mnn_subset: np.ndarray | torch.Tensor,
    prior_gene_embedding_tensor: np.ndarray | torch.Tensor,
    gene_knn_indices: torch.LongTensor,
    cfg: TrainConfig = TrainConfig(),
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns dict with:
      model, losses_history (list), embeddings_history (list), losses_df (pd.DataFrame)
    """
    import pandas as pd

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # tensors
    codex_pca_tensor = (
        codex_pca_mnn_subset if isinstance(codex_pca_mnn_subset, torch.Tensor)
        else torch.tensor(codex_pca_mnn_subset, dtype=torch.float32)
    ).to(device)

    prior_gene_tensor = (
        prior_gene_embedding_tensor if isinstance(prior_gene_embedding_tensor, torch.Tensor)
        else torch.tensor(prior_gene_embedding_tensor, dtype=torch.float32)
    ).to(device)

    edges_protein = edges_protein.to(device)
    edges_gene = edges_gene.to(device)
    gene_knn_indices = gene_knn_indices.to(device)

    # prior sim (detach)
    prior_cell_sim = compute_similarity_matrix(codex_pca_tensor, beta=cfg.cell_beta, eps=cfg.eps).detach()

    num_proteins = codex_pca_tensor.shape[0]
    num_genes = prior_gene_tensor.shape[0]

    model = BipartiteEmbeddingModel(num_proteins, num_genes, cfg.embedding_dim, prior_cell_embedding=codex_pca_tensor).to(device)

    optimizer_cells = torch.optim.Adam([model.protein_emb.weight], lr=cfg.lr)
    optimizer_genes = torch.optim.Adam([model.gene_emb.weight], lr=cfg.lr)

    # logging
    cycles: List[int] = []
    losses_history: List[Dict[str, float]] = []
    embeddings_history: List[Dict[str, np.ndarray]] = []

    # cycle 0
    P0, G0 = get_embeddings_snapshot(model)
    with torch.no_grad():
        kl0 = kl_divergence_rowwise(prior_cell_sim, compute_similarity_matrix(model.protein_emb.weight, beta=cfg.cell_beta, eps=cfg.eps)).item()
    cycles.append(0)
    embeddings_history.append({"protein": P0, "gene": G0})
    losses_history.append({"cell_kl_pre": kl0, "cell_kl_post": kl0})

    # training
    for epoch in range(cfg.num_epochs):
        model.train()

        # 1) fix cells, optimize genes (edge loss)
        model.protein_emb.weight.requires_grad = False
        model.gene_emb.weight.requires_grad = True
        loss_edge_genes_total = 0.0
        for _ in range(cfg.edge_steps):
            optimizer_genes.zero_grad()
            p_vec, g_vec = model(edges_protein, edges_gene)
            loss_edge_genes = bipartite_laplacian_loss(p_vec.detach(), g_vec)
            loss_edge_genes.backward()
            optimizer_genes.step()
            loss_edge_genes_total += loss_edge_genes.item()
        loss_edge_genes_avg = loss_edge_genes_total / cfg.edge_steps

        # 2) gene local structure
        optimizer_genes.zero_grad()
        loss_gene_local = local_structure_loss(model.gene_emb.weight, gene_knn_indices)
        loss_gene_local.backward()
        optimizer_genes.step()

        # 3) fix genes, optimize cells (edge loss)
        for g in optimizer_cells.param_groups:
            g["lr"] = cfg.lr_cells_edge
        model.protein_emb.weight.requires_grad = True
        model.gene_emb.weight.requires_grad = False

        loss_edge_cells_total = 0.0
        for _ in range(cfg.edge_steps):
            optimizer_cells.zero_grad()
            p_vec, g_vec = model(edges_protein, edges_gene)
            loss_edge_cells = bipartite_laplacian_loss(p_vec, g_vec.detach())
            loss_edge_cells.backward()
            optimizer_cells.step()
            loss_edge_cells_total += loss_edge_cells.item()
        loss_edge_cells_avg = loss_edge_cells_total / cfg.edge_steps

        # KL pre
        with torch.no_grad():
            kl_pre = kl_divergence_rowwise(
                prior_cell_sim,
                compute_similarity_matrix(model.protein_emb.weight, beta=cfg.cell_beta, eps=cfg.eps),
                eps=cfg.eps
            ).item()

        # 4) cell KL update
        for g in optimizer_cells.param_groups:
            g["lr"] = cfg.lr_cells_kl
        optimizer_cells.zero_grad()
        current_cell_sim = compute_similarity_matrix(model.protein_emb.weight, beta=cfg.cell_beta, eps=cfg.eps)
        loss_cell_kl = kl_divergence_rowwise(prior_cell_sim, current_cell_sim, eps=cfg.eps)
        loss_cell_kl.backward()
        optimizer_cells.step()

        with torch.no_grad():
            kl_post = kl_divergence_rowwise(
                prior_cell_sim,
                compute_similarity_matrix(model.protein_emb.weight, beta=cfg.cell_beta, eps=cfg.eps),
                eps=cfg.eps
            ).item()

        # restore lr
        for g in optimizer_cells.param_groups:
            g["lr"] = cfg.lr_cells_edge

        # snapshot
        Pc, Gc = get_embeddings_snapshot(model)
        embeddings_history.append({"protein": Pc, "gene": Gc})
        cycles.append(epoch + 1)
        losses_history.append({
            "edge":        0.5 * (loss_edge_genes_avg + loss_edge_cells_avg),
            "gene_local":  float(loss_gene_local.item()),
            "cell_kl_pre": kl_pre,
            "cell_kl_post": kl_post,
        })

        if cfg.print_every and ((epoch + 1) % cfg.print_every == 0):
            print(f"[epoch {epoch+1}/{cfg.num_epochs}] edge={losses_history[-1].get('edge', float('nan')):.4f} "
                  f"gene_local={losses_history[-1].get('gene_local', float('nan')):.4f} "
                  f"kl={kl_post:.4f}")

    losses_df = pd.DataFrame(losses_history, index=cycles)

    return {
        "model": model,
        "losses_history": losses_history,
        "embeddings_history": embeddings_history,
        "losses_df": losses_df,
        "prior_cell_sim": prior_cell_sim.detach().cpu(),
    }
