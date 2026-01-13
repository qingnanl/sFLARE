# src/yourpkg/projection.py
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class EmbedMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def train_mlp(model, X, Y, n_epochs=500, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        optimizer.step()
    return model

def project_all_cells_with_trained_mlp(
    *,
    codex_pca_all: np.ndarray,
    protein_cells_with_neighbors,
    embeddings_history,
    cyc_selected: int = 20,
    embedding_dim: int = 15,
    n_epochs: int = 500,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, EmbedMLP]:
    """
    Train MLP on (PCA of MNN-used cells -> learned cell embeddings at cycle cyc_selected),
    then project all cells in codex_pca_all.

    Returns:
      Z_c_all: np.ndarray of shape (n_all_cells, embedding_dim)
      cell_mlp: trained torch model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    codex_cell_embeddings = embeddings_history[cyc_selected]["protein"]
    # gene_embeddings are not needed for this projection step

    X_cell_pca_train = codex_pca_all[protein_cells_with_neighbors]
    Y_cell_embed_train = codex_cell_embeddings

    Xc = torch.tensor(X_cell_pca_train, dtype=torch.float32, device=device)
    Yc = torch.tensor(Y_cell_embed_train, dtype=torch.float32, device=device)

    cell_mlp = EmbedMLP(Xc.shape[1], embedding_dim).to(device)
    train_mlp(cell_mlp, Xc, Yc, n_epochs=n_epochs, lr=lr)

    with torch.no_grad():
        X_all = torch.tensor(codex_pca_all, dtype=torch.float32, device=device)
        Z_c_all = cell_mlp(X_all).detach().cpu().numpy()

    return Z_c_all, cell_mlp
