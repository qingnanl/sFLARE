# ===== src/sFLARE/enc_dec_impute.py =====
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse import issparse


# ------------------------------ utilities ------------------------------
def sparse_to_1d(x):
    return x.A1 if issparse(x) else np.asarray(x).ravel()


def knn_edges_cosine(X: np.ndarray, K: int, drop_self: bool = True):
    """
    Build an undirected kNN graph (cosine) over rows of X.
    Returns:
      edge_index: (2, E) np.int64
      edge_weight: (E,) np.float32  (cosine similarity clipped at 0)
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=np.float32)
    nn_ = NearestNeighbors(n_neighbors=K, metric="cosine").fit(X)
    distances, indices = nn_.kneighbors(X)

    rows = np.repeat(np.arange(X.shape[0]), K)
    cols = indices.ravel()
    sim = (1.0 - distances.ravel()).clip(min=0)

    if drop_self:
        keep = rows != cols
        rows, cols, sim = rows[keep], cols[keep], sim[keep]

    # make undirected by mirroring
    edge_index = np.vstack([np.hstack([rows, cols]), np.hstack([cols, rows])]).astype(np.int64)
    edge_weight = np.hstack([sim, sim]).astype(np.float32)
    return edge_index, edge_weight


class ResGCN(nn.Module):
    """
    2-layer residual GCN encoder.
    NOTE: torch_geometric is imported lazily so importing this module won't fail
    unless you actually instantiate the model.
    """
    def __init__(self, d_in: int, hidden: int = 64, drop: float = 0.1):
        super().__init__()
        from torch_geometric.nn import GCNConv  # lazy import

        self.c1 = GCNConv(d_in, hidden)
        self.c2 = GCNConv(hidden, hidden)
        self.drop = drop

    def forward(self, x, ei, ew):
        h1 = self.c1(x, ei, ew).relu()
        h1 = F.dropout(h1, p=self.drop, training=self.training)
        h2 = self.c2(h1, ei, ew)
        return h2 + h1


class BilinearDecoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, H_cells, H_genes):  # (Nc,h),(B,h)->(Nc,B)
        return H_cells @ self.proj(H_genes).T


def pearson_loss(pred, y, eps=1e-8):
    pc = pred - pred.mean(0, keepdim=True)
    yc = y    - y.mean(0, keepdim=True)
    num = (pc * yc).sum(0)
    den = torch.sqrt((pc.pow(2).sum(0) + eps) * (yc.pow(2).sum(0) + eps))
    r = num / (den + eps)
    return 1.0 - r.mean()


def bce_topk_loss(logits, y, pos_q=0.90, neg_q=0.50):
    with torch.no_grad():
        qpos = torch.quantile(y, pos_q, dim=0, keepdim=True)
        qneg = torch.quantile(y, neg_q, dim=0, keepdim=True)
        pos_mask = y >= qpos
        neg_mask = y <= qneg
        mask = pos_mask | neg_mask
        targets = torch.where(pos_mask, torch.ones_like(y), torch.zeros_like(y))
    if mask.sum() == 0:
        return logits.new_tensor(0.0)
    return F.binary_cross_entropy_with_logits(logits[mask], targets[mask])


def low_expression_weights(y, base=0.2, gamma=2.0):
    """
    y: (Ns, B) tensor. Compute percentile per gene across cells, then weight:
        w = base + (1-base) * (1 - percentile)^gamma
    -> low-expression cells get higher weights; weights in [base, 1].
    """
    n, b = y.shape
    _, idx = torch.sort(y, dim=0)  # ascending
    ranks = torch.zeros_like(y, dtype=torch.float32)
    ar = torch.arange(n, device=y.device, dtype=torch.float32).unsqueeze(1).expand(n, b)
    ranks.scatter_(0, idx, ar)
    denom = max(n - 1, 1)
    perc = ranks / denom
    w = base + (1.0 - base) * (1.0 - perc).pow(gamma)
    return w


def step1_train_encdec_and_impute_all(
    codex_cell_embeddings,         # (Ns, d)
    gene_embeddings,               # (Ng, d) in HVG order
    subset_adata,                  # AnnData of subset cells, with proteins in .var_names
    protein_shared,                # ordered protein list in protein_shared.var.index
    rna_shared,                    # ordered gene list in rna_shared.var.index
    hvg_names,                     # list[str] defining HVG order for gene_embeddings
    *,
    device: str | None = None,
    K: int = 20,
    hidden: int = 64,
    epochs: int = 150,
    batch: int = 128,
    lr: float = 1e-3,
    wd: float = 1e-5,
    temp: float = 1.0,
    clamp_nonneg: bool = True,
    # loss weights
    w_mse: float = 0.6,
    w_corr: float = 0.3,
    w_bce: float = 0.1,
    w_l1: float = 1e-6,
    # low-expression emphasis
    wmse_base: float = 0.2,
    wmse_gamma: float = 2.0,
    # imputation batching
    impute_batch: int = 512,
    print_every: int = 20,
):
    """
    Encoder+Decoder version (ResGCN + BilinearDecoder) from your original step1.
    Trains on subset cells using protein/gene supervision, then imputes ALL HVGs for subset.

    Returns:
      enc, dec, Hc_sub, Hg_tr, pred_sub_all, valid_sub_all, info
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    codex_cell_embeddings = np.asarray(codex_cell_embeddings, dtype=np.float32)
    gene_embeddings = np.asarray(gene_embeddings, dtype=np.float32)

    Ns = codex_cell_embeddings.shape[0]
    Ng = gene_embeddings.shape[0]
    assert Ns == subset_adata.n_obs, "subset_adata must be SAME ORDER as codex_cell_embeddings"
    assert Ng == len(hvg_names), "gene_embeddings must match HVG order/length"

    # Node feats + type flags
    X_nodes = np.vstack([codex_cell_embeddings, gene_embeddings]).astype(np.float32)
    type_feat = np.zeros((Ns + Ng, 2), np.float32)
    type_feat[:Ns, 0] = 1.0
    type_feat[Ns:, 1] = 1.0
    X_nodes = np.hstack([X_nodes, type_feat]).astype(np.float32)
    Xt = torch.as_tensor(X_nodes, dtype=torch.float32, device=device)

    # Graph: kNN on original embeddings (drop gene↔gene edges)
    ei, ew = knn_edges_cosine(np.vstack([codex_cell_embeddings, gene_embeddings]), K=K)
    r, c = ei[0], ei[1]
    keep = ~((r >= Ns) & (c >= Ns))  # remove gene-gene edges
    ei = np.vstack([r[keep], c[keep]])
    ew = ew[keep]
    ei_t = torch.as_tensor(ei, dtype=torch.long, device=device)
    ew_t = torch.as_tensor(ew, dtype=torch.float32, device=device)

    # Ordered matched pairs: protein_shared.var.index ↔ rna_shared.var.index
    prot_list = list(protein_shared.var.index.to_list())
    gene_list = list(rna_shared.var.index.to_list())
    assert len(prot_list) == len(gene_list)

    g2hvg = {g: i for i, g in enumerate(hvg_names)}
    pairs = [(g2hvg[g], p) for g, p in zip(gene_list, prot_list)
             if (g in g2hvg) and (p in subset_adata.var_names)]
    if len(pairs) == 0:
        raise ValueError("No overlapping gene/protein pairs for Step 1 (enc+dec).")

    G_idx = np.array([gi for gi, _ in pairs], dtype=int)
    P_names = [p for _, p in pairs]

    # RAW protein targets (Ns x M)
    Y = np.stack([sparse_to_1d(subset_adata[:, p].X).astype(np.float32) for p in P_names], axis=1)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)

    # Models
    enc = ResGCN(d_in=X_nodes.shape[1], hidden=hidden, drop=0.1).to(device)
    dec = BilinearDecoder(hidden=hidden).to(device)
    opt = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr, weight_decay=wd)

    M = len(G_idx)
    B = min(batch, M)

    for ep in range(1, epochs + 1):
        enc.train()
        dec.train()
        opt.zero_grad()

        H = enc(Xt, ei_t, ew_t)     # (Ns+Ng, hidden)
        Hc = H[:Ns]

        idx = np.random.choice(M, size=B, replace=False)
        gi = torch.as_tensor(G_idx[idx], device=device)
        Hg = H[Ns + gi]             # (B, hidden)
        y = Yt[:, idx]              # (Ns, B)

        pred = dec(Hc, Hg) / temp

        w_low = low_expression_weights(y, base=wmse_base, gamma=wmse_gamma)
        mse_term = ((pred - y) ** 2 * w_low).sum() / (w_low.sum() + 1e-12)
        corr_term = pearson_loss(pred, y)
        bce_term = bce_topk_loss(pred, y, pos_q=0.90, neg_q=0.50)
        l1_term = pred.abs().mean()

        loss = w_mse * mse_term + w_corr * corr_term + w_bce * bce_term + w_l1 * l1_term
        loss.backward()
        opt.step()

        if print_every and (ep % print_every == 0):
            print(f"[Step1-encdec] ep {ep:03d} | total {loss.item():.4f} | wmse {mse_term.item():.4f} "
                  f"| corr {corr_term.item():.4f} | bce {bce_term.item():.4f}")

    # Freeze latents & impute ALL HVGs for subset
    enc.eval()
    dec.eval()
    with torch.no_grad():
        H = enc(Xt, ei_t, ew_t)
        Hc_sub = H[:Ns].detach()    # (Ns, hidden)
        Hg_tr  = H[Ns:].detach()    # (Ng, hidden) in HVG order

    pred_sub_all = np.empty((Ns, Ng), np.float32)
    with torch.no_grad():
        for s in range(0, Ng, impute_batch):
            Hg_chunk = Hg_tr[s:s + impute_batch]
            S = dec(Hc_sub, Hg_chunk)
            if clamp_nonneg:
                S = torch.clamp(S, min=0)
            pred_sub_all[:, s:s + Hg_chunk.shape[0]] = S.detach().cpu().numpy().astype(np.float32)

    valid_sub_all = list(hvg_names)
    info = {
        "num_supervised_pairs": M,
        "protein_names_supervised": P_names,
        "gene_indices_supervised": G_idx,
        "K": K,
        "hidden": hidden,
    }
    return enc, dec, Hc_sub, Hg_tr, pred_sub_all, valid_sub_all, info
