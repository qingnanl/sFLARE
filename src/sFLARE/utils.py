# ===== src/sFLARE/viz.py =====
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np


# ============================================================
# 1) Spatial feature plotting (ordered rows, no ID matching)
# ============================================================
def plot_spatial_features_ordered(
    adata,
    spa_df,
    features,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    ncols: int = 2,
    cmap: str = "viridis",
    robust: bool = True,
    q: Tuple[float, float] = (1, 99),
    point_max: int = 6,
    show: bool = True,
    return_fig: bool = False,
):
    """
    Assumes spa_df rows are in the SAME ORDER as adata.obs (no ID matching).

    Parameters
    ----------
    adata : AnnData
    spa_df : pd.DataFrame
    features : str or list[str]
    x_col, y_col : coordinate columns in spa_df
    robust : if True, clip color scale to percentiles q
    q : (low, high) percentiles when robust=True
    point_max : max point size (auto scaled by n_obs)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from scipy.sparse import issparse

    if isinstance(features, str):
        features = [features]
    assert adata.n_obs == len(spa_df), "adata.n_obs must equal len(spa_df)."

    x = spa_df[x_col].to_numpy()
    y = spa_df[y_col].to_numpy()
    pt = max(1, min(point_max, 50000 // max(1, adata.n_obs)))

    n = len(features)
    ncols = max(1, int(ncols))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows), dpi=150)
    axes = np.atleast_1d(axes).ravel()

    for i, feat in enumerate(features):
        ax = axes[i]
        if feat not in adata.var_names:
            ax.text(0.5, 0.5, f"'{feat}' not found", ha="center", va="center")
            ax.axis("off")
            continue

        vals = adata[:, feat].X
        vals = vals.A1 if issparse(vals) else np.asarray(vals).ravel()

        if robust:
            vmin, vmax = np.nanpercentile(vals, q[0]), np.nanpercentile(vals, q[1])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        else:
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)

        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        ax.scatter(x, y, c=vals, s=pt, lw=0, cmap=cmap, norm=norm)
        ax.set_title(feat)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.035, pad=0.02)

    # hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    if show:
        plt.show()
        if not return_fig:
            plt.close(fig)
            return None
    return fig


# ============================================================
# 2) Recovery / precision@k and plotting over cycles
# ============================================================
def build_pos_by_cell(edges_protein, edges_gene):
    i = np.asarray(edges_protein, dtype=int)
    j = np.asarray(edges_gene, dtype=int)
    pos = defaultdict(set)
    for ci, gj in zip(i, j):
        pos[ci].add(gj)
    return pos


def l2_topk_indices(P_block: np.ndarray, G: np.ndarray, K: int) -> np.ndarray:
    """
    Fast top-K by Euclidean distance using argpartition (unordered).
    P_block: (B, d), G: (N, d) -> returns indices (B, K)
    """
    P_block = np.asarray(P_block, dtype=np.float32)
    G = np.asarray(G, dtype=np.float32)

    A2 = (P_block * P_block).sum(axis=1, keepdims=True)   # (B,1)
    B2 = (G * G).sum(axis=1, keepdims=True).T             # (1,N)
    D2 = A2 + B2 - 2.0 * (P_block @ G.T)
    np.maximum(D2, 0, out=D2)

    K = min(int(K), G.shape[0])
    return np.argpartition(D2, kth=K - 1, axis=1)[:, :K]


def precision_at_k_euclidean(
    P: np.ndarray,
    G: np.ndarray,
    edges_protein,
    edges_gene,
    *,
    ks: Sequence[int] = (10, 50, 100, 500),
    batch: int = 1024,
    include_cells_without_pos: bool = False,
):
    """
    Per cell precision@k(i) = |TopK(i) ∩ Pos(i)| / k.
    Returns:
      per_cell: dict k -> np.ndarray per-cell precision
      summary : dict mean/median per k
    """
    P = np.asarray(P, dtype=np.float32)
    G = np.asarray(G, dtype=np.float32)

    pos_by_cell = build_pos_by_cell(edges_protein, edges_gene)
    all_cells = np.array(sorted(pos_by_cell.keys()), dtype=int)

    if not include_cells_without_pos:
        cells = np.array([c for c in all_cells if len(pos_by_cell[c]) > 0], dtype=int)
    else:
        cells = all_cells

    if cells.size == 0:
        raise ValueError("No cells with positives to evaluate.")

    ks = tuple(sorted({int(k) for k in ks}))
    Kmax = min(max(ks), G.shape[0])

    per_cell = {k: np.zeros(cells.size, dtype=np.float32) for k in ks}

    for s in range(0, cells.size, batch):
        cb = cells[s:s + batch]
        top_idx = l2_topk_indices(P[cb], G, Kmax)
        for r, ci in enumerate(cb):
            true_set = pos_by_cell[ci]
            if len(true_set) == 0:
                for k in ks:
                    per_cell[k][s + r] = 0.0
                continue
            for k in ks:
                topk = set(top_idx[r, :k].tolist())
                per_cell[k][s + r] = len(topk.intersection(true_set)) / float(k)

    summary: Dict[str, float] = {}
    for k in ks:
        vals = per_cell[k]
        summary[f"precision@{k}_mean"] = float(np.mean(vals))
        summary[f"precision@{k}_median"] = float(np.median(vals))
    return per_cell, summary


def add_recovery_to_losses_df(
    losses_df,
    embeddings_history,
    *,
    edges_protein,
    edges_gene,
    k: int = 500,
    start_cycle: int = 1,
    end_cycle: Optional[int] = None,
    batch: int = 1024,
):
    """
    Returns a new DataFrame: losses_df joined with precision@k_mean by cycle index.
    """
    import pandas as pd

    if end_cycle is None:
        end_cycle = len(embeddings_history) - 1

    rows = []
    for cyc in range(int(start_cycle), int(end_cycle) + 1):
        P = embeddings_history[cyc]["protein"]
        G = embeddings_history[cyc]["gene"]
        _, summary = precision_at_k_euclidean(P, G, edges_protein, edges_gene, ks=(k,), batch=batch)
        rows.append({"cycle": cyc, f"precision@{k}_mean": summary[f"precision@{k}_mean"]})

    rec_df = pd.DataFrame(rows).set_index("cycle")

    out = losses_df.copy()
    try:
        out.index = out.index.astype(int)
    except Exception:
        pass
    out = out.join(rec_df, how="left")
    return out


def save_line_dot_pdf(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    ylabel: str,
    outfile: str,
    xlabel: str = "Cycle",
):
    import matplotlib.pyplot as plt
    import os

    x = np.asarray(x)
    y = np.asarray(y)

    fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=150)
    ax.plot(x, y, marker="o", linestyle="-", linewidth=1.8, markersize=5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.35)

    if len(x) > 20:
        step = max(1, len(x) // 10)
        ax.set_xticks(x[::step])

    ax.set_position([0.17, 0.28, 0.78, 0.62])
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {os.path.abspath(outfile)}")


def plot_losses_and_recovery_pdfs(
    combined_df,
    *,
    cols: Sequence[str] = ("gene_local", "cell_kl_post", "precision@500_mean"),
    outfiles: Optional[Dict[str, str]] = None,
    start_cycle: int = 1,
):
    """
    Save line-dot PDFs for given columns in combined_df.
    """
    import matplotlib.pyplot as plt

    df = combined_df.copy()
    try:
        df.index = df.index.astype(int)
    except Exception:
        pass
    df = df.sort_index()
    df = df.loc[df.index >= int(start_cycle)]

    if outfiles is None:
        outfiles = {c: f"{c}_over_cycles.pdf" for c in cols}

    plt.rcParams.update({
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    x = df.index.values
    for c in cols:
        if c not in df.columns:
            print(f"Skip '{c}' (not in combined_df)")
            continue
        y = df[c].astype(float).fillna(0.0).values
        save_line_dot_pdf(
            x, y,
            title=c.replace("_", " ").title(),
            ylabel=c,
            outfile=outfiles[c],
        )


# ============================================================
# 3) UMAP co-embedding (cells + genes) with marker labels
# ============================================================
def _as_1d_str(a) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 1:
        a = a.ravel()
    return a.astype(str)


def _pick_palette(n: int):
    try:
        import seaborn as sns
        if n <= 20:
            return sns.color_palette("tab20", n_colors=n)
        return sns.color_palette("husl", n_colors=n)
    except Exception:
        import matplotlib.pyplot as plt
        if n <= 20:
            cmap = plt.get_cmap("tab20")
            return [cmap(i) for i in range(n)]
        cmap = plt.get_cmap("hsv")
        return [cmap(i / max(n, 1)) for i in range(n)]


def plot_umap_coembed_cells_genes(
    *,
    embeddings_history,
    cyc_selected: int,
    hvg_names: Sequence[str],
    subset_adata,
    subset_values: np.ndarray,
    marker_gene_list: Sequence[str],
    out_base: str,
    metric: str = "cosine",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    gene_point_size: float = 10,
    cell_point_size: float = 4,
    marker_point_size: float = 30,
    gene_alpha: float = 0.4,
    cell_alpha: float = 0.8,
    marker_alpha: float = 0.9,
    gene_color: str = "lightgray",
    marker_color: str = "red",
    marker_edgecolor: str = "black",
    label_markers: bool = True,
    label_fontsize: int = 8,
    label_color: str = "darkred",
    legend: bool = True,
    figsize: Tuple[float, float] = (12, 9),
    save_pdf: bool = True,
    save_png: bool = False,
    png_dpi: int = 300,
    save_svg: bool = False,
    show: bool = False,
):
    """
    UMAP co-embedding of:
      - protein/codex cell embeddings (embeddings_history[cyc_selected]["protein"])
      - gene embeddings (embeddings_history[cyc_selected]["gene"], assumed HVG order)

    Assumptions:
      - subset_adata.obs.index contains GLOBAL integer indices (or strings castable to int)
        into subset_values (len=subset_values = #ALL cells).
      - subset_adata is in the SAME ORDER as the protein embeddings for this cycle.
      - gene embeddings correspond to hvg_names order.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["axes.grid"] = False

    # embeddings
    codex_cell_embeddings = np.asarray(embeddings_history[cyc_selected]["protein"], dtype=np.float32)
    gene_embeddings = np.asarray(embeddings_history[cyc_selected]["gene"], dtype=np.float32)

    Ns = codex_cell_embeddings.shape[0]
    Ng = gene_embeddings.shape[0]
    if Ng != len(hvg_names):
        raise ValueError(f"gene_embeddings has {Ng} rows but len(hvg_names)={len(hvg_names)}.")

    # labels for subset cells
    sel_idx = np.asarray(subset_adata.obs.index)
    try:
        sel_idx_int = sel_idx.astype(int)
    except Exception as e:
        raise ValueError("subset_adata.obs.index must be integer-like global indices.") from e

    subset_values = np.asarray(subset_values)
    if subset_values.ndim != 1:
        raise ValueError("subset_values must be a 1D array of labels for ALL cells.")
    if sel_idx_int.max() >= subset_values.shape[0]:
        raise ValueError("subset_adata.obs.index contains global indices out of range for subset_values.")

    mnn_protein_labels = _as_1d_str(subset_values[sel_idx_int])
    if len(mnn_protein_labels) != Ns:
        raise ValueError(
            f"Label length mismatch: got {len(mnn_protein_labels)} labels from subset_adata, "
            f"but Ns={Ns} protein embeddings. Ensure subset_adata matches protein embedding order/size."
        )

    # UMAP on combined embeddings
    all_embeddings = np.vstack([codex_cell_embeddings, gene_embeddings])
    categories = np.array(["protein"] * Ns + ["gene"] * Ng)

    try:
        import umap
    except Exception as e:
        raise ImportError("Please install umap-learn (pip install umap-learn).") from e

    reducer = umap.UMAP(
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=str(metric),
        random_state=int(random_state),
    )
    embedding_umap = reducer.fit_transform(all_embeddings)

    # marker genes
    gene_names_for_embeddings = np.asarray(hvg_names)
    gene_idx_start = Ns
    marker_set = set(marker_gene_list)

    highlight_gene_indices: List[int] = []
    highlight_gene_labels: List[str] = []
    for i, g in enumerate(gene_names_for_embeddings):
        if g in marker_set:
            highlight_gene_indices.append(gene_idx_start + i)
            highlight_gene_labels.append(str(g))

    # colors for cell labels
    unique_celltypes = np.unique(mnn_protein_labels)
    palette = _pick_palette(len(unique_celltypes))
    celltype_to_color = dict(zip(unique_celltypes, palette))

    # plot
    fig = plt.figure(figsize=figsize, dpi=150)

    # genes
    gene_idx = np.where(categories == "gene")[0]
    plt.scatter(
        embedding_umap[gene_idx, 0], embedding_umap[gene_idx, 1],
        s=gene_point_size, c=gene_color, alpha=gene_alpha, label="Genes"
    )

    # highlighted marker genes
    if len(highlight_gene_indices) > 0:
        plt.scatter(
            embedding_umap[highlight_gene_indices, 0], embedding_umap[highlight_gene_indices, 1],
            s=marker_point_size, c=marker_color, edgecolor=marker_edgecolor,
            alpha=marker_alpha, label="Marker Genes"
        )

    # cells by label
    protein_idx = np.where(categories == "protein")[0]
    for celltype in unique_celltypes:
        mask = (mnn_protein_labels == celltype)
        if np.any(mask):
            umap_indices = protein_idx[np.where(mask)[0]]
            plt.scatter(
                embedding_umap[umap_indices, 0], embedding_umap[umap_indices, 1],
                s=cell_point_size, color=celltype_to_color[celltype],
                alpha=cell_alpha, label=str(celltype)
            )

    # labels + adjustText
    if label_markers and len(highlight_gene_indices) > 0:
        texts = []
        for idx, label in zip(highlight_gene_indices, highlight_gene_labels):
            x, y = embedding_umap[idx, 0], embedding_umap[idx, 1]
            texts.append(plt.text(x, y, label, fontsize=label_fontsize, weight="bold", color=label_color))
        try:
            from adjustText import adjust_text
            adjust_text(
                texts,
                only_move={"points": "y", "text": "y"},
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            )
        except Exception:
            pass

    if legend:
        plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title(
        f"UMAP Co-embedding (cycle {cyc_selected})\n"
        "Selected CODEX cells colored by cluster; marker genes highlighted"
    )
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.tight_layout()

    # save
    if save_pdf:
        plt.savefig(f"{out_base}.pdf", bbox_inches="tight")
    if save_png:
        plt.savefig(f"{out_base}.png", dpi=int(png_dpi), bbox_inches="tight")
    if save_svg:
        plt.savefig(f"{out_base}.svg", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "embedding_umap": embedding_umap,
        "highlight_gene_indices": np.asarray(highlight_gene_indices, dtype=int),
        "highlight_gene_labels": highlight_gene_labels,
        "unique_celltypes": unique_celltypes,
    }
