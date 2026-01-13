# src/yourpkg/data.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np


@dataclass(frozen=True)
class MultiModalData:
    rna_X_norm: np.ndarray
    protein_X_norm: np.ndarray
    rna_adata: Any

    # NEW: RNA latent coordinates for RNA–RNA neighbor search (e.g., rna_adata.obsm["X_pca_harmony"])
    rna_latent: Optional[np.ndarray] = None

    # Optional: store CODEX AnnData if you have it in the workflow
    codex_adata: Optional[Any] = None
