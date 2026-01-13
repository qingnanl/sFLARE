# src/yourpkg/data.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

# data.py
@dataclass(frozen=True)
class MultiModalData:
    rna_X_norm: np.ndarray
    protein_X_norm: np.ndarray
    rna_adata: Any
    codex_adata: Any | None = None