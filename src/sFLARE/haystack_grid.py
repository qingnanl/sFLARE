# src/yourpkg/haystack_grid.py
from __future__ import annotations
from typing import Tuple
import numpy as np

def compute_grid_density_from_coord(
    coord: np.ndarray,
    n_grid_points: int = 100,
    random_state: int = 1,
    verbose: bool = True,
):
    """
    Thin wrapper around singleCellHaystack grid helpers.
    Returns (grid_points, grid_dist, grid_density).
    """
    from singleCellHaystack._grid import (
        calculate_grid_points,
        calculate_dist_to_cells,
        calculate_density,
    )

    grid_points = calculate_grid_points(coord, n_grid_points, random_state=random_state, verbose=verbose)
    grid_dist = calculate_dist_to_cells(coord, grid_points, verbose=verbose)
    grid_density = calculate_density(grid_dist, verbose=verbose)
    return grid_points, grid_dist, grid_density
