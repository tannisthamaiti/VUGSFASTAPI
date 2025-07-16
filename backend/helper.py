"""
helper.py – utilities for depth-aware FMI plotting
"""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")             # head-less server backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io

# ─── config ────────────────────────────────────────────────────────────
MISSING_THRESHOLD = -5000         # ≤ −5000 → missing / NaN


# ─── loaders ───────────────────────────────────────────────────────────
# def load_depth(depth_path: Path) -> np.ndarray:
#     """
#     Read single-column CSV → (N,) float64 vector.
#     """
#     if not depth_path.exists():
#         raise FileNotFoundError(depth_path)
#     depth = pd.read_csv(depth_path, header=None,nrows=50000).astype(float).values.squeeze()
#     return depth


# def load_and_scale(array_path: Path) -> np.ndarray:
#     """
#     Read 2-D CSV, replace sentinel values with NaN, then Min-Max scale
#     each column independently. NaNs become 0 after scaling.
#     """
#     if not array_path.exists():
#         raise FileNotFoundError(array_path)
#     arr = pd.read_csv(array_path, header=None,nrows=50000).astype(float).values
#     arr[arr <= MISSING_THRESHOLD] = np.nan

#     col_min = np.nanmin(arr, axis=0, keepdims=True)
#     col_max = np.nanmax(arr, axis=0, keepdims=True)
#     scaled  = (arr - col_min) / (col_max - col_min)
#     return np.nan_to_num(scaled, nan=0.0)


MISSING_THRESHOLD = -998.0  # or your defined threshold

def load_depth(depth_input: np.ndarray) -> np.ndarray:
    """
    Accepts a NumPy array and returns a 1D float64 depth vector.
    """
    return depth_input


def load_and_scale(array_input: np.ndarray) -> np.ndarray:
    """
    Accepts a 2D NumPy array, replaces missing values, and returns
    a Min-Max scaled array with NaNs replaced by 0.
    """
    print(array_input)
    # arr = np.asarray(array_input, dtype=float)
    # arr[arr <= MISSING_THRESHOLD] = np.nan

    # col_min = np.nanmin(arr, axis=0, keepdims=True)
    # col_max = np.nanmax(arr, axis=0, keepdims=True)
    # scaled  = (arr - col_min) / (col_max - col_min)

    #return np.nan_to_num(scaled, nan=0.0)
    return array_input



# ─── rendering ─────────────────────────────────────────────────────────

def array_to_png(arr: np.ndarray, depth: np.ndarray) -> bytes:
    """
    Return a PNG byte-string of a depth-referenced heat-map.
    """
    if arr.shape[0] != depth.size:
        raise ValueError("Depth vector length and array row count mismatch")

    extent = [0, arr.shape[1], depth[-1], depth[0]]  # Flip Y-axis

    # Adjust height for clarity
    height_per_row = 0.004
    fig_height = max(6, arr.shape[0] * height_per_row)
    fig_width = 6

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

    im = ax.imshow(arr, cmap="YlOrBr", aspect="auto", extent=extent)
    #ax.set_title("FMI")
    ax.set_ylabel("Depth (m)")
    ax.set_xlabel("Column Index")

    # Add horizontal colorbar on top
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.2)
    cbar.set_label("Scaled Intensity (0-1)")

    # Configure Y-ticks
    num_ticks = 10
    y_ticks = np.linspace(depth.min(), depth.max(), num_ticks)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='both', which='major', labelsize=8)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
