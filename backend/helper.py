"""
helper.py – utilities for depth-aware FMI plotting
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")             # head-less server backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
from multiprocessing import Pool, cpu_count
from PIL import Image
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

def render_batch(batch_arr: np.ndarray, batch_depth: np.ndarray) -> bytes:
    """Render a single batch to a PNG byte stream (no colorbar)."""
    extent = [0, batch_arr.shape[1], batch_depth[-1], batch_depth[0]]

    # Adjust height for clarity
    height_per_row = 0.004
    fig_height = max(4, batch_arr.shape[0] * height_per_row)
    fig_width = 6

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    im = ax.imshow(batch_arr, cmap="YlOrBr", aspect="auto", extent=extent)

    ax.set_ylabel("Depth (m)")
    ax.set_xlabel("Column Index")

    # Y-ticks
    num_ticks = min(6, len(batch_depth))
    y_ticks = np.linspace(batch_depth.min(), batch_depth.max(), num_ticks)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='both', which='major', labelsize=8)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def array_to_png_batches_parallel(arr: np.ndarray, depth: np.ndarray, batch_size: int = 200) -> bytes:
    """
    Process FMI array in batches and return a stitched PNG image byte stream (parallel, no colorbar).
    """
    if arr.shape[0] != depth.size:
        raise ValueError("Depth vector length and array row count mismatch")

    batch_inputs = []
    for start_idx in range(0, arr.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, arr.shape[0])
        batch_arr = arr[start_idx:end_idx, :]
        batch_depth = depth[start_idx:end_idx]
        batch_inputs.append((batch_arr, batch_depth))

    # Use multiprocessing to process batches in parallel
    with Pool(processes=min(cpu_count(), len(batch_inputs))) as pool:
        batch_byte_images = pool.starmap(render_batch, batch_inputs)

    # Convert byte-images to PIL and stitch vertically
    batch_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in batch_byte_images]
    total_width = max(img.width for img in batch_images)
    total_height = sum(img.height for img in batch_images)

    stitched_image = Image.new("RGB", (total_width, total_height))
    y_offset = 0
    for img in batch_images:
        stitched_image.paste(img, (0, y_offset))
        y_offset += img.height

    # Final image as byte stream
    final_buf = io.BytesIO()
    stitched_image.save(final_buf, format="PNG")
    final_buf.seek(0)
    return final_buf.read()


def save_contours_to_csv(contour_csv_outputs, sha_short, timestamp, directory="/app/well_files"):
    """
    Flatten and save contour data (x, depth_m) to a CSV file with sha and timestamp in filename.

    Parameters:
    - contour_csv_outputs: List of lists of dicts (contour points).
    - sha_short: Short SHA string to uniquely identify the file.
    - timestamp: UTC timestamp string in format YYYYMMDDTHHMMSSZ.
    - directory: Output directory path. Default is '/app/well_files'.

    Returns:
    - full_path: The path where the CSV was saved.
    """
    all_contour_points = []

    for contour_list in contour_csv_outputs:
        for point in contour_list:
            all_contour_points.append({
                "contour_id": point["contour_id"],
                "x": int(point["x"]),
                "depth_m": float(point["depth_m"])
            })

    df = pd.DataFrame(all_contour_points)

    # Build the output path
    filename = f"/app/well_files/vug_contours_{sha_short}_{timestamp}.csv"
    full_path = os.path.join(directory, filename)

    df.to_csv(full_path, index=False)
    print(f"[INFO] Saved {len(all_contour_points)} contour points to {full_path}")

    return full_path
