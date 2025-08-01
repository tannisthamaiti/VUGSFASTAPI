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


def save_contours_to_csv(contour_csv_outputs, sha_short, directory="/app/well_files"):
    """
    Save grouped contour data (x[], depth_m[]) to a CSV with native Python types.
    """
    all_contour_rows = []

    for contour_group in contour_csv_outputs:
        for contour in contour_group:
            contour_id = int(contour["contour_id"])
            x_list = contour["x"] #[int(x) for x in contour["x"]]  # Convert np.int64 to int
            y_list = contour["depth_m"] #[float(y) for y in contour["depth_m"]]  # Convert np.float64 to float
            area = float(contour["area"])
            hole_radius=float(contour["hole_radius"])

            all_contour_rows.append({
                "contour_id": contour_id,
                "x": x_list,
                "depth_m": y_list,
                "area": area,
                "hole_radius": hole_radius
            })

    df = pd.DataFrame(all_contour_rows)

    os.makedirs(directory, exist_ok=True)
    filename = f"vug_contours_{sha_short}.csv"
    full_path = os.path.join(directory, filename)

    df.to_csv(full_path, index=False)
    print(f"[INFO] Saved cleaned contour data to {full_path}")

    return full_path



