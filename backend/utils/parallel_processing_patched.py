
import matplotlib.pyplot as plt
import numpy as np
import io
import os
from PIL import Image
from multiprocessing import Pool, cpu_count
from utils.contours import get_contours
from utils.processing import apply_adaptive_thresholding


def create_colorbar(ax, im, fontsize=12):
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.2)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label("Scaled Intensity", fontsize=fontsize)


def plot_single_image_contours(
    img, circles, linewidth=2, cmap='YlOrBr', color='black',
    legend=False, colorbar=False, title='', fontsize=12, labelsize=12,
    axis_off=False, depth_vector=None
):
    height_per_row = 0.004
    fig_height = max(4, img.shape[0] * height_per_row)
    fig_width = 6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

    if depth_vector is not None and len(depth_vector) == img.shape[0]:
        extent = [0, img.shape[1], depth_vector[-1], depth_vector[0]]
        im = ax.imshow(img, cmap=cmap, aspect='auto', extent=extent)
        ax.set_ylabel("Depth (m)")
    else:
        im = ax.imshow(img, cmap=cmap, aspect='auto')
        ax.set_ylabel("Pixel Index")

    for i, pts in enumerate(circles):
        x = pts[:, 0, 0]
        y = pts[:, 0, 1]
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        ax.plot(x, y, color=color, linewidth=linewidth)

    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)

    if axis_off:
        ax.axis('off')
    if colorbar:
        create_colorbar(ax, im, fontsize=labelsize)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def process_patch(args):
    (
        fmi_patch, tdep_patch, diff_thresh, block_size, c_threshold,
        depth_start, depth_end, radius_cm, pix_len_cm,
        min_vug_area, max_vug_area, min_circ_ratio, max_circ_ratio,
        fontsize, labelsize, colorbar, idx
    ) = args
    try:
        threshold_img, _ = apply_adaptive_thresholding(fmi_patch, diff_thresh, block_size=block_size, c=c_threshold)
        contours, centroids, vugs = get_contours(
            threshold_img,
            depth_to=depth_start,
            depth_from=depth_end,
            radius=radius_cm,
            pix_len=pix_len_cm,
            min_vug_area=min_vug_area,
            max_vug_area=max_vug_area,
            min_circ_ratio=min_circ_ratio,
            max_circ_ratio=max_circ_ratio
        )
        if len(contours) == 0:
            return idx, None
        mode_subtracted = np.abs(fmi_patch - diff_thresh)
        png = plot_single_image_contours(
            mode_subtracted, contours,
            cmap='YlOrBr',
            linewidth=2,
            colorbar=colorbar,
            fontsize=fontsize,
            labelsize=labelsize,
            axis_off=False,
            title=f"Patch {idx} Thresh {diff_thresh:.2f}",
            depth_vector=tdep_patch
        )
        return idx, png
    except Exception as e:
        print(f"[ERROR] Patch {idx} failed: {e}")
        return idx, None


def stitch_images_vertically(image_bytes_list):
    from PIL import Image
    images = [Image.open(io.BytesIO(b)) for b in image_bytes_list if b]
    widths, heights = zip(*(img.size for img in images))
    stitched_img = Image.new('RGB', (max(widths), sum(heights)))
    y = 0
    for im in images:
        stitched_img.paste(im, (0, y))
        y += im.height
    out_path = "/mnt/data/final_stitched_fmi.png"
    stitched_img.save(out_path)
    return out_path


def plot_fmi_parallel_patchwise(
    fmi_array, tdep_array, patch_height,
    diff_thresh, block_size, c_threshold,
    depth_start, depth_end, radius_cm, pix_len_cm,
    min_vug_area, max_vug_area, min_circ_ratio, max_circ_ratio,
    fontsize=10, labelsize=10, colorbar=True
):
    patches = []
    step = patch_height
    for i in range(0, fmi_array.shape[0], step):
        patch_img = fmi_array[i:i+step]
        patch_depth = tdep_array[i:i+step]
        if patch_img.shape[0] < 5:
            continue
        patches.append((
            patch_img, patch_depth, diff_thresh, block_size, c_threshold,
            depth_start, depth_end, radius_cm, pix_len_cm,
            min_vug_area, max_vug_area, min_circ_ratio, max_circ_ratio,
            fontsize, labelsize, colorbar, i
        ))

    with Pool(min(cpu_count(), len(patches))) as pool:
        results = pool.map(process_patch, patches)

    ordered_imgs = [img for _, img in sorted(results) if img]
    if not ordered_imgs:
        print("[WARNING] No images were generated.")
        return None
    return stitch_images_vertically(ordered_imgs)
