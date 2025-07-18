import matplotlib.pyplot as plt
import numpy as np
import io
from multiprocessing import Pool
from utils.contours import get_contours, get_centeroid
from utils.processing import apply_adaptive_thresholding
import os
from multiprocessing import Pool, cpu_count

def create_colorbar(ax, im, fontsize=12):
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.2)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label("Scaled Intensity", fontsize=fontsize)

def plot_single_image_contours(
        img, circles, linewidth=2, cmap='YlOrBr', color='black',
        legend=False, colorbar=False, title='', fontsize=12, labelsize=12,
        axis_off=False, depth_vector=None
):
    # Set dynamic height
    height_per_row = 0.004
    fig_height = max(4, img.shape[0] * height_per_row)
    fig_width = 6

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

    # Handle depth (y-axis) scaling
    if depth_vector is not None and len(depth_vector) == img.shape[0]:
        extent = [0, img.shape[1], depth_vector[-1], depth_vector[0]]  # Flip y-axis
        im = ax.imshow(img, cmap=cmap, aspect='auto', extent=extent)
        ax.set_ylabel("Depth (m)")
        ax.set_yticks(np.linspace(depth_vector[0], depth_vector[-1], num=10))
    else:
        im = ax.imshow(img, cmap=cmap, aspect='auto')
        ax.set_ylabel("Pixel Row Index")

    for i, pts in enumerate(circles):
        x = pts[:, 0, 0]
        y = pts[:, 0, 1]
        x = np.append(x, x[0])
        y = np.append(y, y[0])

        if isinstance(legend, bool) and legend:
            legend_label = str(i)
        elif legend:
            legend_label = legend
        else:
            legend_label = None

        ax.plot(x, y, color=color, linewidth=linewidth, label=legend_label)

    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)

    if axis_off:
        ax.axis('off')

    if colorbar:
        create_colorbar(ax, im, fontsize=labelsize)

    plt.tight_layout()

    save_path = "/app/well_files/test.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    with open(save_path, 'wb') as f:
        f.write(buf.getvalue())

    plt.close(fig)
    return buf.getvalue()



def plot_worker(args):
    img, circles, idx = args
    try:
        img_bytes = plot_single_image_contours(
            img, circles, title=f"Image {idx}", axis_off=True
        )
        return idx, img_bytes
    except Exception as e:
        print(f"[ERROR] Image {idx} failed: {e}")
        return idx, None
    
def process_single_threshold(args):
    (
        fmi_array, diff_thresh, block_size, c_threshold,
        depth_start, depth_end, radius_cm, pix_len_cm,
        min_vug_area, max_vug_area, min_circ_ratio, max_circ_ratio,
        fontsize, labelsize, colorbar
    ) = args

    try:
        threshold_img, _ = apply_adaptive_thresholding(
            fmi_array, diff_thresh, block_size=block_size, c=c_threshold
        )

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
            print(f"[WARNING] No contours at threshold {diff_thresh:.2f}")
            return None

        mode_subtracted = np.abs(fmi_array - diff_thresh)
        png = plot_single_image_contours(
            mode_subtracted, contours,
            cmap='YlOrBr',
            linewidth=2,
            colorbar=colorbar,
            fontsize=fontsize,
            labelsize=labelsize,
            axis_off=False,
            title=f"Threshold {diff_thresh:.2f}"
        )
        return png

    except Exception as e:
        print(f"[ERROR] Failed at threshold {diff_thresh:.2f}: {e}")
        return None

def plot_fmi_with_area_circularity_filtered_contours_parallel(
    fmi_array_one_meter_zone,
    fmi_array_unscaled_one_meter_zone,
    different_thresholds,
    block_size,
    c_threshold,
    one_meter_zone_start,
    one_meter_zone_end,
    well_radius_one_meter_zone,
    tdep_array_one_meter_zone,
    min_vug_area,
    max_vug_area,
    min_circ_ratio,
    max_circ_ratio,
    depth,
    depth_in_name=False,
    picture_format='png',
    custom_fname='',
    colorbar=True,
    labelsize=10,
    fontsize=10,
    figsize=(6, 6),
    well_name=''
):
    print("[INFO] Starting parallel contour plotting...")

    try:
        hole_radius_cm = well_radius_one_meter_zone.mean() * 100
        pixel_length_cm = np.diff(tdep_array_one_meter_zone).mean() * 100
        print(f"[DEBUG] Hole radius (cm): {hole_radius_cm}, Pixel length (cm): {pixel_length_cm}")
    except Exception as e:
        print(f"[ERROR] Error computing physical params: {e}")
        return []

    args_list = [
        (
            fmi_array_one_meter_zone, diff_thresh, block_size, c_threshold,
            one_meter_zone_start, one_meter_zone_end,
            hole_radius_cm, pixel_length_cm,
            min_vug_area, max_vug_area, min_circ_ratio, max_circ_ratio,
            fontsize, labelsize, colorbar
        )
        for diff_thresh in different_thresholds
    ]

    with Pool(processes=min(len(different_thresholds), cpu_count())) as pool:
        results = pool.map(process_single_threshold, args_list)

    # Filter out None results
    png_outputs = [res for res in results if res is not None]

    print(f"[INFO] Total PNG images generated: {len(png_outputs)}")
    return png_outputs


