import matplotlib.pyplot as plt
import numpy as np
import io
from multiprocessing import Pool
from utils.contours import get_contours, get_centeroid
from utils.processing import apply_adaptive_thresholding
import os

def create_colorbar(ax, im, fontsize=12):
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.2)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label("Scaled Intensity", fontsize=fontsize)

def plot_single_image_contours(
        img, circles, linewidth=2, cmap='YlOrBr', color='black',
        legend=False, colorbar=False, title='', fontsize=12, labelsize=12,
        axis_off=False
):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    im = ax.imshow(img, cmap=cmap)

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
    if axis_off:
        ax.axis('off')

    if colorbar:
        create_colorbar(ax, im, fontsize=labelsize)

    plt.tight_layout()

    # Save image to hardcoded path
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

def parallel_plot_contours(image_list, contour_list, num_workers=4):
    """
    Parallel plotting of images with contours.
    Returns: List of PNG byte-strings (ordered by index).
    """
    assert len(image_list) == len(contour_list)
    inputs = [(img, circ, i) for i, (img, circ) in enumerate(zip(image_list, contour_list))]

    with Pool(processes=num_workers) as pool:
        results = pool.map(plot_worker, inputs)

    results.sort(key=lambda x: x[0])  # sort by index
    return [r[1] for r in results if r[1] is not None]

def plot_fmi_with_area_circularity_filtered_contours(
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
    """
    Return a list of PNG bytes for each thresholded FMI image with filtered contours.
    """
    png_outputs = []

    # Format file naming suffixes
    depth_str = str(round(depth, 2)) if depth_in_name else ''
    custom_str = f'_{custom_fname}' if custom_fname else ''
    well_str = f'_{well_name}' if well_name else ''

    print("[DEBUG] Generating file suffixes:", depth_str, custom_str, well_str)

    try:
        hole_radius_cm = well_radius_one_meter_zone.mean() * 100
        pixel_length_cm = np.diff(tdep_array_one_meter_zone).mean() * 100
        print(f"[DEBUG] Hole radius (cm): {hole_radius_cm}, Pixel length (cm): {pixel_length_cm}")
    except Exception as e:
        print(f"[ERROR] Error calculating physical parameters: {e}")
        return []

    for i, diff_thresh in enumerate(different_thresholds):
        print(f"[INFO] Processing threshold {i+1}/{len(different_thresholds)}: {diff_thresh:.2f}")

        try:
            threshold_img, C_value = apply_adaptive_thresholding(
                fmi_array_one_meter_zone, diff_thresh, block_size=block_size, c=c_threshold
            )
            print(f"[DEBUG] Adaptive thresholding done. C value used: {C_value}")
        except Exception as e:
            print(f"[ERROR] Adaptive thresholding failed at index {i}: {e}")
            continue

        try:
            contours, centroids, vugs = get_contours(
                threshold_img,
                depth_to=one_meter_zone_start,
                depth_from=one_meter_zone_end,
                radius=hole_radius_cm,
                pix_len=pixel_length_cm,
                min_vug_area=min_vug_area,
                max_vug_area=max_vug_area,
                min_circ_ratio=min_circ_ratio,
                max_circ_ratio=max_circ_ratio
            )
            print(f"[DEBUG] Contours length: {len(contours)}")
            print(f"[DEBUG] Centroids length: {len(centroids)}")
            print(f"[DEBUG] Vugs length: {len(vugs)}")
            print(f"[DEBUG] Contours found: {len(contours)}, Vugs filtered: {len(vugs)}")
        except Exception as e:
            print(f"[ERROR] Contour extraction failed at index {i}: {e}")
            continue

        try:
            mode_subtracted_fmi = np.abs(fmi_array_one_meter_zone - diff_thresh)
            if len(contours) == 0:
                print(f"[WARNING] No contours found at threshold {diff_thresh:.2f}, skipping plot.")
                continue
            png_bytes = plot_single_image_contours(
                mode_subtracted_fmi, contours,
                cmap='YlOrBr',
                linewidth=2,
                colorbar=colorbar,
                fontsize=fontsize,
                labelsize=labelsize,
                axis_off=False,
                title=f"Threshold {diff_thresh:.2f}"
            )

            if png_bytes is not None:
                print(f"[DEBUG] PNG image generated successfully at threshold {diff_thresh:.2f}")
                png_outputs.append(png_bytes)
            else:
                print(f"[WARNING] PNG generation returned None at threshold {diff_thresh:.2f}")
        except Exception as e:
            print(f"[ERROR] PNG generation failed at index {i}: {e}")

    print(f"[INFO] Total PNG images generated: {len(png_outputs)}")
    return png_outputs


    return png_outputs
def batch_worker(args):
    # args = tuple of required inputs from outer scope
    return plot_fmi_with_area_circularity_filtered_contours(*args)