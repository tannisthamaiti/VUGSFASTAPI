import matplotlib.pyplot as plt
import numpy as np
import io
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
#from multiprocessing import Pool
from utils.contours import get_contours, get_centeroid
from utils.processing import apply_adaptive_thresholding
import os
#from multiprocessing import Pool, cpu_count
from multiprocessing import get_context, cpu_count

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
    print(f"[INFO] depth_vector: {np.size(depth_vector)}, {img.shape[0]}")
    extent = [0, 360, depth_vector[-1], depth_vector[0]]
    im = ax.imshow(img, cmap=cmap,aspect="auto", extent=extent)
    ### save contours ####
    all_contour_points = []
    for i, pts in enumerate(circles):
        x = pts[:, 0, 0]
        y = pts[:, 0, 1]
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        y = depth_vector[y.astype(int)]  # convert pixel row to depth
        print(f"[DEBUG] Contour {i}: x = {x}, y = {y}")

        if isinstance(legend, bool) and legend:
            legend_label = str(i)
        elif legend:
            legend_label = legend
        else:
            legend_label = None

        ax.plot(x, y, color=color, linewidth=linewidth, label=legend_label)
        for xi, yi in zip(x, y):
            all_contour_points.append({"contour_id": i, "x": xi, "depth_m": yi})
    ax.set_ylabel("Depth (m)")
    start_tick = int(np.ceil(depth_vector[0]))
    end_tick = int(np.floor(depth_vector[-1]))

    # Generate tick positions at every 1 meter
    tick_positions = np.arange(start_tick, end_tick + 1, 1)

    # Apply to axis
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([str(tick) for tick in tick_positions]) 
    #ax.set_yticks(np.linspace(depth_vector[0], depth_vector[-1], num=100))
    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)

    if axis_off:
        ax.axis('off')

    if colorbar:
        create_colorbar(ax, im, fontsize=labelsize)

    plt.tight_layout()

    # save_path = "/app/well_files/test.png"
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    # with open(save_path, 'wb') as f:
    #     f.write(buf.getvalue())

    plt.close(fig)
    return buf.getvalue(), all_contour_points




def process_single_threshold(args):
    (
        fmi_array, tdep_array,diff_thresh, block_size, c_threshold,
        depth_start, depth_end, radius_cm, pix_len_cm,
        min_vug_area, max_vug_area, min_circ_ratio, max_circ_ratio,
        fontsize, labelsize, colorbar
    ) = args
    print(f"[DEBUG] Entering process_single_threshold for threshold {diff_thresh}")
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
        print(f"[INFO] plotting contours now!")
        mode_subtracted = np.abs(fmi_array - diff_thresh)
        png,contour_csv = plot_single_image_contours(
            mode_subtracted, contours,
            cmap='YlOrBr',
            linewidth=2,
            colorbar=False,
            fontsize=fontsize,
            labelsize=labelsize,
            axis_off=False,
            title=f"Threshold {diff_thresh:.2f}",
            depth_vector=tdep_array
           
        )
        return png ,contour_csv

    except Exception as e:
        print(f"[ERROR] Failed at threshold {diff_thresh:.2f}: {e}")
        return None

def plot_fmi_with_area_circularity_filtered_contours_parallel(
    fmi_array_one_meter_zone,
    #fmi_array_unscaled_one_meter_zone,
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
            fmi_array_one_meter_zone, tdep_array_one_meter_zone,diff_thresh, block_size, c_threshold,
            one_meter_zone_start, one_meter_zone_end,
            hole_radius_cm, pixel_length_cm,
            min_vug_area, max_vug_area, min_circ_ratio, max_circ_ratio,
            fontsize, labelsize, colorbar
        )
        for diff_thresh in different_thresholds
    ]
    with get_context("spawn").Pool(processes=min(len(different_thresholds), cpu_count())) as pool:
    # with Pool(processes=min(len(different_thresholds), cpu_count())) as pool:
        results = pool.map(process_single_threshold, args_list)
    print(f"[INFO]: filter none results")
    # Filter out None results
    # Filter out None results first
    valid_results = [r for r in results if r is not None]
    png_outputs, contour_csv_outputs = zip(*valid_results)
    

    print(f"[INFO] Total PNG images generated: {len(png_outputs)}")
    return png_outputs,contour_csv_outputs


