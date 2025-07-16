import numpy as np
from utils.data import get_data
from utils.pre_processing import preprocessing, get_one_meter_fmi_and_GT
from utils.vugs import extract_contours_multi_thresholded_fmi_after_area_circularity_filtering
from utils.plotting import plot_fmi_with_area_circularity_filtered_contours
from os.path import join as pjoin
import os
import matplotlib.pyplot as plt

def extract_and_plot_contours(
    data_path: str,
    depth_path: str,
    start_depth: float,
    end_depth: float,
    block_size: int = 21,
    c_threshold='mean',
    min_vug_area=1,
    max_vug_area=100,
    min_circ_ratio=0.5,
    max_circ_ratio=1.0,
    centroid_threshold=5,
    plot=True
):
    # Load full data
    #fmi_array, tdep_array, well_radius, gt = get_data(data_path)
    #start, end = int(np.ceil(gt.Depth.min())), int(gt.Depth.max())

    fmi_array=data_path
    tdep_array = depth_path
    well_radius = np.full_like(tdep_array, 8.0)
    start = start_depth
    end = end_depth
    

    
    # Preprocess
    fmi_array_doi_unscaled, fmi_array_doi, tdep_array_doi, well_radius_doi = preprocessing(
        fmi_array, tdep_array, well_radius, start, end, scale_individual_patch=True
    )

    # Mask zone of interest
    outputs = get_one_meter_fmi_and_GT(
        start_depth, end_depth, tdep_array_doi, fmi_array_doi, well_radius_doi, scale_individual_patch=True
    )
    
    fmi_array_zone, tdep_array_zone, well_radius_zone, mask = outputs
    fmi_array_unscaled = fmi_array_doi_unscaled[mask]

    # Thresholds
    different_thresholds = [np.median(fmi_array_zone)]  # simplified threshold for demo

    # Extract vug contours
    outputs = extract_contours_multi_thresholded_fmi_after_area_circularity_filtering(
        fmi_array_zone, tdep_array_zone, well_radius_zone, start_depth, end_depth,
        different_thresholds, block_size, c_threshold,
        min_vug_area, max_vug_area, min_circ_ratio, max_circ_ratio,
        centroid_threshold, print_duplicates_info=False
    )

    final_contours, final_vugs, _, _, _, _ = outputs
    save_path="."
    # Save path
    os.makedirs(save_path, exist_ok=True)
    fig_path = pjoin(save_path, f"vugs_{start_depth:.1f}_{end_depth:.1f}.png")

    # Plot
    if plot:
        fig_path=plot_fmi_with_area_circularity_filtered_contours(
            fmi_array_zone, fmi_array_unscaled, different_thresholds, block_size, c_threshold,
            start_depth, end_depth, well_radius_zone, tdep_array_zone,
            min_vug_area, max_vug_area, min_circ_ratio, max_circ_ratio,
            save_path=save_path, save=True, depth=start_depth, picture_format="png"
        )
    print(fig_path)
    return fig_path

def plotfmi(data_path: str,
    depth_path: str,
    well_radius:str,
    start_depth: float,
    end_depth: float,):
    fmi_array=data_path
    tdep_array = depth_path
    well_radius = well_radius
    start = start_depth
    end = end_depth
    fmi_array_doi_unscaled, fmi_array_doi, tdep_array_doi, well_radius_doi = preprocessing(
        fmi_array, tdep_array, well_radius, start, end, scale_individual_patch=True
    )
    return (fmi_array_doi,tdep_array_doi)
