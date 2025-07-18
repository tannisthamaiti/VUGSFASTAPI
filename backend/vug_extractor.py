import numpy as np
from utils.data import get_data
from utils.pre_processing import preprocessing, get_one_meter_fmi_and_GT
from utils.vugs import extract_contours_multi_thresholded_fmi_after_area_circularity_filtering
from utils.parallel_processing import plot_fmi_with_area_circularity_filtered_contours_parallel
from utils.processing import get_mode_of_interest_from_image
from os.path import join as pjoin
import os
import matplotlib.pyplot as plt

def extract_and_plot_contours(
    data_path: str,
    data_path_unscaled: str,
    depth_path: str,
    well_radius: str,
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
    

    fmi_array=data_path
    fmi_array_unscaled=data_path_unscaled
    tdep_array = depth_path
    well_radius = well_radius
    start = start_depth
    end = end_depth
    

    
    # Preprocess
    #fmi_array_doi_unscaled, fmi_array_doi, tdep_array_doi, well_radius_doi = preprocessing(
    #    fmi_array, tdep_array, well_radius, start, end, scale_individual_patch=True
    #)

    # Mask zone of interest
    outputs = get_one_meter_fmi_and_GT(
        start_depth, end_depth, tdep_array, fmi_array, well_radius, scale_individual_patch=True
    )
    
    fmi_array_zone, tdep_array_zone, well_radius_zone, mask = outputs
    #fmi_array_unscaled = fmi_array_doi_unscaled[mask]

    # Thresholds
    different_thresholds = [np.median(fmi_array_zone)]  # simplified threshold for demo
    different_thresholds, count_of_different_thresholds = get_mode_of_interest_from_image(fmi_array_zone, 5, 5)
    print(f"[INFO] threshold: {different_thresholds}")

    # # Extract vug contours
    # outputs = extract_contours_multi_thresholded_fmi_after_area_circularity_filtering(
    #     fmi_array_zone, tdep_array_zone, well_radius_zone, start_depth, end_depth,
    #     different_thresholds, block_size, c_threshold,
    #     min_vug_area, max_vug_area, min_circ_ratio, max_circ_ratio,
    #     centroid_threshold, print_duplicates_info=False
    # )

    # final_contours, final_vugs, _, _, _, _ = outputs
    # save_path="."
    # # Save path
    # os.makedirs(save_path, exist_ok=True)
    # fig_path = pjoin(save_path, f"vugs_{start_depth:.1f}_{end_depth:.1f}.png")

    # Plot
    if plot:
        fig_path=plot_fmi_with_area_circularity_filtered_contours_parallel(
    fmi_array_one_meter_zone=fmi_array_zone,
    fmi_array_unscaled_one_meter_zone=fmi_array_unscaled,
    different_thresholds=different_thresholds,
    block_size=block_size,
    c_threshold=c_threshold,
    one_meter_zone_start=start_depth,
    one_meter_zone_end=end_depth,
    well_radius_one_meter_zone=well_radius_zone,
    tdep_array_one_meter_zone=tdep_array_zone,
    min_vug_area=min_vug_area,
    max_vug_area=max_vug_area,
    min_circ_ratio=min_circ_ratio,
    max_circ_ratio=max_circ_ratio,
    depth=start_depth,
    depth_in_name=False,
    picture_format="png",
    custom_fname="",
    colorbar=True,
    labelsize=10,
    fontsize=10,
    figsize=(6, 6),
    well_name="")
    
    
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
    return (fmi_array_doi,fmi_array_doi_unscaled,tdep_array_doi, well_radius_doi)
