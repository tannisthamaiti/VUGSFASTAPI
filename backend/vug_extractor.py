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
    #data_path_unscaled: str,
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
    tdep_array = depth_path
    well_radius = well_radius
    start = start_depth
    end = end_depth
    

    

    # Mask zone of interest
    outputs = get_one_meter_fmi_and_GT(
        start_depth, end_depth, tdep_array, fmi_array, well_radius, scale_individual_patch=True
    )
    
    fmi_array_zone, tdep_array_zone, well_radius_zone, mask = outputs
   
    # Thresholds
    different_thresholds = [np.median(fmi_array_zone)]  # simplified threshold for demo
    different_thresholds, count_of_different_thresholds = get_mode_of_interest_from_image(fmi_array_zone, 5, 5)
    print(f"[INFO] threshold: {different_thresholds}")

    

    # Plot
    if plot:
        fig_path,contour_csv=plot_fmi_with_area_circularity_filtered_contours_parallel(
    fmi_array_one_meter_zone=fmi_array_zone,
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
    
    # print(f"[DEBUG] Type of fig_list: {type(fig_path)}")
    # if not isinstance(fig_path, (list, tuple)):
    #     raise TypeError(f"[ERROR] fig_path is not a list or tuple, got: {type(fig_path)}")

    # for i, item in enumerate(fig_path):
    #     if not isinstance(item, str):
    #         raise TypeError(f"[ERROR] fig_path[{i}] is not a string (HTML), got: {type(item)}")
    #     if "<html" not in item.lower() and "plotly" not in item.lower():
    #         raise ValueError(f"[ERROR] fig_path[{i}] does not appear to be valid HTML content.")

    return fig_path, contour_csv
    
    

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
