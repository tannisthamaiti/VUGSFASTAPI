
import copy
import numpy as np
import pandas as pd
import cv2 as cv
import pickle
from os.path import join as pjoin
from IPython.display import clear_output


from utils.plotting import (
    plot_original, 
    plot_mode_subtracted_histogram, 
    plot_thresholded_fmi, 
    plot_fmi_with_original_contours, 
    plot_fmi_with_area_circularity_filtered_contours, 
    plot_combined_contour_on_fmi, 
    plot_contour_after_laplacian_filtering, 
    plot_contour_after_mean_filtering, 
    comparison_plot,
    plot_contours_from_different_thresholds,
    plot_heatmap
)
from utils.contours import (
    extract_contours_multi_thresholded_fmi_after_area_circularity_filtering,
    merge_and_filter_contours
)
from utils.filter import (
    filter_contours_based_on_original_image, 
    filter_contour_based_on_mean_pixel_in_and_around_original_contour,
    filter_overlapping_contours_based_on_iou,
    get_contrast_heatmap,
    get_mean_diff_heatmap
)
from utils.processing import get_mode_of_interest_from_image
from utils.pre_processing import get_one_meter_fmi_and_GT

def detected_percent_vugs(
        filtered_contour, fmi_array_one_meter_zone, tdep_array_one_meter_zone, 
        one_meter_zone_start, one_meter_zone_end
):
    """calculates the percentage of vugs detected in the one meter zone by the contours

    Parameters
    ----------

    filtered_contour : List
        Filtered contour is a list of contours. Each element of list is a dictionary of contours with 
        keys being 'id', 'area', 'depth', 'centroid_x', 'centroid_y'
    fmi_array_one_meter_zone : numpy array
        fmi array of one meter zone
    tdep_array_one_meter_zone : numpy array
        tdep array of one meter zone
    one_meter_zone_start : int
        start of the one meter zone
    one_meter_zone_end : int
        end of the one meter zone

    Returns
    -------
    df : DataFrame
        dataframe of vugs which contain two columns, depth and area
    """
    height, width = fmi_array_one_meter_zone.shape

    blank_im = np.zeros((height, width))
    blank_im[blank_im==0]=255

    cv.drawContours(blank_im, filtered_contour, -1, (0,0,0), thickness=cv.FILLED)
    df = pd.DataFrame()
    for start in np.arange(one_meter_zone_start, one_meter_zone_end, 0.1):
        end = start + 0.1
        mask = (tdep_array_one_meter_zone>=start) & (tdep_array_one_meter_zone<end)
        blank_im_point_one_meter_zone = blank_im[mask]
        white_px = cv.countNonZero(blank_im_point_one_meter_zone)
        total_px = blank_im_point_one_meter_zone.shape[0] * width
        df = pd.concat([df, pd.DataFrame({'Depth': [start], 'Vugs': [(1 - white_px/total_px)*100]})], ignore_index=True)
        # plt.imshow(blank_im_point_one_meter_zone, cmap='gray')
        # plt.title(f"{total_px}-{white_px}")
        # np.save(f"vugs_{start}.npy", blank_im_point_one_meter_zone)
        # plt.show()
    return df

def detect_vugs(
        one_meter_zone_start, one_meter_zone_end, tdep_array_doi, fmi_array_doi, fmi_array_doi_unscaled, well_radius_doi, gt, 
        save_path, pred_df, height_idx, contour_x, contour_y, vug_area_list, vugs_circulairty_list, radius_list, pixel_len_list, 
        stride_mode=5, k=5, block_size=21, c_threshold='mean', min_vug_area=1, max_vug_area=15, min_circ_ratio=0.3,max_circ_ratio=1, 
        centroid_threshold=5, laplacian_variance_threshold=0.2, mean_diff_thresh=0.1, merge_overlapping_contours=True,
        save=False, depth_in_name=False, picture_format='png', workflows_to_be_plotted=None, print_duplicates_info=False,
        custom_fname='', figsize=(10, 10), colorbar=True, labelsize=20, fontsize=20, plot=True, max_bar_scale=25,
        laplacian_heatmap_vmax=1, heatmap_cmap='plasma', overlapping_bbox_based_on_mean=True, background_laplacian_code=0,
        smoothen_contrast_heatmap=True, smoothen_mean_diff_heatmap=False, contrast_smoothening_sigma=2, mean_diff_smoothening_sigma=2,
        well_name='', mud_type=None, background_mean_code=0, mean_diff_heatmap_vmax=None, epsilon=0.01, mode_value_in_legend=False,
        scale_individual_patch=False
):

    '''
    Detect vugs in the given one meter zone
    '''
    ####FOR PROCESSING####
    #-------GET FMI, DEPTH, WELL RADIUS AND GROUND TRUTH FOR ONE METER ZONE-------#
    outputs = get_one_meter_fmi_and_GT(
        one_meter_zone_start, 
        one_meter_zone_end, 
        tdep_array_doi, 
        fmi_array_doi, 
        well_radius_doi, 
        gt,
        scale_individual_patch
    )
    (
        fmi_array_one_meter_zone, tdep_array_one_meter_zone, 
        well_radius_one_meter_zone, gtZone, mask_one_meter_zone
    ) = outputs
    fmi_array_one_meter_zone_unscaled = fmi_array_doi_unscaled[mask_one_meter_zone]

    ####FOR PLOTTING####
    #-------PLOT ORIGINAL IMAGE AND HISTOGRAM-------#
    if workflows_to_be_plotted['original_histogram']:
        plot_original(
            fmi_array_one_meter_zone, save_path=save_path, save=save, depth = one_meter_zone_start, 
            depth_in_name=depth_in_name, picture_format=picture_format, custom_fname=custom_fname,
            colorbar=colorbar, labelsize=labelsize, fontsize=fontsize, plot=plot, well_name=well_name
        )

    ####FOR PROCESSING####
    #-------GET MODE OF INTEREST FROM THE IMAGE-------#
    different_thresholds, count_of_mode_of_interest = get_mode_of_interest_from_image(
        fmi_array_one_meter_zone, stride_mode, k
    )


    ####FOR PLOTTING####
    #-------PLOT LOCAL VARIATION ENHANCED HISTOGRAM; THRESHOLDED IMAGE; EXTRACTED CONTOURS; CONOURS AFTER AREA CIRCULARITY FILTERING-------#
    if workflows_to_be_plotted['local_variation_enhanced_histogram']:
        plot_mode_subtracted_histogram(
            fmi_array_one_meter_zone, different_thresholds, save_path=save_path, save = save, 
            depth = one_meter_zone_start, depth_in_name=depth_in_name, picture_format=picture_format, 
            custom_fname=custom_fname, plot=plot, well_name=well_name, mode_value_in_legend=mode_value_in_legend
        )
    if workflows_to_be_plotted['thresholded_image']:
        plot_thresholded_fmi(
            fmi_array_one_meter_zone, fmi_array_one_meter_zone_unscaled, different_thresholds, block_size, c_threshold, save_path=save_path, 
            save = save, depth = one_meter_zone_start, depth_in_name=depth_in_name, picture_format=picture_format, 
            custom_fname=custom_fname, colorbar=colorbar, labelsize=labelsize, fontsize=fontsize, plot=plot, well_name=well_name
        )
    if workflows_to_be_plotted['extracted_contours']:
        plot_fmi_with_original_contours(
            fmi_array_one_meter_zone, fmi_array_one_meter_zone_unscaled, different_thresholds, block_size, c_threshold, save_path=save_path, 
            save=save, depth = one_meter_zone_start, depth_in_name=depth_in_name, picture_format=picture_format, 
            custom_fname=custom_fname, colorbar=colorbar, labelsize=labelsize, fontsize=fontsize, plot=plot, well_name=well_name
        )
    if workflows_to_be_plotted['area_circularity_filtering']:
        plot_fmi_with_area_circularity_filtered_contours(
            fmi_array_one_meter_zone, fmi_array_one_meter_zone_unscaled, different_thresholds, block_size, c_threshold, one_meter_zone_start, one_meter_zone_end, 
            well_radius_one_meter_zone, tdep_array_one_meter_zone, min_vug_area, max_vug_area, min_circ_ratio, 
            max_circ_ratio, save_path=save_path, save=save, depth = one_meter_zone_start, 
            depth_in_name=depth_in_name, picture_format=picture_format, custom_fname=custom_fname,
            colorbar=colorbar, labelsize=labelsize, fontsize=fontsize, plot=plot, well_name=well_name
        )

    ####FOR PROCESSING####
    #-------EXTRACT CONTOURS AND VUGS AFTER AREA CIRCULARITY FILTERING-------#
    outputs = extract_contours_multi_thresholded_fmi_after_area_circularity_filtering(
        fmi_array_one_meter_zone, tdep_array_one_meter_zone, well_radius_one_meter_zone, one_meter_zone_start, one_meter_zone_end, 
        different_thresholds, block_size, c_threshold, min_vug_area, max_vug_area, 
        min_circ_ratio, max_circ_ratio, centroid_threshold, print_duplicates_info = print_duplicates_info
    )
    final_combined_contour, final_combined_vugs, holeR, pixLen, contours_from_different_thresholds, C_value = outputs
    radius_list.append(holeR)
    pixel_len_list.append(pixLen)

    ####FOR PLOTTING####
    #-------PLOT CONTOURS FROM DIFFERENT THRESHOLDS-------#
    if workflows_to_be_plotted['unique_contours_from_different_thresholds']:
        plot_contours_from_different_thresholds(
            fmi_array_one_meter_zone_unscaled, contours_from_different_thresholds, save_path=save_path, save=save, 
            depth = one_meter_zone_start, depth_in_name=depth_in_name, picture_format=picture_format, custom_fname=custom_fname,
            colorbar=colorbar, labelsize=labelsize, fontsize=fontsize, plot=plot, well_name=well_name
        )

    ####FOR PROCESSING####
    #-------COMBINE CONTOURS FROM DIFFERENT THRESHOLDS-------#
    if merge_overlapping_contours:
        final_combined_contour, final_combined_vugs = merge_and_filter_contours(
            fmi_array_one_meter_zone, final_combined_contour, final_combined_vugs, 
            holeR, pixLen, one_meter_zone_start, one_meter_zone_end
        )
    else:
        # Remove contours that are inside other contours
        final_combined_contour, final_combined_vugs = filter_overlapping_contours_based_on_iou(
            final_combined_contour, final_combined_vugs, threshold=0.2
        )

    ####FOR PLOTTING####
    #-------PLOT COMBINED CONTOUR ON FMI-------#
    if workflows_to_be_plotted['combined_contours']:
        plot_combined_contour_on_fmi(
            fmi_array_one_meter_zone_unscaled, final_combined_contour, save_path=save_path, save=save, depth = one_meter_zone_start, 
            depth_in_name=depth_in_name, picture_format=picture_format, custom_fname=custom_fname,
            colorbar=colorbar, labelsize=labelsize, fontsize=fontsize, plot=plot, well_name=well_name
        )

    ####FOR PROCESSING####
    #-------FILTER CONTOURS BASED ON ORIGINAL IMAGE CONTRAST VALUE DERIVED FROM LAPLACIAN-------#
    filtered_contour, filtered_vugs = filter_contours_based_on_original_image(
        final_combined_contour, final_combined_vugs, fmi_array_one_meter_zone, laplacian_variance_threshold
    )
    laplacian_based_filtering = copy.deepcopy(filtered_contour)
    if workflows_to_be_plotted['contrast_heatmap']:
        contrast_heatmap, contrast_values_list = get_contrast_heatmap(
            fmi_array_one_meter_zone, final_combined_contour, 
            overlapping_bbox_based_on_mean=overlapping_bbox_based_on_mean, 
            background_laplacian_code=background_laplacian_code,
            smoothen_contrast_heatmap=smoothen_contrast_heatmap,
            sigma=contrast_smoothening_sigma
        )

    ####FOR PLOTTING####
    #-------PLOT CONTOUR AFTER LAPLACIAN FILTERING AND ITS HEATMAP-------#
    if workflows_to_be_plotted['contrast_heatmap']:
        plot_heatmap(
            fmi_array_one_meter_zone_unscaled, contrast_heatmap, final_combined_contour, laplacian_based_filtering, vmax=laplacian_heatmap_vmax, 
            depth=one_meter_zone_start, threshold=laplacian_variance_threshold, cmap=heatmap_cmap, fontsize=labelsize, 
            labelsize=labelsize, nrows=1, ncols=4, figsize=(25, 5), save=save, save_path=save_path, 
            custom_fname=custom_fname, picture_format=picture_format, depth_in_name=depth_in_name, well_name=well_name,
            laplacian_heatmap=True
        )
    if workflows_to_be_plotted['laplacian_filtering']:
        plot_contour_after_laplacian_filtering(
            fmi_array_one_meter_zone_unscaled, filtered_contour, save_path=save_path, save=save, 
            depth = one_meter_zone_start, depth_in_name=depth_in_name, picture_format=picture_format, 
            custom_fname=custom_fname, colorbar=colorbar, labelsize=labelsize, fontsize=fontsize, plot=plot, well_name=well_name
        )

    ####FOR PROCESSING####
    #-------FILTER THE CONTOURS BASED ON THE MEAN PIXEL IN AND AROUND THE ORIGINAL CONTOUR-------#
    filtered_contour, filtered_vugs = filter_contour_based_on_mean_pixel_in_and_around_original_contour(
        fmi_array_one_meter_zone, filtered_contour, filtered_vugs, 
        threshold = mean_diff_thresh, mud_type = mud_type
    )
    if workflows_to_be_plotted['mean_diff_heatmap']:
        mean_diff_heatmap, mean_diff_values_list = get_mean_diff_heatmap(
            fmi_array_one_meter_zone, laplacian_based_filtering, 
            overlapping_circle_based_on_mean=overlapping_bbox_based_on_mean, 
            background_mean_code=background_mean_code,
            smoothen_mean_diff_heatmap=smoothen_mean_diff_heatmap,
            sigma=mean_diff_smoothening_sigma
        )

    ####FOR PLOTTING####
    #-------PLOT CONTOUR AFTER MEAN FILTERING-------#
    if workflows_to_be_plotted['mean_diff_heatmap']:
        plot_heatmap(
            fmi_array_one_meter_zone_unscaled, mean_diff_heatmap, laplacian_based_filtering, filtered_contour, vmax=mean_diff_heatmap_vmax, 
            depth=one_meter_zone_start, threshold=mean_diff_thresh, cmap=heatmap_cmap, fontsize=labelsize, 
            labelsize=labelsize, nrows=1, ncols=4, figsize=(25, 5), save=save, save_path=save_path, 
            custom_fname=custom_fname, picture_format=picture_format, depth_in_name=depth_in_name, well_name=well_name,
            mud_type=mud_type, laplacian_heatmap=False, epsilon = epsilon
        )
    if workflows_to_be_plotted['mean_based_filtering']:
        plot_contour_after_mean_filtering(
            fmi_array_one_meter_zone_unscaled, filtered_contour, save_path=save_path, save=save, 
            depth = one_meter_zone_start, depth_in_name=depth_in_name, picture_format=picture_format, 
            custom_fname=custom_fname, colorbar=colorbar, labelsize=labelsize, fontsize=fontsize, plot=plot, well_name=well_name
        )


    ####FOR PROCESSING####
    #-------GET THE FINAL INFORMATION ABOUT THE VUGS DETECTED IN THE ZONE-------#
    contour_x, contour_y, vug_area_list, vugs_circulairty_list, height_idx = process_contours_and_vugs(
        filtered_contour, filtered_vugs, height_idx, contour_x, 
        contour_y, vug_area_list, vugs_circulairty_list
    )
    detected_vugs_percentage = detected_percent_vugs(
        filtered_contour, fmi_array_one_meter_zone, tdep_array_one_meter_zone, 
        one_meter_zone_start, one_meter_zone_end
    )
    pred_df = pd.concat([pred_df, detected_vugs_percentage], axis=0)
    pred_df_zone = pred_df[(pred_df.Depth>=one_meter_zone_start) & (pred_df.Depth<one_meter_zone_end)]

    ####FOR PLOTTING####
    #-------PLOT THE COMPARISON OF THE PREDICTED VUGS AND GROUND TRUTH-------#
    if workflows_to_be_plotted['final']:
        comparison_plot(
            fmi_array_one_meter_zone_unscaled, filtered_contour, pred_df_zone, gtZone, save_path=save_path, save=save, 
            depth = one_meter_zone_start, depth_in_name=depth_in_name, picture_format=picture_format, 
            custom_fname=custom_fname, colorbar=colorbar, labelsize=labelsize, fontsize=fontsize,
            plot=plot, max_bar_scale=max_bar_scale, well_name=well_name, depth_in_yaxis=False
        )

    return (
        pred_df, height_idx, contour_x, contour_y, vug_area_list, vugs_circulairty_list, radius_list, pixel_len_list,
        fmi_array_one_meter_zone, fmi_array_one_meter_zone_unscaled,
        pred_df_zone, gtZone, contours_from_different_thresholds, C_value
    )

def process_contours_and_vugs(
        filtered_contour, filtered_vugs, height_idx = 0, contour_x = [], contour_y = [], 
        vug_area_list = [], vugs_circulairty_list = []
):
    '''
    Function to process the contours and vugs for further use

    1. Get the contours and centroids from the filtered contours and save them in a list for further use
    2. These saved contours are not relative to 1m zone, but to the whole image as we are 
       using the height_idx to adjust the height of the contours
    3. Save the vug area and modified contours in a list for further use

    Parameters
    ----------
    filtered_contour: list
        List of filtered contours
    filtered_vugs: list
        List of filtered vugs
    height_idx: int
        Height index to adjust the height of the contours
    contour_x: list
        List to save the x coordinates of the contours
    contour_y: list
        List to save the y coordinates of the contours
    vug_area_list: list
        List to save the vug area
    vugs_circulairty_list: list
        List to save the vugs circularity

    Returns
    -------
    contour_x: list
        List of x coordinates of the contours
    contour_y: list
        List of y coordinates of the contours
    vug_area_list: list
        List of vug area
    vugs_circulairty_list: list
        List of vugs circularity
    height_idx: int
        Height index to adjust the height of the contours
    '''
    
    # get the contours and centroids from the filtered contours and save them in a list for further use
    # these saved contours are not relative to 1m zone, but to the whole image
    for pts, vugs_individual in zip(filtered_contour, filtered_vugs):
        x = pts[:, 0, 0]
        y = pts[:, 0, 1]
        x = np.append(x, x[0])
        y = np.append(y, y[0])

        y+=height_idx

        contour_x.append(x)
        contour_y.append(y)
        vug_area_list.append(vugs_individual['area'])
        vugs_circulairty_list.append(vugs_individual['circularity'])

    return contour_x, contour_y, vug_area_list, vugs_circulairty_list, height_idx

def convert_vugs_to_df(filtered_vugs):
    """converts the vugs to dataframe
    dataframe should consist two columns, depth and area

    Parameters
    ----------
    filtered_vugs : List
        Filtered vugs is a list of vugs. Each element of list is a dictionary of vugs with keys being 
        'id', 'area', 'depth', 'centroid_x', 'centroid_y'

    Returns
    -------
    df : DataFrame
        dataframe of vugs which contain two columns, depth and area
    """
    df = pd.DataFrame({'depth': [v['depth'] for v in filtered_vugs], 'area': [v['area'] for v in filtered_vugs]})
    df = df.sort_values(by=['depth'])
    return df

def convert_df_to_zone(df, start, end, zone_len):
    """converts the dataframe to zone. 
    
    Parameters
    ----------
    df : DataFrame
        df contains two columns, depth and area.
        depth is in meters and it's value can range anywhere between start and end.
    start : int
        actual start depth of the df, which is not necessarily present
        df can start from 2652.2 and end at 2652.4, but start can be 2652.0
    end : int
        actual end depth of the df, which is not necessarily present
        df can start from 2652.2 and end at 2652.4, but end can be 2653.0
    zone_len : int
        length of the zone, this is the zone in which df needs to be recreated
        if zone_len is 0.1 then df will be recreated in 0.1 meter depth intervals
        and area will be summed up for each 0.1 meter depth interval

    Returns
    -------
    zone : DataFrame
        dataframe of vugs which contain two columns, depth and area
    """
    zone_depth = np.arange(start, end, zone_len)
    zone = pd.DataFrame()
    for i in zone_depth:
        start, end = i, i+0.1
        area = df[(df.depth>=start) & (df.depth<end)]['area'].sum()
        zone = pd.concat([zone, pd.DataFrame({'depth': [start], 'area': [area]})], ignore_index=True)
    return zone

def fine_tune_vugs(
        one_meter_zone_start, one_meter_zone_end, tdep_array_doi, fmi_array_doi, fmi_array_doi_unscaled, well_radius_doi, gt, save_path, 
        pred_df, height_idx, contour_x, contour_y, vug_area_list, vugs_circulairty_list, radius_list, pixel_len_list, stride_mode, 
        k, block_size, c_threshold, min_vug_area, max_vug_area, min_circ_ratio, max_circ_ratio, centroid_threshold, 
        laplacian_variance_threshold, mean_diff_thresh, merge_overlapping_contours=True, save=False, depth_in_name=False, 
        picture_format='png', workflows_to_be_plotted=None, print_duplicates_info=False, plot=True, max_bar_scale=25,
        laplacian_heatmap_vmax=1, heatmap_cmap='plasma', overlapping_bbox_based_on_mean=True, background_laplacian_code=0,
        smoothen_contrast_heatmap=True, contrast_smoothening_sigma=2, well_name='', mud_type=None, mean_diff_heatmap_vmax=None, 
        epsilon=0.01, smoothen_mean_diff_heatmap=False, mean_diff_smoothening_sigma=2, background_mean_code=0,
        mode_value_in_legend=False, scale_individual_patch=False
):
    """
    Function to fine tune the vugs detection parameters
    """
    mean_diff_thresh_orig = copy.deepcopy(mean_diff_thresh)
    block_size_orig = copy.deepcopy(block_size)
    c_threshold_orig = copy.deepcopy(c_threshold)
    min_circ_ratio_orig = copy.deepcopy(min_circ_ratio)
    max_circ_ratio_orig = copy.deepcopy(max_circ_ratio)
    min_vug_area_orig = copy.deepcopy(min_vug_area)
    max_vug_area_orig = copy.deepcopy(max_vug_area)
    laplacian_variance_threshold_orig = copy.deepcopy(laplacian_variance_threshold)
    
    pred_df_orig = copy.deepcopy(pred_df)
    height_idx_orig = copy.deepcopy(height_idx)
    contour_x_orig = copy.deepcopy(contour_x)
    contour_y_orig = copy.deepcopy(contour_y)
    vug_area_list_orig = copy.deepcopy(vug_area_list)
    vugs_circulairty_list_orig = copy.deepcopy(vugs_circulairty_list)
    radius_list_orig = copy.deepcopy(radius_list)
    pixel_len_list_orig = copy.deepcopy(pixel_len_list)
    not_done = True

    while not_done:
        outputs = detect_vugs(
            one_meter_zone_start, one_meter_zone_end, tdep_array_doi, fmi_array_doi, fmi_array_doi_unscaled, well_radius_doi, gt, save_path, 
            pred_df, height_idx, contour_x, contour_y, vug_area_list, vugs_circulairty_list, radius_list, pixel_len_list, 
            stride_mode, k, block_size, c_threshold, min_vug_area, max_vug_area, min_circ_ratio, max_circ_ratio, centroid_threshold, laplacian_variance_threshold, 
            mean_diff_thresh, merge_overlapping_contours=merge_overlapping_contours, save=save, depth_in_name=depth_in_name, picture_format=picture_format, 
            workflows_to_be_plotted=workflows_to_be_plotted, print_duplicates_info=print_duplicates_info, plot=plot, max_bar_scale=max_bar_scale,
            laplacian_heatmap_vmax=laplacian_heatmap_vmax, heatmap_cmap=heatmap_cmap, overlapping_bbox_based_on_mean=overlapping_bbox_based_on_mean,
            background_laplacian_code=background_laplacian_code, smoothen_contrast_heatmap=smoothen_contrast_heatmap, 
            contrast_smoothening_sigma=contrast_smoothening_sigma, well_name=well_name, mud_type=mud_type, 
            mean_diff_heatmap_vmax=mean_diff_heatmap_vmax, epsilon=epsilon, smoothen_mean_diff_heatmap=smoothen_mean_diff_heatmap,
            mean_diff_smoothening_sigma=mean_diff_smoothening_sigma, background_mean_code=background_mean_code,
            mode_value_in_legend=mode_value_in_legend, scale_individual_patch=scale_individual_patch
        )
        (
            pred_df, height_idx, contour_x, contour_y, vug_area_list, vugs_circulairty_list, radius_list, pixel_len_list,
            fmi_array_one_meter_zone, fmi_array_one_meter_zone_unscaled,
            pred_df_zone, gtZone, contours_from_different_thresholds, C_value
        ) = outputs

        not_done = input(f"Do you want to re-iterate for {one_meter_zone_start}? (0/1): ")
        not_done = bool(int(not_done))
        if not_done:
            mean_diff_thresh = input(f"Enter the mean diff threshold (Default: {mean_diff_thresh_orig} & Previous: {mean_diff_thresh}): ")
            block_size = input(f"Enter the block size (Default: {block_size_orig} & Previous: {block_size}): ")
            c_threshold = input(f"Enter the c threshold (Default: {c_threshold_orig}/{C_value} & Previous: {c_threshold}): ")
            min_circ_ratio = input(f"Enter the min circularity ratio (Default: {min_circ_ratio_orig} & Previous: {min_circ_ratio}): ")
            max_circ_ratio = input(f"Enter the max circularity ratio (Default: {max_circ_ratio_orig} & Previous: {max_circ_ratio}): ")
            min_vug_area = input(f"Enter the min vug area (Default: {min_vug_area_orig} & Previous: {min_vug_area}): ")
            max_vug_area = input(f"Enter the max vug area (Default: {max_vug_area_orig} & Previous: {max_vug_area}): ")
            laplacian_variance_threshold = input(
                f"Enter the laplacian variance threshold (Default: {laplacian_variance_threshold_orig} & Previous: {laplacian_variance_threshold}): "
            )

            mean_diff_thresh = float(mean_diff_thresh) if mean_diff_thresh else mean_diff_thresh_orig
            block_size = int(block_size) if block_size else block_size_orig
            c_threshold = float(c_threshold) if c_threshold else c_threshold_orig
            min_circ_ratio = float(min_circ_ratio) if min_circ_ratio else min_circ_ratio_orig
            max_circ_ratio = float(max_circ_ratio) if max_circ_ratio else max_circ_ratio_orig
            min_vug_area = float(min_vug_area) if min_vug_area else min_vug_area_orig
            max_vug_area = float(max_vug_area) if max_vug_area else max_vug_area_orig
            laplacian_variance_threshold = float(laplacian_variance_threshold) if laplacian_variance_threshold else laplacian_variance_threshold_orig
            clear_output(wait=True)
            pred_df = copy.deepcopy(pred_df_orig)
            height_idx = copy.deepcopy(height_idx_orig)
            contour_x = copy.deepcopy(contour_x_orig)
            contour_y = copy.deepcopy(contour_y_orig)
            vug_area_list = copy.deepcopy(vug_area_list_orig)
            vugs_circulairty_list = copy.deepcopy(vugs_circulairty_list_orig)
            radius_list = copy.deepcopy(radius_list_orig)
            pixel_len_list = copy.deepcopy(pixel_len_list_orig)
    return (
        pred_df, height_idx, contour_x, contour_y, vug_area_list, vugs_circulairty_list, radius_list, pixel_len_list,
        fmi_array_one_meter_zone, fmi_array_one_meter_zone_unscaled,
        pred_df_zone, gtZone, contours_from_different_thresholds
    )

def get_extracted_vug_info(save_path, vug_info_fname):
    '''
    Extracts the vug information from the saved pickle file from part 1.
    '''

    with open(pjoin(save_path, vug_info_fname), 'rb') as file:
        loaded_data = pickle.load(file)

    height_idx = loaded_data['height_idx']
    contour_x = loaded_data['contour_x']
    contour_y = loaded_data['contour_y']
    vug_area_list = loaded_data['vug_area_list']
    vugs_circulairty_list = loaded_data['vugs_circulairty_list']
    pred_df = loaded_data['pred_df']
    save_path_folder = loaded_data['save_path']
    holeR = loaded_data['holeR']
    pixLen = loaded_data['pixLen']
    start = loaded_data['start']
    end = loaded_data['end']

    return (
        height_idx,
        contour_x,
        contour_y,
        vug_area_list,
        vugs_circulairty_list,
        pred_df,
        save_path_folder,
        holeR,
        pixLen,
        start,
        end
    )

def get_vugs_final_info(
        pred_df_zone, contour_x_list, contour_y_list, centroid_list, 
        vugs_area_list_filtered, vugs_circulairty_list_filtered, vug_info_zone_thicknes=0.1
):
    '''
    Get the final information about the vugs detected in the zone

    1. Get the depth of the zone
    2. create a dataframe with the depth as vugs centroid and vugs area as individual vugs area
    3. Sort the dataframe based on the depth
    4. Apply a for loop to get the total counts, mean area and std area of the vugs in the 0.1 meter zone
    
    Parameters
    ----------
    pred_df_zone : pandas DataFrame
        Predicted dataframe with 'Depth' column
    contour_x_list : list
        List of x coordinates of the vugs contours
    contour_y_list : list
        List of y coordinates of the vugs contours
    centroid_list : list
        List of vugs centroids
    vugs_area_list_filtered : list
        List of vugs area
    vugs_circulairty_list_filtered : list
        List of vugs circularity

    Returns
    -------
    vugs_info_final : pandas DataFrame
        Final information about the vugs detected in the zone
    '''
    # create a dataframe of the extracted vugs information
    vugs_info_df = pd.DataFrame({
        'Depth': centroid_list, 
        'VugsArea': vugs_area_list_filtered, 
        'VugsCircularity': vugs_circulairty_list_filtered,
        'ContourX': contour_x_list,
        'ContourY': contour_y_list
    })
    vugs_info_df = vugs_info_df.sort_values(by='Depth')

    (
        total_counts, vugs_mean_area, vugs_std_area, vugs_area, 
        all_vugs_area, all_vugs_circularity, all_vugs_contour_x, all_vugs_contour_y
    ) = [], [], [], [], [], [], [], []

    # get the vugs information in the 0.1 meter or the specified thickness zone
    reference_depth = pred_df_zone.Depth
    for depth_ in reference_depth:
        start_zone, end_zone = depth_, depth_+vug_info_zone_thicknes
        vugs_info_zone = vugs_info_df[((vugs_info_df.Depth>=start_zone) & (vugs_info_df.Depth<end_zone))]

        total_counts.append(vugs_info_zone.shape[0])
        vugs_area.append(vugs_info_zone.VugsArea.sum())
        vugs_mean_area.append(vugs_info_zone.VugsArea.mean())
        vugs_std_area.append(vugs_info_zone.VugsArea.std())
        all_vugs_area.append(vugs_info_zone.VugsArea.values)
        all_vugs_circularity.append(vugs_info_zone.VugsCircularity.values)
        all_vugs_contour_x.append(vugs_info_zone.ContourX.values)
        all_vugs_contour_y.append(vugs_info_zone.ContourY.values)

    vugs_info_final = pd.DataFrame(
        {
            'Depth': reference_depth,
            'TotalCounts': total_counts,
            'MeanArea': vugs_mean_area,
            'StdArea': vugs_std_area,
            'VugsArea': vugs_area,
            'AllVugsArea': all_vugs_area,
            'AllVugsCircularity': all_vugs_circularity,
            'AllVugsContourX': all_vugs_contour_x,
            'AllVugsContourY': all_vugs_contour_y
        }
    )
    vugs_info_final = vugs_info_final.fillna(0)
    return vugs_info_final

def get_final_vugs_for_zone(
        fmi_array_doi, fmi_array_doi_unscaled, tdep_array_doi, gt, pred_df, 
        contour_x, contour_y, vug_area_list, vugs_circulairty_list, 
        zone_start, zone_end, img_idx
):
    from utils.contours import get_centeroid
    from utils.vugs import get_vugs_final_info
    from sklearn.preprocessing import MinMaxScaler
    '''
    Get the final vugs information and plot the final vugs information for the zone

    1. Get the data (fmi, pred_df, gt) for the zone
    2. Get the image height
    3. Create a scaler to scale the centroid_y to the zone_start and zone_end range
    4. Filter the contours based on the height of the image and get the centroids of the vugs
    5. Get the vugs information in the zone
    6. Get the final information about the vugs detected in the zone

    Parameters
    ----------
    fmi_array_doi : numpy array
        FMI array of depth of interest
    fmi_array_doi_unscaled : numpy array
        FMI array of depth of interest unscaled
    tdep_array_doi : numpy array
        TDEP array of depth of interest
    fmi_array : numpy array
        FMI array
    tdep_array : numpy array
        TDEP array
    gt : pandas DataFrame
        Ground truth dataframe
    pred_df : pandas DataFrame
        Predicted dataframe
    contour_x : list
        List of x coordinates of the contours
    contour_y : list
        List of y coordinates of the contours
    vug_area_list : list
        List of vugs area
    vugs_circulairty_list : list
        List of vugs circularity
    zone_start : float
        Zone start
    zone_end : float
        Zone end
    img_idx : int
        Image index

    Returns
    -------
    img_idx : int
        Image index
    vugs_info_final : pandas DataFrame
        Final vugs information for the zone
    fmi_zone : numpy array
        FMI array for the zone
    fmi_zone_unscaled : numpy array
        FMI array for the zone unscaled
    pred_df_zone : pandas DataFrame
        Predicted dataframe for the zone
    gt_zone : pandas DataFrame
        Ground truth dataframe for the zone
    contour_x_list : list
        List of x coordinates of the contours
    contour_y_list : list
        List of y coordinates of the contours
    '''

    # get the data (fmi, pred_df, gt) for the zone
    temp_mask = (tdep_array_doi>=zone_start) & (tdep_array_doi<=zone_end)
    fmi_zone = fmi_array_doi[temp_mask]
    fmi_zone_unscaled = fmi_array_doi_unscaled[temp_mask]
    pred_df_zone = pred_df[(pred_df.Depth>=zone_start) & (pred_df.Depth<zone_end)]
    gt_zone = gt[(gt.Depth>=zone_start) & (gt.Depth<zone_end)]

    # get the image height
    img_height = fmi_zone.shape[0]

    # create a scaler to scale the centroid_y to the zone_start and zone_end range
    scaler = MinMaxScaler((zone_start, zone_end))
    scaler.fit([[0], [img_height]])

    # filter the contours based on the height of the image and get the centroids of the vugs
    # This is checking if the contour is within the img_idx and img_idx+img_height
    # Basically, earlier we converted the contour to the whole FMI data level, 
    # now we are extracting the contours that are within the zone
    coord = [
        [k, j] for i, (k, j) in enumerate(zip(contour_x, contour_y)) 
            if (j.min()>=img_idx) & (j.max()<=(img_idx+img_height))
    ]

    centroid_list = []
    vugs_area_list_filtered = []
    vugs_circulairty_list_filtered = []
    contour_x_list, contour_y_list = [], []
    for (x_, y_), vugs_area_individual, vugs_circularity_individual in zip(
        coord, vug_area_list, vugs_circulairty_list
    ):
        # get the y-coordinate of the centroid of the contour
        y_new = y_-img_idx
        centroid_y = get_centeroid(
            np.concatenate([x_.reshape(-1, 1), (y_new).reshape(-1, 1)], axis=1)
        )[1]
        # rescale the centroid_y to the zone_start and zone_end range
        centroid_depth = scaler.transform([[centroid_y]])[0][0]
        # get the index where the centroid_depth should be inserted
        depth_values, target_value = pred_df_zone.Depth.values, centroid_depth
        # Find the index where the target_value should be inserted
        insert_index = np.searchsorted(
            depth_values, target_value, side='right'
        ) - 1

        # Check if the target_value is greater than the last depth value, in that case, it will be inserted at the end
        if insert_index == len(depth_values) - 1 and target_value > depth_values[-1]:
            insert_index = len(depth_values) - 1

        if pred_df_zone.iloc[insert_index].Vugs != 0:
            centroid_list.append(target_value)
            vugs_area_list_filtered.append(vugs_area_individual)
            vugs_circulairty_list_filtered.append(vugs_circularity_individual)
            contour_x_list.append(x_)
            contour_y_list.append(y_new)

    vugs_info_final = get_vugs_final_info(
        pred_df_zone, contour_x_list, contour_y_list, centroid_list, 
        vugs_area_list_filtered, vugs_circulairty_list_filtered
        )

    img_idx+=img_height

    return (
        img_idx, vugs_info_final, fmi_zone, fmi_zone_unscaled, 
        pred_df_zone, gt_zone, contour_x_list, contour_y_list,
        vugs_area_list_filtered, vugs_circulairty_list_filtered
    )

def get_vugs_azimuthal_distribution(vugs_info_final, azimuthal_divisions=4):
    """
    Get the azimuthal distribution of vugs in the image.
    It divides the expected vuugs into azimuthal_divisions number of zones and counts the number of vugs in each zone.
    It basically counts the number of vugs in each azimuthal zone for each vug.

    Parameters
    ----------
    vugs_info_final : DataFrame
        The final DataFrame containing the information about the vugs.
    azimuthal_divisions : int, optional
        The number of azimuthal divisions to divide the vugs into. The default is 4.

    Returns
    -------
    azimuthal_counts_zone : numpy array
        The number of vugs in each azimuthal zone.
    """
    from utils.contours import get_centeroid
    azimuth_division_points = np.linspace(0, 360, azimuthal_divisions + 1)

    azimuthal_counts_list = []

    for i in range(vugs_info_final.shape[0]):

        vugs_conours_x_zone = vugs_info_final['AllVugsContourX'][i]
        vugs_conours_y_zone = vugs_info_final['AllVugsContourY'][i]

        vugs_azimuth_list = []
        for contour_x_, contour_y_ in zip(vugs_conours_x_zone, vugs_conours_y_zone):
            contours_zone = np.concatenate([contour_x_.reshape(-1, 1), (contour_y_).reshape(-1, 1)], axis=1)
            centroid_zone = get_centeroid(contours_zone)
            centroid_zone_x = centroid_zone[0]
            vugs_azimuth_list.append(centroid_zone_x)

        inds = np.digitize(vugs_azimuth_list, azimuth_division_points, right=False) - 1
        azimuthal_counts = np.bincount(inds, minlength=azimuthal_divisions)

        azimuthal_counts_list.append(azimuthal_counts)

    azimuthal_counts_zone = np.asarray(azimuthal_counts_list).sum(axis=0)

    return azimuthal_counts_zone

def get_vugs_spectrum_data(vugs_info_final, i, azimuth_division_points, azimuthal_divisions=4):
    '''
    This function is used to get the vugs spectrum data for each zone.
    It gives the vugs area, vugs circularity and azimuthal counts for each azimuthal division.

    Parameters:
    vugs_info_final (DataFrame): The final vugs information DataFrame.
    i (int): The index of the zone.
    azimuth_division_points (list): The azimuth division points.
    azimuthal_divisions (int): The number of azimuthal divisions.

    Returns:
    vugs_area_zone (float): The total vugs area in the zone.
    vugs_circularity_zone (float): The average vugs circularity in the zone.
    azimuthal_counts (array): The azimuthal counts for each azimuth
    '''
    from utils.contours import get_centeroid
    
    vugs_area_zone = vugs_info_final['AllVugsArea'][i]
    vugs_circularity_zone = vugs_info_final['AllVugsCircularity'][i]
    vugs_conours_x_zone = vugs_info_final['AllVugsContourX'][i]
    vugs_conours_y_zone = vugs_info_final['AllVugsContourY'][i]

    vugs_azimuth_list = []
    for contour_x_, contour_y_ in zip(vugs_conours_x_zone, vugs_conours_y_zone):
        contours_zone = np.concatenate([contour_x_.reshape(-1, 1), (contour_y_).reshape(-1, 1)], axis=1)
        centroid_zone = get_centeroid(contours_zone)
        centroid_zone_x = centroid_zone[0]
        vugs_azimuth_list.append(centroid_zone_x)

    inds = np.digitize(vugs_azimuth_list, azimuth_division_points, right=False) - 1
    azimuthal_counts = np.bincount(inds, minlength=azimuthal_divisions)

    return vugs_area_zone, vugs_circularity_zone, azimuthal_counts