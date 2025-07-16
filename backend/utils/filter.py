import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter
from scipy.stats import skew, kurtosis

def apply_vicinity_filtering(pred_df, vicinity_threshold=1, vugs_threshold=1):
    '''
    Apply vicinity filtering to the predicted vugs values.
    If the vugs value at the center of the vicinity is less than or equal to the vugs_threshold,
    and all the vugs values in the vicinity are less than or equal to the vugs_threshold,
    then set the vugs value at the center of the vicinity to 0.

    1. Get the vicinity of the current vugs value.
    2. If the vugs value at the center of the vicinity is less than or equal to the vugs_threshold, 
       and all the vugs values in the vicinity are less than or equal to the vugs_threshold, then set the vugs 
       value at the center of the vicinity to 0.
    3. Repeat the above steps for all the vugs values in the predicted dataframe.

    Parameters
    ----------
    pred_df : pandas.DataFrame
        Predicted dataframe with 'Vugs' column.
    vicinity_threshold : int, optional
        The vicinity threshold, by default 1.
    vugs_threshold : int, optional
        The vugs threshold, by default 1.

    Returns
    -------
    pandas.DataFrame
        Predicted dataframe with 'Vugs' column after applying vicinity filtering.
    '''
    num_rows = (vicinity_threshold*2)+1
    pred_df = pred_df.reset_index(drop=True)
    for i in range(len(pred_df)-vicinity_threshold):
        pred_df_zone = pred_df.iloc[i:i+num_rows]
        idx_voi = list(pred_df_zone.index)[1]
        if pred_df.iloc[idx_voi].Vugs<=vugs_threshold:
            if pred_df_zone.Vugs.max()<=vugs_threshold:
                pred_df.loc[idx_voi, 'Vugs'] = 0
    return pred_df

def filter_contours_based_on_original_image(contours, vugs, original_image, thresold, expansion_factor=0.1):
    from utils.misc import get_expanded_bbox_dim

    """"Filters the contours based on the original image
        For every contour, it checks the contrast of the original image, 
        if the contrast is high, it is a valid contour
        else it is a false positive and removed from the list of contours
        and icrease the bounding box by 10% in all direction

    Parameters
    ----------
    contours : list
        list of contours, each element of contours is a numpy array of shape (N, 1, 2) which represents a cotour of polygon, 
            it is not a rectangle
    vugs : list
        list of vugs, each element of vugs is a dictionary with keys 'id', 'area', 'depth', 'centroid_x', 'centroid_y'
    original_image : numpy array
        original image
    thresold : int
        thresold value for the contrast

    Returns
    -------
    filtered_contours : list
        list of contours, each element of contours is a numpy array of shape (N, 1, 2) which represents a cotour of polygon
    filtered_vugs : list
        list of vugs, each element of vugs is a dictionary with keys 'id', 'area', 'depth', 'centroid_x', 'centroid_y'
    """
    filtered_contours = []
    filtered_vugs = []

    # for every contour, get the bounding box and check the contrast of the bounding box
    for contour, vug in zip(contours, vugs):
        x, y, w, h = get_expanded_bbox_dim(contour, original_image.shape, expansion_factor=expansion_factor)
        roi = original_image[y:y+h, x:x+w]
        contrast = np.nanvar(cv.Laplacian(roi, cv.CV_32F))
        if contrast > thresold:
            filtered_contours.append(contour)
            filtered_vugs.append(vug)
    return filtered_contours, filtered_vugs

def get_elements_from_circular_roi_based_on_center_and_radius(fmi, center, radius):
    """
    Get elements of fmi image from circular ROI based on center and radius without nans

    Parameters
    ----------
    fmi : np.ndarray
        FMI data
    center : tuple
        Center of the circle
    radius : int
        Radius of the circle

    Returns
    -------
    roi : np.ndarray
        Elements from circular ROI
    """
    
    h, w = fmi.shape

    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Calculate the Euclidean distance
    distances = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # Create a mask
    mask = distances <= radius

    # Apply the mask to the image
    roi = np.zeros_like(fmi)
    roi[mask] = fmi[mask]

    # drop 0s from the numpy array
    roi = roi[roi!=0]

    # drop nans from the numpy array
    roi = roi[~np.isnan(roi)]

    return roi

def filter_contour_based_on_mean_pixel_in_and_around_original_contour(fmi, contours, vugs, threshold = 1, mud_type = None):
    """
    Filter contours based on the mean pixel in and around the original contour

    Parameters
    ----------
    fmi : np.ndarray
        FMI data
    contours : list
        List of contours
    vugs : list
        List of vugs
    threshold : int, optional
        Threshold value, by default 1

    Returns
    -------
    filtered_contour : list
        List of filtered contours
    filtered_vugs : list
        List of filtered vugs
    """
    
    filtered_contour = []
    filtered_vugs = []

    # iterate over each contour
    for contour_test, vugs_test in zip(contours, vugs):

        # get the center and radius of the contour
        (x,y),radius = cv.minEnclosingCircle(contour_test)
        center = (int(x),int(y))
        radius = int(radius)

        # Create a binary mask based on the contour coordinates
        mask = np.zeros_like(fmi, dtype=np.uint8)
        contour_points = np.array(contour_test)

        # fill the contour with white color
        mask = cv.fillPoly(mask, [contour_points], 255)

        # get the pixels from the contour
        contour_pixels = fmi[mask > 0]

        # drop nans from the numpy array
        contour_pixels = contour_pixels[~np.isnan(contour_pixels)]

        # get the pixels from the bounding padded circle
        bounding_padded_circle = get_elements_from_circular_roi_based_on_center_and_radius(fmi, center, radius*2)

        # get the mean of the pixels from the contour and bounding padded circle
        contour_mean = contour_pixels.mean()
        bounding_padded_circle_mean = bounding_padded_circle.mean()

        # if the difference between the mean of the pixels from the contour and 
        # bounding padded circle is greater than 1 then keep the contour
        difference = bounding_padded_circle_mean - contour_mean
        if mud_type == 'water':
            # difference <= -threshold; since in water mud based vugs are darker (YlOrBr), meaning they have larger pixel values
            # and the surrounding area is lighter, so when we subtract the mean of the surrounding area (which should be lighter, lower) 
            # from the mean of the contour (which should be darker, higher), the difference should be negative
            if difference <= 0 and abs(difference) >= threshold:
                filtered_contour.append(contour_test)
                filtered_vugs.append(vugs_test)
            else:
                pass
        elif mud_type == 'oil':
            # difference >= threshold; since in oil mud based vugs are lighter (YlOrBr), meaning they have lower pixel values
            # and the surrounding area is darker, so when we subtract the mean of the surrounding area (which should be darker, higher)
            # from the mean of the contour (which should be lighter, lower), the difference should be positive
            if difference >= 0 and abs(difference) >= threshold:
                filtered_contour.append(contour_test)
                filtered_vugs.append(vugs_test)
            else:
                pass
        else:
            raise ValueError("Invalid mud type. Valid mud types are 'water' and 'oil'.")
        
    return filtered_contour, filtered_vugs

def filter_contours_based_on_statistics_of_image(
        contours, vugs, std, var, skew, kurt, mins, maxs, means, std_thes=3.0, var_thes=5.0, skew_thes=8.0, 
        kurt_thes=70.0, skew_thes_low=0.0, kurt_thes_low=0.0,min_high = 49, 
        max_low = 75, means_low = 60,
):
    """
    Filter contours based on statistics of image, 
    if the statistics of image doesn't satisfy the codition then contour is removed
    if std is greater than 3, then contour is removed
    if var is greater than 5, then contour is removed
    if skew is negative or greater than 8, then contour is removed
    if kurt is greater than 70, then contour is removed

    Parameters
    ----------
    contours: list of contours
    vugs: list of vugs
    std: standard deviation of image
    var: variance of image
    skew: skewness of image
    kurt: kurtosis of image

    Returns
    -------
        filtered_contours: list of filtered contours
        filtered_vugs: list of filtered vugs
    """
    if std >= std_thes:
        return [], []
    if var >= var_thes:
        return [], []
    if mins<=min_high:
        return [], []
    if maxs>=max_low:
        return [], []
    if means>=means_low:
        return [], []
    if skew <= skew_thes_low or skew >= skew_thes:
        return [], []
    if kurt <= kurt_thes_low or kurt >= kurt_thes:
        return [], []
    return contours, vugs

def calculate_statistical_features(img):
    """calculate the statistical features of the image."""

    img = img.reshape(-1)
    img = img[~np.isnan(img)]
    mean = np.mean(img)
    std = np.std(img)
    var = np.var(img)
    skewness = skew(img.reshape(-1))
    kurtosis_val = kurtosis(img.reshape(-1))
    return img.min(), img.max(), np.float16(mean), np.float16(std), np.float16(var), np.float16(skewness), np.float16(kurtosis_val)

def remove_contour_from_bedding_planes(
        fmi, tdep, zone_start, zone_end, thresh, filtered_contour, 
        filtered_vugs, zoi = 0.05, thresh_percent = 15
):
    from utils.misc import check
    """
    Remove Vugs from bedding planes

    Parameters
    ----------
    fmi : np.ndarray
        FMI data
    tdep : np.ndarray
        TDEP data
    zone_start : float
        Starting depth of the zone
    zone_end : float
        Ending depth of the zone
    thresh : float
        Threshold value that is selected based on first mode
    filtered_contour : list
        List of contours
    filtered_vugs : list
        List of vugs
    zoi : float, optional
        Zone of interest length, by default 0.05

    Returns
    -------
    None
    """
    total_starts = []
    start_ = 0
    end_ = 0
    total_percent = []
    drop_idx = []
    for i, zoi_start in enumerate(np.arange(zone_start, zone_end, zoi)):
        zoi_end = zoi_start + zoi
        mask_zoi = (tdep>=zoi_start) & (tdep<zoi_end)
        fmi_zoi = fmi[mask_zoi]
        tdep_zoi = tdep[mask_zoi]
        start_=end_
        end_+=tdep_zoi.shape[0]
        numerator_mask = np.logical_and(fmi_zoi>=thresh[0]-0.1, fmi_zoi<=thresh[0]+0.1)
        percent = (numerator_mask.sum()/(fmi_zoi.reshape(-1).shape[0]))*100
        total_percent.append(percent)
        total_starts.append(zoi_start)

        filtering_required = True if percent<=thresh_percent else False
        if filtering_required:
            for i, (cnt, vgs) in enumerate(zip(filtered_contour, filtered_vugs)):
                cnt_y = cnt[:, 0, :][:, 1]
                cnt_y_range = range(cnt_y.min(), cnt_y.max())
                if sum([check(i, cnt_y_range) for i in range(start_-1, end_+1)]) != 0:
                    second_condition = filter_contour_based_on_bedding_plane(cnt, tdep, fmi, k = 2, zoi = 0.1, low = -1, high = 1)
                    if second_condition:
                        drop_idx.append(i)

    filtered_contour = [j for i, j in enumerate(filtered_contour) if i not in drop_idx]
    filtered_vugs = [j for i, j in enumerate(filtered_vugs) if i not in drop_idx]

    return total_percent, total_starts, filtered_contour, filtered_vugs

def filter_contour_based_on_bedding_plane(
        cnt, tdep_array_one_meter_zone, fmi_array_one_meter_zone, 
        k = 2, zoi = 0.1, low = -0.1, high = 0.1
):
    filtering_required = False
    cnt_x = cnt[:, 0, :][:, 0]
    cnt_y = cnt[:, 0, :][:, 1]
    cnt_y_min, cnt_y_max = cnt_y.min(), cnt_y.max()
    cnt_x_min, cnt_x_max = cnt_x.min(), cnt_x.max()
    cnt_dept = tdep_array_one_meter_zone[cnt_y_min:cnt_y_max]
    window_size = int(np.ceil(zoi/(cnt_dept[1:] - cnt_dept[:-1]).mean()))
    cnt_heigt = cnt_y_max - cnt_y_min
    cnt_width = cnt_x_max - cnt_x_min

    padding_required_height = window_size - cnt_heigt
    padding_required_width = window_size - cnt_width
    pad_top_bottom = int(np.ceil(padding_required_height/2))
    pad_left_right = int(np.ceil(padding_required_width/2))
    cnt_y_min = cnt_y_min - pad_top_bottom if cnt_y_min - pad_top_bottom > 0 else 0
    cnt_y_max = cnt_y_max + pad_top_bottom if cnt_y_max + pad_top_bottom < tdep_array_one_meter_zone.shape[0] else tdep_array_one_meter_zone.shape[0]
    cnt_x_min = cnt_x_min - pad_left_right if cnt_x_min - pad_left_right > 0 else 0
    cnt_x_max = cnt_x_max + pad_left_right if cnt_x_max + pad_left_right < 360 else 360
    fmi_window = fmi_array_one_meter_zone[cnt_y_min:cnt_y_max, :]
    fmi_contour = fmi_window[:, cnt_x_min:cnt_x_max]

    cnt_x_left_max = cnt_x_min
    cnt_x_left_min = cnt_x_min - fmi_contour.shape[1]*k if cnt_x_min - fmi_contour.shape[1]*k > 0 else 0

    cnt_x_right_min = cnt_x_max
    cnt_x_right_max = cnt_x_max + fmi_contour.shape[1]*k if cnt_x_max + fmi_contour.shape[1]*k < 360 else 360

    left, right = cnt_x_left_max - cnt_x_left_min, cnt_x_right_max - cnt_x_right_min
    optmial_pad = min(left, right)

    if cnt_x_min > 20 and cnt_x_max < 340:
        if left>right:
            fmi_contour_right = fmi_window[:, cnt_x_right_min:cnt_x_right_max]
            fmi_contour_left = fmi_window[:, cnt_x_left_max-optmial_pad:cnt_x_left_max]
        else:
            fmi_contour_left = fmi_window[:, cnt_x_left_min:cnt_x_left_max]
            fmi_contour_right = fmi_window[:, cnt_x_right_min:cnt_x_right_min+optmial_pad]
        col_wise_diff_mean = np.nanmean(np.mean(fmi_contour_left, axis = 0) - np.mean(fmi_contour_right, axis = 0))
        if low < col_wise_diff_mean < high:
            filtering_required = True
    return filtering_required

def filter_overlapping_contours_based_on_iou(contours, vugs, threshold=0.2):
    '''
    Function to filter overlapping contours based on IOU

    1. Sort the contours by area in descending order
    2. Iterate over all the contours
    3. Check if the contour is inside any of the already filtered contours
    4. If not, then add the contour to the filtered list only if it is not inside any of the already filtered contours

    Parameters
    ----------
    contours : list
        List of contours
    vugs : list
        List of vugs
    threshold : float
        Threshold for IOU

    Returns
    -------
    filtered_contours : list
        List of filtered contours
    '''
    from utils.contours import is_contour_inside
    # Sort contours by area in descending order
    contours = sorted(contours, key=lambda c: cv.contourArea(c), reverse=True)
    filtered_contours = []
    filtered_vugs = []
    
    for contour, vug in (contours, vugs):
        # Only add contour if it is not inside any of the already filtered contours
        if not is_contour_inside(contour, filtered_contours, threshold=threshold):
            filtered_contours.append(contour)
            filtered_vugs.append(vug)
    
    return filtered_contours, filtered_vugs

def get_contrast_heatmap(
        image, contours, 
        overlapping_bbox_based_on_mean=False, 
        background_laplacian_code=0.0, 
        smoothen_contrast_heatmap=True, sigma=2,
        expansion_factor=0.1
):
    '''
    This function generates a contrast heatmap based on the variance of the Laplacian of the image.
    It takes the image and the contour of the object as input and returns the contrast heatmap.
    It essentially calculates the variance of the Laplacian of the image within the bounding box for each contour.
    It then populates the contrast heatmap with the calculated contrast values.
    If overlapping_bbox_based_on_mean is set to True, it calculates the mean of the contrast values for overlapping bounding boxes.

    Parameters:
    ----------------
    image: 2D numpy array
        The input image
    contour: list
        The contour of the object
    overlapping_bbox_based_on_mean: bool
        If True, calculates the mean of the contrast values for overlapping bounding boxes
    background_laplacian_code: float
        The code to be used for the background in the contrast heatmap

    Returns:
    ----------------
    contrast_heatmap: 2D numpy array
        The contrast heatmap
    contrast_values_list: list
        The list of contrast values for each contour
    '''
    from utils.misc import get_expanded_bbox_dim
    
    contrast_heatmap = np.zeros_like(image)*np.nan
    contrast_values_list = []
    for contour in contours:
        x, y, w, h = get_expanded_bbox_dim(contour, image.shape, expansion_factor=expansion_factor)
        roi = image[y:y+h, x:x+w]
        contrast = np.nanvar(cv.Laplacian(roi, cv.CV_32F))
        if overlapping_bbox_based_on_mean:
            contrast_heatmap[y:y+h, x:x+w] = np.nanmean(
                [np.nanmean(contrast_heatmap[y:y+h, x:x+w]), contrast], axis=0
            )
        else:
            contrast_heatmap[y:y+h, x:x+w] = contrast

        contrast_values_list.append(contrast)

    contrast_heatmap = np.nan_to_num(contrast_heatmap, nan=background_laplacian_code)

    # Apply smoothing to the contrast heatmap if necessary
    if smoothen_contrast_heatmap:
        contrast_heatmap = gaussian_filter(contrast_heatmap, sigma=sigma)

    return contrast_heatmap, contrast_values_list

def get_mean_diff_heatmap(
        image, contours, 
        overlapping_circle_based_on_mean=False, 
        background_mean_code=0.0, 
        smoothen_mean_diff_heatmap=True, sigma=2
):
    '''
    This function generates a mean difference heatmap based on the difference between the mean of the pixels from the contour and the bounding padded circle.
    It takes the image and the contour of the object as input and returns the mean difference heatmap.
    It essentially calculates the mean of the pixels from the contour and the bounding padded circle for each contour.
    It then populates the mean difference heatmap with the calculated mean difference values.
    If overlapping_circle_based_on_mean is set to True, it calculates the mean of the mean difference values for overlapping bounding circles.

    Parameters:
    ----------------
    image: 2D numpy array
        The input image
    contours: list
        The contour of the object
    overlapping_circle_based_on_mean: bool
        If True, calculates the mean of the mean difference values for overlapping bounding circles
    background_mean_code: float
        The code to be used for the background in the mean difference heatmap

    Returns:
    ----------------
    mean_value_heatmap: 2D numpy array
        The mean difference heatmap
    mean_values_list: list
        The list of mean difference values for each contour
    '''
    mean_value_heatmap = np.zeros_like(image)*np.nan
    mean_values_list = []
    for contour in contours:
        mean_value_heatmap_temp = np.zeros_like(image)*np.nan
       # get the center and radius of the contour
        (x,y),radius = cv.minEnclosingCircle(contour)
        center = (int(x),int(y))
        radius = int(radius)
        # Create a binary mask based on the contour coordinates
        mask = np.zeros_like(image, dtype=np.uint8)
        contour_points = np.array(contour)
        # fill the contour with white color
        mask = cv.fillPoly(mask, [contour_points], 255)
        # get the pixels from the contour
        contour_pixels = image[mask > 0]
        # drop nans from the numpy array
        contour_pixels = contour_pixels[~np.isnan(contour_pixels)]
        # get the pixels from the bounding padded circle
        bounding_padded_circle = get_elements_from_circular_roi_based_on_center_and_radius(
            image, center, radius*2
        )
        # get the mean of the pixels from the contour and bounding padded circle
        contour_mean = contour_pixels.mean()
        bounding_padded_circle_mean = bounding_padded_circle.mean()
        # if the difference between the mean of the pixels from the contour and 
        # bounding padded circle is greater than 1 then keep the contour
        difference = bounding_padded_circle_mean - contour_mean

        if overlapping_circle_based_on_mean:
            mean_value_heatmap_temp[mask!=0] = difference
            mean_value_heatmap = np.nanmean(
                [mean_value_heatmap, mean_value_heatmap_temp], axis=0
            )
        else:
            mean_value_heatmap[mask!=0] = difference

        mean_values_list.append(difference)
    mean_value_heatmap = np.nan_to_num(mean_value_heatmap, nan=background_mean_code)

    # Apply smoothing to the contrast heatmap if necessary
    if smoothen_mean_diff_heatmap:
        mean_value_heatmap = gaussian_filter(mean_value_heatmap, sigma=sigma)

    return mean_value_heatmap, mean_values_list