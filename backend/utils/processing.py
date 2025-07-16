import numpy as np
import cv2 as cv

def get_outliers_from_image(
        fmi_array_one_meter_zone, std_no, k, dtype=np.uint8, nearest_point_5=True
):
    from utils.misc import convert_an_array_to_nearest_point_5_or_int
    """
    Returns the outliers from the image

    Parameters
    ----------

    fmi_array_one_meter_zone : numpy array
        FMI array of one meter zone
    std_no : int
        Number of standard deviations to consider as outliers
    k : int
        Number of outliers to return
    dtype : numpy dtype
        Data type of the returned array

    Returns
    -------

    outlier_sorted : numpy array
        Array of outliers

    """
    mean_image = np.nanmean(fmi_array_one_meter_zone)
    std_image = np.nanstd(fmi_array_one_meter_zone)

    z_score = (fmi_array_one_meter_zone - mean_image) / std_image
    outliers = fmi_array_one_meter_zone[np.abs(z_score) > std_no]
    outliers = outliers[outliers>mean_image].astype(dtype)

    outliers = convert_an_array_to_nearest_point_5_or_int(outliers, nearest_point_5=nearest_point_5)

    val_outlier, count_outlier = np.unique(outliers, return_counts=True)
    idx_outlier = np.argsort(count_outlier)[::-1]
    outlier_sorted = val_outlier[idx_outlier][:k]
    return outlier_sorted

def get_block_size_area(M, pixLen, holeR):
    '''Get the area of the block in the cm^2'''
    return M*pixLen*((2*3.14*holeR)/360)*M

def get_mode_of_interest_from_image(fmi_array_one_meter_zone, stride_mode, k):
    from utils.misc import get_every_nth_element
    """
    Returns the mode of interest from the image
    
    Parameters
    ----------
    
    fmi_array_one_meter_zone : numpy array
        FMI array of one meter zone
    stride_mode : int
        Stride for the mode of interest
    k : int
        Number of modes to return
        
    Returns
    -------
        
    mode_of_interest : list
        List of modes of interest
        
    """
    value, counts = np.unique(
        fmi_array_one_meter_zone[~np.isnan(fmi_array_one_meter_zone)].astype(np.float16), 
        return_counts=True
    )
    idx = np.argsort(counts)[::-1]
    mode_of_interest = get_every_nth_element(value[idx], stride_mode)[:k]
    count_of_mode_of_interest = get_every_nth_element(counts[idx], stride_mode)[:k]
    return mode_of_interest, count_of_mode_of_interest

def get_block_size_area(block_size, fmi_array, tdep_array, well_radius):
    '''
    Function to get the area in cm^2 of the block size
    This function essentially calculates the area to which the adaptive thresholding looks at
    when calculating the thresholded FMI

    Parameters:
    block_size (int): The block size in pixels
    fmi_array (np.array): The FMI array
    tdep_array (np.array): The TDEP array
    well_radius (np.array): The well radius array

    Returns:
    block_size_area (float): The area in cm^2 of the block
    '''

    height_per_pixel = (tdep_array[1] - tdep_array[0])*100
    R = (np.nanmean(well_radius)+np.nanstd(well_radius))*100
    width = circumference = 2*np.pi*R
    width_per_pixel = width/len(fmi_array[1])
    height_in_cm = height_per_pixel*block_size
    width_in_cm = width_per_pixel*block_size
    block_size_area = height_in_cm*width_in_cm
    return block_size_area

def apply_adaptive_thresholding(
        fmi_array_one_meter_zone, moi, block_size = 21, 
        c = 'mean', separate_workflow = False
):
    from utils.plotting import plot_single_image
    """
    Returns the thresholded image

    1. Subtract the mode of interest from the FMI image
    2. Apply absolute scaling....**Rethink this!**
    3. Apply adaptive thresholding using Gaussian method and inverse binary thresholding

    Parameters
    ----------

    fmi_array_one_meter_zone : numpy array
        FMI array of one meter zone
    moi : int
        Mode of interest
    block_size : int
        Block size for adaptive thresholding
    c : int or str
        Constant for adaptive thresholding. If 'mean', then the mean of the image is used
    
    Returns
    -------
    thresold_img : numpy array
        Thresholded image
    """

    fmi_array_one_meter_mode_subtracted = fmi_array_one_meter_zone-moi
    if separate_workflow:
        plot_single_image(fmi_array_one_meter_mode_subtracted, f'workflow/2_mode_subtracted_mode_{moi}.png', cmap='YlOrBr')
    fmi_array_one_meter_mode_subtracted = cv.convertScaleAbs(fmi_array_one_meter_mode_subtracted)
    if isinstance(c, str):
        C = fmi_array_one_meter_mode_subtracted.mean()
    else:
        C = c
    thresold_img = cv.adaptiveThreshold(
        fmi_array_one_meter_mode_subtracted, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv.THRESH_BINARY_INV, block_size, C
    )

    if separate_workflow:
        plot_single_image(thresold_img, f'workflow/3_adaptive_thresholding_moi_{moi}.png', cmap='gray')

    return thresold_img, C
