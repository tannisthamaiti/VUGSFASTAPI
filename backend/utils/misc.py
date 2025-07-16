import numpy as np
import cv2 as cv

def get_expanded_bbox_dim(contour, image_shape, expansion_factor=0.1):
    '''
    This function takes a contour and the shape of the image and returns the expanded bounding box dimensions.
    The bounding box is expanded by 10% in all directions.
    This is useful when we want to crop the image around the contour and 
    we want to include some extra area around the contour to do some processing.

    Parameters:
    -----------
    contour: numpy array
        The contour for which the bounding box is to be calculated.
    image_shape: tuple
        The shape of the image.

    Returns:
    --------
    x, y, w, h: int
        The expanded bounding box dimensions.
    '''

    # get the bounding box for each contour
    # (x,y) -> top-left coordinate 
    # (w,h) -> width and height. 
    # taken from https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
    x, y, w, h = cv.boundingRect(contour)

    # increase the bounding box by 10% in all direction
    x -= int(expansion_factor*w) # shift x to left by 10% of width 
    y -= int(expansion_factor*h) # shift y to top by 10% of height
    w += int(2*expansion_factor*w) # after shifting x and y to left and top, increase width by 20%, first 10% is for left and next 10% is for right
    h += int(2*expansion_factor*h) # after shifting x and y to left and top, increase height by 20%, first 10% is for top and next 10% is for bottom
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x+w > image_shape[1]:
        w = image_shape[1] - x
    if y+h > image_shape[0]:
        h = image_shape[0] - y
    return x, y, w, h

def inch_to_meter(tdep_array, radius = False):
    """
    converts the depth values from inches to meters.

    parameters
    ----------
    tdep_array : numpy array
        the depth values in inches.
    radius : bool, optional
        if True, the depth values are in meters. the default is False.

    returns
    -------
    tdep_array : numpy array
        the depth values in meters.
    """

    print('converting inch to meters')
    if not radius:
        tdep_array = tdep_array/10
    depth_ft = tdep_array*0.0833333
    tdep_array = depth_ft*0.3048
    return tdep_array

def MinMaxScalerCustom(X, min = 0, max = 1):
    """
    Scales the input data between the specified min and max values.

    Parameters
    ----------
    X : numpy array
        The input data.
    min : int, optional
        The minimum value. The default is 0.
    max : int, optional
        The maximum value. The default is 1.

    Returns
    -------
    X_scaled : numpy array
        The scaled data.
    """
    X_std = (X - np.nanmin(X)) / (np.nanmax(X) - np.nanmin(X))
    X_scaled = X_std * (max - min) + min
    return X_scaled

def get_depth_from_pixel(val, scaled_max, scaled_min, raw_max, raw_min=0):
    """
    Converts the pixel value to depth value.

    Parameters
    ----------
    val : int
        The pixel value.
    scaled_max : int
        The maximum depth value.
    scaled_min : int
        The minimum depth value.
    raw_max : int
        The maximum pixel value.
    raw_min : int, optional
        The minimum pixel value. The default is 0.

    Returns
    -------
    int
        The depth value.
    """
    return ((scaled_max-scaled_min)/raw_max) * (val-raw_min) + scaled_min

def get_every_nth_element(array, n):
    """Get every nth element from array

    Parameters
    ----------
    array : list
        List of elements
    n : int
        nth element

    Returns
    -------
    list
        List of every nth element
    """
    return array[::n]

def check(value, blind_depth_range):
    """
    checks if the value is within the blind depth range.
    
    parameters
    ----------
    value : float
        the value to be checked.
    blind_depth_range : tuple
        the blind depth range.

    returns
    -------
    bool
        True if the value is within the blind depth range, False otherwise.
    """
    
    if blind_depth_range.start <= value <= blind_depth_range.stop:
        return True
    return False

def convert_an_array_to_nearest_point_5_or_int(array, nearest_point_5=True):
    """
    Converts an array to nearest point 5 or int

    Parameters
    ----------

    array : numpy array
        Array to be converted
    nearest_point_5 : bool
        If True, then the array is converted to nearest point 5. 
        If False, then the array is converted to nearest int

    Returns
    -------

    array : numpy array
        Converted array

    """
    if nearest_point_5:
        return np.round(array*2)/2
    else:
        return np.round(array)