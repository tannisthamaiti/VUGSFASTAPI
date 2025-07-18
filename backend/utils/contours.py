import numpy as np
import cv2 as cv

def is_contour_inside(contour, contours_list, threshold = 0.2):
    '''
    Function to check if a contour is inside another contour based on IOU

    1. Iterate over all the contours
    2. Check if the intersection over union is greater than 0.2
    3. If yes, then return True. This means the contour is inside another contour

    Parameters
    ----------
    contour : numpy array
        Contour to check
    contours : list
        List of contours

    Returns
    -------
    bool
        True if the contour is inside another contour, False otherwise
    '''
    for other_contour in contours_list:
        iou = intersection(contour, other_contour)/union(contour, other_contour)
        if other_contour is not contour and iou >= threshold:
            return True
    return False

def combine_two_contours(c1, c2):
    """combines two contours/centroids
    """
    combined_contour = []
    for c in c1:
        combined_contour.append(c)
    for c in c2:
        combined_contour.append(c)
    return combined_contour

def combine_two_centroids(centroids1, centroids2):
    """Combine two centroids and return combined centroids
    takes
    centroids1: list of centroid1
        len(centroid1) = n
    centroids2: list of centroid2
        len(centroid2) = m

    return:
        combined_centroids: list of combined centroids
    """
    combined_centroids = []
    for centroid1 in centroids1:
        combined_centroids.append(list(centroid1))
    for centroid2 in centroids2:
        combined_centroids.append(list(centroid2))
    return combined_centroids

def compare_two_centroids(reference_centroid, main_centroid, threshold):
    """Compare two centroids and return index of those centroid which are beyond threshold distance
    takes
    reference_centroid: list of refrence centroid from which comparison is to be fone
        len(centroid1) = n
    main_centroid: list of main centroid in which nearby or duplicate centroid is to be found and removed
        len(centroid2) = m

    return:
        idx: index of main_centroid which are beyond threshold distance from reference_centroid
    """
    idx = []
    if len(reference_centroid) != 0:
        for i, centroid in enumerate(main_centroid):
            distance = [np.linalg.norm(np.asarray(centroid) - np.asarray(ref_centroid)) for ref_centroid in reference_centroid]
            if min(distance) > threshold:
                idx.append(i)
    else:
        idx = list(range(len(main_centroid)))
    return idx

def intersection(contour1, contour2):
    '''
    Function to calculate the intersection area between two contours

    1. Create binary masks for both contours
    2. Bitwise AND operation to find the intersection. This will give all the common pixels between the two contours
    3. Calculate the intersecting area by counting the non-zero pixels

    Parameters
    ----------
    contour1 : numpy array
        Contour 1
    contour2 : numpy array
        Contour 2

    Returns
    -------
    intersecting_area : int
        Intersecting area
    '''
    # Create binary masks for both contours
    h, w = max(max(contour1[:, 0, 0]), max(contour2[:, 0, 0])), max(max(contour1[:, 0, 1]), max(contour2[:, 0, 1]))
    mask1 = np.zeros((w, h), dtype=np.uint8)
    mask2 = np.zeros((w, h), dtype=np.uint8)
    cv.drawContours(mask1, [contour1], -1, 255, thickness=cv.FILLED)
    cv.drawContours(mask2, [contour2], -1, 255, thickness=cv.FILLED)
    
    # Bitwise AND operation to find the intersection
    intersection = cv.bitwise_and(mask1, mask2)
    
    # Calculate the intersecting area
    intersecting_area = cv.countNonZero(intersection)
    
    return intersecting_area

def union(contour1, contour2):
    '''
    Function to calculate the union area between two contours

    1. Create binary masks for both contours
    2. Draw the contours on the masks
    3. Calculate the union area by counting the non-zero pixels

    Parameters
    ----------
    contour1 : numpy array
        Contour 1
    contour2 : numpy array
        Contour 2

    Returns
    -------
    union_area : int
        Union area
    '''
    # Create binary masks for both contours
    h, w = max(max(contour1[:, 0, 0]), max(contour2[:, 0, 0])), max(max(contour1[:, 0, 1]), max(contour2[:, 0, 1]))
    mask = np.zeros((w, h), dtype=np.uint8)
    cv.drawContours(mask, [contour1], -1, (255,255,255), thickness=cv.FILLED)
    cv.drawContours(mask, [contour2], -1, (255,255,255), thickness=cv.FILLED)
    
    # Calculate the union area
    union_area = cv.countNonZero(mask)
    
    return union_area

def get_combined_contours_and_centroids(
    contours, centroids, vugs, combined_centroids, final_combined_contour, final_combined_vugs, 
    contours_from_different_thresholds, i, threshold = 5, print_duplicates_info = False
):
    """
    Returns the combined contours and centroids

    Parameters
    ----------

    contours : list
        List of contours
    centroids : list
        List of centroids
    vugs : list
        List of vugs
    combined_centroids : list
        List of combined centroids
    final_combined_contour : list
        List of combined contours
    final_combined_vugs : list
        List of combined vugs
    i : int
        Index of the contours and centroids
    threshold : int
        Threshold for comparing the centroids
        
    Returns
    -------

    combined_centroids : list
        List of combined centroids
    final_combined_contour : list
        List of combined contours

    """
    old_size = len(final_combined_vugs)
    main_size = len(vugs)

    if i == 0:
        combined_centroids = centroids
        final_combined_contour = contours
        final_combined_vugs = vugs
        contours_from_different_thresholds['new'].append(contours)
    else:
        idx = compare_two_centroids(combined_centroids, centroids, threshold)
        combined_centroids = combine_two_centroids(combined_centroids, centroids)
        contours = [contours[i] for i in idx]
        vugs = [vugs[i] for i in idx]
        final_combined_contour = combine_two_contours(final_combined_contour, contours)
        final_combined_vugs = combine_two_contours(final_combined_vugs, vugs)
        contours_from_different_thresholds['new'].append(contours)

    new_size = len(final_combined_vugs)
    contours_from_different_thresholds['duplicates'].append(main_size - (new_size - old_size))

    if print_duplicates_info:
        # old combined vugs size - old_size
        # current mode subtracted image vugs size - main size
        # new combined vugs size - new_size
        # number of duplicates = main_size - (new_size - old_size)
        # new vugs additon = new_size - old_size
        print(
            f"Duplicates Info for Mode-Subtracted Image: {i + 1}:\n"
            f"\tTotal Previous Vugs: {old_size}\n"
            f"\tTotal Potential Vugs In Present Mode-Subtracted Image: {main_size}\n"
            f"\tUnique Vugs After Removing Duplicates: {new_size}\n"
            f"\tTotal Duplicates Present: {main_size - (new_size - old_size)}\n"
            f"\tNew Vugs Added: {new_size - old_size}"
        )

    return combined_centroids, final_combined_contour, final_combined_vugs, contours_from_different_thresholds

def extract_contours_multi_thresholded_fmi_after_area_circularity_filtering(
        fmi_array_one_meter_zone, tdep_array_one_meter_zone, well_radius_one_meter_zone, one_meter_zone_start, 
        one_meter_zone_end, different_thresholds, block_size, c_threshold, min_vug_area, max_vug_area, min_circ_ratio, 
        max_circ_ratio, centroid_threshold, print_duplicates_info = False
):
    """
    Function to extract contours from the FMI after applying different thresholds and filtering based on area and circularity

    1. Apply adaptive thresholding to the FMI
    2. Get well radius and pixel length for filtering purposes
    3. Get contours and centroids from the thresholded image. this is after filtering based on area and circularity
    4. Combine the contours and centroids from different thresholds

    Parameters:
    fmi_array_one_meter_zone: np.array
        FMI array for the derived one meter zone
    tdep_array_one_meter_zone: np.array
        TDEP array for the derived one meter zone
    well_radius_one_meter_zone: np.array
        Well radius for the derived one meter zone
    one_meter_zone_start: float
        Start depth of the derived one meter zone
    one_meter_zone_end: float
        End depth of the derived one meter zone
    different_thresholds: list
        List of different thresholds to apply
    block_size: int
        Block size for adaptive thresholding
    c_threshold: int
        Constant to subtract from the mean
    min_vug_area: int
        Minimum area of the vug
    max_vug_area: int
        Maximum area of the vug
    min_circ_ratio: float
        Minimum circularity ratio of the vug
    max_circ_ratio: float
        Maximum circularity ratio of the vug
    centroid_threshold: int
        Threshold for the centroids

    Returns:
    final_combined_contour: list
        List of combined contours
    final_combined_vugs: list
        List of combined vugs
    holeR: float
        Well radius
    pixLen: float
        Pixel length
    """
    from utils.processing import apply_adaptive_thresholding

    combined_centroids, final_combined_contour, final_combined_vugs = [], [], []
    contours_from_different_thresholds = {
        'new': [],
        'duplicates': []
    }
    # get all the contours from different thresholds for the derived one meter zone
    for i, diff_thresh in enumerate(different_thresholds):

        thresold_img, C_value = apply_adaptive_thresholding(
            fmi_array_one_meter_zone, diff_thresh, block_size = block_size, c = c_threshold
        ) #values changed here
        
        # get well parameters
        holeR, pixLen = well_radius_one_meter_zone.mean()*100, (np.diff(tdep_array_one_meter_zone)*100).mean()
        
        # get contours and centroids from the thresholded image of the derived one meter zone
        contours, centroids, vugs = get_contours(
            thresold_img, depth_to=one_meter_zone_start, depth_from=one_meter_zone_end, radius = holeR, 
            pix_len = pixLen, min_vug_area = min_vug_area, max_vug_area = max_vug_area, 
            min_circ_ratio=min_circ_ratio, max_circ_ratio=max_circ_ratio
        ) #values changed here

        output = get_combined_contours_and_centroids(
            contours, centroids, vugs,combined_centroids, final_combined_contour, final_combined_vugs, 
            contours_from_different_thresholds, i, threshold = centroid_threshold, print_duplicates_info = print_duplicates_info
        )
        combined_centroids, final_combined_contour, final_combined_vugs, contours_from_different_thresholds = output
        
    return final_combined_contour, final_combined_vugs, holeR, pixLen, contours_from_different_thresholds, C_value

def get_centeroid(cnt):
    """Get centroid from a given contour"""
    length = len(cnt)
    sum_x = np.sum(cnt[..., 0])
    sum_y = np.sum(cnt[..., 1])
    return int(sum_x / length), int(sum_y / length)

def get_contours(
    thresold_img, depth_to, depth_from, radius, pix_len,
    min_vug_area=1.2, max_vug_area=10.28,
    min_circ_ratio=0.5, max_circ_ratio=1.0
):
    from utils.misc import get_depth_from_pixel

    height, width = thresold_img.shape
    contours, _ = cv.findContours(thresold_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    print(f"[DEBUG] Total raw contours detected: {len(contours)}")

    circles = []
    centroids = []
    total_vugg = []
    ix = 1          

    HOLE_R = radius
    PIXEL_LEN = pix_len
    PIXEL_SCALE = np.radians(1) * HOLE_R * PIXEL_LEN 

    for idx, contour in enumerate(contours):
        epsilon = 0.01 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:
            #print(f"[DEBUG] Contour {idx}: Rejected — fewer than 3 points (len={len(approx)})")
            continue

        area = cv.contourArea(contour) * PIXEL_SCALE
        (x, y), enclosing_radius = cv.minEnclosingCircle(contour)
        enclosing_radius = int(enclosing_radius)
        area_enclosing = np.pi * enclosing_radius ** 2 * PIXEL_SCALE
        circularity = area / area_enclosing if area_enclosing > 0 else 0

        if area_enclosing == 0:
            #print(f"[DEBUG] Contour {idx}: Rejected — enclosing area is zero")
            continue

        if not (min_vug_area < area <= max_vug_area):
            #print(f"[DEBUG] Contour {idx}: Rejected — area={area:.4f} not in ({min_vug_area}, {max_vug_area})")
            continue

        if not (min_circ_ratio <= circularity <= max_circ_ratio):
            #print(f"[DEBUG] Contour {idx}: Rejected — circularity={circularity:.4f} not in ({min_circ_ratio}, {max_circ_ratio})")
            continue

        # ✅ Passed all filters
        print(f"[DEBUG] Contour {idx}: Accepted — area={area:.4f}, circularity={circularity:.4f}")

        circles.append(contour)
        try:
            cx, cy = get_centeroid(contour)
            centroids.append((cx, cy))
            print(f"[DEBUG] Centroid {idx}: cx={cx:.4f}, cy={cy:.4f}")
        except Exception as e:
            print(f"[ERROR] Contour {idx}: Failed to compute centroid — {e}")
            continue

        depth_val = get_depth_from_pixel(
            val=height - cy,
            scaled_max=depth_to,
            scaled_min=depth_from,
            raw_max=height
        )

        vugg = {
            'id': ix,
            'area': area,
            'depth': depth_val,
            'centroid_x': cx,
            'centroid_y': cy,
            'contour': contour,
            'circularity': circularity,
            'hole_radius': HOLE_R,
            'pix_len': PIXEL_LEN
        }
        total_vugg.append(vugg)
        ix += 1

    print(f"[INFO] Total vugs accepted: {len(total_vugg)}")
    return circles, centroids, total_vugg

def group_overlapping_or_inside_contours(contours, threshold=0.2):
    '''
    This function groups contours that are either overlapping or inside each other.
    It basically checks if a contour is inside another contour and groups them together.
    It returns a list of groups of contours.
    Eacg group is a list of contours that are either overlapping or inside each other.

    Parameters:
    -----------
    contours: list
        List of contours to group
    threshold: float    
        The threshold to consider if a contour is inside another contour

    Returns:
    --------
    grouped_contours: list
        List of groups of contours 
    '''
    grouped_contours = []  # Final list of groups
    processed_indices = set()  # Set to track processed contour indices
    
    for i, contour in enumerate(contours):
        if i in processed_indices:
            continue  # Skip contours that have already been grouped

        group = [contour]  # Start a new group with the current contour
        processed_indices.add(i)  # Mark this contour index as processed

        for j in range(i + 1, len(contours)):
            if j in processed_indices:
                continue  # Skip if it's already in another group

            other_contour = contours[j]

            # Check if the two contours satisfy the "inside" relation
            if is_contour_inside(contour, [other_contour], threshold) or is_contour_inside(other_contour, [contour], threshold):
                group.append(other_contour)
                processed_indices.add(j)  # Mark the index as processed

        # Add the group to the list if it has more than one contour
        if len(group) > 1:
            grouped_contours.append(group)

    return grouped_contours


def merge_contours(grouped_contours, image):
    '''
    This function merges the contours in a group of contours.'
    Basically if there are two, three or more contours that are inside or overlapping with each other,
    then this function merges them into a single contour.

    Parameters:
    -----------
    grouped_contours: list
        List of groups of contours
    image: np.array
        The image on which the contours are drawn

    Returns:
    --------
    merged_contour_list: list
        List of merged contours
    '''
    merged_contour_list = []
    for grp in grouped_contours:
        # Create a blank canvas of same size as the FMI array
        canvas_size = image.shape 
        canvas_merged = np.zeros(canvas_size, dtype=np.uint8)

        for cnt in grp:
            cv.drawContours(canvas_merged, [cnt], -1, 255, thickness=cv.FILLED)

        # Find the merged outer contour
        contours, _ = cv.findContours(canvas_merged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        merged_contour = contours[0]
        merged_contour_list.append(merged_contour)

    return merged_contour_list

def get_contour_size_and_circularity(contour_x, contour_y, holeR, pixLen):
    '''
    This function calculates the area of the contour and the circularity ratio of the contour.
    '''
    ######################HERE WE"RE ONLY TAKING SINGLE HOLE RADIUS, BUT THIS SHOULD BE FIXED LATER DEPENDING ON THE HOLE RADIUS FOR EACH CONTOUR FOUND IN SPECIFIC DEPTH####################
    # Assuming contour_x and contour_y are lists containing x and y coordinates of the contour points
    contour_points = [np.asarray(list(zip(i,j))).reshape(-1, 1, 2) for i,j in zip(contour_x, contour_y)]
    PIXLE_SCALE = np.radians(1) * holeR * pixLen

    contour_area = [cv.contourArea(contour)*PIXLE_SCALE for contour in contour_points]
    min_enclosing_circle_area = [np.pi*(cv.minEnclosingCircle(contour)[-1]**2)*PIXLE_SCALE for contour in contour_points]
    circularity_ratio = [area/min_area for area, min_area in zip(contour_area, min_enclosing_circle_area)]

    return contour_points, contour_area, circularity_ratio

def get_contours_dict(image, contours, holeR, pixLen, start, end):
    '''
    This function takes in the contours and returns a list of dictionaries containing the information about that contour.
    The information includes the area, depth, centroid_x, centroid_y, contour, and circularity of the contour.
    The depth is calculated using the get_depth_from_pixel function.
    The area is calculated using the cv.contourArea function.
    The centroid_x and centroid_y are calculated using the get_centeroid function.
    The circularity is calculated using the formula area/area_enclosing where area_enclosing is the area of the circle enclosing the contour.
    The area_enclosing is calculated using the cv.minEnclosingCircle function.

    Parameters:
    image (numpy.ndarray): The image on which the contours are drawn.
    contours (list): The list of contours.
    holeR (float): The radius of the hole.
    pixLen (float): The pixel length.
    start (float): The start depth.
    end (float): The end depth.

    Returns:
    list: The list of dictionaries containing the information about the contours.
    '''
    from utils.misc import get_depth_from_pixel
    contours_dict = []
    PIXLE_SCALE = np.radians(1) * holeR * pixLen
    for contour in contours:
        #dict_keys(['id', 'area', 'depth', 'centroid_x', 'centroid_y', 'contour', 'circularity'])

        height, width = image.shape
        area = cv.contourArea(contour) * PIXLE_SCALE
        (_, _), radius = cv.minEnclosingCircle(contour)
        radius = int(radius)
        area_enclosing = np.pi * radius**2 * PIXLE_SCALE
        y = get_centeroid(contour)[1]
        x = get_centeroid(contour)[0]
        temp = {
                'id':None, 
                'area': area, 
                'depth': get_depth_from_pixel(
                    val=height-y, scaled_max=start, 
                    scaled_min=end, raw_max=height
                ), 
                'centroid_x':x, 
                'centroid_y': y,
                'contour': contour,
                'circularity': area/area_enclosing
        }
        contours_dict.append(temp)
    return contours_dict

def merge_and_filter_contours(
    fmi_array_one_meter_zone, final_combined_contour, final_combined_vugs, 
    holeR, pixLen, one_meter_zone_start, one_meter_zone_end
):
    '''
    This function merges contours that are inside other contours and filters out contours that are inside other contours.
    It first filters out contours that are inside other contours
    Then it filters out contours that are not inside other contours, it also filters out vugs that are not inside other contours
    Then it groups contours that are inside other contours
    Then it merges contours that are inside other
    It then based on the merged contours, it gets the vugs information
    then it combines the non-overlapping contours with the merged contours

    Args:
    fmi_array_one_meter_zone (np.array): 2D array of one meter zone
    final_combined_contour (list): list of contours
    final_combined_vugs (list): list of vugs
    holeR (int): hole radius
    pixLen (int): pixel length
    one_meter_zone_start (tuple): start coordinate of one meter zone
    one_meter_zone_end (tuple): end coordinate of one meter zone

    Returns:
    final_combined_contour (list): list of contours
    final_combined_vugs (list): list of vugs
    '''
    # merge contours that are inside other contours
    removed_contour = [
        contour for contour in final_combined_contour 
        if is_contour_inside(contour, final_combined_contour)
    ]
    non_overlapping_contour = [
        contour for contour in final_combined_contour 
        if not is_contour_inside(contour, final_combined_contour)
    ]
    non_overlapping_vugs = [
        vug for contour, vug in zip(final_combined_contour, final_combined_vugs)
        if not is_contour_inside(contour, final_combined_contour)
    ]
    grouped_contours = group_overlapping_or_inside_contours(removed_contour)
    merged_contours = merge_contours(grouped_contours, fmi_array_one_meter_zone)
    merged_vugs = get_contours_dict(fmi_array_one_meter_zone, merged_contours, holeR, pixLen, one_meter_zone_start, one_meter_zone_end)

    final_combined_contour = non_overlapping_contour + merged_contours
    final_combined_vugs = non_overlapping_vugs + merged_vugs
    return final_combined_contour, final_combined_vugs