import numpy as np

def get_one_meter_fmi_and_GT(
        one_meter_zone_start, one_meter_zone_end, tdep_array_doi, 
        fmi_array_doi, well_radius, scale_individual_patch = False
):
    """
    Returns the fmi and GT for a one meter zone
    
    Parameters
    ----------
    
    one_meter_zone_start : int
        Start of the one meter zone
    one_meter_zone_end : int
        End of the one meter zone
    tdep_array_doi : numpy array
        Time dependent array of depth of interest
    fmi_array_doi : numpy array
        FMI array of depth of interest
    well_radius : numpy array
        Radius of the well
    gt : pandas dataframe
        Ground truth dataframe
        
    Returns
    -------
        
    fmi_array_one_meter_zone : numpy array
        FMI array of one meter zone
    tdep_array_one_meter_zone : numpy array
        Depth array of one meter zone
    well_radius_one_meter_zone : numpy array
        Radius of the well in one meter zone
    gtZone : pandas dataframe
        Ground truth dataframe of one meter zone
    mask_one_meter_zone : numpy array
        Mask of one meter zone
    """
    from utils.misc import MinMaxScalerCustom
    
    mask_one_meter_zone = (tdep_array_doi>=one_meter_zone_start) & (tdep_array_doi<one_meter_zone_end)
    fmi_array_one_meter_zone = fmi_array_doi[mask_one_meter_zone]
    tdep_array_one_meter_zone = tdep_array_doi[mask_one_meter_zone]
    well_radius_one_meter_zone = well_radius[mask_one_meter_zone]
    #gtZone = gt[(gt.Depth>=one_meter_zone_start) & (gt.Depth<one_meter_zone_end)]

    if scale_individual_patch:
        fmi_array_one_meter_zone = MinMaxScalerCustom(fmi_array_one_meter_zone, min = 0, max = 255)

    return fmi_array_one_meter_zone, tdep_array_one_meter_zone, well_radius_one_meter_zone, mask_one_meter_zone

def preprocessing(fmi_array, tdep_array, well_radius, start, end, scale_individual_patch = False):
    from utils.misc import MinMaxScalerCustom
    '''
    Preprocess the data for further analysis

    1. Mask the data based on the start and end depth
    2. Filter the data based on the mask
    3. Fill the missing values with NaN
    4. Normalize the data using MinMaxScalerCustom

    Parameters
    ----------
    fmi_array : numpy array
        FMI image
    tdep_array : numpy array
        Depth Information
    well_radius : float
        Well radius
    start : float
        Start depth
    end : float
        End depth

    Returns
    -------
    fmi_array_doi_ : numpy array
        FMI image without normalization
    fmi_array_doi : numpy array
        FMI image after preprocessing
    tdep_array_doi : numpy array
        Depth Information after preprocessing
    well_radius_doi : float
        Well radius after preprocessing
    '''
    mask = (tdep_array>=start) & (tdep_array<=end)

    tdep_array_doi = tdep_array[mask]
    fmi_array_doi_ = fmi_array[mask]
    well_radius_doi = well_radius[mask]

    fmi_array_doi_[fmi_array_doi_ == -9999.] = np.nan#np.ma.masked_equal(fmi_array_doi, -9999.0, copy=False).min()

    if not scale_individual_patch:
        fmi_array_doi = MinMaxScalerCustom(fmi_array_doi_, min = 0, max = 255)
    else:
        fmi_array_doi = fmi_array_doi_

    return fmi_array_doi_, fmi_array_doi, tdep_array_doi, well_radius_doi