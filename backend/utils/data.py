import os
import numpy as np
import pandas as pd
from dlisio import dlis
from os.path import join as pjoin

def get_logical_file(dlis_file_name, data_path, dyn):
    """
    This function is used to get the logical file from the DLIS file and 
    filter it based on the type of data (static or dynamic) provided by the user.

    Parameters
    ----------

    dlis_file_name : list
        List of DLIS file names
    data_path : str
        Path of the data folder
    dyn : bool
        True if dynamic data is required, False if static data is required

    Returns
    -------
    logical_file : dlisio.logicalfile
        Logical file containing the data of the type specified by the user

    """
    #get correct file name as per user input
    if len(dlis_file_name)>1:
        if dyn:
            fmi_file_name = [i for i in dlis_file_name if 'dyn' in i.lower()]
            print('dynamic dlis file name retrieved')
        else:
            fmi_file_name = [i for i in dlis_file_name if 'stat' in i.lower()]
            print('static dlis file name retrieved')
    elif len(dlis_file_name)==1:
        fmi_file_name = dlis_file_name
        print('common dlis file name retrieved')
    else:
        print('no dlis file found for this well')
        return

    dlis_path = pjoin(data_path, fmi_file_name[0])
    print("path for dlis file: {}".format(dlis_path))

    dlis_file = dlis.load(dlis_path)

    #get correct logical file as per user input
    if len(dlis_file)>1:
        if dyn:
            logical_file = [i for i in dlis_file if 'dyn' in str(i).lower()][0]
            print('dynamic logical file retrieved')
        else:
            logical_file = [i for i in dlis_file if 'stat' in str(i).lower()][0]
            print('static logical file retrieved')
    else:
        logical_file = dlis_file[0]
        print('logical file retrieved')

    return logical_file

def get_fmi_with_depth_and_radius(logical_file, dyn, reverse=True):
    from utils.misc import inch_to_meter
    import numpy as np

    fmi_array = None
    tdep_array = None
    well_radius = None

    if not hasattr(logical_file, "channels") or not logical_file.channels:
        raise ValueError("logical_file has no channels or channels list is empty.")

    print("logical_file.channels content:")
    for idx, channel in enumerate(logical_file.channels):
        print(f"{idx+1}. Name: {channel.name}, Unit: {getattr(channel, 'units', 'N/A')}, Sample count: {len(channel.curves())}")

    # Find image data
    for channel in logical_file.channels:
        if dyn:
            if channel.name in ['FMI_DYN', 'CMI_DYN', 'TB_IMAGE_DYN_DMS_IMG']:
                fmi_array = channel.curves()
                print(f'{channel.name} loaded, shape: {fmi_array.shape}')
        else:
            if channel.name in ['FMI_STAT', 'CMI_STAT', 'TB_IMAGE_STA_DMS_IMG', 'FMI_-STAT']:
                fmi_array = channel.curves()
                print(f'{channel.name} loaded, shape: {fmi_array.shape}')

        if channel.name in ['TDEP', 'MD', 'DEPTH']:
            tdep_array = channel.curves()
            print(f'{channel.name} loaded, shape: {tdep_array.shape}')
            print(f'Original depth: {tdep_array}')
            print(f'Unit: {channel.units}')

            if channel.units.endswith('in') or tdep_array[0] > 5000:
                tdep_array = inch_to_meter(tdep_array)
                print(f'Depth after conversion: {tdep_array}')

    if fmi_array is None or tdep_array is None:
        raise ValueError("Missing required FMI or depth data in logical_file")

    # Well radius logic
    channels = logical_file.channels
    channel_names = [c.name for c in channels]

    if 'ASSOC_CAL' in channel_names:
        print('Getting radius from ASSOC_CAL channel')
        radius_channel = channels[channel_names.index('ASSOC_CAL')]
        well_diameter = radius_channel.curves()
        well_diameter[well_diameter == -9999.] = np.nan
        if 'in' in radius_channel.units:
            well_diameter = inch_to_meter(well_diameter, radius=True)
        well_radius = well_diameter / 2
    else:
        caliper_log = ['C1_13', 'C1_24', 'C2_13', 'C2_24', 'C3_13', 'C3_24']
        try:
            well_diameter = np.asarray([channels[channel_names.index(i)].curves() for i in caliper_log])
            well_diameter[well_diameter == -9999.] = np.nan
            well_diameter = well_diameter.mean(axis=0)
            if 'in' in channels[channel_names.index(caliper_log[0])].units:
                well_diameter = inch_to_meter(well_diameter, radius=True)
            well_radius = well_diameter / 2
        except ValueError:
            raise ValueError("Could not retrieve caliper data for well radius estimation.")

    # Reverse logic
    if reverse:
        if tdep_array[0] < tdep_array[-1]:
            print("Reversing depth and image arrays...")
            fmi_array = fmi_array[::-1]
            tdep_array = tdep_array[::-1]
            well_radius = well_radius[::-1]
    else:
        if tdep_array[0] > tdep_array[-1]:
            fmi_array = fmi_array[::-1]
            tdep_array = tdep_array[::-1]
            well_radius = well_radius[::-1]

    return fmi_array, tdep_array, well_radius


def get_data(data_path, dyn=False, reverse=False):
    '''
    This function reads the dlis file and returns the FMI image, TDEP image and 
    well radius along with the ground truth data

    1. Get the actual dlis from the folder
    2. Get the appropriate logical file
    3. Get the FMI image, TDEP image and well radius
    4. Get the ground truth data using the folder

    Parameters
    ----------
    data_path : str
        Path to the dlis file
    dyn : bool, optional
        Whether the data is dynamic or static, by default False
    reverse : bool, optional
        Whether to reverse the FMI image based on the depth, by default False

    Returns
    -------
    fmi_array : numpy array
        FMI image
    tdep_array : numpy array
        Depth Information
    well_radius : float
        Well radius
    gt : pandas DataFrame
        Ground truth data
    '''
    dlis_file_name = [i for i in os.listdir(data_path) if i.lower().endswith('.dlis')]
    logical_file = get_logical_file(dlis_file_name, data_path, dyn=dyn)
    fmi_array, tdep_array, well_radius = get_fmi_with_depth_and_radius(logical_file, dyn=dyn, reverse=reverse)

    gt_path = [i for i in os.listdir(data_path) if i.endswith('.csv')][0]
    gt = pd.read_csv(pjoin(data_path, gt_path)).dropna()[1:].astype('float')
    return fmi_array, tdep_array, well_radius, gt

def get_mud_info(data_path, dyn):
    dlis_file_name = [i for i in os.listdir(data_path) if i.lower().endswith('.dlis')]
    logical_file = get_logical_file(dlis_file_name, data_path, dyn=dyn)

    full_param_info = {}
    mud_param_info = {}
    ohm_related_param_info = {}

    for param in logical_file.parameters:
        (
            long_name, name, values
        ) = (
            param.long_name, param.name, 
            param.describe().info.strip().split('\n\n')[-1].split(':')[-1].strip()
        )
        
        full_param_info[name] = {
            'long_name': long_name,
            'values': values
        }

        if 'mud' in param.long_name.lower():
            mud_param_info[name] = {
                'long_name': long_name,
                'values': values
            }

        if 'ohm' in values.lower():
            ohm_related_param_info[name] = {
                'long_name': long_name,
                'values': values
            }

    return full_param_info, mud_param_info, ohm_related_param_info