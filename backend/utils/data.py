import os
import numpy as np
import pandas as pd
from dlisio import dlis
from os.path import join as pjoin
from utils.misc import inch_to_meter
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def get_fmi_with_depth_and_radius(logical_file, dyn=True, reverse=True, max_samples=None, downsample_factor=None):
   

    if not hasattr(logical_file, "channels") or not logical_file.channels:
        raise ValueError("logical_file has no channels.")

    channels = logical_file.channels
    channel_lookup = {ch.name: ch for ch in channels}

    # --- Channel priorities
    fmi_keys = ['FMI_DYN', 'CMI_DYN', 'TB_IMAGE_DYN_DMS_IMG'] if dyn else ['FMI_STAT', 'CMI_STAT', 'TB_IMAGE_STA_DMS_IMG', 'FMI_-STAT']
    depth_keys = ['TDEP', 'MD', 'DEPTH']
    caliper_keys = ['ASSOC_CAL', 'C1_13', 'C1_24', 'C2_13', 'C2_24', 'C3_13', 'C3_24']

    targets = fmi_keys + depth_keys + caliper_keys

    def safe_read_curve(name):
        ch = channel_lookup.get(name)
        if ch:
            try:
                data = ch.curves()
                if max_samples:
                    data = data[:max_samples]
                if downsample_factor:
                    data = data[::downsample_factor]
                unit = getattr(ch, 'units', '')
                return name, data, unit
            except Exception as e:
                print(f"⚠️ Failed to read {name}: {e}")
        return name, None, None

    # --- Parallel safe reads
    results = {}
    with ThreadPoolExecutor() as ex:
        futures = {ex.submit(safe_read_curve, name): name for name in targets}
        for future in futures:
            name, data, unit = future.result()
            if data is not None:
                results[name] = (data, unit)

    # --- Select depth
    tdep_array, tdep_unit = next(((results[k][0], results[k][1]) for k in depth_keys if k in results), (None, None))
    if tdep_array is None:
        raise ValueError("No depth channel found")

    if tdep_unit and (tdep_unit.endswith("in") or tdep_array[0] > 5000):
        tdep_array = inch_to_meter(tdep_array)

    # --- Select FMI image
    fmi_array = next((results[k][0] for k in fmi_keys if k in results), None)
    if fmi_array is None:
        raise ValueError("No FMI image channel found")

    # --- Select well radius (single channel or average)
    if 'ASSOC_CAL' in results:
        cal = results['ASSOC_CAL'][0]
        if results['ASSOC_CAL'][1].endswith('in'):
            cal = inch_to_meter(cal, radius=True)
        well_radius = cal / 2
    else:
        diameters = []
        for k in caliper_keys[1:]:
            if k in results:
                d = results[k][0]
                if results[k][1].endswith('in'):
                    d = inch_to_meter(d, radius=True)
                diameters.append(d)

        if diameters:
            # Memory-safe mean across axis=0
            shape_ok = all(len(d) == len(diameters[0]) for d in diameters)
            if not shape_ok:
                raise ValueError("Caliper logs have mismatched lengths.")
            well_radius = np.nanmean(np.vstack(diameters), axis=0) / 2
        else:
            raise ValueError("No valid caliper logs found.")

    # --- Reverse logic
    if reverse and tdep_array[0] < tdep_array[-1]:
        fmi_array = fmi_array[::-1]
        tdep_array = tdep_array[::-1]
        well_radius = well_radius[::-1]
    elif not reverse and tdep_array[0] > tdep_array[-1]:
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

    #gt_path = [i for i in os.listdir(data_path) if i.endswith('.csv')][0]
    #gt = pd.read_csv(pjoin(data_path, gt_path)).dropna()[1:].astype('float')
    #return fmi_array, tdep_array, well_radius, gt
    return fmi_array, tdep_array, well_radius

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