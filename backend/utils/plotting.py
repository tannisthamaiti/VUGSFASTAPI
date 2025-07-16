import seaborn as sns
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from os.path import join as pjoin
from sklearn.preprocessing import MinMaxScaler
import io
from utils.processing import apply_adaptive_thresholding
from utils.contours import get_contours, get_centeroid
from utils.processing import apply_adaptive_thresholding


def plot_original(
        fmi_array_one_meter_zone, depth, save_path, save=False, depth_in_name=False, 
        picture_format='png', custom_fname='', colorbar=True, labelsize=20, fontsize=20, plot=True,
        well_name=''
):
    '''
    Plot the original FMI image and its histogram.
    '''
    if depth_in_name:
        depth = str(round(depth, 2))
    else:
        depth = ''
    if custom_fname:
        custom_fname = f'_{custom_fname}'
    if well_name:
        well_name = f'_{well_name}'

    _, ax = plt.subplots(1, 2, figsize=(15, 7))
    im = ax[0].imshow(fmi_array_one_meter_zone, cmap='YlOrBr')
    sns.histplot(fmi_array_one_meter_zone.reshape(-1), bins=100, ax=ax[1])
    ax[0].get_yaxis().set_visible(False)
    ax[0].set_xlabel('Azimuth\n(a)', fontsize=fontsize)
    ax[1].set_xlabel('Pixel Intensity\n(b)', fontsize=fontsize)
    ax[1].set_ylabel('Frequency', fontsize=fontsize)
    ax[0].tick_params(axis='both', which='major', labelsize=labelsize)
    ax[1].tick_params(axis='both', which='major', labelsize=labelsize)
    ax[0].set_xticks(np.linspace(0, fmi_array_one_meter_zone.shape[1], 5))
    ax[1].set_xticks(np.linspace(np.nanmin(fmi_array_one_meter_zone), np.nanmax(fmi_array_one_meter_zone), 5))
    if colorbar:
        create_colorbar(ax[0], im, fontsize=labelsize)
    plt.tight_layout()
    if save:
        plt.savefig(pjoin(save_path, f'1_fmi_original_hist_{depth}{custom_fname}{well_name}.'+picture_format), format=picture_format, dpi=500, bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close()

def plot_mode_subtracted_histogram(
        fmi_array_one_meter_zone, different_thresholds, depth, save_path, save = False, 
        depth_in_name=False, picture_format='png', figsize=(10, 7), label_fontsize=20, 
        tick_fontsize=20, legend_fontsize=20, dpi=500, ncols_legend=1, custom_fname='', 
        plot=True, absolute_value=False, well_name='', mode_value_in_legend=False
):
    '''
    Plot histogram of FMI values after subtracting different thresholds/modes
    '''
    if depth_in_name:
        depth = str(round(depth, 2))
    else:
        depth = ''
    if custom_fname:
        custom_fname = f'_{custom_fname}'
    if well_name:
        well_name = f'_{well_name}'

    plt.figure(figsize=figsize)
    for i, diff_thresh in enumerate(different_thresholds):
        if absolute_value:
            mode_subtracted_fmi = np.abs(fmi_array_one_meter_zone-diff_thresh)
        else:
            mode_subtracted_fmi = fmi_array_one_meter_zone-diff_thresh
        label = f'Mode {i+1}'
        if mode_value_in_legend:
            label += f': {diff_thresh:.2f}'
        sns.kdeplot(mode_subtracted_fmi.reshape(-1), label=label)
    plt.xlabel('Pixel Intensity', fontsize=label_fontsize)
    plt.ylabel('Density/Frequency', fontsize=label_fontsize)
    plt.legend(fontsize=legend_fontsize, ncol=ncols_legend)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.tight_layout()
    if save:
        plt.savefig(
            pjoin(save_path, f'2_fmi_mode_subtracted_hist_{depth}{custom_fname}{well_name}.'+picture_format), 
            format=picture_format, dpi=dpi, bbox_inches='tight'
        )
    if plot: 
        plt.show()
    else:
        plt.close()

def plot_thresholded_fmi(
        fmi_array_one_meter_zone, fmi_array_unscaled_one_meter_zone, different_thresholds, block_size, c_threshold, depth, 
        save_path, save = False, depth_in_name=False, picture_format='png', custom_fname='',
        colorbar=True, labelsize=20, fontsize=20, plot=True, well_name=''
):
    from utils.processing import apply_adaptive_thresholding
    """
    This function plots the original FMI and the thresholded FMI for different thresholds.
    """
    if depth_in_name:
        depth = str(round(depth, 2))
    else:
        depth = ''
    if custom_fname:
        custom_fname = f'_{custom_fname}'
    if well_name:
        well_name = f'_{well_name}'
    _, ax1 = plt.subplots(1, len(different_thresholds) + 1, figsize=(25, 7))
    im = ax1[0].imshow(fmi_array_unscaled_one_meter_zone, cmap='YlOrBr')
    ax1[0].get_yaxis().set_visible(False)
    ax1[0].set_title('Original FMI', fontsize=fontsize)
    ax1[0].set_xticks(np.linspace(0, fmi_array_one_meter_zone.shape[1], 5))
    ax1[0].tick_params(axis='both', which='major', labelsize=labelsize)
    if colorbar:
        create_colorbar(ax1[0], im, fontsize=labelsize)

    # get all the contours from different thresholds for the derived one meter zone
    for i, diff_thresh in enumerate(different_thresholds):

        thresold_img, C_value = apply_adaptive_thresholding(
            fmi_array_one_meter_zone, diff_thresh, block_size = block_size, c = c_threshold
        ) #values changed here

        ax1[i+1].imshow(thresold_img, cmap='gray')
        ax1[i+1].set_title(f'Mode {i+1}', fontsize=fontsize)
        ax1[i+1].get_yaxis().set_visible(False)
        ax1[i+1].set_xticks(np.linspace(0, fmi_array_one_meter_zone.shape[1], 5))
        ax1[i+1].tick_params(axis='both', which='major', labelsize=labelsize)
    plt.tight_layout()
    if save: 
        plt.savefig(
            pjoin(save_path, f'3_fmi_thresholded_{depth}{custom_fname}{well_name}.'+picture_format), 
            format=picture_format, dpi=500, bbox_inches='tight'
        )
    if plot:
        plt.show()
    else:
        plt.close()

def create_custom_cmap(
        base_cmap='plasma', 
        custom_color_picker = [0, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.999999],
        threshold = 0.8, min_value=-2.56, max_value=0.62, mudtype=None,
        laplacian_heatmap=True, epsilon = 0.01
):
    '''
    This function creates a custom colormap based on the base colormap. 
    The custom colormap is created by selecting specific colors from the base colormap.
    This function is useful when you want to create a custom colormap with specific schemes for better visualization.
    '''
    from matplotlib.colors import LinearSegmentedColormap

    if custom_color_picker:
        # Get the plasma colormap
        cmap = plt.get_cmap(base_cmap)
        
        # Custom color list based on the plasma colormap
        custom_colors = [cmap(i) for i in custom_color_picker]
    else:
        if laplacian_heatmap:
            if threshold > max_value:
                threshold = max_value
            # scale the threshold between 0 and 1 based on the max value
            threshold = threshold/max_value
            # Define custom colormap with smooth transitions
            custom_colors = [
                (0.0, '#0D0887'),  #----> Dark Blue
                (0.0000001, '#FF0000'), #----> Red
                (threshold, '#FAA0A0'), #----> Light Red
                (threshold, '#7CFC00'), #----> Light Green
                (1.0, '#008000'), #----> Dark Green
            ]
        else:
            if mudtype == 'water':
                threshold = -threshold
            elif mudtype == 'oil':
                threshold = threshold
            else:
                raise ValueError('Mud type not recognized. Please provide a valid mud type')
                
            if threshold < min_value:
                threshold = min_value
            elif threshold > max_value:
                threshold = max_value
            
            if 0 < min_value:
                min_value = 0

            threshold = (threshold - min_value) / (max_value - min_value)
            zero_rescaled = (0 - min_value) / (max_value - min_value)

            if mudtype == 'water':
                # Define custom colormap with smooth transitions
                custom_colors = [
                    (0.0, '#7CFC00'),          # Light Green
                    (threshold, '#008000'),     # Dark Green
                    (threshold + epsilon, '#FAA0A0'),  # Light Red (a tiny increment after threshold)
                    (zero_rescaled - epsilon, '#FF0000'),  # Red (just before zero_rescaled)
                    (zero_rescaled, '#0D0887'), # Dark Blue (exactly at zero_rescaled)
                    (min(zero_rescaled + epsilon, 1.0), '#FFE5B4'),  # Light Orange (just after zero_rescaled) # fix this later, this is a quick hack
                    (1.0, '#FFC000')            # Dark Orange
                ]
            elif mudtype == 'oil':
                # Define custom colormap with smooth transitions
                custom_colors = [
                    (0.0, '#FFC000'),          # Dark Orange
                    (max(zero_rescaled - epsilon, 0.0), '#FFE5B4'),  # Light Orange (just before zero_rescaled)
                    (zero_rescaled, '#0D0887'), # Dark Blue (exactly at zero_rescaled)
                    (zero_rescaled + epsilon, '#FF0000'),  # Red (just after zero_rescaled)
                    (threshold - epsilon, '#FAA0A0'),  # Light Red (just before threshold)
                    (threshold, '#7CFC00'),     # Light Green
                    (1.0, '#008000')            # Dark Green
                ]
            else:
                raise ValueError('Mud type not recognized. Please provide a valid mud type')

    # Create custom colormap
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_colors, N=256)

    return custom_cmap

def create_colorbar(ax, img, fontsize=12):
    '''
    This function creates a colorbar for the given image and axis.
    '''
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.tick_params(labelsize=fontsize)


def plot_single_image_contours(
        img, circles, ax, linewidth=2, cmap='YlOrBr', color='black', legend=False, colorbar=False, 
        title='', fontsize=12, labelsize=12, axis_off=False
):
    '''
    Plot the image with the contours
    '''
    im = ax.imshow(img, cmap=cmap)

    for i, pts in enumerate(circles):
        x = pts[:, 0, 0]
        y = pts[:, 0, 1]
        x = np.append(x, x[0])
        y = np.append(y, y[0])

        # if the legend is boolean, then use the index as the label, else use the label provided, 
        # for that first check the type of the legend
        if type(legend) == bool and legend:
            legend_label = str(i)
        elif type(legend) != bool and legend:
            legend_label = legend
        else:
            legend_label = None
        ax.plot(x, y, color=color, linewidth=linewidth, label=legend_label)

    ax.set_title(title, fontsize=fontsize)
    if axis_off:
        ax.axis('off')

    if colorbar:
        create_colorbar(ax, im, fontsize=labelsize)

def histogram_plot(
        contrast_values_list, filter_cutoffs, ax, color='royalblue', kde=True, edgecolor='black', 
        title='', xlim=(0, 5), vline_color='red', vline_linestyle='--', vline_linewidth=1.5
):
    
    sns.histplot(
        contrast_values_list, 
        color=color, 
        kde=kde, 
        edgecolor=edgecolor,
        ax=ax
    )

    y_min, y_max = ax.get_ylim()
    
    ax.vlines(
        filter_cutoffs, y_min, y_max, color=vline_color, 
        linestyle=vline_linestyle, linewidth=vline_linewidth
    )
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_title(title)
    ax.set_box_aspect(1)

def laplacian_kde_plot(
        contrast_values_list, filter_cutoffs, ax, color='royalblue', fill=True, 
        title='', xlim=(0, 5), vline_color='red', vline_linestyle='--', vline_linewidth=1.5
):
    
    sns.kdeplot(
        contrast_values_list, 
        color=color, 
        fill=fill,
        ax=ax
    )

    y_min, y_max = ax.get_ylim()
    
    ax.vlines(
        filter_cutoffs, y_min, y_max, color=vline_color, 
        linestyle=vline_linestyle, linewidth=vline_linewidth
    )
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_title(title)
    ax.set_box_aspect(1)

def plot_image(image, cmap, vmax, ax, title='', colorbar=False, fontsize=12, labelsize=12, axis_off=False):
    im = ax.imshow(image, cmap=cmap, vmax=vmax)
    if colorbar:
        create_colorbar(ax, im, fontsize=labelsize)
    ax.set_title(title, fontsize=fontsize)
    if axis_off:
        ax.axis('off')

def plot_fmi_with_original_contours(
        fmi_array_one_meter_zone, fmi_array_unscaled_one_meter_zone, different_thresholds, block_size, c_threshold, depth, 
        save_path, save=False, depth_in_name=False, picture_format='png', custom_fname='',
        colorbar=True, labelsize=20, fontsize=20, figsize=(25, 7), plot=True, well_name=''
):
    
    """
    Function to plot the FMI with contours for different thresholds
    """
    if depth_in_name:
        depth = str(round(depth, 2))
    else:
        depth = ''
    if custom_fname:
        custom_fname = f'_{custom_fname}'
    if well_name:
        well_name = f'_{well_name}'
    _, ax1 = plt.subplots(1, len(different_thresholds) + 1, figsize=figsize)
    im = ax1[0].imshow(fmi_array_unscaled_one_meter_zone, cmap='YlOrBr')
    ax1[0].get_yaxis().set_visible(False)
    ax1[0].set_title('Original FMI', fontsize=fontsize)
    ax1[0].set_xticks(np.linspace(0, fmi_array_one_meter_zone.shape[1], 5))
    ax1[0].tick_params(axis='both', which='major', labelsize=labelsize)
    if colorbar:
        create_colorbar(ax1[0], im, fontsize=labelsize)


    # get all the contours from different thresholds for the derived one meter zone
    for i, diff_thresh in enumerate(different_thresholds):

        thresold_img, C_value = apply_adaptive_thresholding(
            fmi_array_one_meter_zone, diff_thresh, block_size = block_size, c = c_threshold
        ) #values changed here
        contours, _ = cv.findContours(thresold_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        mode_subtracted_fmi = np.abs(fmi_array_one_meter_zone-diff_thresh)
        plot_single_image_contours(
            mode_subtracted_fmi, contours, ax1[i+1], linewidth=2, cmap='YlOrBr', 
            colorbar=colorbar, fontsize=fontsize, labelsize=labelsize
        )
        ax1[i+1].set_title(f'Mode {i+1}', fontsize=fontsize)
        ax1[i+1].get_yaxis().set_visible(False)
        ax1[i+1].set_xticks(np.linspace(0, fmi_array_one_meter_zone.shape[1], 5))
        ax1[i+1].tick_params(axis='both', which='major', labelsize=labelsize)
    plt.tight_layout()
    if save:
        plt.savefig(
            pjoin(save_path, f'4_fmi_contour_orig_{depth}{custom_fname}{well_name}.'+picture_format), 
            format=picture_format, dpi=500, bbox_inches='tight'
        )
    if plot:
        plt.show()
    else:
        plt.close()

def plot_fmi_with_area_circularity_filtered_contours(
    fmi_array_one_meter_zone,
    fmi_array_unscaled_one_meter_zone,
    different_thresholds,
    block_size,
    c_threshold,
    one_meter_zone_start,
    one_meter_zone_end,
    well_radius_one_meter_zone,
    tdep_array_one_meter_zone,
    min_vug_area,
    max_vug_area,
    min_circ_ratio,
    max_circ_ratio,
    depth,
    save_path,
    save=False,
    depth_in_name=False,
    picture_format='png',
    custom_fname='',
    colorbar=True,
    labelsize=10,
    fontsize=10,
    figsize=(6,6),
    plot=True,
    well_name=''
):
    """
    Plots FMI images with area- and circularity-filtered contours using adaptive thresholding.
    The left panel shows the original FMI; subsequent panels show processed FMI at different thresholds.
    """
    print( one_meter_zone_start,one_meter_zone_end)
    print(np.shape(tdep_array_one_meter_zone))
    # Validate threshold list
    assert len(different_thresholds) == 1, "Pass exactly one threshold value."

    # Format file naming suffixes
    depth_str = str(round(depth, 2)) if depth_in_name else ''
    custom_str = f'_{custom_fname}' if custom_fname else ''
    well_str = f'_{well_name}' if well_name else ''

    

    # Precompute geometry
    hole_radius_cm = well_radius_one_meter_zone.mean() * 100
    pixel_length_cm = np.diff(tdep_array_one_meter_zone).mean() * 100

    # Loop over thresholds and generate plots
   # Apply thresholding
    threshold = different_thresholds[0]
    threshold_img, _ = apply_adaptive_thresholding(
        fmi_array_one_meter_zone, threshold, block_size=block_size, c=c_threshold
    )

    contours, _, _ = get_contours(
            threshold_img,
            depth_to=one_meter_zone_start,
            depth_from=one_meter_zone_end,
            radius=hole_radius_cm,
            pix_len=pixel_length_cm,
            min_vug_area=min_vug_area,
            max_vug_area=max_vug_area,
            min_circ_ratio=min_circ_ratio,
            max_circ_ratio=max_circ_ratio
        )

    mode_subtracted_fmi = np.abs(fmi_array_one_meter_zone - threshold)
    print(np.shape(mode_subtracted_fmi))
    print(type(contours))
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=150)
    plot_single_image_contours(
            mode_subtracted_fmi,
            contours,
            ax,
            linewidth=2,
            cmap='YlOrBr',
            colorbar=colorbar,
            fontsize=fontsize,
            labelsize=labelsize
        )
    ax.set_title(f'Contours (Threshold: {f"{threshold:.1f}"})', fontsize=fontsize)
    #ax.get_yaxis().set_visible(False)
    ax.set_xticks(np.linspace(0, fmi_array_one_meter_zone.shape[1], 5))
    yticks_idx = np.linspace(0, len(tdep_array_one_meter_zone) - 1, 5, dtype=int)
    ytick_vals = tdep_array_one_meter_zone[yticks_idx]
    ax.set_yticks(yticks_idx)
    ax.set_yticklabels([f"{val:.1f}" for val in ytick_vals])
    ax.tick_params(axis='both', which='major', labelsize=labelsize)

    plt.tight_layout()

    # Save if needed
    if save:
        filename = f'5_fmi_contour_area_circularity_filtered_{depth_str}{custom_str}{well_str}.{picture_format}'
        full_path = pjoin(save_path, filename)
        plt.savefig(
            pjoin(save_path, filename),
            format=picture_format,
            dpi=500,
            bbox_inches='tight'
        )

    # if plot:
    #     plt.show()
    # else:
    #     plt.close()

    return full_path


def plot_combined_contour_on_fmi(
        fmi_array_one_meter_zone, final_combined_contour, depth, save_path, 
        save=False, depth_in_name=False, picture_format='png', custom_fname='',
        colorbar=True, labelsize=10, fontsize=10, plot=True, well_name='',
        figsize=(15, 7)
):
    """
    Function to plot the combined contours on the FMI
    """
    if depth_in_name:
        depth = str(round(depth, 2))
    else:
        depth = ''
    if custom_fname:
        custom_fname = f'_{custom_fname}'
    if well_name:
        well_name = f'_{well_name}'
    _, ax = plt.subplots(1, 2, figsize=figsize)
    im = ax[0].imshow(fmi_array_one_meter_zone, cmap='YlOrBr')
    if colorbar:
        create_colorbar(ax[0], im, fontsize=labelsize)
    plot_single_image_contours(
        fmi_array_one_meter_zone, final_combined_contour, ax[1], linewidth=2, cmap='YlOrBr',
        colorbar=colorbar, fontsize=fontsize, labelsize=labelsize
    )
    ax[0].get_yaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[0].tick_params(axis='both', which='major', labelsize=labelsize)
    ax[1].tick_params(axis='both', which='major', labelsize=labelsize)
    ax[0].set_xticks(np.linspace(0, fmi_array_one_meter_zone.shape[1], 5))
    ax[1].set_xticks(np.linspace(0, fmi_array_one_meter_zone.shape[1], 5))
    ax[0].set_title('Original FMI', fontsize=fontsize)
    ax[1].set_title('Combined Contours', fontsize=fontsize)
    plt.tight_layout()
    if save:
        plt.savefig(
            pjoin(save_path, f'6_fmi_combined_contours_{depth}{custom_fname}{well_name}.'+picture_format), format=picture_format, 
            dpi=500, bbox_inches='tight'
        )
    if plot:
        plt.show()
    else:
        plt.close()

def plot_contours_from_different_thresholds(
        fmi_array_one_meter_zone, contours_from_different_thresholds, depth, save_path, save=False, 
        depth_in_name=False, picture_format='png', custom_fname='', figsize=(25, 7),
        colorbar=True, labelsize=20, fontsize=20, plot=True, well_name=''
):
    """
    Function to plot unique contours from different thresholds
    """
    if depth_in_name:
        depth = str(round(depth, 2))
    else:
        depth = ''
    if custom_fname:
        custom_fname = f'_{custom_fname}'
    if well_name:
        well_name = f'_{well_name}'
    _, ax = plt.subplots(1, len(contours_from_different_thresholds['new']) + 1, figsize=figsize)
    for i in range(len(contours_from_different_thresholds['new'])+1):
        if i == 0:
            im = ax[i].imshow(fmi_array_one_meter_zone, cmap='YlOrBr')
            ax[i].set_title('Original FMI', fontsize=labelsize)
            if colorbar:
                create_colorbar(ax[i], im, fontsize=labelsize)
        else:
            plot_single_image_contours(
                fmi_array_one_meter_zone, contours_from_different_thresholds['new'][i-1], ax[i], linewidth=2, cmap='YlOrBr',
                colorbar=colorbar, fontsize=fontsize, labelsize=labelsize
            )
            ax[i].set_title(f'Mode {i}', fontsize=fontsize)
        
        ax[i].get_yaxis().set_visible(False)
        ax[i].tick_params(axis='both', which='major', labelsize=labelsize)
        ax[i].set_xticks(np.linspace(0, fmi_array_one_meter_zone.shape[1], 5))
    plt.tight_layout()
    if save:
        plt.savefig(
            pjoin(save_path, f'fmi_contours_from_diff_thresholds_{depth}{custom_fname}{well_name}.'+picture_format), 
            format=picture_format, dpi=500, bbox_inches='tight'
        )
    if plot:
        plt.show()
    else:
        plt.close()

def plot_contour_after_laplacian_filtering(
        fmi_array_one_meter_zone, filtered_contour, depth, save_path, save=False, 
        depth_in_name=False, picture_format='png', custom_fname='',
        colorbar=True, labelsize=20, fontsize=20, plot=True, well_name='',
        figsize=(15, 7)
):
    """
    Function to plot the contour after applying laplacian filtering
    """
    if depth_in_name:
        depth = str(round(depth, 2))
    else:
        depth = ''
    if custom_fname:
        custom_fname = f'_{custom_fname}'
    if well_name:
        well_name = f'_{well_name}'
    _, ax = plt.subplots(1, 2, figsize=figsize)
    im = ax[0].imshow(fmi_array_one_meter_zone, cmap='YlOrBr')
    if colorbar:
        create_colorbar(ax[0], im, fontsize=labelsize)
    plot_single_image_contours(
        fmi_array_one_meter_zone, filtered_contour, ax[1], linewidth=2, cmap='YlOrBr',
        colorbar=colorbar, fontsize=fontsize, labelsize=labelsize
    )
    ax[0].get_yaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)

    ax[0].tick_params(axis='both', which='major', labelsize=labelsize)
    ax[1].tick_params(axis='both', which='major', labelsize=labelsize)

    ax[0].set_xticks(np.linspace(0, fmi_array_one_meter_zone.shape[1], 5))
    ax[1].set_xticks(np.linspace(0, fmi_array_one_meter_zone.shape[1], 5))

    ax[0].set_title('Original FMI', fontsize=fontsize)
    ax[1].set_title('Laplacian Filtering', fontsize=fontsize)
    plt.tight_layout()
    if save:
        plt.savefig(
            pjoin(save_path, f'7_fmi_laplacian_filtered_{depth}{custom_fname}{well_name}.'+picture_format), 
            format=picture_format, dpi=500, bbox_inches='tight'
        )
    if plot:
        plt.show()
    else:
        plt.close()

def plot_contour_after_mean_filtering(
        fmi_array_one_meter_zone, filtered_contour, depth, save_path, save=False, 
        depth_in_name=False, picture_format='png', custom_fname='',
        colorbar=True, labelsize=20, fontsize=20, plot=True, well_name='',
        figsize=(15, 7)
):
    """
    Function to plot the contour after applying mean filtering
    """
    if depth_in_name:
        depth = str(round(depth, 2))
    else:
        depth = ''
    if custom_fname:
        custom_fname = f'_{custom_fname}'
    if well_name:
        well_name = f'_{well_name}'
    _, ax = plt.subplots(1, 2, figsize=figsize)
    im = ax[0].imshow(fmi_array_one_meter_zone, cmap='YlOrBr')
    if colorbar:
        create_colorbar(ax[0], im, fontsize=labelsize)
    plot_single_image_contours(
        fmi_array_one_meter_zone, filtered_contour, ax[1], linewidth=2, cmap='YlOrBr',
        colorbar=colorbar, fontsize=fontsize, labelsize=labelsize
    )
    ax[0].get_yaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[0].tick_params(axis='both', which='major', labelsize=labelsize)
    ax[1].tick_params(axis='both', which='major', labelsize=labelsize)
    ax[0].set_xticks(np.linspace(0, fmi_array_one_meter_zone.shape[1], 5))
    ax[1].set_xticks(np.linspace(0, fmi_array_one_meter_zone.shape[1], 5))
    ax[0].set_title('Original FMI', fontsize=fontsize)
    ax[1].set_title('Mean Based Filtering', fontsize=fontsize)
    plt.tight_layout()
    if save:
        plt.savefig(
            pjoin(save_path, f'8_fmi_mean_filtered_{depth}{custom_fname}{well_name}.'+picture_format), 
            format=picture_format, dpi=500, bbox_inches='tight'
        )
    if plot:
        plt.show()
    else:
        plt.close()

def round_to_ceil_nearest(number, nearest=5):
    return nearest * ((number + nearest - 1) // nearest)

def comparison_plot(
        fmi_array_one_meter_zone, filtered_contour, pred_df, gt_df, depth, save_path, 
        save=False, depth_in_name=False, picture_format='png', custom_fname='',
        colorbar=True, labelsize=20, fontsize=20, plot=True, max_bar_scale=25, well_name='',
        depth_in_yaxis=True
):
    """
    Function to plot the comparison between the predicted and ground truth vugs
    """
    if depth_in_name:
        depth = str(round(depth, 2))
    else:
        depth = ''
    if custom_fname:
        custom_fname = f'_{custom_fname}'
    if well_name:
        well_name = f'_{well_name}'
    _, ax = plt.subplots(1, 5, figsize=(16, 6), gridspec_kw = {'width_ratios': [5,5,1,1,1], 'height_ratios': [1]})
    im = ax[0].imshow(fmi_array_one_meter_zone, cmap='YlOrBr')
    if colorbar:
        create_colorbar(ax[0], im, fontsize=labelsize)
    if depth_in_yaxis:
        ax[0].set_yticks(np.linspace(0, fmi_array_one_meter_zone.shape[0], 11))
        y_labels = [f'{x:.1f}' for x in np.linspace(gt_df.Depth.min(), gt_df.Depth.max()+0.1, 11)]
        ax[0].set_yticklabels([f'XXX{i[3:]}' for i in y_labels])
        ax[0].tick_params(axis='both', which='major', labelsize=labelsize)
        
    plot_single_image_contours(
        fmi_array_one_meter_zone, filtered_contour, ax[1], linewidth=2, cmap='YlOrBr',
        colorbar=colorbar, fontsize=fontsize, labelsize=labelsize
    )
    if not depth_in_yaxis:
        ax[0].get_yaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[0].set_xticks(np.linspace(0, fmi_array_one_meter_zone.shape[1], 5))
    ax[1].set_xticks(np.linspace(0, fmi_array_one_meter_zone.shape[1], 5))
    ax[0].tick_params(axis='both', which='major', labelsize=labelsize)
    ax[1].tick_params(axis='both', which='major', labelsize=labelsize)
    ax[0].set_title('Original FMI', fontsize=fontsize)
    ax[1].set_title('Final Extracted Vugs', fontsize=fontsize)
    plot_barh(
        ax[2], pred_df.Depth.values, pred_df['Vugs'].values, pred_df.Depth.values[0], 
        pred_df.Depth.values[-1], "Estimated\n%", max_scale=max_bar_scale, fontsize = fontsize, yticks=False
    )
        
    plot_barh(
        ax[3], gt_df.Depth.values, gt_df['Vugs'].values, pred_df.Depth.values[0], 
        pred_df.Depth.values[-1], "WellCAD\n%", max_scale=max_bar_scale, fontsize = fontsize, yticks=False
    )

    # # Scatter plot with improved aesthetics
    # ax[4].scatter(gt_df.Vugs, pred_df.Vugs, color='blue', edgecolor='black', s=50, alpha=0.7, label='Vugs')
    # # Add y=x reference line
    # max_val = max(gt_df.Vugs.max(), pred_df.Vugs.max())
    # ax[4].plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='y=x')
    # # Setting labels with enhanced font size
    # ax[4].set_xlabel('WellCAD Vugs (%)', fontsize=fontsize)
    # ax[4].set_ylabel('Estimated Vugs (%)', fontsize=fontsize)
    # # Tick parameters for readability
    # ax[4].tick_params(axis='both', which='major', labelsize=fontsize)
    # # Add grid and legend
    # ax[4].grid(True, linestyle='--', alpha=0.6)

    plot_vug_comparison(
        ax[4], gt_df, pred_df, gt_df.Depth.min(), gt_df.Depth.max(), fontsize=20, labelsize=20, vline=5, title='Comparison'
    )

    plt.tight_layout()
    if save:
        plt.savefig(
            pjoin(save_path, f'9_fmi_vugs_comparison_{depth}{custom_fname}{well_name}.'+picture_format), format=picture_format, 
            dpi=500, bbox_inches='tight'
        )
    if plot:
        plt.show()
    else:
        plt.close()

def plot_vug_comparison(
        ax, gt_df, pred_df, one_meter_zone_start, 
        one_meter_zone_end, fontsize=20, labelsize=20, vline=5,
        title=False, legend=False
):
    """
    Plot the comparison between the ground truth and estimated vugs.

    Parameters:
    ax (matplotlib.axes.Axes): The axes on which to plot the comparison.
    gt_df (pandas.DataFrame): DataFrame containing the ground truth vugs with columns 'Depth' and 'Vugs'.
    pred_df (pandas.DataFrame): DataFrame containing the estimated vugs with columns 'Depth' and 'Vugs'.
    one_meter_zone_start (int): The start depth of the one meter zone.
    one_meter_zone_end (int): The end depth of the one meter zone.
    fontsize (int, optional): Font size for the plot titles and labels. Default is 20.
    labelsize (int, optional): Label size for the plot ticks. Default is 20.
    vline (int, optional): Interval for vertical dashed lines. Default is 5.
    """
    ax.plot(gt_df.Vugs.values, gt_df.Depth.values, 'r-', label='WellCAD')
    ax.plot(pred_df.Vugs.values, pred_df.Depth.values, 'g-', label='Estimated')
    if title:
        ax.set_title('Comparison', fontsize=fontsize)
    ax.set_yticks([])
    if legend:
        ax.legend(loc='upper right', fontsize=fontsize-12)
    ax.set_ylim([gt_df.Depth.min(), gt_df.Depth.max()])
    ax.invert_yaxis()
    ax.tick_params(axis='x', which='major', labelsize=labelsize)
    
    for i in range(vline, int(round_to_ceil_nearest(max(gt_df.Vugs.max(), pred_df.Vugs.max()), vline)), vline):
        ax.vlines(
            i, one_meter_zone_start, one_meter_zone_end, colors='k', 
            linestyles='dashed', linewidth=1
        )

def plot_barh(
        ax, y, x, one_meter_zone_start, one_meter_zone_end, title, max_scale=25, fontsize = 8, 
        colors='k', linestyles='dashed', linewidth=1, vlines=5, yticks=True
):
    """
    Plots the barh plot of the detected and GT vugs

    Parameters
    ----------

    ax: matplotlib axis
        axis on which the plot is to be plotted
    y: numpy array
        y axis of the plot
    x: numpy array
        x axis of the plot
    one_meter_zone_start: int
        start of the one meter zone
    one_meter_zone_end: int
        end of the one meter zone
    title: str
        title of the plot
    max_scale: int
        max scale of the plot
    fontsize: int
        fontsize of the plot
    colors: str
        color of the plot
    linestyles: str
        linestyles of the plot
    linewidth: int
        linewidth of the plot

    Returns
    -------
    None
    """
    ax.barh(y, x, align='center', height=0.08)
    for i in range(vlines, 100, vlines):
        ax.vlines(
            i, one_meter_zone_start, one_meter_zone_end, colors=colors, 
            linestyles=linestyles, linewidth=linewidth
        )
    ax.invert_yaxis()
    ax.tick_params(axis='x', labelsize=fontsize)

    if yticks:
        ax.tick_params(axis='y', labelsize=fontsize)
    else:
        ax.set_yticks([])
    
    ax.set_xlim(0, max_scale)
    ax.set_ylim(one_meter_zone_end, one_meter_zone_start)

    ax.set_xticks([0, max_scale])

    if title:
        ax.set_title(title, fontsize = fontsize)

def plot_single_image(img, fpath, cmap='YlOrBr'):
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(fpath, dpi=500, bbox_inches='tight')
    plt.close()

def get_and_plot_final_vugs(
        fmi_array_doi, fmi_array_doi_unscaled, tdep_array_doi, gt, pred_df, contour_x, contour_y, vug_area_list, 
        vugs_circulairty_list, zone_start, zone_end, img_idx, save_path, save = False, 
        fontsize=20, depth_in_name=False, picture_format='png', custom_fname='', well_name=''
):
    from utils.contours import get_centeroid
    from utils.vugs import get_vugs_final_info
    '''
    Get the final vugs information and plot the final vugs information for the zone

    1. Get the data (fmi, pred_df, gt) for the zone
    2. Get the image height
    3. Create a scaler to scale the centroid_y to the zone_start and zone_end range
    4. Filter the contours based on the height of the image and get the centroids of the vugs
    5. Plot the final vugs information for the zone

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
    well_radius : numpy array
        Well radius
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
    zone_start : int
        Start depth of the zone
    zone_end : int
        End depth of the zone
    img_idx : int
        Image index
    save_path : str
        Path to save the final vugs information plot
    fontsize : int, optional
        Fontsize for the plot, by default 20

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
    '''
    if depth_in_name:
        depth = f"{round(zone_start, 2)}-{round(zone_end, 2)}"
    else:
        depth = ''
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
    # Basically, earlier we converted the contour to the whole FMI data level, now we are extracting the contours that are within the zone
    coord = [[k, j] for i, (k, j) in enumerate(zip(contour_x, contour_y)) if (j.min()>=img_idx) & (j.max()<=(img_idx+img_height))]

    _, ax = plt.subplots(1, 8, figsize=(20, 27), gridspec_kw = {'width_ratios': [4,4,1,1,1,1,1,1], 'height_ratios': [1]})
    ax[0].imshow(fmi_zone, cmap='YlOrBr')
    ax[1].imshow(fmi_zone, cmap='YlOrBr')
    centroid_list = []
    vugs_area_list_filtered = []
    vugs_circulairty_list_filtered = []
    contour_x_list, contour_y_list = [], []
    for (x_, y_), vugs_area_individual, vugs_circularity_individual in zip(coord, vug_area_list, vugs_circulairty_list):
        # get the y-coordinate of the centroid of the contour
        y_new = y_-img_idx
        centroid_y = get_centeroid(np.concatenate([x_.reshape(-1, 1), (y_new).reshape(-1, 1)], axis=1))[1]
        # rescale the centroid_y to the zone_start and zone_end range
        centroid_depth = scaler.transform([[centroid_y]])[0][0]
        # get the index where the centroid_depth should be inserted
        depth_values, target_value = pred_df_zone.Depth.values, centroid_depth
        # Find the index where the target_value should be inserted
        insert_index = np.searchsorted(depth_values, target_value, side='right') - 1

        # Check if the target_value is greater than the last depth value, in that case, it will be inserted at the end
        if insert_index == len(depth_values) - 1 and target_value > depth_values[-1]:
            insert_index = len(depth_values) - 1
        if pred_df_zone.iloc[insert_index].Vugs != 0:
            ax[1].plot(x_, y_new, color='black', linewidth=2)
            centroid_list.append(target_value)
            vugs_area_list_filtered.append(vugs_area_individual)
            vugs_circulairty_list_filtered.append(vugs_circularity_individual)
            contour_x_list.append(x_)
            contour_y_list.append(y_new)

    ax[1].set_yticks([])
    ax[0].set_yticks([])

    vugs_info_final = get_vugs_final_info(
        pred_df_zone, centroid_list, vugs_area_list_filtered, vugs_circulairty_list_filtered
    )

    plot_barh(
        ax[2], pred_df_zone.Depth.values, pred_df_zone['Vugs'].values, zone_start, zone_end-0.1, 
        "Pred%", max_scale=15, fontsize = fontsize, yticks=False
    )
    plot_barh(
        ax[3], gt_zone.Depth.values, gt_zone['Vugs'].values, zone_start, zone_end-0.1, 
        "GT%", max_scale=25, fontsize = fontsize, yticks=False
    )
    plot_barh(
        ax[4], vugs_info_final.Depth.values, vugs_info_final['TotalCounts'].values, zone_start, zone_end-0.1, 
        "Count", max_scale=50, fontsize = fontsize, vlines=10, yticks=False
    )
    plot_barh(
        ax[5], vugs_info_final.Depth.values, vugs_info_final['VugsArea'].values, zone_start, zone_end-0.1, 
        "Area", max_scale=70, fontsize = fontsize, vlines=10, yticks=False
    )
    plot_barh(
        ax[6], vugs_info_final.Depth.values, vugs_info_final['MeanArea'].values,zone_start, zone_end-0.1, 
        "Mean A.", max_scale=10, fontsize = fontsize, vlines=2, yticks=False
    )
    plot_barh(
        ax[7], vugs_info_final.Depth.values, vugs_info_final['StdArea'].values,zone_start, zone_end-0.1, 
        "S.Dev. A.", max_scale=10, fontsize = fontsize, vlines=2, yticks=False
    )

    ax[0].set_title("Original FMI", fontsize=fontsize)
    ax[1].set_title("Detected Vugs", fontsize=fontsize)

    ax[0].set_xticks(np.linspace(0, fmi_zone.shape[1], 5))
    ax[1].set_xticks(np.linspace(0, fmi_zone.shape[1], 5))

    ax[0].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[1].tick_params(axis='both', which='major', labelsize=fontsize)

    if save:
        plt.savefig(
            pjoin(save_path, f'10_fmi_final_{depth}{custom_fname}{well_name}.'+picture_format), format=picture_format, 
            dpi=500, bbox_inches='tight'
        )
    plt.show()
    img_idx+=img_height

    return img_idx, vugs_info_final, fmi_zone, fmi_zone_unscaled, pred_df_zone, gt_zone

def plot_contours(circles, ax, linewidth=2):
    """Plot contours on ax
    
    Parameters
    ----------
    circles : list
        List of contours
    ax : matplotlib.axes._subplots.AxesSubplot
        Axes to plot on

    Returns
    -------
    None
    """
    for pts in circles:
        x = pts[:, 0, 0]
        y = pts[:, 0, 1]
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        ax.plot(x, y, color='black', linewidth=linewidth)

def turn_off_particular_subplot(ax, which='all'):
    """Turn off particular subplot

    Parameters
    ----------
    ax: matplotlib axes
    which: str
        'all': turn off all
        'x': turn off x axis
        'y': turn off y axis
    """
    if which == 'all':
        ax.set_axis_off()
    elif which == 'x':
        ax.xaxis.set_visible(False)
    elif which == 'y':
        ax.yaxis.set_visible(False)
    else:
        raise ValueError("which should be 'all', 'x' or 'y'")

def plot_distribution_of_fmi(
        fmi_array_one_meter_zone, fmi_array_doi, one_meter_zone_start, mean, std, 
        var, skewness, kurtosis_val, fig_size=(15, 2),save = True, custom_fname=''
):
    """
    Plots the distribution of FMI

    Parameters
    ----------

    fmi_array_one_meter_zone : numpy array
        FMI array of one meter zone
    fmi_array_doi : numpy array
        FMI array of depth of interest
    one_meter_zone_start : int
        Start of the one meter zone
    mean : float
        Mean of the FMI array
    std : float
        Standard deviation of the FMI array
    var : float
        Variance of the FMI array
    skewness : float
        Skewness of the FMI array
    kurtosis_val : float
        Kurtosis of the FMI array
    fig_size : tuple
        Figure size
        
    Returns
    -------

    None

    """
    plt.figure(figsize=fig_size)
    sns.histplot(fmi_array_one_meter_zone.reshape(-1))
    plt.title(f"Mean: {mean}, Std: {std}, Var: {var}, Skewness: {skewness}, Kurtosis: {kurtosis_val}")
    plt.xlim(np.nanmin(fmi_array_doi), np.nanmax(fmi_array_doi))
    plt.tight_layout()
    if save:
        plt.savefig(f"hist/{one_meter_zone_start}{custom_fname}.png", dpi=400)
        plt.close()
    else:
        plt.show()

def plot_threshold_axis(thresold_img, ax, title, fontsize = 8):
    """Plot the threshold image"""
    ax.imshow(thresold_img, cmap='gray')
    ax.set_title(title, fontsize = fontsize)

def plot_original_image_axis(image, start, end, ax, title, fontsize = 8, confidential = True, colorbar = True, labelsize=8):
    """Plot the original image"""
    im = ax.imshow(image, cmap='YlOrBr')
    ax.set_title(title, fontsize = fontsize)
    yticks_new = np.linspace(start, end, 10).round(2)
    ax.set_yticks(np.linspace(0, image.shape[0], 10).round(2), yticks_new)
    if confidential:
        yticks_new_str = ['X'*2+str(i)[2:] for i in yticks_new]
        ax.set_yticklabels(yticks_new_str)

    if colorbar:
        create_colorbar(ax, im, fontsize=labelsize)
    

def plot_contour_axis(image, contours, title, ax, fontsize = 8, linewidth = 2):
    """
    Plot the contours on the image
    """
    ax.imshow(image, cmap='YlOrBr')
    ax.set_title(title, fontsize = fontsize)
    plot_contours(contours, ax, linewidth = linewidth)

def turn_off_all_the_ticks(ax, except_ax):
    """Turn off all the ticks of the axis"""
    
    for i in range(len(ax)):
        if ax[i]!=except_ax:
            ax[i].set_xticks([])
            if i!=0:
                ax[i].set_yticks([])


def plot_contours_info_from_multiple_modes_line_plot(
        x1, x2, x1_new_contours, x1_duplicate_contours, x2_new_contours, x2_duplicate_contours, depth, save_path, 
        save = False, depth_in_name=False, picture_format='png', custom_fname='', well_name='', fontsize=20
):
    if depth_in_name:
        depth = str(round(depth, 2))
    else:
        depth = ''
    if custom_fname:
        custom_fname = f'_{custom_fname}'
    if well_name:
        well_name = f'_{well_name}'
    plt.figure(figsize=(20, 5))

    plt.plot(x1, x1_new_contours, 'red')
    plt.plot(x1, x1_duplicate_contours, '--r')
    plt.plot(x2, x2_new_contours, 'green', marker='*', markersize=15)
    plt.plot(x2, x2_duplicate_contours, '--g', marker='*', markersize=15)

    plt.grid()
    plt.xlabel('Mode of Interest', fontsize=fontsize)
    plt.ylabel('Contour Counts', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if save: 
        plt.savefig(
            pjoin(save_path, f'contour_counts_comparison_line_plot_{depth}{custom_fname}{well_name}.'+picture_format),
            format=picture_format, dpi=500, bbox_inches='tight'
        )
    plt.show()

def plot_contours_info_from_multiple_modes_bar_plot(
        x1, x2, x1_new_contours, x1_duplicate_contours, x2_new_contours, x2_duplicate_contours, different_thresholds, max_k_selected, 
        depth, save_path, bar_width_double=0.45, offset=0.0, fontsize=30, very_bad_color='#B22222', bad_color='#FF6347', 
        good_color='#FF9999', very_good_color='#228B22', text_color='#000000', arrow_color_ok='#006400', arrow_color_not_ok='#FF0000', 
        figsize=(30, 10), save = False, depth_in_name=False, picture_format='png', custom_fname='', ylim = 362, well_name=''
):
    if depth_in_name:
        depth = str(round(depth, 2))
    else:
        depth = ''
    if custom_fname:
        custom_fname = f'_{custom_fname}'
    if well_name:
        well_name = f'_{well_name}'
    fig, ax = plt.subplots(figsize=figsize)

    accepted_mode_number = 0
    # Lists to store handles and labels for legends
    handles_1, labels_1, handles_2, labels_2 = [], [], [], []

    for i, (x, moi) in enumerate(zip(x1, different_thresholds)):
        if x in x2:
            # If both bars are present, position them with spacing using the offset
            idx = np.where(x2 == x)[0][0]

            # Shifted positions with spacing
            bar1 = ax.bar(
                x - bar_width_double / 2 - offset, x1_duplicate_contours[i], 
                width=bar_width_double, color=very_bad_color
            )
            bar2 = ax.bar(
                x - bar_width_double / 2 - offset, x1_new_contours[i], width=bar_width_double, 
                bottom=x1_duplicate_contours[i], color=bad_color
            )

            bar3 = ax.bar(
                x + bar_width_double / 2 + offset, x2_duplicate_contours[idx], 
                width=bar_width_double, color=good_color
            )
            bar4 = ax.bar(
                x + bar_width_double / 2 + offset, x2_new_contours[idx], width=bar_width_double, 
                bottom=x2_duplicate_contours[idx], color=very_good_color
            )

            # Add handles and labels for the first legend
            if i == 0:
                handles_1.append(bar1[0])
                labels_1.append('Discarded Modes - Duplicate Contours')
                handles_1.append(bar2[0])
                labels_1.append('Discarded Modes - New Contours')

            # Add handles and labels for the second legend
            if idx == 0:
                handles_2.append(bar3[0])
                labels_2.append('Selected Modes - Duplicate Contours')
                handles_2.append(bar4[0])
                labels_2.append('Selected Modes - New Contours')

            total_contours = x2_duplicate_contours[idx] + x2_new_contours[idx]
            accepted_mode_number += 1
            if accepted_mode_number <= max_k_selected:
                arrow_color = arrow_color_ok
                ax.text(x - 0.15, total_contours + 25 + 2, accepted_mode_number, fontsize=fontsize, color=arrow_color, fontweight='bold')
            else:
                arrow_color = arrow_color_not_ok
            ax.arrow(x, total_contours + 25, 0, -10, head_width=0.3, head_length=15, fc=arrow_color, ec=arrow_color)
        else:
            # If only k=46, stride=1 bar is present, use full width (1.0) and center it
            width = 0.9
            bar1 = ax.bar(x, x1_duplicate_contours[i], width=width, color=very_bad_color)
            bar2 = ax.bar(x, x1_new_contours[i], width=width, bottom=x1_duplicate_contours[i], color=bad_color)

        ax.text(x - 0.2, 50, f"{moi:.2f}", fontsize=fontsize, rotation=90, color=text_color, fontweight='bold')

    # Labeling and formatting
    ax.set_xlabel('Mode of Interest', fontsize=fontsize)
    ax.set_ylabel('Contour Counts', fontsize=fontsize)
    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=fontsize)

    # Create legends at two different locations
    legend1 = ax.legend(handles_1, labels_1, fontsize=fontsize, loc='upper left')  # First two legends
    ax.add_artist(legend1)  # Add the first legend

    legend2 = ax.legend(handles_2, labels_2, fontsize=fontsize, loc='upper right')  # Last two legends
    ax.add_artist(legend2)  # Add the second legend

    ax.grid(True, axis='y')

    ax.set_ylim(0, ylim)
    plt.tight_layout()
    if save:
        plt.savefig(
            pjoin(save_path, f'contour_counts_comparison_bar_plot_{depth}{custom_fname}{well_name}.'+picture_format), 
            format=picture_format, dpi=500, bbox_inches='tight'
        )
    plt.show()

def plot_mode_of_interest_frequency_histogram(
        different_thresholds, count_not_ok, count_ok, stride, save_path, save, depth,
        depth_in_name, picture_format, dpi=500, custom_fname = '', figsize=(15, 5), fontsize=20, well_name=''
):
    if depth_in_name:
        depth = str(round(depth, 2))
    else:
        depth = ''
    if custom_fname:
        custom_fname = f'_{custom_fname}'
    if well_name:
        well_name = f'_{well_name}'
    
    plt.figure(figsize=figsize)
    x = range(len(count_not_ok))
    plt.bar(x, count_not_ok, color='red')
    plt.bar(x[::stride], count_ok, color='green')
    plt.xticks(x)
    for _x, moi, cnt in zip(x, different_thresholds, count_not_ok):
        plt.text(_x-0.15, (cnt/5)*2, str(moi), fontsize=fontsize, rotation=90)
    plt.xlabel('Mode of Interest', fontsize=fontsize)
    plt.ylabel('Counts/Frequency', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks([])
    plt.legend(['Discarded Modes', 'Selected Modes'], fontsize=fontsize)
    plt.grid(axis = 'y')
    plt.tight_layout()
    if save:
        plt.savefig(
            pjoin(save_path, f'mode_of_interest_frequency_histogram_{depth}{custom_fname}{well_name}.'+picture_format), 
            format=picture_format, dpi=dpi, bbox_inches='tight'
        )
    plt.show()

def plot_heatmap(
        image, heatmap, contour_pre, contour_post, vmax, depth, threshold, cmap='plasma',
        fontsize=20, labelsize=20, nrows=1, ncols=5, figsize=(20, 5), save=False, save_path='.', 
        custom_fname='', picture_format='png', dpi=500, depth_in_name=False, well_name='', 
        mud_type=None, laplacian_heatmap=True, epsilon = 0.01, axis_off=False
):
    '''
    This function plots the original FMI image, the contrast heatmap, the overlayed contrast heatmap,
    the overlayed contrast heatmap with no filtering, the overlayed contrast heatmap with filtering,
    and the filtered contours. The function is used to visualize the contrast heatmap and the contours
    before and after filtering.

    Parameters:
    -----------
    image: np.ndarray
        The original FMI image.
    heatmap: np.ndarray
        The contrast heatmap.
    cmap: str
        The colormap to use for the contrast heatmap.
    contour_pre: list
        The contours before filtering.
    contour_post: list
        The contours after filtering.
    vmax: float
        The maximum value for the contrast heatmap.
    depth: int
        The depth of the FMI image.

    Returns:
    --------
    None
    '''
    from utils.plotting import plot_image, plot_single_image_contours, create_custom_cmap

    if depth_in_name:
        depth = str(round(depth, 2))
    else:
        depth = ''
    if custom_fname:
        custom_fname = f'_{custom_fname}'
    if well_name:
        well_name = f'_{well_name}'

    # Set up figure with specified width and height ratios
    fig, ax = plt.subplots(1, 5, figsize=figsize, gridspec_kw={'width_ratios': [1]*5})

    # Display the original FMI image with a colorbar
    plot_image(
        image, 'YlOrBr', vmax=None, ax=ax[0], title='Original FMI', 
        colorbar=True, fontsize=fontsize, labelsize=labelsize, axis_off=axis_off
    )
    ax[0].set_xticks(np.linspace(0, image.shape[1], 5), fontsize=fontsize)
    ax[0].set_yticks([])

    if laplacian_heatmap:
        axis_2_title = 'Combined Contour'
        axis_3_title = 'Laplacian Variance Heatmap'
        axis_4_title = 'Thresholded Heatmap'
        axis_5_title = 'Laplacian Filtering'
    else:
        axis_2_title = 'Laplacian Filtering'
        axis_3_title = 'Mean Based Heatmap'
        axis_4_title = 'Thresholded Heatmap'
        axis_5_title = 'Mean Based Filtering'

    # Display unfiltered contours with overlay
    plot_single_image_contours(
        image, contour_pre, ax[1], linewidth=1, colorbar=True, 
        title=axis_2_title, fontsize=fontsize, labelsize=labelsize
    )
    ax[1].set_xticks(np.linspace(0, image.shape[1], 5), fontsize=fontsize)
    ax[1].set_yticks([])

    # Display the contrast heatmap with a colorbar
    plot_image(
        heatmap, cmap, vmax, ax[2], title=axis_3_title, 
        colorbar=True, fontsize=fontsize, labelsize=labelsize,
        axis_off=axis_off
    )
    ax[2].set_xticks(np.linspace(0, image.shape[1], 5), fontsize=fontsize)
    ax[2].set_yticks([])

    # Display the contrast heatmap with a colorbar
    if laplacian_heatmap:
        custom_cmap = create_custom_cmap(
            base_cmap=None, custom_color_picker = None, 
            threshold = threshold, max_value = np.nanmax(heatmap),
            min_value=None, mudtype=None, laplacian_heatmap=True
        )
    else:
        custom_cmap = create_custom_cmap(
            base_cmap=None, custom_color_picker = None, 
            threshold = threshold, min_value=np.nanmin(heatmap), 
            max_value=np.nanmax(heatmap), mudtype=mud_type, 
            laplacian_heatmap=False, epsilon = epsilon
        )
        
    plot_image(
        heatmap, custom_cmap, None, ax[3], title=axis_4_title,
        colorbar=True, fontsize=fontsize, labelsize=labelsize, axis_off=axis_off
    )
    ax[3].set_xticks(np.linspace(0, image.shape[1], 5), fontsize=fontsize)
    ax[3].set_yticks([])

    ax[0].tick_params(axis='both', which='major', labelsize=labelsize)
    ax[1].tick_params(axis='both', which='major', labelsize=labelsize)
    ax[2].tick_params(axis='both', which='major', labelsize=labelsize)
    ax[3].tick_params(axis='both', which='major', labelsize=labelsize)

    # Display filtered contours with overlay
    plot_single_image_contours(
        image, contour_post, ax[4], linewidth=1, colorbar=True, 
        title=axis_5_title, fontsize=fontsize, labelsize=labelsize
    )
    ax[4].set_xticks(np.linspace(0, image.shape[1], 5), fontsize=fontsize)
    ax[4].set_yticks([])

    plt.tight_layout()
    if save:
        if laplacian_heatmap:
            prefix = 'contrast_heatmap_'
        else:
            prefix = 'mean_based_heatmap_'
        plt.savefig(
            pjoin(save_path, f'{prefix}{depth}{custom_fname}{well_name}.'+picture_format),
            format=picture_format, dpi=dpi, bbox_inches='tight'
        )
    plt.show()

def plot_heatmap_v0(
        image, heatmap, contour_pre, contour_post, vmax, depth, threshold, cmap='plasma',
        fontsize=20, labelsize=20, nrows=1, ncols=7, figsize=(20, 5), save=False, save_path='.', 
        custom_fname='', picture_format='png', dpi=500, depth_in_name=False, well_name='', 
        mud_type=None, laplacian_heatmap=True, epsilon = 0.01, axis_off=False
):
    '''
    This function plots the original FMI image, the contrast heatmap, the overlayed contrast heatmap,
    the overlayed contrast heatmap with no filtering, the overlayed contrast heatmap with filtering,
    and the filtered contours. The function is used to visualize the contrast heatmap and the contours
    before and after filtering.

    Parameters:
    -----------
    image: np.ndarray
        The original FMI image.
    heatmap: np.ndarray
        The contrast heatmap.
    cmap: str
        The colormap to use for the contrast heatmap.
    contour_pre: list
        The contours before filtering.
    contour_post: list
        The contours after filtering.
    vmax: float
        The maximum value for the contrast heatmap.
    depth: int
        The depth of the FMI image.

    Returns:
    --------
    None
    '''
    if depth_in_name:
        depth = str(round(depth, 2))
    else:
        depth = ''
    if custom_fname:
        custom_fname = f'_{custom_fname}'
    if well_name:
        well_name = f'_{well_name}'

    # Set up figure with specified width and height ratios
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, gridspec_kw={'width_ratios': [1]*ncols})

    # Display the original FMI image with a colorbar
    plot_image(
        image, 'YlOrBr', vmax=None, ax=ax[0], title='Original FMI', 
        colorbar=True, fontsize=fontsize, labelsize=labelsize, axis_off=axis_off
    )
    ax[0].set_xticks(np.linspace(0, image.shape[1], 5), fontsize=fontsize)
    ax[0].set_yticks([])

    if laplacian_heatmap:
        axis_2_title = 'Laplacian Variance Heatmap'
        axis_3_title = 'Thresholded Heatmap'
        axis_4_title = 'Overlayed Heatmap'
        axis_5_title = 'Overlayed Heatmap\nCombined'
        axis_6_title = 'Overlayed Heatmap\nLaplacian Filtered'
        axis_7_title = 'Filtered Contours'
    else:
        axis_2_title = 'Mean Based Heatmap'
        axis_3_title = 'Thresholded Heatmap'
        axis_4_title = 'Overlayed Heatmap'
        axis_5_title = 'Overlayed Heatmap\nLaplacian Filtered'
        axis_6_title = 'Overlayed Heatmap\nMean Filtered'
        axis_7_title = 'Filtered Contours'

    # Display the contrast heatmap with a colorbar
    plot_image(
        heatmap, cmap, vmax, ax[1], title=axis_2_title, 
        colorbar=True, fontsize=fontsize, labelsize=labelsize,
        axis_off=axis_off
    )
    ax[1].set_xticks(np.linspace(0, image.shape[1], 5), fontsize=fontsize)
    ax[1].set_yticks([])

    # Display the contrast heatmap with a colorbar
    if laplacian_heatmap:
        custom_cmap = create_custom_cmap(
            base_cmap=None, custom_color_picker = None, 
            threshold = threshold, max_value = np.nanmax(heatmap),
            min_value=None, mudtype=None, laplacian_heatmap=True
        )
    else:
        custom_cmap = create_custom_cmap(
            base_cmap=None, custom_color_picker = None, 
            threshold = threshold, min_value=np.nanmin(heatmap), 
            max_value=np.nanmax(heatmap), mudtype=mud_type, 
            laplacian_heatmap=False, epsilon = epsilon
        )
        
    plot_image(
        heatmap, custom_cmap, None, ax[2], title=axis_3_title,
        colorbar=True, fontsize=fontsize, labelsize=labelsize, axis_off=axis_off
    )
    ax[2].set_xticks(np.linspace(0, image.shape[1], 5), fontsize=fontsize)
    ax[2].set_yticks([])

    # Overlay contrast heatmap on the third subplot
    plot_image(
        image, 'YlOrBr', vmax=None, ax=ax[3], title=axis_4_title, 
        colorbar=True, fontsize=fontsize, labelsize=labelsize, axis_off=axis_off
    )
    ax[3].imshow(
        heatmap, cmap=cmap, vmax=vmax, alpha=0.3
    )
    ax[3].set_xticks(np.linspace(0, image.shape[1], 5), fontsize=fontsize)
    ax[3].set_yticks([])

    ax[0].tick_params(axis='both', which='major', labelsize=labelsize)
    ax[1].tick_params(axis='both', which='major', labelsize=labelsize)
    ax[2].tick_params(axis='both', which='major', labelsize=labelsize)
    ax[3].tick_params(axis='both', which='major', labelsize=labelsize)

    if contour_pre and contour_post:
        # Display unfiltered contours with overlay
        plot_single_image_contours(
            image, contour_pre, ax[4], linewidth=1, colorbar=True, 
            title=axis_5_title, fontsize=fontsize, labelsize=labelsize
        )
        ax[4].imshow(
            heatmap, cmap=cmap, alpha=0.5, vmax=vmax
        )
        ax[4].set_xticks(np.linspace(0, image.shape[1], 5), fontsize=fontsize)
        ax[4].set_yticks([])

        # Display filtered contours with overlay
        plot_single_image_contours(
            image, contour_post, ax[5], linewidth=1, colorbar=True, 
            title=axis_6_title, fontsize=fontsize, labelsize=labelsize
        )
        ax[5].imshow(
            heatmap, cmap=cmap, alpha=0.5, vmax=vmax
        )
        ax[5].set_xticks(np.linspace(0, image.shape[1], 5), fontsize=fontsize)
        ax[5].set_yticks([])

        # Display only filtered contours
        plot_single_image_contours(
            image, contour_post, ax[6], linewidth=1, colorbar=True, 
            title=axis_7_title, fontsize=fontsize, labelsize=labelsize
        )
        ax[6].set_xticks(np.linspace(0, image.shape[1], 5), fontsize=fontsize)
        ax[6].set_yticks([])

    plt.tight_layout()
    if save:
        if laplacian_heatmap:
            prefix = 'contrast_heatmap_'
        else:
            prefix = 'mean_based_heatmap_'
        plt.savefig(
            pjoin(save_path, f'{prefix}{depth}{custom_fname}{well_name}.'+picture_format),
            format=picture_format, dpi=dpi, bbox_inches='tight'
        )
    plt.show()

def plot_violine_boxplot_histogram_for_heatmap(
        values_to_plot, depth, cfg, figsize=(15, 7), violin_color="lightgrey", 
        box_color="deepskyblue", line_color="red", fontsize=15, labelsize=12, 
        save=False, save_path=".", custom_fname="", dpi=300, picture_format="png", well_name='',
        depth_in_name=False
):
    '''
    This function plots a violin plot, a box plot, and a histogram for the contrast values
    extracted from the heatmap. The function is used to visualize the distribution of the contrast
    values.

    Parameters:
    -----------
    values_to_plot: list
        The list of contrast values to plot.
    depth: int
        The depth of the contrast values.
    cfg: OmegaConf
        The configuration object.

    Returns:
    --------
    None
    '''
    if depth_in_name:
        depth = str(round(depth, 2))
    else:
        depth = ''
    if custom_fname:
        custom_fname = f'_{custom_fname}'
    if well_name:
        well_name = f'_{well_name}'

    # Initialize the plot with a larger figure size for clarity
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # First subplot: Violin + Box Plot with horizontal orientation
    sns.violinplot(
        data=values_to_plot, color=violin_color, orient="h", ax=ax[0]
    )
    sns.boxplot(
        data=values_to_plot, color=box_color, width=0.2, orient="h", ax=ax[0]
    )
    ax[0].vlines(
        cfg.filter_cutoffs.LAPLACIAN_VARIANCE, -0.4, 0.4, color=line_color, linestyles="--", linewidth=1.5
    )
    ax[0].set_xlabel("Contrast Value (Log Scale)", fontsize=fontsize)
    ax[0].set_xscale('log')
    ax[0].tick_params(axis="both", which="major", labelsize=labelsize)

    # Second subplot: Histogram with KDE
    histogram_plot(
        values_to_plot, cfg.filter_cutoffs.LAPLACIAN_VARIANCE, ax[1], 
        title='', xlim=(0, 2), kde=False
    )
    ax[1].set_xlabel("Contrast Value", fontsize=fontsize)
    ax[1].set_ylabel("Frequency", fontsize=fontsize)
    ax[1].tick_params(axis="both", which="major", labelsize=labelsize)

    # Final layout adjustments for a clean presentation
    plt.tight_layout()
    if save:
        plt.savefig(
            pjoin(save_path, f'contrast_values_hist_violine_box_{depth}{custom_fname}{well_name}.'+picture_format),
            format=picture_format, dpi=dpi, bbox_inches='tight'
        )
    plt.show()

def set_axis_style(ax, not_last_axis=False):
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    if not_last_axis:
        ax.set_xticks([])

def plot_vugs_statistics(
        pred_df_zone, gt_zone, vugs_info_final, zone_start, zone_end, 
        axises, titles, max_bar_scales, vlines, fontsize, minor_zone_length=0.1
):
    from utils.plotting import plot_barh, plot_vug_comparison

    plot_barh(
        axises['estimated'], pred_df_zone.Depth.values, pred_df_zone['Vugs'].values, 
        zone_start, zone_end-minor_zone_length, titles['estimated'], max_scale=max_bar_scales['estimated'],
        fontsize = fontsize, yticks=False, vlines=vlines['estimated']
    )

    plot_barh(
        axises['wellcad'], gt_zone.Depth.values, gt_zone['Vugs'].values, 
        zone_start, zone_end-minor_zone_length, titles['wellcad'], max_scale=max_bar_scales['wellcad'],
        fontsize = fontsize, yticks=False, vlines=vlines['wellcad']
    )

    plot_vug_comparison(
        axises['comp'], gt_zone, pred_df_zone, gt_zone.Depth.min(), 
        gt_zone.Depth.max(), fontsize=20, labelsize=20, vline=5
    )

    plot_barh(
        axises['count'], vugs_info_final.Depth.values, vugs_info_final['TotalCounts'].values,
        zone_start, zone_end-minor_zone_length, titles['count'], max_scale=max_bar_scales['count'],
        fontsize = fontsize, yticks=False, vlines=vlines['count']
    )

    plot_barh(
        axises['area'], vugs_info_final.Depth.values, vugs_info_final['VugsArea'].values, 
        zone_start, zone_end-minor_zone_length, titles['area'], max_scale=max_bar_scales['area'],
        fontsize = fontsize, yticks=False, vlines=vlines['area']
    )

    plot_barh(
        axises['mean'], vugs_info_final.Depth.values, vugs_info_final['MeanArea'].values,zone_start, 
        zone_end-minor_zone_length, titles['mean'], max_scale=max_bar_scales['mean'],
        fontsize = fontsize, yticks=False, vlines=vlines['mean']
    )

    plot_barh(
        axises['std'], vugs_info_final.Depth.values, vugs_info_final['StdArea'].values,zone_start, 
        zone_end-minor_zone_length, titles['std'], max_scale=max_bar_scales['std'],
        fontsize = fontsize, yticks=False, vlines=vlines['std']
    )

# Function to plot KDE with dynamic azimuth divisions
def plot_vug_azimuthal_presence_kde(counts, ax, xticks_fontsize=15, xticks_rotation=0):
    '''
    This function plots the KDE plot of the azimuthal presence of the vugs.
    '''
    from operator import add
    # Create division indices (1 to number of divisions)
    divisions = np.arange(1, len(counts) + 1)
    xticks = np.arange(0.5, len(counts) + 1 + 0.5)
    
    # Generate azimuth ranges dynamically
    azimuth_division_points = np.linspace(0, 360, len(counts) + 1).astype(int)

    # Repeat the division index according to the count
    data = np.repeat(divisions, counts)
    if len(np.unique(data))==1:
        jitter = (abs(np.random.randn(len(data)))*0.01).tolist()
        data = list(map(add, data, jitter))

    # Plot the KDE plot
    sns.kdeplot(data, fill=True, bw_adjust=0.5, ax=ax)

    # Customize x-ticks to show azimuth ranges
    ax.set_xticks(ticks=xticks, labels=azimuth_division_points, fontsize=xticks_fontsize, rotation=xticks_rotation)
    

    # Add vertical lines at azimuth division boundaries
    for boundary in divisions[:-1]:  # Skip the last division (360°)
        ax.axvline(x=boundary + 0.5, color='red', linestyle='-', alpha=0.7)

    # Set x-axis limits based on the number of divisions
    ax.set_xlim(0.5, len(counts) + 0.5)  # Extending a bit beyond division indices for better visualization

def set_axis_style(ax, not_last_axis=False):
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    if not_last_axis:
        ax.set_xticks([])

def plot_vugs_spectrum(
        vugs_info_final, spectrum_grid, area_xlim, 
        circularity_xlim, fig, azimuthal_divisions=4,
        fontsize=25
):  
    from utils.vugs import get_vugs_spectrum_data
    from matplotlib.ticker import FormatStrFormatter

    azimuth_division_points = np.linspace(0, 360, azimuthal_divisions + 1)

    area_grid = spectrum_grid['area']
    circularity_grid = spectrum_grid['circularity']
    azimuth_grid = spectrum_grid['azimuth']

    for i in range(vugs_info_final.shape[0]):
        outputs = get_vugs_spectrum_data(
            vugs_info_final, i, azimuth_division_points, azimuthal_divisions=azimuthal_divisions
        )
        vugs_area_zone, vugs_circularity_zone, azimuthal_counts = outputs

        area_ax = fig.add_subplot(area_grid[i, 0])
        circularity_ax = fig.add_subplot(circularity_grid[i, 0])
        azimuth_ax = fig.add_subplot(azimuth_grid[i, 0])

        sns.kdeplot(vugs_area_zone, ax=area_ax, fill=True)
        sns.kdeplot(vugs_circularity_zone, ax=circularity_ax, fill=True)
        plot_vug_azimuthal_presence_kde(azimuthal_counts, azimuth_ax)

        not_last_axis = i!=vugs_info_final.shape[0]-1
        set_axis_style(area_ax, not_last_axis)
        set_axis_style(circularity_ax, not_last_axis)
        set_axis_style(azimuth_ax, not_last_axis)


        area_ax.set_xlim(area_xlim[0], area_xlim[1])
        circularity_ax.set_xlim(circularity_xlim[0], circularity_xlim[1])

        if not not_last_axis:
            area_ax.set_xticks(np.linspace(area_xlim[0], area_xlim[1], 4))
            area_ax.tick_params(axis='x', labelsize=fontsize)

            circularity_ax.set_xticks(np.linspace(circularity_xlim[0], circularity_xlim[1], 4))
            circularity_ax.tick_params(axis='x', labelsize=fontsize)
            circularity_ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            # azimuth_ax.set_xticks(np.linspace(0, 360, 4))
            azimuth_ax.tick_params(axis='x', labelsize=fontsize)
            

def get_axis_for_plotting(main_grid, vugs_info_final, fig):
    '''
    This function creates the axes for the main images plot, spectrum plots, statistical plots and overall statistics plots.
    '''
    # axes for the three main images
    fmi_ax = fig.add_subplot(main_grid[0, 0])
    contour_ax = fig.add_subplot(main_grid[0, 1])
    mask_ax = fig.add_subplot(main_grid[0, 2])

    # Create a nested GridSpec within the last main column for 10 rows
    area_grid = main_grid[0, 3].subgridspec(vugs_info_final.shape[0], 1, hspace=0.2)
    circularity_grid = main_grid[0, 4].subgridspec(vugs_info_final.shape[0], 1, hspace=0.2)
    azimuth_grid = main_grid[0, 5].subgridspec(vugs_info_final.shape[0], 1, hspace=0.2)

    spectrum_grid = {
        'area': area_grid,
        'circularity': circularity_grid,
        'azimuth': azimuth_grid
    }

    # Get axes for the statistical plots
    estimated_ax = fig.add_subplot(main_grid[0, 6])
    wellcad_ax = fig.add_subplot(main_grid[0, 7])
    comp_ax = fig.add_subplot(main_grid[0, 8])
    count_ax = fig.add_subplot(main_grid[0, 9])
    area_ax = fig.add_subplot(main_grid[0, 10])
    mean_area_ax = fig.add_subplot(main_grid[0, 11])
    std_area_ax = fig.add_subplot(main_grid[0, 12])

    statistical_axises = {
        'estimated': estimated_ax,
        'wellcad': wellcad_ax,
        'comp': comp_ax,
        'count': count_ax,
        'area': area_ax,
        'mean': mean_area_ax,
        'std': std_area_ax
    }

    return (
        fmi_ax,
        contour_ax,
        mask_ax,
        spectrum_grid,
        statistical_axises,
    )

def plot_vugs_circularity_ratio_distribution(
        circularity_ratio, zone_start, zone_end, depth_in_name=True, 
        save=False, save_path='.', picture_format='png',
        fontsize=20, add_labelling=True
):
    if depth_in_name:
        depth = f"{round(zone_start, 2)}-{round(zone_end, 2)}"
    else:
        depth = ''
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(circularity_ratio, bins=30, kde=True)
    ax.lines[0].set_color('red')
    if add_labelling:
        plt.xlabel('Vugs Circularity Ratio', fontsize=fontsize)
        plt.ylabel('Vugs Count', fontsize=fontsize)
        plt.title('Vugs Circularity Ratio Distribution', fontsize=fontsize)
    else:
        plt.ylabel('')  # Remove the y-axis label
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if save:
        plt.savefig(
            pjoin(save_path, f'11_vugs_circularity_ratio_distribution_{depth}.'+picture_format), 
            format=picture_format, dpi=500, bbox_inches='tight'
        )
    plt.show()


def plot_vugs_area_distribution(
        contour_area, zone_start, zone_end, depth_in_name=True, 
        save=False, save_path='.', picture_format='png',
        fontsize=20, add_labelling=True
):
    if depth_in_name:
        depth = f"{round(zone_start, 2)}-{round(zone_end, 2)}"
    else:
        depth = ''

    plt.figure(figsize=(10, 6))
    ax = sns.histplot(contour_area, bins=30, kde=True)
    ax.lines[0].set_color('red')
    if add_labelling:
        plt.xlabel('Vugs Area (cm$^2$)', fontsize=fontsize)
        plt.ylabel('Vugs Count', fontsize=fontsize)
        plt.title('Vugs Area Distribution', fontsize=fontsize)
    else:
        plt.ylabel('')  # Remove the y-axis label
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if save:
        plt.savefig(
            pjoin(save_path, f'12_vugs_area_distribution_{depth}.'+picture_format), 
            format=picture_format, dpi=500, bbox_inches='tight'
        )
    plt.show()

def plot_fmi_vs_vugs_histogram_scaled(
    fmi_zone, mask, zone_start, zone_end, save=False, 
    save_path='.', picture_format='png', depth_in_name=False, fontsize = 20
):
    if depth_in_name:
        depth = f"{round(zone_start, 2)}-{round(zone_end, 2)}"
    else:
        depth = ''

    plt.figure(figsize=(10, 6))
    sns.histplot(fmi_zone.reshape(-1), label='FMI', kde=True)
    sns.histplot(fmi_zone[mask.astype(bool)].reshape(-1), label='Vugs', kde=True)
    plt.xlabel('Static FMI Log Values', fontsize=fontsize)
    plt.ylabel('Count/Frequency', fontsize=fontsize)
    plt.title('FMI Distribution', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if save:
        plt.savefig(
            pjoin(save_path, f'13_fmi_vs_vugs_histogram_scaled_{depth}.'+picture_format), 
            format=picture_format, dpi=500, bbox_inches='tight'
        )
    plt.show()

def plot_fmi_vs_vugs_histogram_unscaled(
    fmi_zone_unscaled, mask, zone_start, zone_end, depth_in_name=False, 
    save=False, save_path='.', picture_format='png', fontsize=20
):
    if depth_in_name:
        depth = f"{round(zone_start, 2)}-{round(zone_end, 2)}"
    else:
        depth = ''

    plt.figure(figsize=(10, 6))
    sns.histplot(fmi_zone_unscaled.reshape(-1), label='FMI', kde=True)
    sns.histplot(fmi_zone_unscaled[mask.astype(bool)].reshape(-1), label='Vugs', kde=True)
    plt.xlabel('Static FMI Log Values', fontsize=fontsize)
    plt.ylabel('Count/Frequency', fontsize=fontsize)
    plt.title('FMI Distribution', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if save:
        plt.savefig(
            pjoin(save_path, f'14_fmi_vs_vugs_histogram_unscaled_{depth}.'+picture_format), 
            format=picture_format, dpi=500, bbox_inches='tight'
        )
    plt.show()

def plot_fmi_vs_vugs_kde_unscaled(
    fmi_zone_unscaled, mask, zone_start, zone_end, save=False, 
    save_path=None, picture_format='png', depth_in_name=False,
    fontsize=20, add_labelling=True, image_log_type='FMI'
):
    if depth_in_name:
        depth = f"{round(zone_start, 2)}-{round(zone_end, 2)}"
    else:
        depth = ''

    plt.figure(figsize=(10, 6))
    sns.kdeplot(fmi_zone_unscaled.reshape(-1), label=image_log_type, fill=True)
    sns.kdeplot(fmi_zone_unscaled[mask.astype(bool)].reshape(-1), label='Vugs', fill=True)
    plt.legend(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if add_labelling:
        plt.title('FMI Distribution', fontsize=fontsize)
        plt.ylabel('Density/Frequency', fontsize=fontsize)
        plt.xlabel('Static FMI Log Values', fontsize=fontsize)
    else:
        plt.ylabel('')  # Remove the y-axis label
    if save:
        plt.savefig(
            pjoin(save_path, f'15_fmi_vs_vugs_kde_unscaled_{depth}.'+picture_format), 
            format=picture_format, dpi=500, bbox_inches='tight'
        )
    plt.show()

def plot_fmi_vs_vugs_kde_scaled(
    fmi_zone, mask, zone_start, zone_end, save=False, 
    save_path=None, picture_format='png', depth_in_name=False, fontsize=20
):
    if depth_in_name:
        depth = f"{round(zone_start, 2)}-{round(zone_end, 2)}"
    else:
        depth = ''

    plt.figure(figsize=(10, 6))
    sns.kdeplot(fmi_zone.reshape(-1), label='FMI', fill=True)
    sns.kdeplot(fmi_zone[mask.astype(bool)].reshape(-1), label='Vugs', fill=True)
    plt.legend(fontsize=fontsize)
    plt.title('FMI Distribution', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('Density/Frequency', fontsize=fontsize)
    plt.xlabel('Static FMI Log Values', fontsize=fontsize)
    if save:
        plt.savefig(
            pjoin(save_path, f'16_fmi_vs_vugs_kde_scaled_{depth}.'+picture_format), 
            format=picture_format, dpi=500, bbox_inches='tight'
        )
    plt.show()