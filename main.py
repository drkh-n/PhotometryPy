import argparse
import os
import csv
from astropy.io import fits
from astropy.wcs import WCS

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from psf import PSF
from idl_circapphot import circ_apphot

def get_PSF(psf_file_path, channel=1):
    """
    Reads a PSF FITS file and returns the PSF image as a numpy array.

    Parameters
    ----------
    psf_file_path : str
        The file path to the PSF FITS file.
    channel : int
        The channel number (1, 2, 3, or 4).

    Returns
    -------
    np.ndarray
        The PSF image data.
    """
    try:
        with fits.open(psf_file_path) as hdul:
            psf_data = hdul[0].data
            return psf_data
    except FileNotFoundError:
        raise FileNotFoundError(f"PSF file not found at {psf_file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading PSF FITS file: {e}")

def get_Image(image_file_path):
    """
    Reads an astronomical image FITS file and returns the image data as a numpy array.

    Parameters
    ----------
    image_file_path : str
        The file path to the image FITS file.

    Returns
    -------
    np.ndarray
        The image data.
    """
    try:
        with fits.open(image_file_path) as hdul:
            image_data = hdul[0].data
            return image_data
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found at {image_file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading image FITS file: {e}")

def get_configs():
    """
    Placeholder function to get configuration parameters.
    In a real scenario, this could read from a config file or user input.

    Returns
    -------
    dict
        A dictionary of configuration parameters.
    """
    configs = {
        'psf_file_path': 'apex_sh_IRAC1_col129_row129_x100.fits',
        'name': '1E2259+586',
        'image_file_path': '1e2259+586/1e2259+586/mosaici1/Combine/mosaic.fits',
        'channel': [1],
        'ap_radius': 2.0,
        'inner_ann_radius': 2.2,
        'outer_ann_radius': 3.0,
        'x_coord': 345.28455,
        'y_coord': 58.854317,
        'norms': [1.0, 5.0, 10.0, 20.0, 30.0],
        'spacing': 2.828,
        'grid': 3,
        'save_results': 'photometry_results.csv'
    }
    return configs

def place_PSF(image_data, processed_psf_image, x, y):
    image_shape = image_data.shape
    source_x, source_y = int(x), int(y)
    psf_half_size = processed_psf_image.shape[0] // 2
    x_start = max(source_x - psf_half_size, 0)
    x_end = min(source_x + psf_half_size + 1, image_shape[1])
    y_start = max(source_y - psf_half_size, 0)
    y_end = min(source_y + psf_half_size + 1, image_shape[0])
    psf_x_start = max(psf_half_size - source_x, 0)
    psf_x_end = psf_x_start + (x_end - x_start)
    psf_y_start = max(psf_half_size - source_y, 0)
    psf_y_end = psf_y_start + (y_end - y_start)
    image_data[y_start:y_end, x_start:x_end] += processed_psf_image[psf_y_start:psf_y_end, psf_x_start:psf_x_end]
    print(f"Simulated image updated with PSF at ({source_x}, {source_y}).")
    return image_data

def make_PSF(psf_data, x, y, channel, norm=1.0):
    # Initialize PSF object
    psf_obj = PSF(psf_data, channel=channel) # Assuming channel 1, adjust if needed

    # PSF Pipeline (example operations, adjust as per your requirements)
    print(f"Original PSF shape: {psf_obj.get_psf_image().shape}")

    # Example: Trim PSF
    trim_pixels = 51
    try:
        psf_obj.trim_prf(trim_pixels)
        print(f"PSF shape after trimming by {trim_pixels} pixels: {psf_obj.get_psf_image().shape}")
    except ValueError as e:
        print(f"Error trimming PSF: {e}")
        return
    
    new_dimensions = psf_obj.compute_dimensions()
    print(f"Computed new dimensions for PSF: {new_dimensions}")

    # Example: Shift PSF
    frac_x, frac_y = x - int(x), y - int(y)
    psf_obj.shift_prf(frac_x, frac_y)
    print(f"PSF shifted by ({frac_x}, {frac_y}) fractional pixels.")

    # Example: Normalize PSF
    psf_obj.congrid(new_dimensions, method='linear')
    print(f"PSF resampled to shape: {psf_obj.get_psf_image().shape}")

    psf_obj.normalize_prf(norm=1.0)
    print(f"PSF normalized. Sum: {np.sum(psf_obj.get_psf_image()):.4f}")

    # Get the processed PSF image
    return psf_obj.get_psf_image()

def make_plot(simulated_image, xc, yc, ap_radius, inner_ann_radius, outer_ann_radius):
    box_size = 50
    half_box = box_size // 2
    cx, cy = int(xc), int(yc)

    y0 = max(cy - half_box, 0)
    y1 = min(cy + half_box, simulated_image.shape[0])
    x0 = max(cx - half_box, 0)
    x1 = min(cx + half_box, simulated_image.shape[1])

    sub_image = simulated_image[y0:y1, x0:x1]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(sub_image, origin='lower', cmap='gray', interpolation='nearest', vmax=0.4)
    ax.set_title(f"PRF placement ({xc:.2f}, {yc:.2f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Coordinates in the sub-image frame
    sub_cx = cx - x0
    sub_cy = cy - y0

    # Aperture (solid) and annuli (dashed)
    aperture = Circle((sub_cx, sub_cy), ap_radius, edgecolor='red', facecolor='none', lw=1.5, linestyle='-')
    ann_inner = Circle((sub_cx, sub_cy), inner_ann_radius, edgecolor='yellow', facecolor='none', lw=1.2, linestyle='--')
    ann_outer = Circle((sub_cx, sub_cy), outer_ann_radius, edgecolor='yellow', facecolor='none', lw=1.2, linestyle='--')

    ax.add_patch(aperture)
    ax.add_patch(ann_inner)
    ax.add_patch(ann_outer)

    # Mark the exact center
    ax.plot(sub_cx, sub_cy, marker='+', color='cyan', markersize=10, markeredgewidth=1.5)
    plt.tight_layout()
    plt.show()

def save_results(filename, data):
    header_list = [
    'name', 'ra', 'dec', 'channel', 'x_pos', 'y_pos', 
    'expected_flux', 'phot_flux_(µJy)', 'phot_sigma_(µJy)'
    ]

    file_exists = os.path.isfile(filename)

    # Open the file in 'append' mode. 'newline=""' is important to prevent
    # blank rows from being inserted in Windows.
    with open(filename, 'a', newline='') as csvfile:
        # Use the csv module to handle proper formatting and quoting
        writer = csv.writer(csvfile)

        # If the file is new, write the header first
        if not file_exists:
            writer.writerow(header_list)

        # Write the data row
        writer.writerow(data)
    

# ========================================================================
# Main Execution
# =======================================================================

def main(plot_enabled):

    configs = get_configs()

    # Extract configurations
    x_deg = configs['x_coord']
    y_deg = configs['y_coord']
    target_file_path = configs['image_file_path']
    psf_file_path = configs['psf_file_path']
    norms = configs['norms']
    channels = configs['channel']
    
    # ========================================================================
    # 1. Read FITS images
    wcs = WCS(target_file_path) 
    xc, yc = wcs.wcs_world2pix(x_deg, y_deg, 0)
    print(f"Target coordinates in pixels: x={xc:.2f}, y={yc:.2f}")

    try:
        psf_data = get_PSF(psf_file_path, channel=1)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        return
    
    try:
        image_data = get_Image(target_file_path)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        return

    # ========================================================================
    apcor = [1.125, 1.120, 1.135, 1.221]
    psize_asec = [1.221, 1.213, 1.222, 1.220]

    for channel in channels:
        if channel not in [1, 2, 3, 4]:
            print(f"Invalid channel: {channel}. Must be 1, 2, 3, or 4.")
            return
        
        # Aperture and annulus sizes in pixels
        ap_radius = configs['ap_radius'] * (psize_asec[channel-1] / 0.6)
        inner_ann_radius = configs['inner_ann_radius'] * (psize_asec[channel-1] / 0.6)
        outer_ann_radius = configs['outer_ann_radius'] * (psize_asec[channel-1] / 0.6)

        for norm in norms:
            for ii in range(configs['grid']):
                for jj in range(configs['grid']):
                    x_offset = (ii - 1) * configs['spacing']
                    y_offset = (jj - 1) * configs['spacing']
                    x_pos = xc + x_offset
                    y_pos = yc + y_offset
                    print(x_pos, y_pos)
                    exit()
                    # ========================================================================
                    # 2. Place PRF at specified coordinates in the image                
                    processed_psf_image = make_PSF(psf_data, x_pos, y_pos, channel, norm)
                    simulated_image = place_PSF(image_data, processed_psf_image, x_pos, y_pos)

                    # ========================================================================
                    # 3. Perform Circular Aperture Photometry
                    try:
                        result = circ_apphot(
                            simulated_image, x_pos, y_pos, ap_radius, 1.0, bgndwidth=outer_ann_radius-inner_ann_radius, 
                            quiet=False, rbackin=inner_ann_radius
                        )
                        print(f"Photometry result: {result['total_counts'] * apcor[channel-1] * 8.47}")
                        print(f"Photometric error (sigma): {result['sigma'] * apcor[channel-1] * 8.47:.4f}")
                    except Exception as e:
                        print(f"Error in aperture photometry: {e}")
                        return
                    
                    # =======================================================================
                    # 4. Save results to CSV
                    save_results(configs['save_results'],
                        [
                        configs['name'],
                        x_deg,
                        y_deg,
                        channel,
                        x_pos,
                        y_pos,
                        norm,
                        result['total_counts'] * apcor[channel-1] * 8.47,
                        result['sigma'] * apcor[channel-1] * 8.47
                        ])

    # fig, ax = plt.subplots(figsize=(6, 6))
    # im = ax.imshow(processed_psf_image, origin='lower', cmap='gray', interpolation='nearest')
    # ax.set_title(f"Processed PSF Image")
    # fig.colorbar(im, ax=ax)   
    


    # =======================================================================
    # 4. PLOTTING
    if plot_enabled:
        make_plot(simulated_image, xc, yc, ap_radius, inner_ann_radius, outer_ann_radius)


    
if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Simulate PSF placement and perform aperture photometry.")
    argparse.add_argument('--plot', action='store_true', help="Enable plotting of the results.")
    args = argparse.parse_args()
    main(args.plot)