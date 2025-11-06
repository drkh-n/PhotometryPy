import argparse
import pandas as pd



from src.config import load_yaml_config
from src.flux_snr5 import process_all_magnetars
from src.ensemble_photometry import ensemble_photometry
from src.logging_setup import configure_logging
import logging
            

# ========================================================================
# Main Execution
# =======================================================================

def main(plot_enabled , config_file='configs/default.yml', verbose=False):

    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting PhotometryPy run")
    logger.debug(f"Loading config from {config_file}")

    cfg = load_yaml_config(filename=config_file)
    logger.debug(f"Loaded configuration: {cfg.as_dict()}")
    mag_path = cfg.magnetars_list_path_file

    logger.debug(f"Reading magnetar list from {mag_path}")
    mag_list = pd.read_csv(mag_path, comment='#', sep='\s+',
                           names=['name','ra','dec'])
    logger.info(f"Found {len(mag_list['name'])} targets to process")

    for i in range(len(mag_list['name'])):        
        logger.info(f"Processing target {i+1}/{len(mag_list['name'])}: {mag_list['name'][i]}")
        mag_config = {
            'psf_file_path': cfg.prf_path,
            'name': mag_list['name'][i],
            'image_file_path': cfg.data_path_folder+'/'+mag_list['name'][i],
            'channels': cfg.channels,
            'ap_radius': cfg.rap_cam_pix,
            'inner_ann_radius': cfg.rbackin_cam_pix,
            'outer_ann_radius': cfg.rbackout_cam_pix,
            'x_coord': mag_list['ra'][i],
            'y_coord': mag_list['dec'][i],
            'spacing': cfg.spacing,
            'grid': cfg.grid,
            'intermediate_path_file' : cfg.intermediate_path_file
        }

        logger.debug(f"Target config: {mag_config}")
        ensemble_photometry(configs=mag_config)
    
    # =======================================================================
    # 7. Calculate SNR=5 fluxes from the photometry results
    logger.info("Combining results and computing SNR=5 fluxes")
    process_all_magnetars(cfg.intermediate_path_file, cfg.result_path_file)
    logger.info("PhotometryPy run completed successfully")

    
if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Simulate PSF placement and perform aperture photometry.")
    argparse.add_argument('-i', '--config', type=str, default="configs/default.yml", help="YAML config file with circapphot parameters")
    argparse.add_argument('--plot', action='store_true', help="Enable plotting of the results.")
    args = argparse.parse_args()

    main(args.plot, args.config)