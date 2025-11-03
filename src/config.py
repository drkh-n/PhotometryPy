import pandas as pd

def get_configs(filename='configs/config_sample.csv'):
    """
    Placeholder function to get configuration parameters.
    In a real scenario, this could read from a config file or user input.

    Returns
    -------
    dict
        A dictionary of configuration parameters.
    """
    # configs = {
    #     'psf_file_path': 'psf',
    #     'name': '1E2259+586',
    #     'image_file_path': '1e2259+586',
    #     'channels': [1],
    #     'ap_radius': 2.0,
    #     'inner_ann_radius': 2.2,
    #     'outer_ann_radius': 3.0,
    #     'x_coord': 345.28455,
    #     'y_coord': 58.854317,
    #     'norms': [1.0, 5.0, 10.0, 20.0, 30.0],
    #     'spacing': 2.828,
    #     'grid': 3,
    #     'save_results': 'test_new.csv'
    # }

    # config_test = {
    #     'psf_file_path': 'psf',
    #     'name': 'TEST',
    #     'image_file_path': 'test',
    #     'channels': [1],
    #     'ap_radius': 2.0,
    #     'inner_ann_radius': 2.2,
    #     'outer_ann_radius': 3.0,
    #     'x_coord': 63,
    #     'y_coord': 63,
    #     'norms': [1.0],
    #     'spacing': 2.828,
    #     'grid': 3,
    #     'save_results': 'test100.csv'
    # }
    
    configs = pd.read_csv(filename, comment='#', sep='\s+',
                          names=['magnetars_list_path_file','intermediate_path_file','data_path_folder','rap_(cam_pix)','rbackin_(cam_pix)','rbackout_(cam_pix)','spacing','prf_path','result_path_file','channels','grid'])

    return configs