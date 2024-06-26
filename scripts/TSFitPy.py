from __future__ import annotations
from scripts.loading_configs import TSFitPyConfig, SpectraParameters
import numpy as np
from scripts.create_window_linelist_function import create_window_linelist
import shutil, os, sys
from dask.distributed import Client
from scripts.dask_client import get_dask_client
import logging
import pandas as pd
from scripts.auxiliary_functions import *


def fit_one_spectra():
    pass


def run_tsfitpy(output_folder, config_file_location):
    parsed_config_file = TSFitPyConfig(config_file_location, output_folder_title=output_folder)
    parsed_config_file.load_config(True)
    run_tsfitpy_with_config(output_folder, parsed_config_file)


def run_tsfitpy_with_config(output_folder, config: TSFitPyConfig):

    debug_modes = {
        -1: {"night_mode": True, "debug_mode_fortran": False, "debug_mode_python": False},
        0: {"night_mode": False, "debug_mode_fortran": False, "debug_mode_python": False},
        1: {"night_mode": False, "debug_mode_fortran": False, "debug_mode_python": True},
        2: {"night_mode": False, "debug_mode_fortran": True, "debug_mode_python": True}
    }

    # gets the debug mode configuration, if weird value is given, it will default to 0
    mode_config = debug_modes.get(config.debug_mode, debug_modes[0])

    night_mode = mode_config["night_mode"]
    debug_mode_fortran = mode_config["debug_mode_fortran"]
    debug_mode_python = mode_config["debug_mode_python"]
    debug_mode_python = True

    if debug_mode_python:
        logging.basicConfig(level=logging.DEBUG)

    logging.debug(f"Running TSFitPy with the following configuration: {config.__dict__}")

    # step 1: load linemasks
    # step 2: load elements
    linemasks = []
    for linemask in config.linemasks_files:
        linemask_path = os.path.join(config.linemasks_path, linemask)
        linemasks.append(linemask_path)
    elements = config.elements_to_fit

    all_lines_center = []
    all_lines_begin = []
    all_lines_end = []
    all_segments_begin = []
    all_segments_end = []
    all_lines_elements = []

    for linemask_index, linemask in enumerate(linemasks):
        logging.debug(f"Loading linemask {linemask}")
        line_wavelength, line_begin, line_end = np.loadtxt(linemask, comments=";", usecols=(0, 1, 2), unpack=True, dtype=float)
        all_lines_center.extend(list(line_wavelength))
        all_lines_begin.extend(list(line_begin))
        all_lines_end.extend(list(line_end))
        all_lines_elements.extend([elements[linemask_index]] * len(line_wavelength))

    for line_begin, line_end in zip(all_lines_begin, all_lines_end):
        all_segments_begin.append(line_begin - config.segment_size)
        all_segments_end.append(line_end + config.segment_size)

    # convert all to numpy arrays
    all_lines_center = np.array(all_lines_center)
    all_lines_begin = np.array(all_lines_begin)
    all_lines_end = np.array(all_lines_end)
    all_segments_begin = np.array(all_segments_begin)
    all_segments_end = np.array(all_segments_end)

    if config.pretrim_linelist:
        logging.debug("Pretrimming linelist")
        linelist_pretrimmed_path = os.path.join(config.temporary_directory_path, "linelist_pretrimmed", "")
        if config.fitting_mode == "all":
            lbl_mode = False
        else:
            lbl_mode = True
        create_window_linelist(all_segments_begin, all_segments_end, config.line_list_path, linelist_pretrimmed_path, config.include_molecules, lbl_mode, True, folder_element_names=all_lines_elements)
        logging.debug(f"Trimming done; new linelist paths: {linelist_pretrimmed_path}")
    else:
        linelist_pretrimmed_path = config.line_list_path

    client = None

    if config.number_of_cpus != 1:
        logging.debug("Creating dask client")
        client = get_dask_client(
            client_type=config.cluster_type,
            cluster_name=config.cluster_name,
            workers_amount_cpus=config.number_of_cpus,
            night_mode=night_mode,
            nodes=config.number_of_nodes,
            slurm_script_commands=config.script_commands,
            slurm_memory_per_core=config.memory_per_cpu_gb,
            time_limit_hours=config.time_limit_hours,
            slurm_partition=config.slurm_partition
        )

    spectra_parameters = SpectraParameters(os.path.join(config.fitlist_input_path, config.input_fitlist_filename), True)
    logging.debug(f"Running TSFitPy with the following parameters: {spectra_parameters}")

    final_results = pd.DataFrame()

    spectra_params_to_fit = spectra_parameters.get_spectra_parameters_for_fit(True, True, True)

    results = []

    for spectra_to_fit in spectra_params_to_fit:
        # specname_list, rv_list, teff_list, logg_list, feh_list, vmic_list, vmac_list, rotation_list, abundance_list, resolution_list, snr_list
        specname, rv, teff, logg, feh, vmic, vmac, rotation, abundance, resolution, snr = spectra_to_fit
        logging.debug(f"Fitting spectra {specname}")

        spectra_parameters = {"specname": specname, "rv": rv, "teff": teff, "logg": logg, "feh": feh, "vmic": vmic, "vmac": vmac, "rotation": rotation, "abundance": abundance, "resolution": resolution, "snr": snr}

        try:
            wavelength_obs, flux_obs, flux_error_obs = np.loadtxt(specname, usecols=(0, 1, 2), unpack=True, dtype=float, comments="#")
        except ValueError:
            wavelength_obs, flux_obs = np.loadtxt(specname, usecols=(0, 1), unpack=True, dtype=float, comments="#")

        for idx, element, linemask_center, linemask_left, linemask_right, segment_left, segment_right in enumerate(zip(all_lines_elements, all_lines_center, all_lines_begin, all_lines_end, all_segments_begin, all_segments_end)):
            if client is not None:
                results.append(client.submit(fit_one_spectra, wavelength_obs, flux_obs, flux_error_obs, linelist_pretrimmed_path, element, linemask_center, linemask_left, linemask_right, segment_left, segment_right, spectra_parameters))
            else:
                results.append(fit_one_spectra(wavelength_obs, flux_obs, flux_error_obs, linelist_pretrimmed_path, element, linemask_center, linemask_left, linemask_right, segment_left, segment_right, spectra_parameters))

    if client is not None:
        results = client.gather(results)



    if config.pretrim_linelist:
        logging.debug(f"Removing temp directory {config.temporary_directory_path}")
        shutil.rmtree(config.temporary_directory_path, ignore_errors=True)






if __name__ == '__main__':
    raise RuntimeError("This file is not meant to be run as main. Please run TSFitPy/main.py instead.")  # this is a module
