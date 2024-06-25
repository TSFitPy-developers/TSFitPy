from __future__ import annotations
from scripts.loading_configs import TSFitPyConfig, SpectraParameters
import numpy as np
from scripts.create_window_linelist_function import create_window_linelist
import shutil, os, sys
from dask.distributed import Client
from scripts.dask_client import get_dask_client
import logging
import pandas as pd


def fit_one_spectra():
    pass


def run_tsfitpy(output_folder, config_file_location):
    parsed_config_file = TSFitPyConfig(config_file_location)
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

    if debug_mode_python:
        logging.basicConfig(level=logging.DEBUG)

    logging.debug(f"Running TSFitPy with the following configuration: {config}")

    # step 1: load linemasks
    # step 2: load elements
    linemasks = os.path.join(config.linemasks_path, config.linemask_file)
    elements = config.elements_to_fit

    all_lines_center = []
    all_lines_begin = []
    all_lines_end = []
    all_lines_elements = []

    for linemask_index, linemask in enumerate([linemasks]):
        logging.debug(f"Loading linemask {linemask}")
        line_wavelength, line_begin, line_end = np.loadtxt(linemask, comments=";", usecols=(0, 1, 2), unpack=True, dtype=float)
        all_lines_center.extend(list(line_wavelength))
        all_lines_begin.extend(list(line_begin))
        all_lines_end.extend(list(line_end))
        all_lines_elements.extend([elements[linemask_index]] * len(line_wavelength))

    original_linelist_path = config.line_list_path

    if config.pretrim_linelist:
        logging.debug("Pretrimming linelist")
        linelist_pretrimmed_path = os.path.join(config.temporary_directory_path, "linelist_pretrimmed", "")
        create_window_linelist(all_lines_begin, all_lines_end, config.line_list_path, linelist_pretrimmed_path, config.include_molecules, True, True)
        logging.debug(f"New linelist paths: {linelist_pretrimmed_path}")
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

    spectra_parameters = SpectraParameters(config.fitlist_input_path, True)
    logging.debug(f"Running TSFitPy with the following parameters: {spectra_parameters}")

    final_results = pd.DataFrame()

    spectra_params_to_fit = spectra_parameters.get_spectra_parameters_for_fit(True, True, True)

    for spectra_to_fit in spectra_parameters.spectra_to_fit:
        logging.debug(f"Fitting spectra {spectra_to_fit}")
        fit_one_spectra()

    for element_index, element in enumerate(elements):
        for linemask_index, linemask in enumerate(linemasks):
            pass



    if config.pretrim_linelist:
        for new_linelist_path in new_linelist_paths:
            logging.debug(f"Removing linelist {new_linelist_path}")
            #shutil.rmtree(new_linelist_path, ignore_errors=True)






if __name__ == '__main__':
    raise RuntimeError("This file is not meant to be run as main. Please run TSFitPy/main.py instead.")  # this is a module
