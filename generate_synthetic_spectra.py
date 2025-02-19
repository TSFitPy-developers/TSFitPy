from __future__ import annotations

import logging
import pickle
import sys
from configparser import ConfigParser

from scripts.dask_client import get_dask_client
from scripts.turbospectrum_class_nlte import TurboSpectrum
from scripts.synthetic_code_class import fetch_marcs_grid
from scripts.convolve import *
import datetime
from scripts.create_window_linelist_function import create_window_linelist
import shutil
from dask.distributed import Client
import socket
import os
from scripts.run_wrapper import run_and_save_wrapper
from time import perf_counter
from scripts.loading_configs import SpectraParameters
from scripts.solar_abundances import periodic_table, solar_abundances
from scripts.auxiliary_functions import calculate_vturb
from scripts.loading_configs import SyntheticSpectraConfig

if __name__ == '__main__':
    # load config file from command line
    today = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  # used to not conflict with other instances of fits
    today = f"{today}_{np.random.random(1)[0]}"

    if len(sys.argv) == 1:
        config_file = "input_files/synthetic_spectra_generation_configuration.cfg"
    else:
        config_file = sys.argv[1]
    config_synthetic_spectra = SyntheticSpectraConfig(config_file, today)
    config_synthetic_spectra.load_config()
    config_synthetic_spectra.validate_input()

    # if debug_mode is >= 1, then change logging level to debug
    if config_synthetic_spectra.debug_mode >= 1:
        logging.basicConfig(level=logging.DEBUG)
    if config_synthetic_spectra.debug_mode >= 2:
        verbose = True
    else:
        verbose = False

    if config_synthetic_spectra.compiler.lower() == "m3dis":
        m3dis_flag = True
    else:
        m3dis_flag = False

    spectra_parameters_class = SpectraParameters(config_synthetic_spectra.input_parameter_path, first_row_name=False)
    spectra_parameters = spectra_parameters_class.get_spectra_parameters_for_grid_generation()

    logging.debug(f"Input parameters: \n{spectra_parameters_class}")

    output_dir = config_synthetic_spectra.output_folder_path
    os.makedirs(output_dir)

    print(f"Output directory: {output_dir}")
    print(f"Input parameters file: {config_synthetic_spectra.input_parameter_path}")
    # print when the grid generation started with the current time and date
    print(f"Grid generation started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # load NLTE data
    nlte_config = ConfigParser()
    nlte_config.read(config_synthetic_spectra.departure_file_config_path)

    depart_bin_file_dict, depart_aux_file_dict, model_atom_file_dict = {}, {}, {}

    for element in config_synthetic_spectra.nlte_elements:
        if config_synthetic_spectra.atmosphere_type == "1D":
            bin_config_name, aux_config_name = "1d_bin", "1d_aux"
        else:
            bin_config_name, aux_config_name = "3d_bin", "3d_aux"
        depart_bin_file_dict[element] = nlte_config[element][bin_config_name]
        depart_aux_file_dict[element] = nlte_config[element][aux_config_name]
        model_atom_file_dict[element] = nlte_config[element]["atom_file"]

    depart_bin_file, depart_aux_file, model_atom_file = depart_bin_file_dict, depart_aux_file_dict, model_atom_file_dict

    model_temperatures, model_logs, model_mets, marcs_value_keys, marcs_models, marcs_values = fetch_marcs_grid(config_synthetic_spectra.model_atmosphere_list, TurboSpectrum.marcs_parameters_to_ignore)
    aux_file_length_dict = {}
    if config_synthetic_spectra.nlte_flag:
        for element in model_atom_file:
            aux_file_length_dict[element] = len(np.loadtxt(os.path.join(config_synthetic_spectra.departure_file_path, depart_aux_file[element]), dtype='str'))

    line_list_path_trimmed = os.path.join(config_synthetic_spectra.temporary_directory_path, "linelist_for_fitting_trimmed", "")
    line_list_path_trimmed = os.path.join(line_list_path_trimmed, "all", today, '')

    print("Trimming")
    include_molecules = config_synthetic_spectra.include_molecules
    create_window_linelist([config_synthetic_spectra.wavelength_min - 5], [config_synthetic_spectra.wavelength_max + 5], config_synthetic_spectra.line_list_path, line_list_path_trimmed, include_molecules, False)
    print("trimming done")

    line_list_path_trimmed = os.path.join(line_list_path_trimmed, "0", "")

    logging.debug(config_synthetic_spectra.__dict__)

    # create mupoint_path
    mupoint_path = os.path.join(config_synthetic_spectra.temporary_directory_path, "mupoints.dat")
    if config_synthetic_spectra.compute_intensity_flag:
        print(f"IMPORTANT!! Intensity calculations require TS to be updated to version v20.1")
        with open(mupoint_path, "w") as f:
            f.write(f"{len(config_synthetic_spectra.intensity_angles)}\n")
            # write all angles, separated by , and with a space at the end
            f.write(f"{', '.join([str(i) for i in config_synthetic_spectra.intensity_angles])} ")

    ts_config = {"turbospec_path": config_synthetic_spectra.spectral_code_path,
                 "interpol_path": config_synthetic_spectra.interpolators_path,
                 "line_list_paths": line_list_path_trimmed,
                 "model_atmosphere_grid_path": config_synthetic_spectra.model_atmosphere_grid_path,
                 "model_atmosphere_grid_list": config_synthetic_spectra.model_atmosphere_list,
                 "model_atom_path": config_synthetic_spectra.model_atoms_path,
                 "model_temperatures": model_temperatures,
                 "model_logs": model_logs,
                 "model_mets": model_mets,
                 "marcs_value_keys": marcs_value_keys,
                 "marcs_models": marcs_models,
                 "marcs_values": marcs_values,
                 "aux_file_length_dict": aux_file_length_dict,
                 "departure_file_path": config_synthetic_spectra.departure_file_path,
                 "atmosphere_type": config_synthetic_spectra.atmosphere_type,
                 "windows_flag": False,
                 "segment_file": None,
                 "line_mask_file": None,
                 "depart_bin_file": depart_bin_file,
                 "depart_aux_file": depart_aux_file,
                 "model_atom_file": model_atom_file,
                 "global_temporary_directory": config_synthetic_spectra.temporary_directory_path,
                 "compute_intensity_flag": config_synthetic_spectra.compute_intensity_flag,
                 "mupoint_path": mupoint_path,
                 "m3dis_flag": m3dis_flag}

    with open(os.path.join(config_synthetic_spectra.temporary_directory_path, "tsfitpy_configuration.pkl"), "wb") as f:
        pickle.dump(ts_config, f)
    tsfitpy_pickled_configuration_path = os.path.join(config_synthetic_spectra.temporary_directory_path, "tsfitpy_configuration.pkl")

    # time to run the code
    time_start = perf_counter()


    client = get_dask_client(config_synthetic_spectra.cluster_type, config_synthetic_spectra.cluster_name,
                             config_synthetic_spectra.number_of_cpus, nodes=config_synthetic_spectra.number_of_nodes,
                             slurm_script_commands=config_synthetic_spectra.script_commands,
                             slurm_memory_per_core=config_synthetic_spectra.memory_per_cpu_gb,
                             time_limit_hours=config_synthetic_spectra.time_limit_hours,
                             slurm_partition=config_synthetic_spectra.slurm_partition)

    temp_df = spectra_parameters_class.spectra_parameters_df.copy()

    # in spectra_parameters_class.spectra_parameters_df change the column names and add .spec to the specname
    temp_df["specname"] = temp_df["specname"].apply(lambda x: f"{x}.spec")

    # if vmic is not in the columns, add it by calculating using the function calculate_vturb
    if "vmic" not in temp_df.columns:
        temp_df["vmic"] = temp_df.apply(lambda x: calculate_vturb(x["teff"], x["logg"], x["feh"]), axis=1)

    # change the columns names for elements in df from X to X_Fe
    # go through columns in the df
    for column in temp_df.columns:
        # if the column name is in the list of elements
        if column in periodic_table:
            # add _Fe to the column name
            temp_df.rename(columns={column: f"{column}_Fe"}, inplace=True)

    # if column Fe_Fe is present, then rename it to A(Fe)
    if "Fe_Fe" in temp_df.columns:
        temp_df.rename(columns={"Fe_Fe": "A(Fe)"}, inplace=True)
        # and change its value by adding FeH and solar abundance of Fe
        temp_df["A(Fe)"] = temp_df["A(Fe)"] + temp_df["feh"] + solar_abundances["Fe"]

    # save the spectra parameters
    temp_df.to_csv(os.path.join(output_dir, "spectra_parameters_temp.csv"), index=False)

    futures = []
    for one_spectra_parameter in spectra_parameters:
        specname, teff, logg, feh, vmic, vmac, rotation, abundances_dict = one_spectra_parameter
        spectrum_name = f"{specname}.spec"
        if config_synthetic_spectra.number_of_cpus > 1:
            future = client.submit(run_and_save_wrapper, tsfitpy_pickled_configuration_path, teff, logg, feh, config_synthetic_spectra.wavelength_min,
                                   config_synthetic_spectra.wavelength_max, config_synthetic_spectra.wavelength_delta,
                                   spectrum_name, config_synthetic_spectra.nlte_flag, config_synthetic_spectra.resolution, vmac, rotation, output_dir, vmic, abundances_dict, config_synthetic_spectra.save_unnormalised_spectra, verbose, config_synthetic_spectra.lpoint_turbospectrum)
            futures.append(future)  # prepares to get values
        else:
            future = run_and_save_wrapper(tsfitpy_pickled_configuration_path, teff, logg, feh, config_synthetic_spectra.wavelength_min,
                                 config_synthetic_spectra.wavelength_max, config_synthetic_spectra.wavelength_delta,
                                 spectrum_name, config_synthetic_spectra.nlte_flag, config_synthetic_spectra.resolution, vmac, rotation, output_dir, vmic, abundances_dict, config_synthetic_spectra.save_unnormalised_spectra, verbose, config_synthetic_spectra.lpoint_turbospectrum)
            futures.append(future)

    if config_synthetic_spectra.number_of_cpus > 1:
        print("Start gathering")  # use http://localhost:8787/status to check status. the port might be different
        futures = np.array(client.gather(futures))  # starts the calculations (takes a long time here)
        results = futures
        print("Worker calculation done")  # when done, save values
    else:
        results = futures

    # in spectra_parameters_class.spectra_parameters_df change the column names and add .spec to the specname
    spectra_parameters_class.spectra_parameters_df["specname"] = spectra_parameters_class.spectra_parameters_df["specname"].apply(lambda x: f"{x}.spec")

    # delete any specname columns that are not in the results
    spectra_parameters_class.spectra_parameters_df = spectra_parameters_class.spectra_parameters_df[spectra_parameters_class.spectra_parameters_df["specname"].isin(results)]

    # if vmic is not in the columns, add it by calculating using the function calculate_vturb
    if "vmic" not in spectra_parameters_class.spectra_parameters_df.columns:
        spectra_parameters_class.spectra_parameters_df["vmic"] = spectra_parameters_class.spectra_parameters_df.apply(lambda x: calculate_vturb(x["teff"], x["logg"], x["feh"]), axis=1)

    # change the columns names for elements in df from X to X_Fe
    # go through columns in the df
    for column in spectra_parameters_class.spectra_parameters_df.columns:
        # if the column name is in the list of elements
        if column in periodic_table:
            # add _Fe to the column name
            spectra_parameters_class.spectra_parameters_df.rename(columns={column: f"{column}_Fe"}, inplace=True)

    # if column Fe_Fe is present, then rename it to A(Fe)
    if "Fe_Fe" in spectra_parameters_class.spectra_parameters_df.columns:
        spectra_parameters_class.spectra_parameters_df.rename(columns={"Fe_Fe": "A(Fe)"}, inplace=True)
        # and change its value by adding FeH and solar abundance of Fe
        spectra_parameters_class.spectra_parameters_df["A(Fe)"] = spectra_parameters_class.spectra_parameters_df["A(Fe)"] + spectra_parameters_class.spectra_parameters_df["feh"] + solar_abundances["Fe"]

    # save the spectra parameters 
    spectra_parameters_class.spectra_parameters_df.to_csv(os.path.join(output_dir, "spectra_parameters.csv"), index=False)

    # delete temporary os.path.join(output_dir, "spectra_parameters_temp.csv")
    try:
        os.remove(os.path.join(output_dir, "spectra_parameters_temp.csv"))
    except FileNotFoundError:
        pass

    time_end = perf_counter()
    # with 4 decimals, the time is converted to hours, minutes and seconds
    print(f"Time elapsed: {datetime.timedelta(seconds=time_end - time_start)}")

    # print ending date and time
    print("Ending date and time: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    shutil.rmtree(line_list_path_trimmed)  # clean up trimmed line list