from __future__ import annotations

import logging
import pickle
import sys
from configparser import ConfigParser

from scripts.dask_client import get_dask_client
from scripts.turbospectrum_class_nlte import TurboSpectrum, fetch_marcs_grid
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
from scripts.solar_abundances import periodic_table

class SyntheticSpectraConfig:
    def __init__(self, config_location: str, output_folder_title: str):
        self.config_parser = ConfigParser()
        self.config_location: str = config_location
        self.output_folder_title: str = output_folder_title

        self.compiler: str = None
        self.turbospectrum_path: str = None
        self.interpolators_path: str = None
        self.line_list_path: str = None
        self.model_atmosphere_grid_path_1d: str = None
        self.model_atmosphere_grid_path_3d: str = None
        self.model_atoms_path: str = None
        self.departure_file_path: str = None
        self.departure_file_config_path: str = None
        self.output_folder_path: str = None
        self.input_parameter_path: str = None
        self.temporary_directory_path: str = None

        self.atmosphere_type: str = None
        self.include_molecules: bool = None
        self.nlte_flag: bool = None
        self.nlte_elements: list[str] = []
        self.wavelength_min: float = None
        self.wavelength_max: float = None
        self.wavelength_delta: float = None

        self.debug_mode: int = None
        self.number_of_cpus: int = None
        self.cluster_name: str = None

        self.input_parameters_filename: str = None
        self.resolution: float = None
        self.save_unnormalised_spectra: bool = None

        # other things not in config
        self.output_folder_path_global: str = None

        self.model_atmosphere_list: str = None
        self.model_atmosphere_grid_path: str = None

        self.depart_bin_file_dict: dict = None
        self.depart_aux_file_dict: dict = None
        self.model_atom_file_dict: dict = None

        self.aux_file_length_dict: dict = None
        self.ndimen: int = None

        self.model_temperatures = None
        self.model_logs = None
        self.model_mets = None
        self.marcs_value_keys = None
        self.marcs_models = None
        self.marcs_values = None

        # new options for the slurm cluster
        self.cluster_type = "local"
        self.number_of_nodes = 1
        self.memory_per_cpu_gb = 3.6
        self.script_commands = [            # Additional commands to run before starting dask worker
            'module purge',
            'module load basic-path',
            'module load intel',
            'module load anaconda3-py3.10']
        self.time_limit_hours = 71
        self.slurm_partition = "debug"

    def load_config(self):
        # read the configuration file
        self.config_parser.read(self.config_location)
        # intel or gnu compiler
        self.compiler = self.validate_string_input(self.config_parser["turbospectrum_compiler"]["compiler"], ["intel", "gnu"])
        self.turbospectrum_path = self.config_parser["MainPaths"]["turbospectrum_path"]
        self.interpolators_path = self.config_parser["MainPaths"]["interpolators_path"]
        self.line_list_path = self.config_parser["MainPaths"]["line_list_path"]
        self.model_atmosphere_grid_path_1d = self.config_parser["MainPaths"]["model_atmosphere_grid_path_1d"]
        self.model_atmosphere_grid_path_3d = self.config_parser["MainPaths"]["model_atmosphere_grid_path_3d"]
        self.model_atoms_path = self.config_parser["MainPaths"]["model_atoms_path"]
        self.departure_file_path = self.config_parser["MainPaths"]["departure_file_path"]
        self.departure_file_config_path = self.config_parser["MainPaths"]["departure_file_config_path"]
        self.output_folder_path = self.config_parser["MainPaths"]["output_path"]
        self.input_parameter_path = self.config_parser["MainPaths"]["input_parameter_path"]
        self.temporary_directory_path = os.path.join(self.config_parser["MainPaths"]["temporary_directory_path"], self.output_folder_title, '')

        self.atmosphere_type = self.validate_string_input(self.config_parser["AtmosphereParameters"]["atmosphere_type"], ["1d", "3d"])
        self.include_molecules = self.convert_string_to_bool(self.config_parser["AtmosphereParameters"]["include_molecules"])
        self.nlte_flag = self.convert_string_to_bool(self.config_parser["AtmosphereParameters"]["nlte"])
        self.nlte_elements = self.split_string_to_string_list(self.config_parser["AtmosphereParameters"]["nlte_elements"])
        self.wavelength_min = float(self.config_parser["AtmosphereParameters"]["wavelength_min"])
        self.wavelength_max = float(self.config_parser["AtmosphereParameters"]["wavelength_max"])
        self.wavelength_delta = float(self.config_parser["AtmosphereParameters"]["wavelength_delta"])
        self.resolution = float(self.config_parser["AtmosphereParameters"]["resolution"])

        self.debug_mode = int(self.config_parser["ExtraParameters"]["debug_mode"])
        self.number_of_cpus = int(self.config_parser["ExtraParameters"]["number_of_cpus"])
        self.cluster_name = self.config_parser["ExtraParameters"]["cluster_name"]
        self.save_unnormalised_spectra = self.convert_string_to_bool(self.config_parser["ExtraParameters"]["save_unnormalised_spectra"])

        self.input_parameters_filename = self.config_parser["InputFile"]["input_filename"]

        try:
            self.cluster_type = self.config_parser["SlurmClusterParameters"]["cluster_type"].lower()
            self.number_of_nodes = int(self.config_parser["SlurmClusterParameters"]["number_of_nodes"])
            self.memory_per_cpu_gb = float(self.config_parser["SlurmClusterParameters"]["memory_per_cpu_gb"])
            self.script_commands = self._split_string_to_string_list_with_semicolons(self.config_parser["SlurmClusterParameters"]["script_commands"])
            self.time_limit_hours = float(self.config_parser["SlurmClusterParameters"]["time_limit_hours"])
            self.slurm_partition = self.config_parser["SlurmClusterParameters"]["partition"]
        except KeyError:
            self.cluster_type = "local"
            self.number_of_nodes = 1
            self.memory_per_cpu_gb = 3.6
            self.script_commands = [            # Additional commands to run before starting dask worker
                'module purge',
                'module load basic-path',
                'module load intel',
                'module load anaconda3-py3.10']
            self.time_limit_hours = 71
            self.slurm_partition = "debug"



    def check_valid_input(self):
        self.atmosphere_type = self.atmosphere_type.upper()
        self.include_molecules = self.include_molecules
        self.nlte_flag = self.nlte_flag
        self.wavelength_min = float(self.wavelength_min)
        self.wavelength_max = float(self.wavelength_max)
        if self.wavelength_min > self.wavelength_max:
            raise ValueError("Wavelength min is greater than wavelength max")
        self.wavelength_delta = float(self.wavelength_delta)
        self.resolution = float(self.resolution)
        self.temporary_directory_path = self.convert_to_absolute_path(self.temporary_directory_path)
        self.number_of_cpus = int(self.number_of_cpus)

        self.debug_mode = int(self.debug_mode)
        if self.compiler.lower() == "intel":
            self.turbospectrum_path = os.path.join(self._check_if_path_exists(self.turbospectrum_path),
                                                   "exec", "")
        elif self.compiler.lower() == "gnu":
            self.turbospectrum_path = os.path.join(self._check_if_path_exists(self.turbospectrum_path),
                                                   "exec-gf", "")
        else:
            raise ValueError("Compiler not recognized")
        self.turbospectrum_path = self.turbospectrum_path

        if os.path.exists(self.interpolators_path):
            self.interpolators_path = os.path.join(os.getcwd(), self.interpolators_path)
        else:
            raise ValueError(f"Interpolators path {self.interpolators_path} does not exist")

        if self.atmosphere_type.upper() == "1D":
            self.model_atmosphere_grid_path = self._check_if_path_exists(self.model_atmosphere_grid_path_1d)
            self.model_atmosphere_list = os.path.join(self.model_atmosphere_grid_path,
                                                                "model_atmosphere_list.txt")
        elif self.atmosphere_type.upper() == "3D":
            self.model_atmosphere_grid_path = self._check_if_path_exists(self.model_atmosphere_grid_path_3d)
            self.model_atmosphere_list = os.path.join(self.model_atmosphere_grid_path,
                                                                "model_atmosphere_list.txt")
        else:
            raise ValueError(f"Expected atmosphere type 1D or 3D, got {self.atmosphere_type.upper()}")
        self.model_atoms_path = self._check_if_path_exists(self.model_atoms_path)
        self.departure_file_path = self._check_if_path_exists(self.departure_file_path)
        self.output_folder_path_global = self.convert_to_absolute_path(self.output_folder_path)
        self.line_list_path = self._check_if_path_exists(self.line_list_path)

        nlte_flag_to_save = "NLTE" if self.nlte_flag else "LTE"

        self.output_folder_title = f"{self.output_folder_title}_{nlte_flag_to_save}_{self.input_parameters_filename}"

        self.output_folder_path = os.path.join(self.output_folder_path_global, self.output_folder_title, "")
        self.input_parameter_path = os.path.join(self._check_if_path_exists(self.input_parameter_path), self.input_parameters_filename)


    @staticmethod
    def split_string_to_float_list(string_to_split: str) -> list[float]:
        # remove commas from the string if they exist and split the string into a list
        string_to_split = string_to_split.replace(",", " ").split()
        # convert the list of strings to a list of floats
        string_to_split = [float(i) for i in string_to_split]
        return string_to_split

    @staticmethod
    def split_string_to_string_list(string_to_split: str) -> list[str]:
        # remove commas from the string if they exist and split the string into a list
        string_to_split = string_to_split.replace(",", " ").split()
        return string_to_split

    @staticmethod
    def convert_string_to_bool(string_to_convert: str) -> bool:
        if string_to_convert.lower() in ["true", "yes", "y", "1"]:
            return True
        elif string_to_convert.lower() in ["false", "no", "n", "0"]:
            return False
        else:
            raise ValueError(f"Configuration: could not convert {string_to_convert} to a boolean")

    @staticmethod
    def validate_string_input(input_to_check: str, allowed_values: list[str]) -> str:
        # check if input is in the list of allowed values
        if input_to_check.lower() in allowed_values:
            # return string in lower case with first letter capitalised
            return input_to_check.lower().capitalize()
        else:
            raise ValueError(f"Configuration: {input_to_check} is not a valid input. Allowed values are {allowed_values}")

    @staticmethod
    def convert_to_absolute_path(path):
        if os.path.isabs(path):
            return path
        else:
            # otherwise just return the temp_directory
            new_path = os.path.join(os.getcwd(), path)
            return new_path

    @staticmethod
    def _check_if_path_exists(path_to_check: str, check_valid_path=True) -> str:
        # check if path is absolute
        if os.path.isabs(path_to_check):
            # check if path exists or file exists
            if os.path.exists(os.path.join(path_to_check, "")) or os.path.isfile(path_to_check):
                return path_to_check
        else:
            # if path is relative, check if it exists in the current directory
            if os.path.exists(os.path.join(path_to_check, "")) or os.path.isfile(path_to_check):
                # returns absolute path
                return os.path.join(os.getcwd(), path_to_check, "")
            else:
                # if it starts with ../ convert to ./ and check again
                if path_to_check.startswith("../"):
                    path_to_check = path_to_check[3:]
                    if os.path.exists(os.path.join(path_to_check, "")) or os.path.isfile(path_to_check):
                        return os.path.join(os.getcwd(), path_to_check, "")
        if check_valid_path:
            raise FileNotFoundError(f"Configuration: {path_to_check} does not exist")
        else:
            return ""

    @staticmethod
    def convert_list_to_str(list_to_convert: list) -> str:
        string_to_return = ""
        for element_ in list_to_convert:
            string_to_return = f"{string_to_return} {element_}"
        return string_to_return

    @staticmethod
    def _split_string_to_string_list_with_semicolons(string_to_split: str) -> list[str]:
        # separate based on semicolons and remove them from the list
        string_to_split = string_to_split.split(';')
        return string_to_split

if __name__ == '__main__':
    # load config file from command line
    today = datetime.datetime.now().strftime("%b-%d-%Y-%H-%M-%S")  # used to not conflict with other instances of fits
    today = f"{today}_{np.random.random(1)[0]}"

    if len(sys.argv) == 0:
        config_file = "input_files/synthetic_spectra_generation_configuration.cfg"
    else:
        config_file = sys.argv[1]
    config_synthetic_spectra = SyntheticSpectraConfig(config_file, today)
    config_synthetic_spectra.load_config()
    config_synthetic_spectra.check_valid_input()

    # if debug_mode is >= 1, then change logging level to debug
    if config_synthetic_spectra.debug_mode >= 1:
        logging.basicConfig(level=logging.DEBUG)
    if config_synthetic_spectra.debug_mode >= 2:
        verbose = True
    else:
        verbose = False

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

    ts_config = {"turbospec_path": config_synthetic_spectra.turbospectrum_path,
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
                 "global_temporary_directory": config_synthetic_spectra.temporary_directory_path}

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

    futures = []
    for one_spectra_parameter in spectra_parameters:
        specname, teff, logg, feh, vmic, vmac, rotation, abundances_dict = one_spectra_parameter
        spectrum_name = f"{specname}.spec"
        future = client.submit(run_and_save_wrapper, tsfitpy_pickled_configuration_path, teff, logg, feh, config_synthetic_spectra.wavelength_min,
                               config_synthetic_spectra.wavelength_max, config_synthetic_spectra.wavelength_delta,
                               spectrum_name, config_synthetic_spectra.nlte_flag, config_synthetic_spectra.resolution, vmac, rotation, output_dir, vmic, abundances_dict, config_synthetic_spectra.save_unnormalised_spectra, verbose)
        futures.append(future)  # prepares to get values

    print("Start gathering")  # use http://localhost:8787/status to check status. the port might be different
    futures = np.array(client.gather(futures))  # starts the calculations (takes a long time here)
    results = futures
    print("Worker calculation done")  # when done, save values

    # in spectra_parameters_class.spectra_parameters_df change the column names and add .spec to the specname
    spectra_parameters_class.spectra_parameters_df["specname"] = spectra_parameters_class.spectra_parameters_df["specname"].apply(lambda x: f"{x}.spec")

    # change the columns names for elements in df from X to X_Fe
    # go through columns in the df
    for column in spectra_parameters_class.spectra_parameters_df.columns:
        # if the column name is in the list of elements
        if column in periodic_table:
            # add _Fe to the column name
            spectra_parameters_class.spectra_parameters_df.rename(columns={column: f"{column}_Fe"}, inplace=True)

    # save the spectra parameters 
    spectra_parameters_class.spectra_parameters_df.to_csv(os.path.join(output_dir, "spectra_parameters.csv"), index=False)

    time_end = perf_counter()
    # with 4 decimals, the time is converted to hours, minutes and seconds
    print(f"Time elapsed: {datetime.timedelta(seconds=time_end - time_start)}")

    # print ending date and time
    print("Ending date and time: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    shutil.rmtree(line_list_path_trimmed)  # clean up trimmed line list