from __future__ import annotations
import sys
from configparser import ConfigParser
from scripts.turbospectrum_class_nlte import TurboSpectrum, fetch_marcs_grid
from scripts.convolve import *
import datetime
from scripts.create_window_linelist_function import create_window_linelist
import shutil
from dask.distributed import Client
import socket
import os
from scripts.run_wrapper_v2 import run_and_save_wrapper
import pandas as pd
from time import perf_counter
from scripts.TSFitPy import create_segment_file

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

        self.temporary_directory_path: str = None
        self.fitlist_input_path: str = None
        self.spectra_input_path: str = None
        self.linemasks_path: str = None
        self.output_folder_path: str = None
        self.departure_file_config_path: str = None
        self.departure_file_path: str = None

        self.atmosphere_type: str = None
        self.fitting_mode: str = None
        self.include_molecules: bool = None
        self.nlte_flag: bool = None
        self.fit_vmac: bool = None
        self.vmac_input: bool = None
        self.elements_to_fit: list[str] = None
        self.fit_feh: bool = None
        self.nlte_elements: list[str] = []
        self.linemask_file: str = None
        self.wavelength_delta: float = None
        self.segment_size: float = 5  # default value

        self.segment_file: str = None  # path to the temp place where segment is saved

        self.debug_mode: int = None
        self.number_of_cpus: int = None
        self.experimental_parallelisation: bool = None
        self.cluster_name: str = None

        self.input_parameters_filename: str = None
        self.output_filename: str = None

        self.resolution: float = None
        self.vmac: float = None
        self.rotation: float = None
        self.init_guess_elements: list[str] = []
        self.init_guess_elements_path: list[str] = []
        self.input_elements_abundance: list[str] = []
        self.input_elements_abundance_path: list[str] = []

        self.wavelength_min: float = None
        self.wavelength_max: float = None

        self.fit_vmic: str = None
        self.fit_rotation: bool = None
        self.bounds_vmic: list[float] = None
        self.guess_range_vmic: list[float] = None

        self.bounds_teff: list[float] = None
        self.guess_range_teff: list[float] = None

        self.bounds_vmac: list[float] = None
        self.bounds_rotation: list[float] = None
        self.bounds_abundance: list[float] = None
        self.bounds_feh: list[float] = None
        self.bounds_doppler: list[float] = None

        self.guess_range_vmac: list[float] = None
        self.guess_range_rotation: list[float] = None
        self.guess_range_abundance: list[float] = None
        self.guess_range_doppler: list[float] = None

        self.oldconfig_nlte_config_outdated: bool = False
        self.oldconfig_need_to_add_new_nlte_config: bool = True  # only if nlte_config_outdated == True
        self.oldconfig_model_atom_file: list[str] = []
        self.oldconfig_model_atom_file_input_elem: list[str] = []

        self.fit_teff: bool = None
        self.nelement: int = None
        self.model_atmosphere_list: str = None
        self.model_atmosphere_grid_path: str = None

        self.depart_bin_file_dict: dict = None
        self.depart_aux_file_dict: dict = None
        self.model_atom_file_dict: dict = None

        self.line_begins_sorted: list[float] = None
        self.line_ends_sorted: list[float] = None
        self.line_centers_sorted: list[float] = None

        self.seg_begins: list[float] = None
        self.seg_ends: list[float] = None

        self.aux_file_length_dict: dict = None
        self.ndimen: int = None

        self.model_temperatures = None
        self.model_logs = None
        self.model_mets = None
        self.marcs_value_keys = None
        self.marcs_models = None
        self.marcs_values = None

        self.global_temporary_directory = None  # used only to convert old config to new config
        self.output_folder_path_global = None  # used only to convert old config to new config
        self.turbospectrum_global_path = None  # used only to convert old config to new config

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
        self.spectra_input_path = self.config_parser["MainPaths"]["input_parameter_path"]
        self.temporary_directory_path = os.path.join(self.config_parser["MainPaths"]["temporary_directory_path"], self.output_folder_title, '')

        self.atmosphere_type = self.validate_string_input(self.config_parser["AtmosphereParameters"]["atmosphere_type"], ["1d", "3d"])
        self.include_molecules = self.convert_string_to_bool(self.config_parser["AtmosphereParameters"]["include_molecules"])
        self.nlte_flag = self.convert_string_to_bool(self.config_parser["AtmosphereParameters"]["nlte"])
        self.nlte_elements = self.split_string_to_string_list(self.config_parser["AtmosphereParameters"]["nlte_elements"])
        self.wavelength_min = float(self.config_parser["AtmosphereParameters"]["wavelength_min"])
        self.wavelength_max = float(self.config_parser["AtmosphereParameters"]["wavelength_max"])
        self.wavelength_delta = float(self.config_parser["AtmosphereParameters"]["wavelength_delta"])

        self.debug_mode = int(self.config_parser["ExtraParameters"]["debug_mode"])
        self.number_of_cpus = int(self.config_parser["ExtraParameters"]["number_of_cpus"])
        self.cluster_name = self.config_parser["ExtraParameters"]["cluster_name"]

        self.input_parameters_filename = self.config_parser["InputFile"]["input_filename"]

        self.resolution = float(self.config_parser["SpectraParameters"]["resolution"])
        self.vmac = float(self.config_parser["SpectraParameters"]["vmac"])
        self.rotation = float(self.config_parser["SpectraParameters"]["rotation"])


    def check_valid_input(self):
        self.atmosphere_type = self.atmosphere_type.upper()
        self.include_molecules = self.include_molecules
        self.nlte_flag = self.nlte_flag
        self.wavelength_min = float(self.wavelength_min)
        self.wavelength_max = float(self.wavelength_max)
        self.wavelength_delta = float(self.wavelength_delta)
        self.resolution = float(self.resolution)
        self.rotation = float(self.rotation)
        self.temporary_directory_path = self.convert_to_absolute_path(self.temporary_directory_path)
        self.number_of_cpus = int(self.number_of_cpus)

        self.debug_mode = int(self.debug_mode)
        if self.compiler.lower() == "intel":
            self.turbospectrum_path = os.path.join(self.convert_to_absolute_path(self.turbospectrum_path),
                                                   "exec", "")
        elif self.compiler.lower() == "gnu":
            self.turbospectrum_path = os.path.join(self.convert_to_absolute_path(self.turbospectrum_path),
                                                   "exec-gf", "")
        else:
            raise ValueError("Compiler not recognized")
        self.turbospectrum_path = self.turbospectrum_path

        if os.path.exists(self.interpolators_path):
            self.interpolators_path = os.path.join(os.getcwd(), self.interpolators_path)
        else:
            raise ValueError(f"Interpolators path {self.interpolators_path} does not exist")

        if self.atmosphere_type.upper() == "1D":
            self.model_atmosphere_grid_path = self.convert_to_absolute_path(self.model_atmosphere_grid_path_1d)
            self.model_atmosphere_list = os.path.join(self.model_atmosphere_grid_path,
                                                                "model_atmosphere_list.txt")
        elif self.atmosphere_type.upper() == "3D":
            self.model_atmosphere_grid_path = self.convert_to_absolute_path(self.model_atmosphere_grid_path_3d)
            self.model_atmosphere_list = os.path.join(self.model_atmosphere_grid_path,
                                                                "model_atmosphere_list.txt")
        else:
            raise ValueError(f"Expected atmosphere type 1D or 3D, got {self.atmosphere_type.upper()}")
        self.model_atoms_path = self.convert_to_absolute_path(self.model_atoms_path)
        self.departure_file_path = self.convert_to_absolute_path(self.departure_file_path)
        self.output_folder_path_global = self.convert_to_absolute_path(self.output_folder_path)

        nlte_flag_to_save = "NLTE" if self.nlte_flag else "LTE"

        self.output_folder_title = f"{self.output_folder_title}_{nlte_flag_to_save}_{self.input_parameters_filename}"

        self.output_folder_path = os.path.join(self.convert_to_absolute_path(self.output_folder_path),
                                                    self.output_folder_title)
        self.spectra_input_path = self.convert_to_absolute_path(self.spectra_input_path)
        self.line_list_path = self.convert_to_absolute_path(self.line_list_path)


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
            if os.path.exists(path):
                return path
            else:
                raise ValueError(f"Configuration: {path} does not exist")
        else:
            # otherwise just return the temp_directory
            new_path = os.path.join(os.getcwd(), path)
            if os.path.exists(new_path):
                return new_path
            else:
                raise ValueError(f"Configuration: {new_path} does not exist")

    @staticmethod
    def convert_list_to_str(list_to_convert: list) -> str:
        string_to_return = ""
        for element in list_to_convert:
            string_to_return = f"{string_to_return} {element}"
        return string_to_return

if __name__ == '__main__':
    # load config file from command line
    today = datetime.datetime.now().strftime("%b-%d-%Y-%H-%M-%S")  # used to not conflict with other instances of fits
    today = f"{today}_{np.random.random(1)[0]}"

    config_file = sys.argv[1]
    config_synthetic_spectra = SyntheticSpectraConfig(config_file, today)
    config_synthetic_spectra.load_config()
    config_synthetic_spectra.check_valid_input()

    """ts_compiler = "intel" #needs to be "intel" or "gnu"
    atmosphere_type = "1D"
    nlte_elements = [] # e.g. ["Mg", "Si", "Ca", "Mn", "Ti", "Fe", "Y", "Ni", "Ba"]

    #set directories
    if ts_compiler == "intel":
        turbospec_path = "/turbospectrum/exec/"
    elif ts_compiler == "gnu":
        turbospec_path = "/turbospectrum/exec-gf/"
    interpol_path = "./scripts/model_interpolators/"
    line_list_path = "/input_files/linelists/linelist_for_fitting/"
    if atmosphere_type == "1D":
        model_atmosphere_grid_path = "/input_files/model_atmospheres/1D/"
        model_atmosphere_list = model_atmosphere_grid_path + "model_atmosphere_list.txt"
    elif atmosphere_type == "3D":
        model_atmosphere_grid_path = "/input_files/model_atmospheres/3D/"
        model_atmosphere_list = model_atmosphere_grid_path + "model_atmosphere_list.txt"
    model_atom_path = "/input_files/nlte_data/model_atoms/"
    departure_file_path = "/input_files/nlte_data/"""

    teff = 5777
    logg = 4.4
    # met = 0.0      # met chosen below
    met_list = [0.0]  # metallicities to generate
    vturb = 1.0

    #nlte_flag = False

    """line_to_check = 4883
    lmin = line_to_check - 20  # change lmin/lmax here
    lmax = line_to_check + 20
    lmin = 4800  # or change lmin/lmax here
    lmax = 5500
    ldelta = 0.005

    cpus_to_use = 1  # how many cpus to use (Dask workers)"""

    output_dir = os.path.join(, today, "")
    #temp_directory = './temp_direectory/'
    #login_node_address = "gemini-login.mpia.de"

    # load NLTE data
    nlte_config = ConfigParser()
    nlte_config.read(os.path.join(departure_file_path, "nlte_filenames.cfg"))

    depart_bin_file_dict, depart_aux_file_dict, model_atom_file_dict = {}, {}, {}

    for element in nlte_elements:
        if atmosphere_type == "1D":
            bin_config_name, aux_config_name = "1d_bin", "1d_aux"
        else:
            bin_config_name, aux_config_name = "3d_bin", "3d_aux"
        depart_bin_file_dict[element] = nlte_config[element][bin_config_name]
        depart_aux_file_dict[element] = nlte_config[element][aux_config_name]
        model_atom_file_dict[element] = nlte_config[element]["atom_file"]

    depart_bin_file, depart_aux_file, model_atom_file = depart_bin_file_dict, depart_aux_file_dict, model_atom_file_dict


    # read in the atmosphere grid to compute for the synthetic spectra
    # Read the file
    df = pd.read_csv('input.txt', delim_whitespace=True)

    # Create a dictionary that maps non-standard names to standard ones
    name_variants = {'vmic': ['vturb', 'vturbulence', 'vmicro', 'vm'],
                     'teff': ['temp', 'temperature', 't'],
                     'vmac': ['vmacroturb', 'vmacro', 'vmacro_turb', 'vmacro_turbulence', 'vmacroturbulence'],
                     'logg': ['logg', 'grav'],
                     'feh': ['met'],
                     'rotation': ['vsini', 'vrot', 'rot', 'vrotini', 'vrotini', 'vrotini']}

    # Reverse the dictionary: map variants to standard names
    name_dict = {variant: standard for standard, variants in name_variants.items() for variant in variants}

    for col in df.columns:
        # Replace the column name if it's in the dictionary, otherwise leave it unchanged
        standard_name = name_dict.get(col.lower(), col)
        df.rename(columns={col: standard_name}, inplace=True)

    df.columns = df.columns.str.lower().capitalize()

    """met = np.arange(-1.5, 0.5, 0.1)
    nlte_flag = np.array([True, False])
    one, two = np.meshgrid(met, nlte_flag)
    met_list = one.flatten()
    nlte_flag_list = two.flatten()"""

    model_temperatures, model_logs, model_mets, marcs_value_keys, marcs_models, marcs_values = fetch_marcs_grid(model_atmosphere_list, TurboSpectrum.marcs_parameters_to_ignore)
    aux_file_length_dict = {}
    if nlte_flag:
        for element in model_atom_file:
            aux_file_length_dict[element] = len(np.loadtxt(os.path.join(departure_file_path, depart_aux_file[element]), dtype='str'))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    line_list_path_trimmed = os.path.join(temp_directory, "linelist_for_fitting_trimmed", "")
    line_list_path_trimmed = os.path.join(line_list_path_trimmed, "all", today, '')

    print("Trimming")
    include_molecules = True
    create_window_linelist([lmin], [lmax], line_list_path, line_list_path_trimmed, include_molecules, False)
    print("trimming done")

    line_list_path_trimmed = os.path.join(line_list_path_trimmed, "0", "")

    print("Preparing workers")
    client = Client(threads_per_worker=1, n_workers=config_synthetic_spectra.number_of_cpus)  # if # of threads are not equal to 1, then may break the program
    print(client)

    host = client.run_on_scheduler(socket.gethostname)
    port = client.scheduler_info()['services']['dashboard']
    print(f"Assuming that the cluster is ran at {config_synthetic_spectra.cluster_name} (change in config if not the case)")

    # print(logger.info(f"ssh -N -L {port}:{host}:{port} {login_node_address}"))
    print(f"ssh -N -L {port}:{host}:{port} {config_synthetic_spectra.cluster_name}")

    print("Worker preparation complete")

    resolution, macro, rotation = 0, 0, 0

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
                 "model_atom_file": model_atom_file}

    # time to run the code
    time_start = perf_counter()

    futures = []
    for metal in met_list:
        spectrum_name = f"spectra_output_{teff}_{logg}_"
        future = client.submit(run_and_save_wrapper, ts_config, teff, logg, metal, config_synthetic_spectra.wavelength_min,
                               config_synthetic_spectra.wavelength_max, config_synthetic_spectra.wavelength_delta,
                               spectrum_name, config_synthetic_spectra.nlte_flag, resolution, macro, rotation, output_dir, vturb)
        futures.append(future)  # prepares to get values

    print("Start gathering")  # use http://localhost:8787/status to check status. the port might be different
    futures = np.array(client.gather(futures))  # starts the calculations (takes a long time here)
    results = futures
    print("Worker calculation done")  # when done, save values

    time_end = perf_counter()
    # with 4 decimals, the time is converted to hours, minutes and seconds
    print(f"Time elapsed: {datetime.timedelta(seconds=time_end - time_start)}")

    shutil.rmtree(line_list_path_trimmed)  # clean up trimmed line list