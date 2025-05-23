from __future__ import annotations

import logging
import os
from _warnings import warn
from configparser import ConfigParser
import pandas as pd
import numpy as np
from .solar_abundances import periodic_table, solar_abundances

class SpectraParameters:
    def __init__(self, input_file_path: str, first_row_name: bool):
        # read in the atmosphere grid to compute for the synthetic spectra
        # Read the file
        df = pd.read_csv(input_file_path, index_col=False, comment=';', sep='\s+')

        with open(input_file_path, 'r') as f:
            header = f.readline()

        header = header.split()
        if header[0] == '#':
            header.pop(0)
            # also remove the last column in the df
            df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)

        header = np.asarray(header)

        # need to also remove following words from header: (not needed if fitting fe)
        words_to_remove = ['(not', 'needed', 'if', 'fitting', 'fe)']
        for word in words_to_remove:
            # check if word is in header
            if word in header:
                # remove word from header
                header = np.delete(header, np.where(header == word))
                # remove column from df
                df.drop(df.columns[-1], axis=1, inplace=True)

        # put header into the df
        df.columns = header

        # Create a dictionary that maps non-standard names to standard ones
        name_variants = {'vmic': ['vturb', 'vturbulence', 'vmicro', 'vm', 'inputvmicroturb', 'inputvmic', 'vt'],
                         'rv': ['radvel', 'radialvelocity', 'radial_velocity', "rv", "radvel", "radialvelocity", "radialvelocity"],
                         'teff': ['temp', 'temperature', 't'],
                         'vmac': ['vmac', 'vmacroturb', 'vmacro', 'vmacroturb', 'vmacroturbulence', 'vmacroturbulence', 'inputvmacroturb', 'inputvmacroturbulence'],
                         'logg': ['grav'],
                         'feh': ['met', 'metallicity', 'metallicityfeh', 'metallicityfeh', 'mh', 'mh', 'mh'],
                         'rotation': ['vsini', 'vrot', 'rot', 'vrotini', 'vrotini', 'vrotini'],
                         'specname': ['spec_name', 'spectrum_name', 'spectrumname', 'spectrum', 'spectrumname', 'spectrumname'],
                         'resolution': ['res', 'resolution', 'resolvingpower', 'resolvingpower', 'r'],
                         'snr': ['snr', 'signaltonoiseratio'],}

        # Reverse the dictionary: map variants to standard names
        name_dict = {variant: standard for standard, variants in name_variants.items() for variant in variants}

        if first_row_name:
            # replace first column name with 'name'
            df.rename(columns={df.columns[0]: 'specname'}, inplace=True)
        else:
            # add column name 'name'
            df.insert(0, 'specname', df.index)

        abundances_xfe_given = []
        abundances_xh_given = []
        abundances_x_given = []
        self.abundance_elements_given: list[str] = []

        for col in df.columns:
            # Replace the column name if it's in the dictionary, otherwise leave it unchanged
            standard_name = name_dict.get(self._strip_string(col.lower()))
            if standard_name is None:
                standard_name = self._strip_string(col.lower())
                if standard_name not in name_variants.keys():
                    testing_col = self._strip_string(col)
                    ending_element = testing_col[-2:]
                    starting_element = testing_col[:-2]
                    if ending_element.lower() == "fe" and starting_element in periodic_table:
                        # means X/Fe
                        standard_name = f"{starting_element.lower()}"
                        abundances_xfe_given.append(standard_name)
                        self.abundance_elements_given.append(standard_name.capitalize())
                    elif ending_element[-1].lower() == 'h' and testing_col[:-1].lower().capitalize() in periodic_table:
                        # means X/H
                        standard_name = f"{testing_col[:-1].lower()}"
                        abundances_xh_given.append(standard_name)
                        self.abundance_elements_given.append(standard_name.capitalize())
                    elif (np.size(testing_col) <= 3 and testing_col[1:].capitalize() in periodic_table and
                          testing_col[0].lower() == 'a' and col[1] == "(" and col[-1] == ")"):
                        # 01.08.2023: added check for brackets, because A(C) would be parsed as Ac, not A(C)
                        # means just elemental abundance perhaps because A(X) is given
                        standard_name = f"{testing_col[1:].lower()}"
                        abundances_x_given.append(standard_name)
                        self.abundance_elements_given.append(standard_name.capitalize())
                    elif np.size(testing_col) <= 2 and testing_col.capitalize() in periodic_table:
                        # means just elemental abundance perhaps
                        standard_name = f"{testing_col.lower()}"
                        abundances_x_given.append(standard_name)
                        self.abundance_elements_given.append(standard_name.capitalize())
                    else:
                        # could not parse, not element?
                        raise ValueError(f"Could not parse {col} as any known parameter or element")
            df.rename(columns={col: standard_name}, inplace=True)

        # make all columns lower
        df.columns = df.columns.str.lower()

        for element in abundances_xfe_given:
            # capitalise
            df.rename(columns={element: element.capitalize()}, inplace=True)

        # convert xh to xfe
        for element in abundances_xh_given:
            # xfe = xh - feh
            df[element] = df[element] - df['feh']
            # convert to capital letter
            df.rename(columns={element: element.capitalize()}, inplace=True)

        # convert A(X) to xfe
        for element in abundances_x_given:
            # [X/H]_star = A(X)_star - A(X)_sun
            # xfe = [X/H]_star - feh
            df[element] = df[element] - solar_abundances[element.capitalize()] - df['feh']
            # convert to capital letter
            df.rename(columns={element: element.capitalize()}, inplace=True)

        # if rv is not present, add it
        if 'rv' not in df.columns:
            df.insert(1, 'rv', 0.0)
        if 'feh' not in df.columns:
            df.insert(2, 'feh', 0.0)

        self.spectra_parameters_df = df

        # get amount of rows
        self.number_of_rows = len(self.spectra_parameters_df.index)

        # check if all columns are present
        self._check_if_all_columns_present(['specname', 'rv', 'teff', 'logg', 'feh'])

    def _check_if_all_columns_present(self, columns: list[str]):
        """
        checks if all columns are present in the dataframe
        :param columns: list of column names to check
        :return:
        """
        for column in columns:
            if column not in self.spectra_parameters_df.columns:
                raise ValueError(f"Column {column} is not present in the dataframe")


    def get_spectra_parameters_for_fit(self, vmic_output: bool, vmac_output: bool, rotation_output: bool) -> np.ndarray:
        """
        returns spectra parameters as a numpy array, where each entry is:
        specname, rv, teff, logg, met, vmic, vmac, input_abundance_dict
        :param vmic_output: if True, vmic is outputted, otherwise None
        :param vmac_output: if True, vmac is outputted, otherwise 0
        :param rotation_output: if True, rotation is outputted, otherwise 0
        :return:  [[specname, rv, teff, logg, feh, vmic, vmac, rotation, input_abundance_dict], ...]
        """

        specname_list = self.spectra_parameters_df['specname'].values
        rv_list = self.spectra_parameters_df['rv'].values
        teff_list = self.spectra_parameters_df['teff'].values
        logg_list = self.spectra_parameters_df['logg'].values
        feh_list = self.spectra_parameters_df['feh'].values
        if 'vmic' in self.spectra_parameters_df.columns and vmic_output:
            vmic_list = self.spectra_parameters_df['vmic'].values
        else:
            # else is list of None
            vmic_list = [None] * self.number_of_rows
        if 'vmac' in self.spectra_parameters_df.columns and vmac_output:
            vmac_list = self.spectra_parameters_df['vmac'].values
        else:
            vmac_list = np.zeros(len(specname_list))
        if 'rotation' in self.spectra_parameters_df.columns and rotation_output:
            rotation_list = self.spectra_parameters_df['rotation'].values
        else:
            rotation_list = np.zeros(len(specname_list))
        # if resolution is in the columns, add it
        if 'resolution' in self.spectra_parameters_df.columns:
            resolution_list = self.spectra_parameters_df['resolution'].values
        else:
            resolution_list = np.zeros(len(specname_list))
        # if snr is in the columns, add it
        if 'snr' in self.spectra_parameters_df.columns:
            snr_list = self.spectra_parameters_df['snr'].values
        else:
            snr_list = np.zeros(len(specname_list))

        # get abundance elements, put in dictionary and then list, where each entry is a dictionary
        abundance_list = self._get_abundance_list()

        # stack all parameters
        stacked_parameters = np.stack((specname_list, rv_list, teff_list, logg_list, feh_list, vmic_list, vmac_list, rotation_list, abundance_list, resolution_list, snr_list), axis=1)

        return stacked_parameters

    def _get_abundance_list(self):
        abundance_list = []
        for i in range(self.number_of_rows):
            abundance_dict = {}
            for element in self.abundance_elements_given:
                elemental_abundance = self.spectra_parameters_df[element][i]
                # replace nan with 0
                if np.isnan(elemental_abundance):
                    elemental_abundance = 0
                abundance_dict[element] = elemental_abundance
            abundance_list.append(abundance_dict)
        return abundance_list

    def get_spectra_parameters_for_grid_generation(self) -> np.ndarray:
        """
        returns spectra parameters as a numpy array, where each entry is:
        specname, rv, teff, logg, met, vmic, vmac, input_abundance_dict
        :return:  [[specname, teff, logg, feh, vmic, vmac, rotation, input_abundance_dict], ...]
        """

        specname_list = self.spectra_parameters_df['specname'].values
        teff_list = self.spectra_parameters_df['teff'].values
        logg_list = self.spectra_parameters_df['logg'].values
        feh_list = self.spectra_parameters_df['feh'].values
        if 'vmic' in self.spectra_parameters_df.columns:
            vmic_list = self.spectra_parameters_df['vmic'].values
        else:
            # else is list of None
            vmic_list = [None] * self.number_of_rows
        if 'vmac' in self.spectra_parameters_df.columns:
            vmac_list = self.spectra_parameters_df['vmac'].values
        else:
            vmac_list = np.zeros(len(specname_list))
        if 'rotation' in self.spectra_parameters_df.columns:
            rotation_list = self.spectra_parameters_df['rotation'].values
        else:
            rotation_list = np.zeros(len(specname_list))
        # get abundance elements, put in dictionary and then list, where each entry is a dictionary
        abundance_list = self._get_abundance_list()

        # stack all parameters
        stacked_parameters = np.stack((specname_list, teff_list, logg_list, feh_list, vmic_list, vmac_list, rotation_list, abundance_list), axis=1)

        return stacked_parameters


    @staticmethod
    def _strip_string(string_to_strip: str) -> str:
        bad_characters = ["[", "]", "/", "\\", "(", ")", "{", "}", "_", "#", 'nlte', 'lte', 'mean', 'median']
        for character_to_remove in bad_characters:
            string_to_strip = string_to_strip.replace(character_to_remove, '')
        return string_to_strip

    def __str__(self):
        # print the dataframe
        return self.spectra_parameters_df.to_string()


class GenericConfig:
    def __init__(self, config_location: str, output_folder_title: str):
        self.config_parser = ConfigParser()
        self.config_location: str = config_location

        if output_folder_title is not None:
            self.output_folder_title: str = output_folder_title
        else:
            self.output_folder_title: str = "none"

        self.compiler: str = None
        self.spectral_code_path: str = None
        self.interpolators_path: str = None
        self.line_list_path: str = None
        self.model_atmosphere_grid_path_1d: str = None
        self.model_atmosphere_grid_path_3d: str = None
        self.model_atoms_path: str = None

        self.atmosphere_type: str = None
        self.include_molecules: bool = None
        self.nlte_flag: bool = None
        self.nlte_elements: list[str] = []
        self.wavelength_min: float = None
        self.wavelength_max: float = None
        self.wavelength_delta: float = None
        self.resolution: float = None

        self.cluster_name = None
        self.number_of_cpus = None
        self.debug_mode = None

        self.departure_file_path: str = None
        self.departure_file_config_path: str = None
        self.output_folder_path: str = None
        self.temporary_directory_path: str = None

        self.debug_mode: int = None
        self.number_of_cpus: int = None
        self.cluster_name: str = None

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
        self.script_commands = [  # Additional commands to run before starting dask worker
            'module purge',
            'module load basic-path',
            'module load intel',
            'module load anaconda3-py3.10']
        self.time_limit_hours = 71
        self.slurm_partition = "debug"

        # new options for the m3dis
        self.n_nu = 1
        self.hash_table_size = 10
        self.mpi_cores = 1
        self.iterations_max = 0
        self.iterations_max_precompute = 0
        self.convlim = 0.01
        self.snap = 1
        self.dims = 23
        self.nx = 10
        self.ny = 10
        self.nz = 230
        # m3dis parameters
        self.m3dis_python_package_name = "m3dis"

        # Value of lpoint for turbospectrum in spectrum.inc file
        self.lpoint_turbospectrum = 500_000

    def load_config(self):

        # check if the config file exists
        if not os.path.isfile(self.config_location):
            raise ValueError(f"The configuration file {self.config_location} does not exist.")

        # read the configuration file
        self.config_parser.read(self.config_location)
        # intel or gnu compiler
        self.compiler = self._validate_string_input(self.config_parser["turbospectrum_compiler"]["compiler"],
                                                    ["intel", "gnu", "m3dis", "ifort", "ifx"])
        # 08.02.2024: ifx is the new intel compiler. So replacing intel with ifort
        if self.compiler == "intel":
            self.compiler = "ifort"
        try:
            self.spectral_code_path = self.config_parser["MainPaths"]["code_path"]
        except KeyError:
            self.spectral_code_path = self.config_parser["MainPaths"]["turbospectrum_path"]
        self.interpolators_path = self.config_parser["MainPaths"]["interpolators_path"]
        self.line_list_path = self.config_parser["MainPaths"]["line_list_path"]
        self.model_atmosphere_grid_path_1d = self.config_parser["MainPaths"]["model_atmosphere_grid_path_1d"]
        self.model_atmosphere_grid_path_3d = self.config_parser["MainPaths"]["model_atmosphere_grid_path_3d"]
        self.model_atoms_path = self.config_parser["MainPaths"]["model_atoms_path"]
        self.departure_file_path = self.config_parser["MainPaths"]["departure_file_path"]
        self.departure_file_config_path = self.config_parser["MainPaths"]["departure_file_config_path"]
        self.output_folder_path = self.config_parser["MainPaths"]["output_path"]
        self.temporary_directory_path = os.path.join(self.config_parser["MainPaths"]["temporary_directory_path"],
                                                     self.output_folder_title, '')

        self.debug_mode = int(self.config_parser["ExtraParameters"]["debug_mode"])
        self.number_of_cpus = int(self.config_parser["ExtraParameters"]["number_of_cpus"])
        self.cluster_name = self.config_parser["ExtraParameters"]["cluster_name"]

        try:
            self.cluster_type = self.config_parser["SlurmClusterParameters"]["cluster_type"].lower()
            self.number_of_nodes = int(self.config_parser["SlurmClusterParameters"]["number_of_nodes"])
            self.memory_per_cpu_gb = float(self.config_parser["SlurmClusterParameters"]["memory_per_cpu_gb"])
            self.script_commands = self._split_string_to_string_list_with_semicolons(self.config_parser["SlurmClusterParameters"]["script_commands"])
            self.time_limit_hours = float(self.config_parser["SlurmClusterParameters"]["time_limit_hours"])
            self.slurm_partition = self.config_parser["SlurmClusterParameters"]["partition"]
        except KeyError:
            pass

        try:
            self.lpoint_turbospectrum = int(self.config_parser["AdvancedOptions"]["lpoint_turbospectrum"])
        except KeyError:
            pass

        # m3dis stuff
        try:
            self.n_nu = int(self.config_parser["m3disParameters"]["n_nu"])
            self.hash_table_size = int(self.config_parser["m3disParameters"]["hash_table_size"])
            self.mpi_cores = int(self.config_parser["m3disParameters"]["mpi_cores"])
            self.iterations_max = int(self.config_parser["m3disParameters"]["iterations_max"])
            self.iterations_max_precompute = int(self.config_parser["m3disParameters"]["iterations_max_precompute"])
            self.convlim = float(self.config_parser["m3disParameters"]["convlim"])
            self.snap = int(self.config_parser["m3disParameters"]["snap"])
            self.dims = int(self.config_parser["m3disParameters"]["dims"])
            self.nx = int(self.config_parser["m3disParameters"]["nx"])
            self.ny = int(self.config_parser["m3disParameters"]["ny"])
            self.nz = int(self.config_parser["m3disParameters"]["nz"])
        except KeyError:
            pass

        try:
            self.m3dis_python_package_name = self.config_parser["AdvancedOptions"]["m3dis_python_package_name"]
        except KeyError:
            pass

    def validate_input(self, check_valid_path=True):
        self.departure_file_config_path = self._check_if_file_exists(self.departure_file_config_path,
                                                                     check_valid_path=check_valid_path)
        self.atmosphere_type = self.atmosphere_type.upper()
        self.include_molecules = self.include_molecules
        if len(self.nlte_elements) == 0 and self.nlte_flag:
            print("\nNo NLTE elements were provided, setting NLTE flag to False!!\n")
            self.nlte_flag = False
        else:
            self.nlte_flag = self.nlte_flag
        self.wavelength_min = float(self.wavelength_min)
        self.wavelength_max = float(self.wavelength_max)
        if self.wavelength_min > self.wavelength_max:
            self.wavelength_min, self.wavelength_max = self.wavelength_max, self.wavelength_min
        self.wavelength_delta = float(self.wavelength_delta)
        self.resolution = float(self.resolution)

        self.number_of_cpus = int(self.number_of_cpus)

        self.debug_mode = self.debug_mode
        if self.debug_mode >= 1:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.WARNING)

        if self.spectral_code_path is None:
            self.spectral_code_path = "../turbospectrum/"
        old_turbospectrum_global_path = os.path.join(os.getcwd(),
                                                     self._check_if_path_exists(self.spectral_code_path,
                                                                                check_valid_path))
        if self.compiler.lower() == "ifort" or self.compiler.lower() == "intel":
            spectral_code_path = os.path.join(old_turbospectrum_global_path, "exec", "")
            # check if path exists
            if os.path.exists(spectral_code_path):
                self.spectral_code_path = spectral_code_path
            else:
                self.spectral_code_path = os.path.join(old_turbospectrum_global_path, "exec-intel", "")
        elif self.compiler.lower() == "ifx":
            self.spectral_code_path = os.path.join(old_turbospectrum_global_path,
                                                   "exec-ifx", "")
        elif self.compiler.lower() == "gnu":
            self.spectral_code_path = os.path.join(old_turbospectrum_global_path,
                                                   "exec-gf", "")
        elif self.compiler.lower() == "m3dis":
            _ = self._check_if_file_exists(os.path.join(self.spectral_code_path, "dispatch.x"), check_valid_path)
            self.spectral_code_path = os.path.join(os.getcwd(), self.spectral_code_path, "")
        else:
            raise ValueError("Compiler not recognized")
        self.spectral_code_path = self.spectral_code_path

        if os.path.exists(self.interpolators_path):
            self.interpolators_path = os.path.join(os.getcwd(), self.interpolators_path)
        else:
            if self.interpolators_path.startswith("./"):
                self.interpolators_path = self.interpolators_path[2:]
                self.interpolators_path = os.path.join(os.getcwd(), "scripts", self.interpolators_path)

                if not os.path.exists(self.interpolators_path):
                    raise ValueError(f"Interpolators path {self.interpolators_path} does not exist")

        if self.atmosphere_type.upper() == "1D":
            self.model_atmosphere_grid_path = self._check_if_path_exists(self.model_atmosphere_grid_path_1d, check_valid_path)
            self.model_atmosphere_list = os.path.join(self.model_atmosphere_grid_path,
                                                                "model_atmosphere_list.txt")
        elif self.atmosphere_type.upper() == "3D":
            self.model_atmosphere_grid_path = self._check_if_path_exists(self.model_atmosphere_grid_path_3d, check_valid_path)
            self.model_atmosphere_list = os.path.join(self.model_atmosphere_grid_path,
                                                                "model_atmosphere_list.txt")
        else:
            raise ValueError(f"Expected atmosphere type 1D or 3D, got {self.atmosphere_type.upper()}")

        self.model_atoms_path = self._check_if_path_exists(self.model_atoms_path, check_valid_path)
        self.departure_file_path = self._check_if_path_exists(self.departure_file_path, check_valid_path)

        if self.n_nu <= 0:
            raise ValueError("n_nu must be greater than 0")
        if self.hash_table_size <= 0:
            raise ValueError("hash_table_size must be greater than 0")
        if self.mpi_cores <= 0:
            raise ValueError("mpi_cores must be greater than 0")
        if self.iterations_max < 0:
            raise ValueError("iterations_max must be greater than or equal to 0")
        if self.iterations_max_precompute < 0:
            raise ValueError("iterations_max_precompute must be greater than or equal to 0")
        if self.convlim < 0:
            raise ValueError("convlim must be greater than or equal to 0")
        if self.snap < 0:
            raise ValueError("snap must be greater than or equal to 0")
        if self.dims <= 0:
            raise ValueError("dims must be greater than 0")
        if self.nx <= 0:
            raise ValueError("nx must be greater than 0")
        if self.ny <= 0:
            raise ValueError("ny must be greater than 0")
        if self.nz <= 0:
            raise ValueError("nz must be greater than 0")
        if self.lpoint_turbospectrum <= 0:
            raise ValueError("lpoint_turbospectrum must be greater than 0")

    @staticmethod
    def _split_string_to_float_list(string_to_split: str) -> list[float]:
        # remove commas from the string if they exist and split the string into a list
        string_to_split = string_to_split.replace(",", " ").split()
        # convert the list of strings to a list of floats
        string_to_split = [float(i) for i in string_to_split]
        return string_to_split

    @staticmethod
    def _split_string_to_string_list(string_to_split: str) -> list[str]:
        # remove commas from the string if they exist and split the string into a list
        string_to_split = string_to_split.replace(",", " ").split()
        return string_to_split

    @staticmethod
    def _convert_string_to_bool(string_to_convert: str) -> bool:
        if string_to_convert.lower() in ["true", "yes", "y", "1"]:
            return True
        elif string_to_convert.lower() in ["false", "no", "n", "0"]:
            return False
        else:
            raise ValueError(f"Configuration: could not convert {string_to_convert} to a boolean")

    @staticmethod
    def _validate_string_input(input_to_check: str, allowed_values: list[str]) -> str:
        # check if input is in the list of allowed values
        if input_to_check.lower() in allowed_values:
            # return string in lower case with first letter capitalised
            return input_to_check.lower().capitalize()
        else:
            raise ValueError(f"Configuration: {input_to_check} is not a valid input. Allowed values are {allowed_values}")

    @staticmethod
    def _check_if_file_exists(path_to_check: str, check_valid_path=True) -> str:
        # check if path is absolute
        if os.path.isabs(path_to_check):
            # check if path exists or file exists
            if os.path.isfile(path_to_check):
                return path_to_check
        else:
            # if path is relative, check if it exists in the current directory
            if os.path.isfile(path_to_check):
                # returns absolute path
                return os.path.join(os.getcwd(), path_to_check)
            else:
                # if it starts with ../ convert to ./ and check again
                if path_to_check.startswith("../"):
                    path_to_check = path_to_check[3:]
                    if os.path.isfile(path_to_check):
                        return os.path.join(os.getcwd(), path_to_check)
        # try to add ../ to the path and check if it exists
        if os.path.isfile(os.path.join("..", path_to_check)):
            return os.path.join(os.getcwd(), "..", path_to_check)
        if check_valid_path:
            raise FileNotFoundError(f"Configuration: {path_to_check} file does not exist")
        else:
            return ""

    @staticmethod
    def _check_if_path_exists(path_to_check: str, check_valid_path=True) -> str:
        # check if path is absolute
        if os.path.isabs(path_to_check):
            # check if path exists or file exists
            if os.path.exists(os.path.join(path_to_check, "")):
                return path_to_check
        else:
            # if path is relative, check if it exists in the current directory
            if os.path.exists(os.path.join(path_to_check, "")):
                # returns absolute path
                return os.path.join(os.getcwd(), path_to_check, "")
            else:
                # if it starts with ../ convert to ./ and check again
                if path_to_check.startswith("../"):
                    path_to_check = path_to_check[3:]
                    if os.path.exists(os.path.join(path_to_check, "")):
                        return os.path.join(os.getcwd(), path_to_check, "")
        # try to add ../ to the path and check if it exists
        if os.path.exists(os.path.join("..", path_to_check, "")):
            return os.path.join(os.getcwd(), "..", path_to_check, "")
        if check_valid_path:
            raise FileNotFoundError(f"Configuration: {path_to_check} folder does not exist")
        else:
            return ""

    @staticmethod
    def _find_path_temporary_directory(temp_directory):
        # find the path to the temporary directory by finding if /scripts/ is located adjacent to the input directory
        # if it is, change temp_directory path to that one
        # check if path is absolute then return it
        if os.path.isabs(temp_directory):
            return temp_directory

        # first check if path above path directory contains /scripts/
        if os.path.exists(os.path.join(temp_directory, "..", "scripts", "")):
            return os.path.join(os.getcwd(), temp_directory)
        elif temp_directory.startswith("../"):
            # if it doesnt, and temp_directory contains ../ remove the ../ and return that
            return os.path.join(os.getcwd(), temp_directory[3:])
        else:
            # otherwise just return the temp_directory
            return os.path.join(os.getcwd(), temp_directory)

    @staticmethod
    def _convert_list_to_str(list_to_convert: list) -> str:
        string_to_return = ""
        for element in list_to_convert:
            string_to_return = f"{string_to_return} {element}"
        return string_to_return

    @staticmethod
    def _split_string_to_string_list_with_semicolons(string_to_split: str) -> list[str]:
        # separate based on semicolons and remove them from the list
        string_to_split = string_to_split.split(';')
        return string_to_split


    @staticmethod
    def _convert_to_absolute_path(path):
        if os.path.isabs(path):
            return path
        else:
            # otherwise just return the temp_directory
            new_path = os.path.join(os.getcwd(), path)
            return new_path


class TSFitPyConfig(GenericConfig):
    def __init__(self, config_location: str, output_folder_title=None, spectra_location=None):
        super().__init__(config_location, output_folder_title)

        self.fitlist_input_path: str = None
        if spectra_location is not None:
            self.spectra_input_path: str = spectra_location
        else:
            self.spectra_input_path: str = None
        self.linemasks_path: str = None

        self.fitting_mode: str = None
        self.fit_vmac: bool = None
        self.vmac_input: bool = None
        self.elements_to_fit: list[str] = None
        self.fit_feh: bool = None
        self.linemask_file: list[str] = None
        self.segment_size: float = 5  # default value

        self.segment_file: str = None  # path to the temp place where segment is saved

        self.experimental_parallelisation: bool = False

        self.input_fitlist_filename: str = None
        self.output_filename: str = None

        self.vmac: float = None
        self.rotation: float = None
        self.init_guess_elements: list[str] = []
        self.init_guess_elements_path: list[str] = []
        self.input_elements_abundance: list[str] = []
        self.input_elements_abundance_path: list[str] = []

        self.fit_vmic: str = None
        self.vmic_input: bool = None
        self.fit_rotation: bool = None
        self.rotation_input: bool = None
        self.bounds_vmic: list[float] = None
        self.guess_range_vmic: list[float] = None

        self.bounds_teff: list[float] = None
        self.guess_range_teff: list[float] = None

        self.bounds_logg: list[float] = None
        self.guess_range_logg: list[float] = None

        self.bounds_vmac: list[float] = None
        self.bounds_rotation: list[float] = None
        self.bounds_abundance: list[float] = None
        self.bounds_feh: list[float] = None
        self.bounds_doppler: list[float] = None

        self.guess_range_vmac: list[float] = None
        self.guess_range_rotation: list[float] = None
        self.guess_range_abundance: list[float] = None
        self.guess_range_doppler: list[float] = None

        self.fit_teff: bool = None
        self.nelement: int = None

        self.line_begins_sorted: np.ndarray[float] = None
        self.line_ends_sorted: np.ndarray[float] = None
        self.line_centers_sorted: np.ndarray[float] = None

        self.seg_begins: np.ndarray[float] = None
        self.seg_ends: np.ndarray[float] = None

        # dictionaries with either abundance guess or abundance input
        self.init_guess_spectra_dict: dict = None
        self.input_elem_abundance_dict: dict = None

        self.find_upper_limit: bool = False
        self.sigmas_upper_limit: float = 5

        self.find_teff_errors: bool = False
        self.teff_error_sigma: float = 3

        self.find_logg_errors: bool = False
        self.logg_error_sigma: float = 3

        # advanced options
        # scipy xatol and fatol for the minimisation, different methods
        self.xatol_all = 0.0001
        self.fatol_all = 0.00001
        self.xatol_lbl = 0.0001
        self.fatol_lbl = 0.00001
        self.xatol_teff = 0.0001
        self.fatol_teff = 0.00001
        self.xatol_logg = 0.0001
        self.fatol_logg = 0.00001
        self.xatol_vmic = 0.0001
        self.fatol_vmic = 0.00001
        self.maxfev = 50 # scipy maxfev for the minimisation
        self.margin = 3.0
        self.guess_ratio_to_add = 0.1
        # whether to save different parameters
        self.save_original_spectra = True
        self.save_fitted_spectra = True
        self.save_convolved_fitted_spectra = True
        self.save_results = True
        self.save_linemask = True
        self.save_fitlist = True
        self.save_config_file = True
        self.make_output_directory = True
        self.pretrim_linelist = True
        self.lightweight_ts_run = False
        self.compute_blend_spectra = True
        self.sensitivity_abundance_offset = 0.2
        self.just_blend_reduce_abundance = -10

    def load_config(self, check_valid_path=True):
        # if last 3 characters are .cfg then new config file, otherwise old config file
        if self.config_location[-4:] != ".cfg":
            raise ValueError("Old config files are not supported anymore. Please convert to new config file format using older version.")

        super().load_config()

        if check_valid_path:
            self.linemasks_path = self._check_if_path_exists(self.config_parser["MainPaths"]["linemasks_path"])
            self.fitlist_input_path = self._check_if_path_exists(self.config_parser["MainPaths"]["fitlist_input_path"])
        else:
            self.linemasks_path = self.config_parser["MainPaths"]["linemasks_path"]
            self.fitlist_input_path = self.config_parser["MainPaths"]["fitlist_input_path"]
        if self.spectra_input_path is None:
            self.spectra_input_path = self.config_parser["MainPaths"]["spectra_input_path"]

        self.temporary_directory_path = os.path.join(self.config_parser["MainPaths"]["temporary_directory_path"], self.output_folder_title, '')
        self.atmosphere_type = self._validate_string_input(self.config_parser["FittingParameters"]["atmosphere_type"], ["1d", "3d"])
        self.fitting_mode = self._validate_string_input(self.config_parser["FittingParameters"]["fitting_mode"], ["all", "lbl", "teff", "lbl_quick", "vmic", "logg"])
        self.include_molecules = self._convert_string_to_bool(self.config_parser["FittingParameters"]["include_molecules"])
        self.nlte_flag = self._convert_string_to_bool(self.config_parser["FittingParameters"]["nlte"])
        vmic_fitting_mode = self._validate_string_input(self.config_parser["FittingParameters"]["fit_vmic"], ["yes", "no", "input"])
        self.fit_vmic, self.vmic_input = self._get_fitting_mode(vmic_fitting_mode)
        vmac_fitting_mode = self._validate_string_input(self.config_parser["FittingParameters"]["fit_vmac"], ["yes", "no", "input"])
        self.fit_vmac, self.vmac_input = self._get_fitting_mode(vmac_fitting_mode)
        rotation_fitting_mode = self._validate_string_input(self.config_parser["FittingParameters"]["fit_rotation"], ["yes", "no", "input"])
        self.fit_rotation, self.rotation_input = self._get_fitting_mode(rotation_fitting_mode)
        self.elements_to_fit = self._split_string_to_string_list(self.config_parser["FittingParameters"]["element_to_fit"])
        if 'Fe' in self.elements_to_fit:
            self.fit_feh = True
        else:
            self.fit_feh = False

        self.nlte_elements = self._split_string_to_string_list(self.config_parser["FittingParameters"]["nlte_elements"])
        self.linemask_file = self.config_parser["FittingParameters"]["linemask_file"]
        self.wavelength_delta = float(self.config_parser["FittingParameters"]["wavelength_delta"])
        self.segment_size = float(self.config_parser["FittingParameters"]["segment_size"])

        self.experimental_parallelisation = self._convert_string_to_bool(self.config_parser["ExtraParameters"]["experimental_parallelisation"])

        self.input_fitlist_filename = self.config_parser["InputAndOutputFiles"]["input_filename"]
        self.output_filename = self.config_parser["InputAndOutputFiles"]["output_filename"]

        self.resolution = float(self.config_parser["SpectraParameters"]["resolution"])
        self.vmac = float(self.config_parser["SpectraParameters"]["vmac"])
        self.rotation = float(self.config_parser["SpectraParameters"]["rotation"])
        self.init_guess_elements = self._split_string_to_string_list(self.config_parser["SpectraParameters"]["init_guess_elements"])
        self.init_guess_elements_path = self._split_string_to_string_list(self.config_parser["SpectraParameters"]["init_guess_elements_path"])
        self.input_elements_abundance = self._split_string_to_string_list(self.config_parser["SpectraParameters"]["input_elements_abundance"])
        self.input_elements_abundance_path = self._split_string_to_string_list(self.config_parser["SpectraParameters"]["input_elements_abundance_path"])

        self.wavelength_min = float(self.config_parser["ParametersForModeAll"]["wavelength_min"])
        self.wavelength_max = float(self.config_parser["ParametersForModeAll"]["wavelength_max"])

        self.bounds_vmic = self._split_string_to_float_list(self.config_parser["ParametersForModeLbl"]["bounds_vmic"])
        self.guess_range_vmic = self._split_string_to_float_list(self.config_parser["ParametersForModeLbl"]["guess_range_vmic"])
        try:
            self.find_upper_limit = self._convert_string_to_bool(self.config_parser["ParametersForModeLbl"]["find_upper_limit"])
            self.sigmas_upper_limit = float(self.config_parser["ParametersForModeLbl"]["upper_limit_sigma"])
        except KeyError:
            self.find_upper_limit = False
            self.sigmas_upper_limit = 5.0

        self.bounds_teff = self._split_string_to_float_list(self.config_parser["ParametersForModeTeff"]["bounds_teff"])
        self.guess_range_teff = self._split_string_to_float_list(self.config_parser["ParametersForModeTeff"]["guess_range_teff"])
        try:
            self.find_teff_errors = self._convert_string_to_bool(self.config_parser["ParametersForModeTeff"]["find_teff_errors"])
            self.teff_error_sigma = float(self.config_parser["ParametersForModeTeff"]["teff_error_sigma"])
        except KeyError:
            self.find_teff_errors = False
            self.teff_error_sigma = 5.0

        try:
            self.bounds_logg = self._split_string_to_float_list(self.config_parser["ParametersForModeLogg"]["bounds_logg"])
            self.guess_range_logg = self._split_string_to_float_list(self.config_parser["ParametersForModeLogg"]["guess_range_logg"])
            self.find_logg_errors = self._convert_string_to_bool(self.config_parser["ParametersForModeLogg"]["find_logg_errors"])
            self.logg_error_sigma = float(self.config_parser["ParametersForModeLogg"]["logg_error_sigma"])
        except KeyError:
            self.bounds_logg = [-0.5, 5]
            self.guess_range_logg = [-0.5, 0.5]
            self.find_logg_errors = False
            self.logg_error_sigma = 5.0

        self.bounds_vmac = self._split_string_to_float_list(self.config_parser["Bounds"]["bounds_vmac"])
        self.bounds_rotation = self._split_string_to_float_list(self.config_parser["Bounds"]["bounds_rotation"])
        self.bounds_abundance = self._split_string_to_float_list(self.config_parser["Bounds"]["bounds_abundance"])
        self.bounds_feh = self._split_string_to_float_list(self.config_parser["Bounds"]["bounds_feh"])
        self.bounds_doppler = self._split_string_to_float_list(self.config_parser["Bounds"]["bounds_doppler"])

        self.guess_range_vmac = self._split_string_to_float_list(self.config_parser["GuessRanges"]["guess_range_vmac"])
        self.guess_range_rotation = self._split_string_to_float_list(self.config_parser["GuessRanges"]["guess_range_rotation"])
        self.guess_range_abundance = self._split_string_to_float_list(self.config_parser["GuessRanges"]["guess_range_abundance"])
        self.guess_range_doppler = self._split_string_to_float_list(self.config_parser["GuessRanges"]["guess_range_doppler"])

        # advanced options
        try:
            self.xatol_all = float(self.config_parser["AdvancedOptions"]["xatol_all"])
            self.fatol_all = float(self.config_parser["AdvancedOptions"]["fatol_all"])
            self.xatol_lbl = float(self.config_parser["AdvancedOptions"]["xatol_lbl"])
            self.fatol_lbl = float(self.config_parser["AdvancedOptions"]["fatol_lbl"])
            self.xatol_teff = float(self.config_parser["AdvancedOptions"]["xatol_teff"])
            self.fatol_teff = float(self.config_parser["AdvancedOptions"]["fatol_teff"])
            self.xatol_logg = float(self.config_parser["AdvancedOptions"]["xatol_logg"])
            self.fatol_logg = float(self.config_parser["AdvancedOptions"]["fatol_logg"])
            self.xatol_vmic = float(self.config_parser["AdvancedOptions"]["xatol_vmic"])
            self.fatol_vmic = float(self.config_parser["AdvancedOptions"]["fatol_vmic"])
            self.maxfev = int(self.config_parser["AdvancedOptions"]["maxfev"])
            self.margin = float(self.config_parser["AdvancedOptions"]["margin"])
            self.guess_ratio_to_add = float(self.config_parser["AdvancedOptions"]["guess_ratio_to_add"])
            self.save_original_spectra = self._convert_string_to_bool(self.config_parser["AdvancedOptions"]["save_original_spectra"])
            self.save_fitted_spectra = self._convert_string_to_bool(self.config_parser["AdvancedOptions"]["save_fitted_spectra"])
            self.save_convolved_fitted_spectra = self._convert_string_to_bool(self.config_parser["AdvancedOptions"]["save_convolved_fitted_spectra"])
            self.save_results = self._convert_string_to_bool(self.config_parser["AdvancedOptions"]["save_results"])
            self.save_linemask = self._convert_string_to_bool(self.config_parser["AdvancedOptions"]["save_linemask"])
            self.save_fitlist = self._convert_string_to_bool(self.config_parser["AdvancedOptions"]["save_fitlist"])
            self.save_config_file = self._convert_string_to_bool(self.config_parser["AdvancedOptions"]["save_config_file"])
            self.pretrim_linelist = self._convert_string_to_bool(self.config_parser["AdvancedOptions"]["pretrim_linelist"])
            self.lightweight_ts_run = self._convert_string_to_bool(self.config_parser["AdvancedOptions"]["lightweight_ts_run"])
        except KeyError:
            pass

        try:
            self.compute_blend_spectra = self._convert_string_to_bool(self.config_parser["AdvancedOptions"]["compute_blend_spectra"])
            self.sensitivity_abundance_offset = float(self.config_parser["AdvancedOptions"]["sensitivity_abundance_offset"])
            self.just_blend_reduce_abundance = float(self.config_parser["AdvancedOptions"]["just_blend_reduce_abundance"])
        except KeyError:
            pass

    def warn_on_config_issues(self):
        print("\n\nChecking inputs\n")

        if np.size(self.seg_begins) != np.size(self.seg_ends):
            print("Segment beginning and end are not the same length")
        if np.size(self.line_centers_sorted) != np.size(self.line_begins_sorted) or np.size(self.line_centers_sorted) != np.size(self.line_ends_sorted):
            print("Line center, beginning and end are not the same length")
        if self.guess_range_teff[0] > 0:
            print(
                f"You requested your {self.guess_range_teff[0]} to be positive. That will result in the lower "
                f"guess value to be bigger than the expected star temperature. Consider changing the number to negative.")
        if self.guess_range_teff[1] < 0:
            print(
                f"You requested your {self.guess_range_teff[1]} to be negative. That will result in the upper "
                f"guess value to be smaller than the expected star temperature. Consider changing the number to positive.")
        if min(self.guess_range_vmac) < min(self.bounds_vmac) or max(
                self.guess_range_vmac) > max(self.bounds_vmac):
            print(f"You requested your macro bounds as {self.bounds_vmac}, but guesses"
                  f"are {self.guess_range_vmac}, which is outside hard bound range. Consider"
                  f"changing bounds or guesses.")
        if min(self.guess_range_vmic) < min(self.bounds_vmic) or max(
                self.guess_range_vmic) > max(self.bounds_vmic):
            print(f"You requested your micro bounds as {self.bounds_vmic}, but guesses"
                  f"are {self.guess_range_vmic}, which is outside hard bound range. Consider"
                  f"changing bounds or guesses.")
        if min(self.guess_range_abundance) < min(self.bounds_abundance) or max(
                self.guess_range_abundance) > max(self.bounds_abundance):
            print(f"You requested your abundance bounds as {self.bounds_abundance}, but guesses"
                  f"are {self.guess_range_abundance} , which is outside hard bound range. Consider"
                  f"changing bounds or guesses if you fit elements except for Fe.")
        if min(self.guess_range_abundance) < min(self.bounds_feh) or max(
                self.guess_range_abundance) > max(self.bounds_feh):
            print(f"You requested your metallicity bounds as {self.bounds_feh}, but guesses"
                  f"are {self.guess_range_abundance}, which is outside hard bound range. Consider"
                  f"changing bounds or guesses IF YOU FIT METALLICITY.")
        if min(self.guess_range_doppler) < min(self.bounds_doppler) or max(
                self.guess_range_doppler) > max(self.bounds_doppler):
            print(f"You requested your RV bounds as {self.bounds_doppler}, but guesses"
                  f"are {self.guess_range_doppler}, which is outside hard bound range. Consider"
                  f"changing bounds or guesses.")
        if self.rotation < 0:
            print(
                f"Requested rotation of {self.rotation}, which is less than 0. Consider changing it.")
        if self.resolution < 0:
            print(
                f"Requested resolution of {self.resolution}, which is less than 0. Consider changing it.")
        if self.vmac < 0:
            print(
                f"Requested macroturbulence input of {self.vmac}, which is less than 0. Consider changing it if "
                f"you fit it.")
        # check done in tsfitpyconfiguration
        if self.nlte_flag:
            if self.compiler.lower() != "m3dis":
                for file in self.depart_bin_file_dict:
                    if not os.path.isfile(os.path.join(self.departure_file_path,
                                                       self.depart_bin_file_dict[file])):
                        print(
                            f"{self.depart_bin_file_dict[file]} does not exist! Check the spelling or if the file exists")
                for file in self.depart_aux_file_dict:
                    if not os.path.isfile(os.path.join(self.departure_file_path,
                                                       self.depart_aux_file_dict[file])):
                        print(
                            f"{self.depart_aux_file_dict[file]} does not exist! Check the spelling or if the file exists")
            for file in self.model_atom_file_dict:
                if not os.path.isfile(os.path.join(self.model_atoms_path,
                                                   self.model_atom_file_dict[file])):
                    print(
                        f"{self.model_atom_file_dict[file]} does not exist! Check the spelling or if the file exists")

        for line_start, line_end in zip(self.line_begins_sorted,
                                        self.line_ends_sorted):
            index_location = np.where(np.logical_and(self.seg_begins <= line_start,
                                                     line_end <= self.seg_ends))[0]
            if np.size(index_location) > 1:
                print(f"{line_start} {line_end} linemask has more than 1 segment!")
            if np.size(index_location) == 0:
                print(f"{line_start} {line_end} linemask does not have any corresponding segment")

        print(
            "\nDone doing some basic checks. Consider reading the messages above, if there are any. Can be useful if it "
            "crashes.\n\n")

    @staticmethod
    def _get_fitting_mode(fitting_mode: str) -> (bool, bool):
        fit_variable, input_variable = None, None  # both booleans
        if fitting_mode == "Yes" or fitting_mode == "True":
            fit_variable = True
            input_variable = False
        elif fitting_mode == "No" or fitting_mode == "False":
            fit_variable = False
            input_variable = False
        elif fitting_mode == "Input":
            fit_variable = False
            input_variable = True
        else:
            raise ValueError(f"Fitting mode {fitting_mode} not recognized")
        return fit_variable, input_variable

    def validate_input(self, check_valid_path=True):
        super().validate_input(check_valid_path=check_valid_path)

        self.fitting_mode = self.fitting_mode.lower()
        self.fit_vmic = self.fit_vmic
        self.fit_vmac = self.fit_vmac
        self.fit_rotation = self.fit_rotation
        self.elements_to_fit = np.asarray(self.elements_to_fit)
        self.fit_feh = self.fit_feh
        self.rotation = float(self.rotation)
        self.temporary_directory_path = self._find_path_temporary_directory(self.temporary_directory_path)
        self.segment_file = os.path.join(self.temporary_directory_path, "segment_file.txt")

        self.experimental_parallelisation = self.experimental_parallelisation

        self.nelement = len(self.elements_to_fit)

        if self.nlte_flag:
            if self.compiler.lower() == "m3dis":
                if self.iterations_max_precompute > 0 or self.iterations_max > 0:
                    nlte_flag_to_save = "NLTE"
                else:
                    nlte_flag_to_save = "LTE"
            else:
                nlte_flag_to_save = "NLTE"
        else:
            nlte_flag_to_save = "LTE"

        self.output_folder_title = f"{self.output_folder_title}_{nlte_flag_to_save}_{self.elements_to_fit[0]}_{self.atmosphere_type.upper()}"

        self.output_folder_path = os.path.join(self._check_if_path_exists(self.output_folder_path, check_valid_path),
                                               self.output_folder_title)
        self.spectra_input_path = self._check_if_path_exists(self.spectra_input_path, check_valid_path)
        self.line_list_path = self._check_if_path_exists(self.line_list_path, check_valid_path)

        if self.fitting_mode == "teff":
            self.fit_teff = True
        else:
            self.fit_teff = False

        if self.fit_teff:
            self.fit_feh = False

        if self.xatol_all <= 0:
            raise ValueError("xatol_all must be greater than 0")
        if self.fatol_all <= 0:
            raise ValueError("fatol_all must be greater than 0")
        if self.xatol_lbl <= 0:
            raise ValueError("xatol_lbl must be greater than 0")
        if self.fatol_lbl <= 0:
            raise ValueError("fatol_lbl must be greater than 0")
        if self.xatol_teff <= 0:
            raise ValueError("xatol_teff must be greater than 0")
        if self.fatol_teff <= 0:
            raise ValueError("fatol_teff must be greater than 0")
        if self.xatol_logg <= 0:
            raise ValueError("xatol_logg must be greater than 0")
        if self.fatol_logg <= 0:
            raise ValueError("fatol_logg must be greater than 0")
        if self.xatol_vmic <= 0:
            raise ValueError("xatol_vmic must be greater than 0")
        if self.fatol_vmic <= 0:
            raise ValueError("fatol_vmic must be greater than 0")
        if self.maxfev <= 0:
            raise ValueError("maxfev must be greater than 0")

        if not np.any(self.save_original_spectra + self.save_fitted_spectra + self.save_convolved_fitted_spectra + self.save_results + self.save_linemask + self.save_fitlist + self.save_config_file):
            self.make_output_directory = False


class SyntheticSpectraConfig(GenericConfig):
    def __init__(self, config_location: str, output_folder_title: str):
        super().__init__(config_location, output_folder_title)

        self.input_parameter_path: str = None

        self.input_parameters_filename: str = None
        self.save_unnormalised_spectra: bool = None

        # other things not in config
        self.output_folder_path_global: str = None

        self.compute_intensity_flag = False
        self.intensity_angles = [0.010018, 0.052035, 0.124619, 0.222841, 0.340008, 0.468138, 0.598497, 0.722203, 0.830825, 0.916958, 0.974726, 1.000000]

    def load_config(self):
        super().load_config()

        self.input_parameter_path = self.config_parser["MainPaths"]["input_parameter_path"]

        self.atmosphere_type = self._validate_string_input(self.config_parser["AtmosphereParameters"]["atmosphere_type"], ["1d", "3d"])
        self.include_molecules = self._convert_string_to_bool(self.config_parser["AtmosphereParameters"]["include_molecules"])
        self.nlte_flag = self._convert_string_to_bool(self.config_parser["AtmosphereParameters"]["nlte"])
        self.nlte_elements = self._split_string_to_string_list(self.config_parser["AtmosphereParameters"]["nlte_elements"])
        self.wavelength_min = float(self.config_parser["AtmosphereParameters"]["wavelength_min"])
        self.wavelength_max = float(self.config_parser["AtmosphereParameters"]["wavelength_max"])
        self.wavelength_delta = float(self.config_parser["AtmosphereParameters"]["wavelength_delta"])
        self.resolution = float(self.config_parser["AtmosphereParameters"]["resolution"])

        self.save_unnormalised_spectra = self._convert_string_to_bool(self.config_parser["ExtraParameters"]["save_unnormalised_spectra"])

        self.input_parameters_filename = self.config_parser["InputFile"]["input_filename"]

        try:
            self.compute_intensity_flag = self._convert_string_to_bool(self.config_parser["AdvancedOptions"]["compute_intensity_flag"])
        except KeyError:
            pass

        try:
            self.intensity_angles = self._split_string_to_float_list(self.config_parser["AdvancedOptions"]["intensity_angles"])
        except KeyError:
            pass

    def validate_input(self, check_valid_path=True):
        super().validate_input(check_valid_path=check_valid_path)

        self.temporary_directory_path = self._convert_to_absolute_path(self.temporary_directory_path)

        self.output_folder_path_global = self._convert_to_absolute_path(self.output_folder_path)
        self.line_list_path = self._check_if_path_exists(self.line_list_path)

        if self.nlte_flag:
            if self.compiler.lower() == "m3dis":
                if self.iterations_max_precompute > 0 or self.iterations_max > 0:
                    nlte_flag_to_save = "NLTE"
                else:
                    nlte_flag_to_save = "LTE"
            else:
                nlte_flag_to_save = "NLTE"
        else:
            nlte_flag_to_save = "LTE"

        self.output_folder_title = f"{self.output_folder_title}_{nlte_flag_to_save}_{self.input_parameters_filename}"

        self.output_folder_path = os.path.join(self.output_folder_path_global, self.output_folder_title, "")
        self.input_parameter_path = os.path.join(self._check_if_path_exists(self.input_parameter_path), self.input_parameters_filename)