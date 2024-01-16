from __future__ import annotations

import logging
import os
from _warnings import warn
from configparser import ConfigParser
import pandas as pd
import numpy as np
from scripts.solar_abundances import periodic_table, solar_abundances

class SpectraParameters:
    def __init__(self, input_file_path: str, first_row_name: bool):
        # read in the atmosphere grid to compute for the synthetic spectra
        # Read the file
        df = pd.read_csv(input_file_path, delim_whitespace=True, index_col=False, comment=';')

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
        name_variants = {'vmic': ['vturb', 'vturbulence', 'vmicro', 'vm', 'inputvmicroturb', 'inputvmic'],
                         'rv': ['radvel', 'radialvelocity', 'radial_velocity', "rv", "radvel", "radialvelocity", "radialvelocity"],
                         'teff': ['temp', 'temperature', 't'],
                         'vmac': ['vmac', 'vmacroturb', 'vmacro', 'vmacroturb', 'vmacroturbulence', 'vmacroturbulence', 'inputvmacroturb', 'inputvmacroturbulence'],
                         'logg': ['grav'],
                         'feh': ['met', 'metallicity', 'metallicityfeh', 'metallicityfeh', 'mh', 'mh', 'mh'],
                         'rotation': ['vsini', 'vrot', 'rot', 'vrotini', 'vrotini', 'vrotini'],
                         'specname': ['spec_name', 'spectrum_name', 'spectrumname', 'spectrum', 'spectrumname', 'spectrumname'],
                         'resolution': ['res', 'resolution', 'resolvingpower', 'resolvingpower', 'r'],}

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
                    testing_col = self._strip_string(col.lower())
                    ending_element = testing_col[-2:]
                    starting_element = testing_col[:-2].capitalize()
                    if ending_element == "fe" and starting_element in periodic_table:
                        # means X/Fe
                        standard_name = f"{starting_element.lower()}"
                        abundances_xfe_given.append(standard_name)
                        self.abundance_elements_given.append(standard_name.capitalize())
                    elif ending_element[-1] == 'h' and testing_col[:-1].capitalize() in periodic_table:
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
        # get abundance elements, put in dictionary and then list, where each entry is a dictionary
        abundance_list = self._get_abundance_list()

        # stack all parameters
        stacked_parameters = np.stack((specname_list, rv_list, teff_list, logg_list, feh_list, vmic_list, vmac_list, rotation_list, abundance_list, resolution_list), axis=1)

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

if __name__ == '__main__':
    fitlist = SpectraParameters('../input_files/fitlist_test2', False)
    print(fitlist)
    print(fitlist.get_spectra_parameters_for_grid_generation())


class TSFitPyConfig:
    def __init__(self, config_location: str, output_folder_title=None, spectra_location=None):
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

        self.temporary_directory_path: str = None
        self.fitlist_input_path: str = None
        if spectra_location is not None:
            self.spectra_input_path: str = spectra_location
        else:
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
        self.experimental_parallelisation: bool = False
        self.cluster_name: str = None

        self.input_fitlist_filename: str = None
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

        # dictionaries with either abundance guess or abundance input
        self.init_guess_spectra_dict: dict = None
        self.input_elem_abundance_dict: dict = None

        self.old_global_temporary_directory = None  # used only to convert old config to new config
        self.old_output_folder_path_global = None  # used only to convert old config to new config
        self.old_turbospectrum_global_path = None  # used only to convert old config to new config

        self.find_upper_limit: bool = False
        self.sigmas_upper_limit: float = 5

        self.find_teff_errors: bool = False
        self.teff_error_sigma: float = 3

        self.find_logg_errors: bool = False
        self.logg_error_sigma: float = 3

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
        # Value of lpoint for turbospectrum in spectrum.inc file
        self.lpoint_turbospectrum = 1000000
        # m3dis parameters
        self.m3dis_python_package_name = "m3dis"
        self.margin = 3.0

    def load_config(self, check_valid_path=True):
        # if last 3 characters are .cfg then new config file, otherwise old config file
        if self.config_location[-4:] == ".cfg":
            self.load_new_config(check_valid_path=check_valid_path)
        else:
            self.load_old_config()

    def load_old_config(self):
        model_atom_file = []
        init_guess_elements = []
        input_elem_abundance = []
        model_atom_file_input_elem = []
        elements_to_do_in_nlte = []
        self.bounds_rotation = [0, 15] # default value
        self.bounds_vmic = [0, 5] # default value
        self.bounds_vmac = [0, 15] # default value
        self.bounds_abundance = [-40, 40] # default value
        self.bounds_feh = [-5, 1] # default value
        self.bounds_doppler = [-2, 2] # default value
        self.guess_range_rotation = [0, 15] # default value
        self.guess_range_vmic = [0.8, 2.0] # default value
        self.guess_range_vmac = [0, 15] # default value
        self.guess_range_abundance = [-2, 2] # default value
        self.guess_range_doppler = [2, -2] # default value
        self.bounds_teff = [2000, 8000] # default value
        self.guess_range_teff = [-250, 250] # default value

        self.spectral_code_path = "../turbospectrum/"
        self.cluster_name = "None"
        self.debug_mode = 0

        #nlte_config_outdated = False
        #need_to_add_new_nlte_config = True  # only if nlte_config_outdated == True

        #initial_guess_string = None

        with open(self.config_location) as fp:
            line = fp.readline()
            while line:
                if len(line) > 1:
                    fields = line.strip().split()
                    field_name = fields[0].lower()
                    if field_name == "title":
                        self.output_folder_title = fields[2]
                    if field_name == "interpol_path":
                        self.interpolators_path = fields[2]
                    if field_name == "line_list_path":
                        self.line_list_path = fields[2]
                    if field_name == "model_atmosphere_grid_path_1d":
                        self.model_atmosphere_grid_path_1d = fields[2]
                    if field_name == "model_atmosphere_grid_path_3d":
                        self.model_atmosphere_grid_path_3d = fields[2]
                    if field_name == "model_atom_path":
                        self.model_atoms_path = fields[2]
                    if field_name == "departure_file_path":
                        self.departure_file_path = fields[2]
                        self.departure_file_config_path = os.path.join(self.departure_file_path, "nlte_filenames.cfg")
                    if field_name == "output_folder":
                        self.output_folder_path = fields[2]
                    if field_name == "linemask_file_folder_location":
                        self.linemasks_path = self._check_if_path_exists(fields[2])
                    #if field_name == "segment_file_folder_location":
                        #self.segment_file_og = fields[2]
                    if field_name == "spec_input_path":
                        if self.spectra_input_path is None:
                            self.spectra_input_path = fields[2]
                    if field_name == "fitlist_input_folder":
                        self.fitlist_input_path = self._check_if_path_exists(fields[2])
                    if field_name == "turbospectrum_compiler":
                        self.compiler = fields[2]
                    if field_name == "atmosphere_type":
                        self.atmosphere_type = fields[2].lower()
                    if field_name == "mode":
                        self.fitting_mode = fields[2].lower()
                    if field_name == "include_molecules":
                        # spectra_to_fit.include_molecules = fields[2]
                        if fields[2].lower() in ["yes", "true"]:
                            self.include_molecules = True
                        elif fields[2].lower() in ["no", "false"]:
                            self.include_molecules = False
                        else:
                            raise ValueError(f"Expected True/False for including molecules, got {fields[2]}")
                    if field_name == "nlte":
                        nlte_flag = fields[2].lower()
                        if nlte_flag in ["yes", "true"]:
                            self.nlte_flag = True
                        elif nlte_flag in ["no", "false"]:
                            self.nlte_flag = False
                        else:
                            raise ValueError(f"Expected True/False for nlte flag, got {fields[2]}")
                    if field_name == "fit_microturb":  # Yes No Input
                        self.fit_vmic = fields[2]
                        if self.fit_vmic not in ["Yes", "No", "Input"]:
                            raise ValueError(f"Expected Yes/No/Input for micro fit, got {fields[2]}")
                    if field_name == "fit_macroturb":  # Yes No Input
                        if fields[2].lower() in ["yes", "true"]:
                            self.fit_vmac = True
                            self.vmac_input = False
                        elif fields[2].lower() in ["no", "false"]:
                            self.fit_vmac = False
                            self.vmac_input = False
                        elif fields[2].lower() == "input":
                            self.fit_vmac = False
                            self.vmac_input = True
                        else:
                            raise ValueError(f"Expected Yes/No/Input for macro fit, got {fields[2]}")
                    if field_name == "fit_rotation":
                        if fields[2].lower() in ["yes", "true"]:
                            self.fit_rotation = True
                        elif fields[2].lower() in ["no", "false"]:
                            self.fit_rotation = False
                        else:
                            raise ValueError(f"Expected Yes/No for rotation fit, got {fields[2]}")
                    if field_name == "element":
                        elements_to_fit = []
                        for i in range(len(fields) - 2):
                            elements_to_fit.append(fields[2 + i])
                        self.elements_to_fit = np.asarray(elements_to_fit)
                        if 'Fe' in self.elements_to_fit:
                            self.fit_feh = True
                        else:
                            self.fit_feh = False
                        """if "Fe" in elements_to_fit:
                            spectra_to_fit.fit_feh = True
                        else:
                            Spectra.fit_feh = False
                        Spectra.nelement = len(Spectra.elem_to_fit)"""
                    if field_name == "linemask_file":
                        self.linemask_file = fields[2]
                    #if field_name == "segment_file":
                        #self.segment_file = fields[2]

                    if field_name == "model_atom_file":
                        self.oldconfig_nlte_config_outdated = True
                        for i in range(2, len(fields)):
                            model_atom_file.append(fields[i])
                        self.oldconfig_model_atom_file = model_atom_file
                    if field_name == "input_elem_model_atom_file":
                        self.oldconfig_nlte_config_outdated = True
                        for i in range(2, len(fields)):
                            model_atom_file_input_elem.append(fields[i])
                        self.oldconfig_model_atom_file_input_elem = model_atom_file_input_elem
                    if field_name == "nlte_elements":
                        self.oldconfig_need_to_add_new_nlte_config = False
                        for i in range(len(fields) - 2):
                            elements_to_do_in_nlte.append(fields[2 + i])
                        self.nlte_elements = elements_to_do_in_nlte
                    if field_name == "wavelength_minimum":
                        self.wavelength_min = float(fields[2])
                    if field_name == "wavelength_maximum":
                        self.wavelength_max = float(fields[2])
                    if field_name == "wavelength_delta":
                        self.wavelength_delta = float(fields[2])
                    if field_name == "resolution":
                        self.resolution = float(fields[2])
                    if field_name == "macroturbulence":
                        self.vmac = float(fields[2])
                    if field_name == "rotation":
                        self.rotation = float(fields[2])
                    if field_name == "temporary_directory":
                        temp_directory = fields[2]
                        self.old_global_temporary_directory = os.path.join(".", temp_directory, "")
                        temp_directory = os.path.join(os.path.join("..", temp_directory, ""), self.output_folder_title, '')
                        self.temporary_directory_path = os.path.join("..", temp_directory, "")
                    if field_name == "input_file":
                        self.input_fitlist_filename = fields[2]
                    if field_name == "output_file":
                        self.output_filename = fields[2]
                    if field_name == "workers":
                        workers = int(fields[2])  # should be the same as cores; use value of 1 if you do not want to use multithprocessing
                        self.number_of_cpus = workers
                    if field_name == "init_guess_elem":
                        init_guess_elements = []
                        for i in range(len(fields) - 2):
                            init_guess_elements.append(fields[2 + i])
                        self.init_guess_elements = np.asarray(init_guess_elements)
                    if field_name == "init_guess_elem_location":
                        init_guess_elements_location = []
                        for i in range(len(init_guess_elements)):
                            init_guess_elements_location.append(fields[2 + i])
                        self.init_guess_elements_path = np.asarray(init_guess_elements_location)
                    if field_name == "input_elem_abundance":
                        input_elem_abundance = []
                        for i in range(len(fields) - 2):
                            input_elem_abundance.append(fields[2 + i])
                        self.input_elements_abundance = np.asarray(input_elem_abundance)
                    if field_name == "input_elem_abundance_location":
                        input_elem_abundance_location = []
                        for i in range(len(input_elem_abundance)):
                            input_elem_abundance_location.append(fields[2 + i])
                        self.input_elements_abundance_path = np.asarray(input_elem_abundance_location)
                    if field_name == "bounds_macro":
                        self.bounds_vmac = [min(float(fields[2]), float(fields[3])), max(float(fields[2]), float(fields[3]))]
                    if field_name == "bounds_rotation":
                        self.bounds_rotation = [min(float(fields[2]), float(fields[3])), max(float(fields[2]), float(fields[3]))]
                    if field_name == "bounds_micro":
                        self.bounds_vmic = [min(float(fields[2]), float(fields[3])), max(float(fields[2]), float(fields[3]))]
                    if field_name == "bounds_abund":
                        self.bounds_abundance = [min(float(fields[2]), float(fields[3])), max(float(fields[2]), float(fields[3]))]
                    if field_name == "bounds_met":
                        self.bounds_feh = [min(float(fields[2]), float(fields[3])), max(float(fields[2]), float(fields[3]))]
                    if field_name == "bounds_teff":
                        self.bounds_teff = [min(float(fields[2]), float(fields[3])),  max(float(fields[2]), float(fields[3]))]
                    if field_name == "bounds_doppler":
                        self.bounds_doppler = [min(float(fields[2]), float(fields[3])), max(float(fields[2]), float(fields[3]))]
                    if field_name == "guess_range_microturb":
                        self.guess_range_vmic = [min(float(fields[2]), float(fields[3])), max(float(fields[2]), float(fields[3]))]
                    if field_name == "guess_range_macroturb":
                        self.guess_range_vmac = [min(float(fields[2]), float(fields[3])), max(float(fields[2]), float(fields[3]))]
                    if field_name == "guess_range_rotation":
                        self.guess_range_rotation = [min(float(fields[2]), float(fields[3])), max(float(fields[2]), float(fields[3]))]
                    if field_name == "guess_range_abundance":
                        self.guess_range_abundance = [min(float(fields[2]), float(fields[3])), max(float(fields[2]), float(fields[3]))]
                    if field_name == "guess_range_rv":
                        self.guess_range_doppler = [min(float(fields[2]), float(fields[3])), max(float(fields[2]), float(fields[3]))]
                    if field_name == "guess_range_teff":
                        self.guess_range_teff = [min(float(fields[2]), float(fields[3])), max(float(fields[2]), float(fields[3]))]
                    if field_name == "debug":
                        self.debug_mode = int(fields[2])
                    if field_name == "experimental":
                        if fields[2].lower() == "true" or fields[2].lower() == "yes":
                            self.experimental_parallelisation = True
                        else:
                            self.experimental_parallelisation = False
                line = fp.readline()

    def load_new_config(self, check_valid_path=True):
        # check if the config file exists
        if not os.path.isfile(self.config_location):
            raise ValueError(f"The configuration file {self.config_location} does not exist.")

        # read the configuration file
        self.config_parser.read(self.config_location)
        # intel or gnu compiler
        self.compiler = self._validate_string_input(self.config_parser["turbospectrum_compiler"]["compiler"], ["intel", "gnu", "m3dis"])
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

        self.debug_mode = int(self.config_parser["ExtraParameters"]["debug_mode"])
        self.number_of_cpus = int(self.config_parser["ExtraParameters"]["number_of_cpus"])
        self.experimental_parallelisation = self._convert_string_to_bool(self.config_parser["ExtraParameters"]["experimental_parallelisation"])
        self.cluster_name = self.config_parser["ExtraParameters"]["cluster_name"]

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
            self.lpoint_turbospectrum = int(self.config_parser["AdvancedOptions"]["lpoint_turbospectrum"])
            self.m3dis_python_package_name = self.config_parser["AdvancedOptions"]["m3dis_python_package_name"]
            self.margin = float(self.config_parser["AdvancedOptions"]["margin"])
        except KeyError:
            pass


    @staticmethod
    def _get_fitting_mode(fitting_mode: str):
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

    def convert_old_config(self):
        self.config_parser.add_section("turbospectrum_compiler")
        self.config_parser["turbospectrum_compiler"]["compiler"] = self.compiler

        self.config_parser.add_section("MainPaths")
        self.config_parser["MainPaths"]["code_path"] = self.old_turbospectrum_global_path
        self.config_parser["MainPaths"]["interpolators_path"] = self.interpolators_path
        self.config_parser["MainPaths"]["line_list_path"] = self.line_list_path
        self.config_parser["MainPaths"]["model_atmosphere_grid_path_1d"] = self.model_atmosphere_grid_path_1d
        self.config_parser["MainPaths"]["model_atmosphere_grid_path_3d"] = self.model_atmosphere_grid_path_3d
        self.config_parser["MainPaths"]["model_atoms_path"] = self.model_atoms_path
        self.config_parser["MainPaths"]["departure_file_path"] = self.departure_file_path
        self.config_parser["MainPaths"]["departure_file_config_path"] = self.departure_file_config_path
        self.config_parser["MainPaths"]["output_path"] = self.old_output_folder_path_global
        self.config_parser["MainPaths"]["linemasks_path"] = self.linemasks_path
        self.config_parser["MainPaths"]["spectra_input_path"] = self.spectra_input_path
        self.config_parser["MainPaths"]["fitlist_input_path"] = self.fitlist_input_path
        self.config_parser["MainPaths"]["temporary_directory_path"] = self.old_global_temporary_directory

        self.config_parser.add_section("FittingParameters")
        self.config_parser["FittingParameters"]["atmosphere_type"] = self.atmosphere_type.upper()
        self.config_parser["FittingParameters"]["fitting_mode"] = self.fitting_mode
        self.config_parser["FittingParameters"]["include_molecules"] = str(self.include_molecules)
        self.config_parser["FittingParameters"]["nlte"] = str(self.nlte_flag)
        self.config_parser["FittingParameters"]["fit_vmic"] = self.fit_vmic
        if self.vmac_input:
            vmac_fitting_mode = "Input"
        elif self.fit_vmac:
            vmac_fitting_mode = "Yes"
        else:
            vmac_fitting_mode = "No"
        self.config_parser["FittingParameters"]["fit_vmac"] = vmac_fitting_mode
        if self.fit_rotation:
            rotation_fitting_mode = "Input"
        elif self.fit_rotation:
            rotation_fitting_mode = "Yes"
        else:
            rotation_fitting_mode = "No"
        self.config_parser["FittingParameters"]["fit_rotation"] = rotation_fitting_mode
        self.config_parser["FittingParameters"]["element_to_fit"] = self._convert_list_to_str(self.elements_to_fit)

        nlte_elements_to_write = []
        if self.oldconfig_need_to_add_new_nlte_config:
            for element in self.oldconfig_model_atom_file + self.oldconfig_model_atom_file_input_elem:
                if ".ba" in element:
                    nlte_elements_to_write.append("Ba")
                if ".ca" in element:
                    nlte_elements_to_write.append("Ca")
                if ".co" in element:
                    nlte_elements_to_write.append("Co")
                if ".fe" in element:
                    nlte_elements_to_write.append("Fe")
                if ".h" in element:
                    nlte_elements_to_write.append("H")
                if ".mg" in element:
                    nlte_elements_to_write.append("Mg")
                if ".mn" in element:
                    nlte_elements_to_write.append("Mn")
                if ".na" in element:
                    nlte_elements_to_write.append("Na")
                if ".ni" in element:
                    nlte_elements_to_write.append("Ni")
                if ".o" in element:
                    nlte_elements_to_write.append("O")
                if ".si" in element:
                    nlte_elements_to_write.append("Si")
                if ".sr" in element:
                    nlte_elements_to_write.append("Sr")
                if ".ti" in element:
                    nlte_elements_to_write.append("Ti")
                if ".y" in element:
                    nlte_elements_to_write.append("Y")
        else:
            nlte_elements_to_write = self.nlte_elements
        self.config_parser["FittingParameters"]["nlte_elements"] = self._convert_list_to_str(nlte_elements_to_write)
        self.config_parser["FittingParameters"]["linemask_file"] = self.linemask_file
        self.config_parser["FittingParameters"]["wavelength_delta"] = str(self.wavelength_delta)
        self.config_parser["FittingParameters"]["segment_size"] = str(self.segment_size)

        self.config_parser.add_section("ExtraParameters")
        self.config_parser["ExtraParameters"]["debug_mode"] = str(self.debug_mode)
        self.config_parser["ExtraParameters"]["number_of_cpus"] = str(self.number_of_cpus)
        self.config_parser["ExtraParameters"]["experimental_parallelisation"] = str(self.experimental_parallelisation)
        self.config_parser["ExtraParameters"]["cluster_name"] = self.cluster_name

        self.config_parser.add_section("InputAndOutputFiles")
        self.config_parser["InputAndOutputFiles"]["input_filename"] = self.input_fitlist_filename
        self.config_parser["InputAndOutputFiles"]["output_filename"] = self.output_filename

        self.config_parser.add_section("SpectraParameters")
        self.config_parser["SpectraParameters"]["resolution"] = str(self.resolution)
        self.config_parser["SpectraParameters"]["vmac"] = str(self.vmac)
        self.config_parser["SpectraParameters"]["rotation"] = str(self.rotation)
        self.config_parser["SpectraParameters"]["init_guess_elements"] = self._convert_list_to_str(self.init_guess_elements)
        self.config_parser["SpectraParameters"]["init_guess_elements_path"] = self._convert_list_to_str(self.init_guess_elements_path)
        self.config_parser["SpectraParameters"]["input_elements_abundance"] = self._convert_list_to_str(self.input_elements_abundance)
        self.config_parser["SpectraParameters"]["input_elements_abundance_path"] = self._convert_list_to_str(self.input_elements_abundance_path)

        self.config_parser.add_section("ParametersForModeAll")
        self.config_parser["ParametersForModeAll"]["wavelength_min"] = str(self.wavelength_min)
        self.config_parser["ParametersForModeAll"]["wavelength_max"] = str(self.wavelength_max)

        self.config_parser.add_section("ParametersForModeLbl")
        self.config_parser["ParametersForModeLbl"]["bounds_vmic"] = self._convert_list_to_str(self.bounds_vmic)
        self.config_parser["ParametersForModeLbl"]["guess_range_vmic"] = self._convert_list_to_str(self.guess_range_vmic)
        self.config_parser["ParametersForModeLbl"]["find_upper_limit"] = 'False'
        self.config_parser["ParametersForModeLbl"]["upper_limit_sigma"] = '5.0'

        self.config_parser.add_section("ParametersForModeTeff")
        self.config_parser["ParametersForModeTeff"]["bounds_teff"] = self._convert_list_to_str(self.bounds_teff)
        self.config_parser["ParametersForModeTeff"]["guess_range_teff"] = self._convert_list_to_str(self.guess_range_teff)
        self.config_parser["ParametersForModeTeff"]["find_teff_errors"] = 'False'
        self.config_parser["ParametersForModeTeff"]["teff_error_sigma"] = '1.0'

        self.config_parser.add_section("Bounds")
        self.config_parser["Bounds"]["bounds_vmac"] = self._convert_list_to_str(self.bounds_vmac)
        self.config_parser["Bounds"]["bounds_rotation"] = self._convert_list_to_str(self.bounds_rotation)
        self.config_parser["Bounds"]["bounds_abundance"] = self._convert_list_to_str(self.bounds_abundance)
        self.config_parser["Bounds"]["bounds_feh"] = self._convert_list_to_str(self.bounds_feh)
        self.config_parser["Bounds"]["bounds_doppler"] = self._convert_list_to_str(self.bounds_doppler)

        self.config_parser.add_section("GuessRanges")
        self.config_parser["GuessRanges"]["guess_range_vmac"] = self._convert_list_to_str(self.guess_range_vmac)
        self.config_parser["GuessRanges"]["guess_range_rotation"] = self._convert_list_to_str(self.guess_range_rotation)
        self.config_parser["GuessRanges"]["guess_range_abundance"] = self._convert_list_to_str(self.guess_range_abundance)
        self.config_parser["GuessRanges"]["guess_range_doppler"] = self._convert_list_to_str(self.guess_range_doppler)

        if self.config_location[-4:] == ".txt":
            converted_config_location = f"{self.config_location[:-4]}"
        else:
            converted_config_location = f"{self.config_location}"

        print("\n\nConverting old config into new one")

        while os.path.exists(f"{converted_config_location}.cfg"):
            print(f"{converted_config_location}.cfg already exists trying {converted_config_location}0.cfg")
            converted_config_location = f"{converted_config_location}0"
        converted_config_location = f"{converted_config_location}.cfg"

        with open(converted_config_location, "w") as new_config_file:
            new_config_file.write(f"# Converted from old file {self.config_location} to a new format\n")
            self.config_parser.write(new_config_file)

        print(f"Converted old config file into new one and save at {converted_config_location}\n\n")
        warn(f"Converted old config file into new one and save at {converted_config_location}", DeprecationWarning, stacklevel=2)

    def validate_input(self, check_valid_path=True):
        self.departure_file_config_path = self._check_if_file_exists(self.departure_file_config_path, check_valid_path=check_valid_path)

        self.atmosphere_type = self.atmosphere_type.upper()
        self.fitting_mode = self.fitting_mode.lower()
        self.include_molecules = self.include_molecules
        if len(self.nlte_elements) == 0 and self.nlte_flag:
            print("\nNo NLTE elements were provided, setting NLTE flag to False!!\n")
            self.nlte_flag = False
        else:
            self.nlte_flag = self.nlte_flag
        self.fit_vmic = self.fit_vmic
        self.fit_vmac = self.fit_vmac
        self.fit_rotation = self.fit_rotation
        self.elements_to_fit = np.asarray(self.elements_to_fit)
        self.fit_feh = self.fit_feh
        self.wavelength_min = float(self.wavelength_min)
        self.wavelength_max = float(self.wavelength_max)
        self.wavelength_delta = float(self.wavelength_delta)
        self.resolution = float(self.resolution)
        self.rotation = float(self.rotation)
        self.temporary_directory_path = self._find_path_temporary_directory(self.temporary_directory_path)
        self.number_of_cpus = int(self.number_of_cpus)

        self.segment_file = os.path.join(self.temporary_directory_path, "segment_file.txt")

        self.debug_mode = self.debug_mode
        if self.debug_mode >= 1:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.WARNING)
        self.experimental_parallelisation = self.experimental_parallelisation

        self.nelement = len(self.elements_to_fit)

        if self.spectral_code_path is None:
            self.spectral_code_path = "../turbospectrum/"
        self.old_turbospectrum_global_path = os.path.join(os.getcwd(), self._check_if_path_exists(self.spectral_code_path, check_valid_path))
        if self.compiler.lower() == "intel":
            self.spectral_code_path = os.path.join(os.getcwd(), self._check_if_path_exists(self.spectral_code_path, check_valid_path),
                                                   "exec", "")
        elif self.compiler.lower() == "gnu":
            self.spectral_code_path = os.path.join(os.getcwd(), self._check_if_path_exists(self.spectral_code_path, check_valid_path),
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
        self.old_output_folder_path_global = self._check_if_path_exists(self.output_folder_path, check_valid_path)

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

        self.output_folder_title = f"{self.output_folder_title}_{nlte_flag_to_save}_{self._convert_list_to_str(self.elements_to_fit).replace(' ', '')}_{self.atmosphere_type.upper()}"

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
        if self.lpoint_turbospectrum <= 0:
            raise ValueError("lpoint_turbospectrum must be greater than 0")

    def load_spectra_config(self, spectra_object):
        # not used anymore
        spectra_object.atmosphere_type = self.atmosphere_type
        spectra_object.fitting_mode = self.fitting_mode
        spectra_object.include_molecules = self.include_molecules
        spectra_object.nlte_flag = self.nlte_flag

        # TODO: redo as booleans instead of strings
        if self.fit_vmic:
            spectra_object.fit_vmic = "Yes"
        elif self.vmic_input:
            spectra_object.fit_vmic = "Input"
        else:
            spectra_object.fit_vmic = "No"

        spectra_object.fit_vmac = self.fit_vmac
        spectra_object.fit_rotation = self.fit_rotation
        spectra_object.input_vmic = self.vmic_input
        spectra_object.input_vmac = self.vmac_input
        spectra_object.input_rotation = self.rotation_input
        spectra_object.elem_to_fit = self.elements_to_fit
        spectra_object.fit_feh = self.fit_feh
        spectra_object.lmin = self.wavelength_min
        spectra_object.lmax = self.wavelength_max
        spectra_object.ldelta = self.wavelength_delta
        spectra_object.resolution = self.resolution
        spectra_object.rotation = self.rotation
        spectra_object.vmac = self.vmac
        spectra_object.global_temp_dir = self.temporary_directory_path
        spectra_object.dask_workers = self.number_of_cpus
        spectra_object.bound_min_vmac = self.bounds_rotation[0]
        spectra_object.bound_max_vmac = self.bounds_rotation[1]
        spectra_object.bound_min_rotation = self.bounds_rotation[0]
        spectra_object.bound_max_rotation = self.bounds_rotation[1]
        spectra_object.bound_min_vmic = self.bounds_vmic[0]
        spectra_object.bound_max_vmic = self.bounds_vmic[1]
        spectra_object.bound_min_abund = self.bounds_abundance[0]
        spectra_object.bound_max_abund = self.bounds_abundance[1]
        spectra_object.bound_min_feh = self.bounds_feh[0]
        spectra_object.bound_max_feh = self.bounds_feh[1]
        spectra_object.bound_min_teff = self.bounds_teff[0]
        spectra_object.bound_max_teff = self.bounds_teff[1]
        spectra_object.bound_min_doppler = self.bounds_doppler[0]
        spectra_object.bound_max_doppler = self.bounds_doppler[1]
        spectra_object.guess_min_vmic = self.guess_range_vmic[0]
        spectra_object.guess_max_vmic = self.guess_range_vmic[1]
        spectra_object.guess_min_vmac = self.guess_range_rotation[0]
        spectra_object.guess_max_vmac = self.guess_range_rotation[1]
        spectra_object.guess_min_rotation = self.guess_range_rotation[0]
        spectra_object.guess_max_rotation = self.guess_range_rotation[1]
        spectra_object.guess_min_abund = self.guess_range_abundance[0]
        spectra_object.guess_max_abund = self.guess_range_abundance[1]
        spectra_object.guess_min_doppler = self.guess_range_doppler[0]
        spectra_object.guess_max_doppler = self.guess_range_doppler[1]
        spectra_object.guess_plus_minus_neg_teff = self.guess_range_teff[0]
        spectra_object.guess_plus_minus_pos_teff = self.guess_range_teff[1]
        spectra_object.debug_mode = self.debug_mode
        spectra_object.experimental_parallelisation = self.experimental_parallelisation

        spectra_object.nelement = self.nelement
        spectra_object.spectral_code_path = self.spectral_code_path
        spectra_object.compiler = self.compiler

        spectra_object.interpol_path = self.interpolators_path

        spectra_object.model_atmosphere_grid_path = self.model_atmosphere_grid_path
        spectra_object.model_atmosphere_list = self.model_atmosphere_list

        spectra_object.model_atom_path = self.model_atoms_path
        spectra_object.departure_file_path = self.departure_file_path
        spectra_object.output_folder = self.output_folder_path
        spectra_object.spec_input_path = self.spectra_input_path

        spectra_object.fit_teff = self.fit_teff

        spectra_object.line_begins_sorted = self.line_begins_sorted
        spectra_object.line_ends_sorted = self.line_ends_sorted
        spectra_object.line_centers_sorted = self.line_centers_sorted

        spectra_object.linemask_file = self.linemask_file
        spectra_object.segment_file = self.segment_file
        spectra_object.seg_begins = self.seg_begins
        spectra_object.seg_ends = self.seg_ends

        spectra_object.depart_bin_file_dict = self.depart_bin_file_dict
        spectra_object.depart_aux_file_dict = self.depart_aux_file_dict
        spectra_object.model_atom_file_dict = self.model_atom_file_dict
        spectra_object.aux_file_length_dict = self.aux_file_length_dict
        spectra_object.ndimen = self.ndimen

        spectra_object.model_temperatures = self.model_temperatures
        spectra_object.model_logs = self.model_logs
        spectra_object.model_mets = self.model_mets
        spectra_object.marcs_value_keys = self.marcs_value_keys
        spectra_object.marcs_models = self.marcs_models
        spectra_object.marcs_values = self.marcs_values

        spectra_object.init_guess_dict = self.init_guess_spectra_dict
        spectra_object.input_elem_abundance = self.input_elem_abundance_dict

        spectra_object.find_upper_limit = self.find_upper_limit
        spectra_object.sigmas_upper_limit = self.sigmas_upper_limit
        spectra_object.find_teff_errors = self.find_teff_errors
        spectra_object.teff_error_sigma = self.teff_error_sigma

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
