import pandas as pd
import numpy as np
from scripts.solar_abundances import periodic_table, solar_abundances

class SpectraParameters:
    def __init__(self, input_file_path: str, first_row_name: bool):
        # read in the atmosphere grid to compute for the synthetic spectra
        # Read the file
        df = pd.read_csv(input_file_path, delim_whitespace=True, index_col=False)

        # Create a dictionary that maps non-standard names to standard ones
        name_variants = {'vmic': ['vturb', 'vturbulence', 'vmicro', 'vm', 'input_vmicroturb', 'input_vmic'],
                         'rv': ['radvel', 'radialvelocity', 'radial_velocity', "#rv", "#radvel", "#radialvelocity", "#radial_velocity"],
                         'teff': ['temp', 'temperature', 't'],
                         'vmac': ['vmacroturb', 'vmacro', 'vmacro_turb', 'vmacro_turbulence', 'vmacroturbulence', 'input_vmacroturb', 'input_vmacroturbulence'],
                         'logg': ['logg', 'grav'],
                         'feh': ['met', 'fe/h', '[fe/h]', 'feh', 'metallicity', 'metallicity_fe_h', 'metallicity_feh', 'mh', 'm/h', '[m/h]'],
                         'rotation': ['vsini', 'vrot', 'rot', 'vrotini', 'vrotini', 'vrotini'],
                         'specname': ['specname', 'spec_name', 'spectrum_name', 'spectrumname', 'spectrum', 'spectrum_name', 'spectrumname']}

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
            standard_name = name_dict.get(col.lower())
            if standard_name is None:
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
                elif np.size(testing_col) <= 2 and testing_col in periodic_table:
                    # means just elemental abundance perhaps
                    standard_name = f"{testing_col.lower()}"
                    abundances_x_given.append(standard_name)
                    self.abundance_elements_given.append(standard_name.capitalize())
                elif np.size(testing_col) <= 3 and testing_col[1:].capitalize() in periodic_table and testing_col[0].lower() == 'a':
                    # means just elemental abundance perhaps because A(X) is given
                    standard_name = f"{testing_col[1:].lower()}"
                    abundances_x_given.append(standard_name)
                    self.abundance_elements_given.append(standard_name.capitalize())
                else:
                    # could not parse, not element?
                    standard_name = col
                    if col not in name_variants.keys():
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
        # get abundance elements, put in dictionary and then list, where each entry is a dictionary
        abundance_list = []
        for i in range(self.number_of_rows):
            abundance_dict = {}
            for element in self.abundance_elements_given:
                abundance_dict[element] = self.spectra_parameters_df[element][i]
            abundance_list.append(abundance_dict)

        # stack all parameters
        stacked_parameters = np.stack((specname_list, rv_list, teff_list, logg_list, feh_list, vmic_list, vmac_list, rotation_list, abundance_list), axis=1)

        return stacked_parameters

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
        abundance_list = []
        for i in range(self.number_of_rows):
            abundance_dict = {}
            for element in self.abundance_elements_given:
                abundance_dict[element] = self.spectra_parameters_df[element][i]
            abundance_list.append(abundance_dict)

        # stack all parameters
        stacked_parameters = np.stack((specname_list, teff_list, logg_list, feh_list, vmic_list, vmac_list, rotation_list, abundance_list), axis=1)

        return stacked_parameters


    @staticmethod
    def _strip_string(string_to_strip: str) -> str:
        bad_characters = ["[", "]", "/", "\\", "(", ")", "{", "}", "_"]
        for character_to_remove in bad_characters:
            string_to_strip = string_to_strip.replace(character_to_remove, '')
        return string_to_strip

    def __str__(self):
        # print the dataframe
        return self.spectra_parameters_df.to_string()

if __name__ == '__main__':
    fitlist = SpectraParameters('../input_files/fitlist_test', False)
    print(fitlist)
    print(fitlist.get_spectra_parameters_for_grid_generation())