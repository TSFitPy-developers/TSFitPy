import pandas as pd
import numpy as np
from scripts.solar_abundances import periodic_table

class FitList:
    def __init__(self, input_file_path: str, first_row_name: bool):
        # read in the atmosphere grid to compute for the synthetic spectra
        # Read the file
        df = pd.read_csv(input_file_path, delim_whitespace=True)

        # Create a dictionary that maps non-standard names to standard ones
        name_variants = {'vmic': ['vturb', 'vturbulence', 'vmicro', 'vm', 'input_vmicroturb', 'input_vmic'],
                         'rv': ['radvel', 'radialvelocity', 'radial_velocity'],
                         'teff': ['temp', 'temperature', 't'],
                         'vmac': ['vmacroturb', 'vmacro', 'vmacro_turb', 'vmacro_turbulence', 'vmacroturbulence', 'input_vmacroturb', 'input_vmacroturbulence'],
                         'logg': ['logg', 'grav'],
                         'feh': ['met', 'fe/h', '[fe/h]', 'feh', 'metallicity', 'metallicity_fe_h', 'metallicity_feh', 'mh', 'm/h', '[m/h]'],
                         'rotation': ['vsini', 'vrot', 'rot', 'vrotini', 'vrotini', 'vrotini']}

        # Reverse the dictionary: map variants to standard names
        name_dict = {variant: standard for standard, variants in name_variants.items() for variant in variants}

        abundances_xfe_given = []
        abundances_xh_given = []
        abundances_x_given = []

        for col in df.columns:
            # Replace the column name if it's in the dictionary, otherwise leave it unchanged
            standard_name = name_dict.get(col.lower())
            if standard_name is None:
                testing_col = self._strip_string(col.lower())
                ending_element = testing_col[-2:]
                starting_element = testing_col[:-2].capitalize()
                if ending_element == "fe" and starting_element in periodic_table:
                    # means X/Fe
                    standard_name = f'{starting_element}fe'
                    abundances_xfe_given.append(standard_name)
                elif ending_element[-1] == 'h' and starting_element[:-1] in periodic_table:
                    # means X/H
                    standard_name = f'{starting_element[:-1]}h'
                    abundances_xh_given.append(standard_name)
                elif np.size(testing_col) <= 2 and testing_col in periodic_table:
                    # means just elemental abundance perhaps
                    standard_name = testing_col
                    abundances_x_given.append(standard_name)
                elif:

                else:
                    # could not parse, not element?
                    standard_name = col
            df.rename(columns={col: standard_name}, inplace=True)

        #xfe = xh - feh

        # convert xh to xfe
        for element in abundances_xh_given:
            df[element] = df[element] - df['feh']

        # convert A(X) to xfe
        for element in abundances_x_given

        # make all columns lower
        df.columns = df.columns.str.lower()

        if first_row_name:
            # replace first column name with 'name'
            df.rename(columns={df.columns[0]: 'specname'}, inplace=True)

        self.spectra_parameters_df = df

        # get amount of rows
        self.number_of_rows = len(self.spectra_parameters_df.index)



    def get_spectra_parameters_for_fit(self) -> np.ndarray:
        """
        returns spectra parameters as a numpy array, where each entry is:
        specname, rv, teff, logg, met, vmic, vmac, input_abundance_dict
        :return:  [[specname, rv, teff, logg, feh, vmic, vmac, input_abundance_dict], ...]
        """

        specname_list = self.spectra_parameters_df['specname'].values
        rv_list = self.spectra_parameters_df['rv'].values
        teff_list = self.spectra_parameters_df['teff'].values
        feh_list = self.spectra_parameters_df['feh'].values
        vmic_list = self.spectra_parameters_df['vmic'].values
        vmac_list = self.spectra_parameters_df['vmac'].values

        stacked_parameters = np.stack((specname_list, rv_list, teff_list, feh_list, vmic_list, vmac_list), axis=1)

        return stacked_parameters


    @staticmethod
    def _strip_string(string_to_strip: str) -> str:
        bad_characters = ["[", "]", "/", "\\"]
        for character_to_remove in bad_characters:
            string_to_strip = string_to_strip.replace(character_to_remove, '')
        return string_to_strip

    def __str__(self):
        # print the dataframe
        return self.spectra_parameters_df.to_string()

if __name__ == '__main__':
    fitlist = FitList('../input_files/fitlist', True)
    print(fitlist)
    print(fitlist.get_spectra_parameters_for_fit())