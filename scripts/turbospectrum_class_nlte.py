from __future__ import annotations

import subprocess
import os
from os import path as os_path
import glob
from operator import itemgetter
from typing import Tuple

import numpy as np
import math
import logging

from .auxiliary_functions import closest_available_value
from .solar_abundances import solar_abundances, periodic_table, molecules_atomic_number
from .solar_isotopes import solar_isotopes
from .synthetic_code_class import SyntheticSpectrumGenerator


class TurboSpectrum(SyntheticSpectrumGenerator):
    """
    A class which wraps Turbospectrum.

    This wrapper currently does not include any provision for macro-turbulence, which should be applied subsequently, by
    convolving the output spectrum with some appropriate line spread function.
    """

    # Default MARCS model settings to look for. These are fixed parameters which we don't (currently) allow user to vary

    # In this context, "turbulence" refers to the micro turbulence assumed in the MARCS atmosphere. It should closely
    # match the micro turbulence value passed to babsma below.
    marcs_parameters = {"turbulence": 1, "model_type": "st",
                        "a": 0, "c": 0, "n": 0, "o": 0, "r": 0, "s": 0}

    # It is safe to ignore these parameters in MARCS model descriptions
    # This includes interpolating between models with different values of these settings
    marcs_parameters_to_ignore = ["a", "c", "n", "o", "r", "s"]


    def __init__(self, turbospec_path: str, interpol_path: str, line_list_paths: str, marcs_grid_path: str,
                 marcs_grid_list: str, model_atom_path: str, departure_file_path: str, aux_file_length_dict: dict,
                 marcs_value_keys: list, marcs_values: dict, marcs_models: dict, model_temperatures: np.ndarray,
                 model_logs: np.ndarray, model_mets: np.ndarray, night_mode: bool=False):
        """
        Instantiate a class for generating synthetic stellar spectra using Turbospectrum.

        :param turbospec_path: Path where the turbospectrum binaries 'babsma' and 'bsyn' can be found.
        :param interpol_path: Path where the compiled interpol_modeles.f binary can be found.
        :param line_list_paths: Path(s) where line lists for synthetic spectra can be found. Specify as either a string, or a list of strings.
        :param marcs_grid_path: Path where a grid of MARCS .mod files can be found. These contain the model atmospheres we use.
        :param model_atom_path: Path to the model atom paths
        :param departure_file_path: Path to the NLTE departure file paths
        """
        super().__init__(turbospec_path, interpol_path, line_list_paths, marcs_grid_path, marcs_grid_list,
                         model_atom_path, marcs_value_keys, marcs_values, marcs_models, model_temperatures,
                         model_logs, model_mets, night_mode)

        self.departure_file_path = departure_file_path
        self.aux_file_length_dict = aux_file_length_dict

        # Default spectrum parameters
        self.lpoint = 1000000  # number of points in TS

        # parameters needed for nlte and <3D> calculations
        self.windows_flag: bool = False
        self.depart_bin_file = None
        self.depart_aux_file = None
        self.model_atom_file = None
        self.segment_file = None
        self.cont_mask_file = None
        self.line_mask_file = None

        self.run_babsma_flag: bool = True
        self.run_bsyn_flag: bool = True

        self.compute_intensity_flag: bool = False

        self.mupoint_path: str = None

    def configure(self, lambda_min: float=None, lambda_max:float=None, lambda_delta: float=None,
                  metallicity: float=None, log_g: float=None, t_eff: float=None, stellar_mass: float=None,
                  turbulent_velocity: float=None, free_abundances=None, free_isotopes=None,
                  sphere=None, alpha=None, s_process=None, r_process=None,
                  line_list_paths=None, line_list_files=None,
                  verbose=None, counter_spectra=None, temp_directory=None, nlte_flag: bool = None, atmosphere_dimension=None,
                  windows_flag=None,
                  depart_bin_file=None, depart_aux_file=None, model_atom_file=None,
                  segment_file=None, cont_mask_file=None, line_mask_file=None, lpoint=None):
        """
        Set the stellar parameters of the synthetic spectra to generate. This can be called as often as needed
        to generate many synthetic spectra with one class instance. All arguments are optional; any which are not
        supplied will remain unchanged.

        :param lambda_min:
            Short wavelength limit of the synthetic spectra we generate. Unit: A.
        :param lambda_max:
            Long wavelength limit of the synthetic spectra we generate. Unit: A.
        :param lambda_delta:
            Wavelength step of the synthetic spectra we generate. Unit: A.
        :param metallicity:
            Metallicity of the star we're synthesizing.
        :param t_eff:
            Effective temperature of the star we're synthesizing.
        :param log_g:
            Log(gravity) of the star we're synthesizing.
        :param stellar_mass:
            Mass of the star we're synthesizing (solar masses).
        :param turbulent_velocity:
            Micro turbulence velocity in km/s
        :param free_abundances:
            List of elemental abundances to use in stellar model. These are passed to Turbospectrum.
        :param sphere:
            Select whether to use a spherical model (True) or a plane-parallel model (False).
        :param alpha:
            Alpha enhancement to use in stellar model.
        :param s_process:
            S-Process element enhancement to use in stellar model.
        :param r_process:
            R-Process element enhancement to use in stellar model.
        :param line_list_paths:
            List of paths where we should search for line lists.
        :param line_list_files:
            List of line list files to use. If not specified, we use all files in `line_list_paths`
        :param verbose:
            Let Turbospectrum print debugging information to terminal?
        :return:
            None
        """

        if lambda_min is not None:
            self.lambda_min = lambda_min
        if lambda_max is not None:
            self.lambda_max = lambda_max
        if lambda_delta is not None:
            self.lambda_delta = lambda_delta
        if metallicity is not None:
            self.metallicity = metallicity
        if t_eff is not None:
            self.t_eff = t_eff
        if log_g is not None:
            self.log_g = log_g
        if stellar_mass is not None:
            self.stellar_mass = stellar_mass
        if turbulent_velocity is not None:
            self.turbulent_velocity = turbulent_velocity
        if free_abundances is not None:
            self.free_abundances = free_abundances  # [X/H]
        if free_isotopes is not None:
            self.free_isotopes = free_isotopes
        if sphere is not None:
            self.sphere = sphere
        if alpha is not None:
            self.alpha = alpha
        if s_process is not None:
            self.s_process = s_process
        if r_process is not None:
            self.r_process = r_process
        if line_list_paths is not None:
            if not isinstance(line_list_paths, (list, tuple)):
                line_list_paths = [line_list_paths]
            self.line_list_paths = line_list_paths
        if line_list_files is not None:
            self.line_list_files = line_list_files
        if verbose is not None:
            self.verbose = verbose
        if counter_spectra is not None:
            self.counter_spectra = counter_spectra
        if temp_directory is not None:
            self.tmp_dir = temp_directory
        if nlte_flag is not None:
            self.nlte_flag = nlte_flag
        if atmosphere_dimension is not None:
            self.atmosphere_dimension = atmosphere_dimension
        if windows_flag is not None:
            self.windows_flag = windows_flag
        if depart_bin_file is not None:
            self.depart_bin_file = depart_bin_file
        if depart_aux_file is not None:
            self.depart_aux_file = depart_aux_file
        if model_atom_file is not None:
            self.model_atom_file = model_atom_file
        if segment_file is not None:
            self.segment_file = segment_file
        if cont_mask_file is not None:
            self.cont_mask_file = cont_mask_file
        if line_mask_file is not None:
            self.line_mask_file = line_mask_file
        if self.atmosphere_dimension == "3D":
            self.turbulent_velocity = 2.0
            #print("turbulent_velocity is not used since model atmosphere is 3D")
        if lpoint is not None:
            self.lpoint = lpoint


    def make_species_lte_nlte_file(self):
        """
        Generate the SPECIES_LTE_NLTE.dat file for TS to determine what elements are NLTE
        """
        data_path = self.tmp_dir
        file = open("{}/SPECIES_LTE_NLTE_{:08d}.dat".format(data_path, self.counter_spectra), 'w')
        file.write("# This file controls which species are treated in LTE/NLTE\n")
        file.write("# It also gives the path to the model atom and the departure files\n")
        file.write("# First created 2021-02-22\n")
        file.write("# if a species is absent it is assumed to be LTE\n")
        file.write("#\n")
        file.write("# each line contains :\n")
        file.write("# atomic number / name / (n)lte / model atom / departure file / binary or ascii departure file\n")
        file.write("#\n")
        file.write("# path for model atom files     ! don't change this line !\n")
        file.write(f"{self.model_atom_path}\n")
        file.write("#\n")
        file.write("# path for departure files      ! don't change this line !\n")
        file.write(f"{self.tmp_dir}\n")
        file.write("#\n")
        file.write("# atomic (N)LTE setup\n")
        if self.nlte_flag:
            for element in self.model_atom_file:
                # write all nlte elements
                if element in molecules_atomic_number:
                    # so if a molecule is given, get "atomic number" from the separate dictionary #TODO improve to do automatically not just for select molecules?
                    atomic_number = molecules_atomic_number[element][0]
                    element_name_to_write = molecules_atomic_number[element][1]
                else:
                    atomic_number = periodic_table.index(element)
                    element_name_to_write = element
                file.write(f"{atomic_number}  '{element_name_to_write}'  'nlte'  '{self.model_atom_file[element]}'   '{self.marcs_model_name}_{element}_coef.dat' 'ascii'\n")
            for element in self.free_abundances:
                # now check for any lte elements which have a specific given abundance and write them too
                atomic_number = periodic_table.index(element)
                if element not in self.model_atom_file:
                    file.write(f"{atomic_number}  '{element}'  'lte'  ''   '' 'ascii'\n")
        else:
            for element in self.free_abundances:
                atomic_number = periodic_table.index(element)
                file.write(f"{atomic_number}  '{element}'  'lte'  ''   '' 'ascii'\n")
        file.close()

    def _interpolate_atmosphere(self, marcs_model_list: list):
        if not self.atmosphere_properties['flag_dont_interp_microturb']:
            # interpolate model atmosphere between two microturbulence values
            logging.debug(f"Interpolating model atmosphere between two microturbulence values")
            marcs_models_low = self.atmosphere_properties['marcs_model_list_low']
            marcs_models_high = self.atmosphere_properties['marcs_model_list_high']
            marcs_model_name_low = self.atmosphere_properties['marcs_model_name_low']
            marcs_model_name_high = self.atmosphere_properties['marcs_model_name_high']

            microturbulence = self.turbulent_velocity

            turbulence_low = self.atmosphere_properties['turbulence_low']
            turbulence_high = self.atmosphere_properties['turbulence_high']

            self.turbulent_velocity = turbulence_low
            atmosphere_properties_low = self._interpolate_one_atmosphere(marcs_models_low, marcs_model_name_low)
            low_model_name = os_path.join(self.tmp_dir, marcs_model_name_low)
            low_model_name += '.interpol'
            if atmosphere_properties_low['errors']:
                return atmosphere_properties_low
            self.turbulent_velocity = turbulence_high
            atmosphere_properties_high = self._interpolate_one_atmosphere(marcs_models_high, marcs_model_name_high)
            high_model_name = os_path.join(self.tmp_dir, marcs_model_name_high)
            high_model_name += '.interpol'
            if atmosphere_properties_high['errors']:
                return atmosphere_properties_high

            self.turbulent_velocity = microturbulence

            # interpolate and find a model atmosphere for the microturbulence
            self.marcs_model_name = "marcs_tef{:.1f}_g{:.2f}_z{:.2f}_tur{:.2f}".format(self.t_eff, self.log_g,
                                                                                       self.metallicity,
                                                                                       self.turbulent_velocity)
            f_low = open(low_model_name, 'r')
            lines_low = f_low.read().splitlines()
            f_low.close()
            t_low, temp_low, pe_low, pt_low, micro_low, lum_low, spud_low = np.loadtxt(
                open(low_model_name, 'rt').readlines()[:-8], skiprows=1, unpack=True)

            t_high, temp_high, pe_high, pt_high, micro_high, lum_high, spud_high = np.loadtxt(
                open(high_model_name, 'rt').readlines()[:-8], skiprows=1, unpack=True)

            fxhigh = (microturbulence - turbulence_low) / (turbulence_high - turbulence_low)
            fxlow = 1.0 - fxhigh

            t_interp = t_low * fxlow + t_high * fxhigh
            temp_interp = temp_low * fxlow + temp_high * fxhigh
            pe_interp = pe_low * fxlow + pe_high * fxhigh
            pt_interp = pt_low * fxlow + pt_high * fxhigh
            lum_interp = lum_low * fxlow + lum_high * fxhigh
            spud_interp = spud_low * fxlow + spud_high * fxhigh

            interp_model_name = os_path.join(self.tmp_dir, self.marcs_model_name)
            interp_model_name += '.interpol'
            g = open(interp_model_name, 'w')
            print(lines_low[0], file=g)
            for i in range(len(t_interp)):
                print(" {:.4f}  {:.2f}  {:.4f}   {:.4f}   {:.4f}    {:.6e}  {:.4f}".format(t_interp[i],
                                                                                           temp_interp[i],
                                                                                           pe_interp[i],
                                                                                           pt_interp[i],
                                                                                           microturbulence,
                                                                                           lum_interp[i],
                                                                                           spud_interp[i]), file=g)
            print(lines_low[-8], file=g)
            print(lines_low[-7], file=g)
            print(lines_low[-6], file=g)
            print(lines_low[-5], file=g)
            print(lines_low[-4], file=g)
            print(lines_low[-3], file=g)
            print(lines_low[-2], file=g)
            print(lines_low[-1], file=g)
            g.close()

            # generate models for low and high parts
            if self.nlte_flag:
                #  {self.model_atom_file}
                logging.debug(f"self.model_atom_file inside ts_class_nlte.py: {self.model_atom_file}")
                for element in self.model_atom_file:
                    logging.debug(f"now low/high parts calling element: {element}")
                    low_coef_dat_name = low_model_name.replace('.interpol', '_{}_coef.dat'.format(element))
                    logging.debug(f"low_coef_dat_name: {low_coef_dat_name}")
                    f_coef_low = open(low_coef_dat_name, 'r')
                    lines_coef_low = f_coef_low.read().splitlines()
                    f_coef_low.close()

                    self.check_nan_in_coefficients(lines_coef_low, low_coef_dat_name)

                    high_coef_dat_name = os_path.join(self.tmp_dir, self.marcs_model_name)
                    high_coef_dat_name += '_{}_coef.dat'.format(element)

                    high_coef_dat_name = high_model_name.replace('.interpol', '_{}_coef.dat'.format(element))
                    logging.debug(f"high_coef_dat_name: {high_coef_dat_name}")
                    f_coef_high = open(high_coef_dat_name, 'r')
                    lines_coef_high = f_coef_high.read().splitlines()
                    f_coef_high.close()

                    self.check_nan_in_coefficients(lines_coef_high, high_coef_dat_name)

                    interp_coef_dat_name = os_path.join(self.tmp_dir, self.marcs_model_name)
                    interp_coef_dat_name += '_{}_coef.dat'.format(element)

                    g = open(interp_coef_dat_name, 'w')
                    logging.debug(f"interp_coef_dat_name: {interp_coef_dat_name}")
                    for i in range(11):
                        print(lines_coef_low[i], file=g)
                    for i in range(len(t_interp)):
                        print(" {:7.4f}".format(t_interp[i]), file=g)
                    for i in range(10 + len(t_interp) + 1, 10 + 2 * len(t_interp) + 1):
                        fields_low = lines_coef_low[i].strip().split()
                        fields_high = lines_coef_high[i].strip().split()
                        fields_interp = []
                        for j in range(len(fields_low)):
                            fields_interp.append(float(fields_low[j]) * fxlow + float(fields_high[j]) * fxhigh)
                        fields_interp_print = ['   {:.5f} '.format(elem) for elem in fields_interp]
                        print(*fields_interp_print, file=g)
                    for i in range(10 + 2 * len(t_interp) + 1, len(lines_coef_low)):
                        print(lines_coef_low[i], file=g)
                    g.close()
        else:
            logging.debug(f"Interpolating single model atmosphere")
            self._interpolate_one_atmosphere(marcs_model_list, self.marcs_model_name)

    def _interpolate_one_atmosphere(self, marcs_model_list: list, marcs_model_name: str):
        if self.verbose:
            stdout = None
            stderr = subprocess.STDOUT
        else:
            stdout = open('/dev/null', 'w')
            stderr = subprocess.STDOUT

        output = os_path.join(self.tmp_dir, marcs_model_name)
        model_test = "{}.test".format(output)

        if self.nlte_flag:
            for element in self.model_atom_file:
                element_abundance = self._get_element_abundance(element)
                # Write configuration input for interpolator
                interpol_config = ""
                for line in marcs_model_list:
                    interpol_config += "'{}{}'\n".format(self.marcs_grid_path, line)
                interpol_config += "'{}.interpol'\n".format(output)
                interpol_config += "'{}.alt'\n".format(output)
                interpol_config += "'{}_{}_coef.dat'\n".format(output, element)  # needed for nlte interpolator
                interpol_config += "'{}'\n".format(os_path.join(self.departure_file_path, self.depart_bin_file[
                    element]))  # needed for nlte interpolator
                interpol_config += "'{}'\n".format(os_path.join(self.departure_file_path, self.depart_aux_file[
                    element]))  # needed for nlte interpolator
                interpol_config += "{}\n".format(self.aux_file_length_dict[element])
                interpol_config += "{}\n".format(self.t_eff)
                interpol_config += "{}\n".format(self.log_g)
                interpol_config += "{:.6f}\n".format(round(float(self.metallicity), 6))
                interpol_config += "{:.6f}\n".format(round(float(element_abundance), 6))
                interpol_config += ".false.\n"  # test option - set to .true. if you want to plot comparison model (model_test)
                interpol_config += ".false.\n"  # MARCS binary format (.true.) or MARCS ASCII web format (.false.)?
                interpol_config += "'{}'\n".format(model_test)

                # Now we run the FORTRAN model interpolator
                try:
                    if self.atmosphere_dimension == "1D":
                        p = subprocess.Popen([os_path.join(self.interpol_path, 'interpol_modeles_nlte')],
                                             stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
                        p.stdin.write(bytes(interpol_config, 'utf-8'))
                        stdout, stderr = p.communicate()
                    elif self.atmosphere_dimension == "3D":
                        p = subprocess.Popen([os_path.join(self.interpol_path, 'interpol_multi_nlte')],
                                             stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
                        p.stdin.write(bytes(interpol_config, 'utf-8'))
                        stdout, stderr = p.communicate()
                except subprocess.CalledProcessError:
                    return {
                        "interpol_config": interpol_config,
                        "errors": "MARCS model atmosphere interpolation failed."
                    }

                coef_dat_name = "'{}_{}_coef.dat'\n".format(output, element).replace('.interpol', '_{}_coef.dat'.format(element)).strip().replace("'", "")

                coef_element = open(coef_dat_name, 'r')
                lines_coef = coef_element.read().splitlines()
                coef_element.close()

                self.check_nan_in_coefficients(lines_coef, coef_dat_name)
        else:
            # Write configuration input for interpolator
            interpol_config = ""
            # print(marcs_model_list)
            for line in marcs_model_list:
                interpol_config += "'{}{}'\n".format(self.marcs_grid_path, line)
            interpol_config += "'{}.interpol'\n".format(output)
            interpol_config += "'{}.alt'\n".format(output)
            interpol_config += "{}\n".format(self.t_eff)
            interpol_config += "{}\n".format(self.log_g)
            interpol_config += "{}\n".format(self.metallicity)
            interpol_config += ".false.\n"  # test option - set to .true. if you want to plot comparison model (model_test)
            interpol_config += ".false.\n"  # MARCS binary format (.true.) or MARCS ASCII web format (.false.)?
            interpol_config += "'{}'\n".format(model_test)

            # Now we run the FORTRAN model interpolator
            try:
                if self.atmosphere_dimension == "1D":
                    p = subprocess.Popen([os_path.join(self.interpol_path, 'interpol_modeles')],
                                         stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
                    p.stdin.write(bytes(interpol_config, 'utf-8'))
                    stdout, stderr = p.communicate()
                elif self.atmosphere_dimension == "3D":
                    p = subprocess.Popen([os_path.join(self.interpol_path, 'interpol_multi')],
                                         stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
                    p.stdin.write(bytes(interpol_config, 'utf-8'))
                    stdout, stderr = p.communicate()
            except subprocess.CalledProcessError:
                return {
                    "interpol_config": interpol_config,
                    "errors": "MARCS model atmosphere interpolation failed."
                }
        return {"errors": None}

    @staticmethod
    def check_nan_in_coefficients(lines_coef, file_to_save):
        # read through the lines. first skip comments that start with #
        idx_to_read = 0
        while lines_coef[idx_to_read][0] == "#":
            idx_to_read += 1
        idx_to_read += 1
        number_of_depths = int(lines_coef[idx_to_read])
        number_of_levels = int(lines_coef[idx_to_read + 1])
        idx_to_read += 2 + number_of_depths
        lines_coef_to_check = lines_coef[idx_to_read:idx_to_read + number_of_depths][::-1]

        save_new_lines = False

        for line_idx, line in enumerate(lines_coef_to_check):
            if "        NaN" in line:
                logging.debug(f"Found NaN in line {line_idx}")
                save_new_lines = True
                # we need to find location of every nan and take the value that is in the same column, but a row below, if it is not a nan
                # if reaching the end of rows, take value of 1 and replace it
                # first find the indices of every nan
                if line_idx == 0:
                    # replace all nans with 1
                    lines_coef_to_check[line_idx] = line.replace("        NaN", "1.00000E+00")
                else:
                    # take the value from the line below
                    # first need to find indices within the string where it contains each "        NaN", then take the value from the line below at the same index
                    line_below = lines_coef_to_check[line_idx - 1]
                    idx_contains_nan = [line.find("        NaN", i) for i in range(len(line)) if
                                        line.find("        NaN", i) != -1]
                    idx_contains_nan = sorted(list(set(idx_contains_nan)))
                    for idx in idx_contains_nan:
                        # find the value in the line below
                        value_below = line_below[idx:idx + 12]
                        line = line[:idx] + value_below + line[idx + 12:]
                    lines_coef_to_check[line_idx] = line
            elif " -" in line:
                logging.debug(f"Found - in line {line_idx} in file {file_to_save}. Probably spectrum will not compute.")

        if save_new_lines:
            new_lines = lines_coef[:idx_to_read] + lines_coef_to_check[::-1] + lines_coef[
                                                                               idx_to_read + number_of_depths:]
            with open(file_to_save, 'w') as f:
                for line in new_lines:
                    f.write("%s\n" % line)

    def calculate_atmosphere(self):
        # figure out if we need to interpolate the model atmosphere for microturbulence
        possible_turbulence = [0.0, 1.0, 2.0, 5.0]
        flag_dont_interp_microturb = False
        for i in range(len(possible_turbulence)):
            if self.turbulent_velocity == possible_turbulence[i]:
                flag_dont_interp_microturb = True

        if self.log_g < 3:
            flag_dont_interp_microturb = True

        if not (self.turbulent_velocity < 2.0 and (self.turbulent_velocity > 1.0 or (self.turbulent_velocity < 1.0 and self.t_eff < 3900.))):
            flag_dont_interp_microturb = True

        atmosphere_properties = {}

        logging.debug(f"flag_dont_interp_microturb: {flag_dont_interp_microturb} {self.turbulent_velocity} {self.t_eff} {self.log_g}")

        if not flag_dont_interp_microturb:
            # Bracket the microturbulence to figure out what two values to generate the models to interpolate between using Andy's code
            turbulence_low = 0.0
            microturbulence = self.turbulent_velocity
            for i in range(len(possible_turbulence)):
                if self.turbulent_velocity > possible_turbulence[i]:
                    turbulence_low = possible_turbulence[i]
                    place = i
            turbulence_high = possible_turbulence[place + 1]

            self.turbulent_velocity = turbulence_low
            atmosphere_properties_low = self._generate_model_atmosphere()
            if atmosphere_properties_low['errors']:
                return atmosphere_properties_low
            low_marcs_model_name = self.marcs_model_name
            low_marcs_list = atmosphere_properties_low['marcs_model_list']

            self.turbulent_velocity = turbulence_high
            atmosphere_properties_high = self._generate_model_atmosphere()
            if atmosphere_properties_high['errors']:
                return atmosphere_properties_high
            high_marcs_model_name = self.marcs_model_name
            high_marcs_list = atmosphere_properties_high['marcs_model_list']

            atmosphere_properties = atmosphere_properties_low
            atmosphere_properties['turbulence_low'] = turbulence_low
            atmosphere_properties['turbulence_high'] = turbulence_high
            atmosphere_properties['marcs_model_list_low'] = low_marcs_list
            atmosphere_properties['marcs_model_list_high'] = high_marcs_list
            atmosphere_properties['marcs_model_name_low'] = low_marcs_model_name
            atmosphere_properties['marcs_model_name_high'] = high_marcs_model_name

            self.turbulent_velocity = microturbulence

        elif self.turbulent_velocity > 2.0:  # not enough models to interp if higher than 2
            microturbulence = self.turbulent_velocity  # just use 2.0 for the model if between 2 and 3
            self.turbulent_velocity = 2.0
            atmosphere_properties = self._generate_model_atmosphere()
            if atmosphere_properties['errors']:
                return atmosphere_properties
            self.turbulent_velocity = microturbulence

        elif self.turbulent_velocity < 1.0 and self.t_eff >= 3900.:  # not enough models to interp if lower than 1 and t_eff > 3900
            microturbulence = self.turbulent_velocity
            self.turbulent_velocity = 1.0
            atmosphere_properties = self._generate_model_atmosphere()
            if atmosphere_properties['errors']:
                return atmosphere_properties
            self.turbulent_velocity = microturbulence

        elif flag_dont_interp_microturb:
            if self.log_g < 3:
                microturbulence = self.turbulent_velocity
                self.turbulent_velocity = 2.0
            atmosphere_properties = self._generate_model_atmosphere()
            if self.log_g < 3:
                self.turbulent_velocity = microturbulence
            if atmosphere_properties['errors']:
                # print('spud')
                if not self.night_mode:
                    print(atmosphere_properties['errors'])
                return atmosphere_properties
        else:
            print("Unexpected error?")
        atmosphere_properties['flag_dont_interp_microturb'] = flag_dont_interp_microturb
        self.atmosphere_properties = atmosphere_properties

    def make_babsma_bsyn_file(self, spherical):
        """
        Generate the configurations files for both the babsma and bsyn binaries in Turbospectrum.
        """

        # If we've not been given an explicit alpha enhancement value, assume one
        alpha = self.alpha
        if alpha is None:
            if self.metallicity < -1.0:
                alpha = 0.4
            elif -1.0 < self.metallicity < 0.0:
                alpha = -0.4 * self.metallicity
            else:
                alpha = 0

        # Updated abundances to below to allow user to set solar abundances through solar_abundances.py and not have to adjust make_abund.f

        individual_abundances = "'INDIVIDUAL ABUNDANCES:'   '{:d}'\n".format(len(periodic_table) - 1)
        
        item_abund = {'H': 12.00, periodic_table[2]: float(solar_abundances[periodic_table[2]])}
        for i in range(3, len(periodic_table)):
            # first take solar scaled abundances as A(X)
            item_abund[periodic_table[i]] = float(solar_abundances[periodic_table[i]]) + round(float(self.metallicity), 6)
        if self.free_abundances is not None:
            # and if any abundance is passed, take it and convert to A(X)
            for element, abundance in self.free_abundances.items():
                item_abund[element] = float(solar_abundances[element]) + round(float(abundance), 6)
        for i in range(1, len(periodic_table)):
            individual_abundances += "{:d}  {:.6f}\n".format(i, item_abund[periodic_table[i]])

        # Allow for user input isotopes as a dictionary (similar to abundances)

        individual_isotopes = f"'ISOTOPES : ' '{len(solar_isotopes)}'\n"
        if self.free_isotopes is None:
            for isotope, ratio in solar_isotopes.items():
                individual_isotopes += "{}  {:6f}\n".format(isotope, ratio)
        else:
            for isotope, ratio in self.free_isotopes.items():
                solar_isotopes[isotope] = ratio
            for isotope, ratio in solar_isotopes.items():
                individual_isotopes += "{}  {:6f}\n".format(isotope, ratio)

        # Make a list of line-list files
        # We start by getting a list of all files in the line list directories we've been pointed towards,
        # excluding any text files we find.
        line_list_files = []
        for line_list_path in self.line_list_paths:
            line_list_files.extend([i for i in glob.glob(os_path.join(line_list_path, "*")) if not i.endswith(".txt")])

        # If an explicit list of line_list_files is set, we treat this as a list of filenames within the specified
        # line_list_path, and we only allow files with matching filenames
        if self.line_list_files is not None:
            line_list_files = [item for item in line_list_files if os_path.split(item)[1] in self.line_list_files]

        # Encode list of line lists into a string to pass to bsyn
        line_lists = "'NFILES   :' '{:d}'\n".format(len(line_list_files))
        for item in line_list_files:
            line_lists += "{}\n".format(item)

        # Build bsyn configuration file
        spherical_boolean_code = "T" if spherical else "F"
        if self.atmosphere_dimension == "3D":
            spherical_boolean_code = "F"
        xifix_boolean_code = "T" if self.atmosphere_dimension == "1D" else "F"
        nlte_boolean_code = ".true." if self.nlte_flag == True else ".false."
        # NS 16.05.2025: changed to .false. to keep scattering on (see GitHub issue #88)
        pure_lte_boolean_code = ".false."

        if self.windows_flag:
            segment_file_string = f"'SEGMENTSFILE:'     '{self.segment_file}'\n"
        else:
            segment_file_string = ""

        intensity_or_flux = "Intensity" if self.compute_intensity_flag else "Flux"

        if self.compute_intensity_flag:
            mu_point_path = f"\n'MU-POINTS:' '{self.mupoint_path}'"
        else:
            mu_point_path = ""

        # Build babsma configuration file
        babsma_config = f"""\
'PURE-LTE  :'  '{pure_lte_boolean_code}'
'LAMBDA_MIN:'    '{self.lambda_min:.4f}'
'LAMBDA_MAX:'    '{self.lambda_max:.4f}'
'LAMBDA_STEP:'    '{self.lambda_delta:.5f}'
'MODELINPUT:' '{self.tmp_dir}{self.marcs_model_name}.interpol'
'MARCS-FILE:' '.false.'
'MODELOPAC:' '{self.tmp_dir}model_opacity_{self.counter_spectra:08d}.opac'
'METALLICITY:'    '{self.metallicity:.2f}'
'ALPHA/Fe   :'    '{alpha:.2f}'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '{self.r_process:.2f}'
'S-PROCESS  :'    '{self.s_process:.2f}'
{individual_abundances.strip()}
'XIFIX:' '{xifix_boolean_code}'
{self.turbulent_velocity:.2f}
"""
        # Build bsyn configuration file
        bsyn_config = f"""\
'PURE-LTE  :'  '{pure_lte_boolean_code}'
'NLTE :'          '{nlte_boolean_code}'
'NLTEINFOFILE:'  '{self.tmp_dir}SPECIES_LTE_NLTE_{self.counter_spectra:08d}.dat'
{segment_file_string}'LAMBDA_MIN:'    '{self.lambda_min:.4f}'
'LAMBDA_MAX:'    '{self.lambda_max:.4f}'
'LAMBDA_STEP:'   '{self.lambda_delta:.5f}'
'INTENSITY/FLUX:' '{intensity_or_flux}'{mu_point_path}
'COS(THETA)    :' '1.00'
'ABFIND        :' '.false.'
'MODELOPAC:' '{self.tmp_dir}model_opacity_{self.counter_spectra:08d}.opac'
'RESULTFILE :' '{self.tmp_dir}/spectrum_{self.counter_spectra:08d}.spec'
'METALLICITY:'    '{self.metallicity:.2f}'
'ALPHA/Fe   :'    '{alpha:.2f}'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '{self.r_process:.2f}'
'S-PROCESS  :'    '{self.s_process:.2f}'
{individual_abundances.strip()}
{individual_isotopes.strip()}
{line_lists.strip()}
'SPHERICAL:'  '{spherical_boolean_code}'
  30
  300.00
  15
  1.30
"""
        # print(babsma_config)
        # print(bsyn_config)
        return babsma_config, bsyn_config

    def load_generated_spectra(self, location):
        if not self.compute_intensity_flag:
            wave, flux_norm, flux = np.loadtxt(location, unpack=True, dtype=float)
            return wave, flux_norm, flux
        else:
            results = np.loadtxt(location, dtype=float)
            wavelength = results[:, 0]
            intensities = results[:, 1:]
            return wavelength, intensities

    def stitch(self, specname1, specname2, lmin, lmax, new_range, count):  # toss a coin to your stitcher
        data1 = np.loadtxt(specname1)
        data2 = np.loadtxt(specname2)

        if np.size(data1) == 0 or np.size(data2) == 0:
            raise ValueError("Empty spectrum file")

        # print(lmin, lmin+(count*new_range))

        # Define masks for clipping
        mask1 = (data1[:, 0] < lmin + (count * new_range)) & (data1[:, 0] >= lmin)
        mask2 = (data2[:, 0] >= lmin + (count * new_range)) & (data2[:, 0] <= lmax)

        # Clip each full 2D array by row
        data1_clipped = data1[mask1]
        data2_clipped = data2[mask2]

        # Concatenate along the row axis (axis=0)
        data_stitched = np.concatenate((data1_clipped, data2_clipped), axis=0)

        # Separate out wavelength (first column) and the rest
        wave = data_stitched[:, 0]
        other_data = data_stitched[:, 1:]

        return wave, other_data

    def synthesize(self, run_babsma_flag=True, run_bsyn_flag=True):
        babsma_in, bsyn_in = self.make_babsma_bsyn_file(spherical=self.atmosphere_properties['spherical'])

        logging.debug("babsma input:\n{}".format(babsma_in))
        logging.debug("bsyn input:\n{}".format(bsyn_in))

        # Start making dictionary of output data
        output = self.atmosphere_properties
        output["errors"] = None
        output["babsma_config"] = babsma_in
        output["bsyn_config"] = bsyn_in

        # We need to run babsma and bsyn with working directory set to root of Turbospectrum install. Otherwise
        # it cannot find its data files.
        cwd = os.getcwd()
        turbospec_root = os_path.join(self.code_path, "..")

        # Select whether we want to see all the output that babsma and bsyn send to the terminal
        if self.verbose:
            stdout = None
            stderr = subprocess.STDOUT
        else:
            stdout = open('/dev/null', 'w')
            stderr = subprocess.STDOUT

        if run_babsma_flag:
            logging.debug("Running babsma")

            # Generate configuration files to pass to babsma and bsyn
            if self.nlte_flag:
                self.make_species_lte_nlte_file()

            # Run babsma. This creates an opacity file .opac from the MARCS atmospheric model
            try:  # chdir is NECESSARY, turbospectrum cannot run from other directories sadly
                os.chdir(turbospec_root)
                pr1, stderr_bytes = self.run_babsma(babsma_in, stderr, stdout)
            except subprocess.CalledProcessError:
                output["errors"] = "babsma failed with CalledProcessError"
                return output
            finally:
                os.chdir(cwd)
            if stderr_bytes is None:
                stderr_bytes = b''
            if pr1.returncode != 0:
                output["errors"] = "babsma failed"
                # logging.info("Babsma failed. Return code {}. Error text <{}>".
                #             format(pr1.returncode, stderr_bytes.decode('utf-8')))
                return output
            output["return_code"] = pr1.returncode

        if run_bsyn_flag:
            logging.debug("Running bsyn")

            # Run bsyn. This synthesizes the spectrum
            try:
                os.chdir(turbospec_root)
                pr, stderr_bytes = self.run_bsyn(bsyn_in, stderr, stdout)
            except subprocess.CalledProcessError:
                output["errors"] = "bsyn failed with CalledProcessError"
                return output
            finally:
                os.chdir(cwd)
            if stderr_bytes is None:
                stderr_bytes = b''
            if pr.returncode != 0:
                output["errors"] = "bsyn failed"
                # logging.info("Bsyn failed. Return code {}. Error text <{}>".
                #             format(pr.returncode, stderr_bytes.decode('utf-8')))
                return output

            # Return output
            output["return_code"] = pr.returncode
        output["output_file"] = os_path.join(self.tmp_dir, "spectrum_{:08d}.spec".format(self.counter_spectra))
        return output

    def run_bsyn(self, bsyn_in, stderr, stdout):
        pr = subprocess.Popen([os_path.join(self.code_path, 'bsyn_lu')],
                              stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
        pr.stdin.write(bytes(bsyn_in, 'utf-8'))
        stdout_bytes, stderr_bytes = pr.communicate()
        return pr, stderr_bytes

    def run_babsma(self, babsma_in, stderr, stdout):
        pr1 = subprocess.Popen([os_path.join(self.code_path, 'babsma_lu')],
                               stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
        pr1.stdin.write(bytes(babsma_in, 'utf-8'))
        stdout_bytes, stderr_bytes = pr1.communicate()
        return pr1, stderr_bytes

    def run_turbospectrum(self, run_babsma_flag=True, run_bsyn_flag=True):
        lmin_orig = self.lambda_min
        lmax_orig = self.lambda_max
        lmin = self.lambda_min
        lmax = self.lambda_max

        lpoint_max = self.lpoint * 0.99  # first number comes from turbospectrum spectrum.inc : lpoint. 0.99 is to give some extra room so that bsyn does not fail for sure
        points_in_new_spectra_to_generate = int((lmax - lmin) / self.lambda_delta)

        if points_in_new_spectra_to_generate > lpoint_max:
            print("WARNING. The range or delta wavelength asked is too big. Trying to break down into smaller "
                  "segments and stitch them together at the end.")
            number = math.ceil(points_in_new_spectra_to_generate / lpoint_max)
            new_range = round((lmax - lmin) / number)
            extra_wavelength_for_stitch = 30  # generats with extra wavlength so that stitch can be nice i guess (i did not write this originally)

            for i in range(number):
                self.configure(lambda_min=lmin - extra_wavelength_for_stitch,
                               lambda_max=lmin + new_range + extra_wavelength_for_stitch, counter_spectra=i)
                self.synthesize(run_babsma_flag=run_babsma_flag, run_bsyn_flag=run_bsyn_flag)
                lmin = lmin + new_range
            for i in range(number - 1):
                spectrum1 = os_path.join(self.tmp_dir, "spectrum_{:08d}.spec".format(0))
                spectrum2 = os_path.join(self.tmp_dir, "spectrum_{:08d}.spec".format(i + 1))
                wave, data = self.stitch(spectrum1, spectrum2, lmin_orig, lmax_orig, new_range, i + 1)

                # Combine wave (shape [N]) with data (shape [N, M]) into a single 2D array (shape [N, M+1])
                combined = np.column_stack((wave, data))

                # Save to a text file using numpy
                np.savetxt(spectrum1, combined, fmt="%.6f")
        else:
            self.synthesize(run_babsma_flag=run_babsma_flag, run_bsyn_flag=run_bsyn_flag)

    def synthesize_spectra(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        run_babsma_flag = self.run_babsma_flag
        run_bsyn_flag = self.run_bsyn_flag
        try:
            logging.debug("Running Turbospectrum and atmosphere")
            logging.debug("Cleaning temp directory")
            # clean temp directory
            temp_spectra_location = os.path.join(self.tmp_dir, "spectrum_00000000.spec")
            # delete the temporary directory if it exists
            if os_path.exists(temp_spectra_location):
                os.remove(temp_spectra_location)
            if run_babsma_flag:
                # generate the model atmosphere
                logging.debug("Calculating atmosphere")
                self.calculate_atmosphere()
            if (run_bsyn_flag and self.nlte_flag) or run_babsma_flag:
                logging.debug("Interpolating atmosphere")
                self._interpolate_atmosphere(self.atmosphere_properties['marcs_model_list'])
            try:
                logging.debug("Running Turbospectrum")
                self.run_turbospectrum(run_babsma_flag=run_babsma_flag, run_bsyn_flag=run_bsyn_flag)
                # NS 12.01.2024: now we return the fitted spectrum
                temp_spectra_location = os.path.join(self.tmp_dir, "spectrum_{:08d}.spec".format(0))
                if os_path.exists(temp_spectra_location):
                    if os.stat(temp_spectra_location).st_size != 0:
                        return self.load_generated_spectra(temp_spectra_location)
                    else:
                        # return 3 empty arrays, because the file exists but is empty
                        if not self.compute_intensity_flag:
                            return np.array([]), np.array([]), np.array([])
                        else:
                            return np.array([]), np.array([])
            except AttributeError:
                if not self.night_mode:
                    print("No attribute, fail of generation?")
        except (FileNotFoundError, ValueError, TypeError) as error:
            if not self.night_mode:
                print(f"Interpolation failed? {error}")
                if error == ValueError:
                    print("ValueError can sometimes imply problem with the departure coefficients grid")
        if not self.compute_intensity_flag:
            return np.array([]), np.array([]), np.array([])
        else:
            return np.array([]), np.array([])