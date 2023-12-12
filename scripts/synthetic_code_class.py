from __future__ import annotations

import subprocess
import os
from os import path as os_path
import glob
import re
from operator import itemgetter
import numpy as np
import math
import logging
import abc
from scripts.auxiliary_functions import closest_available_value
from scripts.solar_abundances import solar_abundances, periodic_table, molecules_atomic_number
from scripts.solar_isotopes import solar_isotopes

class SyntheticSpectrumGenerator:
    """
        A class which wraps any code to run.

        This wrapper currently does not include any provision for macro-turbulence, which should be applied subsequently, by
        convolving the output spectrum with some appropriate line spread function.
    """

    # Default MARCS model settings to look for. These are fixed parameters which we don't (currently) allow user to vary

    # In this context, "turbulence" refers to the micro turbulence assumed in the MARCS atmosphere. It should closely
    # match the micro turbulence value passed to codes.
    marcs_parameters = {"turbulence": 1, "model_type": "st",
                        "a": 0, "c": 0, "n": 0, "o": 0, "r": 0, "s": 0}

    # It is safe to ignore these parameters in MARCS model descriptions
    # This includes interpolating between models with different values of these settings
    marcs_parameters_to_ignore = ["a", "c", "n", "o", "r", "s"]

    def __init__(self, code_path: str, interpol_path: str, line_list_paths: str, marcs_grid_path: str,
                 marcs_grid_list: str, model_atom_path: str,
                 marcs_value_keys: list, marcs_values: dict, marcs_models: dict, model_temperatures: np.ndarray,
                 model_logs: np.ndarray, model_mets: np.ndarray):
        """
        Instantiate a class for generating synthetic stellar spectra using Turbospectrum.

        :param turbospec_path: Path where the turbospectrum binaries 'babsma' and 'bsyn' can be found.
        :param interpol_path: Path where the compiled interpol_modeles.f binary can be found.
        :param line_list_paths: Path(s) where line lists for synthetic spectra can be found. Specify as either a string, or a list of strings.
        :param marcs_grid_path: Path where a grid of MARCS .mod files can be found. These contain the model atmospheres we use.
        :param model_atom_path: Path to the model atom paths
        :param departure_file_path: Path to the NLTE departure file paths
        """

        self.atmosphere_properties = None
        if not isinstance(line_list_paths, (list, tuple)):
            line_list_paths = [line_list_paths]

        self.code_path = code_path
        self.interpol_path = interpol_path
        self.line_list_paths = line_list_paths
        self.marcs_grid_path = marcs_grid_path
        self.marcs_grid_list = marcs_grid_list
        self.model_atom_path = model_atom_path

        # Default spectrum parameters
        self.lambda_min: float = None  # Angstrom
        self.lambda_max: float = None
        self.lambda_delta: float = None
        self.metallicity: float = None
        self.stellar_mass: float = 1
        self.log_g: float = None
        self.t_eff: float = None
        self.turbulent_velocity: float = None  # micro turbulence, km/s
        self.free_abundances: dict = None  # [X/H] for each element
        self.free_isotopes: dict = None
        self.sphere: bool = None
        self.alpha = None  # not used?
        self.s_process = 0  # not used?
        self.r_process = 0  # not used?
        self.verbose: bool = False
        self.line_list_files = None

        # parameters needed for nlte and <3D> calculations
        self.nlte_flag: bool = False
        self.atmosphere_dimension: str = None
        ##self.windows_flag: bool = False
        ##self.depart_bin_file = None
        ##self.depart_aux_file = None
        ##self.model_atom_file = None
        ##self.segment_file = None
        ##self.cont_mask_file = None
        ##self.line_mask_file = None

        # Create temporary directory
        self.id_string = None
        self.tmp_dir = None

        # Look up what MARCS models we have
        # self.counter_marcs = 0
        self.marcs_model_name = None
        self.counter_spectra = 0
        self.marcs_value_keys = marcs_value_keys
        self.marcs_values = {
            "spherical": [], "temperature": [], "log_g": [], "mass": [], "turbulence": [], "model_type": [],
            "metallicity": [], "a": [], "c": [], "n": [], "o": [], "r": [], "s": []}
        self.marcs_values = marcs_values
        self.marcs_models = marcs_models
        self.model_temperatures = model_temperatures
        self.model_logs = model_logs
        self.model_mets = model_mets
        self.marcs_model_list_global = []  # needed for microturbulence interpolation

    @abc.abstractmethod
    def configure(self):
        """
        Configure the code to run.
        """
        pass

    def _get_element_abundance(self, element):
        if element in self.free_abundances:
            # if element abundance was given, then pass it to the NLTE
            # self.free_abundances[element] = [X/Fe] + [Fe/H] = [X/H] (already scaled from before)
            # solar_abundances[element] = abundance as A(X)
            element_abundance = self.free_abundances[element] + float(solar_abundances[element])
        else:
            # else, take solar abundance and scale with metallicity
            # solar_abundances[element] = abundance as A(X)
            # self.metallicity = [Fe/H]
            if element in molecules_atomic_number:
                # so if a molecule is given, get "atomic number" from the separate dictionary #TODO improve to do automatically not just for select molecules?
                if element == "CN":
                    element_abundance = float(solar_abundances["N"]) + self.metallicity
                elif element == "CH":
                    element_abundance = float(solar_abundances["C"]) + self.metallicity
                else:
                    raise ValueError(f"Molecule {element} not supported.")
            else:
                if element.lower() not in ["h", "he"]:
                    # if not H or He, then scale with metallicity
                    # self.metallicity = [Fe/H]
                    element_abundance = float(solar_abundances[element]) + self.metallicity
                else:
                    # if H or He, then just take solar abundance
                    element_abundance = float(solar_abundances[element])
        return element_abundance


    def _generate_model_atmosphere(self):
        """
        Generates an interpolated model atmosphere from the MARCS grid using the interpol.f routine developed by
        T. Masseron (Masseron, PhD Thesis, 2006). This is a python wrapper for that fortran code.
        """
        # self.counter_marcs += 1
        # self.marcs_model_name = "marcs_{:08d}".format(self.counter_marcs)
        self.marcs_model_name = "marcs_tef{:.1f}_g{:.2f}_z{:.2f}_tur{:.2f}".format(self.t_eff, self.log_g,
                                                                                   self.metallicity,
                                                                                   self.turbulent_velocity)
        #global marcs_model_list_global

        #        if self.verbose:
        #            stdout = None
        #            stderr = subprocess.STDOUT
        #        else:
        #            stdout = open('/dev/null', 'w')
        #            stderr = subprocess.STDOUT

        # Defines default point at which plane-parallel vs spherical model atmosphere models are used
        spherical: bool = self.sphere
        if spherical is None:
            spherical = (self.log_g < 3)

        # Create dictionary of the MARCS model parameters we're looking for in grid
        marcs_parameters = self.marcs_parameters.copy()
        marcs_parameters['turbulence'] = self.turbulent_velocity  # JMG line to make microturbulence an adjustable variable
        # print(marcs_parameters)
        if spherical:
            marcs_parameters['spherical'] = "s"
            marcs_parameters['mass'] = closest_available_value(self.stellar_mass, self.marcs_values['mass'])
            microturbulence = self.turbulent_velocity
            self.turbulent_velocity = 2.0
            # print(marcs_parameters['mass'])
            # marcs_parameters['mass'] = self.closest_available_value(self.marcs_values['mass'])
        else:
            marcs_parameters['spherical'] = "p"
            marcs_parameters['mass'] = 0  # All plane-parallel models have mass set to zero

        # Pick MARCS settings which bracket requested stellar parameters
        interpolate_parameters = ("metallicity", "log_g", "temperature")

        interpolate_parameters_around = {"temperature": self.t_eff,
                                         "log_g": self.log_g,
                                         "metallicity": self.metallicity,
                                         }
        # go through each value that is interpolated
        for key in interpolate_parameters:
            value = interpolate_parameters_around[key]
            options = self.marcs_values[key]  # get what values exist in marcs values
            if (value < options[0]) or (
                    value > options[-1]):  # checks that the value is within the marcs possible values
                return {
                    "errors": "Value of parameter <{}> needs to be in range {} to {}. You requested {}.".
                    format(key, options[0], options[-1], value)
                }
            for index in range(len(options) - 1):
                if value < options[index + 1]:
                    break
            # Mar. 11, 2022 added if statement for what to do if parameter is on the vertex. model interpolator needs the value for both models to be on that vertex or else will falsely think it's extrapolating
            if value == options[index]:
                marcs_parameters[key] = [options[index], options[index], index, index]
            elif value == options[index + 1]:
                marcs_parameters[key] = [options[index + 1], options[index + 1], index + 1, index + 1]
            else:
                marcs_parameters[key] = [options[index], options[index + 1], index, index + 1]

        # Loop over eight vertices of cuboidal cell in parameter space, collecting MARCS models
        marcs_model_list = []
        failures = True
        while failures:
            marcs_model_list = []
            failures = 0
            n_vertices = 2 ** len(interpolate_parameters)
            for vertex in range(n_vertices):  # Loop over 8 vertices
                # Variables used to produce informative error message if we can't find a particular model
                model_description = []
                failed_on_parameter = ("None", "None", "None")
                value = "None"
                parameter = "None"
                logg_chosen = None

                # Start looking for a model that sits at this particular vertex of the cube
                dict_iter = self.marcs_models  # Navigate through dictionary tree of MARCS models we have
                try:
                    for parameter in self.marcs_value_keys:
                        value = marcs_parameters[parameter]
                        # When we encounter Teff, log_g or metallicity, we get two options, not a single value
                        # Choose which one to use by looking at the binary bits of <vertex> as it counts from 0 to 7
                        # This tells us which particular vertex of the cube we're looking for
                        if isinstance(value, (list, tuple)):
                            option_number = int(bool(vertex & (2 ** interpolate_parameters.index(parameter))))  # 0 or 1
                            value = value[option_number]

                        if parameter == "log_g":
                            logg_chosen = value

                        # Step to next level of dictionary tree
                        model_description.append("{}={}".format(parameter, str(value)))

                        # NS: this ugly if statement is to deal with the fact that the 3D models do not have p/s in reality,
                        # but the names of the files are p/s. So we need to change the name of the parameter to match the file name
                        # since for 1D models, we dont interpolate between p/s, but rather between p/p or s/s
                        if parameter == "mass" and self.atmosphere_dimension == "3D":
                            # take all values no matter the mass
                            if logg_chosen < 3:
                                dict_iter = dict_iter[1.0]
                            else:
                                dict_iter = dict_iter[0.0]
                        elif parameter == "spherical" and self.atmosphere_dimension == "3D":
                            try:
                                dict_iter = dict_iter[value]
                            except KeyError:
                                if value == "p":
                                    dict_iter = dict_iter["s"]
                                else:
                                    dict_iter = dict_iter["p"]
                        else:
                            dict_iter = dict_iter[value]

                    # Success -- we've found a model which matches all requested parameter.
                    # Extract filename of model we've found.
                    dict_iter = dict_iter['filename']

                except KeyError:
                    # We get a KeyError if there is no model matching the parameter combination we tried
                    # failed_on_parameter = (parameter, value, list(dict_iter.keys()))
                    # print(failed_on_parameter)
                    dict_iter = None
                    failures += 1
                marcs_model_list.append(dict_iter)
                model_description = "<" + ", ".join(model_description) + ">"

                # Produce debugging information about how we did finding models, but only if we want to be verbose
                # if False:
                #    if not failures:
                #        logging.info("Tried {}. Success.".format(model_description))
                #    else:
                #        logging.info("Tried {}. Failed on <{}>. Wanted {}, but only options were: {}.".
                #                     format(model_description, failed_on_parameter[0],
                #                            failed_on_parameter[1], failed_on_parameter[2]))
            # logging.info("Found {:d}/{:d} model atmospheres.".format(n_vertices - failures, n_vertices))

            # If there are MARCS models missing from the corners of the cuboid we tried, see which face had the most
            # corners missing, and move that face out by one grid row
            if failures:
                n_faces = 2 * len(interpolate_parameters)
                failures_per_face = []
                for cuboid_face_no in range(n_faces):  # Loop over 6 faces of cuboid
                    failure_count = 0
                    parameter_no = int(cuboid_face_no / 2)
                    option_no = cuboid_face_no & 1
                    for vertex in range(n_vertices):  # Loop over 8 vertices
                        if marcs_model_list[vertex] is None:
                            failure_option_no = int(bool(vertex & (2 ** parameter_no)))  # This is 0/1
                            if option_no == failure_option_no:
                                failure_count += 1
                    failures_per_face.append([failure_count, parameter_no, option_no])
                failures_per_face.sort(key=itemgetter(0))

                face_to_move = failures_per_face[-1]
                failure_count, parameter_no, option_no = face_to_move
                parameter_to_move = interpolate_parameters[parameter_no]
                options = self.marcs_values[parameter_to_move]
                parameter_descriptor = marcs_parameters[parameter_to_move]

                if option_no == 0:
                    parameter_descriptor[2] -= 1
                    if parameter_descriptor[2] < 0:
                        return {
                            "errors":
                                "Value of parameter <{}> needs to be in range {} to {}. You requested {}, " \
                                "and due to missing models we could not interpolate.". \
                                    format(parameter_to_move, options[0], options[-1],
                                           interpolate_parameters_around[parameter_to_move])
                        }
                    # logging.info("Moving lower bound of parameter <{}> from {} to {} and trying again. "
                    #             "This setting previously had {} failures.".
                    #             format(parameter_to_move, parameter_descriptor[0],
                    #                    options[parameter_descriptor[2]], failure_count))
                    parameter_descriptor[0] = options[parameter_descriptor[2]]
                else:
                    parameter_descriptor[3] += 1
                    if parameter_descriptor[3] >= len(options):
                        return {
                            "errors":
                                "Value of parameter <{}> needs to be in range {} to {}. You requested {}, " \
                                "and due to missing models we could not interpolate.". \
                                    format(parameter_to_move, options[0], options[-1],
                                           interpolate_parameters_around[parameter_to_move])
                        }
                    # logging.info("Moving upper bound of parameter <{}> from {} to {} and trying again. "
                    #             "This setting previously had {} failures.".
                    #             format(parameter_to_move, parameter_descriptor[1],
                    #                    options[parameter_descriptor[3]], failure_count))
                    parameter_descriptor[1] = options[parameter_descriptor[3]]

        logging.debug(marcs_model_list)
        # save marcs model list in a file as one line at a time
        #with open(os_path.join("test12312123.txt"), 'a') as f:
        #    for item in marcs_model_list:
        #        f.write("%s\n" % item)

        # print(len(np.loadtxt(os_path.join(self.departure_file_path,self.depart_aux_file[element]), dtype='str')))
        if self.nlte_flag:
            for element in self.model_atom_file:
                element_abundance = self._get_element_abundance(element)

                if self.verbose:
                    stdout = None
                    stderr = subprocess.STDOUT
                else:
                    stdout = open('/dev/null', 'w')
                    stderr = subprocess.STDOUT
                # Write configuration input for interpolator
                output = os_path.join(self.tmp_dir, self.marcs_model_name)
                model_test = "{}.test".format(output)
                interpol_config = ""
                self.marcs_model_list_global = marcs_model_list
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
                if spherical:
                    self.turbulent_velocity = microturbulence
        else:
            if self.verbose:
                stdout = None
                stderr = subprocess.STDOUT
            else:
                stdout = open('/dev/null', 'w')
                stderr = subprocess.STDOUT
            # print(len(np.loadtxt(os_path.join(self.departure_file_path,self.depart_aux_file[element]), dtype='str')))
            # Write configuration input for interpolator
            output = os_path.join(self.tmp_dir, self.marcs_model_name)
            # output = os_path.join('Testout/', self.marcs_model_name)
            # print(output)
            model_test = "{}.test".format(output)
            interpol_config = ""
            self.marcs_model_list_global = marcs_model_list
            # print(marcs_model_list)
            # print(self.free_abundances["Ca"]+float(solar_abundances["Ca"]))
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
            # print(self.free_abundances["Ba"])
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
                # print("spud")
                return {
                    "interpol_config": interpol_config,
                    "errors": "MARCS model atmosphere interpolation failed."
                }
            # print("spud")
            if spherical:
                self.turbulent_velocity = microturbulence

        return {
            "interpol_config": interpol_config,
            "spherical": spherical,
            "errors": None
        }


    def make_atmosphere_properties(self, spherical, element):
        logging.debug(f"make_atmosphere_properties: spherical={spherical}, element={element}, {self.free_abundances}")
        if self.nlte_flag:
            # Write configuration input for interpolator
            output = os_path.join(self.tmp_dir, self.marcs_model_name)
            model_test = "{}.test".format(output)
            interpol_config = ""
            for line in self.marcs_model_list_global:
                interpol_config += "'{}{}'\n".format(self.marcs_grid_path, line)
            interpol_config += "'{}.interpol'\n".format(output)
            interpol_config += "'{}.alt'\n".format(output)
            interpol_config += "'{}_{}_coef.dat'\n".format(output, element)  # needed for nlte interpolator
            interpol_config += "'{}'\n".format(
                os_path.join(self.departure_file_path, self.depart_bin_file[element]))  # needed for nlte interpolator
            interpol_config += "'{}'\n".format(
                os_path.join(self.departure_file_path, self.depart_aux_file[element]))  # needed for nlte interpolator
            interpol_config += "{}\n".format(self.aux_file_length_dict[element])
            interpol_config += "{}\n".format(self.t_eff)
            interpol_config += "{}\n".format(self.log_g)
            interpol_config += "{:.6f}\n".format(round(float(self.metallicity), 6))
            element_abundance = self._get_element_abundance(element)
            interpol_config += "{:.6f}\n".format(round(float(element_abundance), 6))
            interpol_config += ".false.\n"  # test option - set to .true. if you want to plot comparison model (model_test)
            interpol_config += ".false.\n"  # MARCS binary format (.true.) or MARCS ASCII web format (.false.)?
            interpol_config += "'{}'\n".format(model_test)
        else:
            output = os_path.join(self.tmp_dir, self.marcs_model_name)
            model_test = "{}.test".format(output)
            interpol_config = ""
            for line in self.marcs_model_list_global:
                interpol_config += "'{}{}'\n".format(self.marcs_grid_path, line)
            interpol_config += "'{}.interpol'\n".format(output)
            interpol_config += "'{}.alt'\n".format(output)
            interpol_config += "{}\n".format(self.t_eff)
            interpol_config += "{}\n".format(self.log_g)
            interpol_config += "{}\n".format(self.metallicity)
            interpol_config += ".false.\n"  # test option - set to .true. if you want to plot comparison model (model_test)
            interpol_config += ".false.\n"  # MARCS binary format (.true.) or MARCS ASCII web format (.false.)?
            interpol_config += "'{}'\n".format(model_test)

        return {
            "interpol_config": interpol_config,
            "spherical": spherical,
            "errors": None
        }

    def calculate_atmosphere(self):
        # figure out if we need to interpolate the model atmosphere for microturbulence
        possible_turbulence = [0.0, 1.0, 2.0, 5.0]
        flag_dont_interp_microturb = False
        for i in range(len(possible_turbulence)):
            if self.turbulent_velocity == possible_turbulence[i]:
                flag_dont_interp_microturb = True

        if self.log_g < 3:
            flag_dont_interp_microturb = True

        logging.debug(f"flag_dont_interp_microturb: {flag_dont_interp_microturb} {self.turbulent_velocity} {self.t_eff} {self.log_g}")

        if not flag_dont_interp_microturb and self.turbulent_velocity < 2.0 and (
                self.turbulent_velocity > 1.0 or (self.turbulent_velocity < 1.0 and self.t_eff < 3900.)):
            # Bracket the microturbulence to figure out what two values to generate the models to interpolate between using Andy's code
            turbulence_low = 0.0
            microturbulence = self.turbulent_velocity
            for i in range(len(possible_turbulence)):
                if self.turbulent_velocity > possible_turbulence[i]:
                    turbulence_low = possible_turbulence[i]
                    place = i
            turbulence_high = possible_turbulence[place + 1]
            # print(turbulence_low,turbulence_high)

            self.turbulent_velocity = turbulence_low
            atmosphere_properties_low = self._generate_model_atmosphere()
            # print(marcs_model_list_global)
            low_model_name = os_path.join(self.tmp_dir, self.marcs_model_name)
            low_model_name += '.interpol'
            if atmosphere_properties_low['errors']:
                return atmosphere_properties_low
            self.turbulent_velocity = turbulence_high
            atmosphere_properties_high = self._generate_model_atmosphere()
            high_model_name = os_path.join(self.tmp_dir, self.marcs_model_name)
            high_model_name += '.interpol'
            if atmosphere_properties_high['errors']:
                return atmosphere_properties_high

            self.turbulent_velocity = microturbulence
            # self.tmp_dir = temp_dir

            # interpolate and find a model atmosphere for the microturbulence
            self.marcs_model_name = "marcs_tef{:.1f}_g{:.2f}_z{:.2f}_tur{:.2f}".format(self.t_eff, self.log_g,
                                                                                       self.metallicity,
                                                                                       self.turbulent_velocity)
            f_low = open(low_model_name, 'r')
            lines_low = f_low.read().splitlines()
            t_low, temp_low, pe_low, pt_low, micro_low, lum_low, spud_low = np.loadtxt(
                open(low_model_name, 'rt').readlines()[:-8], skiprows=1, unpack=True)

            f_high = open(high_model_name, 'r')
            lines_high = f_high.read().splitlines()
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
            # print(interp_model_name)
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
                    atmosphere_properties = self.make_atmosphere_properties(atmosphere_properties_low['spherical'],
                                                                            element)
                    low_coef_dat_name = low_model_name.replace('.interpol', '_{}_coef.dat'.format(element))
                    logging.debug(f"low_coef_dat_name: {low_coef_dat_name}")
                    f_coef_low = open(low_coef_dat_name, 'r')
                    lines_coef_low = f_coef_low.read().splitlines()
                    f_coef_low.close()

                    high_coef_dat_name = os_path.join(self.tmp_dir, self.marcs_model_name)
                    high_coef_dat_name += '_{}_coef.dat'.format(element)

                    high_coef_dat_name = high_model_name.replace('.interpol', '_{}_coef.dat'.format(element))
                    logging.debug(f"high_coef_dat_name: {high_coef_dat_name}")
                    f_coef_high = open(high_coef_dat_name, 'r')
                    lines_coef_high = f_coef_high.read().splitlines()
                    f_coef_high.close()

                    interp_coef_dat_name = os_path.join(self.tmp_dir, self.marcs_model_name)
                    interp_coef_dat_name += '_{}_coef.dat'.format(element)

                    #num_lines = np.loadtxt(low_coef_dat_name, unpack=True, skiprows=9, max_rows=1)

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
                        # TODO check if any nans or negative values
                        print(*fields_interp_print, file=g)
                    for i in range(10 + 2 * len(t_interp) + 1, len(lines_coef_low)):
                        print(lines_coef_low[i], file=g)
                    g.close()
            else:
                # atmosphere_properties = atmosphere_properties_low
                atmosphere_properties = self.make_atmosphere_properties(atmosphere_properties_low['spherical'], 'Fe')

        elif not flag_dont_interp_microturb and self.turbulent_velocity > 2.0:  # not enough models to interp if higher than 2
            microturbulence = self.turbulent_velocity  # just use 2.0 for the model if between 2 and 3
            self.turbulent_velocity = 2.0
            atmosphere_properties = self._generate_model_atmosphere()
            if atmosphere_properties['errors']:
                return atmosphere_properties
            self.turbulent_velocity = microturbulence

        elif not flag_dont_interp_microturb and self.turbulent_velocity < 1.0 and self.t_eff >= 3900.:  # not enough models to interp if lower than 1 and t_eff > 3900
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
                print(atmosphere_properties['errors'])
                return atmosphere_properties
        else:
            print("Unexpected error?")
        self.atmosphere_properties = atmosphere_properties
        # print(self.atmosphere_properties)

    @abc.abstractmethod
    def synthesize_spectra(self):
        """Abstract method that should be implemented by child classes."""
        pass


def fetch_marcs_grid(marcs_grid_list: str, marcs_parameters_to_ignore: list):
    """
    Get a list of all of the MARCS models we have.

    :return:
        None
    """

    marcs_values = {
        "spherical": [], "temperature": [], "log_g": [], "mass": [], "turbulence": [], "model_type": [],
        "metallicity": [], "a": [], "c": [], "n": [], "o": [], "r": [], "s": []}

    model_temperatures = []
    model_logs = []
    model_mets = []

    pattern = r"([sp])(\d\d\d\d)_g(....)_m(...)_t(..)_(..)_z(.....)_" \
              r"a(.....)_c(.....)_n(.....)_o(.....)_r(.....)_s(.....).mod"

    marcs_value_keys = [i for i in list(marcs_values.keys()) if i not in marcs_parameters_to_ignore]
    marcs_value_keys.sort()
    marcs_models = {}

    # marcs_models = glob.glob(os_path.join(self.marcs_grid_path, "*"))  # 18.11.22 NS: Takes several seconds here per star, is not used anywhere though? Uncommented for now at least
    marcs_nlte_models = np.loadtxt(marcs_grid_list, dtype='str', usecols=(0,), unpack=True)
    spud_models = []
    for i in range(len(marcs_nlte_models)):
        aux_pattern = r"(\d\d\d\d)_g(....)_m(...)_t(..)_(..)_z(.....)_" \
                      r"a(.....)_c(.....)_n(.....)_o(.....)_r(.....)_s(.....)"
        re_test_aux = re.match(aux_pattern, marcs_nlte_models[i])
        mass = float(re_test_aux.group(3))
        if mass == 0.0:
            spud = "p" + marcs_nlte_models[i] + ".mod"
        else:
            spud = "s" + marcs_nlte_models[i] + ".mod"
        spud_models.append(spud)

    marcs_nlte_models = spud_models

    for item in marcs_nlte_models:

        # Extract model parameters from .mod filename
        filename = os_path.split(item)[1]
        # filename = item
        re_test = re.match(pattern, filename)
        assert re_test is not None, "Could not parse MARCS model filename <{}>".format(filename)

        try:
            model = {
                "spherical": re_test.group(1),
                "temperature": float(re_test.group(2)),
                "log_g": float(re_test.group(3)),
                "mass": float(re_test.group(4)),
                "turbulence": float(re_test.group(5)),  # micro turbulence assumed in MARCS atmosphere, km/s
                "model_type": re_test.group(6),
                "metallicity": float(re_test.group(7)),
                "a": float(re_test.group(8)),
                "c": float(re_test.group(9)),
                "n": float(re_test.group(10)),
                "o": float(re_test.group(11)),
                "r": float(re_test.group(12)),
                "s": float(re_test.group(13))
            }
            model_temperatures.append(model["temperature"])
            model_logs.append(model["log_g"])
            model_mets.append(model["metallicity"])
        except ValueError:
            # logging.info("Could not parse MARCS model filename <{}>".format(filename))
            raise

        # Keep a list of all of the parameter values we've seen
        for parameter, value in model.items():
            if value not in marcs_values[parameter]:
                marcs_values[parameter].append(value)

        # Keep a list of all the models we've got in the grid
        dict_iter = marcs_models
        # print(dict_iter)
        for parameter in marcs_value_keys:
            value = model[parameter]
            if value not in dict_iter:
                dict_iter[value] = {}
            dict_iter = dict_iter[value]
        # if "filename" in dict_iter:
        # logging.info("Warning: MARCS model <{}> duplicates one we already have.".format(item))
        dict_iter["filename"] = item

    model_temperatures = np.asarray(model_temperatures)
    model_logs = np.asarray(model_logs)
    model_mets = np.asarray(model_mets)

    # Sort model parameter values into order
    for parameter in marcs_value_keys:
        marcs_values[parameter].sort()

    model_temperatures, model_logs, model_mets = None, None, None  # i think not used, but eats memory

    return model_temperatures, model_logs, model_mets, marcs_value_keys, marcs_models, marcs_values

