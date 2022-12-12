from __future__ import annotations

import collections
import subprocess
import os
from os import path as os_path
import glob
import re
from operator import itemgetter
import numpy as np
import math

from solar_abundances import solar_abundances, periodic_table
from solar_isotopes import solar_isotopes


def closest_available_value(target: float, options: list[float]) -> float:
    """
    Return the option from a list which most closely matches some target value.

    :param target:
        The target value that we're trying to match.
    :param options:
        The list of possible values that we can try to match to target.
    :return:
        The option value which is closest to <target>.
    """
    options = np.asarray(options)
    idx = (np.abs(options - target)).argmin()
    return options[idx]


class TurboSpectrum:
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
    marcs_model_list_global = [] #needed for microturbulence interpolation

    # It is safe to ignore these parameters in MARCS model descriptions
    # This includes interpolating between models with different values of these settings
    marcs_parameters_to_ignore = ["a", "c", "n", "o", "r", "s"]

    def __init__(self, turbospec_path: str, interpol_path: str, line_list_paths: str, marcs_grid_path: str,
                 marcs_grid_list: str, model_atom_path: str, departure_file_path: str):
        """
        Instantiate a class for generating synthetic stellar spectra using Turbospectrum.

        :param turbospec_path: Path where the turbospectrum binaries 'babsma' and 'bsyn' can be found.
        :param interpol_path: Path where the compiled interpol_modeles.f binary can be found.
        :param line_list_paths: Path(s) where line lists for synthetic spectra can be found. Specify as either a string, or a list of strings.
        :param marcs_grid_path: Path where a grid of MARCS .mod files can be found. These contain the model atmospheres we use.
        :param model_atom_path: Path to the model atom paths
        :param departure_file_path: Path to the NLTE departure file paths
        """

        if not isinstance(line_list_paths, (list, tuple)):
            line_list_paths = [line_list_paths]

        self.turbospec_path = turbospec_path
        self.interpol_path = interpol_path
        self.line_list_paths = line_list_paths
        self.marcs_grid_path = marcs_grid_path
        self.marcs_grid_list = marcs_grid_list
        self.model_atom_path = model_atom_path
        self.departure_file_path = departure_file_path

        # Default spectrum parameters
        self.lambda_min: float = 5100  # Angstrom
        self.lambda_max: float = 5200
        self.lambda_delta: float = 0.05
        self.metallicity: float = -1.5
        self.stellar_mass: float = 1
        self.log_g: float = 2.0
        self.t_eff: float = 5100.0
        self.turbulent_velocity: float = 2.0  # micro turbulence, km/s
        self.free_abundances: dict = None
        self.free_isotopes: dict = None
        self.sphere: bool = None
        self.alpha = None   # not used?
        self.s_process = 0  # not used?
        self.r_process = 0  # not used?
        self.verbose: bool = False
        self.line_list_files = None

        # parameters needed for nlte and <3D> calculations
        self.nlte_flag: bool = False
        self.atmosphere_dimension: str = "1D"
        self.windows_flag: bool = False
        self.depart_bin_file = None
        self.depart_aux_file = None
        self.model_atom_file = None
        self.segment_file = "spud.txt"
        self.cont_mask_file = "spud.txt"
        self.line_mask_file = "spud.txt"

        # Create temporary directory
        self.id_string = "turbospec_{:d}".format(os.getpid())
        self.tmp_dir = os_path.join("/tmp", self.id_string)
        #self.tmp_dir = os_path.join("/Users/gerber/gitprojects/SAPP/tests/", "current_run")
        os.system("mkdir -p {}".format(self.tmp_dir))

        # Look up what MARCS models we have
        #self.counter_marcs = 0
        self.marcs_model_name = "default"
        self.counter_spectra = 0
        self.marcs_values = None
        self.marcs_value_keys = []
        self.marcs_models = {}
        self.model_temperatures = []
        self.model_logs = []
        self.model_mets = []
        self._fetch_marcs_grid()


    def _fetch_marcs_grid(self):
        """
        Get a list of all of the MARCS models we have.

        :return:
            None
        """
        #TODO: calling it every time. But we only need to run once.
        pattern = r"([sp])(\d\d\d\d)_g(....)_m(...)_t(..)_(..)_z(.....)_" \
                  r"a(.....)_c(.....)_n(.....)_o(.....)_r(.....)_s(.....).mod"

        self.marcs_values = {
            "spherical": [], "temperature": [], "log_g": [], "mass": [], "turbulence": [], "model_type": [],
            "metallicity": [], "a": [], "c": [], "n": [], "o": [], "r": [], "s": []
        }

        self.marcs_value_keys = [i for i in list(self.marcs_values.keys()) if i not in self.marcs_parameters_to_ignore]
        self.marcs_value_keys.sort()
        self.marcs_models = {}

        #marcs_models = glob.glob(os_path.join(self.marcs_grid_path, "*"))  # 18.11.22 NS: Takes several seconds here per star, is not used anywhere though? Uncommented for now at least
        marcs_nlte_models = np.loadtxt(self.marcs_grid_list, dtype='str', usecols=(0,), unpack=True)
        spud_models = []
        for i in range(len(marcs_nlte_models)):
            aux_pattern = r"(\d\d\d\d)_g(....)_m(...)_t(..)_(..)_z(.....)_" \
                          r"a(.....)_c(.....)_n(.....)_o(.....)_r(.....)_s(.....)"
            re_test_aux = re.match(aux_pattern, marcs_nlte_models[i])
            mass = float(re_test_aux.group(3))
            if mass == 0.0:
                spud = "p"+marcs_nlte_models[i]+".mod"
            else:
                spud = "s"+marcs_nlte_models[i]+".mod"
            spud_models.append(spud)

        marcs_nlte_models = spud_models

        for item in marcs_nlte_models:

            # Extract model parameters from .mod filename
            filename = os_path.split(item)[1]
            #filename = item
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
                self.model_temperatures.append(model["temperature"])
                self.model_logs.append(model["log_g"])
                self.model_mets.append(model["metallicity"])
            except ValueError:
                #logging.info("Could not parse MARCS model filename <{}>".format(filename))
                raise

            # Keep a list of all of the parameter values we've seen
            for parameter, value in model.items():
                if value not in self.marcs_values[parameter]:
                    self.marcs_values[parameter].append(value)

            # Keep a list of all the models we've got in the grid
            dict_iter = self.marcs_models
            #print(dict_iter)
            for parameter in self.marcs_value_keys:
                value = model[parameter]
                if value not in dict_iter:
                    dict_iter[value] = {}
                dict_iter = dict_iter[value]
            #if "filename" in dict_iter:
                #logging.info("Warning: MARCS model <{}> duplicates one we already have.".format(item))
            dict_iter["filename"] = item

        self.model_temperatures = np.asarray(self.model_temperatures)
        self.model_logs = np.asarray(self.model_logs)
        self.model_mets = np.asarray(self.model_mets)

        # Sort model parameter values into order
        for parameter in self.marcs_value_keys:
            self.marcs_values[parameter].sort()

    def configure(self, lambda_min=None, lambda_max=None, lambda_delta=None,
                  metallicity=None, log_g=None, t_eff=None, stellar_mass=None,
                  turbulent_velocity=None, free_abundances=None, free_isotopes=None,
                  sphere=None, alpha=None, s_process=None, r_process=None,
                  line_list_paths=None, line_list_files=None,
                  verbose=None, counter_spectra=None, temp_directory=None, nlte_flag=None, atmosphere_dimension=None, windows_flag=None,
                  depart_bin_file=None, depart_aux_file=None, model_atom_file=None,
                  segment_file=None, cont_mask_file=None, line_mask_file=None):
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
            self.free_abundances = free_abundances
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
            print("turbulent_velocity is not used since model atmosphere is 3D")

    def _generate_model_atmosphere(self):
        """
        Generates an interpolated model atmosphere from the MARCS grid using the interpol.f routine developed by
        T. Masseron (Masseron, PhD Thesis, 2006). This is a python wrapper for that fortran code.
        """
        #self.counter_marcs += 1
        #self.marcs_model_name = "marcs_{:08d}".format(self.counter_marcs)
        self.marcs_model_name = "marcs_tef{:.1f}_g{:.2f}_z{:.2f}_tur{:.2f}".format(self.t_eff, self.log_g, self.metallicity, self.turbulent_velocity)
        global marcs_model_list_global

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
        marcs_parameters['turbulence'] = self.turbulent_velocity #JMG line to make microturbulence an adjustable variable
        #print(marcs_parameters)
        if spherical:
            if self.atmosphere_dimension == "1D":
                marcs_parameters['spherical'] = "s"
                marcs_parameters['mass'] = closest_available_value(self.stellar_mass, self.marcs_values['mass'])
            else:
                marcs_parameters['spherical'] = "p"
                marcs_parameters['mass'] = 0
            microturbulence = self.turbulent_velocity
            self.turbulent_velocity = 2.0
            #print(marcs_parameters['mass'])
            #marcs_parameters['mass'] = self.closest_available_value(self.marcs_values['mass'])
        else:
            marcs_parameters['spherical'] = "p"
            marcs_parameters['mass'] = 0  # All plane-parallel models have mass set to zero

        #quick setting to reduce temperature in case temperature is higher than grid allows, will give warning that it has happened
        # TODO: logg == 4.0? should be inequality sign maybe?
        if self.t_eff >= 6500 and self.log_g == 4.0 and self.atmosphere_dimension == "3D":
            print("warning temp was {} and the highest value available is 6500. setting temp to 6500 to interpolate model atmosphere. will be {:.2f} for spectrum generation".format(self.t_eff, self.t_eff))
            temp_teff = self.t_eff
            self.t_eff = 6499

        # Pick MARCS settings which bracket requested stellar parameters
        interpolate_parameters = ("metallicity", "log_g", "temperature")

        interpolate_parameters_around = {"temperature": self.t_eff,
                                         "log_g": self.log_g,
                                         "metallicity": self.metallicity,
                                         }
        # go through each value that is interpolated
        for key in interpolate_parameters:
            value = interpolate_parameters_around[key]
            options = self.marcs_values[key]    # get what values exist in marcs values
            if (value < options[0]) or (value > options[-1]):   # checks that the value is within the marcs possible values
                return {
                    "errors": "Value of parameter <{}> needs to be in range {} to {}. You requested {}.".
                        format(key, options[0], options[-1], value)
                }
            for index in range(len(options) - 1):
                if value < options[index + 1]:
                    break
            #Mar. 11, 2022 added if statement for what to do if parameter is on the vertex. model interpolator needs the value for both models to be on that vertex or else will falsely think it's extrapolating
            if value == options[index]:
                marcs_parameters[key] = [options[index], options[index], index, index]
            elif value == options[index+1]:
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

                        # Step to next level of dictionary tree
                        model_description.append("{}={}".format(parameter, str(value)))
                        dict_iter = dict_iter[value]

                    # Success -- we've found a model which matches all requested parameter.
                    # Extract filename of model we've found.
                    dict_iter = dict_iter['filename']

                except KeyError:
                    # We get a KeyError if there is no model matching the parameter combination we tried
                    #failed_on_parameter = (parameter, value, list(dict_iter.keys()))
                    #print(failed_on_parameter)
                    dict_iter = None
                    failures += 1
                marcs_model_list.append(dict_iter)
                model_description = "<" + ", ".join(model_description) + ">"

                # Produce debugging information about how we did finding models, but only if we want to be verbose
                #if False:
                #    if not failures:
                #        logging.info("Tried {}. Success.".format(model_description))
                #    else:
                #        logging.info("Tried {}. Failed on <{}>. Wanted {}, but only options were: {}.".
                #                     format(model_description, failed_on_parameter[0],
                #                            failed_on_parameter[1], failed_on_parameter[2]))
            #logging.info("Found {:d}/{:d} model atmospheres.".format(n_vertices - failures, n_vertices))

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

                if self.atmosphere_dimension == "1D" or self.atmosphere_dimension == "3D":
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
                        #logging.info("Moving lower bound of parameter <{}> from {} to {} and trying again. "
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
                        #logging.info("Moving upper bound of parameter <{}> from {} to {} and trying again. "
                        #             "This setting previously had {} failures.".
                        #             format(parameter_to_move, parameter_descriptor[1],
                        #                    options[parameter_descriptor[3]], failure_count))
                        parameter_descriptor[1] = options[parameter_descriptor[3]]
                elif self.atmosphere_dimension == "3D":
                    # TODO: remove? refactor?
                    # This part was written to be able to interpolate 3D models when they are not at equal vertices.
                    # Now it seems like it is not needed? Not removing, but this part of the code will currently
                    # never run.
                    marcs_values_new = {}
                    marcs_values_new["temperature"] = self.model_temperatures
                    marcs_values_new["log_g"] = self.model_logs
                    marcs_values_new["metallicity"] = self.model_mets

                    interpolate_parameters = ("temperature", "log_g", "metallicity")

                    def get_args(value_to_search: float, array: np.ndarray[float], values_to_dlt: np.ndarray[float]) -> tuple[int, int]:
                        uniq_array = np.unique(array)
                        new_uniq_array = uniq_array[np.isin(uniq_array, values_to_dlt, invert=True)]
                        if value_to_search in new_uniq_array:
                            args_to_use_first = np.where(array == value_to_search)[0]
                            args_to_use_second = args_to_use_first
                        else:
                            new_uniq_array = uniq_array[np.where(uniq_array < value_to_search)[0]]
                            new_uniq_array = new_uniq_array[np.isin(new_uniq_array, values_to_dlt, invert=True)]
                            first_closest_value = new_uniq_array[
                                (np.abs(new_uniq_array - value_to_search)).argmin()]

                            new_uniq_array = uniq_array[np.where(uniq_array > value_to_search)[0]]
                            new_uniq_array = new_uniq_array[np.isin(new_uniq_array, values_to_dlt, invert=True)]
                            second_closest_value = new_uniq_array[
                                (np.abs(new_uniq_array - value_to_search)).argmin()]

                            args_to_use_first = np.where(array == first_closest_value)[0]
                            args_to_use_second = np.where(array == second_closest_value)[0]
                        # print(args_to_use_first, args_to_use_second)
                        return args_to_use_first, args_to_use_second

                    def find_interp_indices(value: float, options: np.ndarray[float], values_to_ignore: np.ndarray[float]) -> tuple[int, int, list[float, float]]:
                        # value = interpolate_parameters_around[key]
                        # options = marcs_values[key]
                        # options = options[args_to_use]
                        if (value < np.min(options[np.isin(options, values_to_ignore, invert=True)])) or (
                                value > np.max(options[np.isin(options, values_to_ignore, invert=True)])):  # checks that the value is within the marcs possible values
                            return None, None, {
                                "errors": f"Value of parameter  needs to be in range {np.min(options)} to {np.max(options)}. You requested {value}. OR the other parameters are not within the range"
                            }
                        args_to_use_first, args_to_use_second = get_args(value, options, values_to_ignore)
                        return args_to_use_first, args_to_use_second, [options[args_to_use_first][0],
                                                                       options[args_to_use_second][0]]

                    temperatures_to_ignore = np.array([])
                    loggs_to_ignore = np.array([])
                    metallicities_to_ignore = np.array([])

                    def find_new_marcs_models(temperatures_to_ignore, loggs_to_ignore, metallicities_to_ignore):
                        # temperature
                        value_temp = interpolate_parameters_around[interpolate_parameters[0]]
                        options_temp = marcs_values_new[interpolate_parameters[0]]
                        args_to_use_first, args_to_use_second, out_values_temp = find_interp_indices(value_temp,
                                                                                                     options_temp,
                                                                                                     temperatures_to_ignore)

                        # logg
                        value_logg = interpolate_parameters_around[interpolate_parameters[1]]
                        options_logg_first = marcs_values_new[interpolate_parameters[1]][args_to_use_first]
                        args_to_use_first_1, args_to_use_second_1, out_values_logg_1 = find_interp_indices(value_logg,
                                                                                                           options_logg_first,
                                                                                                           loggs_to_ignore)

                        if args_to_use_first_1 is None:
                            return np.append(temperatures_to_ignore, out_values_temp[0]), loggs_to_ignore, metallicities_to_ignore, False

                        options_logg_second = marcs_values_new[interpolate_parameters[1]][args_to_use_second]
                        args_to_use_first_2, args_to_use_second_2, out_values_logg_2 = find_interp_indices(value_logg,
                                                                                                           options_logg_second,
                                                                                                           loggs_to_ignore)

                        if args_to_use_first_2 is None:
                            return np.append(temperatures_to_ignore, out_values_temp[1]), loggs_to_ignore, metallicities_to_ignore, False

                        """if args_to_use_first_1 is None:
                            #options_temp = marcs_values_new[interpolate_parameters[0]]
                            options_temp = np.delete(options_temp, np.where(options_temp == out_values_temp[0]), axis=0)
                            args_to_use_first, args_to_use_second, out_values_temp = find_interp_indices(value_temp,
                                                                                                         options_temp)

                            # logg
                            options_logg_first = marcs_values_new[interpolate_parameters[1]][args_to_use_first]
                            args_to_use_first_1, args_to_use_second_1, out_values_logg_1 = find_interp_indices(value_logg,
                                                                                                               options_logg_first)

                            options_logg_second = marcs_values_new[interpolate_parameters[1]][args_to_use_second]
                            args_to_use_first_2, args_to_use_second_2, out_values_logg_2 = find_interp_indices(value_logg,
                                                                                                               options_logg_second)

                        if args_to_use_first_2 is None:
                            #options_temp = marcs_values_new[interpolate_parameters[0]]
                            options_temp = np.delete(options_temp, np.where(options_temp == out_values_temp[1]), axis=0)
                            args_to_use_first, args_to_use_second, out_values_temp = find_interp_indices(value_temp,
                                                                                                         options_temp)

                            # logg
                            options_logg_first = marcs_values_new[interpolate_parameters[1]][args_to_use_first]
                            args_to_use_first_1, args_to_use_second_1, out_values_logg_1 = find_interp_indices(value_logg,
                                                                                                               options_logg_first)

                            options_logg_second = marcs_values_new[interpolate_parameters[1]][args_to_use_second]
                            args_to_use_first_2, args_to_use_second_2, out_values_logg_2 = find_interp_indices(value_logg,
                                                                                                               options_logg_second)"""

                        # metallicity
                        value_met = interpolate_parameters_around[interpolate_parameters[2]]
                        options_met_11 = marcs_values_new[interpolate_parameters[2]][args_to_use_first][
                            args_to_use_first_1]
                        args_to_use_first_10, args_to_use_second_10, out_values_met_10 = find_interp_indices(value_met,
                                                                                                             options_met_11,
                                                                                                             metallicities_to_ignore)

                        if args_to_use_first_10 is None:
                            return temperatures_to_ignore, np.append(loggs_to_ignore, out_values_logg_1[0]), metallicities_to_ignore, False

                        options_met_12 = marcs_values_new[interpolate_parameters[2]][args_to_use_first][
                            args_to_use_second_1]
                        args_to_use_first_20, args_to_use_second_20, out_values_met_20 = find_interp_indices(value_met,
                                                                                                             options_met_12,
                                                                                                             metallicities_to_ignore)

                        if args_to_use_first_20 is None:
                            return temperatures_to_ignore, np.append(loggs_to_ignore, out_values_logg_1[1]), metallicities_to_ignore, False

                        options_met_21 = marcs_values_new[interpolate_parameters[2]][args_to_use_second][
                            args_to_use_first_2]
                        args_to_use_first_30, args_to_use_second_30, out_values_met_30 = find_interp_indices(value_met,
                                                                                                             options_met_21,
                                                                                                             metallicities_to_ignore)

                        if args_to_use_first_30 is None:
                            return temperatures_to_ignore, np.append(loggs_to_ignore, out_values_logg_2[0]), metallicities_to_ignore, False

                        options_met_22 = marcs_values_new[interpolate_parameters[2]][args_to_use_second][
                            args_to_use_second_2]
                        args_to_use_first_40, args_to_use_second_40, out_values_met_40 = find_interp_indices(value_met,
                                                                                                             options_met_22,
                                                                                                             metallicities_to_ignore)

                        if args_to_use_first_40 is None:
                            return temperatures_to_ignore, np.append(loggs_to_ignore,out_values_logg_2[1]), metallicities_to_ignore, False

                        return [out_values_temp, out_values_logg_1, out_values_logg_2, out_values_met_10, out_values_met_20, out_values_met_30, out_values_met_40], None, None, True

                    while True:
                        temperatures_to_ignore, loggs_to_ignore, metallicities_to_ignore, completed = find_new_marcs_models(temperatures_to_ignore, loggs_to_ignore, metallicities_to_ignore)
                        if completed:
                            out_values_temp, out_values_logg_1, out_values_logg_2, out_values_met_10, \
                                out_values_met_20, out_values_met_30, out_values_met_40 = temperatures_to_ignore[0], temperatures_to_ignore[1], temperatures_to_ignore[2], temperatures_to_ignore[3], temperatures_to_ignore[4], temperatures_to_ignore[5], temperatures_to_ignore[6]
                            break
                        else:
                            if collections.Counter(temperatures_to_ignore) == collections.Counter(marcs_values_new[interpolate_parameters[0]]) \
                                    or np.min(marcs_values_new[interpolate_parameters[0]]) in temperatures_to_ignore \
                                    or np.max(marcs_values_new[interpolate_parameters[0]]) in temperatures_to_ignore:
                                return {"errors": "Could not find the models for interpolation"}

                    def get_marcs_model_atmosphere(model):
                        dict_iter = self.marcs_models

                        for parameter in self.marcs_value_keys:
                            if parameter == "spherical":
                                try:
                                    value = 'p'
                                    dict_iter = dict_iter[value]
                                except KeyError:
                                    value = 's'
                                    dict_iter = dict_iter[value]
                            elif parameter == "mass":
                                try:
                                    value = 1.0
                                    dict_iter = dict_iter[value]
                                except KeyError:
                                    value = 0.0
                                    dict_iter = dict_iter[value]
                            else:
                                value = model[parameter]
                                dict_iter = dict_iter[value]

                        dict_iter = dict_iter['filename']
                        return dict_iter

                    marcs_models = []


                    try:
                        model = {"temperature": out_values_temp[0], "log_g": out_values_logg_1[0], "metallicity": out_values_met_10[0],
                                  "model_type": 'st', "spherical": 'p', 'turbulence': 2.0, 'mass': 0.0}

                        marcs_models.append(get_marcs_model_atmosphere(model))
                        model["metallicity"] = out_values_met_10[1]
                        marcs_models.append(get_marcs_model_atmosphere(model))
                        model["log_g"] = out_values_logg_1[1]
                        model["metallicity"] = out_values_met_20[0]
                        marcs_models.append(get_marcs_model_atmosphere(model))
                        model["metallicity"] = out_values_met_20[1]
                        marcs_models.append(get_marcs_model_atmosphere(model))
                        model["temperature"] = out_values_temp[1]
                        model["log_g"] = out_values_logg_2[0]
                        model["metallicity"] = out_values_met_30[0]
                        marcs_models.append(get_marcs_model_atmosphere(model))
                        model["metallicity"] = out_values_met_30[1]
                        marcs_models.append(get_marcs_model_atmosphere(model))
                        model["log_g"] = out_values_logg_2[1]
                        model["metallicity"] = out_values_met_40[0]
                        marcs_models.append(get_marcs_model_atmosphere(model))
                        model["metallicity"] = out_values_met_40[1]
                        marcs_models.append(get_marcs_model_atmosphere(model))
                    except KeyError:
                        return {"errors": f"Value of parameter needs to be in range. No clue where though haha"}

                    print("NEWW models:")
                    for model in marcs_models:
                        print(model)

                    marcs_model_list = marcs_models
                    failures = False



            #print(marcs_model_list)

        #print(len(np.loadtxt(os_path.join(self.departure_file_path,self.depart_aux_file[element]), dtype='str')))
        if self.nlte_flag == True:
            for element, abundance in self.free_abundances.items():
                #print(element,self.model_atom_file[element])
                #print("*******************")
                #print(abundance, self.free_abundances[element])
                #print("{:.2f}".format(round(float(self.free_abundances[element]),2)+float(solar_abundances[element])))
                #print("{:.2f}".format(round(float(abundance),2) + float(solar_abundances[element])))
                #print("{}".format(float(self.metallicity)))
                #print("{:.2f}".format(round(float(self.metallicity),2)))
                #print("{}".format(abundance))
                #print("{:.2f}".format(round(float(abundance),2)))
                #print("{:.2f}".format(round(float(self.free_abundances[element]),2)+float(solar_abundances[element])))
                #print("*******************")
                #print(element,self.model_atom_file[element])
                if self.model_atom_file[element] != "":
                    if self.verbose:
                        stdout = None
                        stderr = subprocess.STDOUT
                    else:
                        stdout = open('/dev/null', 'w')
                        stderr = subprocess.STDOUT
                    #print(len(np.loadtxt(os_path.join(self.departure_file_path,self.depart_aux_file[element]), dtype='str')))
                    # Write configuration input for interpolator
                    output = os_path.join(self.tmp_dir, self.marcs_model_name)
                    #output = os_path.join('Testout/', self.marcs_model_name)
                    #print(output)
                    model_test = "{}.test".format(output)
                    interpol_config = ""
                    marcs_model_list_global = marcs_model_list
                    #print(marcs_model_list)
                    #print(self.free_abundances["Ca"]+float(solar_abundances["Ca"]))
                    for line in marcs_model_list:
                        interpol_config += "'{}{}'\n".format(self.marcs_grid_path,line)
                    interpol_config += "'{}.interpol'\n".format(output)
                    interpol_config += "'{}.alt'\n".format(output)
                    interpol_config += "'{}_{}_coef.dat'\n".format(output, element) #needed for nlte interpolator
                    interpol_config += "'{}'\n".format(os_path.join(self.departure_file_path,self.depart_bin_file[element])) #needed for nlte interpolator
                    interpol_config += "'{}'\n".format(os_path.join(self.departure_file_path,self.depart_aux_file[element])) #needed for nlte interpolator
                    #interpol_config += "'/Users/gerber/gitprojects/TurboSpectrum2020/interpol_modeles_nlte/NLTEdata/1D_NLTE_grid_Fe_mean3D.bin'\n" #needed for nlte interpolator
                    #interpol_config += "'/Users/gerber/gitprojects/TurboSpectrum2020/interpol_modeles_nlte/NLTEdata/auxData_Fe_mean3D_marcs_names.txt'\n" #needed for nlte interpolator
                    #interpol_config += "'1D_NLTE_grid_Fe_MARCSfullGrid.bin'\n" #needed for nlte interpolator
                    #interpol_config += "'auxData_Fe_MARCSfullGrid.txt'\n" #needed for nlte interpolator
                    interpol_config += "{}\n".format(len(np.loadtxt(os_path.join(self.departure_file_path,self.depart_aux_file[element]), dtype='str')))
                    interpol_config += "{}\n".format(self.t_eff)
                    interpol_config += "{}\n".format(self.log_g)
                    interpol_config += "{:.2f}\n".format(round(float(self.metallicity),2))
                    interpol_config += "{:.2f}\n".format(round(float(self.free_abundances[element]),2)+float(solar_abundances[element]))
                    interpol_config += ".false.\n"  # test option - set to .true. if you want to plot comparison model (model_test)
                    interpol_config += ".false.\n"  # MARCS binary format (.true.) or MARCS ASCII web format (.false.)?
                    interpol_config += "'{}'\n".format(model_test)

                    # Now we run the FORTRAN model interpolator
                    #print(self.free_abundances["Ba"])
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
                        #print("spud")
                        return {
                            "interpol_config": interpol_config,
                            "errors": "MARCS model atmosphere interpolation failed."
                        }
                    #print("spud")
                    if spherical:
                        self.turbulent_velocity = microturbulence
        elif self.nlte_flag == False:
            if self.verbose:
                stdout = None
                stderr = subprocess.STDOUT
            else:
                stdout = open('/dev/null', 'w')
                stderr = subprocess.STDOUT
            #print(len(np.loadtxt(os_path.join(self.departure_file_path,self.depart_aux_file[element]), dtype='str')))
            # Write configuration input for interpolator
            output = os_path.join(self.tmp_dir, self.marcs_model_name)
            #output = os_path.join('Testout/', self.marcs_model_name)
            #print(output)
            model_test = "{}.test".format(output)
            interpol_config = ""
            marcs_model_list_global = marcs_model_list
            #print(marcs_model_list)
            #print(self.free_abundances["Ca"]+float(solar_abundances["Ca"]))
            for line in marcs_model_list:
                interpol_config += "'{}{}'\n".format(self.marcs_grid_path,line)
            interpol_config += "'{}.interpol'\n".format(output)
            interpol_config += "'{}.alt'\n".format(output)
            interpol_config += "{}\n".format(self.t_eff)
            interpol_config += "{}\n".format(self.log_g)
            interpol_config += "{}\n".format(self.metallicity)
            interpol_config += ".false.\n"  # test option - set to .true. if you want to plot comparison model (model_test)
            interpol_config += ".false.\n"  # MARCS binary format (.true.) or MARCS ASCII web format (.false.)?
            interpol_config += "'{}'\n".format(model_test)

            # Now we run the FORTRAN model interpolator
            #print(self.free_abundances["Ba"])
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
                #print("spud")
                return {
                    "interpol_config": interpol_config,
                    "errors": "MARCS model atmosphere interpolation failed."
                }
            #print("spud")
            if spherical:
                self.turbulent_velocity = microturbulence
        # TODO: equality with logg 4.0 here again?
        if self.t_eff >= 6500 and self.log_g == 4.0 and self.atmosphere_dimension == "3D": #reset temp to what it was before
            self.t_eff = temp_teff

        return {
            "interpol_config": interpol_config,
            "spherical": spherical,
            "errors": None
        }

    def make_atmosphere_properties(self, spherical, element):
        if self.nlte_flag == True:
            # Write configuration input for interpolator
            output = os_path.join(self.tmp_dir, self.marcs_model_name)
            model_test = "{}.test".format(output)
            interpol_config = ""
            for line in marcs_model_list_global:
                interpol_config += "'{}{}'\n".format(self.marcs_grid_path,line)
            interpol_config += "'{}.interpol'\n".format(output)
            interpol_config += "'{}.alt'\n".format(output)
            interpol_config += "'{}_{}_coef.dat'\n".format(output, element) #needed for nlte interpolator
            interpol_config += "'{}'\n".format(os_path.join(self.departure_file_path,self.depart_bin_file[element])) #needed for nlte interpolator
            interpol_config += "'{}'\n".format(os_path.join(self.departure_file_path,self.depart_aux_file[element])) #needed for nlte interpolator
            #interpol_config += "'/Users/gerber/gitprojects/TurboSpectrum2020/interpol_modeles_nlte/NLTEdata/1D_NLTE_grid_Fe_mean3D.bin'\n" #needed for nlte interpolator
            #interpol_config += "'/Users/gerber/gitprojects/TurboSpectrum2020/interpol_modeles_nlte/NLTEdata/auxData_Fe_mean3D_marcs_names.txt'\n" #needed for nlte interpolator
            interpol_config += "{}\n".format(len(np.loadtxt(os_path.join(self.departure_file_path,self.depart_aux_file[element]), dtype='str')))
            interpol_config += "{}\n".format(self.t_eff)
            interpol_config += "{}\n".format(self.log_g)
            interpol_config += "{:.2f}\n".format(round(float(self.metallicity),2))
            interpol_config += "{:.2f}\n".format(round(float(self.free_abundances[element]),2)+float(solar_abundances[element]))
            interpol_config += ".false.\n"  # test option - set to .true. if you want to plot comparison model (model_test)
            interpol_config += ".false.\n"  # MARCS binary format (.true.) or MARCS ASCII web format (.false.)?
            interpol_config += "'{}'\n".format(model_test)
        elif self.nlte_flag == False:
            output = os_path.join(self.tmp_dir, self.marcs_model_name)
            model_test = "{}.test".format(output)
            interpol_config = ""
            for line in marcs_model_list_global:
                interpol_config += "'{}{}'\n".format(self.marcs_grid_path,line)
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
        flag_dont_interp_microturb = 0
        for i in range(len(possible_turbulence)):
            if self.turbulent_velocity == possible_turbulence[i]:
                flag_dont_interp_microturb = 1

        if self.log_g < 3:
            flag_dont_interp_microturb = 1

        if flag_dont_interp_microturb == 0 and self.turbulent_velocity < 2.0 and (self.turbulent_velocity > 1.0 or (self.turbulent_velocity < 1.0 and self.t_eff < 3900.)):
            # Bracket the microturbulence to figure out what two values to generate the models to interpolate between using Andy's code
            turbulence_low = 0.0
            turbulence_high = 5.0
            microturbulence = self.turbulent_velocity
            for i in range(len(possible_turbulence)):
                if self.turbulent_velocity > possible_turbulence[i]:
                    turbulence_low = possible_turbulence[i]
                    place = i
            turbulence_high = possible_turbulence[place+1]
            #print(turbulence_low,turbulence_high)

            #generate models for low and high parts
            #temp_dir = self.tmp_dir
            #self.tmp_dir = os_path.join("/Users/gerber/iwg7_pipeline/4most-4gp-scripts/files_from_synthesis/current_run", "files_for_micro_interp")
            if self.nlte_flag == True:
                #for element, abundance in self.free_abundances.items():
                self.turbulent_velocity = turbulence_low
                atmosphere_properties_low = self._generate_model_atmosphere()
                #print(marcs_model_list_global)
                low_model_name = os_path.join(self.tmp_dir, self.marcs_model_name)
                low_model_name += '.interpol'
                #low_coef_dat_name = os_path.join(self.tmp_dir, self.marcs_model_name)
                #low_coef_dat_name += '_{}_coef.dat'.format(element)
                if atmosphere_properties_low['errors']:
                    return atmosphere_properties_low
                self.turbulent_velocity = turbulence_high
                atmosphere_properties_high = self._generate_model_atmosphere()
                high_model_name = os_path.join(self.tmp_dir, self.marcs_model_name)
                high_model_name += '.interpol'
                #high_coef_dat_name = os_path.join(self.tmp_dir, self.marcs_model_name)
                #high_coef_dat_name += '_{}_coef.dat'.format(element)
                if atmosphere_properties_high['errors']:
                    return atmosphere_properties_high
    
                self.turbulent_velocity = microturbulence
                #self.tmp_dir = temp_dir
    
                #interpolate and find a model atmosphere for the microturbulence
                self.marcs_model_name = "marcs_tef{:.1f}_g{:.2f}_z{:.2f}_tur{:.2f}".format(self.t_eff, self.log_g, self.metallicity, self.turbulent_velocity)
                f_low = open(low_model_name, 'r')
                lines_low = f_low.read().splitlines()
                t_low, temp_low, pe_low, pt_low, micro_low, lum_low, spud_low = np.loadtxt(open(low_model_name, 'rt').readlines()[:-8], skiprows=1, unpack=True)

                f_high = open(high_model_name, 'r')
                lines_high = f_high.read().splitlines()
                t_high, temp_high, pe_high, pt_high, micro_high, lum_high, spud_high = np.loadtxt(open(high_model_name, 'rt').readlines()[:-8], skiprows=1, unpack=True)

                fxhigh = microturbulence - turbulence_low
                fxlow = 1.0 - fxhigh

                t_interp = t_low*fxlow + t_high*fxhigh
                temp_interp = temp_low*fxlow + temp_high*fxhigh
                pe_interp = pe_low*fxlow + pe_high*fxhigh
                pt_interp = pt_low*fxlow + pt_high*fxhigh
                lum_interp = lum_low*fxlow + lum_high*fxhigh
                spud_interp = spud_low*fxlow + spud_high*fxhigh

                interp_model_name = os_path.join(self.tmp_dir, self.marcs_model_name)
                interp_model_name += '.interpol'
                #print(interp_model_name)
                g = open(interp_model_name, 'w')
                print(lines_low[0], file=g)
                for i in range(len(t_interp)):
                    print(" {:.4f}  {:.2f}  {:.4f}   {:.4f}   {:.4f}    {:.6e}  {:.4f}".format(t_interp[i], temp_interp[i], pe_interp[i], pt_interp[i], microturbulence, lum_interp[i], spud_interp[i]), file=g)
                print(lines_low[-8], file=g)
                print(lines_low[-7], file=g)
                print(lines_low[-6], file=g)
                print(lines_low[-5], file=g)
                print(lines_low[-4], file=g)
                print(lines_low[-3], file=g)
                print(lines_low[-2], file=g)
                print(lines_low[-1], file=g)
                g.close()

                #atmosphere_properties = atmosphere_properties_low
                #atmosphere_properties = self.make_atmosphere_properties(atmosphere_properties_low['spherical'], element)

                    #print(atmosphere_properties)

                    #os.system("mv /Users/gerber/iwg7_pipeline/4most-4gp-scripts/files_from_synthesis/current_run/files_for_micro_interp/* ../")


                for element, abundance in self.free_abundances.items():
                    if self.model_atom_file[element] != "":
                        atmosphere_properties = self.make_atmosphere_properties(atmosphere_properties_low['spherical'], element)
                        #low_coef_dat_name = os_path.join(self.tmp_dir, self.marcs_model_name)
                        #low_coef_dat_name += '_{}_coef.dat'.format(element)
                        low_coef_dat_name = low_model_name.replace('.interpol','_{}_coef.dat'.format(element))
                        f_coef_low = open(low_coef_dat_name, 'r')
                        lines_coef_low = f_coef_low.read().splitlines()
                        f_coef_low.close()

                        high_coef_dat_name = os_path.join(self.tmp_dir, self.marcs_model_name)
                        high_coef_dat_name += '_{}_coef.dat'.format(element)
                        high_coef_dat_name = high_model_name.replace('.interpol','_{}_coef.dat'.format(element))
                        f_coef_high = open(high_coef_dat_name, 'r')
                        lines_coef_high = f_coef_high.read().splitlines()
                        f_coef_high.close()

                        interp_coef_dat_name = os_path.join(self.tmp_dir, self.marcs_model_name)
                        interp_coef_dat_name += '_{}_coef.dat'.format(element)

                        num_lines = np.loadtxt(low_coef_dat_name, unpack = True, skiprows=9, max_rows = 1)

                        g = open(interp_coef_dat_name, 'w')
                        for i in range(11):
                            print(lines_coef_low[i], file=g)
                        for i in range(len(t_interp)):
                            print(" {:7.4f}".format(t_interp[i]), file=g)
                        for i in range(10+len(t_interp)+1,10+2*len(t_interp)+1):
                            fields_low = lines_coef_low[i].strip().split()
                            fields_high = lines_coef_high[i].strip().split()
                            fields_interp=[]
                            for j in range(len(fields_low)):
                                fields_interp.append(float(fields_low[j])*fxlow + float(fields_high[j])*fxhigh)
                            fields_interp_print = ['   {:.5f} '.format(elem) for elem in fields_interp]
                            print(*fields_interp_print, file=g)
                        for i in range(10+2*len(t_interp)+1,len(lines_coef_low)):
                            print(lines_coef_low[i], file=g)
                        g.close()
            elif self.nlte_flag == False:
                self.turbulent_velocity = turbulence_low
                atmosphere_properties_low = self._generate_model_atmosphere()
                #print(marcs_model_list_global)
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
                #self.tmp_dir = temp_dir

                #interpolate and find a model atmosphere for the microturbulence
                self.marcs_model_name = "marcs_tef{:.1f}_g{:.2f}_z{:.2f}_tur{:.2f}".format(self.t_eff, self.log_g, self.metallicity, self.turbulent_velocity)
                f_low = open(low_model_name, 'r')
                lines_low = f_low.read().splitlines()
                t_low, temp_low, pe_low, pt_low, micro_low, lum_low, spud_low = np.loadtxt(open(low_model_name, 'rt').readlines()[:-8], skiprows=1, unpack=True)

                f_high = open(high_model_name, 'r')
                lines_high = f_high.read().splitlines()
                t_high, temp_high, pe_high, pt_high, micro_high, lum_high, spud_high = np.loadtxt(open(high_model_name, 'rt').readlines()[:-8], skiprows=1, unpack=True)

                fxhigh = microturbulence - turbulence_low
                fxlow = 1.0 - fxhigh

                t_interp = t_low*fxlow + t_high*fxhigh
                temp_interp = temp_low*fxlow + temp_high*fxhigh
                pe_interp = pe_low*fxlow + pe_high*fxhigh
                pt_interp = pt_low*fxlow + pt_high*fxhigh
                lum_interp = lum_low*fxlow + lum_high*fxhigh
                spud_interp = spud_low*fxlow + spud_high*fxhigh

                interp_model_name = os_path.join(self.tmp_dir, self.marcs_model_name)
                interp_model_name += '.interpol'
                #print(interp_model_name)
                g = open(interp_model_name, 'w')
                print(lines_low[0], file=g)
                for i in range(len(t_interp)):
                    print(" {:.4f}  {:.2f}  {:.4f}   {:.4f}   {:.4f}    {:.6e}  {:.4f}".format(t_interp[i], temp_interp[i], pe_interp[i], pt_interp[i], microturbulence, lum_interp[i], spud_interp[i]), file=g)
                print(lines_low[-8], file=g)
                print(lines_low[-7], file=g)
                print(lines_low[-6], file=g)
                print(lines_low[-5], file=g)
                print(lines_low[-4], file=g)
                print(lines_low[-3], file=g)
                print(lines_low[-2], file=g)
                print(lines_low[-1], file=g)
                g.close()

                #atmosphere_properties = atmosphere_properties_low
                atmosphere_properties = self.make_atmosphere_properties(atmosphere_properties_low['spherical'], 'Fe')


        elif flag_dont_interp_microturb == 0 and self.turbulent_velocity > 2.0: #not enough models to interp if higher than 2
            microturbulence = self.turbulent_velocity                           #just use 2.0 for the model if between 2 and 3
            self.turbulent_velocity = 2.0
            atmosphere_properties = self._generate_model_atmosphere()
            if atmosphere_properties['errors']:
                return atmosphere_properties
            self.turbulent_velocity = microturbulence

        elif flag_dont_interp_microturb == 0 and self.turbulent_velocity < 1.0 and self.t_eff >= 3900.: #not enough models to interp if lower than 1 and t_eff > 3900
            microturbulence = self.turbulent_velocity
            self.turbulent_velocity = 1.0
            atmosphere_properties = self._generate_model_atmosphere()
            if atmosphere_properties['errors']:
                return atmosphere_properties
            self.turbulent_velocity = microturbulence


        elif flag_dont_interp_microturb == 1:
            if self.log_g < 3:
                microturbulence = self.turbulent_velocity
                self.turbulent_velocity = 2.0
            atmosphere_properties = self._generate_model_atmosphere()
            if self.log_g < 3:
                self.turbulent_velocity = microturbulence
            if atmosphere_properties['errors']:
                #print('spud')
                print(atmosphere_properties['errors'])
                return atmosphere_properties

        self.atmosphere_properties = atmosphere_properties
        #print(self.atmosphere_properties)

    def make_species_lte_nlte_file(self):
        """
        Generate the SPECIES_LTE_NLTE.dat file for TS to determine what elements are NLTE
        """
        #data_path = self.turbospec_path.replace("exec/","DATA/")
        data_path = self.tmp_dir

        nlte = "nlte" if self.nlte_flag == True else "lte"

        #if len(self.free_abundances.items()) == 1:
        #    nlte_fe = nlte
        #else:
        #    nlte_fe =

        file = open("{}/SPECIES_LTE_NLTE_{:08d}.dat".format(data_path,self.counter_spectra), 'w')
        #print("# This file controls which species are treated in LTE/NLTE", file=file)
        #print("# It also gives the path to the model atom and the departure files", file=file)
        file.write("# This file controls which species are treated in LTE/NLTE\n")
        file.write("# It also gives the path to the model atom and the departure files\n")
        file.write("# First created 2021-02-22\n")
        file.write("# if a species is absent it is assumed to be LTE\n")
        file.write("#\n")
        file.write("# each line contains :\n")
        file.write("# atomic number / name / (n)lte / model atom / departure file / binary or ascii departure file\n")
        file.write("#\n")
        file.write("# path for model atom files     ! don't change this line !\n")
        file.write("{}\n".format(self.model_atom_path))
        file.write("#\n")
        file.write("# path for departure files      ! don't change this line !\n")
        file.write("{}\n".format(self.tmp_dir))
        file.write("#\n")
        file.write("# atomic (N)LTE setup\n")
        #file.write("1    'H' 'lte'   'atom.h20'  ' ' 'binary'\n")
        if self.nlte_flag == True:
            for element, abundance in self.free_abundances.items():
                atomic_number = periodic_table.index(element)
                if self.model_atom_file[element] == "":
                    file.write("{}  '{}'  'lte'  ''   '' 'ascii'\n".format(atomic_number,element,nlte,self.model_atom_file[element],self.marcs_model_name, element))
                else:
                    file.write("{}  '{}'  '{}'  '{}'   '{}_{}_coef.dat' 'ascii'\n".format(atomic_number,element,nlte,self.model_atom_file[element],self.marcs_model_name, element))
        elif self.nlte_flag == False:
            for element, abundance in self.free_abundances.items():
                atomic_number = periodic_table.index(element)
                file.write("{}  '{}'  '{}'  ''   '' 'ascii'\n".format(atomic_number,element,nlte))
        file.close()

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

        # Allow for user input abundances as a dictionary of the form {element: abundance}
        #deprecated 8-2022
        """
        if self.free_abundances is None:
            individual_abundances = "'INDIVIDUAL ABUNDANCES:'   '0'\n"
        else:
            individual_abundances = "'INDIVIDUAL ABUNDANCES:'   '{:d}'\n".format(len(self.free_abundances))

            for element, abundance in self.free_abundances.items():
                assert element in solar_abundances, "Cannot proceed as solar abundance for element <{}> is unknown". \
                    format(element)

                atomic_number = periodic_table.index(element)
                individual_abundances += "{:d}  {:.2f}\n".format(int(atomic_number),
                                                                 float(solar_abundances[element]) + round(float(abundance),2))
        #print(individual_abundances.strip())
        #print(individual_abundances)
        """

        # Updated abundances to below to allow user to set solar abundances through solar_abundances.py and not have to adjust make_abund.f

        individual_abundances = "'INDIVIDUAL ABUNDANCES:'   '{:d}'\n".format(len(periodic_table)-1)
        if self.free_abundances is None:
            for i in range(1,len(periodic_table)):
                individual_abundances += "{:d}  {:.2f}\n".format(i, float(solar_abundances[periodic_table[i]])+self.metallicity)
        else:
            item_abund = {}
            item_abund['H'] = 12.00
            for i in range(2,len(periodic_table)):
                item_abund[periodic_table[i]] = float(solar_abundances[periodic_table[i]]) + round(float(self.metallicity),2)
            for element, abundance in self.free_abundances.items():
                item_abund[element] = float(solar_abundances[element]) + round(float(abundance),2)
            for i in range(1,len(periodic_table)):
                individual_abundances += "{:d}  {:.2f}\n".format(i, item_abund[periodic_table[i]])
        #print(individual_abundances)

        # Allow for user input isotopes as a dictionary (similar to abundances)

        individual_isotopes = "'ISOTOPES : ' '149'\n"
        if self.free_isotopes is None:
            for isotope, ratio in solar_isotopes.items():
                individual_isotopes += "{}  {:6f}\n".format(isotope, ratio)
        else:
            for isotope, ratio in self.free_isotopes.items():
                solar_isotopes[isotope] = ratio
            for isotope, ratio in solar_isotopes.items():
                individual_isotopes += "{}  {:6f}\n".format(isotope, ratio)


        #if self.free_isotopes is None:
        #    free_isotopes = "'ISOTOPES : ' '{:d}'\n".format(len(self.))
        #else:
        #    individual_abundances = "'INDIVIDUAL ABUNDANCES:'   '{:d}'\n".format(len(self.free_abundances))

        #    for element, abundance in self.free_abundances.items():
        #        assert element in solar_abundances, "Cannot proceed as solar abundance for element <{}> is unknown". \
        #            format(element)

        #        atomic_number = periodic_table.index(element)
        #        individual_abundances += "{:d}  {:.2f}\n".format(int(atomic_number),
         #                                                        float(solar_abundances[element]) + float(abundance))
        #print(individual_abundances.strip())
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

        #print(self.line_list_paths)
        #print(line_list_files)

        # Build bsyn configuration file
        spherical_boolean_code = "T" if spherical else "F"
        if self.atmosphere_dimension == "3D":
            spherical_boolean_code = "F"
        xifix_boolean_code = "T" if self.atmosphere_dimension == "1D" else "F"
        nlte_boolean_code = ".true." if self.nlte_flag == True else ".false."
        pure_lte_boolean_code = ".false." if self.nlte_flag == True else ".true."

        if self.windows_flag == True:
            bsyn_config = """\
'PURE-LTE  :'  '.false.'
'NLTE :'          '{nlte}'
'NLTEINFOFILE:'  '{this[tmp_dir]}SPECIES_LTE_NLTE_{this[counter_spectra]:08d}.dat'
#'MODELATOMFILE:'  '{this[model_atom_path]}{this[model_atom_file]}'
#'DEPARTUREFILE:'  '{this[tmp_dir]}{this[marcs_model_name]}_coef.dat'
#'DEPARTBINARY:'   '.false.'
#'CONTMASKFILE:'     '{this[cont_mask_file]}'
#'LINEMASKFILE:'     '{this[line_mask_file]}'
'SEGMENTSFILE:'     '{this[segment_file]}'
'LAMBDA_MIN:'    '{this[lambda_min]:.3f}'
'LAMBDA_MAX:'    '{this[lambda_max]:.3f}'
'LAMBDA_STEP:'   '{this[lambda_delta]:.3f}'
'INTENSITY/FLUX:' 'Flux'
'COS(THETA)    :' '1.00'
'ABFIND        :' '.false.'
'MODELOPAC:' '{this[tmp_dir]}model_opacity_{this[counter_spectra]:08d}.opac'
'RESULTFILE :' '{this[tmp_dir]}/spectrum_{this[counter_spectra]:08d}.spec'
'METALLICITY:'    '{this[metallicity]:.2f}'
'ALPHA/Fe   :'    '{alpha:.2f}'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '{this[r_process]:.2f}'
'S-PROCESS  :'    '{this[s_process]:.2f}'
{individual_abundances}
{individual_isotopes}
{line_lists}
'SPHERICAL:'  '{spherical}'
  30
  300.00
  15
  1.30
""".format(this=self.__dict__,
           alpha=alpha,
           spherical=spherical_boolean_code,
           individual_abundances=individual_abundances.strip(),
           individual_isotopes=individual_isotopes.strip(),
           line_lists=line_lists.strip(),
           pure_lte=pure_lte_boolean_code,
           nlte=nlte_boolean_code
           )

            # Build babsma configuration file
            babsma_config = """\
'PURE-LTE  :'  '.false.'
'LAMBDA_MIN:'    '{this[lambda_min]:.3f}'
'LAMBDA_MAX:'    '{this[lambda_max]:.3f}'
'LAMBDA_STEP:'    '{this[lambda_delta]:.3f}'
'MODELINPUT:' '{this[tmp_dir]}{this[marcs_model_name]}.interpol'
'MARCS-FILE:' '.false.'
'MODELOPAC:' '{this[tmp_dir]}model_opacity_{this[counter_spectra]:08d}.opac'
'METALLICITY:'    '{this[metallicity]:.2f}'
'ALPHA/Fe   :'    '{alpha:.2f}'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '{this[r_process]:.2f}'
'S-PROCESS  :'    '{this[s_process]:.2f}'
{individual_abundances}
'XIFIX:' '{xifix}'
{this[turbulent_velocity]:.2f}
""".format(this=self.__dict__,
           alpha=alpha,
           individual_abundances=individual_abundances.strip(),
           pure_lte=pure_lte_boolean_code,
           xifix=xifix_boolean_code
           )
        elif self.windows_flag == False:
            bsyn_config = """\
'PURE-LTE  :'  '.false.'
'NLTE :'          '{nlte}'
'NLTEINFOFILE:'  '{this[tmp_dir]}SPECIES_LTE_NLTE_{this[counter_spectra]:08d}.dat'
#'MODELATOMFILE:'  '{this[model_atom_path]}{this[model_atom_file]}'
#'DEPARTUREFILE:'  '{this[tmp_dir]}{this[marcs_model_name]}_coef.dat'
#'DEPARTBINARY:'   '.false.'
#'CONTMASKFILE:'     '/Users/gerber/gitprojects/SAPP/linemasks/ca-cmask.txt'
#'LINEMASKFILE:'     '/Users/gerber/gitprojects/SAPP/linemasks/ca-lmask.txt'
#'SEGMENTSFILE:'     '/Users/gerber/gitprojects/SAPP/linemasks/ca-seg.txt'
'LAMBDA_MIN:'    '{this[lambda_min]:.3f}'
'LAMBDA_MAX:'    '{this[lambda_max]:.3f}'
'LAMBDA_STEP:'   '{this[lambda_delta]:.3f}'
'INTENSITY/FLUX:' 'Flux'
'COS(THETA)    :' '1.00'
'ABFIND        :' '.false.'
'MODELOPAC:' '{this[tmp_dir]}model_opacity_{this[counter_spectra]:08d}.opac'
'RESULTFILE :' '{this[tmp_dir]}/spectrum_{this[counter_spectra]:08d}.spec'
'METALLICITY:'    '{this[metallicity]:.2f}'
'ALPHA/Fe   :'    '{alpha:.2f}'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '{this[r_process]:.2f}'
'S-PROCESS  :'    '{this[s_process]:.2f}'
{individual_abundances}
{individual_isotopes}
{line_lists}
'SPHERICAL:'  '{spherical}'
  30
  300.00
  15
  1.30
""".format(this=self.__dict__,
           alpha=alpha,
           spherical=spherical_boolean_code,
           individual_abundances=individual_abundances.strip(),
           individual_isotopes=individual_isotopes.strip(),
           line_lists=line_lists.strip(),
           pure_lte=pure_lte_boolean_code,
           nlte=nlte_boolean_code
           )

            # Build babsma configuration file
            babsma_config = """\
'PURE-LTE  :'  '.false.'
'LAMBDA_MIN:'    '{this[lambda_min]:.3f}'
'LAMBDA_MAX:'    '{this[lambda_max]:.3f}'
'LAMBDA_STEP:'    '{this[lambda_delta]:.3f}'
'MODELINPUT:' '{this[tmp_dir]}{this[marcs_model_name]}.interpol'
'MARCS-FILE:' '.false.'
'MODELOPAC:' '{this[tmp_dir]}model_opacity_{this[counter_spectra]:08d}.opac'
'METALLICITY:'    '{this[metallicity]:.2f}'
'ALPHA/Fe   :'    '{alpha:.2f}'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '{this[r_process]:.2f}'
'S-PROCESS  :'    '{this[s_process]:.2f}'
{individual_abundances}
'XIFIX:' '{xifix}'
{this[turbulent_velocity]:.2f}
""".format(this=self.__dict__,
           alpha=alpha,
           individual_abundances=individual_abundances.strip(),
           pure_lte=pure_lte_boolean_code,
           xifix=xifix_boolean_code
           )

        #print(babsma_config)
        #print(bsyn_config)
        return babsma_config, bsyn_config

    def stitch(self, specname1, specname2, lmin, lmax, new_range, count): #toss a coin to your stitcher
        wave1, flux_norm1, flux1 = np.loadtxt(specname1, unpack=True)
        wave2, flux_norm2, flux2 = np.loadtxt(specname2, unpack=True)

        #print(lmin, lmin+(count*new_range))

        wave1_clipped = wave1[np.where((wave1 < lmin+(count*new_range))& (wave1 >= lmin))]
        flux_norm1_clipped = flux_norm1[np.where((wave1 < lmin+(count*new_range)) & (wave1 >= lmin))]
        flux1_clipped = flux1[np.where((wave1 < lmin+(count*new_range)) & (wave1 >= lmin))]
        wave2_clipped = wave2[np.where((wave2 >= lmin+(count*new_range)) & (wave2 <= lmax))]
        flux_norm2_clipped = flux_norm2[np.where((wave2 >= lmin+(count*new_range)) & (wave2 <= lmax))]
        flux2_clipped = flux2[np.where((wave2 >= lmin+(count*new_range)) & (wave2 <= lmax))]

        wave = np.concatenate((wave1_clipped,wave2_clipped))
        flux_norm = np.concatenate((flux_norm1_clipped,flux_norm2_clipped))
        flux = np.concatenate((flux1_clipped,flux2_clipped))

        return wave, flux_norm, flux

    def synthesize(self):
            # Generate configuation files to pass to babsma and bsyn
        self.make_species_lte_nlte_file()  # TODO: not create this file every time (same one for each run anyway)
        babsma_in, bsyn_in = self.make_babsma_bsyn_file(spherical=self.atmosphere_properties['spherical'])

        #print(babsma_in)
        #print(bsyn_in)

        # Start making dictionary of output data
        output = self.atmosphere_properties
        output["errors"] = None
        output["babsma_config"] = babsma_in
        output["bsyn_config"] = bsyn_in

        # Select whether we want to see all the output that babsma and bsyn send to the terminal
        if self.verbose:
            stdout = None
            stderr = subprocess.STDOUT
        else:
            stdout = open('/dev/null', 'w')
            stderr = subprocess.STDOUT

        # We need to run babsma and bsyn with working directory set to root of Turbospectrum install. Otherwise
        # it cannot find its data files.
        cwd = os.getcwd()
        turbospec_root = os_path.join(self.turbospec_path, "..")

        # Run babsma. This creates an opacity file .opac from the MARCS atmospheric model
        try:    # chdir is NECESSARY, turbospectrum cannot run from other directories sadly
            os.chdir(turbospec_root)    # Time wasted trying to make asyncio work here: 6 hours. Halts program halfway
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
            #logging.info("Babsma failed. Return code {}. Error text <{}>".
            #             format(pr1.returncode, stderr_bytes.decode('utf-8')))
            return output

        # Run bsyn. This synthesizes the spectrum
        try:
            pr, stderr_bytes = self.run_bsyn(bsyn_in, stderr, stderr_bytes, stdout, turbospec_root)
        except subprocess.CalledProcessError:
            output["errors"] = "bsyn failed with CalledProcessError"
            return output
        finally:
            os.chdir(cwd)
        if stderr_bytes is None:
            stderr_bytes = b''
        if pr.returncode != 0:
            output["errors"] = "bsyn failed"
            #logging.info("Bsyn failed. Return code {}. Error text <{}>".
            #             format(pr.returncode, stderr_bytes.decode('utf-8')))
            return output

        # Return output
        output["return_code"] = pr.returncode
        output["output_file"] = os_path.join(self.tmp_dir, "spectrum_{:08d}.spec".format(self.counter_spectra))
        return output

    def run_bsyn(self, bsyn_in, stderr, stderr_bytes, stdout, turbospec_root):
        os.chdir(turbospec_root)
        pr = subprocess.Popen([os_path.join(self.turbospec_path, 'bsyn_lu')],
                              stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
        pr.stdin.write(bytes(bsyn_in, 'utf-8'))
        stdout_bytes, stderr_bytes = pr.communicate()
        return pr, stderr_bytes

    def run_babsma(self, babsma_in, stderr, stdout):
        pr1 = subprocess.Popen([os_path.join(self.turbospec_path, 'babsma_lu')],
                               stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
        pr1.stdin.write(bytes(babsma_in, 'utf-8'))
        stdout_bytes, stderr_bytes = pr1.communicate()
        return pr1, stderr_bytes

    def run_turbospectrum(self):
        lmin_orig = self.lambda_min
        lmax_orig = self.lambda_max
        lmin = self.lambda_min
        lmax = self.lambda_max

        lpoint_max = 1000000 * 0.99  # first number comes from turbospectrum spectrum.inc : lpoint. 0.99 is to give some extra room so that bsyn does not fail for sure
        points_in_new_spectra_to_generate = int((lmax-lmin) / self.lambda_delta)

        if points_in_new_spectra_to_generate > lpoint_max:
            print("WARNING. The range or delta wavelength asked is too big. Trying to break down into smaller "
                  "segments and stitch them together at the end.")
            number = math.ceil(points_in_new_spectra_to_generate / lpoint_max)
            new_range = round((lmax-lmin)/number)
            extra_wavelength_for_stitch = 30    # generats with extra wavlength so that stitch can be nice i guess (i did not write this originally)
            for i in range(number):
                self.configure(lambda_min=lmin-extra_wavelength_for_stitch, lambda_max=lmin+new_range+extra_wavelength_for_stitch, counter_spectra=i)
                self.synthesize()
                lmin = lmin + new_range
            for i in range(number-1):
                spectrum1 = os_path.join(self.tmp_dir, "spectrum_{:08d}.spec".format(0))
                spectrum2 = os_path.join(self.tmp_dir, "spectrum_{:08d}.spec".format(i+1))
                wave, flux_norm, flux = self.stitch(spectrum1, spectrum2, lmin_orig, lmax_orig, new_range, i+1)
                f = open(spectrum1, 'w')
                for j in range(len(wave)):
                    print("{}  {}  {}".format(wave[j], flux_norm[j], flux[j]), file=f)
                f.close()
        else:
            self.synthesize()

    def run_turbospectrum_and_atmosphere(self):
        try:
            self.calculate_atmosphere()
            try:
                self.run_turbospectrum()
            except AttributeError:
                print("No attribute, fail of generation?")
        except FileNotFoundError as error:
            print(f"Interpolation failed? {error}")


def fetch_marcs_grid(marcs_grid_path, marcs_grid_list):
    """
    Get a list of all of the MARCS models we have.

    :return:
        None
    """
    counter_marcs = 0
    marcs_model_name = "default"
    counter_spectra = 0
    marcs_values = None
    marcs_value_keys = []
    marcs_models = {}

    pattern = r"([sp])(\d\d\d\d)_g(....)_m(...)_t(..)_(..)_z(.....)_" \
              r"a(.....)_c(.....)_n(.....)_o(.....)_r(.....)_s(.....).mod"

    marcs_values = {
        "spherical": [], "temperature": [], "log_g": [], "mass": [], "turbulence": [], "model_type": [],
        "metallicity": [], "a": [], "c": [], "n": [], "o": [], "r": [], "s": []
    }

    marcs_value_keys = [i for i in list(marcs_values.keys()) if i not in TurboSpectrum.marcs_parameters_to_ignore]
    marcs_value_keys.sort()
    marcs_models = {}

    marcs_models = glob.glob(os_path.join(marcs_grid_path, "*"))
    marcs_nlte_models = np.loadtxt(marcs_grid_list, dtype='str', usecols=(0,), unpack=True)
    # marcs_nlte_models = np.loadtxt("/Users/gerber/gitprojects/TurboSpectrum2020/interpol_modeles_nlte/NLTEdata/auxData_Fe_mean3D_marcs_names.txt", dtype='str', usecols=(0,), unpack=True)
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

    # Sort model parameter values into order
    for parameter in marcs_value_keys:
        marcs_values[parameter].sort()

    return marcs_values, marcs_value_keys, marcs_models