from __future__ import annotations

import pickle
from configparser import ConfigParser
from typing import Tuple, Callable
from warnings import warn
import numpy as np
from distributed import get_worker
from scipy.optimize import minimize, root_scalar
from scripts.auxiliary_functions import create_dir, calculate_vturb, calculate_equivalent_width, \
    apply_doppler_correction, create_segment_file, import_module_from_path
from scripts.model_atom_class import ModelAtom
from scripts.solar_abundances import periodic_table
from scripts.turbospectrum_class_nlte import TurboSpectrum
from scripts.m3dis_class import M3disCall
from scripts.synthetic_code_class import SyntheticSpectrumGenerator
from scripts.synthetic_code_class import fetch_marcs_grid
import time
import os
try:
    from dask.distributed import Client, get_client, secede, rejoin
except (ModuleNotFoundError, ImportError):
    raise ModuleNotFoundError("Dask not installed. It is required for multiprocessing. Install using pip install dask[complete]")
import shutil
import collections
from scripts.convolve import conv_rotation, conv_macroturbulence, conv_res
from scripts.create_window_linelist_function import create_window_linelist
from scripts.loading_configs import SpectraParameters, TSFitPyConfig
import logging
from scripts.dask_client import get_dask_client
import pandas as pd

output_default_configuration_name: str = "configuration.cfg"
output_default_fitlist_name: str = "fitlist.txt"
output_default_linemask_name: str = "linemask.txt"


def get_convolved_spectra(wave: np.ndarray, flux: np.ndarray, resolution: float, macro: float, rot: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Convolves spectra with resolution, macroturbulence or rotation if values are non-zero
    :param wave: wavelength array, in ascending order
    :param flux: flux array normalised
    :param resolution: resolution, zero if not required
    :param macro: Macroturbulence in km/s, zero if not required
    :param rot: Rotation in km/s, 0 if not required
    :return: 2 arrays, first is convolved wavelength, second is convolved flux
    """
    if resolution != 0.0:
        wave_mod_conv, flux_mod_conv = conv_res(wave, flux, resolution)
    else:
        wave_mod_conv = wave
        flux_mod_conv = flux
    if macro != 0.0:
        wave_mod_macro, flux_mod_macro = conv_macroturbulence(wave_mod_conv, flux_mod_conv, macro)
    else:
        wave_mod_macro = wave_mod_conv
        flux_mod_macro = flux_mod_conv
    if rot != 0.0:
        wave_mod, flux_mod = conv_rotation(wave_mod_macro, flux_mod_macro, rot)
    else:
        wave_mod = wave_mod_macro
        flux_mod = flux_mod_macro
    return wave_mod, flux_mod


def calculate_all_lines_chi_squared(wave_obs: np.ndarray, flux_obs: np.ndarray, wave_mod: np.ndarray,
                                    flux_mod: np.ndarray, line_begins_sorted: np.ndarray, line_ends_sorted: np.ndarray,
                                    seg_begins: np.ndarray, seg_ends: np.ndarray) -> float:
    """
    Calculates chi squared for all lines fitting by comparing two spectra and calculating the chi_squared based on
    interpolation between the wavelength points
    :param wave_obs: Observed wavelength
    :param flux_obs: Observed normalised flux
    :param wave_mod: Synthetic wavelength
    :param flux_mod: Synthetic normalised flux
    :param line_begins_sorted: Sorted line list, wavelength of a line start
    :param line_ends_sorted: Sorted line list, wavelength of a line end
    :param seg_begins: Segment list where it starts, array
    :param seg_ends: Segment list where it ends, array
    :return: Calculated chi squared at lines
    """
    if wave_mod[1] - wave_mod[0] <= wave_obs[1] - wave_obs[0]:
        flux_mod_interp = np.interp(wave_obs, wave_mod, flux_mod)
        chi_square = 0
        for l in range(len(line_begins_sorted[np.where(
                (line_begins_sorted > np.min(seg_begins)) & (line_begins_sorted < np.max(seg_ends)))])):
            flux_line_obs = flux_obs[
                np.where((wave_obs <= line_ends_sorted[l]) & (wave_obs >= line_begins_sorted[l]))]
            flux_line_mod = flux_mod_interp[
                np.where((wave_obs <= line_ends_sorted[l]) & (wave_obs >= line_begins_sorted[l]))]
            chi_square += np.sum(np.square((flux_line_obs - flux_line_mod)) / flux_line_mod)
    else:
        flux_obs_interp = np.interp(wave_mod, wave_obs, flux_obs)
        chi_square = 0
        for l in range(len(line_begins_sorted[np.where(
                (line_begins_sorted > np.min(seg_begins)) & (line_begins_sorted < np.max(seg_ends)))])):
            flux_line_obs = flux_obs_interp[
                np.where((wave_mod <= line_ends_sorted[l]) & (wave_mod >= line_begins_sorted[l]))]
            flux_line_mod = flux_mod[
                np.where((wave_mod <= line_ends_sorted[l]) & (wave_mod >= line_begins_sorted[l]))]
            chi_square += np.sum(np.square(flux_line_obs - flux_line_mod) / flux_line_mod)
    return chi_square


def calc_ts_spectra_all_lines(obs_name: str, wave_mod_orig: np.ndarray, flux_mod_orig: np.ndarray, temp_directory: str, output_dir: str, wave_obs: np.ndarray,
                              flux_obs: np.ndarray, macro: float, resolution: float, rot: float,
                              line_begins_sorted: np.ndarray, line_ends_sorted: np.ndarray,
                              seg_begins: np.ndarray, seg_ends: np.ndarray) -> float:
    """
    Calculates chi squared by opening a created synthetic spectrum and comparing to the observed spectra. Then
    calculates chi squared. Used for all lines at once within line list
    :param obs_name: Name of the file where to save the new spectra
    :param temp_directory: Directory where TS calculated the spectra
    :param output_dir: Directory where to save the new spectra
    :param wave_obs: Observed wavelength
    :param flux_obs: Observed normalised flux
    :param macro: Macroturbulence in km/s, zero if not required
    :param resolution: resolution, zero if not required
    :param rot: Rotation in km/s, 0 if not required
    :param line_begins_sorted: Sorted line list, wavelength of a line start
    :param line_ends_sorted: Sorted line list, wavelength of a line end
    :param seg_begins: Segment list where it starts, array
    :param seg_ends: Segment list where it ends, array
    :return: chi squared at line (between line start and end). Also creates convolved spectra.
    """
    wave_mod_filled = np.copy(wave_mod_orig)
    flux_mod_filled = np.copy(flux_mod_orig)

    for l in range(len(seg_begins) - 1):
        flux_mod_filled[
            np.logical_and.reduce((wave_mod_orig > seg_ends[l], wave_mod_orig <= seg_begins[l + 1]))] = 1.0

    wave_mod_filled = np.array(wave_mod_filled)
    flux_mod_filled = np.array(flux_mod_filled)

    wave_mod, flux_mod = get_convolved_spectra(wave_mod_filled, flux_mod_filled, resolution, macro, rot)

    chi_square = calculate_all_lines_chi_squared(wave_obs, flux_obs, wave_mod, flux_mod, line_begins_sorted,
                                                 line_ends_sorted, seg_begins, seg_ends)

    os.system(f"mv {os.path.join(temp_directory, 'spectrum_00000000.spec')} {os.path.join(output_dir, obs_name)}")

    out = open(f"{os.path.join(output_dir, f'spectrum_fit_convolved_{obs_name}')}",'w')
    for l in range(len(wave_mod)):
        print(f"{wave_mod[l]}  {flux_mod[l]}", file=out)
    out.close()

    return chi_square


def calculate_lbl_chi_squared(wave_obs: np.ndarray, flux_obs: np.ndarray, error_obs_variance: np.ndarray,
                              wave_synt_orig: np.ndarray, flux_synt_orig: np.ndarray, resolution: float, lmin: float,
                              lmax: float, vmac: float, rotation: float) -> float:
    """
    Calculates chi squared by opening a created synthetic spectrum and comparing to the observed spectra. Then
    calculates chi squared. Used for line by line method, by only looking at a specific line.
    :param wave_obs: Observed wavelength
    :param flux_obs: Observed normalised flux
    :param error_obs_variance: Observed error variance
    :param wave_synt_orig: Synthetic wavelength
    :param flux_synt_orig: Synthetic normalised flux
    :param resolution: resolution, zero if not required
    :param lmin: Wavelength, start of line
    :param lmax: Wavelength, end of line
    :param vmac: Macroturbulence in km/s, zero if not required
    :param rotation: Rotation in km/s, 0 if not required
    :return: Calculated chi squared for a given line
    """
    indices_to_use_mod = np.where((wave_synt_orig <= lmax) & (wave_synt_orig >= lmin))
    indices_to_use_obs = np.where((wave_obs <= lmax) & (wave_obs >= lmin))

    wave_synt_orig, flux_synt_orig = wave_synt_orig[indices_to_use_mod], flux_synt_orig[indices_to_use_mod]
    wave_obs, flux_obs = wave_obs[indices_to_use_obs], flux_obs[indices_to_use_obs]
    error_obs_variance = error_obs_variance[indices_to_use_obs]

    if np.size(flux_obs) == 0:
        return 999999

    try:
        wave_synt, flux_synt = get_convolved_spectra(wave_synt_orig, flux_synt_orig, resolution, vmac, rotation)
    except ValueError:
        return 999999

    flux_synt_interp = np.interp(wave_obs, wave_synt, flux_synt)

    chi_square = np.sum(np.square(flux_obs - flux_synt_interp) / error_obs_variance) / calculate_dof(wave_obs)

    return chi_square


class Result:
    # because other fitting algorithms call result differently
    def __init__(self):
        # res.x: list = [param1 best guess, param2 best guess etc]
        # res.fun: float = function value (chi squared) after the fit
        self.fun: float = 0
        self.x: list = []

def minimize_function(function_to_minimize, input_param_guess: np.ndarray, function_arguments: tuple, bounds: list[tuple], method: str, options: dict):
    """
    Minimizes a function using specified method and options in the function
    :param function_to_minimize: Function to minimize
    :param input_param_guess: Initial guess for the parameters
    :param function_arguments: Arguments for the function
    :param bounds: Bounds for the parameters
    :param method: Method to use for minimization
    :param options: Options for the minimization
    :return: Result of the minimization
    """
    #res.x: list = [param1 best guess, param2 best guess etc]
    #res.fun: float = function value (chi squared) after the fit

    # using Scipy. Nelder-Mead or L-BFGS- algorithm
    res = minimize(function_to_minimize, input_param_guess, args=function_arguments, bounds=bounds, method=method, options=options)

    """
    cma: might work for high dimensions, doesn't work for 1D easily. so the implementation below doesn't work at all
    if input_param_guess.ndim > 1:
        parameter_guess = np.median(input_param_guess, axis=0)
        sigma = (np.max(input_param_guess, axis=0) - np.min(input_param_guess, axis=0)) / 3
    else:
        parameter_guess = input_param_guess
        sigma = (np.max(bounds, axis=0) - np.min(bounds, axis=0)) / 5
    result = cma.fmin(function_to_minimize, parameter_guess, sigma, args=function_arguments, options={'bounds': bounds})
    res.x = result.xbest
    res.fun = result.funbest"""

    """
    NS: Wasted 3 hours testing. Optuna works OK, but results vary up to 0.1 dex. Maybe more trials are needed. 
    OR just dont use it lol.
    Everything below works
    import logging

    # Set the logging level to ERROR to suppress INFO messages
    logging.getLogger("optuna").setLevel(logging.ERROR)

    if input_param_guess.ndim > 1:
        parameter_guess = np.median(input_param_guess, axis=0)
    else:
        parameter_guess = input_param_guess

    def suggest_float(trial, name, bounds, initial):
        if len(initial) == 1:
            lower, upper = bounds[0]
            return [trial.suggest_float(name + '_0', lower, upper)]
        else:
            return [trial.suggest_float(f"{name}_{i}", bounds[i][0], bounds[i][1]) for i in range(len(initial))]

    def objective(trial):
        x = suggest_float(trial, "x", bounds, parameter_guess)
        return function_to_minimize(x, *function_arguments)

    def silent_callback(study, trial):
        pass

    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=20, interval_steps=1)

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=50, callbacks=[silent_callback])

    res = Result()
    res.x = [study.best_params[key] for key in study.best_params.keys()]
    res.fun = study.best_value"""

    return res


class Spectra:
    def __init__(self, specname: str, teff: float, logg: float, rv: float, met: float, micro: float, macro: float,
                 rotation: float, abundances_dict: dict, resolution: float, snr: float, line_list_path_trimmed: str, index_temp_dir: float,
                 tsfitpy_config, n_workers=1, m3dis_python_module=None):
        # Default values
        self.m3dis_python_module = m3dis_python_module  # python module for reading m3dis output
        self.compiler: str = "None"  # intel, gfotran, m3dis
        self.spectral_code_path: str = "None"  # path to the /exec/ file
        self.interpol_path: str = "None"  # path to the model_interpolators folder with fortran code
        self.model_atmosphere_grid_path: str = "None"
        self.model_atmosphere_list: str = "None"
        self.model_atom_path: str = "None"
        self.departure_file_path: str = "None"
        self.linemask_file: str = "None"
        self.segment_file: str = "None"
        self.atmosphere_type: str = "None"  # "1D" or "3D", string
        self.include_molecules: bool = False  # "True" or "False", bool
        self.nlte_flag: bool = False
        self.fit_vmic: bool = False
        self.input_vmic: bool = False
        self.fit_vmac: bool = False
        self.input_vmac: bool = False
        self.fit_rotation: bool = False
        self.input_rotation: bool = False
        self.fit_teff: bool = False  # seems like not used anymore atm
        #self.fit_logg: str = None  # does not work atm
        self.nelement: int = 0  # how many elements to fit (1 to whatever)
        self.fit_feh: bool = False
        self.elem_to_fit: np.ndarray = np.array([])  # only 1 element at a time is support atm, a list otherwise
        self.lmin: float = 0
        self.lmax: float = 0
        self.ldelta: float = 0
        self.resolution: float = 0  # resolution coming from resolution, constant for all stars:  central lambda / FWHM
        self.rotation: float = 0  # rotation km/s, constant for all stars
        self.fitting_mode: str = "None"  # "lbl" = line by line or "all"
        self.output_folder: str = "None"

        self.dask_workers: int = 1  # workers, i.e. CPUs for multiprocessing

        self.global_temp_dir: str = "None"
        self.line_begins_sorted: np.ndarray = np.array([])
        self.line_ends_sorted: np.ndarray = np.array([])
        self.line_centers_sorted: np.ndarray = np.array([])

        self.seg_begins: np.ndarray = np.array([])
        self.seg_ends: np.ndarray = np.array([])

        self.depart_bin_file_dict: dict = {}
        self.depart_aux_file_dict: dict = {}
        self.model_atom_file_dict: dict = {}
        self.aux_file_length_dict: dict = {}  # loads the length of aux file to not do it everytime later
        self.ndimen: int = 0
        self.spec_input_path: str = "None"

        self.init_guess_dict: dict = {}  # initial guess for elements, if given
        self.input_elem_abundance: dict = {}  # input elemental abundance for a spectra, not fitted, just used for TS

        # bounds for the minimization
        self.bound_min_vmac: float = 0  # km/s
        self.bound_max_vmac: float = 30
        self.bound_min_rotation: float = 0  # km/s
        self.bound_max_rotation: float = 30
        self.bound_min_vmic: float = 0.01  # km/s
        self.bound_max_vmic: float = 5
        self.bound_min_abund: float = -40  # [X/Fe]
        self.bound_max_abund: float = 100
        self.bound_min_feh: float = -4  # [Fe/H]
        self.bound_max_feh: float = 0.5
        self.bound_min_doppler: float = -1  # km/s
        self.bound_max_doppler: float = 1

        # guess bounds for the minimization
        self.guess_min_vmac: float = 0.2  # km/s
        self.guess_max_vmac: float = 8
        self.guess_min_rotation: float = 0.2  # km/s
        self.guess_max_rotation: float = 2
        self.guess_min_vmic: float = 0.8  # km/s
        self.guess_max_vmic: float = 1.5
        self.guess_min_abund: float = -1  # [X/Fe] or [Fe/H]
        self.guess_max_abund: float = 0.4
        self.guess_min_doppler: float = -1  # km/s
        self.guess_max_doppler: float = 1

        self.bound_min_teff: float = 2500
        self.bound_max_teff: float = 8000

        self.bound_min_logg: float = -0.5
        self.bound_max_logg: float = 5.0

        self.guess_plus_minus_neg_teff: float = -1000
        self.guess_plus_minus_pos_teff: float = 1000

        self.guess_plus_minus_neg_logg: float = -0.5
        self.guess_plus_minus_pos_logg: float = 0.5

        self.find_upper_limit: bool = False
        self.sigmas_error: float = 5.0

        self.find_teff_errors: bool = False
        self.teff_error_sigma: float = 5.0

        self.find_logg_errors: bool = False
        self.logg_error_sigma: float = 5.0

        self.model_temperatures: np.ndarray = np.array([])
        self.model_logs: np.ndarray = np.array([])
        self.model_mets: np.ndarray = np.array([])
        self.marcs_value_keys: list = []
        self.marcs_models: dict = {}
        self.pickled_marcs_models_location: str = "None"  # location of the MARCS models pickled file
        self.marcs_values: dict = {}

        self.debug_mode: int = 0  # 0: no debug, 1: show Python warnings, 2: turn Fortran TS verbose setting on
        self.experimental_parallelisation: bool = False  # experimental parallelisation of the TS code

        self.m3dis_n_nu: int = 0
        self.m3dis_hash_table_size: int = 0
        self.m3dis_mpi_cores: int = 0
        self.m3dis_iterations_max: int = 0
        self.m3dis_iterations_max_precompute: int = 0
        self.m3dis_convlim: float = 0
        self.m3dis_snap: int = 0
        self.m3dis_dims: int = 0
        self.m3dis_nx: int = 0
        self.m3dis_ny: int = 0
        self.m3dis_nz: int = 0

        self.xatol_all: float = 0
        self.fatol_all: float = 0
        self.xatol_lbl: float = 0
        self.fatol_lbl: float = 0
        self.xatol_teff: float = 0
        self.fatol_teff: float = 0
        self.xatol_logg: float = 0
        self.fatol_logg: float = 0
        self.xatol_vmic: float = 0
        self.fatol_vmic: float = 0
        # scipy maxfev for the minimisation
        self.maxfev: int = 0
        # Value of lpoint for turbospectrum in spectrum.inc file
        self.lpoint_turbospectrum: int = 0
        # m3dis parameters
        self.m3dis_python_package_name: str = "None"
        # margin in AA, how much of the spectra is kept in the memory. less - less memory. more - bigger rv fits allowed
        self.margin_obs_spectra_save: float = 0
        # adds this much randomness to the guess ratio wise to the guess for the parameters. 0 means guess is the same as the input
        self.guess_random_ratio_to_add: float = 0
        # whether to save different results
        self.save_original_spectra: bool = True
        self.save_fitted_spectra: bool = True
        self.save_convolved_fitted_spectra: bool = True
        self.save_results: bool = True
        self.save_linemask: bool = True
        self.save_fitlist: bool = True
        self.save_config_file: bool = True

        # Set values from config
        if n_workers != 1:
            try:
                tsfitpy_config = tsfitpy_config.result()
            except AttributeError:
                pass
        self.load_spectra_config(tsfitpy_config)

        if self.debug_mode >= 1:
            self.python_verbose: bool = True
        else:
            self.python_verbose: bool = False
        if self.debug_mode >= 2:
            self.ssg_verbose: bool = True
        else:
            self.ssg_verbose: bool = False
        if self.debug_mode <= -1:
            # night mode is super quiet
            self.night_mode: bool = True
        else:
            self.night_mode: bool = False

        self.spec_name: str = str(specname)
        self.spec_path: str = os.path.join(self.spec_input_path, str(specname))
        if not self.night_mode:
            print(self.spec_path)
        self.teff: float = float(teff)
        self.logg: float = float(logg)
        self.feh: float = float(met)
        self.stellar_rv: float = float(rv)  # RV of star (given, but is fitted with extra doppler shift)
        if self.input_elem_abundance is None:  # input abundance - NOT fitted, but just accepted as a constant abund for spectra
            self.input_abund: dict = abundances_dict
        else:
            try:
                self.input_abund: dict = {**abundances_dict, **self.input_elem_abundance[self.spec_name]}
            except KeyError: # if no spec_name in self.input_elem_abundance
                self.input_abund: dict = abundances_dict

        # if Input, then takes it from the fitlist. Otherwise takes it from the constant in the config (vmac + rot)
        if self.input_vmic:
            self.vmic: float = float(micro)  # microturbulence. Set if it is given in input
        else:
            self.vmic = None
        if self.input_vmac:
            self.vmac: float = float(macro)  # macroturbulence km/s, constant for all stars if not fitted
        if self.input_rotation:
            self.rotation: float = float(rotation)  # rotation km/s, constant for all stars if not fitted
        if resolution != 0.0 and resolution is not None and not np.isnan(resolution):
            self.resolution: float = float(resolution)  # resolution coming from resolution, if given input here:  central lambda / FWHM
        self.snr: float = float(snr)  # SNR of the spectra, used for error sigma calculation

        self.temp_dir: str = os.path.join(self.global_temp_dir, self.spec_name + str(index_temp_dir),
                                          '')  # temp directory, including date and name of the star fitted
        create_dir(self.temp_dir)  # create temp directory

        self.line_list_path_trimmed: str = line_list_path_trimmed  # location of trimmed files

        self.wavelength_obs: np.ndarray = np.array([])
        self.flux_norm_obs: np.ndarray = np.array([])
        self.error_obs_variance: np.ndarray = np.array([])
        self.load_observed_spectra()

        # for experimental parallelisation need to have dictionary of fitted values so they dont interfere
        # each index is a different line
        self.vmac_fitted_dict: dict = {}
        self.vmic_fitted_dict: dict = {}
        self.rotation_fitted_dict: dict = {}
        self.rv_extra_fitted_dict: dict = {}
        self.elem_abund_fitted_dict: dict = {}
        self.wavelength_fitted_dict: dict = {}
        self.flux_norm_fitted_dict: dict = {}
        self.flux_fitted_dict: dict = {}

    def load_observed_spectra(self):
        """
        Loads the observed spectra and sorts it according to the wavelength. Loads the error sigma if given, otherwise
        sets it to 0.01. Error sigma is converted to variance. It is used in the chi squared calculation.
        """
        wave_ob, flux_ob = np.loadtxt(self.spec_path, usecols=(0, 1), unpack=True, dtype=float)  # observed spectra
        try:
            # if error sigma is given, load it
            error_obs_variance = np.square(np.loadtxt(self.spec_path, usecols=2, unpack=True, dtype=float))
        except (IndexError, ValueError) as e:
            # check if we have SNR
            if self.snr is not None and not np.isnan(self.snr) and self.snr > 0:
                # convert to sigma
                error_obs_variance = np.full(len(wave_ob), np.square(1 / self.snr))
            else:
                # if no error variance is given, set it to 1
                error_obs_variance = np.ones(len(wave_ob)) * 0.0001
                if not self.night_mode:
                    print("No error sigma given in 3rd column, setting to 0.01")
        # sort the observed spectra according to wavelength using numpy argsort
        sorted_obs_wavelength_index = np.argsort(wave_ob)
        wave_ob, flux_ob = wave_ob[sorted_obs_wavelength_index], flux_ob[sorted_obs_wavelength_index]
        error_obs_variance = error_obs_variance[sorted_obs_wavelength_index]
        result_indices = []
        wave_ob_doppler_shifted = apply_doppler_correction(wave_ob, self.stellar_rv)
        for l, r in zip(self.line_begins_sorted, self.line_ends_sorted):
            result_indices.extend(
                np.where((wave_ob_doppler_shifted >= l - self.margin_obs_spectra_save) & (wave_ob_doppler_shifted <= r + self.margin_obs_spectra_save))[0])
        self.wavelength_obs = wave_ob[result_indices]
        self.flux_norm_obs = flux_ob[result_indices]
        self.error_obs_variance = error_obs_variance[result_indices]

    def load_spectra_config(self, tsfitpy_config):
        """
        Loads the config file and sets the values for the class
        :param tsfitpy_config: Config file TSFitPyConfig
        """
        self.atmosphere_type = tsfitpy_config.atmosphere_type
        self.fitting_mode = tsfitpy_config.fitting_mode
        self.include_molecules = tsfitpy_config.include_molecules
        self.nlte_flag = tsfitpy_config.nlte_flag

        self.fit_vmic = tsfitpy_config.fit_vmic
        self.fit_vmac = tsfitpy_config.fit_vmac
        self.fit_rotation = tsfitpy_config.fit_rotation
        self.input_vmic = tsfitpy_config.vmic_input
        self.input_vmac = tsfitpy_config.vmac_input
        self.input_rotation = tsfitpy_config.rotation_input
        self.elem_to_fit = tsfitpy_config.elements_to_fit
        self.fit_feh = tsfitpy_config.fit_feh
        self.lmin = tsfitpy_config.wavelength_min
        self.lmax = tsfitpy_config.wavelength_max
        self.ldelta = tsfitpy_config.wavelength_delta
        self.resolution = tsfitpy_config.resolution
        self.rotation = tsfitpy_config.rotation
        self.vmac = tsfitpy_config.vmac
        self.global_temp_dir = tsfitpy_config.temporary_directory_path
        self.dask_workers = tsfitpy_config.number_of_cpus
        self.bound_min_vmac = tsfitpy_config.bounds_rotation[0]
        self.bound_max_vmac = tsfitpy_config.bounds_rotation[1]
        self.bound_min_rotation = tsfitpy_config.bounds_rotation[0]
        self.bound_max_rotation = tsfitpy_config.bounds_rotation[1]
        self.bound_min_vmic = tsfitpy_config.bounds_vmic[0]
        self.bound_max_vmic = tsfitpy_config.bounds_vmic[1]
        self.bound_min_abund = tsfitpy_config.bounds_abundance[0]
        self.bound_max_abund = tsfitpy_config.bounds_abundance[1]
        self.bound_min_feh = tsfitpy_config.bounds_feh[0]
        self.bound_max_feh = tsfitpy_config.bounds_feh[1]
        self.bound_min_teff = tsfitpy_config.bounds_teff[0]
        self.bound_max_teff = tsfitpy_config.bounds_teff[1]
        self.bound_min_logg = tsfitpy_config.bounds_logg[0]
        self.bound_max_logg = tsfitpy_config.bounds_logg[1]
        self.bound_min_doppler = tsfitpy_config.bounds_doppler[0]
        self.bound_max_doppler = tsfitpy_config.bounds_doppler[1]
        self.guess_min_vmic = tsfitpy_config.guess_range_vmic[0]
        self.guess_max_vmic = tsfitpy_config.guess_range_vmic[1]
        self.guess_min_vmac = tsfitpy_config.guess_range_rotation[0]
        self.guess_max_vmac = tsfitpy_config.guess_range_rotation[1]
        self.guess_min_rotation = tsfitpy_config.guess_range_rotation[0]
        self.guess_max_rotation = tsfitpy_config.guess_range_rotation[1]
        self.guess_min_abund = tsfitpy_config.guess_range_abundance[0]
        self.guess_max_abund = tsfitpy_config.guess_range_abundance[1]
        self.guess_min_doppler = tsfitpy_config.guess_range_doppler[0]
        self.guess_max_doppler = tsfitpy_config.guess_range_doppler[1]
        self.guess_plus_minus_neg_teff = tsfitpy_config.guess_range_teff[0]
        self.guess_plus_minus_pos_teff = tsfitpy_config.guess_range_teff[1]
        self.guess_plus_minus_pos_logg = tsfitpy_config.guess_range_logg[0]
        self.guess_plus_minus_neg_logg = tsfitpy_config.guess_range_logg[1]
        self.debug_mode = tsfitpy_config.debug_mode
        self.experimental_parallelisation = tsfitpy_config.experimental_parallelisation
        self.compiler = tsfitpy_config.compiler

        self.nelement = tsfitpy_config.nelement
        self.spectral_code_path = tsfitpy_config.spectral_code_path

        self.interpol_path = tsfitpy_config.interpolators_path

        self.model_atmosphere_grid_path = tsfitpy_config.model_atmosphere_grid_path
        self.model_atmosphere_list = tsfitpy_config.model_atmosphere_list

        self.model_atom_path = tsfitpy_config.model_atoms_path
        self.departure_file_path = tsfitpy_config.departure_file_path
        self.output_folder = tsfitpy_config.output_folder_path
        self.spec_input_path = tsfitpy_config.spectra_input_path

        self.fit_teff = tsfitpy_config.fit_teff

        self.line_begins_sorted = tsfitpy_config.line_begins_sorted
        self.line_ends_sorted = tsfitpy_config.line_ends_sorted
        self.line_centers_sorted = tsfitpy_config.line_centers_sorted

        self.linemask_file = tsfitpy_config.linemask_file
        self.segment_file = tsfitpy_config.segment_file
        self.seg_begins = tsfitpy_config.seg_begins
        self.seg_ends = tsfitpy_config.seg_ends

        self.depart_bin_file_dict = tsfitpy_config.depart_bin_file_dict
        self.depart_aux_file_dict = tsfitpy_config.depart_aux_file_dict
        self.model_atom_file_dict = tsfitpy_config.model_atom_file_dict
        self.aux_file_length_dict = tsfitpy_config.aux_file_length_dict
        self.ndimen = tsfitpy_config.ndimen

        self.init_guess_dict = tsfitpy_config.init_guess_spectra_dict
        self.input_elem_abundance = tsfitpy_config.input_elem_abundance_dict

        self.find_upper_limit = tsfitpy_config.find_upper_limit
        self.sigmas_error = tsfitpy_config.sigmas_upper_limit
        self.find_teff_errors = tsfitpy_config.find_teff_errors
        self.teff_error_sigma = tsfitpy_config.teff_error_sigma

        self.m3dis_n_nu = tsfitpy_config.n_nu
        self.m3dis_hash_table_size = tsfitpy_config.hash_table_size
        self.m3dis_mpi_cores = tsfitpy_config.mpi_cores
        self.m3dis_iterations_max = tsfitpy_config.iterations_max
        self.m3dis_iterations_max_precompute = tsfitpy_config.iterations_max_precompute
        self.m3dis_convlim = tsfitpy_config.convlim
        self.m3dis_snap = tsfitpy_config.snap
        self.m3dis_dims = tsfitpy_config.dims
        self.m3dis_nx = tsfitpy_config.nx
        self.m3dis_ny = tsfitpy_config.ny
        self.m3dis_nz = tsfitpy_config.nz

        # advanced options
        # scipy xatol and fatol for the minimisation, different methods
        self.xatol_all = tsfitpy_config.xatol_all
        self.fatol_all = tsfitpy_config.fatol_all
        self.xatol_lbl = tsfitpy_config.xatol_lbl
        self.fatol_lbl = tsfitpy_config.fatol_lbl
        self.xatol_teff = tsfitpy_config.xatol_teff
        self.fatol_teff = tsfitpy_config.fatol_teff
        self.xatol_logg = tsfitpy_config.xatol_logg
        self.fatol_logg = tsfitpy_config.fatol_logg
        self.xatol_vmic = tsfitpy_config.xatol_vmic
        self.fatol_vmic = tsfitpy_config.fatol_vmic
        # scipy maxfev for the minimisation
        self.maxfev = tsfitpy_config.maxfev
        # Value of lpoint for turbospectrum in spectrum.inc file
        self.lpoint_turbospectrum = tsfitpy_config.lpoint_turbospectrum
        # m3dis parameters
        self.m3dis_python_package_name = tsfitpy_config.m3dis_python_package_name
        self.margin_obs_spectra_save = tsfitpy_config.margin
        self.guess_random_ratio_to_add = tsfitpy_config.guess_ratio_to_add
        self.save_original_spectra = tsfitpy_config.save_original_spectra
        self.save_fitted_spectra = tsfitpy_config.save_fitted_spectra
        self.save_convolved_fitted_spectra = tsfitpy_config.save_convolved_fitted_spectra
        self.save_results = tsfitpy_config.save_results
        self.save_linemask = tsfitpy_config.save_linemask
        self.save_fitlist = tsfitpy_config.save_fitlist
        self.save_config_file = tsfitpy_config.save_config_file

        self.pickled_marcs_models_location = os.path.join(self.global_temp_dir, "marcs_models.pkl")

    def get_all_guess(self):
        """
        Converts init param guess list to the 2D list for the simplex calculation for all method
        """
        minim_bounds: list = []
        # make an array for initial guess equal to n x ndimen+1
        initial_guess = np.empty((self.ndimen + 1, self.ndimen))
        # 17.11.2022: Tried random guesses. But they DO affect the result if the random guesses are way off.
        # Trying with linspace. Should be better I hope
        min_macroturb = self.guess_min_vmac  # km/s; cannot be less than 0
        max_macroturb = self.guess_max_vmac
        min_abundance = self.guess_min_abund  # either [Fe/H] or [X/Fe] here
        max_abundance = self.guess_max_abund  # for [Fe/H]: hard bounds -4 to 0.5; other elements: bounds are above -40
        min_rv = self.guess_min_doppler  # km/s i think as well
        max_rv = self.guess_max_doppler
        macroturb_guesses = np.linspace(min_macroturb + np.random.random(1)[0] / 2, max_macroturb + np.random.random(1)[0] / 2, self.ndimen + 1)
        abundance_guesses = np.linspace(min_abundance + np.random.random(1)[0] / 10, max_abundance + np.random.random(1)[0] / 10, self.ndimen + 1)
        rv_guesses = np.linspace(min_rv + np.random.random(1)[0] / 10, max_rv + np.random.random(1)[0] / 10, self.ndimen + 1)

        # abund = param[0]
        # dopple = param[1]
        # macroturb = param [2] (if needed)
        initial_guess[:, 0] = abundance_guesses
        if self.fit_feh:
            minim_bounds.append((self.bound_min_feh, self.bound_max_feh))
        else:
            minim_bounds.append((self.bound_min_abund, self.bound_max_abund))
        initial_guess[:, 1] = rv_guesses
        minim_bounds.append((self.bound_min_doppler, self.bound_max_doppler))
        if self.fit_vmac:
            initial_guess[:, 2] = macroturb_guesses
            minim_bounds.append((self.bound_min_vmac, self.bound_max_vmac))


        return initial_guess[0], initial_guess, minim_bounds

    def get_elem_micro_guess(self, min_guess_microturb: float, max_guess_microturb: float, min_guess_abundance: float,
                             max_guess_abundance: float, bound_min_abund: float=None, bound_max_abund: float=None) -> tuple[np.ndarray, list[tuple]]:
        """
        Gets guess if fitting elements and microturbulence
        :param min_guess_microturb: minimum guess for microturbulence
        :param max_guess_microturb: maximum guess for microturbulence
        :param min_guess_abundance: minimum guess for abundance
        :param max_guess_abundance: maximum guess for abundance
        :param bound_min_abund: minimum bound for abundance
        :param bound_max_abund: maximum bound for abundance
        :return: guesses and bounds
        """
        # param[0:nelements-1] = met or abund
        # param[-1] = micro turb

        guess_length = self.nelement
        if self.fit_vmic and self.atmosphere_type != "3D":
            guess_length += 1

        bounds = []

        guesses = np.array([])

        if bound_min_abund is None:
            bound_min_abund = self.bound_min_abund

        if bound_max_abund is None:
            bound_max_abund = self.bound_max_abund

        for i in range(0, self.nelement):
            if self.elem_to_fit[i] == "Fe":
                guess_elem, bound_elem = self.get_simplex_guess(guess_length, min_guess_abundance, max_guess_abundance,
                                                                self.bound_min_feh, self.bound_max_feh)
            else:
                guess_elem, bound_elem = self.get_simplex_guess(guess_length, min_guess_abundance, max_guess_abundance,
                                                                bound_min_abund, bound_max_abund)
            if self.init_guess_dict is not None and self.elem_to_fit[i] in self.init_guess_dict[self.spec_name]:
                abund_guess = self.init_guess_dict[self.spec_name][self.elem_to_fit[i]]
                abundance_guesses = np.linspace(abund_guess - 0.1, abund_guess + 0.1, guess_length + 1)
                if np.size(guesses) == 0:
                    guesses = np.array([abundance_guesses])
                else:
                    guesses = np.append(guesses, [abundance_guesses], axis=0)
                # if initial abundance is given, then linearly give guess +/- 0.1 dex
            else:
                if np.size(guesses) == 0:
                    guesses = np.array([guess_elem])
                else:
                    guesses = np.append(guesses, [guess_elem], axis=0)
            bounds.append(bound_elem)
        if self.fit_vmic and self.atmosphere_type != "3D":  # last is micro
            micro_guess, micro_bounds = self.get_simplex_guess(guess_length, min_guess_microturb, max_guess_microturb, self.bound_min_vmic, self.bound_max_vmic)
            guesses = np.append(guesses, [micro_guess], axis=0)
            bounds.append(micro_bounds)

        guesses = np.transpose(guesses)

        return guesses, bounds

    def get_elem_guess(self, min_abundance: float, max_abundance: float) -> tuple[np.ndarray, list[tuple]]:
        """
        Gets guess if fitting elements (can take several elements at once)
        :param min_abundance: minimum guess for abundance
        :param max_abundance: maximum guess for abundance
        :return: guesses and bounds
        """
        # param[0:nelements-1] = met or abund
        # param[-1] = micro turb

        guess_length = self.nelement

        bounds = []

        guesses = np.array([])

        for i in range(0, self.nelement):
            if self.elem_to_fit[i] == "Fe":
                guess_elem, bound_elem = self.get_simplex_guess(guess_length, min_abundance, max_abundance,
                                                                self.bound_min_feh, self.bound_max_feh)
            else:
                guess_elem, bound_elem = self.get_simplex_guess(guess_length, min_abundance, max_abundance,
                                                                self.bound_min_abund, self.bound_max_abund)
            if self.init_guess_dict is not None and self.elem_to_fit[i] in self.init_guess_dict[self.spec_name]:
                abund_guess = self.init_guess_dict[self.spec_name][self.elem_to_fit[i]]
                abundance_guesses = np.linspace(abund_guess - 0.1, abund_guess + 0.1, guess_length + 1)
                if np.size(guesses) == 0:
                    guesses = np.array([abundance_guesses])
                else:
                    guesses = np.append(guesses, [abundance_guesses], axis=0)
                # if initial abundance is given, then linearly give guess +/- 0.1 dex
            else:
                if np.size(guesses) == 0:
                    guesses = np.array([guess_elem])
                else:
                    guesses = np.append(guesses, [guess_elem], axis=0)
            bounds.append(bound_elem)

        guesses = np.transpose(guesses)

        return guesses, bounds

    def get_vmic_guess(self, min_vmic_guess: float, max_vmic_guess: float) -> tuple[np.ndarray, list[tuple]]:
        """
        Gets guess if fitting microturbulence
        :param min_vmic_guess: minimum guess for microturbulence
        :param max_vmic_guess: maximum guess for microturbulence
        :return: guesses and bounds
        """
        # param[0:nelements-1] = met or abund
        # param[-1] = vmic

        guess_length = 1

        bounds = []

        guesses = np.array([])

        if self.atmosphere_type != "3D":  # last is micro
            micro_guess, micro_bounds = self.get_simplex_guess(guess_length, min_vmic_guess, max_vmic_guess, self.bound_min_vmic, self.bound_max_vmic)
            guesses = np.array([micro_guess])
            bounds.append(micro_bounds)

        guesses = np.transpose(guesses)

        return guesses, bounds

    def get_simplex_guess(self, length: int, min_guess: float, max_guess: float, min_bound: float, max_bound: float) -> tuple[np.ndarray, tuple]:
        """
        Gets guess if it is fitted for simplex guess
        :param length: number of dimensions (output length+1 array)
        :param min_guess: minimum guess
        :param max_guess: maximum guess
        :param min_bound: minimum bound
        :param max_bound: maximum bound
        :return: Initial guess and minimum bound
        """
        if min_guess < min_bound:
            min_guess = min_bound
        if max_guess > max_bound:
            max_guess = max_bound

        minim_bounds = (min_bound, max_bound)
        # basically adds a bit of randomness to the guess up to this % of the diff of guesses
        guess_difference = np.abs(max_guess - min_guess) * self.guess_random_ratio_to_add

        initial_guess = np.linspace(min_guess + np.random.random() * guess_difference,
                                    max_guess - np.random.random() * guess_difference, length + 1)

        return initial_guess, minim_bounds

    def get_rv_macro_rotation_guess(self, min_rv: float=None, max_rv: float=None, min_macroturb: float=None,
                                    max_macroturb: float=None, min_rotation: float=None,
                                    max_rotation: float=None) -> tuple[np.ndarray, list[tuple]]:
        """
        Gets rv and macroturbulence guess if it is fitted for simplex guess. np.median(guesses[:, 0]) == 0 is checked
        and if it is 0, then the guess is changed to be halfway between 0 and the max/min, depending whichever is not 0
        Use np.median(guesses, axis=0) to get the median of each parameter for L-BFGS-B minimisation
        :param min_rv: minimum RV for guess (not bounds)
        :param max_rv: maximum RV for guess (not bounds)
        :param min_macroturb: minimum macro for guess (not bounds)
        :param max_macroturb: maximum macro for guess (not bounds)
        :param min_rotation: minimum rotation for guess (not bounds)
        :param max_rotation: maximum rotation for guess (not bounds)
        :return: Initial guess and minimum bound
        """
        # param[0] = rv
        # param[1] = macro IF FITTED
        # param[-1] = rotation IF FITTED

        if min_rv is None:
            min_rv = self.guess_min_doppler  # km/s
        if max_rv is None:
            max_rv = self.guess_max_doppler
        if min_macroturb is None:
            min_macroturb = self.guess_min_vmac
        if max_macroturb is None:
            max_macroturb = self.guess_max_vmac
        if min_rotation is None:
            min_rotation = self.guess_min_rotation
        if max_rotation is None:
            max_rotation = self.guess_max_rotation

        guess_length = 1
        if self.fit_vmac:
            guess_length += 1
        if self.fit_rotation:
            guess_length += 1

        bounds = []

        rv_guess, rv_bounds = self.get_simplex_guess(guess_length, min_rv, max_rv, self.bound_min_doppler, self.bound_max_doppler)
        guesses = np.array([rv_guess])
        bounds.append(rv_bounds)
        if self.fit_vmac:
            macro_guess, macro_bounds = self.get_simplex_guess(guess_length, min_macroturb, max_macroturb, self.bound_min_vmac, self.bound_max_vmac)
            guesses = np.append(guesses, [macro_guess], axis=0)
            bounds.append(macro_bounds)
        if self.fit_rotation:
            rotation_guess, rotation_bounds = self.get_simplex_guess(guess_length, min_rotation, max_rotation, self.bound_min_rotation, self.bound_max_rotation)
            guesses = np.append(guesses, [rotation_guess], axis=0)
            bounds.append(rotation_bounds)

        guesses = np.transpose(guesses)

        # check if median of the guess is 0 for either of the parameters, if so, then add a value to the guess that is
        # halfway between 0 and the max/min, depending whichever is not 0
        # otherwise, if the median is 0, the L-BFGS-B minimisation will for some reason get stuck and not move much
        if np.median(guesses[:, 0]) == 0:
            if np.abs(min_rv) > np.abs(max_rv):
                guesses[:, 0][int(np.size(guesses[:, 0]) / 2)] = np.abs(min_rv) / 2
            else:
                guesses[:, 0][int(np.size(guesses[:, 0]) / 2)] = np.abs(max_rv) / 2
        if self.fit_vmac:
            if np.median(guesses[:, 1]) <= 0.5:
                if np.abs(min_macroturb) > np.abs(max_macroturb):
                    guesses[:, 1][int(np.size(guesses[:, 1]) / 2)] = np.abs(min_macroturb) / 2
                else:
                    guesses[:, 1][int(np.size(guesses[:, 1]) / 2)] = np.abs(max_macroturb) / 2
        if self.fit_rotation:
            if np.median(guesses[:, -1]) <= 0.5:
                if np.abs(min_rotation) > np.abs(max_rotation):
                    guesses[:, -1][int(np.size(guesses[:, -1]) / 2)] = np.abs(min_rotation) / 2
                else:
                    guesses[:, -1][int(np.size(guesses[:, -1]) / 2)] = np.abs(max_rotation) / 2

        return guesses, bounds

    def save_observed_spectra(self, path: str):
        """
        save observed spectra using np.savetxt, with up to 5 decimals
        :param path: path to save the spectra
        """
        if self.save_original_spectra:
            np.savetxt(path, np.transpose([self.wavelength_obs, self.flux_norm_obs, np.sqrt(self.error_obs_variance)]),
                       fmt='%.5f', header="wavelength flux error_sigma")

    def configure_and_run_synthetic_code(self, spectrumclass: SyntheticSpectrumGenerator, feh: float, elem_abund: dict,
                                         vmic: float, lmin: float, lmax: float, windows_flag: bool=False, temp_dir: str=None,
                                         teff: float=None, logg: float=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Configures spectrumclass depending on input parameters and runs either NLTE or LTE
        :param spectrumclass: SyntheticSpectrumGenerator
        :param feh: metallicity of star
        :param elem_abund: dictionary with iron and elemental abundances as [X/H]
        :param vmic: microturbulence parameter
        :param lmin: minimum wavelength to synthesise spectra
        :param lmax: maximum wavelength to synthesise spectra
        :param windows_flag: if True, then uses windows, if False, then creates only between lmin and lmax. Only for TS
        TODO: windows_flag is bugged? some lines dont appear in the windows?
        :param temp_dir: temporary directory to save the spectra/generate files
        :param teff: effective temperature of star (otherwise uses self.teff)
        :param logg: logg of star (otherwise uses self.logg)
        :return: 3 arrays of wavelength, flux_normalised, flux. Empty arrays or Nones if failed
        """
        if temp_dir is None:
            temp_dir = self.temp_dir
        else:
            temp_dir = temp_dir
        create_dir(temp_dir)
        if teff is None:
            teff = self.teff
        else:
            teff = teff
        if logg is None:
            logg = self.logg
        else:
            logg = logg
        if self.nlte_flag:
            logging.debug(f"NLTE model atoms: {self.model_atom_file_dict}")
            spectrumclass.configure(t_eff=teff, log_g=logg, metallicity=feh, turbulent_velocity=vmic,
                                    lambda_delta=self.ldelta, lambda_min=lmin, lambda_max=lmax,
                                    free_abundances=elem_abund, temp_directory=temp_dir, nlte_flag=True,
                                    verbose=self.ssg_verbose,
                                    atmosphere_dimension=self.atmosphere_type, windows_flag=windows_flag,
                                    segment_file=self.segment_file, line_mask_file=self.linemask_file,
                                    depart_bin_file=self.depart_bin_file_dict, depart_aux_file=self.depart_aux_file_dict,
                                    model_atom_file=self.model_atom_file_dict)
        else:
            spectrumclass.configure(t_eff=teff, log_g=logg, metallicity=feh, turbulent_velocity=vmic,
                                    lambda_delta=self.ldelta, lambda_min=lmin, lambda_max=lmax,
                                    free_abundances=elem_abund, temp_directory=temp_dir, nlte_flag=False,
                                    verbose=self.ssg_verbose,
                                    atmosphere_dimension=self.atmosphere_type, windows_flag=windows_flag,
                                    segment_file=self.segment_file, line_mask_file=self.linemask_file)
        return spectrumclass.synthesize_spectra()

    def create_ssg_object(self, marcs_values_tuple, segment_index=None) -> SyntheticSpectrumGenerator:
        """
        Creates the synthetic spectrum generator object depending whether TS or M3DIS is used (or other in the future?)
        :param marcs_values_tuple: unpickled marcs models and other values
        :return: SyntheticSpectrumGenerator object
        """
        model_temperatures, model_logs, model_mets, marcs_value_keys, marcs_models, marcs_values = marcs_values_tuple
        if marcs_models is None:
            marcs_models = self.marcs_models
        if self.compiler.lower() == "m3dis":
            if self.nlte_flag and self.m3dis_iterations_max_precompute == 0 and self.m3dis_iterations_max == 0:
                model_atom_path = os.path.join(self.model_atom_path, f"{segment_index}", "")
            else:
                model_atom_path = self.model_atom_path

            ssg = M3disCall(
                m3dis_path=self.spectral_code_path,
                interpol_path=self.interpol_path,
                line_list_paths=self.line_list_path_trimmed,
                marcs_grid_path=self.model_atmosphere_grid_path,
                marcs_grid_list=self.model_atmosphere_list,
                model_atom_path=model_atom_path,
                departure_file_path=self.departure_file_path,
                aux_file_length_dict=self.aux_file_length_dict,
                model_temperatures=model_temperatures,
                model_logs=model_logs,
                model_mets=model_mets,
                marcs_value_keys=marcs_value_keys,
                marcs_models=marcs_models,
                marcs_values=marcs_values,
                m3dis_python_module=self.m3dis_python_module,
                n_nu=self.m3dis_n_nu,
                hash_table_size=self.m3dis_hash_table_size,
                mpi_cores=self.m3dis_mpi_cores,
                iterations_max=self.m3dis_iterations_max,
                convlim=self.m3dis_convlim,
                snap=self.m3dis_snap,
                dims=self.m3dis_dims,
                nx=self.m3dis_nx,
                ny=self.m3dis_ny,
                nz=self.m3dis_nz,
                night_mode=self.night_mode
            )
            if self.m3dis_iterations_max_precompute <= 0:
                ssg.use_precomputed_depart = False
        else:
            ssg = TurboSpectrum(
                turbospec_path=self.spectral_code_path,
                interpol_path=self.interpol_path,
                line_list_paths=self.line_list_path_trimmed,
                marcs_grid_path=self.model_atmosphere_grid_path,
                marcs_grid_list=self.model_atmosphere_list,
                model_atom_path=self.model_atom_path,
                departure_file_path=self.departure_file_path,
                aux_file_length_dict=self.aux_file_length_dict,
                model_temperatures=model_temperatures,
                model_logs=model_logs,
                model_mets=model_mets,
                marcs_value_keys=marcs_value_keys,
                marcs_models=marcs_models,
                marcs_values=marcs_values,
                night_mode=self.night_mode)
            ssg.lpoint = self.lpoint_turbospectrum
        return ssg

    def precompute_departure(self):
        """
        Precomputes departure coefficients for the first element in the list of elements to fit for M3D.
        Saves the departure coefficients in the temp directory. Calculates it for [X/Fe] = 0 or star's input [Fe/H] (0?)
        """
        if self.nlte_flag and self.m3dis_iterations_max_precompute > 0:
            ssg = self.create_ssg_object(self._get_marcs_models())
            # need to run NLTE run once, so that can reuse precomputed departure coefficients
            if self.elem_to_fit[0] == "Fe":
                input_abund = {"Fe": self.feh}
                feh = 0
            else:
                # 0 because assume [X/Fe] = 0
                input_abund = {"Fe": self.feh, f"{self.elem_to_fit[0]}": 0 + self.feh}
                feh = self.feh
            if self.input_vmic:
                vmic = self.vmic
            else:
                vmic = calculate_vturb(self.teff, self.logg, self.feh)
            temp_dir = os.path.join(self.temp_dir, "precomputed_depart")
            # skips lots of
            ssg.skip_linelist = True
            ssg.save_spectra = False
            ssg.iterations_max = self.m3dis_iterations_max_precompute
            ssg.use_precomputed_depart = False
            self.configure_and_run_synthetic_code(ssg, feh, input_abund, vmic, self.lmin, self.lmax, temp_dir=temp_dir)
            # delete ssg
            del ssg
        return None

    def fit_all(self) -> list:
        """
        Fit all lines at once, trying to minimise chi squared. Changes: 1 element, RV, macroturbulence
        :return: Result is a string containing dictionary with the results: specname, Fe_H, Doppler_Shift_add_to_RV,
        chi_squared, vmac
        """
        # timing how long it took
        time_start = time.perf_counter()

        ssg = self.create_ssg_object(self._get_marcs_models())

        initial_simplex_guess, init_param_guess, minim_bounds = self.get_all_guess()

        function_arguments = (ssg, self)
        minimize_options = {'maxiter': self.ndimen * self.maxfev, 'disp': self.python_verbose,
                            'initial_simplex': init_param_guess, 'xatol': self.xatol_all, 'fatol': self.fatol_all}
        res = minimize_function(calc_chi_sqr_all_mode, initial_simplex_guess, function_arguments, minim_bounds, 'Nelder-Mead', minimize_options)
        # print final result from minimazation
        if not self.night_mode:
            print(res.x)

        if self.fit_feh:
            output_elem_column = "Fe_H"
        else:
            output_elem_column = f"{self.elem_to_fit[0]}_Fe"

        result_dict = {
            "specname": self.spec_name,
            output_elem_column: res.x[0],
            "Doppler_Shift_add_to_RV": res.x[1],
            "chi_squared": res.fun,
            "vmac": res.x[2] if self.fit_vmac else self.vmac
        }
        time_end = time.perf_counter()
        if not self.night_mode:
            print(f"Total runtime was {(time_end - time_start) / 60.:2f} minutes.")
        return [{"result": result_dict}]

    def fit_lbl_function(self, client: Client, fitting_function: Callable, find_sigma_error: bool) -> list:
        """
        Fits line by line, by going through each line in the linelist. This is a generic function that can be used for
        any lbl fitting function. One passes in the fitting function, and it will be run for each line.
        :param client: dask client
        :param fitting_function: function to fit the line
        :param find_sigma_error: if True, then finds sigma error based on self.sigmas_error for chi sqr
        :return: List with the results. Each element is a dict containing the results and the fit
        """
        result_list = []

        for line_number in range(len(self.line_begins_sorted)):
            if self.dask_workers != 1:
                result_one_line = client.submit(fitting_function, line_number)
                final_result = client.submit(self.analyse_lbl_fit, result_one_line, line_number, find_sigma_error, self.sigmas_error)
                result_list.append(final_result)
            else:
                time_start = time.perf_counter()
                if not self.night_mode:
                    print(f"Fitting line at {self.line_centers_sorted[line_number]} angstroms")

                result_one_line = fitting_function(line_number)
                result_list.append(self.analyse_lbl_fit(result_one_line, line_number, find_sigma_error, self.sigmas_error))

                time_end = time.perf_counter()
                if not self.night_mode:
                    print(f"Total runtime was {(time_end - time_start) / 60:.2f} minutes.")

        return result_list

    def analyse_lbl_fit(self, result_one_line: dict, line_number: int, find_error_sigma: bool, sigmas_error: float) -> dict:
        """
        Analyses the result of the lbl fitting. Calculates EW, convolves the spectra, saves the spectra, finds sigma
        error if needed, calculates flags and saves the fitted spectra in the result_spectrum_specname.spec file
        :param result_one_line: dictionary with the results for one line
        :param line_number: number of the line
        :param find_error_sigma: if True, then finds sigma error based on self.sigmas_error for chi sqr
        :param sigmas_error: number of sigmas for error
        :return: dictionary with the results for one line
        """
        #result_list = []
        # {"result": , "fit_wavelength": , "fit_flux_norm": , "fit_flux": , "fit_wavelength_conv": , "fit_flux_norm_conv": }

        flag_error = "00000000"
        """
        1st bit: spectra not fitted at all
        2nd bit: 2 or less points in the line
        3rd bit: all flux points are above 1 or below 0
        4th bit: EW of the line is significantly different from the EW of the line in the model; perhaps within factor of 1.5
        5th bit: if number of iterations of the line is <= 3
        6th bit:
        7th bit: 
        8th bit:
        """
        flag_warning = "00000000"
        """
        1st bit: if the fitted parameters are at the edge of the bounds
        2nd bit: if at least one flux point is above 1.1 or below 0
        3rd bit: 
        4th bit: if EW of the line is significantly different from the EW of the line in the model; perhaps within factor of 1.25
        5th bit: if number of iterations of the line is <= 5
        6th bit:
        7th bit:
        8th bit:
        """

        if len(result_one_line["fit_wavelength"]) > 0 and result_one_line["chi_sqr"] <= 99999:
            if self.save_fitted_spectra:
                with open(os.path.join(self.output_folder, f"result_spectrum_{self.spec_name}.spec"), 'a') as g:
                    indices_argsorted = np.argsort(result_one_line['fit_wavelength'])
                    np.savetxt(g, np.column_stack((result_one_line['fit_wavelength'][indices_argsorted], result_one_line['fit_flux_norm'][indices_argsorted], result_one_line['fit_flux'][indices_argsorted])), fmt=('%.5f', '%.5f', '%.10f'))
            line_left, line_right = self.line_begins_sorted[line_number], self.line_ends_sorted[line_number]
            segment_left, segment_right = self.seg_begins[line_number], self.seg_ends[line_number]

            wavelength_fit_array = result_one_line['fit_wavelength']
            norm_flux_fit_array = result_one_line['fit_flux_norm']

            indices_to_use_cut = np.where(
                (wavelength_fit_array <= segment_right) & (wavelength_fit_array >= segment_left))
            wavelength_fit_array_cut, norm_flux_fit_array_cut = wavelength_fit_array[indices_to_use_cut], \
            norm_flux_fit_array[indices_to_use_cut]
            wavelength_fit_conv, flux_fit_conv = get_convolved_spectra(wavelength_fit_array_cut,
                                                                       norm_flux_fit_array_cut, self.resolution,
                                                                       result_one_line["macroturb"],
                                                                       result_one_line["rotation"])

            equivalent_width = calculate_equivalent_width(wavelength_fit_conv, flux_fit_conv, line_left, line_right)

            extra_wavelength_to_save = 1  # AA extra wavelength to save left and right of the line

            # this will save extra +/- extra_wavelength_to_save in convolved spectra. But just so that it doesn't
            # overlap other lines, I save only up to half of the other linemask if they are close enough
            if line_number > 0:
                line_previous_right = self.line_ends_sorted[line_number - 1]
                left_bound_to_save = max(line_left - extra_wavelength_to_save,
                                         (line_left - line_previous_right) / 2 + line_previous_right)
            else:
                left_bound_to_save = line_left - extra_wavelength_to_save
            if line_number < len(self.line_begins_sorted) - 1:
                line_next_left = self.line_begins_sorted[line_number + 1]
                right_bound_to_save = min(line_right + extra_wavelength_to_save,
                                          (line_next_left - line_right) / 2 + line_right)
            else:
                right_bound_to_save = line_right + extra_wavelength_to_save
            indices_to_save_conv = np.logical_and.reduce(
                (wavelength_fit_conv > left_bound_to_save, wavelength_fit_conv < right_bound_to_save))

            if self.save_convolved_fitted_spectra:
                with open(os.path.join(self.output_folder, f"result_spectrum_{self.spec_name}_convolved.spec"),
                          'a') as h:
                    # first sort the arrays by wavelength
                    indices_argsorted = np.argsort(wavelength_fit_conv[indices_to_save_conv])
                    np.savetxt(h, np.column_stack((wavelength_fit_conv[indices_to_save_conv][indices_argsorted], flux_fit_conv[indices_to_save_conv][indices_argsorted])), fmt='%.5f')

            wave_ob = apply_doppler_correction(self.wavelength_obs, self.stellar_rv + result_one_line["rv"])
            flux_ob = self.flux_norm_obs

            # cut to the line
            indices_to_use_cut = np.where((wave_ob <= line_right) & (wave_ob >= line_left))
            wave_ob_cut, flux_ob_cut = wave_ob[indices_to_use_cut], flux_ob[indices_to_use_cut]

            # ERROR FLAGS
            # see how many points are in the line
            wave_ob_points = len(wave_ob_cut)
            if wave_ob_points <= 2:
                # change the 2nd bit to 1, while not changing the rest
                flag_error = flag_error[:1] + "1" + flag_error[2:]

            # check if all flux points are above 1 or below 0
            if np.all(flux_ob_cut >= 1) or np.all(flux_ob_cut <= 0):
                flag_error = flag_error[:2] + "1" + flag_error[3:]

            # check if EW is significantly different from the EW of the line in the model
            ratio_threshold = 1.5
            # calculate EW of the line in the spectra
            equivalent_width_ob = calculate_equivalent_width(wave_ob_cut, flux_ob_cut, line_left, line_right)
            # check if EW of the line in the spectra is within ratio_threshold of the EW of the line in the model
            if equivalent_width_ob > equivalent_width * ratio_threshold or equivalent_width_ob < equivalent_width / ratio_threshold:
                flag_error = flag_error[:3] + "1" + flag_error[4:]

            # check if number of fits of the line is <= 3, but also check that the dictionary exists
            if "fit_iterations" in result_one_line:
                if result_one_line["fit_iterations"] <= 3:
                    flag_error = flag_error[:4] + "1" + flag_error[5:]

            # WARNING FLAGS
            # if fitting feh, check if the fitted parameters are at the edge of the bounds
            if self.fitting_mode == "lbl":
                if self.fit_feh:
                    if result_one_line["fitted_abund"] == self.bound_min_feh or result_one_line["fitted_abund"] == self.bound_max_feh:
                        flag_warning = "1" + flag_warning[1:]
                else:
                    # check if the fitted parameters are at the edge of the bounds
                    if result_one_line["fitted_abund"] == self.bound_min_abund or result_one_line[
                        "fitted_abund"] == self.bound_max_abund:
                        flag_warning = "1" + flag_warning[1:]

            if result_one_line["result"]["Doppler_Shift_add_to_RV"] == self.bound_min_doppler or result_one_line["result"]["Doppler_Shift_add_to_RV"] == self.bound_max_doppler:
                flag_warning = "1" + flag_warning[1:]

            if self.fit_vmac:
                if result_one_line["result"]["Macroturb"] == self.bound_min_vmac or result_one_line["result"]["Macroturb"] == self.bound_max_vmac:
                    flag_warning = "1" + flag_warning[1:]

            if self.fit_rotation:
                if result_one_line["result"]["rotation"] == self.bound_min_rotation or result_one_line["result"]["rotation"] == self.bound_max_rotation:
                    flag_warning = "1" + flag_warning[1:]

            if self.fit_teff:
                if result_one_line["result"]["Teff"] == self.bound_min_teff or result_one_line["result"]["Teff"] == self.bound_max_teff:
                    flag_warning = "1" + flag_warning[1:]

            if self.fit_vmic:
                if result_one_line["result"]["Microturb"] == self.bound_min_vmic or result_one_line["result"]["Microturb"] == self.bound_max_vmic:
                    flag_warning = "1" + flag_warning[1:]

            # check if at least one flux point is above 1.1 or below 0
            if np.any(flux_ob_cut >= 1.1) or np.any(flux_ob_cut < 0):
                flag_warning = flag_warning[:1] + "1" + flag_warning[2:]

            # check if EW is significantly different from the EW of the line in the model
            ratio_threshold = 1.25
            # check if EW of the line in the spectra is within ratio_threshold of the EW of the line in the model
            if equivalent_width_ob > equivalent_width * ratio_threshold or equivalent_width_ob < equivalent_width / ratio_threshold:
                flag_warning = flag_warning[:3] + "1" + flag_warning[4:]

            # check if number of fits of the line is <= 5, but also check that the dictionary exists
            if "fit_iterations" in result_one_line:
                if result_one_line["fit_iterations"] <= 5:
                    flag_warning = flag_warning[:4] + "1" + flag_warning[5:]

        else:
            equivalent_width = 999999
            flag_error = "10000000"
        result_one_line["result"]['ew'] = equivalent_width * 1000
        result_one_line["result"]["flag_error"] = flag_error
        result_one_line["result"]["flag_warning"] = flag_warning

        if find_error_sigma:
            if flag_error == "00000000":
                result_upper_limit = self.lbl_abund_find_sigma_error(result_one_line, sigmas_error, line_number)
                result_one_line["result"][f"err_{int(sigmas_error)}_err"] = np.abs(result_upper_limit["fitted_abund"] - result_one_line["fitted_abund"])
                result_one_line["result"]["err_chi_sqr_diff"] = result_upper_limit["chi_sqr"] - result_one_line["chi_sqr"]
            else:
                result_one_line["result"][f"err_{int(sigmas_error)}_err"] = 999999
                result_one_line["result"]["err_chi_sqr_diff"] = 999999

        # let's only keep 4 decimals for most of the results in result_one_line["result"]
        for key, value in result_one_line["result"].items():
            keys_to_skip = ["chi_sqr", "err_chi_sqr_diff", "flag_error", "flag_warning", "chi_squared"]
            if isinstance(value, float) and key not in keys_to_skip:
                result_one_line["result"][key] = round(value, 4)

        return result_one_line

    def lbl_abund_find_sigma_error(self, result_one_line, sigmas_upper_limit, line_number: int):
        """
        Finds the sigma error for the abundance of the line, by fitting the line with the abundance offset by
        sigmas_upper_limit ** 2. I.e. it increases the abundance and finds the value which results in chi_sqr
        higher than the chi_sqr of the original fit by sigmas_upper_limit ** 2
        It uses RV, broadening and vmic from the original fit. TODO: Is it really correct to use same broadening?
        :param result_one_line: the result of the lbl fitting with the original abundance
        :param sigmas_upper_limit: number of sigmas offset to find the abundance error
        :param line_number: line's index to fit
        :return: dictionary with the offset abundance and fitted chi_sqr
        """
        if not self.night_mode:
            print(f"Fitting error of {sigmas_upper_limit} sigma at {self.line_centers_sorted[line_number]} angstroms")

        temp_directory = os.path.join(self.temp_dir, str(np.random.random()), "")

        offset_chi_sqr = result_one_line["chi_sqr"] + np.square(sigmas_upper_limit) / calculate_dof(self.wavelength_obs, lmin=self.line_begins_sorted[line_number], lmax=self.line_ends_sorted[line_number])
        start_abundance = result_one_line["fitted_abund"]
        # we are trying to find abundance which results in chi_sqr = offset_chi_sqr. so we need to add some constant
        # that overshoots this offset_chi_sqr, so that we can find the abundance that results in offset_chi_sqr
        # thus we add 3 + sigmas_upper_limit to the offset_chi_sqr
        # Hopefully that will be enough to overshoot the offset_chi_sqr
        end_abundance = start_abundance + sigmas_upper_limit + 3

        segment_index = np.where(np.logical_and(self.seg_begins <= self.line_centers_sorted[line_number],
                                        self.line_centers_sorted[line_number] <= self.seg_ends))[0][0]

        ssg = self.create_ssg_object(self._get_marcs_models(), segment_index=segment_index)

        if not self.night_mode:
            print(self.line_centers_sorted[line_number], self.line_begins_sorted[line_number],
                  self.line_ends_sorted[line_number])

        ssg.line_list_paths = [get_trimmed_lbl_path_name(self.line_list_path_trimmed, segment_index)]

        function_arguments = (ssg, self, self.line_begins_sorted[line_number], self.line_ends_sorted[line_number],
                              self.seg_begins[segment_index], self.seg_ends[segment_index], temp_directory,
                              result_one_line["rv"], result_one_line["macroturb"], result_one_line["rotation"],
                              result_one_line["vmic"], offset_chi_sqr)
        try:
            res = root_scalar(calc_chi_sqr_abund_error, args=function_arguments, bracket=[start_abundance, end_abundance],
                              method='brentq', options={'disp': False}, xtol=0.0025)
            if not self.night_mode:
                print(res)
            fitted_abund = res.root
            result_error = {"chi_sqr": offset_chi_sqr, "fitted_abund": fitted_abund}
        except IndexError as error:
            if not self.night_mode:
                print(f"{error} is line in the spectrum? {self.line_centers_sorted[line_number]}")
            result_error = {"fitted_abund": 999999, "chi_sqr": 999999}
        except ValueError as error:
            if not self.night_mode:
                print(f"{error} in finding sigma error for the line {self.line_centers_sorted[line_number]}")
            result_error = {"fitted_abund": 999999, "chi_sqr": 999999}

        shutil.rmtree(temp_directory)

        return result_error

    def fit_lbl_teff(self, line_number: int) -> dict:
        """
        Fits lbl by changing teff
        :param line_number: Which line number/index in line_center_sorted is being fitted
        :return: best fit result dictionary
        """
        temp_directory = os.path.join(self.temp_dir, str(np.random.random()), "")

        # here we find which segment the line is in
        segment_index = np.where(np.logical_and(self.seg_begins <= self.line_centers_sorted[line_number],
                                        self.line_centers_sorted[line_number] <= self.seg_ends))[0][0]
        if not self.night_mode:
            print(self.line_centers_sorted[line_number], self.line_begins_sorted[line_number], self.line_ends_sorted[line_number])

        param_guess = np.array([[self.teff + self.guess_plus_minus_neg_teff], [self.teff + self.guess_plus_minus_pos_teff]])
        min_bounds = [(self.bound_min_teff, self.bound_max_teff)]

        ssg = self.create_ssg_object(self._get_marcs_models(), segment_index=segment_index)

        ssg.line_list_paths = [get_trimmed_lbl_path_name(self.line_list_path_trimmed, segment_index)]

        function_arguments = (ssg, self, self.line_begins_sorted[line_number], self.line_ends_sorted[line_number],
                              self.seg_begins[segment_index], self.seg_ends[segment_index], temp_directory, line_number)
        minimize_options = {'maxfev': self.maxfev, 'disp': self.python_verbose, 'initial_simplex': param_guess,
                            'xatol': self.xatol_teff, 'fatol': self.fatol_teff}
        try:
            res = minimize_function(calc_chi_sqr_teff, param_guess[0], function_arguments, min_bounds, 'Nelder-Mead', minimize_options)
            if not self.night_mode:
                print(res.x)

            teff = res.x[0]
            chi_squared = res.fun

            feh = self.feh
            doppler_fit = self.rv_extra_fitted_dict[line_number]
            if self.vmic is not None:  # Input given
                microturb = self.vmic
            else:
                microturb = calculate_vturb(teff, self.logg, feh)

            if self.fit_vmac:
                macroturb = self.vmac_fitted_dict[line_number]
            else:
                macroturb = self.vmac
            if self.fit_rotation:
                rotation = self.rotation_fitted_dict[line_number]
            else:
                rotation = self.rotation

            wave_result = self.wavelength_fitted_dict[line_number]
            flux_norm_result = self.flux_norm_fitted_dict[line_number]
            flux_result = self.flux_fitted_dict[line_number]
        except IndexError:
            if not self.night_mode:
                print(f"Line {line_number} not fitted, is your line in the spectrum?")
            teff = 999999
            doppler_fit = 999999
            microturb = 999999
            macroturb = 999999
            rotation = 999999
            chi_squared = 999999
            self.wavelength_fitted_dict[line_number] = np.array([])
            self.flux_norm_fitted_dict[line_number] = np.array([])
            self.flux_fitted_dict[line_number] = np.array([])
            wave_result = np.array([])
            flux_norm_result = np.array([])
            flux_result = np.array([])


        if np.size(wave_result) == 0 or teff >= 999998:
            if not self.night_mode:
                print(f"Failed spectra generation completely, line is not fitted at all, not saving spectra then")

        if self.find_teff_errors and teff <= 999998:
            dof = calculate_dof(self.wavelength_obs, lmin=self.line_begins_sorted[line_number], lmax=self.line_ends_sorted[line_number])
            if not self.night_mode:
                print(f"Fitting {self.teff_error_sigma} sigma at {self.line_centers_sorted[line_number]} angstroms")
            try:
                teff_error = np.abs(self.lbl_teff_find_sigma_error(line_number, doppler_fit,
                                                                   macroturb, rotation,
                                                                   microturb,
                                                                   offset_chisqr=(chi_squared + np.square(self.teff_error_sigma) / dof),
                                                                   bound_min_teff=teff,
                                                                   bound_max_teff=teff + 1000) - teff)
            except ValueError as err:
                if not self.night_mode:
                    print(err)
                try:
                    teff_error = np.abs(self.lbl_teff_find_sigma_error(line_number, doppler_fit,
                                                                       macroturb, rotation,
                                                                       microturb,
                                                                       offset_chisqr=(chi_squared + np.square(self.teff_error_sigma) / dof),
                                                                       bound_min_teff=teff - 1000,
                                                                       bound_max_teff=teff) - teff)
                except ValueError as err:
                    if not self.night_mode:
                        print(err)
                    teff_error = 1000
        else:
            teff_error = 999999

        result_dict = {
            "specname": self.spec_name,
            "Teff": teff,
            "Teff_error": teff_error,
            "wave_center": self.line_centers_sorted[line_number],
            "wave_start": self.line_begins_sorted[line_number],
            "wave_end": self.line_ends_sorted[line_number],
            "Doppler_Shift_add_to_RV": doppler_fit,
            "Microturb": microturb,
            "Macroturb": macroturb,
            "rotation": rotation,
            "chi_squared": chi_squared
        }

        one_result = {"result": result_dict, "rv": doppler_fit, "vmic": microturb, "fit_wavelength": wave_result,
                      "fit_flux_norm": flux_norm_result, "fit_flux": flux_result, "macroturb": macroturb,
                      "rotation": rotation, "chi_sqr": chi_squared}
        shutil.rmtree(temp_directory)

        return one_result


    def lbl_teff_find_sigma_error(self, line_number: int, fitted_rv: float, fitted_vmac: float, fitted_rotation: float,
                                  fitted_vmic: float, offset_chisqr: float, bound_min_teff: float,
                                  bound_max_teff: float) -> float:
        """
        Finds the sigma error for the teff of the line, by fitting the line with the teff offset by sigmas_upper_limit ** 2.
        :param line_number: line's index to fit
        :param fitted_rv: fitted rv
        :param fitted_vmac: fitted vmac
        :param fitted_rotation: fitted rotation
        :param fitted_vmic: fitted vmic
        :param offset_chisqr: chi_sqr to fit
        :param bound_min_teff: minimum teff to fit
        :param bound_max_teff: maximum teff to fit
        :return: fitted teff with the offset chi_sqr
        """
        temp_directory = os.path.join(self.temp_dir, str(np.random.random()), "")

        segment_index = np.where(np.logical_and(self.seg_begins <= self.line_centers_sorted[line_number],
                                        self.line_centers_sorted[line_number] <= self.seg_ends))[0][0]

        ssg = self.create_ssg_object(self._get_marcs_models(), segment_index=segment_index)

        if not self.night_mode:
            print(self.line_centers_sorted[line_number], self.line_begins_sorted[line_number], self.line_ends_sorted[line_number])
        ssg.line_list_paths = [get_trimmed_lbl_path_name(self.line_list_path_trimmed, segment_index)]

        function_arguments = (ssg, self, self.line_begins_sorted[line_number], self.line_ends_sorted[line_number],
                              self.seg_begins[segment_index], self.seg_ends[segment_index], temp_directory, fitted_rv, fitted_vmac, fitted_rotation, fitted_vmic, offset_chisqr)
        try:
            res = root_scalar(calc_chi_sqr_teff_error, args=function_arguments, bracket=[bound_min_teff, bound_max_teff], method='brentq')
            if not self.night_mode:
                print(res)
            fitted_teff = res.root
        except IndexError as error:
            if not self.night_mode:
                print(f"{error} is line in the spectrum? {self.line_centers_sorted[line_number]}")
            fitted_teff = -999999

        shutil.rmtree(temp_directory)
        return fitted_teff

    def fit_lbl_logg(self, line_number: int) -> dict:
        """
        Fits lbl by changing logg
        :param line_number: Which line number/index in line_center_sorted is being fitted
        :return: best fit result dict for that line
        """
        temp_directory = os.path.join(self.temp_dir, str(np.random.random()), "")

        segment_index = np.where(np.logical_and(self.seg_begins <= self.line_centers_sorted[line_number],
                                        self.line_centers_sorted[line_number] <= self.seg_ends))[0][0]
        if not self.night_mode:
            print(self.line_centers_sorted[line_number], self.line_begins_sorted[line_number], self.line_ends_sorted[line_number])

        param_guess = np.array([[self.logg + self.guess_plus_minus_neg_logg], [self.logg + self.guess_plus_minus_pos_logg]])
        min_bounds = [(self.bound_min_logg, self.bound_max_logg)]

        ssg = self.create_ssg_object(self._get_marcs_models(), segment_index=segment_index)

        ssg.line_list_paths = [get_trimmed_lbl_path_name(self.line_list_path_trimmed, segment_index)]

        function_arguments = (ssg, self, self.line_begins_sorted[line_number], self.line_ends_sorted[line_number],
                              self.seg_begins[segment_index], self.seg_ends[segment_index], temp_directory, line_number)
        minimize_options = {'maxfev': self.maxfev, 'disp': self.python_verbose, 'initial_simplex': param_guess,
                            'xatol': self.xatol_logg, 'fatol': self.fatol_logg}
        try:
            res = minimize_function(calc_chi_sqr_logg, param_guess[0], function_arguments, min_bounds, 'Nelder-Mead', minimize_options)
            if not self.night_mode:
                print(res.x)

            logg = res.x[0]
            chi_squared = res.fun

            met = self.feh
            doppler_fit = self.rv_extra_fitted_dict[line_number]
            if self.vmic is not None:  # Input given
                microturb = self.vmic
            else:
                microturb = calculate_vturb(self.teff, logg, met)

            if self.fit_vmac:
                macroturb = self.vmac_fitted_dict[line_number]
            else:
                macroturb = self.vmac
            if self.fit_rotation:
                rotation = self.rotation_fitted_dict[line_number]
            else:
                rotation = self.rotation

            wave_result = self.wavelength_fitted_dict[line_number]
            flux_norm_result = self.flux_norm_fitted_dict[line_number]
            flux_result = self.flux_fitted_dict[line_number]
        except IndexError:
            if not self.night_mode:
                print(f"Line {line_number} not fitted, is your line in the spectrum?")
            logg = 999999
            doppler_fit = 999999
            microturb = 999999
            macroturb = 999999
            rotation = 999999
            chi_squared = 999999
            self.wavelength_fitted_dict[line_number] = np.array([])
            self.flux_norm_fitted_dict[line_number] = np.array([])
            self.flux_fitted_dict[line_number] = np.array([])
            wave_result = np.array([])
            flux_norm_result = np.array([])
            flux_result = np.array([])


        if np.size(wave_result) == 0 or logg >= 999998:
            if not self.night_mode:
                print(f"Failed spectra generation completely, line is not fitted at all, not saving spectra then")

        if self.find_logg_errors and logg <= 99999 and not True:
            # TODO: add ability to fit logg error
            dof = calculate_dof(self.wavelength_obs, lmin=self.line_begins_sorted[line_number], lmax=self.line_ends_sorted[line_number])

            if not self.night_mode:
                print(f"Fitting {self.logg_error_sigma} sigma at {self.line_centers_sorted[line_number]} angstroms")
            try:
                logg_error = np.abs(self.find_logg_error_one_line(line_number, doppler_fit,
                                                           macroturb, rotation,
                                                           microturb,
                                                           offset_chisqr=(chi_squared + np.square(self.teff_error_sigma) / dof),
                                                           bound_min_teff=logg,
                                                           bound_max_teff=logg + 1) - logg)
            except ValueError as err:
                if not self.night_mode:
                    print(err)
                try:
                    logg_error = np.abs(self.find_logg_error_one_line(line_number, doppler_fit,
                                                               macroturb, rotation,
                                                               microturb,
                                                               offset_chisqr=(chi_squared + np.square(self.teff_error_sigma) / dof),
                                                               bound_min_teff=logg - 1,
                                                               bound_max_teff=logg) - logg)
                except ValueError as err:
                    if not self.night_mode:
                        print(err)
                    logg_error = 1
        else:
            print("logg error not implemented yet")
            logg_error = 999999

        # Create a dictionary with column names as keys and corresponding values
        result_dict = {
            "specname": self.spec_name,
            "logg": logg,
            "logg_error": logg_error,
            "wave_center": self.line_centers_sorted[line_number],
            "wave_start": self.line_begins_sorted[line_number],
            "wave_end": self.line_ends_sorted[line_number],
            "Doppler_Shift_add_to_RV": doppler_fit,
            "Microturb": microturb,
            "Macroturb": macroturb,
            "rotation": rotation,
            "chi_squared": chi_squared
        }

        one_result = {"result": result_dict, "rv": doppler_fit, "vmic": microturb, "fit_wavelength": wave_result,
                      "fit_flux_norm": flux_norm_result, "fit_flux": flux_result, "macroturb": macroturb,
                      "rotation": rotation, "chi_sqr": chi_squared}

        shutil.rmtree(temp_directory)

        return one_result

    def _get_marcs_models(self, force_pickle_load=False) -> Tuple[None, None, None, list, dict, dict]:
        """
        I hate this function, but it's the only way to get the marcs models to the workers with the least memory usage
        I wasted probably too long to count trying to get this to work with dask distributed
        But it returns the marcs models inside this class
        Please don't judge me
        Also please rewrite this if you know how to do it better
        :param force_pickle_load: if True, then forces to load the pickle file
        :return: marcs models as a dictionary
        """
        # usually size of standard marcs models is 8 MB
        if self.dask_workers != 1 and not force_pickle_load:
            worker = get_worker()
            try:
                model_temperatures, model_logs, model_mets, marcs_value_keys, marcs_models, marcs_values = worker.model_temperatures, worker.model_logs, worker.model_mets, worker.marcs_value_keys, worker.marcs_models, worker.marcs_values
            except AttributeError:
                with open(os.path.join(self.pickled_marcs_models_location), 'rb') as pickled_marcs_models:
                    model_temperatures, model_logs, model_mets, marcs_value_keys, marcs_models, marcs_values = pickle.load(pickled_marcs_models)
                    worker.marcs_models, worker.model_temperatures, worker.model_logs, worker.model_mets, worker.marcs_value_keys, worker.marcs_values = marcs_models, model_temperatures, model_logs, model_mets, marcs_value_keys, marcs_values
        else:
            with open(os.path.join(self.pickled_marcs_models_location), 'rb') as pickled_marcs_models:
                model_temperatures, model_logs, model_mets, marcs_value_keys, marcs_models, marcs_values = pickle.load(pickled_marcs_models)
        return model_temperatures, model_logs, model_mets, marcs_value_keys, marcs_models, marcs_values

    def fit_lbl_abund_vmic(self, line_number: int) -> dict:
        """
        Fits a single line by first calling abundance calculation and inside it fitting macro + doppler shift
        It can also technically fit vmic at the same time, but it's not recommended
        :param line_number: Which line number/index in line_center_sorted is being fitted
        :return: result dictionary for that fit
        """
        temp_directory = os.path.join(self.temp_dir, str(np.random.random()), "")

        segment_index = np.where(np.logical_and(self.seg_begins <= self.line_centers_sorted[line_number],
                                        self.line_centers_sorted[line_number] <= self.seg_ends))[0][0]

        ssg = self.create_ssg_object(self._get_marcs_models(), segment_index=segment_index)

        if not self.night_mode:
            print(self.line_centers_sorted[line_number], self.line_begins_sorted[line_number], self.line_ends_sorted[line_number])
        ssg.line_list_paths = [get_trimmed_lbl_path_name(self.line_list_path_trimmed, segment_index)]

        param_guess, min_bounds = self.get_elem_micro_guess(self.guess_min_vmic, self.guess_max_vmic, self.guess_min_abund, self.guess_max_abund)

        function_arguments = (ssg, self, self.line_begins_sorted[line_number], self.line_ends_sorted[line_number],  self.seg_begins[segment_index], self.seg_ends[segment_index], temp_directory, line_number)
        minimization_options = {'maxfev': self.nelement * self.maxfev, 'disp': self.python_verbose, 'initial_simplex': param_guess, 'xatol': self.xatol_lbl, 'fatol': self.fatol_lbl, 'adaptive': True}
        try:
            res = minimize_function(calc_chi_sqr_abund_and_vmic, param_guess[0], function_arguments, min_bounds, 'Nelder-Mead', minimization_options)
            print_result = "Converged:"
            for elem, value in zip(self.elem_to_fit, res.x):
                # here element is [X/Fe], unless it's Fe, then it's [Fe/H]
                print_result += f" {elem}: {value:.2f}"

            print_result += f" Number of iterations: {res.nit}"
            fit_iterations = res.nit
            if not self.night_mode:
                print(print_result)
            if self.fit_feh:
                met_index = np.where(self.elem_to_fit == "Fe")[0][0]
                feh = res.x[met_index]
            else:
                feh = self.feh
            elem_abund_dict = {"Fe": feh}
            for i in range(self.nelement):
                # self.elem_to_fit[i] = element name
                # param[1:nelement] = abundance of the element
                elem_name = self.elem_to_fit[i]
                if elem_name != "Fe":
                    # here element is [X/Fe], unless it's Fe, then it's [Fe/H]
                    elem_abund_dict[elem_name] = res.x[i]
            doppler_fit = self.rv_extra_fitted_dict[line_number]
            if self.atmosphere_type == "3D":
                vmic = 2.0
            else:
                if self.input_vmic:  # Input given
                    vmic = self.vmic
                elif not self.fit_vmic:
                    vmic = calculate_vturb(self.teff, self.logg, feh)
                elif self.fit_vmic:
                    vmic = res.x[-1]  # last param is vmic
                else:
                    raise ValueError("vmic is not set, input_vmic, fit_vmic and vmic are all False")
            if self.fit_vmac:
                vmac = self.vmac_fitted_dict[line_number]
            else:
                vmac = self.vmac
            if self.fit_rotation:
                rotation = self.rotation_fitted_dict[line_number]
            else:
                rotation = self.rotation
            chi_squared = res.fun

            wave_result = self.wavelength_fitted_dict[line_number]
            flux_norm_result = self.flux_norm_fitted_dict[line_number]
            flux_result = self.flux_fitted_dict[line_number]
        except IndexError as error:
            if not self.night_mode:
                print(f"{error} is line in the spectrum? {self.line_centers_sorted[line_number]}")
            elem_abund_dict = {"Fe": 999999}
            for i in range(self.nelement):
                # self.elem_to_fit[i] = element name
                # param[1:nelement] = abundance of the element
                elem_name = self.elem_to_fit[i]
                if elem_name != "Fe":
                    elem_abund_dict[elem_name] = 999999
            doppler_fit = 999999
            vmic = 999999
            vmac = 999999
            rotation = 999999
            chi_squared = 999999
            fit_iterations = 0
            self.wavelength_fitted_dict[line_number] = np.array([])
            self.flux_norm_fitted_dict[line_number] = np.array([])
            self.flux_fitted_dict[line_number] = np.array([])
            wave_result = np.array([])
            flux_norm_result = np.array([])
            flux_result = np.array([])
            # Create a dictionary with column names as keys and corresponding values
        result_dict = {
            "specname": self.spec_name,
            "wave_center": self.line_centers_sorted[line_number],
            "wave_start": self.line_begins_sorted[line_number],
            "wave_end": self.line_ends_sorted[line_number],
            "Doppler_Shift_add_to_RV": doppler_fit,
        }

        # Add elemental abundances to the dictionary
        for key in elem_abund_dict:
            result_dict[key] = elem_abund_dict[key]

        result_dict["Microturb"] = vmic
        result_dict["Macroturb"] = vmac
        result_dict["rotation"] = rotation
        result_dict["chi_squared"] = chi_squared


        if np.size(wave_result) == 0:
            if not self.night_mode:
                print(f"Failed spectra generation completely, line is not fitted at all, not saving spectra then")

        shutil.rmtree(temp_directory)
        return {"result": result_dict, "rv": doppler_fit, "vmic": vmic, "fit_wavelength": wave_result,
                "fit_flux_norm": flux_norm_result, "fit_flux": flux_result,  "macroturb": vmac,
                "rotation": rotation, "chi_sqr": chi_squared, "fitted_abund": elem_abund_dict[self.elem_to_fit[0]],
                "fit_iterations": fit_iterations}

    def fit_lbl_vmic(self, line_number: int) -> dict:
        """
        Fits a single line by first calling abundance calculation and inside it fitting macro + doppler shift
        :param line_number: Which line number/index in line_center_sorted is being fitted
        :return: best fit result string for that line
        """
        temp_directory = os.path.join(self.temp_dir, str(np.random.random()), "")

        segment_index = np.where(np.logical_and(self.seg_begins <= self.line_centers_sorted[line_number],
                                        self.line_centers_sorted[line_number] <= self.seg_ends))[0][0]

        ssg = self.create_ssg_object(self._get_marcs_models(), segment_index=segment_index)

        if not self.night_mode:
            print(self.line_centers_sorted[line_number], self.seg_begins[segment_index], self.seg_ends[segment_index])
        ssg.line_list_paths = [get_trimmed_lbl_path_name(self.line_list_path_trimmed, segment_index)]

        param_guess, min_bounds = self.get_elem_guess(self.guess_min_abund, self.guess_max_abund)

        self.wavelength_fitted_dict[line_number] = np.array([])
        self.flux_norm_fitted_dict[line_number] = np.array([])
        self.flux_fitted_dict[line_number] = np.array([])

        function_arguments = (ssg, self, self.line_begins_sorted[line_number], self.line_ends_sorted[line_number], self.seg_begins[segment_index], self.seg_ends[segment_index], temp_directory, line_number, self.maxfev, self.xatol_lbl, self.fatol_lbl)
        minimization_options = {'maxfev': self.nelement * self.maxfev, 'disp': self.python_verbose, 'initial_simplex': param_guess, 'xatol': self.xatol_vmic, 'fatol': self.fatol_vmic, 'adaptive': False}
        res = minimize_function(calc_chi_sqr_abund_with_vmic_inside, param_guess[0], function_arguments, min_bounds, 'Nelder-Mead', minimization_options)
        if not self.night_mode:
            print(res.x)
        if self.fit_feh:
            met_index = np.where(self.elem_to_fit == "Fe")[0][0]
            met = res.x[met_index]
        else:
            met = self.feh
        elem_abund_dict = {"Fe": met}
        for i in range(self.nelement):
            # self.elem_to_fit[i] = element name
            # param[1:nelement] = abundance of the element
            elem_name = self.elem_to_fit[i]
            if elem_name != "Fe":
                elem_abund_dict[elem_name] = res.x[i]  # + met
        doppler_fit = self.rv_extra_fitted_dict[line_number]
        microturb = self.vmic_fitted_dict[line_number]
        if self.fit_vmac:
            macroturb = self.vmac_fitted_dict[line_number]
        else:
            macroturb = self.vmac
        if self.fit_rotation:
            rotation = self.rotation_fitted_dict[line_number]
        else:
            rotation = self.rotation
        # Create a dictionary with column names as keys and corresponding values
        result_dict = {
            "specname": self.spec_name,
            "wave_center": self.line_centers_sorted[line_number],
            "wave_start": self.line_begins_sorted[line_number],
            "wave_end": self.line_ends_sorted[line_number],
            "Doppler_Shift_add_to_RV": doppler_fit,
        }

        # Add elemental abundances to the dictionary
        for key in elem_abund_dict:
            result_dict[key] = elem_abund_dict[key]

        result_dict["Microturb"] = microturb
        result_dict["Macroturb"] = macroturb
        result_dict["rotation"] = rotation
        result_dict["chi_squared"] = res.fun

        one_result = result_dict

        wave_result = self.wavelength_fitted_dict[line_number]
        flux_norm_result = self.flux_norm_fitted_dict[line_number]
        flux_result = self.flux_fitted_dict[line_number]

        if np.size(wave_result) == 0:
            if not self.night_mode:
                print(f"Failed spectra generation completely, line is not fitted at all, not saving spectra then")

        shutil.rmtree(temp_directory)
        return {"result": one_result, "fit_wavelength": wave_result, "fit_flux_norm": flux_norm_result,
                "fit_flux": flux_result,  "macroturb": macroturb, "rotation": rotation, "chi_sqr": res.fun, "rv": doppler_fit}


def check_if_spectra_generated(wavelength_fitted: np.ndarray, night_mode=False) -> Tuple[bool, float]:
    """
    Checks if the spectra was generated correctly, depending if it's None or empty
    :param wavelength_fitted:  Wavelength of synthetic spectra
    :param night_mode: if True, then doesn't print anything
    :return: True and 0 if input array is not None or empty, False and 999999 otherwise
    """
    if wavelength_fitted is None:
        if not night_mode:
            print("didn't generate spectra or atmosphere")
        return False, 999999.9999
    elif np.size(wavelength_fitted) == 0:
        if not night_mode:
            print("empty spectrum file.")
        return False, 999999.99
    else:
        return True, 0


def print_intermediate_results(intermediate_results: dict, atmosphere_type: str, night_mode: bool) -> None:
    """
    Prints intermediate results for the line
    :param intermediate_results: Dictionary with the results for the line
    :param atmosphere_type: 1D or 3D, depending on it vmic is printed or not
    :param night_mode: if True, then doesn't print anything
    :return: None
    """
    vmic_possible_columns = ["vmic", "microturb"]
    chi_sqr_possible_columns = ["chi_squared", "chi_sqr", "chisqr", "chisquared", "fitted_chisqr"]
    abundances_possible_columns = ["abundances", "abund", "abundance", "abundances", "elem_abund_dict", "abund_dict", "elem"]
    if not night_mode:
        # we want to print it, by going through each key. each key is a column name
        # 4 for everything with 7 characters total and 8 decimals for chi squared
        string_to_print = ""
        for key in intermediate_results:
            if key in chi_sqr_possible_columns:
                string_to_print += f"chi_sqr= {intermediate_results[key]:>14.8f} "
            else:
                if key.lower() in vmic_possible_columns and atmosphere_type == "3D":
                    # don't print vmic for 3D
                    pass
                else:
                    if key in abundances_possible_columns:
                        for element in intermediate_results[key]:
                            string_to_print += f"[{element}/H]= {intermediate_results[key][element]:>7.4f} "
                    else:
                        string_to_print += f"{key}= {intermediate_results[key]:>.4f} "
        print(string_to_print)


def calculate_dof(wavelength: np.ndarray, lmin: float=None, lmax: float=None, parameters_count: int=1) -> int:
    """
    Calculates the degrees of freedom for the chi squared calculation
    :param wavelength: Wavelength of spectra
    :param lmin: Start of the line [AA] where to calculate chi squared
    :param lmax: End of the line [AA] where to calculate chi squared
    :param parameters_count: Number of parameters to fit, default is 1
    :return: Degrees of freedom
    """
    if lmin is not None and lmax is not None:
        wavelength = wavelength[np.logical_and(lmin <= wavelength, wavelength <= lmax)]
    dof = np.size(wavelength) - parameters_count
    if dof <= 0:
        dof = 1
    return dof


def get_input_xh_abund(feh: float, spectra_to_fit: Spectra) -> dict:
    elem_abund_dict_xh = {"Fe": feh}
    # first it takes the input abundances and then adds the fit abundances to it, so priority is given to fitted abundances
    for element in spectra_to_fit.input_abund:
        if element != "Fe":
            # add input abundances to dict [X/H]
            elem_abund_dict_xh[element] = spectra_to_fit.input_abund[element] + feh
    return elem_abund_dict_xh

def calc_chi_sq_broadening(param: list, spectra_to_fit: Spectra, lmin: float, lmax: float,
                           wavelength_fitted: np.ndarray, flux_norm_fitted: np.ndarray) -> float:
    """
    Calculates the chi squared for the given broadening parameters and small doppler shift
    :param param: Parameters list with the current evaluation guess
    :param spectra_to_fit: Spectra to fit
    :param lmin: Start of the line [AA] where to calculate chi squared
    :param lmax: End of the line [AA] where to calculate chi squared
    :param wavelength_fitted: Wavelength of synthetic spectra
    :param flux_norm_fitted: Flux of synthetic spectra
    :return: Best fit chi squared
    """
    # param[0] = doppler
    # param[1] = vmac
    # param[-1] = rotation fit

    doppler = spectra_to_fit.stellar_rv + param[0]

    if spectra_to_fit.fit_vmac:
        macroturb = param[1]
    else:
        macroturb = spectra_to_fit.vmac

    if spectra_to_fit.fit_rotation:
        rotation = param[-1]
    else:
        rotation = spectra_to_fit.rotation

    wavelength_obs_rv = apply_doppler_correction(spectra_to_fit.wavelength_obs, doppler)

    chi_square = calculate_lbl_chi_squared(wavelength_obs_rv, spectra_to_fit.flux_norm_obs,
                                           spectra_to_fit.error_obs_variance, wavelength_fitted, flux_norm_fitted,
                                           spectra_to_fit.resolution, lmin, lmax, macroturb, rotation)

    return chi_square


def calc_chi_sqr_generic_lbl(feh: float, elem_abund_dict_xh: dict, vmic: float, teff: float, logg: float,
                             ssg: SyntheticSpectrumGenerator, spectra_to_fit: Spectra, lmin: float, lmax: float,
                             lmin_segment: float, lmax_segment: float, temp_directory: str, line_number: int) -> float:
    """
    Calculates the chi squared for the given broadening parameters and small doppler shift for the generic lbl
    :param feh: Fe/H
    :param elem_abund_dict_xh: Abundances of the elements
    :param vmic: Microturbulence
    :param teff: Effective temperature
    :param logg: Log g
    :param ssg: Synthetic spectrum generator object
    :param spectra_to_fit: Spectra to fit
    :param lmin: Start of the line [AA], where to calculate chi squared
    :param lmax: End of the line [AA], where to calculate chi squared
    :param lmin_segment: Start of the segment, where spectra is generated [AA]
    :param lmax_segment: End of the segment, where spectra is generated [AA]
    :param temp_directory: Temporary directory where code is being run
    :param line_number: Which line number/index in line_center_sorted is being fitted
    :return: best fit chi squared
    """
    macroturb = 999999    # for printing only here, in case not fitted
    rotation = 999999
    rv = 999999
    # generates spectra
    wavelength_fitted, flux_norm_fitted, flux_fitted = spectra_to_fit.configure_and_run_synthetic_code(ssg, feh, elem_abund_dict_xh, vmic,
                                                                                                       lmin_segment, lmax_segment, False,
                                                                                                       temp_dir=temp_directory, teff=teff, logg=logg)

    spectra_generated, chi_squared = check_if_spectra_generated(wavelength_fitted, spectra_to_fit.night_mode)
    if spectra_generated:
        param_guess, min_bounds = spectra_to_fit.get_rv_macro_rotation_guess(min_macroturb=spectra_to_fit.guess_min_vmac, max_macroturb=spectra_to_fit.guess_max_vmac)
        # now for the generated abundance it tries to fit best fit macro + doppler shift.
        # Thus, macro should not be dependent on the abundance directly, hopefully
        # Seems to work way better
        function_args = (spectra_to_fit, lmin, lmax, wavelength_fitted, flux_norm_fitted)
        minimize_options = {'maxiter': spectra_to_fit.ndimen * 50, 'disp': False}
        res = minimize_function(calc_chi_sq_broadening, np.median(param_guess, axis=0),
                                function_args, min_bounds, 'L-BFGS-B', minimize_options)

        spectra_to_fit.rv_extra_fitted_dict[line_number] = res.x[0]
        rv = spectra_to_fit.rv_extra_fitted_dict[line_number]
        if spectra_to_fit.fit_vmac:
            spectra_to_fit.vmac_fitted_dict[line_number] = res.x[1]
            macroturb = spectra_to_fit.vmac_fitted_dict[line_number]
        else:
            macroturb = spectra_to_fit.vmac
        if spectra_to_fit.fit_rotation:
            spectra_to_fit.rotation_fitted_dict[line_number] = res.x[-1]
            rotation = spectra_to_fit.rotation_fitted_dict[line_number]
        else:
            rotation = spectra_to_fit.rotation
        chi_squared = res.fun

        spectra_to_fit.wavelength_fitted_dict[line_number] = wavelength_fitted
        spectra_to_fit.flux_norm_fitted_dict[line_number] = flux_norm_fitted
        spectra_to_fit.flux_fitted_dict[line_number] = flux_fitted
    else:
        spectra_to_fit.wavelength_fitted_dict[line_number] = np.array([])
        spectra_to_fit.flux_norm_fitted_dict[line_number] = np.array([])
        spectra_to_fit.flux_fitted_dict[line_number] = np.array([])
        spectra_to_fit.rv_extra_fitted_dict[line_number] = rv
        spectra_to_fit.vmac_fitted_dict[line_number] = macroturb
        spectra_to_fit.rotation_fitted_dict[line_number] = rotation

    intermediate_results = {"abundances": elem_abund_dict_xh, "teff": teff, "logg": logg, "rv": rv, "vmic": vmic,
                            "vmac": macroturb, "rotation": rotation, "chi_sqr": chi_squared}

    print_intermediate_results(intermediate_results, spectra_to_fit.atmosphere_type, spectra_to_fit.night_mode)

    return chi_squared


def calc_chi_sqr_abund_and_vmic(param: list, ssg: SyntheticSpectrumGenerator, spectra_to_fit: Spectra, lmin: float,
                                lmax: float, lmin_segment: float, lmax_segment: float, temp_directory: str,
                                line_number: int) -> float:
    """
    Goes line by line, tries to call turbospectrum and find best fit spectra by varying parameters: abundance, and if
    needed micro. It also finds best broadening and rv by calling another minimisation inside
    :param param: Parameters list with the current evaluation guess
    :param ssg: Synthetic spectrum generator object
    :param spectra_to_fit: Spectra to fit
    :param lmin: Start of the line [AA], where to calculate chi squared
    :param lmax: End of the line [AA], where to calculate chi squared
    :param lmin_segment: Start of the segment, where spectra is generated [AA]
    :param lmax_segment: End of the segment, where spectra is generated [AA]
    :param temp_directory: Temporary directory where code is being run
    :param line_number: Which line number/index in line_center_sorted is being fitted
    :return: best fit chi squared
    """
    # new: now includes several elements
    # param[-1] = vmicro
    # param[0:nelements - 1] = feh or abund
    teff = spectra_to_fit.teff
    logg = spectra_to_fit.logg

    if spectra_to_fit.fit_feh:
        met_index = np.where(spectra_to_fit.elem_to_fit == "Fe")[0][0]
        feh = param[met_index]  # no offset, first is always element
    else:
        feh = spectra_to_fit.feh

    elem_abund_dict_xh = get_input_xh_abund(feh, spectra_to_fit)

    for i in range(spectra_to_fit.nelement):
        # spectra_to_fit.elem_to_fit[i] = element name
        # param[0:nelement - 1] = abundance of the element
        elem_name = spectra_to_fit.elem_to_fit[i]
        if elem_name != "Fe":
            elem_abund_dict_xh[elem_name] = param[i] + feh     # convert [X/Fe] to [X/H]

    if spectra_to_fit.atmosphere_type == "3D":
        vmic = 2.0
    else:
        if spectra_to_fit.input_vmic:  # Input given
            vmic = spectra_to_fit.vmic
        elif not spectra_to_fit.fit_vmic:
            vmic = calculate_vturb(spectra_to_fit.teff, spectra_to_fit.logg, feh)
        elif spectra_to_fit.fit_vmic:
            vmic = param[-1]  # last param is vmic
        else:
            raise ValueError("vmic is not set, input_vmic, fit_vmic and vmic are all False")

    return calc_chi_sqr_generic_lbl(feh, elem_abund_dict_xh, vmic, teff, logg, ssg, spectra_to_fit, lmin, lmax,
                                    lmin_segment, lmax_segment, temp_directory, line_number)


def calc_chi_sqr_error_generic(feh: float, elem_abund_dict_xh: dict, vmic: float, teff: float, logg: float,
                               ssg: SyntheticSpectrumGenerator, spectra_to_fit: Spectra, lmin: float,
                               lmax: float, lmin_segment: float, lmax_segment: float, temp_directory: str, rv: float,
                               vmac: float, rotation: float, offset_chisqr: float) -> float:
    """
    Goes line by line, tries to call turbospectrum and find best fit spectra by varying parameters: abundance, doppler
    shift and if needed micro + macro turbulence. This specific function handles abundance + micro. Calls macro +
    doppker inside
    :param feh: Fe/H
    :param elem_abund_dict_xh: Abundances of the elements
    :param vmic: Microturbulence
    :param teff: Effective temperature
    :param logg: Log g
    :param ssg: Synthetic spectrum generator object
    :param spectra_to_fit: Spectra to fit
    :param lmin: Start of the line [AA] where to calculate chi squared
    :param lmax: End of the line [AA] where to calculate chi squared
    :param lmin_segment: Start of the segment, where spectra is generated [AA]
    :param lmax_segment: End of the segment, where spectra is generated [AA]
    :param temp_directory: Temporary directory where code is being run
    :param rv: Doppler shift, fitted to add to the stellar rv
    :param vmac: Fitted macroturbulence
    :param rotation: Fitted rotation
    :param offset_chisqr: Offset chi squared, to subtract from the chi squared. This is the fitted chi squared + sigma^2 / dof
    :return: best fit chi squared
    """

    wavelength_fitted, flux_norm_fitted, _ = spectra_to_fit.configure_and_run_synthetic_code(ssg, feh, elem_abund_dict_xh, vmic,
                                                                                             lmin_segment, lmax_segment, False,
                                                                                             temp_dir=temp_directory, teff=teff, logg=logg)

    spectra_generated, chi_square = check_if_spectra_generated(wavelength_fitted, spectra_to_fit.night_mode)
    if spectra_generated:
        wave_obs_shifted = apply_doppler_correction(spectra_to_fit.wavelength_obs, rv + spectra_to_fit.stellar_rv)
        chi_square = calculate_lbl_chi_squared(wave_obs_shifted, spectra_to_fit.flux_norm_obs, spectra_to_fit.error_obs_variance, wavelength_fitted, flux_norm_fitted, spectra_to_fit.resolution, lmin, lmax, vmac, rotation)

    intermediate_results = {"abundances": elem_abund_dict_xh, "teff": teff, "logg": logg, "rv": rv, "vmic": vmic,
                            "vmac": vmac, "rotation": rotation, "offset": offset_chisqr, "fitted_chisqr": chi_square,
                            "chi_sqr": chi_square - offset_chisqr}

    print_intermediate_results(intermediate_results, spectra_to_fit.atmosphere_type, spectra_to_fit.night_mode)

    return chi_square - offset_chisqr

def calc_chi_sqr_teff_error(param: float, ssg: SyntheticSpectrumGenerator, spectra_to_fit: Spectra, lmin: float,
                            lmax: float, lmin_segment: float, lmax_segment: float, temp_directory: str, rv: float,
                            vmac: float, rotation: float, vmic: float, offset_chisqr: float) -> float:
    """
    Calculates the chi squared subtracted with offset one for the given broadening parameters and small doppler shift.
    Varies teff only.
    :param param: Parameters list with the current evaluation guess
    :param ssg: Synthetic spectrum generator object
    :param spectra_to_fit: Spectra to fit
    :param lmin: Start of the line [AA] where to calculate chi squared
    :param lmax: End of the line [AA] where to calculate chi squared
    :param lmin_segment: Start of the segment, where spectra is generated [AA]
    :param lmax_segment: End of the segment, where spectra is generated [AA]
    :param temp_directory: Temporary directory where code is being run
    :param rv: Doppler shift, fitted to add to the stellar rv
    :param vmac: Fitted macroturbulence
    :param rotation: Fitted rotation
    :param vmic: Fitted microturbulence
    :param offset_chisqr: Offset chi squared, to subtract from the chi squared. This is the fitted chi squared + sigma^2 / dof
    :return: best fit chi squared
    """
    # param = teff

    teff = param
    logg = spectra_to_fit.logg
    feh = spectra_to_fit.feh
    elem_abund_dict_xh = get_input_xh_abund(feh, spectra_to_fit)

    return calc_chi_sqr_error_generic(feh, elem_abund_dict_xh, vmic, teff, logg, ssg, spectra_to_fit, lmin, lmax,
                                      lmin_segment, lmax_segment, temp_directory, rv, vmac, rotation, offset_chisqr)


def calc_chi_sqr_abund_error(param: list, ssg: SyntheticSpectrumGenerator, spectra_to_fit: Spectra, lmin: float,
                             lmax: float, lmin_segment: float, lmax_segment: float, temp_directory: str, rv: float,
                             vmac: float, rotation: float, vmic: float, offset_chisqr: float) -> float:
    """
    Goes line by line, tries to call turbospectrum and find best fit spectra by varying parameters: abundance, doppler
    shift and if needed micro + macro turbulence. This specific function handles abundance + micro. Calls macro +
    doppker inside
    :param param: Parameters list with the current evaluation guess
    :param ssg: Synthetic spectrum generator object
    :param spectra_to_fit: Spectra to fit
    :param lmin: Start of the line [AA] where to calculate chi squared
    :param lmax: End of the line [AA] where to calculate chi squared
    :param lmin_segment: Start of the segment, where spectra is generated [AA]
    :param lmax_segment: End of the segment, where spectra is generated [AA]
    :param temp_directory: Temporary directory where code is being run
    :param rv: Doppler shift, fitted to add to the stellar rv
    :param vmac: Fitted macroturbulence
    :param rotation: Fitted rotation
    :param vmic: Fitted microturbulence
    :param offset_chisqr: Offset chi squared, to subtract from the chi squared. This is the fitted chi squared + sigma^2 / dof
    :return: best fit chi squared
    """
    # new: now includes several elements
    # param[0:nelements - 1] = feh or abund

    teff = spectra_to_fit.teff
    logg = spectra_to_fit.logg

    if spectra_to_fit.fit_feh:
        feh = param
    else:
        feh = spectra_to_fit.feh

    elem_abund_dict_xh = get_input_xh_abund(feh, spectra_to_fit)

    for i in range(spectra_to_fit.nelement):
        # spectra_to_fit.elem_to_fit[i] = element name
        # param[0:nelement - 1] = abundance of the element
        elem_name = spectra_to_fit.elem_to_fit[i]
        if elem_name != "Fe":
            elem_abund_dict_xh[elem_name] = param + feh     # convert [X/Fe] to [X/H]

    return calc_chi_sqr_error_generic(feh, elem_abund_dict_xh, vmic, teff, logg, ssg, spectra_to_fit, lmin, lmax,
                                      lmin_segment, lmax_segment, temp_directory, rv, vmac, rotation, offset_chisqr)

def calc_chi_sqr_abund_with_vmic_inside(param: list, ssg: SyntheticSpectrumGenerator, spectra_to_fit: Spectra,
                                        lmin: float, lmax: float, lmin_segment: float, lmax_segment: float,
                                        temp_directory: str, line_number: int, maxfev_vmic: int, xatol_vmic: float,
                                        fatol_vmic: float) -> float:
    """
    Calculates the chi squared for the given broadening parameters and small doppler shift. Varies abundance and vmic.
    Here it changes the abundance, but for each abundance it calls another minimisation to find best fit vmic
    :param param: Parameters list with the current evaluation guess
    :param ssg: Synthetic spectrum generator object
    :param spectra_to_fit: Spectra to fit
    :param lmin: Start of the line [AA], where to calculate chi squared
    :param lmax: End of the line [AA], where to calculate chi squared
    :param lmin_segment: Start of the segment, where spectra is generated [AA]
    :param lmax_segment: End of the segment, where spectra is generated [AA]
    :param temp_directory: Temporary directory where code is being run
    :param line_number: Which line number/index in line_center_sorted is being fitted
    :param maxfev_vmic: Maximum number of iterations for vmic minimisation
    :param xatol_vmic: Absolute error in vmic for convergence
    :param fatol_vmic: Absolute error in chi squared for convergence
    :return: best fit chi squared
    """
    # param[0:nelements - 1] = feh or abund

    if spectra_to_fit.fit_feh:
        met_index = np.where(spectra_to_fit.elem_to_fit == "Fe")[0][0]
        feh = param[met_index]  # no offset, first is always element
    else:
        feh = spectra_to_fit.feh

    elem_abund_dict = get_input_xh_abund(feh, spectra_to_fit)

    for i in range(spectra_to_fit.nelement):
        # spectra_to_fit.elem_to_fit[i] = element name
        # param[0:nelement - 1] = abundance of the element
        elem_name = spectra_to_fit.elem_to_fit[i]
        if elem_name != "Fe":
            elem_abund_dict[elem_name] = param[i] + feh  # convert [X/Fe] to [X/H]

    spectra_to_fit.elem_abund_fitted_dict[line_number] = elem_abund_dict

    param_guess, min_bounds = spectra_to_fit.get_vmic_guess(spectra_to_fit.guess_min_vmic, spectra_to_fit.guess_max_vmic)
    function_arguments = (ssg, spectra_to_fit, lmin, lmax, lmin_segment, lmax_segment, temp_directory, line_number)
    minimization_options = {'maxfev': maxfev_vmic, 'disp': spectra_to_fit.python_verbose, 'initial_simplex': param_guess, 'xatol': xatol_vmic, 'fatol': fatol_vmic, 'adaptive': False}
    res = minimize_function(calc_chi_sqr_vmic, param_guess[0], function_arguments, min_bounds, 'Nelder-Mead', minimization_options)

    spectra_to_fit.vmic_fitted_dict[line_number] = res.x[0]
    vmic = spectra_to_fit.vmic_fitted_dict[line_number]
    doppler_shift = spectra_to_fit.rv_extra_fitted_dict[line_number]
    if spectra_to_fit.fit_vmac:
        macroturb = spectra_to_fit.vmac_fitted_dict[line_number]
    else:
        macroturb = spectra_to_fit.vmac
    if spectra_to_fit.fit_rotation:
        rotation = spectra_to_fit.rotation_fitted_dict[line_number]
    else:
        rotation = spectra_to_fit.rotation
    chi_square = res.fun

    intermediate_results = {"abundances": elem_abund_dict, "rv": doppler_shift, "vmic": vmic, "vmac": macroturb,
                            "rotation": rotation, "chisqr": chi_square}

    print_intermediate_results(intermediate_results, spectra_to_fit.atmosphere_type, spectra_to_fit.night_mode)

    return chi_square

def calc_chi_sqr_vmic(param: list, ssg: SyntheticSpectrumGenerator, spectra_to_fit: Spectra, lmin: float, lmax: float,
                      lmin_segment: float, lmax_segment: float, temp_directory: str, line_number: int) -> float:
    """
    Calculates the chi squared by varying vmic and fitting broadening inside
    :param param: Parameters list with the current evaluation guess
    :param ssg: Synthetic spectrum generator object
    :param spectra_to_fit: Spectra to fit
    :param lmin: Start of the line [AA], where to calculate chi squared
    :param lmax: End of the line [AA], where to calculate chi squared
    :param lmin_segment: Start of the segment, where spectra is generated [AA]
    :param lmax_segment: End of the segment, where spectra is generated [AA]
    :param temp_directory: Temporary directory where code is being run
    :param line_number: Which line number/index in line_center_sorted is being fitted
    :return: best fit chi squared
    """
    # param[0] = vmic

    vmic = param[0]

    feh = spectra_to_fit.elem_abund_fitted_dict[line_number]["Fe"]
    elem_abund_dict_xh = spectra_to_fit.elem_abund_fitted_dict[line_number]
    teff = spectra_to_fit.teff
    logg = spectra_to_fit.logg

    return calc_chi_sqr_generic_lbl(feh, elem_abund_dict_xh, vmic, teff, logg, ssg, spectra_to_fit, lmin, lmax,
                                    lmin_segment, lmax_segment, temp_directory, line_number)

def calc_chi_sqr_teff(param: list, ssg: SyntheticSpectrumGenerator, spectra_to_fit: Spectra, lmin: float, lmax: float,
                      lmin_segment: float, lmax_segment: float, temp_directory: str, line_number: int) -> float:
    """
    Calculates the chi squared by varying teff and fitting broadening inside
    :param param: Parameters list with the current evaluation guess
    :param ssg: Synthetic spectrum generator object
    :param spectra_to_fit: Spectra to fit
    :param lmin: Start of the line [AA], where to calculate chi squared
    :param lmax: End of the line [AA], where to calculate chi squared
    :param lmin_segment: Start of the segment, where spectra is generated [AA]
    :param lmax_segment: End of the segment, where spectra is generated [AA]
    :param temp_directory: Temporary directory where code is being run
    :param line_number: Which line number/index in line_center_sorted is being fitted
    :return: best fit chi squared
    """
    # param[0] = teff

    teff = param[0]
    logg = spectra_to_fit.logg

    if spectra_to_fit.atmosphere_type == "3D":
        vmic = 2.0
    else:
        if spectra_to_fit.input_vmic:  # Input given
            vmic = spectra_to_fit.vmic
        elif not spectra_to_fit.fit_vmic:
            vmic = calculate_vturb(teff, spectra_to_fit.logg, spectra_to_fit.feh)
        else:
            raise ValueError("vmic is not set, input_vmic, fit_vmic and vmic are all False")

    elem_abund_dict_xh = get_input_xh_abund(spectra_to_fit.feh, spectra_to_fit)

    return calc_chi_sqr_generic_lbl(spectra_to_fit.feh, elem_abund_dict_xh, vmic, teff, logg, ssg, spectra_to_fit,
                                    lmin, lmax, lmin_segment, lmax_segment, temp_directory, line_number)


def calc_chi_sqr_logg(param: list, ssg: SyntheticSpectrumGenerator, spectra_to_fit: Spectra, lmin: float, lmax: float,
                      lmin_segment: float, lmax_segment: float, temp_directory: str, line_number: int) -> float:
    """
    Calculates the chi squared by varying logg and fitting broadening inside
    :param param: Parameters list with the current evaluation guess
    :param ssg: Synthetic spectrum generator object
    :param spectra_to_fit: Spectra to fit
    :param lmin: Start of the line [AA], where to calculate chi squared
    :param lmax: End of the line [AA], where to calculate chi squared
    :param lmin_segment: Start of the segment, where spectra is generated [AA]
    :param lmax_segment: End of the segment, where spectra is generated [AA]
    :param temp_directory: Temporary directory where code is being run
    :param line_number: Which line number/index in line_center_sorted is being fitted
    :return: best fit chi squared
    """
    # param[0] = logg

    teff = spectra_to_fit.teff
    logg = param[0]

    if spectra_to_fit.atmosphere_type == "3D":
        vmic = 2.0
    else:
        if spectra_to_fit.input_vmic:  # Input given
            vmic = spectra_to_fit.vmic
        elif not spectra_to_fit.fit_vmic:
            vmic = calculate_vturb(spectra_to_fit.teff, logg, spectra_to_fit.feh)
        else:
            raise ValueError("vmic is not set, input_vmic, fit_vmic and vmic are all False")

    elem_abund_dict_xh = get_input_xh_abund(spectra_to_fit.feh, spectra_to_fit)

    return calc_chi_sqr_generic_lbl(spectra_to_fit.feh, elem_abund_dict_xh, vmic, teff, logg, ssg, spectra_to_fit,
                                    lmin, lmax, lmin_segment, lmax_segment, temp_directory, line_number)


def calc_chi_sqr_all_mode(param, ssg: SyntheticSpectrumGenerator, spectra_to_fit: Spectra) -> float:
    """
    Calculates best fit parameters for all lines at once. Changes RV, vmac and abundance
    :param param: Parameter guess
    :param ssg: Synthetic spectrum generator object
    :param spectra_to_fit: Spectra to fit
    :return: Best fit chi squared
    """
    # abund = param[0]
    # dopple = param[1]
    # macroturb = param[2] (if needed)
    abund = param[0]
    doppler = spectra_to_fit.stellar_rv + param[1]
    if spectra_to_fit.fit_vmac:
        macroturb = param[2]
    else:
        macroturb = spectra_to_fit.vmac

    wave_obs = apply_doppler_correction(spectra_to_fit.wavelength_obs, doppler)

    if spectra_to_fit.fit_feh:
        feh = abund
    else:
        feh = spectra_to_fit.feh

    elem_abund_dict = get_input_xh_abund(feh, spectra_to_fit)

    if not spectra_to_fit.fit_feh:
        elem_abund_dict[spectra_to_fit.elem_to_fit[0]] = abund + feh

    if spectra_to_fit.atmosphere_type == "3D":
        vmic = 2.0
    else:
        if spectra_to_fit.input_vmic:  # Input given
            vmic = spectra_to_fit.vmic
        elif not spectra_to_fit.fit_vmic:
            vmic = calculate_vturb(spectra_to_fit.teff, spectra_to_fit.logg, feh)
        else:
            raise ValueError("vmic is not set, input_vmic, fit_vmic and vmic are all False")

    wavelength_fitted, flux_norm_fitted, _ = spectra_to_fit.configure_and_run_synthetic_code(ssg, feh, elem_abund_dict, vmic, spectra_to_fit.lmin, spectra_to_fit.lmax, True)

    spectra_generated, chi_square = check_if_spectra_generated(wavelength_fitted, spectra_to_fit.night_mode)
    if spectra_generated:
        chi_square = calc_ts_spectra_all_lines(spectra_to_fit.spec_name, wavelength_fitted, flux_norm_fitted, spectra_to_fit.temp_dir,
                                               spectra_to_fit.output_folder,
                                               wave_obs, spectra_to_fit.flux_norm_obs,
                                               macroturb, spectra_to_fit.resolution, spectra_to_fit.rotation,
                                               spectra_to_fit.line_begins_sorted, spectra_to_fit.line_ends_sorted,
                                               spectra_to_fit.seg_begins, spectra_to_fit.seg_ends)

    intermediate_results = {"abundances": elem_abund_dict, "rv": doppler, "vmic": vmic, "vmac": macroturb,
                            "rotation": spectra_to_fit.rotation, "chisqr": chi_square}

    print_intermediate_results(intermediate_results, spectra_to_fit.atmosphere_type, spectra_to_fit.night_mode)

    return chi_square

def get_trimmed_lbl_path_name(line_list_path_trimmed: str, segment_index: float) -> os.path:
    """
    Gets the name for the lbl trimmed path. Consistent algorithm to always get the same folder name.
    :param line_list_path_trimmed: Path to the trimmed line list
    :param segment_index: Segment's numbering
    :return: path to the folder where to save/already saved trimmed files can exist.
    """
    return os.path.join(line_list_path_trimmed, f"{segment_index}", '')



def load_spectra(specname: str, teff: float, logg: float, rv: float, met: float, microturb: float,
                           macroturb: float, rotation1: float, abundances_dict1: dict, resolution1: float, snr1: float, line_list_path_trimmed: str,
                           index: float, tsfitpy_configuration, m3dis_python_module, debug_mode, tsfitpy_compiler, n_workers) -> Spectra:
    spectra = Spectra(specname, teff, logg, rv, met, microturb, macroturb, rotation1, abundances_dict1, resolution1, snr1,
                      line_list_path_trimmed, index, tsfitpy_configuration, n_workers=n_workers,
                      m3dis_python_module=m3dis_python_module)

    if tsfitpy_compiler == "m3dis":
        spectra.precompute_departure()
        logging.debug(f"Precomputed departure coefficients {spectra.m3dis_iterations_max_precompute}")

    spectra.save_observed_spectra(os.path.join(spectra.output_folder, spectra.spec_name))

    return spectra

def create_and_fit_spectra(dask_client: Client, spectra: Spectra) -> list:
    """
    Creates spectra object and fits based on requested fitting mode
    :param dask_client: Dask client
    :param spectra: Spectra object
    :return: result of the fit with the best fit parameters and chi squared
    """

    if spectra.debug_mode >= 0:
        print(f"Fitting {spectra.spec_name}")
        print(f"Teff = {spectra.teff}; logg = {spectra.logg}; RV = {spectra.stellar_rv}")

    if spectra.dask_workers == 1:
        if spectra.fitting_mode == "all":
            result = spectra.fit_all()
        elif spectra.fitting_mode == "lbl":
            result = spectra.fit_lbl_function(None, spectra.fit_lbl_abund_vmic, spectra.find_upper_limit)
        elif spectra.fitting_mode == "teff":
            result = spectra.fit_lbl_function(None, spectra.fit_lbl_teff, False)
        elif spectra.fitting_mode == "vmic":
            result = spectra.fit_lbl_function(None, spectra.fit_lbl_vmic, False)
        elif spectra.fitting_mode == "logg":
            result = spectra.fit_lbl_function(None, spectra.fit_lbl_logg, False)
        else:
            raise ValueError(f"unknown fitting mode {spectra.fitting_mode}, need all or lbl or teff")
    else:
        if spectra.fitting_mode == "all":
            result = dask_client.submit(spectra.fit_all)
        elif spectra.fitting_mode == "lbl":
            result = spectra.fit_lbl_function(dask_client, spectra.fit_lbl_abund_vmic, spectra.find_upper_limit)
        elif spectra.fitting_mode == "teff":
            result = spectra.fit_lbl_function(dask_client, spectra.fit_lbl_teff, False)
        elif spectra.fitting_mode == "vmic":
            result = spectra.fit_lbl_function(dask_client, spectra.fit_lbl_vmic, False)
        elif spectra.fitting_mode == "logg":
            result = spectra.fit_lbl_function(dask_client, spectra.fit_lbl_logg, False)
        else:
            raise ValueError(f"unknown fitting mode {spectra.fitting_mode}, need all or lbl or teff")
    del spectra
    return result

def run_tsfitpy(output_folder_title: str, config_location: str, spectra_location=None):
    # read the configuration file
    tsfitpy_configuration = TSFitPyConfig(config_location, output_folder_title, spectra_location)
    tsfitpy_configuration.load_config()
    tsfitpy_configuration.validate_input()
    launch_tsfitpy_with_config(tsfitpy_configuration, output_folder_title, config_location)

def launch_tsfitpy_with_config(tsfitpy_configuration: TSFitPyConfig, output_folder_title: str, config_location: str) -> pd.DataFrame:
    if tsfitpy_configuration.debug_mode >= 0:
        print("\nIMPORTANT UPDATE:")
        print("Update 24.05.2023. Currently the assumption is that the third column in the observed spectra is sigma "
              "i.e. the error in the observed spectra (sqrt(variance)). If this is not the case, please change the spectra."
              "If none is given, the error will be assumed to be 0.01. This error is taken into account in chisqr calculation\n\n\n")

    logging.debug(f"Configuration: {tsfitpy_configuration.__dict__}")

    do_hydrogen_linelist = True

    if tsfitpy_configuration.compiler.lower() == "m3dis":
        module_path = os.path.join(tsfitpy_configuration.spectral_code_path, f"{tsfitpy_configuration.m3dis_python_package_name}/__init__.py")
        m3dis_python_module = import_module_from_path("m3dis", module_path)

        if tsfitpy_configuration.cluster_type.lower() == 'slurm':
            #TODO what if i have several running in parallel, will they delete each other
            # create a symlink to the m3dis executable
            m3dis_executable_path = os.path.join(tsfitpy_configuration.spectral_code_path, "m3dis")
            # check if m3dis_executable_path exists
            if not os.path.exists(m3dis_executable_path):
                raise ValueError(f"m3dis executable path {m3dis_executable_path} does not exist!")

            # get current working directory
            cwd = os.getcwd()
            # convert to absolute path
            m3dis_executable_path_symlink = os.path.join(cwd, "m3dis")
            if os.path.exists(m3dis_executable_path_symlink):
                if os.path.islink(m3dis_executable_path_symlink):
                    os.remove(m3dis_executable_path_symlink)
                else:
                    raise ValueError(f"m3dis symlink {m3dis_executable_path_symlink} exists but is not a symlink! Stopped to prevent accidental deletion of files.")
            os.symlink(m3dis_executable_path, m3dis_executable_path_symlink)

        # set molecules to False
        tsfitpy_configuration.include_molecules = False
        do_hydrogen_linelist = False
    else:
        m3dis_python_module = None

    if not config_location[-4:] == ".cfg":
        logging.debug("Configuration: Config file does not end with .cfg. Converting to new format.")
        tsfitpy_configuration.convert_old_config()

    if tsfitpy_configuration.debug_mode >= 0:
        print(f"Fitting data at {tsfitpy_configuration.spectra_input_path} with resolution {tsfitpy_configuration.resolution} and rotation {tsfitpy_configuration.rotation}")

    # set directories
    line_list_path_orig = tsfitpy_configuration.line_list_path
    line_list_path_trimmed = os.path.join(tsfitpy_configuration.temporary_directory_path, "linelist_for_fitting_trimmed", "") # f"{line_list_path}../linelist_for_fitting_trimmed/"

    # load NLTE data dicts
    if tsfitpy_configuration.nlte_flag:
        logging.debug("Configuration: NLTE flag is set to True. Loading NLTE data.")
        nlte_elements_add_to_og_config = []
        if tsfitpy_configuration.oldconfig_nlte_config_outdated is not False:
            print("\n\nDEPRECATION WARNING PLEASE CHECK IT\n\n")
            warn("There is no need to specify paths of NLTE elements. Now you can just specify which elements you want "
                 "in NLTE and the code will load everything. This will cause error in the future.", DeprecationWarning, stacklevel=2)
            if not os.path.exists(os.path.join(tsfitpy_configuration.departure_file_path, "nlte_filenames.cfg")):
                nlte_config_to_write = ConfigParser()

                nlte_items_config = {"Ba": [
                                            'atom.ba111',
                                            'Ba/NLTEgrid_Ba_MARCS_May-10-2021.bin',
                                            'Ba/auxData_Ba_MARCS_May-10-2021.txt',
                                            'Ba/NLTEgrid_Ba_STAGGERmean3D_May-10-2021.bin',
                                            'Ba/auxData_Ba_mean3D_May-10-2021_marcs_names.txt'
                                        ],
                                     "Ca": [
                                         'atom.ca105b',
                                         'Ca/NLTEgrid4TS_Ca_MARCS_Jun-02-2021.bin',
                                         'Ca/auxData_Ca_MARCS_Jun-02-2021.dat',
                                         'Ca/NLTEgrid4TS_Ca_STAGGERmean3D_May-18-2021.bin',
                                         'Ca/auxData_Ca_STAGGERmean3D_May-18-2021_marcs_names.txt'
                                    ],
                                     "Co": [
                                         'atom.co247',
                                         'Co/NLTEgrid4TS_CO_MARCS_Mar-15-2023.bin',
                                         'Co/auxData_CO_MARCS_Mar-15-2023.dat',
                                         '',
                                         ''
                                        ],
                                     "Fe": [
                                         'atom.fe607a',
                                         'Fe/NLTEgrid4TS_Fe_MARCS_May-07-2021.bin',
                                         'Fe/auxData_Fe_MARCS_May-07-2021.dat',
                                         'Fe/NLTEgrid4TS_Fe_STAGGERmean3D_May-21-2021.bin',
                                         'Fe/auxData_Fe_STAGGERmean3D_May-21-2021_marcs_names.txt'
                                     ],
                                     "H": [
                                         'atom.h20',
                                         'H/NLTEgrid_H_MARCS_May-10-2021.bin',
                                         'H/auxData_H_MARCS_May-10-2021.txt',
                                         'H/NLTEgrid4TS_H_STAGGERmean3D_Jun-17-2021.bin',
                                         'H/auxData_H_STAGGERmean3D_Jun-17-2021_marcs_names.txt'
                                     ],
                                     "Mg": [
                                         'atom.mg86b',
                                         'Mg/NLTEgrid4TS_Mg_MARCS_Jun-02-2021.bin',
                                         'Mg/auxData_Mg_MARCS_Jun-02-2021.dat',
                                         'Mg/NLTEgrid_Mg_STAGGERmean3D_May-17-2021.bin',
                                         'Mg/auxData_Mg_STAGGEmean3D_May-17-2021_marcs_names.txt'
                                     ],
                                     "Mn": [
                                         'atom.mn281kbc',
                                         'Mn/NLTEgrid4TS_MN_MARCS_Mar-15-2023.bin',
                                         'Mn/auxData_MN_MARCS_Mar-15-2023.dat',
                                         'Mn/NLTEgrid4TS_Mn_STAGGERmean3D_May-17-2021.bin',
                                         'Mn/auxData_Mn_STAGGERmean3D_May-17-2021_marcs_names.txt'
                                     ],
                                    "Na": [
                                        'atom.na102',
                                        'Na/NLTEgrid4TS_NA_MARCS_Feb-20-2022.bin',
                                        'Na/auxData_Na_MARCS_Feb-20-2022.dat',
                                        '',
                                        ''
                                    ],
                                    "Ni": [
                                        'atom.ni538qm',
                                        'Ni/NLTEgrid4TS_Ni_MARCS_Jan-31-2022.bin',
                                        'Ni/auxData_Ni_MARCS_Jan-21-2022.txt',
                                        'Ni/NLTEgrid4TS_NI_STAGGERmean3D_Jun-10-2021.bin',
                                        'Ni/auxData_NI_STAGGERmean3DJun-10-2021_marcs_names.txt'
                                    ],
                                    "O": [
                                        'atom.o41f',
                                        'O/NLTEgrid4TS_O_MARCS_May-21-2021.bin',
                                        'O/auxData_O_MARCS_May-21-2021.txt',
                                        'O/NLTEgrid4TS_O_STAGGERmean3D_May-18-2021.bin',
                                        'O/auxData_O_STAGGER_May-18-2021_marcs_names.txt'
                                    ],
                                    "Si": [
                                        'atom.si340',
                                        'Si/NLTEgrid4TS_Si_MARCS_Feb-13-2022.bin',
                                        'Si/auxData_Si_MARCS_Feb-13-2022.dat',
                                        '',
                                        ''
                                    ],
                                    "Sr": [
                                        'atom.sr191',
                                        'Sr/NLTEgrid4TS_Sr_MARCS_Mar-15-2023.bin',
                                        'Sr/auxData_Sr_MARCS_Mar-15-2023.dat',
                                        '',
                                        ''
                                    ],
                                    "Ti": [
                                        'atom.ti503',
                                        'Ti/NLTEgrid4TS_TI_MARCS_Feb-21-2022.bin',
                                        'Ti/auxData_TI_MARCS_Feb-21-2022.dat',
                                        '',
                                        ''
                                    ],
                                    "Y": [
                                        'atom.y423',
                                        'Y/NLTEgrid4TS_Y_MARCS_Mar-27-2023.bin',
                                        'Y/auxData_Y_MARCS_Mar-27-2023.dat',
                                        'Y/NLTEgrid4TS_Y_STAGGERmean3D_May-08-2023.bin',
                                        'Y/auxData_Y_STAGGERmean3D_May-08-2023_marcs_names.dat'
                                    ]
                                     }


                for elements_to_save in nlte_items_config:
                    nlte_config_to_write.add_section(elements_to_save)
                    nlte_config_to_write[elements_to_save]["1d_bin"] = nlte_items_config[elements_to_save][1]
                    nlte_config_to_write[elements_to_save]["1d_aux"] = nlte_items_config[elements_to_save][2]
                    nlte_config_to_write[elements_to_save]["3d_bin"] = nlte_items_config[elements_to_save][3]
                    nlte_config_to_write[elements_to_save]["3d_aux"] = nlte_items_config[elements_to_save][4]
                    nlte_config_to_write[elements_to_save]["atom_file"] = nlte_items_config[elements_to_save][0]

                with open(os.path.join(tsfitpy_configuration.departure_file_path, "nlte_filenames.cfg"), "w") as new_config_file:
                    new_config_file.write("# You can add more or change models paths/names here if needed\n"
                                          "#\n"
                                          "# Changelog:\n"
                                          "# 2023 Apr 18: File creation date\n"
                                          "\n"
                                          "# 14 elements\n"
                                          "# 3D and 1D models: Ba, Ca, Fe, H, Mg, Mn, Ni, O\n"
                                          "# 1D models only: Co, Na, Si, Sr, Ti, Y\n\n")
                    nlte_config_to_write.write(new_config_file)

                warn(f"Added {tsfitpy_configuration.departure_file_path, 'nlte_filenames.cfg'} with paths. Please check it or maybe "
                     f"download updated one from the GitHub", DeprecationWarning, stacklevel=2)
            if tsfitpy_configuration.oldconfig_need_to_add_new_nlte_config:
                for element in tsfitpy_configuration.oldconfig_model_atom_file + tsfitpy_configuration.oldconfig_model_atom_file_input_elem:
                    if ".ba" in element:
                        nlte_elements_add_to_og_config.append("Ba")
                    if ".ca" in element:
                        nlte_elements_add_to_og_config.append("Ca")
                    if ".co" in element:
                        nlte_elements_add_to_og_config.append("Co")
                    if ".fe" in element:
                        nlte_elements_add_to_og_config.append("Fe")
                    if ".h" in element:
                        nlte_elements_add_to_og_config.append("H")
                    if ".mg" in element:
                        nlte_elements_add_to_og_config.append("Mg")
                    if ".mn" in element:
                        nlte_elements_add_to_og_config.append("Mn")
                    if ".na" in element:
                        nlte_elements_add_to_og_config.append("Na")
                    if ".ni" in element:
                        nlte_elements_add_to_og_config.append("Ni")
                    if ".o" in element:
                        nlte_elements_add_to_og_config.append("O")
                    if ".si" in element:
                        nlte_elements_add_to_og_config.append("Si")
                    if ".sr" in element:
                        nlte_elements_add_to_og_config.append("Sr")
                    if ".ti" in element:
                        nlte_elements_add_to_og_config.append("Ti")
                    if ".y" in element:
                        nlte_elements_add_to_og_config.append("Y")

                nlte_elements_to_write = ""
                for element in nlte_elements_add_to_og_config:
                    nlte_elements_to_write = f"{nlte_elements_to_write} {element}"

                with open(config_location, "a") as og_config_file:
                    og_config_file.write(f"\n#elements to have in NLTE (just choose whichever elements you want, whether you fit them or not, as few or many as you want). E.g. :"
                                         f"# nlte_elements = Mg Ca Fe\n"
                                         f"nlte_elements = {nlte_elements_to_write}\n"
                                         f"#\n")
                    warn(f"Added how to add NLTE elements now in the {config_location}", DeprecationWarning, stacklevel=2)

        if len(tsfitpy_configuration.nlte_elements) == 0 and len(nlte_elements_add_to_og_config) > 0:
            tsfitpy_configuration.nlte_elements = nlte_elements_add_to_og_config

        nlte_config = ConfigParser()
        # check if nlte_filenames.cfg exists
        if not os.path.isfile(tsfitpy_configuration.departure_file_config_path):
            raise FileNotFoundError(f"NLTE config file not found in {tsfitpy_configuration.departure_file_config_path}. Please download it from the GitHub")
        nlte_config.read(tsfitpy_configuration.departure_file_config_path)

        depart_bin_file_dict, depart_aux_file_dict, model_atom_file_dict = {}, {}, {}

        for element in tsfitpy_configuration.nlte_elements:
            if tsfitpy_configuration.atmosphere_type == "1D":
                bin_config_name, aux_config_name = "1d_bin", "1d_aux"
            else:
                bin_config_name, aux_config_name = "3d_bin", "3d_aux"
            depart_bin_file_dict[element] = nlte_config[element][bin_config_name]
            depart_aux_file_dict[element] = nlte_config[element][aux_config_name]
            model_atom_file_dict[element] = nlte_config[element]["atom_file"]

        print("NLTE loaded. Please check that elements correspond to their correct binary files:")
        for key in depart_bin_file_dict:
            print(f"{key}: {depart_bin_file_dict[key]} {depart_aux_file_dict[key]} {model_atom_file_dict[key]}")

        print(f"If files do not correspond, please check config file {os.path.join(tsfitpy_configuration.departure_file_path, 'nlte_filenames.cfg')}. "
              f"Elements without NLTE binary files do not need them.")

        tsfitpy_configuration.depart_bin_file_dict = depart_bin_file_dict
        tsfitpy_configuration.depart_aux_file_dict = depart_aux_file_dict
        tsfitpy_configuration.model_atom_file_dict = model_atom_file_dict

    tsfitpy_configuration.linemask_file = os.path.join(tsfitpy_configuration.linemasks_path, tsfitpy_configuration.linemask_file)

    if tsfitpy_configuration.make_output_directory:
        # prevent overwriting
        if os.path.exists(tsfitpy_configuration.output_folder_path):
            raise FileExistsError("Error: output folder already exists. Run was stopped to prevent overwriting")
        create_dir(tsfitpy_configuration.output_folder_path)

    if tsfitpy_configuration.debug_mode >= 0:
        print(f"Temporary directory name: {tsfitpy_configuration.temporary_directory_path}")
    create_dir(tsfitpy_configuration.temporary_directory_path)

    # copy config file into output folder (for easier plotting)
    if tsfitpy_configuration.save_config_file:
        if not config_location[-4:] == ".cfg":
            shutil.copyfile(config_location, os.path.join(tsfitpy_configuration.output_folder_path, output_default_configuration_name.replace(".cfg", ".txt")))
        else:
            shutil.copyfile(config_location, os.path.join(tsfitpy_configuration.output_folder_path, output_default_configuration_name))

    fitlist = os.path.join(tsfitpy_configuration.fitlist_input_path, tsfitpy_configuration.input_fitlist_filename)

    # copy fitlist file into output folder (for easier plotting)
    if tsfitpy_configuration.save_fitlist:
        shutil.copyfile(fitlist, os.path.join(tsfitpy_configuration.output_folder_path, output_default_fitlist_name))

    # copy linemask file into output folder (for easier plotting)
    if tsfitpy_configuration.save_linemask:
        shutil.copyfile(tsfitpy_configuration.linemask_file, os.path.join(tsfitpy_configuration.output_folder_path, output_default_linemask_name))

    tsfitpy_configuration.ndimen = 1  # first dimension is RV fit
    if not tsfitpy_configuration.fit_teff:
        if tsfitpy_configuration.fit_vmic and tsfitpy_configuration.fitting_mode == "lbl" and not tsfitpy_configuration.atmosphere_type == "3D":
            tsfitpy_configuration.ndimen += 1  # if fitting micro for lbl, not 3D
        if tsfitpy_configuration.fitting_mode == "lbl" or tsfitpy_configuration.fitting_mode == "vmic":  # TODO: if several elements fitted for other modes, change here
            tsfitpy_configuration.ndimen += tsfitpy_configuration.nelement
            if tsfitpy_configuration.debug_mode >= 0:
                print(f"Fitting {tsfitpy_configuration.nelement} element(s): {tsfitpy_configuration.elements_to_fit}")
        else:
            tsfitpy_configuration.ndimen += 1
            if tsfitpy_configuration.debug_mode >= 0:
                print(f"Fitting {1} element: {tsfitpy_configuration.elements_to_fit[0]}")
        if tsfitpy_configuration.fit_vmac:
            tsfitpy_configuration.ndimen += 1
    else:
        if tsfitpy_configuration.debug_mode >= 0:
            print("Fitting Teff based on the linelist provided. Ignoring element fitting.")

    logging.debug("Reading fitlist")

    fitlist_data = SpectraParameters(fitlist, True)

    logging.debug(f"{fitlist_data}")
    logging.debug(f"Input vmic: {tsfitpy_configuration.vmic_input}, input vmac: {tsfitpy_configuration.vmac_input}, input rotation: {tsfitpy_configuration.rotation_input}")
    fitlist_spectra_parameters = fitlist_data.get_spectra_parameters_for_fit(tsfitpy_configuration.vmic_input, tsfitpy_configuration.vmac_input, tsfitpy_configuration.rotation_input)

    if np.size(tsfitpy_configuration.init_guess_elements) > 0:
        init_guess_spectra_dict = collections.defaultdict(dict)

        for init_guess_elem, init_guess_loc in zip(tsfitpy_configuration.init_guess_elements, tsfitpy_configuration.init_guess_elements_path):
            init_guess_data = np.loadtxt(init_guess_loc, dtype=str, usecols=(0, 1))
            if init_guess_data.ndim == 1:
                init_guess_data = np.array([init_guess_data])
            init_guess_spectra_names, init_guess_values = init_guess_data[:, 0], init_guess_data[:, 1].astype(float)

            for spectra in fitlist_data.spectra_parameters_df['specname'].values:
                spectra_loc_index = np.where(init_guess_spectra_names == spectra)[0][0]
                init_guess_spectra_dict[spectra][init_guess_elem] = init_guess_values[spectra_loc_index]

        tsfitpy_configuration.init_guess_spectra_dict = dict(init_guess_spectra_dict)

    if np.size(tsfitpy_configuration.input_elements_abundance) > 0:
        input_elem_abundance_dict = collections.defaultdict(dict)

        for input_elem, init_elem_loc in zip(tsfitpy_configuration.input_elements_abundance, tsfitpy_configuration.input_elements_abundance_path):
            input_abund_data = np.loadtxt(init_elem_loc, dtype=str, usecols=(0, 1))
            if input_abund_data.ndim == 1:
                input_abund_data = np.array([input_abund_data])
            input_abund_data_spectra_names, input_abund_data_values = input_abund_data[:, 0], input_abund_data[:, 1].astype(float)

            for spectra in fitlist_data.spectra_parameters_df['specname'].values:
                spectra_loc_index = np.where(input_abund_data_spectra_names == spectra)[0]
                if np.size(spectra_loc_index) == 1:
                    input_elem_abundance_dict[spectra][input_elem] = input_abund_data_values[spectra_loc_index]
                else:
                    if tsfitpy_configuration.debug_mode >= 0:
                        print(f"{spectra} does not have element {input_elem} in the input spectra. Using [{input_elem}/Fe]=0")
                    input_elem_abundance_dict[spectra][input_elem] = 0

        tsfitpy_configuration.input_elem_abundance_dict = dict(input_elem_abundance_dict)

    logging.debug("Reading linemask")

    line_centers, line_begins, line_ends = np.loadtxt(tsfitpy_configuration.linemask_file, comments=";", usecols=(0, 1, 2),
                                                      unpack=True)

    if line_centers.size > 1:
        tsfitpy_configuration.line_begins_sorted = np.array(sorted(line_begins))
        tsfitpy_configuration.line_ends_sorted = np.array(sorted(line_ends))
        tsfitpy_configuration.line_centers_sorted = np.array(sorted(line_centers))
    elif line_centers.size == 1:
        tsfitpy_configuration.line_begins_sorted = np.array([line_begins])
        tsfitpy_configuration.line_ends_sorted = np.array([line_ends])
        tsfitpy_configuration.line_centers_sorted = np.array([line_centers])

    tsfitpy_configuration.seg_begins, tsfitpy_configuration.seg_ends = create_segment_file(tsfitpy_configuration.segment_size, tsfitpy_configuration.line_begins_sorted, tsfitpy_configuration.line_ends_sorted)
    # save segment in a separate file where each line is an index of the seg_begins and seg_ends
    np.savetxt(tsfitpy_configuration.segment_file, np.column_stack((tsfitpy_configuration.seg_begins, tsfitpy_configuration.seg_ends)), fmt="%d")

    # check inputs
    if tsfitpy_configuration.debug_mode >= 0:
        tsfitpy_configuration.warn_on_config_issues()

        print("Trimming down the linelist to only lines within segments for faster fitting")
    if tsfitpy_configuration.fitting_mode == "all":
        line_list_path_trimmed = os.path.join(line_list_path_trimmed, "all", output_folder_title, '')
        create_window_linelist(tsfitpy_configuration.seg_begins, tsfitpy_configuration.seg_ends, line_list_path_orig, line_list_path_trimmed,
                               tsfitpy_configuration.include_molecules, lbl=False, do_hydrogen=do_hydrogen_linelist)
        line_list_path_trimmed =  os.path.join(line_list_path_trimmed, "0", "")
    elif tsfitpy_configuration.fitting_mode == "lbl" or tsfitpy_configuration.fitting_mode == "teff" or tsfitpy_configuration.fitting_mode == "vmic" or tsfitpy_configuration.fitting_mode == "logg":
        line_list_path_trimmed = os.path.join(line_list_path_trimmed, "lbl", output_folder_title, '')
        create_window_linelist(tsfitpy_configuration.seg_begins, tsfitpy_configuration.seg_ends, line_list_path_orig,
                               line_list_path_trimmed,
                               tsfitpy_configuration.include_molecules, lbl=True, do_hydrogen=do_hydrogen_linelist)
    else:
        raise ValueError("Unknown fitting method")
    if tsfitpy_configuration.compiler.lower() == "m3dis":
        # if m3dis, then combine all linelists into one
        # go into line_list_path_trimmed and each folder and combine all linelists into one in each of the folders
        for folder in os.listdir(line_list_path_trimmed):
            if os.path.isdir(os.path.join(line_list_path_trimmed, folder)):
                # go into each folder and combine all linelists into one
                combined_linelist = os.path.join(line_list_path_trimmed, folder, "combined_linelist.bsyn")
                with open(combined_linelist, "w") as combined_linelist_file:
                    for file in os.listdir(os.path.join(line_list_path_trimmed, folder)):
                        if file.endswith(".bsyn") and not file == "combined_linelist.bsyn":
                            with open(os.path.join(line_list_path_trimmed, folder, file), "r") as linelist_file:
                                combined_linelist_file.write(linelist_file.read())
                            # delete the file
                            os.remove(os.path.join(line_list_path_trimmed, folder, file))
                            #print(os.path.join(line_list_path_trimmed, folder, file))
        # also want to precut model atom if LTE, but using model atom
        if tsfitpy_configuration.nlte_flag and tsfitpy_configuration.iterations_max_precompute == 0 and tsfitpy_configuration.iterations_max == 0:
            # if NLTE, then do nothing
            # if LTE, then precut model atom
            # can only do 1 element for now
            # go through all segments
            for segment_index, (segment_begin, segment_end) in enumerate(zip(tsfitpy_configuration.seg_begins, tsfitpy_configuration.seg_ends)):
                # load model atom
                model_atom = ModelAtom()
                model_atom.read_model_atom(os.path.join(tsfitpy_configuration.model_atoms_path, tsfitpy_configuration.model_atom_file_dict[tsfitpy_configuration.nlte_elements[0]]))
                # cut model atom
                model_atom.leave_only_bb_transitions_between_wavelength(segment_begin, segment_end)
                # lets create new temporary directory, where for each segment we will save model atom
                temp_dir_model_atom = os.path.join(tsfitpy_configuration.temporary_directory_path, "model_atom", f"{segment_index}", "")
                # create directory
                create_dir(temp_dir_model_atom)
                # save model atom
                model_atom.write_model_atom(os.path.join(temp_dir_model_atom, model_atom.atom_filename))
            # change the path of config's model atom file to temporary
            tsfitpy_configuration.model_atoms_path = os.path.join(tsfitpy_configuration.temporary_directory_path, "model_atom", "")

    if tsfitpy_configuration.debug_mode >= 0:
        print("Finished trimming linelist")

    model_temperatures, model_logs, model_mets, marcs_value_keys, marcs_models, marcs_values = fetch_marcs_grid(tsfitpy_configuration.model_atmosphere_list, TurboSpectrum.marcs_parameters_to_ignore)
    pickled_marcs_models_location = os.path.join(tsfitpy_configuration.temporary_directory_path, "marcs_models.pkl")

    with open(pickled_marcs_models_location, 'wb') as f:
        # dump all the data into the file, including temperatures, logs, mets, value keys, models and values
        pickle.dump([model_temperatures, model_logs, model_mets, marcs_value_keys, marcs_models, marcs_values], f)
    # clear the variables
    del model_temperatures, model_logs, model_mets, marcs_value_keys, marcs_models, marcs_values

    if tsfitpy_configuration.nlte_flag:
        tsfitpy_configuration.aux_file_length_dict = {}

        if not tsfitpy_configuration.compiler.lower() == "m3dis":
            for element in model_atom_file_dict:
                tsfitpy_configuration.aux_file_length_dict[element] = len(np.loadtxt(os.path.join(tsfitpy_configuration.departure_file_path, depart_aux_file_dict[element]), dtype='str'))
        else:
            for element in model_atom_file_dict:
                tsfitpy_configuration.aux_file_length_dict[element] = 0

    # pickle the configuration file into the temp folder
    #with open(os.path.join(tsfitpy_configuration.temporary_directory_path, "tsfitpy_configuration.pkl"), "wb") as f:
    #    pickle.dump(tsfitpy_configuration, f)
    #tsfitpy_pickled_configuration_path = os.path.join(tsfitpy_configuration.temporary_directory_path, "tsfitpy_configuration.pkl")

    logging.debug("Finished preparing the configuration file")

    if tsfitpy_configuration.number_of_cpus != 1:
        if tsfitpy_configuration.debug_mode >= 0:
            night_mode = False
        else:
            night_mode = True
        client = get_dask_client(tsfitpy_configuration.cluster_type, tsfitpy_configuration.cluster_name,
                                 tsfitpy_configuration.number_of_cpus, nodes=tsfitpy_configuration.number_of_nodes,
                                 slurm_script_commands=tsfitpy_configuration.script_commands,
                                 slurm_memory_per_core=tsfitpy_configuration.memory_per_cpu_gb,
                                 time_limit_hours=tsfitpy_configuration.time_limit_hours,
                                 slurm_partition=tsfitpy_configuration.slurm_partition, night_mode=night_mode)

        if tsfitpy_configuration.compiler.lower() == "m3dis":
            module_path = os.path.join(tsfitpy_configuration.spectral_code_path, f"{tsfitpy_configuration.m3dis_python_package_name}/__init__.py")
            client.run(import_module_from_path, "m3dis", module_path)

        tsfitpy_configuration_scattered = client.scatter(tsfitpy_configuration)

        futures = []
        for idx, one_spectra_parameters in enumerate(fitlist_spectra_parameters):
            # specname_list, rv_list, teff_list, logg_list, feh_list, vmic_list, vmac_list, abundance_list
            specname1, rv1, teff1, logg1, met1, microturb1, macroturb1, rotation1, abundances_dict1, resolution1, snr1 = one_spectra_parameters
            logging.debug(f"specname1: {specname1}, rv1: {rv1}, teff1: {teff1}, logg1: {logg1}, met1: {met1}, "
                          f"microturb1: {microturb1}, macroturb1: {macroturb1}, rotation1: {rotation1}, snr1: {snr1},"
                          f"abundances_dict1: {abundances_dict1}, resolution1: {resolution1}")
            spectra_future = client.submit(load_spectra, specname1, teff1, logg1, rv1, met1, microturb1, macroturb1,
                                           rotation1, abundances_dict1, resolution1, snr1, line_list_path_trimmed, idx,
                                           tsfitpy_configuration_scattered, m3dis_python_module,
                                           tsfitpy_configuration.debug_mode, tsfitpy_configuration.compiler.lower(),
                                           tsfitpy_configuration.number_of_cpus)
            futures.append(spectra_future)  # prepares to get values
        all_spectra_futures = client.gather(futures)
        futures = []
        for idx, spectra in enumerate(all_spectra_futures):
            future = create_and_fit_spectra(client, spectra)
            futures.append(future)  # prepares to get values

        if tsfitpy_configuration.debug_mode >= 0:
            print("Start gathering")  # use http://localhost:8787/status to check status. the port might be different
        futures = np.array(client.gather(futures))  # starts the calculations (takes a long time here)
        results = futures
        if tsfitpy_configuration.debug_mode >= 0:
            print("Worker calculation done")  # when done, save values
        client.close()  # close dask client
    else:
        results = []
        for idx, one_spectra_parameters in enumerate(fitlist_spectra_parameters):
            specname1, rv1, teff1, logg1, met1, microturb1, macroturb1, rotation1, abundances_dict1, resolution1, snr1 = one_spectra_parameters
            logging.debug(
                f"specname1: {specname1}, rv1: {rv1}, teff1: {teff1}, logg1: {logg1}, met1: {met1}, microturb1: {microturb1}, macroturb1: {macroturb1}, rotation1: {rotation1}, abundances_dict1: {abundances_dict1}, resolution1: {resolution1}, snr1: {snr1},")

            one_spectra = load_spectra(specname1, teff1, logg1, rv1, met1, microturb1, macroturb1,
                                           rotation1, abundances_dict1, resolution1, snr1, line_list_path_trimmed, idx,
                                           tsfitpy_configuration, m3dis_python_module, tsfitpy_configuration.debug_mode,
                                           tsfitpy_configuration.compiler.lower(), tsfitpy_configuration.number_of_cpus)

            results.append(create_and_fit_spectra(None, one_spectra))
            del one_spectra

    logging.debug("Finished fitting, now saving results")

    output = os.path.join(tsfitpy_configuration.output_folder_path, tsfitpy_configuration.output_filename)

    # convert mess above to a pd.DataFrame
    df_results = pd.DataFrame()
    for i in range(len(results)):
        for j in range(len(results[i])):
            # do concat such that each row has a different index
            df_results = pd.concat([df_results, pd.DataFrame(results[i][j]['result'], index=[i])], ignore_index=True)

    # reset the index
    df_results = df_results.reset_index(drop=True)

    # go through all columns. if the element is in the periodic table, replace it with element_Fe
    for column in df_results.columns:
        if column in periodic_table:
            if column == 'Fe':
                df_results.rename(columns={column: f"{column}_H"}, inplace=True)
            else:
                df_results.rename(columns={column: f"{column}_Fe"}, inplace=True)

    # save results to csv without index and with tab delimiter
    if tsfitpy_configuration.save_results:
        df_results.to_csv(output, index=False, sep='\t')

        logging.debug("Finished saving results, now removing temporary files")

    shutil.rmtree(tsfitpy_configuration.temporary_directory_path)  # clean up temp directory
    try:
        shutil.rmtree(line_list_path_trimmed)   # clean up trimmed line list
    except FileNotFoundError:
        pass    # because now trimmed files are in the temp directory, might give error

    if tsfitpy_configuration.debug_mode >= 0:
        print("TSFitPy had normal termination")

    return df_results

if __name__ == '__main__':
    raise RuntimeError("This file is not meant to be run as main. Please run TSFitPy/main.py instead.")  # this is a module
