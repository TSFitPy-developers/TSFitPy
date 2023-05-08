from __future__ import annotations

import pickle
from configparser import ConfigParser
from warnings import warn
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scripts.turbospectrum_class_nlte import TurboSpectrum, fetch_marcs_grid
import time
import os
from os import path as os_path
import datetime
try:
    from dask.distributed import Client, get_client, secede, rejoin
except (ModuleNotFoundError, ImportError):
    raise ModuleNotFoundError("Dask not installed. It is required for multiprocessing. Install using pip install dask[complete]")
import shutil
import socket
from sys import argv
import collections
import scipy
from scripts.convolve import conv_rotation, conv_macroturbulence, conv_res
from scripts.create_window_linelist_function import create_window_linelist


def create_dir(directory: str):
    """
    Creates a directory if it does not exist
    :param directory: Path to the directory that can be created (relative or not)
    """
    if not os.path.exists(directory):
        try:
            os.mkdir(directory)
        except FileNotFoundError:  # if one needs to create folders in-between
            os.makedirs(directory)  # can't recall why I just don't call this function directly


def calculate_vturb(teff: float, logg: float, met: float) -> float:
    """
    Calculates micro turbulence based on the input parameters
    :param teff: Temperature in kelvin
    :param logg: log(g) in dex units
    :param met: metallicity [Fe/H] scaled by solar
    :return: micro turbulence in km/s
    """
    t0 = 5500.
    g0 = 4.

    v_mturb: float = 0

    if teff >= 5000.:
        v_mturb = 1.05 + 2.51e-4 * (teff - t0) + 1.5e-7 * (teff - t0) * (teff - t0) - 0.14 * (logg - g0) - 0.005 * (
                logg - g0) * (logg - g0) + 0.05 * met + 0.01 * met * met
    elif teff < 5000. and logg >= 3.5:
        v_mturb = 1.05 + 2.51e-4 * (teff - t0) + 1.5e-7 * (5250. - t0) * (5250. - t0) - 0.14 * (logg - g0) - 0.005 * (
                logg - g0) * (logg - g0) + 0.05 * met + 0.01 * met * met
    elif teff < 5500. and logg < 3.5:
        v_mturb = 1.25 + 4.01e-4 * (teff - t0) + 3.1e-7 * (teff - t0) * (teff - t0) - 0.14 * (logg - g0) - 0.005 * (
                logg - g0) * (logg - g0) + 0.05 * met + 0.01 * met * met

    if v_mturb <= 0.0:
        print("error in calculating micro turb, setting it to 1.0")
        return 1.0

    return v_mturb


def get_convolved_spectra(wave: np.ndarray, flux: np.ndarray, resolution: float, macro: float, rot: float) -> tuple[
    np.ndarray, np.ndarray]:
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


def calc_ts_spectra_all_lines(obs_name: str, temp_directory: str, output_dir: str, wave_obs: np.ndarray,
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
    if os_path.exists(f'{temp_directory}/spectrum_00000000.spec') and os.stat(
            f'{temp_directory}/spectrum_00000000.spec').st_size != 0:
        wave_mod_orig, flux_mod_orig = np.loadtxt(f'{temp_directory}/spectrum_00000000.spec', usecols=(0, 1),
                                                  unpack=True)
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

        os.system(
            f"mv {temp_directory}spectrum_00000000.spec {output_dir}spectrum_fit_{obs_name.replace('../input_files/observed_spectra/', '')}")
        out = open(f"{output_dir}spectrum_fit_convolved_{obs_name.replace('../input_files/observed_spectra/', '')}",
                   'w')
        for l in range(len(wave_mod)):
            print(f"{wave_mod[l]}  {flux_mod[l]}", file=out)
        out.close()
    elif os_path.exists(f'{temp_directory}/spectrum_00000000.spec') and os.stat(
            f'{temp_directory}/spectrum_00000000.spec').st_size == 0:
        chi_square = 999.99
        print("empty spectrum file.")
    else:
        chi_square = 9999.9999
        print("didn't generate spectra")
    return chi_square


def calculate_lbl_chi_squared(temp_directory: str, wave_obs: np.ndarray, flux_obs: np.ndarray,
                              wave_mod_orig: np.ndarray, flux_mod_orig: np.ndarray, resolution: float, lmin: float,
                              lmax: float, macro: float, rot: float, save_convolved=True) -> float:
    """
    Calculates chi squared by opening a created synthetic spectrum and comparing to the observed spectra. Then
    calculates chi squared. Used for line by line method, by only looking at a specific line.
    :param temp_directory:
    :param wave_obs: Observed wavelength
    :param flux_obs: Observed normalised flux
    :param wave_mod_orig: Synthetic wavelength
    :param flux_mod_orig: Synthetic normalised flux
    :param resolution: resolution, zero if not required
    :param lmax: Wavelength, start of segment (will calculate at +5 AA to this)
    :param lmin: Wavelength, end of segment  (will calculate at -5 AA to this)
    :param macro: Macroturbulence in km/s, zero if not required
    :param rot: Rotation in km/s, 0 if not required
    :param save_convolved: whether to save convolved spectra or not (default True)
    :return: Calculated chi squared for a given line
    """
    indices_to_use_mod = np.where((wave_mod_orig <= lmax) & (wave_mod_orig >= lmin))
    indices_to_use_obs = np.where((wave_obs <= lmax) & (wave_obs >= lmin))

    wave_mod_orig, flux_mod_orig = wave_mod_orig[indices_to_use_mod], flux_mod_orig[indices_to_use_mod]
    wave_obs, flux_obs = wave_obs[indices_to_use_obs], flux_obs[indices_to_use_obs]

    wave_mod, flux_mod = get_convolved_spectra(wave_mod_orig, flux_mod_orig, resolution, macro, rot)
    if wave_mod[1] - wave_mod[0] <= wave_obs[1] - wave_obs[0]:
        flux_mod_interp = np.interp(wave_obs, wave_mod, flux_mod)
        wave_line = wave_obs[
            np.where((wave_obs <= lmax - 5.) & (wave_obs >= lmin + 5.))]  # 5 AA i guess to remove extra edges??
        flux_line_obs = flux_obs[np.where((wave_obs <= lmax - 5.) & (wave_obs >= lmin + 5.))]
        flux_line_mod = flux_mod_interp[np.where((wave_obs <= lmax - 5.) & (wave_obs >= lmin + 5.))]
        chi_square = np.sum(((flux_line_obs - flux_line_mod) * (flux_line_obs - flux_line_mod)) / flux_line_mod)
    else:
        flux_obs_interp = np.interp(wave_mod, wave_obs, flux_obs)
        wave_line = wave_mod[np.where((wave_mod <= lmax - 5.) & (wave_mod >= lmin + 5.))]
        flux_line_obs = flux_obs_interp[np.where((wave_mod <= lmax - 5.) & (wave_mod >= lmin + 5.))]
        flux_line_mod = flux_mod[np.where((wave_mod <= lmax - 5.) & (wave_mod >= lmin + 5.))]
        chi_square = np.sum(((flux_line_obs - flux_line_mod) * (flux_line_obs - flux_line_mod)) / flux_line_mod)
    # os.system(f"mv {temp_directory}spectrum_00000000.spec ../output_files/spectrum_fit_{obs_name.replace('../input_files/observed_spectra/', '')}")

    if save_convolved:
        out = open(f"{temp_directory}spectrum_00000000_convolved.spec", 'w')

        for i in range(len(wave_line)):
            print("{}  {}".format(wave_line[i], flux_line_mod[i]), file=out)
        out.close()
    return chi_square


def calculate_equivalent_width(fit_wavelength: np.ndarray, fit_flux: np.ndarray, left_bound: float, right_bound: float) -> float:
    line_func = interp1d(fit_wavelength, fit_flux, kind='linear', assume_sorted=True,
                                     fill_value=1, bounds_error=False)
    total_area = (right_bound - left_bound) * 1.0   # continuum
    try:
        integration_points = fit_wavelength[np.logical_and.reduce((fit_wavelength > left_bound, fit_wavelength < right_bound))]
        area_under_line = integrate.quad(line_func, left_bound, right_bound, points=integration_points, limit=len(integration_points) * 5)
    except ValueError:
        return -9999

    return total_area - area_under_line[0]

class Result:
    # because other fitting algorithms call result differently
    def __init__(self):
        # res.x: list = [param1 best guess, param2 best guess etc]
        # res.fun: float = function value (chi squared) after the fit
        self.fun: float = None
        self.x: list = None

def minimize_function(function_to_minimize, input_param_guess: np.ndarray, function_arguments: tuple, bounds: list[tuple], method: str, options: dict):
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
                 line_list_path_trimmed: str, index_temp_dir: float, tsfitpy_config, elem_abund=None):
        # Default values
        self.turbospec_path: str = None  # path to the /exec/ file
        self.interpol_path: str = None  # path to the model_interpolators folder with fortran code
        self.model_atmosphere_grid_path: str = None
        self.model_atmosphere_list: str = None
        self.model_atom_path: str = None
        self.departure_file_path: str = None
        self.linemask_file: str = None
        self.segment_file: str = None  # TODO: add this to the config file
        self.atmosphere_type: str = None  # "1D" or "3D", string
        self.include_molecules: bool = None  # "True" or "False", bool
        self.nlte_flag: bool = None
        self.fit_vmic: str = "No"  # TODO: redo as bool. It expects, "Yes", "No" or "Input". Add extra variable if input?
        self.fit_vmac: bool = False
        self.fit_rotation: bool = False
        self.fit_teff: bool = None
        #self.fit_logg: str = None  # does not work atm
        self.nelement: int = None  # how many elements to fit (1 to whatever)
        self.fit_feh: bool = None
        self.elem_to_fit: np.ndarray = None  # only 1 element at a time is support atm, a list otherwise
        self.lmin: float = None
        self.lmax: float = None
        self.ldelta: float = None
        self.resolution: float = None  # resolution coming from resolution, constant for all stars:  central lambda / FWHM
        # macroturb: float = None  # macroturbulence km/s, constant for all stars if not fitted
        self.rotation: float = None  # rotation km/s, constant for all stars
        self.fitting_mode: str = None  # "lbl" = line by line or "all" or "lbl_quick"
        self.output_folder: str = None

        self.dask_workers: int = None  # workers, i.e. CPUs for multiprocessing

        self.global_temp_dir: str = None
        self.line_begins_sorted: np.ndarray = None
        self.line_ends_sorted: np.ndarray = None
        self.line_centers_sorted: np.ndarray = None

        self.seg_begins: np.ndarray = None
        self.seg_ends: np.ndarray = None

        self.depart_bin_file_dict: dict = None
        self.depart_aux_file_dict: dict = None
        self.model_atom_file_dict: dict = None
        self.aux_file_length_dict: dict = None  # loads the length of aux file to not do it everytime later
        self.ndimen: int = None
        self.spec_input_path: str = None

        self.grids_amount: int = 25
        self.abund_bound: float = 0.2

        self.init_guess_dict: dict = None  # initial guess for elements, if given
        self.input_elem_abundance: dict = None  # input elemental abundance for a spectra, not fitted, just used for TS

        # bounds for the minimization
        self.bound_min_vmac = 0  # km/s
        self.bound_max_vmac = 30
        self.bound_min_rotation = 0  # km/s
        self.bound_max_rotation = 30
        self.bound_min_vmic = 0.01  # km/s
        self.bound_max_vmic = 5
        self.bound_min_abund = -40  # [X/Fe]
        self.bound_max_abund = 100
        self.bound_min_feh = -4  # [Fe/H]
        self.bound_max_feh = 0.5
        self.bound_min_doppler = -1  # km/s
        self.bound_max_doppler = 1

        # guess bounds for the minimization
        self.guess_min_vmac = 0.2  # km/s
        self.guess_max_vmac = 8
        self.guess_min_rotation = 0.2  # km/s
        self.guess_max_rotation = 2
        self.guess_min_vmic = 0.8  # km/s
        self.guess_max_vmic = 1.5
        self.guess_min_abund = -1  # [X/Fe] or [Fe/H]
        self.guess_max_abund = 0.4
        self.guess_min_doppler = -1  # km/s
        self.guess_max_doppler = 1

        self.bound_min_teff = 2500
        self.bound_max_teff = 8000

        self.guess_plus_minus_neg_teff = -1000
        self.guess_plus_minus_pos_teff = 1000

        self.model_temperatures: np.ndarray = None
        self.model_logs: np.ndarray = None
        self.model_mets: np.ndarray = None
        self.marcs_value_keys: list = None
        self.marcs_models: dict = None
        self.marcs_values: dict = None

        self.debug_mode = 0  # 0: no debug, 1: show Python warnings, 2: turn Fortran TS verbose setting on
        self.experimental_parallelisation = False  # experimental parallelisation of the TS code

        # Set values from config
        tsfitpy_config.load_spectra_config(self)

        self.spec_name: str = str(specname)
        self.spec_path: str = os.path.join(self.spec_input_path, str(specname))
        print(self.spec_path)
        self.teff: float = float(teff)
        self.logg: float = float(logg)
        self.met: float = float(met)
        self.rv: float = float(rv)  # RV of star (given, but is fitted with extra doppler shift)
        self.doppler_shift: float = 0.0  # doppler shift; added to RV (fitted)
        if elem_abund is not None:
            self.elem_abund_input: float = float(elem_abund)  # initial abundance of element as a guess if lbl quick
        else:
            self.elem_abund_input = None
        if self.input_elem_abundance is None:  # input abundance - NOT fitted, but just accepted as a constant abund for spectra
            self.input_abund: dict = {}
        else:
            try:
                self.input_abund: dict = self.input_elem_abundance[self.spec_name]
            except KeyError:
                self.input_abund: dict = {}
        if self.fit_vmic == "Input":
            self.vmic: float = float(micro)  # microturbulence. Set if it is given in input
        else:
            self.vmic = None
        self.vmac: float = float(macro)  # macroturbulence km/s, constant for all stars if not fitted
        self.temp_dir: str = os.path.join(self.global_temp_dir, self.spec_name + str(index_temp_dir),
                                          '')  # temp directory, including date and name of the star fitted
        create_dir(self.temp_dir)  # create temp directory

        self.abund_to_gen = None  # array with generated abundances for lbl quick

        self.init_param_guess: list = None  # guess for minimzation
        self.initial_simplex_guess: list = None
        self.minim_bounds: list = []
        self.set_param_guess()

        self.line_list_path_trimmed = line_list_path_trimmed  # location of trimmed files

        """self.ts = TurboSpectrum(
            turbospec_path=self.turbospec_path,
            interpol_path=self.interpol_path,
            line_list_paths=self.line_list_path_trimmed,
            marcs_grid_path=self.model_atmosphere_grid_path,
            marcs_grid_list=self.model_atmosphere_list,
            model_atom_path=self.model_atom_path,
            departure_file_path=self.departure_file_path)"""

        self.wave_ob, self.flux_ob = np.loadtxt(self.spec_path, usecols=(0, 1), unpack=True,
                                                dtype=float)  # observed spectra

        if self.debug_mode >= 1:
            self.python_verbose = True
        else:
            self.python_verbose = False
        if self.debug_mode >= 2:
            self.turbospectrum_verbose = True
        else:
            self.turbospectrum_verbose = False

        # for experimental parallelisation need to have dictionary of fitted values so they dont interfere
        # each index is a different line
        self.vmac_dict = {}
        self.vmic_dict = {}
        self.rotation_dict = {}
        self.doppler_shift_dict = {}
        self.elem_abund_dict_fitting = {}

    def set_param_guess(self):
        """
        Converts init param guess list to the 2D list for the simplex calculation
        """
        # make an array for initial guess equal to n x ndimen+1
        initial_guess = np.empty((self.ndimen + 1, self.ndimen))
        # 17.11.2022: Tried random guesses. But they DO affect the result if the random guesses are way off.
        # Trying with linspace. Should be better I hope
        min_microturb = self.guess_min_vmic  # set bounds for all elements here, change later if needed
        max_microturb = self.guess_max_vmic  # km/s ? cannot be less than 0
        min_macroturb = self.guess_min_vmac  # km/s; cannot be less than 0
        max_macroturb = self.guess_max_vmac
        min_abundance = self.guess_min_abund  # either [Fe/H] or [X/Fe] here
        max_abundance = self.guess_max_abund  # for [Fe/H]: hard bounds -4 to 0.5; other elements: bounds are above -40
        min_rv = self.guess_min_doppler  # km/s i think as well
        max_rv = self.guess_max_doppler
        #microturb_guesses = np.linspace(min_microturb, max_microturb, self.ndimen + 1)
        macroturb_guesses = np.linspace(min_macroturb + np.random.random(1)[0] / 2, max_macroturb + np.random.random(1)[0] / 2, self.ndimen + 1)
        abundance_guesses = np.linspace(min_abundance + np.random.random(1)[0] / 10, max_abundance + np.random.random(1)[0] / 10, self.ndimen + 1)
        rv_guesses = np.linspace(min_rv + np.random.random(1)[0] / 10, max_rv + np.random.random(1)[0] / 10, self.ndimen + 1)

        """# fill the array with input from config file # OLD
        for j in range(self.ndimen):
            for i in range(j, len(init_param_guess), self.ndimen):
                initial_guess[int(i / self.ndimen)][j] = float(init_param_guess[i])"""

        # TODO: order depends on the fitting mode. Make more universal?

        if self.fitting_mode == "all":
            # abund = param[0]
            # dopple = param[1]
            # macroturb = param [2] (if needed)
            initial_guess[:, 0] = abundance_guesses
            if self.fit_feh:
                self.minim_bounds.append((self.bound_min_feh, self.bound_max_feh))
            else:
                self.minim_bounds.append((self.bound_min_abund, self.bound_max_abund))
            initial_guess[:, 1] = rv_guesses
            self.minim_bounds.append((self.bound_min_doppler, self.bound_max_doppler))
            if self.fit_vmac:
                initial_guess[:, 2] = macroturb_guesses
                self.minim_bounds.append((self.bound_min_vmac, self.bound_max_vmac))
        elif self.fitting_mode == "lbl":
            # param[0] = added doppler to rv
            # param[1:nelements] = met or abund
            # param[-1] = macro turb IF MACRO FIT
            # param[-2] = micro turb IF MACRO FIT
            # param[-1] = micro turb IF NOT MACRO FIT
            initial_guess, self.minim_bounds = self.get_rv_elem_micro_macro_guess(min_rv, max_rv,
                                                                                  min_macroturb, max_macroturb,
                                                                                  min_microturb, max_microturb,
                                                                                  min_abundance, max_abundance)
        elif self.fitting_mode == "lbl_quick":
            # param[0] = doppler
            # param[1] = macro turb
            initial_guess, self.minim_bounds = self.get_rv_macro_rotation_guess()
        else:
            ValueError("Unknown fitting mode, choose all or lbl")

        self.init_param_guess = initial_guess[0]
        self.initial_simplex_guess = initial_guess


    def get_elem_micro_guess(self, min_microturb: float, max_microturb: float, min_abundance: float,
                             max_abundance: float) -> tuple[np.ndarray, list[tuple]]:
        # param[0:nelements-1] = met or abund
        # param[-1] = micro turb

        guess_length = self.nelement
        if self.fit_vmic == "Yes" and self.atmosphere_type != "3D":
            guess_length += 1

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
        if self.fit_vmic == "Yes" and self.atmosphere_type != "3D":  # last is micro
            micro_guess, micro_bounds = self.get_simplex_guess(guess_length, min_microturb, max_microturb, self.bound_min_vmic, self.bound_max_vmic)
            guesses = np.append(guesses, [micro_guess], axis=0)
            bounds.append(micro_bounds)

        guesses = np.transpose(guesses)

        return guesses, bounds
    def get_elem_guess(self, min_abundance: float, max_abundance: float) -> tuple[np.ndarray, list[tuple]]:
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
    def get_micro_guess(self, min_microturb: float, max_microturb: float) -> tuple[np.ndarray, list[tuple]]:
        # param[0:nelements-1] = met or abund
        # param[-1] = micro turb

        guess_length = 1

        bounds = []

        guesses = np.array([])

        if self.atmosphere_type != "3D":  # last is micro
            micro_guess, micro_bounds = self.get_simplex_guess(guess_length, min_microturb, max_microturb, self.bound_min_vmic, self.bound_max_vmic)
            guesses = np.array([micro_guess])
            bounds.append(micro_bounds)

        guesses = np.transpose(guesses)

        return guesses, bounds

    def get_rv_elem_micro_macro_guess(self, min_rv: float, max_rv: float, min_macroturb: float,
                           max_macroturb: float, min_microturb: float, max_microturb: float, min_abundance: float,
                           max_abundance: float) -> tuple[np.ndarray, list[tuple]]:
        # param[0] = added doppler to rv
        # param[1:nelements] = met or abund
        # param[-1] = macro turb IF MACRO FIT
        # param[-2] = micro turb IF MACRO FIT
        # param[-1] = micro turb IF NOT MACRO FIT

        guess_length = self.ndimen
        bounds = []

        rv_guess, rv_bounds = self.get_simplex_guess(guess_length, min_rv, max_rv, self.bound_min_doppler, self.bound_max_doppler)
        guesses = np.array([rv_guess])
        bounds.append(rv_bounds)
        for i in range(1, self.nelement + 1):
            if self.elem_to_fit[i - 1] == "Fe":
                guess_elem, bound_elem = self.get_simplex_guess(guess_length, min_abundance, max_abundance,
                                                                self.bound_min_feh, self.bound_max_feh)
            else:
                guess_elem, bound_elem = self.get_simplex_guess(guess_length, min_abundance, max_abundance,
                                                                self.bound_min_abund, self.bound_max_abund)
            if self.init_guess_dict is not None and self.elem_to_fit[i - 1] in self.init_guess_dict[self.spec_name]:
                abund_guess = self.init_guess_dict[self.spec_name][self.elem_to_fit[i - 1]]
                abundance_guesses = np.linspace(abund_guess - 0.1, abund_guess + 0.1, guess_length + 1)
                guesses = np.append(guesses, [abundance_guesses], axis=0)
                # if initial abundance is given, then linearly give guess +/- 0.1 dex
            else:
                guesses = np.append(guesses, [guess_elem], axis=0)
            bounds.append(bound_elem)
        if self.fit_vmic == "Yes" and not self.atmosphere_type == "3D":  # first adding micro
            micro_guess, micro_bounds = self.get_simplex_guess(guess_length, min_microturb, max_microturb, self.bound_min_vmic, self.bound_max_vmic)
            guesses = np.flip(np.append(guesses, [micro_guess], axis=0))
            bounds.append(micro_bounds)
        if self.fit_vmac:  # last is macro
            macro_guess, macro_bounds = self.get_simplex_guess(guess_length, min_macroturb, max_macroturb, self.bound_min_vmac, self.bound_max_vmac)
            guesses = np.append(guesses, [macro_guess], axis=0)
            bounds.append(macro_bounds)

        guesses = np.transpose(guesses)

        return guesses, bounds

    @staticmethod
    def get_simplex_guess(length: int, min_guess: float, max_guess: float, min_bound: float, max_bound: float) -> tuple[
        np.ndarray, tuple]:
        """
        Gets guess if it is fitted for simplex guess
        :param length: number of dimensions (output length+1 array)
        :param min_guess: minimum guess
        :param max_guess: maximum guess
        :param min_bound: minimum bound
        :param max_bound: maximum bound
        :return: Initial guess and minimum bound
        """
        percentage_of_difference_to_add = 10  # basically adds a bit of randomness to the guess up to this % of the diff of guesses

        if min_guess < min_bound:
            min_guess = min_bound
        if max_guess > max_bound:
            max_guess = max_bound

        minim_bounds = (min_bound, max_bound)

        guess_difference = np.abs(max_guess - min_guess) / percentage_of_difference_to_add

        initial_guess = np.linspace(min_guess + np.random.random() * guess_difference,
                                    max_guess - np.random.random() * guess_difference, length + 1)

        return initial_guess, minim_bounds

    def get_rv_macro_rotation_guess(self, min_rv=None, max_rv=None, min_macroturb=None, max_macroturb=None, min_rotation=None, max_rotation=None) -> tuple[np.ndarray, list[tuple]]:
        """
        Gets rv and macroturbulence guess if it is fitted for simplex guess
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

        return guesses, bounds

    def configure_and_run_ts(self, ts:TurboSpectrum, met: float, elem_abund: dict, vmicro: float, lmin: float, lmax: float,
                             windows_flag: bool, temp_dir=None, teff=None):
        """
        Configures TurboSpectrum depending on input parameters and runs either NLTE or LTE
        :param met: metallicity of star
        :param elem_abund: dictionary with iron and elemental abundances
        :param vmicro: microturbulence parameter
        :param lmin: minimum wavelength where spectra are computed
        :param lmax: maximum wavelength where spectra are computed
        :param windows_flag - False for lbl, True for all lines. TODO: uh does windows flag remove calculation of specific elements/molecules from the spectra?
        :param temp_dir: Temporary directory where to save, if not given, then self.temp_dir is used
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
        if self.nlte_flag:
            ts.configure(t_eff=teff, log_g=self.logg, metallicity=met, turbulent_velocity=vmicro,
                         lambda_delta=self.ldelta, lambda_min=lmin, lambda_max=lmax,
                         free_abundances=elem_abund, temp_directory=temp_dir, nlte_flag=True,
                         verbose=self.turbospectrum_verbose,
                         atmosphere_dimension=self.atmosphere_type, windows_flag=windows_flag,
                         segment_file=self.segment_file, line_mask_file=self.linemask_file,
                         depart_bin_file=self.depart_bin_file_dict, depart_aux_file=self.depart_aux_file_dict,
                         model_atom_file=self.model_atom_file_dict)
        else:
            ts.configure(t_eff=teff, log_g=self.logg, metallicity=met, turbulent_velocity=vmicro,
                         lambda_delta=self.ldelta, lambda_min=lmin, lambda_max=lmax,
                         free_abundances=elem_abund, temp_directory=temp_dir, nlte_flag=False,
                         verbose=self.turbospectrum_verbose,
                         atmosphere_dimension=self.atmosphere_type, windows_flag=windows_flag,
                         segment_file=self.segment_file, line_mask_file=self.linemask_file)
        ts.run_turbospectrum_and_atmosphere()

    def fit_all(self) -> str:
        """
        Fit all lines at once, trying to minimise chi squared
        :return: Result is a string containing Fitted star name, abundance, RV, chi squared and macroturbulence
        """
        # timing how long it took
        time_start = time.perf_counter()

        ts = self.create_ts_object()

        function_arguments = (ts, self)
        minimize_options = {'maxiter': self.ndimen * 50, 'disp': self.python_verbose,
                            'initial_simplex': self.initial_simplex_guess, 'xatol': 0.05, 'fatol': 0.05}
        res = minimize_function(all_abund_rv, self.init_param_guess, function_arguments, self.minim_bounds, 'Nelder-Mead', minimize_options)
        # print final result from minimazation
        print(res.x)

        if self.fit_vmac:  # if fitted macroturbulence, return it
            result = f"{self.spec_name} {res.x[0]} {res.x[1]} {res.fun} {res.x[2]}"
        else:  # otherwise return whatever constant macroturbulence was given in the config
            result = f"{self.spec_name} {res.x[0]} {res.x[1]} {res.fun} {self.vmac}"

        time_end = time.perf_counter()
        print(f"Total runtime was {(time_end - time_start) / 60.:2f} minutes.")
        # remove all temporary files
        #shutil.rmtree(self.temp_dir)
        return result

    def create_ts_object(self):
        ts = TurboSpectrum(
            turbospec_path=self.turbospec_path,
            interpol_path=self.interpol_path,
            line_list_paths=self.line_list_path_trimmed,
            marcs_grid_path=self.model_atmosphere_grid_path,
            marcs_grid_list=self.model_atmosphere_list,
            model_atom_path=self.model_atom_path,
            departure_file_path=self.departure_file_path,
            aux_file_length_dict=self.aux_file_length_dict,
            model_temperatures=self.model_temperatures,
            model_logs=self.model_logs,
            model_mets=self.model_mets,
            marcs_value_keys=self.marcs_value_keys,
            marcs_models=self.marcs_models,
            marcs_values=self.marcs_values)
        return ts

    def generate_grid_for_lbl(self, abund_to_gen: np.ndarray) -> list:
        """
        Generates grids for lbl quick method. Grids are centered at input metallicity/abundance. Number of grids and
        bounds depend on self.abund_bound, self.grids_amount
        :return: List corresponding to self.abund_to_gen with same locations. True: generation success. False: not
        """
        success = []

        for abund_to_use in abund_to_gen:
            if self.met > 0.5 or self.met < -4.0 or abund_to_use < -40 or (
                    self.fit_feh and (abund_to_use < -4.0 or abund_to_use > 0.5)):
                success.append(False)  # if metallicity or abundance too weird, then fail
            else:
                if self.fit_feh:
                    item_abund = {"Fe": abund_to_use}
                    met = abund_to_use
                else:
                    item_abund = {"Fe": self.met, self.elem_to_fit[0]: abund_to_use + self.met}
                    met = self.met

                if self.vmic is not None:  # sets microturbulence here
                    vmicro = self.vmic
                else:
                    vmicro = calculate_vturb(self.teff, self.logg, met)

                ts = self.create_ts_object()

                temp_dir = os.path.join(self.temp_dir, f"{abund_to_use}", '')
                #create_dir(temp_dir)
                self.configure_and_run_ts(ts, met, item_abund, vmicro, self.lmin, self.lmax, False, temp_dir=temp_dir)

                if os_path.exists(f"{temp_dir}spectrum_00000000.spec") and \
                        os.stat(f"{temp_dir}spectrum_00000000.spec").st_size != 0:
                    success.append(True)
                else:
                    success.append(False)
        return success

    def fit_lbl_quick(self) -> list:
        """
        lbl quick called here. It generates grids based on input parameters and then tries to find best chi-squared for
        each grid (with best fit doppler shift and if requested macroturbulence). Then it gives chi-squared and best
        fit parameters for each grid point. Also saves spectra for best fit chi squared for corresponding abundances.
        :return: List full of grid parameters with corresponding best fit values and chi squared
        """
        print("Generating grids")
        if self.fit_feh:  # grids generated centered on input metallicity or abundance
            input_abund = self.met
        else:
            input_abund = self.elem_abund_input
        self.abund_to_gen = np.linspace(input_abund - self.abund_bound, input_abund + self.abund_bound,
                                        self.grids_amount)
        success_grid_gen = self.generate_grid_for_lbl(self.abund_to_gen)  # generate grids
        print("Generation successful")
        result = []
        grid_spectra = {}
        # read spectra from generated grids and keep in memory to not waste time reading them each time
        for abund, success in zip(self.abund_to_gen, success_grid_gen):
            if success:
                spectra_grid_path = os.path.join(self.temp_dir, f"{abund}", '')
                wave_mod_orig, flux_mod_orig = np.loadtxt(f'{spectra_grid_path}/spectrum_00000000.spec',
                                                          usecols=(0, 1), unpack=True)
                grid_spectra[abund] = [wave_mod_orig, flux_mod_orig]

        for j in range(len(self.line_begins_sorted)):
            # each line contains spectra name and fitted line. then to the right of it abundance with chi-sqr are added
            result_one_line = f"{self.spec_name} {self.line_centers_sorted[j]} {self.line_begins_sorted[j]} " \
                              f"{self.line_ends_sorted[j]}"

            chi_squares = []

            for abund, success in zip(self.abund_to_gen, success_grid_gen):
                if success:  # for each successful grid find chi squared with best fit parameters
                    wave_abund, flux_abund = grid_spectra[abund][0], grid_spectra[abund][1]
                    function_argsuments=(self, self.line_begins_sorted[j] - 5., self.line_ends_sorted[j] + 5.,
                                         wave_abund, flux_abund)
                    minimize_options = {'maxiter': self.ndimen * 50, 'disp': False}
                    res = minimize_function(lbl_rv_vmac_rot, self.init_param_guess,
                                            function_argsuments, self.minim_bounds, 'L-BFGS-B',
                                            minimize_options)
                    #print(res.x)
                    if self.fit_vmac:  # if fitted macroturbulence
                        macroturb = res.x[1]
                    else:
                        macroturb = self.vmac
                    if self.vmic is not None:  # if microturbulence was given or finds whatever input was used
                        vmicro = self.vmic
                    else:
                        if self.fit_feh:
                            met = abund
                        else:
                            met = self.met
                        vmicro = calculate_vturb(self.teff, self.logg, met)
                    result_one_line += f" {abund} {res.x[0]} {vmicro} {macroturb} {res.fun}"  # saves additionally here
                    chi_squares.append(res.fun)
                else:
                    #print(f"Abundance {abund} did not manage to generate a grid")  # if no grid was generated
                    result_one_line += f" {abund} {9999} {9999} {9999} {9999}"
                    chi_squares.append(9999)

            result.append(result_one_line)
            # finds best fit chi squared for the line and corresponding abundance
            # 01.12.2022 NS: removed the next few lines, because takes 10 MB/star, which is quite a bit
            """index_min_chi_square = np.argmin(chi_squares)
            min_chi_sqr_spectra_path = os.path.join(self.temp_dir, f"{self.abund_to_gen[index_min_chi_square]}",
                                                    'spectrum_00000000.spec')
            # appends that best fit spectra to the total output spectra. NOT convolved. separate abundance for each line
            wave_result, flux_norm_result, flux_result = np.loadtxt(min_chi_sqr_spectra_path,
                                                                    unpack=True)  # TODO asyncio here? or just print at the very end?
            with open(f"{self.output_folder}result_spectrum_{self.spec_name}.spec", 'a') as g:
                # g = open(f"{self.output_folder}result_spectrum_{self.spec_name}.spec", 'a')
                for k in range(len(wave_result)):
                    print("{}  {}  {}".format(wave_result[k], flux_norm_result[k], flux_result[k]), file=g)
            """
            #time_end = time.perf_counter()
            #print("Total runtime was {:.2f} minutes.".format((time_end - time_start) / 60.))

        # g.close()

        return result

    def fit_lbl(self) -> list:
        """
        Fits line by line, by going through each line in the linelist and computing best abundance/met with chi sqr.
        Also fits doppler shift and can fit micro and macro turbulence. New method, faster and more accurate TM.
        :return: List with the results. Each element is a string containing file name, center start and end of the line,
        Best fit abundance/met, doppler shift, microturbulence, macroturbulence and chi-squared.
        """
        if self.fit_vmac and self.vmac == 0:
            self.vmac = 10

        result = {}

        if self.dask_workers > 1 and self.experimental_parallelisation:
            #TODO EXPERIMENTAL attempt: will make it way faster for single/few star fitting with many lines
            # Maybe Dask will break this in the future? Then remove whatever within this if statement
            client = get_client()
            for line_number in range(len(self.line_begins_sorted)):

                res1 = client.submit(self.fit_one_line, line_number)
                result[line_number] = res1

            secede()
            result = client.gather(result)
            rejoin()
        else:
            for line_number in range(len(self.line_begins_sorted)):
                time_start = time.perf_counter()
                print(f"Fitting line at {self.line_centers_sorted[line_number]} angstroms")

                result[line_number] = self.fit_one_line(line_number)

                time_end = time.perf_counter()
                print("Total runtime was {:.2f} minutes.".format((time_end - time_start) / 60.))

        # g.close()
        # h.close()

        result_list = []
        #{"result": , "fit_wavelength": , "fit_flux_norm": , "fit_flux": , "fit_wavelength_conv": , "fit_flux_norm_conv": }
        for line_number in range(len(self.line_begins_sorted)):
            if len(result[line_number]["fit_wavelength"]) > 0 and result[line_number]["chi_sqr"] < 999:
                with open(os.path.join(self.output_folder, f"result_spectrum_{self.spec_name}.spec"), 'a') as g:
                    # g = open(f"{self.output_folder}result_spectrum_{self.spec_name}.spec", 'a')
                    for k in range(len(result[line_number]["fit_wavelength"])):
                        print(f"{result[line_number]['fit_wavelength'][k]} {result[line_number]['fit_flux_norm'][k]} {result[line_number]['fit_flux'][k]}", file=g)

                line_left, line_right = self.line_begins_sorted[line_number], self.line_ends_sorted[line_number]

                wavelength_fit_array = result[line_number]['fit_wavelength']
                norm_flux_fit_array = result[line_number]['fit_flux_norm']

                indices_to_use_cut = np.where((wavelength_fit_array <= line_right + 5) & (wavelength_fit_array >= line_left - 5))
                wavelength_fit_array_cut, norm_flux_fit_array_cut = wavelength_fit_array[indices_to_use_cut], norm_flux_fit_array[indices_to_use_cut]
                wavelength_fit_conv, flux_fit_conv = get_convolved_spectra(wavelength_fit_array_cut, norm_flux_fit_array_cut, self.resolution, result[line_number]["macroturb"], result[line_number]["rotation"])

                equivalent_width = calculate_equivalent_width(wavelength_fit_conv, flux_fit_conv, line_left, line_right)

                extra_wavelength_to_save = 1  # AA extra wavelength to save left and right of the line

                # this will save extra +/- extra_wavelength_to_save in convolved spectra. But just so that it doesn't
                # overlap other lines, I save only up to half of the other linemask if they are close enough
                if line_number > 0:
                    line_previous_right = self.line_ends_sorted[line_number - 1]
                    left_bound_to_save = max(line_left - extra_wavelength_to_save, (line_left - line_previous_right) / 2 + line_previous_right)
                else:
                    left_bound_to_save = line_left - extra_wavelength_to_save
                if line_number < len(self.line_begins_sorted) - 1:
                    line_next_left = self.line_begins_sorted[line_number + 1]
                    right_bound_to_save = min(line_right + extra_wavelength_to_save, (line_next_left - line_right) / 2 + line_right)
                else:
                    right_bound_to_save = line_right + extra_wavelength_to_save
                indices_to_save_conv = np.logical_and.reduce((wavelength_fit_conv > left_bound_to_save, wavelength_fit_conv < right_bound_to_save))

                with open(os.path.join(self.output_folder, f"result_spectrum_{self.spec_name}_convolved.spec"), 'a') as h:
                    # h = open(f"{self.output_folder}result_spectrum_{self.spec_name}_convolved.spec", 'a')
                    for k in range(len(wavelength_fit_conv[indices_to_save_conv])):
                        print(f"{wavelength_fit_conv[indices_to_save_conv][k]} {flux_fit_conv[indices_to_save_conv][k]}", file=h)
            else:
                equivalent_width = 9999
            result_list.append(f"{result[line_number]['result']} {equivalent_width * 1000}")

        return result_list

    def fit_teff_function(self) -> list:
        """
        Fits line by line, by going through each line in the linelist and computing best abundance/met with chi sqr.
        Also fits doppler shift and can fit micro and macro turbulence. New method, faster and more accurate TM.
        :return: List with the results. Each element is a string containing file name, center start and end of the line,
        Best fit abundance/met, doppler shift, microturbulence, macroturbulence and chi-squared.
        """
        if self.fit_vmac and self.vmac == 0:
            self.vmac = 10

        result = []

        for line_number in range(len(self.line_begins_sorted)):
            time_start = time.perf_counter()
            print(f"Fitting line at {self.line_centers_sorted[line_number]} angstroms")

            result.append(self.fit_teff_one_line(line_number))

            time_end = time.perf_counter()
            print("Total runtime was {:.2f} minutes.".format((time_end - time_start) / 60.))

        # g.close()
        # h.close()

        return result


    def fit_teff_one_line(self, line_number: int) -> str:
        """
        Fits a single line by first calling abundance calculation and inside it fitting macro + doppler shift
        :param line_number: Which line number/index in line_center_sorted is being fitted
        :return: best fit result string for that line
        """
        start = np.where(np.logical_and(self.seg_begins <= self.line_centers_sorted[line_number],
                                        self.line_centers_sorted[line_number] <= self.seg_ends))[0][0]
        print(self.line_centers_sorted[line_number], self.seg_begins[start], self.seg_ends[start])

        param_guess = np.array([[self.teff + self.guess_plus_minus_neg_teff], [self.teff + self.guess_plus_minus_pos_teff]])
        min_bounds = [(self.bound_min_teff, self.bound_max_teff)]

        ts = self.create_ts_object()

        ts.line_list_paths = [get_trimmed_lbl_path_name(self.line_list_path_trimmed, start)]

        function_argsuments = (ts, self, self.line_begins_sorted[line_number] - 5., self.line_ends_sorted[line_number] + 5.)
        minimize_options = {'maxfev': 50, 'disp': self.python_verbose, 'initial_simplex': param_guess, 'xatol': 0.01, 'fatol': 0.01}
        res = minimize_function(lbl_teff, param_guess[0], function_argsuments, min_bounds, 'Nelder-Mead', minimize_options)

        print(res.x)

        teff = res.x[0]

        met = self.met
        doppler_fit = self.doppler_shift
        if self.vmic is not None:  # Input given
            microturb = self.vmic
        else:
            microturb = calculate_vturb(self.teff, self.logg, met)

        macroturb = self.vmac
        result_output = f"{self.spec_name} {teff} {self.line_centers_sorted[line_number]} {self.line_begins_sorted[line_number]} " \
                        f"{self.line_ends_sorted[line_number]} {doppler_fit} {microturb} {macroturb} {res.fun}"

        one_result = result_output  # out = open(f"{temp_directory}spectrum_00000000_convolved.spec", 'w')
        try:
            wave_result, flux_norm_result, flux_result = np.loadtxt(f"{self.temp_dir}spectrum_00000000.spec",
                                                                    unpack=True)
            with open(os.path.join(self.output_folder, f"result_spectrum_{self.spec_name}.spec"), 'a') as g:
                # g = open(f"{self.output_folder}result_spectrum_{self.spec_name}.spec", 'a')
                for k in range(len(wave_result)):
                    print("{}  {}  {}".format(wave_result[k], flux_norm_result[k], flux_result[k]), file=g)
            wave_result, flux_norm_result = np.loadtxt(f"{self.temp_dir}spectrum_00000000_convolved.spec", unpack=True)
            with open(os.path.join(self.output_folder, f"result_spectrum_{self.spec_name}_convolved.spec"), 'a') as h:
                # h = open(f"{self.output_folder}result_spectrum_{self.spec_name}_convolved.spec", 'a')
                for k in range(len(wave_result)):
                    print("{}  {}".format(wave_result[k], flux_norm_result[k]), file=h)
            # os.system("rm ../output_files/spectrum_{:08d}_convolved.spec".format(i + 1))
        except (OSError, ValueError) as error:
            print("Failed spectra generation completely, line is not fitted at all, not saving spectra then")
        return one_result

    def fit_one_line(self, line_number: int) -> dict:
        """
        Fits a single line by first calling abundance calculation and inside it fitting macro + doppler shift
        :param line_number: Which line number/index in line_center_sorted is being fitted
        :return: best fit result string for that line
        """
        temp_directory = os.path.join(self.temp_dir, str(np.random.random()), "")

        ts = self.create_ts_object()

        start = np.where(np.logical_and(self.seg_begins <= self.line_centers_sorted[line_number],
                                        self.line_centers_sorted[line_number] <= self.seg_ends))[0][0]
        print(self.line_centers_sorted[line_number], self.seg_begins[start], self.seg_ends[start])
        ts.line_list_paths = [get_trimmed_lbl_path_name(self.line_list_path_trimmed, start)]

        param_guess, min_bounds = self.get_elem_micro_guess(self.guess_min_vmic, self.guess_max_vmic, self.guess_min_abund, self.guess_max_abund)

        function_arguments = (ts, self, self.line_begins_sorted[line_number] - 5., self.line_ends_sorted[line_number] + 5., temp_directory, line_number)
        minimization_options = {'maxfev': self.nelement * 100, 'disp': self.python_verbose, 'initial_simplex': param_guess, 'xatol': 0.005, 'fatol': 0.000001, 'adaptive': True}
        res = minimize_function(lbl_abund_vmic, param_guess[0], function_arguments, min_bounds, 'Nelder-Mead', minimization_options)
        print(res.x)
        if self.fit_feh:
            met_index = np.where(self.elem_to_fit == "Fe")[0][0]
            met = res.x[met_index]
        else:
            met = self.met
        elem_abund_dict = {"Fe": met}
        for i in range(self.nelement):
            # self.elem_to_fit[i] = element name
            # param[1:nelement] = abundance of the element
            elem_name = self.elem_to_fit[i]
            if elem_name != "Fe":
                elem_abund_dict[elem_name] = res.x[i]  # + met
        doppler_fit = self.doppler_shift_dict[line_number]
        if self.vmic is not None:  # Input given
            microturb = self.vmic
        else:
            if self.fit_vmic == "No" and self.atmosphere_type == "1D":
                microturb = calculate_vturb(self.teff, self.logg, met)
            elif self.fit_vmic == "Yes" and self.atmosphere_type == "1D":
                microturb = res.x[-1]  # if no macroturb fit, then last param is microturb
            elif self.fit_vmic == "Input":  # just for safety's sake, normally should take in the input above anyway
                raise ValueError(
                    "Microturb not given? Did you remember to set microturbulence in parameters? Or is there "
                    "a problem in the code?")
            else:
                microturb = 2.0
        if self.fit_vmac:
            macroturb = self.vmac_dict[line_number]
        else:
            macroturb = self.vmac
        if self.fit_rotation:
            rotation = self.rotation_dict[line_number]
        else:
            rotation = self.rotation
        result_output = f"{self.spec_name} {self.line_centers_sorted[line_number]} {self.line_begins_sorted[line_number]} " \
                        f"{self.line_ends_sorted[line_number]} {doppler_fit}"
        for key in elem_abund_dict:
            result_output += f" {elem_abund_dict[key]}"
        result_output += f" {microturb} {macroturb} {rotation} {res.fun}"
        one_result = result_output  # out = open(f"{temp_directory}spectrum_00000000_convolved.spec", 'w')
        try:
            wave_result, flux_norm_result, flux_result = np.loadtxt(f"{temp_directory}spectrum_00000000.spec",
                                                                    unpack=True)
            #with open(f"{self.output_folder}result_spectrum_{self.spec_name}.spec", 'a') as g:
                # g = open(f"{self.output_folder}result_spectrum_{self.spec_name}.spec", 'a')
            #    for k in range(len(wave_result)):
            #        print("{}  {}  {}".format(wave_result[k], flux_norm_result[k], flux_result[k]), file=g)
            #wave_result_conv, flux_norm_result_conv = np.loadtxt(f"{temp_directory}spectrum_00000000_convolved.spec", unpack=True)
            #with open(f"{self.output_folder}result_spectrum_{self.spec_name}_convolved.spec", 'a') as h:
                # h = open(f"{self.output_folder}result_spectrum_{self.spec_name}_convolved.spec", 'a')
            #    for k in range(len(wave_result)):
            #        print("{}  {}".format(wave_result[k], flux_norm_result[k]), file=h)
            # os.system("rm ../output_files/spectrum_{:08d}_convolved.spec".format(i + 1))
        except (OSError, ValueError) as error:
            print(f"{error} Failed spectra generation completely, line is not fitted at all, not saving spectra then")
            wave_result = np.array([])
            flux_norm_result = np.array([])
            flux_result = np.array([])
        shutil.rmtree(temp_directory)
        return {"result": one_result, "fit_wavelength": wave_result, "fit_flux_norm": flux_norm_result,
                "fit_flux": flux_result,  "macroturb": macroturb, "rotation": rotation, "chi_sqr": res.fun} #"fit_wavelength_conv": wave_result_conv, "fit_flux_norm_conv": flux_norm_result_conv,

    def fit_one_line_vmic(self, line_number: int) -> dict:
        """
        Fits a single line by first calling abundance calculation and inside it fitting macro + doppler shift
        :param line_number: Which line number/index in line_center_sorted is being fitted
        :return: best fit result string for that line
        """
        temp_directory = os.path.join(self.temp_dir, str(np.random.random()), "")

        ts = self.create_ts_object()

        start = np.where(np.logical_and(self.seg_begins <= self.line_centers_sorted[line_number],
                                        self.line_centers_sorted[line_number] <= self.seg_ends))[0][0]
        print(self.line_centers_sorted[line_number], self.seg_begins[start], self.seg_ends[start])
        ts.line_list_paths = [get_trimmed_lbl_path_name(self.line_list_path_trimmed, start)]

        param_guess, min_bounds = self.get_elem_guess(self.guess_min_abund, self.guess_max_abund)

        function_arguments = (ts, self, self.line_begins_sorted[line_number] - 5., self.line_ends_sorted[line_number] + 5., temp_directory, line_number)
        minimization_options = {'maxfev': self.nelement * 100, 'disp': self.python_verbose, 'initial_simplex': param_guess, 'xatol': 0.005, 'fatol': 0.000001, 'adaptive': False}
        res = minimize_function(lbl_abund, param_guess[0], function_arguments, min_bounds, 'Nelder-Mead', minimization_options)
        print(res.x)
        if self.fit_feh:
            met_index = np.where(self.elem_to_fit == "Fe")[0][0]
            met = res.x[met_index]
        else:
            met = self.met
        elem_abund_dict = {"Fe": met}
        for i in range(self.nelement):
            # self.elem_to_fit[i] = element name
            # param[1:nelement] = abundance of the element
            elem_name = self.elem_to_fit[i]
            if elem_name != "Fe":
                elem_abund_dict[elem_name] = res.x[i]  # + met
        doppler_fit = self.doppler_shift_dict[line_number]
        microturb = self.vmic_dict[line_number]
        if self.fit_vmac:
            macroturb = self.vmac_dict[line_number]
        else:
            macroturb = self.vmac
        if self.fit_rotation:
            rotation = self.rotation_dict[line_number]
        else:
            rotation = self.rotation
        result_output = f"{self.spec_name} {self.line_centers_sorted[line_number]} {self.line_begins_sorted[line_number]} " \
                        f"{self.line_ends_sorted[line_number]} {doppler_fit}"
        for key in elem_abund_dict:
            result_output += f" {elem_abund_dict[key]}"
        result_output += f" {microturb} {macroturb} {rotation} {res.fun}"
        one_result = result_output  # out = open(f"{temp_directory}spectrum_00000000_convolved.spec", 'w')
        try:
            wave_result, flux_norm_result, flux_result = np.loadtxt(f"{temp_directory}spectrum_00000000.spec",
                                                                    unpack=True)
            #with open(f"{self.output_folder}result_spectrum_{self.spec_name}.spec", 'a') as g:
                # g = open(f"{self.output_folder}result_spectrum_{self.spec_name}.spec", 'a')
            #    for k in range(len(wave_result)):
            #        print("{}  {}  {}".format(wave_result[k], flux_norm_result[k], flux_result[k]), file=g)
            #wave_result_conv, flux_norm_result_conv = np.loadtxt(f"{temp_directory}spectrum_00000000_convolved.spec", unpack=True)
            #with open(f"{self.output_folder}result_spectrum_{self.spec_name}_convolved.spec", 'a') as h:
                # h = open(f"{self.output_folder}result_spectrum_{self.spec_name}_convolved.spec", 'a')
            #    for k in range(len(wave_result)):
            #        print("{}  {}".format(wave_result[k], flux_norm_result[k]), file=h)
            # os.system("rm ../output_files/spectrum_{:08d}_convolved.spec".format(i + 1))
        except (OSError, ValueError) as error:
            print(f"{error} Failed spectra generation completely, line is not fitted at all, not saving spectra then")
            wave_result = np.array([])
            flux_norm_result = np.array([])
            flux_result = np.array([])
        shutil.rmtree(temp_directory)
        return {"result": one_result, "fit_wavelength": wave_result, "fit_flux_norm": flux_norm_result,
                "fit_flux": flux_result,  "macroturb": macroturb, "rotation": rotation, "chi_sqr": res.fun} #"fit_wavelength_conv": wave_result_conv, "fit_flux_norm_conv": flux_norm_result_conv,


    def fit_vmic_slow(self):
        # this is a slow version of the fit_vmic = True function, but it is more accurate
        if self.fit_vmac and self.vmac == 0:
            self.vmac = 10

        result = {}

        if self.dask_workers > 1 and self.experimental_parallelisation:
            #TODO EXPERIMENTAL attempt: will make it way faster for single/few star fitting with many lines
            # Maybe Dask will break this in the future? Then remove whatever within this if statement
            client = get_client()
            for line_number in range(len(self.line_begins_sorted)):

                res1 = client.submit(self.fit_one_line_vmic, line_number)
                result[line_number] = res1

            secede()
            result = client.gather(result)
            rejoin()
        else:
            for line_number in range(len(self.line_begins_sorted)):
                time_start = time.perf_counter()
                print(f"Fitting line at {self.line_centers_sorted[line_number]} angstroms")

                result[line_number] = self.fit_one_line_vmic(line_number)

                time_end = time.perf_counter()
                print("Total runtime was {:.2f} minutes.".format((time_end - time_start) / 60.))

        # g.close()
        # h.close()

        result_list = []
        #{"result": , "fit_wavelength": , "fit_flux_norm": , "fit_flux": , "fit_wavelength_conv": , "fit_flux_norm_conv": }
        for line_number in range(len(self.line_begins_sorted)):
            if len(result[line_number]["fit_wavelength"]) > 0 and result[line_number]["chi_sqr"] < 999:
                with open(os.path.join(self.output_folder, f"result_spectrum_{self.spec_name}.spec"), 'a') as g:
                    # g = open(f"{self.output_folder}result_spectrum_{self.spec_name}.spec", 'a')
                    for k in range(len(result[line_number]["fit_wavelength"])):
                        print(f"{result[line_number]['fit_wavelength'][k]} {result[line_number]['fit_flux_norm'][k]} {result[line_number]['fit_flux'][k]}", file=g)

                line_left, line_right = self.line_begins_sorted[line_number], self.line_ends_sorted[line_number]

                wavelength_fit_array = result[line_number]['fit_wavelength']
                norm_flux_fit_array = result[line_number]['fit_flux_norm']

                indices_to_use_cut = np.where((wavelength_fit_array <= line_right + 5) & (wavelength_fit_array >= line_left - 5))
                wavelength_fit_array_cut, norm_flux_fit_array_cut = wavelength_fit_array[indices_to_use_cut], norm_flux_fit_array[indices_to_use_cut]
                wavelength_fit_conv, flux_fit_conv = get_convolved_spectra(wavelength_fit_array_cut, norm_flux_fit_array_cut, self.resolution, result[line_number]["macroturb"], result[line_number]["rotation"])

                equivalent_width = calculate_equivalent_width(wavelength_fit_conv, flux_fit_conv, line_left, line_right)

                extra_wavelength_to_save = 1  # AA extra wavelength to save left and right of the line

                # this will save extra +/- extra_wavelength_to_save in convolved spectra. But just so that it doesn't
                # overlap other lines, I save only up to half of the other linemask if they are close enough
                if line_number > 0:
                    line_previous_right = self.line_ends_sorted[line_number - 1]
                    left_bound_to_save = max(line_left - extra_wavelength_to_save, (line_left - line_previous_right) / 2 + line_previous_right)
                else:
                    left_bound_to_save = line_left - extra_wavelength_to_save
                if line_number < len(self.line_begins_sorted) - 1:
                    line_next_left = self.line_begins_sorted[line_number + 1]
                    right_bound_to_save = min(line_right + extra_wavelength_to_save, (line_next_left - line_right) / 2 + line_right)
                else:
                    right_bound_to_save = line_right + extra_wavelength_to_save
                indices_to_save_conv = np.logical_and.reduce((wavelength_fit_conv > left_bound_to_save, wavelength_fit_conv < right_bound_to_save))

                with open(os.path.join(self.output_folder, f"result_spectrum_{self.spec_name}_convolved.spec"), 'a') as h:
                    # h = open(f"{self.output_folder}result_spectrum_{self.spec_name}_convolved.spec", 'a')
                    for k in range(len(wavelength_fit_conv[indices_to_save_conv])):
                        print(f"{wavelength_fit_conv[indices_to_save_conv][k]} {flux_fit_conv[indices_to_save_conv][k]}", file=h)
            else:
                equivalent_width = 9999
            result_list.append(f"{result[line_number]['result']} {equivalent_width * 1000}")

        return result_list


def get_second_degree_polynomial(x: list, y: list) -> tuple[int, int, int]:
    """
    Takes a list of x and y of length 3 each and calculates perfectly fitted second degree polynomial through it.
    Returns a, b, c that are related to the function ax^2+bx+c = y
    :param x: x values, length 3
    :param y: y values, length 3
    :return a,b,c -> ax^2+bx+c = y 2nd degree polynomial
    """
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    y1 = y[0]
    y2 = y[1]
    y3 = y[2]

    a = (x1 * (y3 - y2) + x2 * (y1 - y3) + x3 * (y2 - y1)) / ((x1 - x2) * (x1 - x3) * (x2 - x3))
    b = (y2 - y1) / (x2 - x1) - a * (x1 + x2)
    c = y1 - a * x1 * x1 - b * x2

    return a, b, c

def lbl_rv_vmac_rot(param: list, spectra_to_fit: Spectra, lmin: float, lmax: float,
                    wave_mod_orig: np.ndarray, flux_mod_orig: np.ndarray) -> float:
    """
    Line by line quick. Takes precalculated synthetic spectra (i.e. 1 grid) and finds chi-sqr for observed spectra.
    Also fits doppler shift and can fit macroturbulence if needed.
    :param param: Parameters list with the current evaluation guess
    :param spectra_to_fit: Spectra to fit
    :param lmin: Start of the line [AA]
    :param lmax: End of the line [AA]
    :param wave_mod_orig: Wavelength of synthetic spectra
    :param flux_mod_orig: Flux of synthetic spectra
    :return: Best fit chi squared
    """
    # param[0] = doppler
    # param[1] = macro turb
    # param[-1] = rotation fit

    doppler = spectra_to_fit.rv + param[0]

    if spectra_to_fit.fit_vmac:
        macroturb = param[1]
    else:
        macroturb = spectra_to_fit.vmac

    if spectra_to_fit.fit_rotation:
        rotation = param[-1]
    else:
        rotation = spectra_to_fit.rotation

    wave_ob = apply_doppler_correction(spectra_to_fit.wave_ob, doppler)


    try:
        chi_square = calculate_lbl_chi_squared(None, wave_ob, spectra_to_fit.flux_ob, wave_mod_orig, flux_mod_orig,
                                               spectra_to_fit.resolution, lmin, lmax, macroturb, rotation,
                                               save_convolved=False)
    except IndexError as e:
        chi_square = 9999.99
        print(f"{e} Is your segment seen in the observed spectra?")
    #print(param[0], chi_square, macroturb)  # takes 50%!!!! extra time to run if using print statement here

    return chi_square


def apply_doppler_correction(wave_ob: np.ndarray, doppler: float) -> np.ndarray:
    return wave_ob / (1 + (doppler / 299792.))


def lbl_abund_vmic(param: list, ts: TurboSpectrum, spectra_to_fit: Spectra, lmin: float, lmax: float, temp_directory: str, line_number: int) -> float:
    """
    Goes line by line, tries to call turbospectrum and find best fit spectra by varying parameters: abundance, doppler
    shift and if needed micro + macro turbulence. This specific function handles abundance + micro. Calls macro +
    doppker inside
    :param param: Parameters list with the current evaluation guess
    :param spectra_to_fit: Spectra to fit
    :param lmin: Start of the line [AA]
    :param lmax: End of the line [AA]
    :return: best fit chi squared
    """
    # new: now includes several elements
    # param[-1] = vmicro
    # param[0:nelements - 1] = met or abund

    if spectra_to_fit.fit_feh:
        met_index = np.where(spectra_to_fit.elem_to_fit == "Fe")[0][0]
        met = param[met_index]  # no offset, first is always element
    else:
        met = spectra_to_fit.met
    elem_abund_dict = {"Fe": met}

    #abundances = [met]

    for i in range(spectra_to_fit.nelement):
        # spectra_to_fit.elem_to_fit[i] = element name
        # param[0:nelement - 1] = abundance of the element
        elem_name = spectra_to_fit.elem_to_fit[i]
        if elem_name != "Fe":
            elem_abund_dict[elem_name] = param[i] + met
            #abundances.append(param[i])

    for element in spectra_to_fit.input_abund:
        elem_abund_dict[element] = spectra_to_fit.input_abund[element] + met

    if spectra_to_fit.vmic is not None:  # Input given
        microturb = spectra_to_fit.vmic
    else:
        if spectra_to_fit.fit_vmic == "No" and spectra_to_fit.atmosphere_type == "1D":
            microturb = calculate_vturb(spectra_to_fit.teff, spectra_to_fit.logg, met)
        elif spectra_to_fit.fit_vmic == "Yes" and spectra_to_fit.atmosphere_type == "1D":
            microturb = param[-1]
        elif spectra_to_fit.fit_vmic == "Input":  # just for safety's sake, normally should take in the input above anyway
            raise ValueError("Microturb not given? Did you remember to set microturbulence in parameters? Or is there "
                             "a problem in the code?")
        else:
            microturb = 2.0

    macroturb = 9999    # for printing only here, in case not fitted
    rotation = 9999
    doppler_shift = 9999
    spectra_to_fit.doppler_shift_dict[line_number] = doppler_shift
    spectra_to_fit.vmac_dict[line_number] = macroturb
    spectra_to_fit.rotation_dict[line_number] = rotation

    spectra_to_fit.configure_and_run_ts(ts, met, elem_abund_dict, microturb, lmin, lmax, False, temp_dir=temp_directory)     # generates spectra

    if os_path.exists('{}/spectrum_00000000.spec'.format(temp_directory)) and os.stat(
            '{}/spectrum_00000000.spec'.format(temp_directory)).st_size != 0:
        wave_mod_orig, flux_mod_orig = np.loadtxt(f'{temp_directory}/spectrum_00000000.spec',
                                                  usecols=(0, 1), unpack=True)
        param_guess, min_bounds = spectra_to_fit.get_rv_macro_rotation_guess(min_macroturb=spectra_to_fit.guess_min_vmac, max_macroturb=spectra_to_fit.guess_max_vmac)
        # now for the generated abundance it tries to fit best fit macro + doppler shift.
        # Thus, macro should not be dependent on the abundance directly, hopefully
        # Seems to work way better
        function_args = (spectra_to_fit, lmin, lmax, wave_mod_orig, flux_mod_orig)
        minimize_options = {'maxiter': spectra_to_fit.ndimen * 50, 'disp': False}
        res = minimize_function(lbl_rv_vmac_rot, np.median(param_guess, axis=0),
                                function_args, min_bounds, 'L-BFGS-B', minimize_options)

        spectra_to_fit.doppler_shift_dict[line_number] = res.x[0]
        doppler_shift = spectra_to_fit.doppler_shift_dict[line_number]
        if spectra_to_fit.fit_vmac:
            spectra_to_fit.vmac_dict[line_number] = res.x[1]
            macroturb = spectra_to_fit.vmac_dict[line_number]
        else:
            macroturb = spectra_to_fit.vmac
        if spectra_to_fit.fit_rotation:
            spectra_to_fit.rotation_dict[line_number] = res.x[-1]
            rotation = spectra_to_fit.rotation_dict[line_number]
        else:
            rotation = spectra_to_fit.rotation
        chi_square = res.fun
    elif os_path.exists('{}/spectrum_00000000.spec'.format(temp_directory)) and os.stat(
            '{}/spectrum_00000000.spec'.format(temp_directory)).st_size == 0:
        chi_square = 999.99
        print("empty spectrum file.")
    else:
        chi_square = 9999.9999
        print("didn't generate spectra or atmosphere")

    output_print = f""
    for key in elem_abund_dict:
        output_print += f" [{key}/H]={elem_abund_dict[key]}"
    print(f"{output_print} rv={doppler_shift} vmic={microturb} vmac={macroturb} rotation={rotation} chisqr={chi_square}")

    return chi_square

def lbl_abund(param: list, ts: TurboSpectrum, spectra_to_fit: Spectra, lmin: float, lmax: float, temp_directory: str, line_number: int) -> float:
    """
    Goes line by line, tries to call turbospectrum and find best fit spectra by varying parameters: abundance, doppler
    shift and if needed micro + macro turbulence. This specific function handles abundance + micro. Calls macro +
    doppker inside
    :param param: Parameters list with the current evaluation guess
    :param spectra_to_fit: Spectra to fit
    :param lmin: Start of the line [AA]
    :param lmax: End of the line [AA]
    :return: best fit chi squared
    """
    # new: now includes several elements
    # param[0:nelements - 1] = met or abund

    if spectra_to_fit.fit_feh:
        met_index = np.where(spectra_to_fit.elem_to_fit == "Fe")[0][0]
        met = param[met_index]  # no offset, first is always element
    else:
        met = spectra_to_fit.met
    elem_abund_dict = {"Fe": met}

    #abundances = [met]

    for i in range(spectra_to_fit.nelement):
        # spectra_to_fit.elem_to_fit[i] = element name
        # param[0:nelement - 1] = abundance of the element
        elem_name = spectra_to_fit.elem_to_fit[i]
        if elem_name != "Fe":
            elem_abund_dict[elem_name] = param[i] + met
            #abundances.append(param[i])

    for element in spectra_to_fit.input_abund:
        elem_abund_dict[element] = spectra_to_fit.input_abund[element] + met

    spectra_to_fit.elem_abund_dict_fitting[line_number] = elem_abund_dict

    param_guess, min_bounds = spectra_to_fit.get_micro_guess(spectra_to_fit.guess_min_vmic, spectra_to_fit.guess_max_vmic)
    function_arguments = (ts, spectra_to_fit, spectra_to_fit.line_begins_sorted[line_number] - 5., spectra_to_fit.line_ends_sorted[line_number] + 5., temp_directory, line_number)
    minimization_options = {'maxfev': 50, 'disp': spectra_to_fit.python_verbose, 'initial_simplex': param_guess, 'xatol': 0.05, 'fatol': 0.00001, 'adaptive': False}
    res = minimize_function(lbl_vmic, param_guess[0], function_arguments, min_bounds, 'Nelder-Mead', minimization_options)

    spectra_to_fit.vmic_dict[line_number] = res.x[0]
    microturb = spectra_to_fit.vmic_dict[line_number]
    doppler_shift = spectra_to_fit.doppler_shift_dict[line_number]
    if spectra_to_fit.fit_vmac:
        macroturb = spectra_to_fit.vmac_dict[line_number]
    else:
        macroturb = spectra_to_fit.vmac
    if spectra_to_fit.fit_rotation:
        rotation = spectra_to_fit.rotation_dict[line_number]
    else:
        rotation = spectra_to_fit.rotation
    chi_square = res.fun

    output_print = f""
    for key in elem_abund_dict:
        output_print += f" [{key}/H]={elem_abund_dict[key]}"
    print(f"{output_print} rv={doppler_shift} vmic={microturb} vmac={macroturb} rotation={rotation} chisqr={chi_square}")

    return chi_square

def lbl_vmic(param: list, ts: TurboSpectrum, spectra_to_fit: Spectra, lmin: float, lmax: float, temp_directory: str, line_number: int) -> float:
    """
    Goes line by line, tries to call turbospectrum and find best fit spectra by varying parameters: abundance, doppler
    shift and if needed micro + macro turbulence. This specific function handles abundance + micro. Calls macro +
    doppker inside
    :param param: Parameters list with the current evaluation guess
    :param spectra_to_fit: Spectra to fit
    :param lmin: Start of the line [AA]
    :param lmax: End of the line [AA]
    :return: best fit chi squared
    """
    # param[0] = vmicro

    microturb = param[0]

    macroturb = 9999    # for printing only here, in case not fitted
    rotation = 9999
    doppler_shift = 9999    # for printing only here, in case not fitted
    spectra_to_fit.doppler_shift_dict[line_number] = doppler_shift
    spectra_to_fit.vmac_dict[line_number] = macroturb
    spectra_to_fit.rotation_dict[line_number] = rotation

    met = spectra_to_fit.elem_abund_dict_fitting[line_number]["Fe"]
    elem_abund_dict = spectra_to_fit.elem_abund_dict_fitting[line_number]

    spectra_to_fit.configure_and_run_ts(ts, met, elem_abund_dict, microturb, lmin, lmax, False, temp_dir=temp_directory)     # generates spectra

    if os_path.exists('{}/spectrum_00000000.spec'.format(temp_directory)) and os.stat(
            '{}/spectrum_00000000.spec'.format(temp_directory)).st_size != 0:
        wave_mod_orig, flux_mod_orig = np.loadtxt(f'{temp_directory}/spectrum_00000000.spec',
                                                  usecols=(0, 1), unpack=True)
        param_guess, min_bounds = spectra_to_fit.get_rv_macro_rotation_guess(min_macroturb=spectra_to_fit.guess_min_vmac, max_macroturb=spectra_to_fit.guess_max_vmac)
        function_args = (spectra_to_fit, lmin, lmax, wave_mod_orig, flux_mod_orig)
        minimize_options = {'maxiter': spectra_to_fit.ndimen * 50, 'disp': False}
        res = minimize_function(lbl_rv_vmac_rot, np.median(param_guess, axis=0),
                                function_args, min_bounds, 'L-BFGS-B', minimize_options)

        spectra_to_fit.doppler_shift_dict[line_number] = res.x[0]
        doppler_shift = spectra_to_fit.doppler_shift_dict[line_number]
        if spectra_to_fit.fit_vmac:
            spectra_to_fit.vmac_dict[line_number] = res.x[1]
            macroturb = spectra_to_fit.vmac_dict[line_number]
        else:
            macroturb = spectra_to_fit.vmac
        if spectra_to_fit.fit_rotation:
            spectra_to_fit.rotation_dict[line_number] = res.x[-1]
            rotation = spectra_to_fit.rotation_dict[line_number]
        else:
            rotation = spectra_to_fit.rotation
        chi_square = res.fun
    elif os_path.exists('{}/spectrum_00000000.spec'.format(temp_directory)) and os.stat(
            '{}/spectrum_00000000.spec'.format(temp_directory)).st_size == 0:
        chi_square = 999.99
        print("empty spectrum file.")
    else:
        chi_square = 9999.9999
        print("didn't generate spectra or atmosphere")

    output_print = f""
    for key in elem_abund_dict:
        output_print += f" [{key}/H]={elem_abund_dict[key]}"
    print(
        f"{output_print} rv={doppler_shift} vmic={microturb} vmac={macroturb} rotation={rotation} chisqr={chi_square}")

    return chi_square

def lbl_teff(param: list, ts, spectra_to_fit: Spectra, lmin: float, lmax: float) -> float:
    """
    Goes line by line, tries to call turbospectrum and find best fit spectra by varying parameters: teff.
    Calls macro + doppler inside
    :param param: Parameters list with the current evaluation guess
    :param spectra_to_fit: Spectra to fit
    :param lmin: Start of the line [AA]
    :param lmax: End of the line [AA]
    :return: best fit chi squared
    """
    # param[0] = teff

    teff = param[0]

    if spectra_to_fit.vmic is not None:  # Input given
        microturb = spectra_to_fit.vmic
    else:
        microturb = calculate_vturb(spectra_to_fit.teff, spectra_to_fit.logg, spectra_to_fit.met)

    spectra_to_fit.configure_and_run_ts(ts, spectra_to_fit.met, {"H": 0, "Fe": spectra_to_fit.met}, microturb, lmin, lmax, False, teff=teff)     # generates spectra

    macroturb = 9999  # for printing if fails
    rotation = 9999
    if os_path.exists('{}/spectrum_00000000.spec'.format(spectra_to_fit.temp_dir)) and os.stat(
            '{}/spectrum_00000000.spec'.format(spectra_to_fit.temp_dir)).st_size != 0:
        wave_mod_orig, flux_mod_orig = np.loadtxt(f'{spectra_to_fit.temp_dir}/spectrum_00000000.spec',
                                                  usecols=(0, 1), unpack=True)
        ndimen = 1
        if spectra_to_fit.fit_vmac:
            ndimen += 1
        param_guess, min_bounds = spectra_to_fit.get_rv_macro_rotation_guess(min_macroturb=spectra_to_fit.guess_min_vmac, max_macroturb=spectra_to_fit.guess_max_vmac)
        # now for the generated abundance it tries to fit best fit macro + doppler shift.
        # Thus macro should not be dependent on the abundance directly, hopefully
        # Seems to work way better
        function_args = (spectra_to_fit, lmin, lmax, wave_mod_orig, flux_mod_orig)
        minimize_options = {'maxiter': spectra_to_fit.ndimen * 50, 'disp': False}
        res = minimize_function(lbl_rv_vmac_rot, param_guess[0],
                                function_args, min_bounds, 'L-BFGS-B', minimize_options)

        spectra_to_fit.doppler_shift = res.x[0]
        if spectra_to_fit.fit_vmac:
            spectra_to_fit.vmac = res.x[1]
        macroturb = spectra_to_fit.vmac
        if spectra_to_fit.fit_rotation:
            spectra_to_fit.rotation = res.x[-1]
        rotation = spectra_to_fit.rotation

        chi_square = res.fun
    elif os_path.exists('{}/spectrum_00000000.spec'.format(spectra_to_fit.temp_dir)) and os.stat(
            '{}/spectrum_00000000.spec'.format(spectra_to_fit.temp_dir)).st_size == 0:
        chi_square = 999.99
        print("empty spectrum file.")
    else:
        chi_square = 9999.9999
        print("didn't generate spectra or atmosphere")

    print(f"Teff={teff}, RV={spectra_to_fit.doppler_shift}, micro={microturb}, macro={macroturb}, rotation={rotation}, chisqr={chi_square}")

    return chi_square


def get_trimmed_lbl_path_name(line_list_path_trimmed: str, segment_index: float) -> os.path:
    """
    Gets the name for the lbl trimmed path. Consistent algorithm to always get the same folder name.
    :param line_list_path_trimmed: Path to the trimmed line list
    :param segment_index: Segment's numbering
    :return: path to the folder where to save/already saved trimmed files can exist.
    """
    return os.path.join(line_list_path_trimmed, f"{segment_index}", '')


def all_abund_rv(param, ts, spectra_to_fit: Spectra) -> float:
    """
    Calculates best fit parameters for all lines at once by calling TS and varying abundance/met and doppler shift.
    Can also vary macroturbulence if needed
    :param param: Parameter guess
    :param spectra_to_fit: Spectra to fit
    :return: Best fit chi squared
    """
    # abund = param[0]
    # dopple = param[1]
    # macrorurb = param [2] (if needed)
    abund = param[0]
    doppler = spectra_to_fit.rv + param[1]
    if spectra_to_fit.fit_vmac:
        macroturb = param[2]
    else:
        macroturb = spectra_to_fit.vmac

    #wave_obs = spectra_to_fit.wave_ob / (1 + (doppler / 299792.))
    wave_obs = apply_doppler_correction(spectra_to_fit.wave_ob, doppler)

    if spectra_to_fit.fit_feh:
        item_abund = {"Fe": abund}
        met = abund
        if spectra_to_fit.vmic is not None:
            vmicro = spectra_to_fit.vmic
        else:
            vmicro = calculate_vturb(spectra_to_fit.teff, spectra_to_fit.logg, spectra_to_fit.met)
    else:   # Fe: [Fe/H]. X: [X/Fe]. But TS takes [X/H]. Thus convert [X/H] = [X/Fe] + [Fe/H]
        item_abund = {"Fe": spectra_to_fit.met, spectra_to_fit.elem_to_fit[0]: abund + spectra_to_fit.met}
        met = spectra_to_fit.met
        if spectra_to_fit.vmic is not None:
            vmicro = spectra_to_fit.vmic
        else:
            vmicro = calculate_vturb(spectra_to_fit.teff, spectra_to_fit.logg, spectra_to_fit.met)

    spectra_to_fit.configure_and_run_ts(ts, met, item_abund, vmicro, spectra_to_fit.lmin, spectra_to_fit.lmax, True)

    chi_square = calc_ts_spectra_all_lines(spectra_to_fit.spec_path, spectra_to_fit.temp_dir,
                                           spectra_to_fit.output_folder,
                                           wave_obs, spectra_to_fit.flux_ob,
                                           macroturb, spectra_to_fit.resolution, spectra_to_fit.rotation,
                                           spectra_to_fit.line_begins_sorted, spectra_to_fit.line_ends_sorted,
                                           spectra_to_fit.seg_begins, spectra_to_fit.seg_ends)

    #print(abund, doppler, chi_square, macroturb)

    return chi_square


def create_and_fit_spectra(specname: str, teff: float, logg: float, rv: float, met: float, microturb: float,
                           macroturb: float, line_list_path_trimmed: str, input_abundance: float, index: float,
                           tsfitpy_pickled_configuration_path: str) -> list:
    """
    Creates spectra object and fits based on requested fitting mode
    :param specname: Name of the textfile
    :param teff: Teff in K
    :param logg: logg in dex
    :param rv: radial velocity (km/s)
    :param met: metallicity (doesn't matter what if fitting for Fe)
    :param microturb: Microturbulence if given (None is not known or fitted)
    :param macroturb: Macroturbulence if given (None is not known or fitted)
    :param line_list_path_trimmed: Path to the root of the trimmed line list
    :param input_abundance: Input abundance for grid calculation for lbl quick (doesn't matter what for other stuff)
    :return: result of the fit with the best fit parameters and chi squared
    """
    # Load TS configuration
    with open(tsfitpy_pickled_configuration_path, 'rb') as f:
        tsfitpy_configuration = pickle.load(f)

    spectra = Spectra(specname, teff, logg, rv, met, microturb, macroturb, line_list_path_trimmed, index, tsfitpy_configuration,
                      elem_abund=input_abundance)

    print(f"Fitting {spectra.spec_name}")
    print(f"Teff = {spectra.teff}; logg = {spectra.logg}; RV = {spectra.rv}")

    if spectra.fitting_mode == "all":
        result = spectra.fit_all()
    elif spectra.fitting_mode == "lbl":
        result = spectra.fit_lbl()
    elif spectra.fitting_mode == "lbl_quick":
        result = spectra.fit_lbl_quick()
    elif spectra.fitting_mode == "teff":
        result = spectra.fit_teff_function()
    elif spectra.fitting_mode == "vmic":
        result = spectra.fit_vmic_slow()
    else:
        raise ValueError(f"unknown fitting mode {spectra.fitting_mode}, need all or lbl or teff")
    del spectra
    return result


def load_nlte_files_in_dict(elements_to_fit: list, depart_bin_file: list, depart_aux_file: list, model_atom_file: list, fit_feh, load_fe=True) -> tuple[dict, dict, dict]:
    """
    Loads and sorts NLTE elements to fit into respective dictionaries
    :param elements_to_fit: Array of elements to fit
    :param depart_bin_file: Departure binary file location (Fe last if not fitted)
    :param depart_aux_file: Departure aux file location (Fe last if not fitted)
    :param model_atom_file: Model atom file location (Fe last if not fitted)
    :param load_fe: loads Fe in the dict as well with it being the last element even if not fitted
    :return: 3 dictionaries: NLTE location of elements that exist with keys as element names
    """
    depart_bin_file_dict = {}  # assume that element locations are in the same order as the element to fit
    if load_fe:
        if fit_feh:
            iterations_for_nlte_elem = min(len(elements_to_fit), len(depart_bin_file))
        else:
            iterations_for_nlte_elem = min(len(elements_to_fit), len(depart_bin_file) - 1)
    else:
        iterations_for_nlte_elem = len(elements_to_fit)
    for i in range(iterations_for_nlte_elem):
        depart_bin_file_dict[elements_to_fit[i]] = depart_bin_file[i]
    depart_aux_file_dict = {}
    for i in range(iterations_for_nlte_elem):
        depart_aux_file_dict[elements_to_fit[i]] = depart_aux_file[i]
    model_atom_file_dict = {}
    for i in range(iterations_for_nlte_elem):
        model_atom_file_dict[elements_to_fit[i]] = model_atom_file[i]
    for i in range(iterations_for_nlte_elem, len(elements_to_fit)):
        depart_bin_file_dict[elements_to_fit[i]] = ""
        depart_aux_file_dict[elements_to_fit[i]] = ""
        model_atom_file_dict[elements_to_fit[i]] = ""
    if load_fe:
        if "Fe" not in elements_to_fit:  # if Fe is not fitted, then the last NLTE element should be
            depart_bin_file_dict["Fe"] = depart_bin_file[-1]
            depart_aux_file_dict["Fe"] = depart_aux_file[-1]
            model_atom_file_dict["Fe"] = model_atom_file[-1]
    return depart_bin_file_dict, depart_aux_file_dict, model_atom_file_dict

class TSFitPyConfig:
    def __init__(self, config_location: str, spectra_location: str, output_folder_title: str):
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
        self.experimental_parallelisation: bool = None
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

    def load_config(self):
        # if last 3 characters are .cfg then new config file, otherwise old config file
        if self.config_location[-4:] == ".cfg":
            self.load_new_config()
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

        self.turbospectrum_path = "../turbospectrum/"
        self.cluster_name = "None"

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
                        self.linemasks_path = self.check_if_path_exists(fields[2])
                    #if field_name == "segment_file_folder_location":
                        #self.segment_file_og = fields[2]
                    if field_name == "spec_input_path":
                        if self.spectra_input_path is None:
                            self.spectra_input_path = fields[2]
                    if field_name == "fitlist_input_folder":
                        self.fitlist_input_path = self.check_if_path_exists(fields[2])
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
                        self.wavelength_max = float(fields[2])
                    if field_name == "wavelength_maximum":
                        self.wavelength_min = float(fields[2])
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
                        self.global_temporary_directory = os.path.join(".", temp_directory, "")
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

    def load_new_config(self):
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
        self.linemasks_path = self.check_if_path_exists(self.config_parser["MainPaths"]["linemasks_path"])
        if self.spectra_input_path is None:
            self.spectra_input_path = self.config_parser["MainPaths"]["spectra_input_path"]
        self.fitlist_input_path = self.check_if_path_exists(self.config_parser["MainPaths"]["fitlist_input_path"])
        self.temporary_directory_path = os.path.join(self.config_parser["MainPaths"]["temporary_directory_path"], self.output_folder_title, '')

        self.atmosphere_type = self.validate_string_input(self.config_parser["FittingParameters"]["atmosphere_type"], ["1d", "3d"])
        self.fitting_mode = self.validate_string_input(self.config_parser["FittingParameters"]["fitting_mode"], ["all", "lbl", "teff", "lbl_quick", "vmic"])
        self.include_molecules = self.convert_string_to_bool(self.config_parser["FittingParameters"]["include_molecules"])
        self.nlte_flag = self.convert_string_to_bool(self.config_parser["FittingParameters"]["nlte"])
        self.fit_vmic = self.validate_string_input(self.config_parser["FittingParameters"]["fit_vmic"], ["yes", "no", "input"])
        vmac_fitting_mode = self.validate_string_input(self.config_parser["FittingParameters"]["fit_vmac"], ["yes", "no", "input"])
        if vmac_fitting_mode == "Yes":
            self.fit_vmac = True
            self.vmac_input = False
        elif vmac_fitting_mode == "No":
            self.fit_vmac = False
            self.vmac_input = False
        elif vmac_fitting_mode == "Input":
            self.fit_vmac = False
            self.vmac_input = True
        else:
            raise ValueError(f"Vmac fitting mode {vmac_fitting_mode} not recognized")
        self.fit_rotation = self.convert_string_to_bool(self.config_parser["FittingParameters"]["fit_rotation"])
        self.elements_to_fit = self.split_string_to_string_list(self.config_parser["FittingParameters"]["element_to_fit"])
        if 'Fe' in self.elements_to_fit:
            self.fit_feh = True
        else:
            self.fit_feh = False

        self.nlte_elements = self.split_string_to_string_list(self.config_parser["FittingParameters"]["nlte_elements"])
        self.linemask_file = self.config_parser["FittingParameters"]["linemask_file"]
        self.wavelength_delta = float(self.config_parser["FittingParameters"]["wavelength_delta"])
        self.segment_size = float(self.config_parser["FittingParameters"]["segment_size"])

        self.debug_mode = int(self.config_parser["ExtraParameters"]["debug_mode"])
        self.number_of_cpus = int(self.config_parser["ExtraParameters"]["number_of_cpus"])
        self.experimental_parallelisation = self.convert_string_to_bool(self.config_parser["ExtraParameters"]["experimental_parallelisation"])
        self.cluster_name = self.config_parser["ExtraParameters"]["cluster_name"]

        self.input_fitlist_filename = self.config_parser["InputAndOutputFiles"]["input_filename"]
        self.output_filename = self.config_parser["InputAndOutputFiles"]["output_filename"]

        self.resolution = float(self.config_parser["SpectraParameters"]["resolution"])
        self.vmac = float(self.config_parser["SpectraParameters"]["vmac"])
        self.rotation = float(self.config_parser["SpectraParameters"]["rotation"])
        self.init_guess_elements = self.split_string_to_float_list(self.config_parser["SpectraParameters"]["init_guess_elements"])
        self.init_guess_elements_path = self.split_string_to_float_list(self.config_parser["SpectraParameters"]["init_guess_elements_path"])
        self.input_elements_abundance = self.split_string_to_float_list(self.config_parser["SpectraParameters"]["input_elements_abundance"])
        self.input_elements_abundance_path = self.split_string_to_float_list(self.config_parser["SpectraParameters"]["input_elements_abundance_path"])

        self.wavelength_min = float(self.config_parser["ParametersForModeAll"]["wavelength_min"])
        self.wavelength_max = float(self.config_parser["ParametersForModeAll"]["wavelength_max"])

        self.bounds_vmic = self.split_string_to_float_list(self.config_parser["ParametersForModeLbl"]["bounds_vmic"])
        self.guess_range_vmic = self.split_string_to_float_list(self.config_parser["ParametersForModeLbl"]["guess_range_vmic"])

        self.bounds_teff = self.split_string_to_float_list(self.config_parser["ParametersForModeTeff"]["bounds_teff"])
        self.guess_range_teff = self.split_string_to_float_list(self.config_parser["ParametersForModeTeff"]["guess_range_teff"])

        self.bounds_vmac = self.split_string_to_float_list(self.config_parser["Bounds"]["bounds_vmac"])
        self.bounds_rotation = self.split_string_to_float_list(self.config_parser["Bounds"]["bounds_rotation"])
        self.bounds_abundance = self.split_string_to_float_list(self.config_parser["Bounds"]["bounds_abundance"])
        self.bounds_feh = self.split_string_to_float_list(self.config_parser["Bounds"]["bounds_feh"])
        self.bounds_doppler = self.split_string_to_float_list(self.config_parser["Bounds"]["bounds_doppler"])

        self.guess_range_vmac = self.split_string_to_float_list(self.config_parser["GuessRanges"]["guess_range_vmac"])
        self.guess_range_rotation = self.split_string_to_float_list(self.config_parser["GuessRanges"]["guess_range_rotation"])
        self.guess_range_abundance = self.split_string_to_float_list(self.config_parser["GuessRanges"]["guess_range_abundance"])
        self.guess_range_doppler = self.split_string_to_float_list(self.config_parser["GuessRanges"]["guess_range_doppler"])

    def convert_old_config(self):
        self.config_parser.add_section("turbospectrum_compiler")
        self.config_parser["turbospectrum_compiler"]["compiler"] = self.compiler

        self.config_parser.add_section("MainPaths")
        self.config_parser["MainPaths"]["turbospectrum_path"] = self.turbospectrum_path
        self.config_parser["MainPaths"]["interpolators_path"] = self.interpolators_path
        self.config_parser["MainPaths"]["line_list_path"] = self.line_list_path
        self.config_parser["MainPaths"]["model_atmosphere_grid_path_1d"] = self.model_atmosphere_grid_path_1d
        self.config_parser["MainPaths"]["model_atmosphere_grid_path_3d"] = self.model_atmosphere_grid_path_3d
        self.config_parser["MainPaths"]["model_atoms_path"] = self.model_atoms_path
        self.config_parser["MainPaths"]["departure_file_path"] = self.departure_file_path
        self.config_parser["MainPaths"]["departure_file_config_path"] = self.departure_file_config_path
        self.config_parser["MainPaths"]["output_path"] = self.output_folder_path
        self.config_parser["MainPaths"]["linemasks_path"] = self.linemasks_path
        self.config_parser["MainPaths"]["spectra_input_path"] = self.spectra_input_path
        self.config_parser["MainPaths"]["fitlist_input_path"] = self.fitlist_input_path
        self.config_parser["MainPaths"]["temporary_directory_path"] = self.global_temporary_directory

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
        self.config_parser["FittingParameters"]["element_to_fit"] = self.convert_list_to_str(self.elements_to_fit)

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
        self.config_parser["FittingParameters"]["nlte_elements"] = self.convert_list_to_str(nlte_elements_to_write)
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
        self.config_parser["InputAndOutputFiles"]["output_filename"] = self.output_folder_path_global

        self.config_parser.add_section("SpectraParameters")
        self.config_parser["SpectraParameters"]["resolution"] = str(self.resolution)
        self.config_parser["SpectraParameters"]["vmac"] = str(self.vmac)
        self.config_parser["SpectraParameters"]["rotation"] = str(self.rotation)
        self.config_parser["SpectraParameters"]["init_guess_elements"] = self.convert_list_to_str(self.init_guess_elements)
        self.config_parser["SpectraParameters"]["init_guess_elements_path"] = self.convert_list_to_str(self.init_guess_elements_path)
        self.config_parser["SpectraParameters"]["input_elements_abundance"] = self.convert_list_to_str(self.input_elements_abundance)
        self.config_parser["SpectraParameters"]["input_elements_abundance_path"] = self.convert_list_to_str(self.input_elements_abundance_path)

        self.config_parser.add_section("ParametersForModeAll")
        self.config_parser["ParametersForModeAll"]["wavelength_min"] = str(self.wavelength_min)
        self.config_parser["ParametersForModeAll"]["wavelength_max"] = str(self.wavelength_max)

        self.config_parser.add_section("ParametersForModeLbl")
        self.config_parser["ParametersForModeLbl"]["bounds_vmic"] = self.convert_list_to_str(self.bounds_vmic)
        self.config_parser["ParametersForModeLbl"]["guess_range_vmic"] = self.convert_list_to_str(self.guess_range_vmic)

        self.config_parser.add_section("ParametersForModeTeff")
        self.config_parser["ParametersForModeTeff"]["bounds_teff"] = self.convert_list_to_str(self.bounds_teff)
        self.config_parser["ParametersForModeTeff"]["guess_range_teff"] = self.convert_list_to_str(self.guess_range_teff)

        self.config_parser.add_section("Bounds")
        self.config_parser["Bounds"]["bounds_vmac"] = self.convert_list_to_str(self.bounds_vmac)
        self.config_parser["Bounds"]["bounds_rotation"] = self.convert_list_to_str(self.bounds_rotation)
        self.config_parser["Bounds"]["bounds_abundance"] = self.convert_list_to_str(self.bounds_abundance)
        self.config_parser["Bounds"]["bounds_feh"] = self.convert_list_to_str(self.bounds_feh)
        self.config_parser["Bounds"]["bounds_doppler"] = self.convert_list_to_str(self.bounds_doppler)

        self.config_parser.add_section("GuessRanges")
        self.config_parser["GuessRanges"]["guess_range_vmac"] = self.convert_list_to_str(self.guess_range_vmac)
        self.config_parser["GuessRanges"]["guess_range_rotation"] = self.convert_list_to_str(self.guess_range_rotation)
        self.config_parser["GuessRanges"]["guess_range_abundance"] = self.convert_list_to_str(self.guess_range_abundance)
        self.config_parser["GuessRanges"]["guess_range_doppler"] = self.convert_list_to_str(self.guess_range_doppler)

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

    def check_valid_input(self):
        self.atmosphere_type = self.atmosphere_type.upper()
        self.fitting_mode = self.fitting_mode.lower()
        self.include_molecules = self.include_molecules
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
        self.temporary_directory_path = self.find_path_temporary_directory(self.temporary_directory_path)
        self.number_of_cpus = int(self.number_of_cpus)

        self.segment_file = os.path.join(self.temporary_directory_path, "segment_file.txt")

        self.debug_mode = self.debug_mode
        self.experimental_parallelisation = self.experimental_parallelisation

        self.nelement = len(self.elements_to_fit)

        if self.turbospectrum_path is None:
            self.turbospectrum_path = "../turbospectrum/"
        if self.compiler.lower() == "intel":
            self.turbospectrum_path = os.path.join(os.getcwd(), self.check_if_path_exists(self.turbospectrum_path),
                                                   "exec", "")
        elif self.compiler.lower() == "gnu":
            self.turbospectrum_path = os.path.join(os.getcwd(), self.check_if_path_exists(self.turbospectrum_path),
                                                   "exec-gf", "")
        else:
            raise ValueError("Compiler not recognized")
        self.turbospectrum_path = self.turbospectrum_path

        if os.path.exists(self.interpolators_path):
            self.interpolators_path = os.path.join(os.getcwd(), self.interpolators_path)
        else:
            if self.interpolators_path.startswith("./"):
                self.interpolators_path = self.interpolators_path[2:]
                self.interpolators_path = os.path.join(os.getcwd(), "scripts", self.interpolators_path)

        if self.atmosphere_type.upper() == "1D":
            self.model_atmosphere_grid_path = self.check_if_path_exists(self.model_atmosphere_grid_path_1d)
            self.model_atmosphere_list = os.path.join(self.model_atmosphere_grid_path,
                                                                "model_atmosphere_list.txt")
        elif self.atmosphere_type.upper() == "3D":
            self.model_atmosphere_grid_path = self.check_if_path_exists(self.model_atmosphere_grid_path_3d)
            self.model_atmosphere_list = os.path.join(self.model_atmosphere_grid_path,
                                                                "model_atmosphere_list.txt")
        else:
            raise ValueError(f"Expected atmosphere type 1D or 3D, got {self.atmosphere_type.upper()}")
        self.model_atoms_path = self.check_if_path_exists(self.model_atoms_path)
        self.departure_file_path = self.check_if_path_exists(self.departure_file_path)
        self.output_folder_path_global = self.check_if_path_exists(self.output_folder_path)
        self.output_folder_path = os.path.join(self.check_if_path_exists(self.output_folder_path),
                                                    self.output_folder_title)
        self.spectra_input_path = self.check_if_path_exists(self.spectra_input_path)
        self.line_list_path = self.check_if_path_exists(self.line_list_path)

        if self.fitting_mode == "teff":
            self.fit_teff = True
        else:
            self.fit_teff = False

        if self.fit_teff:
            self.fit_feh = False

    def load_spectra_config(self, spectra_object: Spectra):
        spectra_object.atmosphere_type = self.atmosphere_type
        spectra_object.fitting_mode = self.fitting_mode
        spectra_object.include_molecules = self.include_molecules
        spectra_object.nlte_flag = self.nlte_flag
        spectra_object.fit_vmic = self.fit_vmic
        spectra_object.fit_vmac = self.fit_vmac
        spectra_object.fit_rotation = self.fit_rotation
        spectra_object.elem_to_fit = self.elements_to_fit
        spectra_object.fit_feh = self.fit_feh
        spectra_object.lmin = self.wavelength_min
        spectra_object.lmax = self.wavelength_max
        spectra_object.ldelta = self.wavelength_delta
        spectra_object.resolution = self.resolution
        spectra_object.rotation = self.rotation
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
        spectra_object.turbospec_path = self.turbospectrum_path

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
    def check_if_path_exists(path_to_check: str) -> str:
        # check if path is absolute
        if os.path.isabs(path_to_check):
            if os.path.exists(os.path.join(path_to_check, "")):
                return path_to_check
            else:
                raise ValueError(f"Configuration: {path_to_check} does not exist")
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
                else:
                    raise ValueError(f"Configuration: {path_to_check} does not exist")
            else:
                raise ValueError(f"Configuration: {path_to_check} does not exist")

    @staticmethod
    def find_path_temporary_directory(temp_directory):
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
    def convert_list_to_str(list_to_convert: list) -> str:
        string_to_return = ""
        for element in list_to_convert:
            string_to_return = f"{string_to_return} {element}"
        return string_to_return


def create_segment_file(segment_size: float, line_begins_list, line_ends_list) -> tuple[np.ndarray, np.ndarray]:
    segments_left = []
    segments_right = []
    start = line_begins_list[0] - segment_size
    end = line_ends_list[0] + segment_size

    for (line_left, line_right) in zip(line_begins_list, line_ends_list):
        if line_left > end + segment_size:
            segments_left.append(start)
            segments_right.append(end)
            start = line_left - segment_size
            end = line_right + segment_size
        else:
            end = line_right + segment_size

    segments_left.append(start)
    segments_right.append(end)

    return np.asarray(segments_left), np.asarray(segments_right)

def run_tsfitpy(output_folder_title, config_location, spectra_location, dask_mpi_installed):
    #nlte_config_outdated = False
    #need_to_add_new_nlte_config = True  # only if nlte_config_outdated == True

    #initial_guess_string = None

    # read the configuration file
    tsfitpy_configuration = TSFitPyConfig(config_location, spectra_location, output_folder_title)
    tsfitpy_configuration.load_config()
    tsfitpy_configuration.check_valid_input()

    if not config_location[-4:] == ".cfg":
        tsfitpy_configuration.convert_old_config()

    print(f"Fitting data at {tsfitpy_configuration.spectra_input_path} with resolution {tsfitpy_configuration.resolution} and rotation {tsfitpy_configuration.rotation}")

    # set directories
    line_list_path_orig = tsfitpy_configuration.line_list_path
    line_list_path_trimmed = os.path.join(tsfitpy_configuration.temporary_directory_path, "linelist_for_fitting_trimmed", "") # f"{line_list_path}../linelist_for_fitting_trimmed/"

    # load NLTE data dicts
    if tsfitpy_configuration.nlte_flag:
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
                                            'Ba/auxData_output_Ba_mean3D_May-10-2021_marcs_names.txt'
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
                                        '',
                                        ''
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
        nlte_config.read(os.path.join(tsfitpy_configuration.departure_file_path, "nlte_filenames.cfg"))

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

    #prevent overwriting
    if os.path.exists(tsfitpy_configuration.output_folder_path):
        raise FileExistsError("Error: output folder already exists. Run was stopped to prevent overwriting")

    tsfitpy_configuration.linemask_file = os.path.join(tsfitpy_configuration.linemasks_path, tsfitpy_configuration.linemask_file)
    #Spectra.segment_file = f"{segment_file_og}{segment_file}"

    print(f"Temporary directory name: {tsfitpy_configuration.temporary_directory_path}")
    create_dir(tsfitpy_configuration.temporary_directory_path)
    create_dir(tsfitpy_configuration.output_folder_path)

    # copy config file into output folder (for easier plotting)
    shutil.copyfile(config_location, os.path.join(tsfitpy_configuration.output_folder_path, "configuration.txt"))

    fitlist = os.path.join(tsfitpy_configuration.fitlist_input_path, tsfitpy_configuration.input_fitlist_filename)

    tsfitpy_configuration.ndimen = 1  # first dimension is RV fit
    if not tsfitpy_configuration.fit_teff:
        if tsfitpy_configuration.fit_vmic == "Yes" and (
                tsfitpy_configuration.fitting_mode == "lbl" or tsfitpy_configuration.fitting_mode == "lbl_quick") and not tsfitpy_configuration.atmosphere_type == "3D":
            tsfitpy_configuration.ndimen += 1  # if fitting micro for lbl, not 3D
        if tsfitpy_configuration.fitting_mode == "lbl" or tsfitpy_configuration.fitting_mode == "vmic":  # TODO: if several elements fitted for other modes, change here
            tsfitpy_configuration.ndimen += tsfitpy_configuration.nelement
            print(f"Fitting {tsfitpy_configuration.nelement} element(s): {tsfitpy_configuration.elements_to_fit}")
        elif tsfitpy_configuration.fitting_mode == "lbl_quick":
            pass    # element is not fitted using minimization, no need for ndimen
        else:
            tsfitpy_configuration.ndimen += 1
            print(f"Fitting {1} element: {tsfitpy_configuration.elements_to_fit[0]}")
        if tsfitpy_configuration.fit_vmac:
            tsfitpy_configuration.ndimen += 1
    else:
        print("Fitting Teff based on the linelist provided. Ignoring element fitting.")

    fitlist_data = np.loadtxt(fitlist, dtype='str')

    if fitlist_data.ndim == 1:
        fitlist_data = np.array([fitlist_data])

    specname_fitlist, rv_fitlist, teff_fitlist, logg_fitlist = fitlist_data[:, 0], fitlist_data[:, 1], \
                                                               fitlist_data[:, 2], fitlist_data[:, 3]

    fitlist_next_column = 4     # next loaded column #TODO not perfect solution? what if user gives metal but fits it too?

    input_abundances = np.zeros(fitlist_data.shape[0])  # if lbl_quick they will be used as center guess, otherwise means nothing
    if not tsfitpy_configuration.fitting_mode == "lbl_quick":
        if tsfitpy_configuration.fit_feh:
            met_fitlist = np.zeros(fitlist_data.shape[0])  # fitting metallicity: just give it 0
        else:
            met_fitlist = fitlist_data[:, fitlist_next_column]  # metallicity [Fe/H], scaled to solar; not fitting metallicity: load it
            fitlist_next_column += 1
    else:
        met_fitlist = fitlist_data[:, fitlist_next_column]
        fitlist_next_column += 1
        if not tsfitpy_configuration.fit_feh:
            input_abundances = fitlist_data[:, fitlist_next_column]  # guess for abundance for lbl quick, [X/Fe]
            fitlist_next_column += 1

    if tsfitpy_configuration.fit_vmic == "Input":
        vmic_input = fitlist_data[:, fitlist_next_column]
        fitlist_next_column += 1
    else:
        vmic_input = np.zeros(fitlist_data.shape[0])

    if tsfitpy_configuration.vmac_input:
        vmac_input = fitlist_data[:, fitlist_next_column]  # input macroturbulence in km/s
        fitlist_next_column += 1
    else:
        vmac_input = np.ones(fitlist_data.shape[0]) * tsfitpy_configuration.vmac

    if np.size(tsfitpy_configuration.init_guess_elements) > 0:
        init_guess_spectra_dict = collections.defaultdict(dict)

        for init_guess_elem, init_guess_loc in zip(tsfitpy_configuration.init_guess_elements, tsfitpy_configuration.init_guess_elements_path):
            init_guess_data = np.loadtxt(init_guess_loc, dtype=str, usecols=(0, 1))
            if init_guess_data.ndim == 1:
                init_guess_data = np.array([init_guess_data])
            init_guess_spectra_names, init_guess_values = init_guess_data[:, 0], init_guess_data[:, 1].astype(float)

            for spectra in specname_fitlist:
                spectra_loc_index = np.where(init_guess_spectra_names == spectra)[0][0]
                init_guess_spectra_dict[spectra][init_guess_elem] = init_guess_values[spectra_loc_index]

        tsfitpy_configuration.init_guess_dict = dict(init_guess_spectra_dict)

    if np.size(tsfitpy_configuration.input_elements_abundance) > 0:
        input_elem_abundance_dict = collections.defaultdict(dict)

        for input_elem, init_elem_loc in zip(tsfitpy_configuration.input_elements_abundance, tsfitpy_configuration.input_elements_abundance_path):
            input_abund_data = np.loadtxt(init_elem_loc, dtype=str, usecols=(0, 1))
            if input_abund_data.ndim == 1:
                input_abund_data = np.array([input_abund_data])
            input_abund_data_spectra_names, input_abund_data_values = input_abund_data[:, 0], input_abund_data[:, 1].astype(float)

            for spectra in specname_fitlist:
                spectra_loc_index = np.where(input_abund_data_spectra_names == spectra)[0]
                if np.size(spectra_loc_index) == 1:
                    input_elem_abundance_dict[spectra][input_elem] = input_abund_data_values[spectra_loc_index]
                else:
                    print(f"{spectra} does not have element {input_elem} in the input spectra. Using [{input_elem}/Fe]=0")
                    input_elem_abundance_dict[spectra][input_elem] = 0

        tsfitpy_configuration.input_elem_abundance = dict(input_elem_abundance_dict)

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
    #if tsfitpy_configuration.seg_begins.size == 1:
    #    tsfitpy_configuration.seg_begins = np.array([tsfitpy_configuration.seg_begins])
    #    tsfitpy_configuration.seg_ends = np.array([tsfitpy_configuration.seg_ends])

    # check inputs

    print("\n\nChecking inputs\n")

    if np.size(tsfitpy_configuration.seg_begins) != np.size(tsfitpy_configuration.seg_ends):
        print("Segment beginning and end are not the same length")
    if np.size(tsfitpy_configuration.line_centers_sorted) != np.size(tsfitpy_configuration.line_begins_sorted) or np.size(tsfitpy_configuration.line_centers_sorted) != np.size(tsfitpy_configuration.line_ends_sorted):
        print("Line center, beginning and end are not the same length")
    """if workers < np.size(specname_fitlist.size):
        print(f"You requested {workers}, but you only need to fit {specname_fitlist.size} stars. Requesting more CPUs "
              f"(=workers) than the spectra will just result in idle workers.")"""
    if tsfitpy_configuration.guess_range_teff[0] > 0:
        print(f"You requested your {tsfitpy_configuration.guess_range_teff[0]} to be positive. That will result in the lower "
              f"guess value to be bigger than the expected star temperature. Consider changing the number to negative.")
    if tsfitpy_configuration.guess_range_teff[1] < 0:
        print(f"You requested your {tsfitpy_configuration.guess_range_teff[1]} to be negative. That will result in the upper "
              f"guess value to be smaller than the expected star temperature. Consider changing the number to positive.")
    if min(tsfitpy_configuration.guess_range_vmac) < min(tsfitpy_configuration.bounds_vmac) or max(tsfitpy_configuration.guess_range_vmac) > max(tsfitpy_configuration.bounds_vmac):
        print(f"You requested your macro bounds as {tsfitpy_configuration.bounds_vmac}, but guesses"
              f"are {tsfitpy_configuration.guess_range_vmac}, which is outside hard bound range. Consider"
              f"changing bounds or guesses.")
    if min(tsfitpy_configuration.guess_range_vmic) < min(tsfitpy_configuration.bounds_vmic) or max(tsfitpy_configuration.guess_range_vmic) > max(tsfitpy_configuration.bounds_vmic):
        print(f"You requested your micro bounds as {tsfitpy_configuration.bounds_vmic}, but guesses"
              f"are {tsfitpy_configuration.guess_range_vmic}, which is outside hard bound range. Consider"
              f"changing bounds or guesses.")
    if min(tsfitpy_configuration.guess_range_abundance) < min(tsfitpy_configuration.bounds_abundance) or max(tsfitpy_configuration.guess_range_abundance) > max(tsfitpy_configuration.bounds_abundance):
        print(f"You requested your abundance bounds as {tsfitpy_configuration.bounds_abundance}, but guesses"
              f"are {tsfitpy_configuration.guess_range_abundance} , which is outside hard bound range. Consider"
              f"changing bounds or guesses if you fit elements except for Fe.")
    if min(tsfitpy_configuration.guess_range_abundance) < min(tsfitpy_configuration.bounds_feh) or max(tsfitpy_configuration.guess_range_abundance) > max(tsfitpy_configuration.bounds_feh):
        print(f"You requested your metallicity bounds as {tsfitpy_configuration.bounds_feh}, but guesses"
              f"are {tsfitpy_configuration.guess_range_abundance}, which is outside hard bound range. Consider"
              f"changing bounds or guesses IF YOU FIT METALLICITY.")
    if min(tsfitpy_configuration.guess_range_doppler) < min(tsfitpy_configuration.bounds_doppler) or max(tsfitpy_configuration.guess_range_doppler) > max(tsfitpy_configuration.bounds_doppler):
        print(f"You requested your RV bounds as {tsfitpy_configuration.bounds_doppler}, but guesses"
              f"are {tsfitpy_configuration.guess_range_doppler}, which is outside hard bound range. Consider"
              f"changing bounds or guesses.")
    if tsfitpy_configuration.rotation < 0:
        print(f"Requested rotation of {tsfitpy_configuration.rotation}, which is less than 0. Consider changing it.")
    if tsfitpy_configuration.resolution < 0:
        print(f"Requested resolution of {tsfitpy_configuration.resolution}, which is less than 0. Consider changing it.")
    if tsfitpy_configuration.vmac < 0:
        print(f"Requested macroturbulence input of {tsfitpy_configuration.vmac}, which is less than 0. Consider changing it if "
              f"you fit it.")
    # check done in tsfitpyconfiguration
    #if tsfitpy_configuration.fitting_mode not in ["all", "lbl", "lbl_quick", "teff"]:
    #    print(f"Expected fitting mode 'all', 'lbl', 'lbl_quick', 'teff', but got {tsfitpy_configuration.fitting_mode} instead")
    if tsfitpy_configuration.nlte_flag:
        for file in tsfitpy_configuration.depart_bin_file_dict:
            if not os.path.isfile(os.path.join(tsfitpy_configuration.departure_file_path, tsfitpy_configuration.depart_bin_file_dict[file])):
                print(f"{tsfitpy_configuration.depart_bin_file_dict[file]} does not exist! Check the spelling or if the file exists")
        for file in tsfitpy_configuration.depart_aux_file_dict:
            if not os.path.isfile(os.path.join(tsfitpy_configuration.departure_file_path, tsfitpy_configuration.depart_aux_file_dict[file])):
                print(f"{tsfitpy_configuration.depart_aux_file_dict[file]} does not exist! Check the spelling or if the file exists")
        for file in tsfitpy_configuration.model_atom_file_dict:
            if not os.path.isfile(os.path.join(tsfitpy_configuration.model_atoms_path, tsfitpy_configuration.model_atom_file_dict[file])):
                print(f"{tsfitpy_configuration.model_atom_file_dict[file]} does not exist! Check the spelling or if the file exists")

    for line_start, line_end in zip(tsfitpy_configuration.line_begins_sorted, tsfitpy_configuration.line_ends_sorted):
        index_location = np.where(np.logical_and(tsfitpy_configuration.seg_begins <= line_start, line_end <= tsfitpy_configuration.seg_ends))[0]
        if np.size(index_location) > 1:
            print(f"{line_start} {line_end} linemask has more than 1 segment!")
        if np.size(index_location) == 0:
            print(f"{line_start} {line_end} linemask does not have any corresponding segment")

    print("\nDone doing some basic checks. Consider reading the messages above, if there are any. Can be useful if it "
          "crashes.\n\n")

    print("Trimming down the linelist to only lines within segments for faster fitting")
    if tsfitpy_configuration.fitting_mode == "all" or tsfitpy_configuration.fitting_mode == "lbl_quick":
        # os.system("rm {}/*".format(line_list_path_trimmed))
        line_list_path_trimmed = os.path.join(line_list_path_trimmed, "all", output_folder_title, '')
        create_window_linelist(tsfitpy_configuration.seg_begins, tsfitpy_configuration.seg_ends, line_list_path_orig, line_list_path_trimmed,
                               tsfitpy_configuration.include_molecules, lbl=False)
        line_list_path_trimmed =  os.path.join(line_list_path_trimmed, "0", "")
    elif tsfitpy_configuration.fitting_mode == "lbl" or tsfitpy_configuration.fitting_mode == "teff" or tsfitpy_configuration.fitting_mode == "vmic":
        line_list_path_trimmed = os.path.join(line_list_path_trimmed, "lbl", output_folder_title, '')
        """for j in range(len(tsfitpy_configuration.line_begins_sorted)):
            start = np.where(np.logical_and(tsfitpy_configuration.seg_begins <= tsfitpy_configuration.line_centers_sorted[j],
                                            tsfitpy_configuration.line_centers_sorted[j] <= tsfitpy_configuration.seg_ends))[0][0]
            line_list_path_trimmed_new = get_trimmed_lbl_path_name(tsfitpy_configuration.elem_to_fit, line_list_path_trimmed,
                                                                   tsfitpy_configuration.segment_file, j, start)"""
        #line_list_path_trimmed_new = get_trimmed_lbl_path_name(tsfitpy_configuration.elem_to_fit, line_list_path_trimmed,
        #                                                       tsfitpy_configuration.segment_file, j, start)
        create_window_linelist(tsfitpy_configuration.seg_begins, tsfitpy_configuration.seg_ends, line_list_path_orig,
                               line_list_path_trimmed,
                               tsfitpy_configuration.include_molecules, lbl=True)
    else:
        raise ValueError("Unknown fitting method")
    print("Finished trimming linelist")

    model_temperatures, model_logs, model_mets, marcs_value_keys, marcs_models, marcs_values = fetch_marcs_grid(tsfitpy_configuration.model_atmosphere_list, TurboSpectrum.marcs_parameters_to_ignore)
    tsfitpy_configuration.model_temperatures = model_temperatures
    tsfitpy_configuration.model_logs = model_logs
    tsfitpy_configuration.model_mets = model_mets
    tsfitpy_configuration.marcs_value_keys = marcs_value_keys
    tsfitpy_configuration.marcs_models = marcs_models
    tsfitpy_configuration.marcs_values = marcs_values
    if tsfitpy_configuration.nlte_flag:
        tsfitpy_configuration.aux_file_length_dict = {}

        for element in model_atom_file_dict:
            tsfitpy_configuration.aux_file_length_dict[element] = len(np.loadtxt(os_path.join(tsfitpy_configuration.departure_file_path, depart_aux_file_dict[element]), dtype='str'))

    # pickle the configuration file into the temp folder
    with open(os.path.join(tsfitpy_configuration.temporary_directory_path, "tsfitpy_configuration.pkl"), "wb") as f:
        pickle.dump(tsfitpy_configuration, f)
    tsfitpy_pickled_configuration_path = os.path.join(tsfitpy_configuration.temporary_directory_path, "tsfitpy_configuration.pkl")

    if tsfitpy_configuration.number_of_cpus != 1:
        print("Preparing workers")  # TODO check memory issues? set higher? give warnings?
        if dask_mpi_installed:
            print("Ignoring requested number of CPUs in the config file and launching based on CPUs requested in the slurm script")
            dask_mpi_initialize()
            client = Client(threads_per_worker=1)  # if # of threads are not equal to 1, then may break the program
        else:
            if tsfitpy_configuration.number_of_cpus > 1:
                client = Client(threads_per_worker=1, n_workers=tsfitpy_configuration.number_of_cpus)
            else:
                client = Client(threads_per_worker=1)
        print(client)

        host = client.run_on_scheduler(socket.gethostname)
        port = client.scheduler_info()['services']['dashboard']
        print(f"Assuming that the cluster is ran at {tsfitpy_configuration.cluster_name} (change in config if not the case)")

        # print(logger.info(f"ssh -N -L {port}:{host}:{port} {login_node_address}"))
        print(f"ssh -N -L {port}:{host}:{port} {tsfitpy_configuration.cluster_name}")

        print("Worker preparation complete")

        futures = []
        for i in range(specname_fitlist.size):
            specname1, teff1, logg1, rv1, met1, microturb1 = specname_fitlist[i], teff_fitlist[i], logg_fitlist[i], \
                                                             rv_fitlist[i], met_fitlist[i], vmic_input[i]
            macroturb1 = vmac_input[i]
            input_abundance = input_abundances[i]
            future = client.submit(create_and_fit_spectra, specname1, teff1, logg1, rv1, met1, microturb1, macroturb1,
                                   line_list_path_trimmed, input_abundance, i, tsfitpy_pickled_configuration_path)
            futures.append(future)  # prepares to get values

        print("Start gathering")  # use http://localhost:8787/status to check status. the port might be different
        futures = np.array(client.gather(futures))  # starts the calculations (takes a long time here)
        results = futures
        print("Worker calculation done")  # when done, save values
    else:
        results = []
        for i in range(specname_fitlist.size):
            specname1, teff1, logg1, rv1, met1, microturb1 = specname_fitlist[i], teff_fitlist[i], logg_fitlist[i], \
                                                             rv_fitlist[i], met_fitlist[i], vmic_input[i]
            input_abundance = input_abundances[i]
            macroturb1 = vmac_input[i]
            results.append(create_and_fit_spectra(specname1, teff1, logg1, rv1, met1, microturb1, macroturb1,
                                                  line_list_path_trimmed, input_abundance, i, tsfitpy_pickled_configuration_path))

    output = os.path.join(tsfitpy_configuration.output_folder_path, tsfitpy_configuration.output_filename)

    f = open(output, 'a')

    # result = f"{self.spec_name} {res.x[0]} {res.x[1]} {res.fun} {self.macroturb}"
    # result.append(f"{self.spec_name} {tsfitpy_configuration.line_centers_sorted[j]} {tsfitpy_configuration.line_begins_sorted[j]} "
    #                      f"{tsfitpy_configuration.line_ends_sorted[j]} {res.x[0]} {res.x[1]} {microturb} {macroturb} {res.fun}")

    if tsfitpy_configuration.fitting_mode == "lbl":
        output_elem_column = f"Fe_H"

        for i in range(tsfitpy_configuration.nelement):
            # tsfitpy_configuration.elem_to_fit[i] = element name
            elem_name = tsfitpy_configuration.elements_to_fit[i]
            if elem_name != "Fe":
                output_elem_column += f"\t{elem_name}_Fe"
    else:
        if tsfitpy_configuration.fit_feh:
            output_elem_column = "Fe_H"
        else:
            output_elem_column = f"{tsfitpy_configuration.elements_to_fit[0]}_Fe"

    if tsfitpy_configuration.fitting_mode == "all":
        print(f"#specname\t{output_elem_column}\tDoppler_Shift_add_to_RV\tchi_squared\tMacroturb", file=f)
    elif tsfitpy_configuration.fitting_mode == "lbl" or tsfitpy_configuration.fitting_mode == "vmic":
        print(
            f"#specname\twave_center\twave_start\twave_end\tDoppler_Shift_add_to_RV\t{output_elem_column}\tMicroturb\tMacroturb\trotation\tchi_squared\tew",
            file=f)
    elif tsfitpy_configuration.fitting_mode == "lbl_quick":  # f" {res.x[0]} {vmicro} {macroturb} {res.fun}"
        output_columns = "#specname\twave_center\twave_start\twave_end"
        for i in range(tsfitpy_configuration.grids_amount):     # TODO fix lbl_quick
            output_columns += f"\tabund_{i}\tdoppler_shift_{i}\tmicroturb_{i}\tmacroturb_{i}\tchi_square_{i}"
        # f"#specname        wave_center  wave_start  wave_end  {element[0]}_Fe   Doppler_Shift_add_to_RV Microturb   Macroturb    chi_squared"
        print(output_columns, file=f)
    elif tsfitpy_configuration.fitting_mode == "teff":
        output_columns = "#specname\tTeff\twave_center\twave_start\twave_end\tDoppler_Shift_add_to_RV\tMicroturb\tMacroturb\tchi_squared"
        print(output_columns, file=f)

    results = np.array(results)

    if np.ndim(results) == 1:
        for i in range(np.size(results)):
            print(results[i], file=f)
    else:
        for i in range(int(np.size(results) / np.size(results[0]))):
            for j in range(np.size(results[0])):
                print(results[i][j], file=f)

    f.close()

    shutil.rmtree(tsfitpy_configuration.temporary_directory_path)  # clean up temp directory
    try:
        shutil.rmtree(line_list_path_trimmed)   # clean up trimmed line list
    except FileNotFoundError:
        pass    # because now trimmed files are in the temp directory, might give error


if __name__ == '__main__':
    raise RuntimeError("This file is not meant to be run as main. Please run TSFitPy/main.py instead.")  # this is a module
    """major_version_scipy, minor_version_scipy, patch_version_scipy = scipy.__version__.split(".")
    if int(major_version_scipy) < 1 or (int(major_version_scipy) == 1 and int(minor_version_scipy) < 7) or (
            int(major_version_scipy) == 1 and int(minor_version_scipy) == 7 and int(patch_version_scipy) == 0):
        raise ImportError(f"Scipy has to be at least version 1.7.1, otherwise bounds are not considered in mimisation. "
                          f"That will lead to bad fits. Please update to scipy 1.7.1 OR higher. Your version: "
                          f"{scipy.__version__}")

    try:
        raise ModuleNotFoundError
        from dask_mpi import initialize as dask_mpi_initialize
        dask_mpi_installed = True
    except ModuleNotFoundError:
        #print("Dask MPI not installed. Job launching only on 1 node. Ignore if not using a cluster.")
        dask_mpi_installed = False

    if len(argv) > 1:   # when calling the program, can now add extra argument with location of config file, easier to call
        config_location = argv[1]
    else:
        config_location = "../input_files/tsfitpy_input_configuration.txt"  # location of config file
    if len(argv) > 2:  # when calling the program, can now add extra argument with location of observed spectra, easier to call
        obs_location = argv[2]
    else:
        obs_location = None  # otherwise defaults to the input one
    print(config_location)
    # TODO explain lbl quick
    output_folder_title_date = datetime.datetime.now().strftime("%b-%d-%Y-%H-%M-%S")  # used to not conflict with other instances of fits
    output_folder_title_date = f"{output_folder_title_date}_{np.random.random(1)[0]}"     # in case if someone calls the function several times per second
    print(f"Start of the fitting: {output_folder_title_date}")
    login_node_address = "gemini-login.mpia.de"  # Change this to the address/domain of your login node
    try:
        run_TSFitPy(output_folder_title_date, config_location, obs_location)
        print("Fitting completed")
    except KeyboardInterrupt:
        print(f"KeyboardInterrupt detected. Terminating job.")  #TODO: cleanup temp folders here?
    finally:
        print(f"End of the fitting: {datetime.datetime.now().strftime('%b-%d-%Y-%H-%M-%S')}")"""

# TODO:
# - fix pathing in calculate_nlte_correction_line
# - fix pathing in run_wrapper
# - fix pathing in run_wrapper_v2
# - fix pathing in scripts_for_plotting and corresponding jupyter notebook
# - add conversion of old config into new one <- To check
# - test other fitting modes: all, teff
# - save segments in a file <- To check
# - add changing name of the output folder to what is being done