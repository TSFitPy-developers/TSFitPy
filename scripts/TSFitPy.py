from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
# from multiprocessing import Pool
# import h5py
# import matplotlib.pyplot as plt
from turbospectrum_class_nlte import TurboSpectrum
# from turbospectrum_class_3d import TurboSpectrum_3D
import time
# import math
import os
from os import path as os_path
# import glob
import datetime
from dask.distributed import Client
import shutil
import socket
from typing import Union
from sys import argv
import collections

from solar_abundances import solar_abundances, periodic_table

from convolve import *
from create_window_linelist_function import *


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

    if teff == 5771 and logg == 4.44:
        v_mturb = 0.9

    if v_mturb <= 0.0:
        print("error in calculating micro turb, setting it to 1.0")
        return 1.0

    return v_mturb


def get_convolved_spectra(wave: np.ndarray, flux: np.ndarray, fwhm: float, macro: float, rot: float) -> tuple[
    np.ndarray, np.ndarray]:
    """
    Convolves spectra with FWHM, macroturbulence or rotation if values are non-zero
    :param wave: wavelength array, in ascending order
    :param flux: flux array normalised
    :param fwhm: FWHM, zero if not required
    :param macro: Macroturbulence in km/s, zero if not required
    :param rot: Rotation in km/s, 0 if not required
    :return: 2 arrays, first is convolved wavelength, second is convolved flux
    """
    if fwhm != 0.0:
        wave_mod_conv, flux_mod_conv = conv_res(wave, flux, fwhm)
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
                              flux_obs: np.ndarray, macro: float, fwhm: float, rot: float,
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
    :param fwhm: FWHM, zero if not required
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

        wave_mod, flux_mod = get_convolved_spectra(wave_mod_filled, flux_mod_filled, fwhm, macro, rot)

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
                              wave_mod_orig: np.ndarray, flux_mod_orig: np.ndarray, fwhm: float, lmax: float,
                              lmin: float, macro: float, rot: float, save_convolved=True) -> float:
    """
    Calculates chi squared by opening a created synthetic spectrum and comparing to the observed spectra. Then
    calculates chi squared. Used for line by line method, by only looking at a specific line.
    :param temp_directory:
    :param wave_obs: Observed wavelength
    :param flux_obs: Observed normalised flux
    :param wave_mod_orig: Synthetic wavelength
    :param flux_mod_orig: Synthetic normalised flux
    :param fwhm: FWHM, zero if not required
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

    wave_mod, flux_mod = get_convolved_spectra(wave_mod_orig, flux_mod_orig, fwhm, macro, rot)
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


class Spectra:
    turbospec_path: str = None  # path to the /exec/ file
    interpol_path: str = None  # path to the model_interpolators folder with fortran code
    model_atmosphere_grid_path: str = None
    model_atmosphere_list: str = None
    model_atom_path: str = None
    departure_file_path: str = None
    linemask_file: str = None
    segment_file: str = None
    atmosphere_type: str = None  # "1D" or "3D", string
    include_molecules: str = None  # "True" or "False", string
    nlte_flag: bool = None
    fit_microturb: str = "No"
    fit_macroturb: bool = False
    fit_teff: str = None  # does not work atm
    fit_logg: str = None  # does not work atm
    nelement: int = None  # how many elements to fit (1 to whatever)
    fit_met: bool = None
    elem_to_fit: np.ndarray = None  # only 1 element at a time is support atm, a list otherwise
    lmin: float = None
    lmax: float = None
    ldelta: float = None
    fwhm: float = None  # FWHM coming from resolution, constant for all stars
    #macroturb: float = None  # macroturbulence km/s, constant for all stars if not fitted
    rot: float = None  # rotation km/s, constant for all stars
    fitting_mode: str = None  # "lbl" = line by line or "all" or "lbl_quick"
    output_folder: str = None

    global_temp_dir: str = None
    line_begins_sorted: np.ndarray = None
    line_ends_sorted: np.ndarray = None
    line_centers_sorted: np.ndarray = None

    seg_begins: np.ndarray = None
    seg_ends: np.ndarray = None

    depart_bin_file_dict: dict = None
    depart_aux_file_dict: dict = None
    model_atom_file_dict: dict = None
    ndimen: int = None
    spec_input_path: str = None

    grids_amount: int = 50
    abund_bound: float = 1.0

    init_guess_dict: dict = None    # initial guess for elements, if given

    # bounds for the minimization
    bound_min_macro = 0         # km/s
    bound_max_macro = 30
    bound_min_micro = 0.01         # km/s
    bound_max_micro = 5
    bound_min_abund = -40       # [X/Fe]
    bound_max_abund = 100
    bound_min_met = -4          # [Fe/H]
    bound_max_met = 0.5
    bound_min_doppler = -1      # km/s
    bound_max_doppler = 1

    def __init__(self, specname: str, teff: float, logg: float, rv: float, met: float, micro: float, macro: float,
                 line_list_path_trimmed: str, init_param_guess: list, elem_abund=None):
        self.spec_name: str = str(specname)
        self.spec_path: str = os.path.join(self.spec_input_path, str(specname))
        self.teff: float = float(teff)
        self.logg: float = float(logg)
        self.met: float = float(met)
        self.rv: float = float(rv)  # RV of star (given, but is fitted with extra doppler shift)
        self.doppler_shift: float = 0.0  # doppler shift; added to RV (fitted)
        if elem_abund is not None:
            self.elem_abund: float = float(elem_abund)  # initial abundance of element as a guess if lbl quick
        else:
            self.elem_abund = None
        if Spectra.fit_microturb == "Input":
            self.vmicro: float = float(micro)  # microturbulence. Set if it is given in input
        else:
            self.vmicro = None
        self.macroturb: float = float(macro)  # macroturbulence km/s, constant for all stars if not fitted
        self.temp_dir: str = os.path.join(Spectra.global_temp_dir, self.spec_name,
                                          '')  # temp directory, including date and name of the star fitted
        create_dir(self.temp_dir)  # create temp directory

        self.abund_to_gen = None  # array with generated abundances for lbl quick

        self.init_param_guess: list = None  # guess for minimzation
        self.initial_simplex_guess: list = None
        self.minim_bounds: list = []
        self.set_param_guess(init_param_guess)

        self.line_list_path_trimmed = line_list_path_trimmed  # location of trimmed files

        self.ts = TurboSpectrum(
            turbospec_path=self.turbospec_path,
            interpol_path=self.interpol_path,
            line_list_paths=self.line_list_path_trimmed,
            marcs_grid_path=self.model_atmosphere_grid_path,
            marcs_grid_list=self.model_atmosphere_list,
            model_atom_path=self.model_atom_path,
            departure_file_path=self.departure_file_path)

        self.wave_ob, self.flux_ob = np.loadtxt(self.spec_path, usecols=(0, 1), unpack=True,
                                                dtype=float)  # observed spectra

    def set_param_guess(self, init_param_guess: list):
        """
        Converts init param guess list to the 2D list for the simplex calculation
        :param init_param_guess: Initial list equal to n x ndimen+1, where ndimen = number of fitted parameters
        """
        # make an array for initial guess equal to n x ndimen+1
        initial_guess = np.empty((self.ndimen + 1, self.ndimen))
        # 17.11.2022: Tried random guesses. But they DO affect the result if the random guesses are way off.
        # Trying with linspace. Should be better I hope
        min_microturb = 0.9  # set bounds for all elements here, change later if needed
        max_microturb = 1.3  # km/s ? cannot be less than 0
        min_macroturb = 0.2  # km/s; cannot be less than 0
        max_macroturb = 8.0
        min_abundance = -1  # either [Fe/H] or [X/Fe] here
        max_abundance = 0.4  # for [Fe/H]: hard bounds -4 to 0.5; other elements: bounds are above -40
        min_rv = -1  # km/s i think as well
        max_rv = 1
        # TODO: check that not the same value every time? chance of not fitting at all if all values are same
        microturb_guesses = np.linspace(min_microturb, max_microturb, self.ndimen + 1)
        macroturb_guesses = np.linspace(min_macroturb + np.random.random(1)[0] / 2, max_macroturb + np.random.random(1)[0] / 2, self.ndimen + 1)
        abundance_guesses = np.linspace(min_abundance + np.random.random(1)[0] / 10, max_abundance + np.random.random(1)[0] / 10, self.ndimen + 1)
        rv_guesses = np.linspace(min_rv + np.random.random(1)[0] / 10, max_rv + np.random.random(1)[0] / 10, self.ndimen + 1)

        """# fill the array with input from config file # OLD
        for j in range(Spectra.ndimen):
            for i in range(j, len(init_param_guess), Spectra.ndimen):
                initial_guess[int(i / Spectra.ndimen)][j] = float(init_param_guess[i])"""

        # TODO: order depends on the fitting mode. Make more universal?

        if self.fitting_mode == "all":
            # abund = param[0]
            # dopple = param[1]
            # macroturb = param [2] (if needed)
            initial_guess[:, 0] = abundance_guesses
            if self.fit_met:
                self.minim_bounds.append((self.bound_min_met, self.bound_max_met))
            else:
                self.minim_bounds.append((self.bound_min_abund, self.bound_max_abund))
            initial_guess[:, 1] = rv_guesses
            self.minim_bounds.append((self.bound_min_doppler, self.bound_max_doppler))
            if self.fit_macroturb:
                initial_guess[:, 2] = macroturb_guesses
                self.minim_bounds.append((self.bound_min_macro, self.bound_max_macro))
        elif self.fitting_mode == "lbl":
            # param[0] = added doppler to rv
            # param[1:nelements] = met or abund
            # param[-1] = macro turb IF MACRO FIT
            # param[-2] = micro turb IF MACRO FIT
            # param[-1] = micro turb IF NOT MACRO FIT
            initial_guess, self.minim_bounds = self.get_rv_elem_micro_macro_guess(self.ndimen, min_rv, max_rv,
                                                                                  min_macroturb, max_macroturb,
                                                                                  min_microturb, max_microturb,
                                                                                  min_abundance, max_abundance)
        elif self.fitting_mode == "lbl_quick":
            # param[0] = doppler
            # param[1] = macro turb
            initial_guess, self.minim_bounds = self.get_rv_macro_guess(self.ndimen, min_rv, max_rv, min_macroturb, max_macroturb)
        else:
            ValueError("Unknown fitting mode, choose all or lbl")

        self.init_param_guess = initial_guess[0]
        self.initial_simplex_guess = initial_guess


    def get_elem_micro_guess(self, length: int, min_microturb: float, max_microturb: float, min_abundance: float,
                             max_abundance: float) -> tuple[np.ndarray, list[tuple]]:
        # param[0:nelements-1] = met or abund
        # param[-1] = micro turb

        initial_guess = np.empty((length + 1, length))

        microturb_guesses = np.linspace(min_microturb, max_microturb, length + 1)
        abundance_guesses = np.linspace(min_abundance + np.random.random(1)[0] / 10,
                                        max_abundance + np.random.random(1)[0] / 10, length + 1)

        minim_bounds = []

        for i in range(0, self.nelement):
            if self.init_guess_dict is not None and self.elem_to_fit[i] in self.init_guess_dict[self.spec_name]:
                abund_guess = self.init_guess_dict[self.spec_name][self.elem_to_fit[i]]
                abundance_guesses = np.linspace(abund_guess - 0.1, abund_guess + 0.1, length + 1)
                initial_guess[:, i] = abundance_guesses
                # if initial abundance is given, then linearly give guess +/- 0.1 dex
            else:
                initial_guess[:, i] = abundance_guesses
                # to create new abundance guess for every element
            abundance_guesses = np.random.uniform(min_abundance, max_abundance, length + 1)
            if self.elem_to_fit[i] == "Fe":
                minim_bounds.append((self.bound_min_met, self.bound_max_met))
            else:
                minim_bounds.append((self.bound_min_abund, self.bound_max_abund))

        if self.fit_microturb == "Yes" and not self.atmosphere_type == "3D":
            initial_guess[:, -1] = microturb_guesses
            minim_bounds.append((self.bound_min_micro, self.bound_max_micro))  # micro

        return initial_guess, minim_bounds

    def get_rv_elem_micro_macro_guess(self, length: int, min_rv: float, max_rv: float, min_macroturb: float,
                           max_macroturb: float, min_microturb: float, max_microturb: float, min_abundance: float,
                           max_abundance: float) -> tuple[np.ndarray, list[tuple]]:
        # param[0] = added doppler to rv
        # param[1:nelements] = met or abund
        # param[-1] = macro turb IF MACRO FIT
        # param[-2] = micro turb IF MACRO FIT
        # param[-1] = micro turb IF NOT MACRO FIT

        initial_guess = np.empty((length + 1, length))

        microturb_guesses = np.linspace(min_microturb, max_microturb, length + 1)
        macroturb_guesses = np.linspace(min_macroturb + np.random.random(1)[0] / 2,
                                        max_macroturb + np.random.random(1)[0] / 2, length + 1)
        abundance_guesses = np.linspace(min_abundance + np.random.random(1)[0] / 10,
                                        max_abundance + np.random.random(1)[0] / 10, length + 1)
        rv_guesses = np.linspace(min_rv + np.random.random(1)[0] / 10, max_rv + np.random.random(1)[0] / 10,
                                 length + 1)

        minim_bounds = []

        initial_guess[:, 0] = rv_guesses
        minim_bounds.append((self.bound_min_doppler, self.bound_max_doppler))
        for i in range(1, self.nelement + 1):
            if self.init_guess_dict is not None and self.elem_to_fit[i - 1] in self.init_guess_dict[self.spec_name]:
                abund_guess = self.init_guess_dict[self.spec_name][self.elem_to_fit[i - 1]]
                abundance_guesses = np.linspace(abund_guess - 0.1, abund_guess + 0.1, length + 1)
                initial_guess[:, i] = abundance_guesses
                # if initial abundance is given, then linearly give guess +/- 0.1 dex
            else:
                initial_guess[:, i] = abundance_guesses
                # to create new abundance guess for every element
            abundance_guesses = np.random.uniform(min_abundance, max_abundance, length + 1)
            if self.elem_to_fit[i - 1] == "Fe":
                minim_bounds.append((self.bound_min_met, self.bound_max_met))
            else:
                minim_bounds.append((self.bound_min_abund, self.bound_max_abund))
        if self.fit_macroturb:
            initial_guess[:, -1] = macroturb_guesses
            if self.fit_microturb == "Yes" and not self.atmosphere_type == "3D":
                initial_guess[:, -2] = microturb_guesses
                minim_bounds.append((self.bound_min_micro, self.bound_max_micro))  # first adding micro
            minim_bounds.append((self.bound_min_macro, self.bound_max_macro))  # last one macro
        else:
            if self.fit_microturb == "Yes" and not self.atmosphere_type == "3D":
                initial_guess[:, -1] = microturb_guesses
                minim_bounds.append((self.bound_min_micro, self.bound_max_micro))

        return initial_guess, minim_bounds

    def get_rv_macro_guess(self, length: int, min_rv: float, max_rv: float, min_macroturb: float,
                           max_macroturb: float) -> tuple[np.ndarray, list[tuple]]:
        # param[0] = doppler
        # param[1] = macro turb
        initial_guess = np.empty((length + 1, length))
        minim_bounds = [(self.bound_min_doppler, self.bound_max_doppler)]

        rv_guesses = np.linspace(min_rv + np.random.random(1)[0] / 10, max_rv - np.random.random(1)[0] / 10, length + 1)
        if np.max(rv_guesses) > self.bound_max_doppler or np.min(rv_guesses) < self.bound_min_doppler:
            rv_guesses = np.linspace(self.bound_min_doppler + np.random.random(1)[0] / 10,
                                     self.bound_max_doppler - np.random.random(1)[0] / 10, length + 1)
        initial_guess[:, 0] = rv_guesses

        if self.fit_macroturb:
            macroturb_guesses = np.linspace(min_macroturb + np.random.random(1)[0] / 2,
                                            max_macroturb + np.random.random(1)[0] / 2, length + 1)
            minim_bounds.append((self.bound_min_macro, self.bound_max_macro))
            if np.max(macroturb_guesses) > self.bound_max_macro or np.min(macroturb_guesses) < self.bound_min_macro:
                macroturb_guesses = np.linspace(self.bound_min_macro + np.random.random(1)[0] / 2,
                                                self.bound_max_macro - np.random.random(1)[0] / 2, length + 1)

            initial_guess[:, 1] = macroturb_guesses

        return initial_guess, minim_bounds

    def configure_and_run_ts(self, met: float, elem_abund: dict, vmicro: float, lmin: float, lmax: float,
                             windows_flag: bool, temp_dir=None):
        """
        Configures TurboSpectrum depending on input parameters and runs either NLTE or LTE
        :param met: metallicity of star
        :param elem_abund: dictionary with iron and elemental abundances
        :param vmicro: microturbulence parameter
        :param lmin: minimum wavelength where spectra are computed
        :param lmax: maximum wavelength where spectra are computed
        :param windows_flag - False for lbl, True for all lines or lbl quick
        :param temp_dir: Temporary directory where to save, if not given, then self.temp_dir is used
        """
        if temp_dir is None:
            temp_dir = self.temp_dir
        else:
            temp_dir = temp_dir
        if self.nlte_flag:
            self.ts.configure(t_eff=self.teff, log_g=self.logg, metallicity=met, turbulent_velocity=vmicro,
                              lambda_delta=self.ldelta, lambda_min=lmin, lambda_max=lmax,
                              free_abundances=elem_abund, temp_directory=temp_dir, nlte_flag=True, verbose=False,
                              atmosphere_dimension=self.atmosphere_type, windows_flag=windows_flag,
                              segment_file=self.segment_file, line_mask_file=self.linemask_file,
                              depart_bin_file=self.depart_bin_file_dict, depart_aux_file=self.depart_aux_file_dict,
                              model_atom_file=self.model_atom_file_dict)
        else:
            self.ts.configure(t_eff=self.teff, log_g=self.logg, metallicity=met, turbulent_velocity=vmicro,
                              lambda_delta=self.ldelta, lambda_min=lmin, lambda_max=lmax,
                              free_abundances=elem_abund, temp_directory=temp_dir, nlte_flag=False, verbose=False,
                              atmosphere_dimension=self.atmosphere_type, windows_flag=windows_flag,
                              segment_file=self.segment_file, line_mask_file=self.linemask_file)
        self.ts.run_turbospectrum_and_atmosphere()

    def fit_all(self):
        """
        Fit all lines at once, trying to minimise chi squared
        :return: Result is a string containing Fitted star name, abundance, RV, chi squared and macroturbulence
        """
        # timing how long it took
        time_start = time.perf_counter()

        res = minimize(all_broad_abund_chi_sqr, self.init_param_guess, args=self, method='Nelder-Mead', bounds=self.minim_bounds,
                       options={'maxiter': self.ndimen * 50, 'disp': True,
                                'initial_simplex': self.initial_simplex_guess, 'xatol': 0.05, 'fatol': 0.05})
        # print final result from minimazation
        print(res.x)

        if self.fit_macroturb:  # if fitted macroturbulence, return it
            result = f"{self.spec_name} {res.x[0]} {res.x[1]} {res.fun} {res.x[2]}"
        else:  # otherwise return whatever constant macroturbulence was given in the config
            result = f"{self.spec_name} {res.x[0]} {res.x[1]} {res.fun} {self.macroturb}"

        time_end = time.perf_counter()
        print(f"Total runtime was {(time_end - time_start) / 60.:2f} minutes.")
        # remove all temporary files
        #shutil.rmtree(self.temp_dir)
        return result

    def generate_grid_for_lbl(self, abund_to_gen) -> list:
        """
        Generates grids for lbl quick method. Grids are centered at input metallicity/abundance. Number of grids and
        bounds depend on self.abund_bound, self.grids_amount
        :return: List corresponding to self.abund_to_gen with same locations. True: generation success. False: not
        """
        success = []

        for abund_to_use in abund_to_gen:
            if self.met > 0.5 or self.met < -4.0 or abund_to_use < -40 or (
                    Spectra.fit_met and (abund_to_use < -4.0 or abund_to_use > 0.5)):
                success.append(False)  # if metallicity or abundance too weird, then fail
            else:
                if Spectra.fit_met:
                    item_abund = {"Fe": abund_to_use}
                    met = abund_to_use
                else:
                    item_abund = {"Fe": self.met, Spectra.elem_to_fit[0]: abund_to_use + self.met}
                    met = self.met

                if self.vmicro is not None:  # sets microturbulence here
                    vmicro = self.vmicro
                else:
                    vmicro = calculate_vturb(self.teff, self.logg, met)

                temp_dir = os.path.join(self.temp_dir, f"{abund_to_use}", '')
                create_dir(temp_dir)
                self.configure_and_run_ts(met, item_abund, vmicro, self.lmin, self.lmax, False, temp_dir=temp_dir)

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
        if self.fit_met:  # grids generated centered on input metallicity or abundance
            input_abund = self.met
        else:
            input_abund = self.elem_abund
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
                                                          usecols=(0, 1), unpack=True)  # TODO asyncio here?
                grid_spectra[abund] = [wave_mod_orig, flux_mod_orig]

        for j in range(len(Spectra.line_begins_sorted)):
            #time_start = time.perf_counter()
            #print(f"Fitting line at {Spectra.line_centers_sorted[j]} angstroms")
            # TODO: improve next 3 lines of code.
            #for k in range(len(Spectra.seg_begins)):
            #    if Spectra.seg_ends[k] >= Spectra.line_centers_sorted[j] > Spectra.seg_begins[k]:
            #        start = k
            #print(Spectra.line_centers_sorted[j], Spectra.seg_begins[start], Spectra.seg_ends[start])
            # each line contains spectra name and fitted line. then to the right of it abundance with chi-sqr are added
            result_one_line = f"{self.spec_name} {Spectra.line_centers_sorted[j]} {Spectra.line_begins_sorted[j]} " \
                              f"{Spectra.line_ends_sorted[j]}"

            chi_squares = []

            for abund, success in zip(self.abund_to_gen, success_grid_gen):
                if success:  # for each successful grid find chi squared with best fit parameters
                    wave_abund, flux_abund = grid_spectra[abund][0], grid_spectra[abund][1]
                    res = minimize(lbl_broad_abund_chi_sqr_quick, self.init_param_guess, args=(self,
                                                                                               Spectra.line_begins_sorted[j] - 5.,
                                                                                               Spectra.line_ends_sorted[j] + 5.,
                                                                                               wave_abund,
                                                                                               flux_abund),
                                   bounds=self.minim_bounds,
                                   method='Nelder-Mead',
                                   options={'maxiter': Spectra.ndimen * 50, 'disp': True,
                                            'initial_simplex': self.initial_simplex_guess,
                                            'xatol': 0.05, 'fatol': 0.05})
                    #print(res.x)
                    if Spectra.fit_macroturb:  # if fitted macroturbulence
                        macroturb = res.x[1]
                    else:
                        macroturb = self.macroturb
                    if self.vmicro is not None:  # if microturbulence was given or finds whatever input was used
                        vmicro = self.vmicro
                    else:
                        if self.fit_met:
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
            index_min_chi_square = np.argmin(chi_squares)
            min_chi_sqr_spectra_path = os.path.join(self.temp_dir, f"{self.abund_to_gen[index_min_chi_square]}",
                                                    'spectrum_00000000.spec')
            # appends that best fit spectra to the total output spectra. NOT convolved. separate abundance for each line
            wave_result, flux_norm_result, flux_result = np.loadtxt(min_chi_sqr_spectra_path,
                                                                    unpack=True)  # TODO asyncio here? or just print at the very end?
            with open(f"{self.output_folder}result_spectrum_{self.spec_name}.spec", 'a') as g:
                # g = open(f"{self.output_folder}result_spectrum_{self.spec_name}.spec", 'a')
                for k in range(len(wave_result)):
                    print("{}  {}  {}".format(wave_result[k], flux_norm_result[k], flux_result[k]), file=g)

            #time_end = time.perf_counter()
            #print("Total runtime was {:.2f} minutes.".format((time_end - time_start) / 60.))

        # g.close()

        return result

    def fit_lbl_v3(self):
        result = []

        for j in range(len(Spectra.line_begins_sorted)):
            time_start = time.perf_counter()
            print(f"Fitting line at {Spectra.line_centers_sorted[j]} angstroms")

            result.append(self.fit_one_line_v3_better_guess(j))

            time_end = time.perf_counter()
            print("Total runtime was {:.2f} minutes.".format((time_end - time_start) / 60.))

        # g.close()
        # h.close()

        return result

    def fit_one_line_v3_better_guess(self, j):
        if self.fit_macroturb:
            ndimen = 2
        else:
            ndimen = 1
        initial_guess_quick = np.empty((ndimen + 1, ndimen))
        min_macroturb = 0.2  # km/s; cannot be less than 0
        max_macroturb = 8.0
        min_rv = -1  # km/s i think as well
        max_rv = 1

        macroturb_guesses = np.linspace(min_macroturb, max_macroturb, ndimen + 1)
        rv_guesses = np.linspace(min_rv, max_rv, ndimen + 1)
        minim_bounds = []

        # param[0] = doppler
        # param[1] = macro turb
        initial_guess_quick[:, 0] = rv_guesses
        minim_bounds.append((self.bound_min_doppler, self.bound_max_doppler))
        if self.fit_macroturb:
            initial_guess_quick[:, 1] = macroturb_guesses
            minim_bounds.append((self.bound_min_macro, self.bound_max_macro))

        for k in range(len(Spectra.seg_begins)):  # TODO redo this one, very ugly
            if Spectra.seg_ends[k] >= Spectra.line_centers_sorted[j] > Spectra.seg_begins[k]:
                start = k
        print(Spectra.line_centers_sorted[j], Spectra.seg_begins[start], Spectra.seg_ends[start])
        self.ts.line_list_paths = [
            get_trimmed_lbl_path_name(self.elem_to_fit, self.line_list_path_trimmed, Spectra.segment_file, j,
                                      start)]
        grids_for_func = 3
        delta_abund = 0.3
        abund_to_gen_center = 0.0
        min_limit_abund = -2.5
        abund_to_gen = np.linspace(abund_to_gen_center - delta_abund, abund_to_gen_center + delta_abund, grids_for_func)
        success_grid_gen = self.generate_grid_for_lbl(abund_to_gen)  # generate grids
        abund_to_gen = abund_to_gen[success_grid_gen]
        while np.size(abund_to_gen) < grids_for_func:
            success_grid_gen = np.asarray(success_grid_gen)[success_grid_gen]
            min_abund_to_gen = min_abund_to_gen - delta_abund
            if min_abund_to_gen < min_limit_abund:
                return self.fit_one_line(j, self.init_param_guess, self.initial_simplex_guess)
            abund_to_gen = np.append(min_abund_to_gen)
            success_grid_gen_new = self.generate_grid_for_lbl(abund_to_gen)  # generate grids
            success_grid_gen = np.insert(success_grid_gen, 0, success_grid_gen_new[0], axis=0)
            abund_to_gen = abund_to_gen[success_grid_gen]
        grid_spectra = {}
        # read spectra from generated grids and keep in memory to not waste time reading them each time
        for abund in abund_to_gen:
            spectra_grid_path = os.path.join(self.temp_dir, f"{abund}", '')
            wave_mod_orig, flux_mod_orig = np.loadtxt(f'{spectra_grid_path}/spectrum_00000000.spec',
                                                      usecols=(0, 1), unpack=True)  # TODO asyncio here?
            grid_spectra[abund] = [wave_mod_orig, flux_mod_orig]
        chi_squares = []
        macroturbs = []
        doppler_shifts = []
        for abund in abund_to_gen:
            wave_abund, flux_abund = grid_spectra[abund][0], grid_spectra[abund][1]
            res = minimize(lbl_broad_abund_chi_sqr_quick, initial_guess_quick[0], args=(self,
                                                                                       Spectra.line_begins_sorted[
                                                                                           j] - 5.,
                                                                                       Spectra.line_ends_sorted[
                                                                                           j] + 5.,
                                                                                       wave_abund,
                                                                                       flux_abund),
                           bounds=minim_bounds,
                           method='Nelder-Mead',
                           options={'maxiter': Spectra.ndimen * 50, 'disp': True,
                                    'initial_simplex': initial_guess_quick,
                                    'xatol': 0.05, 'fatol': 0.05})

            chi_squares.append(res.fun)
            doppler_shifts.append(res.x[0])
            if self.fit_macroturb:
                macroturbs.append(res.x[1])
        a, b, c = get_second_degree_polynomial(abund_to_gen, chi_squares)
        if a <= 0:
            return self.fit_one_line(j, self.init_param_guess, self.initial_simplex_guess)
        chi_sqr_func = lambda t: a * t * t + b * t + c
        min_abund = minimize(chi_sqr_func, [0]).x[0]
        min_chi_sqr = chi_sqr_func(min_abund)
        print(min_abund, min_chi_sqr)

        initial_guess = np.empty((self.ndimen + 1, self.ndimen))

        if self.fit_macroturb:
            macroturb_guesses = np.linspace(min(macroturbs), max(macroturbs), self.ndimen + 1)
        abundance_guesses = np.linspace(min_abund - 0.1, min_abund + 0.1, self.ndimen + 1)
        rv_guesses = np.linspace(min(doppler_shifts), max(doppler_shifts), self.ndimen + 1)

        initial_guess[:, 0] = rv_guesses
        initial_guess[:, 1] = abundance_guesses

        if self.fit_macroturb:
            initial_guess[:, -1] = macroturb_guesses
            if self.fit_microturb == "Yes" and not self.atmosphere_type == "3D":
                initial_guess[:, -2] = self.initial_simplex_guess[:, -2]

        else:
            if self.fit_microturb == "Yes" and not self.atmosphere_type == "3D":
                initial_guess[:, -1] = self.initial_simplex_guess[:, -1]
        return self.fit_one_line(j, initial_guess[0], initial_guess)

    def fit_lbl(self) -> list:
        """
        Fits line by line, by going through each line in the linelist and computing best abundance/met with chi sqr.
        Also fits doppler shift and can fit micro and macro turbulence
        :return: List with the results. Each element is a string containing file name, center start and end of the line,
        Best fit abundance/met, doppler shift, microturbulence, macroturbulence and chi-squared.
        """
        result = []

        for line_number in range(len(Spectra.line_begins_sorted)):
            time_start = time.perf_counter()
            print(f"Fitting line at {Spectra.line_centers_sorted[line_number]} angstroms")

            result.append(self.fit_one_line(line_number, self.init_param_guess, self.initial_simplex_guess))

            time_end = time.perf_counter()
            print("Total runtime was {:.2f} minutes.".format((time_end - time_start) / 60.))

        # g.close()
        # h.close()

        return result

    def fit_lbl_v2(self) -> list:
        """
        Fits line by line, by going through each line in the linelist and computing best abundance/met with chi sqr.
        Also fits doppler shift and can fit micro and macro turbulence
        :return: List with the results. Each element is a string containing file name, center start and end of the line,
        Best fit abundance/met, doppler shift, microturbulence, macroturbulence and chi-squared.
        """
        if self.fit_macroturb and self.macroturb == 0:
            self.macroturb = 10

        result = []

        for line_number in range(len(Spectra.line_begins_sorted)):
            time_start = time.perf_counter()
            print(f"Fitting line at {Spectra.line_centers_sorted[line_number]} angstroms")

            result.append(self.fit_one_line_v2(line_number))

            time_end = time.perf_counter()
            print("Total runtime was {:.2f} minutes.".format((time_end - time_start) / 60.))

        # g.close()
        # h.close()

        return result

    def fit_one_line(self, line_number, init_param_guess, initial_simplex_guess):
        for k in range(len(Spectra.seg_begins)):  # TODO redo this one, very ugly
            if Spectra.seg_ends[k] >= Spectra.line_centers_sorted[line_number] > Spectra.seg_begins[k]:
                start = k
        print(Spectra.line_centers_sorted[line_number], Spectra.seg_begins[start], Spectra.seg_ends[start])
        self.ts.line_list_paths = [
            get_trimmed_lbl_path_name(self.elem_to_fit, self.line_list_path_trimmed, Spectra.segment_file, line_number,
                                      start)]
        res = minimize(lbl_broad_abund_chi_sqr, init_param_guess, args=(self,
                                                                             Spectra.line_begins_sorted[line_number] - 5.,
                                                                             Spectra.line_ends_sorted[line_number] + 5.),
                       bounds=self.minim_bounds,
                       method='Nelder-Mead',
                       options={'maxiter': Spectra.ndimen * 50, 'disp': True,
                                'initial_simplex': initial_simplex_guess,
                                'xatol': 0.05, 'fatol': 0.05})
        print(res.x)
        if Spectra.fit_met:
            met_index = np.where(Spectra.elem_to_fit == "Fe")[0][0]
            met = res.x[met_index + 1]  # offset 1: since 0th parameter is always doppler
        else:
            met = self.met
        elem_abund_dict = {"Fe": met}
        for i in range(Spectra.nelement):
            # Spectra.elem_to_fit[i] = element name
            # param[1:nelement] = abundance of the element
            elem_name = Spectra.elem_to_fit[i]
            if elem_name != "Fe":
                elem_abund_dict[elem_name] = res.x[i + 1]  # + met #TODO need metallicity here or not?
        doppler_fit = res.x[0]
        if self.vmicro is not None:  # Input given
            microturb = self.vmicro
        else:
            if Spectra.fit_microturb == "No" and Spectra.atmosphere_type == "1D":
                microturb = calculate_vturb(self.teff, self.logg, met)
            elif Spectra.fit_microturb == "Yes" and Spectra.atmosphere_type == "1D":
                if Spectra.fit_macroturb:
                    microturb = res.x[-2]  # if macroturb fit, then last param is macroturb
                else:
                    microturb = res.x[-1]  # if no macroturb fit, then last param is microturb
            elif Spectra.fit_microturb == "Input":  # just for safety's sake, normally should take in the input above anyway
                raise ValueError(
                    "Microturb not given? Did you remember to set microturbulence in parameters? Or is there "
                    "a problem in the code?")
            else:
                microturb = 2.0
        if Spectra.fit_macroturb:
            macroturb = res.x[-1]  # last is always macroturb, if fitted
        else:
            macroturb = self.macroturb
        result_output = f"{self.spec_name} {Spectra.line_centers_sorted[line_number]} {Spectra.line_begins_sorted[line_number]} " \
                        f"{Spectra.line_ends_sorted[line_number]} {doppler_fit}"
        for key in elem_abund_dict:
            result_output += f" {elem_abund_dict[key]}"
        result_output += f" {microturb} {macroturb} {res.fun}"
        one_result = result_output  # out = open(f"{temp_directory}spectrum_00000000_convolved.spec", 'w')
        wave_result, flux_norm_result, flux_result = np.loadtxt(f"{self.temp_dir}spectrum_00000000.spec",
                                                                unpack=True)
        with open(f"{self.output_folder}result_spectrum_{self.spec_name}.spec", 'a') as g:
            # g = open(f"{self.output_folder}result_spectrum_{self.spec_name}.spec", 'a')
            for k in range(len(wave_result)):
                print("{}  {}  {}".format(wave_result[k], flux_norm_result[k], flux_result[k]), file=g)
        wave_result, flux_norm_result = np.loadtxt(f"{self.temp_dir}spectrum_00000000_convolved.spec", unpack=True)
        with open(f"{self.output_folder}result_spectrum_{self.spec_name}_convolved.spec", 'a') as h:
            # h = open(f"{self.output_folder}result_spectrum_{self.spec_name}_convolved.spec", 'a')
            for k in range(len(wave_result)):
                print("{}  {}".format(wave_result[k], flux_norm_result[k]), file=h)
        # os.system("rm ../output_files/spectrum_{:08d}_convolved.spec".format(i + 1))
        return one_result

    def fit_one_line_v2(self, line_number):
        for k in range(len(Spectra.seg_begins)):  # TODO redo this one, very ugly
            if Spectra.seg_ends[k] >= Spectra.line_centers_sorted[line_number] > Spectra.seg_begins[k]:
                start = k
        print(Spectra.line_centers_sorted[line_number], Spectra.seg_begins[start], Spectra.seg_ends[start])
        self.ts.line_list_paths = [
            get_trimmed_lbl_path_name(self.elem_to_fit, self.line_list_path_trimmed, Spectra.segment_file, line_number,
                                      start)]

        ndimen = self.nelement
        if self.fit_microturb:
            ndimen += 1
        param_guess, min_bounds = self.get_elem_micro_guess(ndimen, 0.9, 1.3, -1, 0.4)

        res = minimize(lbl_broad_abund_chi_sqr_v2, param_guess[0], args=(self,
                                                                             Spectra.line_begins_sorted[line_number] - 5.,
                                                                             Spectra.line_ends_sorted[line_number] + 5.),
                       bounds=min_bounds,
                       method='Nelder-Mead',
                       options={'maxfev': Spectra.nelement * 50, 'disp': True,
                                'initial_simplex': param_guess,
                                'xatol': 0.01, 'fatol': 0.01})
        print(res.x)
        if Spectra.fit_met:
            met_index = np.where(Spectra.elem_to_fit == "Fe")[0][0]
            met = res.x[met_index]
        else:
            met = self.met
        elem_abund_dict = {"Fe": met}
        for i in range(Spectra.nelement):
            # Spectra.elem_to_fit[i] = element name
            # param[1:nelement] = abundance of the element
            elem_name = Spectra.elem_to_fit[i]
            if elem_name != "Fe":
                elem_abund_dict[elem_name] = res.x[i]  # + met
        doppler_fit = self.doppler_shift
        if self.vmicro is not None:  # Input given
            microturb = self.vmicro
        else:
            if Spectra.fit_microturb == "No" and Spectra.atmosphere_type == "1D":
                microturb = calculate_vturb(self.teff, self.logg, met)
            elif Spectra.fit_microturb == "Yes" and Spectra.atmosphere_type == "1D":
                microturb = res.x[-1]  # if no macroturb fit, then last param is microturb
            elif Spectra.fit_microturb == "Input":  # just for safety's sake, normally should take in the input above anyway
                raise ValueError(
                    "Microturb not given? Did you remember to set microturbulence in parameters? Or is there "
                    "a problem in the code?")
            else:
                microturb = 2.0
        macroturb = self.macroturb
        result_output = f"{self.spec_name} {Spectra.line_centers_sorted[line_number]} {Spectra.line_begins_sorted[line_number]} " \
                        f"{Spectra.line_ends_sorted[line_number]} {doppler_fit}"
        for key in elem_abund_dict:
            result_output += f" {elem_abund_dict[key]}"
        result_output += f" {microturb} {macroturb} {res.fun}"
        one_result = result_output  # out = open(f"{temp_directory}spectrum_00000000_convolved.spec", 'w')
        wave_result, flux_norm_result, flux_result = np.loadtxt(f"{self.temp_dir}spectrum_00000000.spec",
                                                                unpack=True)
        with open(f"{self.output_folder}result_spectrum_{self.spec_name}.spec", 'a') as g:
            # g = open(f"{self.output_folder}result_spectrum_{self.spec_name}.spec", 'a')
            for k in range(len(wave_result)):
                print("{}  {}  {}".format(wave_result[k], flux_norm_result[k], flux_result[k]), file=g)
        wave_result, flux_norm_result = np.loadtxt(f"{self.temp_dir}spectrum_00000000_convolved.spec", unpack=True)
        with open(f"{self.output_folder}result_spectrum_{self.spec_name}_convolved.spec", 'a') as h:
            # h = open(f"{self.output_folder}result_spectrum_{self.spec_name}_convolved.spec", 'a')
            for k in range(len(wave_result)):
                print("{}  {}".format(wave_result[k], flux_norm_result[k]), file=h)
        # os.system("rm ../output_files/spectrum_{:08d}_convolved.spec".format(i + 1))
        return one_result


def get_second_degree_polynomial(x: list, y: list) -> tuple[int, int, int]:
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

def lbl_broad_abund_chi_sqr_quick(param: list, spectra_to_fit: Spectra, lmin: float, lmax: float,
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

    doppler = spectra_to_fit.rv + param[0]

    if Spectra.fit_macroturb:
        macroturb = param[1]
    else:
        macroturb = spectra_to_fit.macroturb

    wave_ob = spectra_to_fit.wave_ob / (1 + (doppler / 299792.))

    if spectra_to_fit.met < -4.0 or spectra_to_fit.met > 0.5 or macroturb < 0.0:
        chi_square = 9999.9999
    else:
        chi_square = calculate_lbl_chi_squared(None, wave_ob,
                                               spectra_to_fit.flux_ob, wave_mod_orig, flux_mod_orig, Spectra.fwhm, lmax,
                                               lmin, macroturb,
                                               Spectra.rot, save_convolved=False)

    #print(param[0], chi_square, macroturb)  # takes 50%!!!! extra time to run if using print statement here

    return chi_square


def lbl_broad_abund_chi_sqr(param: list, spectra_to_fit: Spectra, lmin: float, lmax: float) -> float:
    """
    Goes line by line, tries to call turbospectrum and find best fit spectra by varying parameters: abundance, doppler
    shift and if needed micro + macro turbulence
    :param param: Parameters list with the current evaluation guess
    :param spectra_to_fit: Spectra to fit
    :param lmin: Start of the line [AA]
    :param lmax: End of the line [AA]
    :return: best fit chi squared
    """
    # new: now includes several elements
    # param[0] = added doppler to rv
    # param[1:nelements] = met or abund
    # param[-1] = macro turb IF MACRO FIT
    # param[-2] = micro turb IF MACRO FIT
    # param[-1] = micro turb IF NOT MACRO FIT

    if Spectra.fit_met:
        met_index = np.where(Spectra.elem_to_fit == "Fe")[0][0]
        met = param[met_index + 1]  # offset 1: since 0th parameter is always doppler
    else:
        met = spectra_to_fit.met
    elem_abund_dict = {"Fe": met}

    abundances = [met]

    for i in range(Spectra.nelement):
        # Spectra.elem_to_fit[i] = element name
        # param[1:nelement] = abundance of the element
        elem_name = Spectra.elem_to_fit[i]
        if elem_name != "Fe":
            elem_abund_dict[elem_name] = param[i + 1] + met
            abundances.append(param[i + 1])
    doppler = spectra_to_fit.rv + param[0]
    if spectra_to_fit.vmicro is not None:  # Input given
        microturb = spectra_to_fit.vmicro
    else:
        if Spectra.fit_microturb == "No" and Spectra.atmosphere_type == "1D":
            microturb = calculate_vturb(spectra_to_fit.teff, spectra_to_fit.logg, met)
        elif Spectra.fit_microturb == "Yes" and Spectra.atmosphere_type == "1D":
            if Spectra.fit_macroturb:
                microturb = param[-2]  # if macroturb fit, then last param is macroturb
            else:
                microturb = param[-1]  # if no macroturb fit, then last param is microturb
        elif Spectra.fit_microturb == "Input":  # just for safety's sake, normally should take in the input above anyway
            raise ValueError("Microturb not given? Did you remember to set microturbulence in parameters? Or is there "
                             "a problem in the code?")
        else:
            microturb = 2.0
    if Spectra.fit_macroturb:
        macroturb = param[-1]  # last is always macroturb, if fitted
    else:
        macroturb = spectra_to_fit.macroturb

    wave_ob = spectra_to_fit.wave_ob / (1 + (doppler / 299792.))

    if microturb <= 0.0 or macroturb < 0.0 or min(abundances) < -40 or met < -4.0 or met > 0.5:
        chi_square = 9999.9999
    else:
        spectra_to_fit.configure_and_run_ts(met, elem_abund_dict, microturb, lmin, lmax, False)

        if os_path.exists('{}/spectrum_00000000.spec'.format(spectra_to_fit.temp_dir)) and os.stat(
                '{}/spectrum_00000000.spec'.format(spectra_to_fit.temp_dir)).st_size != 0:
            wave_mod_orig, flux_mod_orig = np.loadtxt(f'{spectra_to_fit.temp_dir}/spectrum_00000000.spec',
                                                      usecols=(0, 1), unpack=True)
            chi_square = calculate_lbl_chi_squared(spectra_to_fit.temp_dir, wave_ob,
                                                   spectra_to_fit.flux_ob, wave_mod_orig, flux_mod_orig, Spectra.fwhm,
                                                   lmax, lmin, macroturb,
                                                   Spectra.rot)
        elif os_path.exists('{}/spectrum_00000000.spec'.format(spectra_to_fit.temp_dir)) and os.stat(
                '{}/spectrum_00000000.spec'.format(spectra_to_fit.temp_dir)).st_size == 0:
            chi_square = 999.99
            print("empty spectrum file.")
        else:
            chi_square = 9999.9999
            print("didn't generate spectra or atmosphere")

    output_print = f""
    for key in elem_abund_dict:
        output_print += f" {key} {elem_abund_dict[key]}"
    print(output_print, doppler, microturb, macroturb, chi_square)

    return chi_square


def lbl_broad_abund_chi_sqr_v2(param: list, spectra_to_fit: Spectra, lmin: float, lmax: float) -> float:
    """
    Goes line by line, tries to call turbospectrum and find best fit spectra by varying parameters: abundance, doppler
    shift and if needed micro + macro turbulence
    :param param: Parameters list with the current evaluation guess
    :param spectra_to_fit: Spectra to fit
    :param lmin: Start of the line [AA]
    :param lmax: End of the line [AA]
    :return: best fit chi squared
    """
    # new: now includes several elements
    # param[-1] = vmicro
    # param[0:nelements - 1] = met or abund

    if Spectra.fit_met:
        met_index = np.where(Spectra.elem_to_fit == "Fe")[0][0]
        met = param[met_index]  # no offset
    else:
        met = spectra_to_fit.met
    elem_abund_dict = {"Fe": met}

    abundances = [met]

    for i in range(Spectra.nelement):
        # Spectra.elem_to_fit[i] = element name
        # param[1:nelement] = abundance of the element
        elem_name = Spectra.elem_to_fit[i]
        if elem_name != "Fe":
            elem_abund_dict[elem_name] = param[i] + met
            abundances.append(param[i])

    if spectra_to_fit.vmicro is not None:  # Input given
        microturb = spectra_to_fit.vmicro
    else:
        if Spectra.fit_microturb == "No" and Spectra.atmosphere_type == "1D":
            microturb = calculate_vturb(spectra_to_fit.teff, spectra_to_fit.logg, met)
        elif Spectra.fit_microturb == "Yes" and Spectra.atmosphere_type == "1D":
            microturb = param[-1]
        elif Spectra.fit_microturb == "Input":  # just for safety's sake, normally should take in the input above anyway
            raise ValueError("Microturb not given? Did you remember to set microturbulence in parameters? Or is there "
                             "a problem in the code?")
        else:
            microturb = 2.0

    macroturb = 9999    # for printing only here, in case not fitted
    if microturb <= 0.0 or min(abundances) < -40 or met < -4.0 or met > 0.5:
        chi_square = 9999.9999
    else:
        spectra_to_fit.configure_and_run_ts(met, elem_abund_dict, microturb, lmin, lmax, False)

        if os_path.exists('{}/spectrum_00000000.spec'.format(spectra_to_fit.temp_dir)) and os.stat(
                '{}/spectrum_00000000.spec'.format(spectra_to_fit.temp_dir)).st_size != 0:
            wave_mod_orig, flux_mod_orig = np.loadtxt(f'{spectra_to_fit.temp_dir}/spectrum_00000000.spec',
                                                      usecols=(0, 1), unpack=True)
            ndimen = 1
            if spectra_to_fit.fit_macroturb:
                ndimen += 1
            param_guess, min_bounds = spectra_to_fit.get_rv_macro_guess(ndimen,
                                                                        spectra_to_fit.rv - 0.5,
                                                                        spectra_to_fit.rv + 0.5,
                                                                        spectra_to_fit.macroturb - 3,
                                                                        spectra_to_fit.macroturb + 3)
            #print("guesses", param_guess, min_bounds)
            #min_bounds = [(-1, 1)]
            #if spectra_to_fit.fit_macroturb:
            #    min_bounds.append((spectra_to_fit.bound_min_macro, spectra_to_fit.bound_max_macro))

            res = minimize(lbl_broad_abund_chi_sqr_quick, param_guess[0], args=(spectra_to_fit, lmin, lmax,
                                                                                wave_mod_orig, flux_mod_orig),
                           bounds=min_bounds,
                           method='Nelder-Mead',
                           options={'maxiter': Spectra.ndimen * 50, 'disp': True,
                                    'initial_simplex': param_guess,
                                    'xatol': 0.05, 'fatol': 0.05})

            spectra_to_fit.doppler_shift = res.x[0]
            wave_ob = spectra_to_fit.wave_ob / (1 + ((spectra_to_fit.rv + spectra_to_fit.doppler_shift) / 299792.))
            if spectra_to_fit.fit_macroturb:
                spectra_to_fit.macroturb = res.x[1]
            macroturb = spectra_to_fit.macroturb

            chi_square = calculate_lbl_chi_squared(spectra_to_fit.temp_dir, wave_ob,
                                                   spectra_to_fit.flux_ob, wave_mod_orig, flux_mod_orig, Spectra.fwhm,
                                                   lmax, lmin, macroturb,
                                                   Spectra.rot)
        elif os_path.exists('{}/spectrum_00000000.spec'.format(spectra_to_fit.temp_dir)) and os.stat(
                '{}/spectrum_00000000.spec'.format(spectra_to_fit.temp_dir)).st_size == 0:
            chi_square = 999.99
            print("empty spectrum file.")
        else:
            chi_square = 9999.9999
            print("didn't generate spectra or atmosphere")

    output_print = f""
    for key in elem_abund_dict:
        output_print += f" {key} {elem_abund_dict[key]}"
    print(output_print, spectra_to_fit.doppler_shift, microturb, macroturb, chi_square)

    return chi_square


def get_trimmed_lbl_path_name(element: Union[str, np.ndarray], line_list_path_trimmed: str, segment_file: str, j: float,
                              start: float) -> os.path:
    """
    Gets the anem for the lbl trimmed path. Consistent algorithm to always get the same folder name. Takes into account
    element, line center, where molecules are used, segment etc.
    :param element: Name of the element
    :param line_list_path_trimmed: Path to the trimmed line list
    :param segment_file: Name of the segment file
    :param j: center line's numbering
    :param start: Segment's numbering
    :return: path to the folder where to save/already saved trimmed files can exist.
    """
    element_to_print = ""
    if isinstance(element, np.ndarray):
        for elem in element:
            element_to_print += elem
    else:
        element_to_print = element
    return os.path.join(line_list_path_trimmed,
                        f"{segment_file.replace('/', '_').replace('.', '_')}_{element_to_print}_{Spectra.include_molecules}_{j}_{j + 1}_{str(Spectra.line_centers_sorted[j]).replace('.', '_')}_{str(Spectra.seg_begins[start]).replace('.', '_')}_{str(Spectra.seg_ends[start]).replace('.', '_')}",
                        '')


def all_broad_abund_chi_sqr(param, spectra_to_fit: Spectra) -> float:
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
    if Spectra.fit_macroturb:
        macroturb = param[2]
    else:
        macroturb = spectra_to_fit.macroturb

    wave_obs = spectra_to_fit.wave_ob / (1 + (doppler / 299792.))

    if spectra_to_fit.met > 0.5 or spectra_to_fit.met < -4.0 or macroturb < 0.0 or abund < -40 or (
            Spectra.fit_met and (abund < -4.0 or abund > 0.5)):
        chi_square = 9999.9999
    else:
        if Spectra.fit_met:
            item_abund = {"Fe": abund}
            met = abund
            if spectra_to_fit.vmicro is not None:
                vmicro = spectra_to_fit.vmicro
            else:
                vmicro = calculate_vturb(spectra_to_fit.teff, spectra_to_fit.logg, spectra_to_fit.met)
        else:   # Fe: [Fe/H]. X: [X/Fe]. But TS takes [X/H]. Thus convert [X/H] = [X/Fe] + [Fe/H]
            item_abund = {"Fe": spectra_to_fit.met, Spectra.elem_to_fit[0]: abund + spectra_to_fit.met}
            met = spectra_to_fit.met
            if spectra_to_fit.vmicro is not None:
                vmicro = spectra_to_fit.vmicro
            else:
                vmicro = calculate_vturb(spectra_to_fit.teff, spectra_to_fit.logg, spectra_to_fit.met)

        spectra_to_fit.configure_and_run_ts(met, item_abund, vmicro, spectra_to_fit.lmin, spectra_to_fit.lmax, True)

        chi_square = calc_ts_spectra_all_lines(spectra_to_fit.spec_path, spectra_to_fit.temp_dir,
                                               spectra_to_fit.output_folder,
                                               wave_obs, spectra_to_fit.flux_ob,
                                               macroturb, Spectra.fwhm, Spectra.rot,
                                               Spectra.line_begins_sorted, Spectra.line_ends_sorted,
                                               Spectra.seg_begins, Spectra.seg_ends)

    #print(abund, doppler, chi_square, macroturb)

    return chi_square


def create_and_fit_spectra(specname: str, teff: float, logg: float, rv: float, met: float, microturb: float,
                           macroturb: float,
                           initial_guess_string: list, line_list_path_trimmed: str, input_abundance: float) -> list:
    """
    Creates spectra object and fits based on requested fitting mode
    :param specname: Name of the textfile
    :param teff: Teff in K
    :param logg: logg in dex
    :param rv: radial velocity (km/s)
    :param met: metallicity (doesn't matter what if fitting for Fe)
    :param microturb: Microturbulence if given (None is not known or fitted)
    :param macroturb: Macroturbulence if given (None is not known or fitted)
    :param initial_guess_string: initial guess string for simplex fitting minimization method
    :param line_list_path_trimmed: Path to the root of the trimmed line list
    :param input_abundance: Input abundance for grid calculation for lbl quick (doesn't matter what for other stuff)
    :return: result of the fit with the best fit parameters and chi squared
    """
    spectra = Spectra(specname, teff, logg, rv, met, microturb, macroturb, line_list_path_trimmed, initial_guess_string,
                      elem_abund=input_abundance)

    print(f"Fitting {spectra.spec_name}")
    print(f"Teff = {spectra.teff}; logg = {spectra.logg}; RV = {spectra.rv}")

    if Spectra.fitting_mode == "all":
        result = spectra.fit_all()
    elif Spectra.fitting_mode == "lbl":
        if new_lbl_version == 3:
            result = spectra.fit_lbl_v3()
        elif new_lbl_version == 2:
            result = spectra.fit_lbl_v2()
        else:
            result = spectra.fit_lbl()
    elif Spectra.fitting_mode == "lbl_quick":
        result = spectra.fit_lbl_quick()
    else:
        raise ValueError(f"unknown fitting mode {Spectra.fitting_mode}, need all or lbl")
    del spectra
    return result


def load_nlte_files_in_dict(elements_to_fit: list, depart_bin_file: list, depart_aux_file: list, model_atom_file: list) -> tuple[dict, dict, dict]:
    """
    Loads and sorts NLTE elements to fit into respective dictionaries
    :param elements_to_fit: Array of elements to fit
    :param depart_bin_file: Departure binary file location (Fe last if not fitted)
    :param depart_aux_file: Departure aux file location (Fe last if not fitted)
    :param model_atom_file: Model atom file location (Fe last if not fitted)
    :return: 3 dictionaries: NLTE location of elements that exist with keys as element names
    """
    depart_bin_file_dict = {}  # assume that element locations are in the same order as the element to fit
    if Spectra.fit_met:
        iterations_for_nlte_elem = min(len(elements_to_fit), len(depart_bin_file))
    else:
        iterations_for_nlte_elem = min(len(elements_to_fit), len(depart_bin_file) - 1)
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
    if "Fe" not in elements_to_fit:  # if Fe is not fitted, then the last NLTE element should be
        depart_bin_file_dict["Fe"] = depart_bin_file[-1]
        depart_aux_file_dict["Fe"] = depart_aux_file[-1]
        model_atom_file_dict["Fe"] = model_atom_file[-1]
    return depart_bin_file_dict, depart_aux_file_dict, model_atom_file_dict


def run_TSFitPy():
    depart_bin_file = []
    depart_aux_file = []
    model_atom_file = []
    init_guess_elements = []

    initial_guess_string = None

    # read the configuration file
    with open(config_location) as fp:
        line = fp.readline()
        while line:
            fields = line.strip().split()
            # if fields[0][0] == "#":
            # line = fp.readline()
            if len(fields) == 0:
                line = fp.readline()
                fields = line.strip().split()
            # if fields[0] == "turbospec_path":
            #    turbospec_path = fields[2]
            if fields[0] == "interpol_path":
                interpol_path = fields[2]
            if fields[0] == "line_list_path":
                line_list_path = fields[2]
            # if fields[0] == "line_list_folder":
            #    linelist_folder = fields[2]
            if fields[0] == "model_atmosphere_grid_path_1D":
                model_atmosphere_grid_path_1D = fields[2]
            if fields[0] == "model_atmosphere_grid_path_3D":
                model_atmosphere_grid_path_3D = fields[2]
            # if fields[0] == "model_atmosphere_folder":
            #    model_atmosphere_folder = fields[2]
            # if fields[0] == "model_atmosphere_list":
            #    model_atmosphere_list = fields[2]
            if fields[0] == "model_atom_path":
                model_atom_path = fields[2]
            if fields[0] == "departure_file_path":
                departure_file_path = fields[2]
            if fields[0] == "output_folder":
                output_folder_og = fields[2]
            if fields[0] == "linemask_file_folder_location":
                linemask_file_og = fields[2]
            if fields[0] == "segment_file_folder_location":
                segment_file_og = fields[2]
            if fields[0] == "spec_input_path":
                spec_input_path = fields[2]
            if fields[0] == "fitlist_input_folder":
                fitlist_input_folder = fields[2]
            if fields[0] == "turbospectrum_compiler":
                ts_compiler = fields[2]
            if fields[0] == "atmosphere_type":
                Spectra.atmosphere_type = fields[2]
            if fields[0] == "mode":
                Spectra.fitting_mode = fields[2]
            if fields[0] == "include_molecules":
                Spectra.include_molecules = fields[2]
            if fields[0] == "nlte":
                nlte_flag = fields[2]
                if nlte_flag == "True":
                    Spectra.nlte_flag = True
                else:
                    Spectra.nlte_flag = False
            if fields[0] == "fit_microturb":  # Yes No Input
                Spectra.fit_microturb = fields[2]
            if fields[0] == "fit_macroturb":  # Yes No Input
                if fields[2] == "Yes":
                    Spectra.fit_macroturb = True
                else:
                    Spectra.fit_macroturb = False
                if fields[2] == "Input":
                    input_macro = True
                else:
                    input_macro = False
            if fields[0] == "fit_teff":
                Spectra.fit_teff = fields[2]
            if fields[0] == "fit_logg":
                Spectra.fit_logg = fields[2]
            if fields[0] == "element":
                elements_to_fit = []
                for i in range(len(fields) - 2):
                    elements_to_fit.append(fields[2 + i])
                Spectra.elem_to_fit = np.asarray(elements_to_fit)
                if "Fe" in elements_to_fit:
                    Spectra.fit_met = True
                else:
                    Spectra.fit_met = False
                Spectra.nelement = len(Spectra.elem_to_fit)
            if fields[0] == "linemask_file":
                linemask_file = fields[2]
            if fields[0] == "segment_file":
                segment_file = fields[2]
            # if fields[0] == "continuum_file":
            #    continuum_file = fields[2]
            if fields[0] == "departure_coefficient_binary" and Spectra.nlte_flag:
                for i in range(2, len(fields)):
                    depart_bin_file.append(fields[i])
            if fields[0] == "departure_coefficient_aux" and Spectra.nlte_flag:
                for i in range(2, len(fields)):
                    depart_aux_file.append(fields[i])
            if fields[0] == "model_atom_file" and Spectra.nlte_flag:
                for i in range(2, len(fields)):
                    model_atom_file.append(fields[i])
            if fields[0] == "wavelength_minimum":
                Spectra.lmin = float(fields[2])
            if fields[0] == "wavelength_maximum":
                Spectra.lmax = float(fields[2])
            if fields[0] == "wavelength_delta":
                Spectra.ldelta = float(fields[2])
            if fields[0] == "resolution":
                Spectra.fwhm = float(fields[2])
            if fields[0] == "macroturbulence":
                macroturb_input = float(fields[2])
            if fields[0] == "rotation":
                Spectra.rot = float(fields[2])
            if fields[0] == "temporary_directory":
                temp_directory = fields[2]
                temp_directory = os.path.join(temp_directory, today, '')
                Spectra.global_temp_dir = f"../{temp_directory}"
            if fields[0] == "input_file":
                fitlist = fields[2]
            if fields[0] == "output_file":
                output = fields[2]
            if fields[0] == "workers":
                workers = int(fields[2])  # should be the same as cores; use value of 1 if you do not want to use multithprocessing
            if fields[0] == "init_guess_elem":
                init_guess_elements = []
                for i in range(len(fields) - 2):
                    init_guess_elements.append(fields[2 + i])
                init_guess_elements = np.asarray(init_guess_elements)
            if fields[0] == "init_guess_elem_location":
                init_guess_elements_location = []
                for i in range(len(init_guess_elements)):
                    init_guess_elements_location.append(fields[2 + i])
                init_guess_elements_location = np.asarray(init_guess_elements_location)
            line = fp.readline()
        fp.close()

    if Spectra.nlte_flag:
        depart_bin_file_dict, depart_aux_file_dict, model_atom_file_dict = load_nlte_files_in_dict(elements_to_fit,
                                                                                                   depart_bin_file,
                                                                                                   depart_aux_file,
                                                                                                   model_atom_file)

        print("NLTE loaded. Please check that elements correspond to their correct binary files:")
        for key in depart_bin_file_dict:
            print(f"{key}: {depart_bin_file_dict[key]} {depart_aux_file_dict[key]} {model_atom_file_dict[key]}")

        print("If files do not correspond, please check config file. Fitted elements should go in the same order as "
              "the NLTE file locations. If Fe is not fitted, then it should be added last to the NLTE file location. "
              "Elements without NLTE binary files do not need them.")

        Spectra.depart_bin_file_dict = depart_bin_file_dict
        Spectra.depart_aux_file_dict = depart_aux_file_dict
        Spectra.model_atom_file_dict = model_atom_file_dict

    # set directories
    if ts_compiler == "intel":
        Spectra.turbospec_path = "../turbospectrum/exec/"
    elif ts_compiler == "gnu":
        Spectra.turbospec_path = "../turbospectrum/exec-gf/"
    Spectra.interpol_path = interpol_path
    line_list_path_orig = line_list_path
    line_list_path_trimmed = f"{line_list_path}../linelist_for_fitting_trimmed/"
    if Spectra.atmosphere_type == "1D":
        Spectra.model_atmosphere_grid_path = model_atmosphere_grid_path_1D
        Spectra.model_atmosphere_list = Spectra.model_atmosphere_grid_path + "model_atmosphere_list.txt"
    elif Spectra.atmosphere_type == "3D":
        Spectra.model_atmosphere_grid_path = model_atmosphere_grid_path_3D
        Spectra.model_atmosphere_list = Spectra.model_atmosphere_grid_path + "model_atmosphere_list.txt"
    Spectra.model_atom_path = model_atom_path
    Spectra.departure_file_path = departure_file_path
    Spectra.output_folder = f"{output_folder_og}{today}/"
    Spectra.spec_input_path = spec_input_path

    Spectra.linemask_file = f"{linemask_file_og}{linemask_file}"
    Spectra.segment_file = f"{segment_file_og}{segment_file}"

    create_dir(Spectra.global_temp_dir)
    create_dir(Spectra.output_folder)

    fitlist = f"{fitlist_input_folder}{fitlist}"

    Spectra.ndimen = 1  # first dimension is RV fit
    if Spectra.fit_microturb == "Yes" and (
            Spectra.fitting_mode == "lbl" or Spectra.fitting_mode == "lbl_quick") and not Spectra.atmosphere_type == "3D":
        Spectra.ndimen += 1  # if fitting micro for lbl, not 3D
    if Spectra.fitting_mode == "lbl":  # TODO: if several elements fitted for other modes, change here
        Spectra.ndimen += Spectra.nelement
        print(f"Fitting {Spectra.nelement} element(s): {Spectra.elem_to_fit}")
    elif Spectra.fitting_mode == "lbl_quick":
        pass    # element is not fitted using minimization, no need for ndimen
    else:
        Spectra.ndimen += 1
        print(f"Fitting {1} element: {Spectra.elem_to_fit[0]}")
    if Spectra.fit_macroturb:
        Spectra.ndimen += 1

    fitlist_data = np.loadtxt(fitlist, dtype='str')

    if fitlist_data.ndim == 1:
        fitlist_data = np.array([fitlist_data])

    specname_fitlist, rv_fitlist, teff_fitlist, logg_fitlist = fitlist_data[:, 0], fitlist_data[:, 1], \
                                                               fitlist_data[:, 2], fitlist_data[:, 3]

    fitlist_next_column = 4     # next loaded column #TODO not perfect solution? what if user gives metal but fits it too?

    input_abundances = np.zeros(fitlist_data.shape[0])  # if lbl_quick they will be used as center guess, otherwise means nothing
    if not Spectra.fitting_mode == "lbl_quick":
        if Spectra.fit_met:
            met_fitlist = np.zeros(fitlist_data.shape[0])  # fitting metallicity: just give it 0
        else:
            met_fitlist = fitlist_data[:, fitlist_next_column]  # metallicity [Fe/H], scaled to solar; not fitting metallicity: load it
            fitlist_next_column += 1
    else:
        met_fitlist = fitlist_data[:, fitlist_next_column]
        fitlist_next_column += 1
        if not Spectra.fit_met:
            input_abundances = fitlist_data[:, fitlist_next_column]  # guess for abundance for lbl quick, [X/Fe]
            fitlist_next_column += 1

    if Spectra.fit_microturb == "Input":
        microturb_input = fitlist_data[:, fitlist_next_column]
        fitlist_next_column += 1
    else:
        microturb_input = np.zeros(fitlist_data.shape[0])

    if input_macro:
        macroturb = fitlist_data[:, fitlist_next_column]  # input macroturbulence in km/s
        fitlist_next_column += 1
    else:
        macroturb = np.ones(fitlist_data.shape[0]) * macroturb_input

    if np.size(init_guess_elements) > 0:
        init_guess_spectra_dict = collections.defaultdict(dict)

        for init_guess_elem, init_guess_loc in zip(init_guess_elements, init_guess_elements_location):
            init_guess_data = np.loadtxt(init_guess_loc, dtype=str)
            if init_guess_data.ndim == 1:
                init_guess_data = np.array([init_guess_data])
            init_guess_spectra_names, init_guess_values = init_guess_data[:, 0], init_guess_data[:, 1].astype(float)

            for spectra in specname_fitlist:
                spectra_loc_index = np.where(init_guess_spectra_names == spectra)[0][0]
                init_guess_spectra_dict[spectra][init_guess_elem] = init_guess_values[spectra_loc_index]

        Spectra.init_guess_dict = dict(init_guess_spectra_dict)

    line_centers, line_begins, line_ends = np.loadtxt(Spectra.linemask_file, comments=";", usecols=(0, 1, 2),
                                                      unpack=True)

    if line_centers.size > 1:
        Spectra.line_begins_sorted = np.array(sorted(line_begins))
        Spectra.line_ends_sorted = np.array(sorted(line_ends))
        Spectra.line_centers_sorted = np.array(sorted(line_centers))
    elif line_centers.size == 1:
        Spectra.line_begins_sorted = np.array([line_begins])
        Spectra.line_ends_sorted = np.array([line_ends])
        Spectra.line_centers_sorted = np.array([line_centers])

    Spectra.seg_begins, Spectra.seg_ends = np.loadtxt(Spectra.segment_file, comments=";", usecols=(0, 1), unpack=True)
    if Spectra.seg_begins.size == 1:
        Spectra.seg_begins = np.array([Spectra.seg_begins])
        Spectra.seg_ends = np.array([Spectra.seg_ends])

    print("Trimming down the linelist to only lines within segments for faster fitting")
    if Spectra.fitting_mode == "all" or Spectra.fitting_mode == "lbl_quick":  # TODO add explanation of saving trimmed files
        # os.system("rm {}/*".format(line_list_path_trimmed))
        trimmed_start = 0
        trimmed_end = len(Spectra.seg_ends)
        line_list_path_trimmed = f"{line_list_path_trimmed}_{segment_file.replace('/', '_')}_{Spectra.include_molecules}_{trimmed_start}_{trimmed_end}/"
        create_window_linelist(Spectra.seg_begins, Spectra.seg_ends, line_list_path_orig, line_list_path_trimmed,
                               Spectra.include_molecules,
                               trimmed_start, trimmed_end)
    elif Spectra.fitting_mode == "lbl":
        line_list_path_trimmed = os.path.join(line_list_path_trimmed, "lbl", '')
        for j in range(len(Spectra.line_begins_sorted)):
            for k in range(len(Spectra.seg_begins)):
                if Spectra.seg_ends[k] >= Spectra.line_centers_sorted[j] > Spectra.seg_begins[k]:
                    start = k
            line_list_path_trimmed_new = get_trimmed_lbl_path_name(Spectra.elem_to_fit, line_list_path_trimmed,
                                                                   Spectra.segment_file, j, start)
            create_window_linelist(Spectra.seg_begins, Spectra.seg_ends, line_list_path_orig,
                                   line_list_path_trimmed_new,
                                   Spectra.include_molecules, start, start + 1)
    print("Finished trimming linelist")

    if workers > 1:
        print("Preparing workers")  # TODO check memory issues? set higher? give warnings?
        client = Client(threads_per_worker=1,
                        n_workers=workers)  # if # of threads are not equal to 1, then may break the program
        print(client)

        host = client.run_on_scheduler(socket.gethostname)
        port = client.scheduler_info()['services']['dashboard']
        print(f"Assuming that the cluster is ran at {login_node_address} (change in code if not the case)")

        # print(logger.info(f"ssh -N -L {port}:{host}:{port} {login_node_address}"))
        print(f"ssh -N -L {port}:{host}:{port} {login_node_address}")

        print("Worker preparation complete")

        futures = []
        for i in range(specname_fitlist.size):
            specname1, teff1, logg1, rv1, met1, microturb1 = specname_fitlist[i], teff_fitlist[i], logg_fitlist[i], \
                                                             rv_fitlist[i], met_fitlist[i], microturb_input[i]
            macroturb1 = macroturb[i]
            input_abundance = input_abundances[i]
            future = client.submit(create_and_fit_spectra, specname1, teff1, logg1, rv1, met1, microturb1, macroturb1,
                                   initial_guess_string, line_list_path_trimmed, input_abundance)
            futures.append(future)  # prepares to get values

        print("Start gathering")  # use http://localhost:8787/status to check status. the port might be different
        futures = np.array(client.gather(futures))  # starts the calculations (takes a long time here)
        results = futures
        print("Worker calculation done")  # when done, save values
    else:
        results = []
        for i in range(specname_fitlist.size):
            specname1, teff1, logg1, rv1, met1, microturb1 = specname_fitlist[i], teff_fitlist[i], logg_fitlist[i], \
                                                             rv_fitlist[i], met_fitlist[i], microturb_input[i]
            input_abundance = input_abundances[i]
            macroturb1 = macroturb[i]
            results.append(create_and_fit_spectra(specname1, teff1, logg1, rv1, met1, microturb1, macroturb1,
                                                  initial_guess_string, line_list_path_trimmed, input_abundance))

    shutil.rmtree(Spectra.global_temp_dir)  # clean up temp directory

    output = Spectra.output_folder + output

    f = open(output, 'a')

    # result = f"{self.spec_name} {res.x[0]} {res.x[1]} {res.fun} {self.macroturb}"
    # result.append(f"{self.spec_name} {Spectra.line_centers_sorted[j]} {Spectra.line_begins_sorted[j]} "
    #                      f"{Spectra.line_ends_sorted[j]} {res.x[0]} {res.x[1]} {microturb} {macroturb} {res.fun}")

    if Spectra.fitting_mode == "lbl":
        output_elem_column = f"Fe_H"

        for i in range(Spectra.nelement):
            # Spectra.elem_to_fit[i] = element name
            elem_name = Spectra.elem_to_fit[i]
            if elem_name != "Fe":
                output_elem_column += f"\t{elem_name}_Fe"
    else:
        if Spectra.fit_met:
            output_elem_column = "Fe_H"
        else:
            output_elem_column = f"{Spectra.elem_to_fit[0]}_Fe"

    if Spectra.fitting_mode == "all":
        print(f"#specname        {output_elem_column}     Doppler_Shift_add_to_RV    chi_squared Macro_turb", file=f)
    elif Spectra.fitting_mode == "lbl":
        print(
            f"#specname        wave_center  wave_start  wave_end  Doppler_Shift_add_to_RV   {output_elem_column}    Microturb   Macroturb    chi_squared",
            file=f
        )
    elif Spectra.fitting_mode == "lbl_quick":  # f" {res.x[0]} {vmicro} {macroturb} {res.fun}"
        output_columns = "#specname\twave_center\twave_start\twave_end"
        for i in range(Spectra.grids_amount):
            output_columns += f"\tabund_{i}\tdoppler_shift_{i}\tmicroturb_{i}\tmacroturb_{i}\tchi_square_{i}"
        # f"#specname        wave_center  wave_start  wave_end  {element[0]}_Fe   Doppler_Shift_add_to_RV Microturb   Macroturb    chi_squared"
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


if __name__ == '__main__':
    new_lbl_version = 2
    if len(argv) > 1:
        config_location = argv[1]
    else:
        config_location = "../input_files/tsfitpy_input_configuration.txt"  # location of config file
    print(config_location)
    # TODO explain lbl quick
    today = datetime.datetime.now().strftime("%b-%d-%Y-%H-%M-%S")  # used to not conflict with other instances of fits
    today = f"{today}_{np.random.random(1)[0]}"
    print(f"Start of the fitting: {today}")
    login_node_address = "gemini-login.mpia.de"  # Change this to the address/domain of your login node
    run_TSFitPy()
    print(f"End of the fitting: {datetime.datetime.now().strftime('%b-%d-%Y-%H-%M-%S')}")
