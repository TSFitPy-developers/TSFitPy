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
from distributed.scheduler import logger
import socket

from solar_abundances import solar_abundances, periodic_table

from convolve import *
from create_window_linelist_function import *


def create_dir(directory):
    if not os.path.exists(directory):
        try:
            os.mkdir(directory)
        except FileNotFoundError:
            os.makedirs(directory)


def calculate_vturb(teff, logg, met):
    """
    Calculates micro turbulence based on the input parameters
    :param teff: Temperature in kelvin
    :param logg: log(g) in dex units
    :param met: metallicity [Fe/H] scaled by solar
    :return: micro turbulence in km/s
    """
    t0 = 5500.
    g0 = 4.

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


def get_convolved_spectra(wave_mod_filled, flux_mod_filled, fwhm, macro, rot):
    """
    Convolves spectra with FWHM, macroturbulence or rotation if values are non-zero
    :param wave_mod_filled:
    :param flux_mod_filled:
    :param fwhm:
    :param macro:
    :param rot:
    :return: 2 arrays, first is convolved wavelength, second is convolved flux
    """
    if fwhm != 0.0:
        wave_mod_conv, flux_mod_conv = conv_res(wave_mod_filled, flux_mod_filled, fwhm)
    else:
        wave_mod_conv = wave_mod_filled
        flux_mod_conv = flux_mod_filled
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


def calculate_all_lines_chi_squared(wave_obs, flux_obs, wave_mod, flux_mod, line_begins_sorted, line_ends_sorted,
                                    seg_begins, seg_ends):
    """
    Calculates chi squared for all lines fitting by comparing two spectra and calculating the chi_squared
    :param wave_obs:
    :param flux_obs:
    :param wave_mod:
    :param flux_mod:
    :param line_begins_sorted:
    :param line_ends_sorted:
    :param seg_begins:
    :param seg_ends:
    :return:
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


def calc_ts_spectra_all_lines(obs_name, temp_directory, output_dir, wave_obs, flux_obs, macro, fwhm, rot, line_begins_sorted,
                              line_ends_sorted, seg_begins, seg_ends):
    """
    Calculates chi squared by opening a created synthetic spectrum and comparing to the observed spectra. Then calculates chi squared
    :param flux_obs:
    :param fwhm:
    :param line_begins_sorted:
    :param line_ends_sorted:
    :param macro:
    :param obs_name:
    :param rot:
    :param seg_begins:
    :param seg_ends:
    :param temp_directory:
    :param wave_obs:
    :return:
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


def calculate_lbl_chi_squared(temp_directory, wave_obs, flux_obs, wave_mod_orig, flux_mod_orig, fwhm, lmax, lmin, macro, rot, save_convolved=True):
    """
    Calculates chi squared by opening a created synthetic spectrum and comparing to the observed spectra. Then calculates chi squared
    :param flux_obs:
    :param fwhm:
    :param lmax:
    :param lmin:
    :param macro:
    :param rot:
    :param temp_directory:
    :param wave_obs:
    :return:
    """
    indices_to_use_mod = np.where((wave_mod_orig <= lmax) & (wave_mod_orig >= lmin))
    indices_to_use_obs = np.where((wave_obs <= lmax) & (wave_obs >= lmin))

    wave_mod_orig, flux_mod_orig = wave_mod_orig[indices_to_use_mod], flux_mod_orig[indices_to_use_mod]
    wave_obs, flux_obs = wave_obs[indices_to_use_obs], flux_obs[indices_to_use_obs]

    wave_mod, flux_mod = get_convolved_spectra(wave_mod_orig, flux_mod_orig, fwhm, macro, rot)
    if wave_mod[1] - wave_mod[0] <= wave_obs[1] - wave_obs[0]:
        flux_mod_interp = np.interp(wave_obs, wave_mod, flux_mod)
        wave_line = wave_obs[np.where((wave_obs <= lmax - 5.) & (wave_obs >= lmin + 5.))]   # 5 AA i guess to remove extra edges??
        flux_line_obs = flux_obs[np.where((wave_obs <= lmax - 5.) & (wave_obs >= lmin + 5.))]
        flux_line_mod = flux_mod_interp[np.where((wave_obs <= lmax - 5.) & (wave_obs >= lmin + 5.))]
        chi_square = np.sum(((flux_line_obs - flux_line_mod) * (flux_line_obs - flux_line_mod)) / flux_line_mod)
    else:
        flux_obs_interp = np.interp(wave_mod, wave_obs, flux_obs)
        wave_line = wave_mod[np.where((wave_mod <= lmax - 5.) & (wave_mod >= lmin + 5.))]
        flux_line_obs = flux_obs_interp[np.where((wave_mod <= lmax - 5.) & (wave_mod >= lmin + 5.))]
        flux_line_mod = flux_mod[np.where((wave_mod <= lmax - 5.) & (wave_mod >= lmin + 5.))]
        chi_square = np.sum(((flux_line_obs - flux_line_mod) * (flux_line_obs - flux_line_mod)) / flux_line_mod)
    #os.system(f"mv {temp_directory}spectrum_00000000.spec ../output_files/spectrum_fit_{obs_name.replace('../input_files/observed_spectra/', '')}")

    if save_convolved:
        out = open(f"{temp_directory}spectrum_00000000_convolved.spec", 'w')

        for i in range(len(wave_line)):
            print("{}  {}".format(wave_line[i], flux_line_mod[i]), file=out)
        out.close()
    return chi_square


class Spectra:
    turbospec_path = None
    interpol_path = None
    model_atmosphere_grid_path = None
    model_atmosphere_list = None
    model_atom_path = None
    departure_file_path = None
    linemask_file = None
    segment_file = None
    atmosphere_type = None  # "1D" or "3D", string
    include_molecules = None    # "True" or "False", string
    nlte_flag = None
    fit_microturb = "No"
    fit_macroturb = False
    fit_teff = None     # does not work atm
    fit_logg = None     # does not work atm
    fit_met = None
    elem_to_fit = None  # only 1 element at a time is support atm, a list otherwise
    lmin = None
    lmax = None
    ldelta = None
    fwhm = None
    macroturb = None
    rot = None
    fitting_mode = None  # "lbl" = line by line or "all"
    output_folder = None

    global_temp_dir = None
    line_begins_sorted = None
    line_ends_sorted = None
    line_centers_sorted = None

    seg_begins = None
    seg_ends = None

    depart_bin_file_dict = None
    depart_aux_file_dict = None
    model_atom_file_dict = None
    ndimen = None
    spec_input_path = None

    grids_amount = 50
    abund_bound = 0.5

    def __init__(self, specname, teff, logg, rv, met, micro, line_list_path_trimmed, init_param_guess, elem_abund=None):
        self.spec_name = str(specname)
        self.spec_path = os.path.join(self.spec_input_path, str(specname))
        self.teff = float(teff)
        self.logg = float(logg)
        self.met = float(met)
        if elem_abund is not None:
            self.elem_abund = elem_abund
        else:
            self.elem_abund = None
        if Spectra.fit_microturb == "Input":
            self.vmicro = float(micro)
        else:
            self.vmicro = None
        self.temp_dir = os.path.join(Spectra.global_temp_dir, self.spec_name, '')
        create_dir(self.temp_dir)  # create temp directory
        self.rv = float(rv)
        self.param_guess = None

        self.abund_to_gen = None    # array with generated abundances

        self.init_param_guess = None
        self.initial_simplex_guess = None
        self.set_param_guess(init_param_guess)

        self.line_list_path_trimmed = line_list_path_trimmed

        self.ts = TurboSpectrum(
            turbospec_path=self.turbospec_path,
            interpol_path=self.interpol_path,
            line_list_paths=self.line_list_path_trimmed,
            marcs_grid_path=self.model_atmosphere_grid_path,
            marcs_grid_list=self.model_atmosphere_list,
            model_atom_path=self.model_atom_path,
            departure_file_path=self.departure_file_path)

        self.wave_ob, self.flux_ob = np.loadtxt(self.spec_path, usecols=(0, 1), unpack=True, dtype=float)

    def set_param_guess(self, init_param_guess):
        # make an array for initial guess equal to n x ndimen+1
        initial_guess = np.empty((Spectra.ndimen + 1, Spectra.ndimen))

        # fill the array with input from config file
        for j in range(Spectra.ndimen):
            for i in range(j, len(init_param_guess), Spectra.ndimen):
                initial_guess[int(i / Spectra.ndimen)][j] = float(init_param_guess[i])

        self.init_param_guess = initial_guess[0]
        self.initial_simplex_guess = initial_guess

    def configure_and_run_ts(self, met, elem_abund, vmicro, lmin, lmax, windows_flag, temp_dir=None):
        """
        Configures TurboSpectrum depending on input parameters and runs either NLTE or LTE
        :param windows_flag - False for lbl, True for all lines
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
        time_start = time.time()

        res = minimize(all_broad_abund_chi_sqr, self.init_param_guess, args=(self), method='Nelder-Mead',
                       options={'maxiter': self.ndimen * 50, 'disp': True,
                                'initial_simplex': self.initial_simplex_guess, 'xatol': 0.05, 'fatol': 0.05})

        print(res.x)

        if self.fit_macroturb:
            result = f"{self.spec_name} {res.x[0]} {res.x[1]} {res.fun} {res.x[2]}"
        else:
            result = f"{self.spec_name} {res.x[0]} {res.x[1]} {res.fun} {self.macroturb}"

        time_end = time.time()
        print(f"Total runtime was {(time_end - time_start) / 60.:2f} minutes.")

        shutil.rmtree(self.temp_dir)
        return result

    def generate_grid_for_lbl(self):
        print("Generating grids")
        if self.fit_met:
            input_abund = self.met
        else:
            input_abund = self.elem_abund

        self.abund_to_gen = np.linspace(input_abund - self.abund_bound, input_abund + self.abund_bound, self.grids_amount)

        success = []

        for abund_to_use in self.abund_to_gen:
            if self.met > 0.5 or self.met < -4.0 or abund_to_use < -40 or (Spectra.fit_met and (abund_to_use < -4.0 or abund_to_use > 0.5)):
                success.append(False)
            else:
                if Spectra.fit_met:
                    item_abund = {"Fe": abund_to_use}
                    met = abund_to_use
                else:
                    item_abund = {"Fe": self.met, Spectra.elem_to_fit: abund_to_use + self.met}
                    met = self.met

                if self.vmicro is not None:
                    vmicro = self.vmicro
                else:
                    vmicro = calculate_vturb(self.teff, self.logg, met)

                temp_dir = os.path.join(self.temp_dir, f"{abund_to_use}", '')
                create_dir(temp_dir)
                self.configure_and_run_ts(met, item_abund, vmicro, self.lmin, self.lmax, True, temp_dir=temp_dir)

                success.append(True)
        print("Generation successful")
        return success


    def fit_lbl_quick(self):
        success_grid_gen = self.generate_grid_for_lbl()
        result = []
        grid_spectra = {}

        for i, (abund, success) in enumerate(zip(self.abund_to_gen, success_grid_gen)):
            if success:
                spectra_grid_path = os.path.join(self.temp_dir, f"{abund}", '')
                if os_path.exists(f"{spectra_grid_path}spectrum_00000000.spec") and os.stat(
                    f"{spectra_grid_path}spectrum_00000000.spec").st_size != 0:
                    wave_mod_orig, flux_mod_orig = np.loadtxt(f'{spectra_grid_path}/spectrum_00000000.spec',
                                                              usecols=(0, 1), unpack=True)
                    grid_spectra[abund] = [wave_mod_orig, flux_mod_orig]
                else:
                    success_grid_gen[i] = False

        for j in range(len(Spectra.line_begins_sorted)):
            time_start = time.time()
            print(f"Fitting line at {Spectra.line_centers_sorted[j]} angstroms")

            for k in range(len(Spectra.seg_begins)):
                if Spectra.seg_ends[k] >= Spectra.line_centers_sorted[j] > Spectra.seg_begins[k]:
                    start = k
            print(Spectra.line_centers_sorted[j], Spectra.seg_begins[start], Spectra.seg_ends[start])

            result_one_line = f"{self.spec_name} {Spectra.line_centers_sorted[j]} {Spectra.line_begins_sorted[j]} " \
                              f"{Spectra.line_ends_sorted[j]}"

            chi_squares = []

            for abund, success in zip(self.abund_to_gen, success_grid_gen):
                if success:
                    wave_abund, flux_abund = grid_spectra[abund][0], grid_spectra[abund][1]
                    res = minimize(lbl_broad_abund_chi_sqr_quick, self.init_param_guess, args=(self,
                                                                                         Spectra.line_begins_sorted[j] - 5.,
                                                                                         Spectra.line_ends_sorted[j] + 5.,
                                                                                         abund, wave_abund, flux_abund),
                                   method='Nelder-Mead',
                                   options={'maxiter': Spectra.ndimen * 50, 'disp': True, 'initial_simplex': self.initial_simplex_guess,
                                            'xatol': 0.05, 'fatol': 0.05})
                    print(res.x)
                    if Spectra.fit_macroturb:
                        macroturb = res.x[1]
                    else:
                        macroturb = Spectra.macroturb
                    if self.vmicro is not None:
                        vmicro = self.vmicro
                    else:
                        if self.fit_met:
                            met = abund
                        else:
                            met = self.met
                        vmicro = calculate_vturb(self.teff, self.logg, met)
                    result_one_line += f" {abund} {res.x[0]} {vmicro} {macroturb} {res.fun}"
                    chi_squares.append(res.fun)
                else:
                    print(f"Abundance {abund} did not manage to generate a grid")
                    result_one_line += f" {abund} {9999} {9999} {9999} {9999}"
                    chi_squares.append(9999)

            result.append(result_one_line)

            index_min_chi_square = np.argmin(chi_squares)
            min_chi_sqr_spectra_path = os.path.join(self.temp_dir, f"{self.abund_to_gen[index_min_chi_square]}", 'spectrum_00000000.spec')

            wave_result, flux_norm_result, flux_result = np.loadtxt(min_chi_sqr_spectra_path, unpack=True)
            g = open(f"{self.output_folder}result_spectrum_{self.spec_name}.spec", 'a')
            for k in range(len(wave_result)):
                print("{}  {}  {}".format(wave_result[k], flux_norm_result[k], flux_result[k]), file=g)

            time_end = time.time()
            print("Total runtime was {:.2f} minutes.".format((time_end - time_start) / 60.))

        g.close()

        return result


    def fit_lbl(self):
        result = []

        for j in range(len(Spectra.line_begins_sorted)):
            time_start = time.time()
            print(f"Fitting line at {Spectra.line_centers_sorted[j]} angstroms")

            for k in range(len(Spectra.seg_begins)):
                if Spectra.seg_ends[k] >= Spectra.line_centers_sorted[j] > Spectra.seg_begins[k]:
                    start = k
            print(Spectra.line_centers_sorted[j], Spectra.seg_begins[start], Spectra.seg_ends[start])

            self.ts.line_list_paths = [get_trimmed_lbl_path_name(self.elem_to_fit, self.line_list_path_trimmed, Spectra.segment_file, j, start)]

            res = minimize(lbl_broad_abund_chi_sqr, self.init_param_guess, args=(self,
                                                                                 Spectra.line_begins_sorted[j] - 5.,
                                                                                 Spectra.line_ends_sorted[j] + 5.),
                           method='Nelder-Mead',
                           options={'maxiter': Spectra.ndimen * 50, 'disp': True, 'initial_simplex': self.initial_simplex_guess,
                                    'xatol': 0.05, 'fatol': 0.05})

            print(res.x)

            if self.vmicro is not None:
                microturb = self.vmicro
            else:
                if Spectra.fit_microturb == "No" and Spectra.atmosphere_type == "1D":
                    if Spectra.fit_met:
                        microturb = calculate_vturb(self.teff, self.logg, res.x[0])
                    else:
                        microturb = calculate_vturb(self.teff, self.logg, self.met)
                elif Spectra.fit_microturb == "Yes" and Spectra.atmosphere_type == "1D":
                    microturb = res.x[2]
                else:
                    microturb = 2.0
            if Spectra.fit_macroturb:
                macroturb = res.x[-1]
            else:
                macroturb = Spectra.macroturb

            result.append(f"{self.spec_name} {Spectra.line_centers_sorted[j]} {Spectra.line_begins_sorted[j]} "
                          f"{Spectra.line_ends_sorted[j]} {res.x[0]} {res.x[1]} {microturb} {macroturb} {res.fun}") # out = open(f"{temp_directory}spectrum_00000000_convolved.spec", 'w')

            wave_result, flux_norm_result, flux_result = np.loadtxt(f"{self.temp_dir}spectrum_00000000.spec", unpack=True)
            g = open(f"{self.output_folder}result_spectrum_{self.spec_name}.spec", 'a')
            for k in range(len(wave_result)):
                print("{}  {}  {}".format(wave_result[k], flux_norm_result[k], flux_result[k]), file=g)

            wave_result, flux_norm_result = np.loadtxt(f"{self.temp_dir}spectrum_00000000_convolved.spec", unpack=True)
            h = open(f"{self.output_folder}result_spectrum_{self.spec_name}_convolved.spec", 'a')
            for k in range(len(wave_result)):
                print("{}  {}".format(wave_result[k], flux_norm_result[k]), file=h)
            #os.system("rm ../output_files/spectrum_{:08d}_convolved.spec".format(i + 1))

            time_end = time.time()
            print("Total runtime was {:.2f} minutes.".format((time_end - time_start) / 60.))

        g.close()
        h.close()

        return result


def lbl_broad_abund_chi_sqr_quick(param, spectra_to_fit: Spectra, lmin, lmax, abund, wave_mod_orig, flux_mod_orig):
    # param[0] = doppler
    # param[1] = macro turb

    doppler = spectra_to_fit.rv + param[0]

    if Spectra.fit_macroturb:
        macroturb = param[1]
    else:
        macroturb = Spectra.macroturb

    wave_ob = spectra_to_fit.wave_ob / (1 + (doppler / 300000.))

    if spectra_to_fit.met < -4.0 or spectra_to_fit.met > 0.5 or macroturb < 0.0:
        chi_square = 9999.9999
    else:
        chi_square = calculate_lbl_chi_squared(None, wave_ob,
                                                   spectra_to_fit.flux_ob, wave_mod_orig, flux_mod_orig, Spectra.fwhm, lmax, lmin, macroturb,
                                                   Spectra.rot, save_convolved=False)

    print(abund, doppler, chi_square, macroturb)

    return chi_square


def lbl_broad_abund_chi_sqr(param, spectra_to_fit: Spectra, lmin, lmax):
    # param[0] = met or abund
    # param[1] = added doppler to rv
    # param[2] = micro turb
    # param[-1] = macro turb

    abund = param[0]
    doppler = spectra_to_fit.rv + param[1]
    if spectra_to_fit.vmicro is not None:
        microturb = spectra_to_fit.vmicro
    else:
        if Spectra.fit_microturb == "No" and Spectra.atmosphere_type == "1D":
            if Spectra.fit_met:
                microturb = calculate_vturb(spectra_to_fit.teff, spectra_to_fit.logg, abund)
            else:
                microturb = calculate_vturb(spectra_to_fit.teff, spectra_to_fit.logg, spectra_to_fit.met)
        elif Spectra.fit_microturb == "Yes" and Spectra.atmosphere_type == "1D":
            microturb = param[1]
        else:
            microturb = 2.0
    if Spectra.fit_macroturb:
        macroturb = param[-1]
    else:
        macroturb = Spectra.macroturb

    wave_ob = spectra_to_fit.wave_ob / (1 + (doppler / 300000.))

    if spectra_to_fit.met > 0.5 or spectra_to_fit.met < -4.0 or microturb <= 0.0 or macroturb < 0.0 or \
            abund < -40 or (Spectra.fit_met and (abund < -4.0 or abund > 0.5)):
        chi_square = 9999.9999
    else:
        if Spectra.fit_met:
            item_abund = {"Fe": abund}
            met = abund
        else:
            item_abund = {"Fe": spectra_to_fit.met, Spectra.elem_to_fit: abund + spectra_to_fit.met}
            met = spectra_to_fit.met

        spectra_to_fit.configure_and_run_ts(met, item_abund, microturb, lmin, lmax, False)

        if os_path.exists('{}/spectrum_00000000.spec'.format(spectra_to_fit.temp_dir)) and os.stat(
                '{}/spectrum_00000000.spec'.format(spectra_to_fit.temp_dir)).st_size != 0:
            wave_mod_orig, flux_mod_orig = np.loadtxt(f'{spectra_to_fit.temp_dir}/spectrum_00000000.spec',
                                                      usecols=(0, 1), unpack=True)
            chi_square = calculate_lbl_chi_squared(spectra_to_fit.temp_dir, wave_ob,
                                                   spectra_to_fit.flux_ob, wave_mod_orig, flux_mod_orig, Spectra.fwhm, lmax, lmin, macroturb,
                                                   Spectra.rot)
        elif os_path.exists('{}/spectrum_00000000.spec'.format(spectra_to_fit.temp_dir)) and os.stat(
                '{}/spectrum_00000000.spec'.format(spectra_to_fit.temp_dir)).st_size == 0:
            chi_square = 999.99
            print("empty spectrum file.")
        else:
            chi_square = 9999.9999
            print("didn't generate spectra or atmosphere")

    print(abund, microturb, doppler, chi_square, macroturb)

    return chi_square


def get_trimmed_lbl_path_name(element, line_list_path_trimmed, segment_file, j, start):
    return os.path.join(line_list_path_trimmed,
                        f"{segment_file.replace('/', '_').replace('.', '_')}_{element[0]}_{Spectra.include_molecules}_{j}_{j + 1}_{str(Spectra.line_centers_sorted[j]).replace('.', '_')}_{str(Spectra.seg_begins[start]).replace('.', '_')}_{str(Spectra.seg_ends[start]).replace('.', '_')}", '')


def all_broad_abund_chi_sqr(param, spectra_to_fit: Spectra):
    abund = param[0]
    doppler = spectra_to_fit.rv + param[1]
    if Spectra.fit_macroturb:
        macroturb = param[2]
    else:
        macroturb = Spectra.macroturb

    wave_obs = spectra_to_fit.wave_ob / (1 + (doppler / 300000.))

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
        else:
            item_abund = {"Fe": spectra_to_fit.met, Spectra.elem_to_fit: abund + spectra_to_fit.met}
            met = spectra_to_fit.met
            if spectra_to_fit.vmicro is not None:
                vmicro = spectra_to_fit.vmicro
            else:
                vmicro = calculate_vturb(spectra_to_fit.teff, spectra_to_fit.logg, spectra_to_fit.met)

        spectra_to_fit.configure_and_run_ts(met, item_abund, vmicro, spectra_to_fit.lmin, spectra_to_fit.lmax, True)

        chi_square = calc_ts_spectra_all_lines(spectra_to_fit.spec_path, spectra_to_fit.temp_dir, spectra_to_fit.output_folder,
                                               wave_obs, spectra_to_fit.flux_ob,
                                               macroturb, Spectra.fwhm, Spectra.rot,
                                               Spectra.line_begins_sorted, Spectra.line_ends_sorted,
                                               Spectra.seg_begins, Spectra.seg_ends)

    print(abund, doppler, chi_square, macroturb)

    return chi_square


def create_and_fit_spectra(specname, teff, logg, rv, met, microturb, initial_guess_string, line_list_path_trimmed, input_abundance):
    spectra = Spectra(specname, teff, logg, rv, met, microturb, line_list_path_trimmed, initial_guess_string, elem_abund=input_abundance)

    print(f"Fitting {spectra.spec_name}")
    print(f"Teff = {spectra.teff}; logg = {spectra.logg}; RV = {spectra.rv}")

    if Spectra.fitting_mode == "all":
        result = spectra.fit_all()
    elif Spectra.fitting_mode == "lbl":
        result = spectra.fit_lbl()
    elif Spectra.fitting_mode == "lbl_quick":
        result = spectra.fit_lbl_quick()
    else:
        print(f"unknown fitting mode {Spectra.fitting_mode}, need all or lbl")
        return

    return result


def run_TSFitPy():
    depart_bin_file = []
    depart_aux_file = []
    model_atom_file = []

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
            #if fields[0] == "turbospec_path":
            #    turbospec_path = fields[2]
            if fields[0] == "interpol_path":
                interpol_path = fields[2]
            if fields[0] == "line_list_path":
                line_list_path = fields[2]
            #if fields[0] == "line_list_folder":
            #    linelist_folder = fields[2]
            if fields[0] == "model_atmosphere_grid_path_1D":
                model_atmosphere_grid_path_1D = fields[2]
            if fields[0] == "model_atmosphere_grid_path_3D":
                model_atmosphere_grid_path_3D = fields[2]
            #if fields[0] == "model_atmosphere_folder":
            #    model_atmosphere_folder = fields[2]
            #if fields[0] == "model_atmosphere_list":
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
            if fields[0] == "fit_microturb": # Yes No Input
                Spectra.fit_microturb = fields[2]
            if fields[0] == "fit_macroturb": # Yes No Input
                if fields[2] == "Yes":
                    Spectra.fit_macroturb = True
                else:
                    Spectra.fit_macroturb = False
            if fields[0] == "fit_teff":
                Spectra.fit_teff = fields[2]
            if fields[0] == "fit_logg":
                Spectra.fit_logg = fields[2]
            if fields[0] == "element_number":
                nelement = int(fields[2])
            if fields[0] == "element":
                element = fields[2]
                if element != "Fe":
                    element = []
                    for i in range(nelement):
                        element.append(fields[2 * (i + 1)])
                    element.append("Fe")
                else:
                    element = []
                    element.append(fields[2])
                Spectra.elem_to_fit = element
                if element[0] == "Fe" or element[0] == "fe":
                    Spectra.fit_met = True
            if fields[0] == "linemask_file":
                linemask_file = fields[2]
            if fields[0] == "segment_file":
                segment_file = fields[2]
            # if fields[0] == "continuum_file":
            #    continuum_file = fields[2]
            if fields[0] == "departure_coefficient_binary" and element[0] != "Fe" and nlte_flag == "True":
                for i in range(nelement + 1):
                    depart_bin_file.append(fields[2 + i])
            elif fields[0] == "departure_coefficient_binary" and element[0] == "Fe" and nlte_flag == "True":
                for i in range(nelement):
                    depart_bin_file.append(fields[2 + i])
            if fields[0] == "departure_coefficient_aux" and element[0] != "Fe" and nlte_flag == "True":
                for i in range(nelement + 1):
                    depart_aux_file.append(fields[2 + i])
            elif fields[0] == "departure_coefficient_aux" and element[0] == "Fe" and nlte_flag == "True":
                for i in range(nelement):
                    depart_aux_file.append(fields[2 + i])
            if fields[0] == "model_atom_file" and element[0] != "Fe" and nlte_flag == "True":
                for i in range(nelement + 1):
                    model_atom_file.append(fields[2 + i])
            elif fields[0] == "model_atom_file" and element[0] == "Fe" and nlte_flag == "True":
                for i in range(nelement):
                    model_atom_file.append(fields[2 + i])
            if fields[0] == "wavelength_minimum":
                Spectra.lmin = float(fields[2])
            if fields[0] == "wavelength_maximum":
                Spectra.lmax = float(fields[2])
            if fields[0] == "wavelength_delta":
                Spectra.ldelta = float(fields[2])
            if fields[0] == "resolution":
                Spectra.fwhm = float(fields[2])
            if fields[0] == "macroturbulence":
                Spectra.macroturb = float(fields[2])
            if fields[0] == "rotation":
                Spectra.rot = float(fields[2])
            if fields[0] == "temporary_directory":
                temp_directory = fields[2]
                temp_directory = os.path.join(temp_directory, today, '')
                Spectra.global_temp_dir = f"../{temp_directory}"
            if fields[0] == "initial_guess_array":
                initial_guess_string = fields[2].strip().split(",")
            if fields[0] == "ndimen":
                Spectra.ndimen = int(fields[2])
            if fields[0] == "input_file":
                fitlist = fields[2]
            if fields[0] == "output_file":
                output = fields[2]
            if fields[0] == "workers":
                workers = int(fields[2])  # should be the same as cores; use value of 1 if you do not want to use multithprocessing
            line = fp.readline()
        fp.close()

    if nlte_flag:
        depart_bin_file_dict = {}
        for i in range(len(depart_bin_file)):
            depart_bin_file_dict[element[i]] = depart_bin_file[i]
        depart_aux_file_dict = {}
        for i in range(len(depart_aux_file)):
            depart_aux_file_dict[element[i]] = depart_aux_file[i]
        model_atom_file_dict = {}
        for i in range(len(model_atom_file)):
            model_atom_file_dict[element[i]] = model_atom_file[i]

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

    if Spectra.fit_met:
        specname_fitlist, rv_fitlist, teff_fitlist, logg_fitlist, met_fitlist = np.loadtxt(fitlist, dtype='str',
                                                                                           usecols=(0, 1, 2, 3, 4),
                                                                                           unpack=True)
    else:
        specname_fitlist, rv_fitlist, teff_fitlist, logg_fitlist = np.loadtxt(fitlist, dtype='str',
                                                                                           usecols=(0, 1, 2, 3),
                                                                                           unpack=True)
        met_fitlist = np.zeros(np.size(specname_fitlist))

    if np.size(specname_fitlist) == 1:
        specname_fitlist, rv_fitlist, teff_fitlist, logg_fitlist, met_fitlist = np.array([specname_fitlist]), np.array([rv_fitlist]), np.array([teff_fitlist]), np.array([logg_fitlist]), np.array([met_fitlist])

    if Spectra.fit_microturb == "Input":
        microturb_input = np.loadtxt(fitlist, dtype='str', usecols=5, unpack=True)
    else:
        microturb_input = np.zeros(np.size(specname_fitlist))

    if np.size(specname_fitlist) == 1:
        microturb_input = np.array([microturb_input])

    if Spectra.fitting_mode == "lbl_quick" and not Spectra.fit_met:
        if Spectra.fit_microturb == "Input":
            use_col = 6
        else:
            use_col = 5
        input_abundances = np.loadtxt(fitlist, dtype='str', usecols=use_col, unpack=True)
        if np.size(specname_fitlist) == 1:
            input_abundances = np.array([input_abundances])
    else:
        input_abundances = np.zeros(np.size(specname_fitlist))

    line_centers, line_begins, line_ends = np.loadtxt(Spectra.linemask_file, comments=";", usecols=(0, 1, 2), unpack=True)


    if line_centers.size > 1:
        Spectra.line_begins_sorted = np.array(sorted(line_begins))
        Spectra.line_ends_sorted = np.array(sorted(line_ends))
        Spectra.line_centers_sorted = np.array(sorted(line_centers))
    elif line_centers.size == 1:
        Spectra.line_begins_sorted = np.array([line_begins])
        Spectra.line_ends_sorted = np.array([line_ends])
        Spectra.line_centers_sorted = np.array([line_centers])

    Spectra.seg_begins, Spectra.seg_ends = np.loadtxt(Spectra.segment_file, comments=";", usecols=(0, 1), unpack=True)

    if Spectra.fitting_mode == "all" or Spectra.fitting_mode == "lbl_quick":
        print("Trimming down the linelist to only lines within segments for faster fitting")
        # os.system("rm {}/*".format(line_list_path_trimmed))
        trimmed_start = 0
        trimmed_end = len(Spectra.seg_ends)
        line_list_path_trimmed = f"{line_list_path_trimmed}_{segment_file.replace('/', '_')}_{Spectra.include_molecules}_{trimmed_start}_{trimmed_end}/"
        create_window_linelist(Spectra.segment_file, line_list_path_orig, line_list_path_trimmed, Spectra.include_molecules,
                               trimmed_start, trimmed_end)
        print("Finished trimming linelist")
    elif Spectra.fitting_mode == "lbl":
        line_list_path_trimmed = os.path.join(line_list_path_trimmed, "lbl", '')
        for j in range(len(Spectra.line_begins_sorted)):
            for k in range(len(Spectra.seg_begins)):
                if Spectra.line_centers_sorted[j] <= Spectra.seg_ends[k] and Spectra.line_centers_sorted[j] > Spectra.seg_begins[k]:
                    start = k
            line_list_path_trimmed_new = get_trimmed_lbl_path_name(element, line_list_path_trimmed, Spectra.segment_file, j, start)
            create_window_linelist(Spectra.segment_file, line_list_path_orig, line_list_path_trimmed_new,
                                   Spectra.include_molecules, start, start + 1)

    if workers > 1:
        print("Preparing workers")
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
            input_abundance = input_abundances[i]
            future = client.submit(create_and_fit_spectra, specname1, teff1, logg1, rv1, met1, microturb1,
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
            results.append(create_and_fit_spectra(specname1, teff1, logg1, rv1, met1, microturb1, initial_guess_string,
                                                  line_list_path_trimmed, input_abundance))

    shutil.rmtree(Spectra.global_temp_dir)  # clean up temp directory

    output = Spectra.output_folder + output

    f = open(output, 'a')

    #result = f"{self.spec_name} {res.x[0]} {res.x[1]} {res.fun} {self.macroturb}"
    #result.append(f"{self.spec_name} {Spectra.line_centers_sorted[j]} {Spectra.line_begins_sorted[j]} "
    #                      f"{Spectra.line_ends_sorted[j]} {res.x[0]} {res.x[1]} {microturb} {macroturb} {res.fun}")

    if Spectra.fit_met:
        output_elem_column = "Fe_H"
    else:
        output_elem_column = f"{Spectra.elem_to_fit[0]}_Fe"

    if Spectra.fitting_mode == "all":
        print(f"#specname        {output_elem_column}     Doppler_Shift_add_to_RV    chi_squared Macro_turb", file=f)
    elif Spectra.fitting_mode == "lbl" and (element[0] == "Fe" or element[0] == "fe"):
        print(
            f"#specname        wave_center  wave_start  wave_end  {output_elem_column}   Doppler_Shift_add_to_RV Microturb   Macroturb    chi_squared",
            file=f
        )
    elif Spectra.fitting_mode == "lbl_quick": #f" {res.x[0]} {vmicro} {macroturb} {res.fun}"
        output_columns = "#specname\twave_center\twave_start\twave_end"
        for i in range(Spectra.grids_amount):
            output_columns += f"\tabund_{i}\tdoppler_shift_{i}\tmicroturb_{i}\tmacroturb_{i}\tchi_square_{i}"
        #f"#specname        wave_center  wave_start  wave_end  {element[0]}_Fe   Doppler_Shift_add_to_RV Microturb   Macroturb    chi_squared"
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
    config_location = "../input_files/tsfitpy_input_configuration.txt"  # location of config file

    today = datetime.datetime.now().strftime("%b-%d-%Y-%H-%M-%S")  # used to not conflict with other instances of fits
    login_node_address = "gemini-login.mpia.de"  # Change this to the address/domain of your login node
    Spectra.grids_amount = 50    # for lbl quick
    Spectra.abund_bound = 0.5   # for lbl quick
    run_TSFitPy()
