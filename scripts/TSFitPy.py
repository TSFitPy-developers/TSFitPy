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

    return v_mturb


def configure_and_run_ts(ts, abund_name, atmosphere_type, depart_aux_file_list, depart_bin_file_list, item_abund,
                         ldelta, lmax, lmin, logg, mask_file, met, model_atom_file_list, nlte_flag, segment_file, teff,
                         temp_directory, vturb):
    """
    Configures TurboSpectrum depending on input parameters and runs either NLTE or LTE
    :param ts:
    :param abund_name:
    :param atmosphere_type:
    :param depart_aux_file_list:
    :param depart_bin_file_list:
    :param item_abund:
    :param ldelta:
    :param lmax:
    :param lmin:
    :param logg:
    :param mask_file:
    :param met:
    :param model_atom_file_list:
    :param nlte_flag: True/False depending if one needs LTE or NLTE
    :param segment_file:
    :param teff:
    :param temp_directory:
    :param vturb:
    :return:
    """
    if nlte_flag == "False":
        ts.configure(t_eff=teff, log_g=logg, metallicity=met,
                     turbulent_velocity=vturb, lambda_delta=ldelta, lambda_min=lmin, lambda_max=lmax,
                     free_abundances=item_abund, temp_directory=temp_directory, nlte_flag=False, verbose=False,
                     atmosphere_dimension=atmosphere_type, windows_flag=True, segment_file=segment_file,
                     line_mask_file=mask_file)  # , depart_bin_file=depart_bin_file,
        # depart_aux_file=depart_aux_file, model_atom_file=model_atom_file)
    elif nlte_flag == "True":
        depart_bin_file = {}
        for i in range(len(depart_bin_file_list)):
            depart_bin_file[abund_name[i]] = depart_bin_file_list[i]
        depart_aux_file = {}
        for i in range(len(depart_aux_file_list)):
            depart_aux_file[abund_name[i]] = depart_aux_file_list[i]
        model_atom_file = {}
        for i in range(len(model_atom_file_list)):
            model_atom_file[abund_name[i]] = model_atom_file_list[i]
        ts.configure(t_eff=teff, log_g=logg, metallicity=met,
                     turbulent_velocity=vturb, lambda_delta=ldelta, lambda_min=lmin, lambda_max=lmax,
                     free_abundances=item_abund, temp_directory=temp_directory, nlte_flag=True, verbose=False,
                     atmosphere_dimension=atmosphere_type, windows_flag=True, segment_file=segment_file,
                     line_mask_file=mask_file, depart_bin_file=depart_bin_file,
                     depart_aux_file=depart_aux_file, model_atom_file=model_atom_file)
    ts.run_turbospectrum_and_atmosphere()


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


def calc_ts_spectra_all_lines(flux_obs, fwhm, line_begins_sorted, line_ends_sorted, macro, obs_name, rot, seg_begins,
                              seg_ends, temp_directory, wave_obs):
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
    if os_path.exists(f'{temp_directory}/spectrum_00000000.spec') and os.stat(f'{temp_directory}/spectrum_00000000.spec').st_size != 0:
        wave_mod_orig, flux_mod_orig = np.loadtxt(f'{temp_directory}/spectrum_00000000.spec', usecols=(0, 1), unpack=True)
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

        os.system(f"mv {temp_directory}spectrum_00000000.spec ../output_files/spectrum_fit_{obs_name.replace('../input_files/observed_spectra/', '')}")
        out = open(f"../output_files/spectrum_fit_convolved_{obs_name.replace('../input_files/observed_spectra/', '')}", 'w')
        for l in range(len(wave_mod)):
            print(f"{wave_mod[l]}  {flux_mod[l]}", file=out)
        out.close()
    elif os_path.exists(f'{temp_directory}/spectrum_00000000.spec') and os.stat(f'{temp_directory}/spectrum_00000000.spec').st_size == 0:
        chi_square = 999.99
        print("empty spectrum file.")
    else:
        chi_square = 9999.9999
        print("didn't generate spectra")
    return chi_square


def chi_square_broad(param, ts, obs_name, temp_directory, spectrum_count, mask_file, segment_file, depart_bin_file_list,
                     depart_aux_file_list, model_atom_file_list, atmosphere_type, nlte_flag, doppler, teff, logg, met,
                     vturb, macro, fwhm, rot, abund_name, ldelta, lmin, lmax, wave_obs, flux_obs,
                     line_begins_sorted, line_ends_sorted, seg_begins, seg_ends):
    # teff = param[0]
    # logg = param[1]
    # met = param[2]

    abund = param[0]
    doppler = doppler + param[1]
    if len(param) > 2:
        macro = param[2]

    wave_obs = wave_obs / (1 + (doppler / 300000.))

    if met > 0.5 or met < -4.0 or vturb <= 0.0 or macro < 0.0 or abund < -40:
        chi_square = 9999.9999
    else:
        item_abund = {"Fe": met, abund_name[0]: abund + met}

        configure_and_run_ts(ts, abund_name, atmosphere_type, depart_aux_file_list, depart_bin_file_list, item_abund,
                             ldelta, lmax, lmin, logg, mask_file, met, model_atom_file_list, nlte_flag, segment_file,
                             teff, temp_directory, vturb)

        chi_square = calc_ts_spectra_all_lines(flux_obs, fwhm, line_begins_sorted, line_ends_sorted, macro, obs_name, rot,
                                               seg_begins, seg_ends, temp_directory, wave_obs)

    print(abund, doppler, chi_square, macro)

    return chi_square


def chi_square_broad_met(param, ts, obs_name, temp_directory, spectrum_count, mask_file, segment_file, depart_bin_file_list,
                         depart_aux_file_list, model_atom_file_list, atmosphere_type, nlte_flag, doppler, teff, logg,
                         macro, fwhm, rot, ldelta, lmin, lmax, wave_obs, flux_obs, line_begins_sorted, line_ends_sorted,
                         seg_begins, seg_ends):
    met = param[0]
    doppler = doppler + param[1]
    if len(param) > 2:
        macro = param[2]

    wave_obs = wave_obs / (1 + (doppler / 300000.))

    if atmosphere_type == "1D":
        vturb = calculate_vturb(teff, logg, met)
    elif atmosphere_type == "3D":
        vturb = 2.0

    if met > 0.5 or met < -4.0 or vturb <= 0.0 or macro < 0.0:
        chi_square = 9999.9999
    else:
        item_abund = {"Fe": met}

        configure_and_run_ts(ts, ["Fe"], atmosphere_type, depart_aux_file_list, depart_bin_file_list, item_abund,
                             ldelta, lmax, lmin, logg, mask_file, met, model_atom_file_list, nlte_flag, segment_file,
                             teff, temp_directory, vturb)

        chi_square = calc_ts_spectra_all_lines(flux_obs, fwhm, line_begins_sorted, line_ends_sorted, macro, obs_name, rot,
                                               seg_begins, seg_ends, temp_directory, wave_obs)

    print(met, doppler, chi_square, vturb, macro)

    return chi_square


def prepar_lbl_calc(atmosphere_type, doppler, fit_microturb, logg, macro, met, param, teff, wave_obs):
    """
    Prepares the calculation for "line by line" by extracting and calculating initial parameters
    :param atmosphere_type:
    :param doppler:
    :param fit_microturb:
    :param logg:
    :param macro:
    :param met:
    :param param:
    :param teff:
    :param wave_obs:
    :return:
    """
    abund = param[0]
    doppler = doppler + param[-1]
    #if len(param) > 3:  # macro    #TODO add macro calculation
    #    macro = param[3]
    wave_obs = wave_obs / (1 + (doppler / 300000.))
    if atmosphere_type == "1D" and fit_microturb == "No":
        vturb = calculate_vturb(teff, logg, met)
    elif atmosphere_type == "1D" and fit_microturb == "Yes":
        vturb = param[1]
    elif atmosphere_type == "3D":
        vturb = 2.0
    return abund, doppler, macro, vturb, wave_obs


def calculate_lbl_chi_squared(flux_obs, fwhm, lmax, lmin, macro, rot, spectrum_count, temp_directory, wave_obs):
    """
    Calculates chi squared by opening a created synthetic spectrum and comparing to the observed spectra. Then calculates chi squared
    :param flux_obs:
    :param fwhm:
    :param lmax:
    :param lmin:
    :param macro:
    :param rot:
    :param spectrum_count:
    :param temp_directory:
    :param wave_obs:
    :return:
    """
    wave_mod_orig, flux_mod_orig = np.loadtxt('{}/spectrum_00000000.spec'.format(temp_directory),
                                              usecols=(0, 1), unpack=True)
    wave_mod, flux_mod = get_convolved_spectra(wave_mod_orig, flux_mod_orig, fwhm, macro, rot)
    if wave_mod[1] - wave_mod[0] <= wave_obs[1] - wave_obs[0]: #TODO redo chi squared here
        flux_mod_interp = np.interp(wave_obs, wave_mod, flux_mod)
        chi_square = 0
        wave_line = wave_obs[np.where((wave_obs <= lmax - 5.) & (wave_obs >= lmin + 5.))]
        flux_line_obs = flux_obs[np.where((wave_obs <= lmax - 5.) & (wave_obs >= lmin + 5.))]
        flux_line_mod = flux_mod_interp[np.where((wave_obs <= lmax - 5.) & (wave_obs >= lmin + 5.))]
        for j in range(len(wave_line)):
            chi_square += ((flux_line_obs[j] - flux_line_mod[j]) * (flux_line_obs[j] - flux_line_mod[j])) / \
                          flux_line_mod[j]
    else:
        flux_obs_interp = np.interp(wave_mod, wave_obs, flux_obs)
        chi_square = 0
        wave_line = wave_mod[np.where((wave_mod <= lmax - 5.) & (wave_mod >= lmin + 5.))]
        flux_line_obs = flux_obs_interp[np.where((wave_mod <= lmax - 5.) & (wave_mod >= lmin + 5.))]
        flux_line_mod = flux_mod[np.where((wave_mod <= lmax - 5.) & (wave_mod >= lmin + 5.))]
        for j in range(len(wave_line)):
            chi_square += ((flux_line_obs[j] - flux_line_mod[j]) * (flux_line_obs[j] - flux_line_mod[j])) / \
                          flux_line_mod[j]
    os.system("mv {}spectrum_00000000.spec ../output_files/spectrum_{:08d}.spec".format(temp_directory,
                                                                                        spectrum_count + 1))
    out = open("../output_files/spectrum_{:08d}_convolved.spec".format(spectrum_count + 1), 'w')

    for i in range(len(wave_line)):
        print("{}  {}".format(wave_line[i], flux_line_mod[i]), file=out)
    out.close()
    return chi_square


def chi_square_broad_lbl(param, ts, obs_name, temp_directory, spectrum_count, depart_bin_file_list,
                         depart_aux_file_list,
                         model_atom_file_list, atmosphere_type, nlte_flag, doppler, teff, logg, met, fit_microturb,
                         macro, fwhm, rot, abund_name, ldelta, lmin, lmax, wave_obs, flux_obs):
    abund, doppler, macro, vturb, wave_obs = prepar_lbl_calc(atmosphere_type, doppler, fit_microturb, logg, macro, met,
                                                             param, teff, wave_obs)

    if met > 0.5 or met < -4.0 or vturb <= 0.0:
        chi_square = 9999.9999
    elif macro < 0.0:
        chi_square = 99.9999
    elif abund < -40:
        chi_square = 999.999
    else:
        item_abund = {"Fe": met, abund_name[0]: abund + met}

        if nlte_flag == "False":    #TODO make it same as other ones?
            ts.configure(t_eff=teff, log_g=logg, metallicity=met,
                         turbulent_velocity=vturb, lambda_delta=ldelta, lambda_min=lmin, lambda_max=lmax,
                         free_abundances=item_abund, temp_directory=temp_directory, nlte_flag=False, verbose=False,
                         atmosphere_dimension=atmosphere_type,
                         windows_flag=False)  # , depart_bin_file=depart_bin_file,
            # depart_aux_file=depart_aux_file, model_atom_file=model_atom_file)
        elif nlte_flag == "True":
            depart_bin_file = {}
            for i in range(len(depart_bin_file_list)):
                depart_bin_file[abund_name[i]] = depart_bin_file_list[i]
            depart_aux_file = {}
            for i in range(len(depart_aux_file_list)):
                depart_aux_file[abund_name[i]] = depart_aux_file_list[i]
            model_atom_file = {}
            for i in range(len(model_atom_file_list)):
                model_atom_file[abund_name[i]] = model_atom_file_list[i]
            ts.configure(t_eff=teff, log_g=logg, metallicity=met,
                         turbulent_velocity=vturb, lambda_delta=ldelta, lambda_min=lmin, lambda_max=lmax,
                         free_abundances=item_abund, temp_directory=temp_directory, nlte_flag=True, verbose=False,
                         atmosphere_dimension=atmosphere_type, windows_flag=False, depart_bin_file=depart_bin_file,
                         depart_aux_file=depart_aux_file, model_atom_file=model_atom_file)

        # ts.configure(t_eff = teff, log_g = logg, metallicity = met, turbulent_velocity = vturb, lambda_delta = ldelta, lambda_min=lmin, lambda_max=lmax, free_abundances=item_abund, temp_directory = '/Users/gerber/gitprojects/SAPP/UVES_benchmark_3d_ca/', nlte_flag=False, verbose=False)

        ts.run_turbospectrum_and_atmosphere()

        # macro = macro + param[2]

        if os_path.exists('{}/spectrum_00000000.spec'.format(temp_directory)) and os.stat(
                '{}/spectrum_00000000.spec'.format(temp_directory)).st_size != 0:
            chi_square = calculate_lbl_chi_squared(flux_obs, fwhm, lmax, lmin, macro, rot, spectrum_count,
                                                   temp_directory, wave_obs)
        elif os_path.exists('{}/spectrum_00000000.spec'.format(temp_directory)) and os.stat(
                '{}/spectrum_00000000.spec'.format(temp_directory)).st_size == 0:
            chi_square = 999.99
            print("empty spectrum file.")
        else:
            chi_square = 9999.9999
            print("didn't generate spectra")

    print(abund, vturb, macro, doppler, chi_square, macro)

    return chi_square


def chi_square_broad_met_lbl(param, ts, obs_name, temp_directory, spectrum_count, depart_bin_file_list,
                             depart_aux_file_list, model_atom_file_list, atmosphere_type, nlte_flag, doppler, teff,
                             logg, fit_microturb, macro, fwhm, rot, ldelta, lmin, lmax, wave_obs, flux_obs):
    # metallicity here is param[0]
    met, doppler, macro, vturb, wave_obs = prepar_lbl_calc(atmosphere_type, doppler, fit_microturb, logg, macro,
                                                           param[0], param, teff, wave_obs)

    if met > 0.5 or met < -4.0 or vturb <= 0.0:
        chi_square = 9999.9999
    elif macro < 0.0:
        chi_square = 99.9999
    else:
        item_abund = {"Fe": met}

        if nlte_flag == "False":
            ts.configure(t_eff=teff, log_g=logg, metallicity=met,
                         turbulent_velocity=vturb, lambda_delta=ldelta, lambda_min=lmin, lambda_max=lmax,
                         free_abundances=item_abund, temp_directory=temp_directory, nlte_flag=False, verbose=False,
                         atmosphere_dimension=atmosphere_type,
                         windows_flag=False)  # , depart_bin_file=depart_bin_file,
            # depart_aux_file=depart_aux_file, model_atom_file=model_atom_file)
        elif nlte_flag == "True":
            depart_bin_file = {}
            depart_bin_file['Fe'] = depart_bin_file_list[0]
            depart_aux_file = {}
            depart_aux_file['Fe'] = depart_aux_file_list[0]
            model_atom_file = {}
            model_atom_file['Fe'] = model_atom_file_list[0]
            ts.configure(t_eff=teff, log_g=logg, metallicity=met,
                         turbulent_velocity=vturb, lambda_delta=ldelta, lambda_min=lmin, lambda_max=lmax,
                         free_abundances=item_abund, temp_directory=temp_directory, nlte_flag=True, verbose=False,
                         atmosphere_dimension=atmosphere_type, windows_flag=False, depart_bin_file=depart_bin_file,
                         depart_aux_file=depart_aux_file, model_atom_file=model_atom_file)

        ts.run_turbospectrum_and_atmosphere()

        if os_path.exists('{}/spectrum_00000000.spec'.format(temp_directory)) and os.stat(
                '{}/spectrum_00000000.spec'.format(temp_directory)).st_size != 0:
            chi_square = calculate_lbl_chi_squared(flux_obs, fwhm, lmax, lmin, macro, rot, spectrum_count,
                                                   temp_directory, wave_obs)
        elif os_path.exists('{}/spectrum_00000000.spec'.format(temp_directory)) and os.stat(
                '{}/spectrum_00000000.spec'.format(temp_directory)).st_size == 0:
            chi_square = 999.99
            print("empty spectrum file.")
        else:
            chi_square = 9999.9999
            print("didn't generate spectra or atmosphere")

    print(met, vturb, doppler, chi_square, macro)

    return chi_square


def fit_one_spectra(atmosphere_type, depart_aux_file, depart_bin_file, departure_file_path, element, fit_microturb,
                    fitting_mode, fwhm, i, include_molecules, initial_guess, interpol_path, ldelta, line_list_path_orig,
                    line_list_path_trimmed, linemask_file, lmax, lmin, logg_fitlist, macroturb, met_fitlist,
                    model_atmosphere_grid_path, model_atmosphere_list, model_atom_file, model_atom_path, ndimen,
                    nlte_flag, param0, rot, rv_fitlist, segment_file, specname_fitlist, teff_fitlist, temp_directory,
                    turbospec_path):
    # this next step is in case you're only fitting one star
    if specname_fitlist.size > 1 and (element[0] == "Fe" or element[0] == "fe"):
        specname = "../input_files/observed_spectra/" + specname_fitlist[i]
        temp_directory = f"{temp_directory}{specname_fitlist[i]}/"
        teff = teff_fitlist[i]
        logg = logg_fitlist[i]
        rv = rv_fitlist[i]
    elif specname_fitlist.size > 1:
        specname = "../input_files/observed_spectra/" + specname_fitlist[i]
        temp_directory = f"{temp_directory}{specname_fitlist[i]}/"
        teff = teff_fitlist[i]
        logg = logg_fitlist[i]
        rv = rv_fitlist[i]
        met = met_fitlist[i]
    elif specname_fitlist.size == 1 and (element[0] == "Fe" or element[0] == "fe"):
        specname = "../input_files/observed_spectra/" + np.str(specname_fitlist)
        temp_directory = f"{temp_directory}{np.str(specname_fitlist)}/"
        teff = teff_fitlist
        logg = logg_fitlist
        rv = rv_fitlist
    elif specname_fitlist.size == 1:
        specname = "../input_files/observed_spectra/" + np.str(specname_fitlist)
        temp_directory = f"{temp_directory}{np.str(specname_fitlist)}"
        teff = teff_fitlist
        logg = logg_fitlist
        rv = rv_fitlist
        met = met_fitlist
    if not os.path.exists(temp_directory):
        try:
            os.mkdir(temp_directory)
        except FileNotFoundError:
            os.makedirs(temp_directory)
    print(f"Fitting {specname}")
    print(f"Teff = {teff}; logg = {logg}; RV = {rv}")
    # time_start = time.time() used to evaluate computation time
    ts = TurboSpectrum(
        turbospec_path=turbospec_path,
        interpol_path=interpol_path,
        line_list_paths=line_list_path_trimmed,
        marcs_grid_path=model_atmosphere_grid_path,
        marcs_grid_list=model_atmosphere_list,
        model_atom_path=model_atom_path,
        departure_file_path=departure_file_path)
    wave_ob, flux_ob = np.loadtxt(np.str(specname), usecols=(0, 1), unpack=True)
    if fitting_mode == "all":
        line_centers, line_begins, line_ends = np.loadtxt(linemask_file, comments=";", usecols=(0, 1, 2),
                                                          unpack=True)

        line_begins_sorted = np.array(sorted(line_begins))
        line_ends_sorted = np.array(sorted(line_ends))

        seg_begins, seg_ends = np.loadtxt(segment_file, comments=";", usecols=(0, 1), unpack=True)

        if element[0] == "Fe" or element[0] == "fe":
            time_start = time.time()

            res = minimize(chi_square_broad_met, param0, args=(ts,
                                                               np.str(specname), temp_directory, i, linemask_file,
                                                               segment_file, depart_bin_file, depart_aux_file,
                                                               model_atom_file, atmosphere_type, nlte_flag, rv,
                                                               teff, logg, macroturb, fwhm, rot, ldelta, lmin, lmax,
                                                               wave_ob, flux_ob, line_begins_sorted,
                                                               line_ends_sorted, seg_begins, seg_ends),
                           method='Nelder-Mead',
                           options={'maxiter': ndimen * 50, 'disp': True, 'initial_simplex': initial_guess,
                                    'xatol': 0.05, 'fatol': 0.05})

            print(res.x)

            if len(param0) < 3:
                result = f"{specname.replace('../input_files/observed_spectra/', '')} {res.x[0]} {res.x[1]} {res.fun}"
            else:
                result = f"{specname.replace('../input_files/observed_spectra/', '')} {res.x[0]} {res.x[1]} {res.fun} {res.x[2]}"

            time_end = time.time()
            print("Total runtime was {:.2f} minutes.".format((time_end - time_start) / 60.))

        else:
            if atmosphere_type == "1D":
                vturb = calculate_vturb(teff, logg, met)
            elif atmosphere_type == "3D":
                vturb = 2.0

            res = minimize(chi_square_broad, param0, args=(ts,
                                                           np.str(specname), temp_directory, i, linemask_file,
                                                           segment_file, depart_bin_file, depart_aux_file,
                                                           model_atom_file, atmosphere_type, nlte_flag, rv, teff,
                                                           logg, met, vturb, macroturb, fwhm, rot, element,
                                                           ldelta, lmin, lmax, wave_ob,
                                                           flux_ob, line_begins_sorted, line_ends_sorted,
                                                           seg_begins, seg_ends), method='Nelder-Mead',
                           options={'maxiter': ndimen * 50, 'disp': True, 'initial_simplex': initial_guess,
                                    'xatol': 0.05, 'fatol': 0.05})
            print(res.x)

            if len(param0) < 3:
                result = f"{specname.replace('../input_files/observed_spectra/', '')} {res.x[0]} {res.x[1]} {res.fun}"
            else:
                result = f"{specname.replace('../input_files/observed_spectra/', '')} {res.x[0]} {res.x[1]} {res.fun} {res.x[2]}"
    elif fitting_mode == "lbl":
        result = []
        line_centers, line_begins, line_ends = np.loadtxt(linemask_file, comments=";", usecols=(0, 1, 2),
                                                          unpack=True)
        if line_centers.size > 1:
            line_begins_sorted = sorted(line_begins)
            line_ends_sorted = sorted(line_ends)
            line_centers_sorted = sorted(line_centers)
        elif line_centers.size == 1:
            line_begins_sorted = [line_begins]
            line_ends_sorted = [line_ends]
            line_centers_sorted = [line_centers]

        seg_begins, seg_ends = np.loadtxt(segment_file, comments=";", usecols=(0, 1), unpack=True)

        for j in range(len(line_begins_sorted)):
            for k in range(len(seg_begins)):
                if line_centers_sorted[j] <= seg_ends[k] and line_centers_sorted[j] > seg_begins[k]:
                    start = k
            line_list_path_trimmed_element = f"{line_list_path_trimmed}_{segment_file.replace('/', '_')}_{element[0]}_{include_molecules}_{start}_{start + 1}/"
            create_window_linelist(segment_file, line_list_path_orig, line_list_path_trimmed_element, include_molecules,
                                   start, start + 1)

        if element[0] == "Fe" or element[0] == "fe":
            for j in range(len(line_begins_sorted)):
                time_start = time.time()
                print("Fitting line at {} angstroms".format(line_centers_sorted[j]))

                for k in range(len(seg_begins)):
                    if line_centers_sorted[j] <= seg_ends[k] and line_centers_sorted[j] > seg_begins[k]:
                        start = k
                print(line_centers_sorted[j], seg_begins[start], seg_ends[start])

                ts.line_list_paths = f"{line_list_path_trimmed}_{segment_file.replace('/', '_')}_{element[0]}_{include_molecules}_{start}_{start + 1}/"

                #os.system("rm {}*".format(line_list_path_trimmed))

                #create_window_linelist(segment_file, line_list_path_orig, line_list_path_trimmed, include_molecules,
                #                       start, start + 1)    #TODO not recreate window every time here as well

                res = minimize(chi_square_broad_met_lbl, param0, args=(ts,
                                                                       np.str(specname), temp_directory, i,
                                                                       depart_bin_file, depart_aux_file,
                                                                       model_atom_file,
                                                                       atmosphere_type,
                                                                       nlte_flag, rv, teff, logg, fit_microturb,
                                                                       macroturb, fwhm, rot, ldelta,
                                                                       line_begins_sorted[j] - 5.,
                                                                       line_ends_sorted[j] + 5., wave_ob, flux_ob),
                               method='Nelder-Mead',
                               options={'maxiter': ndimen * 50, 'disp': True, 'initial_simplex': initial_guess,
                                        'xatol': 0.05, 'fatol': 0.05})

                print(res.x)

                if fit_microturb == "Yes":
                    vturb = res.x[1]
                elif fit_microturb == "No":
                    vturb = calculate_vturb(teff, logg, res.x[0])

                result.append("{} {} {} {} {} {} {} {}".format(specname.replace("../input_files/observed_spectra/", ""),
                                                       line_centers_sorted[j], line_begins_sorted[j],
                                                       line_ends_sorted[j], res.x[0], vturb, res.x[-1], res.fun))

                wave_result, flux_norm_result, flux_result = np.loadtxt(
                    "../output_files/spectrum_{:08d}.spec".format(i + 1), unpack=True)
                g = open("../output_files/result_spectrum_{:08d}.spec".format(i), 'a')
                for k in range(len(wave_result)): #TODO not opening file?
                    print("{}  {}  {}".format(wave_result[k], flux_norm_result[k], flux_result[k]), file=g)
                os.system("rm ../output_files/spectrum_{:08d}.spec".format(i + 1))

                wave_result, flux_norm_result = np.loadtxt(
                    "../output_files/spectrum_{:08d}_convolved.spec".format(i + 1), unpack=True)
                h = open("../output_files/result_spectrum_{:08d}_convolved.spec".format(i), 'a')
                for k in range(len(wave_result)):
                    print("{}  {}".format(wave_result[k], flux_norm_result[k]), file=h)
                os.system("rm ../output_files/spectrum_{:08d}_convolved.spec".format(i + 1))

                #os.system("rm {}*".format(line_list_path_trimmed))  #TODO not recreate window every time here as well

                time_end = time.time()
                print("Total runtime was {:.2f} minutes.".format((time_end - time_start) / 60.))

            g.close()
            h.close()

        else:
            for j in range(len(line_begins_sorted)):
                time_start = time.time()
                print(f"Fitting line at {line_centers_sorted[j]} angstroms")

                for k in range(len(seg_begins)):
                    if line_centers_sorted[j] <= seg_ends[k] and line_centers_sorted[j] > seg_begins[k]:
                        start = k
                print(line_centers_sorted[j], seg_begins[start], seg_ends[start])

                ts.line_list_paths = f"{line_list_path_trimmed}_{segment_file.replace('/', '_')}_{element[0]}_{include_molecules}_{start}_{start + 1}/"

                #os.system("rm {}/*".format(line_list_path_trimmed))

                #create_window_linelist(segment_file, line_list_path_orig, line_list_path_trimmed, include_molecules,
                #                       start, start + 1)  #TODO not recreate window every time here as well

                res = minimize(chi_square_broad_lbl, param0, args=(ts,
                                                                   np.str(specname), temp_directory, i,
                                                                   depart_bin_file, depart_aux_file,
                                                                   model_atom_file,
                                                                   atmosphere_type,
                                                                   nlte_flag, rv, teff, logg, met, fit_microturb,
                                                                   macroturb, fwhm, rot, element,
                                                                   ldelta, line_begins_sorted[j] - 5.,
                                                                   line_ends_sorted[j] + 5., wave_ob, flux_ob),
                               method='Nelder-Mead',
                               options={'maxiter': ndimen * 50, 'disp': True, 'initial_simplex': initial_guess,
                                        'xatol': 0.05, 'fatol': 0.05})

                print(res.x)

                if fit_microturb == "Yes":
                    vturb = res.x[1]
                elif fit_microturb == "No":
                    vturb = calculate_vturb(teff, logg, met)

                result.append("{} {} {} {} {} {} {} {}".format(specname.replace("../input_files/observed_spectra/", ""),
                                                       line_centers_sorted[j], line_begins_sorted[j],
                                                       line_ends_sorted[j], res.x[0], vturb, res.x[-1], res.fun),
                )

                wave_result, flux_norm_result, flux_result = np.loadtxt(
                    "../output_files/spectrum_{:08d}.spec".format(i + 1), unpack=True)
                g = open("../output_files/result_spectrum_{}.spec".format(
                    specname.replace("../input_files/observed_spectra/", "")), 'a')
                for k in range(len(wave_result)):
                    print("{}  {}  {}".format(wave_result[k], flux_norm_result[k], flux_result[k]), file=g)
                os.system("rm ../output_files/spectrum_{:08d}.spec".format(i + 1))

                wave_result, flux_norm_result = np.loadtxt(
                    "../output_files/spectrum_{:08d}_convolved.spec".format(i + 1), unpack=True)
                h = open("../output_files/result_spectrum_{}_convolved.spec".format(
                    specname.replace("../input_files/observed_spectra/", "")), 'a')
                for k in range(len(wave_result)):
                    print("{}  {}".format(wave_result[k], flux_norm_result[k]), file=h)
                os.system("rm ../output_files/spectrum_{:08d}_convolved.spec".format(i + 1))

                #os.system("rm {}/*".format(line_list_path_trimmed))  #TODO not recreate window every time here as well

                time_end = time.time()
                print("Total runtime was {:.2f} minutes.".format((time_end - time_start) / 60.))

            g.close()
            h.close()
    shutil.rmtree(temp_directory)
    return result

def run_TSFitPy():
    # set defaults
    include_molecules = "True"

    depart_bin_file = []
    depart_aux_file = []
    model_atom_file = []

    # read the configuration file
    with open("../input_files/tsfitpy_input_configuration.txt") as fp:
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
            #    #line = fp.readline()
            # if fields[0] == "interpol_path":
            #    interpol_path = fields[2]
            # if fields[0] == "line_list_path":
            #    line_list_path = fields[2]
            # if fields[0] == "line_list_folder":
            #    linelist_folder = fields[2]
            # if fields[0] == "model_atmosphere_grid_path":
            #    model_atmosphere_grid_path = fields[2]
            # if fields[0] == "model_atmosphere_folder":
            #    model_atmosphere_folder = fields[2]
            # if fields[0] == "model_atmosphere_list":
            #    model_atmosphere_list = fields[2]
            # if fields[0] == "model_atom_path":
            #    model_atom_path = fields[2]
            # if fields[0] == "departure_file_path":
            #    departure_file_path = fields[2]
            if fields[0] == "turbospectrum_compiler":
                ts_compiler = fields[2]
            if fields[0] == "atmosphere_type":
                atmosphere_type = fields[2]
            if fields[0] == "mode":
                fitting_mode = fields[2]
            if fields[0] == "include_molecules":
                include_molecules = fields[2]
            if fields[0] == "nlte":
                nlte_flag = fields[2]
            if fields[0] == "fit_microturb":
                fit_microturb = fields[2]
            if fields[0] == "fit_teff":
                fit_teff = fields[2]
            if fields[0] == "fit_logg":
                fit_logg = fields[2]
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
                lmin = float(fields[2])
            if fields[0] == "wavelength_maximum":
                lmax = float(fields[2])
            if fields[0] == "wavelength_delta":
                ldelta = float(fields[2])
            if fields[0] == "resolution":
                fwhm = float(fields[2])
            if fields[0] == "macroturbulence":
                macroturb = float(fields[2])
            if fields[0] == "rotation":
                rot = float(fields[2])
            if fields[0] == "temporary_directory":
                temp_directory = fields[2]
                today = datetime.datetime.now().strftime("%b-%d-%Y-%H-%M-%S")
                temp_directory = f"../{temp_directory}{today}/"
                if not os.path.exists(temp_directory):
                    try:
                        os.mkdir(temp_directory)
                    except FileNotFoundError:
                        os.makedirs(temp_directory)
            if fields[0] == "initial_guess_array":
                initial_guess_string = fields[2].strip().split(",")
            if fields[0] == "ndimen":
                ndimen = int(fields[2])
            if fields[0] == "input_file":
                fitlist = fields[2]
            if fields[0] == "output_file":
                output = fields[2]
            line = fp.readline()
        fp.close()

    # set directories
    if ts_compiler == "intel":
        turbospec_path = "../turbospectrum/exec/"
    elif ts_compiler == "gnu":
        turbospec_path = "../turbospectrum/exec-gf/"
    interpol_path = "./model_interpolators/"
    line_list_path_orig = "../input_files/linelists/linelist_for_fitting/"
    line_list_path_trimmed = "../input_files/linelists/linelist_for_fitting_trimmed/"
    if atmosphere_type == "1D":
        model_atmosphere_grid_path = "/mnt/beegfs/gemini/groups/bergemann/users/storm/data/TSFitPy_input_model_atmospheres/model_atmospheres/1D/"
        model_atmosphere_list = model_atmosphere_grid_path + "model_atmosphere_list.txt"
    elif atmosphere_type == "3D":
        model_atmosphere_grid_path = "/mnt/beegfs/gemini/groups/bergemann/users/storm/data/TSFitPy_input_model_atmospheres/model_atmospheres/3D/"
        model_atmosphere_list = model_atmosphere_grid_path + "model_atmosphere_list.txt"
    model_atom_path = "/mnt/beegfs/gemini/groups/bergemann/users/storm/data/nlte_data/model_atoms/"
    departure_file_path = "/mnt/beegfs/gemini/groups/bergemann/users/storm/data/nlte_data/"

    linemask_file = "../input_files/linemask_files/" + linemask_file
    segment_file = "../input_files/linemask_files/" + segment_file
    # continuum_file = "../input_files/linemask_files/"+continuum_file

    # make an array for initial guess equal to n x ndimen+1
    initial_guess = np.empty((ndimen + 1, ndimen))

    # fill the array with input from config file
    for j in range(ndimen):
        for i in range(j, len(initial_guess_string), ndimen):
            initial_guess[int(i / ndimen)][j] = float(initial_guess_string[i])

    # time_start_tot = time.time() line used for evaluating computation time

    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)

    param0 = initial_guess[0]

    fitlist = "../input_files/" + fitlist

    specname_fitlist = np.loadtxt(fitlist, dtype='str', usecols=(0), unpack=True)

    # if fit_teff == "Yes" and fit_logg == "No" and element[0] == "Fe":

    if (element[0] == "Fe" or element[0] == "fe") and fit_teff == "No" and fit_logg == "No":
        rv_fitlist, teff_fitlist, logg_fitlist = np.loadtxt(fitlist, usecols=(1, 2, 3), unpack=True)
        met_fitlist = None
    else:
        rv_fitlist, teff_fitlist, logg_fitlist, met_fitlist = np.loadtxt(fitlist, usecols=(1, 2, 3, 4), unpack=True)

    seg_begins, seg_ends = np.loadtxt(segment_file, comments=";", usecols=(0, 1), unpack=True)

    if fitting_mode == "all":
        print("Trimming down the linelist to only lines within segments for faster fitting")
        # os.system("rm {}/*".format(line_list_path_trimmed))
        trimmed_start = 0
        trimmed_end = len(seg_ends)
        line_list_path_trimmed = f"{line_list_path_trimmed}_{segment_file.replace('/', '_')}_{include_molecules}_{trimmed_start}_{trimmed_end}/"
        create_window_linelist(segment_file, line_list_path_orig, line_list_path_trimmed, include_molecules,
                               trimmed_start,
                               trimmed_end)
    else:
        line_list_path_trimmed = line_list_path_trimmed + "lbl/"
        if not os.path.exists(line_list_path_trimmed):
            os.makedirs(line_list_path_trimmed)

    print("Finished trimming linelist")

    if workers > 1:
        print("Preparing workers")
        client = Client(threads_per_worker=threads_per_worker, n_workers=workers)
        print(client)

        host = client.run_on_scheduler(socket.gethostname)
        port = client.scheduler_info()['services']['dashboard']
        print(f"Assuming that the cluster is ran at {login_node_address} (change in code if not the case)")

        logger.info(f"ssh -N -L {port}:{host}:{port} {login_node_address}")

        print("Worker preparation complete")

        futures = []
        for i in range(specname_fitlist.size):
            future = client.submit(fit_one_spectra, atmosphere_type, depart_aux_file, depart_bin_file, departure_file_path, element,
                            fit_microturb, fitting_mode, fwhm, i, include_molecules, initial_guess, interpol_path, ldelta,
                            line_list_path_orig, line_list_path_trimmed, linemask_file, lmax, lmin, logg_fitlist, macroturb,
                            met_fitlist, model_atmosphere_grid_path, model_atmosphere_list, model_atom_file,
                            model_atom_path, ndimen, nlte_flag, param0, rot, rv_fitlist, segment_file, specname_fitlist,
                            teff_fitlist, temp_directory, turbospec_path)

            futures.append(future)  # prepares to get values

        print("start gathering")  # use http://localhost:8787/status to check status. the port might be different
        futures = np.array(client.gather(futures))  # starts the calculations (takes a long time here)
        results = futures
        print("Worker calculation done")  # when done, save values
    else:
        results = []
        for i in range(specname_fitlist.size):
            results.append(fit_one_spectra(atmosphere_type, depart_aux_file, depart_bin_file, departure_file_path, element,
                            fit_microturb, fitting_mode, fwhm, i, include_molecules, initial_guess, interpol_path, ldelta,
                            line_list_path_orig, line_list_path_trimmed, linemask_file, lmax, lmin, logg_fitlist, macroturb,
                            met_fitlist, model_atmosphere_grid_path, model_atmosphere_list, model_atom_file,
                            model_atom_path, ndimen, nlte_flag, param0, rot, rv_fitlist, segment_file, specname_fitlist,
                            teff_fitlist, temp_directory, turbospec_path))

    shutil.rmtree(temp_directory)   # clean up temp directory

    output = "../output_files/" + output

    f = open(output, 'a')

    if fitting_mode == "all" and (
            element[0] == "Fe" or element[0] == "fe"):  # TODO add other parameters? macroturbulence?
        print("#specname        Fe_H     Doppler_Shift_add_to_RV    chi_squared", file=f)
    elif fitting_mode == "all":
        print(f"#specname        {element[0]}_Fe     Doppler_Shift_add_to_RV    chi_squared", file=f)
    elif fitting_mode == "lbl" and (element[0] == "Fe" or element[0] == "fe"):
        print(
            "#specname        wave_center  wave_start  wave_end  Fe_H    Microturb     Doppler_Shift_add_to_RV    chi_squared", file=f
        )
    elif fitting_mode == "lbl":
        print(
            f"#specname        wave_center  wave_start  wave_end  {element[0]}_Fe   Microturb     Doppler_Shift_add_to_RV    chi_squared", file=f)

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
    login_node_address = "gemini-login.mpia.de"  # Change this to the address/domain of your login node
    workers = 1  # should be the same as cores; use value of 1 if you do not want to use multithprocessing
    threads_per_worker = 1  # seemed to work best with 1; play around if you want.
    run_TSFitPy()
