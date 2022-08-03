import numpy as np
from scipy.optimize import minimize
#from multiprocessing import Pool
#import h5py
#import matplotlib.pyplot as plt
from turbospectrum_class_nlte import TurboSpectrum
#from turbospectrum_class_3d import TurboSpectrum_3D
import time
#import math
import os
from os import path as os_path
#import glob

from solar_abundances import solar_abundances, periodic_table

from convolve import *
from create_window_linelist_function import *

def turbospec_atmosphere(teff, logg, met, vturb, ldelta, lmin, lmax):
    ts = TurboSpectrum_3D(
            #turbospec_path="/Users/gerber/iwg7_pipeline/turbospectrum-19.1/exec-gf-v19.1/",
            #turbospec_path="/Users/gerber/iwg7_pipeline/turbospectrum-19.1/exec-v19.1/",
            turbospec_path="/Users/gerber/gitprojects/TurboSpectrum2020_gitversion/Turbospectrum2020/exec/",
            interpol_path="/Users/gerber/gitprojects/TurboSpectrum2020/interpol_modeles_nlte/",
            #interpol_path="/Users/gerber/iwg7_pipeline/interpol_marcs/",
            #line_list_paths="/Users/gerber/iwg7_pipeline/turbospectrum-19.1/COM-v19.1/linelists/UVES_linelists_nlte_fe/",
            line_list_paths="/Users/gerber/iwg7_pipeline/turbospectrum-19.1/COM-v19.1/linelists/UVES_linelists/",
            #marcs_grid_path="/Users/gerber/iwg7_pipeline/fromBengt/marcs_my_search/plane_and_sphere/")
            marcs_grid_path="/Users/gerber/multi_formatted_atmospheres/stagger_mean3D_interpol_headers_marcs_names/")
    
    item_abund = {}
    item_abund["Fe"] = met

    ts.configure(t_eff = teff, log_g = logg, metallicity = met, turbulent_velocity = vturb, lambda_delta = ldelta, lambda_min=lmin, lambda_max=lmax, free_abundances=item_abund, temp_directory = '/Users/gerber/gitprojects/SAPP/UVES_benchmark_3d_ca/', nlte_flag=False, verbose=False)

    #ts.calculate_atmosphere()

    ts.run_turbospectrum_and_atmosphere()

    #ts.run_babsma_and_atmosphere()

def calculate_atmosphere(teff, logg, met, vturb, ldelta, lmin, lmax):
    ts = TurboSpectrum_3D(
            #turbospec_path="/Users/gerber/iwg7_pipeline/turbospectrum-19.1/exec-gf-v19.1/",
            #turbospec_path="/Users/gerber/iwg7_pipeline/turbospectrum-19.1/exec-v19.1/",
            turbospec_path="/Users/gerber/gitprojects/TurboSpectrum2020_gitversion/Turbospectrum2020/exec/",
            interpol_path="/Users/gerber/gitprojects/TurboSpectrum2020/interpol_modeles_nlte/",
            #interpol_path="/Users/gerber/iwg7_pipeline/interpol_marcs/",
            #line_list_paths="/Users/gerber/iwg7_pipeline/turbospectrum-19.1/COM-v19.1/linelists/UVES_linelists_nlte_fe/",
            line_list_paths="/Users/gerber/iwg7_pipeline/turbospectrum-19.1/COM-v19.1/linelists/UVES_linelists/",
            #marcs_grid_path="/Users/gerber/iwg7_pipeline/fromBengt/marcs_my_search/plane_and_sphere/")
            marcs_grid_path="/Users/gerber/multi_formatted_atmospheres/stagger_mean3D_interpol_headers_marcs_names/")
    
    item_abund = {}
    item_abund["Fe"] = met

    ts.configure(t_eff = teff, log_g = logg, metallicity = met, turbulent_velocity = vturb, lambda_delta = ldelta, lambda_min=lmin, lambda_max=lmax, free_abundances=item_abund, temp_directory = '/Users/gerber/gitprojects/SAPP/UVES_benchmark_3d_ca/', nlte_flag=False, verbose=False)

    ts.calculate_atmosphere()

def turbospec(teff, logg, met, vturb, ldelta, lmin, lmax):

    ts.run_turbospectrum()

def calculate_vturb(teff, logg, met):
    t0 = 5500.
    g0 = 4.
    m0 = 0.

    if teff >= 5000.:
        vturb = 1.05 + 2.51e-4*(teff-t0) + 1.5e-7*(teff-t0)*(teff-t0) - 0.14*(logg-g0) - 0.005*(logg-g0)*(logg-g0) + 0.05*met + 0.01*met*met
    elif teff < 5000. and logg >= 3.5:
        vturb = 1.05 + 2.51e-4*(teff-t0) + 1.5e-7*(5250.-t0)*(5250.-t0) - 0.14*(logg-g0) - 0.005*(logg-g0)*(logg-g0) + 0.05*met + 0.01*met*met
    elif teff < 5500. and logg < 3.5:
        vturb = 1.25 + 4.01e-4*(teff-t0) + 3.1e-7*(teff-t0)*(teff-t0) - 0.14*(logg-g0) - 0.005*(logg-g0)*(logg-g0) + 0.05*met + 0.01*met*met

    if teff == 5771 and logg == 4.44:
        vturb = 0.9

    return vturb

def chi_square_broad(param, obs_name, temp_directory, spectrum_count, mask_file, segment_file, depart_bin_file_list, depart_aux_file_list, model_atom_file_list, atmosphere_type, nlte_flag, doppler, teff, logg, met, vturb, macro, fwhm, rot, abund_name, abund_low, abund_high, ldelta, lmin, lmax):
    #teff = param[0]
    #logg = param[1]
    #met = param[2]

    abund = param[0]
    doppler = doppler + param[1]

    wave_obs, flux_obs = np.loadtxt(obs_name, usecols=(0,1), unpack=True)

    wave_obs = wave_obs/(1+(doppler/300000.))

    line_centers, line_begins, line_ends = np.loadtxt(mask_file, comments = ";", usecols=(0,1,2), unpack=True)

    line_begins_sorted = sorted(line_begins)
    line_ends_sorted = sorted(line_ends)

    seg_begins, seg_ends = np.loadtxt(segment_file, comments = ";", usecols=(0,1), unpack=True)

    if atmosphere_type == "1D":
        met_values = [-5, -4, -3, -2.5, -2, -1.5, -1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    elif atmosphere_type == "3D":
        met_values = [-5, -4, -3, -2.5, -2, -1.5, -1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

    for i in range(len(met_values)-1):
        if met >= met_values[i] and met < met_values[i+1]:
            met_low_bracket = met_values[i]
            met_high_bracket = met_values[i+1]

    if met > 0.5 or met < -4.0 or vturb <= 0.0:
        chi_square = 9999.9999
    elif macro < 0.0:
        chi_square = 99.9999
    elif abund < -40:
        chi_square = 999.999
    #elif nlte_flag == "True" and (abund + met + solar_abundances[abund_name[0]] < met_low_bracket + abund_low or abund + met + solar_abundances[abund_name[0]] > met_low_bracket + abund_high or abund + met + solar_abundances[abund_name[0]] < met_high_bracket + abund_low or abund + met + solar_abundances[abund_name[0]] > met_high_bracket + abund_high): #note was 5-8 previously
    #    chi_square = 999.999
    #    print("abundance range outside of values in departure file")
    #elif os_path.exists('{}/marcs_tef{:.1f}_g{:.2f}_z{:.2f}_tur{:.2f}.interpol'.format(temp_directory, teff, logg, met, vturb)):
    else:

        #turbospec_atmosphere(teff, logg, met, vturb, 0.005, 4800, 6800)

        item_abund = {}
        #for i in range(1,len(periodic_table)):
        #    item_abund[periodic_table[i]] = 0.0#solar_abundances[periodic_table[i]] #deleted this and moved to turbospectrum_class_nlte.py (was causing issues with nlte)
        item_abund["Fe"] = met
        item_abund[abund_name[0]] = abund+met

        if nlte_flag == "False":
            ts.configure(t_eff = teff, log_g = logg, metallicity = met, 
                            turbulent_velocity = vturb, lambda_delta = ldelta, lambda_min=lmin, lambda_max=lmax, 
                            free_abundances=item_abund, temp_directory = temp_directory, nlte_flag=False, verbose=False, 
                            atmosphere_dimension=atmosphere_type, windows_flag=True, segment_file=segment_file, 
                            line_mask_file=mask_file)#, depart_bin_file=depart_bin_file, 
                            #depart_aux_file=depart_aux_file, model_atom_file=model_atom_file)
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
            ts.configure(t_eff = teff, log_g = logg, metallicity = met, 
                            turbulent_velocity = vturb, lambda_delta = ldelta, lambda_min=lmin, lambda_max=lmax, 
                            free_abundances=item_abund, temp_directory = temp_directory, nlte_flag=True, verbose=False, 
                            atmosphere_dimension=atmosphere_type, windows_flag=True, segment_file=segment_file, 
                            line_mask_file=mask_file, depart_bin_file=depart_bin_file, 
                            depart_aux_file=depart_aux_file, model_atom_file=model_atom_file)

        #ts.configure(t_eff = teff, log_g = logg, metallicity = met, turbulent_velocity = vturb, lambda_delta = ldelta, lambda_min=lmin, lambda_max=lmax, free_abundances=item_abund, temp_directory = '/Users/gerber/gitprojects/SAPP/UVES_benchmark_3d_ca/', nlte_flag=False, verbose=False)

        ts.run_turbospectrum_and_atmosphere()

        if os_path.exists('{}/spectrum_00000000.spec'.format(temp_directory)) and os.stat('{}/spectrum_00000000.spec'.format(temp_directory)).st_size != 0:
            wave_mod_orig, flux_mod_orig = np.loadtxt('{}/spectrum_00000000.spec'.format(temp_directory), usecols=(0,1), unpack=True)
            wave_mod_filled = []
            flux_mod_filled = []
            for i in range(len(seg_begins)):
                j = 0
                while wave_mod_orig[j] < seg_begins[i]:
                    j+=1
                while wave_mod_orig[j] >= seg_begins[i] and wave_mod_orig[j] <= seg_ends[i]:
                    wave_mod_filled.append(wave_mod_orig[j])
                    flux_mod_filled.append(flux_mod_orig[j])
                    j+=1
                if i < len(seg_begins)-1:
                    k = 1
                    while (seg_begins[i+1] - 0.001 > seg_ends[i]+k*0.005):
                        wave_mod_filled.append(seg_ends[i]+0.005*k)
                        flux_mod_filled.append(1.0)
                        k+=1

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
        
            line_begins_sorted = np.array(line_begins_sorted)
            if wave_mod[1] - wave_mod[0] <= wave_obs[1]-wave_obs[0]:
                flux_mod_interp = np.interp(wave_obs, wave_mod, flux_mod)
                chi_square = 0
                for i in range(len(line_begins_sorted[np.where((line_begins_sorted > np.min(seg_begins)) & (line_begins_sorted < np.max(seg_ends)))])):
                    wave_line = wave_obs[np.where((wave_obs <= line_ends_sorted[i]) & (wave_obs >= line_begins_sorted[i]))]
                    flux_line_obs = flux_obs[np.where((wave_obs <= line_ends_sorted[i]) & (wave_obs >= line_begins_sorted[i]))]
                    flux_line_mod = flux_mod_interp[np.where((wave_obs <= line_ends_sorted[i]) & (wave_obs >= line_begins_sorted[i]))]
                    for j in range(len(wave_line)):
                        chi_square += ((flux_line_obs[j]-flux_line_mod[j])*(flux_line_obs[j]-flux_line_mod[j]))/flux_line_mod[j]
            else:
                flux_obs_interp = np.interp(wave_mod, wave_obs, flux_obs)
                chi_square = 0
                for i in range(len(line_begins_sorted[np.where((line_begins_sorted > np.min(seg_begins)) & (line_begins_sorted < np.max(seg_ends)))])):
                    wave_line = wave_mod[np.where((wave_mod <= line_ends_sorted[i]) & (wave_mod >= line_begins_sorted[i]))]
                    flux_line_obs = flux_obs_interp[np.where((wave_mod <= line_ends_sorted[i]) & (wave_mod >= line_begins_sorted[i]))]
                    flux_line_mod = flux_mod[np.where((wave_mod <= line_ends_sorted[i]) & (wave_mod >= line_begins_sorted[i]))]
                    for j in range(len(wave_line)):
                        chi_square += ((flux_line_obs[j]-flux_line_mod[j])*(flux_line_obs[j]-flux_line_mod[j]))/flux_line_mod[j]
        
            os.system("mv {}spectrum_00000000.spec ../output_files/spectrum_fit_{}".format(temp_directory, obs_name.replace("../input_files/observed_spectra/","")))
            out = open("../output_files/spectrum_fit_convolved_{}".format(obs_name.replace("../input_files/observed_spectra/","")), 'w')
            for i in range(len(wave_mod)):
                print("{}  {}".format(wave_mod[i], flux_mod[i]), file=out)
            out.close()
        elif os_path.exists('{}/spectrum_00000000.spec'.format(temp_directory)) and os.stat('{}/spectrum_00000000.spec'.format(temp_directory)).st_size == 0:
            chi_square = 999.99
            print("empty spectrum file.")
        else:
            chi_square = 9999.9999
            print("didn't generate spectra")
    #else:
    #    chi_square = 9999.9999
    #    print("didn't generate atmosphere")

    print(abund, doppler, chi_square)

    return(chi_square)

def chi_square_broad_met(param, obs_name, temp_directory, spectrum_count, mask_file, segment_file, depart_bin_file_list, depart_aux_file_list, model_atom_file_list, atmosphere_type, nlte_flag, doppler, teff, logg, macro, fwhm, rot, ldelta, lmin, lmax):
    met = param[0]
    doppler = doppler + param[1]

    wave_obs, flux_obs = np.loadtxt(obs_name, usecols=(0,1), unpack=True)

    wave_obs = wave_obs/(1+(doppler/300000.))

    line_centers, line_begins, line_ends = np.loadtxt(mask_file, comments = ";", usecols=(0,1,2), unpack=True)

    line_begins_sorted = sorted(line_begins)
    line_ends_sorted = sorted(line_ends)

    seg_begins, seg_ends = np.loadtxt(segment_file, comments = ";", usecols=(0,1), unpack=True)

    if atmosphere_type == "1D":
        vturb = calculate_vturb(teff, logg, met)
    elif atmosphere_type == "3D":
        vturb = 2.0

    if met > 0.5 or met < -4.0 or vturb <= 0.0:
        chi_square = 9999.9999
    elif macro < 0.0:
        chi_square = 9999.9999
    else:

        item_abund = {}
        #for i in range(1,len(periodic_table)):
        #    item_abund[periodic_table[i]] = 0.0#solar_abundances[periodic_table[i]] #deleted this and moved to turbospectrum_class_nlte.py (was causing issues with nlte)
        item_abund["Fe"] = met

        if nlte_flag == "False":
            ts.configure(t_eff = teff, log_g = logg, metallicity = met, 
                            turbulent_velocity = vturb, lambda_delta = ldelta, lambda_min=lmin, lambda_max=lmax, 
                            free_abundances=item_abund, temp_directory = temp_directory, nlte_flag=False, verbose=False, 
                            atmosphere_dimension=atmosphere_type, windows_flag=True, segment_file=segment_file, 
                            line_mask_file=mask_file)#, depart_bin_file=depart_bin_file, 
                            #depart_aux_file=depart_aux_file, model_atom_file=model_atom_file)
        elif nlte_flag == "True":
            depart_bin_file = {}
            for i in range(len(depart_bin_file_list)):
                depart_bin_file["Fe"] = depart_bin_file_list[i]
            depart_aux_file = {}
            for i in range(len(depart_aux_file_list)):
                depart_aux_file["Fe"] = depart_aux_file_list[i]
            model_atom_file = {}
            for i in range(len(model_atom_file_list)):
                model_atom_file["Fe"] = model_atom_file_list[i]
            ts.configure(t_eff = teff, log_g = logg, metallicity = met, 
                            turbulent_velocity = vturb, lambda_delta = ldelta, lambda_min=lmin, lambda_max=lmax, 
                            free_abundances=item_abund, temp_directory = temp_directory, nlte_flag=True, verbose=False, 
                            atmosphere_dimension=atmosphere_type, windows_flag=True, segment_file=segment_file, 
                            line_mask_file=mask_file, depart_bin_file=depart_bin_file, 
                            depart_aux_file=depart_aux_file, model_atom_file=model_atom_file)
        
        ts.run_turbospectrum_and_atmosphere()

        if os_path.exists('{}/spectrum_00000000.spec'.format(temp_directory)) and os.stat('{}/spectrum_00000000.spec'.format(temp_directory)).st_size != 0:
            wave_mod_orig, flux_mod_orig = np.loadtxt('{}/spectrum_00000000.spec'.format(temp_directory), usecols=(0,1), unpack=True)
            wave_mod_filled = []
            flux_mod_filled = []
            for i in range(len(seg_begins)):
                j = 0
                while wave_mod_orig[j] < seg_begins[i]:
                    j+=1
                while wave_mod_orig[j] >= seg_begins[i] and wave_mod_orig[j] <= seg_ends[i]:
                    wave_mod_filled.append(wave_mod_orig[j])
                    flux_mod_filled.append(flux_mod_orig[j])
                    j+=1
                if i < len(seg_begins)-1:
                    k = 1
                    while (seg_begins[i+1] - 0.001 > seg_ends[i]+k*0.005):
                        wave_mod_filled.append(seg_ends[i]+0.005*k)
                        flux_mod_filled.append(1.0)
                        k+=1

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
        
            line_begins_sorted = np.array(line_begins_sorted)
            if wave_mod[1] - wave_mod[0] <= wave_obs[1]-wave_obs[0]:
                flux_mod_interp = np.interp(wave_obs, wave_mod, flux_mod)
                chi_square = 0
                for i in range(len(line_begins_sorted[np.where((line_begins_sorted > np.min(seg_begins)) & (line_begins_sorted < np.max(seg_ends)))])):
                    #print(line_begins_sorted[i])
                    wave_line = wave_obs[np.where((wave_obs <= line_ends_sorted[i]) & (wave_obs >= line_begins_sorted[i]))]
                    flux_line_obs = flux_obs[np.where((wave_obs <= line_ends_sorted[i]) & (wave_obs >= line_begins_sorted[i]))]
                    flux_line_mod = flux_mod_interp[np.where((wave_obs <= line_ends_sorted[i]) & (wave_obs >= line_begins_sorted[i]))]
                    for j in range(len(wave_line)):
                        chi_square += ((flux_line_obs[j]-flux_line_mod[j])*(flux_line_obs[j]-flux_line_mod[j]))/flux_line_mod[j]
                    #print(chi_square)
            else:
                flux_obs_interp = np.interp(wave_mod, wave_obs, flux_obs)
                chi_square = 0
                for i in range(len(line_begins_sorted[np.where((line_begins_sorted > np.min(seg_begins)) & (line_begins_sorted < np.max(seg_ends)))])):
                    #print(line_begins_sorted[i])
                    wave_line = wave_mod[np.where((wave_mod <= line_ends_sorted[i]) & (wave_mod >= line_begins_sorted[i]))]
                    flux_line_obs = flux_obs_interp[np.where((wave_mod <= line_ends_sorted[i]) & (wave_mod >= line_begins_sorted[i]))]
                    flux_line_mod = flux_mod[np.where((wave_mod <= line_ends_sorted[i]) & (wave_mod >= line_begins_sorted[i]))]
                    for j in range(len(wave_line)):
                        chi_square += ((flux_line_obs[j]-flux_line_mod[j])*(flux_line_obs[j]-flux_line_mod[j]))/flux_line_mod[j]
                    #print(chi_square)
        
            os.system("mv {}spectrum_00000000.spec ../output_files/spectrum_fit_{}".format(temp_directory, obs_name.replace("../input_files/observed_spectra/","")))
            out = open("../output_files/spectrum_fit_convolved_{}".format(obs_name.replace("../input_files/observed_spectra/","")), 'w')
            for i in range(len(wave_mod)):
                print("{}  {}".format(wave_mod[i], flux_mod[i]), file=out)
            out.close()
        elif os_path.exists('{}/spectrum_00000000.spec'.format(temp_directory)) and os.stat('{}/spectrum_00000000.spec'.format(temp_directory)).st_size == 0:
            chi_square = 999.99
            print("empty spectrum file.")
        else:
            chi_square = 9999.9999
            print("didn't generate spectra or atmosphere")

    print(met, doppler, chi_square, vturb)

    return(chi_square)

def chi_square_broad_lbl(param, obs_name, temp_directory, spectrum_count, depart_bin_file_list, depart_aux_file_list, model_atom_file_list, atmosphere_type, nlte_flag, doppler, teff, logg, met, fit_microturb, macro, fwhm, rot, abund_name, abund_low, abund_high, ldelta, lmin, lmax):
    abund = param[0]
    doppler = doppler + param[-1]

    wave_obs, flux_obs = np.loadtxt(obs_name, usecols=(0,1), unpack=True)

    wave_obs = wave_obs/(1+(doppler/300000.))

    if atmosphere_type == "1D" and fit_microturb == "No":
        vturb = calculate_vturb(teff, logg, met)
    elif atmosphere_type == "1D" and fit_microturb == "Yes":
        vturb = param[1]
    elif atmosphere_type == "3D":
        vturb = 2.0

    if atmosphere_type == "1D":
        met_values = [-5, -4, -3, -2.5, -2, -1.5, -1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    elif atmosphere_type == "3D":
        met_values = [-5, -4, -3, -2.5, -2, -1.5, -1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

    for i in range(len(met_values)-1):
        if met >= met_values[i] and met < met_values[i+1]:
            met_low_bracket = met_values[i]
            met_high_bracket = met_values[i+1]

    if met > 0.5 or met < -4.0 or vturb <= 0.0:
        chi_square = 9999.9999
    elif macro < 0.0:
        chi_square = 99.9999
    elif abund < -40:
        chi_square = 999.999
    #elif nlte_flag == "True" and (abund + met + solar_abundances[abund_name[0]] < met_low_bracket + abund_low or abund + met + solar_abundances[abund_name[0]] > met_low_bracket + abund_high or abund + met + solar_abundances[abund_name[0]] < met_high_bracket + abund_low or abund + met + solar_abundances[abund_name[0]] > met_high_bracket + abund_high): #note was 5-8 previously
    #    chi_square = 999.999
    #    print("abundance range outside of values in departure file")
    #elif os_path.exists('{}/marcs_tef{:.1f}_g{:.2f}_z{:.2f}_tur{:.2f}.interpol'.format(temp_directory, teff, logg, met, vturb)):
    else:

        #turbospec_atmosphere(teff, logg, met, vturb, 0.005, 4800, 6800)

        item_abund = {}
        #for i in range(1,len(periodic_table)):
        #    item_abund[periodic_table[i]] = 0.0#solar_abundances[periodic_table[i]] #deleted this and moved to turbospectrum_class_nlte.py (was causing issues with nlte)
        item_abund["Fe"] = met
        item_abund[abund_name[0]] = abund+met

        if nlte_flag == "False":
            ts.configure(t_eff = teff, log_g = logg, metallicity = met, 
                            turbulent_velocity = vturb, lambda_delta = ldelta, lambda_min=lmin, lambda_max=lmax, 
                            free_abundances=item_abund, temp_directory = temp_directory, nlte_flag=False, verbose=False, 
                            atmosphere_dimension=atmosphere_type, windows_flag=False)#, depart_bin_file=depart_bin_file, 
                            #depart_aux_file=depart_aux_file, model_atom_file=model_atom_file)
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
            ts.configure(t_eff = teff, log_g = logg, metallicity = met, 
                            turbulent_velocity = vturb, lambda_delta = ldelta, lambda_min=lmin, lambda_max=lmax, 
                            free_abundances=item_abund, temp_directory = temp_directory, nlte_flag=True, verbose=False, 
                            atmosphere_dimension=atmosphere_type, windows_flag=False, depart_bin_file=depart_bin_file, 
                            depart_aux_file=depart_aux_file, model_atom_file=model_atom_file)

        #ts.configure(t_eff = teff, log_g = logg, metallicity = met, turbulent_velocity = vturb, lambda_delta = ldelta, lambda_min=lmin, lambda_max=lmax, free_abundances=item_abund, temp_directory = '/Users/gerber/gitprojects/SAPP/UVES_benchmark_3d_ca/', nlte_flag=False, verbose=False)

        ts.run_turbospectrum_and_atmosphere()

        #macro = macro + param[2]

        if os_path.exists('{}/spectrum_00000000.spec'.format(temp_directory)) and os.stat('{}/spectrum_00000000.spec'.format(temp_directory)).st_size != 0:
            wave_mod_orig, flux_mod_orig = np.loadtxt('{}/spectrum_00000000.spec'.format(temp_directory), usecols=(0,1), unpack=True)
            if fwhm != 0.0:
                wave_mod_conv, flux_mod_conv = conv_res(wave_mod_orig, flux_mod_orig, fwhm)
            else:
                wave_mod_conv = wave_mod_orig
                flux_mod_conv = flux_mod_orig
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
        
            if wave_mod[1] - wave_mod[0] <= wave_obs[1]-wave_obs[0]:
                flux_mod_interp = np.interp(wave_obs, wave_mod, flux_mod)
                chi_square = 0
                wave_line = wave_obs[np.where((wave_obs <= lmax-5.) & (wave_obs >= lmin+5.))]
                flux_line_obs = flux_obs[np.where((wave_obs <= lmax-5.) & (wave_obs >= lmin+5.))]
                flux_line_mod = flux_mod_interp[np.where((wave_obs <= lmax-5.) & (wave_obs >= lmin+5.))]
                for j in range(len(wave_line)):
                    chi_square += ((flux_line_obs[j]-flux_line_mod[j])*(flux_line_obs[j]-flux_line_mod[j]))/flux_line_mod[j]
            else:
                flux_obs_interp = np.interp(wave_mod, wave_obs, flux_obs)
                chi_square = 0
                wave_line = wave_mod[np.where((wave_mod <= lmax-5.) & (wave_mod >= lmin+5.))]
                flux_line_obs = flux_obs_interp[np.where((wave_mod <= lmax-5.) & (wave_mod >= lmin+5.))]
                flux_line_mod = flux_mod[np.where((wave_mod <= lmax-5.) & (wave_mod >= lmin+5.))]
                for j in range(len(wave_line)):
                    chi_square += ((flux_line_obs[j]-flux_line_mod[j])*(flux_line_obs[j]-flux_line_mod[j]))/flux_line_mod[j]

            os.system("mv {}spectrum_00000000.spec ../output_files/spectrum_{:08d}.spec".format(temp_directory, spectrum_count+1))
            out = open("../output_files/spectrum_{:08d}_convolved.spec".format(spectrum_count+1), 'w')
            #for i in range(len(wave_mod)):
            #    print("{}  {}".format(wave_mod[i], flux_mod[i]), file=out)
            for i in range(len(wave_line)):
                print("{}  {}".format(wave_line[i], flux_line_mod[i]), file=out)
            out.close()
        elif os_path.exists('{}/spectrum_00000000.spec'.format(temp_directory)) and os.stat('{}/spectrum_00000000.spec'.format(temp_directory)).st_size == 0:
            chi_square = 999.99
            print("empty spectrum file.")
        else:
            chi_square = 9999.9999
            print("didn't generate spectra")
    #else:
    #    chi_square = 9999.9999
    #    print("didn't generate atmosphere")

    print(abund, vturb, macro, doppler, chi_square)

    return(chi_square)

def chi_square_broad_met_lbl(param, obs_name, temp_directory, spectrum_count, depart_bin_file_list, depart_aux_file_list, model_atom_file_list, atmosphere_type, nlte_flag, doppler, teff, logg, fit_microturb, macro, fwhm, rot, ldelta, lmin, lmax):
    met = param[0]
    doppler = doppler + param[-1]

    wave_obs, flux_obs = np.loadtxt(obs_name, usecols=(0,1), unpack=True)

    wave_obs = wave_obs/(1+(doppler/300000.))

    if atmosphere_type == "1D" and fit_microturb == "No":
        vturb = calculate_vturb(teff, logg, met)
    elif atmosphere_type == "1D" and fit_microturb == "Yes":
        vturb = param[1]
    elif atmosphere_type == "3D":
        vturb = 2.0

    if met > 0.5 or met < -4.0 or vturb <= 0.0:
        chi_square = 9999.9999
    elif macro < 0.0:
        chi_square = 99.9999
    else:

        item_abund = {}
        #for i in range(1,len(periodic_table)):
        #    item_abund[periodic_table[i]] = 0.0#solar_abundances[periodic_table[i]] #deleted this and moved to turbospectrum_class_nlte.py (was causing issues with nlte)
        item_abund["Fe"] = met

        if nlte_flag == "False":
            ts.configure(t_eff = teff, log_g = logg, metallicity = met, 
                            turbulent_velocity = vturb, lambda_delta = ldelta, lambda_min=lmin, lambda_max=lmax, 
                            free_abundances=item_abund, temp_directory = temp_directory, nlte_flag=False, verbose=False, 
                            atmosphere_dimension=atmosphere_type, windows_flag=False)#, depart_bin_file=depart_bin_file, 
                            #depart_aux_file=depart_aux_file, model_atom_file=model_atom_file)
        elif nlte_flag == "True":
            depart_bin_file = {}
            depart_bin_file['Fe'] = depart_bin_file_list[0]
            depart_aux_file = {}
            depart_aux_file['Fe'] = depart_aux_file_list[0]
            model_atom_file = {}
            model_atom_file['Fe'] = model_atom_file_list[0]
            ts.configure(t_eff = teff, log_g = logg, metallicity = met, 
                            turbulent_velocity = vturb, lambda_delta = ldelta, lambda_min=lmin, lambda_max=lmax, 
                            free_abundances=item_abund, temp_directory = temp_directory, nlte_flag=True, verbose=False, 
                            atmosphere_dimension=atmosphere_type, windows_flag=False, depart_bin_file=depart_bin_file, 
                            depart_aux_file=depart_aux_file, model_atom_file=model_atom_file)
        
        ts.run_turbospectrum_and_atmosphere()

        if os_path.exists('{}/spectrum_00000000.spec'.format(temp_directory)) and os.stat('{}/spectrum_00000000.spec'.format(temp_directory)).st_size != 0:
            wave_mod_orig, flux_mod_orig = np.loadtxt('{}/spectrum_00000000.spec'.format(temp_directory), usecols=(0,1), unpack=True)
            if fwhm != 0.0:
                wave_mod_conv, flux_mod_conv = conv_res(wave_mod_orig, flux_mod_orig, fwhm)
            else:
                wave_mod_conv = wave_mod_orig
                flux_mod_conv = flux_mod_orig
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
        
            if wave_mod[1] - wave_mod[0] <= wave_obs[1]-wave_obs[0]:
                flux_mod_interp = np.interp(wave_obs, wave_mod, flux_mod)
                chi_square = 0
                wave_line = wave_obs[np.where((wave_obs <= lmax-5.) & (wave_obs >= lmin+5.))]
                flux_line_obs = flux_obs[np.where((wave_obs <= lmax-5.) & (wave_obs >= lmin+5.))]
                flux_line_mod = flux_mod_interp[np.where((wave_obs <= lmax-5.) & (wave_obs >= lmin+5.))]
                for j in range(len(wave_line)):
                    chi_square += ((flux_line_obs[j]-flux_line_mod[j])*(flux_line_obs[j]-flux_line_mod[j]))/flux_line_mod[j]
            else:
                flux_obs_interp = np.interp(wave_mod, wave_obs, flux_obs)
                chi_square = 0
                wave_line = wave_mod[np.where((wave_mod <= lmax-5.) & (wave_mod >= lmin+5.))]
                flux_line_obs = flux_obs_interp[np.where((wave_mod <= lmax-5.) & (wave_mod >= lmin+5.))]
                flux_line_mod = flux_mod[np.where((wave_mod <= lmax-5.) & (wave_mod >= lmin+5.))]
                for j in range(len(wave_line)):
                    chi_square += ((flux_line_obs[j]-flux_line_mod[j])*(flux_line_obs[j]-flux_line_mod[j]))/flux_line_mod[j]
        
            os.system("mv {}spectrum_00000000.spec ../output_files/spectrum_{:08d}.spec".format(temp_directory, spectrum_count+1))
            out = open("../output_files/spectrum_{:08d}_convolved.spec".format(spectrum_count+1), 'w')
            #for i in range(len(wave_mod)):
            #    print("{}  {}".format(wave_mod[i], flux_mod[i]), file=out)
            for i in range(len(wave_line)):
                print("{}  {}".format(wave_line[i], flux_line_mod[i]), file=out)
            out.close()
        elif os_path.exists('{}/spectrum_00000000.spec'.format(temp_directory)) and os.stat('{}/spectrum_00000000.spec'.format(temp_directory)).st_size == 0:
            chi_square = 999.99
            print("empty spectrum file.")
        else:
            chi_square = 9999.9999
            print("didn't generate spectra or atmosphere")

    print(met, vturb, doppler, chi_square)

    return(chi_square)

#set defaults
include_molecules = "True"

depart_bin_file = []
depart_aux_file = []
model_atom_file = []

#read the configuration file
with open("../input_files/tsfitpy_input_configuration.txt") as fp:
    line = fp.readline()
    while line:
        fields = line.strip().split()
        #if fields[0][0] == "#":
            #line = fp.readline()
        if len(fields) == 0:
            line = fp.readline()
            fields = line.strip().split()
        #if fields[0] == "turbospec_path":
        #    turbospec_path = fields[2]
        #    #line = fp.readline()
        #if fields[0] == "interpol_path":
        #    interpol_path = fields[2]
        #if fields[0] == "line_list_path":
        #    line_list_path = fields[2]
        #if fields[0] == "line_list_folder":
        #    linelist_folder = fields[2]
        #if fields[0] == "model_atmosphere_grid_path":
        #    model_atmosphere_grid_path = fields[2]
        #if fields[0] == "model_atmosphere_folder":
        #    model_atmosphere_folder = fields[2]
        #if fields[0] == "model_atmosphere_list":
        #    model_atmosphere_list = fields[2]
        #if fields[0] == "model_atom_path":
        #    model_atom_path = fields[2]
        #if fields[0] == "departure_file_path":
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
                    element.append(fields[2*(i+1)])
                element.append("Fe")
            else:
                element = []
                element.append(fields[2])
        if fields[0] == "linemask_file":
            linemask_file = fields[2]
        if fields[0] == "segment_file":
            segment_file = fields[2]
        #if fields[0] == "continuum_file":
        #    continuum_file = fields[2]
        if fields[0] == "departure_coefficient_binary" and element[0] != "Fe" and nlte_flag == "True":
            for i in range(nelement+1):
                depart_bin_file.append(fields[2*(i+1)])
        elif fields[0] == "departure_coefficient_binary" and element[0] == "Fe" and nlte_flag == "True":
            for i in range(nelement):
                depart_bin_file.append(fields[2*(i+1)])
        if fields[0] == "departure_coefficient_aux" and element[0] != "Fe" and nlte_flag == "True":
            for i in range(nelement+1):
                depart_aux_file.append(fields[2*(i+1)])
        elif fields[0] == "departure_coefficient_aux" and element[0] == "Fe" and nlte_flag == "True":
            for i in range(nelement):
                depart_aux_file.append(fields[2*(i+1)])
        if fields[0] == "model_atom_file" and element[0] != "Fe" and nlte_flag == "True":
            for i in range(nelement+1):
                model_atom_file.append(fields[2*(i+1)])
        elif fields[0] == "model_atom_file" and element[0] == "Fe" and nlte_flag == "True":
            for i in range(nelement):
                model_atom_file.append(fields[2*(i+1)])
        if fields[0] == "wavelengh_minimum":
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
            temp_directory = "../"+temp_directory
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

#set model atmosphere folder

#set directories
if ts_compiler == "intel":
    turbospec_path = "../turbospectrum/exec/"
elif ts_compiler == "gnu":
    turbospec_path = "../turbospectrum/exec-gf/"
interpol_path = "./model_interpolators/"
line_list_path_orig = "../input_files/linelists/linelist_for_fitting/"
line_list_path_trimmed = "../input_files/linelists/linelist_for_fitting_trimmed/"
if atmosphere_type == "1D":
    model_atmosphere_grid_path = "../input_files/model_atmospheres/1D/"
    model_atmosphere_list = model_atmosphere_grid_path+"model_atmosphere_list.txt"
elif atmosphere_type == "3D":
    model_atmosphere_grid_path = "../input_files/model_atmospheres/3D/"
    model_atmosphere_list = model_atmosphere_grid_path+"model_atmosphere_list.txt"
model_atom_path = "../input_files/nlte_data/model_atoms/"
departure_file_path = "../input_files/nlte_data/"

linemask_file = "../input_files/linemask_files/"+linemask_file
segment_file = "../input_files/linemask_files/"+segment_file
#continuum_file = "../input_files/linemask_files/"+continuum_file

#make an array for initial guess equal to n x ndimen+1
initial_guess = np.empty((ndimen+1,ndimen))

#fill the array with input from config file
for j in range(ndimen):
    for i in range(j,len(initial_guess_string),ndimen):
        initial_guess[int(i/ndimen)][j] = float(initial_guess_string[i])

#time_start_tot = time.time() line used for evaluating computation time

if not os.path.exists(temp_directory):
    os.makedirs(temp_directory)

param0 = initial_guess[0]

fitlist = "../input_files/"+fitlist

specname_fitlist = np.loadtxt(fitlist, dtype = 'str', usecols = (0), unpack=True)

#if fit_teff == "Yes" and fit_logg == "No" and element[0] == "Fe":

if (element[0] == "Fe" or element[0] == "fe") and fit_teff == "No" and fit_logg == "No":
    rv_fitlist, teff_fitlist, logg_fitlist = np.loadtxt(fitlist, usecols = (1,2,3), unpack=True)
else:
    rv_fitlist, teff_fitlist, logg_fitlist, met_fitlist = np.loadtxt(fitlist, usecols = (1,2,3,4), unpack=True)

output = "../output_files/"+output

f = open(output, 'a')
if fitting_mode == "all" and (element[0] == "Fe" or element[0] == "fe"):
    print("#specname        [Fe/H]     Doppler Shift (add to RV)    chi_squared", file=f)
elif fitting_mode == "all":
    print("#specname        [{}/Fe]     Doppler Shift (add to RV)    chi_squared".format(element[0]), file=f)
elif fitting_mode == "lbl" and (element[0] == "Fe" or element[0] == "fe"):
    print("#specname        wave_center  wave_start  wave_end  [Fe/H]    Microturb     Doppler Shift (add to RV)    chi_squared", file=f)
elif fitting_mode == "lbl":
    print("#specname        wave_center  wave_start  wave_end  [{}/Fe]   Microturb     Doppler Shift (add to RV)    chi_squared".format(element[0]), file=f)

seg_begins, seg_ends = np.loadtxt(segment_file, comments = ";", usecols=(0,1), unpack=True)

if fitting_mode == "all":
    print("Trimming down the linelist to only lines within segments for faster fitting")
    os.system("rm {}/*".format(line_list_path_trimmed))
    create_window_linelist(segment_file, line_list_path_orig, line_list_path_trimmed, include_molecules, 0, len(seg_ends))

print("Finished trimming linelist")

for i in range(specname_fitlist.size):
    # this next step is in case you're only fitting one star
    if specname_fitlist.size > 1 and (element[0] == "Fe" or element[0] == "fe"):
        specname = "../input_files/observed_spectra/"+specname_fitlist[i]
        teff = teff_fitlist[i]
        logg = logg_fitlist[i]
        rv = rv_fitlist[i]
    elif specname_fitlist.size > 1:
        specname = "../input_files/observed_spectra/"+specname_fitlist[i]
        teff = teff_fitlist[i]
        logg = logg_fitlist[i]
        rv = rv_fitlist[i]
        met = met_fitlist[i]
    elif specname_fitlist.size == 1 and (element[0] == "Fe" or element[0] == "fe"):
        specname = "../input_files/observed_spectra/"+np.str(specname_fitlist)
        teff = teff_fitlist
        logg = logg_fitlist
        rv = rv_fitlist
    elif specname_fitlist.size == 1:
        specname = "../input_files/observed_spectra/"+np.str(specname_fitlist)
        teff = teff_fitlist
        logg = logg_fitlist
        rv = rv_fitlist
        met = met_fitlist


    print("Fitting {}".format(specname))
    print("Teff = {}; logg = {}; RV = {}".format(teff, logg, rv))

    #time_start = time.time() used to evaluate computation time

    ts = TurboSpectrum(
            turbospec_path=turbospec_path,
            interpol_path=interpol_path,
            line_list_paths=line_list_path_trimmed,
            marcs_grid_path=model_atmosphere_grid_path,
            marcs_grid_list=model_atmosphere_list,
            model_atom_path=model_atom_path,
            departure_file_path=departure_file_path)

    if fitting_mode == "all":
        windows_flag = True

        seg_begins, seg_ends = np.loadtxt(segment_file, comments = ";", usecols=(0,1), unpack=True)

        if element[0] == "Fe" or element[0] == "fe":

            time_start = time.time()

            res = minimize(chi_square_broad_met, param0, args = (np.str(specname), temp_directory, i, linemask_file, segment_file, depart_bin_file, depart_aux_file, model_atom_file, atmosphere_type, nlte_flag, rv, teff, logg, macroturb, fwhm, rot, ldelta, lmin, lmax), method='Nelder-Mead', options={'maxiter':ndimen*50, 'disp':True, 'initial_simplex':initial_guess, 'xatol':0.05, 'fatol':0.05})

            print(res.x)

            #time_end = time.time()
            #print("Total runtime was {:.2f} minutes.".format((time_end-time_start)/60.))

            print("{} {} {} {}".format(specname.replace("../input_files/observed_spectra/",""), res.x[0], res.x[1], res.fun), file=f)

            time_end = time.time()
            print("Total runtime was {:.2f} minutes.".format((time_end-time_start)/60.))

        else:
            if atmosphere_type == "1D":
                vturb = calculate_vturb(teff, logg, met)
            elif atmosphere_type == "3D":
                vturb = 2.0

            #item_abund = {}
            #item_abund["Fe"] = met
#
            #if nlte_flag == "False":
            #    ts.configure(t_eff = teff, log_g = logg, metallicity = met, 
            #                turbulent_velocity = vturb, lambda_delta = ldelta, lambda_min=lmin, lambda_max=lmax, 
            #                free_abundances=item_abund, temp_directory = temp_directory, nlte_flag=False, verbose=False, 
            #                atmosphere_dimension=atmosphere_type, windows_flag=windows_flag, segment_file=segment_file, 
            #                line_mask_file=linemask_file, cont_mask_file=continuum_file, depart_bin_file=depart_bin_file, 
            #                depart_aux_file=depart_aux_file, model_atom_file=model_atom_file)
            #elif nlte_flag == "True":
            #    ts.configure(t_eff = teff, log_g = logg, metallicity = met, 
            #                turbulent_velocity = vturb, lambda_delta = ldelta, lambda_min=lmin, lambda_max=lmax, 
            #                free_abundances=item_abund, temp_directory = temp_directory, nlte_flag=True, verbose=False, 
            #                atmosphere_dimension=atmosphere_type, windows_flag=windows_flag, segment_file=segment_file, 
            #                line_mask_file=linemask_file, cont_mask_file=continuum_file, depart_bin_file=depart_bin_file, 
            #                depart_aux_file=depart_aux_file, model_atom_file=model_atom_file)
#
            #ts.calculate_atmosphere()

            if nlte_flag == "True":
                met_auxdata, abund_values_auxdata = np.loadtxt("{}{}".format(departure_file_path,depart_aux_file[0]), usecols=(3,7), unpack=True)

                abund_low = np.min(abund_values_auxdata[np.where(met_auxdata==0.0)])
                abund_high = np.max(abund_values_auxdata[np.where(met_auxdata==0.0)])
            else:
                abund_low = -99.
                abund_high = 99.

            #res = minimize(chi_square_broad, param0, args = (np.str(specname), temp_directory, i, linemask_file, segment_file, depart_bin_file, depart_aux_file, model_atom_file, atmosphere_type, nlte_flag, rv, teff, logg, met, vturb, macroturb, fwhm, rot, element, abund_low, abund_high, ldelta, lmin, lmax), method='Nelder-Mead', bounds=[(teff,teff),(logg,logg),(met,met),(-99.,99.),(0.0,99.)], options={'maxiter':ndimen*50,'disp':True, 'initial_simplex':initial_guess, 'xatol':0.05, 'fatol':0.05})

            res = minimize(chi_square_broad, param0, args = (np.str(specname), temp_directory, i, linemask_file, segment_file, depart_bin_file, depart_aux_file, model_atom_file, atmosphere_type, nlte_flag, rv, teff, logg, met, vturb, macroturb, fwhm, rot, element, abund_low, abund_high, ldelta, lmin, lmax), method='Nelder-Mead', options={'maxiter':ndimen*50,'disp':True, 'initial_simplex':initial_guess, 'xatol':0.05, 'fatol':0.05})
            print(res.x)

            #time_end = time.time()
            #print("Total runtime was {:.2f} minutes.".format((time_end-time_start)/60.))

            print("{} {} {} {}".format(specname.replace("../input_files/observed_spectra/",""), res.x[0], res.x[1], res.fun), file=f)
    
    elif fitting_mode == "lbl":
        windows_flag = False

        line_centers, line_begins, line_ends = np.loadtxt(linemask_file, comments = ";", usecols=(0,1,2), unpack=True)
        if line_centers.size > 1:
            line_begins_sorted = sorted(line_begins)
            line_ends_sorted = sorted(line_ends)
            line_centers_sorted = sorted(line_centers)
        elif line_centers.size == 1:
            line_begins_sorted = [line_begins]
            line_ends_sorted = [line_ends]
            line_centers_sorted = [line_centers]

        seg_begins, seg_ends = np.loadtxt(segment_file, comments = ";", usecols=(0,1), unpack=True)

        if element[0] == "Fe" or element[0] == "fe":

            for j in range(len(line_begins_sorted)):
                time_start = time.time()
                print("Fitting line at {} angstroms".format(line_centers_sorted[j]))

                for k in range(len(seg_begins)):
                    if line_centers_sorted[j] <= seg_ends[k] and line_centers_sorted[j] > seg_begins[k]:
                        start = k
                print(line_centers_sorted[j], seg_begins[start], seg_ends[start])

                os.system("rm {}*".format(line_list_path_trimmed))

                create_window_linelist(segment_file, line_list_path_orig, line_list_path_trimmed, include_molecules, start, start+1)

                res = minimize(chi_square_broad_met_lbl, param0, args = (np.str(specname), temp_directory, i, depart_bin_file, depart_aux_file, model_atom_file, atmosphere_type, nlte_flag, rv, teff, logg, fit_microturb, macroturb, fwhm, rot, ldelta, line_begins_sorted[j]-5., line_ends_sorted[j]+5.), method='Nelder-Mead', options={'maxiter':ndimen*50,'disp':True, 'initial_simplex':initial_guess, 'xatol':0.05, 'fatol':0.05})
                
                print(res.x)

                if fit_microturb == "Yes":
                    vturb = res.x[1]
                elif fit_microturb == "No":
                    vturb = calculate_vturb(teff, logg, res.x[0])

                print("{} {} {} {} {} {} {} {}".format(specname.replace("../input_files/observed_spectra/",""), line_centers_sorted[j], line_begins_sorted[j], line_ends_sorted[j], res.x[0], vturb, res.x[-1], res.fun), file=f)

                wave_result, flux_norm_result, flux_result = np.loadtxt("../output_files/spectrum_{:08d}.spec".format(i+1), unpack=True)
                g = open("../output_files/result_spectrum_{:08d}.spec".format(i), 'a')
                for k in range(len(wave_result)):
                    print("{}  {}  {}".format(wave_result[k], flux_norm_result[k], flux_result[k]), file=g)
                os.system("rm ../output_files/spectrum_{:08d}.spec".format(i+1))

                wave_result, flux_norm_result = np.loadtxt("../output_files/spectrum_{:08d}_convolved.spec".format(i+1), unpack=True)
                h = open("../output_files/result_spectrum_{:08d}_convolved.spec".format(i), 'a')
                for k in range(len(wave_result)):
                    print("{}  {}".format(wave_result[k], flux_norm_result[k]), file=h)
                os.system("rm ../output_files/spectrum_{:08d}_convolved.spec".format(i+1))

                #os.system("rm {}*".format(line_list_path_trimmed))

                time_end = time.time()
                print("Total runtime was {:.2f} minutes.".format((time_end-time_start)/60.))

            g.close()
            h.close()

        else:

            if nlte_flag == "True":
                met_auxdata, abund_values_auxdata = np.loadtxt("{}{}".format(departure_file_path,depart_aux_file[0]), usecols=(3,7), unpack=True)

                abund_low = np.min(abund_values_auxdata[np.where(met_auxdata==0.0)])
                abund_high = np.max(abund_values_auxdata[np.where(met_auxdata==0.0)])
            else:
                abund_low = -99.
                abund_high = 99.

            for j in range(len(line_begins_sorted)):
                time_start = time.time()
                print("Fitting line at {} angstroms".format(line_centers_sorted[j]))

                for k in range(len(seg_begins)):
                    if line_centers_sorted[j] <= seg_ends[k] and line_centers_sorted[j] > seg_begins[k]:
                        start = k
                print(line_centers_sorted[j], seg_begins[start], seg_ends[start])

                os.system("rm {}/*".format(line_list_path_trimmed))

                create_window_linelist(segment_file, line_list_path_orig, line_list_path_trimmed, include_molecules, start, start+1)

                res = minimize(chi_square_broad_lbl, param0, args = (np.str(specname), temp_directory, i, depart_bin_file, depart_aux_file, model_atom_file, atmosphere_type, nlte_flag, rv, teff, logg, met, fit_microturb, macroturb, fwhm, rot, element, abund_low, abund_high, ldelta, line_begins_sorted[j]-5., line_ends_sorted[j]+5.), method='Nelder-Mead', options={'maxiter':ndimen*50,'disp':True, 'initial_simplex':initial_guess, 'xatol':0.05, 'fatol':0.05})

                print(res.x)

                if fit_microturb == "Yes":
                    vturb = res.x[1]
                elif fit_microturb == "No":
                    vturb = calculate_vturb(teff, logg, met)

                print("{} {} {} {} {} {} {} {}".format(specname.replace("../input_files/observed_spectra/",""), line_centers_sorted[j], line_begins_sorted[j], line_ends_sorted[j], res.x[0], vturb, res.x[-1], res.fun), file=f)

                wave_result, flux_norm_result, flux_result = np.loadtxt("../output_files/spectrum_{:08d}.spec".format(i+1), unpack=True)
                g = open("../output_files/result_spectrum_{:08d}.spec".format(i), 'a')
                for k in range(len(wave_result)):
                    print("{}  {}  {}".format(wave_result[k], flux_norm_result[k], flux_result[k]), file=g)
                os.system("rm ../output_files/spectrum_{:08d}.spec".format(i+1))

                wave_result, flux_norm_result = np.loadtxt("../output_files/spectrum_{:08d}_convolved.spec".format(i+1), unpack=True)
                h = open("../output_files/result_spectrum_{:08d}_convolved.spec".format(i), 'a')
                for k in range(len(wave_result)):
                    print("{}  {}".format(wave_result[k], flux_norm_result[k]), file=h)
                os.system("rm ../output_files/spectrum_{:08d}_convolved.spec".format(i+1))

                os.system("rm {}/*".format(line_list_path_trimmed))

                time_end = time.time()
                print("Total runtime was {:.2f} minutes.".format((time_end-time_start)/60.))

            g.close()
            h.close()

        #if nlte_flag == "False":
        #    ts.configure(t_eff = teff, log_g = logg, metallicity = met, turbulent_velocity = vturb, lambda_delta = ldelta, lambda_min=lmin, lambda_max=lmax, free_abundances=item_abund, temp_directory = temp_directory, nlte_flag=False, verbose=False, atmosphere_dimension=atmosphere_type, windows_flag=windows_flag)
        #elif nlte_flag == "True":
        #    ts.configure(t_eff = teff, log_g = logg, metallicity = met, turbulent_velocity = vturb, lambda_delta = ldelta, lambda_min=lmin, lambda_max=lmax, free_abundances=item_abund, temp_directory = temp_directory, nlte_flag=False, verbose=False, atmosphere_dimension=atmosphere_type, windows_flag=windows_flag, depart_bin_file=depart_bin_file, depart_aux_file=depart_aux_file, model_atom_file=model_atom_file)

f.close()



#time_end_tot = time.time()

#print("Total runtime was {:.2f} minutes ({:.2f} hours).".format((time_end_tot-time_start_tot)/60., (time_end_tot-time_start_tot)/3600.))
