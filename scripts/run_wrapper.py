from __future__ import annotations

import pickle
from scripts.turbospectrum_class_nlte import TurboSpectrum
from scripts.convolve import *
import datetime
import shutil
import os
from scripts.solar_abundances import solar_abundances
from scripts.auxiliary_functions import calculate_vturb


def run_wrapper(ts_config, spectrum_name, teff, logg, met, lmin, lmax, ldelta, nlte_flag, abundances_dict, resolution=0, macro=0, rotation=0, vmic=None, verbose=False, **kwargs):
    #parameters to adjust

    teff = teff
    logg = logg
    met = met
    if vmic is None:
        vmic = calculate_vturb(teff, logg, met)
    #print(vturb)
    #vturb = 0.9
    lmin = lmin
    lmax = lmax
    ldelta = ldelta
    temp_directory = os.path.join(ts_config["global_temporary_directory"],
                                  f"temp_directory_{datetime.datetime.now().strftime('%b-%d-%Y-%H-%M-%S')}__{np.random.random(1)[0]}", "")

    abundances_dict_xh = abundances_dict.copy()

    # need to convert abundances_dict from X/Fe to X/H
    for key, value in abundances_dict_xh.items():
        if key != "Fe":
            abundances_dict_xh[key] = value + met

    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)

    ts = TurboSpectrum(
                turbospec_path=ts_config["turbospec_path"],
                interpol_path=ts_config["interpol_path"],
                line_list_paths=ts_config["line_list_paths"],
                marcs_grid_path=ts_config["model_atmosphere_grid_path"],
                marcs_grid_list=ts_config["model_atmosphere_grid_list"],
                model_atom_path=ts_config["model_atom_path"],
                departure_file_path=ts_config["departure_file_path"],
                aux_file_length_dict=ts_config["aux_file_length_dict"],
                model_temperatures=ts_config["model_temperatures"],
                model_logs=ts_config["model_logs"],
                model_mets=ts_config["model_mets"],
                marcs_value_keys=ts_config["marcs_value_keys"],
                marcs_models=ts_config["marcs_models"],
                marcs_values=ts_config["marcs_values"],)

    ts.configure(t_eff = teff, log_g = logg, metallicity = met,
                 turbulent_velocity = vmic, lambda_delta = ldelta, lambda_min=lmin, lambda_max=lmax,
                 free_abundances=abundances_dict_xh, temp_directory = temp_directory, nlte_flag=nlte_flag, verbose=verbose,
                 atmosphere_dimension=ts_config["atmosphere_type"],
                 windows_flag=ts_config["windows_flag"],
                 segment_file=ts_config["segment_file"],
                 line_mask_file=ts_config["line_mask_file"],
                 depart_bin_file=ts_config["depart_bin_file"],
                 depart_aux_file=ts_config["depart_aux_file"],
                 model_atom_file=ts_config["model_atom_file"])

    ts.run_turbospectrum_and_atmosphere()

    try:
        wave_mod_orig, flux_norm_mod_orig, flux_mod_orig = np.loadtxt(os.path.join(temp_directory, 'spectrum_00000000.spec'), usecols=(0,1,2), unpack=True)
        if ts_config["windows_flag"]:
            seg_begins, seg_ends = np.loadtxt(ts_config['segment_file'], comments = ";", usecols=(0,1), unpack=True)
            wave_mod_filled = []
            flux_norm_mod_filled = []
            flux_mod_filled = []
            for i in range(len(seg_begins)):
                j = 0
                while wave_mod_orig[j] < seg_begins[i]:
                    j+=1
                while wave_mod_orig[j] >= seg_begins[i] and wave_mod_orig[j] <= seg_ends[i]:
                    wave_mod_filled.append(wave_mod_orig[j])
                    flux_norm_mod_filled.append(flux_norm_mod_orig[j])
                    flux_mod_filled.append(flux_mod_orig[j])
                    j+=1
                if i < len(seg_begins)-1:
                    k = 1
                    while (seg_begins[i+1] - 0.001 > seg_ends[i]+k*0.005):
                        wave_mod_filled.append(seg_ends[i]+0.005*k)
                        flux_norm_mod_filled.append(1.0)
                        flux_mod_filled.append(np.mean(flux_mod_orig))
                        k+=1
        else:
            wave_mod_filled = wave_mod_orig
            flux_norm_mod_filled = flux_norm_mod_orig
            flux_mod_filled = flux_mod_orig

        if resolution != 0.0:
            wave_mod_conv, flux_norm_mod_conv = conv_res(wave_mod_filled, flux_norm_mod_filled, resolution)
            wave_mod_conv, flux_mod_conv = conv_res(wave_mod_filled, flux_mod_filled, resolution)
        else:
            wave_mod_conv = wave_mod_filled
            flux_norm_mod_conv = flux_norm_mod_filled
            flux_mod_conv = flux_mod_filled

        if macro != 0.0:
            wave_mod_macro, flux_norm_mod_macro = conv_macroturbulence(wave_mod_conv, flux_norm_mod_conv, macro)
            wave_mod_macro, flux_mod_macro = conv_macroturbulence(wave_mod_conv, flux_mod_conv, macro)
        else:
            wave_mod_macro = wave_mod_conv
            flux_norm_mod_macro = flux_norm_mod_conv
            flux_mod_macro = flux_mod_conv

        if rotation != 0.0:
            wave_mod, flux_norm_mod = conv_rotation(wave_mod_macro, flux_norm_mod_macro, rotation)
            wave_mod, flux_mod = conv_rotation(wave_mod_macro, flux_mod_macro, rotation)
        else:
            wave_mod = wave_mod_macro
            flux_norm_mod = flux_norm_mod_macro
            flux_mod = flux_mod_macro
    except (FileNotFoundError, OSError, ValueError) as err:
        print(f"FileNotFoundError: {spectrum_name}. Failed to generate spectrum. Error: {err}")
        wave_mod, flux_norm_mod, flux_mod = [], [], []

    shutil.rmtree(temp_directory)

    return wave_mod, flux_norm_mod, flux_mod


def run_and_save_wrapper(tsfitpy_pickled_configuration_path, teff, logg, met, lmin, lmax, ldelta, spectrum_name, nlte_flag, resolution, macro, rotation, new_directory_to_save_to, vturb, abundances_dict: dict, save_unnormalised_spectra, verbose):
    with open(tsfitpy_pickled_configuration_path, 'rb') as f:
        ts_config = pickle.load(f)

    wave_mod, flux_norm_mod, flux_mod = run_wrapper(ts_config, spectrum_name, teff, logg, met, lmin, lmax, ldelta, nlte_flag, abundances_dict, resolution, macro, rotation, vturb, verbose=verbose)
    file_location_output = os.path.join(new_directory_to_save_to, f"{spectrum_name}")
    if len(wave_mod) > 0:
        f = open(file_location_output, 'w')

        # save the parameters used to generate the spectrum
        # print when spectra was generated
        print("#Generated using TurboSpectrum and TSFitPy wrapper", file=f)
        print("#date: {}".format(datetime.datetime.now()), file=f)
        print("#spectrum_name: {}".format(spectrum_name), file=f)
        print("#teff: {}".format(teff), file=f)
        print("#logg: {}".format(logg), file=f)
        print("#[Fe/H]: {}".format(met), file=f)
        print("#vmic: {}".format(vturb), file=f)
        print("#vmac: {}".format(macro), file=f)
        print("#resolution: {}".format(resolution), file=f)
        print("#rotation: {}".format(rotation), file=f)
        print("#nlte_flag: {}".format(nlte_flag), file=f)
        for key, value in abundances_dict.items():
            if key != "Fe":
                print(f"#[{key}/Fe]={value}", file=f)
            else:
                # if Fe, it is given as weird Fe/Fe way, which can be fixed back by:
                # xfe + feh + A(X)_sun = A(X)_star
                print(f"#A({key})={value + met + solar_abundances['Fe']}", file=f)
        print("#", file=f)

        if save_unnormalised_spectra:
            print("#Wavelength Normalised_flux Unnormalised_flux", file=f)
            for i in range(len(wave_mod)):
                print("{}  {}  {}".format(wave_mod[i], flux_norm_mod[i], flux_mod[i]), file=f)
        else:
            print("#Wavelength Normalised_flux", file=f)
            for i in range(len(wave_mod)):
                print("{}  {}".format(wave_mod[i], flux_norm_mod[i]), file=f)
        f.close()
