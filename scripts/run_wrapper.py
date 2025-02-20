from __future__ import annotations

import pickle
from .turbospectrum_class_nlte import TurboSpectrum
from .m3dis_class import M3disCall
from .convolve import *
import datetime
import shutil
import os
from .solar_abundances import solar_abundances
from .auxiliary_functions import calculate_vturb, import_module_from_path


def run_wrapper(ts_config, spectrum_name, teff, logg, met, lmin, lmax, ldelta, nlte_flag, abundances_dict, resolution=0,
                macro=0, rotation=0, vmic=None, verbose=False, lpoint_turbospectrum=500_000, **kwargs):
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
        abundances_dict_xh[key] = value + met

    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)

    if not ts_config["m3dis_flag"]:
        ssg = TurboSpectrum(
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
    else:
        ssg = M3disCall(
            m3dis_path=ts_config["turbospec_path"],
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
            marcs_values=ts_config["marcs_values"],
            m3dis_python_module=import_module_from_path("m3dis.m3dis", ts_config["m3dis_python_module"]),
            n_nu=ts_config["m3dis_n_nu"],
            hash_table_size=ts_config["m3dis_hash_table_size"],
            mpi_cores=ts_config["m3dis_mpi_cores"],
            iterations_max=ts_config["m3dis_iterations_max"],
            convlim=ts_config["m3dis_convlim"],
            snap=ts_config["m3dis_snap"],
            dims=ts_config["m3dis_dims"],
            nx=ts_config["m3dis_nx"],
            ny=ts_config["m3dis_ny"],
            nz=ts_config["m3dis_nz"])
        ssg.use_precomputed_depart = False

        # TODO: support for intensity in m3dis
        ts_config["compute_intensity_flag"] = False

    ssg.configure(t_eff=teff, log_g=logg, metallicity=met,
                  turbulent_velocity=vmic, lambda_delta=ldelta, lambda_min=lmin, lambda_max=lmax,
                  free_abundances=abundances_dict_xh, temp_directory=temp_directory, nlte_flag=nlte_flag,
                  verbose=verbose,
                  atmosphere_dimension=ts_config["atmosphere_type"],
                  windows_flag=ts_config["windows_flag"],
                  segment_file=ts_config["segment_file"],
                  line_mask_file=ts_config["line_mask_file"],
                  depart_bin_file=ts_config["depart_bin_file"],
                  depart_aux_file=ts_config["depart_aux_file"],
                  model_atom_file=ts_config["model_atom_file"])

    ssg.compute_intensity_flag = ts_config["compute_intensity_flag"]
    ssg.mupoint_path = ts_config["mupoint_path"]

    ssg.lpoint = lpoint_turbospectrum

    results = ssg.synthesize_spectra()

    shutil.rmtree(temp_directory)

    if ts_config["compute_intensity_flag"]:
        wavelength, intensities = results
        return wavelength, intensities, []
    else:
        wave_mod_orig, flux_norm_mod_orig, flux_mod_orig = results


    try:
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

    return wave_mod, flux_norm_mod, flux_mod


def run_and_save_wrapper(tsfitpy_pickled_configuration_path, teff, logg, met, lmin, lmax, ldelta, spectrum_name, nlte_flag, resolution, macro, rotation, new_directory_to_save_to, vturb, abundances_dict: dict, save_unnormalised_spectra, verbose, lpoint_turbospectrum):
    with open(tsfitpy_pickled_configuration_path, 'rb') as f:
        ts_config = pickle.load(f)

    wave_mod, flux_norm_mod, flux_mod = run_wrapper(ts_config, spectrum_name, teff, logg, met, lmin, lmax, ldelta, nlte_flag, abundances_dict, resolution, macro, rotation, vturb, verbose=verbose, lpoint_turbospectrum=lpoint_turbospectrum)
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
        if vturb is None:
            vturb = calculate_vturb(teff, logg, met)
        print("#vmic: {}".format(vturb), file=f)
        print("#vmac: {}".format(macro), file=f)
        print("#resolution: {}".format(resolution), file=f)
        print("#rotation: {}".format(rotation), file=f)
        print("#nlte_flag: {}".format(nlte_flag), file=f)

        # get which elements are in NLTE using ts_config["model_atom_file"]
        nlte_elements = ts_config["model_atom_file"].keys()

        for element, value in abundances_dict.items():
            if nlte_flag:
                if element in nlte_elements:
                    nlte_flag_label = "NLTE"
                else:
                    nlte_flag_label = "LTE"
            else:
                nlte_flag_label = "LTE"
            if element != "Fe":
                print(f"#[{element}/Fe]={value:.4f} {nlte_flag_label}", file=f)
            else:
                # if Fe, it is given as weird Fe/Fe way, which can be fixed back by:
                # xfe + feh + A(X)_sun = A(X)_star
                print(f"#A({element})={value + met + solar_abundances['Fe']:.4f} {nlte_flag_label}", file=f)

        # also print NLTE elements that are not in the abundances_dict
        if nlte_flag:
            for element in nlte_elements:
                if element not in abundances_dict.keys():
                    if element != "Fe" or element != "H":
                        print(f"#[{element}/Fe]=0.0 (solar scaled) NLTE", file=f)
                    elif element == "H":
                        print(f"#A({element})=12.0 NLTE", file=f)
                    elif element == "Fe":
                        print(f"#[Fe/H]={met:.4f} NLTE", file=f)

        print("#", file=f)

        if save_unnormalised_spectra and not ts_config["compute_intensity_flag"]:
            print("#Wavelength Normalised_flux Unnormalised_flux", file=f)
            for i in range(len(wave_mod)):
                print("{}  {}  {}".format(wave_mod[i], flux_norm_mod[i], flux_mod[i]), file=f)
        elif not ts_config["compute_intensity_flag"]:
            print("#Wavelength Normalised_flux", file=f)
            for i in range(len(wave_mod)):
                print("{}  {}".format(wave_mod[i], flux_norm_mod[i]), file=f)
        else:
            print("#Wavelength Intensities", file=f)
            # Example: flux_norm_mod is 1D
            if flux_norm_mod.ndim == 1:
                for i in range(len(wave_mod)):
                    print(f"{wave_mod[i]}  {flux_norm_mod[i]}", file=f)
            # Example: flux_norm_mod is 2D (multiple columns)
            elif flux_norm_mod.ndim == 2:
                for i in range(len(wave_mod)):
                    # Convert the row of flux values into strings, separated by spaces
                    flux_str = "  ".join(str(val) for val in flux_norm_mod[i])
                    print(f"{wave_mod[i]}  {flux_str}", file=f)
        f.close()
        return spectrum_name
    else:
        return ""
