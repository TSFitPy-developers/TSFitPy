from __future__ import annotations
import collections
import datetime
import os
import shutil
import socket
import time
from os import path as os_path
import glob
from sys import argv
from dask.distributed import Client
import numpy as np
from scipy.optimize import minimize

from TSFitPy import Spectra, load_nlte_files_in_dict, create_dir, calculate_equivalent_width
from turbospectrum_class_nlte import TurboSpectrum, fetch_marcs_grid


def cut_linelist(seg_begins: list[float], seg_ends: list[float], line_list_file: str, new_path_name: str, elements_to_use: list[str]):
    segment_to_use_begins: np.ndarray = np.asarray(seg_begins)
    segment_to_use_ends: np.ndarray = np.asarray(seg_ends)

    segment_index_order: np.ndarray = np.argsort(segment_to_use_begins)
    segment_to_use_begins: np.ndarray = segment_to_use_begins[segment_index_order]
    segment_to_use_ends: np.ndarray = segment_to_use_ends[segment_index_order]
    segment_min_wavelength: float = np.min(segment_to_use_begins)
    segment_max_wavelength: float = np.max(segment_to_use_ends)

    for i in range(len(seg_begins)):
        new_path_name_one_seg: str = os.path.join(f"{new_path_name}", f"{i}", '')
        if not os.path.exists(new_path_name_one_seg):
            os.makedirs(new_path_name_one_seg)

    line_list_number: int = 0       # just for saving as linelist_0.bsyn
    new_linelist_name: str = f"{new_path_name}" #f"linelist-{line_list_number}.bsyn"
    #new_linelist: str = os_path.join(f"{new_path_name}", f"linelist-{i}.bsyn")
    #with open(new_linelist, "w") as new_file_to_write:
    with open(line_list_file) as fp:
        lines_file: list[str] = fp.readlines()
        all_lines_to_write: dict = {}
        line_number_read_for_element: int = 0
        line_number_read_file: int = 0
        total_lines_in_file: int = len(lines_file)
        while line_number_read_file < total_lines_in_file:		# go through all line
            line: str = lines_file[line_number_read_file]
            fields: list[str] = line.strip().split()

            # it means this is an element
            if all_lines_to_write:  # if there was an element before with segments, then write them first
                write_lines(all_lines_to_write, elem_line_1_to_save, elem_line_2_to_save, new_linelist_name,
                            line_list_number)
                all_lines_to_write: dict = {}
            element_name = f"{fields[0]}{fields[1]}"
            if len(fields[0]) > 1:		# save the first two lines of an element for the future
                elem_line_1_to_save: str = f"{fields[0]} {fields[1]}  {fields[2]}"  # first line of the element
                number_of_lines_element: int = int(fields[3])
            else:
                elem_line_1_to_save: str = f"{fields[0]}   {fields[1]}            {fields[2]}    {fields[3]}"
                number_of_lines_element: int = int(fields[4])
            line_number_read_file += 1
            line: str = lines_file[line_number_read_file]
            elem_line_2_to_save: str = f"{line.strip()}\n"  # second line of the element

            # now we are reading the element's wavelength and stuff
            line_number_read_file += 1
            # lines_for_element = lines_file[line_number_read_file:number_of_lines_element+line_number_read_file]

            # to not redo strip/split every time, save wavelength for the future here
            element_wavelength_dictionary = {}

            # wavelength minimum and maximum for the element (assume sorted)
            wavelength_minimum_element: float = float(lines_file[line_number_read_file].strip().split()[0])
            wavelength_maximum_element: float = float(lines_file[number_of_lines_element+line_number_read_file - 1].strip().split()[0])

            element_wavelength_dictionary[0] = wavelength_minimum_element
            element_wavelength_dictionary[number_of_lines_element - 1] = wavelength_maximum_element
            if element_name.replace("'", "") in elements_to_use:
                # check that ANY wavelengths are within the range at all
                if not (wavelength_maximum_element < segment_min_wavelength or wavelength_minimum_element > segment_max_wavelength):
                    for seg_index, (seg_begin, seg_end) in enumerate(zip(segment_to_use_begins, segment_to_use_ends)):  # wavelength lines write here
                        index_seg_start, element_wavelength_dictionary = binary_search_lower_bound(lines_file[line_number_read_file:number_of_lines_element + line_number_read_file],
                                                                                                   element_wavelength_dictionary, 0, number_of_lines_element - 1, seg_begin)
                        wavelength_current_line: float = element_wavelength_dictionary[index_seg_start]
                        line_stripped: str = lines_file[line_number_read_file + index_seg_start].strip()
                        line_number_read_for_element: int = index_seg_start + line_number_read_file
                        while wavelength_current_line <= seg_end and line_number_read_for_element < number_of_lines_element + line_number_read_file:
                            seg_current_index = seg_index
                            if seg_current_index not in all_lines_to_write:
                                all_lines_to_write[seg_current_index] = [f"{line_stripped} \n"]
                            else:
                                all_lines_to_write[seg_current_index].append(f"{line_stripped} \n")
                            line_number_read_for_element += 1
                            try:
                                line_stripped: str = lines_file[line_number_read_for_element].strip()
                                wavelength_current_line: float = float(line_stripped.split()[0])
                            except (ValueError, IndexError):
                                pass

            line_number_read_file: int = number_of_lines_element + line_number_read_file

        if len(all_lines_to_write) > 0:
            write_lines(all_lines_to_write, elem_line_1_to_save, elem_line_2_to_save, new_linelist_name, line_list_number)

def binary_search_lower_bound(array_to_search: list[str], dict_array_values: dict, low: int, high: int, element_to_search: float) -> tuple[int, dict]:
    """
    Gives out the lower index where the value is located between the ranges. For example, given array [12, 20, 32, 40, 52]
    Value search: 5, result: 0
    Value search: 13, result: 1
    Value search: 20, result: 1
    Value search: 21, result: 2
    Value search: 51 or 52 or 53, result: 4
    :param array_to_search:
    :param dict_array_values:
    :param low:
    :param high:
    :param element_to_search:
    :return:
    """
    while low < high:
        middle: int = low + (high - low) // 2

        if middle not in dict_array_values:
            dict_array_values[middle] = float(array_to_search[middle].strip().split()[0])
        array_element_value: float = dict_array_values[middle]

        if array_element_value < element_to_search:
            low: int = middle + 1
        else:
            high: int = middle
    return low, dict_array_values

def write_lines(all_lines_to_write: dict[list[str]], elem_line_1_to_save: str, elem_line_2_to_save: str, new_path_name: str, line_list_number: float):
    for key in all_lines_to_write:
        new_linelist_name: str = os.path.join(f"{new_path_name}", f"{key}", f"linelist-{line_list_number}.bsyn")
        with open(new_linelist_name, "a") as new_file_to_write:
            new_file_to_write.write(f"{elem_line_1_to_save}	{len(all_lines_to_write[key])}\n")
            new_file_to_write.write(f"{elem_line_2_to_save}")
            for line_to_write in all_lines_to_write[key]:
                #pass
                new_file_to_write.write(line_to_write)


def generate_lte_atmosphere(teff, logg, vturb, met, lmin, lmax, ldelta, line_list_path, element, abundance, nlte_flag):
    #parameters to adjust

    teff = teff
    logg = logg
    met = met
    lmin = lmin
    lmax = lmax
    ldelta = ldelta
    item_abund = {"Fe": met, element: abundance + met}
    temp_directory = f"../temp_directory_{datetime.datetime.now().strftime('%b-%d-%Y-%H-%M-%S')}__{np.random.random(1)[0]}/"

    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)

    ts = TurboSpectrum(
                turbospec_path=Spectra.turbospec_path,
                interpol_path=Spectra.interpol_path,
                line_list_paths=line_list_path,
                marcs_grid_path=Spectra.model_atmosphere_grid_path,
                marcs_grid_list=Spectra.model_atmosphere_list,
                model_atom_path=Spectra.model_atom_path,
                departure_file_path=Spectra.departure_file_path,
                aux_file_length_dict=Spectra.aux_file_length_dict,
                model_temperatures=Spectra.model_temperatures,
                model_logs=Spectra.model_logs,
                model_mets=Spectra.model_mets,
                marcs_value_keys=Spectra.marcs_value_keys,
                marcs_models=Spectra.marcs_models,
                marcs_values=Spectra.marcs_values)

    ts.configure(t_eff = teff, log_g = logg, metallicity = met,
                                turbulent_velocity = vturb, lambda_delta = ldelta, lambda_min=lmin, lambda_max=lmax,
                                free_abundances=item_abund, temp_directory = temp_directory, nlte_flag=nlte_flag, verbose=False,
                                atmosphere_dimension=Spectra.atmosphere_type, windows_flag=False, segment_file=Spectra.segment_file,
                                line_mask_file=Spectra.linemask_file, depart_bin_file=Spectra.depart_bin_file_dict,
                                depart_aux_file=Spectra.depart_aux_file_dict, model_atom_file=Spectra.model_atom_file_dict)

    ts.calculate_atmosphere()
    ts.run_turbospectrum()

    wave_mod_orig, flux_norm_mod_orig, flux_mod_orig = np.loadtxt('{}spectrum_00000000.spec'.format(temp_directory), usecols=(0,1,2), unpack=True)

    shutil.rmtree(temp_directory)

    return wave_mod_orig, flux_norm_mod_orig

def get_nlte_ew(teff, logg, microturb, met, lmin, lmax, ldelta, line_list_path, element, abundance, lte_ew):
    wavelength_nlte, norm_flux_nlte = generate_lte_atmosphere(teff, logg, microturb, met, lmin, lmax, ldelta, line_list_path, element, abundance, True)
    nlte_ew = calculate_equivalent_width(wavelength_nlte, norm_flux_nlte, lmin, lmax)
    return np.square(nlte_ew - lte_ew)

def generate_and_fit_atmosphere(specname, teff, logg, microturb, met, lmin, lmax, ldelta, line_list_path, element, abundance):
    wavelength_lte, norm_flux_lte = generate_lte_atmosphere(teff, logg, microturb, met, lmin, lmax, ldelta, line_list_path, element, abundance, False)
    ew_lte = calculate_equivalent_width(wavelength_lte, norm_flux_lte, lmin, lmax)
    result = minimize(get_nlte_ew, abundance,
                      args=(teff, logg, microturb, met, lmin, lmax, ldelta, line_list_path, element, abundance, True, ew_lte),
                      bounds=(abundance - 3, abundance + 3), method="L-BFGS-B", options={'disp': False})

    nlte_correction = result.x
    ew_nlte = result.fun

    return [specname, ew_lte, ew_nlte, nlte_correction]


def run_nlte_corrections(config_file_name, output_folder_title):
    depart_bin_file = []
    depart_aux_file = []
    model_atom_file = []
    init_guess_elements = []
    input_elem_abundance = []
    depart_bin_file_input_elem = []
    depart_aux_file_input_elem = []
    model_atom_file_input_elem = []

    # read the configuration file
    with open(config_file_name) as fp:
        line = fp.readline()
        while line:
            if len(line) > 1:
                fields = line.strip().split()
                # if fields[0][0] == "#":
                # line = fp.readline()
                # if fields[0] == "turbospec_path":
                #    turbospec_path = fields[2]
                field_name = fields[0].lower()
                if field_name == "title":
                    output_folder_title = fields[2]
                if field_name == "interpol_path":
                    interpol_path = fields[2]
                if field_name == "line_list_path":
                    line_list_path = fields[2]
                # if fields[0] == "line_list_folder":
                #    linelist_folder = fields[2]
                if field_name == "model_atmosphere_grid_path_1d":
                    model_atmosphere_grid_path_1D = fields[2]
                if field_name == "model_atmosphere_grid_path_3d":
                    model_atmosphere_grid_path_3D = fields[2]
                # if fields[0] == "model_atmosphere_folder":
                #    model_atmosphere_folder = fields[2]
                # if fields[0] == "model_atmosphere_list":
                #    model_atmosphere_list = fields[2]
                if field_name == "model_atom_path":
                    model_atom_path = fields[2]
                if field_name == "departure_file_path":
                    departure_file_path = fields[2]
                if field_name == "output_folder":
                    output_folder_og = fields[2]
                if field_name == "linemask_file_folder_location":
                    linemask_file_og = fields[2]
                if field_name == "segment_file_folder_location":
                    segment_file_og = fields[2]
                if field_name == "spec_input_path":
                    spec_input_path = fields[2]
                if field_name == "fitlist_input_folder":
                    fitlist_input_folder = fields[2]
                if field_name == "turbospectrum_compiler":
                    ts_compiler = fields[2]
                if field_name == "atmosphere_type":
                    Spectra.atmosphere_type = fields[2]
                if field_name == "mode":
                    Spectra.fitting_mode = fields[2].lower()
                if field_name == "include_molecules":
                    Spectra.include_molecules = fields[2]
                if field_name == "nlte":
                    nlte_flag = fields[2].lower()
                    if nlte_flag == "true":
                        Spectra.nlte_flag = True
                    else:
                        Spectra.nlte_flag = False
                if field_name == "fit_microturb":  # Yes No Input
                    Spectra.fit_microturb = fields[2]
                if field_name == "fit_macroturb":  # Yes No Input
                    if fields[2].lower() == "yes":
                        Spectra.fit_macroturb = True
                    else:
                        Spectra.fit_macroturb = False
                    if fields[2].lower() == "input":
                        input_macro = True
                    else:
                        input_macro = False
                if field_name == "fit_rotation":
                    if fields[2].lower() == "yes":
                        Spectra.fit_rotation = True
                    else:
                        Spectra.fit_rotation = False
                """if fields[0] == "fit_teff":
                    if fields[2].lower() == "true":
                        Spectra.fit_teff = True
                    else:
                        Spectra.fit_teff = False
                if fields[0] == "fit_logg":
                    Spectra.fit_logg = fields[2]"""
                if field_name == "element":
                    elements_to_fit = []
                    for i in range(len(fields) - 2):
                        elements_to_fit.append(fields[2 + i])
                    Spectra.elem_to_fit = np.asarray(elements_to_fit)
                    if "Fe" in elements_to_fit:
                        Spectra.fit_met = True
                    else:
                        Spectra.fit_met = False
                    Spectra.nelement = len(Spectra.elem_to_fit)
                if field_name == "linemask_file":
                    linemask_file = fields[2]
                if field_name == "segment_file":
                    segment_file = fields[2]
                # if fields[0] == "continuum_file":
                #    continuum_file = fields[2]
                if field_name == "departure_coefficient_binary" and Spectra.nlte_flag:
                    for i in range(2, len(fields)):
                        depart_bin_file.append(fields[i])
                if field_name == "departure_coefficient_aux" and Spectra.nlte_flag:
                    for i in range(2, len(fields)):
                        depart_aux_file.append(fields[i])
                if field_name == "model_atom_file" and Spectra.nlte_flag:
                    for i in range(2, len(fields)):
                        model_atom_file.append(fields[i])
                if field_name == "input_elem_departure_coefficient_binary" and Spectra.nlte_flag:
                    for i in range(2, len(fields)):
                        depart_bin_file_input_elem.append(fields[i])
                if field_name == "input_elem_departure_coefficient_aux" and Spectra.nlte_flag:
                    for i in range(2, len(fields)):
                        depart_aux_file_input_elem.append(fields[i])
                if field_name == "input_elem_model_atom_file" and Spectra.nlte_flag:
                    for i in range(2, len(fields)):
                        model_atom_file_input_elem.append(fields[i])
                if field_name == "wavelength_minimum":
                    Spectra.lmin = float(fields[2])
                if field_name == "wavelength_maximum":
                    Spectra.lmax = float(fields[2])
                if field_name == "wavelength_delta":
                    Spectra.ldelta = float(fields[2])
                if field_name == "resolution":
                    Spectra.resolution = float(fields[2])
                if field_name == "macroturbulence":
                    macroturb_input = float(fields[2])
                if field_name == "rotation":
                    Spectra.rotation = float(fields[2])
                if field_name == "temporary_directory":
                    temp_directory = fields[2]
                    temp_directory = os.path.join(temp_directory, output_folder_title, '')
                    Spectra.global_temp_dir = f"../{temp_directory}"
                if field_name == "input_file":
                    fitlist = fields[2]
                if field_name == "output_file":
                    output = fields[2]
                if field_name == "workers":
                    workers = int(fields[2])  # should be the same as cores; use value of 1 if you do not want to use multithprocessing
                    Spectra.dask_workers = workers
                if field_name == "init_guess_elem":
                    init_guess_elements = []
                    for i in range(len(fields) - 2):
                        init_guess_elements.append(fields[2 + i])
                    init_guess_elements = np.asarray(init_guess_elements)
                if field_name == "init_guess_elem_location":
                    init_guess_elements_location = []
                    for i in range(len(init_guess_elements)):
                        init_guess_elements_location.append(fields[2 + i])
                    init_guess_elements_location = np.asarray(init_guess_elements_location)
                if field_name == "input_elem_abundance":
                    input_elem_abundance = []
                    for i in range(len(fields) - 2):
                        input_elem_abundance.append(fields[2 + i])
                    input_elem_abundance = np.asarray(input_elem_abundance)
                if field_name == "input_elem_abundance_location":
                    input_elem_abundance_location = []
                    for i in range(len(input_elem_abundance)):
                        input_elem_abundance_location.append(fields[2 + i])
                    input_elem_abundance_location = np.asarray(input_elem_abundance_location)
                if field_name == "bounds_macro":
                    Spectra.bound_min_macro = min(float(fields[2]), float(fields[3]))
                    Spectra.bound_max_macro = max(float(fields[2]), float(fields[3]))
                if field_name == "bounds_rotation":
                    Spectra.bound_min_rotation = min(float(fields[2]), float(fields[3]))
                    Spectra.bound_max_rotation = max(float(fields[2]), float(fields[3]))
                if field_name == "bounds_micro":
                    Spectra.bound_min_micro = min(float(fields[2]), float(fields[3]))
                    Spectra.bound_max_micro = max(float(fields[2]), float(fields[3]))
                if field_name == "bounds_abund":
                    Spectra.bound_min_abund = min(float(fields[2]), float(fields[3]))
                    Spectra.bound_max_abund = max(float(fields[2]), float(fields[3]))
                if field_name == "bounds_met":
                    Spectra.bound_min_met = min(float(fields[2]), float(fields[3]))
                    Spectra.bound_max_met = max(float(fields[2]), float(fields[3]))
                if field_name == "bounds_teff":
                    Spectra.bound_min_teff = min(float(fields[2]), float(fields[3]))
                    Spectra.bound_max_teff = max(float(fields[2]), float(fields[3]))
                if field_name == "bounds_doppler":
                    Spectra.bound_min_doppler = min(float(fields[2]), float(fields[3]))
                    Spectra.bound_max_doppler = max(float(fields[2]), float(fields[3]))
                if field_name == "guess_range_microturb":
                    Spectra.guess_min_micro = min(float(fields[2]), float(fields[3]))
                    Spectra.guess_max_micro = max(float(fields[2]), float(fields[3]))
                if field_name == "guess_range_macroturb":
                    Spectra.guess_min_macro = min(float(fields[2]), float(fields[3]))
                    Spectra.guess_max_macro = max(float(fields[2]), float(fields[3]))
                if field_name == "guess_range_rotation":
                    Spectra.guess_min_rotation = min(float(fields[2]), float(fields[3]))
                    Spectra.guess_max_rotation = max(float(fields[2]), float(fields[3]))
                if field_name == "guess_range_abundance":
                    Spectra.guess_min_abund = min(float(fields[2]), float(fields[3]))
                    Spectra.guess_max_abund = max(float(fields[2]), float(fields[3]))
                if field_name == "guess_range_rv":
                    Spectra.guess_min_doppler = min(float(fields[2]), float(fields[3]))
                    Spectra.guess_max_doppler = max(float(fields[2]), float(fields[3]))
                if field_name == "guess_range_teff":
                    Spectra.guess_plus_minus_neg_teff = min(float(fields[2]), float(fields[3]))
                    Spectra.guess_plus_minus_pos_teff = max(float(fields[2]), float(fields[3]))
                if field_name == "debug":
                    Spectra.debug_mode = float(fields[2])
                if field_name == "experimental":
                    if fields[2].lower() == "true" or fields[2].lower() == "yes":
                        Spectra.experimental = True
                    else:
                        Spectra.experimental = False
            line = fp.readline()

    print(f"Fitting data at {spec_input_path} with resolution {Spectra.resolution} and rotation {Spectra.rotation}")


    # set directories
    if ts_compiler == "intel":
        Spectra.turbospec_path = "../turbospectrum/exec/"
    elif ts_compiler == "gnu":
        Spectra.turbospec_path = "../turbospectrum/exec-gf/"
    Spectra.interpol_path = interpol_path
    line_list_path_trimmed = os.path.join(Spectra.global_temp_dir, f'linelist_for_fitting_trimmed_{output_folder_title_date}', "")
    if Spectra.atmosphere_type == "1D":
        Spectra.model_atmosphere_grid_path = model_atmosphere_grid_path_1D
        Spectra.model_atmosphere_list = Spectra.model_atmosphere_grid_path + "model_atmosphere_list.txt"
    elif Spectra.atmosphere_type == "3D":
        Spectra.model_atmosphere_grid_path = model_atmosphere_grid_path_3D
        Spectra.model_atmosphere_list = Spectra.model_atmosphere_grid_path + "model_atmosphere_list.txt"
    Spectra.model_atom_path = model_atom_path
    Spectra.departure_file_path = departure_file_path
    Spectra.output_folder = f"{output_folder_og}{output_folder_title}/"
    Spectra.spec_input_path = spec_input_path

    # load NLTE data dicts
    if Spectra.nlte_flag:
        depart_bin_file_dict, depart_aux_file_dict, model_atom_file_dict = load_nlte_files_in_dict(elements_to_fit,
                                                                                                   depart_bin_file,
                                                                                                   depart_aux_file,
                                                                                                   model_atom_file)

        input_elem_depart_bin_file_dict, input_elem_depart_aux_file_dict, input_elem_model_atom_file_dict = load_nlte_files_in_dict(input_elem_abundance,
                                                                                                   depart_bin_file_input_elem,
                                                                                                   depart_aux_file_input_elem,
                                                                                                   model_atom_file_input_elem, load_fe=False)

        depart_bin_file_dict = {**depart_bin_file_dict, **input_elem_depart_bin_file_dict}
        depart_aux_file_dict = {**depart_aux_file_dict, **input_elem_depart_aux_file_dict}
        model_atom_file_dict = {**model_atom_file_dict, **input_elem_model_atom_file_dict}

        print("NLTE loaded. Please check that elements correspond to their correct binary files:")
        for key in depart_bin_file_dict:
            print(f"{key}: {depart_bin_file_dict[key]} {depart_aux_file_dict[key]} {model_atom_file_dict[key]}")

        print("If files do not correspond, please check config file. Fitted elements should go in the same order as "
              "the NLTE file locations. If Fe is not fitted, then it should be added last to the NLTE file location. "
              "Elements without NLTE binary files do not need them.")

        Spectra.depart_bin_file_dict = depart_bin_file_dict
        Spectra.depart_aux_file_dict = depart_aux_file_dict
        Spectra.model_atom_file_dict = model_atom_file_dict
        Spectra.aux_file_length_dict = {}

        for element in model_atom_file_dict:
            Spectra.aux_file_length_dict[element] = len(np.loadtxt(os_path.join(departure_file_path, depart_aux_file_dict[element]), dtype='str'))

    #prevent overwriting
    if os.path.exists(Spectra.output_folder):
        print("Error: output folder already exists. Run was stopped to prevent overwriting")
        return

    Spectra.linemask_file = f"{linemask_file_og}{linemask_file}"
    Spectra.segment_file = f"{segment_file_og}{segment_file}"

    print(f"Temporary directory name: {Spectra.global_temp_dir}")
    create_dir(Spectra.global_temp_dir)
    create_dir(Spectra.output_folder)

    fitlist = f"{fitlist_input_folder}{fitlist}"

    fitlist_data = np.loadtxt(fitlist, dtype='str')

    if fitlist_data.ndim == 1:
        fitlist_data = np.array([fitlist_data])

    specname_fitlist, teff_fitlist, logg_fitlist, met_fitlist, microturb_fitlist = fitlist_data[:, 0], fitlist_data[:, 1], \
        fitlist_data[:, 2], fitlist_data[:, 3], fitlist_data[:, 4]

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

    # check inputs

    print("\n\nChecking inputs\n")

    if np.size(Spectra.seg_begins) != np.size(Spectra.seg_ends):
        print("Segment beginning and end are not the same length")
    if np.size(Spectra.line_centers_sorted) != np.size(Spectra.line_begins_sorted) or np.size(
            Spectra.line_centers_sorted) != np.size(Spectra.line_ends_sorted):
        print("Line center, beginning and end are not the same length")
    if ts_compiler not in ["intel", "gnu"]:
        print(f"Expected compiler intel or gnu, but got {ts_compiler} instead.")
    if Spectra.nlte_flag:
        for file in Spectra.depart_bin_file_dict:
            if not os.path.isfile(os.path.join(Spectra.departure_file_path, Spectra.depart_bin_file_dict[file])):
                print(f"{Spectra.depart_bin_file_dict[file]} does not exist! Check the spelling or if the file exists")
        for file in Spectra.depart_aux_file_dict:
            if not os.path.isfile(os.path.join(Spectra.departure_file_path, Spectra.depart_aux_file_dict[file])):
                print(f"{Spectra.depart_aux_file_dict[file]} does not exist! Check the spelling or if the file exists")
        for file in Spectra.model_atom_file_dict:
            if not os.path.isfile(os.path.join(Spectra.model_atom_path, Spectra.model_atom_file_dict[file])):
                print(f"{Spectra.model_atom_file_dict[file]} does not exist! Check the spelling or if the file exists")

    for line_start, line_end in zip(Spectra.line_begins_sorted, Spectra.line_ends_sorted):
        index_location = np.where(np.logical_and(Spectra.seg_begins <= line_start, line_end <= Spectra.seg_ends))[0]
        if np.size(index_location) > 1:
            print(f"{line_start} {line_end} linemask has more than 1 segment!")
        if np.size(index_location) == 0:
            print(f"{line_start} {line_end} linemask does not have any corresponding segment")

    print("\nDone doing some basic checks. Consider reading the messages above, if there are any. Can be useful if it "
          "crashes.\n\n")

    print("Trimming")
    cut_linelist(Spectra.seg_begins, Spectra.seg_ends, line_list_path, line_list_path_trimmed, elements_to_use)
    print("Finished trimming linelist")

    model_temperatures, model_logs, model_mets, marcs_value_keys, marcs_models, marcs_values = fetch_marcs_grid(
        Spectra.model_atmosphere_list, TurboSpectrum.marcs_parameters_to_ignore)
    Spectra.model_temperatures = model_temperatures
    Spectra.model_logs = model_logs
    Spectra.model_mets = model_mets
    Spectra.marcs_value_keys = marcs_value_keys
    Spectra.marcs_models = marcs_models
    Spectra.marcs_values = marcs_values

    print("Preparing workers")  # TODO check memory issues? set higher? give warnings?
    client = Client(threads_per_worker=1, n_workers=Spectra.dask_workers)
    print(client)

    host = client.run_on_scheduler(socket.gethostname)
    port = client.scheduler_info()['services']['dashboard']
    print(f"Assuming that the cluster is ran at {login_node_address} (change in code if not the case)")

    # print(logger.info(f"ssh -N -L {port}:{host}:{port} {login_node_address}"))
    print(f"ssh -N -L {port}:{host}:{port} {login_node_address}")

    print("Worker preparation complete")

    futures = []
    for i in range(specname_fitlist.size):
        specname1, teff1, logg1, met1, microturb1 = specname_fitlist[i], teff_fitlist[i], logg_fitlist[i], met_fitlist[i], microturb_fitlist[i]
        for j in range(len(Spectra.seg_begins)):
            future = client.submit(generate_and_fit_atmosphere, specname1, teff1, logg1, microturb1, met1,
                                   Spectra.seg_begins, Spectra.seg_ends, Spectra.ldelta,
                                   os.path.join(line_list_path_trimmed, str(j), ''), element_to_fit, abundance)
            futures.append(future)  # prepares to get values

    print("Start gathering")  # use http://localhost:8787/status to check status. the port might be different
    futures = np.array(client.gather(futures))  # starts the calculations (takes a long time here)
    results = futures
    print("Worker calculation done")  # when done, save values

    for result in results:
        print(result)


if __name__ == '__main__':
    elements_to_use: list[str] = ["39.000"]
    element_to_fit: str = "Y"
    abundance: float = 0.0  # abundance of element in LTE; scaled with metallicity

    if len(argv) > 1:   # when calling the program, can now add extra argument with location of config file, easier to call
        config_location = argv[1]
    else:
        config_location = "../input_files/tsfitpy_input_configuration.txt"  # location of config file

    output_folder_title_date = datetime.datetime.now().strftime(
        "%b-%d-%Y-%H-%M-%S")  # used to not conflict with other instances of fits
    output_folder_title_date = f"{output_folder_title_date}_{np.random.random(1)[0]}"  # in case if someone calls the function several times per second
    print(f"Start of the fitting: {output_folder_title_date}")

    login_node_address = "gemini-login.mpia.de"

    run_nlte_corrections(config_location, output_folder_title_date)
