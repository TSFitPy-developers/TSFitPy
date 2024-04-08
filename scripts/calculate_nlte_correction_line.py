from __future__ import annotations
import datetime
import os
import pickle
import shutil
import socket
from configparser import ConfigParser
from os import path as os_path
import glob
from dask.distributed import Client
import numpy as np
from scipy.optimize import minimize, root_scalar
from .auxiliary_functions import create_dir, calculate_equivalent_width, create_segment_file
from .loading_configs import SpectraParameters, TSFitPyConfig
from .turbospectrum_class_nlte import TurboSpectrum
from .synthetic_code_class import fetch_marcs_grid

def binary_search_lower_bound(array_to_search: list[str], dict_array_values: dict, low: int, high: int,
                              element_to_search: float) -> tuple[int, dict]:
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


def write_lines(all_lines_to_write: dict[list[str]], elem_line_1_to_save: str, elem_line_2_to_save: str,
                new_path_name: str, line_list_number: float):
    for key in all_lines_to_write:
        new_linelist_name: str = os.path.join(f"{new_path_name}", f"{key}", f"linelist-{line_list_number}.bsyn")
        with open(new_linelist_name, "a") as new_file_to_write:
            new_file_to_write.write(f"{elem_line_1_to_save}	{len(all_lines_to_write[key])}\n")
            new_file_to_write.write(f"{elem_line_2_to_save}")
            for line_to_write in all_lines_to_write[key]:
                # pass
                new_file_to_write.write(line_to_write)


def cut_linelist(seg_begins: list[float], seg_ends: list[float], old_path_name: str, new_path_name: str,
                 elements_to_use: list[str]):
    line_list_path: str = old_path_name
    line_list_files_draft: list = []
    line_list_files_draft.extend([i for i in glob.glob(os_path.join(line_list_path, "*")) if not i.endswith(".txt")])

    molecules_flag = False

    segment_to_use_begins: np.ndarray = np.asarray(seg_begins)
    segment_to_use_ends: np.ndarray = np.asarray(seg_ends)

    segment_index_order: np.ndarray = np.argsort(segment_to_use_begins)
    segment_to_use_begins: np.ndarray = segment_to_use_begins[segment_index_order]
    segment_to_use_ends: np.ndarray = segment_to_use_ends[segment_index_order]
    segment_min_wavelength: float = np.min(segment_to_use_begins)
    segment_max_wavelength: float = np.max(segment_to_use_ends)

    line_list_files: list = []
    # opens each file, reads first row, if it is long enough then it is molecule. If fitting molecules, then add to the line list, otherwise ignore molecules FAST
    for i in range(len(line_list_files_draft)):
        with open(line_list_files_draft[i]) as fp:
            line = fp.readline()
            fields = line.strip().split()
            sep = '.'
            element = fields[0] + fields[1]
            elements = element.split(sep, 1)[0]
            if len(elements) > 3 and molecules_flag == 'True':
                line_list_files.append(line_list_files_draft[i])
            elif len(elements) <= 3:
                line_list_files.append(line_list_files_draft[i])
        fp.close()

    for i in range(len(seg_begins)):
        new_path_name_one_seg: str = os.path.join(f"{new_path_name}", f"{i}", '')
        if not os.path.exists(new_path_name_one_seg):
            os.makedirs(new_path_name_one_seg)

    for line_list_number, line_list_file in enumerate(line_list_files):
        new_linelist_name: str = f"{new_path_name}"  # f"linelist-{line_list_number}.bsyn"
        # new_linelist: str = os_path.join(f"{new_path_name}", f"linelist-{i}.bsyn")
        # with open(new_linelist, "w") as new_file_to_write:
        with open(line_list_file) as fp:
            lines_file: list[str] = fp.readlines()
            all_lines_to_write: dict = {}
            line_number_read_for_element: int = 0
            line_number_read_file: int = 0
            total_lines_in_file: int = len(lines_file)
            while line_number_read_file < total_lines_in_file:  # go through all line
                line: str = lines_file[line_number_read_file]
                fields: list[str] = line.strip().split()

                # it means this is an element
                if all_lines_to_write:  # if there was an element before with segments, then write them first
                    write_lines(all_lines_to_write, elem_line_1_to_save, elem_line_2_to_save, new_linelist_name,
                                line_list_number)
                    all_lines_to_write: dict = {}
                element_name = f"{fields[0]}{fields[1]}"
                if len(fields[0]) > 1:  # save the first two lines of an element for the future
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
                wavelength_maximum_element: float = float(
                    lines_file[number_of_lines_element + line_number_read_file - 1].strip().split()[0])

                element_wavelength_dictionary[0] = wavelength_minimum_element
                element_wavelength_dictionary[number_of_lines_element - 1] = wavelength_maximum_element
                if elem_line_2_to_save.strip().replace("'", "").replace("NLTE", "").replace("LTE", "").replace("I", "").replace(" ", "") in elements_to_use:
                    # check that ANY wavelengths are within the range at all
                    if not (wavelength_maximum_element < segment_min_wavelength or wavelength_minimum_element > segment_max_wavelength):
                        for seg_index, (seg_begin, seg_end) in enumerate(
                                zip(segment_to_use_begins, segment_to_use_ends)):  # wavelength lines write here
                            index_seg_start, element_wavelength_dictionary = binary_search_lower_bound(
                                lines_file[line_number_read_file:number_of_lines_element + line_number_read_file],
                                element_wavelength_dictionary, 0, number_of_lines_element - 1, seg_begin)
                            wavelength_current_line: float = element_wavelength_dictionary[index_seg_start]
                            line_stripped: str = lines_file[line_number_read_file + index_seg_start].strip()
                            line_number_read_for_element: int = index_seg_start + line_number_read_file
                            while wavelength_current_line <= seg_end and line_number_read_for_element < number_of_lines_element + line_number_read_file and wavelength_current_line >= seg_begin:
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
            write_lines(all_lines_to_write, elem_line_1_to_save, elem_line_2_to_save, new_linelist_name,
                        line_list_number)

class AbusingClasses:

    def __init__(self):
        pass


def generate_atmosphere(abusingclasses, teff, logg, vturb, met, lmin, lmax, ldelta, line_list_path, element, abundance, abundances_dict1, nlte_flag, verbose=False):
    # parameters to adjust

    teff = teff
    logg = logg
    if element == "Fe":
        met = abundance
    else:
        met = met
    lmin = lmin
    lmax = lmax
    ldelta = ldelta
    item_abund = abundances_dict1.copy()
    for element1 in item_abund:
        # scale to [X/H] from [X/Fe]
        item_abund[element1] = item_abund[element1] + met
    item_abund["Fe"] = met
    if element not in item_abund:
        item_abund[element] = abundance + met
    #temp_directory = f"../temp_directory_{datetime.datetime.now().strftime('%b-%d-%Y-%H-%M-%S')}__{np.random.random(1)[0]}/"
    temp_directory = os.path.join(abusingclasses.global_temp_dir, f"{datetime.datetime.now().strftime('%b-%d-%Y-%H-%M-%S')}__{np.random.random(1)[0]}", "")

    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)

    ts = TurboSpectrum(
        turbospec_path=abusingclasses.spectral_code_path,
        interpol_path=abusingclasses.interpol_path,
        line_list_paths=line_list_path,
        marcs_grid_path=abusingclasses.model_atmosphere_grid_path,
        marcs_grid_list=abusingclasses.model_atmosphere_list,
        model_atom_path=abusingclasses.model_atom_path,
        departure_file_path=abusingclasses.departure_file_path,
        aux_file_length_dict=abusingclasses.aux_file_length_dict,
        model_temperatures=abusingclasses.model_temperatures,
        model_logs=abusingclasses.model_logs,
        model_mets=abusingclasses.model_mets,
        marcs_value_keys=abusingclasses.marcs_value_keys,
        marcs_models=abusingclasses.marcs_models,
        marcs_values=abusingclasses.marcs_values)

    ts.configure(t_eff=teff, log_g=logg, metallicity=met,
                 turbulent_velocity=vturb, lambda_delta=ldelta, lambda_min=lmin, lambda_max=lmax,
                 free_abundances=item_abund, temp_directory=temp_directory, nlte_flag=nlte_flag, verbose=verbose,
                 atmosphere_dimension=abusingclasses.atmosphere_type, windows_flag=False,
                 segment_file=abusingclasses.segment_file,
                 line_mask_file=abusingclasses.linemask_file, depart_bin_file=abusingclasses.depart_bin_file_dict,
                 depart_aux_file=abusingclasses.depart_aux_file_dict,
                 model_atom_file=abusingclasses.model_atom_file_dict)

    ts.synthesize_spectra()
    # ts.run_turbospectrum()

    try:
        if os_path.exists('{}/spectrum_00000000.spec'.format(temp_directory)) and os.stat('{}/spectrum_00000000.spec'.format(temp_directory)).st_size != 0:
            wave_mod_orig, flux_norm_mod_orig = np.loadtxt('{}spectrum_00000000.spec'.format(temp_directory),
                                                                      usecols=(0, 1), unpack=True)
            if np.size(wave_mod_orig) == 0:
                wave_mod_orig, flux_norm_mod_orig = None, None
        else:
            wave_mod_orig, flux_norm_mod_orig = None, None
    except (FileNotFoundError, OSError, ValueError) as error:
        wave_mod_orig, flux_norm_mod_orig = None, None

    shutil.rmtree(temp_directory)

    if np.size(wave_mod_orig) == 0:
        wave_mod_orig, flux_norm_mod_orig = None, None

    return wave_mod_orig, flux_norm_mod_orig


def get_nlte_ew(param, abusingclasses, teff, logg, microturb, met, lmin, lmax, ldelta, line_list_path, element, lte_ew, verbose):
    abundance = param
    wavelength_nlte, norm_flux_nlte = generate_atmosphere(abusingclasses, teff, logg, microturb, met, lmin - 5, lmax + 5, ldelta,
                                                          line_list_path, element, abundance, {}, True, verbose)
    if wavelength_nlte is not None:
        nlte_ew = calculate_equivalent_width(wavelength_nlte, norm_flux_nlte, lmin - 3, lmax + 3) * 1000
        diff = (nlte_ew - lte_ew)
    else:
        nlte_ew = 9999999
        diff = 9999999
    print(f"NLTE abund={abundance} EW_lte={lte_ew} EW_nlte={nlte_ew} EW_diff={diff}")
    return diff



def generate_and_fit_atmosphere(pickle_file_path, specname, teff, logg, microturb, met, lmin, lmax, ldelta, line_list_path, element,
                                abundance, abundances_dict1, line_center, verbose=False):
    # load pickle file
    with open(pickle_file_path, 'rb') as f:
        abusingclasses = pickle.load(f)
    wavelength_lte, norm_flux_lte = generate_atmosphere(abusingclasses, teff, logg, microturb, met, lmin - 5, lmax + 5, ldelta,
                                                        line_list_path, element, abundance, abundances_dict1, False, verbose)
    if element in abundances_dict1 and element != 'Fe':
        abundance = abundances_dict1[element]
    if wavelength_lte is not None:
        ew_lte = calculate_equivalent_width(wavelength_lte, norm_flux_lte, lmin - 3, lmax + 3) * 1000
        print(f"Fitting {specname} Teff={teff} logg={logg} [Fe/H]={met} microturb={microturb} line_center={line_center} ew_lte={ew_lte} LTE_abund={abundance}")
        try:
            result = root_scalar(get_nlte_ew, args=(abusingclasses, teff, logg, microturb, met, lmin, lmax, ldelta, line_list_path, element, ew_lte, verbose),
                                 bracket=[abundance - 3, abundance + 3], method='brentq')
            #result = minimize(get_nlte_ew, [abundance - 0.1, abundance + 0.5],
            #                  args=(abusingclasses, teff, logg, microturb, met, lmin, lmax, ldelta, line_list_path, element, ew_lte),
            #                  bounds=[(abundance - 3, abundance + 3)], method="Nelder-Mead",
            #                  options={'maxiter': 50, 'disp': False, 'fatol': 1e-8, 'xatol': 1e-3})  # 'eps': 1e-8

            nlte_correction = result.root
            ew_nlte = ew_lte
            print(f"Fitted with NLTE correction={nlte_correction - abundance} EW_lte={ew_lte}")
        except ValueError:
            print("Fitting failed")
            ew_nlte = -99999
            nlte_correction = -99999
    else:
        ew_lte = -99999
        ew_nlte = -99999
        nlte_correction = -99999
    return [f"{specname}\t{teff}\t{logg}\t{met}\t{microturb}\t{line_center}\t{ew_lte}\t{ew_nlte}\t{np.abs(ew_nlte - ew_lte)}\t{nlte_correction - abundance}"]


def run_nlte_corrections(config_file_name, output_folder_title, abundance=0):
    login_node_address = "gemini-login.mpia.de"
    tsfitpy_configuration = TSFitPyConfig(config_file_name, output_folder_title)
    tsfitpy_configuration.load_config()
    tsfitpy_configuration.validate_input()
    if not config_file_name[-4:] == ".cfg":
        tsfitpy_configuration.convert_old_config()

    abusingclasses = AbusingClasses()
    abusingclasses.nlte_flag = tsfitpy_configuration.nlte_flag
    abusingclasses.elem_to_fit = tsfitpy_configuration.elements_to_fit
    abusingclasses.ldelta = tsfitpy_configuration.wavelength_delta
    abusingclasses.global_temp_dir = tsfitpy_configuration.temporary_directory_path
    abusingclasses.dask_workers = tsfitpy_configuration.number_of_cpus
    abusingclasses.atmosphere_type = tsfitpy_configuration.atmosphere_type
    abusingclasses.segment_file = None

    print(
        f"Fitting data at {tsfitpy_configuration.spectra_input_path} with resolution {tsfitpy_configuration.resolution} and rotation {tsfitpy_configuration.rotation}")

    abusingclasses.spectral_code_path = tsfitpy_configuration.spectral_code_path
    abusingclasses.interpol_path = tsfitpy_configuration.interpolators_path
    line_list_path_trimmed = os.path.join(tsfitpy_configuration.temporary_directory_path,
                                          f'linelist_for_fitting_trimmed_{output_folder_title}', "")
    abusingclasses.model_atmosphere_grid_path = tsfitpy_configuration.model_atmosphere_grid_path
    abusingclasses.model_atmosphere_list = tsfitpy_configuration.model_atmosphere_list
    abusingclasses.model_atom_path = tsfitpy_configuration.model_atoms_path
    abusingclasses.departure_file_path = tsfitpy_configuration.departure_file_path
    abusingclasses.output_folder = tsfitpy_configuration.output_folder_path
    abusingclasses.spec_input_path = tsfitpy_configuration.spectra_input_path

    nlte_config = ConfigParser()
    nlte_config.read(tsfitpy_configuration.departure_file_config_path)

    depart_bin_file_dict, depart_aux_file_dict, model_atom_file_dict = {}, {}, {}

    for element in tsfitpy_configuration.elements_to_fit:
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

    print(
        f"If files do not correspond, please check config file {os.path.join(tsfitpy_configuration.departure_file_path, 'nlte_filenames.cfg')}. "
        f"Elements without NLTE binary files do not need them.")

    tsfitpy_configuration.depart_bin_file_dict = depart_bin_file_dict
    tsfitpy_configuration.depart_aux_file_dict = depart_aux_file_dict
    tsfitpy_configuration.model_atom_file_dict = model_atom_file_dict
    abusingclasses.depart_bin_file_dict = depart_bin_file_dict
    abusingclasses.depart_aux_file_dict = depart_aux_file_dict
    abusingclasses.model_atom_file_dict = model_atom_file_dict

    abusingclasses.aux_file_length_dict = {}

    for element in model_atom_file_dict:
        abusingclasses.aux_file_length_dict[element] = len(
            np.loadtxt(os_path.join(tsfitpy_configuration.departure_file_path, depart_aux_file_dict[element]), dtype='str'))

    # prevent overwriting
    if os.path.exists(abusingclasses.output_folder):
        print("Error: output folder already exists. Run was stopped to prevent overwriting")
        return

    abusingclasses.linemask_file = os.path.join(tsfitpy_configuration.linemasks_path, tsfitpy_configuration.linemask_file)

    print(f"Temporary directory name: {abusingclasses.global_temp_dir}")
    create_dir(abusingclasses.global_temp_dir)
    create_dir(abusingclasses.output_folder)

    fitlist = os.path.join(tsfitpy_configuration.fitlist_input_path, tsfitpy_configuration.input_fitlist_filename)

    fitlist_data = SpectraParameters(fitlist, True)

    if tsfitpy_configuration.vmic_input:
        output_vmic: bool = True
    else:
        output_vmic: bool = False

    if tsfitpy_configuration.rotation_input:
        output_rotation: bool = True
    else:
        output_rotation: bool = False
    fitlist_spectra_parameters = fitlist_data.get_spectra_parameters_for_fit(output_vmic,
                                                                             tsfitpy_configuration.vmac_input,
                                                                             output_rotation)
    print(fitlist_data)

    line_centers, line_begins, line_ends = np.loadtxt(abusingclasses.linemask_file, comments=";", usecols=(0, 1, 2),
                                                      unpack=True)

    if line_centers.size > 1:
        abusingclasses.line_begins_sorted = np.array(sorted(line_begins))
        abusingclasses.line_ends_sorted = np.array(sorted(line_ends))
        abusingclasses.line_centers_sorted = np.array(sorted(line_centers))
    elif line_centers.size == 1:
        abusingclasses.line_begins_sorted = np.array([line_begins])
        abusingclasses.line_ends_sorted = np.array([line_ends])
        abusingclasses.line_centers_sorted = np.array([line_centers])

    abusingclasses.seg_begins, abusingclasses.seg_ends = create_segment_file(5, abusingclasses.line_begins_sorted, abusingclasses.line_ends_sorted)

    # check inputs

    print("\n\nChecking inputs\n")

    if np.size(abusingclasses.line_centers_sorted) != np.size(abusingclasses.line_begins_sorted) or np.size(
            abusingclasses.line_centers_sorted) != np.size(abusingclasses.line_ends_sorted):
        print("Line center, beginning and end are not the same length")

    for line_start, line_end in zip(abusingclasses.line_begins_sorted, abusingclasses.line_ends_sorted):
        index_location = \
        np.where(np.logical_and(abusingclasses.seg_begins <= line_start, line_end <= abusingclasses.seg_ends))[0]
        if np.size(index_location) > 1:
            print(f"{line_start} {line_end} linemask has more than 1 segment!")
        if np.size(index_location) == 0:
            print(f"{line_start} {line_end} linemask does not have any corresponding segment")

    print("\nDone doing some basic checks. Consider reading the messages above, if there are any. Can be useful if it "
          "crashes.\n\n")

    print("Trimming")
    cut_linelist(abusingclasses.line_begins_sorted, abusingclasses.line_ends_sorted, tsfitpy_configuration.line_list_path,
                 line_list_path_trimmed, abusingclasses.elem_to_fit[0])
    print("Finished trimming linelist")

    model_temperatures, model_logs, model_mets, marcs_value_keys, marcs_models, marcs_values = fetch_marcs_grid(
        abusingclasses.model_atmosphere_list, TurboSpectrum.marcs_parameters_to_ignore)
    abusingclasses.model_temperatures = model_temperatures
    abusingclasses.model_logs = model_logs
    abusingclasses.model_mets = model_mets
    abusingclasses.marcs_value_keys = marcs_value_keys
    abusingclasses.marcs_models = marcs_models
    abusingclasses.marcs_values = marcs_values

    if tsfitpy_configuration.debug_mode >= 2:
        verbose = True
    else:
        verbose = False

    print("Preparing workers")  # TODO check memory issues? set higher? give warnings?
    client = Client(threads_per_worker=1, n_workers=abusingclasses.dask_workers)
    print(client)

    host = client.run_on_scheduler(socket.gethostname)
    port = client.scheduler_info()['services']['dashboard']
    print(f"Assuming that the cluster is ran at {login_node_address} (change in code if not the case)")

    # print(logger.info(f"ssh -N -L {port}:{host}:{port} {login_node_address}"))
    print(f"ssh -N -L {port}:{host}:{port} {login_node_address}")

    print("Worker preparation complete")

    create_dir(tsfitpy_configuration.temporary_directory_path)
    with open(os.path.join(tsfitpy_configuration.temporary_directory_path, 'abusingclasses.pkl'), 'wb') as f:
        pickle.dump(abusingclasses, f)

    futures = []
    for idx, one_spectra_parameters in enumerate(fitlist_spectra_parameters):
        # specname_list, rv_list, teff_list, logg_list, feh_list, vmic_list, vmac_list, abundance_list
        specname1, rv1, teff1, logg1, met1, microturb1, macroturb1, rotation1, abundances_dict1, resolution1, _ = one_spectra_parameters
        # if element is Fe, then take abundance from metallicity
        if abusingclasses.elem_to_fit[0] == "Fe":
            abundance = met1
        for j in range(len(abusingclasses.line_begins_sorted)):
            future = client.submit(generate_and_fit_atmosphere, os.path.join(tsfitpy_configuration.temporary_directory_path, 'abusingclasses.pkl'), specname1, teff1, logg1, microturb1, met1,
                                   abusingclasses.line_begins_sorted[j] - 2, abusingclasses.line_ends_sorted[j] + 2,
                                   abusingclasses.ldelta,
                                   os.path.join(line_list_path_trimmed, str(j), ''), abusingclasses.elem_to_fit[0],
                                   abundance, abundances_dict1, abusingclasses.line_centers_sorted[j], verbose)
            futures.append(future)  # prepares to get values

    print("Start gathering")  # use http://localhost:8787/status to check status. the port might be different
    futures = np.array(client.gather(futures))  # starts the calculations (takes a long time here)
    results = futures
    print("Worker calculation done")  # when done, save values

    for result in results:
        print(result)

    # shutil.rmtree(abusingclasses.global_temp_dir)  # clean up temp directory
    # shutil.rmtree(line_list_path_trimmed)  # clean up trimmed line list

    output = os.path.join(abusingclasses.output_folder, tsfitpy_configuration.output_filename)

    f = open(output, 'a')
    # specname, line_center, ew_lte, ew_nlte, nlte_correction
    output_columns = "#specname\tteff\tlogg\tmet\tmicroturb\twave_center\tew_lte\tew_nlte\tew_diff\tnlte_correction"
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

