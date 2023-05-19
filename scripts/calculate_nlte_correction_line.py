from __future__ import annotations
import datetime
import os
import shutil
import socket
from configparser import ConfigParser
from os import path as os_path
import glob
from sys import argv
from dask.distributed import Client
import numpy as np
from scipy.optimize import minimize
from scripts.create_window_linelist_function import binary_search_lower_bound, write_lines
from scripts.TSFitPy import load_nlte_files_in_dict, create_dir, calculate_equivalent_width, TSFitPyConfig, create_segment_file
from scripts.turbospectrum_class_nlte import TurboSpectrum, fetch_marcs_grid


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


def generate_atmosphere(teff, logg, vturb, met, lmin, lmax, ldelta, line_list_path, element, abundance, nlte_flag):
    # parameters to adjust

    teff = teff
    logg = logg
    met = met
    lmin = lmin
    lmax = lmax
    ldelta = ldelta
    item_abund = {"Fe": met, element: abundance + met}
    #temp_directory = f"../temp_directory_{datetime.datetime.now().strftime('%b-%d-%Y-%H-%M-%S')}__{np.random.random(1)[0]}/"
    temp_directory = os.path.join(AbusingClasses.global_temp_dir, f"{datetime.datetime.now().strftime('%b-%d-%Y-%H-%M-%S')}__{np.random.random(1)[0]}", "")

    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)

    ts = TurboSpectrum(
        turbospec_path=AbusingClasses.turbospec_path,
        interpol_path=AbusingClasses.interpol_path,
        line_list_paths=line_list_path,
        marcs_grid_path=AbusingClasses.model_atmosphere_grid_path,
        marcs_grid_list=AbusingClasses.model_atmosphere_list,
        model_atom_path=AbusingClasses.model_atom_path,
        departure_file_path=AbusingClasses.departure_file_path,
        aux_file_length_dict=AbusingClasses.aux_file_length_dict,
        model_temperatures=AbusingClasses.model_temperatures,
        model_logs=AbusingClasses.model_logs,
        model_mets=AbusingClasses.model_mets,
        marcs_value_keys=AbusingClasses.marcs_value_keys,
        marcs_models=AbusingClasses.marcs_models,
        marcs_values=AbusingClasses.marcs_values)

    ts.configure(t_eff=teff, log_g=logg, metallicity=met,
                 turbulent_velocity=vturb, lambda_delta=ldelta, lambda_min=lmin, lambda_max=lmax,
                 free_abundances=item_abund, temp_directory=temp_directory, nlte_flag=nlte_flag, verbose=False,
                 atmosphere_dimension=AbusingClasses.atmosphere_type, windows_flag=False,
                 segment_file=AbusingClasses.segment_file,
                 line_mask_file=AbusingClasses.linemask_file, depart_bin_file=AbusingClasses.depart_bin_file_dict,
                 depart_aux_file=AbusingClasses.depart_aux_file_dict,
                 model_atom_file=AbusingClasses.model_atom_file_dict)

    ts.run_turbospectrum_and_atmosphere()
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

    return wave_mod_orig, flux_norm_mod_orig


def get_nlte_ew(param, teff, logg, microturb, met, lmin, lmax, ldelta, line_list_path, element, lte_ew):
    abundance = param[0]
    wavelength_nlte, norm_flux_nlte = generate_atmosphere(teff, logg, microturb, met, lmin - 5, lmax + 5, ldelta,
                                                          line_list_path, element, abundance, True)
    if wavelength_nlte is not None:
        nlte_ew = calculate_equivalent_width(wavelength_nlte, norm_flux_nlte, lmin - 3, lmax + 3) * 1000
        diff = np.square((nlte_ew - lte_ew))
    else:
        nlte_ew = 9999999
        diff = 9999999
    print(f"NLTE abund={abundance} EW_lte={lte_ew} EW_nlte={nlte_ew} EW_diff={diff}")
    return diff



def generate_and_fit_atmosphere(specname, teff, logg, microturb, met, lmin, lmax, ldelta, line_list_path, element,
                                abundance, line_center):
    wavelength_lte, norm_flux_lte = generate_atmosphere(teff, logg, microturb, met, lmin - 5, lmax + 5, ldelta,
                                                        line_list_path, element, abundance, False)
    if wavelength_lte is not None:
        ew_lte = calculate_equivalent_width(wavelength_lte, norm_flux_lte, lmin - 3, lmax + 3) * 1000
        print(f"Fitting {specname} Teff={teff} logg={logg} [Fe/H]={met} microturb={microturb} line_center={line_center} ew_lte={ew_lte}")
        result = minimize(get_nlte_ew, [abundance - 0.3, abundance + 0.3],
                          args=(teff, logg, microturb, met, lmin, lmax, ldelta, line_list_path, element, ew_lte),
                          bounds=[(abundance - 3, abundance + 3)], method="Nelder-Mead",
                          options={'maxiter': 50, 'disp': False, 'fatol': 1e-8, 'xatol': 1e-3})  # 'eps': 1e-8

        nlte_correction = result.x[0]
        ew_nlte = np.sqrt(result.fun) + ew_lte
        print(f"Fitted with NLTE correction={nlte_correction} EW_lte={ew_lte} EW_nlte={ew_nlte} EW_diff={result.fun}")
    else:
        ew_lte = -99999
        ew_nlte = -99999
        nlte_correction = -99999
    return [f"{specname}\t{teff}\t{logg}\t{met}\t{microturb}\t{line_center}\t{ew_lte}\t{ew_nlte}\t{np.abs(ew_nlte - ew_lte)}\t{nlte_correction}"]


def run_nlte_corrections(config_file_name, output_folder_title, abundance=0):
    login_node_address = "gemini-login.mpia.de"

    depart_bin_file = []
    depart_aux_file = []
    model_atom_file = []
    input_elem_abundance = []
    depart_bin_file_input_elem = []
    depart_aux_file_input_elem = []
    model_atom_file_input_elem = []

    tsfitpy_configuration = TSFitPyConfig(config_file_name, output_folder_title)
    tsfitpy_configuration.load_config()
    tsfitpy_configuration.validate_input()
    tsfitpy_configuration.convert_old_config()

    AbusingClasses.nlte_flag = tsfitpy_configuration.nlte_flag
    AbusingClasses.elem_to_fit = tsfitpy_configuration.elements_to_fit
    AbusingClasses.ldelta = tsfitpy_configuration.wavelength_delta
    AbusingClasses.global_temp_dir = tsfitpy_configuration.global_temporary_directory
    AbusingClasses.dask_workers = tsfitpy_configuration.number_of_cpus

    print(
        f"Fitting data at {tsfitpy_configuration.spectra_input_path} with resolution {tsfitpy_configuration.resolution} and rotation {tsfitpy_configuration.rotation}")

    AbusingClasses.turbospec_path = tsfitpy_configuration.turbospectrum_path
    AbusingClasses.interpol_path = tsfitpy_configuration.interpolators_path
    line_list_path_trimmed = os.path.join(tsfitpy_configuration.global_temporary_directory,
                                          f'linelist_for_fitting_trimmed_{output_folder_title}', "")
    AbusingClasses.model_atmosphere_grid_path = tsfitpy_configuration.model_atmosphere_grid_path
    AbusingClasses.model_atmosphere_list = tsfitpy_configuration.model_atmosphere_grid_path
    AbusingClasses.model_atom_path = tsfitpy_configuration.model_atoms_path
    AbusingClasses.departure_file_path = tsfitpy_configuration.departure_file_path
    AbusingClasses.output_folder = tsfitpy_configuration.output_folder_path
    AbusingClasses.spec_input_path = tsfitpy_configuration.spectra_input_path

    # load NLTE data dicts
    if tsfitpy_configuration.nlte_flag:
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

        print(
            f"If files do not correspond, please check config file {os.path.join(tsfitpy_configuration.departure_file_path, 'nlte_filenames.cfg')}. "
            f"Elements without NLTE binary files do not need them.")

        tsfitpy_configuration.depart_bin_file_dict = depart_bin_file_dict
        tsfitpy_configuration.depart_aux_file_dict = depart_aux_file_dict
        tsfitpy_configuration.model_atom_file_dict = model_atom_file_dict
        AbusingClasses.depart_bin_file_dict = depart_bin_file_dict
        AbusingClasses.depart_aux_file_dict = depart_aux_file_dict
        AbusingClasses.model_atom_file_dict = model_atom_file_dict

    # prevent overwriting
    if os.path.exists(AbusingClasses.output_folder):
        print("Error: output folder already exists. Run was stopped to prevent overwriting")
        return

    AbusingClasses.linemask_file = tsfitpy_configuration.linemask_file

    print(f"Temporary directory name: {AbusingClasses.global_temp_dir}")
    create_dir(AbusingClasses.global_temp_dir)
    create_dir(AbusingClasses.output_folder)

    fitlist = os.path.join(tsfitpy_configuration.fitlist_input_path, tsfitpy_configuration.input_fitlist_filename)

    fitlist_data = np.loadtxt(fitlist, dtype='str')

    if fitlist_data.ndim == 1:
        fitlist_data = np.array([fitlist_data])

    specname_fitlist, teff_fitlist, logg_fitlist, met_fitlist, microturb_fitlist = fitlist_data[:, 0], fitlist_data[:,
                                                                                                       2].astype(float), \
        fitlist_data[:, 3].astype(float), fitlist_data[:, 4].astype(float), fitlist_data[:, 5].astype(float)

    line_centers, line_begins, line_ends = np.loadtxt(AbusingClasses.linemask_file, comments=";", usecols=(0, 1, 2),
                                                      unpack=True)

    if line_centers.size > 1:
        AbusingClasses.line_begins_sorted = np.array(sorted(line_begins))
        AbusingClasses.line_ends_sorted = np.array(sorted(line_ends))
        AbusingClasses.line_centers_sorted = np.array(sorted(line_centers))
    elif line_centers.size == 1:
        AbusingClasses.line_begins_sorted = np.array([line_begins])
        AbusingClasses.line_ends_sorted = np.array([line_ends])
        AbusingClasses.line_centers_sorted = np.array([line_centers])

    AbusingClasses.seg_begins, AbusingClasses.seg_ends = create_segment_file(5, AbusingClasses.line_begins_sorted, AbusingClasses.line_ends_sorted)

    # check inputs

    print("\n\nChecking inputs\n")

    if np.size(AbusingClasses.line_centers_sorted) != np.size(AbusingClasses.line_begins_sorted) or np.size(
            AbusingClasses.line_centers_sorted) != np.size(AbusingClasses.line_ends_sorted):
        print("Line center, beginning and end are not the same length")

    for line_start, line_end in zip(AbusingClasses.line_begins_sorted, AbusingClasses.line_ends_sorted):
        index_location = \
        np.where(np.logical_and(AbusingClasses.seg_begins <= line_start, line_end <= AbusingClasses.seg_ends))[0]
        if np.size(index_location) > 1:
            print(f"{line_start} {line_end} linemask has more than 1 segment!")
        if np.size(index_location) == 0:
            print(f"{line_start} {line_end} linemask does not have any corresponding segment")

    print("\nDone doing some basic checks. Consider reading the messages above, if there are any. Can be useful if it "
          "crashes.\n\n")

    print("Trimming")
    cut_linelist(AbusingClasses.line_begins_sorted, AbusingClasses.line_ends_sorted, tsfitpy_configuration.line_list_path,
                 line_list_path_trimmed, AbusingClasses.elem_to_fit[0])
    print("Finished trimming linelist")

    model_temperatures, model_logs, model_mets, marcs_value_keys, marcs_models, marcs_values = fetch_marcs_grid(
        AbusingClasses.model_atmosphere_list, TurboSpectrum.marcs_parameters_to_ignore)
    AbusingClasses.model_temperatures = model_temperatures
    AbusingClasses.model_logs = model_logs
    AbusingClasses.model_mets = model_mets
    AbusingClasses.marcs_value_keys = marcs_value_keys
    AbusingClasses.marcs_models = marcs_models
    AbusingClasses.marcs_values = marcs_values

    print("Preparing workers")  # TODO check memory issues? set higher? give warnings?
    client = Client(threads_per_worker=1, n_workers=AbusingClasses.dask_workers)
    print(client)

    host = client.run_on_scheduler(socket.gethostname)
    port = client.scheduler_info()['services']['dashboard']
    print(f"Assuming that the cluster is ran at {login_node_address} (change in code if not the case)")

    # print(logger.info(f"ssh -N -L {port}:{host}:{port} {login_node_address}"))
    print(f"ssh -N -L {port}:{host}:{port} {login_node_address}")

    print("Worker preparation complete")

    futures = []
    for i in range(specname_fitlist.size):
        specname1, teff1, logg1, met1, microturb1 = specname_fitlist[i], teff_fitlist[i], logg_fitlist[i], met_fitlist[
            i], microturb_fitlist[i]
        for j in range(len(AbusingClasses.line_begins_sorted)):
            future = client.submit(generate_and_fit_atmosphere, specname1, teff1, logg1, microturb1, met1,
                                   AbusingClasses.line_begins_sorted[j] - 2, AbusingClasses.line_ends_sorted[j] + 2,
                                   AbusingClasses.ldelta,
                                   os.path.join(line_list_path_trimmed, str(j), ''), AbusingClasses.elem_to_fit[0],
                                   abundance, AbusingClasses.line_centers_sorted[j])
            futures.append(future)  # prepares to get values

    print("Start gathering")  # use http://localhost:8787/status to check status. the port might be different
    futures = np.array(client.gather(futures))  # starts the calculations (takes a long time here)
    results = futures
    print("Worker calculation done")  # when done, save values

    for result in results:
        print(result)

    # shutil.rmtree(AbusingClasses.global_temp_dir)  # clean up temp directory
    # shutil.rmtree(line_list_path_trimmed)  # clean up trimmed line list

    output = os.path.join(AbusingClasses.output_folder, tsfitpy_configuration.output_file_name)

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

