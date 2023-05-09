from __future__ import annotations

import datetime
import shutil
from configparser import ConfigParser

import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
from warnings import warn

from scripts.convolve import conv_macroturbulence, conv_rotation, conv_res
from scripts.create_window_linelist_function import create_window_linelist
from scripts.turbospectrum_class_nlte import TurboSpectrum, fetch_marcs_grid


def apply_doppler_correction(wave_ob: np.ndarray, doppler: float) -> np.ndarray:
    return wave_ob / (1 + (doppler / 299792.))

def get_all_file_names_in_a_folder(path_to_get_files_from: str) -> list:
    """
    Gets a list of all files in a folder

    :param path_to_get_files_from: Folder where to find files
    :return: List of names of files (not paths to them, only names)
    """

    file_names = [f for f in os.listdir(path_to_get_files_from) if os.path.isfile(os.path.join(path_to_get_files_from, f))]
    if '.DS_Store' in file_names:
        file_names.remove('.DS_Store')  # Sometimes these get in the way, so try to remove this file
    return file_names

def load_output_data(output_folder_location: str, old_variable=None) -> dict:
    if old_variable is not None:
        warn("Warning: now the config file is copied into the output folder. There is no need to "
             "pass config file location for new outputs. Thus you only need to write output folder "
             "from now on. In the future this will give an error", DeprecationWarning, stacklevel=2)
        if os.path.isfile(os.path.join(old_variable, "configuration.txt")):
            # trying new way of loading: first variable is output folder with config in it
            print(f"Loading config from {os.path.join(old_variable, 'configuration.txt')}")
            config_file_location = os.path.join(old_variable, "configuration.txt")
            output_folder_location = old_variable
        else:
            # this was an old way of loading. first variable: config file, second variable: output folder
            print(f"Loading config from {output_folder_location}")
            config_file_location = output_folder_location
            output_folder_location = old_variable
    else:
        # new way of loading: first variable is output folder with config in it
        config_file_location = os.path.join(output_folder_location, "configuration.txt")
    with open(config_file_location) as fp:
        line = fp.readline()
        while line:
            if len(line) > 1:
                fields = line.strip().split()
                field_name = fields[0].lower()
                if field_name == "output_folder":
                    output_folder_og = fields[2]
                if field_name == "linemask_file_folder_location":
                    linemask_file_og = fields[2]
                if field_name == "segment_file_folder_location":
                    segment_file_og = fields[2]
                if field_name == "spec_input_path":
                    spec_input_path = fields[2]
                    #if obs_location is not None:
                    #    spec_input_path = obs_location
                if field_name == "fitlist_input_folder":
                    fitlist_input_folder = fields[2]
                if field_name == "atmosphere_type":
                    atmosphere_type = fields[2]
                if field_name == "mode":
                    fitting_mode = fields[2].lower()
                if field_name == "include_molecules":
                    include_molecules = fields[2]
                if field_name == "nlte":
                    nlte_flag = fields[2].lower()
                    if nlte_flag == "true":
                        nlte_flag = True
                    else:
                        nlte_flag = False
                if field_name == "fit_microturb":  # Yes No Input
                    fit_microturb = fields[2]
                if field_name == "fit_macroturb":  # Yes No Input
                    if fields[2].lower() == "yes":
                        fit_macroturb = True
                    else:
                        fit_macroturb = False
                    if fields[2].lower() == "input":
                        input_macro = True
                    else:
                        input_macro = False
                if field_name == "fit_rotation":
                    if fields[2].lower() == "yes":
                        fit_rotation = True
                    else:
                        fit_rotation = False
                if field_name == "element":
                    elements_to_fit = []
                    for i in range(len(fields) - 2):
                        elements_to_fit.append(fields[2 + i])
                    elem_to_fit = np.asarray(elements_to_fit)
                    if "Fe" in elements_to_fit:
                        fit_met = True
                    else:
                        fit_met = False
                    nelement = len(elem_to_fit)
                if field_name == "linemask_file":
                    linemask_file = fields[2]
                if field_name == "wavelength_minimum":
                    lmin = float(fields[2])
                if field_name == "wavelength_maximum":
                    lmax = float(fields[2])
                if field_name == "wavelength_delta":
                    ldelta = float(fields[2])
                if field_name == "resolution":
                    resolution = float(fields[2])
                if field_name == "macroturbulence":
                    macroturb_input = float(fields[2])
                if field_name == "rotation":
                    rotation = float(fields[2])
                if field_name == "input_file":
                    fitlist = fields[2]
                if field_name == "output_file":
                    output = fields[2]
            line = fp.readline()
    #output_data = np.loadtxt(os.path.join(output_folder_location, output), dtype=str)

    if fitting_mode != "lbl":
        raise ValueError("Non-lbl fitting methods are not supported yet")

    output_elem_column = f"Fe_H"

    for i in range(nelement):
        # Spectra.elem_to_fit[i] = element name
        elem_name = elem_to_fit[i]
        if elem_name != "Fe":
            output_elem_column += f"\t{elem_name}_Fe"

    #names = f"#specname\twave_center\twave_start\twave_end\tDoppler_Shift_add_to_RV\t{output_elem_column}\tMicroturb\tMacroturb\trotation\tchi_squared\tew"
    filenames_output_folder: list[str] = get_all_file_names_in_a_folder(output_folder_location)

    filenames_output_folder_convolved = []
    for filename in filenames_output_folder:
        if "_convolved.spec" in filename:
            filenames_output_folder_convolved.append(os.path.join(output_folder_location, filename))

    with open(os.path.join(output_folder_location, output), 'r') as output_file_reading:
        output_file_lines = output_file_reading.readlines()

    # Extract the header and data lines
    output_file_header = output_file_lines[0].strip().split('\t')
    output_file_header[0] = output_file_header[0].replace("#", "")
    output_file_data_lines = [line.strip().split() for line in output_file_lines[1:]]

    if len(output_file_data_lines) == 1:
        output_file_data_lines = [output_file_data_lines]

    # Create a DataFrame from the processed data
    output_file_df = pd.DataFrame(output_file_data_lines, columns=output_file_header)

    # Convert columns to appropriate data types
    output_file_df = output_file_df.apply(pd.to_numeric, errors='ignore')

    specname_fitlist = np.loadtxt(os.path.join(fitlist_input_folder, fitlist), dtype=str, unpack=True, usecols=(0))
    rv_fitlist = np.loadtxt(os.path.join(fitlist_input_folder, fitlist), dtype=float, unpack=True, usecols=(1))
    if specname_fitlist.ndim == 0:
        specname_fitlist = np.array([specname_fitlist])
        rv_fitlist = np.array([rv_fitlist])

    config_dict = {}
    config_dict["filenames_output_folder"]: list[dir] = filenames_output_folder_convolved
    config_dict["linemask_location"]: str = os.path.join(linemask_file_og, linemask_file)
    config_dict["observed_spectra_location"]: str = spec_input_path
    config_dict["specname_fitlist"]: np.ndarray = specname_fitlist
    config_dict["rv_fitlist"]: np.ndarray = rv_fitlist
    config_dict["output_folder_location"] = output_folder_location
    config_dict["output_file_df"] = output_file_df

    return config_dict

def plot_one_star(config_dict: dict, name_of_spectra_to_plot: str, plot_title=True, save_figure=None, xlim=None, ylim=None, font_size=None):
    # unpack the config dict into separate variables
    filenames_output_folder: list[dir] = config_dict["filenames_output_folder"]
    observed_spectra_location: str = config_dict["observed_spectra_location"]
    linemask_location: str = config_dict["linemask_location"]
    specname_fitlist: np.ndarray = config_dict["specname_fitlist"]
    rv_fitlist: np.ndarray = config_dict["rv_fitlist"]
    output_file_df: pd.DataFrame = config_dict["output_file_df"]

    # tries to find the index where the star name is contained in the output folder name. since we do not expect the star name to be exactly the same, we just try to find indices where given name is PART of the output folder names
    # E.g. given arr = ['abc', 'def', 'ghi'], if we try to use name = 'ef', we get index 1 as return, since it is contained within 'def'
    indices_to_plot = np.where(np.char.find(filenames_output_folder, name_of_spectra_to_plot) != -1)[0]
    if len(indices_to_plot) > 1:
        # Warning if part of several specnames, just in case
        print(f"Warning, several specnames were found with that name {name_of_spectra_to_plot}, using first one")
    if len(indices_to_plot) == 0:
        raise ValueError(f"Could not find {name_of_spectra_to_plot} in the spectra to plot")

    # Take first occurrence of the name, hopefully the only one
    index_to_plot = indices_to_plot[0]

    # get the name of the fitted and observed spectra
    filename_fitted_spectra = filenames_output_folder[index_to_plot]
    filename_observed_spectra = filename_fitted_spectra.replace("result_spectrum_", "").replace("_convolved.spec", "").replace(os.path.join(config_dict["output_folder_location"], ""), "")

    # find where output results have the spectra (can be several lines if there are several lines fitted for each star)
    #output_results_correct_specname_indices = np.where(output_results_specname == filename_observed_spectra)[0]
    df_correct_specname_indices = output_file_df["specname"] == filename_observed_spectra

    # find RV in the fitlist that was input into the star
    if filename_observed_spectra not in specname_fitlist:
        raise ValueError(f"{filename_observed_spectra} not found in the fitlist names, which are {specname_fitlist}")
    rv_index = np.where(specname_fitlist == filename_observed_spectra)[0][0]
    rv = rv_fitlist[rv_index]

    # loads fitted and observed wavelength and flux
    wavelength, flux = np.loadtxt(filename_fitted_spectra, dtype=float, unpack=True)  # normalised flux fitted
    wavelength_observed, flux_observed = np.loadtxt(os.path.join(observed_spectra_location, filename_observed_spectra), dtype=float, unpack=True, usecols=(0, 1)) # normalised flux observed

    # loads the linemask
    linemask_center_wavelengths, linemask_left_wavelengths, linemask_right_wavelengths = np.loadtxt(linemask_location, dtype=float, comments=";", usecols=(0, 1, 2), unpack=True)

    # sorts linemask, just like in TSFitPy
    if linemask_center_wavelengths.size > 1:
        linemask_center_wavelengths = np.array(sorted(linemask_center_wavelengths))
        linemask_left_wavelengths = np.array(sorted(linemask_left_wavelengths))
        linemask_right_wavelengths = np.array(sorted(linemask_right_wavelengths))
    elif linemask_center_wavelengths.size == 1:
        linemask_center_wavelengths = np.array([linemask_center_wavelengths])
        linemask_left_wavelengths = np.array([linemask_left_wavelengths])
        linemask_right_wavelengths = np.array([linemask_right_wavelengths])

    # makes a separate plot for each line
    for linemask_center_wavelength, linemask_left_wavelength, linemask_right_wavelength in zip(linemask_center_wavelengths, linemask_left_wavelengths, linemask_right_wavelengths):
        # finds in the output results, which of the wavelengths are equal to the linemask. Comparison is done using argmin to minimise risk of comparing floats. As downside, there is no check if line is actually the same
        output_result_index_to_plot = (np.abs(output_file_df[df_correct_specname_indices]["wave_center"] - linemask_center_wavelength)).argmin()

        # this is the fitted rv in this case then
        fitted_rv = output_file_df[df_correct_specname_indices]["Doppler_Shift_add_to_RV"].values[output_result_index_to_plot]

        # other fitted values
        fitted_chisqr = output_file_df[df_correct_specname_indices]["chi_squared"].values[output_result_index_to_plot]
        column_names = output_file_df.columns.values
        if "_Fe" in column_names[6]:
            abund_column_name = column_names[6]
        else:
            abund_column_name = column_names[5]
        fitted_abund = output_file_df[df_correct_specname_indices][abund_column_name].values[output_result_index_to_plot]
        fitted_ew = output_file_df[df_correct_specname_indices]["ew"].values[output_result_index_to_plot]

        # Doppler shift is RV correction + fitted rv for the line. Corrects observed wavelength for it
        doppler = fitted_rv + rv
        wavelength_observed_rv = apply_doppler_correction(wavelength_observed, doppler)

        abund_column_name = f"[{abund_column_name.replace('_', '/')}]"

        if plot_title:
            plt.title(f"{abund_column_name}={float(f'{fitted_abund:.3g}'):g}; EW={float(f'{fitted_ew:.3g}'):g}; χ2={float(f'{fitted_chisqr:.3g}'):g}")
        plt.plot(wavelength, flux, color='red')
        plt.scatter(wavelength_observed_rv, flux_observed, color='black', marker='o', linewidths=0.5)
        # xlimit is wavelength left/right +/- 0.3 AA
        if xlim is not None:
            plt.xlim(xlim)
        else:
            plt.xlim(linemask_left_wavelength - 0.3, linemask_right_wavelength + 0.3)
        if ylim is not None:
            plt.ylim(ylim)
        else:
            plt.ylim(0, 1.05)
        # plot x-ticks without scientific notation
        plt.ticklabel_format(useOffset=False)
        # change font size
        if font_size is not None:
            plt.rcParams.update({'font.size': font_size})
        plt.plot([linemask_left_wavelength, linemask_left_wavelength], [0, 2], color='green', alpha=0.2)
        plt.plot([linemask_right_wavelength, linemask_right_wavelength], [0, 2], color='green', alpha=0.2)
        plt.plot([linemask_center_wavelength, linemask_center_wavelength], [0, 2], color='grey', alpha=0.35)
        plt.xlabel("Wavelength [Å]")
        plt.ylabel("Normalised flux")
        if save_figure is not None:
            # save figure without cutting off labels
            plt.savefig(f"{str(linemask_center_wavelength)}_{save_figure}", bbox_inches='tight')
        plt.show()
        plt.close()

def plot_scatter_df_results(df_results: pd.DataFrame, x_axis_column: str, y_axis_column: str, xlim=None, ylim=None, **pltargs):
    plt.scatter(df_results[x_axis_column], df_results[y_axis_column], **pltargs)
    plt.xlabel(x_axis_column)
    plt.ylabel(y_axis_column)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()
    plt.close()

def plot_density_df_results(df_results: pd.DataFrame, x_axis_column: str, y_axis_column: str, xlim=None, ylim=None, **pltargs):
    if np.size(x_axis_column) == 1:
        print("Only one point is found, so doing normal scatter plot")
        plot_scatter_df_results(df_results, x_axis_column, y_axis_column, xlim=xlim, ylim=ylim, **pltargs)
        return

    # creates density map for the plot
    x_array = df_results[x_axis_column]
    y_array = df_results[y_axis_column]
    xy_point_density = np.vstack([x_array, y_array])
    z_point_density = gaussian_kde(xy_point_density)(xy_point_density)
    idx_sort = z_point_density.argsort()
    x_plot, y_plot, z_plot = x_array[idx_sort], y_array[idx_sort], z_point_density[idx_sort]

    density = plt.scatter(x_plot, y_plot, c=z_plot, zorder=-1, vmin=0, **pltargs)

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.colorbar(density)
    plt.xlabel(x_axis_column)
    plt.ylabel(y_axis_column)
    plt.show()
    plt.close()


def plot_histogram_df_results(df_results: pd.DataFrame, x_axis_column: str, xlim=None, ylim=None, **pltargs):
    plt.hist(df_results[x_axis_column], **pltargs)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(x_axis_column)
    plt.ylabel("Count")
    plt.show()
    plt.close()


def get_average_of_table(df_results: pd.DataFrame, rv_limits=None, chi_sqr_limits=None, abund_limits=None, abund_to_limit=None,
                         macroturb_limits=None, microturb_limits=None, rotation_limits=None, ew_limits=None,
                         print_columns=None):
    """rv_results = df_results["Doppler_Shift_add_to_RV"]
    microturb_results = df_results["Microturb"]
    macroturb_results = df_results["Macroturb"]
    rotation_results = df_results["rotation"]
    chi_squared_results = df_results["chi_squared"]
    ew_results = df_results["ew"]
    microturb_results = df_results["Microturb"]"""

    if rv_limits is not None:
        df_results = df_results[(df_results["Doppler_Shift_add_to_RV"] >= min(rv_limits)) & (df_results["Doppler_Shift_add_to_RV"] <= max(rv_limits))]
    if chi_sqr_limits is not None:
        df_results = df_results[(df_results["chi_squared"] >= min(chi_sqr_limits)) & (df_results["chi_squared"] <= max(chi_sqr_limits))]
    if macroturb_limits is not None:
        df_results = df_results[(df_results["Macroturb"] >= min(macroturb_limits)) & (df_results["Macroturb"] <= max(macroturb_limits))]
    if microturb_limits is not None:
        df_results = df_results[(df_results["Microturb"] >= min(microturb_limits)) & (df_results["Microturb"] <= max(microturb_limits))]
    if rotation_limits is not None:
        df_results = df_results[(df_results["rotation"] >= min(rotation_limits)) & (df_results["rotation"] <= max(rotation_limits))]
    if ew_limits is not None:
        df_results = df_results[(df_results["ew"] >= min(ew_limits)) & (df_results["ew"] <= max(ew_limits))]
    if abund_limits is not None and abund_to_limit is not None:
        for abund_limit, one_abund_to_limit in zip(abund_limits, abund_to_limit):
            df_results = df_results[(df_results[one_abund_to_limit] >= min(abund_limit)) & (df_results[one_abund_to_limit] <= max(abund_limit))]

    columns = df_results.columns.values
    for column in columns:
        if column not in ["specname", "wave_center", "wave_start", "wave_end"]:
            # go through each unique specname and get the average of the column
            unique_specnames = df_results["specname"].unique()
            for specname in unique_specnames:
                print(f"Specname: {specname}")
                if print_columns is not None:
                    if column in print_columns:
                        print(f"The mean value of the '{column}' column is: {df_results[specname][column].mean()} pm {df_results[specname][column].std() / np.sqrt(df_results[specname][column].size)}")
                else:
                    print(f"The mean value of the '{column}' column is: {df_results[specname][column].mean()} pm {df_results[specname][column].std() / np.sqrt(df_results[specname][column].size)}")
                #print(f"The median value of the '{column}' column is: {df_results[column].median()}")
                #print(f"The std value of the '{column}' column is: {df_results[column].std()}")


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

def plot_synthetic_data(turbospectrum_paths, teff, logg, met, vmic, lmin, lmax, ldelta, atmosphere_type, nlte_flag,
                        elements_in_nlte, element_abundances, include_molecules, resolution=0, macro=0, rotation=0, verbose=False):
    for element in element_abundances:
        element_abundances[element] += met
    temp_directory = f"../temp_directory_{datetime.datetime.now().strftime('%b-%d-%Y-%H-%M-%S')}__{np.random.random(1)[0]}/"

    for path in turbospectrum_paths:
        turbospectrum_paths[path] = check_if_path_exists(turbospectrum_paths[path])

    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)

    if atmosphere_type == "1D":
        model_atmosphere_grid_path = os.path.join(turbospectrum_paths["model_atmosphere_grid_path"], "1D", "")
        model_atmosphere_list = model_atmosphere_grid_path + "model_atmosphere_list.txt"
    elif atmosphere_type == "3D":
        model_atmosphere_grid_path = os.path.join(turbospectrum_paths["model_atmosphere_grid_path"], "3D", "")
        model_atmosphere_list = model_atmosphere_grid_path + "model_atmosphere_list.txt"

    model_temperatures, model_logs, model_mets, marcs_value_keys, marcs_models, marcs_values = fetch_marcs_grid(
        model_atmosphere_list, TurboSpectrum.marcs_parameters_to_ignore)

    nlte_config = ConfigParser()
    nlte_config.read(os.path.join(turbospectrum_paths["departure_file_path"], "nlte_filenames.cfg"))

    depart_bin_file_dict, depart_aux_file_dict, model_atom_file_dict = {}, {}, {}

    for element in elements_in_nlte:
        if atmosphere_type == "1D":
            bin_config_name, aux_config_name = "1d_bin", "1d_aux"
        else:
            bin_config_name, aux_config_name = "3d_bin", "3d_aux"
        depart_bin_file_dict[element] = nlte_config[element][bin_config_name]
        depart_aux_file_dict[element] = nlte_config[element][aux_config_name]
        model_atom_file_dict[element] = nlte_config[element]["atom_file"]

    aux_file_length_dict = {}
    if nlte_flag:
        for element in model_atom_file_dict:
            aux_file_length_dict[element] = len(
                np.loadtxt(os.path.join(turbospectrum_paths["departure_file_path"], depart_aux_file_dict[element]), dtype='str'))

    today = datetime.datetime.now().strftime("%b-%d-%Y-%H-%M-%S")  # used to not conflict with other instances of fits
    today = f"{today}_{np.random.random(1)[0]}"
    line_list_path_trimmed = os.path.join(f"{temp_directory}", "linelist_for_fitting_trimmed", "")
    line_list_path_trimmed = os.path.join(line_list_path_trimmed, "all", today, '')

    print("Trimming")
    create_window_linelist([lmin - 4], [lmax + 4], turbospectrum_paths["line_list_path"], line_list_path_trimmed, include_molecules, False)
    print("Trimming done")

    line_list_path_trimmed = os.path.join(line_list_path_trimmed, "0", "")

    ts = TurboSpectrum(
        turbospec_path=turbospectrum_paths["turbospec_path"],
        interpol_path=turbospectrum_paths["interpol_path"],
        line_list_paths=line_list_path_trimmed,
        marcs_grid_path=model_atmosphere_grid_path,
        marcs_grid_list=model_atmosphere_list,
        model_atom_path=turbospectrum_paths["model_atom_path"],
        departure_file_path=turbospectrum_paths["departure_file_path"],
        aux_file_length_dict=aux_file_length_dict,
        model_temperatures=model_temperatures,
        model_logs=model_logs,
        model_mets=model_mets,
        marcs_value_keys=marcs_value_keys,
        marcs_models=marcs_models,
        marcs_values=marcs_values)

    ts.configure(t_eff=teff, log_g=logg, metallicity=met,
                 turbulent_velocity=vmic, lambda_delta=ldelta, lambda_min=lmin - 3, lambda_max=lmax + 3,
                 free_abundances=element_abundances, temp_directory=temp_directory, nlte_flag=nlte_flag, verbose=verbose,
                 atmosphere_dimension=atmosphere_type, windows_flag=False, segment_file=None,
                 line_mask_file=None, depart_bin_file=depart_bin_file_dict,
                 depart_aux_file=depart_aux_file_dict, model_atom_file=model_atom_file_dict)
    print("Running TS")
    ts.run_turbospectrum_and_atmosphere()
    print("TS completed")
    try:
        wave_mod_orig, flux_norm_mod_orig = np.loadtxt('{}spectrum_00000000.spec'.format(temp_directory),
                                                                  usecols=(0, 1), unpack=True)
        wave_mod_filled = wave_mod_orig
        flux_norm_mod_filled = flux_norm_mod_orig

        if resolution != 0.0:
            wave_mod_conv, flux_norm_mod_conv = conv_res(wave_mod_filled, flux_norm_mod_filled, resolution)
        else:
            wave_mod_conv = wave_mod_filled
            flux_norm_mod_conv = flux_norm_mod_filled

        if macro != 0.0:
            wave_mod_macro, flux_norm_mod_macro = conv_macroturbulence(wave_mod_conv, flux_norm_mod_conv, macro)
        else:
            wave_mod_macro = wave_mod_conv
            flux_norm_mod_macro = flux_norm_mod_conv

        if rotation != 0.0:
            wave_mod, flux_norm_mod = conv_rotation(wave_mod_macro, flux_norm_mod_macro, rotation)
        else:
            wave_mod = wave_mod_macro
            flux_norm_mod = flux_norm_mod_macro

        plt.plot(wave_mod, flux_norm_mod)
        plt.xlim(lmin - 0.2, lmax + 0.2)
        plt.ylim(0, 1.05)
        plt.xlabel("Wavelength")
        plt.ylabel("Normalised flux")
    except FileNotFoundError:
        print("TS failed")
        wave_mod, flux_norm_mod = np.array([]), np.array([])
    shutil.rmtree(temp_directory)
    #shutil.rmtree(line_list_path_trimmed)  # clean up trimmed line list

    return wave_mod, flux_norm_mod


def remove_bad_lines(output_data):
    linemask_location = output_data["linemask_location"]
    output_file_df: pd.DataFrame = output_data["output_file_df"]

    # loads the linemask
    linemask_center_wavelengths = np.loadtxt(linemask_location, dtype=float, comments=";", usecols=0, unpack=True)

    # sorts linemask, just like in TSFitPy
    if linemask_center_wavelengths.size > 1:
        linemask_center_wavelengths = np.array(sorted(linemask_center_wavelengths))
    elif linemask_center_wavelengths.size == 1:
        linemask_center_wavelengths = np.array([linemask_center_wavelengths])

    mask = output_file_df["wave_center"].isin(linemask_center_wavelengths)
    output_file_df = output_file_df[mask]
    output_file_df.to_csv(os.path.join(output_data["output_folder_location"], "output_good_lines"), sep=' ', index=False, header=True)
    return output_file_df


if __name__ == '__main__':
    # CHANGE NEXT TWO LINES
    configuration_file_location: str = "../input_files/tsfitpy_input_configuration_ba_oliver_y_nlte_fenlte.txt"  # CHANGE
    output_folder_location: str = "../output_files/Mar-27-2023-14-11-24_0.23697863971919042_y_nlte_fe_nlte_oliverba/"  # CHANGE
    output_folder_location: str = "../output_files/test"  # CHANGE
    # loads all data from config file and output
    config_dict = load_output_data(configuration_file_location, output_folder_location)
    output_results_pd_df = config_dict["output_file_df"]  # Pandas dataframe for your own use
    print("Column names are:")
    print(output_results_pd_df.columns.values)  # Column names if you want to plot them
    # CHANGE NEXT LINE
    star_name_to_plot: str = "00"  # CHANGE
    # plots all fitted lines for the requested star
    plot_one_star(config_dict, star_name_to_plot)