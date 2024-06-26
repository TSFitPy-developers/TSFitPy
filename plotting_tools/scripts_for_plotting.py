from __future__ import annotations

import datetime
import shutil
from collections import OrderedDict
from configparser import ConfigParser
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.stats import gaussian_kde
from warnings import warn
from scripts.convolve import conv_macroturbulence, conv_rotation, conv_res
from scripts.create_window_linelist_function import create_window_linelist
from scripts.turbospectrum_class_nlte import TurboSpectrum
from scripts.m3dis_class import M3disCall
from scripts.synthetic_code_class import SyntheticSpectrumGenerator
from scripts.synthetic_code_class import fetch_marcs_grid
from scripts.TSFitPy import (output_default_configuration_name, output_default_fitlist_name,
                             output_default_linemask_name)
from scripts.auxiliary_functions import (calculate_equivalent_width, apply_doppler_correction, import_module_from_path,
                                         combine_linelists)
from scripts.loading_configs import SpectraParameters, TSFitPyConfig
from scripts.solar_abundances import periodic_table


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
            print(f"Loading config from {os.path.join(old_variable, output_default_configuration_name.replace('.cfg', '.txt'))}")
            config_file_location = os.path.join(old_variable, output_default_configuration_name.replace(".cfg", ".txt"))
            output_folder_location = old_variable
        elif os.path.isfile(os.path.join(old_variable, output_default_configuration_name)):
            # trying new way of loading: first variable is output folder with config in it
            print(f"Loading config from {os.path.join(old_variable, output_default_configuration_name)}")
            config_file_location = os.path.join(old_variable, output_default_configuration_name)
            output_folder_location = old_variable
        else:
            # this was an old way of loading. first variable: config file, second variable: output folder
            print(f"Loading config from {output_folder_location}")
            config_file_location = output_folder_location
            output_folder_location = old_variable
    else:
        # new way of loading: first variable is output folder with config in it
        if os.path.isfile(os.path.join(output_folder_location, output_default_configuration_name.replace(".cfg", ".txt"))):
            config_file_location = os.path.join(output_folder_location, output_default_configuration_name.replace(".cfg", ".txt"))
        else:
            config_file_location = os.path.join(output_folder_location, output_default_configuration_name)

    tsfitpy_config = TSFitPyConfig(config_file_location, "none")
    tsfitpy_config.load_config(check_valid_path=False)
    tsfitpy_config.validate_input(check_valid_path=False)

    if tsfitpy_config.fitting_mode not in ["lbl", "teff", 'vmic', 'logg']:
        raise ValueError("Non-lbl fitting methods are not supported yet")

    output_elem_column = f"Fe_H"

    for i in range(tsfitpy_config.nelement):
        # Spectra.elem_to_fit[i] = element name
        elem_name = tsfitpy_config.elements_to_fit[i]
        if elem_name != "Fe":
            output_elem_column += f"\t{elem_name}_Fe"

    #names = f"#specname\twave_center\twave_start\twave_end\tDoppler_Shift_add_to_RV\t{output_elem_column}\tMicroturb\tMacroturb\trotation\tchi_squared\tew"
    filenames_output_folder: list[str] = get_all_file_names_in_a_folder(output_folder_location)

    filenames_output_folder_convolved = []
    for filename in filenames_output_folder:
        if "_convolved.spec" in filename:
            filenames_output_folder_convolved.append(os.path.join(output_folder_location, filename))

    with open(os.path.join(output_folder_location, tsfitpy_config.output_filename), 'r') as output_file_reading:
        output_file_lines = output_file_reading.readlines()

    # Extract the header and data lines
    output_file_header = output_file_lines[0].strip().split('\t')
    output_file_header[0] = output_file_header[0].replace("#", "")
    output_file_data_lines = [line.strip().split() for line in output_file_lines[1:]]

    #if len(output_file_data_lines) == 1:
    #    output_file_data_lines = output_file_data_lines[0]

    # Create a DataFrame from the processed data
    output_file_df = pd.DataFrame(output_file_data_lines, columns=output_file_header)

    # Convert columns to appropriate data types
    # TODO: except for columns with names "flag error" and "flag warning"
    def safe_to_numeric(val):
        try:
            return pd.to_numeric(val)
        except ValueError:
            return val

    output_file_df = output_file_df.applymap(safe_to_numeric)

    # check if fitlist exists and if not load old fitlist
    output_filist_location = os.path.join(output_folder_location, output_default_fitlist_name)
    if os.path.isfile(output_filist_location):
        fitlist_path = output_filist_location
    else:
        fitlist_path = os.path.join(tsfitpy_config.fitlist_input_path, tsfitpy_config.input_fitlist_filename)

    fitlist = SpectraParameters(fitlist_path, True)
    specname_fitlist = fitlist.spectra_parameters_df["specname"].values
    rv_fitlist = fitlist.spectra_parameters_df["rv"].values

    if specname_fitlist.ndim == 0:
        specname_fitlist = np.array([specname_fitlist])
        rv_fitlist = np.array([rv_fitlist])

    config_dict = {}
    config_dict["filenames_output_folder"]: list[dir] = filenames_output_folder_convolved
    linemask_output_location = os.path.join(output_folder_location, output_default_linemask_name)
    if os.path.isfile(linemask_output_location):
        config_dict["linemask_location"]: str = linemask_output_location
    else:
        config_dict["linemask_location"]: str = os.path.join(tsfitpy_config.linemasks_path, tsfitpy_config.linemasks_files)
    config_dict["observed_spectra_location"]: str = tsfitpy_config.spectra_input_path
    config_dict["specname_fitlist"]: np.ndarray = specname_fitlist
    config_dict["rv_fitlist"]: np.ndarray = rv_fitlist
    config_dict["output_folder_location"] = output_folder_location
    config_dict["output_file_df"] = output_file_df
    config_dict["fitted_element"] = tsfitpy_config.elements_to_fit[0]
    config_dict["fitting_method"] = tsfitpy_config.fitting_mode
    config_dict["parsed_fitlist"] = fitlist
    config_dict["vmac_input_bool"] = tsfitpy_config.vmac_input
    config_dict["vmic_input_bool"] = tsfitpy_config.vmic_input
    config_dict["rotation_input_bool"] = tsfitpy_config.rotation_input
    config_dict["resolution_constant"] = tsfitpy_config.resolution

    return config_dict

def plot_one_star(config_dict: dict, name_of_spectra_to_plot: str, plot_title=True, save_figure=None, xlim=None, ylim=None, font_size=None, remove_errors=False, remove_warnings=False):
    try:
        # unpack the config dict into separate variables
        filenames_output_folder: list[dir] = config_dict["filenames_output_folder"]
        observed_spectra_location: str = config_dict["observed_spectra_location"]
        linemask_location: str = config_dict["linemask_location"]
        specname_fitlist: np.ndarray = config_dict["specname_fitlist"]
        rv_fitlist: np.ndarray = config_dict["rv_fitlist"]
        output_file_df: pd.DataFrame = config_dict["output_file_df"]
        output_folder_location: str = config_dict["output_folder_location"]

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
        # check if file is located in the output folder, if not, load from the original folder
        if os.path.isfile(os.path.join(output_folder_location, filename_observed_spectra)):
            wavelength_observed, flux_observed = np.loadtxt(os.path.join(output_folder_location, filename_observed_spectra), dtype=float, unpack=True, usecols=(0, 1))  # normalised flux observed
        else:
            wavelength_observed, flux_observed = np.loadtxt(os.path.join(observed_spectra_location, filename_observed_spectra), dtype=float, unpack=True, usecols=(0, 1)) # normalised flux observed

        # sort the observed spectra, just like in TSFitPy
        if wavelength_observed.size > 1:
            sorted_obs_wavelength_index = np.argsort(wavelength_observed)
            wavelength_observed, flux_observed = wavelength_observed[sorted_obs_wavelength_index], flux_observed[sorted_obs_wavelength_index]

            sorted_wavelength_index = np.argsort(wavelength)
            wavelength, flux = wavelength[sorted_wavelength_index], flux[sorted_wavelength_index]


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

            if remove_errors:
                # check if flag error is not 0, then return
                if output_file_df[df_correct_specname_indices]["flag_error"].values[output_result_index_to_plot] != 0 and \
                        output_file_df[df_correct_specname_indices]["flag_error"].values[output_result_index_to_plot] != "0":
                    continue
            if remove_warnings:
                # check if flag warning is not 0, then return
                if output_file_df[df_correct_specname_indices]["flag_warning"].values[output_result_index_to_plot] != 0 and \
                        output_file_df[df_correct_specname_indices]["flag_warning"].values[output_result_index_to_plot] != "0":
                    continue

            # this is the fitted rv in this case then
            fitted_rv = output_file_df[df_correct_specname_indices]["Doppler_Shift_add_to_RV"].values[output_result_index_to_plot]

            # other fitted values
            fitted_chisqr = output_file_df[df_correct_specname_indices]["chi_squared"].values[output_result_index_to_plot]
            fitted_element = config_dict['fitted_element']
            if config_dict["fitting_method"] == "lbl" or config_dict["fitting_method"] == "vmic":
                if fitted_element != "Fe":
                    abund_column_name = f"[{fitted_element}/Fe]"
                    column_name = f"{fitted_element}_Fe"
                else:
                    abund_column_name = "[Fe/H]"
                    column_name = "Fe_H"
                fitted_abund = output_file_df[df_correct_specname_indices][column_name].values[output_result_index_to_plot]
            elif config_dict["fitting_method"] == "teff":
                abund_column_name = "teff"
                fitted_abund = output_file_df[df_correct_specname_indices]["Teff"].values[output_result_index_to_plot]
            elif config_dict["fitting_method"] == "logg":
                abund_column_name = "logg"
                fitted_abund = output_file_df[df_correct_specname_indices]["logg"].values[output_result_index_to_plot]

            fitted_ew = output_file_df[df_correct_specname_indices]["ew"].values[output_result_index_to_plot]

            # Doppler shift is RV correction + fitted rv for the line. Corrects observed wavelength for it
            doppler = fitted_rv + rv
            wavelength_observed_rv = apply_doppler_correction(wavelength_observed, doppler)

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
    except (ValueError, IndexError, FileNotFoundError) as e:
        print(f"Error: {e}")

def plot_scatter_df_results(df_results: pd.DataFrame, x_axis_column: str, y_axis_column: str, xlim=None, ylim=None,
                            color='black', invert_x_axis=False, invert_y_axis=False, **pltargs):
    if color in df_results.columns.values:
        pltargs['c'] = df_results[color]
        pltargs['cmap'] = 'viridis'
        pltargs['vmin'] = df_results[color].min()
        pltargs['vmax'] = df_results[color].max()
        plot_colorbar = True
    else:
        pltargs['color'] = color
        plot_colorbar = False
    plt.scatter(df_results[x_axis_column], df_results[y_axis_column], **pltargs)
    plt.xlabel(x_axis_column)
    plt.ylabel(y_axis_column)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if invert_x_axis:
        plt.gca().invert_xaxis()
    if invert_y_axis:
        plt.gca().invert_yaxis()
    if plot_colorbar:
        # colorbar with label
        plt.colorbar(label=color)
    plt.show()
    plt.close()

def plot_density_df_results(df_results: pd.DataFrame, x_axis_column: str, y_axis_column: str, xlim=None, ylim=None,
                            invert_x_axis=False, invert_y_axis=False, **pltargs):
    if np.size(df_results[x_axis_column]) == 1:
        print("Only one point is found, so doing normal scatter plot")
        plot_scatter_df_results(df_results, x_axis_column, y_axis_column, xlim=xlim, ylim=ylim,
                                invert_x_axis=invert_x_axis, invert_y_axis=invert_y_axis, **pltargs)
        return
    try:
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
        if invert_x_axis:
            plt.gca().invert_xaxis()
        if invert_y_axis:
            plt.gca().invert_yaxis()
        plt.show()
        plt.close()
    except LinAlgError:
        print("LinAlgError, so doing normal scatter plot")
        plot_scatter_df_results(df_results, x_axis_column, y_axis_column, xlim=xlim, ylim=ylim,
                                invert_x_axis=invert_x_axis, invert_y_axis=invert_y_axis, **pltargs)


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
    unique_specnames = df_results["specname"].unique()
    for specname in unique_specnames:
        print(f"Specname: {specname}")
        for column in columns:
            if column not in ["specname", "wave_center", "wave_start", "wave_end"]:
                # go through each unique specname and get the average of the column
                    if print_columns is not None:
                        if column in print_columns:
                            # take only rows with the specname
                            print(f"The mean value of the '{column}' column is: {df_results[df_results['specname'] == specname][column].mean()} pm {df_results[df_results['specname'] == specname][column].std() / np.sqrt(df_results[df_results['specname'] == specname][column].size)}")
                    else:
                        print(f"The mean value of the '{column}' column is: {df_results[df_results['specname'] == specname][column].mean()} pm {df_results[df_results['specname'] == specname][column].std() / np.sqrt(df_results[df_results['specname'] == specname][column].size)}")
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
                        elements_in_nlte, element_abundances, include_molecules, resolution=0, macro=0, rotation=0,
                        verbose=False, return_unnorm_flux=False, do_matplotlib_plot=True):
    for element in element_abundances:
        element_abundances[element] += met
    temp_directory = f"../temp_directory/temp_directory_{datetime.datetime.now().strftime('%b-%d-%Y-%H-%M-%S')}__{np.random.random(1)[0]}/"

    temp_directory = os.path.join(os.getcwd(), temp_directory, "")

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

    depart_bin_file_dict, depart_aux_file_dict, model_atom_file_dict = {}, {}, {}
    aux_file_length_dict = {}

    if nlte_flag:
        nlte_config = ConfigParser()
        nlte_config.read(os.path.join(turbospectrum_paths["departure_file_path"], "nlte_filenames.cfg"))

        for element in elements_in_nlte:
            if atmosphere_type == "1D":
                bin_config_name, aux_config_name = "1d_bin", "1d_aux"
            else:
                bin_config_name, aux_config_name = "3d_bin", "3d_aux"
            depart_bin_file_dict[element] = nlte_config[element][bin_config_name]
            depart_aux_file_dict[element] = nlte_config[element][aux_config_name]
            model_atom_file_dict[element] = nlte_config[element]["atom_file"]

        for element in model_atom_file_dict:
            aux_file_length_dict[element] = len(np.loadtxt(os.path.join(turbospectrum_paths["departure_file_path"], depart_aux_file_dict[element]), dtype='str'))

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
    wave_mod_orig, flux_norm_mod_orig, flux_unnorm = ts.synthesize_spectra()
    print("TS completed")
    if wave_mod_orig is not None:
        if np.size(wave_mod_orig) != 0.0:
            try:
                wave_mod_filled = wave_mod_orig
                flux_norm_mod_filled = flux_norm_mod_orig

                if len(wave_mod_orig) > 0:
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

                    if do_matplotlib_plot:
                        plt.plot(wave_mod, flux_norm_mod)
                        plt.xlim(lmin - 0.2, lmax + 0.2)
                        plt.ylim(0, 1.05)
                        plt.xlabel("Wavelength")
                        plt.ylabel("Normalised flux")
                else:
                    print('TS failed')
                    wave_mod, flux_norm_mod = np.array([]), np.array([])
                    flux_unnorm = np.array([])
            except (FileNotFoundError, ValueError, IndexError) as e:
                print(f"TS failed: {e}")
                wave_mod, flux_norm_mod = np.array([]), np.array([])
                flux_unnorm = np.array([])
        else:
            print('TS failed')
            wave_mod, flux_norm_mod = np.array([]), np.array([])
            flux_unnorm = np.array([])
    else:
        print('TS failed')
        wave_mod, flux_norm_mod = np.array([]), np.array([])
        flux_unnorm = np.array([])
    shutil.rmtree(temp_directory)
    #shutil.rmtree(line_list_path_trimmed)  # clean up trimmed line list
    if return_unnorm_flux:
        return wave_mod, flux_norm_mod, flux_unnorm
    else:
        return wave_mod, flux_norm_mod


def read_element_data(lines):
    i = 0
    elements_data = []
    while i < len(lines):
        line_parts = lines[i].split()
        if len(line_parts) == 0:
            i += 1
            continue
        if line_parts[0] == "'":
            atomic_num = (line_parts[1])
        else:
            atomic_num = (line_parts[0])
        ionization = int(line_parts[-2])
        num_lines = int(line_parts[-1])

        element_name = lines[i + 1].strip().replace("'", "").replace("NLTE", "").replace("LTE", "")

        for _ in range(num_lines):
            i += 1
            data_line = lines[i + 1]
            wavelength, loggf = float(data_line.split()[0]), float(data_line.split()[2])
            #elements_data.append((element_name, atomic_num, ionization, wavelength, loggf))
            elements_data.append((wavelength, f"{element_name}", loggf))

        i += 2

    return elements_data

def find_elements(elements_data, left_wavelength, right_wavelength, loggf_threshold):
    filtered_elements = []
    for element_data in elements_data:
        wavelength, element_name, loggf = element_data
        if left_wavelength <= wavelength <= right_wavelength and loggf >= loggf_threshold:
            filtered_elements.append(element_data)

    sorted_elements = sorted(filtered_elements, key=lambda x: x[0])  # Sort by wavelength

    return sorted_elements

    #for element_data in sorted_elements:
    #    element_name, atomic_num, ionization, wavelength, loggf = element_data
    #    print(element_name.replace("'", "").replace("NLTE", "").replace("LTE", ""), atomic_num, wavelength, loggf)

def plot_synthetic_data_m3dis(m3dis_paths, teff, logg, met, vmic, lmin, lmax, ldelta, atmosphere_type, atmos_format, n_nu, mpi_cores,
                              hash_table_size, nlte_flag, element_in_nlte, element_abundances, snap, dims, nx, ny, nz,
                              nlte_iterations_max, nlte_convergence_limit, resolution=0, macro=0, rotation=0, verbose=False, return_unnorm_flux=False,
                              m3dis_package_name="m3dis", return_parsed_linelist=False, loggf_limit_parsed_linelist=-5.0,
                              plot_output=False):
    for element in element_abundances:
        element_abundances[element] += met
    temp_directory = f"../temp_directory/temp_directory_{datetime.datetime.now().strftime('%b-%d-%Y-%H-%M-%S')}__{np.random.random(1)[0]}/"
    # convert temp directory to absolute path
    temp_directory = os.path.join(os.getcwd(), temp_directory, "")

    for path in m3dis_paths:
        if ((atmosphere_type != "3D" and path != "3D_atmosphere_path") or atmosphere_type == "3D") and not path == "nlte_config_path":
            m3dis_paths[path] = check_if_path_exists(m3dis_paths[path])

    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)

    if atmosphere_type == "1D":
        model_atmosphere_grid_path = os.path.join(m3dis_paths["model_atmosphere_grid_path"], "1D", "")
        model_atmosphere_list = model_atmosphere_grid_path + "model_atmosphere_list.txt"

        model_temperatures, model_logs, model_mets, marcs_value_keys, marcs_models, marcs_values = fetch_marcs_grid(
            model_atmosphere_list, TurboSpectrum.marcs_parameters_to_ignore)
    elif atmosphere_type == "3D":
        model_atmosphere_grid_path = None
        model_atmosphere_list = None

        model_temperatures, model_logs, model_mets, marcs_value_keys, marcs_models, marcs_values = None, None, None, None, None, None

    depart_bin_file_dict, depart_aux_file_dict, model_atom_file_dict = {}, {}, {}

    if nlte_flag:
        nlte_config = ConfigParser()
        nlte_config.read(m3dis_paths["nlte_config_path"])

        model_atom_file_dict[element_in_nlte] = nlte_config[element_in_nlte]["atom_file"]

    today = datetime.datetime.now().strftime("%b-%d-%Y-%H-%M-%S")  # used to not conflict with other instances of fits
    today = f"{today}_{np.random.random(1)[0]}"
    line_list_path_trimmed = os.path.join(f"{temp_directory}", "linelist_for_fitting_trimmed", "")
    line_list_path_trimmed = os.path.join(line_list_path_trimmed, "all", today, '')

    print("Trimming")
    create_window_linelist([lmin - 2], [lmax + 2], m3dis_paths["line_list_path"], line_list_path_trimmed, False, False, do_hydrogen=False)
    # if m3dis, then combine all linelists into one
    # go into line_list_path_trimmed and each folder and combine all linelists into one in each of the folders
    parsed_linelist_data = combine_linelists(line_list_path_trimmed, return_parsed_linelist=return_parsed_linelist)
    parsed_elements_sorted_info = None
    if return_parsed_linelist:
        parsed_model_atom_data = []
        for i in range(len(parsed_linelist_data)):
            parsed_model_atom_data.extend(parsed_linelist_data[i].split("\n"))

        left_wavelength = lmin  # change this to change the range of wavelengths to print
        right_wavelength = lmax
        loggf_threshold = loggf_limit_parsed_linelist           # change this to change the threshold for loggf
        elements_data = read_element_data(parsed_model_atom_data)
        parsed_elements_sorted_info = find_elements(elements_data, left_wavelength, right_wavelength, loggf_threshold)

    print("Trimming done")

    line_list_path_trimmed = os.path.join(line_list_path_trimmed, "0", "")

    module_path = os.path.join(m3dis_paths["m3dis_path"], f"{m3dis_package_name}/__init__.py")
    m3dis_python_module = import_module_from_path("m3dis", module_path)

    m3dis = M3disCall(
        m3dis_path=m3dis_paths["m3dis_path"],
        interpol_path=None,
        line_list_paths=line_list_path_trimmed,
        marcs_grid_path=model_atmosphere_grid_path,
        marcs_grid_list=model_atmosphere_list,
        model_atom_path=m3dis_paths["model_atom_path"],
        departure_file_path=None,
        aux_file_length_dict=None,
        model_temperatures=model_temperatures,
        model_logs=model_logs,
        model_mets=model_mets,
        marcs_value_keys=marcs_value_keys,
        marcs_models=marcs_models,
        marcs_values=marcs_values,
        m3dis_python_module=m3dis_python_module,
        n_nu=n_nu,
        hash_table_size=hash_table_size,
        mpi_cores=mpi_cores,
        iterations_max=nlte_iterations_max,
        convlim=nlte_convergence_limit,
        snap=snap,
        dims=dims,
        nx=nx,
        ny=ny,
        nz=nz
    )

    m3dis.configure(t_eff=teff, log_g=logg, metallicity=met,
                 turbulent_velocity=vmic, lambda_delta=ldelta, lambda_min=lmin - 3, lambda_max=lmax + 3,
                 free_abundances=element_abundances, temp_directory=f"{temp_directory}", nlte_flag=nlte_flag, verbose=verbose,
                 atmosphere_dimension=atmosphere_type, windows_flag=False, segment_file=None,
                 line_mask_file=None, model_atom_file=model_atom_file_dict, atmos_format_3d=atmos_format, atmosphere_path_3d_model=m3dis_paths["3D_atmosphere_path"])
    m3dis.use_precomputed_depart = False
    print("Running m3dis")

    wave_mod_orig, flux_norm_mod_orig, flux_unnorm = m3dis.synthesize_spectra()
    print("m3dis completed")
    if wave_mod_orig is not None:
        if np.size(wave_mod_orig) != 0.0:
            try:
                wave_mod_filled = wave_mod_orig
                flux_norm_mod_filled = flux_norm_mod_orig

                if len(wave_mod_orig) > 0:
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
                    if plot_output:
                        plt.plot(wave_mod, flux_norm_mod)
                        plt.xlim(lmin - 0.2, lmax + 0.2)
                        plt.ylim(0, 1.05)
                        plt.xlabel("Wavelength")
                        plt.ylabel("Normalised flux")
                else:
                    print('m3dis failed')
                    wave_mod, flux_norm_mod = np.array([]), np.array([])
                    flux_unnorm = np.array([])
            except (FileNotFoundError, ValueError, IndexError) as e:
                print(f"m3dis failed: {e}")
                wave_mod, flux_norm_mod = np.array([]), np.array([])
                flux_unnorm = np.array([])
        else:
            print('m3dis failed')
            wave_mod, flux_norm_mod = np.array([]), np.array([])
            flux_unnorm = np.array([])
    else:
        print('m3dis failed')
        wave_mod, flux_norm_mod = np.array([]), np.array([])
        flux_unnorm = np.array([])
    shutil.rmtree(temp_directory)
    #shutil.rmtree(line_list_path_trimmed)  # clean up trimmed line list

    result = [wave_mod, flux_norm_mod]

    if return_unnorm_flux:
        result.append(flux_unnorm)
    if return_parsed_linelist:
        result.append(parsed_elements_sorted_info)

    return tuple(result)


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

def load_output_grid(output_folder):
    # load pandas dataframe .csv file
    output_file_df = pd.read_csv(os.path.join(output_folder, "spectra_parameters.csv"), sep=',', header=0)

    return output_file_df

def plot_synthetic_spectra_from_grid(input_folder, spectra_name, xlim=None, ylim=None, plt_show=True, **kwargs):
    wavelength, flux = np.loadtxt(os.path.join(input_folder, spectra_name), usecols=(0, 1), unpack=True, dtype=float)
    plt.plot(wavelength, flux, **kwargs)
    plt.title(spectra_name)
    plt.xlabel("Wavelength")
    plt.ylabel("Normalised flux")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if plt_show:
        plt.show()
        plt.close()

def plot_many_spectra_same_plot(input_folder, spectra_names, xlim=None, ylim=None, **kwargs):
    for spectra_name in spectra_names:
        plot_synthetic_spectra_from_grid(input_folder, spectra_name, xlim=xlim, ylim=ylim, plt_show=False, **kwargs)
    plt.show()
    plt.close()


class Star:
    # this class will load abundances from several different files and load them into a class for later use
    # it will also load linelist such that we get atomic information about different lines
    def __init__(self, name, input_folders: list, linelist_folder):
        # if input_folders is a string, then convert it to a list
        if isinstance(input_folders, str):
            input_folders = [input_folders]
        self.name = name
        self.linelist_folder = linelist_folder
        # get names of all files in the linelist folder
        self.linelist_filenames = get_all_file_names_in_a_folder(linelist_folder)
        # add the path to the filenames
        for i in range(len(self.linelist_filenames)):
            self.linelist_filenames[i] = os.path.join(linelist_folder, self.linelist_filenames[i])

        molecules_flag = False

        def read_linelist(filenames):
            data = {}
            for filename in filenames:
                with open(filename) as fp:
                    # so that we dont read full file if we are not sure that we use it (if it is a molecule)
                    first_line: str = fp.readline()

                    fields = first_line.strip().split()
                    sep = '.'
                    element = fields[0] + fields[1]
                    elements = element.split(sep, 1)[0]
                    # opens each file, reads first row, if it is long enough then it is molecule. If fitting molecules, then
                    # keep it, otherwise ignore molecules

                    if len(elements) > 3 and molecules_flag or len(elements) <= 3:
                        # now read the whole file
                        lines_file: list[str] = fp.readlines()
                        # append the first line to the lines_file
                        lines_file.insert(0, first_line)
                        line_number_read_for_element: int = 0
                        line_number_read_file: int = 0
                        total_lines_in_file: int = len(lines_file)
                        while line_number_read_file < total_lines_in_file:  # go through all line
                            line: str = lines_file[line_number_read_file]
                            fields: list[str] = line.strip().split()

                            element_name = f"{fields[0]}{fields[1]}"

                            if element_name == "'01.000000'":  # find out whether it is hydrogen
                                hydrogen_element: bool = True
                            else:
                                hydrogen_element: bool = False
                            if len(fields[0]) > 1:  # save the first two lines of an element for the future
                                number_of_lines_element: int = int(fields[3])
                            else:
                                number_of_lines_element: int = int(fields[4])
                            line_number_read_file += 1
                            line: str = lines_file[line_number_read_file]
                            elem_line_2_to_save: str = f"{line.strip()}"  # second line of the element
                            element_name_string = elem_line_2_to_save.split()[0].replace("'", "")
                            ionisation_stage = elem_line_2_to_save.split()[1].replace("'", "")

                            element_name_string = f"{element_name_string}_{ionisation_stage}"

                            if element_name_string not in data:
                                data[element_name_string] = []

                            # now we are reading the element's wavelength and stuff
                            line_number_read_file += 1
                            # lines_for_element = lines_file[line_number_read_file:number_of_lines_element+line_number_read_file]
                            while line_number_read_for_element < number_of_lines_element:
                                line_stripped: str = lines_file[
                                    line_number_read_for_element + line_number_read_file].strip()
                                data[element_name_string].append(line_stripped.split())
                                line_number_read_for_element += 1

                            line_number_read_file: int = number_of_lines_element + line_number_read_file
                            line_number_read_for_element = 0

            # Convert lists to DataFrames
            data_new = {}
            for key in data:
                data[key] = [item[:3] for item in data[key]]
                #print([item[:3] for item in data[key]])

                # Split the element and ionisation stage
                element, ionisation_stage = key.split("_")

                # Create a new DataFrame from the existing data using float
                df = pd.DataFrame(data[key], columns=['wavelength', 'ep', 'loggf']).astype(float)

                # Add a new column for the ionisation stage, setting it to the current ionisation stage for all rows
                df['ionisation_stage'] = ionisation_stage

                # If the element is not in the dictionary, add it
                if element not in data_new:
                    data_new[element] = df
                else:
                    # append to the existing DataFrame using concat
                    data_new[element] = pd.concat([data_new[element], df], ignore_index=True)

            return data_new

        # Usage
        self.parsed_linelist = read_linelist(self.linelist_filenames)

        # load each element using dataframe:
        self.elemental_data = {"wavelength": {}, "ew": {}, "abund": {}, "chisqr": {}, "rv": {}, "vmic": {}, "vmac": {}, "rotation": {}, "flag_error": {}, "flag_warning": {}}
        for input_folder in input_folders:
            config_dict = load_output_data(input_folder)
            fitted_element = config_dict["fitted_element"]
            # find the name of the star in df and only use that one
            df = config_dict["output_file_df"]
            mask = df["specname"] == name
            df = df[mask]
            # get the line wavelength
            line_wavelengths = df["wave_center"].values
            self.elemental_data["wavelength"][fitted_element] = np.asarray(line_wavelengths)
            # get the line equivalent width
            line_ew = df["ew"].values
            self.elemental_data["ew"][fitted_element] = np.asarray(line_ew)
            # get the line abundance
            if fitted_element == "Fe":
                line_abund = df["Fe_H"].values
            else:
                line_abund = df[f"{fitted_element}_Fe"].values
            self.elemental_data["abund"][fitted_element] = np.asarray(line_abund)
            # get the line chi squared
            line_chisqr = df["chi_squared"].values
            self.elemental_data["chisqr"][fitted_element] = np.asarray(line_chisqr)
            # get the line rv
            line_rv = df["Doppler_Shift_add_to_RV"].values
            self.elemental_data["rv"][fitted_element] = np.asarray(line_rv)
            # get the line microturbulence
            line_microturbulence = df["Microturb"].values
            self.elemental_data["vmic"][fitted_element] = np.asarray(line_microturbulence)
            # get the line macroturbulence
            line_macroturbulence = df["Macroturb"].values
            self.elemental_data["vmac"][fitted_element] = np.asarray(line_macroturbulence)
            # get the line rotation
            line_rotation = df["rotation"].values
            self.elemental_data["rotation"][fitted_element] = np.asarray(line_rotation)
            # get the line flag error
            line_flag_error = df["flag_error"].values
            self.elemental_data["flag_error"][fitted_element] = np.asarray(line_flag_error)
            # get the line flag warning
            line_flag_warning = df["flag_warning"].values
            self.elemental_data["flag_warning"][fitted_element] = np.asarray(line_flag_warning)

    def get_line_data(self, element, wavelengths, column, ionisation_stage=None, tolerance=0.1):
        element_data = self.parsed_linelist[element]
        result = []
        ionisation_stages = []

        # Ensure wavelengths is a list
        if not isinstance(wavelengths, list) and not isinstance(wavelengths, np.ndarray):
            wavelengths = [wavelengths]

        for wavelength in wavelengths:
            # Calculate the absolute difference between each wavelength and the provided wavelength
            differences = np.abs(element_data['wavelength'] - wavelength)

            # Find the index of the smallest difference that is within the tolerance
            try:
                idx_min_difference = differences[differences <= tolerance].idxmin()

                # If a matching row was found
                if not np.isnan(idx_min_difference):
                    line = element_data.loc[idx_min_difference]
                    if ionisation_stage is not None:
                        # choose the ionisation stage
                        line = line[line['ionisation_stage'] == ionisation_stage]
                    result.append(line[column])
                    ionisation_stages.append(line['ionisation_stage'])
                else:
                    result.append(None)
            except ValueError:
                result.append(None)

        return result, ionisation_stages

    def remove_data_bad_flags(self, data_list, element, flag_error, flag_warning):
        if flag_error:
            mask = self.elemental_data["flag_error"][element] == 0
            for i in range(len(data_list)):
                data_list[i] = data_list[i][mask]
        if flag_warning:
            mask = self.elemental_data["flag_warning"][element] == 0
            for i in range(len(data_list)):
                data_list[i] = data_list[i][mask]
        return data_list

    def plot_fit_parameters_vs_abundance(self, fit_parameter, element, abund_limits=None, remove_flag_error=True, remove_flag_warning=False):
        allowed_params = ["wavelength", "ew"]
        if fit_parameter not in allowed_params:
            raise ValueError(f"Fit parameter must be {allowed_params}, not {fit_parameter}")
        corresponding_labels = {"wavelength": "Wavelength [Å]", "ew": "Equivalent width"}
        x_data = self.elemental_data[fit_parameter][element]
        y_data = self.elemental_data["abund"][element]
        # if abund_limits is not None, then remove the lines that are outside the limits
        x_data, y_data = self.remove_data_bad_flags([x_data, y_data], element, remove_flag_error, remove_flag_warning)
        if abund_limits is not None:
            mask = (y_data >= abund_limits[0]) & (y_data <= abund_limits[1])
            x_data = x_data[mask]
            y_data = y_data[mask]
        plt.scatter(x_data, y_data)
        plt.xlabel(corresponding_labels[fit_parameter])
        if element == "Fe":
            plt.ylabel("[Fe/H]")
        else:
            plt.ylabel(f"[{element}/Fe]")
        plt.title(f"{self.name}")
        plt.show()

    def plot_vs_abundance(self, element, column, abund_limits=None):
        data, ionisation_stages = self.get_line_data(element, self.elemental_data["wavelength"][element], column)
        stellar_param_data = self.elemental_data["abund"][element]

        # if abund_limits is not None, then remove the lines that are outside the limits
        ionisation_stages = np.array(ionisation_stages)
        data = np.array(data)
        if abund_limits is not None:
            stellar_param_data = np.array(stellar_param_data)
            mask = (stellar_param_data >= abund_limits[0]) & (stellar_param_data <= abund_limits[1])
            data = data[mask]
            stellar_param_data = stellar_param_data[mask]
            ionisation_stages = ionisation_stages[mask]

        # find those with ionisation stage 1
        mask = ionisation_stages == "I"
        if np.sum(mask) != 0:
            data_neutral = data[mask]
            stellar_param_data_neutral = stellar_param_data[mask]
            plt.scatter(data_neutral, stellar_param_data_neutral, label="Neutral", color='black')

        # find those with any other ionisation stage
        mask = ionisation_stages != "I"
        if np.sum(mask) != 0:
            data_other = data[mask]
            stellar_param_data_other = stellar_param_data[mask]
            plt.scatter(data_other, stellar_param_data_other, label="Other", color='red')

        plt.xlabel(column)
        if element == "Fe":
            plt.ylabel("[Fe/H]")
        else:
            plt.ylabel(f"[{element}/Fe]")
        plt.title(f"{self.name}")
        plt.legend()
        plt.show()

    def plot_ep_vs_abundance(self, element, abund_limits=None):
        self.plot_vs_abundance(element, 'ep', abund_limits)

    def plot_loggf_vs_abundance(self, element, abund_limits=None):
        self.plot_vs_abundance(element, 'loggf', abund_limits)

    def plot_abundance_plot(self, abund_limits, fontsize=16):
        # plots all abundances as a function of atomic number
        # get all elements
        elements = self.elemental_data["abund"].keys()
        # get atomic numbers
        atomic_numbers = []
        # use periodic table to get atomic numbers
        for element in elements:
            atomic_numbers.append(periodic_table.index(element) + 1)

        # get abundances
        abundances = []
        for element in elements:
            abundance_element = self.elemental_data["abund"][element]
            if len(abundance_element) > 1:
                # if abund_limits is not None, then remove the lines that are outside the limits
                if abund_limits is not None:
                    abundance_element = np.array(abundance_element)
                    mask = (abundance_element >= abund_limits[0]) & (abundance_element <= abund_limits[1])
                    abundance_element = abundance_element[mask]
                abundance_element = np.mean(abundance_element)
            else:
                abundance_element = abundance_element[0]
            # check that abundance_element is not nan
            if np.isnan(abundance_element):
                abundance_element = None
            abundances.append(abundance_element)
        # plot
        plt.scatter(atomic_numbers, abundances, color='black', zorder=3)
        for i, element in enumerate(elements):
            # every second element has y offset of 0.1
            label_y_def = max(abundances) + 0.4
            if i % 2 == 0:
                label_y = label_y_def + 0.1
            else:
                label_y = label_y_def - 0.1
            plt.text(atomic_numbers[i], label_y, element, horizontalalignment='center',
                     verticalalignment='center', color='black', fontsize=fontsize)
        plt.ylabel('[X/Fe]', fontsize=fontsize)
        plt.xlabel('Atomic Number (Z)', fontsize=fontsize)
        plt.title(f'Chemical Abundance {self.name}', fontsize=fontsize)
        atomic_numbers = [0, 63]
        # grid should repeat every 2 x-values, but x-ticks themselves should be every 10 x-values
        plt.xticks(ticks=np.arange(min(atomic_numbers), max(atomic_numbers) + 1, 10))  # Set x-ticks every 10
        plt.grid(which='major', linestyle='-', linewidth=0.5)  # Major grid lines
        plt.grid(which='minor', axis='x', linestyle=':', linewidth=0.5)  # Minor grid lines
        plt.minorticks_on()  # Turn on the minor ticks
        ax = plt.gca()  # Get the current Axes instance
        ax.set_xticks(np.arange(min(atomic_numbers), max(atomic_numbers) + 1, 2),
                      minor=True)  # Set minor x-ticks every 2
        plt.axhline(0, color='black', linewidth=0.5, zorder=1)
        # change font size of ticks
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.tight_layout()
        # plot y=0 line
        plt.axhline(0, color='black', linewidth=2.0, zorder=1)
        plt.ylim(min(abundances) - 0.4, max(abundances) + 0.6)

        plt.show()

    def get_average_abundances(self, ew_limits=None, chi_sqr_limits=None, remove_flag_error=True, remove_flag_warning=False):
        # gets average abundances by element
        # get all elements
        elements = self.elemental_data["abund"].keys()
        # get average abundances
        average_abundances = {}
        stdev_abundances = {}
        for element in elements:
            abundances_element = self.elemental_data["abund"][element]
            ews_element = self.elemental_data["ew"][element]
            chi_sqrs_element = self.elemental_data["chisqr"][element]
            # if remove_flag_error is True, then remove the lines that have flag_error == 1
            flag_error_element = self.elemental_data["flag_error"][element]
            flag_warning_element = self.elemental_data["flag_warning"][element]
            abundances_element = np.array(abundances_element)
            ews_element = np.array(ews_element)
            chi_sqrs_element = np.array(chi_sqrs_element)
            flag_error_element = np.array(flag_error_element)
            flag_warning_element = np.array(flag_warning_element)
            if remove_flag_error:
                mask = flag_error_element == 0
                abundances_element = abundances_element[mask]
                ews_element = ews_element[mask]
                chi_sqrs_element = chi_sqrs_element[mask]
                flag_warning_element = flag_warning_element[mask]
            if remove_flag_warning:
                mask = flag_warning_element == 0
                abundances_element = abundances_element[mask]
                ews_element = ews_element[mask]
                chi_sqrs_element = chi_sqrs_element[mask]
            # if ew_limits is not None, then remove the lines that are outside the limits
            if ew_limits is not None:
                mask = (ews_element >= ew_limits[0]) & (ews_element <= ew_limits[1])
                abundances_element = abundances_element[mask]
                chi_sqrs_element = chi_sqrs_element[mask]
            # if chi_sqr_limits is not None, then remove the lines that are outside the limits
            if chi_sqr_limits is not None:
                mask = (chi_sqrs_element >= chi_sqr_limits[0]) & (chi_sqrs_element <= chi_sqr_limits[1])
                abundances_element = abundances_element[mask]
            if len(abundances_element) > 1:
                mean_abundances_element = np.mean(abundances_element)
                stdev_abundance_element = np.std(abundances_element) / np.sqrt(len(abundances_element))
            elif len(abundances_element) == 1:
                mean_abundances_element = abundances_element[0]
                stdev_abundance_element = 0
            else:
                mean_abundances_element = None
                stdev_abundance_element = None
            average_abundances[f"{element}_mean"] = mean_abundances_element
            stdev_abundances[f"{element}_stdev"] = stdev_abundance_element

        # Create an ordered dictionary
        ordered_dict = OrderedDict()

        # Iterate over elements in average_abundances
        for element in average_abundances:
            # Add mean and stdev for each element to the ordered dictionary
            ordered_dict[element] = average_abundances[element]
            # Get the element name without '_mean'
            element_name = element.replace("_mean", "")
            ordered_dict[element_name + "_stdev"] = stdev_abundances[element_name + "_stdev"]

        # Create a DataFrame from the ordered dictionary
        df = pd.DataFrame([ordered_dict])

        # add first column as specname usign self.name
        df.insert(0, "specname", self.name)

        return df


def get_average_abundance_all_stars(input_folders, linelist_path, ew_limits=None, chi_sqr_limits=None, remove_flag_error=True, remove_flag_warning=False):
    # if input_folders is a string, then convert it to a list
    if isinstance(input_folders, str):
        input_folders = [input_folders]
    config_dict = load_output_data(input_folders[0])
    # get all spectra names from the first folder
    spectra_names = config_dict["output_file_df"]["specname"].unique()
    # create Star objects for each spectra name and add them to a list
    stars = []
    for spectra_name in spectra_names:
        stars.append(Star(spectra_name, input_folders, linelist_path))
    # get all abundances for different spectra and combine into one dataframe
    df = pd.concat([star.get_average_abundances(ew_limits=ew_limits, chi_sqr_limits=chi_sqr_limits, remove_flag_error=remove_flag_error, remove_flag_warning=remove_flag_warning) for star in stars], axis=0)
    return df



if __name__ == '__main__':
    #test_star = Star("150429001101153.spec", ["../output_files/Nov-17-2023-00-23-55_0.1683492858486244_NLTE_Fe_1D/"], "../input_files/linelists/linelist_for_fitting/")
    #test_star.plot_fit_parameters_vs_abundance("ew", "Fe", abund_limits=(-3, 3))
    #test_star.plot_ep_vs_abundance("Fe")
    #test_star.plot_loggf_vs_abundance("Fe", abund_limits=(-3, 3))
    #test_star.plot_abundance_plot(abund_limits=(-3, 3))
    #print(test_star.get_average_abundances())

    test = get_average_abundance_all_stars(["../output_files/Nov-17-2023-00-23-55_0.1683492858486244_NLTE_Fe_1D/", "../output_files/Nov-17-2023-00-23-55_0.1683492858486244_NLTE_Fe_1D/"], "../input_files/linelists/linelist_for_fitting/")
    print(test)