from __future__ import annotations

import os
import types

import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
import importlib.util
import sys


def create_dir(directory: str):
    """
    Creates a directory if it does not exist
    :param directory: Path to the directory that can be created (relative or not)
    """
    if not os.path.exists(directory):
        try:
            os.mkdir(directory)
        except FileNotFoundError:  # if one needs to create folders in-between
            os.makedirs(directory)  # can't recall why I just don't call this function directly


def calculate_vturb(teff: float, logg: float, met: float) -> float:
    """
    Calculates micro turbulence based on the input parameters
    :param teff: Temperature in kelvin
    :param logg: log(g) in dex units
    :param met: metallicity [Fe/H] scaled by solar
    :return: micro turbulence in km/s
    """
    t0 = 5500.
    g0 = 4.
    tlim = 5000.
    glim = 3.5

    delta_logg = logg - g0

    if logg >= glim:
        # dwarfs
        if teff >= tlim:
            # hot dwarfs
            delta_t = teff - t0
        else:
            # cool dwarfs
            delta_t = tlim - t0

        v_mturb = (1.05 + 2.51e-4 * delta_t + 1.5e-7 * delta_t**2 - 0.14 * delta_logg - 0.005 * delta_logg**2 +
                   0.05 * met + 0.01 * met**2)

    elif logg < glim:
        # giants
        delta_t = teff - t0

        v_mturb = (1.25 + 4.01e-4 * delta_t + 3.1e-7 * delta_t**2 - 0.14 * delta_logg - 0.005 * delta_logg**2 +
                   0.05 * met + 0.01 * met**2)

    return v_mturb


def calculate_equivalent_width(wavelength: np.ndarray, normalised_flux: np.ndarray, left_bound: float, right_bound: float) -> float:
    """
    Calculates the equivalent width of a line based on the input parameters
    :param wavelength: Wavelength array
    :param normalised_flux: Normalised flux array
    :param left_bound: Left bound of the line
    :param right_bound: Right bound of the line
    :return: Equivalent width of the line
    """
    # first cut wavelength and flux to the bounds
    try:
        normalised_flux = normalised_flux[np.logical_and(wavelength >= left_bound, wavelength <= right_bound)]
        wavelength = wavelength[np.logical_and(wavelength >= left_bound, wavelength <= right_bound)]
    except TypeError:
        return -9999
    line_func = interp1d(wavelength, normalised_flux, kind='linear', assume_sorted=True, fill_value=1, bounds_error=False)
    total_area = (right_bound - left_bound) * 1.0   # continuum
    try:
        integration_points = wavelength[np.logical_and.reduce((wavelength > left_bound, wavelength < right_bound))]
        area_under_line = integrate.quad(line_func, left_bound, right_bound, points=integration_points, limit=len(integration_points) * 5)
    except ValueError:
        return -9999

    return total_area - area_under_line[0]


def get_second_degree_polynomial(x: list, y: list) -> tuple[int, int, int]:
    """
    Takes a list of x and y of length 3 each and calculates perfectly fitted second degree polynomial through it.
    Returns a, b, c that are related to the function ax^2+bx+c = y
    :param x: x values, length 3
    :param y: y values, length 3
    :return a,b,c -> ax^2+bx+c = y 2nd degree polynomial
    """
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    y1 = y[0]
    y2 = y[1]
    y3 = y[2]

    a = (x1 * (y3 - y2) + x2 * (y1 - y3) + x3 * (y2 - y1)) / ((x1 - x2) * (x1 - x3) * (x2 - x3))
    b = (y2 - y1) / (x2 - x1) - a * (x1 + x2)
    c = y1 - a * x1 * x1 - b * x2

    return a, b, c


def apply_doppler_correction(wave_ob: np.ndarray, doppler: float) -> np.ndarray:
    return wave_ob / (1 + (doppler / 299792.))


def create_segment_file(segment_size: float, line_begins_list: np.ndarray[float], line_ends_list: np.ndarray[float]) -> tuple[np.ndarray[float], np.ndarray[float]]:
    segments_left = []
    segments_right = []
    start: float = line_begins_list[0] - segment_size
    end: float = line_ends_list[0] + segment_size

    for (line_left, line_right) in zip(line_begins_list, line_ends_list):
        if line_left > end + segment_size:
            segments_left.append(start)
            segments_right.append(end)
            start = line_left - segment_size
            end = line_right + segment_size
        else:
            end = line_right + segment_size

    segments_left.append(start)
    segments_right.append(end)

    return np.asarray(segments_left), np.asarray(segments_right)


def closest_available_value(target: float, options: list[float]) -> float:
    """
    Return the option from a list which most closely matches some target value.

    :param target:
        The target value that we're trying to match.
    :param options:
        The list of possible values that we can try to match to target.
    :return:
        The option value which is closest to <target>.
    """
    options = np.asarray(options)
    idx = (np.abs(options - target)).argmin()
    return options[idx]

def import_module_from_path(module_name, file_path):
    """
    Imports a module by temporarily adding its parent directory to sys.path.

    Parameters:
    module_name (str): The full dotted module name (e.g., 'm3dis.m3dis').
    file_path (str): The file path to the module.

    Returns:
    module: The imported module.
    """
    # Determine the directory containing the module
    module_dir = os.path.dirname(file_path)

    # Calculate how many levels up we need to go to get the package root
    levels_up = module_name.count('.')

    # Get the package root directory
    package_root = module_dir
    for _ in range(levels_up):
        package_root = os.path.dirname(package_root)

    # Add the package root to sys.path temporarily
    sys.path.insert(0, package_root)
    try:
        module = importlib.import_module(module_name)
    finally:
        # Remove the package root from sys.path
        sys.path.pop(0)
    return module


def combine_linelists(line_list_path_trimmed: str, combined_linelist_name: str = "combined_linelist.bsyn", return_parsed_linelist: bool = False, save_combined_linelist: bool = True):
    parsed_linelist_data = []
    for folder in os.listdir(line_list_path_trimmed):
        if os.path.isdir(os.path.join(line_list_path_trimmed, folder)):
            # go into each folder and combine all linelists into one
            combined_linelist = os.path.join(line_list_path_trimmed, folder, combined_linelist_name)
            if save_combined_linelist:
                with open(combined_linelist, "w") as combined_linelist_file:
                    for file in os.listdir(os.path.join(line_list_path_trimmed, folder)):
                        if file.endswith(".bsyn") and not file == combined_linelist_name:
                            with open(os.path.join(line_list_path_trimmed, folder, file), "r") as linelist_file:
                                read_file = linelist_file.read()
                                combined_linelist_file.write(read_file)
                                if return_parsed_linelist:
                                    parsed_linelist_data.append(read_file)
                            # delete the file
                            os.remove(os.path.join(line_list_path_trimmed, folder, file))
            else:
                for file in os.listdir(os.path.join(line_list_path_trimmed, folder)):
                    if file.endswith(".bsyn") and not file == combined_linelist_name:
                        with open(os.path.join(line_list_path_trimmed, folder, file), "r") as linelist_file:
                            try:
                                first_line: str = linelist_file.readline()
                                fields = first_line.strip().split()
                                sep = '.'
                                element = fields[0] + fields[1]
                                elements = element.split(sep, 1)[0]
                                if len(elements) <= 3:
                                    with open(os.path.join(line_list_path_trimmed, folder, file), "r") as linelist_file:
                                        read_file = linelist_file.read()
                                        # opens each file, reads first row, if it is long enough then it is molecule. If fitting molecules, then
                                        # keep it, otherwise ignore molecules
                                        if return_parsed_linelist:
                                            parsed_linelist_data.append(read_file)
                            except UnicodeDecodeError:
                                print(f"LINELIST WARNING! File {linelist_file} is not a valid linelist file")
                                continue
    if return_parsed_linelist:
        return parsed_linelist_data
