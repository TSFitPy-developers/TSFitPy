from __future__ import annotations

import os

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

    v_mturb: float = 0

    if teff >= 5000.:
        v_mturb = 1.05 + 2.51e-4 * (teff - t0) + 1.5e-7 * (teff - t0) * (teff - t0) - 0.14 * (logg - g0) - 0.005 * (
                logg - g0) * (logg - g0) + 0.05 * met + 0.01 * met * met
    elif teff < 5000. and logg >= 3.5:
        v_mturb = 1.05 + 2.51e-4 * (teff - t0) + 1.5e-7 * (5250. - t0) * (5250. - t0) - 0.14 * (logg - g0) - 0.005 * (
                logg - g0) * (logg - g0) + 0.05 * met + 0.01 * met * met
    elif teff < 5500. and logg < 3.5:
        v_mturb = 1.25 + 4.01e-4 * (teff - t0) + 3.1e-7 * (teff - t0) * (teff - t0) - 0.14 * (logg - g0) - 0.005 * (
                logg - g0) * (logg - g0) + 0.05 * met + 0.01 * met * met

    if v_mturb <= 0.0:
        print("error in calculating micro turb, setting it to 1.0")
        return 1.0

    return v_mturb


def calculate_equivalent_width(fit_wavelength: np.ndarray, fit_flux: np.ndarray, left_bound: float, right_bound: float) -> float:
    line_func = interp1d(fit_wavelength, fit_flux, kind='linear', assume_sorted=True,
                                     fill_value=1, bounds_error=False)
    total_area = (right_bound - left_bound) * 1.0   # continuum
    try:
        integration_points = fit_wavelength[np.logical_and.reduce((fit_wavelength > left_bound, fit_wavelength < right_bound))]
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


def create_segment_file(segment_size: float, line_begins_list, line_ends_list) -> tuple[np.ndarray, np.ndarray]:
    segments_left = []
    segments_right = []
    start = line_begins_list[0] - segment_size
    end = line_ends_list[0] + segment_size

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
    Dynamically imports a module or package from a given file path.

    Parameters:
    module_name (str): The name to assign to the module.
    file_path (str): The file path to the module or package.

    Returns:
    module: The imported module.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Module spec not found for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
