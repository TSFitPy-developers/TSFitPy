from __future__ import annotations
from scripts.loading_configs import TSFitPyConfig
import numpy as np


def run_tsfitpy(config_file_location):
    parsed_config_file = TSFitPyConfig(config_file_location)


def run_tsfitpy_with_config(config: TSFitPyConfig):
    # step 1: load linemasks
    # step 2: load elements
    linemasks = config.linemask_file
    elements = config.elements_to_fit

    for linemask in linemasks:
        line_wavelength, line_begin, line_end = np.loadtxt(linemask, comments=";", usecols=(0, 1, 2), unpack=True, dtype=float)





if __name__ == '__main__':
    raise RuntimeError("This file is not meant to be run as main. Please run TSFitPy/main.py instead.")  # this is a module
