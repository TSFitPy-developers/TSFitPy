from __future__ import annotations

from scripts.TSFitPy import run_tsfitpy
import scipy
from sys import argv
import datetime
import numpy as np

if __name__ == '__main__':
    major_version_scipy, minor_version_scipy, patch_version_scipy = scipy.__version__.split(".")
    if int(major_version_scipy) < 1 or (int(major_version_scipy) == 1 and int(minor_version_scipy) < 7) or (
            int(major_version_scipy) == 1 and int(minor_version_scipy) == 7 and int(patch_version_scipy) == 0):
        raise ImportError(f"Scipy has to be at least version 1.7.1, otherwise bounds are not considered in minimisation. "
                          f"That will lead to bad fits. Please update to scipy 1.7.1 OR higher. Your version: "
                          f"{scipy.__version__}")

    if len(argv) > 1:   # when calling the program, can now add extra argument with location of config file, easier to call
        config_location = argv[1]
    else:
        config_location = "./input_files/tsfitpy_input_configuration.cfg"  # location of config file
    if len(argv) > 2:  # when calling the program, can now add extra argument with location of observed spectra, easier to call
        obs_location = argv[2]
    else:
        obs_location = None  # otherwise defaults to the input one 
    print(config_location)
    output_folder_title_date = datetime.datetime.now().strftime("%b-%d-%Y-%H-%M-%S")  # used to not conflict with other instances of fits
    output_folder_title_date = f"{output_folder_title_date}_{np.random.random(1)[0]}"     # in case if someone calls the function several times per second
    print(f"Start of the fitting: {output_folder_title_date}")
    try:
        run_tsfitpy(output_folder_title_date, config_location, obs_location)
        print("Fitting completed")
    except KeyboardInterrupt:
        print(f"KeyboardInterrupt detected. Terminating job.")  #TODO: cleanup temp folders here?
    finally:
        print(f"End of the fitting: {datetime.datetime.now().strftime('%b-%d-%Y-%H-%M-%S')}")
