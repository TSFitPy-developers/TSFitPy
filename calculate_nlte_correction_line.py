from __future__ import annotations
from sys import argv
from scripts.calculate_nlte_correction_line import run_nlte_corrections
import datetime
import numpy as np


if __name__ == '__main__':
    abundance: float = 0.0  # abundance of element in LTE [X/Fe]; scaled with metallicity

    if len(argv) > 1:  # when calling the program, can now add extra argument with location of config file, easier to call
        config_location = argv[1]
    else:
        config_location = "../input_files/tsfitpy_input_configuration.cfg"  # location of config file

    output_folder_title_date = datetime.datetime.now().strftime(
        "%b-%d-%Y-%H-%M-%S")  # used to not conflict with other instances of fits
    output_folder_title_date = f"{output_folder_title_date}_{np.random.random(1)[0]}"  # in case if someone calls the function several times per second
    print(f"Start of the fitting: {output_folder_title_date}")


    run_nlte_corrections(config_location, output_folder_title_date)