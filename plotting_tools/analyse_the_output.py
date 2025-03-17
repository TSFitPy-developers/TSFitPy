from __future__ import annotations

import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from plotting_tools.scripts_for_plotting import load_output_data

matplotlib.use("MacOSX")


# Created by storm at 06.11.24

def get_average_abundance(output_folder_location, remove_errors, remove_warnings, chisqr_limit, ew_limits):
    config_dict = load_output_data(output_folder_location)
    output_file_df = config_dict["output_file_df"]
    fitted_element = config_dict["fitted_element"]
    if fitted_element != "Fe":
        fitted_element = f"{fitted_element}_Fe"
        x_value_plot = "Fe_H"
    else:
        fitted_element = "Fe_H"
        x_value_plot = "wave_center"

    # remove any rows
    if remove_errors:
        output_file_df = output_file_df[output_file_df['flag_error'] == 0]
    if remove_warnings:
        output_file_df = output_file_df[output_file_df['flag_warning'] == 0]
    output_file_df = output_file_df[output_file_df['chi_squared'] <= chisqr_limit]
    output_file_df = output_file_df[output_file_df['ew_just_line'] >= ew_limits[0]]
    output_file_df = output_file_df[output_file_df['ew_just_line'] <= ew_limits[1]]
    # output_file_df that have wave_center > 5500
    #output_file_df = output_file_df[output_file_df['wave_center'] < 5500]

    output_file_df.reset_index(drop=True, inplace=True)

    output_df = pd.DataFrame()
    # new columns: specname, x_value, y_value
    output_df["specname"] = []
    output_df[x_value_plot] = []
    output_df[fitted_element] = []
    output_df[f"{fitted_element}_err"] = []
    output_df["vmac"] = []
    output_df["vmac_err"] = []
    output_df["vsini"] = []
    output_df["vsini_err"] = []
    output_df["ew_line"] = []

    # get the data from the fitted spectra
    specnames = np.unique(output_file_df['specname'].values)

    for specname in specnames:
        # find all rows with the same specname
        indices = np.where(output_file_df['specname'] == specname)[0]

        if np.size(indices) > 0:
            # new row in the dataframe
            output_df.loc[len(output_df)] = [specname, np.mean(output_file_df[x_value_plot][indices]),
                                             np.mean(output_file_df[fitted_element][indices]),
                                             np.std(output_file_df[fitted_element][indices]),
                                             np.mean(output_file_df["Macroturb"][indices]),
                                             np.std(output_file_df["Macroturb"][indices]),
                                             np.mean(output_file_df["rotation"][indices]),
                                             np.std(output_file_df["rotation"][indices]),
                                             np.mean(output_file_df["ew_just_line"][indices]),]
        print(f"specname: {specname}, x_value: {np.mean(output_file_df[x_value_plot][indices])}, y_value: {np.mean(output_file_df[fitted_element][indices])}")

    return output_df

if __name__ == '__main__':
    folder_path = "/Users/storm/PhD_2025/01.20 Victor/2025-03-03-10-29-03_NLTE_Fe_1D_combined_with_old_good"
    output_df = get_average_abundance(folder_path, False, False, 100, (1, 500))
    print(output_df)
    output_df.to_csv(f"{folder_path}/average_abundance.csv", index=False)
    #plt.plot(x_values, y_values, 'o')
    #plt.show()