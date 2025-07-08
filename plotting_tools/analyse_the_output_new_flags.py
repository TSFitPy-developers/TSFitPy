from __future__ import annotations
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', )))

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from plotting_tools.scripts_for_plotting import load_output_data
import argparse
from pathlib import Path


# Created by storm at 06.11.24

def get_average_abundance(output_folder_location, remove_errors, remove_warnings, chisqr_limit, ew_limits, ew_limit_total):
    config_dict = load_output_data(output_folder_location)
    output_file_df = config_dict["output_file_df"]
    fitted_element = config_dict["fitted_element"]
    if fitted_element != "Fe":
        fitted_element = f"{fitted_element}_Fe"
        x_value_plot = "Fe_H"
    else:
        fitted_element = "Fe_H"
        x_value_plot = "wave_center"

    new_flags_df = pd.read_csv(f"{output_folder_location}/new_flags.csv")

    if remove_errors:
        output_file_df = output_file_df[output_file_df['flag_error'] == 0]
    if remove_warnings:
        output_file_df = output_file_df[output_file_df['flag_warning'] == 0]
    output_file_df = output_file_df[output_file_df['chi_squared'] <= chisqr_limit]
    output_file_df = output_file_df[output_file_df['ew_just_line'] >= ew_limits[0]]
    output_file_df = output_file_df[output_file_df['ew_just_line'] <= ew_limits[1]]
    output_file_df = output_file_df[output_file_df['ew'] <= ew_limit_total]

    # remove any rows that have a flag == 1
    # specname,linemask,extra_error columns
    for index, row in new_flags_df.iterrows():
        specname = row["specname"]
        linemask = row["linemask"]
        extra_error = row["extra_error"]
        # find the row in the output_file_df
        indices = np.where((output_file_df['specname'] == specname) & (output_file_df['wave_center'] == linemask))[0]
        if extra_error == 1:
            print(f"Removing {specname} {linemask} {output_file_df['specname'][indices].values} {output_file_df['wave_center'][indices].values}")
            output_file_df = output_file_df.drop(indices)
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
                                                np.mean(output_file_df["ew_just_line"][indices])]
        print(f"specname: {specname}, x_value: {np.mean(output_file_df[x_value_plot][indices])}, y_value: {np.mean(output_file_df[fitted_element][indices])}")

    return output_df


def main(
    folder_path: str,
    *,
    remove_errors: bool = True,
    remove_warnings: bool = False,
    chisqr_limit: float = 50.0,
    ew_limits: tuple[float, float] = (1, 400),
    ew_limit_total: float = 550.0
):
    """Run TSFitPy post-processing and write average_abundance.csv."""
    output_df = get_average_abundance(
        folder_path,
        remove_errors=remove_errors,
        remove_warnings=remove_warnings,
        chisqr_limit=chisqr_limit,
        ew_limits=ew_limits,
        ew_limit_total=ew_limit_total,
    )
    print(output_df)
    output_df.to_csv(Path(folder_path) / "average_abundance.csv", index=False)


# ----------------------------------------------------------------------
# 2.  Command-line interface (all *optional*, with sensible defaults)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # should be used together with flags from TSGuiPy. Put "new_flags.csv" in the output folder and it will remove everything that has a flag == 1 AND other limits.
    # So if you onl want to use the flags, set remove_errors and remove_warnings to False, and set chisqr_limit, ew_limits and ew_limit_total to very high values.

    # remove_errors - if True, remove all rows with flag_error != 0
    # remove_warnings - if True, remove all rows with flag_warning != 0
    # chisqr_limit - maximum chi_squared value to keep (set higher for bigger linemasks, e.g. molecular bands)
    # ew_limits - tuple with minimum and maximum equivalent width to keep for the actual line
    # ew_limit_total - maximum equivalent width to keep for the whole line (including blends)

    parser = argparse.ArgumentParser(
        description="Analyse the output of the TSFitPy fitting process."
    )

    parser.add_argument(
        "folder_path",
        nargs="?",
        default=None,
        help="Path to the output folder containing the fitting results."
             " If omitted, the hard-coded fallback value is used.",
    )

    parser.add_argument(
        "--remove-errors",
        dest="remove_errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove rows with flag_error != 0 (default: True)",
    )
    parser.add_argument(
        "--remove-warnings",
        dest="remove_warnings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove rows with flag_warning != 0 (default: True)",
    )

    parser.add_argument(
        "--chisqr-limit",
        type=float,
        default=50.0,
        help="Maximum chi_squared value to keep (default: 5.0)",
    )
    parser.add_argument(
        "--ew-limits",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(1, 400),
        help="Min and max equivalent width for the line (default: 1 200)",
    )
    parser.add_argument(
        "--ew-limit-total",
        type=float,
        default=550.0,
        help="Max equivalent width for the whole line incl. blends (default: 350)",
    )

    args = parser.parse_args()

    fallback_folder_path = "PATH_TO_YOUR_OUTPUT_FOLDER"  # ← change just this

    main(
        folder_path=args.folder_path or fallback_folder_path,
        remove_errors=args.remove_errors,
        remove_warnings=args.remove_warnings,
        chisqr_limit=args.chisqr_limit,
        ew_limits=tuple(args.ew_limits),
        ew_limit_total=args.ew_limit_total,
    )

