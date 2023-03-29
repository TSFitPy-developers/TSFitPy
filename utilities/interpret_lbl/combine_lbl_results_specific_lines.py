import numpy as np


def str_tab_topcat(value):
    """
    Converts the value to string, replaces commas by dots and adds tab at the end. Used for saving into txt file. Made to be read by TOPCAT (because of dots)
    :param value: Value to be converted
    :return: Cleaned string with tab at the end
    """
    return f"{str(value).replace(',', '.')}\t"


def save_in_txt_topcat(text, filename):
    """
    Saves text in file, separating each element in text by tab and adding a new line below it. To be read by TOPCAT because saves with dots instead of commas.
    :param text: 1D array with words to write
    :param filename: Path and filename where to save
    """
    with open(filename, 'a+') as f:
        for word in text:
            f.write(str_tab_topcat(word))
        f.write('\n')


def combine_lbl(input_location, output_location):
    output = np.loadtxt(input_location, dtype=str)
    star_names = output[:, 0]
    wave_center = output[:, 1].astype(float)
    #elem_abund = output[:, 4].astype(float)
    #doppler_shift = output[:, 5].astype(float)
    #microturb = output[:, 6].astype(float)
    macroturb = output[:, 7].astype(float)
    chi_squared = output[:, 8].astype(float)

    unique_lines = np.unique(wave_center)

    #stars_amount = np.size(np.unique(star_names))

    comment_on_top = "# line_center\tmean_chi_sqr\tmedian_chi_sqr\tst_dev_chi_sqr\tmean_macroturb\tmedian_macroturb\tst_dev_macroturb\t"
    save_in_txt_topcat([comment_on_top], output_location)

    for line_center in unique_lines:
        indices = np.where(wave_center == line_center)[0]
        star_output = [line_center, np.mean(chi_squared[indices]), np.median(chi_squared[indices]), np.std(chi_squared[indices]), np.mean(macroturb[indices]), np.median(macroturb[indices]), np.std(macroturb[indices])]
        save_in_txt_topcat(star_output, output_location)


if __name__ == '__main__':
    # TODO resturcture files
    results_location = None
    new_output_location = None

    if results_location is None or new_output_location is None:
        print("Need to set results and new output location textfiles")

    combine_lbl(results_location, new_output_location)
    