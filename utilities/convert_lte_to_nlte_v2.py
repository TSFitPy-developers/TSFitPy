import numpy as np
import re
from scripts.model_atom_class import ModelAtom, BBTransition
from tqdm import tqdm

def strip_element_name(element_name):
    # strips the element name by removing commas, spaces, apostrophes, and dashes
    element_name = element_name.replace(",", "")
    element_name = element_name.replace(" ", "")
    element_name = element_name.replace("'", "")
    element_name = element_name.replace("-", "")
    return element_name.strip()

def read_atoms(nlte_linelist, transition_names, ion_energies, nlte_model_atoms):
    transition_names = np.asarray(transition_names)
    # go through the 2D array of transition names and strip the element names
    for index, element in enumerate(transition_names):
        for index2, element2 in enumerate(element):
            transition_names[index][index2] = strip_element_name(element2).lower()
    # flatten the transition names
    transition_names_all = transition_names.flatten()
    transition_names_all = list(transition_names_all)

    ion_energies = np.asarray(ion_energies)
    # flatten the ion energies
    ion_energies_all = ion_energies.flatten()
    ion_energies_all = list(ion_energies_all)
    for index, element in enumerate(transition_names_all):
        transition_names_all[index] = strip_element_name(element).lower()

    # write the NLTE linelist
    with open(nlte_linelist, 'w') as new_nlte_file:
        # read the LTE linelist
        with open(lte_linelist, 'r') as fp:
            lines_file: list[str] = fp.readlines()
            # go through all lines
            line_number_read_file: int = 0
            total_lines_in_file: int = len(lines_file)
            while line_number_read_file < total_lines_in_file:  # go through all lines
                # first line is the element name and number of lines
                elem_line_1_to_save = lines_file[line_number_read_file]
                line_number_read_file += 1
                # get the number of lines for this element
                fields: list[str] = elem_line_1_to_save.strip().split()
                number_of_lines_element: int = int(fields[-1])
                # second line is the element name and ionization stage
                elem_line_2_to_save: str = lines_file[line_number_read_file]
                line_number_read_file += 1
                line = elem_line_2_to_save

                if strip_element_name(line[1:6]).lower() in transition_names_all:  # find the transition we want to crossmatch
                    # write the element name and number of lines
                    new_nlte_file.write(elem_line_1_to_save)
                    # write the element and NLTE label
                    elem_line_2_to_save_nlte = elem_line_2_to_save.replace("LTE", "NLTE")
                    if "NLTE" not in elem_line_2_to_save_nlte:
                        # replace only the last ' with NLTE'
                        elem_line_2_to_save_nlte = elem_line_2_to_save_nlte[::-1].replace("'", "'ETLN ", 1)[::-1]
                    new_nlte_file.write(elem_line_2_to_save_nlte)

                    # get the index of the element
                    transition_index: int = transition_names_all.index(strip_element_name(line[1:6]).lower())
                    ionisation_energy = ion_energies_all[transition_index]
                    # find index in which the element is in the transition_names 2D array
                    index_element_in_transition_names = np.where(transition_names == strip_element_name(line[1:6]).lower())[0][0]
                    atom_EC, atom_label, atom_level, atom_upper2j1, atom_bb_transitions = read_model_atom(nlte_model_atoms[index_element_in_transition_names])
                    line = lines_file[line_number_read_file]
                    line_number_read_file += 1
                    # get count of crossmatched lines
                    count_total = 0
                    count_found_both = 0
                    count_found_one = 0
                    for line_number in tqdm(range(number_of_lines_element)):  # pull info from lte linelist and set to units in model atom
                        fields = line.strip().split()
                        lte_wave_air = float(fields[0])
                        lte_EC_ev = float(fields[1])
                        lte_up2j1 = float(fields[4])
                        lte_EC_low = 8065.54429 * (lte_EC_ev + ionisation_energy)
                        sigma2 = (10000. / lte_wave_air) ** 2
                        fact = 1.0 + 8.336624212083e-5 + 2.408926869968e-2 / (1.301065924522e2 - sigma2) + 1.599740894897e-4 / (3.892568793293e1 - sigma2)
                        lte_wave_vac = lte_wave_air * fact
                        lte_EC_up = lte_EC_low + ((1.0 / lte_wave_vac) * 1e8)
                        if len(fields) > 15:
                            lte_label_up = fields[len(fields) - 1].replace("'", "")
                            if lte_label_up == "":
                                lte_label_up = fields[len(fields) - 2].replace("'", "")
                        else:
                            lte_label_up = "spud"

                        convid_low = 'x'
                        convid_high = 'x'
                        nlte_level_low = 0
                        nlte_label_low = 'none'
                        nlte_level_up = 0
                        nlte_label_up = 'none'

                        for one_bb_transition in atom_bb_transitions:
                            if np.abs(one_bb_transition.wavelength_aa - lte_wave_air) < 0.02 and (one_bb_transition.lower_level_ec * energy_factor >= lte_EC_low >= one_bb_transition.lower_level_ec / energy_factor):
                                nlte_level_low = one_bb_transition.lower_level_id
                                nlte_label_low = one_bb_transition.lower_level_label
                                nlte_label_up = one_bb_transition.upper_level_label
                                nlte_level_up = one_bb_transition.upper_level_id
                                convid_low = 'b'
                                convid_high = 'b'
                                break

                        if convid_low == 'x' and convid_high == 'x' and True:
                            if lte_label_up[0].isalpha() and fields[len(fields) - 1][0] != '_':  # determine if matching can be done based on the labels in the lte linelist
                                if lte_label_up[len(lte_label_up) - 1] == "*":
                                    lte_label_up = lte_label_up.replace("*", "") + str(int((lte_up2j1 - 1.0) / 2.0)) + "*"
                                else:
                                    lte_label_up = lte_label_up + str(int((lte_up2j1 - 1.0) / 2.0))
                                nlte_level_low = 0
                                nlte_label_low = 'none'
                                nlte_level_up = 0
                                nlte_label_up = 'none'
                                convid_low = 'x'
                                convid_high = 'x'
                                for j in range(len(atom_EC)):
                                    if atom_EC[j] * energy_factor >= lte_EC_low >= atom_EC[j] / energy_factor:
                                        nlte_level_low = atom_level[j]
                                        nlte_label_low = atom_label[j]
                                        nlte_EC_low = atom_EC[j]
                                        convid_low = 'c'
                                    if lte_label_up == atom_label[j]:
                                        nlte_level_up = atom_level[j]
                                        nlte_label_up = atom_label[j]
                                        convid_high = 'c'
                            elif lte_label_up[0].isalpha() and fields[len(fields) - 1][0] == '_':  # what to do when encounter something like "__:59390even" as a label for the upper level in lte linelist
                                nlte_level_low = 0
                                nlte_label_low = 'none'
                                nlte_level_up = 0
                                nlte_label_up = 'none'
                                convid_low = 'x'
                                convid_high = 'x'
                                lte_EC_up = re.findall(r'\d+', fields[len(fields) - 1])
                                try:
                                    lte_EC_up = float(lte_EC_up[0])
                                    for j in range(len(atom_EC)):
                                        if atom_EC[j] * energy_factor >= lte_EC_low >= atom_EC[j] / energy_factor:
                                            nlte_level_low = atom_level[j]
                                            nlte_label_low = atom_label[j]
                                            nlte_EC_low = atom_EC[j]
                                            convid_low = 'c'
                                        if atom_EC[j] * energy_factor >= lte_EC_up >= atom_EC[j] / energy_factor:
                                            nlte_level_up = atom_level[j]
                                            nlte_label_up = atom_label[j]
                                            convid_high = 'c'
                                except IndexError:
                                    pass
                            elif "/" in lte_label_up:  # what to do if upper label in lte linelist is something like "2[11/2]", means either using accessory file or matching based on energy
                                if accessory_match:
                                    acc_level, acc_energy, acc_label, acc_energy_nist, acc_term, acc_config, acc_percent, acc_ion = np.loadtxt(
                                        accessory_matching_files[index_element_in_transition_names], dtype='str', unpack=True)
                                lte_config = fields[len(fields) - 2]
                                lte_config = lte_config[-2:len(lte_config)]
                                nlte_level_low = 0
                                nlte_label_low = 'none'
                                nlte_level_up = 0
                                nlte_label_up = 'none'
                                convid_low = 'x'
                                convid_high = 'x'
                                for j in range(len(atom_EC)):
                                    if atom_EC[j] * energy_factor >= lte_EC_low >= atom_EC[j] / energy_factor:
                                        nlte_level_low = atom_level[j]
                                        nlte_label_low = atom_label[j]
                                        nlte_EC_low = atom_EC[j]
                                        convid_low = 'c'
                                if accessory_match:
                                    for j in range(len(acc_level)):
                                        if lte_label_up == acc_term[j] and lte_config == acc_config[j] and float(
                                                acc_energy[j]) * energy_factor >= lte_EC_up >= float(
                                                acc_energy[j]) / energy_factor:
                                            nlte_level_up = int(acc_level[j])
                                            nlte_label_up = acc_label[j]
                                            convid_high = 'a'
                                elif match_multiplicity:
                                    for j in range(len(atom_EC)):
                                        if atom_EC[j] * energy_factor >= lte_EC_up >= atom_EC[
                                            j] / energy_factor and lte_up2j1 == atom_upper2j1[j]:
                                            nlte_level_up = atom_level[j]
                                            nlte_label_up = atom_label[j]
                                            convid_high = 'a'
                                else:
                                    for j in range(len(atom_EC)):
                                        if atom_EC[j] * energy_factor >= lte_EC_up >= atom_EC[j] / energy_factor:
                                            nlte_level_up = atom_level[j]
                                            nlte_label_up = atom_label[j]
                                            convid_high = 'a'
                            else:  # what to do when no label information from the lte linelist matches the label naming format in the model atom (use accessory file or energy levels)
                                try:
                                    if accessory_match:
                                        acc_level, acc_energy, acc_label, acc_energy_nist, acc_term, acc_config, acc_percent, acc_ion = np.loadtxt(
                                            accessory_matching_files[index_element_in_transition_names], dtype='str', unpack=True)

                                    place = 0
                                    nlte_level_low = 0
                                    nlte_label_low = 'none'
                                    nlte_level_up = 0
                                    nlte_label_up = 'none'
                                    convid_low = 'x'
                                    convid_high = 'x'

                                    if not accessory_match:
                                        lte_label_up = lte_label_up.replace("*", "")
                                    for j in range(len(atom_EC)):
                                        if atom_EC[j] * energy_factor >= lte_EC_low >= atom_EC[j] / energy_factor:
                                            nlte_level_low = atom_level[j]
                                            nlte_label_low = atom_label[j]
                                            nlte_EC_low = atom_EC[j]
                                            convid_low = 'c'
                                    if accessory_match:
                                        lte_config = fields[len(fields) - 2]
                                        while lte_config:
                                            # print(lte_config[len(lte_config)-1-place])
                                            if lte_config[len(lte_config) - 1 - place] == "." or lte_config[
                                                len(lte_config) - 1 - place] == "-" or lte_config[
                                                len(lte_config) - 1 - place] == ":":
                                                break
                                            place += 1
                                        lte_config = lte_config[-place:len(lte_config)]
                                        lte_config = lte_config.replace("(", "")
                                        lte_config = lte_config.replace(")", "")

                                        for j in range(len(acc_level)):
                                            if lte_label_up == acc_term[j] and lte_config == acc_config[j] and float(
                                                    acc_energy[j]) * energy_factor >= lte_EC_up >= float(
                                                acc_energy[j]) / energy_factor:
                                                nlte_level_up = int(acc_level[j])
                                                nlte_label_up = acc_label[j]
                                                convid_high = 'a'
                                    elif match_multiplicity:
                                        for j in range(len(atom_EC)):
                                            if (lte_label_up in atom_label[j] or lte_label_up.lower() in atom_label[
                                                j]) and lte_EC_up <= float(atom_EC[j]) * energy_factor and lte_EC_up >= float(
                                                    atom_EC[j]) / energy_factor and lte_up2j1 == atom_upper2j1[j]:
                                                nlte_level_up = atom_level[j]
                                                nlte_label_up = atom_label[j]
                                                convid_high = 'a'
                                    else:
                                        for j in range(len(atom_EC)):
                                            if (lte_label_up in atom_label[j] or lte_label_up.lower() in atom_label[j]) and \
                                                    atom_EC[j] * energy_factor >= lte_EC_up >= atom_EC[j] / energy_factor:
                                                nlte_level_up = atom_level[j]
                                                nlte_label_up = atom_label[j]
                                                convid_high = 'a'
                                except IndexError:
                                    pass

                        nlte_line = f"  {line.strip()}  {nlte_level_low} {nlte_level_up}  '{nlte_label_low}' '{nlte_label_up}'  '{convid_low}' '{convid_high}'"
                        count_total += 1
                        if convid_low != 'x' and convid_high != 'x':
                            count_found_both += 1
                            if overwrite_lte:
                                gamma_stark = '0.000'
                                nlte_EC_low_EV = "{:.3f}".format(nlte_EC_low / 8065.54429)
                                line_count = 0
                                with open(model_line_list[index_element_in_transition_names]) as file:
                                    model_line = file.readline()
                                    line_count += 1
                                    while model_line:
                                        if (line_count % 2) == 0:
                                            model_fields = model_line.strip().split()
                                            level_low_model = model_fields[1]
                                            level_up_model = model_fields[0]
                                            GA_model = model_fields[7]
                                            if int(level_low_model) == nlte_level_low and int(
                                                    level_up_model) == nlte_level_up:
                                                wave_vald, loggf_vald, gamma_stark_vald = np.loadtxt(vald_output[index_element_in_transition_names],
                                                                                                     delimiter=',',
                                                                                                     skiprows=2,
                                                                                                     usecols=(1, 3, 5),
                                                                                                     unpack=True)
                                                for z in range(len(wave_vald)):
                                                    if wave_vald[z] * 1.0005 >= lte_wave_air >= wave_vald[
                                                        z] / 1.0005 and float(fields[2]) <= loggf_vald[
                                                        z] * 1.0005 and float(fields[2]) >= loggf_vald[z] / 1.0005:
                                                        gamma_stark = str(gamma_stark_vald[z])
                                                if index_element_in_transition_names == 1:
                                                    convid_high = 'c'
                                                nlte_line = "  " + wave_model + "  " + nlte_EC_low_EV + " " + loggf_model + "  " + \
                                                            fields[3] + "    " + fields[
                                                                4] + "  " + GA_model + "  " + gamma_stark + "  " + line.strip()[
                                                                                                                   49:len(
                                                                                                                       line.strip()) - 1] + "'" + "  " + str(
                                                    nlte_level_low) + " " + str(nlte_level_up) + "  '" + str(
                                                    nlte_label_low) + "' '" + str(nlte_label_up) + "'  '" + str(
                                                    convid_low) + "' '" + str(convid_high) + "'"
                                        else:
                                            model_fields = model_line.strip().split()
                                            wave_model = model_fields[4]
                                            loggf_model = model_fields[7]
                                        model_line = file.readline()
                                        line_count += 1
                                file.close()
                        elif convid_low != 'x' or convid_high != 'x':
                            count_found_one += 1
                            if overwrite_lte:
                                nlte_line = "  " + line.strip() + "  " + str(nlte_level_low) + " " + str(
                                    nlte_level_up) + "  '" + str(nlte_label_low) + "' '" + str(
                                    nlte_label_up) + "'  '" + str(convid_low) + "' '" + str(convid_high) + "'"
                        # write line to new nlte file
                        new_nlte_file.write(f"{nlte_line}\n")
                        # get next line from lte file
                        if line_number_read_file < total_lines_in_file:
                            line = lines_file[line_number_read_file]
                        line_number_read_file += 1
                        if not line:  # added to prevent error if element and transition is last one in lte file
                            break
                    # actually needed to read len(lines_file) - 1 lines, so we need to go back one line
                    line_number_read_file -= 1
                    if count_total == 0:
                        print(f"{strip_element_name(elem_line_2_to_save.replace('LTE', ''))}: No lines found in the linelist to crossmatch")
                    else:
                        print("{}: Found {} of {} lines ({} %). Partial matches are {} of {} lines ({} %)".format(
                            strip_element_name(elem_line_2_to_save.replace('LTE', '')), count_found_both, count_total,
                            100. * (count_found_both / count_total), count_found_one, count_total,
                            100. * (count_found_one / count_total)))
                else:
                    # so if the element is not in the model atoms to match, we just copy the lte linelist over
                    new_nlte_file.write(elem_line_1_to_save)
                    new_nlte_file.write(elem_line_2_to_save)
                    for line_number_lte in range(number_of_lines_element):
                        new_nlte_file.write(lines_file[line_number_read_file])
                        line_number_read_file += 1

def read_model_atom(nlte_model):
    model_atom = ModelAtom()
    model_atom.read_model_atom(nlte_model, read_collisions=False, sort_bb_transitions=False)

    for bb_transition in model_atom.bb_transitions:
        bb_transition.upper_level_label = model_atom.ec_levels[bb_transition.upper_level_id].label_name
        bb_transition.lower_level_label = model_atom.ec_levels[bb_transition.lower_level_id].label_name

    atom_EC = []
    atom_upper2j1 = []
    atom_level = []
    atom_label = []
    atom_bb_transitions = model_atom.bb_transitions

    for level in model_atom.ec_levels:
        atom_EC.append(level.ec_value)
        atom_upper2j1.append(level.g_value)
        atom_level.append(level.level_id)
        atom_label.append(level.label_name)

    return atom_EC, atom_label, atom_level, atom_upper2j1, atom_bb_transitions

if __name__ == '__main__':
    lte_linelist_dir = "12CH-BX_Masseron_v2.list"
    # lte_linelist_dir = "nlte_ges_linelist_jmg18jul2023_I_II"
    lte_linelist = lte_linelist_dir
    output_file = f"{lte_linelist_dir}_nlte"

    nlte_models = ["atom.ch_lte"]

    transition_names = [["chb"]]
    ion_energy = [[0.0]]

    accessory_match = False  # whether or not to use accessory matching files set as True or False, in almost all cases, this will be false (it is outdated)
    accessory_matching_files = []  # can leave empty if not using accessory matching files

    match_multiplicity = False  # whether or not the upper level multiplicity (2J+1) from the lte linelist needs to match the multiplicity in the model atom
    # do not use with accessory_match set to True, is often not needed

    overwrite_lte = False  # whether or not to overwrite the information such as excitation potential, loggf, etc. from the lte linelist with the information found in the model atom
    model_line_list = ["model_lines_ca", "model_lines_fe"]  # needed for overwriting, otherwise can leave blank
    vald_output = ["ca_vald_short", "fe_vald_short"]  # needed for overwriting, otherwise can leave blank

    energy_factor = 1.05  # this parameter can be adjusted to whatever margin of error is wanted (currently at 0.05%)

    read_atoms(output_file, transition_names, ion_energy, nlte_models)

    exit()

    lte_linelist = "turbospec.20180901t20.atoms"
    output_file = f"{lte_linelist}_test_3"

    nlte_models = ["atom.fe607a", "atom.ba111", "atom.ca105b", "atom.co247", "atom.mg86b",
                   "atom.mn281kbc", "atom.ni538qm", "atom.si340", "atom.sr191", "atom.ti503",
                   "atom.y423"]

    transition_names = [["Fe I", "Fe II"], ["BA I", "BA II"], ["CA I", "CA II"], ["CO I", "CO II"],
                        ["MG I", "MG II"], ["MN I", "MN II"], ["NI I", "NI II"], ["SI I", "SI II"],
                        ["SR I", "SR II"], ["TI I", "TI II"], ["Y I", "Y II"]]
    ion_energy = [[0.0, 7.9024681], [0.0, 5.2116646], [0.0, 6.11315547], [0.0, 7.88101],
                  [0.0, 7.646236], [0.0, 7.4340380], [0.0, 7.639878], [0.0, 8.15168],
                  [0.0, 5.69486745], [0.0, 6.828120], [0.0, 6.2173]]

    accessory_match = False  # whether or not to use accessory matching files set as True or False, in almost all cases, this will be false (it is outdated)
    accessory_matching_files = []  # can leave empty if not using accessory matching files

    match_multiplicity = False  # whether or not the upper level multiplicity (2J+1) from the lte linelist needs to match the multiplicity in the model atom
    # do not use with accessory_match set to True, is often not needed

    overwrite_lte = False  # whether or not to overwrite the information such as excitation potential, loggf, etc. from the lte linelist with the information found in the model atom
    model_line_list = ["model_lines_ca", "model_lines_fe"]  # needed for overwriting, otherwise can leave blank
    vald_output = ["ca_vald_short", "fe_vald_short"]  # needed for overwriting, otherwise can leave blank

    energy_factor = 1.0005  # this parameter can be adjusted to whatever margin of error is wanted (currently at 0.05%)

    read_atoms(output_file, transition_names, ion_energy, nlte_models)
