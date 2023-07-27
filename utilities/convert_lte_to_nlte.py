import numpy as np
import re

def read_atoms():
    for i in range(len(nlte_models)):
        #extract the info on the energy levels from the model atom/s
        atom_EC = []
        atom_upper2j1 = []
        atom_level = []
        atom_label = []
        with open(nlte_models[i]) as fp:
            line = fp.readline()
            while line:
                fields = line.strip().split()
                if fields[0] == "*" and len(fields) > 1:
                    if fields[1] == "EC" or fields[1] == "E[cm⁻¹]":
                        while line[0:4] != "* UP" and line[0:11] != "* RADIATIVE":
                            line = fp.readline()
                            fields = line.strip().split()
                            if fields[0] != "*":
                                atom_EC.append(float(fields[0]))
                                atom_upper2j1.append(float(fields[1]))
                                try:
                                    atom_level.append(int(fields[3]))
                                    atom_label.append(fields[5].replace("'", ""))
                                except ValueError:
                                    atom_level.append(int(fields[4]))
                                    atom_label.append(fields[6].replace("'", ""))
                    else:
                        line = fp.readline()
                else:
                    line = fp.readline()
        fp.close()
        #begin crossmatching the lines in the model linelist and filling the nlte line array
        count_found_both = 0  # keep a count of how many we successfully crossmatch for a report at the end
        count_found_one = 0
        count_total = 0
        for k in range(len(transition_names[i])):
            with open(lte_linelist) as fp:
                line = fp.readline()
                while line:
                    if (line[1:6].replace(" ", "") == transition_names[i][k].replace(" ","")): #find the transition we want to crossmatch
                        line = fp.readline()
                        energy_factor = 1.0005 #this parameter can be adjusted to whatever margin of error is wanted (currently at 0.05%)
                        while line[0] != "'": #pull info from lte linelist and set to units in model atom
                            fields = line.strip().split()
                            lte_wave_air = float(fields[0])
                            lte_EC_ev = float(fields[1])
                            lte_up2j1 = float(fields[4])
                            lte_EC_low = 8065.54429*(lte_EC_ev+ion_energy[i][k])
                            sigma2 = (10000./lte_wave_air)**2
                            fact = 1.0 + 8.336624212083e-5 + 2.408926869968e-2/(1.301065924522e2 - sigma2) + 1.599740894897e-4/(3.892568793293e1 - sigma2)
                            lte_wave_vac = lte_wave_air * fact
                            lte_EC_up = lte_EC_low + ((1.0/lte_wave_vac)*1e8)
                            if (len(fields) > 15):
                                lte_label_up = fields[len(fields)-1].replace("'","")
                            else:
                                lte_label_up = "spud"
                            if lte_label_up[0].isalpha() and fields[len(fields)-1][0] != '_': #determine if matching can be done based on the labels in the lte linelist
                                if lte_label_up[len(lte_label_up)-1] == "*":
                                    lte_label_up = lte_label_up.replace("*","") + str(int((lte_up2j1-1.0)/2.0)) + "*"
                                else:
                                    lte_label_up = lte_label_up + str(int((lte_up2j1-1.0)/2.0))
                                nlte_level_low = 0
                                nlte_label_low = 'none'
                                nlte_level_up = 0
                                nlte_label_up = 'none'
                                convid_low = 'x'
                                convid_high = 'x'
                                for j in range(len(atom_EC)):
                                    if lte_EC_low <= atom_EC[j]*energy_factor and lte_EC_low >= atom_EC[j]/energy_factor:
                                        nlte_level_low = atom_level[j]
                                        nlte_label_low = atom_label[j]
                                        nlte_EC_low = atom_EC[j]
                                        convid_low = 'c'
                                    if lte_label_up == atom_label[j]:
                                        nlte_level_up = atom_level[j]
                                        nlte_label_up = atom_label[j]
                                        convid_high = 'c'
                            elif lte_label_up[0].isalpha() and fields[len(fields)-1][0] == '_': #what to do when encounter something like "__:59390even" as a label for the upper level in lte linelist
                                nlte_level_low = 0
                                nlte_label_low = 'none'
                                nlte_level_up = 0
                                nlte_label_up = 'none'
                                convid_low = 'x'
                                convid_high = 'x'
                                lte_EC_up = re.findall(r'\d+', fields[len(fields)-1])
                                try:
                                    lte_EC_up = float(lte_EC_up[0])
                                    for j in range(len(atom_EC)):
                                        if lte_EC_low <= atom_EC[j] * energy_factor and lte_EC_low >= atom_EC[j] / energy_factor:
                                            nlte_level_low = atom_level[j]
                                            nlte_label_low = atom_label[j]
                                            nlte_EC_low = atom_EC[j]
                                            convid_low = 'c'
                                        if lte_EC_up <= atom_EC[j] * energy_factor and lte_EC_up >= atom_EC[j] / energy_factor:
                                            nlte_level_up = atom_level[j]
                                            nlte_label_up = atom_label[j]
                                            convid_high = 'c'
                                except IndexError:
                                    pass
                            elif "/" in lte_label_up: #what to do if upper label in lte linelist is something like "2[11/2]", means either using accessory file or matching based on energy
                                if accessory_match:
                                    acc_level, acc_energy, acc_label, acc_energy_nist, acc_term, acc_config, acc_percent, acc_ion = np.loadtxt(accessory_matching_files[i], dtype='str', unpack=True)
                                lte_config = fields[len(fields)-2]
                                lte_config = lte_config[-2:len(lte_config)]
                                nlte_level_low = 0
                                nlte_label_low = 'none'
                                nlte_level_up = 0
                                nlte_label_up = 'none'
                                convid_low = 'x'
                                convid_high = 'x'
                                for j in range(len(atom_EC)):
                                    if lte_EC_low <= atom_EC[j]*energy_factor and lte_EC_low >= atom_EC[j]/energy_factor:
                                        nlte_level_low = atom_level[j]
                                        nlte_label_low = atom_label[j]
                                        nlte_EC_low = atom_EC[j]
                                        convid_low = 'c'
                                if accessory_match:
                                    for j in range(len(acc_level)):
                                        if lte_label_up == acc_term[j] and lte_config == acc_config[j] and lte_EC_up <= float(acc_energy[j])*energy_factor and lte_EC_up >= float(acc_energy[j])/energy_factor:
                                            nlte_level_up = int(acc_level[j])
                                            nlte_label_up = acc_label[j]
                                            convid_high = 'a'
                                elif match_multiplicity:
                                    for j in range(len(atom_EC)):
                                        if lte_EC_up <= float(atom_EC[j])*energy_factor and lte_EC_up >= float(atom_EC[j])/energy_factor and lte_up2j1 == atom_upper2j1[j]:
                                            nlte_level_up = atom_level[j]
                                            nlte_label_up = atom_label[j]
                                            convid_high = 'a'
                                else:
                                    for j in range(len(atom_EC)):
                                        if lte_EC_up <= float(atom_EC[j])*energy_factor and lte_EC_up >= float(atom_EC[j])/energy_factor:
                                            nlte_level_up = atom_level[j]
                                            nlte_label_up = atom_label[j]
                                            convid_high = 'a'
                            else: #what to do when no label information from the lte linelist matches the label naming format in the model atom (use accessory file or energy levels)
                                try:
                                    if accessory_match:
                                        acc_level, acc_energy, acc_label, acc_energy_nist, acc_term, acc_config, acc_percent, acc_ion = np.loadtxt(accessory_matching_files[i], dtype='str', unpack=True)
                                    lte_config = fields[len(fields)-2]
                                    place = 0
                                    while lte_config:
                                        if lte_config[len(lte_config)-1-place] == "." or lte_config[len(lte_config)-1-place] == "-" or lte_config[len(lte_config)-1-place] == ":":
                                            break
                                        place+=1
                                    lte_config = lte_config[-place:len(lte_config)]
                                    lte_config = lte_config.replace("(","")
                                    lte_config = lte_config.replace(")","")
                                    if accessory_match == False:
                                        lte_label_up = lte_label_up.replace("*","")
                                    nlte_level_low = 0
                                    nlte_label_low = 'none'
                                    nlte_level_up = 0
                                    nlte_label_up = 'none'
                                    convid_low = 'x'
                                    convid_high = 'x'
                                    for j in range(len(atom_EC)):
                                        if lte_EC_low <= atom_EC[j]*energy_factor and lte_EC_low >= atom_EC[j]/energy_factor:
                                            nlte_level_low = atom_level[j]
                                            nlte_label_low = atom_label[j]
                                            nlte_EC_low = atom_EC[j]
                                            convid_low = 'c'
                                    if accessory_match:
                                        for j in range(len(acc_level)):
                                            if lte_label_up == acc_term[j] and lte_config == acc_config[j] and lte_EC_up <= float(acc_energy[j])*energy_factor and lte_EC_up >= float(acc_energy[j])/energy_factor:
                                                nlte_level_up = int(acc_level[j])
                                                nlte_label_up = acc_label[j]
                                                convid_high = 'a'
                                    elif match_multiplicity:
                                        for j in range(len(atom_EC)):
                                            if (lte_label_up in atom_label[j] or lte_label_up.lower() in atom_label[j]) and lte_EC_up <= float(atom_EC[j])*energy_factor and lte_EC_up >= float(atom_EC[j])/energy_factor and lte_up2j1 == atom_upper2j1[j]:
                                                nlte_level_up = atom_level[j]
                                                nlte_label_up = atom_label[j]
                                                convid_high = 'a'
                                    else:
                                        for j in range(len(atom_EC)):
                                            if (lte_label_up in atom_label[j] or lte_label_up.lower() in atom_label[j]) and lte_EC_up <= float(atom_EC[j])*energy_factor and lte_EC_up >= float(atom_EC[j])/energy_factor:
                                                nlte_level_up = atom_level[j]
                                                nlte_label_up = atom_label[j]
                                                convid_high = 'a'
                                except IndexError:
                                    pass

                            nlte_line = "  "+line.strip()[0:48]+"  "+line.strip()[49:len(line.strip())-1]+"'"+"  "+str(nlte_level_low)+" "+str(nlte_level_up)+"  '"+str(nlte_label_low)+"' '"+str(nlte_label_up)+"'  '"+str(convid_low)+"' '"+str(convid_high)+"'"
                            count_total+=1
                            if convid_low != 'x' and convid_high != 'x':
                                count_found_both+=1
                                if overwrite_lte:
                                    gamma_stark = '0.000'
                                    nlte_EC_low_EV = "{:.3f}".format(nlte_EC_low/8065.54429)
                                    line_count = 0
                                    with open(model_line_list[i]) as file:
                                        model_line = file.readline()
                                        line_count+=1
                                        while model_line:
                                            if (line_count % 2) == 0:
                                                model_fields = model_line.strip().split()
                                                level_low_model = model_fields[1]
                                                level_up_model = model_fields[0]
                                                GA_model = model_fields[7]
                                                if int(level_low_model) == nlte_level_low and int(level_up_model) == nlte_level_up:
                                                    wave_vald, loggf_vald, gamma_stark_vald = np.loadtxt(vald_output[i], delimiter = ',', skiprows=2, usecols=(1,3,5), unpack=True)
                                                    for z in range(len(wave_vald)):
                                                        if lte_wave_air <= wave_vald[z]*1.0005 and lte_wave_air >= wave_vald[z]/1.0005 and float(fields[2]) <= loggf_vald[z]*1.0005 and float(fields[2]) >= loggf_vald[z]/1.0005:
                                                            gamma_stark = str(gamma_stark_vald[z])
                                                    if i == 1:
                                                        convid_high = 'c'
                                                    nlte_line = "  "+wave_model+"  "+nlte_EC_low_EV+" "+loggf_model+"  "+fields[3]+"    "+fields[4]+"  "+GA_model+"  "+gamma_stark+"  "+line.strip()[49:len(line.strip())-1]+"'"+"  "+str(nlte_level_low)+" "+str(nlte_level_up)+"  '"+str(nlte_label_low)+"' '"+str(nlte_label_up)+"'  '"+str(convid_low)+"' '"+str(convid_high)+"'"
                                            else:
                                                model_fields = model_line.strip().split()
                                                wave_model = model_fields[4]
                                                loggf_model = model_fields[7]
                                                label_low_model = model_fields[8]
                                                label_up_model = model_fields[9]
                                            model_line = file.readline()
                                            line_count+=1
                                    file.close()
                            elif convid_low != 'x' or convid_high != 'x':
                                count_found_one+=1
                                if overwrite_lte:
                                    nlte_line = "  "+line.strip()+"  "+str(nlte_level_low)+" "+str(nlte_level_up)+"  '"+str(nlte_label_low)+"' '"+str(nlte_label_up)+"'  '"+str(convid_low)+"' '"+str(convid_high)+"'"
                            nlte_lines[i][k].append(nlte_line)
                            line = fp.readline()
                            if not line: #added to prevent error if element and transition is last one in lte file
                                break
                    else:
                        if (line[0] != "'"):
                            line = fp.readline()
                        else:
                            line = fp.readline()
            fp.close()
            if count_total == 0:
                print(f"{transition_names[i][k]}: No lines found in the linelist to crossmatch")
            else:
                print("{}: Found {} of {} lines ({} %). Partial matches are {} of {} lines ({} %)".format(transition_names[i][k], count_found_both, count_total, 100.*(count_found_both/count_total), count_found_one, count_total, 100.*(count_found_one/count_total)))

def print_atoms():
    #now print linelist with nlte values
    g = open(output_file, "w")

    add_lte_flag = "  0 0  'none' 'none'"

    with open(lte_linelist) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            for i in range(len(transition_names)):
                for k in range(len(transition_names[i])):
                    if (line[1:6].replace(" ", "") == transition_names[i][k].replace(" ","") and line[-5:-1] == "LTE'"): #works with newer format
                        g.write("{}".format(line.strip().replace("LTE'","")+" NLTE'\n"))
                        line = fp.readline()
                        while line[0] != "'":
                            line = fp.readline()
                            if not line:
                                break
                        for j in range(len(nlte_lines[i][k])):
                            g.write("{}\n".format(nlte_lines[i][k][j]))
                    elif (line[1:6].replace(" ", "") == transition_names[i][k].replace(" ","") and line[-3:-1] == " '"): #works with older format when NLTE/LTE wasn't specified
                        g.write("{}".format(line.strip().replace(" '","")+" NLTE'\n"))
                        line = fp.readline()
                        while line[0] != "'":
                            line = fp.readline()
                            if not line:
                                break
                        for j in range(len(nlte_lines[i][k])):
                            g.write("{}\n".format(nlte_lines[i][k][j]))
            if not line:
                break
            if (line[0] != "'"):
                g.write("  {}\n".format(line.strip()))
            elif (len(line) > 10):
                g.write("{}\n".format(line.strip()))
            else:
                g.write("{}\n".format("'"+line.strip().replace("'","")+" LTE'"))
            cnt += 1
            line = fp.readline()

    fp.close()

    g.close()

if __name__ == '__main__':
    # input parameters for the cross matching

    lte_linelist = "sample_lte_linelist"
    output_file = "sample_nlte_converted_linelist"

    nlte_models = ["../input_files/nlte_data/model_atoms/atom.o41f", "../input_files/nlte_data/model_atoms/atom.mg86b",
                   "../input_files/nlte_data/model_atoms/atom.caNew"]

    transition_names = [["O I", "O II"], ["Mg I", "Mg II"], ["Ca I",
                                                             "Ca II"]]  # needs to match the length of nlte_models and needs to match the transition names in the lte linelist
    ion_energy = [[0.0, 13.61805], [0.0, 7.646235], [0.0, 6.11316]]  # ionization energy in eV from NIST

    accessory_match = False  # whether or not to use accessory matching files set as True or False, in almost all cases, this will be false (it is outdated)
    accessory_matching_files = []  # can leave empty if not using accessory matching files

    match_multiplicity = False  # whether or not the upper level multiplicity (2J+1) from the lte linelist needs to match the multiplicity in the model atom
    # do not use with accessory_match set to True, is often not needed

    overwrite_lte = False  # whether or not to overwrite the information such as excitation potential, loggf, etc. from the lte linelist with the information found in the model atom
    model_line_list = ["model_lines_ca", "model_lines_fe"]  # needed for overwriting, otherwise can leave blank
    vald_output = ["ca_vald_short", "fe_vald_short"]  # needed for overwriting, otherwise can leave blank

    # get rid of space in transition names for easier matching DEPRECATED
    # atom_names = []
    # for i in range(len(transition_names)):
    #    atom = transition_names[i].replace(" ", "")
    #    atom_names.append(atom)

    # initialize directory for storing nlte lines that will be added to final linelist
    nlte_lines = [[[] for i in range(len(nlte_models) + 1)] for j in
                  range(len(nlte_models) + 1)]  # the +1 allows it to work with an array of one nlte element to identify

    read_atoms()
    print_atoms()
