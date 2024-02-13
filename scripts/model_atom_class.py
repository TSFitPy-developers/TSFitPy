from __future__ import annotations

import os

# creates model atom based on NIST and KURUCZ data

import numpy as np
import pandas as pd

electron_charge = 1.602176634e-19
light_speed = 2.99792458e8
mass_electron = 9.1093837015e-31
epsilon_0 = 8.8541878128e-12


class ECLevel:
    def __init__(self, level_id, ec, g, term, ion, config, ionisation_ec):
        self.level_id: int = level_id
        if self.level_id <= 0:
            raise ValueError("Level ID must be greater than 0")
        self.ec_value: float = ec
        self.g_value: float = g
        self.term_name: str = term
        self.ionisation_stage: int = ion
        self.config_name: str = config
        self.label_name: str = (
            self.config_name.replace(" ", "") + "" + self.term_name.replace(" ", "")
        )
        # ionisation_ec is the energy of the ionisation level below this level
        self.ionisation_ec = ionisation_ec

    def __str__(self):
        return f"ECLevel(id = {self.level_id:>4}, EC = {self.ec_value:>12.3f}, g = {self.g_value:>4.1f}, label = {self.label_name:>20}, ion = {self.ionisation_stage:>1})"

    def write_to_file(self, file, label_length=10):
        file.write(
            f"{self.ec_value:>12.3f}{self.g_value:>9}  '    Level {self.level_id:>4} = {self.label_name:>{label_length}} '   {self.ionisation_stage}\n"
        )

    def __eq__(self, other, tolerance=0.1):
        if isinstance(other, ECLevel):
            # compares the energy levels by EC, g and ionisation stage
            # EC is equal within tolerance

            if (
                abs(self.ec_value - other.ec_value) <= tolerance
                and self.g_value == other.g_value
                and self.ionisation_stage == other.ionisation_stage
            ):
                return True
            else:
                return False
        else:
            return False


class BBTransition:
    def __init__(
        self, wavelength, ion_transition, loggf, lower_level_ec=0.0, upper_level_ec=0.0, bb_id=0
    ):
        self.wavelength_aa: float = wavelength
        self.ion_transition: int = ion_transition

        self.loggf: float = loggf
        self.f_value: float = -1
        self.lower_level_ec: float = lower_level_ec
        self.upper_level_ec: float = upper_level_ec

        self.lower_level_id: int = 1
        self.upper_level_id: int = 1
        self.lower_level_label: str = ""
        self.upper_level_label: str = ""
        self.lower_level_g_value: float = 0.0

        self.bb_id: int = bb_id

        # f"* UP LOW      F        NQ  QMAX   Q0    IW    GA            GV   GQ\n"

        self.nq = 9
        self.qmax = 4.0
        self.q0 = 0.0
        self.iw = 0  # if > 0 then hyperfine structure is included
        self.ga = 0.0
        self.gv = 1.0
        self.gq = 0
        self.line_profile = "VOIGT"

        # hypefine
        self.hyperfine = False
        self.hyperfine_delta_ec = []
        self.hyperfine_weight = []
        self.hyperfine_f_value = []
        self.hyperfine_ga = []
        self.hyperfine_gw = []
        self.hyperfine_gq = []
        self.hyperfine_line_profile = []

    def find_level_id(self, ec_levels, tolerance=0.1):
        lower_levels_within_tolerance: list[ec_levels] = []
        upper_levels_within_tolerance: list[ec_levels] = []

        for ec_level in ec_levels:
            if (
                abs(ec_level.ec_value - ec_level.ionisation_ec - self.lower_level_ec)
                <= tolerance
                and ec_level.ionisation_stage == self.ion_transition
            ):
                lower_levels_within_tolerance.append(ec_level)
            if (
                abs(ec_level.ec_value - ec_level.ionisation_ec - self.upper_level_ec)
                <= tolerance
                and ec_level.ionisation_stage == self.ion_transition
            ):
                upper_levels_within_tolerance.append(ec_level)

        lower_g_value = None
        upper_g_value = None

        if len(lower_levels_within_tolerance) == 0:
            self.lower_level_id = -1
            self.lower_level_label = "???"
            print(f"Could not find lower level for {self}")
        elif len(lower_levels_within_tolerance) == 1:
            self.lower_level_id = lower_levels_within_tolerance[0].level_id
            self.lower_level_label = lower_levels_within_tolerance[0].label_name
            lower_g_value = lower_levels_within_tolerance[0].g_value
            self.lower_level_g_value = lower_g_value
            self.calculate_f_value(lower_g_value)
        else:
            print(f"Found {len(lower_levels_within_tolerance)} lower levels for {self}")
            self.lower_level_id = -2
            self.lower_level_label = "SEV?"

        if len(upper_levels_within_tolerance) == 0:
            self.upper_level_id = -1
            self.upper_level_label = "???"
            print(f"Could not find upper level for {self}")
        elif len(upper_levels_within_tolerance) == 1:
            self.upper_level_id = upper_levels_within_tolerance[0].level_id
            self.upper_level_label = upper_levels_within_tolerance[0].label_name
            upper_g_value = upper_levels_within_tolerance[0].g_value
        else:
            print(f"Found {len(upper_levels_within_tolerance)} upper levels for {self}")
            self.upper_level_id = -2
            self.upper_level_label = "SEV?"

        if lower_g_value is not None and upper_g_value is not None:
            wavelength_m = self.wavelength_aa * 1e-10
            frequency = light_speed / wavelength_m
            # frequency = 1
            einstein_coefficient = (
                (2 * np.pi * frequency**2 * electron_charge**2)
                / (epsilon_0 * mass_electron * light_speed**3)
                * self.f_value
                * (lower_g_value / upper_g_value)
            )

            self.ga = einstein_coefficient
            self.ga = 0
        else:
            self.ga = 0

    def __str__(self):
        return f"BBTransition({self.wavelength_aa}, {self.loggf}, {self.lower_level_id}, {self.upper_level_id})"

    def write_to_file(
        self, file, id_overwrite=None, write_comment=True, label_length=10
    ):
        if id_overwrite is not None:
            self.bb_id = id_overwrite
        # f"* UP LOW      F        NQ  QMAX   Q0    IW    GA            GV   GQ\n"
        # 1195  873   3.349E-07     9   4.0   0.0    0  8.198E+02       1.5  0.00e+00       VOIGT
        if write_comment:
            # * id = 18375   lam (A) =      17619.5841 log gf =  -4.541    'F2e'0  'F2e'3
            file.write(
                f"* id = {self.bb_id:>5}   lam (A) = {self.wavelength_aa:>15.3f} log gf = {self.loggf:>7.3f} "
                f"{self.lower_level_label:>{label_length}} {self.upper_level_label:>{label_length}}\n"
            )
        file.write(
            f"{self.upper_level_id:>4} {self.lower_level_id:>4}{self.f_value:>12.3E}{self.nq:>6}{self.qmax:>6.1f}{self.q0:>6.1f}"
            f"{self.iw:>5}{self.ga:>11.3E}{self.gv:>10.1f}{self.gq:>10.2e}{self.line_profile:>12}\n"
        )
        if self.hyperfine:
            for i in range(len(self.hyperfine_delta_ec)):
                file.write(
                    f"{self.hyperfine_delta_ec[i]:>9.3f} {self.hyperfine_weight[i]:>6.3f} {self.hyperfine_f_value[i]:>11.3E}"
                    f" {self.hyperfine_ga[i]:>27.3E} {self.hyperfine_gw[i]:>9.3f} {self.hyperfine_gq[i]:>9.3e}"
                    f" {self.hyperfine_line_profile[i]:>11}\n"
                )

    def __eq__(self, other, tolerance=0.1):
        if isinstance(other, BBTransition):
            # compares the energy levels by EC, g and ionisation stage
            # EC is equal within tolerance
            # also compares f value

            if (
                abs(self.wavelength_aa - other.wavelength_aa) <= tolerance
                and abs(self.loggf - other.loggf) <= tolerance
                and abs(self.f_value - other.f_value) <= tolerance
            ):
                return True
            else:
                return False
        else:
            return False

    def calculate_f_value(self, lower_g_value: float):
        self.f_value = 10**self.loggf / lower_g_value


class BFTransition:
    def __init__(self, lower_level_ec, upper_level_ec, a0, nq, lmax, lmin=None):
        self.a0 = a0  # F value, cross-section at the edge (cgs)

        self.lower_level_id = lower_level_ec
        self.upper_level_id = upper_level_ec

        # f"*  PHOTO-ION:  UP  LO  F   NQ  QMAX   Q0\n"
        # UPPER LOWER A0 NQ LMAX/QMAX (LMIN)

        self.nq = nq
        self.lmax = (
            lmax  # same as qmax. if lmax is negative, then it is the same as qmax
        )
        self.lmin = lmin

        if self.lmin is not None:
            self.wavelengths = []
            self.alpha_cross_sections = []
        else:
            self.wavelengths = np.zeros(self.nq)
            self.alpha_cross_sections = np.zeros(self.nq)

    def __str__(self):
        # 188   1   1.1303E-17    11     -1.0
        string = f"{self.upper_level_id:>4} {self.lower_level_id:>4} {self.a0:>12.4E} {self.nq:>5}"
        if self.lmin is not None:
            string += f" {self.lmax:>8.3E} {self.lmin:>8.3E} hi\n"
        else:
            string += f" {self.lmax:>6.1f}\n"
            for i in range(self.nq):
                string += f" {self.wavelengths[i]:>12.3f} {self.alpha_cross_sections[i]:>12.5E}\n"
        return string

    def write_to_file(self, file):
        # 188   1   1.1303E-17    11     -1.0
        string = f"{self.upper_level_id:>4} {self.lower_level_id:>4} {self.a0:>12.4E} {self.nq:>5}"
        if self.lmin is not None:
            string += f" {self.lmax:>8.3E} {self.lmin:>8.3E}\n"
            file.write(string)
        else:
            string += f" {self.lmax:>6.1f}\n"
            file.write(string)
            for i in range(self.nq):
                string = f" {self.wavelengths[i]:>12.3f} {self.alpha_cross_sections[i]:>12.5E}\n"
                file.write(string)


class CollisionalTransition:
    def __init__(
        self,
        label: str,
        upper_level: int,
        lower_level: int,
        values_count: int,
        collisional_values: list,
        temperature_values: list,
    ):
        # label, then upper level, then lower level then amount of collisional values for each temperature
        self.label: str = label
        self.upper_level: int = upper_level
        self.lower_level: int = lower_level
        self.values_count: int = values_count
        self.collisional_values: list = collisional_values
        self.temperature_values: list = temperature_values

    def __str__(self):
        return f"CollisionalTransition({self.label}, {self.upper_level}, {self.lower_level}, {self.values_count}, {self.collisional_values})"

    def write_to_file(self, file):
        # Construct the string to be written
        output_str = f"{self.label}\n"
        output_str += f"{self.upper_level:>4} {self.lower_level:>3}"
        for collisional_value in self.collisional_values:
            output_str += f" {collisional_value:>10.2E}"
        output_str += "\n"

        # Write the string to the file
        file.write(output_str)

    def write_temperature_values_to_file(self, file):
        file.write(f"TEMP\n")
        file.write(f"{self.values_count:>4}")
        for temperature_value in self.temperature_values:
            file.write(f" {temperature_value:>10.1f}")
        file.write("\n")

    def check_if_same_temperature_values(
        self, other_temperature_values, temperature_tolerance=0.1
    ):
        if len(self.temperature_values) != len(other_temperature_values):
            return False
        else:
            for i, temperature_value in enumerate(self.temperature_values):
                if (
                    abs(temperature_value - other_temperature_values[i])
                    > temperature_tolerance
                ):
                    return False
        return True


def is_number(a):
    try:
        float(a)
        return True
    except ValueError:
        return False


class ModelAtom:
    def __init__(self, atom_element="", atom_filename=""):
        # atom element is the name of the element
        self.atom_element: str = atom_element
        # file name is the name of the file e.g. "atom.fe"
        self.atom_filename: str = atom_filename

        # header comments
        self.header_comments: list[str] = []

        # abundance and mass
        self.atom_abundance: float = 0.0
        self.atom_mass: float = 0.0

        # energy levels and transitions
        self.ec_levels: list[ECLevel] = []
        self.max_ion: int = 0
        self.bb_transitions: list[BBTransition] = []
        self.bf_transitions: list[BFTransition] = []
        # not sure what these are, but last one is the number of radiative fixations
        self.nrfix: int = 0
        # collisional transitions
        self.collisional_transitions: list[CollisionalTransition] = []

    def ec_length(self) -> int:
        """Get the length of the energy levels"""
        return len(self.ec_levels)

    def bb_length(self) -> int:
        """Get the length of the bb transitions"""
        return len(self.bb_transitions)

    def bf_length(self) -> int:
        """Get the length of the bf transitions"""
        return len(self.bf_transitions)

    def sort_bb_transitions(self):
        """Sort the bb transitions by wavelength and set the ids accordingly"""
        self.bb_transitions = sorted(self.bb_transitions, key=lambda x: x.wavelength_aa)
        # set the ids according to the order
        for i, bb_transition in enumerate(self.bb_transitions):
            bb_transition.bb_id = i + 1

    def add_bb_transition(self, bb_transition: BBTransition):
        """adds a bb transition to the list of bb transitions and sorts them by wavelength"""
        self.bb_transitions.append(bb_transition)
        self.sort_bb_transitions()

    def leave_only_bb_transitions_between_wavelength(self, min_wavelength: float, max_wavelength: float):
        """
        Leaves only the transitions between the given wavelengths
        :param min_wavelength: Minimum wavelength
        :param max_wavelength: Maximum wavelength
        """
        self.bb_transitions = [
            x
            for x in self.bb_transitions
            if min_wavelength <= x.wavelength_aa <= max_wavelength
        ]
        self.sort_bb_transitions()

    def set_atom_parameters(self, atom_abundance, atom_mass):
        """Set the atom parameters for the model atom"""
        self.atom_abundance = atom_abundance
        self.atom_mass = atom_mass

    def read_nist_data(self, nist_filenames_csv):
        ec_all = []
        g_all = []
        term_all = []
        ion_all = []
        config_all = []
        current_ion = 1
        max_ec = 0
        ionisation_level = 0
        current_level = 1

        for filename_csv in nist_filenames_csv:
            # read NIST data
            df = pd.read_csv(filename_csv, comment=";")

            ec_one_file = df["Level (cm-1)"].values
            # remove =" from the beginning and " from the end of the string
            ec_one_file = [float(x[2:-1]) for x in ec_one_file]
            g_one_file = df["g"].values
            # only take the number before "/" and convert to float
            # g = [x[2:] for x in g]
            # g_new = []
            # for g1 in g:
            #    if "/" in g1:
            #        g_new.append(float(g1.split("/")[0]))
            #    elif "---" in g1:
            #        g_new.append(0)
            #    elif "or" in g1:
            #        g_new.append(float(g1.split(" or")[0]))
            #    else:
            #        try:
            #            g_new.append(float(g1[:-1]))
            #        except ValueError:
            #            g_new.append(0)
            # g = g_new

            term_one_file = df["Term"].values
            # remove first 2 and last 2 characters
            term_one_file = [x[2:-1] for x in term_one_file]

            config_one_file = df["Configuration"].values
            # remove first 2 and last 2 characters
            config_one_file = [x[2:-1] for x in config_one_file]

            # find the first occurence of "Limi" and remove everything after it
            for i, term_one in enumerate(term_one_file):
                if term_one == "Limit":
                    term_one_file = term_one_file[: i + 1]
                    ec_one_file = ec_one_file[: i + 1]
                    g_one_file = g_one_file[: i + 1]
                    config_one_file = config_one_file[: i + 1]
                    print(f"Found Limit at {i}")
                    break

            if len(ec_all) == 0:
                ec_all.extend(ec_one_file)
            else:
                # extend ec_all by adding max_ec to all values
                ec_one_file = [x + max_ec for x in ec_one_file]
                ionisation_level = max_ec
                ec_all.extend(ec_one_file)

            # label = config_one.replace(" ", "") + " " + term_one.replace(" ", "")
            #             if term_one != "Limit" and i < len(ec_all) - 1:  # and term_one != "":

            g_all.extend(g_one_file)
            term_all.extend(term_one_file)
            config_all.extend(config_one_file)
            current_ion_one_file = [current_ion] * len(ec_one_file)
            ion_all.extend(current_ion_one_file)
            current_ion += 1

            for i, (e_one, g_one, config_one, term_one, ion_one) in enumerate(
                zip(
                    ec_one_file,
                    g_one_file,
                    config_one_file,
                    term_one_file,
                    current_ion_one_file,
                )
            ):
                if term_one != "Limit" and i < len(ec_all) - 1:  # and term_one != "":
                    # for same ionisation the next level is repeated, so not taking it
                    ec_level = ECLevel(
                        current_level,
                        e_one,
                        g_one,
                        term_one,
                        ion_one,
                        config_one,
                        ionisation_level,
                    )
                    self.ec_levels.append(ec_level)
                    current_level += 1

            max_ec = max(ec_one_file)

        if term_one == "Limit":
            # but if last level is Limit, then we need to add it, because we dont have the next ionisation
            ec_level = ECLevel(
                current_level,
                e_one,
                g_one,
                term_one.replace("Limit", "").replace("<0>", ""),
                ion_one + 1,
                config_one.replace("Limit", "").replace("<0>", ""),
                ionisation_level,
            )
            self.ec_levels.append(ec_level)

        #self.ec_length = len(self.ec_levels)

        self.max_ion = current_ion

    def read_kurucz_data(
        self,
        kurucz_filenames,
        min_wavelength=0,
        max_wavelength=1e99,
        convert_nm_to_aa=True,
    ):
        bb_id = 1
        for filename in kurucz_filenames:
            # read Kurucz data txt file
            with open(filename, "r") as file:
                kurucz_filename_lines = file.readlines()
            linebefore = ""
            for line in kurucz_filename_lines:
                # split_line = line.split()
                try:
                    wavelength = float(line[:11])
                except ValueError:
                    print(line, filename, linebefore)
                    exit()
                linebefore = line
                if convert_nm_to_aa:
                    wavelength *= 10

                if min_wavelength <= wavelength <= max_wavelength:
                    loggf = float(line[11:18])
                    # lower level is 25th to 37th character
                    first_level = float(line[24:36].replace(" ", ""))
                    # upper level is 53rd to 65th character
                    second_level = float(line[52:64].replace(" ", ""))

                    lower_level = min(first_level, second_level)
                    upper_level = max(first_level, second_level)
                    # ion transition is 23rd to 24th character
                    ion_transition = int(line[22:24].replace(" ", "")) + 1
                    bb_transition = BBTransition(
                        wavelength,
                        ion_transition,
                        loggf,
                        lower_level,
                        upper_level,
                        bb_id,
                    )
                    bb_id += 1
                    bb_transition.find_level_id(self.ec_levels)
                    self.bb_transitions.append(bb_transition)
        #self.bb_length = len(self.bb_transitions)
        self.sort_bb_transitions()

    def write_model_atom(
        self, path, label_length=10, remove_unmatched_transitions=True
    ):
        """
        Writes the model atom to a file
        :param path: Path to the file
        :param label_length: Label length of the energy levels
        :param remove_unmatched_transitions: Whether to remove the BB transitions that do not have a matching energy level
        """
        with open(path, "w") as file:
            # write the header comments
            for header_comment in self.header_comments:
                file.write(f"{header_comment}")
            # write top lines
            file.write(f"{self.atom_element}\n")
            file.write(f"{self.atom_abundance:>8.2f}  {self.atom_mass:>7.3f}\n")
            file.write(
                f"{self.ec_length():>6}{self.bb_length():>6}{self.bf_length():>6}{self.nrfix:>6}\n"
            )
            # write the energy levels
            file.write(f"*  EC          G       LABEL                   ION\n")
            for ec_level in self.ec_levels:
                ec_level.write_to_file(file, label_length=label_length)
            # write the BB transitions
            file.write(
                f"* UP LOW      F        NQ  QMAX   Q0    IW    GA            GV   GQ\n"
            )
            bb_transition_id = 1
            removed_bb_transitions = 0
            for bb_transition in self.bb_transitions:
                if (bb_transition.lower_level_id < 0 or bb_transition.upper_level_id < 0) and remove_unmatched_transitions:
                    # delete this transition
                    print(f"Deleting transition {bb_transition}")
                    self.bb_transitions.remove(bb_transition)
                    removed_bb_transitions += 1
                else:
                    bb_transition.write_to_file(file, id_overwrite=bb_transition_id, label_length=label_length)
                    bb_transition_id += 1
            if removed_bb_transitions > 0:
                print(f"Removed {removed_bb_transitions} transitions")
            # write the BF transitions
            file.write(f"*  PHOTO-ION:  UP  LO  F   NQ  QMAX   Q0\n")
            for bf_transition in self.bf_transitions:
                bf_transition.write_to_file(file)
            # write the collisional transitions
            file.write(f"GENCOL\n")
            current_temperature_values = []
            for collisional_transition in self.collisional_transitions:
                # very stupid way, but have to make sure that each collisional data has correct temperature values
                if not collisional_transition.check_if_same_temperature_values(current_temperature_values):
                    collisional_transition.write_temperature_values_to_file(file)
                    current_temperature_values = collisional_transition.temperature_values
                collisional_transition.write_to_file(file)
            file.write(f"END\n")

    def read_model_atom(self, path, atom_version=None, read_collisions=True, sort_bb_transitions=True):
        # possible atom versions are MB and FORMATO2
        potential_atom_versions = ["MB", "FORMATO2"]
        if atom_version is not None and atom_version not in potential_atom_versions:
            raise ValueError(f"Atom version must be one of {potential_atom_versions}")

        # get basename of the path
        basename = os.path.basename(path)
        self.atom_filename = basename
        if atom_version is None:
            try:
                self.read_model_atom_version(path, "MB", read_collisions, sort_bb_transitions)
            except ValueError:
                self.read_model_atom_version(path, "FORMATO2", read_collisions, sort_bb_transitions)
        else:
            self.read_model_atom_version(path, atom_version, read_collisions, sort_bb_transitions)

    def _read_top_lines_atom(self, lines):
        headers = []
        # read lines skipping any comments (starting with *) until * EC
        line_index = 0
        line_index_old = 0
        # first line is the element
        line, line_index = self._read_atom_skip_comments(lines, line_index)
        # append lines[line_index_old:lind_index+1] to headers
        if line_index_old < line_index:
            headers.extend(lines[line_index_old:line_index - 1])
        line_index_old = line_index
        self.atom_element = line.strip().split()[0]
        # second line is the abundance and mass
        line, line_index = self._read_atom_skip_comments(lines, line_index)
        if line_index_old < line_index:
            headers.extend(lines[line_index_old:line_index - 1])
        line_index_old = line_index
        # convert to float
        try:
            self.atom_abundance, self.atom_mass = [float(x) for x in line.split()]
        except ValueError:
            line_strip = line.strip().split()
            # just take random abundance because none is given
            self.atom_abundance = 5.0
            print(f"Warning! Abundance is not given in {self.atom_filename}, using 5.0")
            self.atom_mass = float(line_strip[1])
        # third line is the number of energy levels, number of transitions, number of radiative fixations and number of collisional fixations
        line, line_index = self._read_atom_skip_comments(lines, line_index)
        if line_index_old < line_index:
            headers.extend(lines[line_index_old:line_index - 1])
        line_index_old = line_index
        ec_length, bb_length, bf_length, self.nrfix = [
            int(x) for x in line.split()
        ]
        # read until one of the potential headers is found
        potential_header_ec = [
            "ec",
            "e[cm⁻¹]",
            "energy",
            "ecm",
            "e[cm]"
        ]
        while True:
            if len(lines[line_index].split()) > 1:
                if not any([lines[line_index].split()[1].lower().startswith(x) for x in potential_header_ec]):
                    line_index += 1
                else:
                    break
            else:
                line_index += 1
        if line_index_old < line_index:
            headers.extend(lines[line_index_old:line_index])
        self.header_comments = headers
        return bb_length, bf_length, ec_length, line_index

    def _read_ec_levels(self, ec_length, line_index, lines, atom_version:str):
        """

        :param ec_length:
        :param line_index:
        :param lines:
        :param atom_version: MB or FORMATO2
        :return:
        """
        ionisation_ec = 0
        current_ionisation_level = 1
        ec_levels = []
        # now read the energy levels
        for i in range(ec_length):
            line, line_index = self._read_atom_skip_comments(lines, line_index)
            fields = line.strip().split()

            ec = float(fields[0])
            g = float(fields[1])
            if atom_version == "MB":
                # MB version: level ID is inside the string
                try:
                    atom_level = int(fields[3])
                    atom_label = "".join(fields[5:-1]).replace("'", "")
                except ValueError:
                    atom_level = int(fields[4])
                    atom_label = "".join(fields[6:-1]).replace("'", "")
                ion = int(fields[-1])
            elif atom_version == "FORMATO2":
                # FORMATO2 version: level ID is the last field
                atom_label = fields[2].replace("'", "")
                if len(fields) > 5:
                    atom_label += fields[3].replace("'", "")
                atom_level = int(fields[-1])
                ion = int(fields[-2])
            if ion >= current_ionisation_level and i < ec_length - 1:
                ionisation_ec = ec
                current_ionisation_level += 1
            ec_level = ECLevel(atom_level, ec, g, "", ion, atom_label, ionisation_ec)
            ec_levels.append(ec_level)
        return current_ionisation_level, ec_levels, line_index

    def _read_bb_transitions(self, bb_length, line_index, lines, atom_version:str, sort_bb_transitions=True):
        current_bb_index = 0
        # first check if there is a comment
        while current_bb_index < bb_length:
            if atom_version == "MB":
                while lines[line_index].startswith("*"):
                    # there is a comment
                    line_split = lines[line_index].strip().split()
                    if self.check_if_relevant_comment(line_index, line_split, lines):
                        # means that it is comment relevant to the transition
                        # replace ";" with nothing in line_split
                        line_split = [x.replace(";", "") for x in line_split]
                        try:
                            bb_id, bb_wavelength, bb_loggf = [float(s) for s in line_split if is_number(s)][:3]
                        except ValueError:
                            # replace = with space
                            line_split = (
                                lines[line_index].replace("=", "= ", 1).strip().split()
                            )
                            bb_id, bb_wavelength, bb_loggf = [float(s) for s in line_split if is_number(s)][:3]

                        # if line_split[-1] ends with "AXR" kinda stuff, then it is a lower level label
                        if len(line_split[-1]) == 3 and line_split[-1][1] == "X":
                            lower_level_label = line_split[-3]
                            upper_level_label = f"{line_split[-2]} {line_split[-1]}"
                        else:
                            lower_level_label = line_split[-2]
                            upper_level_label = line_split[-1]

                        line_index += 1
                        break
                    else:
                        # then this is just a comment
                        line_index += 1
            elif atom_version == "FORMATO2":
                # No comments above BB transitions
                pass

            # skip comment
            while lines[line_index][0] == "*":
                line_index += 1

            # now read the transition
            fields = lines[line_index].strip().split()
            line_index += 1
            upper_level_id = int(fields[0])
            lower_level_id = int(fields[1])
            f_value = float(fields[2])
            nq = int(fields[3])
            qmax = float(fields[4])
            q0 = float(fields[5])
            iw = int(fields[6])
            ga = float(fields[7])
            gv = float(fields[8])
            gq = float(fields[9])
            if atom_version == "MB":
                line_profile = fields[10]
            elif atom_version == "FORMATO2":
                # sometimes there is no line profile
                try:
                    bb_wavelength = float(fields[10])
                    line_profile = "VOIGT"
                    bb_id = int(fields[11])
                except ValueError:
                    line_profile = fields[10]
                    bb_wavelength = float(fields[11])
                    bb_id = int(fields[12])

                # need to calculate bb_loggf manually :( from f_value
                bb_loggf = np.log10(
                    f_value * self.ec_levels[lower_level_id - 1].g_value
                )

            # find the ec of the lower and upper level
            lower_level_ec = self.ec_levels[lower_level_id - 1].ec_value
            upper_level_ec = self.ec_levels[upper_level_id - 1].ec_value
            ion_transition = self.ec_levels[lower_level_id - 1].ionisation_stage

            bb_transition = BBTransition(
                bb_wavelength,
                ion_transition,
                bb_loggf,
                lower_level_ec,
                upper_level_ec,
                bb_id,
            )
            bb_transition.lower_level_id = lower_level_id
            bb_transition.upper_level_id = upper_level_id
            bb_transition.f_value = f_value
            bb_transition.nq = nq
            bb_transition.qmax = qmax
            bb_transition.q0 = q0
            bb_transition.iw = iw
            bb_transition.ga = ga
            bb_transition.gv = gv
            bb_transition.gq = gq
            bb_transition.line_profile = line_profile
            if atom_version == "MB":
                bb_transition.lower_level_label = lower_level_label
                bb_transition.upper_level_label = upper_level_label
            elif atom_version == "FORMATO2":
                bb_transition.lower_level_label = self.ec_levels[lower_level_id - 1].label_name
                bb_transition.upper_level_label = self.ec_levels[upper_level_id - 1].label_name
            bb_transition.lower_level_g_value = self.ec_levels[lower_level_id - 1].g_value

            # check if there is hyperfine structure
            if bb_transition.iw > 1:
                bb_transition.hyperfine = True
                # read hyperfine structure line by line
                for j in range(bb_transition.iw):
                    line_split = lines[line_index].strip().split()
                    line_index += 1
                    delta_ec = float(line_split[0])
                    weight = float(line_split[1])
                    f_value = float(line_split[2])
                    ga = float(line_split[3])
                    gw = float(line_split[4])
                    gq = float(line_split[5])
                    if len(line_split) == 7:
                        line_profile = line_split[6]
                    else:
                        line_profile = bb_transition.line_profile
                    bb_transition.hyperfine_delta_ec.append(delta_ec)
                    bb_transition.hyperfine_weight.append(weight)
                    bb_transition.hyperfine_f_value.append(f_value)
                    bb_transition.hyperfine_ga.append(ga)
                    bb_transition.hyperfine_gw.append(gw)
                    bb_transition.hyperfine_gq.append(gq)
                    bb_transition.hyperfine_line_profile.append(line_profile)

            self.bb_transitions.append(bb_transition)
            current_bb_index += 1

        if sort_bb_transitions or atom_version == "FORMATO2":
            self.sort_bb_transitions()
        return line_index

    def _read_bf_transitions(self, bf_length, line_index, lines, atom_version:str):
        current_bf_index = 0
        while current_bf_index < bf_length:
            if not lines[line_index].startswith("*"):
                # skip comment
                line_split = lines[line_index].strip().split()
                line_index += 1
                upper_level_id = int(line_split[0])
                lower_level_id = int(line_split[1])
                a0 = float(line_split[2])
                nq = int(line_split[3])
                lmax = float(line_split[4])
                if len(line_split) == 6 and atom_version == "MB":
                    # then we have lmax and lmin, but nq is probably 0
                    lmin = float(line_split[5])
                    bf_transition = BFTransition(
                        lower_level_id, upper_level_id, a0, nq, lmax, lmin=lmin
                    )
                elif line_split[4] != "-1" and atom_version == "FORMATO2":
                    # then we have lmax and lmin
                    lmin = 0.0
                    bf_transition = BFTransition(
                        lower_level_id, upper_level_id, a0, nq, lmax, lmin=lmin
                    )
                else:
                    bf_transition = BFTransition(
                        lower_level_id, upper_level_id, a0, nq, lmax, None
                    )
                    # now read through the wavelengths and cross-sections
                    for i in range(nq):
                        line_split = lines[line_index].strip().split()
                        line_index += 1
                        wavelength = float(line_split[0])
                        alpha_cross_section = float(line_split[1])
                        bf_transition.wavelengths[i] = wavelength
                        bf_transition.alpha_cross_sections[i] = alpha_cross_section

                self.bf_transitions.append(bf_transition)
                current_bf_index += 1
            else:
                line_index += 1
        return line_index

    def _read_collisional_data(self, line_index, lines):
        current_temperature_values = []
        current_values_count = 0
        # now collisional data.... man this is a mess
        while line_index < len(lines):
            # first ignore comments
            if (
                not lines[line_index].startswith("*")
                and not lines[line_index].startswith("END")
                and not lines[line_index].startswith("GENCOL")
            ):
                if lines[line_index].startswith("TEMP"):
                    current_temperature_values = []
                    line_index += 1
                    line_split = lines[line_index].strip().split()
                    line_index += 1
                    current_values_count = int(line_split[0])
                    for i in range(1, current_values_count + 1):
                        temperature_value = float(line_split[i])
                        current_temperature_values.append(temperature_value)
                else:
                    # first line is the label
                    split_line = lines[line_index].strip().split()
                    label = split_line[0]
                    if label == "CH_CE":
                        label = "CH0"
                    skip_line = False
                    if label == "CI" and len(split_line) != 1:
                        if split_line[1] == "SEATON":
                            # skip CI if Seaton is used, so we delete these lines apparently
                            # Source: MB
                            skip_line = True
                    if not skip_line:
                        line_index += 1
                        # next line is the upper level and lower level
                        line_split = lines[line_index].strip().split()
                        line_index += 1
                        upper_level = int(line_split[0])
                        lower_level = int(line_split[1])
                        # all other values are collisional values
                        collisional_values = []
                        for i in range(2, current_values_count + 2):
                            collisional_value = float(line_split[i])
                            collisional_values.append(collisional_value)
                        collisional_transition = CollisionalTransition(
                            label,
                            upper_level,
                            lower_level,
                            current_values_count,
                            collisional_values,
                            current_temperature_values,
                        )
                        self.collisional_transitions.append(collisional_transition)
                    else:
                        line_index += 2
            else:
                line_index += 1

    def read_model_atom_version(self, path, atom_version:str, read_collisions=True, sort_bb_transitions=True):
        with open(path, "r") as file:
            lines = file.readlines()

        bb_length, bf_length, ec_length, line_index = self._read_top_lines_atom(lines)

        self.max_ion, self.ec_levels, line_index = self._read_ec_levels(ec_length, line_index, lines, atom_version)

        line_index = self._read_atom_skip_comments(lines, line_index)[1] - 2

        line_index = self._read_bb_transitions(bb_length, line_index, lines, atom_version, sort_bb_transitions=sort_bb_transitions)

        line_index = self._read_bf_transitions(bf_length, line_index, lines, atom_version)

        if read_collisions:
            self._read_collisional_data(line_index, lines)

    @staticmethod
    def check_if_relevant_comment(line_index, line_split, lines):
        possible_names_for_id = ["id", "i", "kr"]
        # check if the length of the comment line is at least 2, meaning that it is a comment with at least some stuff. maybe increase?
        check_length = len(line_split) >= 2
        if not check_length:
            return False
        # check if the first element is id or i, meaning that it is a comment relevant to the transition
        check_id = line_split[1] in possible_names_for_id or line_split[0].replace("*", "") in possible_names_for_id
        # check that it is not double comment, because that usually implies that this line was commented out completely
        check_double_comment = lines[line_index][1] != "*"
        # check that the next line is not a comment with a transition
        # we check that by checking that the next 2 lines are comments AND that the next-next line has id or i
        try:
            check_next_line = not (
                lines[line_index + 1].split()[0] == "*"
                and lines[line_index + 2].split()[0] == "*"
                and lines[line_index + 2].split()[1] in possible_names_for_id
            )
        except IndexError:
            # if we are at the end of the transitions or whatever else, then let's just accept it
            check_next_line = True
        comment_is_relevant = (
            check_length and check_id and check_double_comment and check_next_line
        )
        return comment_is_relevant

    @staticmethod
    def _read_atom_skip_comments(lines: list, line_index: int) -> tuple[str, int]:
        while lines[line_index].startswith("*"):
            line_index += 1
        return lines[line_index], line_index + 1



if __name__ == '__main__':
    fe_atom = ModelAtom()
    fe_atom.read_model_atom("atom.fe607a_nq_p")

    # change nq value for lines with following bb_ids
    wavelengths = np.loadtxt("fee", usecols=(0,), unpack=True, dtype=float, comments=";")
    # find ids in the wavelengths
    for wavelength in wavelengths:
        for bb_transition in fe_atom.bb_transitions:
            if abs(bb_transition.wavelength_aa - wavelength) <= 0.01:
                # change nq value
                #bb_transition.nq = 121
                #print(f"Changed nq for {bb_transition.wavelength_aa} {wavelength} to 121")
                print(f"{wavelength}: {int(bb_transition.bb_id - 1)}, ", end="")

    #fe_atom.write_model_atom("atom.fe607a_nq")



    #atom = ModelAtom()
    #atom.read_model_atom("/Users/storm/docker_common_folder/TSFitPy/input_files/nlte_data/model_atoms/atom.mg86b")

    #wavelength, ion_transition, loggf = 5528.411, 1, -0.498
    #lower_level_ec, upper_level_ec = 35051.36, 53134.70
    #ga = 0
    #gv = 0

    #new_line = BBTransition(wavelength, ion_transition, loggf, lower_level_ec, upper_level_ec)
    #new_line.find_level_id(atom.ec_levels)
    #new_line.ga = ga
    #new_line.gv = gv
    #atom.add_bb_transition(new_line)
    #atom.write_model_atom("atom.mg2")
    exit()
    # load Y loggf
    with open("/Users/storm/docker_common_folder/TSFitPy/scripts/y_loggf.txt") as file:
        lines = file.readlines()
    y_loggf = {}
    for line in lines:
        fields = line.strip().split()
        wavelength = float(fields[0]) * 10
        loggf_uncorr = float(fields[7])
        loggf_corr = float(fields[10])
        reference = fields[-1]
        y_loggf[wavelength] = {"reference": reference, "loggf_uncorr": loggf_uncorr, "loggf_corr": loggf_corr}

    atom_y = ModelAtom()
    atom_y.read_model_atom("/Users/storm/docker_common_folder/TSFitPy/input_files/nlte_data/model_atoms/atom.y423qm")
    # change loggf for several lines. go through all BB transitions and check if the wavelength is in the dictionary within 0.1 AA
    for bb_transition in atom_y.bb_transitions:
        for wavelength in y_loggf:
            if abs(bb_transition.wavelength_aa - wavelength) <= 0.001:
                # if reference is T, then take the corrected loggf, otherwise take the uncorrected loggf
                if y_loggf[wavelength]["reference"] == "T" or y_loggf[wavelength]["reference"] == "B":
                    bb_transition.loggf = y_loggf[wavelength]["loggf_corr"]
                else:
                    bb_transition.loggf = y_loggf[wavelength]["loggf_uncorr"]
                # recalculate f_value
                bb_transition.calculate_f_value(bb_transition.lower_level_g_value)
                # remove the wavelength from the dictionary
                del y_loggf[wavelength]
                break

    atom_y.write_model_atom("./atom.y423qmh_pf")