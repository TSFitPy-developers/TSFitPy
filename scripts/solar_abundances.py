# -*- coding: utf-8 -*-

"""
A list of solar abundances.

Changed by Gerber Nov. 2019
Source: Asplund et al. 2009
Updated to Magg et al. 2022 on Feb. 5 2022
"""
solar_abundances = {
    "STD?": "T",
    "[Fe/H]": 0.00,
    "[alpha/Fe]": 0.00,
    "[C/Fe]": 0.00,
    "[N/Fe]": 0.00,
    "[O/Fe]": 0.00,
    "[r/Fe]": 0.00,
    "[s/Fe]": 0.00,
    "C/O": 0.54954,
    "X": 0.73826,
    "Y_": 0.24954,  # because Yttrium is also Y, so Y_ is used for Helium
    "Z": 1.22E-02,
    "H": 12.00,
    "He": 10.93,
    "Li": 1.05,
    "Be": 1.38,
    "B": 2.70,
    "C": 8.56,
    "N": 7.98,
    "O": 8.77,
    "F": 4.40,
    "Ne": 8.16,
    "Na": 6.29,
    "Mg": 7.55,
    "Al": 6.43,
    "Si": 7.59,
    "P": 5.41,
    "S": 7.16,
    "Cl": 5.25,
    "Ar": 6.40,
    "K": 5.14,
    "Ca": 6.37,
    "Sc": 3.07,
    "Ti": 4.94,
    "V": 3.89,
    "Cr": 5.74,
    "Mn": 5.52,
    "Fe": 7.50,
    "Co": 4.95,
    "Ni": 6.24,
    "Cu": 4.19,
    "Zn": 4.56,
    "Ga": 3.04,
    "Ge": 3.65,
    "As": 2.29, #not in asplund, kept old value
    "Se": 3.33, #ditto
    "Br": 2.56, #ditto
    "Kr": 3.25,
    "Rb": 2.52,
    "Sr": 2.87,
    "Y": 2.21,
    "Zr": 2.58,
    "Nb": 1.46,
    "Mo": 1.88,
    "Tc": -99.00, #ditto
    "Ru": 1.75,
    "Rh": 0.91,
    "Pd": 1.57,
    "Ag": 0.94,
    "Cd": 1.77, #ditto
    "In": 0.80,
    "Sn": 2.04,
    "Sb": 1.00, #ditto
    "Te": 2.19, #ditto
    "I": 1.51, #ditto
    "Xe": 2.24,
    "Cs": 1.07, #ditto
    "Ba": 2.18,
    "La": 1.10,
    "Ce": 1.58,
    "Pr": 0.72,
    "Nd": 1.42,
    "Pm": -99.00, #ditto
    "Sm": 0.96,
    "Eu": 0.52,
    "Gd": 1.07,
    "Tb": 0.30,
    "Dy": 1.10,
    "Ho": 0.48,
    "Er": 0.92,
    "Tm": 0.10,
    "Yb": 0.84,
    "Lu": 0.10,
    "Hf": 0.85,
    "Ta": -0.17, #ditto
    "W": 0.85,
    "Re": 0.23, #ditto
    "Os": 1.40,
    "Ir": 1.38,
    "Pt": 1.64, #ditto
    "Au": 0.92,
    "Hg": 1.13, #ditto
    "Tl": 0.90,
    "Pb": 1.75,
    "Bi": 0.65, #ditto
    "Po": -99.00, #ditto
    "At": -99.00, #ditto
    "Rn": -99.00, #ditto
    "Fr": -99.00, #ditto
    "Ra": -99.00, #ditto
    "Ac": -99.00, #ditto
    "Th": 0.02,
    "Pa": -99.00, #ditto
    "U": -0.52 #ditto
}

"""
A list of solar abundances.

These values were supplied by Bengt Edvardsson, June 2017
Source: Grevesse et al. 2007, Space Sci Rev 130,105
"""

#solar_abundances = {
#    "STD?": "T",
#    "[Fe/H]": 0.00,
#    "[alpha/Fe]": 0.00,
#    "[C/Fe]": 0.00,
#    "[N/Fe]": 0.00,
#    "[O/Fe]": 0.00,
#    "[r/Fe]": 0.00,
#    "[s/Fe]": 0.00,
#    "C/O": 0.537,
#    "X": 0.73826,
#    "Y": 0.24954,
#    "Z": 1.22E-02,
#    "H": 12.00,
#    "He": 10.93,
#    "Li": 1.05,
#    "Be": 1.38,
#    "B": 2.70,
#    "C": 8.39,
#    "N": 7.78,
#    "O": 8.66,
#    "F": 4.56,
#    "Ne": 7.84,
#    "Na": 6.17,
#    "Mg": 7.53,
#    "Al": 6.37,
#    "Si": 7.51,
#    "P": 5.36,
#    "S": 7.14,
#    "Cl": 5.50,
#    "Ar": 6.18,
#    "K": 5.08,
#    "Ca": 6.31,
#    "Sc": 3.17,
#    "Ti": 4.90,
#    "V": 4.00,
#    "Cr": 5.64,
#    "Mn": 5.39,
#    "Fe": 7.45,
#    "Co": 4.92,
#    "Ni": 6.23,
#    "Cu": 4.21,
#    "Zn": 4.60,
#    "Ga": 2.88,
#    "Ge": 3.58,
#    "As": 2.29,
#    "Se": 3.33,
#    "Br": 2.56,
#    "Kr": 3.25,
#    "Rb": 2.60,
#    "Sr": 2.92,
#    "Y": 2.21,
#    "Zr": 2.58,
#    "Nb": 1.42,
#    "Mo": 1.92,
#    "Tc": -99.00,
#    "Ru": 1.84,
#    "Rh": 1.12,
#    "Pd": 1.66,
#    "Ag": 0.94,
#    "Cd": 1.77,
#    "In": 1.60,
#    "Sn": 2.00,
#    "Sb": 1.00,
#    "Te": 2.19,
#    "I": 1.51,
#    "Xe": 2.24,
#    "Cs": 1.07,
#    "Ba": 2.17,
#    "La": 1.13,
#    "Ce": 1.70,
#    "Pr": 0.58,
#    "Nd": 1.45,
#    "Pm": -99.00,
#    "Sm": 1.00,
#    "Eu": 0.52,
#    "Gd": 1.11,
#    "Tb": 0.28,
#    "Dy": 1.14,
#    "Ho": 0.51,
#    "Er": 0.93,
#    "Tm": 0.00,
#    "Yb": 1.08,
#    "Lu": 0.06,
#    "Hf": 0.88,
#    "Ta": -0.17,
#    "W": 1.11,
#    "Re": 0.23,
#    "Os": 1.25,
#    "Ir": 1.38,
#    "Pt": 1.64,
#    "Au": 1.01,
#    "Hg": 1.13,
#    "Tl": 0.90,
#    "Pb": 2.00,
#    "Bi": 0.65,
#    "Po": -99.00,
#    "At": -99.00,
#    "Rn": -99.00,
#    "Fr": -99.00,
#    "Ra": -99.00,
#    "Ac": -99.00,
#    "Th": 0.06,
#    "Pa": -99.00,
#    "U": -0.52
#}

periodic_table = [
    "",
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La",
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U"]

molecules_atomic_number = {"CH": ["0106.000000", "C H"], "CN": ["0607.000000", "C N"]}