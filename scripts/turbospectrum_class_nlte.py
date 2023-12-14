from __future__ import annotations

import subprocess
import os
from os import path as os_path
import glob
from operator import itemgetter
import numpy as np
import math
import logging

from scripts.auxiliary_functions import closest_available_value
from scripts.solar_abundances import solar_abundances, periodic_table, molecules_atomic_number
from scripts.solar_isotopes import solar_isotopes
from scripts.synthetic_code_class import SyntheticSpectrumGenerator


class TurboSpectrum(SyntheticSpectrumGenerator):
    """
    A class which wraps Turbospectrum.

    This wrapper currently does not include any provision for macro-turbulence, which should be applied subsequently, by
    convolving the output spectrum with some appropriate line spread function.
    """

    # Default MARCS model settings to look for. These are fixed parameters which we don't (currently) allow user to vary

    # In this context, "turbulence" refers to the micro turbulence assumed in the MARCS atmosphere. It should closely
    # match the micro turbulence value passed to babsma below.
    marcs_parameters = {"turbulence": 1, "model_type": "st",
                        "a": 0, "c": 0, "n": 0, "o": 0, "r": 0, "s": 0}

    # It is safe to ignore these parameters in MARCS model descriptions
    # This includes interpolating between models with different values of these settings
    marcs_parameters_to_ignore = ["a", "c", "n", "o", "r", "s"]


    def __init__(self, turbospec_path: str, interpol_path: str, line_list_paths: str, marcs_grid_path: str,
                 marcs_grid_list: str, model_atom_path: str, departure_file_path: str, aux_file_length_dict: dict,
                 marcs_value_keys: list, marcs_values: dict, marcs_models: dict, model_temperatures: np.ndarray,
                 model_logs: np.ndarray, model_mets: np.ndarray):
        """
        Instantiate a class for generating synthetic stellar spectra using Turbospectrum.

        :param turbospec_path: Path where the turbospectrum binaries 'babsma' and 'bsyn' can be found.
        :param interpol_path: Path where the compiled interpol_modeles.f binary can be found.
        :param line_list_paths: Path(s) where line lists for synthetic spectra can be found. Specify as either a string, or a list of strings.
        :param marcs_grid_path: Path where a grid of MARCS .mod files can be found. These contain the model atmospheres we use.
        :param model_atom_path: Path to the model atom paths
        :param departure_file_path: Path to the NLTE departure file paths
        """
        super().__init__(turbospec_path, interpol_path, line_list_paths, marcs_grid_path, marcs_grid_list,
                         model_atom_path, marcs_value_keys, marcs_values, marcs_models, model_temperatures,
                         model_logs, model_mets)

        self.departure_file_path = departure_file_path
        self.aux_file_length_dict = aux_file_length_dict

        # Default spectrum parameters
        self.lpoint = 1000000  # number of points in TS

        # parameters needed for nlte and <3D> calculations
        self.windows_flag: bool = False
        self.depart_bin_file = None
        self.depart_aux_file = None
        self.model_atom_file = None
        self.segment_file = None
        self.cont_mask_file = None
        self.line_mask_file = None

    def configure(self, lambda_min: float=None, lambda_max:float=None, lambda_delta: float=None,
                  metallicity: float=None, log_g: float=None, t_eff: float=None, stellar_mass: float=None,
                  turbulent_velocity: float=None, free_abundances=None, free_isotopes=None,
                  sphere=None, alpha=None, s_process=None, r_process=None,
                  line_list_paths=None, line_list_files=None,
                  verbose=None, counter_spectra=None, temp_directory=None, nlte_flag: bool = None, atmosphere_dimension=None,
                  windows_flag=None,
                  depart_bin_file=None, depart_aux_file=None, model_atom_file=None,
                  segment_file=None, cont_mask_file=None, line_mask_file=None, lpoint=None):
        """
        Set the stellar parameters of the synthetic spectra to generate. This can be called as often as needed
        to generate many synthetic spectra with one class instance. All arguments are optional; any which are not
        supplied will remain unchanged.

        :param lambda_min:
            Short wavelength limit of the synthetic spectra we generate. Unit: A.
        :param lambda_max:
            Long wavelength limit of the synthetic spectra we generate. Unit: A.
        :param lambda_delta:
            Wavelength step of the synthetic spectra we generate. Unit: A.
        :param metallicity:
            Metallicity of the star we're synthesizing.
        :param t_eff:
            Effective temperature of the star we're synthesizing.
        :param log_g:
            Log(gravity) of the star we're synthesizing.
        :param stellar_mass:
            Mass of the star we're synthesizing (solar masses).
        :param turbulent_velocity:
            Micro turbulence velocity in km/s
        :param free_abundances:
            List of elemental abundances to use in stellar model. These are passed to Turbospectrum.
        :param sphere:
            Select whether to use a spherical model (True) or a plane-parallel model (False).
        :param alpha:
            Alpha enhancement to use in stellar model.
        :param s_process:
            S-Process element enhancement to use in stellar model.
        :param r_process:
            R-Process element enhancement to use in stellar model.
        :param line_list_paths:
            List of paths where we should search for line lists.
        :param line_list_files:
            List of line list files to use. If not specified, we use all files in `line_list_paths`
        :param verbose:
            Let Turbospectrum print debugging information to terminal?
        :return:
            None
        """

        if lambda_min is not None:
            self.lambda_min = lambda_min
        if lambda_max is not None:
            self.lambda_max = lambda_max
        if lambda_delta is not None:
            self.lambda_delta = lambda_delta
        if metallicity is not None:
            self.metallicity = metallicity
        if t_eff is not None:
            self.t_eff = t_eff
        if log_g is not None:
            self.log_g = log_g
        if stellar_mass is not None:
            self.stellar_mass = stellar_mass
        if turbulent_velocity is not None:
            self.turbulent_velocity = turbulent_velocity
        if free_abundances is not None:
            self.free_abundances = free_abundances  # [X/H]
        if free_isotopes is not None:
            self.free_isotopes = free_isotopes
        if sphere is not None:
            self.sphere = sphere
        if alpha is not None:
            self.alpha = alpha
        if s_process is not None:
            self.s_process = s_process
        if r_process is not None:
            self.r_process = r_process
        if line_list_paths is not None:
            if not isinstance(line_list_paths, (list, tuple)):
                line_list_paths = [line_list_paths]
            self.line_list_paths = line_list_paths
        if line_list_files is not None:
            self.line_list_files = line_list_files
        if verbose is not None:
            self.verbose = verbose
        if counter_spectra is not None:
            self.counter_spectra = counter_spectra
        if temp_directory is not None:
            self.tmp_dir = temp_directory
        if nlte_flag is not None:
            self.nlte_flag = nlte_flag
        if atmosphere_dimension is not None:
            self.atmosphere_dimension = atmosphere_dimension
        if windows_flag is not None:
            self.windows_flag = windows_flag
        if depart_bin_file is not None:
            self.depart_bin_file = depart_bin_file
        if depart_aux_file is not None:
            self.depart_aux_file = depart_aux_file
        if model_atom_file is not None:
            self.model_atom_file = model_atom_file
        if segment_file is not None:
            self.segment_file = segment_file
        if cont_mask_file is not None:
            self.cont_mask_file = cont_mask_file
        if line_mask_file is not None:
            self.line_mask_file = line_mask_file
        if self.atmosphere_dimension == "3D":
            self.turbulent_velocity = 2.0
            #print("turbulent_velocity is not used since model atmosphere is 3D")
        if lpoint is not None:
            self.lpoint = lpoint


    def make_species_lte_nlte_file(self):
        """
        Generate the SPECIES_LTE_NLTE.dat file for TS to determine what elements are NLTE
        """
        data_path = self.tmp_dir
        file = open("{}/SPECIES_LTE_NLTE_{:08d}.dat".format(data_path, self.counter_spectra), 'w')
        file.write("# This file controls which species are treated in LTE/NLTE\n")
        file.write("# It also gives the path to the model atom and the departure files\n")
        file.write("# First created 2021-02-22\n")
        file.write("# if a species is absent it is assumed to be LTE\n")
        file.write("#\n")
        file.write("# each line contains :\n")
        file.write("# atomic number / name / (n)lte / model atom / departure file / binary or ascii departure file\n")
        file.write("#\n")
        file.write("# path for model atom files     ! don't change this line !\n")
        file.write(f"{self.model_atom_path}\n")
        file.write("#\n")
        file.write("# path for departure files      ! don't change this line !\n")
        file.write(f"{self.tmp_dir}\n")
        file.write("#\n")
        file.write("# atomic (N)LTE setup\n")
        if self.nlte_flag:
            for element in self.model_atom_file:
                # write all nlte elements
                if element in molecules_atomic_number:
                    # so if a molecule is given, get "atomic number" from the separate dictionary #TODO improve to do automatically not just for select molecules?
                    atomic_number = molecules_atomic_number[element][0]
                    element_name_to_write = molecules_atomic_number[element][1]
                else:
                    atomic_number = periodic_table.index(element)
                    element_name_to_write = element
                file.write(f"{atomic_number}  '{element_name_to_write}'  'nlte'  '{self.model_atom_file[element]}'   '{self.marcs_model_name}_{element}_coef.dat' 'ascii'\n")
            for element in self.free_abundances:
                # now check for any lte elements which have a specific given abundance and write them too
                atomic_number = periodic_table.index(element)
                if element not in self.model_atom_file:
                    file.write(f"{atomic_number}  '{element}'  'lte'  ''   '' 'ascii'\n")
        else:
            for element in self.free_abundances:
                atomic_number = periodic_table.index(element)
                file.write(f"{atomic_number}  '{element}'  'lte'  ''   '' 'ascii'\n")
        file.close()

    def make_babsma_bsyn_file(self, spherical):
        """
        Generate the configurations files for both the babsma and bsyn binaries in Turbospectrum.
        """

        # If we've not been given an explicit alpha enhancement value, assume one
        alpha = self.alpha
        if alpha is None:
            if self.metallicity < -1.0:
                alpha = 0.4
            elif -1.0 < self.metallicity < 0.0:
                alpha = -0.4 * self.metallicity
            else:
                alpha = 0

        # Updated abundances to below to allow user to set solar abundances through solar_abundances.py and not have to adjust make_abund.f

        individual_abundances = "'INDIVIDUAL ABUNDANCES:'   '{:d}'\n".format(len(periodic_table) - 1)
        
        item_abund = {}
        item_abund['H'] = 12.00
        item_abund[periodic_table[2]] = float(solar_abundances[periodic_table[2]])  # Helium is always constant, no matter the metallicity
        for i in range(3, len(periodic_table)):
            # first take solar scaled abundances as A(X)
            item_abund[periodic_table[i]] = float(solar_abundances[periodic_table[i]]) + round(float(self.metallicity), 6)
        if self.free_abundances is not None:
            # and if any abundance is passed, take it and convert to A(X)
            for element, abundance in self.free_abundances.items():
                item_abund[element] = float(solar_abundances[element]) + round(float(abundance), 6)
        for i in range(1, len(periodic_table)):
            individual_abundances += "{:d}  {:.6f}\n".format(i, item_abund[periodic_table[i]])

        # Allow for user input isotopes as a dictionary (similar to abundances)

        individual_isotopes = f"'ISOTOPES : ' '{len(solar_isotopes)}'\n"
        if self.free_isotopes is None:
            for isotope, ratio in solar_isotopes.items():
                individual_isotopes += "{}  {:6f}\n".format(isotope, ratio)
        else:
            for isotope, ratio in self.free_isotopes.items():
                solar_isotopes[isotope] = ratio
            for isotope, ratio in solar_isotopes.items():
                individual_isotopes += "{}  {:6f}\n".format(isotope, ratio)

        # Make a list of line-list files
        # We start by getting a list of all files in the line list directories we've been pointed towards,
        # excluding any text files we find.
        line_list_files = []
        for line_list_path in self.line_list_paths:
            line_list_files.extend([i for i in glob.glob(os_path.join(line_list_path, "*")) if not i.endswith(".txt")])

        # If an explicit list of line_list_files is set, we treat this as a list of filenames within the specified
        # line_list_path, and we only allow files with matching filenames
        if self.line_list_files is not None:
            line_list_files = [item for item in line_list_files if os_path.split(item)[1] in self.line_list_files]

        # Encode list of line lists into a string to pass to bsyn
        line_lists = "'NFILES   :' '{:d}'\n".format(len(line_list_files))
        for item in line_list_files:
            line_lists += "{}\n".format(item)

        # Build bsyn configuration file
        spherical_boolean_code = "T" if spherical else "F"
        if self.atmosphere_dimension == "3D":
            spherical_boolean_code = "F"
        xifix_boolean_code = "T" if self.atmosphere_dimension == "1D" else "F"
        nlte_boolean_code = ".true." if self.nlte_flag == True else ".false."
        pure_lte_boolean_code = ".false." if self.nlte_flag == True else ".true."

        if self.windows_flag == True:
            bsyn_config = """\
'PURE-LTE  :'  '.false.'
'NLTE :'          '{nlte}'
'NLTEINFOFILE:'  '{this[tmp_dir]}SPECIES_LTE_NLTE_{this[counter_spectra]:08d}.dat'
#'MODELATOMFILE:'  '{this[model_atom_path]}{this[model_atom_file]}'
#'DEPARTUREFILE:'  '{this[tmp_dir]}{this[marcs_model_name]}_coef.dat'
#'DEPARTBINARY:'   '.false.'
#'CONTMASKFILE:'     '{this[cont_mask_file]}'
#'LINEMASKFILE:'     '{this[line_mask_file]}'
'SEGMENTSFILE:'     '{this[segment_file]}'
'LAMBDA_MIN:'    '{this[lambda_min]:.3f}'
'LAMBDA_MAX:'    '{this[lambda_max]:.3f}'
'LAMBDA_STEP:'   '{this[lambda_delta]:.3f}'
'INTENSITY/FLUX:' 'Flux'
'COS(THETA)    :' '1.00'
'ABFIND        :' '.false.'
'MODELOPAC:' '{this[tmp_dir]}model_opacity_{this[counter_spectra]:08d}.opac'
'RESULTFILE :' '{this[tmp_dir]}/spectrum_{this[counter_spectra]:08d}.spec'
'METALLICITY:'    '{this[metallicity]:.2f}'
'ALPHA/Fe   :'    '{alpha:.2f}'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '{this[r_process]:.2f}'
'S-PROCESS  :'    '{this[s_process]:.2f}'
{individual_abundances}
{individual_isotopes}
{line_lists}
'SPHERICAL:'  '{spherical}'
  30
  300.00
  15
  1.30
""".format(this=self.__dict__,
           alpha=alpha,
           spherical=spherical_boolean_code,
           individual_abundances=individual_abundances.strip(),
           individual_isotopes=individual_isotopes.strip(),
           line_lists=line_lists.strip(),
           pure_lte=pure_lte_boolean_code,
           nlte=nlte_boolean_code
           )

            # Build babsma configuration file
            babsma_config = """\
'PURE-LTE  :'  '.false.'
'LAMBDA_MIN:'    '{this[lambda_min]:.3f}'
'LAMBDA_MAX:'    '{this[lambda_max]:.3f}'
'LAMBDA_STEP:'    '{this[lambda_delta]:.3f}'
'MODELINPUT:' '{this[tmp_dir]}{this[marcs_model_name]}.interpol'
'MARCS-FILE:' '.false.'
'MODELOPAC:' '{this[tmp_dir]}model_opacity_{this[counter_spectra]:08d}.opac'
'METALLICITY:'    '{this[metallicity]:.2f}'
'ALPHA/Fe   :'    '{alpha:.2f}'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '{this[r_process]:.2f}'
'S-PROCESS  :'    '{this[s_process]:.2f}'
{individual_abundances}
'XIFIX:' '{xifix}'
{this[turbulent_velocity]:.2f}
""".format(this=self.__dict__,
           alpha=alpha,
           individual_abundances=individual_abundances.strip(),
           pure_lte=pure_lte_boolean_code,
           xifix=xifix_boolean_code
           )
        elif self.windows_flag == False:
            bsyn_config = """\
'PURE-LTE  :'  '.false.'
'NLTE :'          '{nlte}'
'NLTEINFOFILE:'  '{this[tmp_dir]}SPECIES_LTE_NLTE_{this[counter_spectra]:08d}.dat'
#'MODELATOMFILE:'  '{this[model_atom_path]}{this[model_atom_file]}'
#'DEPARTUREFILE:'  '{this[tmp_dir]}{this[marcs_model_name]}_coef.dat'
#'DEPARTBINARY:'   '.false.'
#'CONTMASKFILE:'     '/Users/gerber/gitprojects/SAPP/linemasks/ca-cmask.txt'
#'LINEMASKFILE:'     '/Users/gerber/gitprojects/SAPP/linemasks/ca-lmask.txt'
#'SEGMENTSFILE:'     '/Users/gerber/gitprojects/SAPP/linemasks/ca-seg.txt'
'LAMBDA_MIN:'    '{this[lambda_min]:.3f}'
'LAMBDA_MAX:'    '{this[lambda_max]:.3f}'
'LAMBDA_STEP:'   '{this[lambda_delta]:.3f}'
'INTENSITY/FLUX:' 'Flux'
'COS(THETA)    :' '1.00'
'ABFIND        :' '.false.'
'MODELOPAC:' '{this[tmp_dir]}model_opacity_{this[counter_spectra]:08d}.opac'
'RESULTFILE :' '{this[tmp_dir]}/spectrum_{this[counter_spectra]:08d}.spec'
'METALLICITY:'    '{this[metallicity]:.2f}'
'ALPHA/Fe   :'    '{alpha:.2f}'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '{this[r_process]:.2f}'
'S-PROCESS  :'    '{this[s_process]:.2f}'
{individual_abundances}
{individual_isotopes}
{line_lists}
'SPHERICAL:'  '{spherical}'
  30
  300.00
  15
  1.30
""".format(this=self.__dict__,
           alpha=alpha,
           spherical=spherical_boolean_code,
           individual_abundances=individual_abundances.strip(),
           individual_isotopes=individual_isotopes.strip(),
           line_lists=line_lists.strip(),
           pure_lte=pure_lte_boolean_code,
           nlte=nlte_boolean_code
           )

            # Build babsma configuration file
            babsma_config = """\
'PURE-LTE  :'  '.false.'
'LAMBDA_MIN:'    '{this[lambda_min]:.3f}'
'LAMBDA_MAX:'    '{this[lambda_max]:.3f}'
'LAMBDA_STEP:'    '{this[lambda_delta]:.3f}'
'MODELINPUT:' '{this[tmp_dir]}{this[marcs_model_name]}.interpol'
'MARCS-FILE:' '.false.'
'MODELOPAC:' '{this[tmp_dir]}model_opacity_{this[counter_spectra]:08d}.opac'
'METALLICITY:'    '{this[metallicity]:.2f}'
'ALPHA/Fe   :'    '{alpha:.2f}'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '{this[r_process]:.2f}'
'S-PROCESS  :'    '{this[s_process]:.2f}'
{individual_abundances}
'XIFIX:' '{xifix}'
{this[turbulent_velocity]:.2f}
""".format(this=self.__dict__,
           alpha=alpha,
           individual_abundances=individual_abundances.strip(),
           pure_lte=pure_lte_boolean_code,
           xifix=xifix_boolean_code
           )

        # print(babsma_config)
        # print(bsyn_config)
        return babsma_config, bsyn_config

    def stitch(self, specname1, specname2, lmin, lmax, new_range, count):  # toss a coin to your stitcher
        wave1, flux_norm1, flux1 = np.loadtxt(specname1, unpack=True)
        wave2, flux_norm2, flux2 = np.loadtxt(specname2, unpack=True)

        # print(lmin, lmin+(count*new_range))

        wave1_clipped = wave1[np.where((wave1 < lmin + (count * new_range)) & (wave1 >= lmin))]
        flux_norm1_clipped = flux_norm1[np.where((wave1 < lmin + (count * new_range)) & (wave1 >= lmin))]
        flux1_clipped = flux1[np.where((wave1 < lmin + (count * new_range)) & (wave1 >= lmin))]
        wave2_clipped = wave2[np.where((wave2 >= lmin + (count * new_range)) & (wave2 <= lmax))]
        flux_norm2_clipped = flux_norm2[np.where((wave2 >= lmin + (count * new_range)) & (wave2 <= lmax))]
        flux2_clipped = flux2[np.where((wave2 >= lmin + (count * new_range)) & (wave2 <= lmax))]

        wave = np.concatenate((wave1_clipped, wave2_clipped))
        flux_norm = np.concatenate((flux_norm1_clipped, flux_norm2_clipped))
        flux = np.concatenate((flux1_clipped, flux2_clipped))

        return wave, flux_norm, flux

    def synthesize(self):
        # Generate configuation files to pass to babsma and bsyn
        self.make_species_lte_nlte_file()  # TODO: not create this file every time (same one for each run anyway)
        babsma_in, bsyn_in = self.make_babsma_bsyn_file(spherical=self.atmosphere_properties['spherical'])

        logging.debug("babsma input:\n{}".format(babsma_in))
        logging.debug("bsyn input:\n{}".format(bsyn_in))

        # print(babsma_in)
        # print(bsyn_in)

        # Start making dictionary of output data
        output = self.atmosphere_properties
        output["errors"] = None
        output["babsma_config"] = babsma_in
        output["bsyn_config"] = bsyn_in

        # Select whether we want to see all the output that babsma and bsyn send to the terminal
        if self.verbose:
            stdout = None
            stderr = subprocess.STDOUT
        else:
            stdout = open('/dev/null', 'w')
            stderr = subprocess.STDOUT

        # We need to run babsma and bsyn with working directory set to root of Turbospectrum install. Otherwise
        # it cannot find its data files.
        cwd = os.getcwd()
        turbospec_root = os_path.join(self.code_path, "..")

        # Run babsma. This creates an opacity file .opac from the MARCS atmospheric model
        try:  # chdir is NECESSARY, turbospectrum cannot run from other directories sadly
            os.chdir(turbospec_root)
            pr1, stderr_bytes = self.run_babsma(babsma_in, stderr, stdout)
        except subprocess.CalledProcessError:
            output["errors"] = "babsma failed with CalledProcessError"
            return output
        finally:
            os.chdir(cwd)
        if stderr_bytes is None:
            stderr_bytes = b''
        if pr1.returncode != 0:
            output["errors"] = "babsma failed"
            # logging.info("Babsma failed. Return code {}. Error text <{}>".
            #             format(pr1.returncode, stderr_bytes.decode('utf-8')))
            return output

        # Run bsyn. This synthesizes the spectrum
        try:
            pr, stderr_bytes = self.run_bsyn(bsyn_in, stderr, stderr_bytes, stdout, turbospec_root)
        except subprocess.CalledProcessError:
            output["errors"] = "bsyn failed with CalledProcessError"
            return output
        finally:
            os.chdir(cwd)
        if stderr_bytes is None:
            stderr_bytes = b''
        if pr.returncode != 0:
            output["errors"] = "bsyn failed"
            # logging.info("Bsyn failed. Return code {}. Error text <{}>".
            #             format(pr.returncode, stderr_bytes.decode('utf-8')))
            return output

        # Return output
        output["return_code"] = pr.returncode
        output["output_file"] = os_path.join(self.tmp_dir, "spectrum_{:08d}.spec".format(self.counter_spectra))
        return output

    def run_bsyn(self, bsyn_in, stderr, stderr_bytes, stdout, turbospec_root):
        os.chdir(turbospec_root)
        pr = subprocess.Popen([os_path.join(self.code_path, 'bsyn_lu')],
                              stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
        pr.stdin.write(bytes(bsyn_in, 'utf-8'))
        stdout_bytes, stderr_bytes = pr.communicate()
        return pr, stderr_bytes

    def run_babsma(self, babsma_in, stderr, stdout):
        pr1 = subprocess.Popen([os_path.join(self.code_path, 'babsma_lu')],
                               stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
        pr1.stdin.write(bytes(babsma_in, 'utf-8'))
        stdout_bytes, stderr_bytes = pr1.communicate()
        return pr1, stderr_bytes

    def run_turbospectrum(self):
        lmin_orig = self.lambda_min
        lmax_orig = self.lambda_max
        lmin = self.lambda_min
        lmax = self.lambda_max

        lpoint_max = self.lpoint * 0.99  # first number comes from turbospectrum spectrum.inc : lpoint. 0.99 is to give some extra room so that bsyn does not fail for sure
        points_in_new_spectra_to_generate = int((lmax - lmin) / self.lambda_delta)

        if points_in_new_spectra_to_generate > lpoint_max:
            print("WARNING. The range or delta wavelength asked is too big. Trying to break down into smaller "
                  "segments and stitch them together at the end.")
            number = math.ceil(points_in_new_spectra_to_generate / lpoint_max)
            new_range = round((lmax - lmin) / number)
            extra_wavelength_for_stitch = 30  # generats with extra wavlength so that stitch can be nice i guess (i did not write this originally)

            for i in range(number):
                self.configure(lambda_min=lmin - extra_wavelength_for_stitch,
                               lambda_max=lmin + new_range + extra_wavelength_for_stitch, counter_spectra=i)
                self.synthesize()
                lmin = lmin + new_range
            for i in range(number - 1):
                spectrum1 = os_path.join(self.tmp_dir, "spectrum_{:08d}.spec".format(0))
                spectrum2 = os_path.join(self.tmp_dir, "spectrum_{:08d}.spec".format(i + 1))
                wave, flux_norm, flux = self.stitch(spectrum1, spectrum2, lmin_orig, lmax_orig, new_range, i + 1)
                f = open(spectrum1, 'w')
                for j in range(len(wave)):
                    print("{}  {}  {}".format(wave[j], flux_norm[j], flux[j]), file=f)
                f.close()
            '''
            if (lmax-lmin)/self.lambda_delta > self.lpoint:
                print("Whoops! You went over the default maximum number of spectrum points. TSFitPy will break up the wavelength range and stitch together the smaller pieces, but a better solution is to increase the number of points in Turbospectrum in the file spectrum.inc to match what you need. Then adjust the same lpoint parameter next time you call TSFitPy.")
                lmax = (self.lpoint*self.lambda_delta) + lmin
                k = 0
                while lmax < lmax_orig:
                    self.configure(lambda_min = lmin-30., lambda_max=lmax+30, counter_spectra=k)
                    self.synthesize()
                    lmin = lmax
                    lmax = (self.lpoint*self.lambda_delta) + lmin
                    k+=1
                lmax = lmag_orig
                self.configure(lambda_min = lmin-30., lambda_max=lmax+30, counter_spectra=k)
                self.synthesize()
                for i in range(k-1):
                    spectrum1 = os_path.join(self.tmp_dir, "spectrum_{:08d}.spec".format(0))
                    spectrum2 = os_path.join(self.tmp_dir, "spectrum_{:08d}.spec".format(i+1))
                    wave, flux_norm, flux = self.stitch(spectrum1, spectrum2, lmin_orig, lmax_orig, new_range, i+1)
                    f = open(spectrum1, 'w')
                    for j in range(len(wave)):
                        print("{}  {}  {}".format(wave[j], flux_norm[j], flux[j]), file=f)
                    f.close()'''
        else:
            self.synthesize()

    def synthesize_spectra(self):
        try:
            logging.debug("Running Turbospectrum and atmosphere")
            self.calculate_atmosphere()
            try:
                logging.debug("Running Turbospectrum")
                self.run_turbospectrum()
            except AttributeError:
                print("No attribute, fail of generation?")
        except (FileNotFoundError, ValueError, TypeError) as error:
            print(f"Interpolation failed? {error}")
            print("ValueError can sometimes imply problem with the departure coefficients grid")


