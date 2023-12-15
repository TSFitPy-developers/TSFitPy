from __future__ import annotations

import logging
import subprocess

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import time
import tempfile
import importlib.util
import sys

from scripts.solar_abundances import periodic_table, solar_abundances
from scripts.solar_isotopes import solar_isotopes
from scripts.synthetic_code_class import SyntheticSpectrumGenerator




class m3disCall(SyntheticSpectrumGenerator):
    def __init__(self, m3dis_path: str, interpol_path: str, line_list_paths: str, marcs_grid_path: str,
                 marcs_grid_list: str, model_atom_path: str, departure_file_path: str,
                 aux_file_length_dict: dict,
                 marcs_value_keys: list, marcs_values: dict, marcs_models: dict, model_temperatures: np.ndarray,
                 model_logs: np.ndarray, model_mets: np.ndarray, m3dis_python_module, n_nu=None,
                hash_table_size=None,
                mpi_cores=None,
                iterations_max=None,
                convlim=None,
                snap=None,
                dims=None,
                nx=None,
                ny=None,
                nz=None):
        """
        Instantiate a class for generating synthetic stellar spectra using Turbospectrum.

        :param turbospec_path: Path where the turbospectrum binaries 'babsma' and 'bsyn' can be found.
        :param interpol_path: Path where the compiled interpol_modeles.f binary can be found.
        :param line_list_paths: Path(s) where line lists for synthetic spectra can be found. Specify as either a string, or a list of strings.
        :param marcs_grid_path: Path where a grid of MARCS .mod files can be found. These contain the model atmospheres we use.
        :param model_atom_path: Path to the model atom paths
        :param departure_file_path: Path to the NLTE departure file paths
        """
        super().__init__(m3dis_path, interpol_path, line_list_paths, marcs_grid_path,
                 marcs_grid_list, model_atom_path,
                 marcs_value_keys, marcs_values, marcs_models, model_temperatures,
                 model_logs, model_mets)
        self.m3dis_path = self.code_path
        self.mpi_cores: int = mpi_cores
        self.departure_file_path = departure_file_path
        self.aux_file_length_dict = aux_file_length_dict
        self.m3dis_python_module = m3dis_python_module
        self.n_nu = n_nu
        self.hash_table_size = hash_table_size
        self.iterations_max = iterations_max
        self.convlim = convlim
        self.snap = snap
        self.dims = dims
        self.nx = nx
        self.ny = ny
        self.nz = nz

    def configure(self, lambda_min: float=None, lambda_max:float=None, lambda_delta: float=None,
                  metallicity: float=None, log_g: float=None, t_eff: float=None, stellar_mass: float=None,
                  turbulent_velocity: float=None, free_abundances=None, free_isotopes=None,
                  sphere=None, alpha=None, s_process=None, r_process=None,
                  line_list_paths=None, line_list_files=None,
                  verbose=None, temp_directory=None, nlte_flag: bool = None, atmosphere_dimension=None,
                  mpi_cores:int=None,
                  windows_flag=None, segment_file=None, line_mask_file=None, depart_bin_file=None,
                  depart_aux_file=None, model_atom_file=None):
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
        if temp_directory is not None:
            self.tmp_dir = temp_directory
        if nlte_flag is not None:
            self.nlte_flag = nlte_flag
        if atmosphere_dimension is not None:
            self.atmosphere_dimension = atmosphere_dimension
        if self.atmosphere_dimension == "3D":
            self.turbulent_velocity = None
        if mpi_cores is not None:
            self.mpi_cores = mpi_cores
        if model_atom_file is not None:
            self.model_atom_file = model_atom_file
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

    def run_m3dis(self, input_in, stderr, stdout):
        # Write the input data to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(bytes(input_in, "utf-8"))
            temp_file_name = temp.name

        #print(input_in)

        # check if file exists
        #if not os.path.exists("./dispatch.x"):
        #    print("File does not exist")
        #else:
        #    print("File exists")

        # Create a copy of the current environment variables
        env = os.environ.copy()
        # Set OMP_NUM_THREADS for the subprocess

        env['OMP_NUM_THREADS'] = str(self.mpi_cores)

        # Now, you can use temp_file_name as an argument to dispatch.x
        pr1 = subprocess.Popen(
            [
                "./dispatch.x",
                temp_file_name,
            ],
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=stderr,
            env=env,
        )
        # pr1.stdin.write(bytes(input_in, "utf-8"))
        stdout_bytes, stderr_bytes = pr1.communicate()

        # Don't forget to remove the temporary file at some point
        os.unlink(temp_file_name)
        return pr1, stderr_bytes

    def write_abund_file(self):
        # file path is temp_dir + abund
        file_path = os.path.join(self.tmp_dir, "abund")
        # open file
        with open(file_path, "w") as file:
            # write the number of elements
            # write the elements and their abundances
            for element in periodic_table:
                if element != "":
                    if element == "H" or element == "He":
                        abundance_to_write = solar_abundances[element]
                    else:
                        if element in self.free_abundances:
                            abundance_to_write = self.free_abundances[element] + solar_abundances[element] + self.metallicity
                        else:
                            abundance_to_write = solar_abundances[element] + self.metallicity
                    file.write(f"{element:<4} {abundance_to_write:>6.3f}\n")
        return file_path

    def write_isotope_file(self):
        if self.free_isotopes is None:
            self.free_isotopes = solar_isotopes
        elements_atomic_mass_number = self.free_isotopes.keys()

        # elements now consists of e.g. '3.006'. we want to convert 3
        elements_atomic_number = [int(float(element.split(".")[0])) for element in elements_atomic_mass_number]
        # count the number of each element, such that we have e.g. 3: 2, 4: 1, 5: 1
        elements_count = {element: elements_atomic_number.count(element) for element in elements_atomic_number}
        # remove duplicates
        elements_atomic_number_unique = set(elements_atomic_number)
        separator = "_"  # separator between sections in the file from NIST

        atomic_weights = {}
        with open("scripts/atomicweights.dat", "r") as file:
            skip_section = True
            current_element_atomic_number = 0
            for line in file:
                if line[0] != separator and skip_section:
                    continue
                elif line[0] == separator and skip_section:
                    skip_section = False
                    continue
                elif line[0] != separator and not skip_section and current_element_atomic_number == 0:
                    current_element_atomic_number_to_test = int(line.split()[0])
                    if current_element_atomic_number_to_test not in elements_atomic_number_unique:
                        skip_section = True
                        continue
                    current_element_atomic_number = current_element_atomic_number_to_test
                    atomic_weights[current_element_atomic_number] = {}
                    # remove any spaces and anything after (
                    atomic_weights[current_element_atomic_number][int(line[8:12].replace(" ", ""))] = \
                    line[13:32].replace(" ", "").split("(")[0]
                elif line[0] != separator and not skip_section and current_element_atomic_number != 0:
                    atomic_weights[current_element_atomic_number][int(line[8:12].replace(" ", ""))] = atomic_weight = \
                    line[13:32].replace(" ", "").split("(")[0]
                elif line[0] == separator and not skip_section and current_element_atomic_number != 0:
                    current_element_atomic_number = 0

        """
        format:
        Li    2
           6   6.0151   0.0759
           7   7.0160   0.9241
        """

        # open file
        file_path = os.path.join(self.tmp_dir, "isotopes")
        with open(file_path, "w") as file:
            # write element, then number of isotopes. next lines are isotope mass and abundance
            current_element_atomic_number = 0
            for element, isotope in self.free_isotopes.items():
                element_atomic_number = int(float(element.split(".")[0]))
                element_mass_number = int(float(element.split(".")[1]))
                if current_element_atomic_number != element_atomic_number:
                    # elements now consists of e.g. '3.006'. we want to convert 3 to and 6
                    current_element_atomic_number = element_atomic_number
                    file.write(
                        f"{periodic_table[element_atomic_number]:<5}{elements_count[element_atomic_number]:>2}\n")

                file.write(
                    f"{int(element_mass_number):>4} {float(atomic_weights[element_atomic_number][element_mass_number]):>8.4f} {isotope:>8.4f}\n")
        return file_path

    def convert_interpolated_atmo_to_m3dis(self, atmos_path):
        log_taur, teff, log_pe, log_pg, vmic, subt_depth, log_tau5 = np.loadtxt(atmos_path, unpack=True, skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6), comments="/")
        # convert subt_depth to depth. depends whether spherical or plane-parallel
        spherical_model: bool = self.atmosphere_properties["spherical"]
        if not spherical_model:
            depth = 1 - subt_depth
        else:
            raise ValueError("Spherical models not implemented yet")
        # interpolate to the equidistant depth grid. first get the new depth grid based on minimum and maximum depth
        depth_min = np.min(depth)
        depth_max = np.max(depth)
        depth_points = np.size(depth)
        depth_new = np.linspace(depth_min, depth_max, depth_points)
        # interpolate all the other parameters to the new depth grid
        teff_new = np.interp(depth_new, depth, teff)
        log_pe_new = np.interp(depth_new, depth, log_pe)
        log_pg_new = np.interp(depth_new, depth, log_pg)
        # vmic not needed because it is constant
        #vmic_new = np.interp(depth_new, depth, vmic)

        # write the file, format: depth temp pe pg vmic, so need to convert log_pe and log_pg
        file_path = os.path.join(self.tmp_dir, f"atmos.{self.marcs_model_name}")
        with open(file_path, "w") as file:
            # first is name of file
            file.write(f"{self.marcs_model_name}\n")
            # next is number of points as integer
            file.write(f"{depth_points}\n")
            # next is the format
            file.write("* depth      temp       pe        pg      vmic\n")
            for i in range(len(depth_new)):
                file.write(f"{depth_new[i]:>13.6e} {teff_new[i]:>8.1f} {np.power(10, log_pe_new[i]):>12.4E} {np.power(10, log_pg_new[i]):>12.4E} {vmic[i]:>3.1f}\n")
        return file_path


    def call_m3dis(self):
        abund_file_path = self.write_abund_file()
        isotope_file_path = self.write_isotope_file()



        # get all files from self.line_list_paths[0]
        self.line_list_files = os.listdir(self.line_list_paths[0])

        if self.atmosphere_dimension == "1D":
            atmo_param = f"atmos_format='Marcs' vmic={round(self.turbulent_velocity, 5)}"
            self.dims = 1
            #atmos_path = "./input_multi3d/atmos/p5777_g+4.4_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod"

            atmo_param = f"atmos_format='Text' vmic={round(self.turbulent_velocity, 5)}"
            atmos_path = f"{os.path.join(self.tmp_dir, self.marcs_model_name)}.interpol"
            # convert to m3dis format
            atmos_path = self.convert_interpolated_atmo_to_m3dis(atmos_path)
        elif self.atmosphere_dimension == "3D":
            raise ValueError("3D atmospheres not implemented yet")
            atmo_param = "atmos_format='MUST'"
        else:
            raise ValueError("Atmosphere dimension must be either 1D or 3D: m3dis_class.py")

        if self.nlte_flag:
            atom_path = self.model_atom_path
            atom_files = list(self.model_atom_file.keys())
            atom_file_element = atom_files[0]
            if len(atom_files) > 1:
                print(f"Only one atom file is allowed for NLTE: m3dis, using the first one {atom_file_element}")
            atom_params = f"&atom_params        atom_file='{os.path.join(atom_path, self.model_atom_file[atom_file_element])}' convlim={self.convlim} use_atom_abnd=F exclude_trace_cont=F exclude_from_line_list=T /\n"
            # linelist_param_extra
            linelist_param_extra = f"exclude_elements='{atom_file_element}'"
        else:
            atom_params = ""
            linelist_param_extra = ""

        output = {}
        config_m3dis = (f"! -- Parameters defining the run -----------------------------------------------\n\
&io_params          datadir='{self.tmp_dir}' gb_step=100.0 do_trace=F /\n\
&timer_params       sec_per_report=1e8 /\n\
&atmos_params       dims={self.dims} save_atmos=F atmos_file='{atmos_path}' {atmo_param}/\n{atom_params}\
&m3d_params         verbose=0 n_nu={self.n_nu} maxiter={self.iterations_max} quad_scheme='set_a2' long_scheme='lobatto'/\n\
&linelist_params    linelist_file='{os.path.join(self.line_list_paths[0], self.line_list_files[0])}' {linelist_param_extra}/\n\
&spectrum_params    daa={self.lambda_delta} aa_blue={self.lambda_min} aa_red={self.lambda_max} /\n\
&composition_params isotope_file='{isotope_file_path}' abund_file='{abund_file_path}'/\n\
&task_list_params   hash_table_size={self.hash_table_size} /\n")
        #print(config_m3dis)

        if self.verbose:
            stdout = None
            stderr = subprocess.STDOUT
        else:
            stdout = open("/dev/null", "w")
            stderr = subprocess.STDOUT

        cwd = os.getcwd()

        try:  # chdir is NECESSARY, turbospectrum cannot run from other directories sadly
            os.chdir(os.path.join(self.m3dis_path, ""))  #
            #print(os.getcwd())
            pr1, stderr_bytes = self.run_m3dis(config_m3dis, stderr, stdout)
        except subprocess.CalledProcessError:
            output["errors"] = "babsma failed with CalledProcessError"
            return output
        finally:
            os.chdir(cwd)
        if stderr_bytes is None:
            stderr_bytes = b""
        if pr1.returncode != 0:
            output["errors"] = f"m3dis failed with return code {pr1.returncode} {stderr_bytes.decode('utf-8')}"
            return output

        # Return output
        # output["return_code"] = pr.returncode
        # output["output_file"] = os_path.join(
        #    self.tmp_dir, "spectrum_{:08d}.spec".format(self.counter_spectra)
        # )
        return output

    def synthesize_spectra(self):
        try:
            logging.debug("Running m3dis and atmosphere")
            self.calculate_atmosphere()
            logging.debug("Running m3dis")
            output = self.call_m3dis()
            if "errors" in output:
                print(output["errors"], "m3dis failed")
            else:
                try:
                    completed_run = self.m3dis_python_module.read(
                        self.tmp_dir
                    )
                    wavelength, _ = completed_run.get_xx(completed_run.lam)
                    flux, continuum = completed_run.get_yy(norm=False)
                    normalised_flux = flux / continuum
                    # save to file as append
                    file_to_save = os.path.join(self.tmp_dir, "spectrum_00000000.spec")
                    # save to file using numpy, wavelength, normalised_flux, flux
                    np.savetxt(file_to_save, np.transpose([wavelength, normalised_flux, flux]), fmt='%10.5f %10.5f %10.5f')
                except FileNotFoundError as e:
                    print(f"m3dis, cannot find  {e}")
        except (FileNotFoundError, ValueError, TypeError) as error:
            print(f"Interpolation failed? {error}")




if __name__ == "__main__":


    test_class = m3disCall(
        m3dis_path="/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/",
        interpol_path="/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/",
        line_list_paths="/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/data2/input_multi3d/",
        marcs_grid_path="/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/data2/input_multi3d/atmos/",
        marcs_grid_list="/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/data2/input_multi3d/atmos/marcs_grid_list.txt",
        model_atom_path="/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/data2/input_multi3d/atoms/",
        marcs_value_keys=[
            "spherical",
            "temperature",
            "log_g",
            "mass",
            "turbulence",
            "model_type",
            "metallicity",
        ],
        marcs_values={
            "spherical": [],
            "temperature": [],
            "log_g": [],
            "mass": [],
            "turbulence": [],
            "model_type": [],
            "metallicity": [],
            "a": [],
            "c": [],
            "n": [],
            "o": [],
            "r": [],
            "s": [],
        },
        marcs_models={},
        model_temperatures=np.array([]),
        model_logs=np.array([]),
        model_mets=np.array([]),
    )
    # create temp directory
    global_temp_dir = "/Users/storm/docker_common_folder/TSFitPy/temp_directory/"
    temp_directory = tempfile.mkdtemp(dir=global_temp_dir)
    #print(temp_directory)
    test_class.configure(temp_directory=temp_directory, lambda_min=6707, lambda_max=6808, lambda_delta=0.01,
                         line_list_files=["/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/input_multi3d/nlte_ges_linelist_jmg17feb2022_I_II_li"],
                         turbulent_velocity=1.0, free_abundances={}, verbose=True)
    time_start = time.perf_counter()
    test_class.synthesize_spectra()
    time_end = time.perf_counter()
    print("Time taken to read: ", time_end - time_start)




    # Example usage
    # Assuming the package is in a folder named 'somecode' and the main module is 'package.py'
    module_path = "/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/m3dis/__init__.py"  # Replace with the actual path
    m3dis = import_module_from_path("m3dis", module_path)

    # Now you can use pkg as if you imported it normally
    # For example:
    # result = pkg.some_function()

    run = m3dis.read(
        #"/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/data2/input_test/"
        temp_directory
    )
    run.line[0].plot()
    # run.plot_spectrum()
    #plt.show()
