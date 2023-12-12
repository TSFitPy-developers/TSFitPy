import subprocess

#from m3dis_l.m3dis.experiments.Multi3D import m3dis
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import time
import tempfile

from scripts.solar_abundances import periodic_table, solar_abundances
from scripts.solar_isotopes import solar_isotopes
from scripts.synthetic_code_class import SyntheticSpectrumGenerator




class m3disCall(SyntheticSpectrumGenerator):
    def __init__(self, m3dis_path: str, interpol_path: str, line_list_paths: str, marcs_grid_path: str,
                 marcs_grid_list: str, model_atom_path: str,
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
        super().__init__(m3dis_path, interpol_path, line_list_paths, marcs_grid_path,
                 marcs_grid_list, model_atom_path,
                 marcs_value_keys, marcs_values, marcs_models, model_temperatures,
                 model_logs, model_mets)
        self.m3dis_path = self.code_path

    def configure(self, lambda_min: float=None, lambda_max:float=None, lambda_delta: float=None,
                  metallicity: float=None, log_g: float=None, t_eff: float=None, stellar_mass: float=None,
                  turbulent_velocity: float=None, free_abundances=None, free_isotopes=None,
                  sphere=None, alpha=None, s_process=None, r_process=None,
                  line_list_paths=None, line_list_files=None,
                  verbose=None, temp_directory=None, nlte_flag: bool = None, atmosphere_dimension=None,
                  model_atom_file=None):
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

    def run_m3dis(self, input_in, stderr, stdout):
        # Write the input data to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(bytes(input_in, "utf-8"))
            temp_file_name = temp.name

        print(input_in)

        # check if file exists
        if not os.path.exists("./dispatch.x"):
            print("File does not exist")
        else:
            print("File exists")

        # Now, you can use temp_file_name as an argument to dispatch.x
        pr1 = subprocess.Popen(
            [
                "./dispatch.x",
                temp_file_name,
            ],
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=stderr,
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
                    if element in self.free_abundances:
                        file.write(f"{element:<4} {self.free_abundances[element]:>6.3f}\n")
                    else:
                        file.write(f"{element:<4} {solar_abundances[element]:>6.3f}\n")
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
        elements_atomic_number_unique = list(set(elements_atomic_number))

        # parse the atomic weights file
        elements_atomic_number_unique = set(
            elements_atomic_number_unique)  # Convert list to set for faster membership testing
        separator = "_"

        atomic_weights = {}
        with open("atomicweights.dat", "r") as file:
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



    def call_m3dis(self):
        abund_file_path = self.write_abund_file()
        isotope_file_path = self.write_isotope_file()

        output = {}
        config_m3dis = (f"! -- Parameters defining the run -----------------------------------------------\n\
&io_params          datadir='{self.tmp_dir}' gb_step=100.0 do_trace=F /\n\
&timer_params       sec_per_report=1e8 /\n\
&atmos_params       dims=1 atmos_format='Marcs' vmic={self.turbulent_velocity} atmos_file='./input_multi3d/atmos/p5777_g+4.4_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod'/\n\
&atom_params        atom_file='./input_multi3d/atoms/atom.ba06' convlim=1d-2 use_atom_abnd=T /\n\
&m3d_params         verbose=0 n_nu=1 maxiter=0 /\n\
&linelist_params    linelist_file='{self.line_list_files[0]}' /\n\
&spectrum_params    daa={self.lambda_delta} aa_blue={self.lambda_min} aa_red={self.lambda_max} /\n\
&composition_params isotope_file='{isotope_file_path}' abund_file='{abund_file_path}'/\n\
&task_list_params   hash_table_size=10 /\n")

        # Select whether we want to see all the output that babsma and bsyn send to the terminal
        if self.verbose:
            stdout = None
            stderr = subprocess.STDOUT
        else:
            stdout = open("/dev/null", "w")
            stderr = subprocess.STDOUT

        cwd = os.getcwd()

        try:  # chdir is NECESSARY, turbospectrum cannot run from other directories sadly
            os.chdir(os.path.join(self.m3dis_path, "experiments/Multi3D/", ""))  #
            print(os.getcwd())
            pr1, stderr_bytes = self.run_m3dis(config_m3dis, stderr, stdout)
        except subprocess.CalledProcessError:
            output["errors"] = "babsma failed with CalledProcessError"
            return output
        finally:
            os.chdir(cwd)
        if stderr_bytes is None:
            stderr_bytes = b""
        if pr1.returncode != 0:
            output["errors"] = "m3dis failed"
            # logging.info("Babsma failed. Return code {}. Error text <{}>".
            #             format(pr1.returncode, stderr_bytes.decode('utf-8')))
            return output

        # Return output
        # output["return_code"] = pr.returncode
        # output["output_file"] = os_path.join(
        #    self.tmp_dir, "spectrum_{:08d}.spec".format(self.counter_spectra)
        # )
        return output

    def synthesize_spectra(self):
        output = self.call_m3dis()
        if "errors" in output:
            print(output["errors"], "m3dis failed")


if __name__ == "__main__":
    time_start = time.perf_counter()

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
    print(temp_directory)
    test_class.configure(temp_directory=temp_directory, lambda_min=6707, lambda_max=6808, lambda_delta=0.01,
                         line_list_files=["/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/input_multi3d/nlte_ges_linelist_jmg17feb2022_I_II_li"],
                         turbulent_velocity=1.0, free_abundances={}, verbose=True)
    test_class.synthesize_spectra()
    time_end = time.perf_counter()
    print("Time taken to read: ", time_end - time_start)

    import importlib.util
    import sys


    def import_module_from_path(module_name, file_path):
        """
        Dynamically imports a module or package from a given file path.

        Parameters:
        module_name (str): The name to assign to the module.
        file_path (str): The file path to the module or package.

        Returns:
        module: The imported module.
        """
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            raise ImportError(f"Module spec not found for {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


    # Example usage
    # Assuming the package is in a folder named 'somecode' and the main module is 'package.py'
    module_path = "/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/m3dis/__init__.py"  # Replace with the actual path
    m3dis = import_module_from_path("m3dis", module_path)

    # Now you can use pkg as if you imported it normally
    # For example:
    # result = pkg.some_function()

    run = m3dis.read(
        "/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/data2/input_test/"
    )
    run.plot_spectrum()



    # run.plot_spectrum()
    #plt.show()
