from __future__ import annotations

import logging
import shutil
import subprocess
import numpy as np
import os
import tempfile
from scipy.interpolate import LinearNDInterpolator, interp1d
from scripts import marcs_class
from scripts.solar_abundances import periodic_table, solar_abundances
from scripts.solar_isotopes import solar_isotopes
from scripts.synthetic_code_class import SyntheticSpectrumGenerator


class M3disCall(SyntheticSpectrumGenerator):
    def __init__(self, m3dis_path: str, interpol_path: str, line_list_paths: str, marcs_grid_path: str,
                 marcs_grid_list: str, model_atom_path: str, departure_file_path: str,
                 aux_file_length_dict: dict,
                 marcs_value_keys: list, marcs_values: dict, marcs_models: dict, model_temperatures: np.ndarray,
                 model_logs: np.ndarray, model_mets: np.ndarray, m3dis_python_module, night_mode: bool=False, n_nu=None,
                hash_table_size=None, mpi_cores=None, iterations_max=None, convlim=None, snap=None,
                dims=None, nx=None, ny=None, nz=None):
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
                 model_logs, model_mets, night_mode)
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
        self.skip_linelist = False
        self.save_spectra = True
        self.use_precomputed_depart = True
        self.atmosphere_path_3d_model = None
        self.atmos_format_3d = None

    def configure(self, lambda_min: float=None, lambda_max:float=None, lambda_delta: float=None,
                  metallicity: float=None, log_g: float=None, t_eff: float=None, stellar_mass: float=None,
                  turbulent_velocity: float=None, free_abundances=None, free_isotopes=None,
                  sphere=None, alpha=None, s_process=None, r_process=None,
                  line_list_paths=None, line_list_files=None,
                  verbose=None, temp_directory=None, nlte_flag: bool = None, atmosphere_dimension=None,
                  mpi_cores:int=None,
                  windows_flag=None, segment_file=None, line_mask_file=None, depart_bin_file=None,
                  depart_aux_file=None, model_atom_file=None, atmosphere_path_3d_model=None, atmos_format_3d=None):
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
        if atmosphere_path_3d_model is not None:
            self.atmosphere_path_3d_model = atmosphere_path_3d_model
        if atmos_format_3d is not None:
            self.atmos_format_3d = atmos_format_3d

    def run_m3dis(self, input_in, stderr, stdout):
        # Write the input data to a temporary file
        # TODO: check this solution because temp direction might mess up something
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
                            # Here abundance is passed as [X/H], so we need to add the solar abundance to convert to A(X)
                            # A(X)_star = A(X)_solar + [X/H]
                            abundance_to_write = self.free_abundances[element] + solar_abundances[element]
                        else:
                            # If the element is not in the free abundances, we assume it has the solar scaled abundance
                            # A(X)_star = A(X)_solar + [Fe/H]
                            abundance_to_write = solar_abundances[element] + self.metallicity
                        if self.atmosphere_dimension == "3D":
                            # if 3D, we need to subtract the metallicity from the abundance, because it auto scales (adds it) in M3D with FeH already
                            abundance_to_write = abundance_to_write - self.metallicity
                    file.write(f"{element:<4} {abundance_to_write:>6.3f}\n")
                    logging.debug(f"{element:<4} {abundance_to_write:>6.3f}")
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

        atomic_weights_path = "scripts/atomicweights.dat"
        # check if file exists
        if not os.path.exists(atomic_weights_path):
            # add ../ to the path
            atomic_weights_path = os.path.join("../", atomic_weights_path)

        atomic_weights = {}
        with open(atomic_weights_path, "r") as file:
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


    def call_m3dis(self, skip_linelist=False, use_precomputed_depart=False):
        abund_file_path = self.write_abund_file()
        isotope_file_path = self.write_isotope_file()

        # get all files from self.line_list_paths[0]
        self.line_list_files = os.listdir(self.line_list_paths[0])

        if self.atmosphere_dimension == "1D":
            atmo_param = f"atmos_format='Marcs' vmic={round(self.turbulent_velocity, 5)}"
            self.dims = 1
            #atmos_path = "./input_multi3d/atmos/p5777_g+4.4_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod"

            atmo_param = f"atmos_format='Text' vmic={round(self.turbulent_velocity, 5)}"
            atmos_path = f"{os.path.join(self.tmp_dir, self.marcs_model_name)}"
            # convert to m3dis format
            #atmos_path = self.convert_interpolated_atmo_to_m3dis(atmos_path)
        elif self.atmosphere_dimension == "3D":
            # check if teff is about 4600, logg is about 1.39 and feh is about -2.55, within tolerance of 0.01
            # if so, use the 3D model
            if np.isclose(self.t_eff, 4600, atol=0.1) and np.isclose(self.log_g, 1.39, atol=0.02) and np.isclose(self.metallicity, -2.55, atol=0.2):
                atmo_param = f"atmos_format='Stagger' snap={self.snap} FeH={self.metallicity} dims={self.dims} nx={self.nx} ny={self.ny} nz={self.nz}"
                atmos_path = "/mnt/beegfs/gemini/groups/bergemann/users/shared-storage/bergemann-data/Stagger_remo/hd1225623/2013-04-10_nlam48/t46g16m2503"
            else:
                if self.atmosphere_path_3d_model is not None:
                    atmos_path = self.atmosphere_path_3d_model
                    if self.atmos_format_3d.lower == "multi" or self.atmos_format_3d.lower == "muram":
                        atmo_param = f"atmos_format='Multi' dims={self.dims}"
                    elif self.atmos_format_3d.lower == "stagger":
                        atmo_param = f"atmos_format='Stagger' snap={self.snap} dims={self.dims} nx={self.nx} ny={self.ny} nz={self.nz}"
                    elif self.atmos_format_3d.lower == "must":
                        atmo_param = f"atmos_format='MUST' dims={self.dims}"
                    else:
                        raise ValueError(f"Atmosphere format {self.atmos_format_3d} not recognized")
                else:
                    raise ValueError("3D atmospheres not implemented yet")
                    atmo_param = "atmos_format='MUST'"
                    atmo_param = "&atmos_params       dims=10 atmos_format='Multi' atmos_file='/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/input_multi3d/atmos/t5777g44m0005_20.5x5x230'/"
                    atmo_param = f"atmos_format='Multi' dims={self.dims}"
                    atmos_path = "/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/input_multi3d/atmos/t5777g44m0005_20.5x5x230"
                # &atmos_params       dims=1 atmos_format='MUST' atmos_file='input_multi3d/atmos/m3dis_sun_magg22_10x10x280_1' /

                # multi:
                #atmos_format='MUST' dims=23 atmos_file='/shared-storage/bergemann/m3dis/experiments/Multi3D/input_multi3d/atmos/299/magg2022_150x300/m3dis_sun_magg22_80x80x299_1'
                # might need these two as well? use_density=T use_ne=F

                # stagger:
                # atmos_format="Stagger" snap=20 dims=23 FeH=0.0 nx=30 ny=30 nz=230 atmos_file='./input_multi3d/atmos/t5777g44m00/v05/t5777g44m0005'/

                # muram:
                # atmos_format='Multi' dims=23 atmos_file='./input_multi3d/atmos/muram/mDIS_MARCS_v0.5.1_box_MURaM_HDSun'  /



        else:
            raise ValueError("Atmosphere dimension must be either 1D or 3D: m3dis_class.py")

        if self.nlte_flag:
            atom_path = self.model_atom_path
            atom_files = list(self.model_atom_file.keys())
            atom_file_element = atom_files[0]
            if len(atom_files) > 1:
                print(f"Only one atom file is allowed for NLTE: m3dis, using the first one {atom_file_element}")
            if use_precomputed_depart:
                precomputed_depart = f"precomputed_depart='{os.path.join(self.tmp_dir, '../precomputed_depart', '')}'"
            else:
                precomputed_depart = ""
            atom_params = (f"&atom_params        atom_file='{os.path.join(atom_path, self.model_atom_file[atom_file_element])}' "
                           f"convlim={self.convlim} use_atom_abnd=F exclude_trace_cont=F exclude_from_line_list=T "
                           f"{precomputed_depart}/\n")
            # linelist_param_extra
            linelist_param_extra = f"exclude_elements='{atom_file_element}'"
        else:
            atom_params = ""
            linelist_param_extra = ""

        if skip_linelist:
            linelist_parameters = ""
        else:
            linelist_parameters = (f"&linelist_params    linelist_file='{os.path.join(self.line_list_paths[0], self.line_list_files[0])}' {linelist_param_extra}/\n\
                                     &spectrum_params    daa={self.lambda_delta} aa_blue={self.lambda_min} aa_red={self.lambda_max} /\n")

        output = {}
        config_m3dis = (f"! -- Parameters defining the run -----------------------------------------------\n\
&io_params          datadir='{self.tmp_dir}' gb_step=100.0 do_trace=F /\n\
&timer_params       sec_per_report=1e8 /\n\
&atmos_params       dims={self.dims} save_atmos=F atmos_file='{atmos_path}' {atmo_param}/\n{atom_params}\
&m3d_params         verbose=2 n_nu={self.n_nu} maxiter={self.iterations_max} quad_scheme='set_a2' long_scheme='lobatto'/\n\
{linelist_parameters}\
&composition_params isotope_file='{isotope_file_path}' abund_file='{abund_file_path}'/\n\
&task_list_params   hash_table_size={self.hash_table_size} /\n")
        #TODO absmet files?
        logging.debug(config_m3dis)

        # clean temp directory
        save_file_dir = os.path.join(self.tmp_dir, "save")
        if os.path.exists(save_file_dir):
            # just in case it fails, so that it doesn't reuse the old files
            shutil.rmtree(save_file_dir)

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

    def interpolate_m3dis_atmosphere(self, marcs_models_to_load):
        modelAtmGrid = {'teff': [], 'logg': [], 'feh': [], 'vturb': [], 'file': [], 'structure': [],
                        'structure_keys': [], 'mass': []}  # data

        all_marcs_models = []
        marcs_path = self.marcs_grid_path
        for marcs_model in marcs_models_to_load:
            one_model = marcs_class.MARCSModel(os.path.join(marcs_path, marcs_model))
            all_marcs_models.append(one_model)
            modelAtmGrid['teff'].append(one_model.teff)
            modelAtmGrid['logg'].append(one_model.logg)
            modelAtmGrid['feh'].append(one_model.metallicity)
            modelAtmGrid['vturb'].append(one_model.vmicro)
            modelAtmGrid['file'].append(one_model.file)
            modelAtmGrid['structure'].append(np.vstack((one_model.lgTau5, one_model.temperature, one_model.pe,
                                                        np.full(one_model.depth.shape, one_model.vmicro),
                                                        one_model.density, one_model.depth)))
            modelAtmGrid['structure_keys'].append(['tau500', 'temp', 'pe', 'vmic', 'density', 'depth'])
            modelAtmGrid['mass'].append(one_model.mass)

        interpolate_variables = ['teff', 'logg', 'feh']  # , 'vmic'

        # convert all to numpy arrays
        for k in modelAtmGrid:
            modelAtmGrid[k] = np.asarray(modelAtmGrid[k])

        points = []
        norm_coord = {}
        for k in interpolate_variables:
            points.append(modelAtmGrid[k])  # / max(modelAtmGrid[k]) )
            norm_coord.update({k: max(modelAtmGrid[k])})
        points = np.array(points).T
        values = np.array(modelAtmGrid['structure'])
        interpolate_point = [self.t_eff, self.log_g, self.metallicity]

        points, unique_indices = np.unique(points, axis=0, return_index=True)
        values = values[unique_indices]

        indices_to_delete = []

        for i in range(len(interpolate_variables)):
            # get the column
            column = points[:, i]
            # check if all elements are the same
            if np.all(column == column[0]):
                indices_to_delete.append(i)
                #interpolate_point_new.pop(i)
                #interpolate_variables_new.pop(i)
                ## also remove ith column from points
                #points_new = np.delete(points_new, i, axis=1)

        # remove the indices
        points = np.delete(points, indices_to_delete, axis=1)
        interpolate_point = np.delete(interpolate_point, indices_to_delete)
        interpolate_variables = np.delete(interpolate_variables, indices_to_delete)

        if len(interpolate_point) > 1:
            interp_f = LinearNDInterpolator(points, values)
            tau500_new, temp_new, pe_new, vmic_new, density_new, depth_new = interp_f(interpolate_point)[0]
        elif len(interpolate_point) == 1:
            # linear interpolation
            # flatten points
            points = points.flatten()

            # take the first element of the array
            interpolate_point = interpolate_point[0]

            interp_f = interp1d(points, values, axis=0, kind='linear')
            tau500_new, temp_new, pe_new, vmic_new, density_new, depth_new = interp_f(interpolate_point)
        else:
            # only one model, so return that
            tau500_new, temp_new, pe_new, vmic_new, density_new, depth_new = values[0]

        # check if nan
        if np.any(np.isnan(tau500_new)):
            print("NAN in model atmosphere")

        # interpolate all variables to equidistant depth grid
        depth_min = np.min(depth_new)
        depth_max = np.max(depth_new)
        depth_points = np.size(depth_new)
        depth_new_equi = np.linspace(depth_min, depth_max, depth_points)
        tau500_new = np.interp(depth_new_equi, depth_new, tau500_new)
        temp_new = np.interp(depth_new_equi, depth_new, temp_new)
        pe_new = np.interp(depth_new_equi, depth_new, pe_new)
        vmic_new = np.interp(depth_new_equi, depth_new, vmic_new)
        density_new = np.interp(depth_new_equi, depth_new, density_new)
        depth_new = depth_new_equi
        return tau500_new, temp_new, pe_new, vmic_new, density_new, depth_new

    def calculate_atmosphere(self):
        if self.atmosphere_dimension == "1D":
            self.calculate_atmosphere_1d()
        elif self.atmosphere_dimension == "3D":
            self.calculate_atmosphere_3d()
        else:
            raise ValueError("Atmosphere dimension must be either 1D or 3D: m3dis_class.py")

    def calculate_atmosphere_3d(self):
        return

    def calculate_atmosphere_1d(self):
        possible_turbulence = [0.0, 1.0, 2.0, 5.0]
        flag_dont_interp_microturb = False
        for i in range(len(possible_turbulence)):
            if self.turbulent_velocity == possible_turbulence[i]:
                flag_dont_interp_microturb = True

        if self.log_g < 3:
            flag_dont_interp_microturb = True

        logging.debug(
            f"flag_dont_interp_microturb: {flag_dont_interp_microturb} {self.turbulent_velocity} {self.t_eff} {self.log_g}")


        if not flag_dont_interp_microturb and self.turbulent_velocity < 2.0 and (
                self.turbulent_velocity > 1.0 or (self.turbulent_velocity < 1.0 and self.t_eff < 3900.)):
            # Bracket the microturbulence to figure out what two values to generate the models to interpolate between using Andy's code
            turbulence_low = 0.0
            microturbulence = self.turbulent_velocity
            for i in range(len(possible_turbulence)):
                if self.turbulent_velocity > possible_turbulence[i]:
                    turbulence_low = possible_turbulence[i]
                    place = i
            turbulence_high = possible_turbulence[place + 1]

            self.turbulent_velocity = turbulence_low
            marcs_model_list_low = self._generate_model_atmosphere(run_ts_interpolator=False)
            if marcs_model_list_low["errors"] is not None:
                raise ValueError(f"{marcs_model_list_low['errors']}")
            tau500_low, temp_low, pe_low, vmic_low, density_low, depth_low = self.interpolate_m3dis_atmosphere(marcs_model_list_low["marcs_model_list"])
            self.turbulent_velocity = turbulence_high

            marcs_model_list_high = self._generate_model_atmosphere(run_ts_interpolator=False)
            if marcs_model_list_high["errors"] is not None:
                raise ValueError(f"{marcs_model_list_high['errors']}")
            tau500_high, temp_high, pe_high, vmic_high, density_high, depth_high = self.interpolate_m3dis_atmosphere(marcs_model_list_high["marcs_model_list"])
            atmosphere_properties = marcs_model_list_high
            self.turbulent_velocity = microturbulence

            # interpolate and find a model atmosphere for the microturbulence
            fxhigh = (microturbulence - turbulence_low) / (turbulence_high - turbulence_low)
            fxlow = 1.0 - fxhigh

            tau500_interp = tau500_low * fxlow + tau500_high * fxhigh
            temp_interp = temp_low * fxlow + temp_high * fxhigh
            pe_interp = pe_low * fxlow + pe_high * fxhigh
            vmic_interp = vmic_low * fxlow + vmic_high * fxhigh
            density_interp = density_low * fxlow + density_high * fxhigh
            depth_interp = depth_low * fxlow + depth_high * fxhigh

            # print(interp_model_name)
            self.marcs_model_name = "atmos.marcs_tef{:.1f}_g{:.2f}_z{:.2f}_tur{:.2f}".format(self.t_eff, self.log_g,
                                                                                             self.metallicity,
                                                                                             self.turbulent_velocity)
            interp_model_name = os.path.join(self.tmp_dir, self.marcs_model_name)

            self.save_m3dis_model(interp_model_name, depth_interp, temp_interp, pe_interp, density_interp, vmic_interp)


        elif not flag_dont_interp_microturb and self.turbulent_velocity > 2.0:  # not enough models to interp if higher than 2
            microturbulence = self.turbulent_velocity  # just use 2.0 for the model if between 2 and 3
            self.turbulent_velocity = 2.0
            marcs_model_list = self._generate_model_atmosphere(run_ts_interpolator=False)
            atmosphere_properties = marcs_model_list
            if marcs_model_list["errors"] is not None:
                raise ValueError(f"{marcs_model_list['errors']}")
            tau500, temp, pe, vmic, density, depth = self.interpolate_m3dis_atmosphere(marcs_model_list["marcs_model_list"])
            self.marcs_model_name = "atmos.marcs_tef{:.1f}_g{:.2f}_z{:.2f}_tur{:.2f}".format(self.t_eff, self.log_g,
                                                                                             self.metallicity,
                                                                                             self.turbulent_velocity)
            interp_model_name = os.path.join(self.tmp_dir, self.marcs_model_name)

            self.save_m3dis_model(interp_model_name, depth, temp, pe, density, vmic)
            self.turbulent_velocity = microturbulence

        elif not flag_dont_interp_microturb and self.turbulent_velocity < 1.0 and self.t_eff >= 3900.:  # not enough models to interp if lower than 1 and t_eff > 3900
            microturbulence = self.turbulent_velocity
            self.turbulent_velocity = 1.0
            marcs_model_list = self._generate_model_atmosphere(run_ts_interpolator=False)
            atmosphere_properties = marcs_model_list
            if marcs_model_list["errors"] is not None:
                raise ValueError(f"{marcs_model_list['errors']}")
            tau500, temp, pe, vmic, density, depth = self.interpolate_m3dis_atmosphere(marcs_model_list["marcs_model_list"])
            self.marcs_model_name = "atmos.marcs_tef{:.1f}_g{:.2f}_z{:.2f}_tur{:.2f}".format(self.t_eff, self.log_g,
                                                                                             self.metallicity,
                                                                                             self.turbulent_velocity)
            interp_model_name = os.path.join(self.tmp_dir, self.marcs_model_name)

            self.save_m3dis_model(interp_model_name, depth, temp, pe, density, vmic)
            self.turbulent_velocity = microturbulence


        elif flag_dont_interp_microturb:
            if self.log_g < 3:
                microturbulence = self.turbulent_velocity
                self.turbulent_velocity = 2.0
            marcs_model_list = self._generate_model_atmosphere(run_ts_interpolator=False)
            atmosphere_properties = marcs_model_list
            if marcs_model_list["errors"] is not None:
                raise ValueError(f"{marcs_model_list['errors']}")
            tau500, temp, pe, vmic, density, depth = self.interpolate_m3dis_atmosphere(marcs_model_list["marcs_model_list"])
            self.marcs_model_name = "atmos.marcs_tef{:.1f}_g{:.2f}_z{:.2f}_tur{:.2f}".format(self.t_eff, self.log_g,
                                                                                             self.metallicity,
                                                                                             self.turbulent_velocity)
            interp_model_name = os.path.join(self.tmp_dir, self.marcs_model_name)

            self.save_m3dis_model(interp_model_name, depth, temp, pe, density, vmic)

            if self.log_g < 3:
                self.turbulent_velocity = microturbulence
        else:
            print("Unexpected error?")
        self.atmosphere_properties = atmosphere_properties

    def save_m3dis_model(self, interp_model_name, depth_interp, temp_interp, pe_interp, density_interp, vmic_interp):
        with open(interp_model_name, "w") as file:
            # first is name of file
            file.write(f"{self.marcs_model_name}\n")
            # next is number of points as integer
            depth_points = np.size(depth_interp)
            file.write(f"{depth_points}\n")
            # next is the format
            file.write("* depth      temp       pe        pg      vmic\n")
            for i in range(len(depth_interp)):
                file.write(
                    f"{depth_interp[i]:>13.6e} {temp_interp[i]:>8.1f} {pe_interp[i]:>12.4E} {density_interp[i]:>12.4E} {vmic_interp[i]:>5.3f}\n")

    def synthesize_spectra(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        wavelength, normalised_flux, flux = None, None, None
        try:
            logging.debug("Running m3dis and atmosphere")
            self.calculate_atmosphere()
            logging.debug("Running m3dis")
            output = self.call_m3dis(skip_linelist=self.skip_linelist, use_precomputed_depart=self.use_precomputed_depart)
            if "errors" in output:
                if not self.night_mode:
                    print(output["errors"], "m3dis failed")
            else:
                if self.save_spectra:
                    try:
                        completed_run = self.m3dis_python_module.read(
                            self.tmp_dir
                        )
                        wavelength, _ = completed_run.get_xx(completed_run.lam)
                        flux, continuum = completed_run.get_yy(norm=False)
                        normalised_flux = flux / continuum
                    except FileNotFoundError as e:
                        if not self.night_mode:
                            print(f"m3dis, cannot find {e}")
        except (FileNotFoundError, ValueError, TypeError) as error:
            if not self.night_mode:
                print(f"Interpolation failed? {error}")
        return wavelength, normalised_flux, flux

