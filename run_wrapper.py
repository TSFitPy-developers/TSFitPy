from configparser import ConfigParser

from scripts.turbospectrum_class_nlte import TurboSpectrum, fetch_marcs_grid
from scripts.convolve import *
import datetime
from scripts.create_window_linelist_function import create_window_linelist
import shutil
from dask.distributed import Client
import socket
import os
from scripts.run_wrapper_v2 import run_and_save_wrapper

if __name__ == '__main__':
    ts_compiler = "intel" #needs to be "intel" or "gnu"
    atmosphere_type = "1D"
    nlte_elements = [] # e.g. ["Mg", "Si", "Ca", "Mn", "Ti", "Fe", "Y", "Ni", "Ba"]

    windows_flag = False
    #adjust the following only if using windows mode. if not, you can leave alone
    linemask_file = "Fe/fe-lmask.txt"
    segment_file = "Fe/fe-seg.txt"
    linemask_file = "../input_files/linemask_files/" + linemask_file
    segment_file = "../input_files/linemask_files/" + segment_file

    #set directories
    if ts_compiler == "intel":
        turbospec_path = "/turbospectrum/exec/"
    elif ts_compiler == "gnu":
        turbospec_path = "/turbospectrum/exec-gf/"
    interpol_path = "./scripts/model_interpolators/"
    line_list_path = "/input_files/linelists/linelist_for_fitting/"
    if atmosphere_type == "1D":
        model_atmosphere_grid_path = "/input_files/model_atmospheres/1D/"
        model_atmosphere_list = model_atmosphere_grid_path + "model_atmosphere_list.txt"
    elif atmosphere_type == "3D":
        model_atmosphere_grid_path = "/input_files/model_atmospheres/3D/"
        model_atmosphere_list = model_atmosphere_grid_path + "model_atmosphere_list.txt"
    model_atom_path = "/input_files/nlte_data/model_atoms/"
    departure_file_path = "/input_files/nlte_data/"

    nlte_config = ConfigParser()
    nlte_config.read(os.path.join(departure_file_path, "nlte_filenames.cfg"))

    depart_bin_file_dict, depart_aux_file_dict, model_atom_file_dict = {}, {}, {}

    for element in nlte_elements:
        if atmosphere_type == "1D":
            bin_config_name, aux_config_name = "1d_bin", "1d_aux"
        else:
            bin_config_name, aux_config_name = "3d_bin", "3d_aux"
        depart_bin_file_dict[element] = nlte_config[element][bin_config_name]
        depart_aux_file_dict[element] = nlte_config[element][aux_config_name]
        model_atom_file_dict[element] = nlte_config[element]["atom_file"]

    depart_bin_file, depart_aux_file, model_atom_file = depart_bin_file_dict, depart_aux_file_dict, model_atom_file_dict

    teff = 5777
    logg = 4.4
    #met = 0.0      # met chosen below
    met_list = [0.0]    # metallicities to generate
    vturb = 1.0

    nlte_flag = False

    line_to_check = 4883
    lmin = line_to_check - 20   # change lmin/lmax here
    lmax = line_to_check + 20
    lmin = 4800                 # or change lmin/lmax here
    lmax = 5500
    ldelta = 0.005

    cpus_to_use = 1     # how many cpus to use (Dask workers)

    """met = np.arange(-1.5, 0.5, 0.1)
    nlte_flag = np.array([True, False])
    one, two = np.meshgrid(met, nlte_flag)
    met_list = one.flatten()
    nlte_flag_list = two.flatten()"""

    model_temperatures, model_logs, model_mets, marcs_value_keys, marcs_models, marcs_values = fetch_marcs_grid(model_atmosphere_list, TurboSpectrum.marcs_parameters_to_ignore)
    aux_file_length_dict = {}
    if nlte_flag:
        for element in model_atom_file:
            aux_file_length_dict[element] = len(np.loadtxt(os.path.join(departure_file_path, depart_aux_file[element]), dtype='str'))

    output_dir = "../synt_spectra_to_fit/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    today = datetime.datetime.now().strftime("%b-%d-%Y-%H-%M-%S")  # used to not conflict with other instances of fits
    today = f"{today}_{np.random.random(1)[0]}"

    line_list_path_trimmed = os.path.join(f"../temp_directory/", "linelist_for_fitting_trimmed", "")
    line_list_path_trimmed = os.path.join(line_list_path_trimmed, "all", today, '')

    print("Trimming")
    include_molecules = True
    create_window_linelist([lmin], [lmax], line_list_path, line_list_path_trimmed, include_molecules, False)
    print("trimming done")

    line_list_path_trimmed = os.path.join(line_list_path_trimmed, "0", "")

    login_node_address = "gemini-login.mpia.de"

    print("Preparing workers")
    client = Client(threads_per_worker=1, n_workers=cpus_to_use)  # if # of threads are not equal to 1, then may break the program
    print(client)

    host = client.run_on_scheduler(socket.gethostname)
    port = client.scheduler_info()['services']['dashboard']
    print(f"Assuming that the cluster is ran at {login_node_address} (change in code if not the case)")

    # print(logger.info(f"ssh -N -L {port}:{host}:{port} {login_node_address}"))
    print(f"ssh -N -L {port}:{host}:{port} {login_node_address}")

    print("Worker preparation complete")

    resolution, macro, rotation = 0, 0, 0

    ts_config = {"turbospec_path": turbospec_path,
                 "interpol_path": interpol_path,
                 "line_list_paths": line_list_path_trimmed,
                 "model_atmosphere_grid_path": model_atmosphere_grid_path,
                 "model_atmosphere_grid_list": model_atmosphere_list,
                 "model_atom_path": model_atom_path,
                 "model_temperatures": model_temperatures,
                 "model_logs": model_logs,
                 "model_mets": model_mets,
                 "marcs_value_keys": marcs_value_keys,
                 "marcs_models": marcs_models,
                 "marcs_values": marcs_values,
                 "aux_file_length_dict": aux_file_length_dict,
                 "departure_file_path": departure_file_path,
                 "atmosphere_type": atmosphere_type,
                 "windows_flag": windows_flag,
                 "segment_file": segment_file,
                 "line_mask_file": linemask_file,
                 "depart_bin_file": depart_bin_file,
                 "depart_aux_file": depart_aux_file,
                 "model_atom_file": model_atom_file}

    futures = []
    for metal in met_list:
        spectrum_name = f"spectra_output_{teff}_{logg}_"
        future = client.submit(run_and_save_wrapper, ts_config, teff, logg, metal, lmin, lmax, ldelta, spectrum_name, nlte_flag, resolution, macro, rotation, output_dir, vturb)
        futures.append(future)  # prepares to get values

    print("Start gathering")  # use http://localhost:8787/status to check status. the port might be different
    futures = np.array(client.gather(futures))  # starts the calculations (takes a long time here)
    results = futures
    print("Worker calculation done")  # when done, save values


    shutil.rmtree(line_list_path_trimmed)  # clean up trimmed line list