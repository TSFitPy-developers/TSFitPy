from turbospectrum_class_nlte import TurboSpectrum, fetch_marcs_grid
import math
import time
import numpy as np
import os
from convolve import *
import datetime
from create_window_linelist_function import create_window_linelist
import shutil
from dask.distributed import Client
import socket

def calculate_vturb(teff, logg, met):
    t0 = 5500.
    g0 = 4.

    if teff >= 5000.:
        vturb = 1.05 + 2.51e-4*(teff-t0) + 1.5e-7*(teff-t0)*(teff-t0) - 0.14*(logg-g0) - 0.005*(logg-g0)*(logg-g0) + 0.05*met + 0.01*met*met
    elif teff < 5000. and logg >= 3.5:
        vturb = 1.05 + 2.51e-4*(teff-t0) + 1.5e-7*(5250.-t0)*(5250.-t0) - 0.14*(logg-g0) - 0.005*(logg-g0)*(logg-g0) + 0.05*met + 0.01*met*met
    elif teff < 5500. and logg < 3.5:
        vturb = 1.25 + 4.01e-4*(teff-t0) + 3.1e-7*(teff-t0)*(teff-t0) - 0.14*(logg-g0) - 0.005*(logg-g0)*(logg-g0) + 0.05*met + 0.01*met*met

    return vturb

def run_wrapper(teff, logg, met, lmin, lmax, ldelta, nlte_flag, resolution=0, macro=0, rotation=0, vturb=None):
    #parameters to adjust

    teff = teff
    logg = logg
    met = met
    if vturb is None:
        vturb = calculate_vturb(teff, logg, met)
    #print(vturb)
    #vturb = 0.9
    lmin = lmin
    lmax = lmax
    ldelta = ldelta
    item_abund = {}
    #item_abund["H"] = 0.0
    #item_abund["O"] = 0.0 + met
    item_abund["Mg"] = 0.0 + met
    #item_abund["Si"] = 0.0 + met
    #item_abund["Ca"] = 0.0 + met
    #item_abund["Mn"] = 0.0 + met
    item_abund["Ti"] = 0.0 + met
    item_abund["Fe"] = met
    #item_abund["Y"] = 0 + met
    #item_abund["Ni"] = 0.0 + met
    #item_abund["Ba"] = 0.0 + met
    temp_directory = f"../temp_directory_{datetime.datetime.now().strftime('%b-%d-%Y-%H-%M-%S')}__{np.random.random(1)[0]}/"

    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)

    ts = TurboSpectrum(
                turbospec_path=turbospec_path,
                interpol_path=interpol_path,
                line_list_paths=line_list_path_trimmed,
                marcs_grid_path=model_atmosphere_grid_path,
                marcs_grid_list=model_atmosphere_list,
                model_atom_path=model_atom_path,
                departure_file_path=departure_file_path,
                aux_file_length_dict=aux_file_length_dict,
                model_temperatures=model_temperatures,
                model_logs=model_logs,
                model_mets=model_mets,
                marcs_value_keys=marcs_value_keys,
                marcs_models=marcs_models,
                marcs_values=marcs_values)

    ts.configure(t_eff = teff, log_g = logg, metallicity = met,
                                turbulent_velocity = vturb, lambda_delta = ldelta, lambda_min=lmin, lambda_max=lmax,
                                free_abundances=item_abund, temp_directory = temp_directory, nlte_flag=nlte_flag, verbose=False,
                                atmosphere_dimension=atmosphere_type, windows_flag=windows_flag, segment_file=segment_file,
                                line_mask_file=linemask_file, depart_bin_file=depart_bin_file,
                                depart_aux_file=depart_aux_file, model_atom_file=model_atom_file)

    ts.run_turbospectrum_and_atmosphere()

    wave_mod_orig, flux_norm_mod_orig, flux_mod_orig = np.loadtxt('{}spectrum_00000000.spec'.format(temp_directory), usecols=(0,1,2), unpack=True)
    if windows_flag:
        seg_begins, seg_ends = np.loadtxt(segment_file, comments = ";", usecols=(0,1), unpack=True)
        wave_mod_filled = []
        flux_norm_mod_filled = []
        flux_mod_filled = []
        for i in range(len(seg_begins)):
            j = 0
            while wave_mod_orig[j] < seg_begins[i]:
                j+=1
            while wave_mod_orig[j] >= seg_begins[i] and wave_mod_orig[j] <= seg_ends[i]:
                wave_mod_filled.append(wave_mod_orig[j])
                flux_norm_mod_filled.append(flux_norm_mod_orig[j])
                flux_mod_filled.append(flux_mod_orig[j])
                j+=1
            if i < len(seg_begins)-1:
                k = 1
                while (seg_begins[i+1] - 0.001 > seg_ends[i]+k*0.005):
                    wave_mod_filled.append(seg_ends[i]+0.005*k)
                    flux_norm_mod_filled.append(1.0)
                    flux_mod_filled.append(np.mean(flux_mod_orig))
                    k+=1
    else:
        wave_mod_filled = wave_mod_orig
        flux_norm_mod_filled = flux_norm_mod_orig
        flux_mod_filled = flux_mod_orig

    if resolution != 0.0:
        wave_mod_conv, flux_norm_mod_conv = conv_res(wave_mod_filled, flux_norm_mod_filled, resolution)
        wave_mod_conv, flux_mod_conv = conv_res(wave_mod_filled, flux_mod_filled, resolution)
    else:
        wave_mod_conv = wave_mod_filled
        flux_norm_mod_conv = flux_norm_mod_filled
        flux_mod_conv = flux_mod_filled

    if macro != 0.0:
        wave_mod_macro, flux_norm_mod_macro = conv_macroturbulence(wave_mod_conv, flux_norm_mod_conv, macro)
        wave_mod_macro, flux_mod_macro = conv_macroturbulence(wave_mod_conv, flux_mod_conv, macro)
    else:
        wave_mod_macro = wave_mod_conv
        flux_norm_mod_macro = flux_norm_mod_conv
        flux_mod_macro = flux_mod_conv

    if rotation != 0.0:
        wave_mod, flux_norm_mod = conv_rotation(wave_mod_macro, flux_norm_mod_macro, rotation)
        wave_mod, flux_mod = conv_rotation(wave_mod_macro, flux_mod_macro, rotation)
    else:
        wave_mod = wave_mod_macro
        flux_norm_mod = flux_norm_mod_macro
        flux_mod = flux_mod_macro

    shutil.rmtree(temp_directory)

    return wave_mod, flux_norm_mod, flux_mod


def run_and_save_wrapper(teff, logg, met, lmin, lmax, ldelta, spectrum_name, nlte_flag, resolution, macro, rotation, new_directory_to_save_to, vturb):
    wave_mod, flux_norm_mod, flux_mod = run_wrapper(teff, logg, met, lmin, lmax, ldelta, nlte_flag, resolution, macro, rotation, vturb)
    file_location_output = os.path.join(new_directory_to_save_to, f"{spectrum_name}_{met}_{str(nlte_flag)}.spec")
    f = open(file_location_output, 'w')
    for i in range(len(wave_mod)):
        print("{}  {}  {}".format(wave_mod[i], flux_norm_mod[i], flux_mod[i]), file=f)
    f.close()


if __name__ == '__main__':
    ts_compiler = "intel" #needs to be "intel" or "gnu"
    atmosphere_type = "1D"

    windows_flag = False
    #adjust the following only if using windows mode. if not, you can leave alone
    linemask_file = "Fe/fe-lmask.txt"
    segment_file = "Fe/fe-seg.txt"
    linemask_file = "../input_files/linemask_files/" + linemask_file
    segment_file = "../input_files/linemask_files/" + segment_file

    # other files needed for nlte calculations, ignore if not using nlte
    depart_bin_file = {}
    # depart_bin_file["H"] = "H/1D_NLTE_grid_H_MARCSfullGrid_reformat_May-10-2021.bin"
    # depart_bin_file["H"] = "H/H_av3DSTAGGER_NLTEgrid4TS_Jun-17-2021.bin"
    # depart_bin_file["O"] = "O/NLTEgrid4TS_O_MARCS_May-21-2021.bin"
    # depart_bin_file["O"] = "O/NLTEgrid4TS_O_STAGGER_May-18-2021.bin"
    depart_bin_file["Mg"] = "Mg/NLTEgrid4TS_Mg_MARCS_Jun-02-2021.bin"
    # depart_bin_file["Mg"] = "Mg/NLTEgrid_Mg_Mean3D_May-17-2021.bin"
    # depart_bin_file["Ca"] = "Ca/Ca_MARCS_NLTEgrid4TS_Jun-02-2021.bin"
    # depart_bin_file["Ca"] = "Ca/output_NLTEgrid4TS_av3D_STAGGER_May-18-2021.bin"
    # depart_bin_file["Mn"] = "Mn/NLTEgrid_inprogress_output_Mn_MARCS.bin"
    # depart_bin_file["Mn"] = "Mn/NLTEgrid_Mn_mean3D_May-17-2021.bin"
    depart_bin_file["Fe"] = "Fe/NLTEgrid4TS_Fe_MARCS_May-07-2021.bin"
    depart_bin_file["Ti"] = "Ti/NLTEgrid4TS_TI_MARCS_Feb-21-2022.bin"
    # depart_bin_file["Fe"] = "Fe/1D_NLTE_grid_Fe_mean3D_reformat_May-21-2021.bin"
    # depart_bin_file["Ni"] = "Ni/NLTEgrid4TS_Ni_MARCS_Jul-03-2021.bin"
    # depart_bin_file["Ni"] = "Ni/Ni_STAGGER_av3D_NLTEgrid4TS_Jun-10-2021.bin"
    # depart_bin_file["Ba"] = "Ba/NLTEgrid_Ba_MARCS_May-10-2021.bin"
    # depart_bin_file["Ba"] = "Ba/NLTEgrid_output_Ba_mean3D_May-10-2021.bin"
    depart_aux_file = {}
    # depart_aux_file["H"] = "H/auxData_H_MARCSfullGrid_reformat_May-10-2021.txt"
    # depart_aux_file["H"] = "H/H_av3DSTAGGER_auxData_Jun-17-2021_marcs_names.txt"
    # depart_aux_file["O"] = "O/auxData_NLTEgrid4TS_O_MARCS_May-21-2021.txt"
    # depart_aux_file["O"] = "O/auxData_NLTEgrid4TS_O_STAGGER_May-18-2021_marcs_names.txt"
    depart_aux_file["Mg"] = "Mg/auxData_Mg_MARCS_Jun-02-2021.dat"
    # depart_aux_file["Mg"] = "Mg/auxData_Mg_Mean3D_May-17-2021_marcs_names.txt"
    # depart_aux_file["Ca"] = "Ca/auxData_Ca_MARCS_Jun-02-2021.dat"
    # depart_aux_file["Ca"] = "Ca/auxData_NLTEgrid4TS_av3D_STAGGER_May-18-2021_marcs_names.txt"
    # depart_aux_file["Mn"] = "Mn/auxData_inprogress_output_Mn_MARCS.txt"
    # depart_aux_file["Mn"] = "Mn/auxData_Mn_mean3D_May-17-2021.txt"
    depart_aux_file["Fe"] = "Fe/auxData_Fe_MARCS_May-07-2021.dat"
    depart_aux_file["Ti"] = "Ti/auxData_TI_MARCS_Feb-21-2022.dat"
    # depart_aux_file["Fe"] = "Fe/auxData_Fe_mean3D_reformat_May-21-2021_marcs_names.txt"
    # depart_aux_file["Ni"] = "Ni/auxData_Ni_MARCS_Jul-03-2021.txt"
    # depart_aux_file["Ni"] = "Ni/Ni_STAGGER_av3D_auxData_Jun-10-2021_marcs_names.txt"
    # depart_aux_file["Ba"] = "Ba/auxData_Ba_MARCS_May-10-2021.txt"
    # depart_aux_file["Ba"] = "Ba/auxData_output_Ba_mean3D_May-10-2021_marcs_names.txt"
    model_atom_file = {}
    # model_atom_file["H"] = "atom.h20"
    # model_atom_file["O"] = "atom.o41f"
    model_atom_file["Mg"] = "atom.mg86b"
    # model_atom_file["Ca"] = "atom.caNew"
    # model_atom_file["Mn"] = "atom.mn281kbc"
    model_atom_file["Fe"] = "atom.fe607a"
    model_atom_file["Ti"] = "atom.ti503"
    # model_atom_file["Ni"] = "atom.ni538sh0051000fbc"
    # model_atom_file["Ba"] = "atom.ba111"

    #set directories
    if ts_compiler == "intel":
        turbospec_path = "../turbospectrum/exec/"
    elif ts_compiler == "gnu":
        turbospec_path = "../turbospectrum/exec-gf/"
    interpol_path = "./model_interpolators/"
    line_list_path = "../input_files/linelists/linelist_for_fitting/"
    if atmosphere_type == "1D":
        model_atmosphere_grid_path = "../input_files/model_atmospheres/1D/"
        model_atmosphere_list = model_atmosphere_grid_path + "model_atmosphere_list.txt"
    elif atmosphere_type == "3D":
        model_atmosphere_grid_path = "../input_files/model_atmospheres/3D/"
        model_atmosphere_list = model_atmosphere_grid_path + "model_atmosphere_list.txt"
    model_atom_path = "../input_files/nlte_data/model_atoms/"
    departure_file_path = "../input_files/nlte_data/"

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

    futures = []
    for metal in met_list:
        spectrum_name = f"spectra_output_{teff}_{logg}_"
        future = client.submit(run_and_save_wrapper, teff, logg, metal, lmin, lmax, ldelta, spectrum_name, nlte_flag, resolution, macro, rotation, output_dir, vturb)
        futures.append(future)  # prepares to get values

    print("Start gathering")  # use http://localhost:8787/status to check status. the port might be different
    futures = np.array(client.gather(futures))  # starts the calculations (takes a long time here)
    results = futures
    print("Worker calculation done")  # when done, save values


    shutil.rmtree(line_list_path_trimmed)  # clean up trimmed line list