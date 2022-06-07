from turbospectrum_class_nlte import TurboSpectrum
import math
import time
import numpy as np
import os
from convolve import *

def calculate_vturb(teff, logg, met):
    t0 = 5500.
    g0 = 4.
    m0 = 0.

    if teff >= 5000.:
        vturb = 1.05 + 2.51e-4*(teff-t0) + 1.5e-7*(teff-t0)*(teff-t0) - 0.14*(logg-g0) - 0.005*(logg-g0)*(logg-g0) + 0.05*met + 0.01*met*met
    elif teff < 5000. and logg >= 3.5:
        vturb = 1.05 + 2.51e-4*(teff-t0) + 1.5e-7*(5250.-t0)*(5250.-t0) - 0.14*(logg-g0) - 0.005*(logg-g0)*(logg-g0) + 0.05*met + 0.01*met*met
    elif teff < 5500. and logg < 3.5:
        vturb = 1.25 + 4.01e-4*(teff-t0) + 3.1e-7*(teff-t0)*(teff-t0) - 0.14*(logg-g0) - 0.005*(logg-g0)*(logg-g0) + 0.05*met + 0.01*met*met

    if teff == 5771 and logg == 4.44:
        vturb = 0.9

    return vturb

#parameters to adjust
ts_compiler = "intel" #needs to be "intel" or "gnu"
atmosphere_type = "1D"
windows_flag = False
teff = 5771
logg = 4.44
met = 0.00
vturb = calculate_vturb(teff, logg, met) #needs to be > 0
#print(vturb)
#vturb = 0.9
lmin = 4500
lmax = 7000
ldelta = 0.005
item_abund = {}
item_abund["H"] = 0.0
item_abund["O"] = 0.0 + met
item_abund["Mg"] = 0.0 + met
#item_abund["Si"] = 0.0 + met
item_abund["Ca"] = 0.0 + met
#item_abund["Mn"] = 0.0 + met #only has mean3d
#item_abund["Ti"] = 0.0
item_abund["Fe"] = met
#item_abund["Ni"] = 0.0 + met #only has mean3d
#item_abund["Ba"] = 0.0 + met
temp_directory = "../temp_directory_solar/"

if not os.path.exists(temp_directory):
    os.makedirs(temp_directory)

#parameters for convolving if needed, if not set to zero
fwhm = 0.0 #fwhm in milli-angstroms, negative in km/s, 0 means no convolution based on resolution, was 160 for higher convolution h lines
macroturbulence = 3.5 #in km/s

#adjust the following only if using windows mode. if not, you can leave alone
linemask_file = "Fe/fe-lmask.txt"
segment_file = "Fe/fe-seg.txt"

nlte_flag = True
#other files needed for nlte calculations, ignore if not using nlte
depart_bin_file = {}
depart_bin_file["H"] = "H/1D_NLTE_grid_H_MARCSfullGrid_reformat_May-10-2021.bin"
#depart_bin_file["H"] = "H/H_av3DSTAGGER_NLTEgrid4TS_Jun-17-2021.bin"
depart_bin_file["O"] = "O/NLTEgrid4TS_O_MARCS_May-21-2021.bin"
#depart_bin_file["O"] = "O/NLTEgrid4TS_O_STAGGER_May-18-2021.bin"
depart_bin_file["Mg"] = "Mg/NLTEgrid4TS_Mg_MARCS_Jun-02-2021.bin"
#depart_bin_file["Mg"] = "Mg/NLTEgrid_Mg_Mean3D_May-17-2021.bin"
depart_bin_file["Ca"] = "Ca/Ca_MARCS_NLTEgrid4TS_Jun-02-2021.bin"
#depart_bin_file["Ca"] = "Ca/output_NLTEgrid4TS_av3D_STAGGER_May-18-2021.bin"
#depart_bin_file["Mn"] = "Mn/NLTEgrid_inprogress_output_Mn_MARCS.bin"
#depart_bin_file["Mn"] = "Mn/NLTEgrid_Mn_mean3D_May-17-2021.bin"
depart_bin_file["Fe"] = "Fe/output_NLTEgrid4TS_MARCS_May-07-2021.bin"
#depart_bin_file["Fe"] = "Fe/1D_NLTE_grid_Fe_mean3D_reformat_May-21-2021.bin"
#depart_bin_file["Ni"] = "Ni/NLTEgrid4TS_Ni_MARCS_Jul-03-2021.bin"
#depart_bin_file["Ni"] = "Ni/Ni_STAGGER_av3D_NLTEgrid4TS_Jun-10-2021.bin"
#depart_bin_file["Ba"] = "Ba/NLTEgrid_Ba_MARCS_May-10-2021.bin"
#depart_bin_file["Ba"] = "Ba/NLTEgrid_output_Ba_mean3D_May-10-2021.bin"
depart_aux_file = {}
depart_aux_file["H"] = "H/auxData_H_MARCSfullGrid_reformat_May-10-2021.txt"
#depart_aux_file["H"] = "H/H_av3DSTAGGER_auxData_Jun-17-2021_marcs_names.txt"
depart_aux_file["O"] = "O/auxData_NLTEgrid4TS_O_MARCS_May-21-2021.txt"
#depart_aux_file["O"] = "O/auxData_NLTEgrid4TS_O_STAGGER_May-18-2021_marcs_names.txt"
depart_aux_file["Mg"] = "Mg/auxData_NLTEgrid4TS_Mg_MARCS_Jun-02-2021.dat"
#depart_aux_file["Mg"] = "Mg/auxData_Mg_Mean3D_May-17-2021_marcs_names.txt"
depart_aux_file["Ca"] = "Ca/auxData_Ca_MARCS_Jun-02-2021.dat"
#depart_aux_file["Ca"] = "Ca/auxData_NLTEgrid4TS_av3D_STAGGER_May-18-2021_marcs_names.txt"
#depart_aux_file["Mn"] = "Mn/auxData_inprogress_output_Mn_MARCS.txt"
#depart_aux_file["Mn"] = "Mn/auxData_Mn_mean3D_May-17-2021.txt"
depart_aux_file["Fe"] = "Fe/auxData_NLTEgrid4TS_MARCS_May-07-2021.dat"
#depart_aux_file["Fe"] = "Fe/auxData_Fe_mean3D_reformat_May-21-2021_marcs_names.txt"
#depart_aux_file["Ni"] = "Ni/auxData_Ni_MARCS_Jul-03-2021.txt"
#depart_aux_file["Ni"] = "Ni/Ni_STAGGER_av3D_auxData_Jun-10-2021_marcs_names.txt"
#depart_aux_file["Ba"] = "Ba/auxData_Ba_MARCS_May-10-2021.txt"
#depart_aux_file["Ba"] = "Ba/auxData_output_Ba_mean3D_May-10-2021_marcs_names.txt"
model_atom_file = {}
model_atom_file["H"] = "atom.h20"
model_atom_file["O"] = "atom.o41f"
model_atom_file["Mg"] = "atom.mg86b"
model_atom_file["Ca"] = "atom.caNew"
#model_atom_file["Mn"] = "atom.mn281kbc"
model_atom_file["Fe"] = "atom.fe607"
#model_atom_file["Ni"] = "atom.ni538sh0051000fbc"
#model_atom_file["Ba"] = "atom.ba111"

#whether or not you want to rename output file, default is "final_spectrum.spec"
rename_spectrum = True
spectrum_name = "solar_1d_nlte.spec"

#set directories
if ts_compiler == "intel":
    turbospec_path = "../turbospectrum/exec/"
elif ts_compiler == "gnu":
    turbospec_path = "../turbospectrum/exec-gf/"
interpol_path = "./model_interpolators/"
line_list_path = "../input_files/linelists/linelist_for_fitting/"
if atmosphere_type == "1D":
    model_atmosphere_grid_path = "../input_files/model_atmospheres/1D/"
    model_atmosphere_list = model_atmosphere_grid_path+"model_atmosphere_list.txt"
elif atmosphere_type == "3D":
    model_atmosphere_grid_path = "../input_files/model_atmospheres/3D/"
    model_atmosphere_list = model_atmosphere_grid_path+"model_atmosphere_list.txt"
model_atom_path = "../input_files/nlte_data/model_atoms/"
departure_file_path = "../input_files/nlte_data/"

linemask_file = "../input_files/linemask_files/"+linemask_file
segment_file = "../input_files/linemask_files/"+segment_file
#continuum_file = "../input_files/linemask_files/"+continuum_file

ts = TurboSpectrum(
            turbospec_path=turbospec_path,
            interpol_path=interpol_path,
            line_list_paths=line_list_path,
            marcs_grid_path=model_atmosphere_grid_path,
            marcs_grid_list=model_atmosphere_list,
            model_atom_path=model_atom_path,
            departure_file_path=departure_file_path)

time_start = time.time()

ts.configure(t_eff = teff, log_g = logg, metallicity = met, 
                            turbulent_velocity = vturb, lambda_delta = ldelta, lambda_min=lmin, lambda_max=lmax, 
                            free_abundances=item_abund, temp_directory = temp_directory, nlte_flag=nlte_flag, verbose=False, 
                            atmosphere_dimension=atmosphere_type, windows_flag=windows_flag, segment_file=segment_file, 
                            line_mask_file=linemask_file, depart_bin_file=depart_bin_file, 
                            depart_aux_file=depart_aux_file, model_atom_file=model_atom_file)

#time_end = time.time()
#print("Total runtime was {:.2f} seconds.".format((time_end-time_start)))
#time_start = time.time()

ts.calculate_atmosphere()

time_end = time.time()
print("Total runtime babsma was {:.2f} seconds.".format((time_end-time_start)))
time_start = time.time()

ts.run_turbospectrum()
#ts.run_turbospectrum_and_atmosphere()

time_end = time.time()
print("Total runtime bsyn was {:.2f} seconds.".format((time_end-time_start)))
time_start = time.time()

wave_mod_orig, flux_norm_mod_orig, flux_mod_orig = np.loadtxt('{}/spectrum_00000000.spec'.format(temp_directory), usecols=(0,1,2), unpack=True)
if windows_flag == True:
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
            flux_norm_mod_filled.append(flux_norm_mod_filled[j])
            flux_mod_filled.append(flux_mod_orig[j])
            j+=1
        if i < len(seg_begins)-1:
            k = 1
            while (seg_begins[i+1] - 0.001 > seg_ends[i]+k*0.005):
                wave_mod_filled.append(seg_ends[i]+0.005*k)
                flux_norm_mod_filled.append(1.0)
                flux_mod_filled.append(np.mean(flux_mod_orig))
                k+=1
elif windows_flag == False:
    wave_mod_filled = wave_mod_orig
    flux_norm_mod_filled = flux_norm_mod_orig
    flux_mod_filled = flux_mod_orig

if fwhm != 0.0:
    wave_mod_conv, flux_norm_mod_conv = conv_res(wave_mod_filled, flux_norm_mod_filled, fwhm)
    wave_mod_conv, flux_mod_conv = conv_res(wave_mod_filled, flux_mod_filled, fwhm)
else:
    wave_mod_conv = wave_mod_filled
    flux_norm_mod_conv = flux_norm_mod_filled
    flux_mod_conv = flux_mod_filled

wave_mod, flux_norm_mod = conv_macroturbulence(wave_mod_conv, flux_norm_mod_conv, macroturbulence)
wave_mod, flux_mod = conv_macroturbulence(wave_mod_conv, flux_mod_conv, macroturbulence)

if rename_spectrum == True:
    #os.system("mv {}spectrum_00000000.spec {}{}".format(temp_directory, temp_directory, spectrum_name))
    f = open("{}{}".format(temp_directory, spectrum_name), 'w')
    for i in range(len(wave_mod)):
        print("{}  {}  {}".format(wave_mod[i], flux_norm_mod[i], flux_mod[i]), file=f)
    f.close()
elif rename_spectrum == False:
    #os.system("mv {}spectrum_00000000.spec {}{}".format(temp_directory, temp_directory, spectrum_name))
    f = open("{}final_spectrum.spec".format(temp_directory), 'w')
    for i in range(len(wave_mod)):
        print("{}  {}  {}".format(wave_mod[i], flux_norm_mod[i], flux_mod[i]), file=f)
    f.close()

time_end = time.time()

print("Total runtime was {:.2f} minutes.".format((time_end-time_start)/60.))
