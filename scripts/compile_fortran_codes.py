import os
from os import path as os_path

compiler = input("Type of compiler (GNU or IFORT or IFX): ")

if compiler.lower == "ifort":
	os.system("ifort -o faltbon faltbon.f90")
	os.chdir("./model_interpolators/")
	os.system("ifort -o interpol_modeles interpol_modeles.f")
	os.system("ifort -o interpol_modeles_nlte interpol_modeles_nlte.f")
	os.system("ifort -o interpol_multi interpol_multi.f")
	os.system("ifort -o interpol_multi_nlte interpol_multi_nlte.f")
elif compiler.lower == "gnu":
	os.system("gfortran -o faltbon faltbon.f90")
	os.chdir("./model_interpolators/") #need a separate file for GNU and INTEL codes
	os.system("gfortran -o interpol_modeles interpol_modeles.f")
	os.system("gfortran -o interpol_modeles_nlte interpol_modeles_nlte_gfort.f")
	os.system("gfortran -o interpol_multi interpol_multi.f")
	os.system("gfortran -o interpol_multi_nlte interpol_multi_nlte_gfort.f")
elif compiler.lower == "ifx":
	os.system("ifx -o faltbon faltbon.f90")
	os.chdir("./model_interpolators/")
	os.system("ifx -o interpol_modeles interpol_modeles.f")
	os.system("ifx -o interpol_modeles_nlte interpol_modeles_nlte.f")
	os.system("ifx -o interpol_multi interpol_multi.f")
	os.system("ifx -o interpol_multi_nlte interpol_multi_nlte.f")
else:
	print('Write only GNU, IFORT or IFX, you wrote: ')
	print(compiler)