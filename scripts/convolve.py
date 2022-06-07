import numpy as np
import os
import subprocess

def conv_res(wave, flux, fwhm):
	#create a file to feed to ./faltbon of spectrum that needs convolving
	f = open('original.txt', 'w')
	spud = [9.999 for i in range(len(wave))] #faltbon needs three columns of data, but only cares about the first two
	for j in range(len(wave)):
		f.write("{:f}  {:f}  {:f}\n".format(wave[j], flux[j], spud[j]))
	f.close()
	os.system("{ echo original.txt; echo convolve.txt; echo %f; echo 2; } | ./faltbon > faltbon_out_spud.txt" % fwhm)
	os.system("rm faltbon_out_spud.txt")
	wave_conv, flux_conv, spud = np.loadtxt("convolve.txt", unpack='True') #read in our new convolved spectrum
	return wave_conv, flux_conv

def conv_macroturbulence(wave, flux, fwhm):
	#create a file to feed to ./faltbon of spectrum that needs convolving
	f = open('original.txt', 'w')
	spud = [9.999 for i in range(len(wave))] #faltbon needs three columns of data, but only cares about the first two
	for j in range(len(wave)):
		f.write("{:f}  {:f}  {:f}\n".format(wave[j], flux[j], spud[j]))
	f.close()
	os.system("{ echo original.txt; echo convolve.txt; echo %f; echo 3; } | ./faltbon > faltbon_out_spud.txt" % -fwhm)
	os.system("rm faltbon_out_spud.txt")
	wave_conv, flux_conv, spud = np.loadtxt("convolve.txt", unpack='True') #read in our new convolved spectrum
	return wave_conv, flux_conv

def conv_rotation(wave, flux, fwhm):
	#create a file to feed to ./faltbon of spectrum that needs convolving
	f = open('original.txt', 'w')
	spud = [9.999 for i in range(len(wave))] #faltbon needs three columns of data, but only cares about the first two
	for j in range(len(wave)):
		f.write("{:f}  {:f}  {:f}\n".format(wave[j], flux[j], spud[j]))
	f.close()
	os.system("{ echo original.txt; echo convolve.txt; echo %f; echo 4; } | ./faltbon > faltbon_out_spud.txt" % -fwhm)
	os.system("rm faltbon_out_spud.txt")
	wave_conv, flux_conv, spud = np.loadtxt("convolve.txt", unpack='True') #read in our new convolved spectrum
	return wave_conv, flux_conv
#def conv_rotation(spectrum_name, convolve_name, fwhm):
#	stdout = open('/dev/null', 'w')
#	stderr = subprocess.STDOUT
#	faltbon_config = ""
#	faltbon_config += "{}\n".format(spectrum_name)
#	faltbon_config += "{}\n".format(convolve_name)
#	faltbon_config += "{}\n".format(-fwhm)
#	faltbon_config += "4\n"
#	try:
#		p = subprocess.Popen('./faltbon',
#							 stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
#		p.stdin.write(bytes(faltbon_config, 'utf-8'))
#		stdout, stderr = p.communicate()
#	except subprocess.CalledProcessError:
#			print("spud")
#			return {
#				"interpol_config": faltbon_config,
#				"errors": "faltbon convolution failed."
#			}
#	wave_conv, flux_conv, spud = np.loadtxt(convolve_name, unpack='True') #read in our new convolved spectrum
#	return wave_conv, flux_conv
