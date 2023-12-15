from __future__ import annotations

from typing import Union

# class for MARCS model atmospheres
import numpy as np
import matplotlib.pyplot as plt
from scripts.solar_abundances import  periodic_table
import matplotlib
from scipy.interpolate import interp1d

matplotlib.use("MacOSX")

class MARCSModel:
    def __init__(self, file):
        self.file = file
        # read in file. it is a constantly formatted file
        with open(file) as f:
            lines_marcs_model = f.readlines()
        current_line = 0
        # first line is name of the atmosphere model
        self.name = lines_marcs_model[current_line].strip()
        current_line += 1
        # second line is temperature with junk at the end
        self.temperature = float(lines_marcs_model[current_line].strip().split()[0])
        current_line += 1
        # third line if flux in erg/s/cm^2 with junk at the end
        self.flux = float(lines_marcs_model[current_line].strip().split()[0])
        current_line += 1
        # fourth line is log(g) with junk at the end
        self.logg = float(lines_marcs_model[current_line].strip().split()[0])
        current_line += 1
        # fifth line is vmicro with junk at the end
        self.vmicro = float(lines_marcs_model[current_line].strip().split()[0])
        current_line += 1
        # sixth line is mass with junk at the end
        self.mass = float(lines_marcs_model[current_line].strip().split()[0])
        current_line += 1
        # seventh line is metallicity and alpha with junk at the end
        self.metallicity = float(lines_marcs_model[current_line].strip().split()[0])
        self.alpha = float(lines_marcs_model[current_line].strip().split()[1])
        current_line += 1
        # eighth line is 1 cm radius for plane-parallel models with junk at the end
        self.radius = float(lines_marcs_model[current_line].strip().split()[0])
        current_line += 1
        # ninth line is Luminosity [Lsun] FOR A RADIUS OF 1 cm! with junk at the end
        self.luminosity_1cm = float(lines_marcs_model[current_line].strip().split()[0])
        current_line += 1
        # tenth line is are the convection parameters: alpha, nu, y and beta with junk at the end
        self.alpha_convection = float(lines_marcs_model[current_line].strip().split()[0])
        self.nu_convection = float(lines_marcs_model[current_line].strip().split()[1])
        self.y_convection = float(lines_marcs_model[current_line].strip().split()[2])
        self.beta_convection = float(lines_marcs_model[current_line].strip().split()[3])
        current_line += 1
        # eleventh line is are X, Y and Z, 12C/13C=89 (=solar) with junk at the end
        self.x = float(lines_marcs_model[current_line].strip().split()[0])
        self.y = float(lines_marcs_model[current_line].strip().split()[1])
        self.z = float(lines_marcs_model[current_line].strip().split()[2])
        current_line += 1
        # line 12 is header, skipped
        # next 9 lines are the abundances of the elements, 10 elements per line, 92 elements in total
        # convert to dictionary, where the key is the element name and the value is the abundance
        self.abundances = {}
        current_element_atomic_number = 1
        for i in range(9):
            current_line += 1
            for j in range(10):
                element_name = periodic_table[current_element_atomic_number]
                element_abundance = float(lines_marcs_model[current_line].strip().split()[j])
                self.abundances[element_name] = element_abundance
                current_element_atomic_number += 1
        # last line only has 2 elements
        current_line += 1
        for j in range(2):
            element_name = periodic_table[current_element_atomic_number]
            element_abundance = float(lines_marcs_model[current_line].strip().split()[j])
            self.abundances[element_name] = element_abundance
            current_element_atomic_number += 1
        current_line += 1
        # next line is number of depth points with junk at the end
        self.number_depth_points = int(lines_marcs_model[current_line].strip().split()[0])
        current_line += 3
        # next 2 lines is header, skipped
        # now we have number of depth points lines with  k lgTauR  lgTau5    Depth     T        Pe          Pg         Prad       Pturb
        # use numpy to read in the data, by passing the lines_marcs_model list and skipping the first current_line lines
        data = np.loadtxt(lines_marcs_model[current_line:current_line + self.number_depth_points], skiprows=0)
        self.k = data[:, 0].astype(int)
        self.lgTauR = data[:, 1].astype(float)
        self.lgTau5 = data[:, 2].astype(float)
        self.depth = data[:, 3].astype(float)
        self.temperature = data[:, 4].astype(float)
        self.pe = data[:, 5].astype(float)
        self.pg = data[:, 6].astype(float)
        self.prad = data[:, 7].astype(float)
        self.pturb = data[:, 8].astype(float)
        current_line += self.number_depth_points + 1
        # next line is header, skipped
        # now we have number of depth points lines with   k lgTauR    KappaRoss   Density   Mu      Vconv   Fconv/F      RHOX
        # use numpy to read in the data, by passing the lines_marcs_model list and skipping the first current_line lines
        data = np.loadtxt(lines_marcs_model[current_line:current_line + self.number_depth_points], skiprows=0)
        self.lgTauR = data[:, 1].astype(float)
        self.kappaRoss = data[:, 2].astype(float)
        self.density = data[:, 3].astype(float)
        self.mu = data[:, 4].astype(float)
        self.vconv = data[:, 5].astype(float)
        self.fconv_f = data[:, 6].astype(float)
        self.rhox = data[:, 7].astype(float)
        # next lines are elemental pressures, skipped
        # done

    def plot_temperature(self):
        plt.plot(self.depth, self.temperature)
        plt.gca().invert_yaxis()
        plt.ylabel("Temperature [K]")
        plt.xlabel("Depth [cm]")
        # invert y axis
        plt.gca().invert_yaxis()
        plt.show()

    def interpolate_temperature(self, depth_points: Union[list, np.ndarray, int]) -> tuple[np.ndarray, np.ndarray]:
        # if number_depth_points is an integer, then interpolate to that number of depth points
        # if number_depth_points is a list, then interpolate to those depth points
        if isinstance(depth_points, int):
            depth_new, temperature_new = self._interpolate_temperature_number_depth_points(depth_points)
        elif isinstance(depth_points, list) or isinstance(depth_points, np.ndarray):
            depth_new, temperature_new = self._interpolate_temperature_new_depth(depth_points)
        else:
            raise TypeError("number_depth_points must be int or list")
        return depth_new, temperature_new

    def _interpolate_temperature_number_depth_points(self, number_depth_points: int) -> tuple[np.ndarray, np.ndarray]:
        # down or upsample to the number of depth points. returns the interpolated depth and temperature
        # number_depth_points: number of depth points to interpolate to

        # create new depth points
        depth_new = np.linspace(self.depth[0], self.depth[-1], number_depth_points)
        return self._interpolate_temperature_new_depth(depth_new)

    def _interpolate_temperature_new_depth(self, depth_new: Union[list, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        # down or upsample to the number of depth points. returns the interpolated depth and temperature
        # number_depth_points: number of depth points to interpolate to

        # create new depth points
        if type(depth_new) == list:
            depth_new = np.asarray(depth_new)
        # interpolate temperature
        temperature_new = interp1d(self.depth, self.temperature, kind='linear')(depth_new)
        return depth_new, temperature_new

    def write_m3d_type_model(self, new_file: str, depth_points: Union[list, np.ndarray, int]) -> None:
        # write a model in the format of Multi3D
        # file: file to write to
        # depth_points: number of depth points to interpolate to
        # write header
        with open(new_file, "w") as f:
            f.write(f"equidistant_{self.name}\n")
            f.write(f"{depth_points}\n")
            f.write(f"* Depth_[cm] Temperature_[K] Electron_pressure_[cgs] Density_[cgs] Microturbulence_[km/s]\n")
        # interpolate to new depth points
        depth_new, temperature_new, electron_pressure_new, density_new, vmicro_new = self.interpolate_m3d_type_model(depth_points)
        # write data
        with open(new_file, "a") as f:
            for i in range(depth_points):
                f.write(f"{depth_new[i]:>13.6e} {temperature_new[i]:>8.1f} {electron_pressure_new[i]:>12.4E} {density_new[i]:>12.4E} {vmicro_new[i]:>3.1f}\n")

    def interpolate_m3d_type_model(self, depth_points: Union[list, np.ndarray, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # if number_depth_points is an integer, then interpolate to that number of depth points
        # if number_depth_points is a list, then interpolate to those depth points
        if isinstance(depth_points, int):
            depth_new, temperature_new, electron_pressure_new, density_new, vmicro_new = self._interpolate_m3d_number_depth_points(depth_points)
        elif isinstance(depth_points, list) or isinstance(depth_points, np.ndarray):
            depth_new, temperature_new, electron_pressure_new, density_new, vmicro_new = self._interpolate_m3d_new_depth(depth_points)
        else:
            raise TypeError("number_depth_points must be int or list")
        return depth_new, temperature_new, electron_pressure_new, density_new, vmicro_new

    def _interpolate_m3d_number_depth_points(self, number_depth_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # down or upsample to the number of depth points. returns the interpolated depth and temperature
        # number_depth_points: number of depth points to interpolate to

        # create new depth points
        depth_new = np.linspace(self.depth[0], self.depth[-1], number_depth_points)
        return self._interpolate_m3d_new_depth(depth_new)

    def _interpolate_m3d_new_depth(self, depth_new: Union[list, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # down or upsample to the number of depth points. returns the interpolated depth and temperature
        # number_depth_points: number of depth points to interpolate to

        # create new depth points
        if type(depth_new) == list:
            depth_new = np.asarray(depth_new)
        # interpolate temperature
        temperature_new = self.interpolate_temperature(depth_new)[1]
        # interpolate electron pressure
        electron_pressure_new = interp1d(self.depth, self.pe, kind='linear')(depth_new)
        # interpolate density
        density_new = interp1d(self.depth, self.density, kind='linear')(depth_new)
        # vmic is constant float, converted to array of same length as depth_new
        vmic_new = np.full(len(depth_new), self.vmicro)
        return depth_new, temperature_new, electron_pressure_new, density_new, vmic_new


if __name__ == '__main__':
    # example usage
    file = "/Users/storm/docker_common_folder/TSFitPy/input_files/model_atmospheres/1D/p5750_g+4.5_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod"
    file = "/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/input_multi3d/atmos/p5777_g+4.4_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod"
    marcs_model = MARCSModel(file)
    #marcs_model.depth = marcs_model.radius - marcs_model.depth
    #print(marcs_model.depth)
    #marcs_model.plot_temperature()
    #plt.plot(marcs_model.depth, marcs_model.lgTauR)
    # plot y = x
    #plt.plot(marcs_model.depth, marcs_model.depth)
    #plt.ylabel("lgTauR")
    #plt.xlabel("Depth [cm]")
    #plt.show()

    plt.plot(marcs_model.prad, marcs_model.density)
    plt.ylabel("Density [cgs]")
    plt.xlabel("Pg [cgs]")
    plt.show()

    sunint_file = "/Users/storm/docker_common_folder/TSFitPy/temp_directory_Dec-14-2023-11-52-14__0.06273117145467688/marcs_tef5777.0_g4.40_z0.00_tur1.00.interpol"
    sunint_temperature, sunint_depth = np.loadtxt(sunint_file, unpack=True, usecols=(1,5), dtype=float, skiprows=1, comments="/")
    sunint_depth = marcs_model.radius - sunint_depth

    marcs_50_depth, marcs_50_temperature = np.loadtxt("/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/input_multi3d/atmos/atmos.sun_MARCS_50", unpack=True, usecols=(0,1), dtype=float, skiprows=2)
    #plt.plot(marcs_50_depth, marcs_50_temperature, label="MARCS 50", color="red")
    plt.plot(sunint_depth, sunint_temperature, label="MARCS TS interpolated", color="black")
    #xx, yy = marcs_model.interpolate_temperature(50)
    #plt.plot(xx, yy, label="MARCS 50 interpolated", color="blue", linestyle="--")
    xx2, yy2 = marcs_model.interpolate_temperature(marcs_50_depth)
    xx2, yy2 = marcs_model.interpolate_temperature(marcs_model.depth)
    plt.plot(xx2, yy2, label="MARCS 50 interpolated v2", color="red", linestyle="--")
    plt.legend()
    plt.show()

    marcs_model.write_m3d_type_model("test_marcs50.txt", 50)

