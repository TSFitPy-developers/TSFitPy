import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from scripts import marcs_class

modelAtmGrid = {'teff':[], 'logg':[], 'feh':[], 'vturb':[], 'file':[], 'structure':[], 'structure_keys':[], 'mass':[]} # data

marcs_models_to_load = ["p5000_g+4.0_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod",
                        "p5000_g+4.5_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod",
                        "p5000_g+4.0_m0.0_t01_st_z+0.50_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod",
                        "p5000_g+4.5_m0.0_t01_st_z+0.50_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod",
                        "p6000_g+4.0_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod",
                        "p6000_g+4.5_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod",
                        "p6000_g+4.0_m0.0_t01_st_z+0.50_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod",
                        "p6000_g+4.5_m0.0_t01_st_z+0.50_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod"]
all_marcs_models = []
marcs_path = "/Users/storm/docker_common_folder/TSFitPy/input_files/model_atmospheres/1D/"
for marcs_model in marcs_models_to_load:
    one_model = marcs_class.MARCSModel(os.path.join(marcs_path, marcs_model))
    #one_model.interpolate_all_parameters(one_model.number_depth_points)
    all_marcs_models.append(one_model)
    modelAtmGrid['teff'].append(one_model.teff)
    modelAtmGrid['logg'].append(one_model.logg)
    modelAtmGrid['feh'].append(one_model.metallicity)
    modelAtmGrid['vturb'].append(one_model.vmicro)
    modelAtmGrid['file'].append(one_model.file)
    modelAtmGrid['structure'].append(np.vstack((one_model.lgTau5, one_model.temperature, one_model.pe, np.full(one_model.depth.shape, one_model.vmicro))))
    modelAtmGrid['structure_keys'].append(['tau500', 'temp', 'pe', 'vmic'])
    modelAtmGrid['mass'].append(one_model.mass)

    # np.vstack((one_model.teff, one_model.logg, one_model.feh, one_model.vmic, one_model.vturb, one_model.mass, one_model.structure))

#modelAtmGrid = {'teff':[...], 'logg':[...], 'feh':[...], 'vturb':[...], 'file':[...], 'structure':[...], 'structure_keys':[...], 'mass':[...]} # data
#modelAtmGrid = {'teff':[5000, 5000, 5000, 5000, 6000, 6000, 6000, 6000], 'logg':[4.0, 4.0, 4.5, 4.5, 4.0, 4.0, 4.5, 4.5], 'feh':[-1, 1, -1, 1, -1, 1, -1, 1], 'vturb':[1, 1, 1, 1, 1, 1, 1, 1], 'file':[], 'structure':[], 'structure_keys':[], 'mass':[]} # data
interpolCoords = ['teff', 'logg', 'feh'] #, 'vmic'

# convert all to numpy arrays
for k in modelAtmGrid:
    modelAtmGrid[k] = np.asarray(modelAtmGrid[k])

points = []
norm_coord = {}
interpolator = {}
for k in interpolCoords:
    points.append(modelAtmGrid[k] )#/ max(modelAtmGrid[k]) )
    norm_coord.update( { k :  max(modelAtmGrid[k])} )
points = np.array(points).T
values = np.array(modelAtmGrid['structure'])
interp_f = LinearNDInterpolator(points, values)

tau500_new, temp_new, pe_new, vmic_new = interp_f([5777, 4.44, 0])[0]
# plot tau vs temp, pe, vmic in 3 different plots
# together with it plot solar values
solar_model = marcs_class.MARCSModel("/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/input_multi3d/atmos/p5777_g+4.4_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod")
# plot
plt.figure(figsize=(10, 8))
plt.plot(solar_model.temperature, solar_model.lgTau5, label="solar")
plt.plot(temp_new, tau500_new, label="interpolated")
plt.xlabel("temperature")
plt.ylabel("tau500")
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(solar_model.temperature, solar_model.pe, label="solar")
plt.plot(temp_new, pe_new, label="interpolated")
plt.xlabel("temperature")
plt.ylabel("pe")
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(solar_model.temperature, solar_model.vmicro, label="solar")
plt.plot(temp_new, vmic_new, label="interpolated")
plt.xlabel("temperature")
plt.ylabel("vmic")
plt.legend()
plt.show()

# save in a test file
np.savetxt("test.txt", np.array([tau500_new, temp_new, pe_new, vmic_new]).T, fmt='%1.3f')
exit()
interpFunction, normalisedCoord = interp_f, norm_coord

hull = Delaunay(np.array([ modelAtmGrid[k] / normalisedCoord[k] for k in interpolCoords ]).T)

interpolator['modelAtm'] = {'interpFunction' : interpFunction, \
                                'normCoord' : normalisedCoord, \
                                'hull': hull}
def in_hull(p, hull):
    return hull.find_simplex(p) >= 0

inputParams = {}

inputParams.update({'modelAtmInterpol' : np.full(inputParams['count'], None) })

countOutsideHull = 0
for i in range(inputParams['count']):
    point = [ inputParams[k][i] / interpolator['modelAtm']['normCoord'][k] \
            for k in interpolator['modelAtm']['normCoord'] ]
    if not in_hull(np.array(point).T, interpolator['modelAtm']['hull']):
        countOutsideHull += 1
    else:
        values =  interpolator['modelAtm']['interpFunction'](point)[0]
        inputParams['modelAtmInterpol'][i] = values