import matplotlib
import pylab as plt
import numpy as np
from reportlab.pdfgen import canvas
from typing import Union
import os
import subprocess
from astropy import convolution
from astropy import constants as const
from astropy.modeling import Fittable1DModel, Parameter
from scipy import interpolate
from scipy import stats

def conv_res(wave, flux, resolution):
    d_lam = (np.mean(wave)/resolution)
    sigma = d_lam / (2.0 * np.sqrt(2. * np.log(2.)))
    kernel = convolution.Gaussian1DKernel(sigma/(wave[1]-wave[0]))
    flux_conv = convolution.convolve(flux, kernel, fill_value=1)
    wave_conv = np.array(wave)
    return wave_conv, flux_conv

def conv_macroturbulence(wave, flux, vmac):
    wave_conv, flux_conv = None, None
    if vmac == 0:
        pass
    elif vmac < 0:
        print(F"Macroturbulence <0: {vmac}. Can only be positive (km/s).")
    elif not np.isnan(vmac):
        spec_deltaV = (wave[1]-wave[0])/np.mean(wave) * const.c.to('km/s').value
        if (spec_deltaV) > vmac:
            print(F"WARNING: resolution of model spectra {spec_deltaV} is less than Vmac={vmac}. No convolution will be done, Vmac = 0.")
            pass
        elif np.max(wave) - np.min(wave) > 5.:
            #if wavelength is too large, divide and conquer into 5 A windows
            num_intervals = int((np.max(wave) - np.min(wave))/5)
            if (np.max(wave) - np.min(wave)) % 5 != 0:
                num_intervals += 1
            i_start = 0
            i_end = 0
            for i in range(num_intervals):
                #i_end = i_start
                while (i_end < len(wave)) and (wave[i_end] < wave[i_start]+5.):
                    i_end += 1

                offset = int(2./(wave[1]-wave[0]))
                if offset % 2 == 1:
                    offset += 1
                if i_start != 0:
                    i_start = i_start - offset

                fwhm = vmac * np.mean(wave[i_start:i_end]) / const.c.to('km/s').value / (wave[1]-wave[0])

                x_size = int(30*fwhm)
                if x_size % 2 == 0:
                    x_size += 1

                wave_conv_interval = wave[i_start:i_end]
                macro_kernel = convolution.Model1DKernel( rad_tang(fwhm), x_size = x_size)
                flux_conv_interval = convolution.convolve(flux[i_start:i_end], macro_kernel, fill_value=1)

                if i == 0:
                    wave_conv = wave_conv_interval
                    flux_conv = flux_conv_interval
                else:
                    wave_conv = np.concatenate((wave_conv[:-int(offset/2)], wave_conv_interval[int(offset/2):]), axis=None)
                    flux_conv = np.concatenate((flux_conv[:-int(offset/2)], flux_conv_interval[int(offset/2):]), axis=None)
                i_start = i_end
        else:
            # FWHM: km/s --> A --> step
            # assumes constant step along the whole wavelength range
            fwhm = vmac * np.mean(wave) / const.c.to('km/s').value / (wave[1]-wave[0])

            # kernel should always have odd size along all axis
            x_size=int(30*fwhm)
            if x_size % 2 == 0:
                x_size += 1
            macro_kernel = convolution.Model1DKernel( rad_tang(fwhm), x_size=x_size)
            flux_conv = convolution.convolve(flux, macro_kernel, fill_value=1)
    else:
            print(F"Unexpected Vmac={vmac} [km/s]. Stopped.")
            exit(1)

    if wave_conv is None:   # otherwise there was a bug that no wave/flux returned with invalid vmac
        wave_conv, flux_conv = wave, flux

    return wave_conv, flux_conv

def conv_rotation(wave, flux, vrot):
    wave_conv, flux_conv = None, None
    if vrot == 0:
        pass
    elif not np.isnan(vrot):
        spec_deltaV = (wave[1]-wave[0])/np.mean(wave) * const.c.to('km/s').value
        if (spec_deltaV) > vrot:
            print(F"WARNING: resolution of model spectra {spec_deltaV} is less than Vrot={vrot}. No convolution will be done")
            pass
        elif np.max(wave) - np.min(wave) > 5.:
            #if wavelength is too large, divide and conquer into 5 A windows
            num_intervals = int((np.max(wave) - np.min(wave))/5)
            if (np.max(wave) - np.min(wave)) % 5 != 0:
                num_intervals += 1
            i_start = 0
            i_end = 0
            for i in range(num_intervals):
                #i_end = i_start
                while (i_end < len(wave)) and (wave[i_end] < wave[i_start]+5.):
                    i_end += 1

                offset = int(2./(wave[1]-wave[0]))
                if offset % 2 == 1:
                    offset += 1
                if i_start != 0:
                    i_start = i_start - offset

                fwhm = vrot * np.mean(wave[i_start:i_end]) / const.c.to('km/s').value / (wave[1]-wave[0])

                x_size = int(30*fwhm)
                if x_size % 2 == 0:
                    x_size += 1

                wave_conv_interval = wave[i_start:i_end]
                rot_kernel = convolution.Model1DKernel( rotation(fwhm), x_size = x_size)
                flux_conv_interval = convolution.convolve(flux[i_start:i_end], rot_kernel, fill_value=1)

                if i == 0:
                    wave_conv = wave_conv_interval
                    flux_conv = flux_conv_interval
                else:
                    wave_conv = np.concatenate((wave_conv[:-int(offset/2)], wave_conv_interval[int(offset/2):]), axis=None)
                    flux_conv = np.concatenate((flux_conv[:-int(offset/2)], flux_conv_interval[int(offset/2):]), axis=None)
                i_start = i_end
        else:
            # FWHM: km/s --> A --> step
            # assumes constant step along the whole wavelength range
            fwhm = vrot * np.mean(wave) / const.c.to('km/s').value / (wave[1]-wave[0])

            #kernel should always have odd size along all axis
            x_size = int(30*fwhm)
            if x_size % 2 == 0:
                x_size += 1

            rot_kernel = convolution.Model1DKernel(rotation(fwhm), x_size = x_size)
            flux_conv = convolution.convolve(flux, rot_kernel, fill_value=1)
    else:
        print(F"Unexpected Vrot={vrot} [km/s]. Stopped.")
        exit(1)

    if wave_conv is None:  # otherwise there was a bug that no wave/flux returned with invalid vmac
        wave_conv, flux_conv = wave, flux

    return wave_conv, flux_conv

class rotation(Fittable1DModel):
    fwhm = Parameter(default=0)
    # FWHM is v*sin i in wavelength units
    @staticmethod
    def evaluate(x, fwhm):
        f = np.zeros(len(x))
        mask = np.where( 1-(x/fwhm)**2  >=0 )
        eps=1.-0.3*x[mask]/5000.
        f[mask] =  2*(1-eps)*np.sqrt(1-(x[mask]/fwhm)**2)+np.pi/2 * eps * (1.-(x[mask]/fwhm)**2) / (np.pi*fwhm*(1-eps/3))
        return f

class rad_tang(Fittable1DModel):
    fwhm = Parameter(default=0)

    @staticmethod
    def evaluate(x, fwhm):
        # Gray, 'Turbulence in stellar atmospheres', 1978
        rtf = [1.128,.939,.773,.628,.504,.399,.312,.240,.182,.133, \
       .101,.070,.052,.037,.024,.017,.012,.010,.009,.007,.006, \
       .005,.004,.004,.003,.003,.002,.002,.002,.002,.001,.001, \
       .001,.001,.001,.001,.000,.000,.000,.000 ]
        rtf_x = np.arange(len(rtf))
        # step of macroturbulence function (rtf)
        delta=fwhm*1.433/10.

        x = np.abs(x /delta)
        rtf_inter = interpolate.interp1d(rtf_x, rtf, fill_value='extrapolate')

        return rtf_inter(x)

def load_data_star(input_location, star_name):
    output = np.loadtxt(input_location, dtype=str)
    star_names = output[:, 0]
    wave_center = output[:, 1].astype(float)
    elem_abund = output[:, 4].astype(float)
    doppler_shift = output[:, 5].astype(float)
    microturb = output[:, 6].astype(float)
    macroturb = output[:, 7].astype(float)
    chi_squared = output[:, 8].astype(float)

    unique_star_names = np.unique(star_names)

    indices = np.where(star_names == star_name)[0]

    return doppler_shift[indices], wave_center[indices], macroturb[indices]


def load_normal_data_lbl(wave, flux, start_ind_line):
    # not convolved data
    diff_wave = np.diff(wave)
    mode = stats.mode(diff_wave, keepdims=True)[0]
    #sep_index = np.where(diff_wave != mode)[0]
    sep_index = np.array([0])
    sep_index = np.append(sep_index, np.where(np.logical_or.reduce((diff_wave > mode * 1.01, diff_wave < mode * 0.99)) == True)[0])

    start_index = sep_index[start_ind_line] + 1
    #end_index = sep_index[start_ind_line + 1]
    if start_ind_line + 1 >= np.size(sep_index):
        end_index = -1
    else:
        end_index = sep_index[start_ind_line + 1]

    wave_new = wave[start_index:end_index]
    flux_new = flux[start_index:end_index]

    return wave_new, flux_new



#matplotlib.use('macosx')

#seg_begins, seg_ends = np.loadtxt('../../../storm/turbospectrumwrapper/27.10.22/Fe/fe-seg.txt', comments = ";", usecols=(0,1), unpack=True)
line_centers, line_begins, line_ends = np.loadtxt('../../../storm/PhD_2022/gaia_eso_mg_y_stuff/fe-lmask_gaiaeso.txt', comments =";", usecols=(0, 1, 2), unpack=True)

file_name = "15303489-1955354.txt"

data = np.loadtxt(f"../../../storm/PhD_2022/gaia_eso_mg_y_stuff/Nov-07-2022-14-19-34_ff_no_mol_macro/output", dtype=str)


star_names = data[:, 0]
lines = data[:, 1].astype(float)

stars_amount = np.size(np.unique(star_names))
lines_amount = np.size(np.unique(lines))

columns_for_each_abund = 5
start_index = 4

abund_amount = int((np.size(data[0, :]) - start_index) / columns_for_each_abund)

abund_ind_loc = np.linspace(start_index, abund_amount * columns_for_each_abund + start_index - columns_for_each_abund, abund_amount).astype(int)
chi_square_ind_loc = np.linspace(start_index - 1 + columns_for_each_abund, abund_amount * columns_for_each_abund + start_index - 1, abund_amount).astype(int)
macroturbs_ind_loc = np.linspace(start_index - 2 + columns_for_each_abund, abund_amount * columns_for_each_abund + start_index - 2, abund_amount).astype(int)
microturbs_ind_loc = np.linspace(start_index - 3 + columns_for_each_abund, abund_amount * columns_for_each_abund + start_index - 3, abund_amount).astype(int)
dopplershift_ind_loc = np.linspace(start_index - 4 + columns_for_each_abund, abund_amount * columns_for_each_abund + start_index - 4, abund_amount).astype(int)


for i in range(lines_amount):
    index_to_plot = np.logical_and.reduce((star_names == file_name, lines.astype(float) == line_centers[i]))
    abundances = data[index_to_plot, abund_ind_loc].astype(float)
    chi_squares = data[index_to_plot, chi_square_ind_loc].astype(float)
    macroturbs = data[index_to_plot, macroturbs_ind_loc].astype(float)
    microturbs = data[index_to_plot, microturbs_ind_loc].astype(float)
    dopplershift = data[index_to_plot, dopplershift_ind_loc].astype(float)

    indices_to_use = np.where(chi_squares < 99)[0]
    if np.size(indices_to_use) != 0:
        abundances = abundances[indices_to_use]
        chi_squares = chi_squares[indices_to_use]
        macroturbs = macroturbs[indices_to_use]
        microturbs = microturbs[indices_to_use]
        dopplershift = dopplershift[indices_to_use]

    plt.scatter(abundances, chi_squares, c=dopplershift, marker='o', linewidths=0.5)
    plt.xlim(np.min(abundances), np.max(abundances))
    plt.ylim(0, np.max(chi_squares))
    cbar = plt.colorbar()
    cbar.set_label('Doppler shift [km/s]')
    #plt.scatter
    #plt.plot([line_begins[i], line_begins[i]], [0, 2], color='green', alpha=0.2)
    #plt.plot([line_ends[i], line_ends[i]], [0, 2], color='green', alpha=0.2)
    #plt.plot([line_centers[i], line_centers[i]], [0, 2], color='grey', alpha=0.35)
    plt.ylabel("Chi squared")
    plt.xlabel("[Fe/H]")
    plt.title(f"{i + 1}: line {line_centers[i]} AA")
    plt.savefig(f"../../../storm/PhD_2022/gaia_eso_mg_y_stuff/figures/{i + 1}_line_{line_centers[i]}_AA_chi_square_ff.png")
    #plt.show()
    plt.close()

"""
name, rvs = np.loadtxt("../../../storm/PhD_2022/gaia_eso_mg_y_stuff/gaia_eso_stars_to_use_full_info.txt", dtype=str,
                     usecols=[0, 32]).transpose()
rv_index = np.where(file_name[:-4] == name)[0][0]
rv = float(rvs[rv_index])


for i in range(len(line_begins)):
    print(i, line_centers[i])

    index_where_line_is_in_output = np.where(wave_centers == line_centers[i])[0]

    doppler = rv + doppler_shifts[index_where_line_is_in_output]
    wave_obs_tmp = wave_obs / (1 + (doppler / 300000.))

    wave_fit_tmp, flux_fit_tmp = load_normal_data_lbl(wave_fit, flux_fit, i)
    wave_fit_tmp, flux_fit_tmp = conv_macroturbulence(wave_fit_tmp, flux_fit_tmp, macroturbs[index_where_line_is_in_output])

    plt.plot(wave_fit_tmp, flux_fit_tmp, color='red')
    plt.scatter(wave_obs_tmp, flux_obs, color='black', marker='o', linewidths=0.5)
    plt.xlim(line_begins[i] - 0.3, line_ends[i] + 0.3)
    plt.ylim(0.2, 1.05)
    plt.plot([line_begins[i], line_begins[i]], [0, 2], color='green', alpha=0.2)
    plt.plot([line_ends[i], line_ends[i]], [0, 2], color='green', alpha=0.2)
    plt.plot([line_centers[i], line_centers[i]], [0, 2], color='grey', alpha=0.35)
    plt.title(f"{i+1}: line {line_centers[i]} AA")
    plt.savefig(f"../../../storm/PhD_2022/gaia_eso_mg_y_stuff/figures/{i+1}_line_{line_centers[i]}_AA.png")
    #plt.show()
    plt.close()"""

def draw_pdf_graphs(canvas_to_use: canvas.Canvas):
    """
    Draws all graphs for the specific star: lightcurves, SED, HR

    :param canvas_to_use: Canvas object
    :param star_obj: Star object
    :param data_all_gaia_stars: All stars' data from FITS file
    """

    init_x_image = 23

    init_y_image_page_n = 480
    image_width = 260
    image_height = int(image_width * 3 / 4)
    image_spacing = 13

    max_i = len(line_begins) + 1

    for i in range(1, max_i, 8):
        left_images_directory = []
        right_images_directory = []
        for j in range(i, i+8, 2):
            if j < max_i:
                left_images_directory.append(f"../../../storm/PhD_2022/gaia_eso_mg_y_stuff/figures/{j}_line_{line_centers[j-1]}_AA_chi_square_ff.png")
            else:
                left_images_directory.append(None)

            if j + 1 < max_i:
                right_images_directory.append(f"../../../storm/PhD_2022/gaia_eso_mg_y_stuff/figures/{j+1}_line_{line_centers[j]}_AA_chi_square_ff.png")
            else:
                right_images_directory.append(None)
        draw_graphs_on_canvas(canvas_to_use, image_height, image_spacing, image_width, init_x_image,
                              init_y_image_page_n,
                              left_images_directory, right_images_directory)
        canvas_to_use.showPage()

def draw_graphs_on_canvas(canvas_to_use, image_height, image_spacing, image_width, init_x_image, init_y_image,
                          left_images_directory, right_images_directory):
    for i in range(len(left_images_directory)):
        draw_canvas_image(canvas_to_use, left_images_directory[i], init_x_image, init_y_image, image_width)
        draw_canvas_image(canvas_to_use, right_images_directory[i], init_x_image + image_width + image_spacing,
                          init_y_image, image_width)
        init_y_image = init_y_image - image_height - image_spacing


def draw_canvas_image(canvas_to_use: canvas.Canvas, image_directory: str, x_image: float, y_image: float,
                      image_width: float):
    """
    Draws image in the canvas, if image directory is given and the image exists. Otherwise prints that no image is given

    :param canvas_to_use: canvas where to draw
    :param image_directory: path to the image
    :param x_image: x coordinate of image top left corner
    :param y_image: y coordinate of image top left corner
    :param image_width: width of image in pixels
    """
    if image_directory is not None:
        canvas_to_use.drawImage(image_directory, x_image, y_image, width=image_width, preserveAspectRatio=True)
        """try:
            canvas_to_use.drawImage(image_directory, x_image, y_image, width=image_width, preserveAspectRatio=True)
        except:
            print(f"No image at {image_directory}")"""


def create_new_canvas_for_a_star(canvas_name):
    """
    Takes in a specific star's data and all stars' data from FITS file and makes one PDF document

    :param star_obj: Star object
    :param data_all_gaia_stars: All data from FITS file (uses specifically BP, RP, parallax, G)
    """

    c = canvas.Canvas(canvas_name)
    c.setLineWidth(.3)
    c.setFont('Helvetica', 12)

    draw_pdf_graphs(c)

    c.save()

create_new_canvas_for_a_star(f"../../../storm/PhD_2022/gaia_eso_mg_y_stuff/{file_name}_chi.pdf")
# TODO: restructure file