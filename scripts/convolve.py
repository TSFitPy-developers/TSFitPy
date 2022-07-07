import numpy as np
import os
import subprocess
from astropy import convolution
from astropy import constants as const
from astropy.modeling import Fittable1DModel, Parameter
from scipy import interpolate

def conv_res(wave, flux, resolution):
    d_lam = (np.mean(wave)/resolution)
    sigma = d_lam / (2.0 * np.sqrt(2. * np.log(2.)))
    kernel = convolution.Gaussian1DKernel(sigma/(wave[1]-wave[0]))
    flux_conv = convolution.convolve(flux, kernel, fill_value=1)
    return wave, flux_conv

def conv_macroturbulence(wave, flux, vmac):
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

    return wave_conv, flux_conv

def conv_rotation(wave, flux, vrot):
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
        print(F"Unexpected Vrot={self.Vrot} [km/s]. Stopped.")
        exit(1)

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
