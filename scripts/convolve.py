from __future__ import annotations
import numpy as np
from astropy import constants as const
from numpy.fft import fft, ifft
from scipy.interpolate import interp1d

SPEED_OF_LIGHT_KMS = const.c.to('km/s').value
#CONV_RES_CONST_TWOS = (2.0 * np.sqrt(2. * np.log(2.)))  # just not to call it every time
MIN_RD = 0.1 / SPEED_OF_LIGHT_KMS    # RESAMPLING-DISTANCE

def conv_res(wavelength, flux, resolution):
    """
    Applies convolutions to data sx, sy. Uses gaussian doppler broadedning.
    Give resolution in R for gaussian doppler broadening. Converts to doppler broadedning via v_dop = c / R
    Credits: Richard Hoppe
    """
    # convert resolution to doppler velocity
    velocity_doppler = SPEED_OF_LIGHT_KMS / resolution

    sxx = np.log(wavelength.astype(np.float64))  # original xscale in
    syy = flux.astype(np.float64)

    rd = 0.5 * np.min(np.diff(sxx))
    rd = np.max([rd, MIN_RD])

    npres = ((sxx[-1] - sxx[0]) // rd) + 1
    npresn = npres + npres % 2

    rd = (sxx[-1] - sxx[0]) / (npresn - 1)
    sxn = sxx[0] + np.arange(npresn) * rd
    syn = interp1d(sxx, syy)(sxn)

    px = (np.arange(npresn) - npresn // 2) * rd

    width = velocity_doppler / SPEED_OF_LIGHT_KMS

    py = np.exp(- (1.66511 * px / width) ** 2) / (np.sqrt(np.pi) * width)
    sxn, syn = conv_profile(sxn, syn, px, py)

    xx = np.exp(sxn)
    yy = syn

    return xx, yy


def conv_profile(xx, yy, px, py):
    norm = np.trapz(py, x=px)
    n = len(xx)
    dxn = (xx[-1] - xx[0]) / (n - 1)
    conv = dxn * ifft(fft(yy) * fft(np.roll(py / norm, int(n / 2))))

    return xx, np.real(conv)


def conv_macroturbulence(wavelength, flux, vmac):
    """
    Applies convolutions to data sx, sy.
    Give vmac in km/s for convolution with macroturbulence.
    Credits: Richard Hoppe
    """
    sxx = np.log(wavelength.astype(np.float64)) # original xscale in
    syy = flux.astype(np.float64)

    rd = 0.5 * np.min(np.diff(sxx))
    rd = np.max([rd, MIN_RD])

    npres = ((sxx[-1] - sxx[0]) // rd) + 1
    npresn = npres + npres%2

    rd = (sxx[-1] - sxx[0]) / (npresn-1)
    sxn = sxx[0] + np.arange(npresn) * rd
    syn = interp1d(sxx, syy)(sxn)

    px = (np.arange(npresn) - npresn // 2) * rd

    WAVE_RT_VMAC = np.arange(20) / 10.
    FLUX_RT_VMAC = [1.128, 0.939, 0.773, 0.628, 0.504, 0.399, 0.312, 0.240, 0.182, 0.133,
                    0.101, 0.070, 0.052, 0.037, 0.024, 0.017, 0.012, 0.010, 0.009, 0.007]

    WAVE_RT_VMAC = np.concatenate([-WAVE_RT_VMAC[:0:-1],WAVE_RT_VMAC])
    FLUX_RT_VMAC = np.concatenate([ FLUX_RT_VMAC[:0:-1],FLUX_RT_VMAC])
    zeta_rt1 = vmac / SPEED_OF_LIGHT_KMS * 1.433
    WAVE_RT_VMAC = WAVE_RT_VMAC * zeta_rt1
    FLUX_RT_VMAC = FLUX_RT_VMAC / zeta_rt1
    py = interp1d(WAVE_RT_VMAC, FLUX_RT_VMAC, bounds_error=False, fill_value=0)(px)
    mask = (px < WAVE_RT_VMAC[0]) + (px > WAVE_RT_VMAC[-1])
    py[mask] = 0

    sxn, syn = conv_profile(sxn, syn, px, py)

    xx = np.exp(sxn)
    yy = syn

    return xx, yy


def conv_rotation(wavelength, flux, vrot):
    """
    Applies convolutions to data sx, sy.
    Give vrot in km/s for convolution with a rotational profile.
    Credits: Richard Hoppe
    """
    beta = 1.5
    sxx  = np.log(wavelength.astype(np.float64)) # original xscale in
    syy  =        flux.astype(np.float64)

    rd = 0.5 * np.min(np.diff(sxx))
    rd = np.max([rd, MIN_RD])

    npres = ((sxx[-1] - sxx[0]) // rd) + 1
    npresn = npres + npres%2

    rd  = (sxx[-1] - sxx[0]) / (npresn-1)
    sxn = sxx[0] + np.arange(npresn) * rd
    syn = interp1d(sxx, syy)(sxn)

    px = (np.arange(npresn) - npresn // 2) * rd

    normf = SPEED_OF_LIGHT_KMS / vrot
    xi = normf*px

    xi[abs(xi) > 1] = 1

    py = (2*np.sqrt(1-xi**2) / np.pi + beta*(1-xi**2) / 2) * normf / (1 + 6/9*beta)

    sxn, syn = conv_profile(sxn, syn, px, py)

    xx = np.exp(sxn)
    yy = syn

    return xx, yy
