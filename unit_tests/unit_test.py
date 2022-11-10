import unittest
from ..scripts import TSFitPy
import numpy as np


class MyTestCase(unittest.TestCase):
    # TODO: more unittests?
    def test_chi_square_lbl(self):
        wave_obs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_obs = np.array([1, 0.9, .8, .9, 1, 1, 1, 1, 1, 1])
        wave_mod_orig = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_mod_orig = np.array([1, 0.9, .8, .9, 1, 1, 1, 1, 1, 1])
        fwhm, macro, rot = 0, 0, 0
        lmax, lmin = 5000, -10
        res = TSFitPy.calculate_lbl_chi_squared(None, wave_obs, flux_obs, wave_mod_orig, flux_mod_orig, fwhm, lmax, lmin, macro, rot)
        self.assertAlmostEqual(0, res)

        wave_obs = np.array([1, 2, 3, 4, 5,])
        flux_mod_orig = np.array([11.8, 24.67, 39.15, 48.81, 30.57])    # expected
        wave_mod_orig = np.array([1, 2, 3, 4, 5,])
        flux_obs = np.array([17, 25, 39, 42, 32])   # observed
        res = TSFitPy.calculate_lbl_chi_squared(None, wave_obs, flux_obs, wave_mod_orig, flux_mod_orig, fwhm, lmax,
                                                lmin, macro, rot)
        self.assertAlmostEqual(3.314, res, places=3)    # done by hand to check

        wave_obs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_obs = np.array([1, 0.9, .8, .9, 1, 1, 1, 1, 1, 1.1])
        wave_mod_orig = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_mod_orig = np.array([1, 0.9, .8, .9, 1, 1, 1, 1, 1, 1])
        res = TSFitPy.calculate_lbl_chi_squared(None, wave_obs, flux_obs, wave_mod_orig, flux_mod_orig, fwhm, lmax,
                                                lmin, macro, rot)
        self.assertAlmostEqual(0.01 / 1.0, res)

        wave_obs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_obs = np.array([1, 0.9, .8, .9, 1, 1, 1, 1, 1, 1.0])
        wave_mod_orig = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_mod_orig = np.array([1, 0.9, .8, .9, 1, 1, 1, 1, 1, 1.1])
        res = TSFitPy.calculate_lbl_chi_squared(None, wave_obs, flux_obs, wave_mod_orig, flux_mod_orig, fwhm, lmax,
                                                lmin, macro, rot)
        self.assertAlmostEqual(0.01 / 1.1, res)

        wave_obs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_obs = np.array([1, 0.8, .7, .8, 1.1, 1.2, 0.7, 1.4, 1.2, 1.0])
        wave_mod_orig = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_mod_orig = np.array([1, 0.9, .8, .9, 1, 1, 1, 1, 1, 1.1])
        res = TSFitPy.calculate_lbl_chi_squared(None, wave_obs, flux_obs, wave_mod_orig, flux_mod_orig, fwhm, lmax,
                                                lmin, macro, rot)
        self.assertAlmostEqual(0.3838131313131312, res)

        rot = 100000
        wave_obs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_obs = np.array([1, 0.8, .7, .8, 1.1, 1.2, 0.7, 1.4, 1.2, 1.0])
        wave_mod_orig = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_mod_orig = np.array([1, 0.9, .8, .9, 1, 1, 1, 1, 1, 1.1])
        res = TSFitPy.calculate_lbl_chi_squared(None, wave_obs, flux_obs, wave_mod_orig, flux_mod_orig, fwhm, lmax,
                                                lmin, macro, rot)
        self.assertAlmostEqual(0.3674679979008318, res)

        rot = 200000
        wave_obs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_obs = np.array([1, 0.8, .7, .8, 1.1, 1.2, 0.7, 1.4, 1.2, 1.0])
        wave_mod_orig = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_mod_orig = np.array([1, 0.9, .8, .9, 1, 1, 1, 1, 1, 1.1])
        res = TSFitPy.calculate_lbl_chi_squared(None, wave_obs, flux_obs, wave_mod_orig, flux_mod_orig, fwhm, lmax,
                                                lmin, macro, rot)
        self.assertAlmostEqual(0.386082941394771, res)


if __name__ == '__main__':
    unittest.main()
