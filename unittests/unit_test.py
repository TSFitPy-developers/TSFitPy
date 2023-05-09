import os
import shutil
import unittest
from scripts import TSFitPy
import numpy as np
from scripts import turbospectrum_class_nlte
from scripts.create_window_linelist_function import create_window_linelist

class MyTestCase(unittest.TestCase):
    # TODO: more unittests?

    def test_segment_creation(self):
        # tests if the segments are created correctly in TSFitPy.create_segments
        segment_size = 5
        wavelength_start = [4200.0, 4201.0, 4202.0, 4220.0, 4240.0]
        wavelength_end = [4201.0, 4202.0, 4203.0, 4221.0, 4241.0]
        segment_left, segment_right = TSFitPy.create_segment_file(segment_size, wavelength_start, wavelength_end)
        segment_left = list(segment_left)
        segment_right = list(segment_right)

        self.assertListEqual(segment_left, [4195.0, 4215.0, 4235.0])
        self.assertListEqual(segment_right, [4208.0, 4226.0, 4246.0])

        wavelength_start = [4200.0]
        wavelength_end = [4201.0]
        segment_left, segment_right = TSFitPy.create_segment_file(segment_size, wavelength_start, wavelength_end)
        segment_left = list(segment_left)
        segment_right = list(segment_right)

        self.assertListEqual(segment_left, [4195.0])
        self.assertListEqual(segment_right, [4206.0])

    def test_create_window_linst_linelist_function(self):
        def compare_files_ignore_consecutive_spaces(file1, file2):
            with open(file1, 'r') as f1, open(file2, 'r') as f2:
                lines1 = [' '.join(line.split()) for line in f1.readlines()]
                lines2 = [' '.join(line.split()) for line in f2.readlines()]

            if len(lines1) != len(lines2):
                return False

            for line1, line2 in zip(lines1, lines2):
                if line1 != line2:
                    return False

            return True

        lmin, lmax = 4200.0, 4205.0

        # remove directory if it exists
        if os.path.exists("unittests/linelist_testing/test_output/0/"):
            shutil.rmtree("unittests/linelist_testing/test_output/0/")

        create_window_linelist([lmin], [lmax], "unittests/linelist_testing/", "unittests/linelist_testing/test_output/", molecules_flag=True, lbl=True)
        print(os.getcwd())
        self.assertTrue(compare_files_ignore_consecutive_spaces("unittests/linelist_testing/expected_result/atomic_test_data_expected", "unittests/linelist_testing/test_output/0/linelist-0.bsyn"))
        self.assertTrue(compare_files_ignore_consecutive_spaces("unittests/linelist_testing/expected_result/molecular_test_data_expected", "unittests/linelist_testing/test_output/0/linelist-3.bsyn"))
        # get number of files in directory
        num_files = len([f for f in os.listdir("unittests/linelist_testing/test_output/0/") if os.path.isfile(os.path.join("unittests/linelist_testing/test_output/0/", f))])
        self.assertEqual(2, num_files)

        shutil.rmtree("unittests/linelist_testing/test_output/0/")

        # check that molecules are not included
        create_window_linelist([lmin], [lmax], "unittests/linelist_testing/", "unittests/linelist_testing/test_output/",
                               molecules_flag=False, lbl=True)

        num_files = len([f for f in os.listdir("unittests/linelist_testing/test_output/0/") if
                         os.path.isfile(os.path.join("unittests/linelist_testing/test_output/0/", f))])
        self.assertEqual(1, num_files)

        shutil.rmtree("unittests/linelist_testing/test_output/0/")

    def test_get_simplex_guess(self):
        for i in range(1000):
            initial_guess, bounds = TSFitPy.Spectra.get_simplex_guess(1, -10, 10, -5, 5)
            self.assertTupleEqual((-5, 5), bounds)
            self.assertLess(initial_guess[1], bounds[1])
            self.assertGreater(initial_guess[0], bounds[0])
        for i in range(1000):
            initial_guess, bounds = TSFitPy.Spectra.get_simplex_guess(1, -3, 3, -5, 5)
            self.assertTupleEqual((-5, 5), bounds)
            self.assertLess(initial_guess[1], 3)
            self.assertGreater(initial_guess[0], -3)

    def test_calculate_vturb(self):
        self.assertAlmostEqual(TSFitPy.calculate_vturb(5777, 4.44, 0.0), 1.0, delta=0.1)

    def test_chi_square_lbl(self):
        wave_obs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_obs = np.array([1, 0.9, .8, .9, 1, 1, 1, 1, 1, 1])
        wave_mod_orig = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_mod_orig = np.array([1, 0.9, .8, .9, 1, 1, 1, 1, 1, 1])
        fwhm, macro, rot = 0, 0, 0
        lmax, lmin = 5000, -10
        res = TSFitPy.calculate_lbl_chi_squared(None, wave_obs, flux_obs, wave_mod_orig, flux_mod_orig, fwhm, lmin,
                                                lmax, macro, rot, save_convolved=False)
        self.assertAlmostEqual(0, res)

        wave_obs = np.array([1, 2, 3, 4, 5, ])
        flux_mod_orig = np.array([11.8, 24.67, 39.15, 48.81, 30.57])  # expected
        wave_mod_orig = np.array([1, 2, 3, 4, 5, ])
        flux_obs = np.array([17, 25, 39, 42, 32])  # observed
        res = TSFitPy.calculate_lbl_chi_squared(None, wave_obs, flux_obs, wave_mod_orig, flux_mod_orig, fwhm, lmin,
                                                lmax, macro, rot, save_convolved=False)
        self.assertAlmostEqual(3.314, res, places=3)  # done by hand to check

        wave_obs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_obs = np.array([1, 0.9, .8, .9, 1, 1, 1, 1, 1, 1.1])
        wave_mod_orig = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_mod_orig = np.array([1, 0.9, .8, .9, 1, 1, 1, 1, 1, 1])
        res = TSFitPy.calculate_lbl_chi_squared(None, wave_obs, flux_obs, wave_mod_orig, flux_mod_orig, fwhm, lmin,
                                                lmax, macro, rot, save_convolved=False)
        self.assertAlmostEqual(0.01 / 1.0, res)

        wave_obs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_obs = np.array([1, 0.9, .8, .9, 1, 1, 1, 1, 1, 1.0])
        wave_mod_orig = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_mod_orig = np.array([1, 0.9, .8, .9, 1, 1, 1, 1, 1, 1.1])
        res = TSFitPy.calculate_lbl_chi_squared(None, wave_obs, flux_obs, wave_mod_orig, flux_mod_orig, fwhm, lmin,
                                                lmax, macro, rot, save_convolved=False)
        self.assertAlmostEqual(0.01 / 1.1, res)

        wave_obs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_obs = np.array([1, 0.8, .7, .8, 1.1, 1.2, 0.7, 1.4, 1.2, 1.0])
        wave_mod_orig = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_mod_orig = np.array([1, 0.9, .8, .9, 1, 1, 1, 1, 1, 1.1])
        res = TSFitPy.calculate_lbl_chi_squared(None, wave_obs, flux_obs, wave_mod_orig, flux_mod_orig, fwhm, lmin,
                                                lmax, macro, rot, save_convolved=False)
        self.assertAlmostEqual(0.3838131313131312, res)

        rot = 100000
        wave_obs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_obs = np.array([1, 0.8, .7, .8, 1.1, 1.2, 0.7, 1.4, 1.2, 1.0])
        wave_mod_orig = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_mod_orig = np.array([1, 0.9, .8, .9, 1, 1, 1, 1, 1, 1.1])
        res = TSFitPy.calculate_lbl_chi_squared(None, wave_obs, flux_obs, wave_mod_orig, flux_mod_orig, fwhm, lmin,
                                                lmax, macro, rot, save_convolved=False)
        self.assertAlmostEqual(0.3674679979008318, res)

        rot = 200000
        wave_obs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_obs = np.array([1, 0.8, .7, .8, 1.1, 1.2, 0.7, 1.4, 1.2, 1.0])
        wave_mod_orig = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flux_mod_orig = np.array([1, 0.9, .8, .9, 1, 1, 1, 1, 1, 1.1])
        res = TSFitPy.calculate_lbl_chi_squared(None, wave_obs, flux_obs, wave_mod_orig, flux_mod_orig, fwhm, lmin,
                                                lmax, macro, rot, save_convolved=False)
        self.assertAlmostEqual(0.386082941394771, res)

    def test_load_nlte_files_in_dict(self):
        elems = ["Eu", "Fe"]
        files1 = ["Eu1", "Fe1"]
        files2 = ["Eu2", "Fe2"]
        files3 = ["Eu3", "Fe3"]

        expec_dict1 = {"Eu": "Eu1", "Fe": "Fe1"}
        expec_dict2 = {"Eu": "Eu2", "Fe": "Fe2"}
        expec_dict3 = {"Eu": "Eu3", "Fe": "Fe3"}

        dict1, dict2, dict3 = TSFitPy.load_nlte_files_in_dict(elems, files1, files2, files3, True)
        self.assertDictEqual(expec_dict1, dict1)
        self.assertDictEqual(expec_dict2, dict2)
        self.assertDictEqual(expec_dict3, dict3)

        elems = ["Fe", "Eu"]
        files1 = ["Fe1", "Eu1"]
        files2 = ["Fe2", "Eu2"]
        files3 = ["Fe3", "Eu3"]

        expec_dict1 = {"Fe": "Fe1", "Eu": "Eu1"}
        expec_dict2 = {"Fe": "Fe2", "Eu": "Eu2"}
        expec_dict3 = {"Fe": "Fe3", "Eu": "Eu3"}

        dict1, dict2, dict3 = TSFitPy.load_nlte_files_in_dict(elems, files1, files2, files3, True)
        self.assertDictEqual(expec_dict1, dict1)
        self.assertDictEqual(expec_dict2, dict2)
        self.assertDictEqual(expec_dict3, dict3)

        elems = ["Eu", "Fe5"]
        files1 = ["Eu1", "Fe1", "Fe01"]
        files2 = ["Eu2", "Fe2", "Fe02"]
        files3 = ["Eu3", "Fe3", "Fe03"]

        expec_dict1 = {"Eu": "Eu1", "Fe5": "Fe1", "Fe": "Fe01"}
        expec_dict2 = {"Eu": "Eu2", "Fe5": "Fe2", "Fe": "Fe02"}
        expec_dict3 = {"Eu": "Eu3", "Fe5": "Fe3", "Fe": "Fe03"}

        dict1, dict2, dict3 = TSFitPy.load_nlte_files_in_dict(elems, files1, files2, files3, False)
        self.assertDictEqual(expec_dict1, dict1)
        self.assertDictEqual(expec_dict2, dict2)
        self.assertDictEqual(expec_dict3, dict3)

        elems = ["Fe5", "Eu"]
        files1 = ["Fe1", "Eu1", "Fe01"]
        files2 = ["Fe2", "Eu2", "Fe02"]
        files3 = ["Fe3", "Eu3", "Fe03"]

        expec_dict1 = {"Fe5": "Fe1", "Eu": "Eu1", "Fe": "Fe01"}
        expec_dict2 = {"Fe5": "Fe2", "Eu": "Eu2", "Fe": "Fe02"}
        expec_dict3 = {"Fe5": "Fe3", "Eu": "Eu3", "Fe": "Fe03"}

        dict1, dict2, dict3 = TSFitPy.load_nlte_files_in_dict(elems, files1, files2, files3, False)
        self.assertDictEqual(expec_dict1, dict1)
        self.assertDictEqual(expec_dict2, dict2)
        self.assertDictEqual(expec_dict3, dict3)

        elems = ["Fe5", "Eu"]
        files1 = ["Fe1", "Fe01"]
        files2 = ["Fe2", "Fe02"]
        files3 = ["Fe3", "Fe03"]

        expec_dict1 = {"Fe5": "Fe1", "Eu": "", "Fe": "Fe01"}
        expec_dict2 = {"Fe5": "Fe2", "Eu": "", "Fe": "Fe02"}
        expec_dict3 = {"Fe5": "Fe3", "Eu": "", "Fe": "Fe03"}

        dict1, dict2, dict3 = TSFitPy.load_nlte_files_in_dict(elems, files1, files2, files3, False)
        self.assertDictEqual(expec_dict1, dict1)
        self.assertDictEqual(expec_dict2, dict2)
        self.assertDictEqual(expec_dict3, dict3)

    def test_second_degree(self):
        a, b, c = TSFitPy.get_second_degree_polynomial([1, 2, 3], [1, 4, 9])
        func = lambda t: a * t * t + b * t + c
        self.assertEqual(25, func(5))
        self.assertEqual(25, func(-5))

        a, b, c = TSFitPy.get_second_degree_polynomial([1, 2, 3], [2, 5, 10])
        func = lambda t: a * t * t + b * t + c
        self.assertEqual(26, func(5))
        self.assertEqual(26, func(-5))

    def test_closest_available_value(self):
        self.assertEqual(turbospectrum_class_nlte.closest_available_value(1, [0, 2.5, 5, 10]), 0)
        self.assertEqual(turbospectrum_class_nlte.closest_available_value(5, [0, 2.5, 5, 10]), 5)
        self.assertEqual(turbospectrum_class_nlte.closest_available_value(-10, [0, 2.5, 5, 10]), 0)
        self.assertEqual(turbospectrum_class_nlte.closest_available_value(100, [0, 2.5, 5, 10]), 10)
        self.assertEqual(turbospectrum_class_nlte.closest_available_value(7, [0, 2.5, 5, 10]), 5)
        self.assertEqual(turbospectrum_class_nlte.closest_available_value(8, [0, 2.5, 5, 10]), 10)


if __name__ == '__main__':
    unittest.main()
