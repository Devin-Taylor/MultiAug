import unittest
import multiaug.augmenters as aug
import numpy as np


class TestGaussianPerturbations(unittest.TestCase):

    def test_not_variance_throws_exception(self):
        self.assertRaises(NotImplementedError, aug.tabular_augmenters.GaussianPerturbation, "test")

    def test_fraction_value_valid(self):
        _ = aug.tabular_augmenters.GaussianPerturbation(fraction=0.)
        _ = aug.tabular_augmenters.GaussianPerturbation(fraction=1.)
        _ = aug.tabular_augmenters.GaussianPerturbation(fraction=0.5)
        self.assertRaises(RuntimeError, aug.tabular_augmenters.GaussianPerturbation, "variance", 1.1)

    def test_adds_correct_number_new_rows(self):
        augmenter = aug.tabular_augmenters.GaussianPerturbation()
        data = np.ones((10, 5))
        new_data = augmenter.apply(data, [0, 1, 3])
        self.assertEqual(new_data.shape[0], 3)

    def test_new_data_different_from_original(self):
        augmenter = aug.tabular_augmenters.GaussianPerturbation()
        data = np.ones((10, 5))
        data[1, :] = [1, 2, 3, 4, 5]
        data[5, :] = [2, 2, 2, 2, 2]
        locs = [0, 3]
        new_data = augmenter.apply(data, locs)
        self.assertFalse(np.array_equal(data[locs, :], new_data))

if __name__ == '__main__':
    unittest.main()
