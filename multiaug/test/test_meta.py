import unittest
import multiaug.augmenters as aug
import numpy as np

class TestRandomMethods(unittest.TestCase):

    def test_random_state_set(self):
        augmenter = aug.Augmenter()
        test_int = np.random.RandomState(1447).randint(0, 100)
        self.assertEqual(augmenter.random_state.randint(0, 100), test_int)

class TestSampleSelection(unittest.TestCase):

    def test_accepts_lists(self):
        _ = aug.OneOf(augment=[0, 1, 2])

    def test_accepts_floats(self):
        _ = aug.OneOf(augment=0.2)

    def test_does_not_accept_not_list_float(self):
        self.assertRaises(AssertionError, aug.OneOf, 3)

    def test_list_unchanged(self):
        augmenter = aug.OneOf(augment=[0, 1, 2])
        augmenter._determine_indices(10)
        self.assertListEqual(augmenter.indices, [0, 1, 2])

    def test_correct_number_samples(self):
        augmenter = aug.OneOf(augment=0.2)
        augmenter._determine_indices(10)
        self.assertEqual(2, len(augmenter.indices))


if __name__ == '__main__':
    unittest.main()
