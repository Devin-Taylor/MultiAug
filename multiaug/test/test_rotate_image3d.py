import unittest
import multiaug.augmenters as aug
import numpy as np

class TestRotate3d(unittest.TestCase):

    def test_angle_list_three_dimensions(self):
        _ = aug.image3d_augmenters.Rotate3d([0, 1, 2])
        self.assertRaises(AssertionError, aug.image3d_augmenters.Rotate3d, [0, 1])
        self.assertRaises(AssertionError, aug.image3d_augmenters.Rotate3d, [0])

    def test_angle_int_float_valid(self):
        _ = aug.image3d_augmenters.Rotate3d(10)
        _ = aug.image3d_augmenters.Rotate3d(10.5)

    def test_rotation_angle_validation(self):
        aug.image3d_augmenters.rotate._validate_angle(0)
        aug.image3d_augmenters.rotate._validate_angle(360)
        aug.image3d_augmenters.rotate._validate_angle(180)
        self.assertRaises(AssertionError, aug.image3d_augmenters.rotate._validate_angle, -1)
        self.assertRaises(AssertionError, aug.image3d_augmenters.rotate._validate_angle, 361)

    def test_validate_active_axes(self):
        _ = aug.image3d_augmenters.Rotate3d(10, 'nearest', False, False, True)
        _ = aug.image3d_augmenters.Rotate3d(10, 'nearest', False, True, False)
        _ = aug.image3d_augmenters.Rotate3d(10, 'nearest', True, False, False)
        _ = aug.image3d_augmenters.Rotate3d(10, 'nearest', True, True, False)
        _ = aug.image3d_augmenters.Rotate3d(10, 'nearest', True, False, True)
        _ = aug.image3d_augmenters.Rotate3d(10, 'nearest', False, True, True)
        _ = aug.image3d_augmenters.Rotate3d(10, 'nearest', True, True, True)
        self.assertRaises(RuntimeError, aug.image3d_augmenters.Rotate3d, 10, 'nearest', False, False, False)

    def test_merge_bool_sequences(self):
        a = [True, True, True]
        mask = [True, False, True]
        result = [True, False, True]
        self.assertListEqual(result, aug.image3d_augmenters.rotate._merge_bool_sequences(a, mask))
        a = [False, True, True]
        mask = [True, False, True]
        result = [False, False, True]
        self.assertListEqual(result, aug.image3d_augmenters.rotate._merge_bool_sequences(a, mask))

    def test_rotated_image_different(self):
        augmenter = aug.image3d_augmenters.Rotate3d(10, 'nearest', False, False, True)

        data = np.zeros((10, 10, 10))
        data[0, :, 0] = 1
        result = augmenter.apply(data)
        self.assertFalse(np.array_equal(data, result))

    def test_batching_generates_correct_sample_numbers(self):
        augmenter = aug.image3d_augmenters.Rotate3d(10, 'nearest', False, False, True)
        data = np.zeros((10, 10, 10, 10))
        result = augmenter.apply_to_batch(data, [0, 1, 2])
        self.assertEqual(result.shape[0], 3)


if __name__ == '__main__':
    unittest.main()
