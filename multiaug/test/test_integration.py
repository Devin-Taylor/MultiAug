import unittest
import multiaug.augmenters as aug
import numpy as np

class TestIntegration(unittest.TestCase):

    def test_image3d_modality(self):
        augmenter = aug.OneOf(0.2, image3d_transforms=[aug.image3d_augmenters.Rotate3d(10)])
        data = np.zeros((10, 10, 10, 10))
        data[:, 0, :, 0] = 1
        labels = np.ones(10)
        new_data, new_labels = augmenter.apply_image3d(data, labels)
        self.assertEqual(len(new_labels), 12)
        self.assertEqual(len(new_data), 12)

    def test_tabular_modality(self):
        augmenter = aug.OneOf(0.2, tabular_transforms=[aug.tabular_augmenters.GaussianPerturbation()])
        data = np.zeros((10, 10))
        data[0, :] = 1
        labels = np.ones(10)
        new_data, new_labels = augmenter.apply_tabular(data, labels)
        self.assertEqual(len(new_labels), 12)
        self.assertEqual(len(new_data), 12)

    def test_tabular_image_modalities(self):
        augmenter = aug.OneOf(0.2, image3d_transforms=[aug.image3d_augmenters.Rotate3d(10)],
                              tabular_transforms=[aug.tabular_augmenters.GaussianPerturbation()])
        images = np.zeros((10, 10, 10, 10))
        images[:, 0, :, 0] = 1
        tabular = np.zeros((10, 10))
        tabular[0, :] = 1
        labels = np.ones(10)
        new_images, new_labels = augmenter.apply_image3d(images, labels)
        new_tabular, _ = augmenter.apply_tabular(tabular, labels)
        self.assertEqual(len(new_labels), 12)
        self.assertEqual(len(new_images), 12)
        self.assertEqual(len(new_tabular), 12)

    def test_correct_modalities_selected(self):
        augmenter = aug.OneOf([6], image3d_transforms=[aug.image3d_augmenters.Rotate3d(0)],
                              tabular_transforms=[aug.tabular_augmenters.GaussianPerturbation(fraction=0)])
        images = np.zeros((10, 10, 10, 10))
        images[6, 0, :, 0] = 1
        tabular = np.zeros((10, 10))
        tabular[6, :] = 1
        labels = np.ones(10)
        new_images, new_labels = augmenter.apply_image3d(images, labels)
        new_tabular, _ = augmenter.apply_tabular(tabular, labels)

        self.assertEqual(1000, sum(np.isclose(images[augmenter.indices[0]].flatten(), new_images[-1, :].flatten())))
        self.assertEqual(10, sum(np.isclose(tabular[augmenter.indices[0]].flatten(), new_tabular[-1, :].flatten())))


if __name__ == '__main__':
    unittest.main()
