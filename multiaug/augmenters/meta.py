from typing import Union, Tuple
import numpy as np

class Augmenter(object):
    def __init__(self, seed: int = None, **kwargs):
        # only support 3D image transforms and tabular transforms at the moment
        self.image3d_transforms = kwargs.get("image3d_transforms")
        self.tabular_transforms = kwargs.get("tabular_transforms")

        if seed is None:
            seed = 1447
        self.random_state = np.random.RandomState(seed)

class OneOf(Augmenter):
    def __init__(self, augment: Union[float, list], **kwargs):
        super(OneOf, self).__init__(augment=augment, **kwargs)

        self.augment = augment
        self.indices = None
        if isinstance(augment, list):
            self.indices = augment
        elif not isinstance(self.augment, float):
            raise AssertionError("Augment parameter of type {} not supported, expected one of [float, list]".format(type(self.augment)))


    def _determine_indices(self, numel: int, replace: bool = True):
        if self.indices is None:
            num_indices = int(numel * self.augment)
            self.indices = self.random_state.choice(range(numel), num_indices, replace=replace)


    def apply_image3d(self, images: np.ndarray, labels: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        assert self.image3d_transforms is not None, "No transformations available to apply for image3d"
        num_transforms = len(self.image3d_transforms)

        self._determine_indices(len(images))

        tsf_idx = self.random_state.choice(range(num_transforms), len(self.indices))

        tsf_to_sample = {}
        for idx in range(num_transforms):
            if idx not in tsf_to_sample:
                tsf_to_sample[idx] = []
            for row_id, ii in enumerate(tsf_idx):
                if ii == idx:
                    tsf_to_sample[idx].append(self.indices[row_id])

        original_images = images.copy()
        for transform, row_ids in tsf_to_sample.items():
            aug_images = self.image3d_transforms[transform].apply_to_batch(original_images, row_ids)
            if len(aug_images.shape) == 3: # if only single image
                aug_images = np.expand_dims(aug_images, axis=0)
            images = np.concatenate((images, aug_images), axis=0)

            if labels is not None:
                labels = np.concatenate((labels, labels[self.indices])) # NOTE this assumes categorical

        return images, labels

    def apply_tabular(self, data: np.ndarray, labels: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        assert self.tabular_transforms is not None, "No transformations available to apply for tabular"
        num_transforms = len(self.tabular_transforms)

        self._determine_indices(len(data))

        tsf_idx = self.random_state.choice(range(num_transforms), len(self.indices))

        tsf_to_sample = {}
        for idx in range(num_transforms):
            if idx not in tsf_to_sample:
                tsf_to_sample[idx] = []
            for row_id, ii in enumerate(tsf_idx):
                if ii == idx:
                    tsf_to_sample[idx].append(self.indices[row_id])

        original_data = data.copy()
        for transform, row_ids in tsf_to_sample.items():
            aug_data = self.tabular_transforms[transform].apply(original_data, row_ids)
            data = np.concatenate((data, aug_data), axis=0)

            if labels is not None:
                labels = np.concatenate((labels, labels[self.indices])) # NOTE this assumes categorical

        return data, labels
