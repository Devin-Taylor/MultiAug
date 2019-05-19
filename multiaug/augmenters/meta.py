from typing import Union, Tuple
import numpy as np

class Augmenter(object):
    '''
    Base class for all objects that can augment data.

    Parameters
    ----------
    seed : None or int
        random seed for augmenter

            * If None then default seed will be used.
            * If int then random state will be seeded with value provided.
    '''
    def __init__(self, seed: int = None, **kwargs):
        # only support 3D image transforms and tabular transforms at the moment
        self.image3d_transforms = kwargs.get("image3d_transforms")
        self.tabular_transforms = kwargs.get("tabular_transforms")

        if seed is None:
            seed = 1447
        self.random_state = np.random.RandomState(seed)

class OneOf(Augmenter):
    '''
    Augmenter that always executes exactly one of the augmentation methods on each of the modalities provided.

    Parameters
    ----------
    augment : float or list
        Determines portion of dataset that will be used for augmentation

            * If float then that fraction of samples are drawn from the dataset
            * If list then this samples are directly used for augmentation
    '''
    def __init__(self, augment: Union[float, list], **kwargs):
        super(OneOf, self).__init__(augment=augment, **kwargs)

        self.augment = augment
        self.indices = None
        if isinstance(augment, list):
            self.indices = augment
        elif not isinstance(self.augment, float):
            raise AssertionError("Augment parameter of type {} not supported, expected one of [float, list]".format(type(self.augment)))


    def _determine_indices(self, numel: int, replace: bool = True):
        '''
        Selects the indices of the samples to be used for augmentation and sets it to self.indices.

        Parameters
        ----------
        numel : int
            Number of elements in the dataset

        replace : bool
            When sampling points, determines if a sample can be used multiple times or not

                * If True then samples can be repeated
                * If False then samples cannot be repeated
        '''
        if self.indices is None:
            num_indices = int(numel * self.augment)
            self.indices = self.random_state.choice(range(numel), num_indices, replace=replace)


    def apply_image3d(self, images: np.ndarray, labels: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Apply a random transform to a set of images.

        Parameters
        ----------
        images : np.ndarray
            Set of images of dimension N x H x W x D

        labels : list
            Corresponding labels to images. Currently, labels as categorical integers are only supported.

        Returns
        -------
        np.ndarray
            Set of augmented images concatenated on the original images, of shape N x H x W x D

        list
            Corresponding labels to images
        '''
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
            aug_images = self.image3d_transforms[transform].apply(original_images, row_ids)
            if len(aug_images.shape) == 3: # if only single image
                aug_images = np.expand_dims(aug_images, axis=0)
            images = np.concatenate((images, aug_images), axis=0)

            if labels is not None:
                labels = np.concatenate((labels, labels[self.indices])) # NOTE this assumes categorical

        return images, labels

    def apply_tabular(self, data: np.ndarray, labels: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Apply a random transform to a set of images.

        Parameters
        ----------
        data : np.ndarray
            Set of tabular data of dimension N x D

        labels : list
            Corresponding labels to data. Currently, labels as categorical integers are only supported.

        Returns
        -------
        np.ndarray
            Set of augmented tabular data concatenated on the original data, of shape N x D

        list
            Corresponding labels to data
        '''
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
