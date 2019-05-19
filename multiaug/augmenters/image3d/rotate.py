from typing import Union, Tuple

import multiaug
import numpy as np
import scipy.ndimage
from multiaug.augmenters import meta


def _generate_bool_sequence(num: int, random_state: int) -> list:
    return [random_state.choice([True, False], 1)[0] for _ in range(num)]

def _merge_bool_sequences(proposed: list, mask: list) -> list:
    return [x and y for x, y in zip(proposed, mask)]

def _validate_angle(angle: Union[int, float]):
    assert (angle >= 0 and angle <= 360), "Angle not within valid range [0, 360], received {}".format(angle)

class Rotate3d(meta.Augmenter):
    '''
    Apply random rotation to 3D images.

    Parameters
    ----------
    angle : int or float or list
        The maximum angle by which to rotate the image.

            * If 'int' or 'float' then use the angle specified for all axes.
            * If 'list' then must be of format [angle_x, angle_y, angle_z].

    axes : list
        Flags for the x, y, z axes that determine which axes the image can be rotated about.

            * If 'True' then axis can be rotated about
            * If 'False' then axis cannot be rotated about

    interpolation : str
        Interpolation method to use.

            * If 'nearest' then use nearest interpolation.
    '''
    def __init__(self, angle: Union[int, float, list], axes: Tuple[bool, bool, bool] = [True, True, True], interpolation: str = 'nearest'):
        super(Rotate3d, self).__init__()

        if isinstance(angle, list):
            assert len(angle) == 3, "Please specify rotation angle for x, y and z dimensions, only {} angles provided".format(len(angle))
            self.x_max, self.y_max, self.z_max = angle
        elif isinstance(angle, float) or isinstance(angle, int):
            self.x_max = self.y_max = self.z_max = angle
        else:
            raise NotImplementedError("Angle must either be a list of length 3 or a single number, not {}".format(angle))

        _validate_angle(self.x_max)
        _validate_angle(self.y_max)
        _validate_angle(self.z_max)

        self.interpolation = interpolation
        self.active_axes = axes

        if sum(self.active_axes) == 0:
            raise RuntimeError("All axes have been deactivated, please activate atleast one axes for augmentation to take place")

    def apply(self, images: np.ndarray, row_ids: list) -> np.ndarray:
        '''
        Apply transformations to an entire batch.

        Parameters
        ----------
        images : np.ndarray
            Image batch of shape N x H x W x D.

        row_ids : list
            Indices of rows to rotate.

        Returns
        -------
        np.ndarray
            Batch of rotated images.
        '''
        rotated_images = []
        for image in images[row_ids]:
            rotated_images.append(self.apply_to_sample(image))

        return np.array(rotated_images)

    def apply_to_sample(self, image: np.ndarray) -> np.ndarray:
        '''
        Apply transformation to a single image. Randomly samples one or more active axes and applies
        a random rotation about that axes.

        Parameters
        ----------
        image : np.ndarray
            Image to transform.

        Returns
        -------
        np.ndarray
            Rotated image.
        '''
        which_axes = _generate_bool_sequence(3, self.random_state)
        which_axes = _merge_bool_sequences(which_axes, self.active_axes)

        while sum(which_axes) == 0: # at least rotate around one axes
            which_axes = _generate_bool_sequence(3, self.random_state)
            which_axes = _merge_bool_sequences(which_axes, self.active_axes)

        img = image.copy()
        # z-axis
        if which_axes[2]:
            angle = self.random_state.uniform(-self.z_max, self.z_max)
            img = scipy.ndimage.interpolation.rotate(img, angle, mode=self.interpolation, axes=(0, 1), reshape=False)
        # y-axis
        if which_axes[1]:
            angle = self.random_state.uniform(-self.y_max, self.y_max)
            img = scipy.ndimage.interpolation.rotate(img, angle, mode=self.interpolation, axes=(0, 2), reshape=False)
        # x-axis
        if which_axes[0]:
            angle = self.random_state.uniform(-self.x_max, self.x_max)
            img = scipy.ndimage.interpolation.rotate(img, angle, mode=self.interpolation, axes=(1, 2), reshape=False)

        return img
