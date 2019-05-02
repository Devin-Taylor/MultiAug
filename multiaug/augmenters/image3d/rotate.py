from typing import Union

import multiaug
from multiaug.augmenters import meta

import numpy as np
import scipy.ndimage


def _generate_bool_sequence(num: int, random_state: int) -> list:
    return [random_state.choice([True, False], 1)[0] for _ in range(num)]

def _merge_bool_sequences(proposed: list, mask: list) -> list:
    return [x and y for x, y in zip(proposed, mask)]

def _validate_angle(angle: Union[int, float]):
    assert (angle >= 0 and angle <= 360), "Angle not within valid range [0, 360], received {}".format(angle)

class Rotate3d(meta.Augmenter):
    def __init__(self, angle: Union[int, float, list], interpolation: str = 'nearest', rotate_x: bool = True, rotate_y: bool = True, rotate_z: bool = True):
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
        self.active_axes = [rotate_x, rotate_y, rotate_z]

        if sum(self.active_axes) == 0:
            raise RuntimeError("All axes have been deactivated, please activate atleast one axes for augmentation to take place")

    def apply_to_batch(self, images: np.ndarray, row_ids: list) -> np.ndarray:

        rotated_images = []
        for image in images[row_ids]:
            rotated_images.append(self.apply(image))

        return np.array(rotated_images)

    def apply(self, image: np.ndarray) -> np.ndarray:
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
