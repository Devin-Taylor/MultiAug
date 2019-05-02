import numpy as np

import multiaug.augmenters
from multiaug.augmenters import meta


class GaussianPerturbation(meta.Augmenter):
    def __init__(self, method: str = 'variance', fraction: float = 0.1):
        super(GaussianPerturbation, self).__init__()

        if method == "variance":
            self.method = np.std
        else:
            raise NotImplementedError("No implementation for method {}".format(method))

        if fraction < 0 or fraction > 1:
            raise RuntimeError("{} is out of range, fraction must be in range [0, 1]".format(fraction))
        self.fraction = fraction

    def apply(self, data: np.ndarray, row_ids: np.ndarray):
        new_data = np.zeros((len(row_ids), data.shape[1]))

        noise = self.fraction * self.method(data, axis=0)
        for ii, n in enumerate(noise):
            new_data[:, ii] = self.random_state.normal(0, n, len(row_ids))

        new_data = new_data + data[row_ids, :]

        return new_data
