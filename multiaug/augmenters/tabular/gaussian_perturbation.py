import multiaug.augmenters
import numpy as np
from multiaug.augmenters import meta


class GaussianPerturbation(meta.Augmenter):
    '''
    Apply featurewise Gaussian noise to each feature in a sample.

    Parameters
    ----------
    method : str
        Method to use to determine the noise (currently variance is the only supported method).

            * If 'variance' then the noise is determined by taking a fraction
              of the variance (across the individual features) for each feature
              and adding it the the original feature.

    fraction : float
        Fraction of noise to add to sample.

    '''
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
        '''
        Apply the augmentation to the samples.

        Parameters
        ----------
        data : np.ndarray
            Entire dataset such that the noise can be determined relative to the entire dataset.

        row_ids : np.ndarray
            The indices of the samples to which to apply noise.

        Returns
        -------
        np.ndarray
            Augmented data
        '''
        new_data = np.zeros((len(row_ids), data.shape[1]))

        noise = self.fraction * self.method(data, axis=0)
        for ii, n in enumerate(noise):
            new_data[:, ii] = self.random_state.normal(0, n, len(row_ids))

        new_data = new_data + data[row_ids, :]

        return new_data
