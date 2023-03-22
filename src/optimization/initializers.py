from typing import Optional, Union, Tuple

import torch
import numpy as np

from scipy.stats import qmc

from ..utils.base import Initializer


class Random(Initializer):
    """An acquisition function that generates samples in an uniform manner"""

    def __init__(self, domain: np.ndarray):
        super().__init__(domain)

    def forward(self, n_samples: Union[int, Tuple]) -> torch.Tensor:
        """
        Implements the random forward function. As the sampling method is based on the uniform distribution.

        Args:
            n_samples (Union[int, Tuple]): The number of samples to draw.

        Returns:
            Tensor: A tensor of the `self.size` random samples, wrapped in the shape n_batches x n_samples x n_dims
        """
        if self.dimension == 1:
            samples = np.random.uniform(low=self.domain[0], high=self.domain[1], size=n_samples)
            return torch.from_numpy(samples).unsqueeze(1)
        else:
            samples = np.empty((n_samples, self.dimension))
            for i in range(self.dimension):
                samples[:, i] = np.random.uniform(low=self.domain[0, i], high=self.domain[1, i], size=n_samples)
            return torch.from_numpy(samples)


class Sobol(Initializer):
    """An acquisition function that generates samples according to a Sobol sequence [1]. Sobol sequences are the key
    to a pseudorandom sampling technique that samples from the search space with a low discrepancy (i.e. in a more or
    less uniform manner). Sobol sequences are only balanced if they are generated for n = 2^m samples.
    Sobol sequences have been used in [2] to sample the initial batch of points.


    [1] I.M Sobol, "On the distribution of points in a cube and the approximate evaluation of integrals",
    USSR Computational Mathematics and Mathematical Physics, Volume 7, Issue 4, 1967, Pages 86-112, ISSN 0041-5553,
    https://doi.org/10.1016/0041-5553(67)90144-9.
    [2] Letham, Benjamin, et al. "Constrained Bayesian optimization with noisy experiments.", 2019, pp. 495-519.
    """

    def __init__(self, domain: np.ndarray):
        """
        Initialize the Sobol acquisition function.

        Args:
            domain (np.ndarray): The domain where to sample from.
        """
        super().__init__(domain=domain)
        self.sampler = qmc.Sobol(d=self.dimension, scramble=True, seed=42)

    def forward(self, n_samples: int) -> torch.Tensor:
        """
        Implements the random forward function. As the sampling method is based on the uniform distribution, the
        argument 'x' is superfluous (it has been added to implement the `AcquisitionFunction` class).

        Args:
            n_samples (Union[int, Tuple]): The number of samples to draw (recommended to choose a power of 2)


        Returns:
            Tensor: A tensor of the `self.size` random samples, wrapped in the shape n_batches x n_samples x n_dims
        """
        if self.is_power_of_two(n_samples):
            unit_sample = self.sampler.random_base2(m=int(np.log2(n_samples)))
        else:
            unit_sample = self.sampler.random(n=int(n_samples))

        sample = qmc.scale(unit_sample, self.domain[0], self.domain[1])
        return torch.tensor(sample)

    @staticmethod
    def is_power_of_two(n: int) -> bool:
        """
        Bit-based operation to query whether integer `n` is a power of 2.
        Based on: https://stackoverflow.com/questions/600293/how-to-check-if-a-number-is-a-power-of-2/600306#600306

        Args:
            n (int): The argument to the function.

        Returns:
            (bool): True if n is a power of 2.
        """
        return (np.bitwise_and(n, (n - 1)) == 0) and n != 0
