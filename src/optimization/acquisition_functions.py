from typing import Union, Optional

import torch
import numpy as np
from scipy.stats import qmc

from botorch.models.model import Model
from botorch.acquisition.objective import PosteriorTransform
from botorch.acquisition import AcquisitionFunction, AnalyticAcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform


class BoundedUpperConfidenceBound(AnalyticAcquisitionFunction):
    r"""Single-outcome bounded variant of the Upper Confidence Bound (UCB).
    Adapted from the botorch library.

    Analytic upper confidence bound that consists of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `BUCB(x) = (1-beta) * mu(x) + beta * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.
    """
    def __init__(
        self,
        model: Model,
        beta: Union[float, torch.Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        **kwargs,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model (Model): A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta (float): A float regulating the trade-off parameter between mean and covariance. 0<= beta <= 1.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        assert 0 <= beta <= 1, f"beta should be between 0 and 1, received {beta}."
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Upper Confidence Bound values at the
            given design points `X`.
        """
        mean, sigma = self._mean_and_sigma(X)
        return (mean * (1 - self.beta) if self.maximize else -mean * (1 - self.beta)) + self.beta * sigma


class Random(AcquisitionFunction):
    """An acquisition function that generates samples in an uniform manner"""

    def __init__(self, size: int, domain: np.ndarray):
        self.size = size
        self.domain = domain
        self.dimension = self.domain.shape[-1]
        super().__init__(None)

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Implements the random forward function. As the sampling method is based on the uniform distribution, the
        argument 'x' is superfluous (it has been added to implement the `AcquisitionFunction` class).

        Args:
            x (Optional[Tensor]): Not used - default = None.

        Returns:
            Tensor: A tensor of the `self.size` random samples, wrapped in the shape n_batches x n_samples x n_dims
        """
        if self.dimension == 1:
            samples = np.random.uniform(low=self.domain[0], high=self.domain[1], size=self.size)
            return torch.from_numpy(samples).unsqueeze(1)
        else:
            samples = np.empty((self.size, self.dimension))
            for i in range(self.dimension):
                samples[:, i] = np.random.uniform(low=self.domain[0, i], high=self.domain[1, i], size=self.size)
            return torch.from_numpy(samples)


class Sobol(AcquisitionFunction):
    """An acquisition function that generates samples according to a Sobol sequence [1]. Sobol sequences are the key
    to a pseudorandom sampling technique that samples from the search space with a low discrepancy (i.e. in a more or
    less uniform manner). Sobol sequences are only balanced if they are generated for n = 2^m samples.
    Sobol sequences have been used in [2] to sample the initial batch of points.


    [1] I.M Sobol, "On the distribution of points in a cube and the approximate evaluation of integrals",
    USSR Computational Mathematics and Mathematical Physics, Volume 7, Issue 4, 1967, Pages 86-112, ISSN 0041-5553,
    https://doi.org/10.1016/0041-5553(67)90144-9.
    [2] Letham, Benjamin, et al. "Constrained Bayesian optimization with noisy experiments.", 2019, pp. 495-519.
    """

    def __init__(self, size: int, domain: np.ndarray):
        """
        Initialize the Sobol acquisition function.

        Args:
            size (int): The number of samples to draw (recommended to choose a power of 2)
            domain (np.ndarray): The domain where to sample from.
        """
        self.size = size
        self.domain = domain
        self.d = domain.shape[1]
        self.sampler = qmc.Sobol(d=self.d, scramble=True, seed=42)
        super().__init__(None)

    def forward(self, x: torch.Tensor = None) -> torch.Tensor:
        """
        Implements the random forward function. As the sampling method is based on the uniform distribution, the
        argument 'x' is superfluous (it has been added to implement the `AcquisitionFunction` class).

        Args:
            x (Tensor): Not used - default = None.

        Returns:
            Tensor: A tensor of the `self.size` random samples, wrapped in the shape n_batches x n_samples x n_dims
        """
        if self.is_power_of_two(self.size):
            unit_sample = self.sampler.random_base2(m=int(np.log2(self.size)))
        else:
            unit_sample = self.sampler.random(n=int(self.size))

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
