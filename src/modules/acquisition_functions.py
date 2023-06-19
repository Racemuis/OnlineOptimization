from typing import Union, Optional

import torch
import numpy as np
from src.utils.base import RegressionModel
from scipy.stats import qmc, zscore

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
        maximize: Optional[bool] = True,
        center: Optional[bool] = True,
        **kwargs,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model (Model): A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta (float): A float regulating the trade-off parameter between mean and covariance. 0<= beta <= 1.
            posterior_transform (PosteriorTransform): If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize (bool): If True, consider the problem a maximization problem.
        """
        assert 0 <= beta <= 1, f"beta should be between 0 and 1, received {beta}."
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.maximize = maximize
        self.center = center

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
        if self.center:
            mean = mean - torch.mean(mean)
            sigma = sigma - torch.mean(sigma)
        return (mean * (1 - self.beta) if self.maximize else -mean * (1 - self.beta)) + self.beta * sigma


class BoundedUpperConfidenceBoundVar(AnalyticAcquisitionFunction):
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
        maximize: Optional[bool] = True,
        center: Optional[bool] = True,
        **kwargs,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model (Model): A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta (float): A float regulating the trade-off parameter between mean and covariance. 0<= beta <= 1.
            posterior_transform (PosteriorTransform): If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize (bool): If True, consider the problem a maximization problem.
        """
        assert 0 <= beta <= 1, f"beta should be between 0 and 1, received {beta}."
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.maximize = maximize
        self.center = center

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
        r_x = self.model.get_estimated_std(X).squeeze()

        if (sigma < r_x).any():
            sigma = r_x
        else:
            sigma = sigma - r_x

        if self.center:
            mean = mean - torch.mean(mean)
            sigma = sigma - torch.mean(sigma)
        return (mean * (1 - self.beta) if self.maximize else -mean * (1 - self.beta)) + self.beta * sigma
