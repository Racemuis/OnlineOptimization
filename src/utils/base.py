from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union, Tuple, List

import numpy as np
import torch

from botorch.acquisition import AcquisitionFunction
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel


class Source(ABC):
    def __init__(self):
        pass

    @property
    def dimension(self):
        ...

    @property
    @abstractmethod
    def noise_function(self):
        ...

    @property
    @abstractmethod
    def objective_function(self):
        ...

    @abstractmethod
    def sample(self, x: Any, info: bool = False):
        ...

    @abstractmethod
    def get_domain(self) -> np.ndarray:
        ...


class RegressionModel(ABC):
    def __init__(self, input_transform, outcome_transform):
        self.input_transform = input_transform
        self.outcome_transform = outcome_transform

    def get_model(self):
        return self

    @abstractmethod
    def get_estimated_std(self, x_train: torch.Tensor) -> torch.Tensor:
        ...

    @property
    @abstractmethod
    def with_grad(self) -> bool:
        ...

    @abstractmethod
    def fit(self, x_train: Any, y_train: Any):
        ...

    @abstractmethod
    def plot(
        self,
        x_train: Any,
        y_train: Any,
        var_true: Any,
        maximum: float,
        f: Callable,
        domain: np.ndarray,
        acquisition_function: Optional[AcquisitionFunction],
    ):
        ...


class Initializer(ABC):
    def __init__(self, domain: np.ndarray):
        self.domain = domain
        self.dimension = domain.shape[-1]

    @abstractmethod
    def forward(self, n_samples: int):
        ...


class Selector(ABC):
    def __init__(self, model: Optional[BatchedMultiOutputGPyTorchModel], estimated_variance: Optional[torch.Tensor]):
        self.model = model
        self.estimated_variance = estimated_variance

    @abstractmethod
    def forward(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: Optional[torch.Tensor],
        x_replicated: Optional[List[torch.Tensor]],
    ) -> Union[torch.tensor, float]:
        ...


class Replicator(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def forward(
        self,
        x_proposed: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        model: BatchedMultiOutputGPyTorchModel,
        estimated_std: torch.Tensor,
    ):
        ...

    @staticmethod
    def _mean_and_variance(
        X: torch.Tensor, model: BatchedMultiOutputGPyTorchModel, compute_sigma: bool = True, min_var: float = 1e-12,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Computes the first and second moments of the model posterior.
        Retrieved from botorch.analytic.AnalyticAcquisitionFunction

        Args:
            X (torch.Tensor): `batch_shape x q x d`-dim Tensor of model inputs.
            model (BatchedMultiOutputGPyTorchModel): The fitted regression model.
            compute_sigma (bool): Boolean indicating whether to compute the second
                moment (default: True).
            min_var (float): The minimum value the variance is clamped too. Should be positive.

        Returns:
            A tuple of tensors containing the first and second moments of the model
            posterior. Removes the last two dimensions if they have size one. Only
            returns a single tensor of means if compute_sigma is True.
        """
        posterior = model.posterior(X=X)
        mean = posterior.mean.squeeze(-2).squeeze(-1)  # removing redundant dimensions
        if not compute_sigma:
            return mean, None
        sigma = posterior.variance.clamp_min(min_var).sqrt().view(mean.shape)
        return mean, sigma
