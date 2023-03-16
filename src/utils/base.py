from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np
import torch

from botorch.acquisition import AcquisitionFunction


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
