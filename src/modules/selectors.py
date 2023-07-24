from typing import Union, Optional, List
import torch
import numpy as np

from scipy.stats import zscore

from ..utils.base import Selector
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel


class SimpleSelector(Selector):
    """
    Simple implementation of a selector that decides what sample to choose as the final output of the Bayesian
    modules process. The simple selection is based on the sampled y_coordinate and the variance that is
    estimated by the model.
    """

    def __init__(
        self, beta: float, model: BatchedMultiOutputGPyTorchModel, estimated_variance_train: torch.Tensor,
    ):
        """
        Args:
            model (BatchedMultiOutputGPyTorchModel): The fitted model.
            estimated_variance_train (Tensor): The variance that is estimated by the model over the evaluated samples.
        """
        super().__init__(
            beta=beta, model=model, estimated_variance_train=estimated_variance_train,
        )

    def forward(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        y_posterior: Optional[torch.Tensor] = None,
        x_replicated: List[torch.Tensor] = None,
        convergence_measure: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[torch.tensor, float]:
        """
        Yield the most likely parameter values that optimize the black box function.

        Args:
            x_train (torch.Tensor): A `batch_shape x n x d` tensor of training features.
            y_train (torch.Tensor): A `batch_shape x n x m` tensor of training observations.
            y_posterior (torch.Tensor): A `batch_shape x n x m` tensor of training observations.
            x_replicated (List[torch.Tensor]): The replicated samples.
            convergence_measure (ConvergenceMeasure): The convergence measure to use in the selector.


        Returns:
            Union[torch.tensor, float]: The most likely parameter values that optimize the black box function.
        """
        self.model.eval()
        x_posterior = self.model.posterior(x_train).mean
        fitness = y_train.squeeze() * 1 / self.estimated_variance_train.squeeze() + x_posterior.squeeze()
        return x_train[torch.argmax(fitness)]


class VarianceSelector(Selector):
    def __init__(
        self, beta: float, model: BatchedMultiOutputGPyTorchModel = None, estimated_variance_train: torch.Tensor = None,
    ):
        super().__init__(
            beta=beta, model=model, estimated_variance_train=estimated_variance_train,
        )

    def forward(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        y_posterior: Optional[torch.Tensor] = None,
        x_replicated: Optional[List[torch.Tensor]] = None,
        convergence_measure: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[torch.tensor, float]:
        """
        Yield the most likely parameter values that optimize the black box function.

        Args:
            x_train (torch.Tensor): A `batch_shape x n x d` tensor of training features.
            y_train (torch.Tensor): A `batch_shape x n x m` tensor of training observations.
            y_posterior (torch.Tensor): A `batch_shape x n x m` tensor of training observations.
            x_replicated (List[torch.Tensor]): The replicated samples.
            convergence_measure (ConvergenceMeasure): The convergence measure to use in the selector.


        Returns:
            Union[torch.tensor, float]: The most likely parameter values that optimize the black box function.
        """
        # Beliefs that come from the observed outcomes of the objective function
        observed_beliefs = (1 - self.beta) * zscore(y_train[:-1].squeeze())

        # Beliefs that are modelled by the surrogate model
        modelled_beliefs = (1 - self.beta) * zscore(y_posterior.squeeze()[:-1]) - self.beta * zscore(
            self.estimated_variance_train.squeeze()[:-1]
        )

        # Combine beliefs (if convergence_measure == 1, then the observation beliefs are not used)
        y_scaled = (1 - convergence_measure) * observed_beliefs + convergence_measure * modelled_beliefs

        return x_train[torch.argmax(y_scaled)]


class NaiveSelector(Selector):
    def __init__(
        self, beta: float, model: BatchedMultiOutputGPyTorchModel = None, estimated_variance_train: torch.Tensor = None,
    ):
        """
        Args:
            model (BatchedMultiOutputGPyTorchModel): The fitted model.
            estimated_variance_train (Tensor): The variance that is estimated by the model over the evaluated samples.
        """
        super().__init__(
            beta=beta, model=model, estimated_variance_train=estimated_variance_train,
        )

    def forward(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        y_posterior: Optional[torch.Tensor] = None,
        x_replicated: List[torch.Tensor] = None,
        convergence_measure: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[torch.tensor, float]:
        """
        Yield the most likely parameter values that optimize the black box function.

        Args:
            x_train (torch.Tensor): A `batch_shape x n x d` tensor of training features.
            y_train (torch.Tensor): A `batch_shape x n x m` tensor of training observations.
            y_posterior (torch.Tensor): A `batch_shape x n x m` tensor of training observations.
            x_replicated (List[torch.Tensor]): The replicated samples.
            convergence_measure (ConvergenceMeasure): The convergence measure to use in the selector.

        Returns:
            Union[torch.tensor, float]: The most likely parameter values that optimize the black box function.
        """
        return x_train[torch.argmax(y_train)]
