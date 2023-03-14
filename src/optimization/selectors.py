from typing import Union
import torch

from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel


class SimpleSelector:
    """
    Simple implementation of a selector that decides what sample to choose as the final output of the Bayesian
    optimization process. The simple selection is based on the sampled y_coordinate and the variance that is
    estimated by the model.
    """
    def __init__(self, model: BatchedMultiOutputGPyTorchModel, estimated_variance: torch.Tensor):
        """
        Args:
            model (BatchedMultiOutputGPyTorchModel): The fitted model.
            estimated_variance (Tensor): The variance that is estimated by the model.
        """
        self.model = model
        self.estimated_variance = estimated_variance

    def forward(self, x_train: torch.Tensor, y_train: torch.Tensor) -> Union[torch.tensor, float]:
        """
        Yield the most likely parameter values that optimize the black box function.

        Args:
            x_train (torch.Tensor): A `batch_shape x n x d` tensor of training features.
            y_train (torch.Tensor): A `batch_shape x n x m` tensor of training observations.

        Returns:
            Union[torch.tensor, float]: The most likely parameter values that optimize the black box function.
        """
        x_posterior = self.model.posterior(x_train).mean
        fitness = y_train.squeeze() * 1/self.estimated_variance.squeeze() + x_posterior.squeeze()
        return x_train[torch.argmax(fitness)]
