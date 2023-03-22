import torch
import numpy as np

from scipy.spatial.distance import euclidean
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel

from ..utils.base import Replicator


class SequentialReplicator:
    def __init__(self, horizon: int):
        self.horizon = horizon


class MaxReplicator(Replicator):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x_proposed: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        model: BatchedMultiOutputGPyTorchModel,
        estimated_std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Assess whether the proposed x-value should be chosen, or whether the model should resample an existing x-value.
        TODO: Get the variance around the noise estimation to make an assessment for replication.
        TODO: For flat noise type use a normal GP instead.

        Args:
            x_proposed (torch.Tensor): The proposed parameters.
            x_train (torch.Tensor): The parameters that have already been evaluated.
            y_train (torch.Tensor): The y-values that are associated with the evaluated parameters.
            model (BatchedMultiOutputGPyTorchModel): The regression model that is used during the optimization process.
            estimated_std (torch.Tensor): The standard deviation that is estimated by the regression model.

        Returns:
            The x-value that is decided upon by the replicator. Either the proposed value, or a replication.
        """
        # mean, variance = self._mean_and_variance(X=x_proposed, model=model)
        # expected_y = model.posterior(x_proposed).mean

        if x_proposed.shape[-1] > 1:
            distances = np.array([euclidean(x_proposed.squeeze(), x.squeeze()) for x in x_train])
        else:
            distances = np.array([euclidean(x_proposed[0], x) for x in x_train])

        closest_train_idx = np.argmin(distances)
        y_max_idx = torch.argmax(y_train).item()

        # the proposed sample is close to the sample + noise that maximizes the objective function
        if y_max_idx == closest_train_idx:  # and torch.sqrt(variance).item() < estimated_std.mean().item():
            replicate = x_train[y_max_idx].unsqueeze(0)
            replicate_std = torch.sqrt(model.posterior(X=replicate.detach()).variance)
            proposed_std = torch.sqrt(model.posterior(X=x_proposed).variance)
            var_std = torch.var(torch.sqrt(model.posterior(X=x_train.detach()).variance))
            if replicate_std + var_std > proposed_std:
                return x_train[y_max_idx].unsqueeze(0)
        return x_proposed
