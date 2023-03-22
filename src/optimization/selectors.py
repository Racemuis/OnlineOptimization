from typing import Union, Optional, List
import torch

from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN


from ..utils.base import Selector
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel


class SimpleSelector(Selector):
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
        super().__init__(model=model, estimated_variance=estimated_variance)

    def forward(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor = None,
        x_replicated: List[torch.Tensor] = None,
    ) -> Union[torch.tensor, float]:
        """
        Yield the most likely parameter values that optimize the black box function.

        Args:
            x_train (torch.Tensor): A `batch_shape x n x d` tensor of training features.
            y_train (torch.Tensor): A `batch_shape x n x m` tensor of training observations.
            x_test (torch.Tensor): A `batch_shape x n x m` linear space over the domain of the objective function.
            x_replicated (List[torch.Tensor]): The replicated samples.


        Returns:
            Union[torch.tensor, float]: The most likely parameter values that optimize the black box function.
        """
        self.model.eval()
        x_posterior = self.model.posterior(x_train).mean
        fitness = y_train.squeeze() * 1 / self.estimated_variance.squeeze() + x_posterior.squeeze()
        return x_train[torch.argmax(fitness)]


class AveragingSelector(Selector):
    def __init__(self, model: BatchedMultiOutputGPyTorchModel, estimated_variance: torch.Tensor):
        """
        Args:
            model (BatchedMultiOutputGPyTorchModel): The fitted model.
            estimated_variance (Tensor): The variance that is estimated by the model.
        """
        super().__init__(model=model, estimated_variance=estimated_variance)

    def forward(
        self, x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor, x_replicated: List[torch.Tensor]
    ) -> Union[torch.tensor, float]:
        """
        Yield the most likely parameter values that optimize the black box function.
        TODO: Add some (un)certainty about the posterior of the regression model, as the optimum of the posterior is
            taken as starting point, currently, it's just taken as-is...

        Args:
            x_train (torch.Tensor): A `batch_shape x n x d` tensor of training features.
            y_train (torch.Tensor): A `batch_shape x n x m` tensor of training observations.
            x_test (torch.Tensor): A `batch_shape x n x m` linear space over the domain of the objective function.
            x_replicated (List[torch.Tensor]): The replicated samples.

        Returns:
            Union[torch.tensor, float]: The most likely parameter values that optimize the black box function.
        """
        # get the maximum from the posterior
        self.model.eval()
        posterior = self.model.posterior(X=x_test).mean
        x_test_max = x_test[torch.argmax(posterior)]

        # get all evaluated samples that are close to x_test_max (use DBSCAN on the x-axis to cluster the samples)
        closest_train_idx = torch.argmin(torch.tensor([euclidean(x, x_test_max[0]) for x in x_train.cpu().detach()]))
        col_stack_train = torch.hstack([x_train, y_train])
        dist_to_closest, _ = torch.sort(
            torch.tensor([euclidean(x, col_stack_train[closest_train_idx]) for x in col_stack_train.cpu().detach()])
        )

        # cluster the evaluated samples using DBSCAN with a flexible distance parameter eps
        eps = dist_to_closest[1].item()
        clusters = DBSCAN(eps=eps, min_samples=1).fit(col_stack_train)
        max_label = clusters.labels_[closest_train_idx]
        clustered_elements = x_train[clusters.labels_ == max_label]

        # weigh the clustered samples by their inverse variance
        clustered_var = self.model.posterior(X=clustered_elements).variance
        weighted_average = torch.sum(clustered_elements.squeeze() / clustered_var.squeeze()) / torch.sum(
            1 / clustered_var
        )
        return weighted_average.unsqueeze(0).cpu().detach()


class NaiveSelector(Selector):
    def __init__(self, model: Optional = None, estimated_variance: Optional = None):
        super().__init__(model=model, estimated_variance=estimated_variance)

    def forward(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor = None,
        x_replicated: List[torch.Tensor] = None,
    ) -> Union[torch.tensor, float]:
        """
        Yield the most likely parameter values that optimize the black box function.

        Args:
            x_train (torch.Tensor): A `batch_shape x n x d` tensor of training features.
            y_train (torch.Tensor): A `batch_shape x n x m` tensor of training observations.
            x_test (torch.Tensor): A `batch_shape x n x m` linear space over the domain of the objective function.
            x_replicated (List[torch.Tensor]): The replicated samples.

        Returns:
            Union[torch.tensor, float]: The most likely parameter values that optimize the black box function.
        """
        return x_train[torch.argmax(y_train)]
