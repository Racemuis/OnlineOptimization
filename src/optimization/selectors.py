from typing import Union, Optional, List
import torch
import numpy as np

from scipy.spatial.distance import euclidean
from scipy.stats import zscore
from sklearn.cluster import DBSCAN


from ..utils.base import Selector
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel


class SimpleSelector(Selector):
    """
    Simple implementation of a selector that decides what sample to choose as the final output of the Bayesian
    optimization process. The simple selection is based on the sampled y_coordinate and the variance that is
    estimated by the model.
    """

    def __init__(
        self,
        beta: float,
        model: BatchedMultiOutputGPyTorchModel,
        estimated_variance_train: torch.Tensor,
        estimated_variance_test: torch.Tensor,
    ):
        """
        Args:
            model (BatchedMultiOutputGPyTorchModel): The fitted model.
            estimated_variance_train (Tensor): The variance that is estimated by the model over the evaluated samples.
            estimated_variance_test (Tensor): The variance that is estimated by the model over the hypothetical samples.
        """
        super().__init__(
            beta=beta,
            model=model,
            estimated_variance_train=estimated_variance_train,
            estimated_variance_test=estimated_variance_test,
        )

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
        fitness = y_train.squeeze() * 1 / self.estimated_variance_train.squeeze() + x_posterior.squeeze()
        return x_train[torch.argmax(fitness)]


class AveragingSelector(Selector):
    def __init__(
        self,
        beta: float,
        model: BatchedMultiOutputGPyTorchModel,
        estimated_variance_train: torch.Tensor,
        estimated_variance_test: torch.Tensor,
    ):
        """
        Args:
            model (BatchedMultiOutputGPyTorchModel): The fitted model.
            estimated_variance_train (Tensor): The variance that is estimated by the model over the evaluated samples.
            estimated_variance_test (Tensor): The variance that is estimated by the model over the hypothetical samples.
        """
        super().__init__(
            beta=beta,
            model=model,
            estimated_variance_train=estimated_variance_train,
            estimated_variance_test=estimated_variance_test,
        )

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

        argmax_posterior = torch.argmax(posterior)
        argmax_evaluated = torch.argmax(y_train)

        x_post_opt = x_test[argmax_posterior]
        # x_eval_opt = x_train[argmax_evaluated]
        #
        # x_post_opt_var = self.estimated_variance_test[argmax_posterior]
        # x_eval_opt_var = self.estimated_variance_train[argmax_evaluated]

        # get all evaluated samples that are close to x_post_max (use DBSCAN on the x-axis to cluster the samples)
        closest_train_idx = torch.argmin(torch.tensor([euclidean(x, x_post_opt[0]) for x in x_train.cpu().detach()]))
        # print(f"closest eval x to posterior x: {x_train[closest_train_idx]}")
        col_stack_train = torch.hstack([x_train, y_train])
        dist_to_closest, _ = torch.sort(
            torch.tensor([euclidean(x, x_train[closest_train_idx]) for x in x_train.cpu().detach()])
        )

        # cluster the evaluated samples using DBSCAN with a flexible distance parameter eps
        eps = dist_to_closest[1].item() + 1e-5
        clusters = DBSCAN(eps=eps, min_samples=1).fit(x_train)
        max_label = clusters.labels_[closest_train_idx]
        clustered_elements = x_train[clusters.labels_ == max_label]

        # weigh the clustered samples by their inverse variance
        clustered_var = self.model.posterior(X=clustered_elements).variance.cpu().detach().numpy()
        clustered_var = clustered_var.squeeze()[:, np.newaxis]
        clustered_elements = clustered_elements.numpy()

        weighted_average = np.sum(clustered_elements / clustered_var, axis=0) / np.sum(1 / clustered_var)
        return weighted_average


class VarianceSelector(Selector):
    def __init__(
        self,
        beta: float,
        model: BatchedMultiOutputGPyTorchModel = None,
        estimated_variance_train: torch.Tensor = None,
        estimated_variance_test: torch.Tensor = None,
    ):
        super().__init__(
            beta=beta,
            model=model,
            estimated_variance_train=estimated_variance_train,
            estimated_variance_test=estimated_variance_test,
        )

    def forward(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: Optional[torch.Tensor] = None,
        x_replicated: Optional[List[torch.Tensor]] = None,
    ) -> Union[torch.tensor, float]:
        y_scaled = (1 - self.beta) * zscore(y_train.squeeze()) - self.beta * zscore(
            torch.pow(self.estimated_variance_train.squeeze(), 2)
        )
        return x_train[torch.argmax(y_scaled)]


class NaiveSelector(Selector):
    def __init__(
        self,
        beta: float,
        model: BatchedMultiOutputGPyTorchModel = None,
        estimated_variance_train: torch.Tensor = None,
        estimated_variance_test: torch.Tensor = None,
    ):
        """
        Args:
            model (BatchedMultiOutputGPyTorchModel): The fitted model.
            estimated_variance_train (Tensor): The variance that is estimated by the model over the evaluated samples.
            estimated_variance_test (Tensor): The variance that is estimated by the model over the hypothetical samples.
        """
        super().__init__(
            beta=beta,
            model=model,
            estimated_variance_train=estimated_variance_train,
            estimated_variance_test=estimated_variance_test,
        )

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
