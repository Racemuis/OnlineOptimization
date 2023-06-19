import os
from typing import Optional, List, Any, Callable

import torch
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.base import RegressionModel

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble._base import _partition_estimators
from sklearn.utils.parallel import delayed, Parallel
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestRegressor

from botorch.models.model import Model
from botorch.acquisition import AcquisitionFunction
from botorch.posteriors import Posterior
from botorch.posteriors.torch import TorchPosterior
from botorch.acquisition.objective import PosteriorTransform


def _accumulate_prediction(predict, X, out, lock):
    """
    This is a utility function for joblib's Parallel.
    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, check_input=False)
    with lock:
        if len(out) == 1:
            out[0] = prediction
        else:
            for i in range(len(out)):
                out[i] = prediction[i]


def _accumulate_variance(estimator: DecisionTreeRegressor, X: np.ndarray, out: np.ndarray, lock: threading.Lock):
    """
    This is a utility function for joblib's Parallel.
    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    assert len(out) == 1, f"The output should be a wrapped float, and therefore len(out) == 1. Received {len(out)}"
    # get the leaf indices for each training sample
    leaves_indices = estimator.apply(X)
    Y = estimator.predict(X)

    leaves = np.unique(leaves_indices)
    var_leaf = np.empty(leaves.shape)

    # group the training samples per leaf
    for i, leaf in enumerate(leaves):
        var_leaf[i] = np.var(Y[leaves_indices == leaf])

    var_tree = np.mean(var_leaf)
    with lock:
        out[0] = var_tree


class RandomForestWrapper(RandomForestRegressor, Model, RegressionModel):
    """ A wrapper class that adds an estimation of a posterior distribution to the sklearn RandomForestRegressor"""
    def __init__(
        self,
        n_estimators=100,
        *,  # force named
        criterion="squared_error",
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        num_outputs=1,
        min_variance=0.01,
    ):
        RandomForestRegressor.__init__(
            self,
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )
        Model.__init__(self)
        self._num_outputs = num_outputs
        self.min_variance = min_variance
        self._with_grad = False

    @property
    def with_grad(self) -> bool:
        return self._with_grad

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        """
        Fit the Random Forest Regressor model.

        Args:
            X (np.ndarray): The training input samples.
            y (np.ndarray): The target values.
            sample_weight (Any): The sample weights. If None, then the samples are equally weighted.

        Returns:
            object: The fitted RandomForestWrapper instance.
            object: The fitted RandomForestWrapper instance.
        """
        y = y.ravel()
        return RandomForestRegressor.fit(self, X, y, sample_weight)

    def get_model(self):
        """

        Returns: Return the model instance.

        """
        check_is_fitted(self)
        return self

    def subset_output(self, idcs: List[int]) -> Model:
        pass

    def condition_on_observations(self, X: torch.Tensor, Y: torch.Tensor, **kwargs: Any) -> Model:
        pass

    def get_estimated_std(self, x_train: torch.Tensor) -> torch.Tensor:
        return torch.std(self._get_regression_targets(x_train), dim=-1).unsqueeze(-1)

    def get_posterior_std(self, x_train: torch.Tensor) -> torch.Tensor:
        return self.get_estimated_std(x_train=x_train)

    def posterior(
        self,
        X: torch.Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> Posterior:
        """
        Wraps the posterior distribution of the Random Forest Regressor into a Torch Posterior.

        The posterior distribution of a Random Forest Regression model is estimated using the method proposed in
        [1, p16].

        [1] Frank Hutter, Lin Xu, Holger H. Hoos, Kevin Leyton-Brown, Algorithm runtime prediction:
        Methods & evaluation, Artificial Intelligence, Volume 206, 2014, Pages 79-111, ISSN 0004-3702,
        https://doi.org/10.1016/j.artint.2013.10.003.

        TODO: This function starts two parallel processes that involve the base-estimators of the forest, these
          processes can be wrapped into a single process to speed up the computation.

        Parameters
        ----------
         X: A `b x q x d`-dim Tensor, where `d` is the dimension of the
            feature space, `q` is the number of points considered jointly,
            and `b` is the batch dimension.
        output_indices: A list of indices, corresponding to the outputs over
            which to compute the posterior (if the model is multi-output).
            Can be used to speed up computation if only a subset of the
            model's outputs are required for modules. If omitted,
            computes the posterior over all model outputs.
        observation_noise: If True, add observation noise to the posterior.
        posterior_transform: An optional PosteriorTransform.
        Returns
        -------
            A `Posterior` object, representing a batch of `b` joint distributions
            over `q` points and `m` outputs each.
        """
        regression_targets = self._get_regression_targets(X)

        # calculate the posterior mean
        target_mean = torch.mean(regression_targets, dim=-1).unsqueeze(0)

        # calculate the posterior variance
        variance_within_predictions = self._get_empirical_variance(X)
        # decomposed law of total variance
        variance_vector = (torch.mean(
            torch.pow(regression_targets, 2) + torch.pow(variance_within_predictions, 2), dim=-1
        ) - torch.pow(target_mean, 2)).squeeze()

        if variance_vector.shape == torch.Size([]):
            variance_vector = torch.tensor([variance_vector])

        variance_vector = self.bound_variance(variance_vector)
        target_matrix = variance_vector * torch.eye(variance_vector.shape[0])

        mvn = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=target_mean, covariance_matrix=target_matrix,
        )

        return TorchPosterior(distribution=mvn)

    def bound_variance(self, variance_vector: torch.Tensor):
        """
        Set a minimum on the variance to avoid numerical instabilities.

        Args:
            variance_vector (torch.Tensor): The estimated variance that should be bounded.

        Returns:
            torch.Tensor: The bounded estimated variance vector.
        """
        for i, var in enumerate(variance_vector):
            if var < self.min_variance:
                variance_vector[i] = self.min_variance
        return variance_vector

    def _get_empirical_variance(self, X: torch.Tensor):
        """
        Gather the mean variance within each tree in the forest for X. This is the average of the variance per leaf.
        The predicted regression target of an input sample is computed as the

        Parameters
        ----------
        X (np.ndarray): of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        Returns
        -------
        y : ndarray of shape (n_samples, n_estimators,) or (n_samples, n_outputs, n_estimators)
            The predicted regression targets.
        """
        check_is_fitted(self)
        # Check data
        if torch.is_tensor(X):
            X = X.detach().numpy()

        # set the data shape to [n_samples, n_dimensions] i.e., remove the q dimension from botorch
        if len(X.shape) > 2:
            X = X[:, 0, :]

        X = self._validate_X_predict(X)
        X = X.astype(dtype=np.float32)

        # Assign chunks of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # Assign storage for the estimators
        if self.n_outputs_ > 1:
            y_hat = np.zeros((1, self.n_outputs_, self.n_estimators), dtype=np.float64)
        else:
            y_hat = np.zeros((1, self.n_estimators), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_variance)(e, X, y_hat[..., i], lock) for i, e in enumerate(self.estimators_)
        )
        return torch.tensor(y_hat)

    def _get_regression_targets(self, X: torch.Tensor):
        """
        Gather all predicted regression targets of the trees in the forest for X.
        The predicted regression target of an input sample is computed as the mean prediction of all the single
        estimators in the forsest.

        Parameters
        ----------
        X (np.ndarray): of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        Returns
        -------
        y : ndarray of shape (n_samples, n_estimators,) or (n_samples, n_outputs, n_estimators)
            The predicted regression targets.
        """
        check_is_fitted(self)

        # Check data
        if torch.is_tensor(X):
            X = X.detach().numpy()

        # set the data shape to [n_samples, n_dimensions] i.e., remove the q dimension from botorch
        if len(X.shape) > 2:
            X = X[:, 0, :]

        X = self._validate_X_predict(X)
        X = X.astype(dtype=np.float32)

        # Assign chunks of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # Assign storage for the estimators
        if self.n_outputs_ > 1:
            y_hat = np.zeros((X.shape[0], self.n_outputs_, self.n_estimators), dtype=np.float64)
        else:
            y_hat = np.zeros((X.shape[0], self.n_estimators), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict, X, y_hat[..., i], lock) for i, e in enumerate(self.estimators_)
        )
        return torch.tensor(y_hat)

    def plot(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        var_true: Callable,
        f: Callable,
        domain: np.ndarray,
        maximum: float,
        random_sample_size: int,
        informed_sample_size: int,
        participant: Optional[str] = None,
        acquisition_function: Optional[AcquisitionFunction] = None,
    ) -> None:
        """
        Plot the fitted gaussian process, the corresponding training data and the objective function.

        Args:
            x_train (torch.Tensor): A `batch_shape x n x d` tensor of training features.
            y_train (torch.Tensor): A `batch_shape x n x m` tensor of training observations.
            var_true (Callable): The true variance function.
            f (Callable): The objective function.
            domain (np.ndarray): The domain of the objective function.
            maximum (float): The x-coordinate that corresponds to the maximum of the objective function.
            random_sample_size (int): The number of random samples to take.
            informed_sample_size (int): The number of informed samples to take.
            participant (Optional[str]): The identifier of the participant.
            acquisition_function (Optional[AcquisitionFunction]): The acquisition function.

        Returns:
            None
        """
        domain = domain.astype(int)

        # unwrap the domain for easy plotting
        plot_domain = [domain[0][0], domain[1][0]]

        x_test = np.expand_dims(np.linspace(plot_domain[0], plot_domain[1], 1000), 1)
        regression_targets = self._get_regression_targets(torch.tensor(x_test))
        posterior_mean = torch.mean(regression_targets, dim=-1)
        posterior_std = torch.std(regression_targets, dim=-1)
        lower = posterior_mean - 1.96 * posterior_std
        upper = posterior_mean + 1.96 * posterior_std

        fig1, axes = plt.subplots(3, 1)

        ax0 = axes[0]
        ax1 = axes[1]
        ax2 = axes[2]

        ax0.plot(x_train, y_train, "kx", mew=2, label="observed data")
        ax0.plot(x_test, posterior_mean, label="Mean posterior distribution")
        ax0.fill_between(
            x_test.squeeze(), lower, upper, alpha=0.2, label="confidence interval (1.96σ)",
        )
        # ax0.axvline(maximum, color="r", linewidth=0.3, label="global maximum")
        # ax0.plot(x_test, f(x_test), color="black", linestyle="dashed", linewidth=0.6, label="f(x)")
        ax0.set_title(f"Random Forest regression\n{random_sample_size} "
                      f"random samples, {informed_sample_size} informed samples")
        if acquisition_function is not None:
            # plot the acquisition function
            ax0.plot(
                x_test.squeeze(),
                acquisition_function.forward(torch.tensor(x_test).unsqueeze(-1)).detach(),
                label="acquisition function",
                alpha=0.6,
                linewidth=0.5,
            )

        ax0.legend(bbox_to_anchor=(1.01, 0.0), loc="lower left", ncols=1, borderaxespad=0.0)
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")

        ax1.plot(
            x_test, posterior_std,  label="$\sigma(y)$",
        )
        ax1.set_title("estimated observed standard deviation")
        ax1.set_xlabel("x")
        ax1.set_ylabel("$\sigma(y)$")
        ax1.legend(bbox_to_anchor=(1.01, 0.0), loc="lower left", ncols=1, borderaxespad=0.0)

        if participant is None:
            var_true = 0.5 * np.abs(np.sin(10 * x_test))
            ax2.plot(x_test, var_true, label="$\sigma(y)$")
        else:
            results_folder = r'C:\Users\Racemuis\Documents\school\m artificial intelligence\semester ' \
                             r'2\thesis\results\variance_estimation'
            path = os.path.join(results_folder, f'results_{participant}.csv')
            results = pd.read_csv(filepath_or_buffer=path, index_col=0).to_numpy()
            ax2.plot(np.linspace(0, 1, num=results.shape[1]), np.std(results, axis=0), label="$\sigma(y)$")
        ax2.set_title("true standard deviation")
        ax2.set_xlabel("x")
        ax2.set_ylabel("$\sigma(y)$")
        ax2.set_ylabel("$\sigma(y)$")
        ax2.legend(bbox_to_anchor=(1.01, 0.0), loc="lower left", ncols=1, borderaxespad=0.0)
        plt.tight_layout()
        plt.show()
