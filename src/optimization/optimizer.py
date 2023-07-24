import warnings
from typing import Optional, List, Tuple

import torch
import numpy as np

from ..modules.models import MostLikelyHeteroskedasticGP
from ..modules.selectors import NaiveSelector
from ..utils import enums, wrap_acqf

from botorch.optim import optimize_acqf
from sklearn.metrics import mean_squared_error


class Optimizer:
    def __init__(
        self,
        n_random_samples: int,
        domain: List[Tuple],
        beta: float,
        initializer: str,
        acquisition: str,
        selector: str,
        regression_model: str,
        replicator: Optional[str] = None,
        convergence_measure: Optional[str] = "None",
        ignore_warnings: bool = True,
    ):
        assert n_random_samples > 0, f"The number of initializing samples should be > 0, received {n_random_samples}."
        # Initialize components
        self.n_random_samples = n_random_samples
        self.domain = np.array(domain, dtype=float).T
        self.initializer = enums.initializers(initializer)(domain=np.array(domain, dtype=float).T)
        self.beta = beta
        # TODO: Make center and beta optional kwargs for the acqf
        self.acquisition = wrap_acqf.curry(enums.acquisition(acquisition), center=True, beta=self.beta)
        self.selector = enums.selectors(selector)
        self.regression_model = enums.regression_models(regression_model)
        self.replicator = None if replicator is None else enums.replicators(replicator)
        self.convergence_measure = enums.ConvergenceMeasure(convergence_measure)

        # Initialize random sampling
        self.random_samples = self.initializer.forward(n_samples=n_random_samples)

        # Initialize dataset
        self.x_train = torch.tensor([])
        self.y_train = torch.tensor([])

        # Keep track of surrogate status
        self.surrogate_fit = False

        # Store convergence aspects
        self.previous_posterior = None
        self.mses = []
        self.variance_function_family = []

        if ignore_warnings:
            self.ignore_warnings()

    def query(self, n_samples: Optional[int] = 1):
        assert n_samples <= self.n_random_samples or self.n_random_samples == 0, (
            "You attempted to take more samples "
            "than the number of random samples "
            "specified beforehand. Try informing "
            "the optimization process first."
        )
        if self.n_random_samples > 0:
            # Get indices
            i_previous = self.random_samples.shape[0] - self.n_random_samples
            i_current = self.random_samples.shape[0] - (self.n_random_samples - n_samples)

            # Update sample size
            self.n_random_samples = self.n_random_samples - n_samples
            return self.random_samples[i_previous:i_current]

        elif self.regression_model is None:
            return self.initializer.forward(n_samples=n_samples)

        else:
            # Check whether the regression model is initialized
            self.check_initialized()

            # Fit surrogate model and update acquisition function
            fitted_model = self.regression_model.fit(self.x_train, self.y_train)
            acq_function = wrap_acqf.uncurry(
                curried_acquisition=self.acquisition, model=self.regression_model, x_train=self.x_train
            )

            # Get a new candidate
            # TODO: Add acqf kwargs here as well
            bounds = torch.from_numpy(self.domain)
            options = {"with_grad": False} if not self.regression_model.with_grad else None
            x_proposed, acq_value = optimize_acqf(
                acq_function=acq_function, bounds=bounds, q=1, num_restarts=5, raw_samples=20, options=options
            )

            # Make replications if needed
            if self.replicator is not None:
                x_proposed = self.replicator.forward(
                    x_proposed=x_proposed, x_train=self.x_train, y_train=self.y_train, model=self.regression_model,
                )

            # Update attributes
            self.surrogate_fit = True

            # Store the posterior for MSE calculation
            if self.convergence_measure == enums.ConvergenceMeasure.MSE:
                posterior = fitted_model.posterior(self.x_train).mean.detach().numpy().squeeze()
                if self.previous_posterior is not None:
                    self.mses.append(mean_squared_error(posterior[:-1], self.previous_posterior))
                self.previous_posterior = posterior
            return x_proposed

    def check_initialized(self):
        if self.x_train.shape[0] == 0:
            raise RuntimeError(
                f"You are trying to query an uninitialized regression model. Please inform the "
                f"model using optimizer.inform(x, y)."
            )

    def inform(self, x: torch.Tensor, y: torch.Tensor):
        self.x_train = torch.cat((self.x_train, x))
        self.y_train = torch.cat((self.y_train, y))

    def select(self):
        # Naively select the training sample with the highest y for the random "regression model"
        if self.regression_model is None or not self.surrogate_fit:
            selector = NaiveSelector(beta=self.beta)
            return selector.forward(x_train=self.x_train, y_train=self.y_train)

        # Select samples based on posterior mean and std
        else:
            # Check whether the regression model is initialized
            self.check_initialized()

            selector = self.selector(
                beta=self.beta,
                model=self.regression_model.get_model(),
                estimated_variance_train=self.regression_model.get_posterior_std(x_train=self.x_train).detach(),
            )

            return selector.forward(
                x_train=self.x_train,
                y_train=self.y_train,
                y_posterior=self.regression_model.posterior(self.x_train).mean.detach(),
                x_replicated=None,
                convergence_measure=self.get_measure(
                    model=self.regression_model, convergence_measure=self.convergence_measure
                ),
            )

    def get_measure(self, model: MostLikelyHeteroskedasticGP, convergence_measure: enums.ConvergenceMeasure):
        """
        Calculate the convergence measure of the model.

        Args:
            model (MostLikelyHeteroskedasticGP): A trained MostLikelyHeteroskedasticGP instance.
            convergence_measure (ConvergenceMeasure): The selected convergence measure

        Returns (float): The assessment of convergence according to the selected measure
                (in [0, 1], where 1 indicates that the model is likely to be trustworthy.)

        """
        measure = 1
        # Length scales
        if convergence_measure == enums.ConvergenceMeasure.LENGTH_SCALE:
            l_s = model.length_scales
            if len(l_s) > 3:  # take at least 3 samples for the mean calculation

                # Calculate the deviation of each length scale of the mean
                devs = np.array(
                    [np.sqrt(np.sum(np.square(np.mean(l_s[:i], axis=0) - l_s[i])) / i) for i in range(3, len(l_s))]
                )

                # z-score the last deviation given all other deviations
                z_score = (devs[-1] - np.min(devs)) / (np.max(devs) - np.min(devs))

                # average the z-scored length scales over all dimensions
                avg_z_score = np.mean(z_score)

                # measure = 1.0 if the length scale relatively doesn't deviate from the mean, 0.0 if it does
                measure = 1 - avg_z_score
            else:
                measure = 1

        # MSE
        elif convergence_measure == enums.ConvergenceMeasure.MSE:
            if len(self.mses) > 1:
                z_score = (self.mses[-1] - min(self.mses)) / (max(self.mses) - min(self.mses))
                measure = 1 - z_score
            else:
                measure = 1

        return measure

    @staticmethod
    def ignore_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Input data is not contained to the unit cube. Please consider min-max scaling the input data."
        )
        warnings.filterwarnings(
            "ignore",
            message="Input data is not standardized. Please consider scaling the input to zero mean and unit variance.",
        )
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.simplefilter(action="ignore", category=RuntimeWarning)
