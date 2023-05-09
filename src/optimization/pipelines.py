import numpy as np
import torch
from typing import Type, Optional, Tuple, Union, Callable

from sklearn.metrics import mean_squared_error

from src.utils.base import Source, RegressionModel, Initializer, Selector, Replicator, ConvergenceMeasure
from src.optimization.selectors import NaiveSelector
from src.utils.wrap_acqf import uncurry
from src.models.gaussian_processes import MostLikelyHeteroskedasticGP

from botorch.optim import optimize_acqf
from botorch.models.model import Model
from botorch.acquisition import AcquisitionFunction


class BayesOptPipeline:
    """
    An object that connects different aspects of Bayesian optimization in a pipeline.
    """

    def __init__(
        self,
        initialization: Initializer,
        regression_model: Optional[RegressionModel],
        acquisition: Union[
            Callable[[Model], AcquisitionFunction], Callable[[torch.Tensor, Model], AcquisitionFunction]
        ],
        selector: Type[Selector],
        replicator: Replicator,
        beta: Optional[float] = 0.3,
    ):
        """

        Args:
            initialization (Initializer): The initialization sampler.
            regression_model (Optional[RegressionModel]): The regression model that is used for calculating the
            likelihood. If None is provided, then the optimization process resolves to random sampling using the
            initialization function.
            acquisition (Callable): The Bayesian optimization acquisition function.
            replicator (Type[Replicator]): A replicator strategy.
            selector (Selector): A final selection strategy.
        """
        self.initialization = initialization
        self.regression_model = regression_model
        self.acquisition = acquisition
        self.replicator = replicator
        self.selector = selector
        self.beta = beta
        self.replicated_samples = []
        self.previous_posterior = None
        self.mses = []

    def optimize(
        self,
        source: Source,
        random_sample_size: int,
        informed_sample_size: int,
        convergence_measure: ConvergenceMeasure,
        plot: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ) -> Tuple[Union[float, np.ndarray], int]:
        """
        Perform Bayesian optimization on the ``source`` with ``informed_sample_size`` number of samples.

        Args:
            source (Source): A source where samples can be taken from.
            random_sample_size (int): The number of random samples to take.
            informed_sample_size (int): The number of informed samples to take.
            convergence_measure (ConvergenceMeasure): The convergence measure to use in the selector.
            plot (Optional[bool]): True if the optimization process should be plotted.
            verbose (Optional[bool]): True if intermediate samples should be pasted on standard out.

        Returns:
            Union[float, np.ndarray]: The optimal coordinates found by the Bayesian optimization process.
            int: the number of replications made.
        """
        x_test = torch.tensor(np.linspace(source.get_domain()[0], source.get_domain()[1], 100)).unsqueeze(1)

        # TODO: Calculating the posterior using x-test like this is faulty, but predicting the posterior in multiple
        #  dimensions will result in a cartesian product. Thus, make a new selector.

        intermediate_results = np.zeros((random_sample_size + informed_sample_size, source.dimension))

        # Bayesian optimization - phase 1: Random sampling
        x_train = self.initialization.forward(n_samples=random_sample_size)
        y_train = torch.tensor(source.sample(x=x_train.numpy().squeeze())).unsqueeze(1)
        # Bayesian optimization - phase 2: informed sampling
        if self.regression_model is not None:
            for i in range(informed_sample_size):
                x_sample, y_sample = self._optimization_step(
                    x_train=x_train, y_train=y_train, x_test=x_test, source=source
                )

                # Update training data
                x_train = torch.cat((x_train, x_sample))
                y_train = torch.cat((y_train, y_sample))

                # Get convergence measurements for Gaussian Process
                if isinstance(self.regression_model, MostLikelyHeteroskedasticGP):
                    measure = self.get_measure(model=self.regression_model, convergence_measure=convergence_measure)
                else:
                    measure = 1

                if verbose:
                    print(f"Iteration {i} - suggested candidate for maximum: {x_sample[0, 0]:.2f}.")

                selector = self.selector(
                    beta=self.beta,
                    model=self.regression_model.get_model(),
                    estimated_variance_train=self.regression_model.get_estimated_std(x_train=x_train).cpu().detach(),
                    estimated_variance_test=self.regression_model.get_estimated_std(x_train=x_test).cpu().detach(),
                )
                intermediate_results[random_sample_size + i, ...] = selector.forward(
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    x_replicated=self.replicated_samples,
                    convergence_measure=measure,
                )

        # print(self.regression_model.get_observed_information(x_train=x_train, y_train=y_train))
        # print(np.var(np.array(self.regression_model.length_scales)), intermediate_results[-1, ...])

        if plot:
            self.regression_model.plot(
                x_train=x_train,
                y_train=y_train,
                var_true=source.noise_function,
                maximum=source.objective_function.get_maximum(),
                f=source.objective_function.f,
                domain=source.get_domain(),
                acquisition_function=uncurry(
                    curried_acquisition=self.acquisition, model=self.regression_model.get_model(), x_train=x_train
                ),
            )

        # Use the selector to get a verdict for the random samples
        size = random_sample_size if self.regression_model is not None else random_sample_size + informed_sample_size
        for i in range(size):
            selector = NaiveSelector(beta=self.beta)
            intermediate_results[i, ...] = selector.forward(
                x_train=x_train[: i + 1, ...], y_train=y_train[: i + 1, ...]
            )
        return intermediate_results, len(self.replicated_samples)

    def get_measure(self, model: MostLikelyHeteroskedasticGP, convergence_measure: ConvergenceMeasure):
        measure = 1
        # Length scales
        if convergence_measure == ConvergenceMeasure.LENGTH_SCALE:
            l_s = model.length_scales
            if len(l_s) > 3:  # take at least 3 samples for the mean calculation
                # Calculate the deviation of each length scale of the mean
                devs = np.array([np.square(np.mean(l_s[:i], axis=0) - l_s[i]) for i in range(3, len(l_s))])

                # z-score the last deviation given all other deviations
                z_score = (devs[-1] - np.min(devs)) / (np.max(devs) - np.min(devs))

                # average the z-scored length scales over all dimensions
                avg_z_score = np.mean(z_score)

                # 1.0 if the length scale relatively doesn't deviate from the mean, 0.0 if it does
                measure = 1 - avg_z_score
            else:
                measure = 1

        # MSE
        elif convergence_measure == ConvergenceMeasure.MSE:
            if len(self.mses) > 1:
                z_score = (self.mses[-1] - min(self.mses)) / (max(self.mses) - min(self.mses))
                measure = 1 - z_score
            else:
                measure = 1

        # Uncertainty of noise model -> single value \sigma
        elif convergence_measure == convergence_measure.NOISE_UNCERTAINTY:
            noise_uncertainty = model.get_noise_model().likelihood.noise.item()
            measure = 1 / noise_uncertainty

        # Uncertainty of the Most likely Heteroskedastic GP
        # -> \sigma(x), tensor of values of shape x_train.shape[0]
        elif convergence_measure == convergence_measure.MODEL_UNCERTAINTY:
            model_uncertainty = model.get_model().likelihood.noise.cpu().detach().numpy()
            measure = 1 / model_uncertainty

        return measure

    def _optimization_step(
        self, x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor, source: Source,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A Bayesian optimization, optimization step. Use the acquisition function to select a new sample, based on the
        fitted regression model.

        Args:
            x_train (torch.Tensor): A `batch_shape x n x d` tensor of training features.
            y_train (torch.Tensor): A `batch_shape x n x m` tensor of training observations.
            x_test (torch.Tensor): A `batch_shape x n x d` tensor of densely spaced test features.
            source (Source): A source where samples can be taken from.

        Returns:
            torch.Tensor: The x-coordinate of the sample.
            torch.Tensor: The y-coordinate of the sample.
        """
        # Fit surrogate model and update acquisition function
        fitted_model = self.regression_model.fit(x_train, y_train)
        acq_function = uncurry(curried_acquisition=self.acquisition, model=fitted_model, x_train=x_train)

        # Get a new candidate
        bounds = torch.from_numpy(source.get_domain())
        options = {"with_grad": False} if not self.regression_model.with_grad else None
        x_proposed, acq_value = optimize_acqf(
            acq_function=acq_function, bounds=bounds, q=1, num_restarts=5, raw_samples=20, options=options
        )

        # Calculate possible replication
        x_sample = self.replicator.forward(
            x_proposed=x_proposed,
            x_train=x_train,
            y_train=y_train,
            model=self.regression_model,
            estimated_std=self.regression_model.get_estimated_std(x_train=x_test),
        )

        # Store the posterior for MSE calculation
        posterior = fitted_model.posterior(x_train).mean.cpu().detach().numpy().squeeze()
        if self.previous_posterior is not None:
            self.mses.append(mean_squared_error(posterior[:-1], self.previous_posterior))
        self.previous_posterior = posterior

        # Store replicated samples
        if not torch.equal(x_sample, x_proposed):
            self.replicated_samples.append(x_sample)

        y_sample = torch.tensor(source.sample(x=x_sample.numpy()))
        return x_sample, torch.tensor([[y_sample.item()]])

    def calculate_convergence_score(self, score_type: str):
        pass
