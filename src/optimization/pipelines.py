import numpy as np
import torch
from typing import Type, Optional, Tuple, Union, Callable

from src.utils.base import Source, RegressionModel, Initializer, Selector, Replicator
from src.optimization.selectors import NaiveSelector
from src.utils.wrap_acqf import uncurry

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
        self.replicated_samples = []
        self.previous_posterior = None

    def optimize(
        self,
        source: Source,
        random_sample_size: int,
        informed_sample_size: int,
        plot: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ) -> Union[float, np.ndarray]:
        """
        Perform Bayesian optimization on the ``source`` with ``informed_sample_size`` number of samples.

        Args:
            source (Source): A source where samples can be taken from.
            random_sample_size (int): The number of random samples to take.
            informed_sample_size (int): The number of informed samples to take.
            plot (Optional[bool]): True if the optimization process should be plotted.
            verbose (Optional[bool]): True if intermediate samples should be pasted on standard out.

        Returns:
            Union[float, np.ndarray]: The optimal coordinates found by the Bayesian optimization process.
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

                if verbose:
                    print(f"Iteration {i} - suggested candidate for maximum: {x_sample[0, 0]:.2f}.")

                selector = self.selector(
                    model=self.regression_model.get_model(),
                    estimated_variance_train=self.regression_model.get_estimated_std(x_train=x_train).cpu().detach(),
                    estimated_variance_test=self.regression_model.get_estimated_std(x_train=x_test).cpu().detach(),
                )
                intermediate_results[random_sample_size+i, ...] = selector.forward(
                    x_train=x_train, y_train=y_train, x_test=x_test, x_replicated=self.replicated_samples
                )

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
            selector = NaiveSelector()
            intermediate_results[i, ...] = selector.forward(x_train=x_train[:i+1, ...], y_train=y_train[:i+1, ...])

        return intermediate_results

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
        fitted_model = self.regression_model.fit(x_train, y_train)
        acq_function = uncurry(curried_acquisition=self.acquisition, model=fitted_model, x_train=x_train)

        bounds = torch.from_numpy(source.get_domain())

        # Get a new candidate
        options = {"with_grad": False} if not self.regression_model.with_grad else None
        x_proposed, acq_value = optimize_acqf(
            acq_function=acq_function, bounds=bounds, q=1, num_restarts=5, raw_samples=20, options=options
        )

        # Possible replication
        x_sample = self.replicator.forward(
            x_proposed=x_proposed,
            x_train=x_train,
            y_train=y_train,
            model=fitted_model,
            estimated_std=self.regression_model.get_estimated_std(x_train=x_test),
        )

        # Store replicated samples
        if not torch.equal(x_sample, x_proposed):
            self.replicated_samples.append(x_sample)

        posterior = fitted_model.posterior(x_test).mean.cpu().detach().numpy().squeeze()
        if self.previous_posterior is not None:
            pass
            # print(f"MSE: {mean_squared_error(posterior, self.previous_posterior)}")
        self.previous_posterior = posterior

        y_sample = torch.tensor(source.sample(x=x_sample.numpy()))
        return x_sample, torch.tensor([[y_sample.item()]])
