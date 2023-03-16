import numpy as np
import torch
from typing import Type, Optional, Tuple, Union, Callable

from src.optimization.selectors import SimpleSelector
from src.utils.base import Source, RegressionModel
from src.utils.utils import uncurry

from botorch.optim import optimize_acqf
from botorch.models.model import Model
from botorch.acquisition import AcquisitionFunction


class BayesOptPipeline:
    """
    An object that connects different aspects of Bayesian optimization in a pipeline.
    """
    def __init__(
        self,
        initialization: AcquisitionFunction,
        regression_model: RegressionModel,
        acquisition: Union[
            Callable[[Model], AcquisitionFunction], Callable[[torch.Tensor, Model], AcquisitionFunction]
        ],
        selector: Type[SimpleSelector],
        replicator: Optional = None,
    ):
        """

        Args:
            initialization (AcquisitionFunction): The initialization sampler.
            regression_model (RegressionModel): The regression model that is used for calculating the likelihood.
            acquisition (Type[AcquisitionFunction]): The Bayesian optimization acquisition function.
            replicator (Optional): A replicator strategy.
            selector (SimpleSelector): A final selection strategy.
        """
        self.initialization = initialization
        self.regression_model = regression_model
        self.acquisition = acquisition
        self.replicator = replicator
        self.selector = selector

    def optimize(
        self, source: Source, informed_sample_size: int, plot: Optional[bool] = False, verbose: Optional[bool] = False,
    ) -> Union[float, np.ndarray]:
        """
        Perform Bayesian optimization on the ``source`` with ``informed_sample_size`` number of samples.

        Args:
            source (Source): A source where samples can be taken from.
            informed_sample_size (int): The number of informed samples to take.
            plot (Optional[bool]): True if the optimization process should be plotted.
            verbose (Optional[bool]): True if intermediate samples should be pasted on standardout.

        Returns:
            Union[float, np.ndarray]: The optimal coordinates found by the Bayesian optimization process.
        """
        intermediate_results = np.empty((informed_sample_size, source.dimension))

        # Bayesian optimization - phase 1: Random sampling
        x_train = self.initialization.forward()
        y_train = torch.tensor(source.sample(x=x_train.numpy().squeeze())).unsqueeze(1)

        # Bayesian optimization - phase 2: informed sampling
        for i in range(informed_sample_size):
            x_sample, y_sample = self._optimization_step(x_train=x_train, y_train=y_train, source=source)

            # Update training data
            x_train = torch.cat((x_train, x_sample))
            y_train = torch.cat((y_train, y_sample))

            if verbose:
                print(f"Iteration {i} - suggested candidate for maximum: {x_sample[0, 0]:.2f}.")

            selector = self.selector(
                model=self.regression_model.get_model(),
                estimated_variance=self.regression_model.get_estimated_std(x_train=x_train).cpu().detach(),
            )
            intermediate_results[i, :] = selector.forward(x_train=x_train, y_train=y_train)

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

        return intermediate_results

    def _optimization_step(
        self, x_train: torch.Tensor, y_train: torch.Tensor, source: Source,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A Bayesian optimization, optimization step. Use the acquisition function to select a new sample, based on the
        fitted regression model.

        Args:
            x_train (torch.Tensor): A `batch_shape x n x d` tensor of training features.
            y_train (torch.Tensor): A `batch_shape x n x m` tensor of training observations.
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
        x_sample, acq_value = optimize_acqf(
            acq_function=acq_function, bounds=bounds, q=1, num_restarts=5, raw_samples=20, options=options
        )
        y_sample = torch.tensor(source.sample(x=x_sample.numpy()))
        return x_sample, torch.tensor([[y_sample.item()]])
