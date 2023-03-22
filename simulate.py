from typing import Tuple, Type, List, Callable, Optional

import random
import torch
import numpy as np
from tqdm import tqdm

from scipy.spatial.distance import euclidean

from src.utils.utils import curry
from src.models.trees import RandomForestWrapper
from src.optimization.selectors import SimpleSelector
from src.optimization.replicators import MaxReplicator
from src.optimization.pipelines import BayesOptPipeline
from src.optimization.initializers import Random, Sobol
from src.plot_functions.utils import plot_simulation_results
from src.models.gaussian_processes import MostLikelyHeteroskedasticGP
from src.simulation import ObjectiveFunction, Simulator, function_factory
from src.optimization.acquisition_functions import BoundedUpperConfidenceBound
from src.utils.base import Initializer, Selector, RegressionModel, Replicator

from botorch.exceptions.warnings import OptimizationWarning


import warnings

warnings.filterwarnings(
    "ignore", message="Input data is not contained to the unit cube. Please consider min-max scaling the input data."
)
warnings.filterwarnings(
    "ignore",
    message="Input data is not standardized. Please consider scaling the input to zero mean and unit variance.",
)

warnings.filterwarnings("error", category=OptimizationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    """
    Simulate different objective functions to test the Bayesian Optimization pipeline. The objective functions can be
    simulated with different noise patterns. The simulations can be run for different runs to get a more stable estimate
    of the performance of the pipeline.

    Returns:
        None
    """
    # general optimization parameters
    n_random_samples = 8
    n_informed_samples = 15
    n_runs = 15

    # modules
    initialization = Sobol
    regression_models = {
        "Random forest regression": RandomForestWrapper(n_estimators=10, random_state=44),
        "Gaussian process regression": MostLikelyHeteroskedasticGP(normalize=False),
        "Random sampling": None,
    }
    regression_key = "Gaussian process regression"  # "Random forest regression"
    regression_model = regression_models[regression_key]
    acquisition = curry(BoundedUpperConfidenceBound, center=True, beta=0.8)
    selector = SimpleSelector
    replicator = MaxReplicator()

    # create objective_functions
    objective_functions = {
        "Sinusoid and Log": ObjectiveFunction.ObjectiveFunction(
            name="Sinusoid and Log", f=function_factory.sinusoid_and_log, domain=np.array([[0], [10]], dtype=float)
        ),
        "Branin": function_factory.BraninFunction(),
    }
    objective_key = "Sinusoid and Log"
    objective_function = objective_functions[objective_key]

    # create noise functions
    noise_functions = {
        "cosine scale": function_factory.cosine_scale,
        "flat": function_factory.flat,
        "linear": function_factory.linear,
    }

    sample_size = n_informed_samples if regression_model is not None else n_informed_samples + n_random_samples
    distances = np.empty((len(noise_functions), n_runs, sample_size))
    for i, key in enumerate(noise_functions.keys()):
        # reset seed for each new simulation
        np.random.seed(44)
        random.seed(44)
        torch.manual_seed(44)

        simulator = Simulator.Simulator(objective_function=objective_function, noise_function=noise_functions[key])
        results, distance = simulate_batch(
            simulator=simulator,
            initialization=initialization,
            regression_model=regression_model,
            acquisition=acquisition,
            selector=selector,
            replicator=replicator,
            n_runs=n_runs,
            n_informed_samples=n_informed_samples,
            n_random_samples=n_random_samples,
        )
        distances[i, ...] = distance.squeeze()

    # plot results
    plot_simulation_results(distances, n_informed_samples, n_random_samples, n_runs,
                            noise_functions, objective_key, regression_key, sample_size)


def euclidean_distance(batch_results: np.ndarray, optima: List[np.ndarray]):
    """
    Calculate the euclidean distance from the batch results to the closest optimum.

    Args:
        batch_results (np.ndarray): The batch results that have a shape [n_runs, n_informed_samples, data_dim].
        optima (List[np.ndarray]): The optima of the objective function.

    Returns:
        np.ndarray: the euclidean distances stored in an array of shape [n_runs, n_informed_samples].
    """
    if len(optima) == 1:
        return np.abs(batch_results - optima)

    distances_all_optima = np.empty((*batch_results.shape[:-1], len(optima)))
    for i in range(batch_results.shape[0]):
        for j in range(batch_results.shape[1]):
            for k, optimum in enumerate(optima):
                distances_all_optima[i, j, k] = euclidean(batch_results[i, j], optimum)
    return np.min(distances_all_optima, axis=-1)


def simulate_batch(
    simulator: Simulator.Simulator,
    initialization: Type[Initializer],
    regression_model: Optional[RegressionModel],
    acquisition: Callable,
    selector: Type[Selector],
    replicator: Replicator,
    n_runs: int,
    n_informed_samples: int,
    n_random_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make a batch simulation of a Bayesian optimization process. For each informed sample, the optimization process is
    queried to provide the most likely value for the optimum during that time in the process. These guesses are returned
    together with their distance to the true optimum.

    Args:
        simulator (Simulator): The simulator of the unknown optimization function.
        initialization (Initializer): The initialization sampler.
        regression_model (RegressionModel): The regression model that is used for calculating the likelihood.
        acquisition (Callable): The Bayesian optimization acquisition function.
        selector (Selector): A final selection strategy.
        replicator (Replicator): A replicator strategy.
        n_runs (int): The number of runs to simulate.
        n_informed_samples (int): The number of informed samples to take.
        n_random_samples (int): The number of random samples to take.

    Returns:
        batch_results (np.ndarray): The results of the optimization process.
        batch_differences (np.ndarray): The Euclidean distance between the true optimum and the estimated optimum.
    """
    if regression_model is None:
        n_random_samples = n_random_samples + n_informed_samples
        n_informed_samples = 0
        batch_results = np.zeros((n_runs, n_random_samples, simulator.dimension))

    else:
        batch_results = np.zeros((n_runs, n_informed_samples, simulator.dimension))

    for i in tqdm(range(n_runs), desc="Simulating batch"):
        pipe = BayesOptPipeline(
            initialization=initialization(domain=np.array(simulator.get_domain(), dtype=float)),
            regression_model=regression_model,
            acquisition=acquisition,
            selector=selector,
            replicator=replicator,
        )
        batch_results[i, ...] = pipe.optimize(
            source=simulator, random_sample_size=n_random_samples, informed_sample_size=n_informed_samples, plot=False
        )
    batch_differences = euclidean_distance(batch_results, simulator.objective_function.get_maximum())
    return batch_results, batch_differences


if __name__ == "__main__":
    main()
