from typing import Tuple, Type, List

import numpy as np
from tqdm import tqdm

from scipy.spatial.distance import euclidean

from src.utils.utils import curry
from src.models.trees import RandomForestWrapper
from src.models.gaussian_processes import MostLikelyHeteroskedasticGP
from src.optimization.pipelines import BayesOptPipeline
from src.optimization.selectors import SimpleSelector
from src.optimization.acquisition_functions import Random, BoundedUpperConfidenceBound
from src.simulation import ObjectiveFunction, Simulator, function_factory

from botorch.acquisition import AcquisitionFunction, AnalyticAcquisitionFunction


import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler

import warnings

warnings.filterwarnings(
    "ignore", message="Input data is not contained to the unit cube. Please consider min-max scaling the input data."
)
warnings.filterwarnings(
    "ignore",
    message="Input data is not standardized. Please consider scaling the input to zero mean and unit variance.",
)


def main():
    """
    Simulate different objective functions to test the Bayesian Optimization pipeline. The objective functions can be
    simulated with different noise patterns. The simulations can be run for different runs to get a more stable estimate
    of the performance of the pipeline.

    Returns:
        None
    """
    # general optimization parameters
    n_random_samples = 5
    n_informed_samples = 20
    n_runs = 3

    # plot parameters
    colour_cycler = matplotlib.rcParams["axes.prop_cycle"]
    linestyle_cycler = cycler("linestyle", ["-", "--", ":", "-."])

    # create objective_functions
    sin_and_log = ObjectiveFunction.ObjectiveFunction(
        name="Sinusoid and Log", f=function_factory.sinusoid_and_log, domain=np.array([[0], [10]], dtype=float)
    )
    branin = function_factory.BraninFunction()
    noise_functions = {
        "cosine scale": function_factory.cosine_scale,
        "flat": function_factory.flat,
        "linear": function_factory.linear,
    }

    for i, (key, c, ls) in enumerate(zip(noise_functions.keys(), colour_cycler, linestyle_cycler)):
        # reset seed for each new simulation
        np.random.seed(44)

        simulator = Simulator.Simulator(objective_function=branin, noise_function=noise_functions[key])
        results, dist = simulate_batch(
            simulator=simulator, n_runs=n_runs, n_informed_samples=n_informed_samples, n_random_samples=n_random_samples
        )

        # plot results
        mean = np.mean(dist, axis=0).squeeze()
        std = np.std(dist, axis=0).squeeze()
        plt.fill_between(range(n_informed_samples), mean - std, mean + std, color=c["color"], alpha=0.2)
        plt.plot(range(n_informed_samples), mean, color=c["color"], label=key, linestyle=ls["linestyle"])

    plt.legend(title="Noise type")
    plt.xlabel("Number of informed samples")
    plt.ylabel("Euclidean distance")
    plt.title(
        f"The average distance between the estimated and true optimum over {n_runs} runs"
        f"\n{n_random_samples} random initialization samples"
    )
    plt.show()


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
    simulator: Simulator.Simulator, n_runs: int, n_informed_samples: int, n_random_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make a batch simulation of a Bayesian optimization process. For each informed sample, the optimization process is
    queried to provide the most likely value for the optimum during that time in the process. These guesses are returned
    together with their distance to the true optimum.

    Args:
        simulator (Simulator): The simulator of the unknown optimization function.
        n_runs (int): The number of runs to simulate.
        n_informed_samples (int): The number of informed samples to take.
        n_random_samples (int): The number of random samples to take.

    Returns:
        batch_results (np.ndarray): The results of the optimization process.
        batch_differences (np.ndarray): The Euclidean distance between the true optimum and the estimated optimum.
    """
    batch_results = np.zeros((n_runs, n_informed_samples, simulator.dimension))

    # MostLikelyHeteroskedasticGP(normalize=False)
    # TODO: get this abomination up in the hierarchy and make it nicely parameterizable :)
    for i in tqdm(range(n_runs), desc="Simulating batch"):
        pipe = BayesOptPipeline(
            initialization=Random(size=n_random_samples, domain=np.array(simulator.get_domain(), dtype=float)),
            regression_model=RandomForestWrapper(n_estimators=10, random_state=44),
            acquisition=curry(BoundedUpperConfidenceBound, beta=0.7),
            selector=SimpleSelector,
        )
        batch_results[i, ...] = pipe.optimize(source=simulator, informed_sample_size=n_informed_samples, plot=False)

    batch_differences = euclidean_distance(batch_results, simulator.objective_function.get_maximum())
    return batch_results, batch_differences


if __name__ == "__main__":
    main()
