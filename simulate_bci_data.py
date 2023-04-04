from typing import Tuple, Type, List, Callable, Optional

import yaml
import random
import torch
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from src.utils.base import Source
from src.utils.wrap_acqf import curry
from src.models.trees import RandomForestWrapper
from src.data.DataSimulator import DataSimulator
from src.optimization.selectors import AveragingSelector, VarianceSelector
from src.optimization.replicators import MaxReplicator
from src.optimization.pipelines import BayesOptPipeline
from src.optimization import initializers
from src.models.gaussian_processes import MostLikelyHeteroskedasticGP
from src.simulation import Simulator
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
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def main():
    """
    Simulate different objective functions based on BCI data to test the Bayesian Optimization pipeline.
    The simulations can be run for different runs to get a more stable estimate of the performance of the pipeline.

    Returns:
        None
    """
    # general optimization parameters
    n_random_samples = 8
    n_informed_samples = 15
    n_runs = 15
    plot = False

    # data simulator parameters
    experiment = "auditory_aphasia"
    participant = "VPpblz_15_08_14"
    condition = "6D"
    data_config = yaml.load(open("./src/conf/data_config.yaml", "r"), Loader=yaml.FullLoader)
    noise_functions = {
        "σ = 0.05": lambda x: np.array([0.05]),
        # "σ = 0.1": lambda x: np.array([0.1]),
    }
    dimension = 3

    # modules
    initialization = initializers.Random
    regression_models = {
        "Random forest regression": RandomForestWrapper(n_estimators=10, random_state=44),
        "Gaussian process regression": MostLikelyHeteroskedasticGP(normalize=False),
        "Random sampling": None,
    }
    regression_key = "Random sampling"
    regression_model = regression_models[regression_key]
    acquisition = curry(BoundedUpperConfidenceBound, center=True, beta=0.3)
    selector = VarianceSelector if dimension > 1 else AveragingSelector
    replicator = MaxReplicator()

    results = np.empty((len(noise_functions), n_runs, n_informed_samples + n_random_samples, dimension))
    auc_scores = np.empty((len(noise_functions), n_runs, n_informed_samples + n_random_samples))
    simulator = None

    for i, key in enumerate(noise_functions.keys()):
        # reset seed for each new simulation
        np.random.seed(44)
        random.seed(44)
        torch.manual_seed(44)

        simulator = DataSimulator(
            data_config=data_config,
            experiment=experiment,
            participant=participant,
            condition=condition,
            noise_function=noise_functions[key],
            dimension=dimension,
        )
        result = simulate_batch(
            simulator=simulator,
            initialization=initialization,
            regression_model=regression_model,
            acquisition=acquisition,
            selector=selector,
            replicator=replicator,
            n_runs=n_runs,
            n_informed_samples=n_informed_samples,
            n_random_samples=n_random_samples,
            plot=plot,
        )
        results[i, ...] = result
        print("Calculating results...", end='', flush=True)
        for j in range(n_runs):
            auc_scores[i, j, ...] = simulator.sample(x=result[j].squeeze(), noise=False)

    paper_auc = simulator.get_paper_score()
    print(" done!")

    # plot results
    for i, key in enumerate(noise_functions.keys()):
        plt.plot(range(n_random_samples + n_informed_samples), np.mean(auc_scores[i, ...], axis=0), label=key)

    plt.plot(
        range(n_random_samples + n_informed_samples),
        np.ones(n_random_samples + n_informed_samples) * paper_auc,
        label="paper parameters",
        c="tab:red",
        linewidth=0.3
    )
    plt.xlabel("Number of samples")
    plt.ylabel("AUC")
    plt.legend()
    plt.title(f"AUC scores obtained with 5 fold cross validation\n"
              f"{participant}, {condition} - {regression_key}\nrandom samples: {n_random_samples}, "
              f"informed samples: {n_informed_samples}"
              f", number of runs: {n_runs}"
              f", number of dimensions: {dimension}"
              )
    plt.show()


def simulate_batch(
    simulator: Source,
    initialization: Type[Initializer],
    regression_model: Optional[RegressionModel],
    acquisition: Callable,
    selector: Type[Selector],
    replicator: Replicator,
    n_runs: int,
    n_informed_samples: int,
    n_random_samples: int,
    plot: bool,
) -> np.ndarray:
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
        plot (bool): True if the intermediate runs should be plotted.

    Returns:
        batch_results (np.ndarray): The results of the optimization process.
        batch_differences (np.ndarray): The Euclidean distance between the true optimum and the estimated optimum.
    """
    batch_results = np.zeros((n_runs, n_random_samples + n_informed_samples, simulator.dimension))

    for i in tqdm(range(n_runs), desc="Simulating batch"):
        pipe = BayesOptPipeline(
            initialization=initialization(domain=np.array(simulator.get_domain(), dtype=float)),
            regression_model=regression_model,
            acquisition=acquisition,
            selector=selector,
            replicator=replicator,
        )
        batch_results[i, ...] = pipe.optimize(
            source=simulator, random_sample_size=n_random_samples, informed_sample_size=n_informed_samples, plot=plot
        )

    return batch_results


if __name__ == "__main__":
    main()
