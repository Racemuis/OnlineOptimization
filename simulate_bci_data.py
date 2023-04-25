from typing import Type, Callable, Optional

import yaml
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

from src.utils.base import Source
from src.utils.wrap_acqf import curry
from src.models.trees import RandomForestWrapper
from src.data.DataSimulator import DataSimulator
from src.data.CVEP import CVEPSimulator
from src.optimization.selectors import AveragingSelector, VarianceSelector
from src.optimization.replicators import MaxReplicator, SequentialReplicator
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


def run_simulation(
    conf: dict, beta: float, regression_model: RegressionModel, simulator: DataSimulator,
):
    """
    Simulate different objective functions based on BCI data to test the Bayesian Optimization pipeline.
    The simulations can be run for different runs to get a more stable estimate of the performance of the pipeline.

    Args:
     conf (dict): The optimization configuration file, containing:
            - experiment (str): The name of the experiment (matches with a key in the data_config).

            - participant (Optional[str]): The participant to read. If None is provided, all participants are read.
                Default = None.

            - condition (Optional[str]): The condition to read. If None is provided, all conditions are read.
                Default = None.

            - dimension (int): The dimensionality of the optimization problem.
                Only 0 < dimension <= 3 are supported.
     regression_model (RegressionModel): a regression model.
     beta (float): The parameter that determines the exploration vs exploitation tradeoff. (0 <= beta <= 1)
     simulator (Simulator): The simulator of the unknown optimization function.

    Returns:
        np.ndarray: The auc scores of the simulated Bayesian optimization processes.
    """
    # modules
    initialization = initializers.Sobol
    acquisition = curry(BoundedUpperConfidenceBound, center=True, beta=beta)
    selector = VarianceSelector if conf["dimension"] > 1 else AveragingSelector
    replicator = SequentialReplicator(horizon=2)  # MaxReplicator()

    # reset seed for each new simulation
    np.random.seed(44)
    random.seed(44)
    torch.manual_seed(44)

    result = simulate_batch(
        simulator=simulator,
        initialization=initialization,
        regression_model=regression_model,
        acquisition=acquisition,
        selector=selector,
        replicator=replicator,
        n_runs=conf["n_runs"],
        n_informed_samples=conf["informed_sample_size"],
        n_random_samples=conf["random_sample_size"],
        beta=beta,
        plot=conf["plot"],
    )
    print("Calculating results...", end="", flush=True)
    # allocate memory
    auc_scores = np.empty((conf["n_runs"], conf["informed_sample_size"] + conf["random_sample_size"]))
    for j in range(conf["n_runs"]): # TODO: Set CV to true on the cluster
        auc_scores[j, ...] = simulator.sample(x=result[j].squeeze(), noise=False, cv=False)
    print(" done!")
    # to_csv(auc_scores, dimension, noise_functions, regression_key, result)
    return auc_scores


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
    beta: float,
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
        beta (float): The parameter that determines the exploration vs exploitation tradeoff.
        plot (bool): True if the intermediate runs should be plotted.

    Returns:
        batch_results (np.ndarray): The results of the optimization process.
        batch_differences (np.ndarray): The Euclidean distance between the true optimum and the estimated optimum.
    """
    batch_results = np.zeros((n_runs, n_random_samples + n_informed_samples, simulator.dimension))

    for i in tqdm(range(n_runs), desc=f"Simulating batch for beta={beta}"):
        pipe = BayesOptPipeline(
            initialization=initialization(domain=np.array(simulator.get_domain(), dtype=float)),
            regression_model=regression_model,
            acquisition=acquisition,
            selector=selector,
            replicator=replicator,
            beta=beta,
        )
        batch_results[i, ...] = pipe.optimize(
            source=simulator, random_sample_size=n_random_samples, informed_sample_size=n_informed_samples, plot=plot
        )

    return batch_results


def to_csv(auc_scores: np.ndarray, dimension: int, noise_functions: dict, regression_key: str, results: np.ndarray):
    data = {}
    for i, key in enumerate(noise_functions.keys()):
        for j in range(dimension):
            data.update({f"noise_function_{key}_parameter{j}": results[i, :, -1, j].squeeze()})
        data.update({f"noise_function_{key}_auc": auc_scores[i, :, -1].squeeze()})
    df = pd.DataFrame(data=data)
    df.to_csv(path_or_buf=fr"./{regression_key}_{dimension}_parameters.csv", index=False)


def print_conf(config: dict):
    print("Simulating BCI data")
    print(f"\tdimensionality: {config['dimension']}")
    print(f"\trandom samples: {config['random_sample_size']}")
    print(f"\tinformed samples: {config['informed_sample_size']}")
    print(
        f"\texperiment: {config['experiment']}; participant: {config['participant']}; condition: {config['condition']}"
    )


if __name__ == "__main__":
    conf = yaml.load(open("./src/conf/bo_config.yaml", "r"), Loader=yaml.FullLoader)
    data_config = yaml.load(open("./src/conf/data_config.yaml", "r"), Loader=yaml.FullLoader)
    print_conf(config=conf)

    if conf['experiment'] == "auditory_aphasia":
        simulator = DataSimulator(data_config=data_config, bo_config=conf, noise_function=None,)
        paper_auc = simulator.get_paper_score()
    elif conf['experiment'].upper() == "CVEP":
        simulator = CVEPSimulator(data_config=data_config, bo_config=conf, trial=True,)
        paper_auc = simulator.get_paper_score()
    else:
        print(f"The chosen experiment \"{conf['experiment']}\" is not supported, defaulting to \"auditory_aphasia\"")
        simulator = DataSimulator(data_config=data_config, bo_config=conf, noise_function=None,)
        paper_auc = simulator.get_paper_score()

    # loop over these
    regression_models = {
        "Random forest regression": RandomForestWrapper(n_estimators=10, random_state=44),
        "Gaussian process regression": MostLikelyHeteroskedasticGP(normalize=False),
        "Random sampling": None,
    }

    fig, axes = plt.subplots(1, 2, sharey="all")

    betas = np.round(np.linspace(start=0, stop=1, num=11), decimals=1)
    boxplot_data = np.zeros((betas.shape[0], conf["n_runs"]))

    print(f"Performing grid-search for beta in {np.linspace(0, 1, 11)}")
    for i, beta in enumerate(betas):
        auc_score = run_simulation(
            conf=conf, beta=beta, regression_model=regression_models[conf["regression_key"]], simulator=simulator,
        )

        # plot results
        axes[0].plot(
            range(conf["random_sample_size"] + conf["informed_sample_size"]),
            np.mean(auc_score, axis=0),
            label=f"beta: {beta:.1f}",
        )

        # store boxplot data
        boxplot_data[i, :] = auc_score[:, -1]

    axes[0].plot(
        range(conf["random_sample_size"] + conf["informed_sample_size"]),
        np.ones(conf["random_sample_size"] + conf["informed_sample_size"]) * paper_auc,
        label="paper parameters",
        c="tab:red",
        linewidth=0.3,
    )
    axes[0].set_xlabel("Number of samples")
    axes[0].set_ylabel("AUC")
    axes[0].legend()
    axes[0].set_title(
        f"AUC scores obtained with 5 fold cross validation\n"
        f"{conf['participant']}, {conf['condition']} - {conf['regression_key']}"
        f"\nrandom samples: {conf['random_sample_size']}, "
        f"informed samples: {conf['informed_sample_size']}"
        f", number of runs: {conf['n_runs']}"
        f", number of dimensions: {conf['dimension']}"
    )

    axes[1].boxplot([d.flatten() for d in boxplot_data])
    axes[1].set_xlabel("Beta")
    axes[1].set_xticklabels(betas)
    axes[1].set_title(f"Eventual proposals by the optimization algorithm")
    plt.show()
