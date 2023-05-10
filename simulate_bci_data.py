import os
from pathlib import Path

import yaml
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

from src.utils import enums
from src.utils.base import Source
from src.utils.wrap_acqf import curry
from src.data.DataSimulator import DataSimulator

# from src.data.CVEP import CVEPSimulator
from src.optimization.pipelines import BayesOptPipeline
from src.simulation import Simulator
from src.optimization.acquisition_functions import BoundedUpperConfidenceBound

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

plt.rcParams.update({"font.size": 11})


def main() -> None:
    betas = np.round(np.linspace(start=0, stop=1, num=11), decimals=1)

    path = os.path.dirname(os.path.realpath(__file__))
    conf = yaml.load(open(os.path.join(path, "src/conf/bo_config.yaml"), "r"), Loader=yaml.FullLoader)
    data_config = yaml.load(open(os.path.join(path, "src/conf/data_config.yaml"), "r"), Loader=yaml.FullLoader)

    print_conf(config=conf)

    if conf["experiment"] == "auditory_aphasia":
        simulator = DataSimulator(data_config=data_config, bo_config=conf, noise_function=None,)
        paper_auc = simulator.get_paper_score()

    # elif conf["experiment"].upper() == "CVEP":
    #     simulator = CVEPSimulator(data_config=data_config, bo_config=conf, trial=True,)
    #     paper_auc = simulator.get_paper_score()

    else:
        print(f"The chosen experiment \"{conf['experiment']}\" is not supported, defaulting to \"auditory_aphasia\"")
        simulator = DataSimulator(data_config=data_config, bo_config=conf, noise_function=None,)
        paper_auc = simulator.get_paper_score()

    fig, axes = plt.subplots(1, 2, sharey="all", figsize=(15, 8))

    boxplot_data = np.zeros((betas.shape[0], conf["n_runs"]))

    # Create a results directory if it does not exist
    destination_folder = conf["destination_folder"]
    Path(destination_folder).mkdir(parents=True, exist_ok=True)

    # Create dataframes for appending later on
    plot_df = pd.DataFrame([])
    boxplot_df = pd.DataFrame([])

    # Start grid search
    print(f"Performing grid-search for beta in {np.linspace(0, 1, 11)}")
    for i, beta in enumerate(betas):
        auc_score = run_simulation(conf=conf, beta=beta, simulator=simulator,)

        # plot results
        axes[0].plot(
            range(conf["random_sample_size"] + conf["informed_sample_size"]),
            np.mean(auc_score, axis=0),
            label=f"beta: {beta:.1f}",
        )

        # store boxplot data
        boxplot_data[i, :] = auc_score[:, -1]

        # write the results to a csv file
        plot_df = pd.concat([plot_df, pd.DataFrame({beta: np.mean(auc_score, axis=0)})], axis=1)
        boxplot_df = pd.concat([boxplot_df, pd.DataFrame({beta: auc_score[:, -1]})], axis=1)

    # Write results to pandas dataframe
    plot_df.to_csv(
        path_or_buf=os.path.join(
            destination_folder,
            f"results_{conf['experiment']}_{conf['regression_key']}_{conf['convergence_measure']}_dim"
            f"{conf['dimension']}_{conf['participant']}.csv",
        )
    )
    boxplot_df.to_csv(
        path_or_buf=os.path.join(
            destination_folder,
            f"results_boxplot_{conf['experiment']}_{conf['regression_key']}_{conf['convergence_measure']}_dim"
            f"{conf['dimension']}_{conf['participant']}.csv",
        )
    )

    # Plot the results
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
    plt.savefig(
        os.path.join(
            destination_folder,
            f"results_{conf['experiment']}_{conf['regression_key']}_{conf['convergence_measure']}_dim"
            f"{conf['dimension']}_{conf['participant']}.png",
        )
    )


def run_simulation(
    conf: dict, beta: float, simulator: DataSimulator,
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

            - regression_model (str): The regression model that is used for calculating the likelihood.

            - selector (str): A final selection strategy.

            - convergence_measure (str): The convergence measure to use in the selector.

            - replicator (str): A replicator strategy.

            - n_runs (int): The number of runs to simulate.

            - informed_sample_size (int): The number of informed samples to take.

            - random_sample_size (int): The number of random samples to take.
     beta (float): The parameter that determines the exploration vs exploitation tradeoff. (0 <= beta <= 1)
     simulator (Simulator): The simulator of the unknown optimization function.

    Returns:
        np.ndarray: The auc scores of the simulated Bayesian optimization processes.
    """
    # reset seed for each new simulation
    np.random.seed(44)
    random.seed(44)
    torch.manual_seed(44)

    result = simulate_batch(simulator=simulator, conf=conf, beta=beta, plot=conf["plot"],)

    print("Calculating results...", end="", flush=True)
    # allocate memory
    auc_scores = np.empty((conf["n_runs"], conf["informed_sample_size"] + conf["random_sample_size"]))

    # calculate results using all data and cv for "more" reliability
    for j in range(conf["n_runs"]):
        auc_scores[j, ...] = simulator.sample(x=result[j].squeeze(), noise=False, cv=False)
    print(" done!")
    return auc_scores


def simulate_batch(simulator: Source, conf: dict, beta: float, plot: bool,) -> np.ndarray:
    """
    Make a batch simulation of a Bayesian optimization process. For each informed sample, the optimization process is
    queried to provide the most likely value for the optimum during that time in the process. These guesses are returned
    together with their distance to the true optimum.

    Args:
        simulator (Simulator): The simulator of the unknown optimization function.
        conf (dict): The optimization configuration file, containing:
            - experiment (str): The name of the experiment (matches with a key in the data_config).

            - participant (Optional[str]): The participant to read. If None is provided, all participants are read.
                Default = None.

            - condition (Optional[str]): The condition to read. If None is provided, all conditions are read.
                Default = None.

            - dimension (int): The dimensionality of the optimization problem.
                Only 0 < dimension <= 3 are supported.

            - regression_model (str): The regression model that is used for calculating the likelihood.

            - selector (str): A final selection strategy.

            - convergence_measure (str): The convergence measure to use in the selector.

            - replicator (str): A replicator strategy.

            - n_runs (int): The number of runs to simulate.

            - informed_sample_size (int): The number of informed samples to take.

            - random_sample_size (int): The number of random samples to take.

        beta (float): The parameter that determines the exploration vs exploitation tradeoff.

        plot (bool): True if the intermediate runs should be plotted.

    Returns:
        batch_results (np.ndarray): The results of the optimization process.
        batch_differences (np.ndarray): The Euclidean distance between the true optimum and the estimated optimum.
    """
    batch_results = np.zeros(
        (conf["n_runs"], conf["random_sample_size"] + conf["informed_sample_size"], simulator.dimension)
    )
    n_replications = 0

    try:
        convergence_measure = enums.ConvergenceMeasure(conf["convergence_measure"])
    except ValueError as error:
        raise ValueError(str(error) + f". Valid convergence measures are {enums.ConvergenceMeasure.list()}.")

    for i in tqdm(range(conf["n_runs"]), desc=f"Simulating batch for beta={beta}"):
        pipe = BayesOptPipeline(
            initialization=enums.initializers[conf["initializer"]](
                domain=np.array(simulator.get_domain(), dtype=float)
            ),
            regression_model=enums.regression_models[conf["regressor"]],
            acquisition=curry(BoundedUpperConfidenceBound, center=True, beta=beta),
            selector=enums.selectors[conf["selector"]],
            replicator=enums.replicators[conf["replicator"]],
            beta=beta,
        )
        batch_results[i, ...], n_replications = pipe.optimize(
            source=simulator,
            random_sample_size=conf["random_sample_size"],
            informed_sample_size=conf["informed_sample_size"],
            plot=plot,
            convergence_measure=convergence_measure,
        )

    print(f"Average number of replications made over {conf['n_runs']} runs: {n_replications / conf['n_runs']}")

    return batch_results


def print_conf(config: dict):
    print("Simulating BCI data")
    print(f"\tregressor: \t\t\t{config['regressor']}")
    print(f"\tinitializer: \t\t\t{config['initializer']}")
    print(f"\treplicator: \t\t\t{config['replicator']}")
    print(f"\tselector: \t\t\t{config['selector']}")
    print(f"\tconvergence measure: \t\t{config['convergence_measure']}")
    print(f"\tdimensionality: \t\t{config['dimension']}")
    print(f"\trandom samples: \t\t{config['random_sample_size']}")
    print(f"\tinformed samples: \t\t{config['informed_sample_size']}")
    print(
        f"\texperiment: \t\t\t{config['experiment']}; participant: {config['participant']}; condition: {config['condition']}"
    )


if __name__ == "__main__":
    main()
