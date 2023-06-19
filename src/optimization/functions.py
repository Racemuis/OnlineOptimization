from typing import Union

import random
import torch
import numpy as np
from tqdm import tqdm

from src.utils import enums
from src.utils.base import Source
from src.utils.wrap_acqf import curry
from src.modules.models import gaussian_processes, trees
from src.data.DataSimulator import DataSimulator

from src.data.CVEP import CVEPSimulator
from src.optimization.pipelines import BayesOptPipeline
from src.simulation import Simulator
from src.modules.acquisition_functions import BoundedUpperConfidenceBound, BoundedUpperConfidenceBoundVar


def get_simulator(conf: dict, data_config: dict) -> Union[CVEPSimulator, DataSimulator]:
    """
    Create a simulator instance given the selected dataset in the configuration file.

    Args:
        conf (dict): The dictionary containing the contents of the main configuration file.
        data_config (dict): The dictionary containing the contents of the data configuration file.

    Returns:
        Simulator: the data simulator.
    """
    if conf["experiment"] == "auditory_aphasia":
        simulator = DataSimulator(data_config=data_config, bo_config=conf, noise_function=None, )

    elif conf["experiment"].upper() == "CVEP":
        simulator = CVEPSimulator(data_config=data_config, bo_config=conf, trial=True, )

    else:
        print(
            f"The chosen experiment \"{conf['experiment']}\" is not supported, defaulting to \"auditory_aphasia\"")
        simulator = DataSimulator(data_config=data_config, bo_config=conf, noise_function=None, )
    return simulator


def run_simulation(
    conf: dict, beta: float, simulator: DataSimulator,
):
    """
    Simulate different objective functions based on BCI data to test the Bayesian Optimization pipeline.
    The simulations can be run for different runs to get a more stable estimate of the performance of the pipeline.

    Args:
             conf (dict): The modules configuration file, containing:
            - experiment (str): The name of the experiment (matches with a key in the data_config).

            - participant (Optional[str]): The participant to read. If None is provided, all participants are read.
                Default = None.

            - condition (Optional[str]): The condition to read. If None is provided, all conditions are read.
                Default = None.

            - dimension (int): The dimensionality of the modules' problem.
                Only 0 < dimension <= 3 are supported.

            - regression_model (str): The regression model that is used for calculating the likelihood.

            - selector (str): A final selection strategy.

            - convergence_measure (str): The convergence measure to use in the selector.

            - replicator (str): A replicator strategy.

            - n_runs (int): The number of runs to simulate.

            - informed_sample_size (int): The number of informed samples to take.

            - random_sample_size (int): The number of random samples to take.
     beta (float): The parameter that determines the exploration vs exploitation tradeoff. (0 <= beta <= 1)
     simulator (Simulator): The simulator of the unknown modules function.

    Returns:
        np.ndarray: The auc scores of the simulated Bayesian modules processes.
    """
    # reset seed for each new simulation
    np.random.seed(44)
    random.seed(44)
    torch.manual_seed(44)

    if isinstance(simulator, CVEPSimulator):
        simulator.clear_weights()

    result = simulate_batch(simulator=simulator, conf=conf, beta=beta, plot=conf["plot"],)

    print("Calculating results...", end="", flush=True)

    # Clear the CCA weights for the possible application of (noiseless) cv
    if isinstance(simulator, CVEPSimulator):
        simulator.clear_weights()

    # allocate memory
    auc_scores = np.empty((conf["n_runs"], conf["informed_sample_size"] + conf["random_sample_size"]))

    # calculate results
    for j in range(conf["n_runs"]):
        auc_scores[j, ...] = simulator.sample(x=result[j].squeeze(), noise=False, cv=False)
    print(" done!")
    return auc_scores


def simulate_batch(simulator: Source, conf: dict, beta: float, plot: bool,) -> np.ndarray:
    """
    Make a batch simulation of a Bayesian modules process. For each informed sample, the modules process is
    queried to provide the most likely value for the optimum during that time in the process. These guesses are returned
    together with their distance to the true optimum.

    Args:
        simulator (Simulator): The simulator of the unknown modules function.
        conf (dict): The modules configuration file, containing:
            - experiment (str): The name of the experiment (matches with a key in the data_config).

            - participant (Optional[str]): The participant to read. If None is provided, all participants are read.
                Default = None.

            - condition (Optional[str]): The condition to read. If None is provided, all conditions are read.
                Default = None.

            - dimension (int): The dimensionality of the modules' problem.
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
        batch_results (np.ndarray): The results of the modules process.
        batch_differences (np.ndarray): The Euclidean distance between the true optimum and the estimated optimum.
    """
    batch_results = np.zeros(
        (conf["n_runs"], conf["random_sample_size"] + conf["informed_sample_size"], simulator.dimension)
    )

    try:
        convergence_measure = enums.ConvergenceMeasure(conf["convergence_measure"])
    except ValueError as error:
        raise ValueError(str(error) + f". Valid convergence measures are {enums.ConvergenceMeasure.list()}.")

    if conf["acquisition"] == "var":
        acquisition = curry(BoundedUpperConfidenceBoundVar, center=True, beta=beta)
    elif conf["acquisition"] == "ucb":
        acquisition = curry(BoundedUpperConfidenceBound, center=True, beta=beta)
    else:
        print(f'The acquisition function is not recognised, valid acquisition functions are "var" and "ucb"')
        print(f"Defaulting to ucb...")
        acquisition = curry(BoundedUpperConfidenceBound, center=True, beta=beta)

    for i in tqdm(range(conf["n_runs"]), desc=f"Simulating batch for beta={beta}"):
        regression_models = {
            "Random forest regression": trees.RandomForestWrapper(n_estimators=10, random_state=44),
            "Gaussian process regression": gaussian_processes.MostLikelyHeteroskedasticGP(normalize=True),
            "Random sampling": None,
        }

        pipe = BayesOptPipeline(
            participant=conf["participant"],
            initialization=enums.initializers[conf["initializer"]](
                domain=np.array(simulator.get_domain(), dtype=float)
            ),
            regression_model=regression_models[conf["regressor"]],
            acquisition=acquisition,
            selector=enums.selectors[conf["selector"]],
            replicator=enums.replicators[conf["replicator"]],
            beta=beta,
        )
        batch_results[i, ...], _ = pipe.optimize(
            source=simulator,
            random_sample_size=conf["random_sample_size"],
            informed_sample_size=conf["informed_sample_size"],
            plot=plot,
            convergence_measure=convergence_measure,
        )

    return batch_results
