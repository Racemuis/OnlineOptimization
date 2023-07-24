import os
import yaml
from typing import Union
from pathlib import Path

import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data import get_simulator
from src.optimization import Optimizer
from src.output.command_line import print_conf

import warnings


warnings.filterwarnings(
    "ignore", message="Input data is not contained to the unit cube. Please consider min-max scaling the input data."
)
warnings.filterwarnings(
    "ignore",
    message="Input data is not standardized. Please consider scaling the input to zero mean and unit variance.",
)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def main():
    # Set hyperparameters
    domain = [(0, 1)]
    n_random_samples = 11
    n_informed_samples = 5
    n_replicates = 5  # number of replicates for the overzealous replicator

    # Set paths
    path = os.path.dirname(os.path.realpath(__file__))
    conf = yaml.load(open(os.path.join(path, "src/conf/bo_config.yaml"), "r"), Loader=yaml.FullLoader)
    data_config = yaml.load(open(os.path.join(path, "src/conf/data_config.yaml"), "r"), Loader=yaml.FullLoader)

    print_conf(config=conf)
    participants = conf["participant"]

    # Create a results directory if it does not exist
    destination_folder = conf["destination_folder"] + "_" + conf["regressor"] + "_dim" + str(conf["dimension"])
    Path(destination_folder).mkdir(parents=True, exist_ok=True)

    # Loop over the participants
    for p in participants:

        # Specify the config
        conf["participant"] = p

        # Get data simulator
        simulator = get_simulator(conf=conf, data_config=data_config,)

        # Initialize BO process
        beta_opt = Optimizer(
            n_random_samples=n_random_samples,
            domain=domain,
            beta=0.2,
            initializer="sobol",
            acquisition="ucb",
            selector="variance",
            regression_model="Gaussian process regression",
        )

        for _ in tqdm(range(n_random_samples + n_informed_samples), desc="Optimizing beta"):

            # Sample the BO
            beta = np.round(beta_opt.query(n_samples=1), 3)

            # Evaluate the sample
            score = objective_function(beta[0, 0], conf, destination_folder, n_replicates, p, simulator)

            # Inform the BO
            beta_opt.inform(beta, torch.tensor([[score]]))

        # Store the outcomes
        df = pd.DataFrame(
            np.column_stack((beta_opt.x_train.detach().numpy(), beta_opt.y_train.detach().numpy())),
            columns=["x_train", "y_train"],
        )
        df.to_csv(os.path.join(destination_folder, rf"bo_outcomes_{p}.csv"))


def objective_function(b, conf, destination_folder, n_replicates, p, simulator) -> Union[float, np.ndarray]:

    # Reset seeds
    np.random.seed(44)
    random.seed(44)
    torch.manual_seed(44)

    # Create storage framework
    auc_scores = np.zeros((conf["n_runs"], conf["random_sample_size"] + conf["informed_sample_size"]))
    # Loop over the runs
    for run in range(conf["n_runs"]):

        # Initialize BO process
        opt = Optimizer(
            n_random_samples=conf["random_sample_size"],
            domain=simulator.get_domain().T,
            beta=b,
            initializer="sobol",
            acquisition="ucb",
            selector="variance",
            regression_model="Gaussian process regression",
        )

        for i in range(conf["random_sample_size"] + conf["informed_sample_size"]):
            # Sample a newly proposed position from the BO
            sample = opt.query(n_samples=1)

            for _ in range(n_replicates):  # Overzealous replicator @ work
                # Evaluate the sample
                outcome = simulator.sample(sample.detach().numpy(), noise=True)

                # Inform the BO process
                opt.inform(sample, torch.tensor([[outcome]]))

            # Get the selected outcome
            guess = opt.select().detach().numpy()

            # Store the selected outcome
            append_locations(
                path=os.path.join(destination_folder, f"locations_{p}_beta_{b:.3f}.csv"), run=run, location=guess
            )

            # Quick fix guess wrapping
            if len(guess.shape) <= 1:
                guess = guess[np.newaxis, :]

            # Store the associated score
            auc_scores[run, i] = simulator.sample(guess, noise=False)

    # Write the AUC scores to a csv file
    csv_path = os.path.join(destination_folder, f"scores_{p}_beta_{b:.3f}.csv")
    pd.DataFrame(auc_scores).to_csv(csv_path)
    return np.mean(np.max(auc_scores, axis=1))


def append_locations(path: str, run: int, location: np.ndarray) -> None:
    with open(path, "a") as f:
        results = f"{run},"
        for cat in location:
            results += f"{cat},"
        f.write(results + "\n")


if __name__ == "__main__":
    main()
