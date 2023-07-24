import os
import yaml
from pathlib import Path

import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from .uilts import append_locations
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
    """
    Find the optimal convergence measure by comparing None, length scale and mse.

    Returns:
        None

    """
    # Set hyperparameters
    convergence_measures = ["None", "length_scale", "mse"]
    beta = 0.187

    # Set paths
    path = os.path.dirname(os.path.realpath(__file__))
    conf = yaml.load(open(os.path.join(path, "src/conf/bo_config.yaml"), "r"), Loader=yaml.FullLoader)
    data_config = yaml.load(open(os.path.join(path, "src/conf/data_config.yaml"), "r"), Loader=yaml.FullLoader)

    print_conf(config=conf)
    print(f"\tbeta: \t\t\t\t{beta}")
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

        for cv in tqdm(convergence_measures, desc="Optimizing replicator"):
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
                    beta=beta,
                    initializer="sobol",
                    acquisition="ucb",
                    selector="variance",
                    regression_model="Gaussian process regression",
                    replicator="max",
                    convergence_measure=cv,
                )

                for i in range(conf["random_sample_size"] + conf["informed_sample_size"]):
                    # Sample a newly proposed position from the BO
                    sample = opt.query(n_samples=1)

                    # Evaluate the sample
                    outcome = simulator.sample(sample.detach().numpy(), noise=True)

                    # Inform the BO process
                    opt.inform(sample, torch.tensor([[outcome]]))

                    # Get the selected outcome
                    guess = opt.select().detach().numpy()

                    # Store the selected outcome
                    append_locations(
                        path=os.path.join(destination_folder, f"locations_{p}_measure_{cv}.csv"), run=run,
                        location=guess
                    )

                    # Quick fix guess wrapping
                    if len(guess.shape) <= 1:
                        guess = guess[np.newaxis, :]

                    # Store the associated score
                    auc_scores[run, i] = simulator.sample(guess, noise=False)

            # Write the AUC scores to a csv file
            csv_path = os.path.join(destination_folder, f"scores_{p}_measure_{cv}.csv")
            pd.DataFrame(auc_scores).to_csv(csv_path)


if __name__ == "__main__":
    main()
