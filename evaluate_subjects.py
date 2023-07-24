import os
import yaml
from pathlib import Path

import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data import get_simulator
from src.optimization import Optimizer
from src.utils.command_line import print_conf


def main():
    # Set paths
    path = os.path.dirname(os.path.realpath(__file__))
    conf = yaml.load(open(os.path.join(path, "src/conf/bo_config.yaml"), "r"), Loader=yaml.FullLoader)
    data_config = yaml.load(open(os.path.join(path, "src/conf/data_config.yaml"), "r"), Loader=yaml.FullLoader)

    print_conf(config=conf)
    participants = conf['participant']

    # Create a results directory if it does not exist
    destination_folder = conf["destination_folder"] + "_" + conf["regressor"] + "_dim" + str(conf["dimension"])
    Path(destination_folder).mkdir(parents=True, exist_ok=True)

    # Loop over the participants
    for p in participants:
        # Reset seeds
        np.random.seed(44)
        random.seed(44)
        torch.manual_seed(44)

        # Create storage framework
        auc_scores = np.zeros((conf['n_runs'], conf['random_sample_size'] + conf['informed_sample_size']))

        # Specify the config
        conf['participant'] = p

        # Get data simulator
        simulator = get_simulator(conf=conf, data_config=data_config,)

        # Loop over the runs
        for run in range(conf['n_runs']):
            print(f"Subject: {p} - run {run+1}/{conf['n_runs']}")
            # Initialize BO process
            opt = Optimizer(
                n_random_samples=conf['random_sample_size'],
                domain=simulator.get_domain().T,
                initializer=conf['initializer'],
                acquisition=conf['acquisition'],
                selector=conf['selector'],
                regression_model=conf['regressor'],
                beta=0.2,
            )

            for i in tqdm(range(conf['random_sample_size'] + conf['informed_sample_size'])):
                # Sample a newly proposed position from the BO
                sample = opt.query(n_samples=1)

                # Evaluate the sample
                outcome = simulator.sample(sample.detach().numpy(), noise=True)

                # Inform the BO process
                opt.inform(sample, torch.tensor([[outcome]]))

                # Get the selected outcome
                guess = opt.select().detach().numpy()

                # Store the selected outcome
                append_locations(path=os.path.join(destination_folder, f"locations_{p}.csv"), run=run, location=guess)

                # Quick fix guess wrapping
                if len(guess.shape) <= 1:
                    guess = guess[np.newaxis, :]

                # Store the associated score
                auc_scores[run, i] = simulator.sample(guess, noise=False)

        # Write the AUC scores to a csv file
        csv_path = os.path.join(destination_folder, f"scores_{p}.csv")
        pd.DataFrame(auc_scores).to_csv(csv_path)


def append_locations(path: str, run: int, location: np.ndarray) -> None:
    """
    Append `location` to a csv file that will be located at `path`. If the csv file already exists, the locations are
    appended to the file, else a new file is created.

    Args:
        path (str): The path to the the csv file.
        run (int): The index of the run.
        location (np.ndarray): The sampled location.

    Returns:
        None.
    """
    with open(path, 'a') as f:
        results = f"{run},"
        for cat in location:
            results += f"{cat},"
        f.write(results + "\n")


if __name__ == "__main__":
    main()
