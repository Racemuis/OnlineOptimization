import os
import yaml
from pathlib import Path

import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from src.output.command_line import print_conf
from src.modules.models import gaussian_processes
from src.optimization.functions import get_simulator, run_simulation

from botorch.exceptions.warnings import OptimizationWarning

import warnings

from src.plot_functions.utils import plot_GP_1d

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

plt.rcParams["figure.dpi"] = 250


def main() -> None:
    n_samples = 3
    domain = np.array([0, 1])
    path = os.path.dirname(os.path.realpath(__file__))
    conf = yaml.load(open(os.path.join(path, "src/conf/bo_config.yaml"), "r"), Loader=yaml.FullLoader)
    data_config = yaml.load(open(os.path.join(path, "src/conf/data_config.yaml"), "r"), Loader=yaml.FullLoader)

    print_conf(config=conf)
    participants = conf['participant']

    # Create a results directory if it does not exist
    destination_folder = conf["destination_folder"] + "_" + conf["regressor"] + "_dim" + str(conf["dimension"])
    Path(destination_folder).mkdir(parents=True, exist_ok=True)

    # Loop over participants
    for p in participants:
        conf["participant"] = p

        # Load data
        simulator = get_simulator(conf, data_config)

        # Start modules of beta
        betas = np.round(np.linspace(0, 1, num=n_samples), 2)
        ys = np.zeros(conf['n_runs'] * n_samples)

        for i, beta in enumerate(betas):
            auc_scores = run_simulation(conf=conf, beta=beta, simulator=simulator,)

            # Take the mean over all samples
            ys[i*conf['n_runs']:(i+1)*conf['n_runs']] = np.mean(auc_scores, axis=-1)

            # write the original results to a csv file
            pd.DataFrame(auc_scores).to_csv(
                path_or_buf=os.path.join(
                    destination_folder,
                    f"results_{conf['experiment']}_{conf['regressor']}_{conf['convergence_measure']}_dim"
                    f"{conf['dimension']}_beta{beta:.3f}_{conf['participant']}.csv",
                )
            )

        # expand the betas
        betas = np.repeat(betas, conf['n_runs'])

        # Fit GP on the data
        gp = gaussian_processes.MostLikelyHeteroskedasticGP(normalize=True, n_iter=5)
        model = gp.fit(x_train=torch.tensor(betas)[:, np.newaxis], y_train=torch.tensor(ys)[:, np.newaxis])

        # Get posterior variables
        x_test = np.linspace(domain[0], domain[1], num=51)
        posterior_mean = model.posterior(torch.tensor(x_test)[:, np.newaxis]).mean.detach().numpy().squeeze()
        posterior_std = np.sqrt(model.posterior(torch.tensor(x_test)[:, np.newaxis]).variance.detach().numpy().squeeze())
        r_x = gp.get_estimated_std(torch.tensor(x_test)).detach().numpy().squeeze()

        # Plot the GP
        plot_GP_1d(
            x=betas,
            y=ys,
            x_test=x_test,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            title=conf["participant"],
            xlabel="Beta",
            ylabel="Mean AUC",
            path=os.path.join(destination_folder, fr"{conf['participant']}_beta_GP.pdf"),
            r_x=r_x,
        )

if __name__ == "__main__":
    main()
