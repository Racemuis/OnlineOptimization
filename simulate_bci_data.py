import os
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from src.output.command_line import print_conf
from src.optimization.functions import get_simulator, run_simulation

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

    simulator = get_simulator(conf=conf, data_config=data_config)
    paper_auc = simulator.get_paper_score()

    fig, axes = plt.subplots(1, 2, sharey="all", figsize=(15, 8))

    boxplot_data = np.zeros((betas.shape[0], conf["n_runs"]))

    # Create a results directory if it does not exist
    destination_folder = (
        conf["destination_folder"]
        + "_"
        + conf["regressor"]
        + "_"
        + conf["convergence_measure"]
        + "_dim"
        + str(conf["dimension"])
    )
    Path(destination_folder).mkdir(parents=True, exist_ok=True)

    # Create dataframes for appending later on
    plot_df = pd.DataFrame([])
    boxplot_df = pd.DataFrame([])

    # Start grid search
    print(f"Performing grid-search for beta in {betas}")
    for i, beta in enumerate(betas):
        plt.close(fig)
        auc_score = run_simulation(conf=conf, beta=beta, simulator=simulator,)

        # plot results
        plt.figure(fig)
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

        pd.DataFrame(auc_score).to_csv(
            path_or_buf=os.path.join(
                destination_folder,
                f"results_{conf['experiment']}_{conf['regressor']}_{conf['convergence_measure']}_dim"
                f"{conf['dimension']}_beta{beta:.1f}_{conf['participant']}.csv",
            )
        )

        # Write results to pandas dataframe
        plot_df.to_csv(
            path_or_buf=os.path.join(
                destination_folder,
                f"results_{conf['experiment']}_{conf['regressor']}_{conf['convergence_measure']}_dim"
                f"{conf['dimension']}_{conf['participant']}.csv",
            )
        )
        boxplot_df.to_csv(
            path_or_buf=os.path.join(
                destination_folder,
                f"results_boxplot_{conf['experiment']}_{conf['regressor']}_{conf['convergence_measure']}_dim"
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
        f"AUC scores\n"
        f"{conf['participant']}, {conf['condition']} - {conf['regressor']}"
        f"\nrandom samples: {conf['random_sample_size']}, "
        f"informed samples: {conf['informed_sample_size']}"
        f", number of runs: {conf['n_runs']}"
        f", number of dimensions: {conf['dimension']}"
    )

    axes[1].boxplot([d.flatten() for d in boxplot_data])
    axes[1].set_xlabel("Beta")
    axes[1].set_xticklabels(betas)
    axes[1].set_title(f"Eventual proposals by the modules algorithm")
    plt.savefig(
        os.path.join(
            destination_folder,
            f"results_{conf['experiment']}_{conf['regressor']}_{conf['convergence_measure']}_dim"
            f"{conf['dimension']}_{conf['participant']}.png",
        )
    )


if __name__ == "__main__":
    main()
