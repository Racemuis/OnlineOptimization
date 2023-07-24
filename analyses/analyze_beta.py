import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd

from figures.example_traces import best_found
from src.plot_functions.utils import plot_GP_1d
from src.modules.models import MostLikelyHeteroskedasticGP

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

font = {"size": 9}
matplotlib.rc("font", **font)
plt.rcParams["figure.dpi"] = 250


def main():
    """
    Analyze the outcomes for the different beta values by creating trace plots.

    Returns:
        None
    """
    auc = False
    d = 1
    betas = np.round(np.linspace(0, 1, 11), 3)
    domain = [0, 1]
    participants = ["VPpblz_15_08_14", "VPpboc_15_08_17", "VPpbob_15_08_13", "VPpboa_15_08_11"]
    folder = r"C:\Users\Racemuis\Documents\school\m artificial intelligence\semester 2\thesis\results\results_beta"
    file_name = fr"beta_selection_Gaussian process regression_dim{d}"

    fig, axes = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(6.4, 4.8))
    fig2, ax2 = plt.subplots(2, 2, sharex=True)

    for j, p in enumerate(participants):
        maxes = np.zeros((betas.shape[0], 10))
        aucs = np.zeros((betas.shape[0], 10))

        # get results
        for i, beta in enumerate(betas):
            csv = rf"scores_{p}_beta_{beta:.1f}.csv"
            path = os.path.join(folder, file_name, csv)
            df = pd.read_csv(path, index_col=0, header=0)

            ax2[j // 2, j % 2].plot(range(df.shape[1]), np.mean(df.to_numpy(), axis=0), label=beta)

            maxes[i, :] = df.max(axis=1)

            auc_small = np.zeros(aucs.shape[1])
            for k in range(aucs.shape[1]):
                auc_small[k] = np.sum(best_found(df.iloc[k])) / df.shape[1]
            aucs[i, :] = auc_small

        if j // 2 == 1:
            ax2[j // 2, j % 2].set_xlabel("iteration")
        if j % 2 == 0:
            ax2[j // 2, j % 2].set_ylabel("Classification AUC")
        ax2[j // 2, j % 2].set_title(f"{p}, dim={d}")

        # expand the betas
        betas_plot = np.repeat(betas, 10)

        # reshape maxes and aucs
        maxes = maxes.reshape(betas.shape[0] * 10)
        aucs = aucs.reshape(betas.shape[0] * 10)

        y_train = aucs if auc else maxes

        # Fit GP on the data
        gp = MostLikelyHeteroskedasticGP(normalize=True, n_iter=1)
        model = gp.fit(x_train=torch.tensor(betas_plot)[:, np.newaxis], y_train=torch.tensor(y_train)[:, np.newaxis])

        # Get posterior variables
        model.eval()
        x_test = np.linspace(domain[0], domain[1], num=51)
        posterior_mean = model.posterior(torch.tensor(x_test)[:, np.newaxis]).mean.detach().numpy().squeeze()
        posterior_std = np.sqrt(
            model.posterior(torch.tensor(x_test)[:, np.newaxis]).variance.detach().numpy().squeeze()
        )

        ax = axes[j // 2, j % 2]

        ylabel = "Metric: AUC" if auc else "Metric: Best found"

        # Plot the GP
        plot_GP_1d(
            x=betas_plot,
            y=y_train,
            x_test=x_test,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            title=p,
            xlabel=r"$\beta$" if j // 2 == 1 else None,
            ylabel=ylabel if j % 2 == 0 else None,
            r_x=None,
            ax=ax,
        )
        ax.axvline(x_test[np.argmax(posterior_mean)], color="tab:red", label="optimum", linewidth=0.5, zorder=20)
        ax.text(
            x_test[np.argmax(posterior_mean)],
            0.6,
            fr"$\beta$={x_test[np.argmax(posterior_mean)]:.2f}",
            rotation=90,
            color="tab:red",
            horizontalalignment="left",
            fontsize=9,
            zorder=20,
        )

    ax2[-1, -1].legend(title=r"$\beta$", ncols=2, fontsize=6)
    fig2.tight_layout()
    fig2.savefig(f"./analyze_beta_auc_d{d}.pdf", bbox_inches='tight')

    axes[-1, -1].legend()
    path = r"./results"
    Path(path).mkdir(parents=True, exist_ok=True)
    fig.suptitle(rf"Optimization of $\beta$, dimensionality: {d}")
    if auc:
        fig.savefig(os.path.join(path, f"./analyze_beta_d{d}.pdf"), bbox_inches='tight')
    else:
        fig.savefig(os.path.join(path, f"./analyze_beta_best_found_d{d}.pdf"), bbox_inches='tight')


if __name__ == "__main__":
    main()
