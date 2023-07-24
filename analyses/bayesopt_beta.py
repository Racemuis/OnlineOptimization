import os

import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

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

font = {'size': 7.5}
matplotlib.rc('font', **font)
plt.rcParams["figure.dpi"] = 250


def main():
    # Set hyperparameters

    NUM_COLORS = 4
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    cm = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red']

    d = 2
    domain = [0, 1]
    dimensions = [1, 2, 7]
    participants = [
        "VPpblz_15_08_14",
        "VPpboa_15_08_11",
        "VPpbob_15_08_13",
    ]

    # Set paths
    folder = r"C:\Users\Racemuis\Documents\school\m artificial intelligence\semester 2\thesis\results\bayesopt_beta"

    paper_score_path = r"C:\Users\Racemuis\Documents\school\m artificial intelligence\semester " \
                       r"2\thesis\bci_data\auditory_aphasia\paper_scores.csv"

    paper_scores = pd.read_csv(paper_score_path, index_col=0)

    maxes = np.zeros((len(participants), len(dimensions)))

    for k, p in enumerate(participants):
        csv = f"bo_outcomes_{p}.csv"
        fig, axes = plt.subplots(1, 3, sharey='all')

        for i, (ax, d) in enumerate(zip(axes, dimensions)):
            subdir = rf"beta_selection_Gaussian process regression_dim{d}"
            df = pd.read_csv(os.path.join(folder, subdir, csv), index_col=0, header=0)
            ax.scatter(df["x_train"], df["y_train"], label="Observed data", color='tab:orange')
            x_train = df["x_train"].to_numpy()
            y_train = df["y_train"].to_numpy()

            # Fit GP on the data
            gp = MostLikelyHeteroskedasticGP(normalize=True, n_iter=1)
            model = gp.fit(x_train=torch.tensor(x_train)[:, np.newaxis], y_train=torch.tensor(y_train)[:, np.newaxis])

            # Get posterior variables
            model.eval()
            x_test = np.linspace(domain[0], domain[1], num=51)
            posterior_mean = model.posterior(torch.tensor(x_test)[:, np.newaxis]).mean.detach().numpy().squeeze()
            posterior_std = np.sqrt(
                model.posterior(torch.tensor(x_test)[:, np.newaxis]).variance.detach().numpy().squeeze()
            )

            # Plot the GP
            ax.plot(x_test, posterior_mean, label=r"posterior $\mu(x)$")
            ax.fill_between(
                x_test,
                posterior_mean - 1.96 * posterior_std,
                posterior_mean + 1.96 * posterior_std,
                label=r"95% confidence interval",
                color='tab:blue',
                alpha=0.2,
            )
            ax.set_title(f"dimensionality: {d}")
            ax.axvline(x_test[np.argmax(posterior_mean)], linewidth=0.5, label="posterior optimum", color='tab:green')
            ax.axhline(y=paper_scores[p][0], color='tab:red', linewidth=0.2, label="Paper score")

            maxes[k, i] = x_test[np.argmax(posterior_mean)]

            if i > 0:
                axes[i].yaxis.set_ticks_position('none')
            box = axes[i].get_position()
            axes[i].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

            fig2, axes2 = plt.subplots(1, 1)
            for f, beta in enumerate(np.sort(x_train)):
                trace_path = f"scores_{p}_beta_{beta:.3f}.csv"
                traces = pd.read_csv(os.path.join(folder, subdir, trace_path), index_col=0, header=0).to_numpy()
                lines = axes2.plot(range(traces.shape[1]), np.mean(traces, axis=0), label=beta)
                lines[0].set_color(cm[(f // NUM_COLORS)-1])
                lines[0].set_linestyle(LINE_STYLES[(f % NUM_COLORS) % NUM_STYLES])

            axes2.set_xlabel("Iteration index")
            axes2.set_ylabel("Classification AUC")
            axes2.set_title(f"Subject: {p} - dimensionality: {d}")
            # Shrink current axis by 20%
            box = axes2.get_position()
            axes2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            axes2.legend(title=r"$\beta$",loc='center left', bbox_to_anchor=(1, 0.5))
            # fig2.savefig(f"./bayesopt_beta_traces_{p}_d{d}.pdf", bbox_inches='tight')

        for j, ax in enumerate(axes):
            ax.text(
                maxes[k, j],
                np.mean(axes[-1].get_ylim()),
                fr"$\beta$={maxes[k, j]:.2f}",
                rotation=90,
                color="tab:green",
                horizontalalignment="left",
                fontsize=9,
            )

        fig.supxlabel(r"$\beta$", y=0.09)
        fig.supylabel("Best found AUC scores", x=0.04)
        fig.suptitle(rf"Subject {p}")

        lines_labels = [axes[-1].get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

        # Put a legend below current axis
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.08),
                   fancybox=False, shadow=False, ncol=5, handles=lines,
                   labels=labels)
        plt.tight_layout()
        fig.savefig(f"./bayesopt_beta_{p}.pdf", bbox_inches='tight')
    print(np.mean(maxes))


if __name__ == "__main__":
    main()
