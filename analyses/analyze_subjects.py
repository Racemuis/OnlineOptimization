import os
import re
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import wilcoxon, kurtosis, norm
from matplotlib.ticker import FormatStrFormatter

font = {"size": 10}
matplotlib.rc("font", **font)
plt.rcParams["figure.dpi"] = 250


def main():
    # Set hyperparameters
    plot_best_found = False
    d = 1

    participants = [
        "VPpbob_15_08_13",
        "VPpblz_15_08_14",
        "VPpboa_15_08_11",
        "VPpbog_15_08_25",
        "VPpbon_15_09_15",
        "VPpbor_15_10_07",
        "VPpbqb_15_08_06",
        "VPpboc_15_08_17",
        "VPpboi_15_09_01",
        "VPpboo_15_09_22",
        "VPpbos_15_10_09",
        "VPpbqe_15_07_28",
        "VPpbod_15_08_18",
        "VPpboj_15_09_04",
        "VPpbop_15_10_02",
        "VPpbot_15_10_13",
        "VPpboe_15_08_21",
        "VPpbom_15_09_14",
        "VPpboq_15_10_06",
        "VPpbqa_15_08_10",
    ]

    double_participants = ["VPpbob_15_08_13", "VPpblz_15_08_14", "VPpboa_15_08_11"]

    folder = (
        r"C:\Users\Racemuis\Documents\school\m artificial intelligence\semester " r"2\thesis\results\evaluation_results"
    )
    models = ["Gaussian process regression", "Random forest regression", "Random sampling"]
    model_names = ["$BO_{GP}$", "$BO_{RF}$", "$BO_{R}$"]

    fig, axes = plt.subplots(5, 4, sharex=True, sharey=False, figsize=(8, 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    paper_score_path = (
        r"C:\Users\Racemuis\Documents\school\m artificial intelligence\semester "
        r"2\thesis\bci_data\auditory_aphasia\paper_scores.csv"
    )

    paper_scores = pd.read_csv(paper_score_path, index_col=0)

    scores = np.zeros((len(participants), len(models)))
    sems = np.zeros((len(participants), len(models)))

    for i, p in enumerate(participants):
        # Format the axes
        axes[i // 4, i % 4].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        # Make outline gray if participant is used to tune the algorithm
        if p in double_participants:
            axes[i // 4, i % 4].tick_params(color="grey", labelcolor="grey")
            for spine in axes[i // 4, i % 4].spines.values():
                spine.set_edgecolor("grey")
            title_colour = "gray"
        else:
            title_colour = "black"

        # Plot paper score
        axes[i // 4, i % 4].axhline(y=paper_scores[p][0], color="tab:green", linewidth=0.2, label="Paper score")
        axes[i // 4, i % 4].set_title(p, fontsize=10, color=title_colour)

        for j, m in enumerate(models):
            file_name = fr"{m}_dim{d}"
            csv = f"scores_{p}.csv"
            path = os.path.join(folder, file_name, csv)

            df = pd.read_csv(path, index_col=0, header=0)
            if plot_best_found:
                best_found = calculate_best_found(df.to_numpy())
                mean = np.mean(best_found, axis=0)
                std = np.std(best_found, axis=0)
                sem = std / np.sqrt(best_found.shape[0])

            else:
                mean = np.mean(df.to_numpy(), axis=0)
                std = np.std(df.to_numpy(), axis=0)
                sem = std / np.sqrt(df.to_numpy().shape[0])

            axes[i // 4, i % 4].plot(mean, label=model_names[j])
            axes[i // 4, i % 4].fill_between(range(df.shape[1]), mean - sem, mean + sem, alpha=0.2)

            scores[i, j] = np.mean(np.max(df.to_numpy(), axis=1))
            sems[i, j] = np.std(np.max(df.to_numpy(), axis=1)) / df.to_numpy().shape[0]

    scores_to_LaTeX(scores, sems, participants)

    fig.supxlabel("iteration index")
    fig.supylabel("best found mean classification AUC" if plot_best_found else "mean classification AUC")
    fig.suptitle(f"{d}-dimensional objective function")

    lines, labels = axes[-2, -1].get_legend_handles_labels()
    # Put a legend below current axis
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.00),
        fancybox=False,
        shadow=False,
        ncol=4,
        handles=lines,
        labels=labels,
    )
    fig.tight_layout()

    if plot_best_found:
        path = f"./results//bf_evaluate_subjects_d{d}.pdf"
    else:
        path = f"./results/evaluate_subjects_d{d}.pdf"
    Path(path).mkdir(parents=True, exist_ok=True)

    fig.savefig(path, bbox_inches="tight")

    # Create histograms of the distributions
    fig2, axes2 = plt.subplots(1, 3, figsize=(10, 4))
    for i, ax in enumerate(axes2):
        mean = np.mean(scores[:, i])
        std = np.std(scores[:, i])
        kur = kurtosis(scores[:, i], fisher=True)

        ax.set_title(f"{model_names[i]}\nmean: {mean:.2f}, std: {std:.2f}, kur:{kur:.2f}")

        ax.set_xlim(0, 1)
        ax.hist(scores[:, i], bins=5, label="density histogram")

        x = np.linspace(0, 1, 50)
        ax.plot(x, norm.pdf(x, mean, std), label="superimposed Gaussian distribution")

    lines, labels = axes2[-1].get_legend_handles_labels()
    # Put a legend below current axis
    fig2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.00),
        fancybox=False,
        shadow=False,
        ncol=2,
        handles=lines,
        labels=labels,
    )
    fig2.supxlabel("ROC AUC score")
    fig2.supylabel("Frequency")
    fig2.tight_layout()

    path = f"./results"
    Path(path).mkdir(parents=True, exist_ok=True)
    fig2.savefig(os.path.join(path, f"histograms_d{d}.pdf"), bbox_inches="tight")

    for i in range(2):
        print(f"{models[i]} versus Random sampling\n\t{wilcoxon(scores[:, i], scores[:, 2])}")


def calculate_best_found(traces: np.ndarray) -> np.ndarray:
    """
    Calculate the best found scores at each iteration for multiple traces.

    Args:
        traces (np.ndarray): An array of traces of shape N_traces x N_iterations

    Returns:
        np.ndarray: The array of best found scores.
    """
    best_found = np.zeros(traces.shape)
    for i in range(traces.shape[0]):
        best_found[i, 0] = traces[i, 0]
        for j in range(1, traces.shape[1]):
            if traces[i, j] > best_found[i, j - 1]:
                best_found[i, j] = traces[i, j]
            else:
                best_found[i, j] = best_found[i, j - 1]
    return best_found


def scores_to_LaTeX(scores: np.ndarray, sems: np.ndarray, participants: List[str]) -> None:
    """
    Write the scores in a LaTeX table format.

    Args:
        scores (np.ndarray): The scores.
        sems (np.ndarray): The standard errors of the mean associated with the scores.
        participants List[str]: A list of participants.

    Returns:
        None
    """
    print(f"& $\mu$ & SEM & $\mu$ & SEM & $\mu$ & SEM " + r"\\")
    for p, row, sem_row in zip(participants, scores, sems):
        res = re.sub(r"_", r"\_", p)
        for s, sem in zip(row, sem_row):
            res += f" & {s:.4f} & {sem:.4f}"
        res += r"\\"
        print(res)
    mean = np.mean(scores, axis=0)
    std = np.std(scores, axis=0)
    print(
        fr"All & {mean[0]:.4f} & {std[0]/scores.shape[0]:.4f}& {mean[1]:.4f} & {std[1]/scores.shape[0]:.4f} & {mean[2]:.4f} & {std[2]/scores.shape[0]:.4f}\\"
    )


if __name__ == "__main__":
    main()
