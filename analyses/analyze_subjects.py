import os
import yaml

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import wilcoxon

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

font = {"size": 6}
matplotlib.rc("font", **font)
plt.rcParams["figure.dpi"] = 250


def main():
    # Set hyperparameters
    d = 1

    participants = ["VPpbob_15_08_13", "VPpbog_15_08_25", "VPpbon_15_09_15", "VPpbor_15_10_07", "VPpbqb_15_08_06",
                    "VPpboc_15_08_17", "VPpboi_15_09_01", "VPpboo_15_09_22", "VPpbos_15_10_09", "VPpbqe_15_07_28",
                    "VPpblz_15_08_14", "VPpbod_15_08_18", "VPpboj_15_09_04", "VPpbop_15_10_02", "VPpbot_15_10_13",
                    "VPpboa_15_08_11", "VPpboe_15_08_21", "VPpbom_15_09_14", "VPpboq_15_10_06", "VPpbqa_15_08_10"]

    folder = r"C:\Users\Racemuis\Documents\school\m artificial intelligence\semester 2\thesis\results\results_eval"
    models = ["Gaussian process regression", "Random sampling"]

    fig, axes = plt.subplots(5, 4, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    paper_score_path = r"C:\Users\Racemuis\Documents\school\m artificial intelligence\semester " \
                       r"2\thesis\bci_data\auditory_aphasia\paper_scores.csv"

    paper_scores = pd.read_csv(paper_score_path, index_col=0)

    for i, p in enumerate(participants):
        # Plot paper score
        axes[i // 4, i % 4].axhline(y=paper_scores[p][0], color='r', linewidth=0.2, label="Paper score")
        axes[i // 4, i % 4].set_title(p, fontsize=6)

        best_found = np.zeros((2, 10))

        for j, m in enumerate(models):
            file_name = fr"subject_evaluation_{m}_dim{d}"
            csv = f"scores_{p}.csv"
            path = os.path.join(folder, file_name, csv)
            df = pd.read_csv(path, index_col=0, header=0)
            axes[i // 4, i % 4].plot(np.mean(df.to_numpy(), axis=0), label=m)
            best_found[j] = np.max(df.to_numpy(), axis=1)
        differences = best_found[0] - best_found[1]
        print(f"Subject {p}\n\t {wilcoxon(differences)}")

    fig.supxlabel("iteration")
    fig.supylabel("classification AUC")
    fig.suptitle(f"{d}-dimensional objective function")

    lines, labels = axes[-1, -1].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', ncol=3)
    plt.show()
    fig.savefig(f"./evaluate_subjects_d{d}.pdf", bbox_inches='tight')


if __name__ == '__main__':
    main()
