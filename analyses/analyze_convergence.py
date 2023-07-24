import os
from pathlib import Path

import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

font = {"size": 7.5}
matplotlib.rc("font", **font)
plt.rcParams["figure.dpi"] = 250


def main():
    """
    Analyze the different convergence measures by creating box plots.

    Returns:
        None
    """
    dimensions = [1, 2, 7]
    participants = ["VPpblz_15_08_14", "VPpbob_15_08_13", "VPpboa_15_08_11"]
    measures = ["None", "length_scale", "mse"]

    folder = (
        r"C:\Users\Racemuis\Documents\school\m artificial intelligence\semester "
        r"2\thesis\results\results_components"
    )

    paper_score_path = r"C:\Users\Racemuis\Documents\school\m artificial intelligence\semester " \
                       r"2\thesis\bci_data\auditory_aphasia\paper_scores.csv"

    paper_scores = pd.read_csv(paper_score_path, index_col=0)

    data_dict_p = {}
    for p in participants:
        data_dict = {}
        for r in measures:
            maxes = []
            duplicates = 0
            for d in dimensions:
                sub_folder = f"evaluation_measure_Gaussian process regression_dim{d}"
                csv = f"scores_{p}_measure_{r}.csv"
                location_csv = f"locations_{p}_measure_{r}.csv"
                path = os.path.join(folder, sub_folder, csv)

                df = pd.read_csv(path, index_col=0, header=0)
                maxes.append(df.max(axis=1))

            # print(f"replicator: {r} - average number of duplicates {duplicates/len(dimensions)}")
            data_dict[r] = maxes
        data_dict_p[p] = data_dict

        ticks = dimensions

        fig = plt.figure()

        bpl = plt.boxplot(
            data_dict["None"], positions=np.array(range(len(data_dict["None"]))) * 2.0 - 0.4, sym="", widths=0.3
        )
        bpr = plt.boxplot(
            data_dict["length_scale"], positions=np.array(range(len(data_dict["length_scale"]))) * 2.0 + 0.4, sym="",
            widths=0.3
        )
        bpm = plt.boxplot(data_dict["mse"], positions=np.array(range(len(data_dict["mse"]))) * 2.0, sym="", widths=0.3)
        set_box_color(bpl, "tab:orange")
        set_box_color(bpm, "tab:blue")
        set_box_color(bpr, "tab:green")

        # draw temporary red and blue lines and use them to create a legend
        plt.plot([], c="tab:orange", label="none")
        plt.plot([], c="tab:blue", label="mse")
        plt.plot([], c="tab:green", label="length_scale")
        legend1 = plt.legend(title="Convergence measure")

        hline = plt.axhline(y=paper_scores[p][0], color='tab:green', linewidth=0.2, label="Paper score")
        legend2 = plt.legend([hline], ["Paper score"], loc='lower left')
        plt.gca().add_artist(legend1)
        plt.gca().add_artist(legend2)

        plt.xticks(range(0, len(ticks) * 2, 2), ticks)
        plt.xlim(-2, len(ticks) * 2)
        plt.title(fr"Optimization of the convergence measure - subject {p}" + "\n" + fr"$\beta$ = 0.187, replicator = variance")
        plt.xlabel("Dimensionality")
        plt.ylabel("Best found AUC")
        fig.tight_layout()
        path = r"./results"
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(os.path.join(path,f'analyze_conv_{p}.pdf'))


def set_box_color(bp, color):
    plt.setp(bp["boxes"], color=color)
    plt.setp(bp["whiskers"], color=color)
    plt.setp(bp["caps"], color=color)
    plt.setp(bp["medians"], color=color)


if __name__ == "__main__":
    main()
