import os

import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

font = {"size": 7.5}
matplotlib.rc("font", **font)
plt.rcParams["figure.dpi"] = 250


def main():
    d = 1
    participants = ["VPpblz_15_08_14", "VPpboc_15_08_17", "VPpbob_15_08_13", "VPpboa_15_08_11"]
    measures = ["None", "length_scale", "mse"]

    data_dict = {}
    for r in measures:
        # Get the vanilla results from the replicator folder
        if r == "None":
            folder = (
                r"C:\Users\Racemuis\Documents\school\m artificial intelligence\semester "
                r"2\thesis\results\results_replicator_sobol_8_beta_02"
            )

            sub_folder = f"optimization_replicator_Gaussian process regression_dim{d}"
        else:
            folder = (
                r"C:\Users\Racemuis\Documents\school\m artificial intelligence\semester "
                r"2\thesis\results\results_convergence_sobol_8_beta_02"
            )

            sub_folder = f"optimization_convergence_Gaussian process regression_dim{d}"

        maxes = []
        for p in participants:
            if r == "None":
                csv = f"results_beta0.2_replicator_{r}_{p}.csv"
            else:
                csv = f"results_beta0.2_measure_{r}_{p}.csv"
            path = os.path.join(folder, sub_folder, csv)
            df = pd.read_csv(path, index_col=0, header=0)
            maxes.append(df.max(axis=1))
        data_dict[r] = maxes

    ticks = participants

    fig = plt.figure()

    bpl = plt.boxplot(
        data_dict["None"], positions=np.array(range(len(data_dict["None"]))) * 2.0 - 0.4, sym="", widths=0.3
    )
    bpr = plt.boxplot(
        data_dict["length_scale"],
        positions=np.array(range(len(data_dict["length_scale"]))) * 2.0 + 0.4,
        sym="",
        widths=0.3,
    )
    bpm = plt.boxplot(data_dict["mse"], positions=np.array(range(len(data_dict["mse"]))) * 2.0, sym="", widths=0.3)
    set_box_color(bpl, "tab:orange")
    set_box_color(bpm, "tab:blue")
    set_box_color(bpr, "tab:green")

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c="tab:orange", label="none")
    plt.plot([], c="tab:blue", label="mse")
    plt.plot([], c="tab:green", label="length_scale")
    plt.legend(title="Convergence measure")

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks) * 2)
    plt.title(f"Optimization of the convergence measure, dimensionality: {d}")
    plt.xlabel("Participant")
    plt.ylabel("Best found AUC")
    plt.tight_layout()
    plt.show()
    fig.savefig(f"analyze_conv_d{d}.pdf")


def set_box_color(bp, color):
    plt.setp(bp["boxes"], color=color)
    plt.setp(bp["whiskers"], color=color)
    plt.setp(bp["caps"], color=color)
    plt.setp(bp["medians"], color=color)


if __name__ == "__main__":
    main()
