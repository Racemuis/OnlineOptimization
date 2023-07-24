import os

import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

font = {"size": 7.5}
matplotlib.rc("font", **font)
plt.rcParams["figure.dpi"] = 250

# TODO: Count duplicates https://stackoverflow.com/questions/35584085/how-to-count-duplicate-rows-in-pandas-dataframe


def main():
    d = 1
    participants = ["VPpblz_15_08_14", "VPpboc_15_08_17", "VPpbob_15_08_13", "VPpboa_15_08_11"]
    replicators = ["None", "max", "sequential"]
    folder = (
        r"C:\Users\Racemuis\Documents\school\m artificial intelligence\semester "
        r"2\thesis\results\results_replicator_sobol_8_beta_02"
    )

    sub_folder = f"optimization_replicator_Gaussian process regression_dim{d}"

    data_dict = {}
    for r in replicators:
        duplicates = 0
        maxes = []
        for p in participants:
            csv = f"results_beta0.2_replicator_{r}_{p}.csv"
            location_csv = f"locations_beta0.200_replicator_{r}_{p}.csv"
            duplicates += count_duplicates(os.path.join(folder, sub_folder, location_csv))
            path = os.path.join(folder, sub_folder, csv)
            df = pd.read_csv(path, index_col=0, header=0)
            maxes.append(df.max(axis=1))
        print(f"replicator: {r} - average number of duplicates {duplicates/40}")
        data_dict[r] = maxes

    ticks = participants

    fig = plt.figure()

    bpl = plt.boxplot(
        data_dict["None"], positions=np.array(range(len(data_dict["None"]))) * 2.0 - 0.4, sym="", widths=0.3
    )
    bpr = plt.boxplot(
        data_dict["sequential"], positions=np.array(range(len(data_dict["sequential"]))) * 2.0 + 0.4, sym="", widths=0.3
    )
    bpm = plt.boxplot(data_dict["max"], positions=np.array(range(len(data_dict["max"]))) * 2.0, sym="", widths=0.3)
    set_box_color(bpl, "tab:orange")
    set_box_color(bpm, "tab:blue")
    set_box_color(bpr, "tab:green")

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c="tab:orange", label="none")
    plt.plot([], c="tab:blue", label="max")
    plt.plot([], c="tab:green", label="sequential")
    plt.legend(title="Replicator")

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks) * 2)
    plt.title(f"Optimization of the replicator, dimensionality: {d}")
    plt.xlabel("Participant")
    plt.ylabel("Best found AUC")
    plt.tight_layout()
    plt.show()
    fig.savefig(f'analyze_repl_d{d}.pdf')


def set_box_color(bp, color):
    plt.setp(bp["boxes"], color=color)
    plt.setp(bp["whiskers"], color=color)
    plt.setp(bp["caps"], color=color)
    plt.setp(bp["medians"], color=color)

# TODO: not correct due to the way I store the locations
def count_duplicates(path: str) -> int:
    df = pd.read_csv(path, index_col=0, header=0)
    return df.duplicated(keep='first').sum()


if __name__ == "__main__":
    main()
