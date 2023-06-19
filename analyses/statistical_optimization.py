import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon


def main():
    experiment = r"auditory_aphasia"
    dimensions = ["1", "3", "7"]
    participant = r"VPpblz_15_08_14"
    regressor = r"Gaussian process regression"
    measures = ["length_scale", "mse", "noise_uncertainty", "model_uncertainty"]
    path = r"C:\Users\Racemuis\Documents\school\m artificial intelligence\semester 2\thesis\results\transfer_2"
    beta = 0.2
    n_runs = 15

    auc = np.zeros((len(dimensions), len(measures), n_runs))
    auc_random = np.zeros((len(dimensions), n_runs))
    auc_rf = np.zeros((len(dimensions), n_runs))
    best_found = np.zeros((len(dimensions), len(measures), n_runs))
    best_found_random = np.zeros((len(dimensions), n_runs))
    best_found_rf = np.zeros((len(dimensions), n_runs))

    fig, axes = plt.subplots(len(dimensions), 1, sharex=True)

    for i_d, d in enumerate(dimensions):
        for i_m, m in enumerate(measures):
            data_dir = fr"C:\Users\Racemuis\Documents\school\m artificial intelligence\semester " \
                       fr"2\thesis\results\transfer\results_{regressor}_{m}_dim{d}"
            filename = fr"results_{experiment}_{regressor}_{m}_dim{d}_beta{beta}_{participant}.csv"
            df = pd.read_csv(filepath_or_buffer=os.path.join(data_dir, filename), index_col=0)

            mean = np.mean(df.to_numpy(), axis=0)
            std = np.std(df.to_numpy(), axis=0)

            axes[i_d].plot(range(df.shape[1]), mean, label=f"{m}")
            axes[i_d].fill_between(range(df.shape[1]), mean-std, mean+std, alpha=0.2)
            axes[i_d].set_title(f"dimension = {d}")


        rf = rf"C:\Users\Racemuis\Documents\school\m artificial intelligence\semester 2\thesis\results\transfer\results_Random forest regression_mse_dim{d}\results_{experiment}_Random forest regression_mse_dim{d}_beta{beta}_{participant}.csv"
        df = pd.read_csv(filepath_or_buffer=rf, index_col=0)

        mean = np.mean(df.to_numpy(), axis=0)
        std = np.std(df.to_numpy(), axis=0)

        axes[i_d].plot(range(df.shape[1]), mean, label=f"random forest regression")
        axes[i_d].fill_between(range(df.shape[1]), mean - std, mean + std, alpha=0.2)

        noise = fr"C:\Users\Racemuis\Documents\school\m artificial intelligence\semester 2\thesis\results\transfer\random\results_Random sampling_mse_dim{d}\results_{experiment}_Random sampling_mse_dim{d}_beta{beta}_{participant}.csv "
        df = pd.read_csv(filepath_or_buffer=noise, index_col=0)

        mean = np.mean(df.to_numpy(), axis=0)
        std = np.std(df.to_numpy(), axis=0)

        axes[i_d].plot(range(df.shape[1]), mean, label=f"random sampling")
        axes[i_d].fill_between(range(df.shape[1]), mean - std, mean + std, alpha=0.2)

    fig.supylabel("AUC score")
    fig.supxlabel("Iteration (i)")
    axes[0].legend(title="Convergence measure", ncols=3, loc='upper center', bbox_to_anchor=(0.5, 1.7))
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(len(dimensions), 1, sharex=True)
    for i_d, d in enumerate(dimensions):
        for i_m, m in enumerate(measures):
            data_dir = fr"C:\Users\Racemuis\Documents\school\m artificial intelligence\semester " \
                       fr"2\thesis\results\transfer\results_{regressor}_{m}_dim{d}"
            filename = fr"results_{experiment}_{regressor}_{m}_dim{d}_beta{beta}_{participant}.csv"
            df = pd.read_csv(filepath_or_buffer=os.path.join(data_dir, filename), index_col=0)

            best_seen = np.zeros(df.shape)
            for j, data in df.iterrows():
                best_seen[j, :] = [np.max(data[:index]) for index in range(data.size)]
                best_seen[j, 0] = data[0]
            mean = np.mean(best_seen, axis=0)
            std = np.std(best_seen, axis=0)
            axes[i_d].plot(range(df.shape[1]), mean, label=f"{m}")
            axes[i_d].fill_between(range(df.shape[1]), mean - std, mean + std, alpha=0.2)
            auc[i_d, i_m, :] = (1 / df.shape[1]) * np.sum(best_seen, axis=1)  # AUC per run
            print(f"dimension: {d}, convergence measure: {m}, auc: {np.mean(auc[i_d, i_m, :]):.3f}")
            best_found[i_d, i_m] = best_seen[:, -1]  # best found sample per run

        noise = fr"C:\Users\Racemuis\Documents\school\m artificial intelligence\semester 2\thesis\results\transfer\random\results_Random sampling_mse_dim{d}\results_{experiment}_Random sampling_mse_dim{d}_beta{beta}_{participant}.csv "

        df = pd.read_csv(filepath_or_buffer=noise, index_col=0)

        best_seen = np.zeros(df.shape)
        for j, data in df.iterrows():
            best_seen[j, :] = [np.max(data[:index]) for index in range(data.size)]
            best_seen[j, 0] = data[0]
        mean = np.mean(best_seen, axis=0)
        std = np.std(best_seen, axis=0)

        axes[i_d].plot(range(df.shape[1]), mean, label=f"random sampling")
        axes[i_d].fill_between(range(df.shape[1]), mean - std, mean + std, alpha=0.2)
        axes[i_d].set_title(f"dimension = {d}")
        auc_random[i_d, :] = (1 / df.shape[1]) * np.sum(best_seen, axis=1)  # AUC per run
        print(f"Random sampling, dimension: {d}, auc: {np.mean(auc[i_d, :]):.3f}")
        best_found_random[i_d] = best_seen[:, -1]

        rf = rf"C:\Users\Racemuis\Documents\school\m artificial intelligence\semester 2\thesis\results\transfer\results_Random forest regression_mse_dim{d}\results_{experiment}_Random forest regression_mse_dim{d}_beta{beta}_{participant}.csv"

        df = pd.read_csv(filepath_or_buffer=rf, index_col=0)
        best_seen = np.zeros(df.shape)
        for j, data in df.iterrows():
            best_seen[j, :] = [np.max(data[:index]) for index in range(data.size)]
            best_seen[j, 0] = data[0]
        mean = np.mean(best_seen, axis=0)
        std = np.std(best_seen, axis=0)

        axes[i_d].plot(range(df.shape[1]), mean, label=f"random forest regression")
        axes[i_d].fill_between(range(df.shape[1]), mean - std, mean + std, alpha=0.2)
        axes[i_d].set_title(f"dimension = {d}")
        auc_rf[i_d, :] = (1 / df.shape[1]) * np.sum(best_seen, axis=1)  # AUC per run
        print(f"Random forest regression, dimension: {d}, auc: {np.mean(auc[i_d, :]):.3f}")
        best_found_rf[i_d] = best_seen[:, -1]

    fig.supylabel("Best seen")
    fig.supxlabel("Iteration (i)")
    axes[0].legend(title="Convergence measure", ncols=4, loc='upper center', bbox_to_anchor=(0.5, 1.7))
    plt.tight_layout()
    plt.show()

    print("\nStatistics")
    for i, d in enumerate(dimensions):
        res = wilcoxon(best_found[i, 0, :], best_found_random[i])
        print(f"Dimension {d}, GP, mse - sobol:\n\tstatistic {res.statistic}, pvalue {res.pvalue}")
        res = wilcoxon(best_found_rf[i, :], best_found_random[i])
        print(f"Dimension {d}, Random forest - sobol:\n\tstatistic {res.statistic}, pvalue {res.pvalue}")


if __name__ == "__main__":
    main()
