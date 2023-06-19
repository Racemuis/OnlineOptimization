import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
plt.rcParams['figure.dpi'] = 250



def main():
    experiment = r"auditory_aphasia"
    dimension = r"3"
    participant = r"VPpblz_15_08_14"
    regressor = r"Random sampling"
    measure = r"mse"
    data_dir = fr"C:\Users\Racemuis\Documents\school\m artificial intelligence\semester " \
               fr"2\thesis\results\transfer\random\results_{regressor}_{measure}_dim{dimension}"
    n_runs = 15

    betas = np.round(np.linspace(start=0, stop=1, num=11), decimals=1)

    betas = np.array([0.2])

    best_found = np.zeros((betas.shape[0], n_runs))
    auc = np.zeros((betas.shape[0], n_runs))

    fig = plt.figure()
    ax = plt.subplot(111)

    for i, beta in enumerate(betas):
        filename = fr"results_{experiment}_{regressor}_{measure}_dim{dimension}_beta{beta}_{participant}.csv"
        df = pd.read_csv(filepath_or_buffer=os.path.join(data_dir, filename), index_col=0)

        best_seen = np.zeros(df.shape)
        for j, data in df.iterrows():
            best_seen[j, :] = [np.max(data[:m]) for m in range(data.size)]
            best_seen[j, 0] = data[0]
        mean = np.mean(best_seen, axis=0)
        std = np.std(best_seen, axis=0)
        # mean = np.mean(df.to_numpy(), axis=0)
        # std = np.std(df.to_numpy(), axis=0)
        # ax.errorbar(range(df.shape[1]), mean, yerr=std/np.sqrt(n_runs), label=f"{beta}")
        ax.plot(range(df.shape[1]), mean, label=f"{beta}")
        # ax.fill_between(range(df.shape[1]), mean-std, mean+std, alpha=0.2)
        auc[i] = (1 / df.shape[1]) * np.sum(best_seen, axis=1)  # AUC per run
        best_found[i] = best_seen[:, -1]  # best found sample per run
        print(f"Beta: {beta}, auc: {np.mean(auc[i]):.3f}, {np.sum(mean)}")

    # print("\nStatistics")
    # for i, beta_1 in enumerate(betas):
    #     for j, beta_2 in enumerate(betas[i+1:]):
    #         best_found_s, best_found_p = mannwhitneyu(best_found[i], best_found[j])
    #         # if best_found_p < 1:
    #         #     print(f"Beta = {beta_1} vs Beta = {beta_2}")
    #         #     print(f"\tbest found: statistic={best_found_s}, pvalue={best_found_p}")
    #         #     auc_s, auc_p = mannwhitneyu(auc[i], auc[j])
    #         #     print(f"\tauc: statistic={auc_s}, pvalue={auc_p}")
    #
    plt.ylabel("Best seen (f)")
    # plt.ylabel("AUC corresponding to selected sample")
    ax.set_xlabel("Iteration (i)")

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width * 0.8, box.height * 0.9])

    # Put a legend to the right of the current axis
    # plt.legend(title="Beta", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Random sampling\ndimension: 3")
    plt.show()
    fig.savefig("./Random_d3_cumulative.pdf", bbox_inches='tight')


if __name__ == "__main__":
    main()
