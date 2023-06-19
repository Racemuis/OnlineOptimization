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
    regressor = r"Gaussian process regression"
    measure = r"None"
    data_dir = fr"C:\Users\Racemuis\Documents\school\m artificial intelligence\semester " \
               fr"2\thesis\results\transfer_3\results_{regressor}_{measure}_dim{dimension}"
    n_runs = 40
    cumulative = False

    betas = np.round(np.linspace(start=0, stop=1, num=11), decimals=1)

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
        if cumulative:
            mean = np.mean(best_seen, axis=0)
            std = np.std(best_seen, axis=0)
        else:
            mean = np.mean(df.to_numpy(), axis=0)
            std = np.std(df.to_numpy(), axis=0)
        # ax.errorbar(range(df.shape[1]), mean, yerr=std/np.sqrt(n_runs), label=f"{beta}")
        ax.plot(range(df.shape[1]), mean, label=f"{beta}")
        auc[i] = (1 / df.shape[1]) * np.sum(best_seen, axis=1)  # AUC per run
        best_found[i] = best_seen[:, -1]  # best found sample per run
        print(f"Beta: {beta}, auc: {np.mean(auc[i]):.3f}, {np.sum(mean)}")

    if cumulative:
        plt.ylabel("Best seen (f)")
    else:
        plt.ylabel("AUC corresponding to selected sample")

    ax.set_xlabel("Iteration (i)")

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width * 0.8, box.height * 0.9])

    # Put a legend to the right of the current axis
    plt.legend(title="Beta", loc='center left', bbox_to_anchor=(1, 0.5))

    title = f"{regressor}\ndimension: {dimension}"
    if regressor == "Gaussian process regression" and measure != "None":
        title += f", measure: {measure}"

    plt.title(f"{regressor}\ndimension: {dimension}")
    plt.show()
    figname = fr"./{regressor}_d{dimension}" + ("_cumulative" if cumulative else "") + ".pdf"
    fig.savefig(figname, bbox_inches='tight')


if __name__ == "__main__":
    main()
