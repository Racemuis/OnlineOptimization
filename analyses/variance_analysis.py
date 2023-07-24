import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    """
    Plot the estimated sampling variance.

    Returns:
        None
    """
    save_path = r"./results"
    Path(save_path).mkdir(exist_ok=True, parents=True)
    for participant in [r'VPpblz_15_08_14', r'VPpboc_15_08_17', r'VPpboa_15_08_11', r'VPpbob_15_08_13']:
        results_folder = r'C:\Users\Racemuis\Documents\school\m artificial intelligence\semester ' \
                         r'2\thesis\results\variance_estimation'
        path = os.path.join(results_folder, f'results_{participant}.csv')

        results = pd.read_csv(filepath_or_buffer=path, index_col=0).to_numpy()

        plt.plot(np.linspace(0, 1, num=results.shape[1]), np.std(results, axis=0))
        plt.xlabel('Shrinkage parameter')
        plt.ylabel('σ')
        plt.title(participant)

        plt.savefig(os.path.join(save_path, f"{participant}.pdf"))


if __name__ == '__main__':
    main()
