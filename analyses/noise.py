import os
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


from src.data.ERPSimulator import DataSimulator

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def main():
    """
    Assess the degree of sampling noise by using grid search.

    Returns:
        None
    """
    n_samples = 500

    # read yaml
    path = os.path.dirname(os.path.realpath(__file__))
    conf = yaml.load(open(os.path.join(path, "../src/conf/bo_config.yaml"), "r"), Loader=yaml.FullLoader)
    data_config = yaml.load(open(os.path.join(path, "../src/conf/data_config.yaml"), "r"), Loader=yaml.FullLoader)

    simulator = DataSimulator(data_config=data_config, bo_config=conf, noise_function=None, )

    grid = np.round(np.linspace(0, 1, num=31), decimals=4)
    results = np.zeros((n_samples, grid.shape[0]))
    for i, g in enumerate(tqdm(grid, desc="Performing grid evaluation")):
        results[:, i] = simulator.sample(np.array([[g]]*n_samples), noise=True)

    path = r"./results"
    Path(path).mkdir(exist_ok=True, parents=True)
    pd.DataFrame(data=results, columns=grid).to_csv(os.path.join(path, f"./results_{conf['participant']}.csv"))

    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)
    plt.plot(grid, mean)
    plt.fill_between(grid, mean-std, mean+std, alpha=0.2)
    plt.show()


if __name__ == '__main__':
    main()
