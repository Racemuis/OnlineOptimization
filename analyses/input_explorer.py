import os

import pandas as pd
import yaml
import numpy as np
import itertools

from ..src.data.ERPSimulator import DataSimulator
from ..src.data.CVEPSimulator import CVEPSimulator


def main() -> None:
    n_grid_samples = 20
    path = os.path.dirname(os.path.realpath(__file__))
    conf = yaml.load(open(os.path.join(path, "src/conf/bo_config.yaml"), "r"), Loader=yaml.FullLoader)
    data_config = yaml.load(open(os.path.join(path, "src/conf/data_config.yaml"), "r"), Loader=yaml.FullLoader)

    if conf["experiment"] == "auditory_aphasia":
        simulator = DataSimulator(
            data_config=data_config, bo_config=conf, noise_function=None, augmentation=None
        )

    elif conf["experiment"].upper() == "CVEP":
        simulator = CVEPSimulator(data_config=data_config, bo_config=conf, trial=True,)

    else:
        print(f"The chosen experiment \"{conf['experiment']}\" is not supported, defaulting to \"auditory_aphasia\"")
        simulator = DataSimulator(data_config=data_config, bo_config=conf, noise_function=None,)

    domains = simulator.get_domain().T
    grid = None
    for i, domain in enumerate(domains):
        if grid is None:
            grid = np.linspace(domain[0], domain[1], n_grid_samples)
        else:
            grid = np.column_stack((grid, np.linspace(domain[0], domain[1], n_grid_samples)))

    results = np.zeros((n_grid_samples**domains.shape[0], domains.shape[0] + 1))

    for i, element in enumerate(itertools.product(*grid.T)):
        results[i, :-1] = element
        results[i, -1] = simulator.sample(np.array([list(element)]), noise=False)

    pd.DataFrame(results).to_csv(r"./space_search.csv", index=False, header=False)


if __name__ == "__main__":
    main()
