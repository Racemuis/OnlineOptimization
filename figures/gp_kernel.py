import os
from pathlib import Path

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern


def main():
    font = {'size': 7.5}
    matplotlib.rc('font', **font)
    plt.rcParams["figure.dpi"] = 250
    np.random.seed(42)
    n_samples = 10
    n_figs = 3

    x = np.linspace(0, 5, 6)[:, np.newaxis]
    fig, axes = plt.subplots(1, n_figs, sharey=True, sharex=True)

    im = None
    for i in range(n_figs):
        kernel = Matern(length_scale=i + 1, nu=2.5)
        K = kernel(X=x)
        im = axes[i].matshow(K)
        axes[i].set_title(f"$\ell$ = {i+1}")
        axes[i].xaxis.set_ticks_position('bottom')
        if i > 0:
            axes[i].yaxis.set_ticks_position('none')

    plt.xticks(x.squeeze())
    plt.yticks(x.squeeze())

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes(
        [axes[n_figs-1].get_position().x1 + 0.01, axes[n_figs-1].get_position().y0 - 0.035, 0.02, axes[1].get_position().height + 0.072]
    )
    fig.colorbar(im, cax=cbar_ax)
    plt.show()

    path = r"./results"
    Path(path).mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(path, r"kernels.pdf"), bbox_inches='tight')


if __name__ == '__main__':
    main()
