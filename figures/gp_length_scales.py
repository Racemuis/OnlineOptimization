import os

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.gaussian_process.kernels import Matern


def main():
    """
    Create example figures for GPs with different length scales.

    Returns:
        None
    """
    font = {'size': 8}
    matplotlib.rc('font', **font)
    plt.rcParams["figure.dpi"] = 250
    np.random.seed(42)
    n_samples = 10
    n_figs = 3

    x = np.linspace(0, 5, 101)[:, np.newaxis]
    fig, axes = plt.subplots(1, n_figs, sharey=True, sharex=True, figsize=(12, 3))

    for i in range(n_figs):
        kernel = Matern(length_scale=i + 1, nu=2.5)
        K = kernel(X=x)

        # Plot prior
        samples = np.random.multivariate_normal(mean=np.zeros(K.shape[0]), cov=K, size=n_samples)
        std_prior = np.sqrt(K.diagonal())

        axes[i].set_title(f"$\ell$ = {i+1}")
        axes[i].plot(x.squeeze(), np.zeros(x.shape[0]), label="m(x)")
        axes[i].plot(x.squeeze(), samples.T, "C0", label="prior samples", linewidth=0.3)
        axes[i].fill_between(
            x.squeeze(),
            0 - 1.96 * std_prior,
            0 + 1.96 * std_prior,
            alpha=0.2,
            label=r"95% confidence interval",
        )
        axes[i].set_xlim(0, 5)
        axes[i].set_ylim(-3, 3)
        if i > 0:
            axes[i].yaxis.set_ticks_position('none')

        # Shrink current axis's height by 10% on the bottom
        box = axes[i].get_position()
        axes[i].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    fig.supxlabel("x", y=0.09)
    fig.supylabel("$f(x)$", x=0.06)

    lines_labels = [axes[-1].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(-0.7, -0.14),
              fancybox=False, shadow=False, ncol=3, handles=lines[:2] + lines[-1:], labels=labels[:2] + labels[-1:])
    plt.show()
    fig.tight_layout()

    path = f"./results/"
    Path(path).mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(path, r"prior.pdf"), bbox_inches='tight')


if __name__ == '__main__':
    main()
