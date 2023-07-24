import os

import numpy as np
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 250
font = {'size': 15}
matplotlib.rc('font', **font)

np.random.seed(42)


def main():
    """
    Create figures of example traces to illustrate the methods used in:

    Dewancker, I., McCourt, M.J., Clark, S.C., Hayes, P., Johnson, A., & Ke, G. (2016).
    A Stratified Analysis of Bayesian Optimization Methods. ArXiv, abs/1603.09441.
    Returns:

    """
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    linspace_a = np.linspace(0.2, 0.8, 58)
    linspace_b = np.linspace(0.1, 0.5, 58)
    std = 0.05
    trace_a = best_found(np.random.normal(linspace_a, std, 58))
    trace_b = best_found(np.random.normal(linspace_b, std, 58))

    # trace_a = np.random.normal(linspace_a, std, 58)
    # trace_b = np.random.normal(linspace_b, std, 58)

    ax.plot(trace_a, '-o', label="A")
    ax.plot(trace_b, '-o', label="B")
    ax.fill_between(np.linspace(0, 57, 58), trace_a-std, trace_a + std*1.98, alpha=0.2)
    ax.fill_between(np.linspace(0, 57, 58), trace_b - std, trace_b + std*1.98, alpha=0.2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("best found $f(\hat{x}^*)$")
    ax.legend(title="Optimization process", ncols=2)
    plt.tight_layout()
    plt.show()

    path = r"./results/"
    Path(path).mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(path, r"best_found.pdf"), bbox_inches='tight')


def best_found(trace: np.ndarray) -> np.ndarray:
    """
    For each index, calculate the best found element thus far.

    Args:
        trace (np.ndarray): An array of scores.

    Returns:
        np.ndarray: An array of the best found scores.
    """
    best = 0
    results = np.zeros(trace.shape)
    for i, element in enumerate(trace):
        if element > best:
            best = element
        results[i] = best
    return results


if __name__ == '__main__':
    main()
