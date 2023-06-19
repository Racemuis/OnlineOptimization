from typing import Callable

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt


def plot_1D(f: Callable, domain: np.ndarray) -> None:
    """
    Create a one-dimensional plot of the objective function ``f``.

    Args:
        f (Callable): The objective function.
        domain (np.ndarray): The domain on which the objective function is defined.

    Returns:
        None
    """
    x = np.linspace(start=domain[0], stop=domain[1], num=101)[:, np.newaxis]
    plt.plot(x, f(x))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def plot_2D(f: Callable, domain: np.ndarray) -> None:
    """
    Create a two-dimensional plot of the objective function ``f``.

    Args:
        f (Callable): The objective function.
        domain (np.ndarray): The domain on which the objective function is defined.

    Returns:
        None
    """
    x = np.linspace(start=domain[0], stop=domain[1], num=101)

    xx, yy = np.meshgrid(x, x)

    combined = np.empty((x.shape[0], x.shape[0], 2))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            combined[i, j, 0] = x[i]
            combined[i, j, 1] = x[j]
    zz = f(combined)

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()
