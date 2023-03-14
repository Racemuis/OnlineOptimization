from typing import Callable, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import cm, colors, lines

from gpytorch.models.exact_gp import ExactGP

from ..simulation.ObjectiveFunction import ObjectiveFunction
from ..simulation.Simulator import Simulator


def setup_simulation_plot(
    n_simulations: int, objective_function: ObjectiveFunction, cmap: Callable,
) -> Tuple[plt.Figure, plt.Axes, np.ndarray]:
    """
    Set up the plotting backbone for the simulation plot (I extracted a function to remove some clutter).

    Args:
        n_simulations (int): The number of samples to simulate.
        objective_function (ObjectiveFunction): The objective function to simulate.
        cmap (plt.colors.Colormap): The colormap to use to indicate the passing of time in the simulation.

    Returns:
        plt.Figure: The set up figure.
        plt.Axes: The set up axes.
        np.ndarray: The set up colormap.
    """
    fig, ax = plt.subplots(1, 1)

    # Plot the objective function
    domain = objective_function.domain
    x = np.arange(start=domain[0], stop=domain[1], step=0.1)
    f_x = objective_function.f(x)
    ax.plot(x, f_x, label="f(x)")

    # Set up the title settings
    ax.set_title(f"Random samples of the objective function '{objective_function.name}'")
    ax.set_xlabel("Stimulus Onset Asynchrony (SOA)")
    ax.set_ylabel("AUC")
    ax.set_xlim(left=domain[0], right=domain[1])
    ax.set_ylim(
        bottom=0, top=objective_function.f(objective_function.get_maximum()) + 2,
    )

    # Create a legend
    cross = lines.Line2D([], [], color="gray", marker="x", linestyle="None")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(cross)
    labels.append("y(x)")
    ax.legend(handles, labels, loc="lower right")

    # Add a color bar
    color = cmap(np.linspace(0, 1, n_simulations))
    fig.colorbar(
        cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=n_simulations), cmap=cmap),
        ax=ax,
        orientation="vertical",
        label="Sample index",
    )
    return fig, ax, color


def plot_objective_function(
    domain: np.ndarray, objective_function: ObjectiveFunction, simulator: Simulator, num: int = 100
) -> None:
    """
    Visualise the objective function with a few simulated samples.

    Args:
        domain (np.ndarray): The domain of the objective function.
        objective_function (ObjectiveFunction): The objective function to simulate.
        simulator (Simulator): A simulator object.
        num (int): The number of samples that should be visualised.

    Returns:
        None
    """
    plot_x = np.linspace(start=domain[0], stop=domain[1], num=num)
    plot_y = np.array([simulator.sample(x) for x in plot_x])
    plt.scatter(plot_x, plot_y, label="y(x)")
    plt.plot(plot_x, objective_function.f(plot_x), label="f(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def plot_GP(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: torch.Tensor,
    model: ExactGP,
    observation_noise: bool = False,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Plot the posterior mean of a gaussian process regression model, including the data points and two standard errors of
    the mean.

    Args:
        x_train (np.ndarray): the x-coordinates of the observed samples.
        y_train (np.ndarray): the y-coordinates of the observed samples.
        x_test (torch.Tensor): the x-coordinates for which to predict the GP posterior.
        model (SingleTaskGP): the trained GP model.
        observation_noise (bool): True if the standard errors should include observational noise.
        ax (plt.Axes): The axis on which the figure should be plotted.

    Returns:
        plt.Axes: The plotted axis.
    """
    # no need for gradients
    with torch.no_grad():
        # compute posterior
        model.eval()
        posterior = model.posterior(x_test, observation_noise=observation_noise)

        # Get upper and lower confidence bounds (2 standard deviations from the mean)
        lower, upper = posterior.mvn.confidence_region()

        # Plot training points as black crosses
        ax.plot(x_train, y_train, "kx", mew=2, label="observed data")

        # Plot posterior mean as blue line
        ax.plot(x_test.cpu().numpy()[:, 0], posterior.mean.cpu().numpy(), label="posterior mean")

        # Shade between the lower and upper confidence bounds
        ax.fill_between(
            x_test.cpu().numpy()[:, 0],
            lower.cpu().numpy(),
            upper.cpu().numpy(),
            alpha=0.2,
            label="confidence interval (2σ)",
        )

        # Sample from the posterior
        for i in range(10):
            sample = model.posterior(x_test).rsample()
            ax.plot(x_test.cpu().numpy()[:, 0], sample.cpu().numpy().squeeze(), "C0", linewidth=0.3)

    plt.tight_layout()
    return ax
