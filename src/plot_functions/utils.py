from typing import Callable, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import cm, colors, lines
import matplotlib
from cycler import cycler, Cycler


from gpytorch.models.exact_gp import ExactGP

from ..simulation.ObjectiveFunction import ObjectiveFunction

# from ..simulation.Simulator import Simulator


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
    domain: np.ndarray, objective_function: ObjectiveFunction, simulator, num: int = 100
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


def plot_GP_1d(
    x: np.ndarray,
    y: np.ndarray,
    x_test: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_std: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Optional[str] = None,
    r_x: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot posterior distribution of a 1-dimensional Gaussian Process.

    Args:
        x (np.ndarray): The x-coordinates of the observed datapoints.
        y (np.ndarray): The y-coordinates of the observed datapoints.
        posterior_mean: The mean of the GP posterior.
        posterior_std: The standard deviation of the GP posterior.
        x_test (np.ndarray): The x-coordinates corresponding to the posterior mean and std (and r(x)).
        title (str): The title of the figure.
        xlabel (str): The x label.
        ylabel (str): The y label.
        path (Optional[str]): The destination path if the plot needs to be saved. If None, the plot is not saved.
        r_x (Optional[np.ndarray]): The estimated variance of the objective function.
                                    If None, it is not included in the figure.
        ax: (plt.Axes): The axis whereon the plot should be created. If None, a new axis is created.

    Returns:
        plt.Axes: The plotted ax.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(x, y, c="tab:orange", label="Observed data", zorder=10)
    ax.plot(x_test, posterior_mean, label=r"m(x)", zorder=20)
    ax.fill_between(
        x_test,
        posterior_mean - 1.96 * posterior_std,
        posterior_mean + 1.96 * posterior_std,
        alpha=0.2,
        label=r"95% confidence interval",
        zorder=2,
    )

    if r_x is not None:
        ax.fill_between(
            x_test, posterior_mean - r_x, posterior_mean + r_x, alpha=0.6, color="gray", label=r"$r(x)$", zorder=0,
        )
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plt.legend()

    if path is not None:
        plt.savefig(path, bbox_inches="tight")
    return ax


def plot_simulation_results(
    distances: np.ndarray,
    n_informed_samples: int,
    n_random_samples: int,
    n_runs: int,
    noise_functions: dict,
    objective_key: str,
    regression_key: str,
    colors: Cycler = matplotlib.rcParams["axes.prop_cycle"],
    lines: Cycler = cycler("linestyle", ["-", "--", ":", "-."]),
) -> None:
    """
    Visualise the result of the simulations by plotting the distances to the optimum against the number of samples taken
    by the modules algorithm. Create a boxplot to show the distribution of the final outcome of the modules
    algorithm.

    Args:
        distances (np.ndarray): The distances to the optimum. Shape: [len(noise_functions), n_runs, sample_size].
        n_informed_samples (int): The number of informed samples that have been sampled.
        n_random_samples (int): The number of random samples that have been sampled.
        n_runs (int): The number of runs that the results should be averaged over.
        noise_functions (dict): The dictionary containing the noise functions.
        objective_key (str): The string indicating the objective function.
        regression_key (str): The string indicating the regression model.
        colors (Cycler): A color cycler.
        lines (Cycler): A linestyle cycler.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2)
    for dist, key, c, ls in zip(distances, noise_functions.keys(), colors, lines):
        mean = np.mean(dist, axis=0).squeeze()
        axes[0].plot(
            range(n_random_samples + n_informed_samples), mean, color=c["color"], label=key, linestyle=ls["linestyle"]
        )
    axes[0].legend(title="Noise type")
    axes[0].set_xlabel("Number of samples")
    axes[0].set_ylabel("Euclidean distance")
    axes[0].set_title(f"Average distance between the estimated and true optimum")
    axes[0].set_xticks(ticks=np.arange(0, n_random_samples + n_informed_samples, 5))
    boxplot_data = [distances[i, :, -1].flatten() for i in range(len(noise_functions))]
    axes[1].boxplot(boxplot_data)
    axes[1].set_xlabel("Noise function")
    axes[1].set_ylabel("Euclidean distance")
    axes[1].set_xticklabels(noise_functions.keys())
    axes[1].set_title(f"Eventual proposals by the modules algorithm")
    fig.suptitle(
        f"{objective_key} - {regression_key}\nrandom samples: {n_random_samples}, "
        f"informed samples: {n_informed_samples}"
        f", number of runs: {n_runs}",
        fontsize=14,
    )
    plt.show()
