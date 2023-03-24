from typing import Callable, Union

from .ObjectiveFunction import ObjectiveFunction
from ..plot_functions import utils
from ..utils.base import Source

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class Simulator(Source):
    """Simulator of an unknown objective function."""

    def __init__(self, objective_function: ObjectiveFunction, noise_function: Callable):
        super().__init__()
        self._objective_function = objective_function
        self._noise_function = noise_function
        self._dimension = (
            self._objective_function.domain.shape[-1] if len(self._objective_function.domain.shape) > 1 else 1
        )

    @property
    def dimension(self):
        return self._dimension

    @property
    def noise_function(self):
        return self._noise_function

    @property
    def objective_function(self):
        return self._objective_function

    def simulate(self, n_simulations: int) -> None:
        """
        Plot an animation of a simulated sampling strategy.

        Args:
            n_simulations (int): The number of samples to simulate.

        Returns:
            None
        """
        domain = self.objective_function.domain
        fig, ax, color = utils.setup_simulation_plot(
            n_simulations=n_simulations, objective_function=self.objective_function, cmap=cm.Reds,
        )
        for i, c in enumerate(color):
            sample_x = np.random.uniform(low=domain[0], high=domain[1])
            sample_y = self.sample(x=sample_x)
            ax.scatter(x=sample_x, y=sample_y, color=c, marker="X", edgecolors="black")
            plt.draw()
            plt.pause(0.75)
        plt.show()

    def sample(self, x: Union[float, np.ndarray], info: bool = False) -> Union[float, np.ndarray]:
        """
        Sample the objective function for the value 'x'.

        Args:
            x (Union[float, np.ndarray]): The value that is used as an input for the objective function.
            info (bool): True if the (noisy) value of the objective function should be printed.

        Returns:
            Union[float, np.ndarray]: The value of the (noisy) objective function at `x`.
        """
        f_x = self.objective_function.evaluate(x=x)
        scale_x = x[:, 0] if self._dimension > 1 else x
        scale = np.abs(self.noise_function(scale_x))
        y_x = np.random.normal(loc=f_x, scale=scale)
        if info:
            print(f"Evaluated: x = {x}\n\tf(x) = {f_x}\n\ty(x) = {y_x}")
        return y_x

    def get_domain(self) -> np.ndarray:
        """
        The domain of the simulated objective function.

        Returns:
            np.ndarray: The domain of the simulated objective function.
        """
        return self.objective_function.domain

    def __str__(self) -> str:
        return f"Simulator of an unknown objective function '{self.objective_function.name}' with noise."
