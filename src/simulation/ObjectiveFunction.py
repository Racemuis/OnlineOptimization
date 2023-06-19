from dataclasses import dataclass
from typing import Callable, Union, Optional

import scipy
import numpy as np


class ObjectiveFunction:
    """An objective function class"""
    def __init__(
        self,
        name: str,
        f: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
        domain: Optional[np.ndarray] = np.array([0, 10], dtype=float),
    ):
        self.name = name
        self.f = f
        self.domain = domain
        self.dimension = self.domain.shape[-1]

    def evaluate(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the value of the objective function given an input value `x`.

        Args:
            x (Union[float, np.ndarray]): The input value that is evaluated.

        Returns:
            (Union[float, np.ndarray]): The value of the objective function at x, can be a float or a numpy array.
        """
        return self.f(x)

    def get_maximum(self) -> float:
        """
        Calculate the maximum of the objective function.

        Returns:
            float: The objective of the maximum. Returns 0 if the modules was unsuccessful.
        """
        bounds = scipy.optimize.Bounds(lb=self.domain[0], ub=self.domain[1])
        naive_guess = self.domain[1] - self.domain[0]
        opt_result = scipy.optimize.minimize(fun=lambda x: -self.f(x), x0=naive_guess, bounds=bounds)

        if opt_result.success:
            return opt_result.x
        print(f"Optimization ended unsuccessfully with termination message:\n{opt_result.message}")
        return 0

    def __str__(self):
        return f"Objective function '{self.name}':\ndomain:\t{list(self.domain)}\noptimum:{self.get_maximum()}"
