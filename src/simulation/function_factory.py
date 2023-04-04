from typing import Union, List
import numpy as np

from .ObjectiveFunction import ObjectiveFunction


def sinusoid_and_log(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    A function implementing |sin(x) + log(x+1)|.

    Args:
        x (float): the function parameter.

    Returns:
        float: the function value for x.
    """
    return np.abs(np.sin(x) + np.log(x + 1))


def cosine_scale(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    A function implementing: exp(cos(x))

    Args:
        x (float): the function parameter.

    Returns:
        float: the function value for x.
    """
    return np.exp(np.cos(x))


def flat(x: Union[float, np.ndarray], offset: float = 1.0) -> Union[float, np.ndarray]:
    """
    A flat function that implements ``f(x) = offset``.
    Args:
        x (float): the function parameter.
        offset (float): the evaluated value of the function.

    Returns:
        Union[float, np.ndarray]: the evaluated function value.
    """
    try:
        return np.ones(x.shape[0]) + offset
    except AttributeError:
        return offset


def linear(x: Union[float, np.ndarray], scalar: float = 0.1) -> Union[float, np.ndarray]:
    """
    A linear function that implements ``f(x) = scalar * x``.
    Args:
        x (float): the function parameter.
        scalar (float): the scalar in the function.

    Returns:
        Union[float, np.ndarray]: the evaluated function value.
    """
    return scalar * x


def ackley_function(
    x: Union[float, np.ndarray], a: float = 20, b: float = 0.2, c: float = 2 * np.pi, complement: bool = True
) -> Union[float, np.ndarray]:
    """
    An optimization benchmark function proposed by David Ackley [1].

    Common input domain: x_i ∈ [-32.768, 32.768], though smaller cubes are also possible.

    Optimum: x^* = [0, ..., 0]

    [1] David H. Ackley. 1987. A connectionist machine for genetic hill climbing. Kluwer Academic Publishers, USA.

    Args:
        x (Union[float, np.ndarray]): the function parameter
        a (float): a parameter in the ackley function.
        b (float): a parameter in the ackley function.
        c (float): a parameter in the ackley function.
        complement (bool): True if the function needs to be negated. Default = True for a global maximum.

    Returns:
        Union[float, np.ndarray]: the function value for x.
    """
    sign = -1 if complement else 1
    return sign * (
        -a * np.exp(-b * np.sqrt(np.mean(np.power(x, 2), axis=-1)))
        - np.exp(np.mean(np.cos(c * x), axis=-1))
        + a
        + np.exp(1)
    )


class BraninFunction(ObjectiveFunction):
    """
    ObjectiveFunction wrapper for the Branin function.
    """
    def __init__(self):
        ObjectiveFunction.__init__(
            self, name="branin", f=self.branin_function, domain=np.array([[-5, 0], [10, 15]], dtype=float)
        )

    def get_maximum(self) -> List[np.ndarray]:
        """
        Returns the maxima of the objective function.

        Returns:
            np.ndarray: The objective of the maximum. Returns 0 if the optimization was unsuccessful.
        """
        return [np.array([-np.pi, 12.275]), np.array([np.pi, 2.275]), np.array([9.42478, 2.475])]

    @staticmethod
    def branin_function(
        x: Union[float, np.ndarray],
        a: float = 1,
        b: float = 5.1 / (4 * np.pi ** 2),
        c: float = 5 / np.pi,
        r: float = 6,
        s: float = 10,
        t: float = 1 / (8 / np.pi),
        complement: bool = True,
    ) -> Union[float, np.ndarray]:
        """
        An optimization benchmark function retrieved from https://www.sfu.ca/~ssurjano/branin.html.

        Common input domain: The square x1 ∈ [-5, 10], x2 ∈ [0, 15].

        Optima: x^* ∈ {(-pi, 12.275), (pi, 2.275), (9.42478, 2.475)}.

        Args:
            x (Union[float, np.ndarray]): the function parameter. x should have the shape [n_samples_x, n_samples_y, 2]
            a (float): a parameter in the branin function.
            b (float): a parameter in the branin function.
            c (float): a parameter in the branin function.
            r (float): a parameter in the branin function.
            s (float): a parameter in the branin function.
            t (float): a parameter in the branin function.
            complement (bool): True if the function needs to be negated. Default = True for a global maximum.

        Returns:
            Union[float, np.ndarray]: the function value for x.
        """
        assert x.shape[-1] == 2, f"x should have the dimension [n_samples_x, n_samples_y, 2], received {x.shape}."
        sign = -1 if complement else 1
        return sign * (
            a * np.power(x[..., 1] - b * np.power(x[..., 0], 2) + c * x[..., 0] - r, 2)
            + s * (1 - t) * np.cos(x[..., 0])
            + s
        )
