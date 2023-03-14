from typing import Union
import numpy as np


def sinusoid_and_log(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    A function implementing |sin(x) + log(x+1)|.

    Args:
        x (float): the function parameter.

    Returns:
        float: the function value for x.
    """
    return np.abs(np.sin(x) + np.log(x+1))


def cosine_scale(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    A function implementing: exp(cos(x))

    Args:
        x (float): the function parameter.

    Returns:
        float: the function value for x.
    """
    return np.exp(np.cos(x))
