from typing import Callable, Type, Optional, Union

from torch import Tensor
from botorch.models.model import Model
from botorch.acquisition import AcquisitionFunction


def curry(
    acquisition_function: Type[AcquisitionFunction], **kwargs: Union[bool, float]
) -> Union[Callable[[Model], AcquisitionFunction], Callable[[Tensor, Model], AcquisitionFunction]]:
    """
    Function that curries an acquisition function such that the acquisition function can be initialized later on, when
    the parameters of the acquisition function are available.

    `I too, consider this a horrible, hacky solution`.

    Args:
        acquisition_function (Type[Acquisition]): An uninitialized botorch acquisition function.
        **kwargs (dict): The arguments of the acquisition function that are already known, such as the beta for the UCB.

    Returns:
        Union[Callable[[Model], AcquisitionFunction], Callable[[Tensor, Model], AcquisitionFunction]]:
                A curried instance of the acquisition function that can be further
                initialized later on.
    """

    def init_acquisition_function(model: Model, x_train: Optional[Tensor] = None) -> AcquisitionFunction:
        if x_train is not None:  # For acquisition functions such as NEI that take the data as input.
            acquisition = acquisition_function(model, X_observed=x_train)
        else:  # For acquisition functions such as UCB, that just take args that are known beforehand.
            acquisition = acquisition_function(model, **kwargs)
        return acquisition

    return init_acquisition_function


def uncurry(
    curried_acquisition: Union[Callable[[Model], AcquisitionFunction], Callable[[Tensor, Model], AcquisitionFunction]],
    model: Model,
    x_train: Tensor,
):
    """
    Initialize the acquisition function by uncurrying the curried `acquisition function` function.

    Args:
        curried_acquisition (Union[Callable[[Model], AcquisitionFunction],
        Callable[[Tensor, Model], AcquisitionFunction]]): A curried instance of the acquisition function that can be
                                                          further initialized later on.
        model (Model): A fitted botorch model.
        x_train (Tensor): The x-coordinates of the training data.

    Returns:

    """
    try:
        acquisition = curried_acquisition(model)
    except TypeError:
        acquisition = curried_acquisition(x_train=x_train, model=model)
    return acquisition
