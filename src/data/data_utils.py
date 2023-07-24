from typing import Union, Optional

from src.simulation import Simulator
from src.data.ERPSimulator import DataSimulator
from src.data.CVEPSimulator import CVEPSimulator


def get_simulator(
    conf: dict, data_config: dict, noise_function: Optional[bool] = False, augmentation: Optional[bool] = False
) -> Union[CVEPSimulator, DataSimulator]:
    """
    Create a simulator instance given the selected dataset in the configuration file.

    Args:
        conf (dict): The dictionary containing the contents of the main configuration file.
        data_config (dict): The dictionary containing the contents of the data configuration file.
        noise_function (Callable[[np.ndarray], np.ndarray]): The noise function describing the scale of the Gaussian
             distribution that is superimposed on the simulated data.
        augmentation (Optional[bool]): Add augmentation of a sinusoid to the objective function.
    Returns:
        Simulator: the data simulator.
    """
    if conf["experiment"] == "auditory_aphasia":
        simulator = DataSimulator(
            data_config=data_config, bo_config=conf, noise_function=noise_function, augmentation=augmentation
        )

    elif conf["experiment"].upper() == "CVEP":
        simulator = CVEPSimulator(data_config=data_config, bo_config=conf, trial=True,)

    else:
        print(f"The chosen experiment \"{conf['experiment']}\" is not supported, defaulting to \"auditory_aphasia\"")
        simulator = DataSimulator(
            data_config=data_config, bo_config=conf, noise_function=noise_function, augmentation=augmentation
        )
    return simulator
