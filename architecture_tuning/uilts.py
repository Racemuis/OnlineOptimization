import numpy as np


def append_locations(path: str, run: int, location: np.ndarray) -> None:
    """
    Append the locations of `run` to the file found at `path`.

    Args:
        path (str): The path to the results file.
        run (int): The index of the run.
        location (np.ndarray): The array representing the (multidimensional) location.

    Returns:
        None
    """
    with open(path, "a") as f:
        results = f"{run},"
        for cat in location:
            results += f"{cat},"
        f.write(results + "\n")
