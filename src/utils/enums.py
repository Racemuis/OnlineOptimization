from typing import Type

from enum import Enum
from src import modules
from src.utils import base
from src.modules.models import gaussian_processes, trees


class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class ConvergenceMeasure(ExtendedEnum):
    """
    Enumerate class for the different convergence measures.
    """
    LENGTH_SCALE = 'length_scale'
    MSE = 'mse'
    NOISE_UNCERTAINTY = 'noise_uncertainty'
    MODEL_UNCERTAINTY = 'model_uncertainty'
    NOISE_R = 'noise_r'
    NONE = "None"


def initializers(i: str) -> Type[base.Initializer]:
    d = {
        "random": modules.initializers.Random,
        "sobol": modules.initializers.Sobol,
    }
    return d[i]


def replicators(r: str) -> base.Replicator:
    d = {
        "fixed_n": modules.replicators.FixedNReplicator(n_replications=5),
        "max": modules.replicators.MaxReplicator(),
        "sequential": modules.replicators.SequentialReplicator(horizon=2),
        "None": None,
    }
    return d[r]


def regression_models(rm: str) -> base.RegressionModel:
    d = {
        "Random forest regression": trees.RandomForestWrapper(n_estimators=10, random_state=44),
        "Gaussian process regression": gaussian_processes.MostLikelyHeteroskedasticGP(normalize=True),
        "Random sampling": None,
    }
    return d[rm]


def selectors(s: str) -> Type[base.Selector]:
    d = {
        "variance": modules.VarianceSelector,
    }
    return d[s]


def acquisition(a: str) -> Type[base.AcquisitionFunction]:
    d = {
        "variance": modules.BoundedUpperConfidenceBoundVar,
        "ucb": modules.BoundedUpperConfidenceBound
    }
    return d[a]


