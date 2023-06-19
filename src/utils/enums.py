from enum import Enum
from src import modules
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


initializers = {
    "random": modules.initializers.Random,
    "sobol": modules.initializers.Sobol,
}

    
replicators = {
    "fixed_n": modules.replicators.FixedNReplicator(n_replications=5),
    "max": modules.replicators.MaxReplicator(),
    "sequential": modules.replicators.SequentialReplicator(horizon=2)
}


regression_models = {
    "Random forest regression": trees.RandomForestWrapper(n_estimators=10, random_state=44),
    "Gaussian process regression": gaussian_processes.MostLikelyHeteroskedasticGP(normalize=False),
    "Random sampling": None,
}


selectors = {
    "variance": modules.selectors.VarianceSelector,
}


