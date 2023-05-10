from enum import Enum
from src import optimization
from src.models import trees, gaussian_processes


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
    NONE = None


initializers = {
    "random": optimization.initializers.Random,
    "sobol": optimization.initializers.Sobol,
}

    
replicators = {
    "max": optimization.replicators.MaxReplicator(),
    "sequential": optimization.replicators.SequentialReplicator(horizon=2)
}


regression_models = {
    "Random forest regression": trees.RandomForestWrapper(n_estimators=10, random_state=44),
    "Gaussian process regression": gaussian_processes.MostLikelyHeteroskedasticGP(normalize=False),
    "Random sampling": None,
}


selectors = {
    "variance": optimization.selectors.VarianceSelector,
    "posterior": optimization.selectors.AveragingSelector,
}


