import yaml
import numpy as np

from src.utils.wrap_acqf import curry
from src.models.trees import RandomForestWrapper
from src.models.gaussian_processes import MostLikelyHeteroskedasticGP
from src.optimization.pipelines import BayesOptPipeline
from src.optimization.selectors import SimpleSelector
from src.optimization.initializers import Random
from src.optimization.acquisition_functions import BoundedUpperConfidenceBound
from src.simulation import ObjectiveFunction, Simulator, function_factory
from src.data.DataSimulator import DataSimulator

import warnings

warnings.filterwarnings(
    "ignore", message="Input data is not contained to the unit cube. Please consider min-max scaling the input data."
)
warnings.filterwarnings(
    "ignore",
    message="Input data is not standardized. Please consider scaling the input to zero mean and unit variance.",
)


def main():
    np.random.seed(44)
    random_sample_size = 5
    informed_sample_size = 10

    # create simulator
    with open("src/conf/bo_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # create "unknown" objective function and simulator
    objective = ObjectiveFunction.ObjectiveFunction(
        name="Sinusoid and Log", f=function_factory.sinusoid_and_log, domain=np.array(config["domain"], dtype=float)
    )
    simulator = Simulator.Simulator(objective_function=objective, noise_function=function_factory.cosine_scale)

    # create pipeline
    pipe = BayesOptPipeline(
        initialization=Random(domain=np.array(config["domain"], dtype=float)),
        # regression_model=RandomForestWrapper(n_estimators=100, max_depth=4, random_state=0),
        regression_model=MostLikelyHeteroskedasticGP(normalize=False),
        replicator=None,
        acquisition=curry(BoundedUpperConfidenceBound, beta=0.7),
        # acquisition=curry(NoisyExpectedImprovement),
        selector=SimpleSelector,
    )

    # optimize the unknown objective function
    presumed_x_max = pipe.optimize(
        source=simulator, random_sample_size=random_sample_size, informed_sample_size=informed_sample_size, plot=True
    )
    print(f"Presumed optimum: {presumed_x_max[0].item():.2f}\nTrue optimum: {objective.get_maximum()[0]:.2f}")
    print(f"Difference: {np.abs(presumed_x_max[0].item() - objective.get_maximum()[0]):.2f}")


if __name__ == "__main__":
    main()
