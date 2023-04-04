from typing import Any, Union, Callable, Optional

import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .Reader import Reader
from ..utils.base import Source


# TODO: parameters to optimize
#  Shrinkage (float) ∈ [0, 1]  <-
#  Decimation (int) ∈ [0, 150] <-
#  Baseline correction (float) ∈ [-0.2, 0]      (lower bound)
#  Baseline correction (float) ∈ [-0.1, 0.1]    (upper bound)
#  Frequency filtering (float) ∈ [0, 3]         (lower bound)
#  Frequency filtering (float) ∈ [10, 20]       (upper bound)


class DataSimulator(Source):
    def __init__(
        self,
        data_config: dict,
        experiment: str,
        participant: str,
        condition: str,
        noise_function: Optional[Callable[[np.ndarray], np.ndarray]],
        dimension: int,
        n_intervals: int = 5,
    ):
        assert dimension <= 3, "Only the dimensionalities of 1, 2 and 3 are supported."
        super().__init__()
        self.data_config = data_config
        self.experiment = experiment
        self.participant = participant
        self.condition = condition
        self._noise_function = noise_function
        self._dimension = dimension
        self.n_intervals = n_intervals

        self.reader = Reader(self.experiment)
        print("Reading data...", end='', flush=True)
        self.epoch_dict = self.reader.read(
            data_config=self.data_config, participant=self.participant, condition=self.condition, verbose=False
        )
        print(" done!")

    @property
    def dimension(self):
        return self._dimension

    @property
    def noise_function(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._noise_function

    @property
    def objective_function(self):
        return None

    def sample(self, x: np.ndarray, info: bool = False, noise: bool = True) -> Union[float, np.ndarray]:
        if not x.shape == (1, 1) and x.squeeze().shape[0] != self.dimension:
            results = np.zeros(x.shape[0])
            for i, sample in enumerate(x):
                results[i] = self.single_sample(x=sample, noise=noise)
            return results
        return self.single_sample(x=x, noise=noise)

    def single_sample(self, x: np.ndarray, noise: bool) -> float:
        if self.dimension == 1:
            shrinkage = x
            boundaries = np.array([0.1, 0.17, 0.23, 0.3, 0.41, 0.5])
            # shrinkage = "auto"
            # boundaries = np.array([self.get_item(x), 0.17, 0.23, 0.3, 0.41, 0.5])

        elif self.dimension == 2:
            shrinkage, temporal_start = x.squeeze()
            boundaries = np.array([temporal_start, 0.17, 0.23, 0.3, 0.41, 0.5])

        elif self.dimension == 3:
            shrinkage, temporal_start, temporal_interval = x.squeeze()
            boundaries = np.zeros(self.n_intervals + 1)
            boundaries[0] = temporal_start
            boundaries[1:] = temporal_interval
            boundaries = np.cumsum(boundaries, axis=0)

        else:
            raise NotImplementedError(f"The chosen dimension {self.dimension} is not supported.")

        epoch = self.epoch_dict[self.participant][self.condition]

        x_train = self.reader.average_temporal_intervals(epoch, boundaries=boundaries)
        x_train = x_train.reshape((-1, epoch.info["nchan"] * self.n_intervals))
        y_train = epoch.events[:, 2]

        # Initialize classifier with estimated shrinkage parameter
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=self.get_item(shrinkage))

        # Evaluate classifier, save average score over folds
        f_x = cross_val_score(lda, x_train, y_train, cv=5, scoring="roc_auc").mean().squeeze()

        noise_scale = self.noise_function(shrinkage)

        if noise and noise_scale > 0.0:
            y_x = max(0, min(1, np.random.normal(loc=f_x, scale=noise_scale)))
            return y_x
        return f_x

    def get_paper_score(self):
        epoch = self.epoch_dict[self.participant][self.condition]
        x_train = self.reader.average_temporal_intervals(epoch, boundaries=np.array([0.1, 0.17, 0.23, 0.3, 0.41, 0.5]))
        x_train = x_train.reshape((-1, epoch.info["nchan"] * self.n_intervals))
        y_train = epoch.events[:, 2]

        # Initialize classifier with estimated shrinkage parameter
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

        # Evaluate classifier, save average score over folds
        f_x = cross_val_score(lda, x_train, y_train, cv=5, scoring="roc_auc").mean().squeeze()

        return f_x

    def get_domain(self) -> np.ndarray:
        domains = np.array(
            [
                [0, 1],  # shrinkage
                [0, 0.1],  # temporal averaging start
                [0.03, 0.4/self.n_intervals],  # temporal averaging interval
                [1, 25],  # decimation
                [-0.2, 0],  # baseline correction (lower bound)
                [-0.1, 0.1],  # baseline correction (upper bound)
                [0, 3],  # frequency filtering (lower bound)
                [10, 20],  # frequency filtering (upper bound)
            ]
        )
        return domains[: self.dimension].transpose()

    @staticmethod
    def get_item(x: Union[float, np.ndarray]) -> float:
        while isinstance(x, np.ndarray):
            x = x[0]
        return x
