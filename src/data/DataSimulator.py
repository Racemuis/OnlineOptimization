from typing import Union, Callable, Optional

import numpy as np

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .Reader import Reader
from ..utils.base import Source


# Parameters to optimize
#  Shrinkage (float) ∈ [0, 1]  <-
#  Decimation (int) ∈ [0, 150] <-
#  Baseline correction (float) ∈ [-0.2, 0]      (lower bound)
#  Baseline correction (float) ∈ [-0.1, 0.1]    (upper bound)
#  Frequency filtering (float) ∈ [0, 3]         (lower bound)
#  Frequency filtering (float) ∈ [10, 20]       (upper bound)


class DataSimulator(Source):
    """A class that can be used to simulate BCI data."""

    def __init__(
        self,
        data_config: dict,
        bo_config: dict,
        noise_function: Optional[Callable[[np.ndarray], np.ndarray]],
        n_intervals: int = 5,
    ):
        """

        Args:
            data_config (dict): The data configuration file.
            bo_config (dict): The optimization configuration file, containing:
                - experiment (str): The name of the experiment (matches with a key in the data_config).
                - participant (Optional[str]): The participant to read. If None is provided, all participants are read.
                    Default = None.
                - condition (Optional[str]): The condition to read. If None is provided, all conditions are read.
                    Default = None.
                - dimension (int): The dimensionality of the optimization problem.
                    Only 0 < dimension <= 3 are supported.
            noise_function (Callable[[np.ndarray], np.ndarray]): The noise function describing the scale of the Gaussian
             distribution that is superimposed on the simulated data.
            n_intervals (int): The number of temporal intervals that should be averaged over within the epoch.
        """
        assert bo_config['dimension'] <= 3 or bo_config['dimension'] == 7, \
            "Only the dimensionalities of 1, 2 and 3 are supported."
        super().__init__()
        self.data_config = data_config
        self.experiment = bo_config['experiment']
        self.participant = bo_config['participant']
        self.condition = bo_config['condition']
        self._noise_function = noise_function
        self._dimension = bo_config['dimension']
        self.n_intervals = n_intervals

        self.reader = Reader(self.experiment)
        print("Reading data...", end="", flush=True)
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

    def sample(
        self, x: np.ndarray, info: bool = False, noise: bool = True, cv: bool = False
    ) -> Union[float, np.ndarray]:
        """
        Sample the objective function for the value 'x'.

        Args:
            x (Union[float, np.ndarray]): The value that is used as an input for the objective function.
            info (bool): True if the (noisy) value of the objective function should be printed.
            noise (bool): True if noise should be superimposed on the sampled value of the objective function.
                          Default: True
            cv (bool): True if cross validation should be used to calculate the AUC. Default: False

        Returns:
            Union[float, np.ndarray]: The value of the (noisy) objective function at `x`.
        """
        if not x.shape == (1, 1) and x.squeeze().shape[0] != self.dimension:
            results = np.zeros(x.shape[0])
            for i, sample in enumerate(x):
                results[i] = self.single_sample(x=sample, noise=noise, cv=cv)
            return results
        return self.single_sample(x=x, noise=noise, cv=cv)

    def single_sample(self, x: np.ndarray, noise: bool, cv: bool) -> float:
        """
        Take a single sample of the objective function for the value of `x`.

        Args:
            x (Union[float, np.ndarray]): The value that is used as an input for the objective function.
            noise (bool): True if noise should be superimposed on the sampled value of the objective function.
            cv (bool): True if cross validation should be used to calculate the AUC. Default: False

        Returns:
            Union[float, np.ndarray]: The value of the (noisy) objective function at `x`.

        """
        if self.dimension == 1:
            shrinkage = x
            boundaries = np.array([0.1, 0.17, 0.23, 0.3, 0.41, 0.5])

        elif self.dimension == 2:
            shrinkage, temporal_start = x.squeeze()
            boundaries = np.array([temporal_start, 0.17, 0.23, 0.3, 0.41, 0.5])

        elif self.dimension == 3:
            shrinkage, temporal_start, temporal_interval = x.squeeze()
            boundaries = np.zeros(self.n_intervals + 1)
            boundaries[0] = temporal_start
            boundaries[1:] = temporal_interval
            boundaries = np.cumsum(boundaries, axis=0)

        elif self.dimension == 7:
            shrinkage, temporal_start, iv1, iv2, iv3, iv4, iv5 = x.squeeze()
            boundaries = np.array([temporal_start, iv1, iv2, iv3, iv4, iv5])
            boundaries = np.cumsum(boundaries, axis=0)

        else:
            raise NotImplementedError(f"The chosen dimension {self.dimension} is not supported.")

        epoch = self.epoch_dict[self.participant][self.condition]

        x_train = self.reader.average_temporal_intervals(epoch, boundaries=boundaries)
        x_train = x_train.reshape((-1, epoch.info["nchan"] * self.n_intervals))
        y_train = epoch.events[:, 2]

        # Initialize classifier with estimated shrinkage parameter
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=self.get_item(shrinkage))

        # Simulate noise by taking random subsets of the training data
        if noise and self.noise_function is None:
            x_sampled, _, y_sampled, _ = train_test_split(x_train, y_train, train_size=450)
            x_train = x_sampled
            y_train = y_sampled

        if cv:
            # Evaluate classifier, save average score over folds
            f_x = cross_val_score(lda, x_train, y_train, cv=5, scoring="roc_auc").mean().squeeze()
        else:
            # Evaluate classifier over a single fold
            x_train, x_test, y_train, y_test = train_test_split(x_train, y_train)
            lda.fit(x_train, y_train)
            f_x = roc_auc_score(y_test, lda.predict_proba(x_test)[:, 1])

        # if self.noise_function is not None:
        #     noise_scale = self.noise_function(shrinkage)
        #     if noise and noise_scale > 0.0:
        #         y_x = max(0, min(1, np.random.normal(loc=f_x, scale=noise_scale)))
        #         return y_x
        return f_x

    def get_paper_score(self):
        """
        Calculate the AUC score that can be associated with the parameter settings that have been used in [1].

        [1] Sosulski, J., & Tangermann, M. (2022). Introducing block-Toeplitz covariance matrices to remaster linear
        discriminant analysis for event-related potential brain-computer interfaces. Journal of neural engineering,
        19(6), 10.1088/1741-2552/ac9c98. https://doi.org/10.1088/1741-2552/ac9c98

        Returns:
            float: the AUC score that can be associated with the parameter settings that have been used in [1]
        """
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
        """
        The domain of the simulated objective function.

        Returns:
            np.ndarray: The domain of the simulated objective function.
        """
        domains = np.array(
            [
                [0, 1],  # shrinkage
                [0, 0.1],  # temporal averaging start
                [0.03, 0.4 / self.n_intervals],  # temporal averaging interval
                [0.03, 0.4 / self.n_intervals],  # temporal averaging interval
                [0.03, 0.4 / self.n_intervals],  # temporal averaging interval
                [0.03, 0.4 / self.n_intervals],  # temporal averaging interval
                [0.03, 0.4 / self.n_intervals],  # temporal averaging interval
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
        """
        Unwrap numpy arrays.

        Args:
            x (Union[float, np.ndarray]): an input value that is possibly wrapped in multiple input arrays.

        Returns:
            float: The unwrapped item.
        """
        while isinstance(x, np.ndarray):
            x = x[0]
        return x
