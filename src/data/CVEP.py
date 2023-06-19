import os
from typing import Union, List
import numpy as np
import pynt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import train_test_split

from ..utils.base import Source, Callable, Optional


class CVEPReader:
    def __init__(self):
        self.experiment = "CVEP"

    def load_data(self, conf: dict):
        subject = conf[self.experiment]["participant_list"][0]
        fn = os.path.join(conf[self.experiment]["data_path"], "derivatives", "offline", subject, f"{subject}_gdf.npz")
        tmp = np.load(fn)
        X = tmp["X"]
        y = tmp["y"]
        V = tmp["V"]
        fs = tmp["fs"]

        # Read cap file
        channels = []
        capfile = os.path.join(conf[self.experiment]["pynt_path"], "pynt", "capfiles", "nt_cap8.loc")
        fid = open(capfile, "r")
        for line in fid.readlines():
            channels.append(line.split("\t")[-1].strip())

        return X, y, V, fs, channels


class CVEPSimulator(Source):
    def __init__(
        self, data_config: dict, bo_config: dict, trial: bool = True,
    ):
        """
        Args:
            data_config (dict): The data configuration file.
            bo_config (dict): The modules configuration file, containing:
                - experiment (str): The name of the experiment (matches with a key in the data_config).
                - participant (Optional[str]): The participant to read. If None is provided, all participants are read.
                   Default = None.
                - dimension (int): The dimensionality of the modules problem.
                   Only 0 < dimension <= 3 are supported.
            trial (bool): True if the accuracy of the trial level should be calculated.
        """
        super().__init__()
        self.trial = trial
        self.data_config = data_config
        self.experiment = bo_config["experiment"]
        self.participant = bo_config["participant"]
        self._dimension = bo_config["dimension"]

        self.reader = CVEPReader()
        self.x_train, self.y_train, self.V, self.fs, self.channels = self.reader.load_data(conf=data_config)
        self.weights = None

    @property
    def dimension(self):
        return self._dimension

    @property
    def noise_function(self) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        return None

    @property
    def objective_function(self):
        return None

    def clear_weights(self) -> None:
        self.weights = None

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
            epoch_multiplier = 0.3

        elif self.dimension == 2:
            shrinkage, epoch_multiplier = x.squeeze()

        else:
            raise NotImplementedError(f"The chosen dimension {self.dimension} is not supported.")

        if self.weights is None:
            self.weights = self.train_CCA(epoch_multiplier=epoch_multiplier, noise=noise, cv=cv)

        # Set trial duration
        n_samples = int(4.2 * self.fs)

        # Set epoch size
        epoch_size = int(np.ceil(epoch_multiplier * self.fs))
        step_size = int(1 / 60 * self.fs)

        # Set up codebook for trial classification
        n = int(np.ceil(n_samples / self.V.shape[0]))
        _V = np.tile(self.V, (n, 1)).astype("float32")[: n_samples - epoch_size: step_size]

        # Initialize classifier with estimated shrinkage parameter
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=self.get_item(shrinkage))

        # Simulate noise by taking random subsets of the training data
        if noise and self.noise_function is None:
            x_sampled, _, y_sampled, _ = train_test_split(
                self.x_train, self.y_train, train_size=int(np.floor(3 * self.x_train.shape[0] / 4))
            )
        else:
            x_sampled = self.x_train
            y_sampled = self.y_train

        if cv:
            # Setup cross-validation
            n_folds = 5
            folds = np.repeat(np.arange(n_folds), x_sampled.shape[0] / n_folds)

            # Loop folds
            accuracy_epoch = np.zeros(n_folds)
            accuracy_trial = np.zeros(n_folds)
            for i_fold in range(n_folds):
                # Split data to train and valid set
                X_trn, y_trn = x_sampled[folds != i_fold, :, :n_samples], y_sampled[folds != i_fold]
                X_tst, y_tst = x_sampled[folds == i_fold, :, :n_samples], y_sampled[folds == i_fold]

                # Slice trials to epochs
                X_sliced_trn, y_sliced_trn = pynt.utilities.trials_to_epochs(
                    X_trn, y_trn, self.V, epoch_size, step_size
                )
                X_sliced_tst, y_sliced_tst = pynt.utilities.trials_to_epochs(
                    X_tst, y_tst, self.V, epoch_size, step_size
                )

                w = self.weights[i_fold]

                # Apply CCA (on epoch level)
                X_sliced_filtered_trn = np.dot(
                    w, X_sliced_trn.transpose((2, 0, 1, 3)).reshape((x_sampled.shape[1], -1))
                ).reshape(-1, epoch_size)
                X_sliced_filtered_tst = np.dot(
                    w, X_sliced_tst.transpose((2, 0, 1, 3)).reshape((x_sampled.shape[1], -1))
                ).reshape(-1, epoch_size)

                # Train LDA (on epoch level)
                # N.B.: spatio-temporal features are flattened
                # N.B.: all epochs of all trials are concatenated
                lda.fit(X_sliced_filtered_trn, y_sliced_trn.flatten())

                # Apply LDA (on epoch level)
                yh_sliced_tst = lda.predict(X_sliced_filtered_tst)

                if self.trial:
                    # Apply LDA (on trial level)
                    ph_tst = lda.predict_proba(X_sliced_filtered_tst)[:, 1]
                    ph_tst = np.reshape(ph_tst, y_sliced_tst.shape)
                    rho = pynt.utilities.correlation(ph_tst, _V.T)
                    yh_tst = np.argmax(rho, axis=1)
                    accuracy_trial[i_fold] = 100 * np.mean(yh_tst == y_tst)
                else:
                    # Compute accuracy (on epoch level)
                    accuracy_epoch[i_fold] = 100 * np.mean(yh_sliced_tst == y_sliced_tst.flatten())

            if self.trial:
                f_x = accuracy_trial.mean()
            else:
                f_x = accuracy_epoch.mean()
            return f_x

        else:
            # Evaluate classifier over a single fold
            X_trn, X_tst, y_trn, y_tst = train_test_split(x_sampled, y_sampled)
            X_trn = X_trn[:, :, :n_samples]
            X_tst = X_tst[:, :, :n_samples]

            # Slice trials to epochs
            X_sliced_trn, y_sliced_trn = pynt.utilities.trials_to_epochs(
                X_trn, y_trn, self.V, epoch_size, step_size
            )
            X_sliced_tst, y_sliced_tst = pynt.utilities.trials_to_epochs(
                X_tst, y_tst, self.V, epoch_size, step_size
            )

            w = self.weights

            # Apply CCA (on epoch level)
            X_sliced_filtered_trn = np.dot(
                w, X_sliced_trn.transpose((2, 0, 1, 3)).reshape((x_sampled.shape[1], -1))
            ).reshape(-1, epoch_size)
            X_sliced_filtered_tst = np.dot(
                w, X_sliced_tst.transpose((2, 0, 1, 3)).reshape((x_sampled.shape[1], -1))
            ).reshape(-1, epoch_size)

            # Train LDA (on epoch level)
            # N.B.: spatio-temporal features are flattened
            # N.B.: all epochs of all trials are concatenated
            lda.fit(X_sliced_filtered_trn, y_sliced_trn.flatten())

            if self.trial:
                # Apply LDA (on trial level)
                ph_tst = lda.predict_proba(X_sliced_filtered_tst)[:, 1]
                ph_tst = np.reshape(ph_tst, y_sliced_tst.shape)
                rho = pynt.utilities.correlation(ph_tst, _V.T)
                yh_tst = np.argmax(rho, axis=1)
                accuracy_trial = 100 * np.mean(yh_tst == y_tst)
                f_x = accuracy_trial

            else:
                # Apply LDA (on epoch level)
                yh_sliced_tst = lda.predict(X_sliced_filtered_tst)

                # Compute accuracy (on epoch level)
                accuracy_epoch = 100 * np.mean(yh_sliced_tst == y_sliced_tst.flatten())
                f_x = accuracy_epoch

        return f_x

    def train_CCA(self, epoch_multiplier: float, noise: bool, cv: bool) -> Union[List[np.ndarray], np.ndarray]:
        # Set trial duration
        n_samples = int(4.2 * self.fs)

        # Set epoch size
        epoch_size = int(np.ceil(epoch_multiplier * self.fs))
        step_size = int(1 / 60 * self.fs)

        # Set up codebook for trial classification
        n = int(np.ceil(n_samples / self.V.shape[0]))
        _V = np.tile(self.V, (n, 1)).astype("float32")[: n_samples - epoch_size: step_size]

        # Setup CCA
        cca = CCA(n_components=1)

        # Simulate noise by taking random subsets of the training data
        if noise and self.noise_function is None:
            x_sampled, _, y_sampled, _ = train_test_split(
                self.x_train, self.y_train, train_size=int(np.floor(3 * self.x_train.shape[0] / 4))
            )
        else:
            x_sampled = self.x_train
            y_sampled = self.y_train

        if cv:
            # Setup cross-validation
            n_folds = 5
            folds = np.repeat(np.arange(n_folds), x_sampled.shape[0] / n_folds)

            weights = []

            # Loop folds
            for i_fold in range(n_folds):
                # Split data to train and valid set
                X_trn, y_trn = x_sampled[folds != i_fold, :, :n_samples], y_sampled[folds != i_fold]

                # Slice trials to epochs
                X_sliced_trn, y_sliced_trn = pynt.utilities.trials_to_epochs(
                    X_trn, y_trn, self.V, epoch_size, step_size
                )

                # Train CCA (on epoch level)
                erp_noflash = np.mean(X_sliced_trn[y_sliced_trn == 0, :, :], axis=0, keepdims=True)
                erp_flash = np.mean(X_sliced_trn[y_sliced_trn == 1, :, :], axis=0, keepdims=True)
                erps = np.concatenate((erp_noflash, erp_flash), axis=0)[y_sliced_trn, :, :]
                cca.fit(
                    X_sliced_trn.transpose((0, 1, 3, 2)).reshape((-1, x_sampled.shape[1])),
                    erps.transpose((0, 1, 3, 2)).reshape(-1, x_sampled.shape[1]),
                )
                w = cca.x_weights_.flatten()

                weights.append(w)
            return weights

        else:
            # Evaluate classifier over a single fold
            X_trn, X_tst, y_trn, y_tst = train_test_split(x_sampled, y_sampled)
            X_trn = X_trn[:, :, :n_samples]

            # Slice trials to epochs
            X_sliced_trn, y_sliced_trn = pynt.utilities.trials_to_epochs(
                X_trn, y_trn, self.V, epoch_size, step_size
            )

            # Train CCA (on epoch level)
            erp_noflash = np.mean(X_sliced_trn[y_sliced_trn == 0, :, :], axis=0, keepdims=True)
            erp_flash = np.mean(X_sliced_trn[y_sliced_trn == 1, :, :], axis=0, keepdims=True)
            erps = np.concatenate((erp_noflash, erp_flash), axis=0)[y_sliced_trn, :, :]
            cca.fit(
                X_sliced_trn.transpose((0, 1, 3, 2)).reshape((-1, x_sampled.shape[1])),
                erps.transpose((0, 1, 3, 2)).reshape(-1, x_sampled.shape[1]),
            )
            w = cca.x_weights_.flatten()
            return w

    def get_paper_score(self):
        # Set trial duration
        n_samples = int(4.2 * self.fs)

        # Set epoch size
        epoch_size = int(0.3 * self.fs)
        step_size = int(1 / 60 * self.fs)

        # Set up codebook for trial classification
        n = int(np.ceil(n_samples / self.V.shape[0]))
        _V = np.tile(self.V, (n, 1)).astype("float32")[: n_samples - epoch_size: step_size]

        # Initialize classifier with estimated shrinkage parameter
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

        # Setup CCA
        cca = CCA(n_components=1)

        # Simulate noise by taking random subsets of the training data
        x_sampled = self.x_train
        y_sampled = self.y_train

        # Setup cross-validation
        n_folds = 5
        folds = np.repeat(np.arange(n_folds), x_sampled.shape[0] / n_folds)

        # Loop folds
        accuracy_epoch = np.zeros(n_folds)
        accuracy_trial = np.zeros(n_folds)
        for i_fold in range(n_folds):
            # Split data to train and valid set
            X_trn, y_trn = x_sampled[folds != i_fold, :, :n_samples], y_sampled[folds != i_fold]
            X_tst, y_tst = x_sampled[folds == i_fold, :, :n_samples], y_sampled[folds == i_fold]

            # Slice trials to epochs
            X_sliced_trn, y_sliced_trn = pynt.utilities.trials_to_epochs(
                X_trn, y_trn, self.V, epoch_size, step_size
            )
            X_sliced_tst, y_sliced_tst = pynt.utilities.trials_to_epochs(
                X_tst, y_tst, self.V, epoch_size, step_size
            )

            # Train CCA (on epoch level)
            erp_noflash = np.mean(X_sliced_trn[y_sliced_trn == 0, :, :], axis=0, keepdims=True)
            erp_flash = np.mean(X_sliced_trn[y_sliced_trn == 1, :, :], axis=0, keepdims=True)
            erps = np.concatenate((erp_noflash, erp_flash), axis=0)[y_sliced_trn, :, :]
            cca.fit(
                X_sliced_trn.transpose((0, 1, 3, 2)).reshape((-1, x_sampled.shape[1])),
                erps.transpose((0, 1, 3, 2)).reshape(-1, x_sampled.shape[1]),
            )
            w = cca.x_weights_.flatten()

            # Apply CCA (on epoch level)
            X_sliced_filtered_trn = np.dot(
                w, X_sliced_trn.transpose((2, 0, 1, 3)).reshape((x_sampled.shape[1], -1))
            ).reshape(-1, epoch_size)
            X_sliced_filtered_tst = np.dot(
                w, X_sliced_tst.transpose((2, 0, 1, 3)).reshape((x_sampled.shape[1], -1))
            ).reshape(-1, epoch_size)

            # Train LDA (on epoch level)
            # N.B.: spatio-temporal features are flattened
            # N.B.: all epochs of all trials are concatenated
            lda.fit(X_sliced_filtered_trn, y_sliced_trn.flatten())

            # Apply LDA (on epoch level)
            yh_sliced_tst = lda.predict(X_sliced_filtered_tst)

            # Compute accuracy (on epoch level)
            accuracy_epoch[i_fold] = 100 * np.mean(yh_sliced_tst == y_sliced_tst.flatten())

            # Apply LDA (on trial level)
            ph_tst = lda.predict_proba(X_sliced_filtered_tst)[:, 1]
            ph_tst = np.reshape(ph_tst, y_sliced_tst.shape)
            rho = pynt.utilities.correlation(ph_tst, _V.T)
            yh_tst = np.argmax(rho, axis=1)
            accuracy_trial[i_fold] = 100 * np.mean(yh_tst == y_tst)

        if self.trial:
            f_x = accuracy_trial.mean()
        else:
            f_x = accuracy_epoch.mean()
        return f_x

    def get_domain(self) -> np.ndarray:
        """
        The domain of the simulated objective function.

        Returns:
            np.ndarray: The domain of the simulated objective function.
        """
        domains = np.array([[0, 1], [0.1, 0.4], ])  # shrinkage  # number of CCA components
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
