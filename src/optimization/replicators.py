from typing import Optional, List, Tuple
from copy import deepcopy

import torch
import numpy as np

from scipy.spatial.distance import euclidean
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel

from ..models.gaussian_processes import MostLikelyHeteroskedasticGP
from ..utils.base import Replicator, RegressionModel
from ..utils.integrate_mat_52 import p1, p3, p4


class SequentialReplicator(Replicator):
    """
    A replicator that has been based on the work by Binois et al. [1], where the replication of a proposed sample is
    based on the Integrated Mean Squared Prediction Error (IMSPE).

    [1] Binois, M., Huang, J., Gramacy, R. B., & Ludkovski, M. (2019). Replication or exploration? Sequential design for
     stochastic simulation experiments. Technometrics, 61(1), 7-23.
    """

    def __init__(self, horizon: int):
        """

        Args:
            horizon (int): The lookahead horizon.

        """
        self.horizon = horizon
        super(SequentialReplicator, self).__init__()

    def forward(
        self,
        x_train: torch.Tensor,
        x_proposed: torch.Tensor,
        y_train: torch.Tensor,
        model: MostLikelyHeteroskedasticGP,
        estimated_std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Assess whether x_proposed should be chosen as the next value to evaluate, or whether the model should resample
        an existing value from x_train.

        Args:
            x_train (torch.Tensor): The tensor containing all unique design locations X: (x_0, x_1, ...,  x_n-1) where
            each entry x_i is a vector of shape (d, ).
            x_proposed (torch.Tensor):  The new sample, a vector of shape (d, ).
            y_train (torch.Tensor): Not used, added for compatibility with the Replicator superclass.
            model (MostLikelyHeteroskedasticGP): An instance of the class MostLikelyHeteroskedasticGP that has been
                fitted on x_train.
            estimated_std (torch.Tensor): Not used, added for compatibility with the Replicator superclass.

        Returns:
            torch.Tensor: The x-value that is decided upon by the replicator. Either an existing, or the proposed value.
        """
        x_train_unique = self.unique_preserve_order(x_train)

        r_train = torch.pow(model.get_estimated_std(x_train_unique), 2)
        r_sample = torch.pow(model.get_estimated_std(x_proposed), 2)

        with torch.no_grad():
            return self.rollout(
                horizon=self.horizon,
                x_train=x_train,
                x_proposed=x_proposed[0],
                r_train=r_train,
                r_proposed=r_sample,
                model=model.get_model(),
            )

    def rollout(
        self,
        horizon: int,
        x_train: torch.Tensor,
        x_proposed: torch.Tensor,
        r_train: torch.tensor,
        r_proposed: torch.tensor,
        model: BatchedMultiOutputGPyTorchModel,
    ) -> torch.Tensor:
        """
        Perform the lookahead strategy described in [1, p11]. The strategy has been implemented in a recursive fashion.

        [1] Binois, M., Huang, J., Gramacy, R. B., & Ludkovski, M. (2019). Replication or exploration? Sequential design for
        stochastic simulation experiments. Technometrics, 61(1), 7-23.

        Args:
            horizon (int): The lookahead horizon.
            x_train (torch.Tensor): The tensor containing all unique design locations X: (x_0, x_1, ...,  x_n-1) where
            each entry x_i is a vector of shape (d, ).
            x_proposed (torch.Tensor):  The new sample, a vector of shape (d, ).
            r_train (torch.Tensor): The variance of the noise distribution around x_train.
            r_proposed (torch.Tensor): The variance of the noise distribution around x_proposed.
            model (BatchedMultiOutputGPyTorchModel): The trained Gaussian process.

        Returns:
            torch.Tensor: The x-value that is decided upon by the replicator. Either an existing, or the proposed value.
        """
        assert horizon > 0, f"The rollout horizon should be > 0, received: {horizon}."

        # Cast input to numpy
        x_mu = x_train.numpy()
        r_mu = r_train.numpy()
        x_proposed = x_proposed.cpu().detach().numpy()
        r_proposed = r_proposed.cpu().detach().numpy()

        # TODO: Tests do not work :(, something wrong with the block partition inverse, probably matrix multiplication
        #  order
        # _K_n = self.K_n(x_mu=x_mu, r=r_mu, model=model)
        # _K_i = np.linalg.inv(_K_n)
        #
        # _K_next = self.K_next(K=_K_n, x_tilde=x_proposed, x_mu=x_mu, r_tilde=r_proposed, model=model)
        # _K_i_next = self.K_inv_next(K_i=_K_i, x_mu=x_mu, x_tilde=x_proposed, r_tilde=r_proposed, model=model)
        #
        # print("\nMethode 1")
        # print(np.linalg.inv(_K_next))
        # print()
        # print("Methode 2")
        # print(_K_i_next)
        #
        # print()
        # print(np.linalg.inv(_K_next)@_K_next)
        # print()
        # print(_K_i_next@_K_next)

        path, score = self._rollout(
            horizon=horizon,
            x_mu=x_mu,
            x_tilde=x_proposed,
            r_mu=r_mu,
            r_tilde=r_proposed,
            _K_n=self.K_n(x_mu=x_mu, r=r_mu, model=model),
            _W_n=self.W_mat_52_n(x_mu=x_mu, theta=model.covar_module.base_kernel.lengthscale[0]),
            model=model,
            path=[],
            replicated_samples=[],
            accepted_proposal=False,
        )
        return torch.tensor(path[0][np.newaxis, :])

    def _rollout(
        self,
        horizon: int,
        x_mu: np.ndarray,
        x_tilde: np.ndarray,
        r_mu: np.ndarray,
        r_tilde: np.ndarray,
        _K_n: np.ndarray,
        _W_n: np.ndarray,
        model: BatchedMultiOutputGPyTorchModel,
        path: List[np.ndarray],
        replicated_samples: List[np.ndarray],
        accepted_proposal: bool,
    ) -> Tuple[List[np.ndarray], float]:
        """
        A recursive rollout-helper function. If the proposed sample has already been included in the path, the function
        calls a one-tailed recursion with merely replicates. If the proposed sample has not been included in the path,
        then a two-tailed recursion is called, where one recursive call includes the proposed sample, and the other call
        includes a replicate. A sample can only be contained once in each path.

        Args:
            horizon (int): The lookahead horizon.
            x_mu (np.ndarray): The tensor containing all unique design locations X: (x_0, x_1, ...,  x_n-1) where
            each entry x_i is a vector of shape (d, ).
            x_tilde (np.ndarray):  The new sample, a vector of shape (d, ).
            r_mu (np.ndarray): The variance of the noise distribution around x_mu.
            r_tilde (np.ndarray): The variance of the noise distribution around x_tilde.
            _K_n (np.ndarray): The covariance matrix of the data-generating mechanism according to the model.
            _W_n (np.ndarray): The double integral of the matérn 5/2 kernel over a [0,1]^d domain.
            model (BatchedMultiOutputGPyTorchModel): The trained Gaussian process.
            path (List[np.ndarray]): The path that is rolled out by the recursive call.
            replicated_samples (List[np.ndarray]): The samples that have already been replicated within this call.
            accepted_proposal (bool): True if the path contains x_tilde.

        Returns:
            List[np.ndarray]: The final path.
            float: The IMSPE that is associated with the final path.
        """
        # print(f"Horizon: {horizon}\npath: {path}\n")
        E = model.covar_module.base_kernel.lengthscale[0].numpy()

        if accepted_proposal:
            # Find the best replicate
            min_imspe, min_x, min_r = self.get_best_replicate(E, _K_n, _W_n, model, r_mu, x_mu, replicated_samples)
            replicate_path = deepcopy(path)
            replicate_path.append(min_x)

            replicated_samples_new = deepcopy(replicated_samples)
            replicated_samples_new.append(min_x)

            if horizon == 0:  # Return the path with the best replicate
                # print(f"Submitting path: {replicate_path}, score: {min_imspe}")
                return replicate_path, min_imspe
            else:
                # Start recursion with one tail
                _K_next = self.K_next(K=_K_n, x_tilde=min_x, x_mu=x_mu, r_tilde=min_r, model=model)
                _W_next = self.W_mat_52_next(W=_W_n, x_tilde=min_x, x_mu=x_mu, theta=E)
                path_left_tail, score = self._rollout(
                    horizon=horizon - 1,
                    x_mu=np.vstack([x_mu, min_x]),  # vstack with previous
                    x_tilde=x_tilde,
                    r_mu=np.vstack([r_mu, min_r]),
                    r_tilde=r_tilde,
                    _K_n=_K_next,
                    _W_n=_W_next,
                    model=model,
                    path=replicate_path,
                    replicated_samples=replicated_samples_new,
                    accepted_proposal=True,
                )
                return path_left_tail, score
        else:
            # Get the score for an accepted sample
            _K_next = self.K_next(K=_K_n, x_tilde=x_tilde, x_mu=x_mu, r_tilde=r_tilde, model=model)
            _W_next = self.W_mat_52_next(W=_W_n, x_tilde=x_tilde, x_mu=x_mu, theta=E)

            score = self.IMSPE(E=E, K_i=np.linalg.inv(_K_next), W=_W_next)

            path_accepted = deepcopy(path)
            path_accepted.append(x_tilde)

            if horizon == 0:  # Return the path with the accepted sample
                # print(f"Submitting path: {path_accepted}, score: {score}")
                return path_accepted, score
            else:
                # Start recursion with two tails
                # 1: Start the path for the accepted sample
                path_accepted, score_accepted = self._rollout(
                    horizon=horizon - 1,
                    x_mu=np.vstack([x_mu, x_tilde]),
                    x_tilde=x_tilde,
                    r_mu=np.vstack([r_mu, r_tilde]),
                    r_tilde=r_tilde,
                    _K_n=_K_next,
                    _W_n=_W_next,
                    model=model,
                    path=path_accepted,
                    replicated_samples=replicated_samples,
                    accepted_proposal=True,
                )

                # 2: Start a path with the best replicate
                min_imspe, min_x, min_r = self.get_best_replicate(E, _K_n, _W_n, model, r_mu, x_mu, replicated_samples)
                _K_next = self.K_next(K=_K_n, x_tilde=min_x, x_mu=x_mu, r_tilde=min_r, model=model)
                _W_next = self.W_mat_52_next(W=_W_n, x_tilde=min_x, x_mu=x_mu, theta=E)

                path_rejected = deepcopy(path)
                path_rejected.append(min_x)  # add the replicate to the path

                replicated_samples_new = deepcopy(replicated_samples)
                replicated_samples_new.append(min_x)

                path_replicated, score_replicated = self._rollout(
                    horizon=horizon - 1,
                    x_mu=np.vstack([x_mu, min_x]),
                    x_tilde=x_tilde,
                    r_mu=np.vstack([r_mu, min_r]),
                    r_tilde=r_tilde,
                    _K_n=_K_next,
                    _W_n=_W_next,
                    model=model,
                    path=path_rejected,
                    replicated_samples=replicated_samples_new,
                    accepted_proposal=False,
                )

                # Return path that yields the lowest IMSPE
                if score_accepted <= score_replicated:
                    return path_accepted, score_accepted
                else:
                    return path_replicated, score_replicated

    def get_best_replicate(
        self,
        E: np.ndarray,
        _K_n: np.ndarray,
        _W_n: np.ndarray,
        model: BatchedMultiOutputGPyTorchModel,
        r_mu: np.ndarray,
        x_mu: np.ndarray,
        replicated_samples: List[np.ndarray],
    ):
        """
        Get the replicate that is not in the path and minimizes the IMSPE of the path.

        Args:
            E (np.ndarray): The length scale parameter of the kernel.
            _K_n (np.ndarray): The covariance matrix of the data-generating mechanism according to the model.
            _W_n (np.ndarray): The double integral of the matérn 5/2 kernel over a [0,1]^d domain.
            model (BatchedMultiOutputGPyTorchModel): The trained Gaussian process.
            r_mu (np.ndarray): The variance of the noise distribution around x_mu.
            x_mu (np.ndarray): The tensor containing all unique design locations X: (x_0, x_1, ...,  x_n-1) where
            each entry x_i is a vector of shape (d, ).
            replicated_samples (List[np.ndarray]): The samples that have already been replicated within this call.

        Returns:
            The replicate that minimizes the IMSPE of the path.
        """
        min_imspe = np.inf
        min_x = x_mu[0]
        min_r = r_mu[0]
        # print("Finding best replicate")
        # print(replicated_samples)
        for x, r in zip(x_mu, r_mu):
            if len(replicated_samples) > 0 and np.any(np.all(x == replicated_samples, axis=1)):
                continue
            _K_next = self.K_next(K=_K_n, x_tilde=x, x_mu=x_mu, r_tilde=r, model=model)
            _W_next = self.W_mat_52_next(W=_W_n, x_tilde=x, x_mu=x_mu, theta=E)
            score = self.IMSPE(E=E, K_i=np.linalg.inv(_K_next), W=_W_next)

            # print(f"x: {x}, score: {score}, k.shape: {_K_next.shape}")

            if score < min_imspe:
                min_imspe = score
                min_x = x
                min_r = r
        # print()
        return min_imspe, min_x, min_r

    @staticmethod
    def IMSPE(E: np.ndarray, K_i: np.ndarray, W: np.ndarray) -> float:
        """
        Calculate the integrated mean squared prediction error (IMSPE), which is defined as the "de-noised" model
        variance that is integrated over the domain D for the N training data points:

        .. math::
            IMSPE(x_0, x_1, ..., x_N) = \int_{x \in D} \check{\sigma^2}_N(x) dx = \mathbb{E}[\check{\sigma^2}_N(x)]

        As demonstrated in Lemma 3.1 in [1], the IMSPE can be rewritten as

        .. math::
            IMSPE(x_0, x_1, ..., x_n) = E - tr(K^{-1}_n W_n)

        where n is the number of unique design points.

        Args:
            E (torch.Tensor): The scale parameter of the Gaussian process kernel.
                Take heed: E only reduces to the scale parameter for the kernel families: Gaussian, Matérn 5/2,
                Matérn 3/2 and Matérn 1/2.
            K_i (torch.Tensor): The inverse of the Gaussian process covariance matrix.
            W (torch.Tensor): The double integral of the kernel over the [0,1]^d domain.

        Returns:
            float: The integrated mean squared prediction error.
        """
        return np.mean(E) - np.trace(np.matmul(K_i, W))

    @staticmethod
    def K_n(x_mu: np.ndarray, model: BatchedMultiOutputGPyTorchModel, r: Optional[np.ndarray]) -> np.ndarray:
        """
        Calculate the covariance matrix :math:`K_n` of the data-generating mechanism :math:`Y \sim \mathcal{N}(0, K_n)`
        for the noisy observations Y, according to the model [1]. That is, K is an N x N matrix where each entry

        .. math::
           k_ij = k(x_i, x_j) + \delta * r(x_i)

        Args:
            x_mu (np.ndarray): The tensor containing all unique design locations X: (x_0, x_1, ...,  x_n-1) where each
                entry x_i is a vector of shape (d, ).
            model (BatchedMultiOutputGPyTorchModel): The trained Gaussian process.
            r (Optional[torch.Tensor]): The variance of the noise distribution. If None is provided, the noise that is
                learned by the model is used.

        Returns:
            (torch.tensor): The covariance matrix of shape (n, n).
        """
        # cast to torch
        x_mu = torch.tensor(x_mu, requires_grad=False)
        K = np.zeros((x_mu.shape[0], x_mu.shape[0]))
        for i, x1 in enumerate(x_mu):
            for j, x2 in enumerate(x_mu):
                dirac_delta = 1 if i == j else 0
                K[i, j] = model.covar_module.base_kernel.forward(x1, x2) + dirac_delta * r[i]
        return K

    @staticmethod
    def K_next(
        K: np.ndarray,
        x_tilde: np.ndarray,
        x_mu: np.ndarray,
        r_tilde: Optional[np.ndarray],
        model: BatchedMultiOutputGPyTorchModel,
    ) -> np.ndarray:
        """
        Calculate the updated covariance matrix :math:`K_{n+1}` of the data-generating mechanism Y ~ N(0, K) for the
        noisy observations Y, according to the model [1], where the original covariance matrix has been extended with
        the covariance of `x_sample`.

        Args:
            K (np.ndarray): The original covariance matrix of the data generating process of the n unique design
                points. K has a shape of (n, n).
            x_tilde (np.ndarray): The new sample, a vector of shape (d, ).
            x_mu (np.ndarray): The tensor containing all unique design locations X: (x_0, x_1, ...,  x_n-1) where each
                entry x_i is a vector of shape (d, ).
            r_tilde (np.ndarray): The variance of the noise distribution. If None is provided, the noise that is
                learned by the model is used.
            model (BatchedMultiOutputGPyTorchModel): The trained Gaussian process.

        Returns:
            (np.ndarray): The updated covariance matrix of shape (n+1, n+1).
        """
        # cast to tensors
        x_mu = torch.tensor(x_mu, requires_grad=False)
        x_tilde = torch.tensor(x_tilde, requires_grad=False)

        K_row = np.array([model.covar_module.base_kernel.forward(x_i, x_tilde).squeeze() for x_i in x_mu])
        _K_next = np.ones((K.shape[0] + 1, K.shape[1] + 1))
        _K_next[: K.shape[0], : K.shape[0]] = K
        _K_next[K.shape[0], : K.shape[0]] = K_row.T
        _K_next[: K.shape[0], K.shape[0]] = K_row
        _K_next[K.shape[0], K.shape[0]] = model.covar_module.base_kernel.forward(x_tilde, x_tilde) + r_tilde
        return _K_next

    @staticmethod
    def K_inv_next(
        K_i: np.ndarray,
        x_mu: np.ndarray,
        x_tilde: np.ndarray,
        r_tilde: np.ndarray,
        model: BatchedMultiOutputGPyTorchModel,
    ) -> np.ndarray:
        """
        Calculate the inverse of the updated covariance matrix :math:`K_{n+1}` given the inverse of the current
        covariance matrix :math:`K_n^{-1}` using the (block matrix) partition inverse equations [2].
        Calculating the covariance matrix in this fashion requires an :math:`\mathcal{O}(n^2)` computation.
        (Versus :math:`\mathcal{O}(n^3)` for a normal matrix inverse.

        [2] Barnett, S. (1979). Matrix Methods for Engineers and Scientists. McGraw-Hill.

        Args:
            K_i: The inverse of the current covariance matrix of shape (n, n).
            x_mu (np.ndarray): The tensor containing all unique design locations X: (x_0, x_1, ...,  x_n-1) where each
                entry x_i is a vector of shape (d, ).
            x_tilde (np.ndarray): The new sample, a vector of shape (d, ).
            r_tilde (np.ndarray): The variance of the noise distribution for the new sample.
            model (BatchedMultiOutputGPyTorchModel): The trained Gaussian process.


        Returns:
            The inverse of the next covariance matrix that includes the observation x_tilde.
        """
        # cast to tensors
        x_mu = torch.tensor(x_mu, requires_grad=False)
        x_tilde = torch.tensor(x_tilde, requires_grad=False)

        # This is actually wrong as this model should be trained on the unique design locations only,
        # now it's trained on all training data
        K_row = np.array([model.covar_module.base_kernel.forward(x_i, x_tilde).squeeze() for x_i in x_mu])

        # var_n_tilde = model.posterior(x_tilde[np.newaxis, :]).variance.numpy().squeeze()
        var_n_tilde = (
            model.covar_module.base_kernel.forward(x_tilde, x_tilde).detach().cpu().numpy()
            + r_tilde
            - K_row.T @ K_i @ K_row
        ).squeeze()

        g_tilde = (-1 / var_n_tilde) * np.dot(K_i, K_row)

        K_i_next = np.ones((K_i.shape[0] + 1, K_i.shape[1] + 1))
        K_i_next[: K_i.shape[0], : K_i.shape[0]] = K_i + (1/var_n_tilde) * (K_i@K_row)@(K_row.T@K_i)  # np.dot(g_tilde, g_tilde.T) * var_n_tilde
        K_i_next[K_i.shape[0], : K_i.shape[0]] = g_tilde.T
        K_i_next[: K_i.shape[0], K_i.shape[0]] = g_tilde
        K_i_next[K_i.shape[0], K_i.shape[0]] = 1 / var_n_tilde

        return K_i_next

    @staticmethod
    def W_mat_52_n(x_mu: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute double integral of the matérn 5/2 kernel over a [0,1]^d domain.
        - Translated from c++ from the hetGP package: https://github.com/cran/hetGP/src/EMSE.cpp.
        - Original author: Mickaël Binois.

        Args:
            x_mu (np.ndarray): The tensor containing all unique design locations X: (x_0, x_1, ...,  x_n-1) where each
                entry x_i is a vector of shape (d, ).
            theta (np.ndarray): The length scale parameter of the matérn 5/2 kernel.

        Returns:
            torch.Tensor: The double integral of the matérn 5/2 kernel of shape (n, n).
        """
        W = np.ones((x_mu.shape[0], x_mu.shape[0]))
        for i in range(x_mu.shape[0]):  # number of data points
            for j in range(i + 1):
                for k in range(x_mu.shape[1]):  # dimensionality of each data point
                    t = theta[k]
                    a = x_mu[j, k]
                    b = x_mu[i, k]
                    if b < a:
                        tmp = b
                        b = a
                        a = tmp
                    t2 = t * t
                    a2 = a * a
                    b2 = b * b
                    if i == j:
                        W[i, j] *= (
                            np.exp(-2 * np.sqrt(5.0) * a / t)
                            * (
                                63 * t2 * t2 * np.exp(2 * np.sqrt(5.0) * a / t)
                                - 50 * a2 * a2
                                - 16 * 5 * np.sqrt(5.0) * t * a2 * a
                                - 270 * t2 * a2
                                - 18 * 5 * np.sqrt(5.0) * t2 * t * a
                                - 63 * t2 * t2
                            )
                            - np.exp(-2 * np.sqrt(5.0) / t)
                            * (
                                (
                                    t
                                    * (
                                        t
                                        * (
                                            10 * (5 * a2 - 27 * a + 27)
                                            + 9 * t * (7 * t - 5 * np.sqrt(5.0) * (2 * a - 2))
                                            + 10 * a * (22 * a - 27)
                                        )
                                        - 8 * 5 * np.sqrt(5.0) * (a - 1) * (a - 1) * (2 * a - 2)
                                    )
                                    + 50 * (a - 2) * (a - 1) * (a - 1) * a
                                    + 50 * (a - 1) * (a - 1)
                                )
                                * np.exp(2 * np.sqrt(5.0) * a / t)
                                - 63 * t2 * t2 * np.exp(2 * np.sqrt(5.0) / t)
                            )
                        ) / (36 * np.sqrt(5.0) * t2 * t)
                    else:
                        p_1 = p1(a, a2, b, b2, t2, t)
                        p_3 = p3(a, a2, b, b2, t2, t)
                        p_4 = p4(a, a2, b, t2, t)
                        W[j, i] *= p_1 + p_3 + p_4
                        W[i, j] *= p_1 + p_3 + p_4
        return W

    def W_mat_52_next(self, W: np.ndarray, x_tilde: np.ndarray, x_mu: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute the updated double integral of the matérn 5/2 kernel over a [0,1]^d domain.
        - Translated from c++ from the hetGP package: https://github.com/cran/hetGP/src/EMSE.cpp.
        - Original author: Mickaël Binois.

        Args:
            W (np.ndarray): The original double integral of the matérn 5/2 kernel of shape (n, n).
            x_tilde (np.ndarray): The new sample, a vector of shape (d, ).
            x_mu (np.ndarray): The tensor containing all unique design locations X: (x_0, x_1, ...,  x_n-1) where each
                entry x_i is a vector of shape (d, ).
            theta (np.ndarray): The length scale parameter of the matérn 5/2 kernel.

        Returns:
            torch.Tensor: The updated double integral of the matérn 5/2 kernel of shape (n+1, n+1).
        """
        W_row = self.W_mat_52(x_tilde=x_tilde, x_mu=x_mu, theta=theta)

        W_sample = 1
        for k in range(x_tilde.shape[0]):
            t = theta[k]
            a = x_tilde[k]
            b = x_tilde[k]

            t2 = t * t
            a2 = a * a
            b2 = b * b

            W_sample *= p1(a, a2, b, b2, t2, t) + p3(a, a2, b, b2, t2, t) + p4(a, a2, b, t2, t)

        W_next = np.ones((W.shape[0] + 1, W.shape[1] + 1))
        W_next[: W.shape[0], : W.shape[0]] = W
        W_next[W.shape[0], : W.shape[0]] = W_row.T
        W_next[: W.shape[0], W.shape[0]] = W_row
        W_next[W.shape[0], W.shape[0]] = W_sample
        return W_next

    @staticmethod
    def W_mat_52(x_tilde: np.ndarray, x_mu: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Calculate a single row of the double integral matrix W.

        .. math::
            w(x_tilde) = w(x_tilde, x_{mu, i})_{0 \leq i < n}

        Args:
            x_tilde (np.ndarray): The new sample, a vector of shape (d, ).
            x_mu (np.ndarray): The tensor containing all unique design locations X: (x_0, x_1, ...,  x_n-1) where each
                entry x_i is a vector of shape (d, ).
            theta (np.ndarray): The length scale parameter of the matérn 5/2 kernel.

        Returns:
            np.ndarray: A single row of the double integral matrix W.
        """
        w_vect = np.ones(x_mu.shape[0])
        for j in range(x_mu.shape[0]):
            for k in range(x_mu.shape[1]):
                t = theta[k]
                a = x_mu[j, k]
                b = x_tilde[k]
                if b < a:
                    tmp = b
                    b = a
                    a = tmp
                t2 = t * t
                a2 = a * a
                b2 = b * b
                w_vect[j] *= p1(a, a2, b, b2, t2, t) + p3(a, a2, b, b2, t2, t) + p4(a, a2, b, t2, t)
        return w_vect

    @staticmethod
    def unique_preserve_order(x_train: torch.Tensor) -> torch.Tensor:
        """
        Return the unique values in x_train while preserving the order.

        Args:
            x_train:

        Returns:

        """
        accepted_x = x_train[0]
        for x in x_train[1:]:
            for y in accepted_x:
                if torch.equal(x, y):
                    continue
            accepted_x = torch.vstack((accepted_x, x))
        return accepted_x


class MaxReplicator(Replicator):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x_proposed: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        model: RegressionModel,
        estimated_std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Assess whether the proposed x-value should be chosen, or whether the model should resample an existing x-value.
        TODO: Get the variance around the noise estimation to make an assessment for replication.
        TODO: For flat noise type use a normal GP instead.

        Args:
            x_proposed (torch.Tensor): The proposed parameters.
            x_train (torch.Tensor): The parameters that have already been evaluated.
            y_train (torch.Tensor): The y-values that are associated with the evaluated parameters.
            model (BatchedMultiOutputGPyTorchModel): The regression model that is used during the optimization process.
            estimated_std (torch.Tensor): The standard deviation that is estimated by the regression model.

        Returns:
            The x-value that is decided upon by the replicator. Either the proposed value, or a replication.
        """
        model = model.get_model()
        # mean, variance = self._mean_and_variance(X=x_proposed, model=model)
        # expected_y = model.posterior(x_proposed).mean

        if x_proposed.shape[-1] > 1:
            distances = np.array([euclidean(x_proposed.squeeze(), x.squeeze()) for x in x_train])
        else:
            distances = np.array([euclidean(x_proposed[0], x) for x in x_train])

        closest_train_idx = np.argmin(distances)
        y_max_idx = torch.argmax(y_train).item()

        # the proposed sample is close to the sample + noise that maximizes the objective function
        if y_max_idx == closest_train_idx:  # and torch.sqrt(variance).item() < estimated_std.mean().item():
            replicate = x_train[y_max_idx].unsqueeze(0)
            replicate_std = torch.sqrt(model.posterior(X=replicate.detach()).variance)
            proposed_std = torch.sqrt(model.posterior(X=x_proposed).variance)
            var_std = torch.var(torch.sqrt(model.posterior(X=x_train.detach()).variance))

            # print(f"Proposed_sample {x_proposed}, proposed replicate: {replicate}")
            # print(replicate_std, var_std, proposed_std)

            if replicate_std > proposed_std:
                return x_train[y_max_idx].unsqueeze(0)
        return x_proposed
