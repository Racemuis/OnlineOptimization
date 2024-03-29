import os
import warnings
from typing import Optional, Callable, List, Any

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from botorch.exceptions import OptimizationWarning
from botorch.acquisition.objective import PosteriorTransform

from src.plot_functions.utils import plot_GP
from src.utils.base import RegressionModel

from botorch.models import transforms
from botorch.acquisition import AcquisitionFunction
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, FixedNoiseGP
from gpytorch.constraints import GreaterThan
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform

from gpytorch import ExactMarginalLogLikelihood
from gpytorch import Module


from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel


class HomGP(RegressionModel):
    def __init__(
        self,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        normalize: Optional[bool] = False,
    ):
        self.normalize = normalize
        super().__init__(input_transform, outcome_transform)
        self.model = None

    def get_estimated_std(self, x_train: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.tensor([self.model.likelihood.noise.item()])).repeat(x_train.shape[0])

    def get_posterior_std(self, x_train: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return torch.sqrt(self.model.posterior(x_train).variance)

    @property
    def with_grad(self) -> bool:
        return True

    @property
    def num_outputs(self):
        return self.model.num_outputs

    def posterior(
        self,
        X: torch.Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ):
        """
        Computes the posterior over model outputs at the provided points.

        Note: The input transforms should be applied here using
            `self.transform_inputs(X)` after the `self.eval()` call and before
            any `model.forward` or `model.likelihood` calls.

        Args:
            X: A `b x q x d`-dim Tensor, where `d` is the dimension of the
                feature space, `q` is the number of points considered jointly,
                and `b` is the batch dimension.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for modules. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add observation noise to the posterior.
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `Posterior` object, representing a batch of `b` joint distributions
            over `q` points and `m` outputs each.
        """
        if self.model is None:
            print("The model has not been initialized yet. Consider training the models first.")
            return None

        else:
            self.model.eval()
            return self.model.posterior(
                X=X,
                output_indices=output_indices,
                observation_noise=observation_noise,
                posterior_transform=posterior_transform,
                kwargs=kwargs,
            )

    def get_model(self) -> FixedNoiseGP:
        """
        Getter function for the main model.

        Returns:
            FixedNoiseGP: The fitted "Most likely" model.
        """
        if self.model is None:
            print("The model has not been initialized yet. Consider training the models first.")
        return self.model

    def fit(self, x_train: torch.Tensor, y_train: torch.Tensor) -> SingleTaskGP:

        if self.normalize:
            self.input_transform = transforms.input.Normalize(d=x_train.shape[-1])
            self.outcome_transform = transforms.outcome.Standardize(m=y_train.shape[-1])

        covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=x_train.shape[-1], lengthscale_prior=GammaPrior(3.5, 2.0),),
            outputscale_prior=GammaPrior(2.0, 0.5),
        )
        self.model = SingleTaskGP(
            x_train,
            y_train,
            # input_transform=self.input_transform,
            # outcome_transform=self.outcome_transform,
            covar_module=covar_module,
        )
        return self.model

    def plot(
        self,
        x_train: Any,
        y_train: Any,
        var_true: Any,
        maximum: float,
        f: Callable,
        domain: np.ndarray,
        random_sample_size: int,
        informed_sample_size: int,
        acquisition_function: Optional[AcquisitionFunction],
    ):
        pass


class MostLikelyHeteroskedasticGP(RegressionModel):
    """
    The implementation of the heteroskedastic Gaussian process regression as proposed in [1]. The model accounts
    for heteroskedastic noise by

    - Fitting a Gaussian process, g1, to the training data.
    - Training a second Gaussian process, g2, to estimate the empirical noise levels based on the training data and g1.
    - Training a third Gaussian process, g3, that combines the training data and the noise levels that are estimated
      by g2.
    The algorithm performs multiple iterations of this process, in an EM-like fashion.

    [1] K. Kersting, C. Plagemann, P. Pfaff, and W. Burgard, “Most Likely Heteroscedastic Gaussian
    Process Regression,” in Proceedings of the 24th International Conference on Machine Learning,
    ser. ICML ’07, Corvalis, Oregon, USA: Association for Computing Machinery, 2007, pp. 393–400,
    isbn: 9781595937933. doi: 10.1145/1273496.1273546.
    """

    def __init__(
        self,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        normalize: Optional[bool] = False,
        n_iter: int = 1,
    ):
        """

        Args:
            input_transform (Optional[InputTransform]): The input transform, typically Normalize().
            outcome_transform (Optional[OutcomeTransform]): The output transform, typically Standardize().
            n_iter (int): The number of iterations in the EM algorithm.
        """
        super().__init__(input_transform, outcome_transform)
        self.n_iter = n_iter
        self.normalize = normalize
        self._composite_model = None
        self._noise_model = None
        self._mll = None
        self.is_trained = False
        self.log_noise = False
        self._with_grad = True
        self.length_scales = []

    @property
    def with_grad(self) -> bool:
        return self._with_grad

    @property
    def num_outputs(self):
        return self._composite_model.num_outputs

    def fit(self, x_train: torch.Tensor, y_train: torch.Tensor) -> FixedNoiseGP:
        """
        Fit a botorch FixedNoiseGP according to the implementation proposed in [1].

        Args:
            x_train (torch.Tensor): A `batch_shape x n x d` tensor of training features.
            y_train (torch.Tensor): A `batch_shape x n x m` tensor of training observations.

        Returns:
            FixedNoiseGP: A fitted instance of a FixedNoiseGP, trained as Most Likely GP regression algorithm.
        """
        g2 = g3 = mll_g3 = None
        optimization_problem = False

        if self.normalize:
            self.input_transform = transforms.input.Normalize(d=x_train.shape[-1])
            self.outcome_transform = transforms.outcome.Standardize(m=y_train.shape[-1])

        # Create a kernel with a prior on relatively large length scales for the noise GPs
        covar_module = MaternKernel(nu=2.5, ard_num_dims=x_train.shape[-1], lengthscale_prior=GammaPrior(4.0, 1),)

        # Create g1 to estimate the empirical noise levels for the training data
        g1 = SingleTaskGP(
            train_X=x_train,
            train_Y=y_train,
            input_transform=self.input_transform,
            outcome_transform=self.outcome_transform,
            covar_module=covar_module,
        )
        g1.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))

        # Create likelihood function for g1
        mll_g1 = ExactMarginalLogLikelihood(likelihood=g1.likelihood, model=g1)

        # set mll and all submodules to the specified dtype and device
        mll_g1 = mll_g1.to(x_train)

        # Fit g1
        g1.train()
        _ = fit_gpytorch_mll(mll_g1)

        for i in range(self.n_iter):

            # Create new labels z = log(var[t_i, g1(x_i, D)])
            g1.eval()
            with torch.no_grad():
                z = torch.pow(g1.posterior(x_train).mean - y_train, 2)

            # Estimate g2 on the new dataset (noise model)
            covar_module_g2 = MaternKernel(
                nu=2.5, ard_num_dims=x_train.shape[-1], lengthscale_prior=GammaPrior(1.5, 4),
            )
            try:
                g2 = self.train_noise_model(covar_module_g2, x_train, z)

                # Predict the empirical noise levels using g2
                g2.eval()
                with torch.no_grad():
                    train_y_var = g2.posterior(x_train).mean
            except OptimizationWarning:
                optimization_problem = True

            if optimization_problem or train_y_var.lt(0).any():
                self.log_noise = True
                if optimization_problem:
                    warnings.warn(
                        f"Scipy minimize encountered an error due to too large differences in the values of z,"
                        f" training g2 on the log variance instead."
                    )
                else:
                    warnings.warn(
                        f"The estimated empirical variances have become too small, which causes them to be negative"
                        f", training g2 on log variance instead."
                    )

                # Estimate g2 on the new dataset (noise model)
                g2 = self.train_noise_model(covar_module_g2, x_train, torch.log(z))

                # Predict the empirical noise levels using g2
                g2.eval()
                with torch.no_grad():
                    train_y_var = torch.exp(g2.posterior(x_train).mean)
            else:
                self.log_noise = False

            # Estimate the combined GP g3
            g3 = FixedNoiseGP(
                train_X=x_train,
                train_Y=y_train,
                train_Yvar=train_y_var.detach(),
                input_transform=self.input_transform,
                outcome_transform=self.outcome_transform,
                covar_module=ScaleKernel(
                    MaternKernel(nu=2.5, ard_num_dims=x_train.shape[-1], lengthscale_prior=GammaPrior(3.5, 2.0),),
                    outputscale_prior=GammaPrior(2.0, 0.5),
                ),
            )

            mll_g3 = ExactMarginalLogLikelihood(likelihood=g3.likelihood, model=g3)
            mll_g3 = mll_g3.to(x_train)

            try:
                g3.train()
                mll_g3 = fit_gpytorch_mll(mll_g3)
                g3.eval()

            except RuntimeError:
                print(
                    "Warning: Trying to backward through the graph a second time "
                    + "(or directly access saved tensors after they have already been freed).\n Skipping iteration."
                )
                continue

            # Set g1 <- g3 for the next iteration
            g1 = g3

        # update all attributes
        self._mll = mll_g3
        self._noise_model = g2
        self._composite_model = g3
        self.is_trained = True

        # save final length scale
        self.length_scales.append(g3.covar_module.base_kernel.lengthscale.squeeze().detach().cpu().numpy())

        return g3

    def train_noise_model(self, covar_module: Module, x_train: torch.Tensor, z: torch.Tensor) -> SingleTaskGP:
        """
        Train a noise model given the input and noise levels.

        Args:
            covar_module (Module): The kernel for the noise model.
            x_train (torch.Tensor): The x-coordinates of the input.
            z (torch.Tensor): The noise levels that are associated with the x-coordinates.

        Returns:
            SingleTaskGP: A trained noise model (in training mode).
        """

        g2 = SingleTaskGP(
            train_X=x_train,
            train_Y=z,
            # input_transform=self.input_transform,
            # outcome_transform=self.outcome_transform,
            covar_module=covar_module,
        )
        mll_g2 = ExactMarginalLogLikelihood(likelihood=g2.likelihood, model=g2)
        mll_g2 = mll_g2.to(x_train)

        # Fit g2
        g2.train()
        _ = fit_gpytorch_mll(mll_g2)
        return g2

    def get_noise_model(self) -> SingleTaskGP:
        """
        Getter function for the noise model.

        Returns:
            SingleTaskGP: The fitted noise model.

        """
        if self._noise_model is None:
            print("The noise model has not been initialized yet. Consider training the models first.")
        return self._noise_model

    def get_model(self) -> FixedNoiseGP:
        """
        Getter function for the main model.

        Returns:
            FixedNoiseGP: The fitted "Most likely" model.
        """
        if self._composite_model is None:
            print("The model has not been initialized yet. Consider training the models first.")
        return self._composite_model

    def posterior(
        self,
        X: torch.Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ):
        """
        Computes the posterior over model outputs at the provided points.

        Note: The input transforms should be applied here using
            `self.transform_inputs(X)` after the `self.eval()` call and before
            any `model.forward` or `model.likelihood` calls.

        Args:
            X: A `b x q x d`-dim Tensor, where `d` is the dimension of the
                feature space, `q` is the number of points considered jointly,
                and `b` is the batch dimension.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for modules. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add observation noise to the posterior.
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `Posterior` object, representing a batch of `b` joint distributions
            over `q` points and `m` outputs each.
        """
        self._composite_model.eval()
        return self._composite_model.posterior(
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            kwargs=kwargs,
        )

    def get_estimated_std(self, x_train: torch.Tensor) -> torch.Tensor:
        self._noise_model.eval()
        if self.log_noise:
            return torch.sqrt(torch.exp(self.get_noise_model().posterior(x_train).mean))
        else:
            return torch.sqrt(self.get_noise_model().posterior(x_train).mean)

    def get_posterior_std(self, x_train: torch.Tensor) -> torch.Tensor:
        self._composite_model.eval()
        return torch.sqrt(self._composite_model.posterior(x_train).variance)

    def get_observed_information(self, x_train: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
        """
        Calculate the negative second derivative of the marginal log-likelihood of the fitted model.

        Fisher information is expected value of observed information, observed information is
        typically estimated at the MLE (= Fisher Information) https://en.wikipedia.org/wiki/Observed_information

        TODO: The hessian is now calculated with respect to the input data (x_train and y_train). However, it could be
          more informative to calculate the derivative of marginal log likelihood with respect to the length scale.

        Args:
            x_train (torch.Tensor): A `batch_shape x n x d` tensor of training features.
            y_train (torch.Tensor): A `batch_shape x n x m` tensor of training observations.

        Returns:
            torch.Tensor: The Hessian matrix of the marginal log-likelihood of the model given the data.
        """
        return -torch.autograd.functional.hessian(self.get_marginal_log_likelihood, torch.stack((x_train, y_train)))

    def get_marginal_log_likelihood(self, params: torch.Tensor) -> torch.Tensor:
        """
        Calculate the marginal log-likelihood of the fitted model over the training data x_train dna y_train.

        Args:
            params (torch.Tensor): A stacked vector of
                - x_train (torch.Tensor): A `batch_shape x n x d` tensor of training features.
                - y_train (torch.Tensor): A `batch_shape x n x m` tensor of training observations.
                That can be created using torch.stack((x_train, y_train)).

        Returns:
            torch.Tensor: The marginal log-likelihood of the fitted model given the data.
        """
        x_train = params[0, ...]
        y_train = params[1, ...]

        if self.is_trained:
            self._composite_model.eval()
            posterior_dist = self._composite_model(x_train)
            return self._mll.forward(function_dist=posterior_dist, target=y_train).sum()
        else:
            print("The model has not been trained yet. Consider training the models first.")
            return torch.tensor([0])

    def plot(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        var_true: Callable,
        f: Optional[Callable],
        domain: np.ndarray,
        maximum: float,
        random_sample_size: int,
        informed_sample_size: int,
        participant: Optional[str] = None,
        acquisition_function: Optional[AcquisitionFunction] = None,
    ) -> None:
        """
        Plot the fitted gaussian process, the corresponding training data and the objective function.

        Args:
            x_train (torch.Tensor): A `batch_shape x n x d` tensor of training features.
            y_train (torch.Tensor): A `batch_shape x n x m` tensor of training observations.
            var_true (Callable): The true variance function.
            f (Callable): The objective function.
            domain (np.ndarray): The domain of the objective function.
            maximum (float): The x-coordinate that corresponds to the maximum of the objective function.
            random_sample_size (int): The number of random samples to take.
            informed_sample_size (int): The number of informed samples to take.
            participant (Optional[str]): The identifier of the participant.
            acquisition_function (Optional[AcquisitionFunction]): The acquisition function.

        Returns:
            None
        """
        composite_model = self.get_model().eval()
        domain = domain.astype(int)

        # unwrap the domain for easy plotting
        plot_domain = [domain[0][0], domain[1][0]]

        x_test = torch.tensor(np.linspace(plot_domain[0], plot_domain[1], 1000)).unsqueeze(1)

        fig1, axes = plt.subplots(3, 1)

        ax0 = axes[0]
        ax1 = axes[1]
        ax2 = axes[2]

        ax0 = plot_GP(
            x_train=x_train.cpu().numpy(), y_train=y_train.cpu().numpy(), x_test=x_test, model=composite_model, ax=ax0
        )
        ax0.set_title(
            f"Most-likely GP regression\n{random_sample_size} "
            f"random samples, {informed_sample_size} informed samples"
        )

        if acquisition_function is not None:
            # plot the acquisition function
            ax0.plot(
                x_test.detach().squeeze(),
                acquisition_function.forward(x_test.unsqueeze(-1)).detach(),
                alpha=0.6,
                label="acquisition function",
                linewidth=0.5,
            )

        ax0.legend(bbox_to_anchor=(1.01, 0.0), loc="lower left", ncols=1, borderaxespad=0.0)
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")

        ax1.plot(
            x_test.numpy(), self.get_estimated_std(x_train=x_test).cpu().detach().numpy(), label="$\sigma(y)$",
        )
        ax1.set_title("estimated observed standard deviation")
        ax1.set_xlabel("x")
        ax1.set_ylabel("$\sigma(y)$")
        ax1.legend(bbox_to_anchor=(1.01, 0.0), loc="lower left", ncols=1, borderaxespad=0.0)

        if participant is None:
            var_true = 0.5 * np.abs(np.sin(10 * x_test.numpy()))
            ax2.plot(x_test.numpy(), var_true, label="$\sigma(y)$")
        else:
            results_folder = (
                r"C:\Users\Racemuis\Documents\school\m artificial intelligence\semester "
                r"2\thesis\results\variance_estimation"
            )
            path = os.path.join(results_folder, f"results_{participant}.csv")
            results = pd.read_csv(filepath_or_buffer=path, index_col=0).to_numpy()
            ax2.plot(np.linspace(0, 1, num=results.shape[1]), np.std(results, axis=0), label="$\sigma(y)$")
        ax2.set_title("true standard deviation")
        ax2.set_xlabel("x")
        ax2.set_ylabel("$\sigma(y)$")
        ax2.legend(bbox_to_anchor=(1.01, 0.0), loc="lower left", ncols=1, borderaxespad=0.0)
        plt.tight_layout()
        plt.show()
        fig1.savefig("./modules.pdf", bbox_inches="tight")
