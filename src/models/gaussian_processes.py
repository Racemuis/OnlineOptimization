import warnings
from typing import Optional, Callable

import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.exceptions import OptimizationWarning

from ..plot_functions.utils import plot_GP
from ..utils.base import RegressionModel

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
        self.is_trained = False
        self.log_noise = False
        self._with_grad = True

    @property
    def with_grad(self) -> bool:
        return self._with_grad

    def fit(self, x_train: torch.Tensor, y_train: torch.Tensor) -> FixedNoiseGP:
        """
        Fit a botorch FixedNoiseGP according to the implementation proposed in [1].

        Args:
            x_train (torch.Tensor): A `batch_shape x n x d` tensor of training features.
            y_train (torch.Tensor): A `batch_shape x n x m` tensor of training observations.

        Returns:
            FixedNoiseGP: A fitted instance of a FixedNoiseGP, trained as Most Likely GP regression algorithm.
        """
        g2 = g3 = None
        optimization_problem = False

        if self.normalize:
            self.input_transform = transforms.input.Normalize(d=x_train.shape[-1])
            self.outcome_transform = transforms.outcome.Standardize(m=x_train.shape[-1])

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
            try:
                g2 = self.train_noise_model(covar_module, x_train, z)

                # Predict the empirical noise levels using g2
                g2.eval()
                with torch.no_grad():
                    train_y_var = g2.posterior(x_train).mean
            except OptimizationWarning:
                optimization_problem = True

            # TODO: Convergence is added when the estimated noise values become too small, but this should actually be
            #  solved by predicting the log noise. Though, log -> train g2 -> exp(g2.posterior) tends to underestimate
            #  high variances

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
                g2 = self.train_noise_model(covar_module, x_train, torch.log(z))

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
                _ = fit_gpytorch_mll(mll_g3)
                g3.eval()

            except RuntimeError:
                print(
                    "Warning: Trying to backward through the graph a second time "
                    + "(or directly access saved tensors after they have already been freed).\n Skipping iteration."
                )
                continue

            # Set g1 <- g3 for the next iteration
            g1 = g3

        self._noise_model = g2
        self._composite_model = g3
        self.is_trained = True

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
            input_transform=self.input_transform,
            outcome_transform=self.outcome_transform,
            covar_module=covar_module,
        )
        # g2.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
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

    def get_estimated_std(self, x_train: torch.Tensor) -> torch.Tensor:
        self._noise_model.eval()
        if self.log_noise:
            return torch.sqrt(torch.exp(self.get_noise_model().posterior(x_train).mean))
        else:
            return torch.sqrt(self.get_noise_model().posterior(x_train).mean)

    def plot(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        var_true: Callable,
        f: Callable,
        domain: np.ndarray,
        maximum: float,
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
            acquisition_function (Optional[AcquisitionFunction]): The acquisition function.

        Returns:
            None
        """
        noise_model = self.get_noise_model().eval()
        composite_model = self.get_model().eval()
        domain = domain.astype(int)

        # unwrap the domain for easy plotting
        plot_domain = [domain[0][0], domain[1][0]]

        x_test = torch.tensor(np.linspace(plot_domain[0], plot_domain[1], 1000)).unsqueeze(1)

        ax0 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        ax1 = plt.subplot2grid((2, 2), (1, 0))
        ax2 = plt.subplot2grid((2, 2), (1, 1))

        ax0 = plot_GP(
            x_train=x_train.cpu().numpy(), y_train=y_train.cpu().numpy(), x_test=x_test, model=composite_model, ax=ax0
        )
        ax0.axvline(maximum, color="r", linewidth=0.3, label="global maximum")
        ax0.plot(x_test.numpy(), f(x_test.numpy()), color="black", linestyle="dashed", linewidth=0.6, label="f(x)")
        ax0.set_title(f"Most-likely GP regression")

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
            x_test.numpy(),
            self.get_estimated_std(x_train=x_test).cpu().detach().numpy(),
            label="estimated observed standard deviation",
        )
        ax1.set_title("estimated observed standard deviation")
        ax1.set_xlabel("x")
        ax1.set_xticks(np.linspace(plot_domain[0], plot_domain[1], (plot_domain[1] - plot_domain[0]) // 2 + 1))
        ax1.set_ylabel("var(y)")

        ax2.plot(x_test.numpy(), var_true(x_test.numpy()), label="true std")
        ax2.set_title("true standard deviation")
        ax2.set_xlabel("x")
        ax2.set_xticks(np.linspace(plot_domain[0], plot_domain[1], (plot_domain[1] - plot_domain[0]) // 2 + 1))
        plt.tight_layout()
        plt.show()
