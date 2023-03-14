import warnings
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import torch
import gpytorch

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, FixedNoiseGP, transforms

from gpytorch.models import ExactGP
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood

from src.plot_functions.utils import plot_GP


def initialize_SingleTaskGP(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    likelihood: gpytorch.likelihoods.Likelihood,
    kernel: gpytorch.kernels.Kernel,
    input_transform: Optional[transforms.input.InputTransform] = None,
    outcome_transform: Optional[transforms.outcome.OutcomeTransform] = None,
) -> Tuple[SingleTaskGP, gpytorch.mlls.ExactMarginalLogLikelihood]:
    """
    Initialize the BoTorch SingleTaskGP model with 'likelihood' and 'kernel'.

    Args:
        train_x (Tensor): The x-coordinates of the training data.
        train_y (Tensor): The y-coordinates that are associated with the training data.
        likelihood (Likelihood): The noise-distribution that is assumed around the model likelihood.
        kernel (Kernel): The kernel of the GP.
        input_transform (Optional[InputTransform]): The input transform, typically Normalize().
        outcome_transform (Optional[OutcomeTransform]): The output transform, typically Standardize().

    Returns:
        a tuple containing
            - SingleTaskGP: The initialized SingleTaskGP object.
            - ExactMarginalLogLikelihood: A GPyTorch MarginalLogLikelihood instance.
    """
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        likelihood=likelihood,
        covar_module=kernel,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
    )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return model, mll


def most_likely_heteroskedasticGP(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    n_iter: int,
    y_true: np.ndarray,
    var_true: np.ndarray,
    normalize: Optional[bool] = False,
    plot: Optional[bool] = False,
    verbose: Optional[bool] = False,
) -> Tuple[ExactGP, ExactMarginalLogLikelihood, ExactGP]:
    """
    The implementation of the heteroscedastic Gaussian process regression as proposed in [1]. The model accounts
    for heteroscedastic noise by

    - Fitting a Gaussian process, g1, to the training data.
    - Training a second Gaussian process, g2, to estimate the empirical noise levels based on the training data and g1.
    - Training a third Gaussian process, g3, that combines the training data and the noise levels that are estimated
      by g2.
    The algorithm performs multiple iterations of this process, in an EM-like fashion.

    [1] K. Kersting, C. Plagemann, P. Pfaff, and W. Burgard, “Most Likely Heteroscedastic Gaussian
    Process Regression,” in Proceedings of the 24th International Conference on Machine Learning,
    ser. ICML ’07, Corvalis, Oregon, USA: Association for Computing Machinery, 2007, pp. 393–400,
    isbn: 9781595937933. doi: 10.1145/1273496.1273546.

    TODO: Look into get_fantasy_model(inputs, targets, **kwargs) for updating the training set instead of
          retraining the whole model every time.

    Args:
        x_train (torch.Tensor): A `batch_shape x n x d` tensor of training features.
        y_train (torch.Tensor): A `batch_shape x n x m` tensor of training observations.
        n_iter (int): The number of iterations in the EM algorithm.
        y_true (np.ndarray): The value of the objective function at x_train.
        var_true (np.array): The value of the true variance around the objective function.
        normalize (Optional[bool]): True if the inputs should be normalized to the unit cube and the outcomes should be
                                    standardized to a mean of 0 and a var of 1. (Performs worse.)
        plot (Optional[bool]): True if the resulting GP g3 should be plotted at the end of the iterations.
        verbose (Optional[bool]): True if the MSE of the most-likely GP regression optimization iterations should be
                                  printed.

    Returns:
        ExactGP: The fitted most-likely heteroscedastic model.
        ExactMarginalLogLikelihood: The likelihood of the fitted model.
        ExactGP: The fitted noise model.
    """
    if normalize:
        input_transform = transforms.input.Normalize(d=x_train.shape[-1])
        outcome_transform = transforms.outcome.Standardize(m=x_train.shape[-1])
    else:
        input_transform = None
        outcome_transform = None

    g3 = mll_g3 = None

    # Create g1 to estimate the empirical noise levels for the training data
    g1 = SingleTaskGP(
        train_X=x_train, train_Y=y_train, input_transform=input_transform, outcome_transform=outcome_transform
    )
    g1.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))

    # Create likelihood function for g1
    mll_g1 = ExactMarginalLogLikelihood(likelihood=g1.likelihood, model=g1)

    # set mll and all submodules to the specified dtype and device
    mll_g1 = mll_g1.to(x_train)

    # Fit g1
    g1.train()
    _ = fit_gpytorch_mll(mll_g1)

    for i in range(n_iter):
        # Create new labels z = log(var[t_i, g1(x_i, D)])
        g1.eval()

        with torch.no_grad():
            z = torch.pow(g1.posterior(x_train).mean - y_train, 2)
            n_samples = 1000
            z_paper = (
                1
                / 2
                * torch.mean(torch.pow(g1.posterior(x_train).rsample(torch.Size([n_samples])) - y_train, 2), dim=0)
            )
            # print("[DEBUG]:", z[0], z_paper[0])

        # Estimate g2 on the new dataset (noise model)
        g2 = SingleTaskGP(
            train_X=x_train, train_Y=z, input_transform=input_transform, outcome_transform=outcome_transform
        )
        g2.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))

        # Create likelihood function
        mll_g2 = ExactMarginalLogLikelihood(likelihood=g2.likelihood, model=g2)
        mll_g2 = mll_g2.to(x_train)

        # Fit g2
        g2.train()
        _ = fit_gpytorch_mll(mll_g2)

        # Predict the empirical noise levels using g2
        g2.eval()
        with torch.no_grad():
            train_y_var = g2.posterior(x_train).mean

        # TODO: Convergence is added when the estimated noise values become too small, but this should actually be
        #  solved by predicting the log noise. Though, log -> train g2 -> exp(g2.posterior) tends to underestimate
        #  high variances

        if train_y_var.lt(0).any():
            warnings.warn(
                f"The estimated empirical variances have become too small, which causes them to be negative"
                f" due to numerical instability. Ceasing EM algorithm at iteration {i + 1}."
            )
            if g3 is None:
                g3 = SingleTaskGP(
                    train_X=x_train,
                    train_Y=y_train,
                    input_transform=input_transform,
                    outcome_transform=outcome_transform,
                )
                mll_g3 = ExactMarginalLogLikelihood(likelihood=g3.likelihood, model=g3)
            break

        # Estimate the combined GP g3
        g3 = FixedNoiseGP(
            train_X=x_train,
            train_Y=y_train,
            train_Yvar=train_y_var.detach(),
            input_transform=input_transform,
            outcome_transform=outcome_transform,
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

        if verbose:
            with torch.no_grad():
                mse = mean_squared_error(y_true, g3.posterior(x_train).mean.cpu().detach().numpy())
                print(f"\tMost likely GP regression; iteration {i + 1:>2} - MSE: {mse:>4.5f}")

        # Set g1 <- g3 for the next iteration
        g1 = g3

    if plot:
        # Plot g3
        ax0 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        ax1 = plt.subplot2grid((2, 2), (1, 0))
        ax2 = plt.subplot2grid((2, 2), (1, 1))

        ax0 = plot_GP(x_train=x_train.cpu().numpy(), y_train=y_train.cpu().numpy(), x_test=x_train, model=g3, ax=ax0)
        ax0.set_title(f"Most-likely GP regression - iteration {i + 1}")
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")
        ax1.plot(
            x_train.cpu().numpy(), g3.posterior(x_train).variance.cpu().detach().numpy(), label="posterior variance"
        )
        ax1.set_title("posterior variance")
        ax1.set_xlabel("x")
        ax1.set_xticks(np.linspace(0, 12, 7))
        ax1.set_ylabel("var(y)")
        ax2.plot(x_train.cpu().numpy(), var_true, label="true variance")
        ax2.set_title("true variance")
        ax2.set_xlabel("x")
        ax2.set_xticks(np.linspace(0, 12, 7))
        plt.tight_layout()
        plt.show()

    return g3, mll_g3, g2
