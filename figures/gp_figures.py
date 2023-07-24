import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

from src.modules.models import MostLikelyHeteroskedasticGP
import warnings

warnings.filterwarnings(
    "ignore", message="Input data is not contained to the unit cube. Please consider min-max scaling the input data."
)
warnings.filterwarnings(
    "ignore",
    message="Input data is not standardized. Please consider scaling the input to zero mean and unit variance.",
)


def main():
    """
    Create figures that illustrate Gaussian processes.

    Returns:
        None
    """
    font = {'size': 8}
    matplotlib.rc('font', **font)
    plt.rcParams['figure.dpi'] = 250
    np.random.seed(42)
    n_samples = 10

    path = r"./results/"
    Path(path).mkdir(parents=True, exist_ok=True)

    x = np.linspace(0, 10, 100)[:, np.newaxis]

    kernel = Matern(length_scale=1.0, nu=2.5)
    K = kernel(X=x)

    # Plot prior
    samples = np.random.multivariate_normal(mean=np.zeros(K.shape[0]), cov=K, size=n_samples)
    std_prior = np.sqrt(K.diagonal())

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x.squeeze(), np.zeros(x.shape[0]), label="m(x)")
    ax.plot(x.squeeze(), samples.T, "C0", label="prior samples", linewidth=0.3)
    ax.fill_between(
        x.squeeze(),
        0 - 1.96 * std_prior,
        0 + 1.96 * std_prior,
        alpha=0.2,
        label=r"95% confidence interval",
    )
    ax.set_xlim(0, 10)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("x")
    ax.set_ylabel("$f(x)$")

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.9])

    # remove redundant labels
    h, l = ax.get_legend_handles_labels()
    h = h[:2] + h[-1:]
    l = l[:2] + l[-1:]

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
              fancybox=False, shadow=False, ncol=4, handles=h, labels=l)
    plt.show()

    fig.savefig(os.path.join(path, "prior.pdf"), bbox_inches='tight')

    # Plot posterior
    gp = GaussianProcessRegressor(kernel=kernel)
    x_train = np.random.uniform(0, 10, 5)[:, np.newaxis]
    y_train = np.sin(x_train[:, 0])

    gp.fit(x_train, y_train)
    y_mean, y_std = gp.predict(x, return_std=True)
    y_samples = gp.sample_y(x, n_samples)


    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(x_train, y_train, c="tab:orange", label="Observed data", zorder=1)
    ax.plot(x.squeeze(), y_mean, label="m(x)")
    ax.plot(x.squeeze(), y_samples, "C0", linewidth=0.3, label="posterior samples")
    ax.fill_between(
        x.squeeze(),
        y_mean - 1.96 * y_std,
        y_mean + 1.96 * y_std,
        alpha=0.2,
        label=r"95% confidence interval",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("$f(x)$")
    ax.set_xlim(0, 10)
    ax.set_ylim(-3, 3)

    # remove redundant labels
    h, l = ax.get_legend_handles_labels()
    h = h[:3] + [h[-1]]
    l = l[:3] + [l[-1]]

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
              fancybox=False, shadow=False, ncol=4, handles=h, labels=l)
    plt.show()
    fig.savefig(os.path.join(path, "posterior.pdf"), bbox_inches='tight')


    # Plot posterior + noise
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1)

    gp.fit(x_train, y_train)
    y_mean, y_std = gp.predict(x, return_std=True)
    y_samples = gp.sample_y(x, n_samples)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(x_train, y_train, c="tab:orange", label="Observed data", zorder=10)
    ax.plot(x.squeeze(), y_mean, label="m(x)", zorder=20)
    ax.plot(x.squeeze(), y_samples, "C0", linewidth=0.3, label="posterior samples")
    ax.fill_between(
        x.squeeze(),
        y_mean - 1.96 * y_std,
        y_mean + 1.96 * y_std,
        alpha=0.2,
        label=r"95% confidence interval",
        zorder=2,
    )

    ax.fill_between(
        x.squeeze(),
        y_mean - 0.1,
        y_mean + 0.1,
        alpha=0.6,
        color='gray',
        label=r"$\sigma^2_{noise}$",
        zorder=0,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("$f(x)$")
    ax.set_xlim(0, 10)
    ax.set_ylim(-3, 3)
    ax.vlines(x_train, gp.predict(x_train, return_std=False), y_train, colors='tab:red', label="residuals", zorder=30)

    # remove redundant labels
    h, l = ax.get_legend_handles_labels()
    h = h[:3] + h[-3:]
    l = l[:3] + l[-3:]

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
              fancybox=False, shadow=False, ncol=3, handles=h, labels=l)
    plt.show()
    fig.savefig(os.path.join(path, "hom_posterior.pdf"), bbox_inches='tight')

    # Plot posterior + heteroskedastic noise
    gp = MostLikelyHeteroskedasticGP()

    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)[:, np.newaxis]

    fitted_model = gp.fit(x_train, y_train)
    y_mean = fitted_model.posterior(torch.tensor(x)).mean.cpu().detach().numpy().squeeze()
    y_std = np.sqrt(fitted_model.posterior(torch.tensor(x)).variance.cpu().detach().numpy().squeeze())
    y_samples = fitted_model.posterior(torch.tensor(x)).sample(torch.Size([10])).cpu().detach().numpy().squeeze()

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(x_train, y_train, c="tab:orange", label="Observed data", zorder=10)
    ax.plot(x.squeeze(), y_mean, label="m(x)", zorder=20)
    ax.plot(x.squeeze(), y_samples.T, "C0", linewidth=0.3, label="posterior samples")
    ax.fill_between(
        x.squeeze(),
        y_mean - 1.96 * y_std,
        y_mean + 1.96 * y_std,
        alpha=0.2,
        label=r"95% confidence interval",
        zorder=2
    )

    estimated_std = gp.get_estimated_std(torch.tensor(x)).cpu().detach().numpy().squeeze()

    ax.fill_between(
        x.squeeze(),
        y_mean - estimated_std,
        y_mean + estimated_std,
        alpha=0.6,
        color='gray',
        label=r"$r(x)$",
        zorder=0,
    )

    ax.vlines(x_train, fitted_model.posterior(x_train).mean.cpu().detach().numpy().squeeze(), y_train,
              colors='tab:red', label="residuals", zorder=30)

    ax.set_xlabel("x")
    ax.set_ylabel("$f(x)$")
    ax.set_xlim(0, 10)
    ax.set_ylim(-3, 3)

    # remove redundant labels
    h, l = ax.get_legend_handles_labels()
    h = h[:3] + h[-3:]
    l = l[:3] + l[-3:]

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17),
              fancybox=False, shadow=False, ncol=3, handles=h, labels=l)
    plt.show()
    fig.savefig(os.path.join(path, "het_posterior.pdf"), bbox_inches='tight')


if __name__ == '__main__':
    main()