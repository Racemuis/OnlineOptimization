import torch
import numpy as np

import matplotlib.pyplot as plt

from src.modules.models import MostLikelyHeteroskedasticGP

import warnings

warnings.filterwarnings(
    "ignore", message="Input data is not contained to the unit cube. Please consider min-max scaling the input data."
)
warnings.filterwarnings(
    "ignore",
    message="Input data is not standardized. Please consider scaling the input to zero mean and unit variance.",
)

# Set seeds
np.random.seed(0)
torch.manual_seed(0)


def main():
    """
    Test the MostLikelyHeteroskedasticGP model.

    Returns:
        None.
    """
    # Set hyperparameters
    n_samples = 151

    # Create data domain
    x_train = np.linspace(0, 4 * np.pi, n_samples)

    # Compute loc and scale as functions of input data
    loc = np.sin(x_train)
    scale = np.abs(np.sin(x_train / 2))

    # Create the noisy training outcomes
    y_train = np.random.normal(loc, scale)

    # Fit the Gaussian process
    gp = MostLikelyHeteroskedasticGP(normalize=True, n_iter=1)
    model = gp.fit(x_train=torch.tensor(x_train)[:, np.newaxis], y_train=torch.tensor(y_train)[:, np.newaxis])

    # Get posterior mean and std
    mu = model.posterior(torch.tensor(x_train)[:, np.newaxis]).mean.detach().numpy().squeeze()
    std = np.sqrt(model.posterior(torch.tensor(x_train)[:, np.newaxis]).variance.detach().numpy()).squeeze()

    # Plot the distribution
    fig, axes = plt.subplots(3, 1, sharex='all')
    axes[0].scatter(x_train, y_train, label="Observed data, y(x)", color="tab:orange")
    axes[0].plot(x_train, loc, label="Objective function, f(x)", color="tab:red")
    axes[0].plot(x_train, mu, label="Posterior mean, $\mu(x)$", color="tab:blue",)
    axes[0].fill_between(
        x_train, mu - 1.96 * std, mu + 1.96 * std, color="tab:blue", alpha=0.2, label="95% confidence interval"
    )
    axes[0].set_title(r"Variance estimation with Gaussian Process regression")
    axes[0].legend()

    axes[1].plot(x_train, scale)
    axes[1].set_title("True standard deviation: r(x)")
    axes[2].plot(
        x_train,
        gp.get_estimated_std(x_train=torch.tensor(x_train)[:, np.newaxis]).cpu().detach().numpy(),
        label="estimated standard deviation",
    )
    axes[2].set_title("Estimated standard deviation: $\hat{r}(x)$")
    fig.supxlabel("x")
    fig.supylabel("y")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
