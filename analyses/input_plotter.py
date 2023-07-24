from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.stats import multivariate_normal

plt.rcParams.update({"font.size": 14})


def paraboloid(x, y, z, a, b):
    x = (x - np.min(x))/(np.max(x)-np.min(x))
    x = (x - 0.5) * 4
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    y = (y - 0.5) * 4
    p = -0.1*(x ** 2 / a ** 2 + y ** 2 / b ** 2 - a) + 0.2 + z
    p[p < 0] = 0
    p[p > 1] = 1

    # mean = np.array([np.mean(x), np.mean(y)])
    # cov = np.eye(2) * np.array([(np.max(x)-np.min(x))/2, (np.max(y)-np.min(y))/2])
    # p = multivariate_normal.pdf(np.column_stack((x, y)), mean=mean, cov=cov)*0.3 + z

    return p


def plot_figure(
    df: pd.DataFrame,
    x_label: str,
    y_label: str,
    participant: str,
    superimpose_parabola: bool,
    ax: Optional[plt.Axes] = None,
):
    no_ax = False
    if ax is None:
        no_ax = True
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    if superimpose_parabola:
        auc = paraboloid(df[x_label], df[y_label], df["AUC score"], 1, 1)
    else:
        auc = df["AUC score"]

    # tricontour
    surf = ax.plot_trisurf(df[x_label], df[y_label], auc, cmap=cm.coolwarm, vmin=0.5, vmax=0.75)
    ax.set_xlabel(x_label.replace("_", " "), labelpad=15)
    ax.set_xticks(np.round(np.linspace(df[x_label].min(), df[x_label].max(), 3), 2))
    ax.set_yticks(np.round(np.linspace(df[y_label].min(), df[y_label].max(), 3), 2))
    ax.set_ylabel(y_label.replace("_", " "), labelpad=15)
    ax.set_zlabel("AUC score", labelpad=15, rotation=95)
    if superimpose_parabola:
        pass
    else:
        ax.set_zlim(0.5, 0.75)
        ax.set_zticks(np.round(np.linspace(0.45, 0.75, 3), 2))

    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.zaxis.set_tick_params(labelsize=12)

    if no_ax:
        # ax.set_title(f"Participant {participant}, condition: 6D")
        # fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)
        # plt.subplots_adjust(left=0, right=1, bottom=-0.5, top=0.9)
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width * 0.8, box.height * 0.9])
        # plt.tight_layout()
        ax.set_box_aspect(aspect=None, zoom=0.75)
        fig.savefig(f"./space_results/{x_label}_vs_{y_label}_{participant}.pdf", bbox_inches="tight")
        return ax
    return ax, surf


def main():
    superimpose_parabola = True
    participants = ["VPpblz_15_08_14", "VPpboa_15_08_11", "VPpbob_15_08_13", "VPpboc_15_08_17"]
    Path(r"space_results").mkdir(parents=True, exist_ok=True)

    for participant in participants:
        path = (
            rf"C:\Users\Racemuis\Documents\school\m artificial intelligence\semester"
            rf" 2\thesis\results\space_map\space_search_{participant}.csv "
        )
        df = pd.read_csv(path, names=["shrinkage_parameter", "offset", "interval", "AUC score"], header=None)

        # plot shrinkage parameter vs offset
        plot_figure(
            df=df[df["interval"] == 0.08], x_label="shrinkage_parameter", y_label="offset", participant=participant, superimpose_parabola=superimpose_parabola
        )

        # plot shrinkage parameter vs interval
        plot_figure(
            df=df[df["offset"] == 0.1], x_label="shrinkage_parameter", y_label="interval", participant=participant, superimpose_parabola=superimpose_parabola,
        )

        # plot offset vs interval
        plot_figure(
            df=df[df["shrinkage_parameter"] == 0.0526315789473684],
            x_label="offset",
            y_label="interval",
            participant=participant,
            superimpose_parabola=superimpose_parabola
        )

        # # combined plot
        # fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "3d"})
        # # plot shrinkage parameter vs offset
        # axes[0], _ = plot_figure(
        #     df=df[df["interval"] == 0.08],
        #     x_label="shrinkage_parameter",
        #     y_label="offset",
        #     participant=participant,
        #     ax=axes[0],
        #     superimpose_parabola=superimpose_parabola,
        # )
        #
        # # plot shrinkage parameter vs interval
        # axes[1], _ = plot_figure(
        #     df=df[df["offset"] == 0.1],
        #     x_label="shrinkage_parameter",
        #     y_label="interval",
        #     participant=participant,
        #     ax=axes[1],
        #     superimpose_parabola=superimpose_parabola,
        # )
        #
        # # plot offset vs interval
        # axes[2], surf = plot_figure(
        #     df=df[df["shrinkage_parameter"] == 0],
        #     x_label="offset",
        #     y_label="interval",
        #     participant=participant,
        #     ax=axes[2],
        #     superimpose_parabola=superimpose_parabola,
        # )
        #
        # fig.suptitle(f"Participant {participant}")
        # fig.colorbar(surf, ax=axes.ravel().tolist())
        # fig.savefig(f"./space_results/all_{participant}.pdf", bbox_inches="tight")
        #
        # a = np.array([[0.5, 0.8]])
        # plt.figure(figsize=(9, 1.5))
        # _ = plt.imshow(a, cmap="coolwarm")
        # plt.gca().set_visible(False)
        # cax = plt.axes([0.1, 0.2, 0.8, 0.6])
        # plt.colorbar(orientation="horizontal", cax=cax, label="AUC score", shrink=0.5)
        # plt.savefig("./space_results/colorbar.pdf")


if __name__ == "__main__":
    main()
