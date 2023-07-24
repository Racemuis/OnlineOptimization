import os
import yaml
from pathlib import Path

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from src.data.Reader import Reader

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

sns.set_style('whitegrid')
font = {'size': 11.7}
matplotlib.rc('font', **font)
plt.rcParams["figure.dpi"] = 250


def main():
    """
    Plot an example event-related potential.

    Returns:
        None
    """
    # Set paths
    participant = "VPpbqe_15_07_28"
    condition = "6D"
    path, _ = os.path.split(os.path.dirname(os.path.realpath(__file__)))
    data_config = yaml.load(open(os.path.join(path, "src/conf/data_config.yaml"), "r"), Loader=yaml.FullLoader)

    reader = Reader("auditory_aphasia")
    plot_channels = ['Cz', 'FC1']
    channel_styles = ['-', '--', ':']
    cp = sns.color_palette()

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    data = reader.read(data_config=data_config, participant=participant, condition=condition, verbose=True)
    epo = data[participant][condition]
    print(epo)

    # Compute target and non-target ERP at channels
    evo_t = epo['Target'].average(picks=plot_channels)
    evo_nt = epo['Non-target'].average(picks=plot_channels)

    # Plot target and non-target ERP for channel
    for ch_i, ch in enumerate(plot_channels):
        ax.plot(evo_t.times * 1000, evo_t.data[ch_i, :],
                linestyle=channel_styles[ch_i], color=cp[1], label=f'{ch} Target')
        ax.plot(evo_nt.times * 1000, evo_nt.data[ch_i, :],
                linestyle=channel_styles[ch_i], color=cp[0], label=f'{ch} Non-target')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title(f'Subject: {participant} - condition: {condition}')
    ax.legend()
    path = r"./results"
    Path(path).mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(path, r"erp.pdf"), bbox_inches='tight')


if __name__ == '__main__':
    main()
