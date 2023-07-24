from typing import Tuple, List, Union, Optional

import os
import mne
import glob
import numpy as np

from pathlib import Path


class Reader:
    """
    A reader for EEG recordings. The default arguments from the Reader are based on the work by Sosulski and Tangermann
    in [1].

    [1] Sosulski, Jan & Tangermann, Michael. (2022). Introducing Block-Toeplitz Covariance Matrices to Remaster Linear
        Discriminant Analysis for Event-related Potential Brain-computer Interfaces.
    """

    def __init__(
        self,
        experiment: str,
        filter_band: Optional[Union[List, Tuple]] = (0.5, 16),
        baseline: Optional[Union[List, Tuple]] = (-0.2, 0),
        decimate: Optional[int] = 25,
        temporal_intervals: Optional[Union[np.ndarray, List]] = np.array([0.1, 0.17, 0.23, 0.3, 0.41, 0.5]),
    ):
        """

        Args:
            experiment (str): The name of the experiment (should match an experiment name from the config file).
            filter_band (Optional[Union[List, Tuple]]): The band used to filter the EEG data.
            baseline (Optional[Union[List, Tuple]]): The interval wherein baseline correction should be applied.
            decimate (Optional[int]): The decimation value.
            temporal_intervals (Optional[Union[np.ndarray, List]]): The intervals of timepoints that should be averaged.
                The time points should be provided in seconds. An interval list of [0.1, 0.2, 0.3] specifies that the
                time points between 100 and 200, and the timepoints between 200 and 300 ms should be averaged.
        """
        assert (
            len(filter_band) == 2
        ), f"The filter band should contain 2 elements, obtained {len(filter_band)} elements."
        assert (
            baseline is None or len(baseline) == 2
        ), f"The baseline correction should contain 2 elements, obtained {len(baseline)} elements."
        self.experiment = experiment
        self.filter_band = filter_band
        self.baseline = baseline
        self.decimate = decimate
        self.temporal_intervals = temporal_intervals
        self.non_eeg_channels = ["EOGvu", "x_EMGl", "x_GSR", "x_Respi", "x_Pulse", "x_Optic"]
        mne.set_log_level("WARNING")

    def read(
        self,
        data_config: dict,
        participant: Optional[str] = None,
        condition: Optional[str] = None,
        verbose: bool = False,
    ) -> dict:
        """
        Read the EEG data into a dictionary of mne.Epochs objects, indexed with the participant and the session as keys.

        Args:
            data_config (dict): The data configuration file.
            participant (Optional[str]): The participant to read. If None is provided, all participants are read.
                Default = None.
            condition (Optional[str]): The condition to read. If None is provided, all conditions are read.
                Default = None.
            verbose (bool): True if the information regarding the read epoch should be printed to standard out.

        Returns:
            dict: a dictionary of mne.Epochs objects.
        """
        # create data stack
        data = dict()

        participants = data_config[self.experiment]["participant_list"] if participant is None else [participant]
        conditions = data_config[self.experiment]["condition_list"] if condition is None else [condition]

        for session in participants:
            for condition in conditions:

                # Find files
                eeg_filepaths = self.get_file_list(data_config[self.experiment]["data_path"], session, condition)
                stimulus_ids, class_ids = self.get_marker_format(config=data_config)

                # Read, preprocess, and slice the data
                epo_arr = []
                for eeg_filepath in eeg_filepaths:
                    eeg_data = mne.io.read_raw_brainvision(eeg_filepath, misc=self.non_eeg_channels)
                    eeg_data.set_montage("standard_1020").load_data()
                    eeg_data.filter(self.filter_band[0], self.filter_band[1], method="iir")
                    eeg_data.pick_types(eeg=True)
                    epoch = self.to_epoch(eeg_data, stimulus_ids=stimulus_ids, class_ids=class_ids,)
                    epo_arr.append(epoch)

                # Add to data stack
                if session not in data:
                    data[session] = dict()
                if condition not in data[session]:
                    data[session][condition] = mne.concatenate_epochs(epo_arr) if len(epo_arr) > 1 else epo_arr

        if verbose:
            print(data[participants[0]][conditions[0]].info)

        return data

    def to_epoch(self, data, stimulus_ids: dict, class_ids: dict, reject=None) -> mne.Epochs:
        if reject is None:
            reject = dict()

        return mne.Epochs(
            raw=data,
            events=mne.events_from_annotations(raw=data, event_id=stimulus_ids)[0],
            event_id=class_ids,
            baseline=self.baseline,
            decim=self.decimate,
            reject=reject,
            proj=False,
            preload=True,
            tmin=-0.2,
            tmax=0.8,
        )

    def eeg_to_numpy(
        self, participant: str, condition: str, data_config: dict, verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read the EEG data into a flattened numpy array. The array is flattened by reshaping the temporal and spatial
        dimension in a single dimension of n_averaged_time_intervals x n_channels.

        Args:
            participant (str): The participant to read.
            condition (str): The condition to read.
            data_config (dict): The data configuration file.
            verbose (bool): True if the information regarding the read epoch should be printed to standard out.

        Returns:
            np.ndarray: The flattened EEG signals.
            np.ndarray: The associated labels.
        """
        epochs = self.read(data_config=data_config, participant=participant, condition=condition)
        epoch = epochs[participant][condition]

        if verbose:
            print(epoch.info)

        x = self.average_temporal_intervals(epoch, boundaries=self.temporal_intervals).reshape(
            (-1, epoch.info["nchan"] * (self.temporal_intervals.shape[0] - 1))
        )
        y = epoch.events[:, 2]

        return x, y

    @staticmethod
    def get_file_list(data_path: Union[Path, str], subject: str, condition: str, verbose: bool = False) -> List[str]:
        """
        Get the list of valid paths given a file path to the data and the subject and condition. Assumes that the data
        is stored like
        <data_folder>
        |---<subject>
            |---<condition>
            |---<condition>
            ...
        |---<subject>
            ...


        Args:
            data_path (str): The path to the data folder.
            subject (str): The identifier of the participant
            condition (str): The identifier of the condition.
            verbose (bool): True if the paths that have been found should be printed. Default = False.

        Returns:

        """
        sep = os.path.sep
        search_path = f"{data_path}{sep}{subject}{sep}*{condition}*.vhdr"
        if verbose:
            print(f"Loading files matching {search_path}")
        return sorted(glob.glob(search_path))

    @staticmethod
    def average_temporal_intervals(epoch: mne.Epochs, boundaries: Union[List, Tuple, np.ndarray],) -> np.ndarray:
        """
        Average the epochs over specified time intervals. The time points should be provided in seconds.
        An interval list of [0.1, 0.2, 0.3] specifies that the time points between 100 and 200, and the timepoints
        between 200 and 300 ms should be averaged.

        Args:
            epoch (mne.Epochs): A mne.Epochs object.
            boundaries (Optional[Union[np.ndarray, List]]): The intervals of timepoints that should be averaged.

        Returns:
            np.ndarray: The averaged data.
        """
        shape_orig = epoch.get_data().shape
        X = np.zeros((shape_orig[0], shape_orig[1], len(boundaries) - 1))
        for i in range(len(boundaries) - 1):
            idx = epoch.time_as_index((boundaries[i], boundaries[i + 1]))
            idx_range = list(range(idx[0], idx[1]))
            X[:, :, i] = epoch.get_data()[:, :, idx_range].mean(axis=2)
        return X

    def get_marker_format(self, config: dict) -> Tuple[dict, dict]:
        """
        Get the marker format given the dataset.

        Args:
            config (dict): The data configuration file.

        Returns:
            dict: the stimulus identifiers.
            dict: the class identifiers.
        """
        if (
            "auditory_aphasia" in config[self.experiment]["data_path"]
            or "Word_ERPs_DirectionStudyDenzer" in config[self.experiment]["data_path"]
        ):
            stimulus_ids = dict([(f"Stimulus/S1{i}{j}", i) for j in range(7) for i in range(2)])
            class_ids = {"Target": 1, "Non-target": 0}
            return stimulus_ids, class_ids
        raise ValueError("The dataset that has been provided is not supported yet.")
