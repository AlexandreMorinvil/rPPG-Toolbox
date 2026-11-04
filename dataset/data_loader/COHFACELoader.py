"""The dataloader for COHFACE datasets.

Details for the COHFACE Dataset see https://www.idiap.ch/en/dataset/cohface
If you use this dataset, please cite the following publication:
Guillaume Heusch, André Anjos, Sébastien Marcel, “A reproducible study on remote heart rate measurement”, arXiv, 2016.
http://publications.idiap.ch/index.php/publications/show/3688
"""
import glob
import os
import re

import cv2
import h5py
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader


class COHFACELoader(BaseLoader):
    """The data loader for the COHFACE dataset."""

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes an COHFACE dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- 1/
                     |      |-- 0/
                     |          |-- data.avi
                     |          |-- data.hdf5
                     |      |...
                     |      |-- 3/
                     |          |-- data.avi
                     |          |-- data.hdf5
                     |...
                     |   |-- n/
                     |      |-- 0/
                     |          |-- data.avi
                     |          |-- data.hdf5
                     |      |...
                     |      |-- 3/
                     |          |-- data.avi
                     |          |-- data.hdf5
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Returns the recordings under the path: sessions 0-3 of every subject folder (the 160 videos of the
        dataset; subject 6 has extra session folders that the dataset's protocols do not list)."""
        dirs = list()
        for subject_entry in os.scandir(data_path):
            if not subject_entry.is_dir() or not subject_entry.name.isdigit():
                continue
            subject = int(subject_entry.name)
            for session in range(4):
                session_dir = os.path.join(subject_entry.path, str(session))
                if os.path.isfile(os.path.join(session_dir, "data.hdf5")):
                    dirs.append({"index": f"{subject:02d}-{session}", "path": session_dir, "subject": subject})
        if not dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        return sorted(dirs, key=lambda data_dir: data_dir["index"])

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values over the sorted subjects, so that no
        subject is in two splits."""
        if begin == 0 and end == 1:
            return data_dirs
        subjects = sorted({data_dir["subject"] for data_dir in data_dirs})
        selected = set(subjects[int(begin * len(subjects)):int(end * len(subjects))])
        return [data_dir for data_dir in data_dirs if data_dir["subject"] in selected]

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Invoked by preprocess_dataset for multi_process."""
        if 'None' not in config_preprocess.DATA_AUG:
            raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! '
                             f'Received {config_preprocess.DATA_AUG}.')
        frames = self.read_video(os.path.join(data_dirs[i]["path"], "data.avi"))
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = self.read_wave(os.path.join(data_dirs[i]["path"], "data.hdf5"))
        # The contact PPG is sampled at 256 Hz; one label per video frame.
        bvps = BaseLoader.resample_ppg(bvps, frames.shape[0])
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, data_dirs[i]["index"])
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while (success):
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frame[np.isnan(frame)] = 0  # TODO: maybe change into avg
            frames.append(frame)
            success, frame = VidObj.read()

        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with h5py.File(bvp_file, 'r') as f:
            return f["pulse"][:]
