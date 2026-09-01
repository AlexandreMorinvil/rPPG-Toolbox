"""Data loader for the participant/segment/viewpoint vHRM dataset."""

import json
import os
import re

import cv2
import numpy as np
import pandas as pd

from dataset.data_loader.BaseLoader import BaseLoader


class VHRMLoader(BaseLoader):
    """Load labeled videos and frame-aligned heart-rate measurements."""

    def __init__(self, name, data_path, config_data, device=None):
        self.views = {self._slug(view) for view in config_data.VHRM.VIEWS}
        self.signal_column = config_data.VHRM.SIGNAL_COLUMN
        super().__init__(name, data_path, config_data, device)

    @staticmethod
    def _slug(value):
        return re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")

    @classmethod
    def _camera_view(cls, camera):
        label = camera.get("label", "")
        return cls._slug(label.removeprefix("Camera_"))

    def get_raw_data(self, data_path):
        recordings = []
        if not os.path.isdir(data_path):
            raise ValueError(f"{self.dataset_name} data path does not exist: {data_path}")

        for participant_entry in sorted(os.scandir(data_path), key=lambda entry: entry.name):
            segments_path = os.path.join(participant_entry.path, "labelled_segments")
            if not participant_entry.is_dir() or not os.path.isdir(segments_path):
                continue
            participant = self._slug(participant_entry.name)
            for segment_entry in sorted(os.scandir(segments_path), key=lambda entry: entry.name):
                metadata_path = os.path.join(segment_entry.path, "meta.json")
                signal_path = os.path.join(segment_entry.path, "separated_signals", "final_signal.csv")
                if not segment_entry.is_dir() or not os.path.isfile(metadata_path) or not os.path.isfile(signal_path):
                    continue
                with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                    metadata = json.load(metadata_file)
                movement = self._slug(metadata.get("label", segment_entry.name))
                segment_index = str(metadata.get("index", segment_entry.name.split("_", 1)[0])).zfill(4)
                for camera in metadata.get("cameras", []):
                    view = self._camera_view(camera)
                    if self.views and view not in self.views:
                        continue
                    video_path = os.path.join(segment_entry.path, camera.get("video_file", ""))
                    if not os.path.isfile(video_path):
                        continue
                    recording_id = "--".join((participant, segment_index, movement, view))
                    recordings.append({
                        "index": recording_id,
                        "participant": participant,
                        "segment_index": segment_index,
                        "movement": movement,
                        "view": view,
                        "path": video_path,
                        "signal_path": signal_path,
                    })

        if not recordings:
            selected = ", ".join(sorted(self.views)) or "all"
            raise ValueError(f"{self.dataset_name} contains no usable recordings for views: {selected}")
        return recordings

    def split_raw_data(self, data_dirs, begin, end):
        if begin == 0 and end == 1:
            return data_dirs
        participants = sorted({recording["participant"] for recording in data_dirs})
        selected = set(participants[int(begin * len(participants)):int(end * len(participants))])
        return [recording for recording in data_dirs if recording["participant"] in selected]

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, index, file_list_dict):
        recording = data_dirs[index]
        frames = self.read_video(recording["path"], self.config_data.FS, config_preprocess)
        heart_rate = self.read_heart_rate(recording["signal_path"], self.signal_column, len(frames), self.config_data.FS)
        frame_clips, heart_rate_clips = self.preprocess_resized(frames, heart_rate, config_preprocess)
        file_list_dict[index] = self.save_multi_process(frame_clips, heart_rate_clips, recording["index"])

    def read_video(self, video_path, target_fps, config_preprocess):
        capture = cv2.VideoCapture(video_path)
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or source_fps <= 0 or target_fps <= 0:
            capture.release()
            raise ValueError(f"Cannot read video or determine FPS: {video_path}")

        frames = []
        source_index = 0
        next_sample_time = 0.0
        face_region = None
        sampled_index = 0
        detection = config_preprocess.CROP_FACE.DETECTION
        if detection.USE_MEDIAN_FACE_BOX:
            capture.release()
            raise ValueError("vHRM streaming preprocessing does not support USE_MEDIAN_FACE_BOX")
        success, frame = capture.read()
        while success:
            frame_time = source_index / source_fps
            if frame_time + (0.5 / source_fps) >= next_sample_time:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detect_now = face_region is None or (
                    detection.DO_DYNAMIC_DETECTION
                    and sampled_index % detection.DYNAMIC_DETECTION_FREQUENCY == 0
                )
                if config_preprocess.CROP_FACE.DO_CROP_FACE and detect_now:
                    face_region = self.face_detection(
                        frame,
                        config_preprocess.CROP_FACE.BACKEND,
                        config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
                        config_preprocess.CROP_FACE.LARGE_BOX_COEF,
                    )
                if config_preprocess.CROP_FACE.DO_CROP_FACE:
                    x, y, width, height = (int(value) for value in face_region)
                    frame = frame[max(y, 0):min(y + height, frame.shape[0]),
                                  max(x, 0):min(x + width, frame.shape[1])]
                frame = cv2.resize(
                    frame,
                    (config_preprocess.RESIZE.W, config_preprocess.RESIZE.H),
                    interpolation=cv2.INTER_AREA,
                )
                frames.append(frame)
                next_sample_time += 1.0 / target_fps
                sampled_index += 1
            source_index += 1
            success, frame = capture.read()
        capture.release()
        if not frames:
            raise ValueError(f"Video contains no readable frames: {video_path}")
        return np.asarray(frames)

    @staticmethod
    def preprocess_resized(frames, heart_rate, config_preprocess):
        data = []
        for data_type in config_preprocess.DATA_TYPE:
            if data_type == "Raw":
                data.append(frames.copy())
            elif data_type == "DiffNormalized":
                data.append(BaseLoader.diff_normalize_data(frames.copy()))
            elif data_type == "Standardized":
                data.append(BaseLoader.standardized_data(frames.copy()))
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
        data = np.concatenate(data, axis=-1).astype(np.float32)
        if config_preprocess.LABEL_TYPE != "Raw":
            raise ValueError("vHRM heart_rate_bpm labels must use LABEL_TYPE: Raw")
        if config_preprocess.DO_CHUNK:
            chunk_length = config_preprocess.CHUNK_LENGTH
            clip_count = len(data) // chunk_length
            frame_clips = [data[index * chunk_length:(index + 1) * chunk_length]
                           for index in range(clip_count)]
            heart_rate_clips = [heart_rate[index * chunk_length:(index + 1) * chunk_length]
                                for index in range(clip_count)]
            return np.asarray(frame_clips), np.asarray(heart_rate_clips)
        return np.asarray([data]), np.asarray([heart_rate])

    @staticmethod
    def read_heart_rate(signal_path, signal_column, frame_count, fps):
        signal = pd.read_csv(signal_path, usecols=["elapsed_time_seconds", signal_column]).dropna()
        if signal.empty:
            raise ValueError(f"Signal file contains no {signal_column} values: {signal_path}")
        frame_times = np.arange(frame_count, dtype=np.float64) / fps
        return np.interp(
            frame_times,
            signal["elapsed_time_seconds"].to_numpy(dtype=np.float64),
            signal[signal_column].to_numpy(dtype=np.float64),
        )

    def save_multi_process(self, frames_clips, heart_rate_clips, filename):
        participant, segment_index, movement, view = filename.split("--", 3)
        output_dir = os.path.join(self.cached_path, participant, movement, view, segment_index)
        os.makedirs(output_dir, exist_ok=True)
        input_paths = []
        for clip_index, (frames, heart_rate) in enumerate(zip(frames_clips, heart_rate_clips)):
            input_path = os.path.join(output_dir, f"{filename}_input{clip_index}.npy")
            label_path = os.path.join(output_dir, f"{filename}_label{clip_index}.npy")
            np.save(input_path, frames)
            np.save(label_path, heart_rate)
            input_paths.append(input_path)
        return input_paths
