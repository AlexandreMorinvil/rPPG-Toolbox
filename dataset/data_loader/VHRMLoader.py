"""Data loader for the participant/segment/viewpoint vHRM dataset."""

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dataset.data_loader.BaseLoader import BaseLoader


class VHRMLoader(BaseLoader):
    """Load labeled videos and frame-aligned heart-rate measurements."""

    LABEL_COLUMNS = ("heart_rate_bpm", "HRV", "respiration_rate_bpm")
    SOURCE_COLUMNS = {
        "heart_rate_bpm": ("heart_rate_bpm",),
        "HRV": ("HRV",),
        "respiration_rate_bpm": ("BR", "RR"),
    }

    def __init__(self, name, data_path, config_data, device=None):
        self.views = {self._slug(view) for view in config_data.VHRM.VIEWS}
        self.signal_column = config_data.VHRM.SIGNAL_COLUMN
        self.video_decoder = config_data.VHRM.VIDEO_DECODER.lower()
        self.ffmpeg_path = config_data.VHRM.FFMPEG_PATH
        self.ffprobe_path = config_data.VHRM.FFPROBE_PATH
        self.incremental_preprocess = config_data.VHRM.INCREMENTAL_PREPROCESS
        self.adopt_legacy_cache = config_data.VHRM.ADOPT_LEGACY_CACHE
        self.manifest_filename = config_data.VHRM.MANIFEST_FILENAME
        self.cache_dtype_name = config_data.VHRM.CACHE_DTYPE.lower()
        self.show_video_progress = config_data.VHRM.SHOW_VIDEO_PROGRESS
        if self.cache_dtype_name not in {"float16", "float32"}:
            raise ValueError("VHRM.CACHE_DTYPE must be 'float16' or 'float32'")
        self.cache_dtype = np.dtype(self.cache_dtype_name)
        self._video_codec_cache = {}
        self._nvdec_decoders = None
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
                        "duration_sec": float(metadata.get("duration_sec", 0)),
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
        recording_id = recording["index"]
        started_at = time.perf_counter()
        progress_position = 1 + index % max(1, config_preprocess.NUM_WORKERS)
        expected_frames = round(recording["duration_sec"] * self.config_data.FS)
        decode_progress = tqdm(
            total=expected_frames or None,
            desc=f"{recording_id} | decode",
            unit="frame",
            position=progress_position,
            leave=False,
            dynamic_ncols=True,
            disable=not self.show_video_progress,
        )
        self._progress_message(
            f"[{recording_id}] Starting: {recording['duration_sec']:.1f}s, "
            f"target {self.config_data.FS} FPS ({expected_frames} frames)."
        )
        try:
            frames = self.read_video(
                recording["path"], self.config_data.FS, config_preprocess, decode_progress)
        finally:
            decode_progress.close()

        self._progress_message(
            f"[{recording_id}] Aligning HR, HRV, and respiration rate to {len(frames)} decoded frames."
        )
        physiological_labels = self.read_physiological_labels(
            recording["signal_path"], len(frames), self.config_data.FS)
        self._progress_message(
            f"[{recording_id}] Computing normalization statistics and writing clips."
        )
        paths = self.transform_and_save(
            frames,
            physiological_labels,
            config_preprocess,
            recording_id,
            progress_position=progress_position,
        )
        file_list_dict[index] = paths
        elapsed = time.perf_counter() - started_at
        self._progress_message(
            f"[{recording_id}] Complete: {len(paths)} clips in {elapsed:.1f}s."
        )

    def _progress_message(self, message):
        if self.show_video_progress:
            tqdm.write(message)

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        if not self.incremental_preprocess:
            return super().preprocess_dataset(data_dirs, config_preprocess, begin, end)

        recordings = self.split_raw_data(data_dirs, begin, end)
        os.makedirs(self.cached_path, exist_ok=True)
        manifest_path = os.path.join(self.cached_path, self.manifest_filename)
        manifest = self._load_manifest(manifest_path)
        manifest_entries = manifest.setdefault("recordings", {})
        config_fingerprint = self._config_fingerprint(config_preprocess)
        previous_dtype_fingerprint = self._config_fingerprint(
            config_preprocess, include_cache_dtype=False)
        reused_paths = {}
        pending_recordings = []
        adopted_count = 0
        migrated_count = 0

        for recording in recordings:
            recording_id = recording["index"]
            signature = self._recording_signature(recording)
            entry = manifest_entries.get(recording_id)
            paths = self._entry_paths(entry)
            reusable = (
                entry is not None
                and entry.get("config_fingerprint") == config_fingerprint
                and entry.get("source") == signature
                and self._validate_outputs(paths, recording, config_preprocess)
            )
            dtype_only_change = (
                not reusable
                and entry is not None
                and entry.get("config_fingerprint") == previous_dtype_fingerprint
                and entry.get("source") == signature
                and self._validate_outputs(
                    paths, recording, config_preprocess, expected_dtype=np.float32)
            )
            if dtype_only_change:
                self._migrate_input_dtype(paths)
                reusable = self._validate_outputs(paths, recording, config_preprocess)
                if reusable:
                    migrated_count += 1
            if not reusable and entry is None and self.adopt_legacy_cache:
                paths = self._legacy_output_paths(recording)
                reusable = self._validate_outputs(paths, recording, config_preprocess)
                if reusable:
                    adopted_count += 1
            if reusable:
                reused_paths[recording_id] = paths
                manifest_entries[recording_id] = self._manifest_entry(
                    recording, paths, signature, config_fingerprint)
            else:
                pending_recordings.append(recording)

        processed_paths = {}
        if pending_recordings:
            num_workers = getattr(config_preprocess, "NUM_WORKERS", 4)
            process_results = dict(self.multi_process_manager(
                pending_recordings, config_preprocess, multi_process_quota=num_workers))
            if len(process_results) != len(pending_recordings):
                missing = [recording["index"] for index, recording in enumerate(pending_recordings)
                           if index not in process_results]
                raise RuntimeError(f"Incremental preprocessing failed for recordings: {missing}")
            for index, recording in enumerate(pending_recordings):
                paths = list(process_results[index])
                if not self._validate_outputs(paths, recording, config_preprocess):
                    raise RuntimeError(f"Incomplete preprocessing output: {recording['index']}")
                self._remove_stale_outputs(recording, paths)
                processed_paths[recording["index"]] = paths
                manifest_entries[recording["index"]] = self._manifest_entry(
                    recording, paths, self._recording_signature(recording), config_fingerprint)

        all_paths = {**reused_paths, **processed_paths}
        manifest["version"] = 1
        self._write_manifest(manifest_path, manifest)
        self.build_file_list({index: all_paths[recording["index"]]
                              for index, recording in enumerate(recordings)})
        self.load_preprocessed_data()
        reused_count = len(reused_paths)
        print(
            f"Incremental preprocessing: {len(processed_paths)} processed, "
            f"{reused_count} reused ({adopted_count} adopted, "
            f"{migrated_count} dtype-migrated).",
            end="\n\n",
        )

    def _config_fingerprint(self, config_preprocess, include_cache_dtype=True):
        crop = config_preprocess.CROP_FACE
        detection = crop.DETECTION
        settings = {
            "fs": self.config_data.FS,
            "signal_column": self.signal_column,
            "label_columns": self.LABEL_COLUMNS,
            "data_type": list(config_preprocess.DATA_TYPE),
            "data_aug": list(config_preprocess.DATA_AUG),
            "label_type": config_preprocess.LABEL_TYPE,
            "do_chunk": config_preprocess.DO_CHUNK,
            "chunk_length": config_preprocess.CHUNK_LENGTH,
            "resize": [config_preprocess.RESIZE.H, config_preprocess.RESIZE.W],
            "crop_face": crop.DO_CROP_FACE,
            "crop_backend": crop.BACKEND,
            "large_box": crop.USE_LARGE_FACE_BOX,
            "large_box_coef": crop.LARGE_BOX_COEF,
            "dynamic_detection": detection.DO_DYNAMIC_DETECTION,
            "detection_frequency": detection.DYNAMIC_DETECTION_FREQUENCY,
            "median_face_box": detection.USE_MEDIAN_FACE_BOX,
        }
        if include_cache_dtype:
            settings["cache_dtype"] = self.cache_dtype_name
        serialized = json.dumps(settings, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _file_signature(path):
        stat = os.stat(path)
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def _recording_signature(self, recording):
        return {
            "video": self._file_signature(recording["path"]),
            "signal": self._file_signature(recording["signal_path"]),
        }

    def _entry_paths(self, entry):
        if not entry:
            return []
        return [os.path.join(self.cached_path, relative_path)
                for relative_path in entry.get("outputs", [])]

    def _legacy_output_paths(self, recording):
        output_dir = self._recording_output_dir(recording)
        pattern = os.path.join(output_dir, f"{recording['index']}_input*.npy")
        return sorted(glob.glob(pattern), key=self._chunk_number)

    @staticmethod
    def _chunk_number(path):
        match = re.search(r"_input(\d+)\.npy$", path)
        return int(match.group(1)) if match else -1

    @staticmethod
    def _label_path(input_path):
        return re.sub(r"_input(\d+)\.npy$", r"_label\1.npy", input_path)

    def _expected_clip_count(self, recording, config_preprocess):
        if not config_preprocess.DO_CHUNK:
            return 1
        frame_count = int(recording["duration_sec"] * self.config_data.FS)
        return frame_count // config_preprocess.CHUNK_LENGTH

    def _validate_outputs(self, input_paths, recording, config_preprocess, expected_dtype=None):
        expected_count = self._expected_clip_count(recording, config_preprocess)
        if expected_count <= 0 or len(input_paths) != expected_count:
            return False
        channels = 3 * len(config_preprocess.DATA_TYPE)
        expected_dtype = np.dtype(expected_dtype or self.cache_dtype)
        for input_path in input_paths:
            label_path = self._label_path(input_path)
            if not os.path.isfile(input_path) or not os.path.isfile(label_path):
                return False
            input_array = None
            label_array = None
            try:
                input_array = np.load(input_path, mmap_mode="r")
                label_array = np.load(label_path, mmap_mode="r")
                if input_array.dtype != expected_dtype:
                    return False
                if config_preprocess.DO_CHUNK:
                    expected_input_shape = (
                        config_preprocess.CHUNK_LENGTH,
                        config_preprocess.RESIZE.H,
                        config_preprocess.RESIZE.W,
                        channels,
                    )
                    if input_array.shape != expected_input_shape:
                        return False
                    if label_array.shape != (config_preprocess.CHUNK_LENGTH, len(self.LABEL_COLUMNS)):
                        return False
            except (OSError, ValueError):
                return False
            finally:
                if input_array is not None and hasattr(input_array, "_mmap"):
                    input_array._mmap.close()
                if label_array is not None and hasattr(label_array, "_mmap"):
                    label_array._mmap.close()
        return True

    def _migrate_input_dtype(self, input_paths):
        for input_path in input_paths:
            temporary_path = input_path + ".dtype-migration.npy"
            input_array = np.load(input_path)
            np.save(temporary_path, input_array.astype(self.cache_dtype, copy=False))
            del input_array
            os.replace(temporary_path, input_path)

    def _recording_output_dir(self, recording):
        return os.path.join(
            self.cached_path,
            recording["participant"],
            recording["movement"],
            recording["view"],
            recording["segment_index"],
        )

    def _manifest_entry(self, recording, paths, signature, config_fingerprint):
        return {
            "source": signature,
            "config_fingerprint": config_fingerprint,
            "outputs": [os.path.relpath(path, self.cached_path) for path in paths],
        }

    def _remove_stale_outputs(self, recording, current_input_paths):
        current_paths = set(current_input_paths)
        for stale_input in self._legacy_output_paths(recording):
            if stale_input not in current_paths:
                stale_label = self._label_path(stale_input)
                os.remove(stale_input)
                if os.path.isfile(stale_label):
                    os.remove(stale_label)

    @staticmethod
    def _load_manifest(manifest_path):
        if not os.path.isfile(manifest_path):
            return {"version": 1, "recordings": {}}
        try:
            with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
            if manifest.get("version") != 1 or not isinstance(manifest.get("recordings"), dict):
                raise ValueError("unsupported manifest format")
            return manifest
        except (json.JSONDecodeError, OSError, ValueError) as error:
            print(f"Ignoring invalid incremental preprocessing manifest: {error}")
            return {"version": 1, "recordings": {}}

    @staticmethod
    def _write_manifest(manifest_path, manifest):
        temporary_path = manifest_path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        os.replace(temporary_path, manifest_path)

    def read_video(self, video_path, target_fps, config_preprocess, progress=None):
        if self._can_use_nvdec(video_path, config_preprocess):
            try:
                if progress is not None:
                    progress.set_description(f"{os.path.basename(video_path)} | NVDEC", refresh=True)
                frames = self._read_video_nvdec(
                    video_path, target_fps, config_preprocess, progress)
                self._progress_message(f"Using FFmpeg/NVDEC: {video_path}")
                return frames
            except (OSError, RuntimeError, ValueError) as error:
                self._progress_message(f"FFmpeg/NVDEC unavailable for {video_path}: {error}")
                self._progress_message("Falling back to OpenCV decoding.")
                if progress is not None:
                    progress.reset()
        if progress is not None:
            progress.set_description(f"{os.path.basename(video_path)} | OpenCV", refresh=True)
        return self._read_video_opencv(video_path, target_fps, config_preprocess, progress)

    def _can_use_nvdec(self, video_path, config_preprocess):
        if self.video_decoder == "opencv":
            return False
        if self.video_decoder not in {"auto", "nvdec"}:
            raise ValueError("VHRM.VIDEO_DECODER must be 'auto', 'nvdec', or 'opencv'")
        detection = config_preprocess.CROP_FACE.DETECTION
        if detection.DO_DYNAMIC_DETECTION or detection.USE_MEDIAN_FACE_BOX:
            return False
        if not str(self._face_det_device).lower().startswith("cuda") or not torch.cuda.is_available():
            return False

        ffmpeg = shutil.which(self.ffmpeg_path)
        ffprobe = shutil.which(self.ffprobe_path)
        if not ffmpeg or not ffprobe:
            return False
        try:
            codec = self._probe_codec(video_path)
            decoder = {"hevc": "hevc_cuvid", "h264": "h264_cuvid"}.get(codec)
            if decoder is None:
                return False
            if self._nvdec_decoders is None:
                self._nvdec_decoders = subprocess.run(
                    [ffmpeg, "-v", "error", "-decoders"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            return decoder in self._nvdec_decoders
        except (json.JSONDecodeError, OSError, subprocess.SubprocessError):
            return False

    def _read_video_nvdec(self, video_path, target_fps, config_preprocess, progress=None):
        capture = cv2.VideoCapture(video_path)
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        success, first_frame = capture.read()
        capture.release()
        if not success or source_width <= 0 or source_height <= 0:
            raise ValueError("OpenCV could not read the reference frame for face detection")
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

        codec = self._probe_codec(video_path)
        decoder = {"hevc": "hevc_cuvid", "h264": "h264_cuvid"}.get(codec)
        if decoder is None:
            raise ValueError(f"NVDEC does not support input codec: {codec}")

        command = [
            shutil.which(self.ffmpeg_path), "-v", "error", "-nostdin", "-noautorotate",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-c:v", decoder,
        ]
        if config_preprocess.CROP_FACE.DO_CROP_FACE:
            x, y, width, height = self.face_detection(
                first_frame,
                config_preprocess.CROP_FACE.BACKEND,
                config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
                config_preprocess.CROP_FACE.LARGE_BOX_COEF,
            )
            left, top, right, bottom = self._even_crop_bounds(
                x, y, width, height, source_width, source_height)
            command.extend([
                "-crop",
                f"{top}x{source_height - bottom}x{left}x{source_width - right}",
            ])
        command.extend([
            "-resize", f"{config_preprocess.RESIZE.W}x{config_preprocess.RESIZE.H}",
            "-i", video_path,
            "-vf", f"fps={target_fps},hwdownload,format=nv12,format=rgb24",
            "-an", "-sn", "-dn", "-f", "rawvideo", "pipe:1",
        ])

        frame_size = config_preprocess.RESIZE.W * config_preprocess.RESIZE.H * 3
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frames = []
        try:
            while True:
                frame_bytes = process.stdout.read(frame_size)
                if not frame_bytes:
                    break
                while len(frame_bytes) < frame_size:
                    remainder = process.stdout.read(frame_size - len(frame_bytes))
                    if not remainder:
                        break
                    frame_bytes += remainder
                if len(frame_bytes) != frame_size:
                    raise RuntimeError("FFmpeg returned a partial video frame")
                frames.append(np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
                    config_preprocess.RESIZE.H, config_preprocess.RESIZE.W, 3))
                if progress is not None:
                    progress.update(1)
            stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
            return_code = process.wait()
        finally:
            if process.poll() is None:
                process.kill()
        if return_code != 0:
            raise RuntimeError(stderr or f"FFmpeg exited with status {return_code}")
        if not frames:
            raise ValueError("FFmpeg returned no video frames")
        return np.asarray(frames)

    def _probe_codec(self, video_path):
        codec_cache = getattr(self, "_video_codec_cache", {})
        if video_path in codec_cache:
            return codec_cache[video_path]
        probe = subprocess.run(
            [shutil.which(self.ffprobe_path), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "json", video_path],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(probe.stdout).get("streams", [])
        if not streams:
            raise ValueError("ffprobe found no video stream")
        codec = streams[0].get("codec_name")
        codec_cache[video_path] = codec
        self._video_codec_cache = codec_cache
        return codec

    @staticmethod
    def _even_crop_bounds(x, y, width, height, frame_width, frame_height):
        left = max(0, min(int(x), frame_width - 2))
        top = max(0, min(int(y), frame_height - 2))
        right = max(left + 2, min(int(x + width), frame_width))
        bottom = max(top + 2, min(int(y + height), frame_height))
        left -= left % 2
        top -= top % 2
        right -= right % 2
        bottom -= bottom % 2
        return left, top, right, bottom

    def _read_video_opencv(self, video_path, target_fps, config_preprocess, progress=None):
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
                if progress is not None:
                    progress.update(1)
            source_index += 1
            success, frame = capture.read()
        capture.release()
        if not frames:
            raise ValueError(f"Video contains no readable frames: {video_path}")
        return np.asarray(frames)

    def transform_and_save(
            self, frames, physiological_labels, config_preprocess, filename, progress_position=1):
        """Apply recording-wide normalization and save one clip-sized buffer at a time."""
        if config_preprocess.LABEL_TYPE != "Raw":
            raise ValueError("vHRM heart_rate_bpm labels must use LABEL_TYPE: Raw")
        supported_types = {"Raw", "DiffNormalized", "Standardized"}
        unsupported = set(config_preprocess.DATA_TYPE) - supported_types
        if unsupported:
            raise ValueError(f"Unsupported data type: {sorted(unsupported)}")

        standard_mean = None
        standard_deviation = None
        if "Standardized" in config_preprocess.DATA_TYPE:
            standard_mean = float(np.mean(frames, dtype=np.float64))
            standard_deviation = float(np.std(frames, dtype=np.float64))
        difference_deviation = None
        if "DiffNormalized" in config_preprocess.DATA_TYPE:
            difference_deviation = self._difference_deviation(
                frames, config_preprocess.CHUNK_LENGTH)

        if config_preprocess.DO_CHUNK:
            chunk_length = config_preprocess.CHUNK_LENGTH
            clip_count = len(frames) // chunk_length
        else:
            chunk_length = len(frames)
            clip_count = 1

        participant, segment_index, movement, view = filename.split("--", 3)
        output_dir = os.path.join(self.cached_path, participant, movement, view, segment_index)
        os.makedirs(output_dir, exist_ok=True)
        input_paths = []
        channels = 3 * len(config_preprocess.DATA_TYPE)
        clip_range = tqdm(
            range(clip_count),
            desc=f"{filename} | transform/save",
            unit="clip",
            position=progress_position,
            leave=False,
            dynamic_ncols=True,
            disable=not self.show_video_progress,
        )
        for clip_index in clip_range:
            start = clip_index * chunk_length
            end = min(start + chunk_length, len(frames))
            transformed = np.empty(
                (end - start, frames.shape[1], frames.shape[2], channels),
                dtype=self.cache_dtype,
            )
            for data_index, data_type in enumerate(config_preprocess.DATA_TYPE):
                output = transformed[..., data_index * 3:(data_index + 1) * 3]
                if data_type == "Raw":
                    output[...] = frames[start:end]
                elif data_type == "Standardized":
                    self._standardize_into(
                        frames[start:end], output, standard_mean, standard_deviation)
                else:
                    self._difference_normalize_into(
                        frames, output, start, end, difference_deviation)

            input_path = os.path.join(output_dir, f"{filename}_input{clip_index}.npy")
            label_path = os.path.join(output_dir, f"{filename}_label{clip_index}.npy")
            np.save(input_path, transformed)
            np.save(label_path, physiological_labels[start:end])
            input_paths.append(input_path)
        return input_paths

    @staticmethod
    def _raw_differences(frames, start, end):
        numerator = np.subtract(
            frames[start + 1:end + 1], frames[start:end], dtype=np.uint8).astype(np.float32)
        denominator = np.add(
            frames[start + 1:end + 1], frames[start:end], dtype=np.uint8).astype(np.float32)
        denominator += np.float32(1e-7)
        np.divide(numerator, denominator, out=numerator)
        return numerator

    @classmethod
    def _difference_deviation(cls, frames, block_length):
        count = 0
        mean = 0.0
        sum_squared_deviations = 0.0
        for start in range(0, len(frames) - 1, block_length):
            end = min(start + block_length, len(frames) - 1)
            differences = cls._raw_differences(frames, start, end)
            block_count = differences.size
            block_mean = float(np.mean(differences, dtype=np.float64))
            block_deviations = differences.astype(np.float64) - block_mean
            block_sum_squared = float(np.sum(np.square(block_deviations)))
            delta = block_mean - mean
            combined_count = count + block_count
            sum_squared_deviations += (
                block_sum_squared + delta * delta * count * block_count / combined_count)
            mean += delta * block_count / combined_count
            count = combined_count
        return np.sqrt(sum_squared_deviations / count) if count else 0.0

    @staticmethod
    def _standardize_into(source, output, mean, deviation):
        if not deviation or not np.isfinite(deviation):
            output.fill(0)
            return
        temporary = source.astype(np.float32)
        temporary -= np.float32(mean)
        temporary /= np.float32(deviation)
        output[...] = temporary

    @classmethod
    def _difference_normalize_into(cls, frames, output, start, end, deviation):
        output.fill(0)
        difference_end = min(end, len(frames) - 1)
        if difference_end <= start or not deviation or not np.isfinite(deviation):
            return
        differences = cls._raw_differences(frames, start, difference_end)
        differences /= np.float32(deviation)
        output[:len(differences)] = differences

    @classmethod
    def read_physiological_labels(cls, signal_path, frame_count, fps):
        signal = pd.read_csv(signal_path)
        if "elapsed_time_seconds" not in signal:
            raise ValueError(f"Signal file contains no elapsed_time_seconds column: {signal_path}")
        frame_times = np.arange(frame_count, dtype=np.float64) / fps
        aligned = []
        for label_column in cls.LABEL_COLUMNS:
            column = next((candidate for candidate in cls.SOURCE_COLUMNS[label_column]
                           if candidate in signal), None)
            if column is None:
                aligned.append(np.full(frame_count, np.nan, dtype=np.float64))
                continue
            times = pd.to_numeric(signal["elapsed_time_seconds"], errors="coerce").to_numpy()
            values = pd.to_numeric(signal[column], errors="coerce").to_numpy()
            valid = np.isfinite(times) & np.isfinite(values)
            if label_column in {"HRV", "respiration_rate_bpm"}:
                valid &= (values >= 0) & (values < 65535)
            if not np.any(valid):
                if label_column == "heart_rate_bpm":
                    raise ValueError(f"Signal file contains no {label_column} values: {signal_path}")
                aligned.append(np.full(frame_count, np.nan, dtype=np.float64))
                continue
            aligned.append(np.interp(frame_times, times[valid], values[valid]))
        return np.column_stack(aligned)

    def save_multi_process(self, frames_clips, heart_rate_clips, filename):
        participant, segment_index, movement, view = filename.split("--", 3)
        output_dir = os.path.join(self.cached_path, participant, movement, view, segment_index)
        os.makedirs(output_dir, exist_ok=True)
        input_paths = []
        for clip_index, (frames, heart_rate) in enumerate(zip(frames_clips, heart_rate_clips)):
            input_path = os.path.join(output_dir, f"{filename}_input{clip_index}.npy")
            label_path = os.path.join(output_dir, f"{filename}_label{clip_index}.npy")
            np.save(input_path, frames.astype(self.cache_dtype, copy=False))
            np.save(label_path, heart_rate)
            input_paths.append(input_path)
        return input_paths

    def load_preprocessed_data(self):
        file_list = pd.read_csv(self.file_list_path)["input_files"].tolist()
        if not file_list:
            raise ValueError(self.dataset_name + " dataset loading data error!")
        self.inputs = sorted(file_list)
        self.labels = [self._label_path(input_path) for input_path in self.inputs]
        self.preprocessed_data_len = len(self.inputs)
