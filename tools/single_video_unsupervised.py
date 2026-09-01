"""Run traditional rPPG methods on one video without ground-truth labels.

This command is intentionally separate from main.py because the stock
unsupervised_method mode is a benchmark/evaluation path that expects dataset
loaders and label files. This script takes a video directly, extracts face ROIs,
runs the unsupervised methods, and writes prediction artifacts.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import scipy.signal

TOOLBOX_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLBOX_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from evaluation.post_process import _calculate_fft_hr, _calculate_peak_hr, _detrend
from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN
from unsupervised_methods.methods.GREEN import GREEN
from unsupervised_methods.methods.ICA_POH import ICA_POH
from unsupervised_methods.methods.LGI import LGI
from unsupervised_methods.methods.OMIT import OMIT
from unsupervised_methods.methods.PBV import PBV
from unsupervised_methods.methods.POS_WANG import POS_WANG


METHODS = {
    "POS": lambda frames, fs: POS_WANG(frames, fs),
    "CHROM": lambda frames, fs: CHROME_DEHAAN(frames, fs),
    "ICA": lambda frames, fs: ICA_POH(frames, fs),
    "GREEN": lambda frames, fs: GREEN(frames),
    "LGI": lambda frames, fs: LGI(frames),
    "PBV": lambda frames, fs: PBV(frames),
    "OMIT": lambda frames, fs: OMIT(frames),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate rPPG HR predictions from a single video without ground truth."
    )
    parser.add_argument("video", type=Path, help="Input video path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV, plots, BVP signals, and ROI archive.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["POS"],
        choices=sorted(METHODS.keys()),
        help="Traditional methods to run.",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=None,
        help="Video frame rate. Defaults to the FPS reported by OpenCV.",
    )
    parser.add_argument(
        "--hr-method",
        choices=["FFT", "peak"],
        default="FFT",
        help="Heart-rate estimator used on each BVP window.",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=10.0,
        help="HR window size in seconds when --use-smaller-window is set.",
    )
    parser.add_argument(
        "--use-smaller-window",
        action="store_true",
        help="Estimate HR on fixed windows instead of the whole video.",
    )
    parser.add_argument("--resize-width", type=int, default=72, help="ROI output width.")
    parser.add_argument("--resize-height", type=int, default=72, help="ROI output height.")
    parser.add_argument(
        "--no-crop-face",
        action="store_true",
        help="Use full frames as ROI instead of face detection.",
    )
    parser.add_argument(
        "--face-detector-backend",
        choices=["HC", "Y5F"],
        default="HC",
        help="Face detector backend for ROI extraction: HC or Y5F.",
    )
    parser.add_argument(
        "--large-box-coef",
        type=float,
        default=1.5,
        help="Scale factor applied to detected face boxes.",
    )
    parser.add_argument(
        "--dynamic-detection-frequency",
        type=int,
        default=30,
        help="Detect a new face box every N frames. Use 0 to detect once.",
    )
    parser.add_argument(
        "--use-median-box",
        action="store_true",
        help="Use the median of dynamically detected boxes for every frame.",
    )
    parser.add_argument("--start-sec", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument(
        "--end-sec",
        type=float,
        default=None,
        help="End time in seconds. Defaults to the end of the video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process after start/end trimming.",
    )
    parser.add_argument(
        "--save-roi-video",
        action="store_true",
        help="Also save a playable ROI preview video.",
    )
    return parser.parse_args()


def read_video(video_path, start_sec=0.0, end_sec=None, max_frames=None):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 0 else 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = None if end_sec is None else max(start_frame, int(round(end_sec * fps)))
    expected_frames = None
    if total_frames > 0:
        effective_end_frame = min(end_frame if end_frame is not None else total_frames, total_frames)
        expected_frames = max(0, effective_end_frame - start_frame)
    elif end_frame is not None:
        expected_frames = max(0, end_frame - start_frame)
    if max_frames is not None:
        expected_frames = min(expected_frames, max_frames) if expected_frames is not None else max_frames

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    frame_index = start_frame
    with tqdm(total=expected_frames, desc="Reading frames", unit="frame") as progress:
        while True:
            if end_frame is not None and frame_index >= end_frame:
                break
            if max_frames is not None and len(frames) >= max_frames:
                break

            success, frame_bgr = capture.read()
            if not success:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            frame_index += 1
            progress.update(1)

    capture.release()
    if not frames:
        raise ValueError("No frames were read from the requested video range.")
    return np.asarray(frames, dtype=np.uint8), fps, start_frame


def video_range_info(capture, start_sec=0.0, end_sec=None, max_frames=None):
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 0 else 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = None if end_sec is None else max(start_frame, int(round(end_sec * fps)))
    expected_frames = None
    if total_frames > 0:
        effective_end_frame = min(end_frame if end_frame is not None else total_frames, total_frames)
        expected_frames = max(0, effective_end_frame - start_frame)
    elif end_frame is not None:
        expected_frames = max(0, end_frame - start_frame)
    if max_frames is not None:
        expected_frames = min(expected_frames, max_frames) if expected_frames is not None else max_frames
    return fps, total_frames, start_frame, end_frame, expected_frames


def read_frame_at(capture, frame_index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame_bgr = capture.read()
    if not success:
        return None
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def cascade_path():
    toolbox_root = Path(__file__).resolve().parents[1]
    local_path = toolbox_root / "dataset" / "haarcascade_frontalface_default.xml"
    if local_path.exists():
        return str(local_path)
    cv2_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if cv2_path.exists():
        return str(cv2_path)
    raise FileNotFoundError("Could not find haarcascade_frontalface_default.xml")


def create_face_detector(backend):
    if backend == "HC":
        return cv2.CascadeClassifier(cascade_path())
    if backend == "Y5F":
        from dataset.data_loader.face_detector.YOLO5Face import YOLO5Face

        return YOLO5Face(backend)
    raise ValueError(f"Unsupported face detection backend: {backend}")


def enlarge_box(x, y, width, height, large_box_coef):
    if large_box_coef != 1.0:
        x = max(0, x - (large_box_coef - 1.0) / 2 * width)
        y = max(0, y - (large_box_coef - 1.0) / 2 * height)
        width = large_box_coef * width
        height = large_box_coef * height
    return np.asarray([x, y, width, height], dtype=np.int32)


def detect_face(frame_rgb, detector, backend, large_box_coef):
    frame_uint8 = frame_rgb[:, :, :3].astype(np.uint8)
    if backend == "Y5F":
        result = detector.detect_face(frame_uint8)
        if result is None:
            print("Warning: no face detected; using the full frame for this detection point.")
            return np.asarray([0, 0, frame_rgb.shape[1], frame_rgb.shape[0]], dtype=np.int32)

        x_min, y_min, x_max, y_max = result
        width = x_max - x_min
        height = y_max - y_min
        center_x = x_min + width // 2
        center_y = y_min + height // 2
        square_size = max(width, height)
        x = center_x - (square_size // 2)
        y = center_y - (square_size // 2)
        return enlarge_box(x, y, square_size, square_size, large_box_coef)

    frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
    face_zones = detector.detectMultiScale(frame_bgr)
    if len(face_zones) < 1:
        print("Warning: no face detected; using the full frame for this detection point.")
        x, y, width, height = 0, 0, frame_rgb.shape[1], frame_rgb.shape[0]
    elif len(face_zones) >= 2:
        max_width_index = np.argmax(face_zones[:, 2])
        x, y, width, height = face_zones[max_width_index]
        print("Warning: more than one face detected; using the largest one.")
    else:
        x, y, width, height = face_zones[0]

    return enlarge_box(x, y, width, height, large_box_coef)


def extract_rois(
    frames,
    crop_face,
    face_detector_backend,
    large_box_coef,
    dynamic_detection_frequency,
    use_median_box,
    resize_width,
    resize_height,
):
    if not crop_face:
        detection_boxes = np.asarray([[0, 0, frames.shape[2], frames.shape[1]]], dtype=np.int32)
        reference_indices = np.zeros(frames.shape[0], dtype=np.int32)
    else:
        detector = create_face_detector(face_detector_backend)
        detection_frequency = dynamic_detection_frequency if dynamic_detection_frequency > 0 else frames.shape[0]
        detection_count = int(np.ceil(frames.shape[0] / detection_frequency))
        detection_boxes = []
        for detection_index in tqdm(range(detection_count), desc="Detecting face boxes", unit="box"):
            frame_index = min(detection_index * detection_frequency, frames.shape[0] - 1)
            detection_boxes.append(detect_face(frames[frame_index], detector, face_detector_backend, large_box_coef))
        detection_boxes = np.asarray(detection_boxes, dtype=np.int32)
        reference_indices = np.minimum(
            np.arange(frames.shape[0]) // detection_frequency, len(detection_boxes) - 1
        )
        if use_median_box:
            detection_boxes = np.asarray([np.median(detection_boxes, axis=0).astype(np.int32)])
            reference_indices = np.zeros(frames.shape[0], dtype=np.int32)

    rois = np.zeros((frames.shape[0], resize_height, resize_width, 3), dtype=np.uint8)
    frame_boxes = np.zeros((frames.shape[0], 4), dtype=np.int32)
    for frame_index, frame in enumerate(tqdm(frames, desc="Cropping/resizing ROIs", unit="frame")):
        x, y, width, height = detection_boxes[reference_indices[frame_index]]
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(frame.shape[1], int(x + width))
        y1 = min(frame.shape[0], int(y + height))
        if x1 <= x0 or y1 <= y0:
            x0, y0, x1, y1 = 0, 0, frame.shape[1], frame.shape[0]
        roi = frame[y0:y1, x0:x1, :3]
        rois[frame_index] = cv2.resize(roi, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
        frame_boxes[frame_index] = [x0, y0, x1 - x0, y1 - y0]

    return rois, frame_boxes


def extract_rois_from_video(
    video_path,
    start_sec,
    end_sec,
    max_frames,
    crop_face,
    face_detector_backend,
    large_box_coef,
    dynamic_detection_frequency,
    use_median_box,
    resize_width,
    resize_height,
):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    detected_fps, _, start_frame, end_frame, expected_frames = video_range_info(
        capture,
        start_sec=start_sec,
        end_sec=end_sec,
        max_frames=max_frames,
    )
    if expected_frames == 0:
        capture.release()
        raise ValueError("No frames were found in the requested video range.")

    detection_frequency = dynamic_detection_frequency if dynamic_detection_frequency > 0 else (expected_frames or 1)
    if expected_frames is None:
        detection_count = None
    else:
        detection_count = int(np.ceil(expected_frames / detection_frequency))

    if not crop_face:
        first_frame = read_frame_at(capture, start_frame)
        if first_frame is None:
            capture.release()
            raise ValueError("No frames were read from the requested video range.")
        detection_boxes = np.asarray([[0, 0, first_frame.shape[1], first_frame.shape[0]]], dtype=np.int32)
    else:
        detector = create_face_detector(face_detector_backend)
        detection_boxes = []
        detection_iterable = range(detection_count) if detection_count is not None else range(1)
        for detection_index in tqdm(detection_iterable, desc="Detecting face boxes", unit="box"):
            frame_index = start_frame + detection_index * detection_frequency
            frame = read_frame_at(capture, frame_index)
            if frame is None:
                break
            detection_boxes.append(detect_face(frame, detector, face_detector_backend, large_box_coef))
        if not detection_boxes:
            capture.release()
            raise ValueError("No frames were available for face detection.")
        detection_boxes = np.asarray(detection_boxes, dtype=np.int32)
        if use_median_box:
            detection_boxes = np.asarray([np.median(detection_boxes, axis=0).astype(np.int32)])

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    rois = []
    frame_boxes = []
    frame_index = start_frame
    local_frame_index = 0
    with tqdm(total=expected_frames, desc="Reading/cropping ROI frames", unit="frame") as progress:
        while True:
            if end_frame is not None and frame_index >= end_frame:
                break
            if max_frames is not None and local_frame_index >= max_frames:
                break

            success, frame_bgr = capture.read()
            if not success:
                break
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if crop_face and not use_median_box:
                reference_index = min(local_frame_index // detection_frequency, len(detection_boxes) - 1)
            else:
                reference_index = 0
            x, y, width, height = detection_boxes[reference_index]
            x0 = max(0, int(x))
            y0 = max(0, int(y))
            x1 = min(frame.shape[1], int(x + width))
            y1 = min(frame.shape[0], int(y + height))
            if x1 <= x0 or y1 <= y0:
                x0, y0, x1, y1 = 0, 0, frame.shape[1], frame.shape[0]

            roi = frame[y0:y1, x0:x1, :3]
            rois.append(cv2.resize(roi, (resize_width, resize_height), interpolation=cv2.INTER_AREA))
            frame_boxes.append([x0, y0, x1 - x0, y1 - y0])

            frame_index += 1
            local_frame_index += 1
            progress.update(1)

    capture.release()
    if not rois:
        raise ValueError("No frames were read from the requested video range.")
    return np.asarray(rois, dtype=np.uint8), np.asarray(frame_boxes, dtype=np.int32), detected_fps, start_frame


def filter_bvp(bvp, fs, low_pass=0.6, high_pass=3.3):
    bvp = np.asarray(bvp, dtype=np.double).reshape(-1)
    if len(bvp) < 9:
        return bvp
    bvp = _detrend(bvp, 100)
    nyquist = fs / 2.0
    high = min(high_pass, nyquist * 0.95)
    low = min(low_pass, high * 0.5)
    if low <= 0 or high <= low:
        return bvp
    b, a = scipy.signal.butter(1, [low / nyquist, high / nyquist], btype="bandpass")
    padlen = 3 * max(len(a), len(b))
    if len(bvp) <= padlen:
        return bvp
    return scipy.signal.filtfilt(b, a, bvp)


def estimate_hr(bvp_window, fs, hr_method):
    if len(bvp_window) < 9:
        return np.nan
    if hr_method == "FFT":
        return float(_calculate_fft_hr(bvp_window, fs=fs))
    try:
        return float(_calculate_peak_hr(bvp_window, fs=fs))
    except Exception:
        return np.nan


def window_predictions(bvp, fs, hr_method, use_smaller_window, window_size_sec):
    if use_smaller_window:
        window_frames = max(9, int(round(window_size_sec * fs)))
    else:
        window_frames = len(bvp)

    rows = []
    window_starts = list(range(0, len(bvp), window_frames))
    for start in tqdm(window_starts, desc="Estimating HR windows", unit="window", leave=False):
        stop = min(start + window_frames, len(bvp))
        if stop - start < 9:
            continue
        rows.append(
            {
                "window_index": len(rows),
                "start_frame": start,
                "end_frame_exclusive": stop,
                "start_time_sec": start / fs,
                "end_time_sec": stop / fs,
                "predicted_hr_bpm": estimate_hr(bvp[start:stop], fs, hr_method),
            }
        )
    return rows


def write_hr_csv(path, all_rows):
    print(f"Writing HR predictions CSV: {path}")
    fieldnames = [
        "method",
        "window_index",
        "start_frame",
        "end_frame_exclusive",
        "start_time_sec",
        "end_time_sec",
        "predicted_hr_bpm",
    ]
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


def write_bvp_csv(path, bvp_by_method, fs):
    print(f"Writing BVP signals CSV: {path}")
    max_len = max(len(bvp) for bvp in bvp_by_method.values())
    methods = list(bvp_by_method.keys())
    with open(path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "time_sec", *methods])
        for frame_index in range(max_len):
            row = [frame_index, frame_index / fs]
            for method in methods:
                bvp = bvp_by_method[method]
                row.append(float(bvp[frame_index]) if frame_index < len(bvp) else "")
            writer.writerow(row)


def plot_hr(path, rows_by_method):
    print(f"Writing HR plot: {path}")
    plt.figure(figsize=(10, 5))
    for method, rows in rows_by_method.items():
        if not rows:
            continue
        times = [(row["start_time_sec"] + row["end_time_sec"]) / 2 for row in rows]
        hrs = [row["predicted_hr_bpm"] for row in rows]
        plt.plot(times, hrs, marker="o", label=method)
    plt.xlabel("Time (s)")
    plt.ylabel("Predicted HR (bpm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_bvp(path, bvp_by_method, fs):
    print(f"Writing BVP plot: {path}")
    plt.figure(figsize=(12, 5))
    for method, bvp in bvp_by_method.items():
        time_axis = np.arange(len(bvp)) / fs
        plt.plot(time_axis, bvp, label=method, linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Filtered BVP (a.u.)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_roi_archive(path, rois, boxes, metadata):
    print(f"Writing ROI archive: {path}")
    np.savez_compressed(
        path,
        roi_frames=rois,
        face_boxes=boxes,
        metadata=json.dumps(metadata, indent=2),
    )


def save_roi_video(path, rois, fs):
    print(f"Writing ROI preview video: {path}")
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fs,
        (rois.shape[2], rois.shape[1]),
    )
    for frame_rgb in tqdm(rois, desc="Encoding ROI preview", unit="frame"):
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    args = parse_args()
    video_path = args.video.resolve()
    output_dir = args.output_dir or Path("outputs") / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input video: {video_path}")
    print(f"Output directory: {output_dir.resolve()}")
    print("Streaming video and extracting ROI frames...")

    rois, boxes, detected_fps, start_frame = extract_rois_from_video(
        video_path,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        max_frames=args.max_frames,
        crop_face=not args.no_crop_face,
        face_detector_backend=args.face_detector_backend,
        large_box_coef=args.large_box_coef,
        dynamic_detection_frequency=args.dynamic_detection_frequency,
        use_median_box=args.use_median_box,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
    )
    fs = float(args.fs) if args.fs else detected_fps
    print(f"Extracted {len(rois)} ROI frames starting at source frame {start_frame}.")
    print(f"Using FPS: {fs:.3f} (OpenCV reported {detected_fps:.3f}).")

    metadata = {
        "source_video": str(video_path),
        "fps": fs,
        "detected_fps": detected_fps,
        "start_frame": start_frame,
        "frame_count": int(rois.shape[0]),
        "roi_width": args.resize_width,
        "roi_height": args.resize_height,
        "crop_face": not args.no_crop_face,
        "face_detector_backend": args.face_detector_backend,
        "large_box_coef": args.large_box_coef,
        "dynamic_detection_frequency": args.dynamic_detection_frequency,
        "use_median_box": args.use_median_box,
    }
    roi_path = output_dir / f"{video_path.stem}.rppgroi.npz"
    save_roi_archive(roi_path, rois, boxes, metadata)
    if args.save_roi_video:
        save_roi_video(output_dir / f"{video_path.stem}_roi_preview.mp4", rois, fs)

    rows_by_method = {}
    bvp_by_method = {}
    flat_rows = []
    for method in tqdm(args.methods, desc="Running methods", unit="method"):
        print(f"\nRunning {method}...")
        raw_bvp = METHODS[method](rois, fs)
        print(f"{method}: extracted raw BVP with {len(raw_bvp)} samples.")
        filtered_bvp = filter_bvp(raw_bvp, fs)
        print(f"{method}: filtered BVP and estimating HR.")
        bvp_by_method[method] = filtered_bvp
        rows = window_predictions(
            filtered_bvp,
            fs,
            args.hr_method,
            args.use_smaller_window,
            args.window_size,
        )
        rows_by_method[method] = rows
        for row in rows:
            flat_rows.append({"method": method, **row})

    write_hr_csv(output_dir / "hr_predictions.csv", flat_rows)
    write_bvp_csv(output_dir / "bvp_signals.csv", bvp_by_method, fs)
    plot_hr(output_dir / "hr_predictions.png", rows_by_method)
    plot_bvp(output_dir / "bvp_signals.png", bvp_by_method, fs)

    with open(output_dir / "metadata.json", "w") as metadata_file:
        json.dump({**metadata, "methods": args.methods, "hr_method": args.hr_method}, metadata_file, indent=2)

    print(f"Wrote outputs to: {output_dir.resolve()}")
    print(f"ROI archive: {roi_path.resolve()}")


if __name__ == "__main__":
    main()