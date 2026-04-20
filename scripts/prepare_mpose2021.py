"""Download MPOSE2021 and convert to DANCE_JOINTS layout.

Downloads MPOSE2021 via the `mpose` pip package (first run only — subsequent
runs use the package's cached download). Maps each sample's keypoints from
COCO-17 or OpenPose-25 layout to our 12 DANCE_JOINTS, pads/crops to the
target sequence length, and saves to a .npz cache used by the pretraining
script.

Output file is RAW (x, y, confidence) without hip/torso normalization — the
pretrain script applies normalization at load time to match the reference-dance
fine-tuning pipeline exactly.

Usage
-----
    python scripts/prepare_mpose2021.py                          # posenet, split 1, T=30
    python scripts/prepare_mpose2021.py --pose-extractor openpose
    python scripts/prepare_mpose2021.py --sequence-length 60 --split 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding.dataset_contrastive import DANCE_JOINTS  # noqa: E402


# MediaPipe keypoint index → COCO-17 keypoint index
# COCO-17: 0 nose, 5 L_shoulder, 6 R_shoulder, 7 L_elbow, 8 R_elbow,
#          9 L_wrist, 10 R_wrist, 11 L_hip, 12 R_hip, 13 L_knee, 14 R_knee,
#          15 L_ankle, 16 R_ankle
COCO17_TO_DANCE = {
    11: 5,   # L_shoulder
    12: 6,   # R_shoulder
    13: 7,   # L_elbow
    14: 8,   # R_elbow
    15: 9,   # L_wrist
    16: 10,  # R_wrist
    23: 11,  # L_hip
    24: 12,  # R_hip
    25: 13,  # L_knee
    26: 14,  # R_knee
    27: 15,  # L_ankle
    28: 16,  # R_ankle
}

# MediaPipe keypoint index → OpenPose-25 keypoint index (BODY_25)
OPENPOSE25_TO_DANCE = {
    11: 5,   # L_shoulder (OpenPose 5)
    12: 2,   # R_shoulder (OpenPose 2)
    13: 6,   # L_elbow
    14: 3,   # R_elbow
    15: 7,   # L_wrist
    16: 4,   # R_wrist
    23: 12,  # L_hip
    24: 9,   # R_hip
    25: 13,  # L_knee
    26: 10,  # R_knee
    27: 14,  # L_ankle
    28: 11,  # R_ankle
}


def _pad_or_crop_time(sample: np.ndarray, sequence_length: int) -> np.ndarray:
    t = sample.shape[0]
    if t == sequence_length:
        return sample
    if t > sequence_length:
        return sample[:sequence_length]
    pad = np.zeros((sequence_length - t, *sample.shape[1:]), dtype=sample.dtype)
    return np.concatenate([sample, pad], axis=0)


def _convert_sample(sample: np.ndarray, sequence_length: int,
                     mapping: dict, conf_threshold: float = 0.0,
                     min_valid_kp: int = 8) -> np.ndarray:
    """Map (T, K, C) sample to (sequence_length, 12, 2) in DANCE_JOINTS order.

    Input channels: typically (x, y, confidence). If source has <3 channels,
    confidence defaults to 1.0.

    Frames with fewer than `min_valid_kp` keypoints above `conf_threshold`
    are zeroed out entirely (all 12 joints → 0). This mirrors the existing
    pretrain_mpose2021 behavior and prevents absurd values after
    hip/torso normalization when detection partially fails.

    Output: (sequence_length, 12, 2) xy only.
    """
    sample = _pad_or_crop_time(np.asarray(sample, dtype=np.float32),
                                sequence_length)
    in_channels = sample.shape[2]
    xy = np.zeros((sequence_length, len(DANCE_JOINTS), 2), dtype=np.float32)
    conf = np.ones((sequence_length, len(DANCE_JOINTS)), dtype=np.float32)
    for out_idx, mediapipe_idx in enumerate(DANCE_JOINTS):
        src = mapping[mediapipe_idx]
        xy[:, out_idx, 0] = sample[:, src, 0]
        xy[:, out_idx, 1] = sample[:, src, 1]
        if in_channels >= 3:
            conf[:, out_idx] = sample[:, src, 2]
    # Frame-level mask: enough confident joints?
    valid_per_frame = (conf > conf_threshold).sum(axis=-1)   # (T,)
    bad_frames = valid_per_frame < min_valid_kp
    xy[bad_frames] = 0.0
    return xy


def _labels_to_ids(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim > 1:
        labels = np.argmax(labels, axis=-1)
    return labels.astype(np.int64).reshape(-1)


def _convert_split(x: np.ndarray, y: np.ndarray,
                    sequence_length: int, mapping: dict,
                    conf_threshold: float = 0.0, min_valid_kp: int = 8):
    x = np.nan_to_num(np.asarray(x, dtype=np.float32))
    labels = _labels_to_ids(y)
    if x.ndim != 4:
        raise ValueError(f"Expected X shape (N, T, K, C), got {x.shape}")
    if x.shape[0] != labels.shape[0]:
        raise ValueError(f"X/y length mismatch: {x.shape[0]} vs {labels.shape[0]}")
    windows = np.stack([
        _convert_sample(sample, sequence_length, mapping,
                        conf_threshold, min_valid_kp)
        for sample in x
    ], axis=0).astype(np.float32)
    # Report how many frames were zeroed out
    all_zero_mask = (np.abs(windows).sum(axis=(-2, -1)) == 0)  # (N, T)
    n_bad = int(all_zero_mask.sum())
    n_total = windows.shape[0] * windows.shape[1]
    print(f"    zeroed {n_bad}/{n_total} frames ({100*n_bad/max(n_total,1):.1f}%) "
          f"due to low confidence")
    return windows, labels


def _default_output_path(extractor: str, split: int) -> Path:
    return (PROJECT_ROOT / "data" / "pretrain" / "mpose2021"
            / f"mpose2021_{extractor}_split{split}_raw.npz")


def prepare(extractor: str = "posenet", split: int = 1,
            sequence_length: int = 30, output_path: Path | None = None,
            overwrite: bool = False) -> Path:
    """Download + convert MPOSE2021 into a .npz cache in DANCE_JOINTS layout.

    Returns the output path.
    """
    output_path = output_path or _default_output_path(extractor, split)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        print(f"[LOAD] cache already exists: {output_path}")
        return output_path

    try:
        import mpose
    except ImportError as exc:
        raise SystemExit(
            "The 'mpose' package is required. Install with:\n"
            "    pip install mpose==1.2\n"
        ) from exc

    print(f"[MPOSE] downloading split={split} extractor={extractor} ...")
    dataset = mpose.MPOSE(
        pose_extractor=extractor,
        split=split,
        preprocess=None,
        config_file=None,
        velocities=False,
        remove_zip=False,
        overwrite=overwrite,
        verbose=True,
    )
    x_train, y_train, x_val, y_val = dataset.get_data()

    num_kp = x_train.shape[2]
    if num_kp == 17:
        mapping = COCO17_TO_DANCE
    elif num_kp == 25:
        mapping = OPENPOSE25_TO_DANCE
    else:
        raise ValueError(f"Unsupported num_keypoints={num_kp} for extractor={extractor}")

    print(f"[CONVERT] mapping {num_kp}-kp → 12 DANCE_JOINTS, T={sequence_length}")
    train_windows, train_labels = _convert_split(x_train, y_train, sequence_length, mapping)
    val_windows, val_labels = _convert_split(x_val, y_val, sequence_length, mapping)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        train_windows=train_windows,
        train_labels=train_labels,
        val_windows=val_windows,
        val_labels=val_labels,
        pose_extractor=np.array(extractor),
        split=np.array(split),
        sequence_length=np.array(sequence_length),
        num_keypoints=np.array(num_kp),
        target_joints=np.asarray(DANCE_JOINTS, dtype=np.int32),
    )
    print(f"[SAVE] {output_path}")
    print(f"       train: {train_windows.shape}, "
          f"{len(np.unique(train_labels))} classes")
    print(f"       val  : {val_windows.shape}")
    return output_path


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pose-extractor", choices=["posenet", "openpose", "movenet"],
                   default="posenet")
    p.add_argument("--split", type=int, default=1)
    p.add_argument("--sequence-length", type=int, default=30)
    p.add_argument("--output-path", type=str, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main():
    args = parse_args()
    prepare(extractor=args.pose_extractor, split=args.split,
            sequence_length=args.sequence_length,
            output_path=Path(args.output_path) if args.output_path else None,
            overwrite=args.overwrite)


if __name__ == "__main__":
    main()
