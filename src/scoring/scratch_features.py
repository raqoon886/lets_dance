"""Shared pose preprocessing for scratch TFLite similarity models."""

import numpy as np

from pose.landmark_utils import DANCE_JOINTS


def normalize_pose_landmarks(landmarks: np.ndarray, target_joints=None,
                             feature_dims: int = 2) -> np.ndarray:
    """Select dance joints and normalize a single pose frame."""
    if landmarks is None:
        raise ValueError("landmarks must not be None")

    target_joints = list(target_joints or DANCE_JOINTS)
    feature_dims = int(feature_dims)
    if feature_dims not in (2, 3, 4):
        raise ValueError("feature_dims must be one of 2, 3, or 4")

    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] <= max(target_joints):
        raise ValueError( 
            f"Expected landmarks shaped at least ({max(target_joints) + 1}, C), "
            f"got {arr.shape}"
        )
    if arr.shape[1] < min(feature_dims, 3):
        raise ValueError(f"Expected at least {feature_dims} channels, got {arr.shape[1]}")

    if feature_dims == 4 and arr.shape[1] < 4:
        visibility = np.ones((arr.shape[0], 1), dtype=np.float32)
        arr = np.concatenate([arr[:, :3], visibility], axis=1)

    pose = arr[target_joints, :feature_dims].copy()
    pose = np.nan_to_num(pose, nan=0.0, posinf=0.0, neginf=0.0)

    left_hip = target_joints.index(23) if 23 in target_joints else 0
    right_hip = target_joints.index(24) if 24 in target_joints else left_hip
    left_shoulder = target_joints.index(11) if 11 in target_joints else 0
    right_shoulder = target_joints.index(12) if 12 in target_joints else left_shoulder

    center = (pose[left_hip, :2] + pose[right_hip, :2]) * 0.5
    pose[:, :2] -= center

    shoulder_width = np.linalg.norm(pose[left_shoulder, :2] - pose[right_shoulder, :2])
    if np.isfinite(shoulder_width) and shoulder_width > 1e-6:
        pose[:, :2] /= shoulder_width
        if feature_dims >= 3:
            pose[:, 2] /= shoulder_width

    return pose.astype(np.float32)


def build_pose_window(sequence: np.ndarray, end_idx: int, sequence_length: int,
                      target_joints=None, feature_dims: int = 2) -> np.ndarray:
    """Build a normalized fixed-length pose window ending at end_idx."""
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 3:
        raise ValueError(f"Expected sequence shape (frames, joints, channels), got {seq.shape}")

    sequence_length = int(sequence_length)
    end_idx = int(end_idx)
    start_idx = end_idx - sequence_length + 1
    if start_idx < 0:
        raise ValueError(
            f"Not enough frames for sequence_length={sequence_length}, end_idx={end_idx}"
        )

    frames = [
        normalize_pose_landmarks(
            seq[i],
            target_joints=target_joints,
            feature_dims=feature_dims,
        )
        for i in range(start_idx, end_idx + 1)
    ]
    return np.stack(frames, axis=0).astype(np.float32)
