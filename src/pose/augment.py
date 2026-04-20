"""
Skeleton augmentation for contrastive training (MediaPipe 33-keypoint).

Design notes
------------
- All ops operate on raw (T, 33, 3) arrays and preserve T exactly so that
  anchor frame t in the original maps to frame t in the augmented file
  (required by the "same position" positive-pair sampling strategy).
- Augmentations are composed stochastically per file: each op fires with
  its own probability, with intensity sampled uniformly from its range.
  Multiple augmented files with independent seeds give diverse positives.
- Invalid keypoints in the source (all-zero detections) are preserved as
  zero after augmentation to avoid fabricating fake-valid points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.random import Generator


# MediaPipe 33-keypoint left/right mirror pairs
FLIP_PAIRS: Tuple[Tuple[int, int], ...] = (
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28),
    (29, 30), (31, 32),
)

# Kinematic tree as (parent, child) in topological order from hip roots.
# Used for bone-length perturbation (child translation propagates to descendants).
BONE_TREE: Tuple[Tuple[int, int], ...] = (
    (23, 11), (24, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
)

_CHILDREN: dict = {}
for _p, _c in BONE_TREE:
    _CHILDREN.setdefault(_p, []).append(_c)


def _hip_center(pose: np.ndarray) -> np.ndarray:
    """(T, 33, 3) -> (T, 1, 3) midpoint of left/right hip."""
    return (pose[:, 23:24, :] + pose[:, 24:25, :]) * 0.5


def _invalid_mask(pose: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """(T, 33, 3) -> (T, 33, 1) bool mask of zero (invalid) keypoints."""
    return (np.abs(pose).sum(axis=-1, keepdims=True) <= eps)


# ---------- individual ops ----------

def horizontal_flip(pose: np.ndarray) -> np.ndarray:
    """Mirror around image x=0.5 and swap left/right keypoint indices."""
    out = pose.copy()
    out[..., 0] = 1.0 - out[..., 0]
    for l, r in FLIP_PAIRS:
        out[:, [l, r]] = out[:, [r, l]]
    return out


def rotate_xy(pose: np.ndarray, deg: float) -> np.ndarray:
    """Rotate in image plane around per-frame hip center."""
    theta = np.deg2rad(deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=pose.dtype)
    hip = _hip_center(pose)
    out = pose.copy()
    xy = out[..., :2] - hip[..., :2]
    out[..., :2] = xy @ R.T + hip[..., :2]
    return out


def scale_iso(pose: np.ndarray, factor: float) -> np.ndarray:
    """Isotropic scale around per-frame hip center."""
    hip = _hip_center(pose)
    return (pose - hip) * factor + hip


def shear_xy(pose: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Shear in x-y plane around per-frame hip center."""
    hip = _hip_center(pose)
    M = np.array([[1.0, sx, 0.0],
                  [sy,  1.0, 0.0],
                  [0.0, 0.0, 1.0]], dtype=pose.dtype)
    return (pose - hip) @ M.T + hip


def joint_jitter(pose: np.ndarray, sigma: float, rng: Generator) -> np.ndarray:
    """Independent Gaussian noise per keypoint (skips zero keypoints)."""
    noise = rng.normal(0.0, sigma, size=pose.shape).astype(pose.dtype)
    valid = ~_invalid_mask(pose)
    return pose + noise * valid


def bone_length_perturb(pose: np.ndarray,
                        low: float, high: float,
                        rng: Generator) -> np.ndarray:
    """Scale each bone length by a random factor; propagate translation to descendants."""
    out = pose.copy()
    for parent, child in BONE_TREE:
        factor = float(rng.uniform(low, high))
        delta = (out[:, child] - out[:, parent]) * (factor - 1.0)  # (T, 3)
        stack = [child]
        seen = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            out[:, node] += delta
            stack.extend(_CHILDREN.get(node, ()))
    return out


def frame_dropout(pose: np.ndarray, frac: float, rng: Generator) -> np.ndarray:
    """Drop a fraction of interior frames and replace via linear interpolation."""
    T = pose.shape[0]
    drop = rng.random(T) < frac
    drop[0] = drop[-1] = False
    if not drop.any():
        return pose.copy()
    out = pose.copy()
    keep = np.where(~drop)[0]
    for t in np.where(drop)[0]:
        lo = keep[keep < t][-1]
        hi = keep[keep > t][0]
        w = (t - lo) / (hi - lo)
        out[t] = (1 - w) * pose[lo] + w * pose[hi]
    return out


def temporal_smooth(pose: np.ndarray, sigma: float) -> np.ndarray:
    """1-D Gaussian smoothing along time axis (edge-padded)."""
    radius = max(1, int(np.ceil(sigma * 3)))
    t = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (t / sigma) ** 2)
    k = (k / k.sum()).astype(pose.dtype)
    padded = np.pad(pose, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    out = np.zeros_like(pose)
    for i, w in enumerate(k):
        out += w * padded[i:i + pose.shape[0]]
    return out


# ---------- stochastic composition ----------

@dataclass
class AugConfig:
    """Per-operation probability and intensity range.

    Tuned for MediaPipe image-normalized coords (x,y ~ [0,1]) at 30 fps.
    """

    # Off by default: this project evaluates whether the user mirrors the reference
    # choreography exactly, so L/R is semantically load-bearing. Enable only for
    # pretraining on L/R-agnostic motion (e.g. generic action recognition warmup).
    flip_p: float = 0.0

    rot_p: float = 0.3
    rot_deg: Tuple[float, float] = (-10.0, 10.0)

    scale_p: float = 0.3
    scale_range: Tuple[float, float] = (0.9, 1.1)

    shear_p: float = 0.3
    shear_range: Tuple[float, float] = (-0.08, 0.08)

    bone_p: float = 0.3
    bone_range: Tuple[float, float] = (0.95, 1.05)

    smooth_p: float = 0.3
    smooth_sigma: Tuple[float, float] = (0.5, 1.5)   # in frames

    drop_p: float = 0.3
    drop_fraction: Tuple[float, float] = (0.02, 0.08)

    jitter_p: float = 0.3
    jitter_sigma: Tuple[float, float] = (0.003, 0.015)  # fraction of image size


def apply_random(pose: np.ndarray,
                 cfg: AugConfig,
                 rng: Generator) -> Tuple[np.ndarray, dict]:
    """Apply a random composition of augmentations.

    Accepts input shape (T, 33, 3) or (T, 33, 4) — if a 4th channel (visibility)
    is present, it is passed through unchanged and reattached at the end.

    Order is intentional:
        structural (flip, bone) -> geometric (rot, scale, shear)
        -> temporal (smooth, drop) -> measurement noise (jitter).

    Returns
    -------
    augmented : same shape and dtype as input (float32)
    meta      : dict recording which ops fired with which parameters
    """
    extra = None
    if pose.shape[-1] > 3:
        extra = pose[..., 3:].astype(np.float32, copy=True)
        pose = pose[..., :3]
    invalid = _invalid_mask(pose)
    out = pose.astype(np.float32, copy=True)
    meta: dict = {}

    if rng.random() < cfg.flip_p:
        out = horizontal_flip(out)
        meta["flip"] = True

    if rng.random() < cfg.bone_p:
        out = bone_length_perturb(out, *cfg.bone_range, rng=rng)
        meta["bone_range"] = list(cfg.bone_range)

    if rng.random() < cfg.rot_p:
        deg = float(rng.uniform(*cfg.rot_deg))
        out = rotate_xy(out, deg)
        meta["rot_deg"] = deg

    if rng.random() < cfg.scale_p:
        f = float(rng.uniform(*cfg.scale_range))
        out = scale_iso(out, f)
        meta["scale"] = f

    if rng.random() < cfg.shear_p:
        sx = float(rng.uniform(*cfg.shear_range))
        sy = float(rng.uniform(*cfg.shear_range))
        out = shear_xy(out, sx, sy)
        meta["shear"] = [sx, sy]

    if rng.random() < cfg.smooth_p:
        sigma = float(rng.uniform(*cfg.smooth_sigma))
        out = temporal_smooth(out, sigma)
        meta["smooth_sigma"] = sigma

    if rng.random() < cfg.drop_p:
        frac = float(rng.uniform(*cfg.drop_fraction))
        out = frame_dropout(out, frac, rng)
        meta["drop_fraction"] = frac

    if rng.random() < cfg.jitter_p:
        sigma = float(rng.uniform(*cfg.jitter_sigma))
        out = joint_jitter(out, sigma, rng)
        meta["jitter_sigma"] = sigma

    # Preserve originally-invalid keypoints as zero
    out = np.where(invalid, 0.0, out).astype(np.float32)
    if extra is not None:
        out = np.concatenate([out, extra], axis=-1)
    return out, meta
