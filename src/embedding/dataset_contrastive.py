"""Contrastive dataset builder for reference dance skeletons.

Loads each whitelisted dance's `reference.npy` and all its precomputed
`augment_data/aug_*.npy`, applies 12-joint selection + hip/torso normalization,
and exposes samplers that emit:

- InfoNCE : ((anchor, positive, dance_idx, end_idx), dummy_y)
            — in-batch others act as negatives; dance_idx/end_idx enable
              false-negative masking in the loss.
- Triplet : ((anchor, positive, negative), dummy_y)
            — explicit negatives sampled per example.

Anchor is drawn from the original; positive is the SAME window position
(± small jitter) taken from a random augment.

Train/val split
---------------
Each dance's frames are split along the time axis:
    train : end ∈ [T-1, cutoff-1]          (front val_fraction of frames = test only)
    val   : end ∈ [cutoff, n-1]
where cutoff = floor(n * (1 - val_fraction)). This guarantees val windows
never overlap with any train window.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


DANCE_JOINTS: Tuple[int, ...] = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)

TARGET_DANCES: Tuple[str, ...] = (
    "404_dance", "basic_movement", "beginner_wave", "cheerup_dance",
    "hiphop_move", "kpop_basic", "rasputin", "shuffle_dance",
)


def normalize_pose(pose_12: np.ndarray) -> np.ndarray:
    """Hip-center + torso-scale normalization over a sequence.

    scale = max(|mid_shoulder - mid_hip|, 0.5 * |L_shoulder - R_shoulder|)
    Torso length is the primary measure (canonical in skeleton literature).
    Half shoulder-width acts as a floor so that when the torso projection
    collapses (e.g., steep forward bend) the scale does not explode.

    Input  : (T, 12, C>=2)
    Output : (T, 12, C)  translation + scale invariant
    """
    out = pose_12.astype(np.float32, copy=True)
    hip = (out[:, 6:7, :] + out[:, 7:8, :]) * 0.5         # (T, 1, C)
    out = out - hip
    shoulder = (out[:, 0:1, :] + out[:, 1:2, :]) * 0.5    # (T, 1, C), hip-centered
    torso = np.linalg.norm(shoulder[..., :2], axis=-1, keepdims=True)  # (T, 1, 1)
    shoulder_vec = out[:, 0:1, :2] - out[:, 1:2, :2]
    shoulder_w = np.linalg.norm(shoulder_vec, axis=-1, keepdims=True)  # (T, 1, 1)
    scale = np.maximum(torso, 0.5 * shoulder_w)
    scale = np.maximum(scale, 1e-6)
    out = out / scale
    return np.nan_to_num(out).astype(np.float32)


def _load_and_process(npy_path: Path, feature_dims: int) -> np.ndarray | None:
    arr = np.load(npy_path)
    if arr.ndim != 3 or arr.shape[1] != 33 or arr.shape[2] < 2:
        return None
    arr = arr[:, list(DANCE_JOINTS), :feature_dims]
    return normalize_pose(arr)


def sample_candidate_indices(num_frames: int, reference_index: int,
                             tolerance_frames: int, stride: int = 1,
                             min_end_index: int = 0) -> List[int]:
    """Match the runtime candidate-window search around a target frame index."""
    num_frames = int(num_frames)
    reference_index = int(reference_index)
    tolerance_frames = max(0, int(tolerance_frames))
    stride = max(1, int(stride))
    min_end_index = max(0, int(min_end_index))
    if num_frames <= 0:
        return []

    lo = max(min_end_index, reference_index - tolerance_frames)
    hi = min(num_frames - 1, reference_index + tolerance_frames)
    if hi < lo:
        return []

    indices = list(range(lo, hi + 1, stride))
    if reference_index < lo or reference_index > hi:
        return indices
    if reference_index not in indices:
        indices.append(reference_index)
        indices.sort()
    return indices


def _apply_joint_dropout(window: np.ndarray, rng: np.random.Generator,
                         dropout_prob: float) -> np.ndarray:
    if dropout_prob <= 0.0:
        return window
    out = window.astype(np.float32, copy=True)
    T, J, _ = out.shape
    drop_mask = rng.random((T, J)) < float(dropout_prob)
    for t in range(T):
        for j in range(J):
            if not drop_mask[t, j]:
                continue
            if t > 0 and rng.random() < 0.7:
                out[t, j] = out[t - 1, j]
            else:
                out[t, j] = 0.0
    return out


def _apply_frame_hold(window: np.ndarray, rng: np.random.Generator,
                      hold_prob: float) -> np.ndarray:
    if hold_prob <= 0.0:
        return window
    out = window.astype(np.float32, copy=True)
    for t in range(1, len(out)):
        if rng.random() < float(hold_prob):
            out[t] = out[t - 1]
    return out


def _apply_temporal_warp(window: np.ndarray, rng: np.random.Generator,
                         warp_prob: float, warp_strength: float) -> np.ndarray:
    if warp_prob <= 0.0 or warp_strength <= 0.0 or rng.random() >= float(warp_prob):
        return window

    out = np.empty_like(window, dtype=np.float32)
    T = int(window.shape[0])
    if T <= 2:
        return window.astype(np.float32, copy=True)

    base = np.arange(T, dtype=np.float32)
    center = 0.5 * (T - 1)
    scale = float(rng.uniform(max(0.5, 1.0 - warp_strength), 1.0 + warp_strength))
    shift = float(rng.uniform(-warp_strength, warp_strength) * (T - 1) * 0.25)
    positions = np.clip(center + (base - center) / scale + shift, 0.0, float(T - 1))

    for j in range(window.shape[1]):
        for c in range(window.shape[2]):
            out[:, j, c] = np.interp(positions, base, window[:, j, c]).astype(np.float32)
    return out


def apply_runtime_window_augment(window: np.ndarray, rng: np.random.Generator,
                                 gaussian_sigma: float = 0.0,
                                 joint_dropout_prob: float = 0.0,
                                 frame_hold_prob: float = 0.0,
                                 temporal_warp_prob: float = 0.0,
                                 temporal_warp_strength: float = 0.0) -> np.ndarray:
    """Apply lightweight detector-style corruption to a normalized pose window."""
    out = np.asarray(window, dtype=np.float32).copy()
    out = _apply_temporal_warp(out, rng, temporal_warp_prob, temporal_warp_strength)
    out = _apply_frame_hold(out, rng, frame_hold_prob)
    out = _apply_joint_dropout(out, rng, joint_dropout_prob)
    if gaussian_sigma > 0.0:
        out += rng.normal(0.0, gaussian_sigma, size=out.shape).astype(np.float32)
    return np.nan_to_num(out).astype(np.float32)


@dataclass
class DanceBundle:
    name: str
    original: np.ndarray
    augments: List[np.ndarray]

    @property
    def num_frames(self) -> int:
        return int(self.original.shape[0])


class ContrastiveSequenceStore:
    def __init__(self, root: str | Path, dances: Sequence[str] = TARGET_DANCES,
                 feature_dims: int = 2, verbose: bool = True):
        self.root = Path(root)
        self.feature_dims = feature_dims
        self.bundles: List[DanceBundle] = []

        for name in dances:
            d = self.root / name
            ref_path = d / "reference.npy"
            if not ref_path.exists():
                if verbose:
                    print(f"[skip] {name}: reference.npy 없음")
                continue
            original = _load_and_process(ref_path, feature_dims)
            if original is None or original.shape[0] < 4:
                if verbose:
                    print(f"[skip] {name}: invalid shape")
                continue
            aug_dir = d / "augment_data"
            augments: List[np.ndarray] = []
            if aug_dir.exists():
                for p in sorted(aug_dir.glob("aug_*.npy")):
                    a = _load_and_process(p, feature_dims)
                    if a is not None and a.shape == original.shape:
                        augments.append(a)
            self.bundles.append(DanceBundle(name, original, augments))
            if verbose:
                print(f"[{name}] T={original.shape[0]}, augments={len(augments)}")

        if not self.bundles:
            raise RuntimeError(f"No usable dances under {root}")

    def __len__(self) -> int:
        return len(self.bundles)

    @property
    def names(self) -> List[str]:
        return [b.name for b in self.bundles]


class ContrastiveBatchGenerator:
    """Yields `(x_tuple, dummy_y)` batches indefinitely.

    For InfoNCE:
        x_tuple = (anchor, positive, dance_idx, end_idx)
    For Triplet:
        x_tuple = (anchor, positive, negative)

    Parameters
    ----------
    split : "train" or "val" — restricts `end_idx` to non-overlapping ranges
    val_fraction : fraction of each sequence's tail reserved for val
    """

    def __init__(self, store: ContrastiveSequenceStore,
                 sequence_length: int, batch_size: int, steps_per_epoch: int,
                 mode: str = "infonce", positive_jitter: int = 2,
                 negative_gap: int = 45, cross_song_prob: float = 0.75,
                 false_negative_gap: int | None = None,
                 hard_negative_min_gap: int | None = None,
                 hard_negative_max_gap: int | None = None,
                 hard_negative_prob: float = 0.0,
                 split: str = "train", val_fraction: float = 0.15,
                 runtime_jitter: float = 0.0,
                 joint_dropout_prob: float = 0.0,
                 frame_hold_prob: float = 0.0,
                 temporal_warp_prob: float = 0.0,
                 temporal_warp_strength: float = 0.0,
                 seed: int = 42):
        self.store = store
        self.T = int(sequence_length)
        self.bs = int(batch_size)
        self.steps = int(steps_per_epoch)
        assert mode in ("infonce", "triplet"), mode
        assert split in ("train", "val"), split
        self.mode = mode
        self.pj = int(positive_jitter)
        self.neg_gap = int(negative_gap)
        self.false_negative_gap = int(false_negative_gap or negative_gap)
        self.hard_negative_min_gap = int(
            hard_negative_min_gap if hard_negative_min_gap is not None
            else max(self.pj + 2, 4))
        self.hard_negative_max_gap = int(
            hard_negative_max_gap if hard_negative_max_gap is not None
            else max(self.hard_negative_min_gap, self.neg_gap))
        self.hard_negative_prob = float(hard_negative_prob)
        self.cross_song_prob = float(cross_song_prob)
        self.split = split
        self.val_fraction = float(val_fraction)
        self.runtime_jitter = float(runtime_jitter)
        self.joint_dropout_prob = float(joint_dropout_prob)
        self.frame_hold_prob = float(frame_hold_prob)
        self.temporal_warp_prob = float(temporal_warp_prob)
        self.temporal_warp_strength = float(temporal_warp_strength)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed + 987654321)
        self._end_ranges: List[Tuple[int, int]] = self._compute_end_ranges()

    def _compute_end_ranges(self) -> List[Tuple[int, int]]:
        """Precompute per-bundle valid end-index range for the chosen split.

        Raises ValueError if any bundle is too short to yield at least one
        window, or if val split ends up with no valid positions. This is
        preferred over silent fallback because a silent fallback would either
        crash later in `_random_end` or cause train/val overlap (leakage).
        """
        ranges: List[Tuple[int, int]] = []
        for b in self.store.bundles:
            n = b.num_frames
            if n < self.T:
                raise ValueError(
                    f"Dance '{b.name}' has {n} frames, which is less than "
                    f"sequence_length={self.T}. Remove it from `dances` or "
                    f"reduce sequence_length.")
            cutoff = max(self.T, int(round(n * (1.0 - self.val_fraction))))
            if self.split == "train":
                lo, hi = self.T - 1, cutoff - 1
            else:  # val
                lo, hi = max(self.T - 1, cutoff), n - 1
            if hi < lo:
                raise ValueError(
                    f"Dance '{b.name}' (T={n}) has no valid windows for "
                    f"split='{self.split}', val_fraction={self.val_fraction}, "
                    f"sequence_length={self.T}. Use a different val_fraction "
                    f"or a longer sequence.")
            ranges.append((lo, hi))
        return ranges

    def __len__(self) -> int:
        return self.steps

    def _window(self, seq: np.ndarray, end: int) -> np.ndarray:
        return seq[end - self.T + 1: end + 1]

    def _random_end(self, i: int) -> int:
        lo, hi = self._end_ranges[i]
        return self.rng.randint(lo, hi)

    def _clamp_end(self, i: int, end: int) -> int:
        lo, hi = self._end_ranges[i]
        return max(lo, min(end, hi))

    def _sample_anchor_positive(self) -> Tuple[np.ndarray, np.ndarray, int, int]:
        i = self.rng.randrange(len(self.store))
        b = self.store.bundles[i]
        end = self._random_end(i)
        anchor = self._window(b.original, end)

        if b.augments:
            aug = self.rng.choice(b.augments)
            jitter = self.rng.randint(-self.pj, self.pj) if self.pj > 0 else 0
            end_p = self._clamp_end(i, end + jitter)
            positive = self._window(aug, end_p)
        else:
            positive = anchor  # fallback
        return anchor, positive, i, end

    def _sample_negative(self, anchor_i: int, anchor_end: int) -> np.ndarray:
        same_song = len(self.store) < 2 or self.rng.random() >= self.cross_song_prob
        if same_song:
            i = anchor_i
            end = self._sample_same_song_negative_end(i, anchor_end)
        else:
            choices = [k for k in range(len(self.store)) if k != anchor_i]
            i = self.rng.choice(choices)
            end = self._random_end(i)
        b = self.store.bundles[i]
        if b.augments and self.rng.random() < 0.5:
            seq = self.rng.choice(b.augments)
        else:
            seq = b.original
        return self._window(seq, end)

    def _sample_same_song_negative_end(self, bundle_idx: int, anchor_end: int) -> int:
        use_hard_negative = (
            self.hard_negative_prob > 0.0 and
            self.rng.random() < self.hard_negative_prob and
            self.hard_negative_max_gap >= self.hard_negative_min_gap
        )
        min_gap = self.hard_negative_min_gap if use_hard_negative else self.neg_gap
        max_gap = self.hard_negative_max_gap if use_hard_negative else None

        for _ in range(40):
            end = self._random_end(bundle_idx)
            dist = abs(end - anchor_end)
            if dist < min_gap:
                continue
            if max_gap is not None and dist > max_gap:
                continue
            return end

        if max_gap is not None:
            target = anchor_end + max_gap if self.rng.random() < 0.5 else anchor_end - max_gap
        else:
            target = anchor_end + min_gap
        return self._clamp_end(bundle_idx, target)

    def _augment_windows(self, arr: np.ndarray) -> np.ndarray:
        out = np.asarray(arr, dtype=np.float32)
        use_runtime_aug = any([
            self.runtime_jitter > 0.0,
            self.joint_dropout_prob > 0.0,
            self.frame_hold_prob > 0.0,
            self.temporal_warp_prob > 0.0,
        ])
        if not use_runtime_aug:
            return out
        return np.stack([
            apply_runtime_window_augment(
                window,
                self.np_rng,
                gaussian_sigma=self.runtime_jitter,
                joint_dropout_prob=self.joint_dropout_prob,
                frame_hold_prob=self.frame_hold_prob,
                temporal_warp_prob=self.temporal_warp_prob,
                temporal_warp_strength=self.temporal_warp_strength,
            )
            for window in out
        ], axis=0).astype(np.float32)

    def _generate_batch(self):
        anchors: List[np.ndarray] = []
        positives: List[np.ndarray] = []
        negatives: List[np.ndarray] = []
        dance_idx: List[int] = []
        end_idx: List[int] = []
        for _ in range(self.bs):
            a, p, i, e = self._sample_anchor_positive()
            anchors.append(a)
            positives.append(p)
            dance_idx.append(i)
            end_idx.append(e)
            if self.mode == "triplet":
                negatives.append(self._sample_negative(i, e))
        A = self._augment_windows(np.stack(anchors).astype(np.float32))
        P = self._augment_windows(np.stack(positives).astype(np.float32))
        dummy = np.zeros((self.bs,), dtype=np.float32)
        if self.mode == "triplet":
            N = self._augment_windows(np.stack(negatives).astype(np.float32))
            return (A, P, N), dummy
        D = np.asarray(dance_idx, dtype=np.int32)
        E = np.asarray(end_idx, dtype=np.int32)
        return (A, P, D, E), dummy

    def batches(self):
        while True:
            for _ in range(self.steps):
                yield self._generate_batch()
