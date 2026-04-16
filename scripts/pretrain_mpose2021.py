#!/usr/bin/env python3
"""Pretrain the scratch TCN/GCN skeleton encoder on MPOSE2021.

The script keeps the same encoder/loss/export path as
``train_scratch_similarity.py`` but samples metric-learning pairs from
MPOSE2021 action labels:

  - positive: two pose windows with the same action label
  - negative: two pose windows with different action labels

MPOSE2021 is downloaded through the optional ``mpose`` package when the
converted cache is not already present.
"""

import argparse
from collections import defaultdict
import json
import os
import random
import sys
import time

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pose.landmark_utils import DANCE_JOINTS
from train_scratch_similarity import (
    build_encoder,
    build_siamese_model,
    build_triplet_model,
    export_tflite,
    import_tensorflow,
    verify_tflite,
)


COCO17_TO_DANCE = {
    11: 5,   # left shoulder
    12: 6,   # right shoulder
    13: 7,   # left elbow
    14: 8,   # right elbow
    15: 9,   # left wrist
    16: 10,  # right wrist
    23: 11,  # left hip
    24: 12,  # right hip
    25: 13,  # left knee
    26: 14,  # right knee
    27: 15,  # left ankle
    28: 16,  # right ankle
}

OPENPOSE25_TO_DANCE = {
    11: 5,   # left shoulder
    12: 2,   # right shoulder
    13: 6,   # left elbow
    14: 3,   # right elbow
    15: 7,   # left wrist
    16: 4,   # right wrist
    23: 12,  # left hip
    24: 9,   # right hip
    25: 13,  # left knee
    26: 10,  # right knee
    27: 14,  # left ankle
    28: 11,  # right ankle
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download/convert MPOSE2021 and pretrain a scratch encoder.")
    parser.add_argument("--pose-extractor", default="posenet",
                        choices=["posenet", "openpose", "movenet"],
                        help="MPOSE2021 pose extractor. posenet/movenet use 17-keypoint layout.")
    parser.add_argument("--split", type=int, default=1, choices=[1, 2, 3],
                        help="MPOSE2021 train/test split.")
    parser.add_argument("--mpose-preprocess", default="none",
                        choices=["none", "scale_and_center", "scale_to_unit"],
                        help="Optional preprocessing passed to mpose.MPOSE.")
    parser.add_argument("--converted-output", default=None,
                        help="Cached converted .npz path. Defaults under data/pretrain/mpose2021.")
    parser.add_argument("--overwrite-cache", action="store_true",
                        help="Re-download/re-convert even when converted-output exists.")
    parser.add_argument("--download-only", action="store_true",
                        help="Download MPOSE2021 through mpose and write converted cache, then exit.")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Write converted cache and metadata without training.")
    parser.add_argument("--train-only", action="store_true",
                        help="Use an existing converted cache and skip mpose download.")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Optional cap for quick smoke runs.")
    parser.add_argument("--max-val-samples", type=int, default=None,
                        help="Optional cap for quick smoke runs.")

    parser.add_argument("--model-name", default="mpose2021_gcn_e64_triplet")
    parser.add_argument("--model-type", choices=["tcn", "gcn"], default="gcn")
    parser.add_argument("--output-dir", default="data/models/pretrain/mpose2021")
    parser.add_argument("--output", default=None)
    parser.add_argument("--keras-output", default=None)
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--feature-dims", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--filters", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--validation-steps", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--loss-type", choices=["bce", "triplet"], default="triplet")
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def project_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def default_converted_output(args):
    filename = f"mpose2021_{args.pose_extractor}_split{args.split}_scratch.npz"
    return os.path.join(PROJECT_ROOT, "data", "pretrain", "mpose2021", filename)


def output_paths(args):
    output_dir = project_path(args.output_dir)
    tflite = args.output or os.path.join(output_dir, f"{args.model_name}.tflite")
    keras = args.keras_output or os.path.join(output_dir, f"{args.model_name}_encoder.keras")
    meta = args.metadata_output or os.path.join(output_dir, f"{args.model_name}_meta.json")
    return project_path(tflite), project_path(keras), project_path(meta)


def labels_to_ids(labels):
    labels = np.asarray(labels)
    if labels.ndim > 1:
        labels = np.argmax(labels, axis=-1)
    return labels.astype(np.int64).reshape(-1)


def cap_samples(windows, labels, max_samples, seed):
    if max_samples is None or max_samples <= 0 or len(labels) <= max_samples:
        return windows, labels
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    indices = indices[:int(max_samples)]
    return windows[indices], labels[indices]


def source_mapping(num_keypoints):
    if num_keypoints == 17:
        return COCO17_TO_DANCE
    if num_keypoints == 25:
        return OPENPOSE25_TO_DANCE
    raise ValueError(
        f"Unsupported MPOSE keypoint count: {num_keypoints}. "
        "Expected 17 for PoseNet/MoveNet or 25 for OpenPose."
    )


def convert_mpose_split(x, y, sequence_length, feature_dims, seed, max_samples=None):
    x = np.nan_to_num(np.asarray(x, dtype=np.float32))
    labels = labels_to_ids(y)
    if x.ndim != 4:
        raise ValueError(f"Expected MPOSE X shape (N, T, K, C), got {x.shape}")
    if x.shape[0] != labels.shape[0]:
        raise ValueError(f"X/y length mismatch: {x.shape[0]} vs {labels.shape[0]}")

    windows = np.asarray([
        convert_mpose_sample(sample, sequence_length, feature_dims)
        for sample in x
    ], dtype=np.float32)
    windows, labels = cap_samples(windows, labels, max_samples, seed)
    return windows, labels


def convert_mpose_sample(sample, sequence_length, feature_dims):
    sample = pad_or_crop_time(np.asarray(sample, dtype=np.float32), sequence_length)
    mapping = source_mapping(sample.shape[1])
    channels = sample.shape[2]
    mapped = np.zeros((sequence_length, len(DANCE_JOINTS), 4), dtype=np.float32)

    for out_idx, mediapipe_idx in enumerate(DANCE_JOINTS):
        src_idx = mapping[mediapipe_idx]
        mapped[:, out_idx, 0] = sample[:, src_idx, 0]
        mapped[:, out_idx, 1] = sample[:, src_idx, 1]
        if channels >= 3:
            mapped[:, out_idx, 3] = sample[:, src_idx, 2]
        else:
            mapped[:, out_idx, 3] = 1.0

    normalized = normalize_dance_window(mapped)
    if feature_dims == 2:
        return normalized[:, :, :2]
    if feature_dims == 3:
        return normalized[:, :, :3]
    return normalized


def pad_or_crop_time(sample, sequence_length):
    if sample.shape[0] == sequence_length:
        return sample
    if sample.shape[0] > sequence_length:
        return sample[:sequence_length]
    pad = np.zeros((sequence_length - sample.shape[0], *sample.shape[1:]), dtype=sample.dtype)
    return np.concatenate([sample, pad], axis=0)


def normalize_dance_window(window):
    """Apply scratch-style per-frame hip centering and shoulder scaling.

    Input is already in DANCE_JOINTS order with channels [x, y, z, confidence].
    """
    out = np.nan_to_num(window.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    for frame in out:
        confidence = frame[:, 3]
        if np.count_nonzero(confidence > 0.0) < 4:
            frame[:, :3] = 0.0
            continue
        center = (frame[6, :2] + frame[7, :2]) * 0.5
        frame[:, :2] -= center
        shoulder_width = np.linalg.norm(frame[0, :2] - frame[1, :2])
        if np.isfinite(shoulder_width) and shoulder_width > 1e-6:
            frame[:, :2] /= shoulder_width
    return out


def fetch_and_convert_mpose(args, converted_path):
    try:
        import mpose
    except ImportError as exc:
        raise SystemExit(
            "The optional 'mpose' package is required to download MPOSE2021.\n"
            "Install it with:\n"
            "  pip install mpose==1.2\n"
            "or run the notebook setup cell first."
        ) from exc

    preprocess = None if args.mpose_preprocess == "none" else args.mpose_preprocess
    print("[MPOSE] downloading/loading dataset")
    dataset = mpose.MPOSE(
        pose_extractor=args.pose_extractor,
        split=args.split,
        preprocess=preprocess,
        config_file=None,
        velocities=False,
        remove_zip=False,
        overwrite=args.overwrite_cache,
        verbose=True,
    )
    try:
        dataset.get_info()
    except Exception:
        pass
    x_train, y_train, x_test, y_test = dataset.get_data()

    train_windows, train_labels = convert_mpose_split(
        x_train,
        y_train,
        args.sequence_length,
        args.feature_dims,
        args.seed,
        args.max_train_samples,
    )
    val_windows, val_labels = convert_mpose_split(
        x_test,
        y_test,
        args.sequence_length,
        args.feature_dims,
        args.seed + 1,
        args.max_val_samples,
    )

    os.makedirs(os.path.dirname(converted_path), exist_ok=True)
    np.savez_compressed(
        converted_path,
        train_windows=train_windows,
        train_labels=train_labels,
        val_windows=val_windows,
        val_labels=val_labels,
        pose_extractor=np.array(args.pose_extractor),
        split=np.array(args.split),
        sequence_length=np.array(args.sequence_length),
        feature_dims=np.array(args.feature_dims),
        target_joints=np.asarray(DANCE_JOINTS, dtype=np.int32),
    )
    print(f"[SAVE] converted MPOSE cache: {converted_path}")
    print_dataset_summary(train_windows, train_labels, val_windows, val_labels)
    return train_windows, train_labels, val_windows, val_labels


def load_or_prepare_mpose(args, converted_path):
    if args.train_only and not os.path.exists(converted_path):
        raise FileNotFoundError(f"Converted cache not found: {converted_path}")
    if os.path.exists(converted_path) and not args.overwrite_cache:
        print(f"[LOAD] converted MPOSE cache: {converted_path}")
        data = np.load(converted_path, allow_pickle=False)
        train_windows = data["train_windows"].astype(np.float32)
        train_labels = data["train_labels"].astype(np.int64)
        val_windows = data["val_windows"].astype(np.float32)
        val_labels = data["val_labels"].astype(np.int64)
        print_dataset_summary(train_windows, train_labels, val_windows, val_labels)
        return train_windows, train_labels, val_windows, val_labels
    if args.train_only:
        raise FileNotFoundError(f"Converted cache not found: {converted_path}")
    return fetch_and_convert_mpose(args, converted_path)


def print_dataset_summary(train_windows, train_labels, val_windows, val_labels):
    print("[DATA] train:", train_windows.shape, "labels=", label_summary(train_labels))
    print("[DATA] val:  ", val_windows.shape, "labels=", label_summary(val_labels))


def label_summary(labels):
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    pairs = [f"{int(label)}:{int(count)}" for label, count in zip(unique[:8], counts[:8])]
    suffix = "..." if len(unique) > 8 else ""
    return "{" + ", ".join(pairs) + suffix + f"}} classes={len(unique)}"


class MposePairBatchGenerator:
    """Sample BCE or triplet batches from converted MPOSE windows."""

    def __init__(self, windows, labels, batch_size, steps_per_epoch,
                 loss_type="triplet", noise_std=0.01, seed=42):
        self.windows = np.asarray(windows, dtype=np.float32)
        self.labels = labels_to_ids(labels)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.loss_type = loss_type
        self.noise_std = float(noise_std)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.by_label = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.by_label[int(label)].append(idx)
        self.labels_unique = sorted(self.by_label)
        if len(self.labels_unique) < 2:
            raise ValueError("At least two labels are required for metric pretraining.")

    def __len__(self):
        return self.steps_per_epoch

    def batches(self):
        while True:
            for idx in range(len(self)):
                yield self[idx]

    def __getitem__(self, _idx):
        if self.loss_type == "triplet":
            return self._triplet_batch()
        return self._pair_batch()

    def _pair_batch(self):
        users, refs, labels = [], [], []
        half = self.batch_size // 2
        for _ in range(half):
            a, p = self._sample_positive_indices()
            users.append(self._window(a, augment=True))
            refs.append(self._window(p, augment=True))
            labels.append(1.0)
        while len(labels) < self.batch_size:
            a, n = self._sample_negative_indices()
            users.append(self._window(a, augment=True))
            refs.append(self._window(n, augment=True))
            labels.append(0.0)
        order = np.arange(len(labels))
        self.np_rng.shuffle(order)
        return (
            np.asarray(users, dtype=np.float32)[order],
            np.asarray(refs, dtype=np.float32)[order],
        ), np.asarray(labels, dtype=np.float32)[order]

    def _triplet_batch(self):
        anchors, positives, negatives = [], [], []
        for _ in range(self.batch_size):
            a, p = self._sample_positive_indices()
            _, n = self._sample_negative_indices(anchor_index=a)
            anchors.append(self._window(a, augment=True))
            positives.append(self._window(p, augment=True))
            negatives.append(self._window(n, augment=True))
        return (
            (
                np.asarray(anchors, dtype=np.float32),
                np.asarray(positives, dtype=np.float32),
                np.asarray(negatives, dtype=np.float32),
            ),
            np.zeros(self.batch_size, dtype=np.float32),
        )

    def _sample_positive_indices(self):
        label = self.rng.choice(self.labels_unique)
        indices = self.by_label[label]
        a = self.rng.choice(indices)
        if len(indices) == 1:
            return a, a
        p = self.rng.choice(indices)
        for _ in range(10):
            if p != a:
                break
            p = self.rng.choice(indices)
        return a, p

    def _sample_negative_indices(self, anchor_index=None):
        if anchor_index is None:
            anchor_index = self.rng.randrange(len(self.labels))
        anchor_label = int(self.labels[anchor_index])
        neg_label = self.rng.choice([label for label in self.labels_unique if label != anchor_label])
        return anchor_index, self.rng.choice(self.by_label[neg_label])

    def _window(self, index, augment=False):
        window = self.windows[int(index)].copy()
        if augment and self.noise_std > 0:
            noise = self.np_rng.normal(0.0, self.noise_std, size=window.shape).astype(np.float32)
            window = window + noise
        return np.nan_to_num(window).astype(np.float32)


def write_metadata(path, args, converted_path, input_shape, output_shape, history=None,
                   smoke_metrics=None, training_summary=None):
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": args.model_name,
        "model_type": args.model_type,
        "runtime_score_method": "embedding",
        "runtime_output_mode": "embedding",
        "pretrain_dataset": "MPOSE2021",
        "mpose": {
            "pose_extractor": args.pose_extractor,
            "split": args.split,
            "preprocess": args.mpose_preprocess,
            "converted_cache": os.path.relpath(converted_path, PROJECT_ROOT),
        },
        "input_layout": "BTJC",
        "input_shape": input_shape,
        "output_shape": output_shape,
        "target_joints": DANCE_JOINTS,
        "config": vars(args),
        "history": history or {},
        "training_summary": training_summary or {},
        "smoke_metrics": smoke_metrics or {},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def smoke_embedding_metrics(encoder, windows, labels, max_pairs=256, seed=42):
    rng = random.Random(seed)
    labels = labels_to_ids(labels)
    by_label = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[int(label)].append(idx)
    unique = sorted(by_label)
    if len(unique) < 2:
        return {}

    same, diff = [], []
    for _ in range(min(max_pairs, len(windows))):
        label = rng.choice(unique)
        indices = by_label[label]
        a = rng.choice(indices)
        p = rng.choice(indices)
        neg_label = rng.choice([item for item in unique if item != label])
        n = rng.choice(by_label[neg_label])
        emb = encoder.predict(
            np.asarray([windows[a], windows[p], windows[n]], dtype=np.float32),
            verbose=0,
        )
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
        same.append(float(np.sum(emb[0] * emb[1])))
        diff.append(float(np.sum(emb[0] * emb[2])))
    return {
        "same_cosine_mean": float(np.mean(same)),
        "different_cosine_mean": float(np.mean(diff)),
        "margin_mean": float(np.mean(np.asarray(same) - np.asarray(diff))),
    }


def summarize_training_history(history):
    """Return explicit best-epoch metadata for the saved/restored model."""
    values = getattr(history, "history", {}) or {}
    val_loss = values.get("val_loss") or []
    train_loss = values.get("loss") or []
    summary = {
        "monitor": "val_loss",
        "restore_best_weights": True,
        "epochs_ran": len(train_loss),
    }
    if val_loss:
        best_index = int(np.argmin(np.asarray(val_loss, dtype=np.float32)))
        summary.update({
            "best_epoch": best_index + 1,
            "best_val_loss": float(val_loss[best_index]),
        })
        if train_loss and best_index < len(train_loss):
            summary["best_epoch_loss"] = float(train_loss[best_index])
    return summary


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    converted_path = project_path(args.converted_output) if args.converted_output else default_converted_output(args)
    train_windows, train_labels, val_windows, val_labels = load_or_prepare_mpose(args, converted_path)
    if args.download_only or args.prepare_only:
        print("[DONE] prepared MPOSE cache")
        return

    sample = train_windows[0]
    output_path, keras_output, metadata_output = output_paths(args)
    if args.dry_run:
        write_metadata(
            metadata_output,
            args,
            converted_path,
            [1, args.sequence_length, len(DANCE_JOINTS), args.feature_dims],
            [1, args.embedding_dim],
        )
        print(f"[DRY-RUN] metadata written: {metadata_output}")
        return

    tf = import_tensorflow()
    tf.keras.utils.set_random_seed(args.seed)
    encoder = build_encoder(
        tf,
        sequence_length=args.sequence_length,
        num_joints=len(DANCE_JOINTS),
        feature_dims=args.feature_dims,
        embedding_dim=args.embedding_dim,
        filters=args.filters,
        blocks=args.blocks,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        model_type=args.model_type,
    )
    if args.loss_type == "triplet":
        model = build_triplet_model(tf, encoder, args.learning_rate, args.triplet_margin)
    else:
        model = build_siamese_model(tf, encoder, args.learning_rate)
    model.summary()

    train_gen = MposePairBatchGenerator(
        train_windows,
        train_labels,
        args.batch_size,
        args.steps_per_epoch,
        loss_type=args.loss_type,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    val_gen = MposePairBatchGenerator(
        val_windows,
        val_labels,
        args.batch_size,
        args.validation_steps,
        loss_type=args.loss_type,
        noise_std=0.0,
        seed=args.seed + 1000,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=max(1, int(args.patience)),
        restore_best_weights=True,
        verbose=1,
    )
    history = model.fit(
        train_gen.batches(),
        validation_data=val_gen.batches(),
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        epochs=args.epochs,
        verbose=1,
        callbacks=[early_stop],
    )

    os.makedirs(os.path.dirname(keras_output), exist_ok=True)
    encoder.save(keras_output)
    print(f"[SAVE] Keras encoder: {keras_output}")

    size = export_tflite(tf, encoder, output_path, quantize=not args.no_quantize)
    input_shape, output_shape, embedding = verify_tflite(tf, output_path, sample)
    smoke_metrics = smoke_embedding_metrics(encoder, val_windows, val_labels, seed=args.seed)
    history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    training_summary = summarize_training_history(history)
    write_metadata(
        metadata_output,
        args,
        converted_path,
        input_shape,
        output_shape,
        history=history_dict,
        smoke_metrics=smoke_metrics,
        training_summary=training_summary,
    )
    print(f"[SAVE] TFLite encoder: {output_path} ({size / 1024:.1f} KiB)")
    print(f"[VERIFY] input={input_shape} output={output_shape} "
          f"embedding_norm={float(np.linalg.norm(embedding)):.4f}")
    print(f"[BEST] {training_summary}")
    print(f"[METRIC] {smoke_metrics}")
    print(f"[SAVE] metadata: {metadata_output}")
    print("\nFine-tune starting point:")
    print(f"  {keras_output}")


if __name__ == "__main__":
    main()
