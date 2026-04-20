"""Supervised contrastive pretraining of scratch MLP/TCN/GCN on MPOSE2021.

Uses Supervised Contrastive Loss (Khosla 2020, SupCon):
  For each anchor i in a batch, positives = {j : label[j] == label[i], j != i}.
  Loss pulls anchor toward in-batch same-class samples and pushes away from
  different-class samples via cross-entropy over the B×B similarity matrix.

After pretraining, the encoder's weights are saved as `.weights.h5` so that
`scripts/train_scratch_contrastive.py --pretrained-weights <path>` can load
them for downstream fine-tuning on reference-dance data.

Architecture is IDENTICAL to scratch (`src/embedding/models_scratch.py`),
so weight transfer is a 1:1 layer copy — no surgery needed.

Usage
-----
    python scripts/pretrain_scratch_mpose2021.py --model mlp
    python scripts/pretrain_scratch_mpose2021.py --model tcn --embedding-dim 64
    python scripts/pretrain_scratch_mpose2021.py --model gcn --prepare-if-missing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf  # noqa: E402

from src.embedding.dataset_contrastive import normalize_pose  # noqa: E402
from src.embedding.models_scratch import build_encoder  # noqa: E402
from scripts.prepare_mpose2021 import prepare as prepare_mpose  # noqa: E402


# ---------------- Supervised contrastive wrapper ----------------

class SupConWrapper(tf.keras.Model):
    """SupCon wrapper around an encoder. Batch format: (windows (B,T,J,C), labels (B,))."""

    def __init__(self, encoder: tf.keras.Model, temperature: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.temperature = float(temperature)
        self._loss_tracker = tf.keras.metrics.Mean(name="loss")
        self._acc_tracker = tf.keras.metrics.Mean(name="class_retrieval_acc")

    @property
    def metrics(self):
        return [self._loss_tracker, self._acc_tracker]

    def _compute(self, inputs, training: bool):
        windows, labels = inputs
        z = self.encoder(windows, training=training)
        B = tf.shape(z)[0]
        sim = tf.matmul(z, z, transpose_b=True) / self.temperature

        self_mask = tf.eye(B, dtype=tf.bool)
        sim_masked = tf.where(self_mask, tf.fill(tf.shape(sim), -1e9), sim)
        log_prob = tf.nn.log_softmax(sim_masked, axis=1)

        labels_i = tf.cast(labels, tf.int32)
        pos_mask = tf.equal(labels_i[:, None], labels_i[None, :])
        pos_mask = tf.logical_and(pos_mask, tf.logical_not(self_mask))
        pos_mask_f = tf.cast(pos_mask, tf.float32)

        num_pos = tf.reduce_sum(pos_mask_f, axis=1)
        pos_log_prob_sum = tf.reduce_sum(log_prob * pos_mask_f, axis=1)
        per_sample_loss = -pos_log_prob_sum / tf.maximum(num_pos, 1.0)
        has_pos = tf.greater(num_pos, 0)
        per_sample_loss = tf.where(has_pos, per_sample_loss,
                                    tf.zeros_like(per_sample_loss))
        loss = tf.reduce_sum(per_sample_loss) / tf.maximum(
            tf.reduce_sum(tf.cast(has_pos, tf.float32)), 1.0)

        preds = tf.argmax(sim_masked, axis=1, output_type=tf.int32)
        pred_labels = tf.gather(labels_i, preds)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred_labels, labels_i), tf.float32))
        return loss, acc

    def call(self, inputs, training=False):
        windows = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        return self.encoder(windows, training=training)

    def train_step(self, data):
        inputs, _ = data
        with tf.GradientTape() as tape:
            loss, acc = self._compute(inputs, training=True)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self._loss_tracker.update_state(loss)
        self._acc_tracker.update_state(acc)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        inputs, _ = data
        loss, acc = self._compute(inputs, training=False)
        self._loss_tracker.update_state(loss)
        self._acc_tracker.update_state(acc)
        return {m.name: m.result() for m in self.metrics}


# ---------------- Data pipeline ----------------

def _normalize_windows(windows: np.ndarray) -> np.ndarray:
    """Apply hip-center + torso-scale normalization to every window."""
    out = np.empty_like(windows)
    for i in range(len(windows)):
        out[i] = normalize_pose(windows[i])
    return out


def _augment_window(window: tf.Tensor, rot_deg_max: float,
                    jitter_sigma: float) -> tf.Tensor:
    """Per-sample augmentation for SupCon pretraining.

    Flip and isotropic scale are intentionally excluded: flip would teach
    the encoder L/R-invariance (harmful for downstream dance where L/R is
    semantic), and scale_iso is a no-op after hip/torso normalization.
    """
    w = window  # (T, J, C)
    if rot_deg_max > 0.0:
        theta = tf.random.uniform([], -rot_deg_max, rot_deg_max) * (np.pi / 180.0)
        c = tf.cos(theta); s = tf.sin(theta)
        R = tf.stack([tf.stack([c, -s]), tf.stack([s, c])])  # (2, 2)
        xy_rot = tf.einsum("tjc,cd->tjd", w[..., :2], R)
        if w.shape[-1] > 2:
            w = tf.concat([xy_rot, w[..., 2:]], axis=-1)
        else:
            w = xy_rot
    if jitter_sigma > 0.0:
        w = w + tf.random.normal(tf.shape(w), 0.0, jitter_sigma, dtype=w.dtype)
    return w


def _make_dataset(windows: np.ndarray, labels: np.ndarray,
                   batch_size: int, shuffle: bool, seed: int,
                   augment: bool = False, rot_deg_max: float = 0.0,
                   jitter_sigma: float = 0.0) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((windows, labels))
    if shuffle:
        ds = ds.shuffle(len(windows), seed=seed, reshuffle_each_iteration=True)
    if augment and (rot_deg_max > 0.0 or jitter_sigma > 0.0):
        ds = ds.map(
            lambda x, y: (_augment_window(x, rot_deg_max, jitter_sigma), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    ds = ds.batch(batch_size, drop_remainder=True).repeat()
    ds = ds.map(lambda x, y: ((x, y), tf.zeros(tf.shape(x)[0], dtype=tf.float32)))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _lr_schedule(base_lr, warmup_epochs, total_epochs, min_lr):
    warmup_epochs = max(0, int(warmup_epochs))
    total_epochs = max(1, int(total_epochs))

    def schedule(epoch, lr):
        del lr
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return float(min_lr + (base_lr - min_lr) * 0.5 *
                     (1.0 + np.cos(np.pi * progress)))

    return schedule


# ---------------- Train ----------------

def train(args: argparse.Namespace) -> Dict[str, Any]:
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    data_path = Path(args.data_path) if args.data_path else None
    if data_path is None:
        data_path = (PROJECT_ROOT / "data" / "pretrain" / "mpose2021" /
                     f"mpose2021_{args.pose_extractor}_split{args.split}_raw.npz")
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    # Download + convert if missing
    if not data_path.exists():
        if args.prepare_if_missing:
            print(f"[PREPARE] cache missing, downloading MPOSE2021...")
            data_path = prepare_mpose(
                extractor=args.pose_extractor,
                split=args.split,
                sequence_length=args.sequence_length,
                output_path=data_path)
        else:
            raise SystemExit(
                f"Cache not found: {data_path}\n"
                f"Run `python scripts/prepare_mpose2021.py` first, "
                f"or pass --prepare-if-missing.")

    npz = np.load(data_path)
    train_windows = npz["train_windows"].astype(np.float32)
    train_labels = npz["train_labels"].astype(np.int32)
    val_windows = npz["val_windows"].astype(np.float32)
    val_labels = npz["val_labels"].astype(np.int32)

    # Drop any extra channels beyond xy (prepare_mpose2021 now saves xy only,
    # but keep defensive slicing for backward compat with older caches).
    if train_windows.shape[-1] > 2:
        train_windows = train_windows[..., :2]
        val_windows = val_windows[..., :2]

    # Hip/torso normalization (identical to reference-dance fine-tune pipeline)
    print("[NORM] applying hip/torso normalization...")
    train_windows = _normalize_windows(train_windows)
    val_windows = _normalize_windows(val_windows)

    T, J, C = train_windows.shape[1:]
    num_classes = int(train_labels.max()) + 1
    print(f"[DATA] train={len(train_windows)}, val={len(val_windows)}, "
          f"T={T}, J={J}, C={C}, num_classes={num_classes}")

    # Build encoder (identical architecture to scratch)
    model_kwargs: Dict[str, Any] = {"dropout": args.dropout}
    if args.model == "mlp":
        model_kwargs["hidden"] = args.mlp_hidden
    elif args.model == "tcn":
        model_kwargs.update(filters=args.tcn_filters, blocks=args.tcn_blocks,
                             kernel_size=args.tcn_kernel)
    else:
        model_kwargs.update(filters=args.gcn_filters, blocks=args.gcn_blocks,
                             kernel_size=args.gcn_kernel,
                             partition_strategy=args.gcn_partition)

    encoder = build_encoder(
        args.model, sequence_length=T, num_joints=J,
        feature_dims=C, embedding_dim=args.embedding_dim, **model_kwargs)
    encoder.summary()

    wrapper = SupConWrapper(encoder, temperature=args.temperature)
    wrapper.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate, clipnorm=1.0))

    train_ds = _make_dataset(
        train_windows, train_labels, args.batch_size,
        shuffle=True, seed=args.seed,
        augment=True,
        rot_deg_max=float(getattr(args, "augment_rot_deg", 0.0) or 0.0),
        jitter_sigma=float(getattr(args, "runtime_jitter", 0.0) or 0.0),
    )
    val_ds = _make_dataset(val_windows, val_labels,
                            args.batch_size, shuffle=False,
                            seed=args.seed + 1000)

    steps_per_epoch = max(1, len(train_windows) // args.batch_size)
    validation_steps = max(1, len(val_windows) // args.batch_size)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name or f"mpose_{args.model}_e{args.embedding_dim}"
    weights_path = output_dir / f"{model_name}.weights.h5"
    meta_path = output_dir / f"{model_name}_meta.json"

    class _BestEncoderCheckpoint(tf.keras.callbacks.Callback):
        def __init__(self, enc, path, monitor="val_loss"):
            super().__init__()
            self.enc = enc
            self.path = str(path)
            self.monitor = monitor
            self.best = float("inf")

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            v = logs.get(self.monitor)
            if v is not None and v < self.best:
                self.best = float(v)
                self.enc.save_weights(self.path)

    lr_sched = _lr_schedule(args.learning_rate, args.warmup_epochs,
                             args.epochs, args.min_learning_rate)
    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lr_sched, verbose=0),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience,
            restore_best_weights=True, verbose=1),
        _BestEncoderCheckpoint(encoder, weights_path),
    ]

    history = wrapper.fit(
        train_ds, validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=args.epochs, callbacks=callbacks, verbose=args.verbose)

    # Final save (EarlyStopping already restored best weights into encoder)
    encoder.save_weights(weights_path)

    meta = {
        "model_name": model_name,
        "model_type": args.model,
        "pretrain_dataset": "mpose2021",
        "pretrain_objective": "SupCon",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_shape": [1, T, J, C],
        "output_shape": [1, args.embedding_dim],
        "num_train": int(len(train_windows)),
        "num_val": int(len(val_windows)),
        "num_classes": num_classes,
        "config": {k: (v if not isinstance(v, Path) else str(v))
                   for k, v in vars(args).items()},
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"[SAVE] weights : {weights_path}")
    print(f"[SAVE] metadata: {meta_path}")
    print()
    print("Fine-tune with:")
    print(f"  python scripts/train_scratch_contrastive.py --model {args.model} "
          f"--embedding-dim {args.embedding_dim} "
          f"--pretrained-weights {weights_path} "
          f"--learning-rate 1e-4 --epochs 20 --warmup-epochs 1")

    return {
        "encoder": encoder, "wrapper": wrapper, "history": history.history,
        "paths": {"weights": str(weights_path), "meta": str(meta_path)},
        "num_classes": num_classes,
    }


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Data
    p.add_argument("--data-path", default=None,
                   help="path to prepared raw npz (default: auto from pose-extractor/split)")
    p.add_argument("--pose-extractor", choices=["posenet", "openpose", "movenet"],
                   default="posenet")
    p.add_argument("--split", type=int, default=1)
    p.add_argument("--sequence-length", type=int, default=30)
    p.add_argument("--prepare-if-missing", action="store_true",
                   help="download MPOSE2021 if cache is missing")
    # Model
    p.add_argument("--model", choices=["mlp", "tcn", "gcn"], required=True)
    p.add_argument("--model-name", default=None)
    p.add_argument("--output-dir", default="data/models/pretrain/mpose2021")
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--tcn-filters", type=int, default=96)
    p.add_argument("--tcn-blocks", type=int, default=4)
    p.add_argument("--tcn-kernel", type=int, default=3)
    p.add_argument("--gcn-filters", type=int, default=64)
    p.add_argument("--gcn-blocks", type=int, default=3)
    p.add_argument("--gcn-kernel", type=int, default=9)
    p.add_argument("--gcn-partition", choices=["distance", "uniform"], default="distance")
    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--min-learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--runtime-jitter", type=float, default=0.005,
                   help="Gaussian σ added to each training sample (train only). "
                        "Matches the downstream contrastive pipeline.")
    p.add_argument("--augment-rot-deg", type=float, default=10.0,
                   help="Max ±deg random rotation per training sample (0=off).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args(argv)


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
