"""Train a scratch MLP / TCN / GCN skeleton encoder with contrastive loss.

Improvements over v1:
- Train/val split along time axis (val_fraction of each dance's tail is held-out)
- InfoNCE wrapper with in-batch false-negative masking (same dance + |Δt|<gap
  is excluded from softmax denominator)
- ModelCheckpoint callback (best val_loss saved mid-training as insurance)
- Per-architecture builders with joint-aware structure (see models_scratch.py)

TFLite export is UNAFFECTED: only the `encoder` module is exported via
@tf.function + TFLiteConverter. The contrastive wrapper (InfoNCE Model
subclass / Triplet functional wrapper) exists only for training.

Usage
-----
    python scripts/train_scratch_contrastive.py --model mlp  --loss infonce
    python scripts/train_scratch_contrastive.py --model tcn  --loss infonce
    python scripts/train_scratch_contrastive.py --model gcn  --loss triplet
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

from src.embedding.dataset_contrastive import (  # noqa: E402
    ContrastiveBatchGenerator, ContrastiveSequenceStore,
    DANCE_JOINTS, TARGET_DANCES,
)
from src.embedding.models_scratch import build_encoder  # noqa: E402


# ---------------- Contrastive wrappers ----------------

class InfoNCEWrapper(tf.keras.Model):
    """Symmetric InfoNCE with in-batch false-negative masking.

    Inputs per batch:
        [anchor (B,T,J,C), positive (B,T,J,C), dance_idx (B,), end_idx (B,)]

    The similarity matrix S[i,j] = (a_i · p_j) / τ. Before computing
    cross-entropy, we mask entries where:
        i ≠ j  AND  dance[i] == dance[j]  AND  |end[i] - end[j]| < negative_gap
    i.e. in-batch samples that are actually near-duplicates of the anchor's
    positive — they would otherwise be treated as negatives.
    """

    def __init__(self, encoder: tf.keras.Model, temperature: float = 0.1,
                 negative_gap: int = 45, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.temperature = float(temperature)
        self.negative_gap = int(negative_gap)
        self._loss_tracker = tf.keras.metrics.Mean(name="loss")
        self._acc_tracker = tf.keras.metrics.Mean(name="retrieval_acc")

    @property
    def metrics(self):
        return [self._loss_tracker, self._acc_tracker]

    def _compute(self, inputs, training: bool):
        anchor, positive, dance_idx, end_idx = inputs
        a = self.encoder(anchor, training=training)
        p = self.encoder(positive, training=training)

        sim = tf.matmul(a, p, transpose_b=True) / self.temperature  # (B, B)
        B = tf.shape(sim)[0]
        d = tf.cast(dance_idx, tf.int32)
        e = tf.cast(end_idx, tf.int32)

        same_dance = tf.equal(d[:, None], d[None, :])                     # (B, B)
        close_t = tf.less(tf.abs(e[:, None] - e[None, :]), self.negative_gap)
        false_neg = tf.logical_and(same_dance, close_t)
        not_diag = tf.logical_not(tf.eye(B, dtype=tf.bool))
        mask = tf.logical_and(false_neg, not_diag)                        # keep diagonal
        sim = tf.where(mask, tf.fill(tf.shape(sim), -1e9), sim)

        labels = tf.range(B)
        loss_ap = tf.keras.losses.sparse_categorical_crossentropy(
            labels, sim, from_logits=True)
        loss_pa = tf.keras.losses.sparse_categorical_crossentropy(
            labels, tf.transpose(sim), from_logits=True)
        loss = tf.reduce_mean((loss_ap + loss_pa) * 0.5)

        preds = tf.argmax(sim, axis=-1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
        return loss, acc

    def call(self, inputs, training=False):
        # Provided for Model interface; returns raw similarity matrix (unmasked).
        anchor, positive = inputs[0], inputs[1]
        a = self.encoder(anchor, training=training)
        p = self.encoder(positive, training=training)
        return tf.matmul(a, p, transpose_b=True) / self.temperature

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


def build_triplet_model(encoder: tf.keras.Model, margin: float,
                        learning_rate: float) -> tf.keras.Model:
    shape = encoder.input_shape[1:]
    a_in = tf.keras.layers.Input(shape=shape, name="anchor")
    p_in = tf.keras.layers.Input(shape=shape, name="positive")
    n_in = tf.keras.layers.Input(shape=shape, name="negative")
    a = encoder(a_in)
    p = encoder(p_in)
    n = encoder(n_in)

    def sims(tensors):
        ta, tp, tn = tensors
        sim_ap = tf.reduce_sum(ta * tp, axis=-1, keepdims=True)
        sim_an = tf.reduce_sum(ta * tn, axis=-1, keepdims=True)
        return tf.concat([sim_ap, sim_an], axis=-1)

    out = tf.keras.layers.Lambda(sims, name="similarities")([a, p, n])
    model = tf.keras.Model([a_in, p_in, n_in], out, name="triplet_wrapper")

    margin = float(margin)

    def triplet_loss(y_true, y_pred):
        del y_true
        sim_ap = y_pred[:, 0]
        sim_an = y_pred[:, 1]
        return tf.reduce_mean(tf.maximum(sim_an - sim_ap + margin, 0.0))

    def pos_margin(y_true, y_pred):
        del y_true
        return tf.reduce_mean(y_pred[:, 0] - y_pred[:, 1])

    def violation_rate(y_true, y_pred):
        del y_true
        return tf.reduce_mean(tf.cast(y_pred[:, 1] + margin > y_pred[:, 0], tf.float32))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate, clipnorm=1.0),
        loss=triplet_loss,
        metrics=[pos_margin, violation_rate],
    )
    return model


# ---------------- LR schedule ----------------

def make_warmup_cosine_schedule(base_lr: float, warmup_epochs: int,
                                total_epochs: int, min_lr: float = 1e-5):
    warmup_epochs = max(0, int(warmup_epochs))
    total_epochs = max(1, int(total_epochs))

    def schedule(epoch: int, lr: float) -> float:
        del lr
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return float(min_lr + (base_lr - min_lr) * 0.5 * (1.0 + np.cos(np.pi * progress)))

    return schedule


# ---------------- Export ----------------

def export_tflite(encoder: tf.keras.Model, output_path: Path, quantize: bool = True) -> int:
    input_shape = [1, *encoder.input_shape[1:]]

    @tf.function(input_signature=[tf.TensorSpec(input_shape, tf.float32, name="pose_window")])
    def encode(pose_window):
        return encoder(pose_window, training=False)

    # NOTE: passing trackable_obj=encoder causes TF 2.21 converter to silently
    # drop variable values → TFLite output all zeros / NaN. Omit it.
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [encode.get_concrete_function()])
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_bytes)
    return len(tflite_bytes)


def verify_tflite(tflite_path: Path, sample_window: np.ndarray):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    x = sample_window[None, ...].astype(in_det["dtype"])
    interpreter.set_tensor(in_det["index"], x)
    interpreter.invoke()
    emb = interpreter.get_tensor(out_det["index"])
    return in_det["shape"].tolist(), out_det["shape"].tolist(), emb


# ---------------- Train ----------------

def train(args: argparse.Namespace) -> Dict[str, Any]:
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    store = ContrastiveSequenceStore(
        data_dir, dances=tuple(args.dances), feature_dims=args.feature_dims,
        verbose=True)
    print(f"[DATA] {len(store)} dances, feature_dims={args.feature_dims}, "
          f"val_fraction={args.val_fraction}")

    train_gen = ContrastiveBatchGenerator(
        store, sequence_length=args.sequence_length, batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch, mode=args.loss,
        positive_jitter=args.positive_jitter, negative_gap=args.negative_gap,
        split="train", val_fraction=args.val_fraction,
        runtime_jitter=args.runtime_jitter, seed=args.seed)
    val_gen = ContrastiveBatchGenerator(
        store, sequence_length=args.sequence_length, batch_size=args.batch_size,
        steps_per_epoch=args.validation_steps, mode=args.loss,
        positive_jitter=0,  # no temporal jitter on val
        negative_gap=args.negative_gap,
        split="val", val_fraction=args.val_fraction,
        runtime_jitter=0.0,  # clean evaluation
        seed=args.seed + 1000)

    model_kwargs: Dict[str, Any] = {"dropout": args.dropout}
    if args.model == "mlp":
        model_kwargs["hidden"] = args.mlp_hidden
    elif args.model == "tcn":
        model_kwargs.update(filters=args.tcn_filters, blocks=args.tcn_blocks,
                             kernel_size=args.tcn_kernel)
    else:  # gcn
        model_kwargs.update(filters=args.gcn_filters, blocks=args.gcn_blocks,
                             kernel_size=args.gcn_kernel,
                             partition_strategy=args.gcn_partition)

    encoder = build_encoder(
        args.model, sequence_length=args.sequence_length, num_joints=12,
        feature_dims=args.feature_dims, embedding_dim=args.embedding_dim,
        **model_kwargs)

    if args.pretrained_weights:
        pw = Path(args.pretrained_weights)
        if not pw.is_absolute():
            pw = PROJECT_ROOT / pw
        if not pw.exists():
            raise SystemExit(f"--pretrained-weights not found: {pw}")
        # Build model by calling once so weights can load
        _ = encoder(tf.zeros(
            [1, args.sequence_length, 12, args.feature_dims], dtype=tf.float32))
        encoder.load_weights(str(pw))
        print(f"[FINETUNE] loaded pretrained weights from {pw}")

    encoder.summary()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name or (
        f"scratch_{args.model}_{args.loss}_e{args.embedding_dim}")
    keras_path = output_dir / f"{model_name}_encoder.keras"
    tflite_path = output_dir / f"{model_name}.tflite"
    meta_path = output_dir / f"{model_name}_meta.json"
    ckpt_path = output_dir / f"{model_name}_checkpoint.weights.h5"

    if args.loss == "infonce":
        wrapper: tf.keras.Model = InfoNCEWrapper(
            encoder, temperature=args.temperature,
            negative_gap=args.negative_gap)
        wrapper.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate, clipnorm=1.0))
    else:
        wrapper = build_triplet_model(encoder, args.triplet_margin,
                                       args.learning_rate)

    lr_schedule = make_warmup_cosine_schedule(
        args.learning_rate, args.warmup_epochs, args.epochs,
        args.min_learning_rate)

    class EncoderCheckpoint(tf.keras.callbacks.Callback):
        """Save encoder weights only on val_loss improvement (subclass-safe)."""
        def __init__(self, encoder_model, filepath, monitor="val_loss"):
            super().__init__()
            self.encoder_model = encoder_model
            self.filepath = str(filepath)
            self.monitor = monitor
            self.best = float("inf")

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            current = logs.get(self.monitor)
            if current is not None and current < self.best:
                self.best = float(current)
                self.encoder_model.save_weights(self.filepath)

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience,
            restore_best_weights=True, verbose=1),
        EncoderCheckpoint(encoder, ckpt_path, monitor="val_loss"),
    ]

    history = wrapper.fit(
        train_gen.batches(),
        validation_data=val_gen.batches(),
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=args.verbose,
    )

    # Save encoder (best weights were restored by EarlyStopping)
    encoder.save(keras_path)

    # Sample window for TFLite verification
    (x_tuple, _) = train_gen._generate_batch()
    sample = x_tuple[0][0]  # first anchor window

    size_bytes = export_tflite(encoder, tflite_path, quantize=not args.no_quantize)
    in_shape, out_shape, sample_emb = verify_tflite(tflite_path, sample)

    meta = {
        "model_name": model_name,
        "model_type": args.model,
        "loss": args.loss,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_shape": in_shape,
        "output_shape": out_shape,
        "target_joints": list(DANCE_JOINTS),
        "dances": list(store.names),
        "config": {k: (v if not isinstance(v, Path) else str(v))
                   for k, v in vars(args).items()},
        "tflite_size_kib": size_bytes / 1024.0,
        "sample_embedding_norm": float(np.linalg.norm(sample_emb)),
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    # Clean up mid-training checkpoint (we saved final encoder already)
    if ckpt_path.exists() and not args.keep_checkpoint:
        ckpt_path.unlink()

    print(f"[SAVE] encoder  : {keras_path}")
    print(f"[SAVE] tflite   : {tflite_path} ({size_bytes/1024:.1f} KiB)")
    print(f"[SAVE] metadata : {meta_path}")
    return {
        "encoder": encoder, "wrapper": wrapper, "history": history.history,
        "paths": {"keras": str(keras_path), "tflite": str(tflite_path),
                   "meta": str(meta_path)},
        "tflite_size_kib": size_bytes / 1024.0,
        "sample_embedding_norm": float(np.linalg.norm(sample_emb)),
    }


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Data
    p.add_argument("--data-dir", default="data/reference_dances")
    p.add_argument("--dances", nargs="+", default=list(TARGET_DANCES))
    p.add_argument("--feature-dims", type=int, choices=[2, 3], default=2)
    p.add_argument("--val-fraction", type=float, default=0.15,
                   help="fraction of each dance's tail reserved for validation")
    # Model
    p.add_argument("--model", choices=["mlp", "tcn", "gcn"], required=True)
    p.add_argument("--model-name", default=None)
    p.add_argument("--output-dir", default="data/models/scratch")
    p.add_argument("--sequence-length", type=int, default=30)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--tcn-filters", type=int, default=96)
    p.add_argument("--tcn-blocks", type=int, default=4)
    p.add_argument("--tcn-kernel", type=int, default=3)
    p.add_argument("--gcn-filters", type=int, default=64)
    p.add_argument("--gcn-blocks", type=int, default=3)
    p.add_argument("--gcn-kernel", type=int, default=9)
    p.add_argument("--gcn-partition", choices=["distance", "uniform"],
                   default="distance")
    # Training
    p.add_argument("--loss", choices=["infonce", "triplet"], default="infonce")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--validation-steps", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--min-learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--triplet-margin", type=float, default=0.2)
    p.add_argument("--positive-jitter", type=int, default=2)
    p.add_argument("--negative-gap", type=int, default=45)
    p.add_argument("--runtime-jitter", type=float, default=0.005,
                   help="on-the-fly Gaussian σ added to A/P/N at every training step "
                        "(0 = disabled). Extra diversity beyond precomputed augments. "
                        "Reasonable range: 0.003~0.01 in hip/torso-normalized coords.")
    # Fine-tuning
    p.add_argument("--pretrained-weights", default=None,
                   help="path to encoder .weights.h5 from pretraining. "
                        "Recommended: --learning-rate 1e-4 --epochs 20 "
                        "--warmup-epochs 1 when using this.")
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-quantize", action="store_true")
    p.add_argument("--keep-checkpoint", action="store_true",
                   help="keep the mid-training best checkpoint file")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args(argv)


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
