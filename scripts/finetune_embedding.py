#!/usr/bin/env python3
"""Fine-tune a pretrained skeleton encoder on dance reference data.

This script continues metric learning from a pretrained ``.keras`` encoder
created by ``pretrain_mpose2021.py`` and exports the fine-tuned encoder as
TFLite for the runtime ``embedding`` score method.
"""

import argparse
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
from scoring.scratch_features import build_pose_window
from train_scratch_similarity import (
    PairBatchGenerator,
    TripletBatchGenerator,
    build_encoder,
    build_siamese_model,
    build_triplet_model,
    export_tflite,
    import_tensorflow,
    load_sequences,
    verify_tflite,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained skeleton embedding encoder on dance data.")
    parser.add_argument("--data-dir", default="data/reference_dances")
    parser.add_argument("--pretrained-encoder", required=True,
                        help="Path to a pretrained .keras encoder.")
    parser.add_argument("--model-name", default="dance_embedding_gcn_e64_triplet")
    parser.add_argument("--model-type", choices=["tcn", "gcn"], default="gcn",
                        help="Used only when --rebuild-from-scratch is set.")
    parser.add_argument("--output-dir", default="data/models/embedding")
    parser.add_argument("--output", default=None)
    parser.add_argument("--keras-output", default=None)
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--feature-dims", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument("--embedding-dim", type=int, default=None,
                        help="Defaults to the pretrained encoder output dimension.")
    parser.add_argument("--filters", type=int, default=64,
                        help="Used only when --rebuild-from-scratch is set.")
    parser.add_argument("--blocks", type=int, default=3,
                        help="Used only when --rebuild-from-scratch is set.")
    parser.add_argument("--kernel-size", type=int, default=3,
                        help="Used only when --rebuild-from-scratch is set.")
    parser.add_argument("--dropout", type=float, default=0.15,
                        help="Used only when --rebuild-from-scratch is set.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps-per-epoch", type=int, default=120)
    parser.add_argument("--validation-steps", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--loss-type", choices=["bce", "triplet"], default="triplet")
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--positive-jitter", type=int, default=6)
    parser.add_argument("--negative-gap", type=int, default=45)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-from-scratch", action="store_true",
                        help="Ignore --pretrained-encoder and build a fresh encoder.")
    return parser.parse_args()


def project_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def output_paths(args):
    output_dir = project_path(args.output_dir)
    tflite = args.output or os.path.join(output_dir, f"{args.model_name}.tflite")
    keras = args.keras_output or os.path.join(output_dir, f"{args.model_name}_encoder.keras")
    meta = args.metadata_output or os.path.join(output_dir, f"{args.model_name}_meta.json")
    return project_path(tflite), project_path(keras), project_path(meta)


def dance_adjacency(num_joints):
    adjacency = np.eye(num_joints, dtype=np.float32)
    edges = [
        (0, 1), (0, 2), (2, 4), (1, 3), (3, 5),
        (0, 6), (1, 7), (6, 7), (6, 8), (8, 10),
        (7, 9), (9, 11),
    ]
    for a, b in edges:
        if a < num_joints and b < num_joints:
            adjacency[a, b] = 1.0
            adjacency[b, a] = 1.0
    return adjacency / np.maximum(adjacency.sum(axis=1, keepdims=True), 1.0)


def load_pretrained_encoder(tf, path):
    path = project_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pretrained encoder not found: {path}")

    adjacency = tf.constant(dance_adjacency(len(DANCE_JOINTS)), dtype=tf.float32)

    def aggregate(x):
        return tf.einsum("ij,btjc->btic", adjacency, x)

    custom_objects = {"aggregate": aggregate}
    try:
        return tf.keras.models.load_model(
            path, safe_mode=False, custom_objects=custom_objects)
    except TypeError:
        return tf.keras.models.load_model(path, custom_objects=custom_objects)


def infer_encoder_config(encoder):
    input_shape = tuple(int(x) for x in encoder.input_shape[1:])
    if len(input_shape) != 3:
        raise ValueError(f"Expected encoder input shape (T, J, C), got {encoder.input_shape}")
    output_shape = encoder.output_shape
    embedding_dim = int(output_shape[-1])
    return {
        "sequence_length": input_shape[0],
        "num_joints": input_shape[1],
        "feature_dims": input_shape[2],
        "embedding_dim": embedding_dim,
    }


def validate_encoder_config(args, encoder_config):
    expected_joints = len(DANCE_JOINTS)
    if encoder_config["num_joints"] != expected_joints:
        raise ValueError(
            f"Pretrained encoder uses {encoder_config['num_joints']} joints, "
            f"but runtime expects {expected_joints} DANCE_JOINTS."
        )
    if args.sequence_length != encoder_config["sequence_length"]:
        raise ValueError(
            f"--sequence-length={args.sequence_length} does not match pretrained encoder "
            f"input length {encoder_config['sequence_length']}."
        )
    if args.feature_dims != encoder_config["feature_dims"]:
        raise ValueError(
            f"--feature-dims={args.feature_dims} does not match pretrained encoder "
            f"feature dims {encoder_config['feature_dims']}."
        )


def build_or_load_encoder(tf, args):
    if not args.rebuild_from_scratch:
        encoder = load_pretrained_encoder(tf, args.pretrained_encoder)
        config = infer_encoder_config(encoder)
        validate_encoder_config(args, config)
        return encoder, config

    embedding_dim = args.embedding_dim or 64
    encoder = build_encoder(
        tf,
        sequence_length=args.sequence_length,
        num_joints=len(DANCE_JOINTS),
        feature_dims=args.feature_dims,
        embedding_dim=embedding_dim,
        filters=args.filters,
        blocks=args.blocks,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        model_type=args.model_type,
    )
    return encoder, {
        "sequence_length": args.sequence_length,
        "num_joints": len(DANCE_JOINTS),
        "feature_dims": args.feature_dims,
        "embedding_dim": embedding_dim,
    }


def summarize_training_history(history):
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


def smoke_embedding_metrics(encoder, sequences, args, max_pairs=128):
    if len(sequences) < 2:
        return {}
    rng = random.Random(args.seed)
    same, diff = [], []
    for _ in range(max_pairs):
        seq = rng.choice(sequences)
        end_a = rng.randint(args.sequence_length - 1, seq.num_frames - 1)
        end_b = min(
            max(end_a + rng.randint(-args.positive_jitter, args.positive_jitter),
                args.sequence_length - 1),
            seq.num_frames - 1,
        )
        other = rng.choice([item for item in sequences if item is not seq])
        end_n = rng.randint(args.sequence_length - 1, other.num_frames - 1)
        windows = np.asarray([
            build_pose_window(seq.data, end_a, args.sequence_length,
                              target_joints=DANCE_JOINTS, feature_dims=args.feature_dims),
            build_pose_window(seq.data, end_b, args.sequence_length,
                              target_joints=DANCE_JOINTS, feature_dims=args.feature_dims),
            build_pose_window(other.data, end_n, args.sequence_length,
                              target_joints=DANCE_JOINTS, feature_dims=args.feature_dims),
        ], dtype=np.float32)
        emb = encoder.predict(windows, verbose=0)
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
        same.append(float(np.sum(emb[0] * emb[1])))
        diff.append(float(np.sum(emb[0] * emb[2])))
    return {
        "same_cosine_mean": float(np.mean(same)),
        "different_cosine_mean": float(np.mean(diff)),
        "margin_mean": float(np.mean(np.asarray(same) - np.asarray(diff))),
    }


def write_metadata(path, args, encoder_config, sequences, input_shape, output_shape,
                   history=None, training_summary=None, smoke_metrics=None):
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": args.model_name,
        "runtime_score_method": "embedding",
        "runtime_output_mode": "embedding",
        "input_layout": "BTJC",
        "input_shape": input_shape,
        "output_shape": output_shape,
        "target_joints": DANCE_JOINTS,
        "pretrained_encoder": (
            os.path.relpath(project_path(args.pretrained_encoder), PROJECT_ROOT)
            if not args.rebuild_from_scratch else None
        ),
        "encoder_config": encoder_config,
        "config": vars(args),
        "references": [
            {
                "name": seq.name,
                "path": os.path.relpath(seq.path, PROJECT_ROOT),
                "frames": seq.num_frames,
            }
            for seq in sequences
        ],
        "history": history or {},
        "training_summary": training_summary or {},
        "smoke_metrics": smoke_metrics or {},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    sequences = load_sequences(args.data_dir, args.sequence_length)
    print(f"[DATA] loaded {len(sequences)} dance reference sequences")
    for seq in sequences:
        print(f"  - {seq.name}: {seq.num_frames} frames {seq.data.shape[1:]}")

    output_path, keras_output, metadata_output = output_paths(args)
    if args.dry_run:
        write_metadata(
            metadata_output,
            args,
            {
                "sequence_length": args.sequence_length,
                "num_joints": len(DANCE_JOINTS),
                "feature_dims": args.feature_dims,
                "embedding_dim": args.embedding_dim,
            },
            sequences,
            [1, args.sequence_length, len(DANCE_JOINTS), args.feature_dims],
            [1, args.embedding_dim],
        )
        print(f"[DRY-RUN] metadata written: {metadata_output}")
        return

    tf = import_tensorflow()
    tf.keras.utils.set_random_seed(args.seed)
    encoder, encoder_config = build_or_load_encoder(tf, args)
    print(f"[MODEL] encoder config: {encoder_config}")

    gen_cls = TripletBatchGenerator if args.loss_type == "triplet" else PairBatchGenerator
    train_gen = gen_cls(
        sequences, args.sequence_length, args.feature_dims, args.batch_size,
        args.steps_per_epoch, args.positive_jitter, args.negative_gap,
        args.noise_std, args.seed)
    val_gen = gen_cls(
        sequences, args.sequence_length, args.feature_dims, args.batch_size,
        args.validation_steps, args.positive_jitter, args.negative_gap,
        0.0, args.seed + 1000)

    if args.loss_type == "triplet":
        model = build_triplet_model(tf, encoder, args.learning_rate, args.triplet_margin)
    else:
        model = build_siamese_model(tf, encoder, args.learning_rate)
    model.summary()

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
    print(f"[SAVE] fine-tuned Keras encoder: {keras_output}")

    sample = build_pose_window(
        sequences[0].data,
        args.sequence_length - 1,
        args.sequence_length,
        target_joints=DANCE_JOINTS,
        feature_dims=args.feature_dims,
    )
    size = export_tflite(tf, encoder, output_path, quantize=not args.no_quantize)
    input_shape, output_shape, embedding = verify_tflite(tf, output_path, sample)
    history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    training_summary = summarize_training_history(history)
    smoke_metrics = smoke_embedding_metrics(encoder, sequences, args)
    write_metadata(
        metadata_output,
        args,
        encoder_config,
        sequences,
        input_shape,
        output_shape,
        history=history_dict,
        training_summary=training_summary,
        smoke_metrics=smoke_metrics,
    )
    print(f"[SAVE] fine-tuned TFLite encoder: {output_path} ({size / 1024:.1f} KiB)")
    print(f"[VERIFY] input={input_shape} output={output_shape} "
          f"embedding_norm={float(np.linalg.norm(embedding)):.4f}")
    print(f"[BEST] {training_summary}")
    print(f"[METRIC] {smoke_metrics}")
    print(f"[SAVE] metadata: {metadata_output}")
    print("\nRun with:")
    print(f"  python src/main.py -s embedding --embedding-model-name {args.model_name}")


if __name__ == "__main__":
    main()
