#!/usr/bin/env python3
"""Train a scratch TCN/GCN skeleton encoder and export it to TFLite."""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pose.landmark_utils import DANCE_JOINTS
from scoring.scratch_features import build_pose_window


@dataclass
class PoseSequence:
    name: str
    path: str
    data: np.ndarray

    @property
    def num_frames(self):
        return int(self.data.shape[0])


class PairBatchGenerator:
    """Sample positive/negative skeleton-window pairs from reference dances."""

    def __init__(self, sequences, sequence_length, feature_dims, batch_size,
                 steps_per_epoch, positive_jitter=6, negative_gap=45,
                 noise_std=0.015, seed=42):
        self.sequences = list(sequences)
        self.sequence_length = int(sequence_length)
        self.feature_dims = int(feature_dims)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.positive_jitter = int(positive_jitter)
        self.negative_gap = int(negative_gap)
        self.noise_std = float(noise_std)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, _idx):
        users, refs, labels = [], [], []
        half = self.batch_size // 2
        for _ in range(half):
            user, ref = self._sample_positive_pair()
            users.append(user)
            refs.append(ref)
            labels.append(1.0)
        while len(labels) < self.batch_size:
            user, ref = self._sample_negative_pair()
            users.append(user)
            refs.append(ref)
            labels.append(0.0)

        order = np.arange(self.batch_size)
        self.np_rng.shuffle(order)
        return (
            np.asarray(users, dtype=np.float32)[order],
            np.asarray(refs, dtype=np.float32)[order],
        ), np.asarray(labels, dtype=np.float32)[order]

    def batches(self):
        while True:
            for idx in range(len(self)):
                yield self[idx]

    def _sample_positive_pair(self):
        seq = self.rng.choice(self.sequences)
        end_a = self._random_end(seq)
        delta = self.rng.randint(-self.positive_jitter, self.positive_jitter)
        end_b = min(max(end_a + delta, self.sequence_length - 1), seq.num_frames - 1)
        return self._window(seq, end_a, augment=True), self._window(seq, end_b, augment=True)

    def _sample_negative_pair(self):
        seq_a = self.rng.choice(self.sequences)
        end_a = self._random_end(seq_a)
        if len(self.sequences) >= 2 and self.rng.random() < 0.75:
            seq_b = self.rng.choice([seq for seq in self.sequences if seq is not seq_a])
            end_b = self._random_end(seq_b)
        else:
            seq_b = seq_a
            end_b = self._random_far_end(seq_b, end_a)
        return self._window(seq_a, end_a, augment=True), self._window(seq_b, end_b, augment=True)

    def _random_end(self, seq):
        return self.rng.randint(self.sequence_length - 1, seq.num_frames - 1)

    def _random_far_end(self, seq, anchor_end):
        valid_min = self.sequence_length - 1
        valid_max = seq.num_frames - 1
        for _ in range(20):
            end = self.rng.randint(valid_min, valid_max)
            if abs(end - anchor_end) >= self.negative_gap:
                return end
        return valid_max if anchor_end < (valid_min + valid_max) // 2 else valid_min

    def _window(self, seq, end_idx, augment=False):
        window = build_pose_window(
            seq.data,
            end_idx,
            self.sequence_length,
            target_joints=DANCE_JOINTS,
            feature_dims=self.feature_dims,
        )
        if augment and self.noise_std > 0:
            window = window + self.np_rng.normal(
                0.0, self.noise_std, size=window.shape).astype(np.float32)
        return np.nan_to_num(window).astype(np.float32)


class TripletBatchGenerator(PairBatchGenerator):
    """Sample (anchor, positive, negative) triplets for triplet-loss training.

    Reuses PairBatchGenerator's sequence pool and sampling helpers. Every
    triplet is drawn such that:
      - anchor: random window from a randomly-chosen dance
      - positive: same dance as anchor, within ±positive_jitter frames
      - negative: 75% of the time a different dance, otherwise same dance but
        at least negative_gap frames away (reusing PairBatchGenerator logic)
    """

    def __getitem__(self, _idx):
        anchors, positives, negatives = [], [], []
        for _ in range(self.batch_size):
            seq_a = self.rng.choice(self.sequences)
            end_a = self._random_end(seq_a)
            delta = self.rng.randint(-self.positive_jitter, self.positive_jitter)
            end_p = min(max(end_a + delta, self.sequence_length - 1), seq_a.num_frames - 1)
            if len(self.sequences) >= 2 and self.rng.random() < 0.75:
                seq_n = self.rng.choice([s for s in self.sequences if s is not seq_a])
                end_n = self._random_end(seq_n)
            else:
                seq_n = seq_a
                end_n = self._random_far_end(seq_n, end_a)
            anchors.append(self._window(seq_a, end_a, augment=True))
            positives.append(self._window(seq_a, end_p, augment=True))
            negatives.append(self._window(seq_n, end_n, augment=True))
        return (
            (
                np.asarray(anchors, dtype=np.float32),
                np.asarray(positives, dtype=np.float32),
                np.asarray(negatives, dtype=np.float32),
            ),
            np.zeros(self.batch_size, dtype=np.float32),  # dummy y (loss는 y_pred만 사용)
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and export a scratch TFLite skeleton similarity encoder.")
    parser.add_argument("--data-dir", default="data/reference_dances")
    parser.add_argument("--model-name", default="scratch_gcn")
    parser.add_argument("--model-type", choices=["tcn", "gcn"], default="gcn")
    parser.add_argument("--output-dir", default="data/models/scratch")
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps-per-epoch", type=int, default=120)
    parser.add_argument("--validation-steps", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5,
                        help="EarlyStopping patience on val_loss. "
                             "best weights는 항상 자동 복원되어 저장된다.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--loss-type", choices=["bce", "triplet"], default="bce",
                        help="bce: Siamese + cosine에 대한 binary cross entropy. "
                             "triplet: (anchor, positive, negative) + margin triplet loss.")
    parser.add_argument("--triplet-margin", type=float, default=0.2,
                        help="Triplet loss의 margin. L2 정규화된 cosine 기반이라 "
                             "일반적으로 0.1~0.5 범위를 사용. "
                             "--loss-type=triplet일 때만 사용된다.")
    parser.add_argument("--positive-jitter", type=int, default=6)
    parser.add_argument("--negative-gap", type=int, default=45)
    parser.add_argument("--noise-std", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--convert-only", action="store_true")
    return parser.parse_args()


def project_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def output_paths(args):
    output_dir = project_path(args.output_dir)
    tflite = args.output or os.path.join(output_dir, f"{args.model_name}.tflite")
    keras = args.keras_output or os.path.join(output_dir, f"{args.model_name}_encoder.keras")
    meta = args.metadata_output or os.path.join(output_dir, f"{args.model_name}_meta.json")
    return project_path(tflite), project_path(keras), project_path(meta)


def load_sequences(data_dir, sequence_length):
    data_dir = project_path(data_dir)
    sequences = []
    for root, _dirs, files in os.walk(data_dir):
        if "reference.npy" not in files:
            continue
        path = os.path.join(root, "reference.npy")
        data = np.load(path, allow_pickle=True)
        if not isinstance(data, np.ndarray) or data.ndim != 3 or data.shape[1] < 29:
            print(f"[WARN] skip invalid reference: {path} shape={getattr(data, 'shape', None)}")
            continue
        if data.shape[0] < sequence_length:
            print(f"[WARN] skip short reference: {path} frames={data.shape[0]}")
            continue
        if data.shape[2] == 3:
            vis = np.ones((*data.shape[:2], 1), dtype=np.float32)
            data = np.concatenate([data, vis], axis=2)
        sequences.append(PoseSequence(
            name=os.path.basename(root),
            path=path,
            data=np.nan_to_num(data.astype(np.float32)),
        ))
    if not sequences:
        raise RuntimeError(f"No usable reference.npy files found under {data_dir}")
    return sequences


def import_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow is required to train/export scratch models.\n"
            "Install training dependencies, for example:\n"
            "  pip install tensorflow==2.16.2"
        ) from exc
    return tf


def tcn_block(tf, x, filters, kernel_size, dilation_rate, dropout, name):
    layers = tf.keras.layers
    residual = x
    x = layers.Conv1D(filters, kernel_size, padding="causal",
                      dilation_rate=dilation_rate, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.SpatialDropout1D(dropout, name=f"{name}_drop1")(x)
    x = layers.Conv1D(filters, kernel_size, padding="causal",
                      dilation_rate=dilation_rate, name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    if residual.shape[-1] != filters:
        residual = layers.Conv1D(filters, 1, padding="same", name=f"{name}_resample")(residual)
    x = layers.Add(name=f"{name}_add")([x, residual])
    return layers.Activation("relu", name=f"{name}_relu2")(x)


def build_tcn_encoder(tf, sequence_length, num_joints, feature_dims, embedding_dim,
                      filters, blocks, kernel_size, dropout):
    layers = tf.keras.layers
    inputs = layers.Input(shape=(sequence_length, num_joints, feature_dims), name="pose_window")
    x = layers.Reshape((sequence_length, num_joints * feature_dims), name="flatten_joints")(inputs)
    x = layers.Dense(filters, activation="relu", name="input_projection")(x)
    for i in range(blocks):
        x = tcn_block(tf, x, filters, kernel_size, 2 ** i, dropout, name=f"tcn_{i + 1}")
    x = layers.GlobalAveragePooling1D(name="temporal_pool")(x)
    x = layers.Dense(filters, activation="relu", name="embedding_hidden")(x)
    outputs = layers.Dense(embedding_dim, name="embedding")(x)
    return tf.keras.Model(inputs, outputs, name="scratch_tcn_encoder")


def graph_aggregate_layer(tf, adjacency, name):
    adjacency = tf.constant(adjacency, dtype=tf.float32)

    def aggregate(x):
        return tf.einsum("ij,btjc->btic", adjacency, x)

    return tf.keras.layers.Lambda(aggregate, name=name)


def build_gcn_encoder(tf, sequence_length, num_joints, feature_dims, embedding_dim,
                      filters, blocks, kernel_size, dropout):
    layers = tf.keras.layers
    inputs = layers.Input(shape=(sequence_length, num_joints, feature_dims), name="pose_window")
    adjacency = np.eye(num_joints, dtype=np.float32)
    edges = [
        (0, 1), (0, 2), (2, 4), (1, 3), (3, 5),
        (0, 6), (1, 7), (6, 7), (6, 8), (8, 10),
        (7, 9), (9, 11),
    ]
    for a, b in edges:
        adjacency[a, b] = 1.0
        adjacency[b, a] = 1.0
    adjacency = adjacency / np.maximum(adjacency.sum(axis=1, keepdims=True), 1.0)

    x = layers.Dense(filters, activation="relu", name="input_projection")(inputs)
    for i in range(blocks):
        residual = x
        x = graph_aggregate_layer(tf, adjacency, name=f"gcn_{i + 1}_aggregate")(x)
        x = layers.Dense(filters, activation="relu", name=f"gcn_{i + 1}_spatial")(x)
        x = layers.Reshape((sequence_length, num_joints * filters),
                           name=f"gcn_{i + 1}_flatten_joints")(x)
        x = layers.Conv1D(filters, kernel_size, padding="same",
                          activation="relu", name=f"gcn_{i + 1}_temporal")(x)
        x = layers.Reshape((sequence_length, 1, filters),
                           name=f"gcn_{i + 1}_restore_time")(x)
        x = layers.UpSampling2D(size=(1, num_joints),
                                name=f"gcn_{i + 1}_restore_joints")(x)
        if residual.shape[-1] != filters:
            residual = layers.Dense(filters, name=f"gcn_{i + 1}_resample")(residual)
        x = layers.Add(name=f"gcn_{i + 1}_add")([x, residual])
        x = layers.Activation("relu", name=f"gcn_{i + 1}_relu")(x)
        x = layers.Dropout(dropout, name=f"gcn_{i + 1}_drop")(x)

    x = layers.GlobalAveragePooling2D(name="spacetime_pool")(x)
    x = layers.Dense(filters, activation="relu", name="embedding_hidden")(x)
    outputs = layers.Dense(embedding_dim, name="embedding")(x)
    return tf.keras.Model(inputs, outputs, name="scratch_gcn_encoder")


def build_encoder(tf, sequence_length, num_joints, feature_dims, embedding_dim,
                  filters, blocks, kernel_size, dropout, model_type):
    if model_type == "tcn":
        return build_tcn_encoder(
            tf, sequence_length, num_joints, feature_dims, embedding_dim,
            filters, blocks, kernel_size, dropout)
    if model_type == "gcn":
        return build_gcn_encoder(
            tf, sequence_length, num_joints, feature_dims, embedding_dim,
            filters, blocks, kernel_size, dropout)
    raise ValueError(f"Unknown model_type: {model_type}")


def build_siamese_model(tf, encoder, learning_rate):
    layers = tf.keras.layers
    user_input = layers.Input(shape=encoder.input_shape[1:], name="user_window")
    ref_input = layers.Input(shape=encoder.input_shape[1:], name="reference_window")
    user_emb = encoder(user_input)
    ref_emb = encoder(ref_input)

    def cosine_to_probability(tensors):
        user, ref = tensors
        user = tf.math.l2_normalize(user, axis=-1)
        ref = tf.math.l2_normalize(ref, axis=-1)
        cosine = tf.reduce_sum(user * ref, axis=-1, keepdims=True)
        return (cosine + 1.0) * 0.5

    score = layers.Lambda(cosine_to_probability, name="cosine_similarity")([user_emb, ref_emb])
    model = tf.keras.Model([user_input, ref_input], score, name="scratch_siamese_similarity")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )
    return model


def build_triplet_model(tf, encoder, learning_rate, margin):
    """Three-branch Siamese with cosine-based triplet loss.

    encoder를 세 번 호출해 (anchor, positive, negative) 임베딩을 만들고,
    L2 정규화된 상태에서 cosine 유사도를 계산한다.
    출력은 (B, 2) 텐서: [sim(a,p), sim(a,n)]. loss는 max(0, sim_an - sim_ap + margin).
    encoder weight는 세 branch가 공유하므로 BCE 학습과 동일한 방식으로
    EarlyStopping(restore_best_weights=True)에 의해 best 상태가 encoder에 복원된다.
    """
    layers = tf.keras.layers
    shape = encoder.input_shape[1:]
    anchor_in = layers.Input(shape=shape, name="anchor_window")
    positive_in = layers.Input(shape=shape, name="positive_window")
    negative_in = layers.Input(shape=shape, name="negative_window")

    a_emb = encoder(anchor_in)
    p_emb = encoder(positive_in)
    n_emb = encoder(negative_in)

    def compute_similarities(tensors):
        a, p, n = tensors
        a = tf.math.l2_normalize(a, axis=-1)
        p = tf.math.l2_normalize(p, axis=-1)
        n = tf.math.l2_normalize(n, axis=-1)
        sim_ap = tf.reduce_sum(a * p, axis=-1, keepdims=True)
        sim_an = tf.reduce_sum(a * n, axis=-1, keepdims=True)
        return tf.concat([sim_ap, sim_an], axis=-1)

    sims = layers.Lambda(compute_similarities, name="triplet_similarities")(
        [a_emb, p_emb, n_emb])

    margin = float(margin)

    def triplet_loss(y_true, y_pred):
        del y_true
        sim_ap = y_pred[:, 0]
        sim_an = y_pred[:, 1]
        return tf.reduce_mean(tf.maximum(sim_an - sim_ap + margin, 0.0))

    def positive_margin(y_true, y_pred):
        """평균적으로 positive가 negative보다 얼마나 더 가까운가 (높을수록 좋음)."""
        del y_true
        return tf.reduce_mean(y_pred[:, 0] - y_pred[:, 1])

    def violation_rate(y_true, y_pred):
        """triplet 제약을 위반한 (loss > 0) 샘플 비율 (낮을수록 좋음)."""
        del y_true
        return tf.reduce_mean(tf.cast(
            y_pred[:, 1] + margin > y_pred[:, 0], tf.float32))

    model = tf.keras.Model(
        [anchor_in, positive_in, negative_in], sims,
        name="scratch_triplet_similarity")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=triplet_loss,
        metrics=[positive_margin, violation_rate],
    )
    return model


def export_tflite(tf, encoder, output_path, quantize=True):
    input_shape = [1, *encoder.input_shape[1:]]

    @tf.function(input_signature=[tf.TensorSpec(input_shape, tf.float32, name="pose_window")])
    def encode(pose_window):
        return encoder(pose_window, training=False)

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [encode.get_concrete_function()], trackable_obj=encoder)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    return len(tflite_model)


def verify_tflite(tf, output_path, sample_window):
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    sample = sample_window[None, ...].astype(input_detail["dtype"])
    interpreter.set_tensor(input_detail["index"], sample)
    interpreter.invoke()
    embedding = interpreter.get_tensor(output_detail["index"])
    return input_detail["shape"].tolist(), output_detail["shape"].tolist(), embedding


def write_metadata(path, args, sequences, input_shape, output_shape, history=None):
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": args.model_name,
        "model_type": args.model_type,
        "runtime_score_method": "scratch",
        "runtime_output_mode": "embedding",
        "input_layout": "BTJC",
        "input_shape": input_shape,
        "output_shape": output_shape,
        "target_joints": DANCE_JOINTS,
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
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    sequences = load_sequences(args.data_dir, args.sequence_length)
    print(f"[DATA] loaded {len(sequences)} reference sequences")
    for seq in sequences:
        print(f"  - {seq.name}: {seq.num_frames} frames {seq.data.shape[1:]}")

    sample = build_pose_window(
        sequences[0].data,
        args.sequence_length - 1,
        args.sequence_length,
        target_joints=DANCE_JOINTS,
        feature_dims=args.feature_dims,
    )
    print(f"[DATA] model input window shape: {sample.shape}")
    print(f"[MODEL] name={args.model_name} type={args.model_type}")

    output_path, keras_output, metadata_output = output_paths(args)
    if args.dry_run:
        write_metadata(
            metadata_output,
            args,
            sequences,
            [1, args.sequence_length, len(DANCE_JOINTS), args.feature_dims],
            [1, args.embedding_dim],
        )
        print(f"[DRY-RUN] metadata written: {metadata_output}")
        return

    tf = import_tensorflow()
    tf.keras.utils.set_random_seed(args.seed)

    if args.convert_only:
        encoder = tf.keras.models.load_model(keras_output)
        size = export_tflite(tf, encoder, output_path, quantize=not args.no_quantize)
        input_shape, output_shape, embedding = verify_tflite(tf, output_path, sample)
        write_metadata(metadata_output, args, sequences, input_shape, output_shape)
        print(f"[SAVE] TFLite encoder: {output_path} ({size / 1024:.1f} KiB)")
        print(f"[VERIFY] input={input_shape} output={output_shape} "
              f"embedding_norm={float(np.linalg.norm(embedding)):.4f}")
        return

    gen_cls = TripletBatchGenerator if args.loss_type == "triplet" else PairBatchGenerator
    train_gen = gen_cls(
        sequences, args.sequence_length, args.feature_dims, args.batch_size,
        args.steps_per_epoch, args.positive_jitter, args.negative_gap,
        args.noise_std, args.seed)
    val_gen = gen_cls(
        sequences, args.sequence_length, args.feature_dims, args.batch_size,
        args.validation_steps, args.positive_jitter, args.negative_gap,
        0.0, args.seed + 1000)

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
    write_metadata(
        metadata_output,
        args,
        sequences,
        input_shape,
        output_shape,
        history={k: [float(x) for x in v] for k, v in history.history.items()},
    )
    print(f"[SAVE] TFLite encoder: {output_path} ({size / 1024:.1f} KiB)")
    print(f"[VERIFY] input={input_shape} output={output_shape} "
          f"embedding_norm={float(np.linalg.norm(embedding)):.4f}")
    print(f"[SAVE] metadata: {metadata_output}")
    print("\nRun with:")
    print(f"  python src/main.py -s scratch --scratch-model-name {args.model_name}")


if __name__ == "__main__":
    main()
