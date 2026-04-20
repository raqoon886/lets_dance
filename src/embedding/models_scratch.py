"""Scratch encoders (MLP, TCN, GCN) for contrastive skeleton learning.

Contract
--------
All three encoders share the same interface so they can be swapped freely:

    input  : (B, T, J, C)   T=sequence_length, J=12 dance joints, C=feature_dims
    output : (B, D)         L2-normalized embedding (cosine-similarity-ready)

Improvements in this version:
- MLP  : per-frame Dense + temporal global pool (≈3× fewer params)
- TCN  : Conv2D (K, 1) preserving J axis (per-joint temporal features)
- GCN  : ST-GCN spatial partition strategy with 3 adjacencies
         (self / centripetal / centrifugal)

All ops used here are TFLite-convertible on TF 2.15+:
- Conv1D / Conv2D (incl. dilated, asymmetric kernels) / Dense / BN / Dropout
- Reshape / Add / GlobalAveragePooling{1D,2D} / Flatten / Activation
- Lambda wrapping tf.einsum and tf.math.l2_normalize
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers


# Skeleton graph on the 12 DANCE_JOINTS subset.
# Index map (src/pose/landmark_utils.DANCE_JOINTS):
#   0: L_shoulder(11)   1: R_shoulder(12)
#   2: L_elbow(13)      3: R_elbow(14)
#   4: L_wrist(15)      5: R_wrist(16)
#   6: L_hip(23)        7: R_hip(24)
#   8: L_knee(25)       9: R_knee(26)
#  10: L_ankle(27)     11: R_ankle(28)
DANCE_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),                          # shoulders
    (0, 2), (2, 4),                  # left arm
    (1, 3), (3, 5),                  # right arm
    (0, 6), (1, 7), (6, 7),          # torso / hip
    (6, 8), (8, 10),                 # left leg
    (7, 9), (9, 11),                 # right leg
)

# Graph distance from the virtual root (hip midpoint) for each of the 12 joints.
# Used for ST-GCN spatial partitioning. Lower = closer to torso core.
ROOT_DISTANCE: Tuple[int, ...] = (
    1, 1,   # shoulders
    2, 2,   # elbows
    3, 3,   # wrists
    0, 0,   # hips (root level)
    1, 1,   # knees
    2, 2,   # ankles
)


def build_symmetric_adjacency(num_joints: int = 12,
                              edges: Tuple[Tuple[int, int], ...] = DANCE_EDGES,
                              ) -> np.ndarray:
    """Single symmetric-normalized adjacency with self-loops: D^{-1/2}(A+I)D^{-1/2}."""
    A = np.eye(num_joints, dtype=np.float32)
    for a, b in edges:
        A[a, b] = 1.0
        A[b, a] = 1.0
    d = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-8))
    D_inv_sqrt = np.diag(d_inv_sqrt).astype(np.float32)
    return (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)


def build_stgcn_partitions(num_joints: int = 12,
                            edges: Tuple[Tuple[int, int], ...] = DANCE_EDGES,
                            distances: Tuple[int, ...] = ROOT_DISTANCE,
                            ) -> np.ndarray:
    """ST-GCN spatial partition strategy (Yan 2018).

    Returns 3 row-normalized (J, J) adjacency matrices stacked as (3, J, J):
        [0] A_self       : identity (self-loops)
        [1] A_centripetal: for i, aggregates from neighbors closer to root
        [2] A_centrifugal: for i, aggregates from neighbors farther from root

    Sibling edges (equal distance, e.g. left↔right shoulder) split equally
    between centripetal and centrifugal.
    """
    r = np.asarray(distances, dtype=np.int32)
    A_self = np.eye(num_joints, dtype=np.float32)
    A_cent = np.zeros((num_joints, num_joints), dtype=np.float32)
    A_outer = np.zeros((num_joints, num_joints), dtype=np.float32)

    bi_edges = list(edges) + [(b, a) for a, b in edges]
    for i, j in bi_edges:
        if r[j] < r[i]:
            A_cent[i, j] = 1.0
        elif r[j] > r[i]:
            A_outer[i, j] = 1.0
        else:  # sibling
            A_cent[i, j] = 0.5
            A_outer[i, j] = 0.5

    def row_norm(A: np.ndarray) -> np.ndarray:
        d = A.sum(axis=1, keepdims=True)
        return (A / np.maximum(d, 1e-8)).astype(np.float32)

    return np.stack([A_self, row_norm(A_cent), row_norm(A_outer)], axis=0)


def _l2_normalize(x: tf.Tensor) -> tf.Tensor:
    return tf.math.l2_normalize(x, axis=-1)


# ---------------- MLP (per-frame + temporal pool) ----------------

def build_mlp_encoder(sequence_length: int, num_joints: int, feature_dims: int,
                      embedding_dim: int, hidden: int = 256,
                      dropout: float = 0.2) -> Model:
    """Per-frame Dense encoder followed by temporal average pooling.

    Each frame's (J*C) features are projected by a shared MLP, then averaged
    over T. This is translation-equivariant across time and uses fewer params
    than flattening T*J*C.
    """
    inputs = layers.Input(shape=(sequence_length, num_joints, feature_dims), name="pose")
    # (B, T, J, C) -> (B, T, J*C)
    x = layers.Reshape((sequence_length, num_joints * feature_dims), name="flatten_joints")(inputs)
    # Per-frame Dense (applied across the last axis at each timestep)
    x = layers.Dense(hidden, name="frame_dense_1")(x)
    x = layers.BatchNormalization(name="frame_bn_1")(x)
    x = layers.Activation("relu", name="frame_relu_1")(x)
    x = layers.Dropout(dropout, name="frame_drop_1")(x)
    x = layers.Dense(hidden, name="frame_dense_2")(x)
    x = layers.BatchNormalization(name="frame_bn_2")(x)
    x = layers.Activation("relu", name="frame_relu_2")(x)
    x = layers.Dropout(dropout, name="frame_drop_2")(x)
    # Temporal average over T
    x = layers.GlobalAveragePooling1D(name="temporal_pool")(x)
    # Head
    x = layers.Dense(hidden // 2, activation="relu", name="head_hidden")(x)
    x = layers.Dense(embedding_dim, name="embedding_raw")(x)
    out = layers.Lambda(_l2_normalize, name="embedding")(x)
    return Model(inputs, out, name="scratch_mlp_encoder")


# ---------------- TCN (per-joint temporal Conv2D) ----------------

def _tcn_block_2d(x, filters: int, kernel_size: int, dilation: int,
                  dropout: float, name: str):
    """Temporal conv on T keeping J axis: Conv2D with (K, 1) kernel, dilation=(d, 1)."""
    residual = x
    x = layers.Conv2D(filters, (kernel_size, 1), padding="same",
                      dilation_rate=(dilation, 1), name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Dropout(dropout, name=f"{name}_drop1")(x)
    x = layers.Conv2D(filters, (kernel_size, 1), padding="same",
                      dilation_rate=(dilation, 1), name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    if residual.shape[-1] != filters:
        residual = layers.Conv2D(filters, 1, padding="same", name=f"{name}_res")(residual)
    x = layers.Add(name=f"{name}_add")([x, residual])
    return layers.Activation("relu", name=f"{name}_relu2")(x)


def build_tcn_encoder(sequence_length: int, num_joints: int, feature_dims: int,
                      embedding_dim: int, filters: int = 96, blocks: int = 4,
                      kernel_size: int = 3, dropout: float = 0.1) -> Model:
    """Per-joint dilated temporal ConvNet.

    Preserves J axis throughout the blocks (each block applies temporal conv
    with (K, 1) kernel, so joints evolve independently). Cross-joint mixing
    happens only at (a) initial 1x1 projection of C→filters, and (b) final
    GlobalAveragePooling2D over both T and J.
    """
    inputs = layers.Input(shape=(sequence_length, num_joints, feature_dims), name="pose")
    x = layers.Conv2D(filters, 1, padding="same", name="input_proj")(inputs)
    x = layers.BatchNormalization(name="input_bn")(x)
    x = layers.Activation("relu", name="input_relu")(x)
    for i in range(blocks):
        x = _tcn_block_2d(x, filters, kernel_size, 2 ** i, dropout,
                          name=f"tcn{i+1}")
    x = layers.GlobalAveragePooling2D(name="spacetime_pool")(x)
    x = layers.Dense(filters, activation="relu", name="head_hidden")(x)
    x = layers.Dense(embedding_dim, name="embedding_raw")(x)
    out = layers.Lambda(_l2_normalize, name="embedding")(x)
    return Model(inputs, out, name="scratch_tcn_encoder")


# ---------------- GCN (ST-GCN multi-partition) ----------------

def _make_graph_agg(adj_tensor):
    """Builds a closure over `adj_tensor` that performs A @ x aggregation."""
    def aggregate(x):
        return tf.einsum("ij,btjc->btic", adj_tensor, x)
    return aggregate


def _stgcn_block(x, adjacencies: List[tf.Tensor], filters: int,
                 kernel_size: int, dropout: float, name: str):
    """ST-GCN block with multi-partition spatial + temporal conv.

    adjacencies: list of tf.constant (J, J) tensors, one per partition.
    """
    residual = x

    # Spatial: aggregate with each partition, project separately, sum
    parts = []
    for p, A_p in enumerate(adjacencies):
        agg = layers.Lambda(_make_graph_agg(A_p),
                            name=f"{name}_agg_p{p}")(x)
        proj = layers.Conv2D(filters, 1, padding="same",
                             name=f"{name}_proj_p{p}")(agg)
        parts.append(proj)
    if len(parts) > 1:
        x = layers.Add(name=f"{name}_partition_sum")(parts)
    else:
        x = parts[0]
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)

    # Temporal: 1D conv on T axis, per-joint (kernel (K, 1))
    x = layers.Conv2D(filters, (kernel_size, 1), padding="same",
                      name=f"{name}_temporal")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Dropout(dropout, name=f"{name}_drop")(x)

    if residual.shape[-1] != filters:
        residual = layers.Conv2D(filters, 1, padding="same",
                                  name=f"{name}_res")(residual)
    x = layers.Add(name=f"{name}_add")([x, residual])
    return layers.Activation("relu", name=f"{name}_relu2")(x)


def build_gcn_encoder(sequence_length: int, num_joints: int, feature_dims: int,
                      embedding_dim: int, filters: int = 64, blocks: int = 3,
                      kernel_size: int = 9, dropout: float = 0.1,
                      partition_strategy: str = "distance") -> Model:
    """ST-GCN encoder with spatial partitioning.

    partition_strategy:
        "distance" (default) : 3 partitions (self / centripetal / centrifugal)
        "uniform"            : 1 symmetric-normalized adjacency
    """
    if partition_strategy == "distance":
        adj_np = build_stgcn_partitions(num_joints)        # (3, J, J)
    elif partition_strategy == "uniform":
        adj_np = build_symmetric_adjacency(num_joints)[None, ...]  # (1, J, J)
    else:
        raise ValueError(f"Unknown partition_strategy: {partition_strategy}")

    adjacencies = [tf.constant(adj_np[p], dtype=tf.float32,
                               name=f"adjacency_p{p}")
                   for p in range(adj_np.shape[0])]

    inputs = layers.Input(shape=(sequence_length, num_joints, feature_dims), name="pose")
    x = layers.Conv2D(filters, 1, padding="same", name="input_proj")(inputs)
    x = layers.BatchNormalization(name="input_bn")(x)
    x = layers.Activation("relu", name="input_relu")(x)
    for i in range(blocks):
        x = _stgcn_block(x, adjacencies, filters, kernel_size, dropout,
                         name=f"stgcn{i+1}")
    x = layers.GlobalAveragePooling2D(name="spacetime_pool")(x)
    x = layers.Dense(filters, activation="relu", name="head_hidden")(x)
    x = layers.Dense(embedding_dim, name="embedding_raw")(x)
    out = layers.Lambda(_l2_normalize, name="embedding")(x)
    return Model(inputs, out, name="scratch_gcn_encoder")


ENCODER_BUILDERS = {
    "mlp": build_mlp_encoder,
    "tcn": build_tcn_encoder,
    "gcn": build_gcn_encoder,
}


def build_encoder(model_type: str, sequence_length: int, num_joints: int,
                  feature_dims: int, embedding_dim: int, **kwargs) -> Model:
    if model_type not in ENCODER_BUILDERS:
        raise ValueError(f"Unknown model_type: {model_type}. "
                         f"Choices: {list(ENCODER_BUILDERS)}")
    return ENCODER_BUILDERS[model_type](
        sequence_length=sequence_length, num_joints=num_joints,
        feature_dims=feature_dims, embedding_dim=embedding_dim, **kwargs,
    )
