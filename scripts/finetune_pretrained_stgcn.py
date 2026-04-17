#!/usr/bin/env python3
"""Fine-tune PYSKL ST-GCN++ (NTU-60 pretrained) on dance data and export TFLite.

Pipeline:
  1. Build minimal ST-GCN++ in PyTorch (no PYSKL/MMCV dependency)
  2. Load pretrained checkpoint with COCO 17→12 DANCE_JOINTS adaptation
  3. Replace classification head with embedding projection
  4. Fine-tune with triplet loss on local dance reference data
  5. Export: PyTorch → ONNX → TFLite
  6. Validate with ScratchPoseSimilarity smoke test

Usage:
  python scripts/finetune_pretrained_stgcn.py \
    --checkpoint data/models/pretrain/stgcnpp/stgcnpp_ntu60_xsub_hrnet_j.pth \
    --model-name stgcnpp_dance_e64 \
    --embedding-dim 64 \
    --epochs 30
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pose.landmark_utils import DANCE_JOINTS
from scoring.scratch_features import build_pose_window, normalize_pose_landmarks

# ---------------------------------------------------------------------------
# COCO 17 → 12 DANCE_JOINTS mapping
# COCO: 0=nose,1=Leye,2=Reye,3=Lear,4=Rear,5=Lshoulder,6=Rshoulder,
#        7=Lelbow,8=Relbow,9=Lwrist,10=Rwrist,11=Lhip,12=Rhip,
#        13=Lknee,14=Rknee,15=Lankle,16=Rankle
# DANCE_JOINTS (MediaPipe indices mapped to body parts):
#   [11,12,13,14,15,16,23,24,25,26,27,28] =
#   Lshoulder,Rshoulder,Lelbow,Relbow,Lwrist,Rwrist,
#   Lhip,Rhip,Lknee,Rknee,Lankle,Rankle
# → COCO indices: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# ---------------------------------------------------------------------------
COCO_TO_DANCE = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
NUM_JOINTS = 12
NUM_COCO_JOINTS = 17


# ===========================================================================
# ST-GCN++ Architecture (minimal, self-contained PyTorch)
# ===========================================================================

def build_coco17_adjacency():
    """COCO 17-joint adjacency (3 subsets: self, centripetal, centrifugal)."""
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 4),       # head
        (0, 5), (0, 6),                         # neck→shoulder
        (5, 7), (7, 9),                         # left arm
        (6, 8), (8, 10),                        # right arm
        (5, 11), (6, 12),                       # torso
        (11, 13), (13, 15),                     # left leg
        (12, 14), (14, 16),                     # right leg
        (5, 6), (11, 12),                       # cross
    ]
    center = 0  # nose as center for partition
    hop_dist = np.full((17, 17), np.inf)
    np.fill_diagonal(hop_dist, 0)
    for u, v in edges:
        hop_dist[u, v] = 1
        hop_dist[v, u] = 1
    # Floyd-Warshall
    for k in range(17):
        for i in range(17):
            for j in range(17):
                if hop_dist[i, k] + hop_dist[k, j] < hop_dist[i, j]:
                    hop_dist[i, j] = hop_dist[i, k] + hop_dist[k, j]

    A = np.zeros((3, 17, 17), dtype=np.float32)
    for i in range(17):
        for j in range(17):
            if hop_dist[i, j] <= 1:
                if hop_dist[j, center] == hop_dist[i, center]:
                    A[0, i, j] = 1  # self-loop subset
                elif hop_dist[j, center] < hop_dist[i, center]:
                    A[1, i, j] = 1  # centripetal
                else:
                    A[2, i, j] = 1  # centrifugal
    # Normalize each subset
    for k in range(3):
        d = A[k].sum(axis=1, keepdims=True)
        d = np.maximum(d, 1e-6)
        A[k] = A[k] / d
    return A


def subsample_adjacency(A_17, joint_indices):
    """Extract sub-adjacency for selected joints from 17-joint graph."""
    n = len(joint_indices)
    A_sub = np.zeros((A_17.shape[0], n, n), dtype=np.float32)
    for k in range(A_17.shape[0]):
        for i, ji in enumerate(joint_indices):
            for j, jj in enumerate(joint_indices):
                A_sub[k, i, j] = A_17[k, ji, jj]
        # Re-normalize
        d = A_sub[k].sum(axis=1, keepdims=True)
        d = np.maximum(d, 1e-6)
        A_sub[k] = A_sub[k] / d
    return A_sub


class SpatialGCN(nn.Module):
    """Adaptive spatial graph convolution with 3 subsets."""

    def __init__(self, in_channels, out_channels, A, adaptive=True):
        super().__init__()
        self.num_subsets = A.shape[0]
        self.out_channels = out_channels
        # Learnable adjacency (initialized from pretrained A)
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32))
        # Conv: in → num_subsets * out
        self.conv = nn.Conv2d(in_channels, self.num_subsets * out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Residual path
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down = None

    def forward(self, x):
        # x: (B, C, T, V)
        residual = self.down(x) if self.down is not None else x
        B, C, T, V = x.shape
        # Project to subsets
        y = self.conv(x)  # (B, K*out, T, V)
        y = y.view(B, self.num_subsets, self.out_channels, T, V)
        # Aggregate with adjacency
        out = torch.zeros(B, self.out_channels, T, V, device=x.device)
        for k in range(self.num_subsets):
            # y_k: (B, out, T, V) @ A_k: (V, V) → (B, out, T, V)
            out = out + torch.einsum("bctv,vw->bctw", y[:, k], self.A[k])
        out = self.relu(self.bn(out) + residual)
        return out


class MultiScaleTCN(nn.Module):
    """Multi-scale temporal convolution with multiple dilation branches."""

    def __init__(self, channels, stride=1):
        super().__init__()
        # Branch channel allocation (matches PYSKL default for base_channels=64)
        # The exact split depends on channels; we compute proportionally
        # PYSKL default: first branch gets ceil(channels * 5/18) rounded,
        # rest get floor(channels/6) each, and a 1x1 branch
        n_dilated = 4  # branches with dilated temporal conv
        n_1x1 = 1      # pooling-like branch (just 1x1)
        ch_first = channels - (n_dilated + n_1x1) * (channels // (n_dilated + n_1x1 + 1))
        ch_rest = (channels - ch_first) // (n_dilated + n_1x1)
        ch_first = channels - ch_rest * (n_dilated + n_1x1)

        # We use the exact channel counts from the checkpoint
        self.branches = nn.ModuleList()

        dilations = [1, 2, 3, 4]
        branch_channels = self._compute_branch_channels(channels)

        for i, (ch, dilation) in enumerate(zip(branch_channels[:n_dilated], dilations)):
            self.branches.append(nn.Sequential(
                nn.Conv2d(channels, ch, 1),     # pointwise
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                TemporalConv(ch, kernel_size=3, stride=stride,
                                      dilation=dilation),
            ))

        # Max-pool branch
        pool_ch = branch_channels[n_dilated]
        pool_branch = nn.Sequential(
            nn.Conv2d(channels, pool_ch, 1),
            nn.BatchNorm2d(pool_ch),
        )
        self.branches.append(pool_branch)
        self.pool_idx = len(self.branches) - 1
        self.pool_stride = stride

        # 1x1 branch (identity temporal, just channel transform)
        self.branches.append(nn.Conv2d(channels, branch_channels[-1], 1))
        self.identity_idx = len(self.branches) - 1

        # Input transform (BN + 1x1 conv for residual)
        self.transform = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
        )
        self.bn = nn.BatchNorm2d(channels)
        self.stride = stride

    def _compute_branch_channels(self, channels):
        """Match PYSKL's channel allocation."""
        # 6 branches total. From checkpoint analysis:
        # 64 → [14, 10, 10, 10, 10, 10]
        # 128 → [23, 21, 21, 21, 21, 21]
        # 256 → [46, 42, 42, 42, 42, 42]
        n_branches = 6
        ch_rest = channels // n_branches
        ch_first = channels - ch_rest * (n_branches - 1)
        return [ch_first] + [ch_rest] * (n_branches - 1)

    def forward(self, x):
        # x: (B, C, T, V)
        res = self.transform(x)
        parts = []
        target_t = None  # determined by first dilated conv branch
        for i, branch in enumerate(self.branches):
            if i == self.pool_idx:
                p = branch(x)
                if self.stride > 1:
                    p = F.max_pool2d(p, (self.stride, 1), stride=(self.stride, 1))
                else:
                    p = F.max_pool2d(p, (3, 1), stride=(1, 1), padding=(1, 0))
                # Pad/truncate temporal dim to match dilated conv output
                if target_t is not None and p.shape[2] != target_t:
                    p = F.adaptive_avg_pool2d(p, (target_t, p.shape[3]))
                parts.append(p)
            elif i == self.identity_idx:
                p = branch(x)
                if self.stride > 1:
                    p = F.adaptive_avg_pool2d(p, (target_t or (x.shape[2] // self.stride), p.shape[3]))
                parts.append(p)
            else:
                p = branch(x)
                if target_t is None:
                    target_t = p.shape[2]
                parts.append(p)
        out = torch.cat(parts, dim=1)  # (B, channels, T', V)
        if self.stride > 1:
            # Match residual temporal dim to output
            res = F.adaptive_avg_pool2d(res, (target_t, res.shape[3]))
        out = self.bn(out) + res
        return out


class TemporalConv(nn.Module):
    """Temporal convolution (regular Conv2d, matching PYSKL)."""

    def __init__(self, channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            channels, channels, (kernel_size, 1),
            stride=(stride, 1), padding=(padding, 0),
            dilation=(dilation, 1),
        )

    def forward(self, x):
        return self.conv(x)


class STGCNPPBlock(nn.Module):
    """One ST-GCN++ block: Spatial GCN + Multi-Scale TCN."""

    def __init__(self, in_channels, out_channels, A, stride=1):
        super().__init__()
        self.gcn = SpatialGCN(in_channels, out_channels, A)
        self.tcn = MultiScaleTCN(out_channels, stride=stride)
        if stride > 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = None

    def forward(self, x):
        out = self.gcn(x)
        out = self.tcn(out)
        return out


class STGCNPPBackbone(nn.Module):
    """ST-GCN++ backbone: 10 blocks in 3 stages."""

    def __init__(self, in_channels, num_joints, base_channels=64):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        self.in_channels = in_channels
        self.num_joints = num_joints

        A_default = subsample_adjacency(build_coco17_adjacency(), COCO_TO_DANCE)

        # 10 blocks: (in, out, stride)
        block_cfg = [
            (in_channels,     base_channels,     1),  # 0
            (base_channels,   base_channels,     1),  # 1
            (base_channels,   base_channels,     1),  # 2
            (base_channels,   base_channels,     1),  # 3
            (base_channels,   base_channels * 2, 2),  # 4 stride
            (base_channels*2, base_channels * 2, 1),  # 5
            (base_channels*2, base_channels * 2, 1),  # 6
            (base_channels*2, base_channels * 4, 2),  # 7 stride
            (base_channels*4, base_channels * 4, 1),  # 8
            (base_channels*4, base_channels * 4, 1),  # 9
        ]
        self.gcn = nn.ModuleList()
        for in_c, out_c, stride in block_cfg:
            self.gcn.append(STGCNPPBlock(in_c, out_c, A_default, stride=stride))

        self.out_channels = base_channels * 4  # 256

    def forward(self, x):
        # x: (B, C, T, V) where C=in_channels, V=num_joints
        B, C, T, V = x.shape
        x = x.reshape(B, C * V, T)
        x = self.data_bn(x)
        x = x.reshape(B, C, T, V)
        for block in self.gcn:
            x = block(x)
        # Global average pool: (B, 256, T', V) → (B, 256)
        x = x.mean(dim=-1).mean(dim=-1)
        return x


class STGCNPPEmbedding(nn.Module):
    """ST-GCN++ backbone + embedding projection head."""

    def __init__(self, in_channels=2, num_joints=12, base_channels=64,
                 embedding_dim=64):
        super().__init__()
        self.backbone = STGCNPPBackbone(in_channels, num_joints, base_channels)
        self.projection = nn.Sequential(
            nn.Linear(self.backbone.out_channels, embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x):
        # x: (B, T, J, C) — BTJC layout from service
        # Convert to (B, C, T, V) for ST-GCN
        x = x.permute(0, 3, 1, 2)  # (B, C, T, V)
        feat = self.backbone(x)      # (B, 256)
        emb = self.projection(feat)  # (B, embedding_dim)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb


# ===========================================================================
# Weight Loading: PYSKL checkpoint → our model (17→12 joint adaptation)
# ===========================================================================

def load_pretrained_weights(model, checkpoint_path, in_channels=2):
    """Load PYSKL ST-GCN++ checkpoint with joint adaptation."""
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]

    our_sd = model.state_dict()
    loaded, skipped, adapted = [], [], []

    for key, param in sd.items():
        # Map PYSKL keys to our keys
        our_key = key
        if our_key.startswith("backbone."):
            our_key = our_key  # same prefix
        if our_key.startswith("cls_head."):
            skipped.append(f"{key} (classification head, skipped)")
            continue

        # Prepend backbone. for our nested model structure
        model_key = our_key
        if model_key.startswith("backbone."):
            model_key = model_key  # backbone.data_bn → backbone.data_bn

        if model_key not in our_sd:
            skipped.append(f"{key} → {model_key} (not in model)")
            continue

        target_shape = our_sd[model_key].shape
        source_shape = param.shape

        if source_shape == target_shape:
            our_sd[model_key] = param
            loaded.append(f"{key} → {model_key}")
        else:
            # Need adaptation
            adapted_param = adapt_weight(key, param, target_shape, in_channels)
            if adapted_param is not None:
                our_sd[model_key] = adapted_param
                adapted.append(f"{key} {tuple(source_shape)} → {model_key} {tuple(target_shape)}")
            else:
                skipped.append(f"{key} {tuple(source_shape)} → {model_key} {tuple(target_shape)} (shape mismatch)")

    model.load_state_dict(our_sd, strict=False)
    return {"loaded": len(loaded), "adapted": len(adapted), "skipped": len(skipped),
            "details": {"adapted": adapted, "skipped": skipped}}


def adapt_weight(key, param, target_shape, in_channels):
    """Adapt a single weight tensor for joint/channel changes."""
    source_shape = param.shape

    # data_bn: (C*17) → (C*12) — select dance joint features
    if "data_bn" in key and len(source_shape) == 1:
        # Original: interleaved as [j0_c0, j0_c1, j0_c2, j1_c0, ...]
        # or [c0_j0, c0_j1, ..., c0_j16, c1_j0, ...] depending on reshape order
        # PYSKL uses reshape(N, C*V, T) with C=3, V=17 → indices are C-major
        src_c = source_shape[0] // NUM_COCO_JOINTS  # 3
        tgt_c = target_shape[0] // NUM_JOINTS       # should be in_channels
        if src_c * NUM_COCO_JOINTS == source_shape[0]:
            p = param.view(src_c, NUM_COCO_JOINTS)
            p = p[:tgt_c, COCO_TO_DANCE]  # (tgt_c, 12)
            return p.reshape(-1)
        return None

    # Adjacency: (3, 17, 17) → (3, 12, 12)
    if ".gcn.A" in key and source_shape[-2:] == (17, 17):
        return param[:, COCO_TO_DANCE][:, :, COCO_TO_DANCE]

    # First block conv: (out, 3, 1, 1) → (out, 2, 1, 1) for in_channels adaptation
    if source_shape != target_shape and len(source_shape) == 4:
        if source_shape[1] != target_shape[1] and source_shape[0] == target_shape[0]:
            # Channel dim mismatch — take first `in_channels` channels
            return param[:, :target_shape[1]]

    return None


# ===========================================================================
# Dance Data Loading + Triplet Sampling
# ===========================================================================

class DanceSequence:
    def __init__(self, name, data, path):
        self.name = name
        self.data = data  # (frames, 33, 4)
        self.path = path
        self.num_frames = data.shape[0]


def load_dance_sequences(data_dir, min_frames=30):
    """Load reference dance sequences from data/reference_dances/."""
    sequences = []
    data_dir = os.path.join(PROJECT_ROOT, data_dir) if not os.path.isabs(data_dir) else data_dir
    for entry in sorted(os.listdir(data_dir)):
        ref_path = os.path.join(data_dir, entry, "reference.npy")
        if not os.path.isfile(ref_path):
            continue
        data = np.load(ref_path, allow_pickle=True).astype(np.float32)
        if data.shape[0] >= min_frames:
            sequences.append(DanceSequence(entry, data, ref_path))
    return sequences


def make_window(seq_data, end_idx, seq_len=30, feature_dims=2):
    """Build preprocessed window: (seq_len, 12, feature_dims)."""
    return build_pose_window(
        seq_data, end_idx, seq_len,
        target_joints=DANCE_JOINTS, feature_dims=feature_dims
    )


def triplet_batch(sequences, batch_size, seq_len=30, feature_dims=2,
                  positive_jitter=6, negative_gap=45, noise_std=0.01):
    """Generate (anchor, positive, negative) triplet batch."""
    anchors, positives, negatives = [], [], []
    for _ in range(batch_size):
        # Pick anchor sequence
        seq = random.choice(sequences)
        end_a = random.randint(seq_len - 1, seq.num_frames - 1)
        # Positive: same dance, nearby frame
        off = random.randint(-positive_jitter, positive_jitter)
        end_p = max(seq_len - 1, min(end_a + off, seq.num_frames - 1))
        # Negative: different dance
        others = [s for s in sequences if s is not seq]
        if not others:
            others = sequences
        neg_seq = random.choice(others)
        end_n = random.randint(seq_len - 1, neg_seq.num_frames - 1)

        a = make_window(seq.data, end_a, seq_len, feature_dims)
        p = make_window(seq.data, end_p, seq_len, feature_dims)
        n = make_window(neg_seq.data, end_n, seq_len, feature_dims)

        if noise_std > 0:
            a = a + np.random.randn(*a.shape).astype(np.float32) * noise_std
            p = p + np.random.randn(*p.shape).astype(np.float32) * noise_std
            n = n + np.random.randn(*n.shape).astype(np.float32) * noise_std

        anchors.append(a)
        positives.append(p)
        negatives.append(n)

    return (
        torch.tensor(np.array(anchors), dtype=torch.float32),
        torch.tensor(np.array(positives), dtype=torch.float32),
        torch.tensor(np.array(negatives), dtype=torch.float32),
    )


# ===========================================================================
# Training
# ===========================================================================

def triplet_loss(anchor, positive, negative, margin=0.2):
    """Triplet margin loss on L2-normalized embeddings."""
    dist_pos = (anchor - positive).pow(2).sum(dim=-1)
    dist_neg = (anchor - negative).pow(2).sum(dim=-1)
    loss = F.relu(dist_pos - dist_neg + margin)
    return loss.mean()


def train_epoch(model, sequences, optimizer, device, args):
    model.train()
    total_loss = 0.0
    total_margin = 0.0
    for step in range(args.steps_per_epoch):
        a, p, n = triplet_batch(
            sequences, args.batch_size, args.sequence_length,
            args.feature_dims, args.positive_jitter, args.negative_gap,
            args.noise_std,
        )
        a, p, n = a.to(device), p.to(device), n.to(device)
        emb_a = model(a)
        emb_p = model(p)
        emb_n = model(n)
        loss = triplet_loss(emb_a, emb_p, emb_n, margin=args.triplet_margin)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            dp = (emb_a - emb_p).pow(2).sum(-1).mean()
            dn = (emb_a - emb_n).pow(2).sum(-1).mean()
            total_loss += loss.item()
            total_margin += (dn - dp).item()

    n = args.steps_per_epoch
    return {"loss": total_loss / n, "margin": total_margin / n}


@torch.no_grad()
def validate(model, sequences, device, args):
    model.eval()
    total_loss = 0.0
    total_margin = 0.0
    for step in range(args.validation_steps):
        a, p, n = triplet_batch(
            sequences, args.batch_size, args.sequence_length,
            args.feature_dims, args.positive_jitter, args.negative_gap, 0.0,
        )
        a, p, n = a.to(device), p.to(device), n.to(device)
        emb_a, emb_p, emb_n = model(a), model(p), model(n)
        loss = triplet_loss(emb_a, emb_p, emb_n, margin=args.triplet_margin)
        dp = (emb_a - emb_p).pow(2).sum(-1).mean()
        dn = (emb_a - emb_n).pow(2).sum(-1).mean()
        total_loss += loss.item()
        total_margin += (dn - dp).item()

    n = args.validation_steps
    return {"val_loss": total_loss / n, "val_margin": total_margin / n}


def smoke_metrics(model, sequences, device, args, n_pairs=128):
    """Compute same/different cosine similarity for quality check."""
    model.eval()
    sames, diffs = [], []
    with torch.no_grad():
        for _ in range(n_pairs):
            seq = random.choice(sequences)
            ea = random.randint(args.sequence_length - 1, seq.num_frames - 1)
            off = random.randint(-args.positive_jitter, args.positive_jitter)
            ep = max(args.sequence_length - 1, min(ea + off, seq.num_frames - 1))
            other = random.choice([s for s in sequences if s is not seq] or sequences)
            en = random.randint(args.sequence_length - 1, other.num_frames - 1)

            windows = torch.tensor(np.stack([
                make_window(seq.data, ea, args.sequence_length, args.feature_dims),
                make_window(seq.data, ep, args.sequence_length, args.feature_dims),
                make_window(other.data, en, args.sequence_length, args.feature_dims),
            ]), dtype=torch.float32).to(device)

            embs = model(windows).cpu().numpy()
            embs = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-8)
            sames.append(float(np.dot(embs[0], embs[1])))
            diffs.append(float(np.dot(embs[0], embs[2])))

    return {
        "same_cosine_mean": float(np.mean(sames)),
        "different_cosine_mean": float(np.mean(diffs)),
        "margin_mean": float(np.mean(np.array(sames) - np.array(diffs))),
    }


# ===========================================================================
# Export: PyTorch → ONNX → TFLite
# ===========================================================================

def export_onnx(model, onnx_path, args, device):
    """Export model to ONNX format."""
    model.eval()
    dummy = torch.randn(1, args.sequence_length, NUM_JOINTS, args.feature_dims).to(device)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
    import onnx
    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)
    print(f"[EXPORT] ONNX: {onnx_path} ({os.path.getsize(onnx_path) / 1024:.1f} KiB)")


def convert_to_tflite(onnx_path, tflite_path, quantize=True):
    """Convert ONNX to TFLite via onnx2tf."""
    tf_dir = tflite_path + "_tf_tmp"
    cmd = ["onnx2tf", "-i", onnx_path, "-o", tf_dir, "-oiqt",
           "-kat", "input"]  # keep input dim order (prevent NCHW→NHWC transpose)
    subprocess.run(cmd, capture_output=True, check=True)

    # Find the float32 tflite (or integer_quant)
    float_tflite = os.path.join(tf_dir, os.path.basename(onnx_path).replace(".onnx", "_float32.tflite"))
    quant_tflite = os.path.join(tf_dir, os.path.basename(onnx_path).replace(".onnx", "_integer_quant.tflite"))

    # Prefer float32 — integer quantization from onnx2tf can produce
    # incompatible tensors for models with adaptive_avg_pool / einsum.
    source = float_tflite
    if not os.path.exists(source):
        # Try to find any tflite file
        for f in os.listdir(tf_dir):
            if f.endswith("_float32.tflite"):
                source = os.path.join(tf_dir, f)
                break

    import shutil
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    shutil.copy2(source, tflite_path)
    print(f"[EXPORT] TFLite: {tflite_path} ({os.path.getsize(tflite_path) / 1024:.1f} KiB)")

    # Cleanup
    shutil.rmtree(tf_dir, ignore_errors=True)
    return tflite_path


def verify_tflite(tflite_path, args):
    """Verify TFLite model runs and produces correct shapes."""
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    # Run dummy inference
    dummy = np.random.randn(*inp_det["shape"]).astype(np.float32)
    interp.set_tensor(inp_det["index"], dummy)
    interp.invoke()
    out = interp.get_tensor(out_det["index"])

    input_shape = [int(x) for x in inp_det["shape"]]
    output_shape = [int(x) for x in out_det["shape"]]
    print(f"[VERIFY] input={input_shape} output={output_shape} "
          f"embedding_norm={float(np.linalg.norm(out)):.4f}")
    return input_shape, output_shape


def runtime_smoke_test(tflite_path, sequences, args):
    """Test with ScratchPoseSimilarity (same interface as service)."""
    from scoring.scratch_similarity import ScratchPoseSimilarity
    ref = sequences[0].data
    other = sequences[min(1, len(sequences) - 1)].data

    comp = ScratchPoseSimilarity(
        tflite_path,
        sequence_length=args.sequence_length,
        feature_dims=args.feature_dims,
        input_layout="BTJC",
    )

    same = None
    for idx in range(args.sequence_length):
        same = comp.compute(ref[idx], ref, idx)

    comp.reset()
    cross = None
    for idx in range(args.sequence_length):
        cross = comp.compute(other[idx], ref, idx)

    print(f"[SMOKE] same={same:.4f} cross={cross:.4f} "
          f"finite={np.isfinite(same) and np.isfinite(cross)}")
    return {"same": float(same) if same else None, "cross": float(cross) if cross else None}


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune pretrained ST-GCN++ on dance data")
    p.add_argument("--checkpoint", required=True, help="Path to PYSKL .pth checkpoint")
    p.add_argument("--data-dir", default="data/reference_dances")
    p.add_argument("--model-name", default="stgcnpp_dance_e64")
    p.add_argument("--output-dir", default="data/models/embedding")
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--sequence-length", type=int, default=30)
    p.add_argument("--feature-dims", type=int, default=2, choices=[2, 3])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps-per-epoch", type=int, default=60)
    p.add_argument("--validation-steps", type=int, default=10)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--triplet-margin", type=float, default=0.2)
    p.add_argument("--positive-jitter", type=int, default=6)
    p.add_argument("--negative-gap", type=int, default=45)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--no-quantize", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="Training device. 'auto' uses CUDA if available.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[DEVICE] {device}")

    # 1. Load dance data
    sequences = load_dance_sequences(args.data_dir)
    print(f"[DATA] {len(sequences)} dance sequences:")
    for seq in sequences:
        print(f"  - {seq.name}: {seq.num_frames} frames {seq.data.shape}")

    # 2. Build model
    model = STGCNPPEmbedding(
        in_channels=args.feature_dims,
        num_joints=NUM_JOINTS,
        base_channels=args.base_channels,
        embedding_dim=args.embedding_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] ST-GCN++ Embedding: {total_params:,} params, embedding_dim={args.embedding_dim}")

    # 3. Load pretrained weights
    ckpt_path = os.path.join(PROJECT_ROOT, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    if os.path.exists(ckpt_path):
        result = load_pretrained_weights(model, ckpt_path, in_channels=args.feature_dims)
        print(f"[PRETRAIN] loaded={result['loaded']}, adapted={result['adapted']}, skipped={result['skipped']}")
        for detail in result["details"]["adapted"]:
            print(f"  [ADAPT] {detail}")
        for detail in result["details"]["skipped"][:5]:
            print(f"  [SKIP] {detail}")
    else:
        print(f"[WARN] Checkpoint not found: {ckpt_path}. Training from scratch.")

    # 4. Fine-tune
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"loss": [], "margin": [], "val_loss": [], "val_margin": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, sequences, optimizer, device, args)
        val_metrics = validate(model, sequences, device, args)
        elapsed = time.time() - t0

        for k, v in {**train_metrics, **val_metrics}.items():
            history.setdefault(k, []).append(v)

        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"loss={train_metrics['loss']:.4f}  margin={train_metrics['margin']:.3f}  "
              f"val_loss={val_metrics['val_loss']:.4f}  val_margin={val_metrics['val_margin']:.3f}  "
              f"({elapsed:.1f}s)")

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[BEST] Restored best weights (val_loss={best_val_loss:.6f})")

    # 5. Smoke metrics
    sm = smoke_metrics(model, sequences, device, args)
    print(f"[METRIC] same={sm['same_cosine_mean']:.4f} diff={sm['different_cosine_mean']:.4f} "
          f"margin={sm['margin_mean']:.4f}")

    # 6. Export
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    onnx_path = os.path.join(output_dir, f"{args.model_name}.onnx")
    tflite_path = os.path.join(output_dir, f"{args.model_name}.tflite")
    meta_path = os.path.join(output_dir, f"{args.model_name}_meta.json")
    pytorch_path = os.path.join(output_dir, f"{args.model_name}.pt")

    # Save PyTorch model
    model_cpu = model.cpu()
    torch.save(model_cpu.state_dict(), pytorch_path)
    print(f"[SAVE] PyTorch: {pytorch_path} ({os.path.getsize(pytorch_path) / 1024:.1f} KiB)")

    # Export ONNX
    export_onnx(model_cpu, onnx_path, args, torch.device("cpu"))

    # Convert to TFLite
    convert_to_tflite(onnx_path, tflite_path, quantize=not args.no_quantize)

    # Verify TFLite
    input_shape, output_shape = verify_tflite(tflite_path, args)

    # Runtime smoke test
    runtime_result = runtime_smoke_test(tflite_path, sequences, args)

    # 7. Save metadata
    best_epoch = int(np.argmin(history["val_loss"])) + 1 if history["val_loss"] else 0
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": args.model_name,
        "model_type": "stgcnpp_pretrained",
        "runtime_score_method": "embedding",
        "runtime_output_mode": "embedding",
        "input_layout": "BTJC",
        "input_shape": input_shape,
        "output_shape": output_shape,
        "target_joints": DANCE_JOINTS,
        "pretrained_checkpoint": os.path.relpath(ckpt_path, PROJECT_ROOT) if os.path.exists(ckpt_path) else None,
        "encoder_config": {
            "architecture": "stgcnpp",
            "base_channels": args.base_channels,
            "sequence_length": args.sequence_length,
            "num_joints": NUM_JOINTS,
            "feature_dims": args.feature_dims,
            "embedding_dim": args.embedding_dim,
            "total_params": total_params,
        },
        "config": vars(args),
        "references": [
            {"name": seq.name, "path": os.path.relpath(seq.path, PROJECT_ROOT), "frames": seq.num_frames}
            for seq in sequences
        ],
        "training_summary": {
            "epochs_ran": len(history.get("loss", [])),
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
        },
        "smoke_metrics": sm,
        "runtime_smoke": runtime_result,
        "history": {k: [float(x) for x in v] for k, v in history.items()},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] Metadata: {meta_path}")

    # Cleanup ONNX (optional, keep for debugging)
    # os.remove(onnx_path)

    print(f"\n{'='*60}")
    print(f"Run service with:")
    print(f"  python src/main.py -s embedding --embedding-model-name {args.model_name}")
    print(f"  python src/main.py -s embedding --embedding-model-path {tflite_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
