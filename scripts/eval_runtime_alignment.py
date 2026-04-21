#!/usr/bin/env python3
"""Evaluate an exported TFLite encoder with service-style candidate ranking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf  # noqa: E402

from src.embedding.dataset_contrastive import (  # noqa: E402
    ContrastiveSequenceStore,
    TARGET_DANCES,
    apply_runtime_window_augment,
    sample_candidate_indices,
)
from src.scoring.scratch_similarity import resolve_scratch_model_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["scratch", "embedding"], default="scratch")
    p.add_argument("--model-name", required=True)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--data-dir", default="data/reference_dances")
    p.add_argument("--dances", nargs="+", default=list(TARGET_DANCES))
    p.add_argument("--feature-dims", type=int, choices=[2, 3, 4], default=2)
    p.add_argument("--sequence-length", type=int, default=30)
    p.add_argument("--candidate-stride", type=int, default=3)
    p.add_argument("--tolerance-frames", type=int, default=12)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--user-runtime-jitter", type=float, default=0.01)
    p.add_argument("--user-joint-dropout-prob", type=float, default=0.04)
    p.add_argument("--user-frame-hold-prob", type=float, default=0.05)
    p.add_argument("--user-temporal-warp-prob", type=float, default=0.25)
    p.add_argument("--user-temporal-warp-strength", type=float, default=0.15)
    p.add_argument("--output", default=None, help="optional JSON output path")
    return p.parse_args()


def _resolve_model_path(args: argparse.Namespace) -> Path:
    default_dir = f"data/models/{args.kind}"
    path = resolve_scratch_model_path(
        model_name=args.model_name,
        model_dir=args.model_dir or default_dir,
        model_path=args.model_path,
    )
    return Path(path)


def _embed_tflite(interpreter: tf.lite.Interpreter, windows: np.ndarray) -> np.ndarray:
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    outputs = []
    for i in range(len(windows)):
        x = windows[i:i + 1].astype(in_det["dtype"])
        interpreter.set_tensor(in_det["index"], x)
        interpreter.invoke()
        outputs.append(interpreter.get_tensor(out_det["index"])[0])
    emb = np.asarray(outputs, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=-1, keepdims=True)
    return emb / np.maximum(norms, 1e-8)


def evaluate_runtime_alignment(args: argparse.Namespace) -> dict:
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    store = ContrastiveSequenceStore(
        data_dir,
        dances=tuple(args.dances),
        feature_dims=args.feature_dims,
        verbose=True,
    )
    model_path = _resolve_model_path(args)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    rng = np.random.default_rng(args.seed)

    target_sims = []
    best_wrong_sims = []
    target_ranks = []
    top1_hits = 0
    top3_hits = 0

    for _ in range(args.samples):
        bundle = store.bundles[int(rng.integers(len(store)))]
        end = int(rng.integers(args.sequence_length - 1, bundle.num_frames))
        base_window = bundle.original[end - args.sequence_length + 1:end + 1]
        if bundle.augments and rng.random() < 0.7:
            aug = bundle.augments[int(rng.integers(len(bundle.augments)))]
            base_window = aug[end - args.sequence_length + 1:end + 1]

        user_window = apply_runtime_window_augment(
            base_window,
            rng,
            gaussian_sigma=args.user_runtime_jitter,
            joint_dropout_prob=args.user_joint_dropout_prob,
            frame_hold_prob=args.user_frame_hold_prob,
            temporal_warp_prob=args.user_temporal_warp_prob,
            temporal_warp_strength=args.user_temporal_warp_strength,
        )
        candidate_indices = sample_candidate_indices(
            bundle.num_frames,
            end,
            tolerance_frames=args.tolerance_frames,
            stride=args.candidate_stride,
            min_end_index=args.sequence_length - 1,
        )
        if end not in candidate_indices:
            continue
        candidate_windows = np.stack([
            bundle.original[idx - args.sequence_length + 1:idx + 1]
            for idx in candidate_indices
        ], axis=0).astype(np.float32)
        user_emb = _embed_tflite(interpreter, user_window[None, ...])[0]
        ref_embs = _embed_tflite(interpreter, candidate_windows)
        sims = np.dot(ref_embs, user_emb)
        order = np.argsort(-sims)
        ordered_indices = [candidate_indices[int(i)] for i in order]
        target_rank = int(ordered_indices.index(end)) + 1
        target_idx = candidate_indices.index(end)
        target_sim = float(sims[target_idx])
        wrong_mask = np.ones(len(candidate_indices), dtype=bool)
        wrong_mask[target_idx] = False
        best_wrong = float(np.max(sims[wrong_mask])) if np.any(wrong_mask) else target_sim

        target_sims.append(target_sim)
        best_wrong_sims.append(best_wrong)
        target_ranks.append(target_rank)
        top1_hits += int(target_rank == 1)
        top3_hits += int(target_rank <= 3)

    result = {
        "kind": args.kind,
        "model_name": args.model_name,
        "model_path": str(model_path),
        "samples": int(len(target_sims)),
    }
    if target_sims:
        target_arr = np.asarray(target_sims, dtype=np.float32)
        wrong_arr = np.asarray(best_wrong_sims, dtype=np.float32)
        rank_arr = np.asarray(target_ranks, dtype=np.float32)
        result.update({
            "target_cosine_mean": float(np.mean(target_arr)),
            "best_wrong_cosine_mean": float(np.mean(wrong_arr)),
            "target_margin_mean": float(np.mean(target_arr - wrong_arr)),
            "target_rank_mean": float(np.mean(rank_arr)),
            "top1_acc": float(top1_hits / len(target_arr)),
            "top3_acc": float(top3_hits / len(target_arr)),
        })
    return result


def main():
    args = parse_args()
    result = evaluate_runtime_alignment(args)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
