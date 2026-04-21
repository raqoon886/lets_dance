#!/usr/bin/env python3
"""Precompute reference dance embeddings for scratch/embedding TFLite models.

The runtime only needs to infer the live user window. Reference windows are
fixed per song, so this script stores their encoder outputs in .npz cache files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for import_path in (PROJECT_ROOT, SRC_DIR):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

from pose.landmark_utils import DANCE_JOINTS  # noqa: E402
from scoring.scratch_features import build_pose_window  # noqa: E402
from scoring.scratch_similarity import ScratchPoseSimilarity  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute per-song reference embeddings for TFLite encoders."
    )
    parser.add_argument("--data-dir", default="data/reference_dances",
                        help="Directory containing */reference.npy dances.")
    parser.add_argument("--scratch-model-dir", default="data/models/scratch",
                        help="Directory containing scratch .tflite models.")
    parser.add_argument("--embedding-model-dir", default="data/models/embedding",
                        help="Directory containing fine-tuned embedding .tflite models.")
    parser.add_argument("--output-dir", default="data/cache/reference_embeddings",
                        help="Directory where .npz cache files are written.")
    parser.add_argument("--model-kind", choices=["scratch", "embedding", "all"],
                        default="all", help="Which model directory to process.")
    parser.add_argument("--model-name", action="append", default=[],
                        help="Model stem to include. Can be repeated. Default: all.")
    parser.add_argument("--dance-name", action="append", default=[],
                        help="Dance folder name to include. Can be repeated. Default: all with reference.npy.")
    parser.add_argument("--sequence-length", type=int, default=None,
                        help="Override model sequence length. Default: read metadata/input_shape.")
    parser.add_argument("--feature-dims", type=int, choices=[2, 3, 4], default=None,
                        help="Override model feature dims. Default: read metadata/input_shape.")
    parser.add_argument("--input-layout", choices=ScratchPoseSimilarity.LAYOUTS,
                        default=None, help="Override input layout. Default: metadata or BTJC.")
    parser.add_argument("--cache-stride", type=int, default=1,
                        help="Reference end-index stride to store. 1 is most flexible.")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float16",
                        help="Embedding dtype stored in the .npz cache.")
    parser.add_argument("--no-normalize-output", action="store_true",
                        help="Store raw model outputs instead of L2-normalized embeddings.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing cache files.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned jobs without running TFLite inference.")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def discover_models(model_kind: str, scratch_dir: Path, embedding_dir: Path,
                    names: Iterable[str]) -> list[dict]:
    wanted = set(names)
    dirs: list[tuple[str, Path]] = []
    if model_kind in ("scratch", "all"):
        dirs.append(("scratch", scratch_dir))
    if model_kind in ("embedding", "all"):
        dirs.append(("embedding", embedding_dir))

    models = []
    for kind, model_dir in dirs:
        if not model_dir.exists():
            continue
        for model_path in sorted(model_dir.glob("*.tflite")):
            name = model_path.stem
            if wanted and name not in wanted:
                continue
            meta_path = model_dir / f"{name}_meta.json"
            models.append({
                "kind": kind,
                "name": name,
                "path": model_path,
                "meta_path": meta_path,
                "meta": load_json(meta_path),
            })
    return models


def discover_dances(data_dir: Path, names: Iterable[str]) -> list[dict]:
    wanted = set(names)
    dances = []
    for ref_path in sorted(data_dir.glob("*/reference.npy")):
        dance_name = ref_path.parent.name
        if wanted and dance_name not in wanted:
            continue
        meta = load_json(ref_path.parent / "metadata.json")
        dances.append({
            "name": dance_name,
            "path": ref_path.parent,
            "reference_path": ref_path,
            "meta": meta,
        })
    return dances


def infer_model_config(model: dict, args: argparse.Namespace) -> dict:
    meta = model.get("meta") or {}
    config = meta.get("encoder_config") or meta.get("config") or {}
    input_shape = meta.get("input_shape") or []
    target_joints = meta.get("target_joints") or DANCE_JOINTS

    sequence_length = (
        args.sequence_length
        or config.get("sequence_length")
        or (input_shape[1] if len(input_shape) >= 2 else 30)
    )
    feature_dims = (
        args.feature_dims
        or config.get("feature_dims")
        or (input_shape[-1] if len(input_shape) >= 4 else 2)
    )
    input_layout = args.input_layout or meta.get("input_layout") or config.get("input_layout") or "BTJC"

    return {
        "sequence_length": int(sequence_length),
        "feature_dims": int(feature_dims),
        "input_layout": str(input_layout),
        "target_joints": [int(j) for j in target_joints],
    }


def cache_filename(dance_name: str, model_kind: str, model_name: str,
                   sequence_length: int, feature_dims: int,
                   input_layout: str, cache_stride: int) -> str:
    return (
        f"{dance_name}__{model_kind}__{model_name}"
        f"__seq{sequence_length}_fd{feature_dims}_{input_layout}_stride{cache_stride}.npz"
    )


def load_reference_sequence(path: Path) -> np.ndarray:
    seq = np.load(path, allow_pickle=True).astype(np.float32)
    if seq.ndim != 3:
        raise ValueError(f"Expected reference.npy shape (frames, joints, channels), got {seq.shape}")
    if seq.shape[2] == 3:
        visibility = np.ones((*seq.shape[:2], 1), dtype=np.float32)
        seq = np.concatenate([seq, visibility], axis=2)
    # 서비스는 cv2.flip(frame, 1) 후 MediaPipe를 실행하므로 (거울 모드),
    # reference.npy는 원본 영상(flip 없음)에서 추출됐기 때문에 x축이 반대.
    # 캐시 임베딩도 flip된 좌표 기준으로 생성해야 서비스 유저 포즈와 일치한다.
    seq[:, :, 0] = 1.0 - seq[:, :, 0]
    return seq


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    embedding = np.nan_to_num(np.asarray(embedding, dtype=np.float32).reshape(-1))
    norm = np.linalg.norm(embedding)
    if norm > 1e-8:
        embedding = embedding / norm
    return embedding.astype(np.float32)


def precompute_one(model: dict, dance: dict, output_dir: Path,
                   args: argparse.Namespace) -> dict:
    model_cfg = infer_model_config(model, args)
    sequence_length = model_cfg["sequence_length"]
    feature_dims = model_cfg["feature_dims"]
    input_layout = model_cfg["input_layout"]
    target_joints = model_cfg["target_joints"]

    cache_path = output_dir / cache_filename(
        dance["name"],
        model["kind"],
        model["name"],
        sequence_length,
        feature_dims,
        input_layout,
        args.cache_stride,
    )

    if cache_path.exists() and not args.overwrite:
        return {
            "status": "skipped",
            "reason": "exists",
            "cache_path": str(cache_path),
            "model": model["name"],
            "dance": dance["name"],
        }

    seq = load_reference_sequence(dance["reference_path"])
    if len(seq) < sequence_length:
        return {
            "status": "skipped",
            "reason": f"too_short:{len(seq)}<{sequence_length}",
            "cache_path": str(cache_path),
            "model": model["name"],
            "dance": dance["name"],
        }

    end_indices = np.arange(sequence_length - 1, len(seq), max(1, args.cache_stride), dtype=np.int32)
    if args.dry_run:
        return {
            "status": "planned",
            "cache_path": str(cache_path),
            "model": model["name"],
            "dance": dance["name"],
            "num_windows": int(len(end_indices)),
        }

    comparator = ScratchPoseSimilarity(
        model_path=str(model["path"]),
        sequence_length=sequence_length,
        feature_dims=feature_dims,
        input_layout=input_layout,
        target_joints=target_joints,
    )

    started = time.time()
    embeddings = []
    for idx, end_idx in enumerate(end_indices, start=1):
        window = build_pose_window(
            seq,
            int(end_idx),
            sequence_length,
            target_joints=target_joints,
            feature_dims=feature_dims,
        )
        embedding = comparator._infer_single(window)
        if not args.no_normalize_output:
            embedding = normalize_embedding(embedding)
        embeddings.append(embedding)
        if idx % 500 == 0:
            print(
                f"    {dance['name']} / {model['name']}: "
                f"{idx}/{len(end_indices)} windows",
                flush=True,
            )

    embeddings_arr = np.stack(embeddings, axis=0).astype(args.dtype)
    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_kind": model["kind"],
        "model_name": model["name"],
        "model_path": str(model["path"]),
        "model_sha256": sha256_file(model["path"]),
        "model_meta_path": str(model["meta_path"]) if model["meta_path"].exists() else "",
        "dance_name": dance["name"],
        "reference_path": str(dance["reference_path"]),
        "reference_sha256": sha256_file(dance["reference_path"]),
        "reference_frames": int(len(seq)),
        "sequence_length": sequence_length,
        "feature_dims": feature_dims,
        "input_layout": input_layout,
        "target_joints": target_joints,
        "cache_stride": int(args.cache_stride),
        "end_index_start": int(end_indices[0]),
        "end_index_stop": int(end_indices[-1]),
        "num_windows": int(len(end_indices)),
        "embedding_dim": int(embeddings_arr.shape[1]),
        "embedding_dtype": args.dtype,
        "l2_normalized": not args.no_normalize_output,
        "elapsed_sec": round(time.time() - started, 3),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        end_indices=end_indices,
        embeddings=embeddings_arr,
        metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
    )

    return {
        "status": "created",
        "cache_path": str(cache_path),
        "model": model["name"],
        "dance": dance["name"],
        "num_windows": int(len(end_indices)),
        "embedding_dim": int(embeddings_arr.shape[1]),
        "elapsed_sec": metadata["elapsed_sec"],
    }


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    scratch_dir = resolve_path(args.scratch_model_dir)
    embedding_dir = resolve_path(args.embedding_model_dir)
    output_dir = resolve_path(args.output_dir)

    models = discover_models(args.model_kind, scratch_dir, embedding_dir, args.model_name)
    dances = discover_dances(data_dir, args.dance_name)
    if not models:
        raise SystemExit("No TFLite models found for the requested filters.")
    if not dances:
        raise SystemExit("No reference.npy dances found for the requested filters.")

    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"models = {len(models)}")
    for model in models:
        cfg = infer_model_config(model, args)
        print(
            f" - {model['kind']}::{model['name']} "
            f"seq={cfg['sequence_length']} fd={cfg['feature_dims']} layout={cfg['input_layout']}"
        )
    print(f"dances = {len(dances)}")
    for dance in dances:
        print(f" - {dance['name']}: {dance['reference_path']}")
    print(f"output_dir = {output_dir}")
    print()

    results = []
    total = len(models) * len(dances)
    job = 0
    for model in models:
        for dance in dances:
            job += 1
            print(f"[{job}/{total}] {model['kind']}::{model['name']} x {dance['name']}")
            result = precompute_one(model, dance, output_dir, args)
            results.append(result)
            print("   ", result)

    print()
    created = sum(1 for r in results if r["status"] == "created")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    planned = sum(1 for r in results if r["status"] == "planned")
    print(f"Done. created={created}, skipped={skipped}, planned={planned}")


if __name__ == "__main__":
    main()
