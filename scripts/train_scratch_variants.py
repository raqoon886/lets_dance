#!/usr/bin/env python3
"""Train multiple scratch TCN/GCN TFLite model variants."""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "train_scratch_similarity.py")


@dataclass(frozen=True)
class Variant:
    name: str
    model_type: str
    filters: int
    blocks: int
    embedding_dim: int
    kernel_size: int
    dropout: float
    learning_rate: float
    positive_jitter: int
    negative_gap: int
    noise_std: float
    seed_offset: int


DEFAULT_VARIANTS = [
    Variant("tcn_e32", "tcn", 32, 3, 32, 3, 0.10, 1e-3, 6, 45, 0.015, 0),
    Variant("tcn_e64", "tcn", 64, 4, 64, 5, 0.15, 8e-4, 8, 60, 0.020, 101),
    Variant("gcn_e32", "gcn", 32, 2, 32, 3, 0.10, 1e-3, 6, 45, 0.015, 202),
    Variant("gcn_e64", "gcn", 64, 3, 64, 5, 0.15, 8e-4, 8, 60, 0.020, 303),
]

PROFILES = {
    "smoke": {"epochs": 1, "batch_size": 8, "steps_per_epoch": 2, "validation_steps": 1},
    "quick": {"epochs": 3, "batch_size": 32, "steps_per_epoch": 20, "validation_steps": 5},
    "full": {"epochs": 30, "batch_size": 64, "steps_per_epoch": 120, "validation_steps": 20},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train scratch TFLite variants.")
    parser.add_argument("--data-dir", default="data/reference_dances")
    parser.add_argument("--output-dir", default="data/models/scratch")
    parser.add_argument("--registry-output", default="data/models/scratch/model_registry.json")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    parser.add_argument("--only", default="", help="Comma-separated model names")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--feature-dims", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override profile epochs for all selected variants")
    parser.add_argument("--embedding-dims", default="",
                        help="Optional comma-separated embedding dims for generated grid")
    parser.add_argument("--model-types", default="tcn,gcn",
                        help="Model types used with --embedding-dims grid")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-quantize", action="store_true")
    return parser.parse_args()


def project_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def rel(path):
    return os.path.relpath(path, PROJECT_ROOT)


def parse_csv_ints(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_grid_variants(args):
    dims = parse_csv_ints(args.embedding_dims)
    if not dims:
        return DEFAULT_VARIANTS
    model_types = [x.strip() for x in args.model_types.split(",") if x.strip()]
    variants = []
    seed = 0
    for model_type in model_types:
        if model_type not in ("tcn", "gcn"):
            raise SystemExit(f"Unknown model type for grid: {model_type}")
        for dim in dims:
            filters = max(32, dim)
            blocks = 3 if model_type == "tcn" else 2
            variants.append(Variant(
                name=f"{model_type}_e{dim}",
                model_type=model_type,
                filters=filters,
                blocks=blocks,
                embedding_dim=dim,
                kernel_size=3,
                dropout=0.12,
                learning_rate=1e-3,
                positive_jitter=6,
                negative_gap=45,
                noise_std=0.015,
                seed_offset=seed,
            ))
            seed += 97
    return variants


def selected_variants(args):
    variants = build_grid_variants(args)
    if not args.only.strip():
        return variants
    names = {name.strip() for name in args.only.split(",") if name.strip()}
    known = {variant.name for variant in variants}
    unknown = sorted(names - known)
    if unknown:
        raise SystemExit(f"Unknown variant(s): {', '.join(unknown)}")
    return [variant for variant in variants if variant.name in names]


def build_command(args, variant, profile):
    output_dir = project_path(args.output_dir)
    tflite_path = os.path.join(output_dir, f"{variant.name}.tflite")
    keras_path = os.path.join(output_dir, f"{variant.name}_encoder.keras")
    metadata_path = os.path.join(output_dir, f"{variant.name}_meta.json")
    command = [
        sys.executable,
        TRAIN_SCRIPT,
        "--data-dir", args.data_dir,
        "--model-name", variant.name,
        "--model-type", variant.model_type,
        "--output-dir", output_dir,
        "--output", tflite_path,
        "--keras-output", keras_path,
        "--metadata-output", metadata_path,
        "--sequence-length", str(args.sequence_length),
        "--feature-dims", str(args.feature_dims),
        "--embedding-dim", str(variant.embedding_dim),
        "--filters", str(variant.filters),
        "--blocks", str(variant.blocks),
        "--kernel-size", str(variant.kernel_size),
        "--dropout", str(variant.dropout),
        "--learning-rate", str(variant.learning_rate),
        "--positive-jitter", str(variant.positive_jitter),
        "--negative-gap", str(variant.negative_gap),
        "--noise-std", str(variant.noise_std),
        "--seed", str(args.seed + variant.seed_offset),
        "--epochs", str(args.epochs if args.epochs is not None else profile["epochs"]),
        "--batch-size", str(profile["batch_size"]),
        "--steps-per-epoch", str(profile["steps_per_epoch"]),
        "--validation-steps", str(profile["validation_steps"]),
    ]
    if args.no_quantize:
        command.append("--no-quantize")
    return command, tflite_path, keras_path, metadata_path


def read_metrics(metadata_path):
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    history = metadata.get("history") or {}
    metrics = {}
    for key in ("binary_accuracy", "val_binary_accuracy", "loss", "val_loss"):
        values = history.get(key) or []
        if values:
            metrics[key] = float(values[-1])
    return metrics


def service_command(model_name):
    return f"python src/main.py -s scratch --scratch-model-name {model_name}"


def write_registry(path, args, entries):
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile": args.profile,
        "sequence_length": args.sequence_length,
        "feature_dims": args.feature_dims,
        "variants": entries,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    profile = PROFILES[args.profile]
    registry_path = project_path(args.registry_output)
    entries = []

    for variant in selected_variants(args):
        command, tflite_path, keras_path, metadata_path = build_command(args, variant, profile)
        print(f"\n[VARIANT] {variant.name} ({variant.model_type})")
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)

        entry = {
            "name": variant.name,
            "model_type": variant.model_type,
            "training_profile": args.profile,
            "parameters": {
                "filters": variant.filters,
                "blocks": variant.blocks,
                "embedding_dim": variant.embedding_dim,
                "kernel_size": variant.kernel_size,
                "dropout": variant.dropout,
                "learning_rate": variant.learning_rate,
                "epochs": args.epochs if args.epochs is not None else profile["epochs"],
            },
            "paths": {
                "tflite": rel(tflite_path),
                "keras": rel(keras_path),
                "metadata": rel(metadata_path),
            },
            "service_command": service_command(variant.name),
            "metrics": read_metrics(metadata_path),
        }
        entries.append(entry)
        write_registry(registry_path, args, entries)
        print(f"[REGISTRY] {rel(registry_path)}")
        print(f"[RUN] {entry['service_command']}")

    print("\n[DONE]")
    for entry in entries:
        val_acc = entry["metrics"].get("val_binary_accuracy")
        suffix = f" val_acc={val_acc:.4f}" if val_acc is not None else ""
        print(f"  - {entry['name']}: {entry['paths']['tflite']}{suffix}")


if __name__ == "__main__":
    main()
