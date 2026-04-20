"""
Precompute augmented skeleton variants per reference dance.

For each dance under data/reference_dances/<id>/, generates K augmented
copies of reference.npy at <id>/augments/aug_{k:02d}.npy together with
a manifest.json that records seeds and applied ops per file.

Usage
-----
    python scripts/precompute_augments.py                      # all dances, K=12, no flip
    python scripts/precompute_augments.py --k 20 --seed 123
    python scripts/precompute_augments.py --dance rasputin     # single dance
    python scripts/precompute_augments.py --flip               # enable flip (L/R-agnostic pretraining only)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pose.augment import AugConfig, apply_random  # noqa: E402


DEFAULT_ROOT = Path("data/reference_dances")


def _stable_seed(name: str, base: int) -> int:
    h = hashlib.md5(name.encode()).hexdigest()
    return (base + int(h[:8], 16)) % (2**31 - 1)


def process_dance(dance_dir: Path, k: int, seed: int, cfg: AugConfig) -> dict | None:
    ref_path = dance_dir / "reference.npy"
    if not ref_path.exists():
        print(f"  [skip] {ref_path} not found")
        return None

    pose = np.load(ref_path)
    if pose.ndim != 3 or pose.shape[1:] != (33, 3):
        print(f"  [skip] unexpected shape {pose.shape}, expected (T, 33, 3)")
        return None

    aug_dir = dance_dir / "augments"
    aug_dir.mkdir(exist_ok=True)

    top_rng = np.random.default_rng(seed)
    manifest = {
        "source": str(ref_path.relative_to(dance_dir.parent.parent)),
        "shape": list(pose.shape),
        "seed": seed,
        "config": cfg.__dict__,
        "augments": [],
    }
    for i in range(k):
        sub_seed = int(top_rng.integers(0, 2**31 - 1))
        sub_rng = np.random.default_rng(sub_seed)
        aug, ops = apply_random(pose, cfg, sub_rng)
        out_path = aug_dir / f"aug_{i:02d}.npy"
        np.save(out_path, aug)
        manifest["augments"].append({
            "file": out_path.name,
            "seed": sub_seed,
            "ops": ops,
        })

    (aug_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    p.add_argument("--k", type=int, default=12, help="augmented files per dance")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dance", type=str, default=None,
                   help="process only this dance id (default: all)")
    p.add_argument("--flip", action="store_true",
                   help="enable horizontal flip (default: off; use only for L/R-agnostic pretraining)")
    p.add_argument("--flip-p", type=float, default=None,
                   help="override flip probability (implies --flip)")
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    if args.dance:
        dances = [root / args.dance]
    else:
        dances = sorted(d for d in root.iterdir() if d.is_dir())

    cfg = AugConfig()
    if args.flip_p is not None:
        cfg.flip_p = args.flip_p
    elif args.flip:
        cfg.flip_p = 0.5

    for d in dances:
        print(f"[{d.name}] K={args.k}")
        m = process_dance(d, args.k, _stable_seed(d.name, args.seed), cfg)
        if m:
            fired = {op for a in m["augments"] for op in a["ops"]}
            print(f"  -> {len(m['augments'])} files, ops seen: {sorted(fired)}")


if __name__ == "__main__":
    main()
