"""
Collect Reference Dance Data
Records webcam + pose data for a reference dance that players will mimic.

Usage:
    python scripts/collect_reference.py --song "my_song" --duration 60
"""

import argparse
import os
import time
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Collect reference dance data")
    parser.add_argument("--song", type=str, required=True,
                        help="Song/dance identifier name")
    parser.add_argument("--duration", type=int, default=60,
                        help="Recording duration in seconds")
    parser.add_argument("--output-dir", type=str,
                        default="data/reference_dances",
                        help="Output directory for reference data")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device ID")
    return parser.parse_args()


def collect_reference(song_name: str, duration: int, output_dir: str,
                      camera_id: int):
    """
    Record reference dance landmarks and embeddings.

    Pipeline:
        1. Open camera and initialize pose detector
        2. Show countdown, then start recording
        3. For each frame: detect pose, normalize landmarks, store
        4. After recording: build sequence graphs, extract embeddings
        5. Save landmarks, embeddings, and metadata to output directory
    """
    song_dir = os.path.join(output_dir, song_name)
    os.makedirs(song_dir, exist_ok=True)

    # TODO:
    # 1. Initialize camera, PoseDetector, LandmarkProcessor
    # 2. Record landmarks for `duration` seconds
    # 3. Save as .npy files:
    #    - {song_dir}/landmarks.npy   (T, N_joints, 3)
    #    - {song_dir}/embeddings.npy  (num_windows, embedding_dim)
    #    - {song_dir}/metadata.json   (song info, duration, fps, timestamp)
    print(f"[MOCK] Would record {duration}s of reference dance for '{song_name}'")
    print(f"[MOCK] Output: {song_dir}/")

    # Placeholder outputs
    np.save(os.path.join(song_dir, "landmarks.npy"),
            np.zeros((duration * 30, 12, 3), dtype=np.float32))
    np.save(os.path.join(song_dir, "embeddings.npy"),
            np.zeros((duration, 128), dtype=np.float32))

    metadata = {
        "song": song_name,
        "duration": duration,
        "fps": 30,
        "num_joints": 12,
        "embedding_dim": 128,
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    import json
    with open(os.path.join(song_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Reference data saved to {song_dir}/")


if __name__ == "__main__":
    args = parse_args()
    collect_reference(args.song, args.duration, args.output_dir, args.camera)
