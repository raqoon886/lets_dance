"""
Let's Dance - Main Entry Point
Loads configuration and launches the game engine.
"""

import sys
import os
import argparse

# Add src to Python path so modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from game.engine import GameEngine


def load_config(config_path: str = None) -> dict:
    """Load game configuration from YAML file."""
    if config_path is None:
        # Resolve relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config", "settings.yaml")

    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"[WARN] Config not found at {config_path}, using defaults.")
        return {}


def main():
    """Application entry point."""
    parser = argparse.ArgumentParser(description="Let's Dance!")
    parser.add_argument("--model", "-m", choices=["movenet", "mediapipe"],
                        default="mediapipe",
                        help="포즈 추출 모델 선택 (기본: mediapipe)")
    parser.add_argument("--compare", "-c", type=str, default=None,
                        metavar="NPY_PATH",
                        help="레퍼런스 .npy 파일과 실시간 비교 모드 실행")
    parser.add_argument("--score-method", "-s",
                        choices=["direct", "embedding"],
                        default="direct",
                        help="유사도 계산 방식: direct(키포인트 비교) / embedding(ST-GCN 임베딩) (기본: direct)")
    parser.add_argument("--similarity", "-S",
                        choices=["cosine", "euclidean", "hybrid", "angle"],
                        default="cosine",
                        help="direct 모드 유사도 메트릭: cosine / euclidean / hybrid / angle (기본: cosine)")
    parser.add_argument("--delay", "-d", type=float, default=1.0,
                        help="반응 딜레이 허용 시간(초). 유저가 보고 따라하는 시간 보정 (기본: 1.0)")
    args = parser.parse_args()

    # Direct Compare 모드
    if args.compare:
        from direct_compare.demo import play_compare
        play_compare(args.compare, model=args.model)
        return

    config = load_config()
    config["pose_backend"] = args.model
    config["score_method"] = args.score_method
    config["similarity_method"] = args.similarity
    config["tolerance_delay"] = args.delay
    engine = GameEngine(config)

    try:
        engine.initialize()
        engine.run()
    except KeyboardInterrupt:
        print("\n[INFO] Game interrupted by user.")
    finally:
        engine.shutdown()
        print("[INFO] Game shut down cleanly.")


if __name__ == "__main__":
    main()
