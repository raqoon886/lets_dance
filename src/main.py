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
                        choices=["direct", "embedding", "scratch"],
                        default="direct",
                        help="유사도 계산 방식: direct / embedding / scratch(TFLite 모델) (기본: direct)")
    parser.add_argument("--similarity", "-S",
                        choices=["cosine", "euclidean", "hybrid", "angle"],
                        default="angle",
                        help="direct 모드 유사도 메트릭: cosine / euclidean / hybrid / angle (기본: cosine)")
    parser.add_argument("--delay", "-d", type=float, default=1.0,
                        help="반응 딜레이 허용 시간(초). 유저가 보고 따라하는 시간 보정 (기본: 1.0)")
    parser.add_argument("--scratch-model-name", type=str, default=None,
                        help="scratch 모드에서 사용할 registry 모델명. 예: gcn_e64")
    parser.add_argument("--scratch-model-dir", type=str, default=None,
                        help="scratch 모델 registry와 .tflite가 있는 디렉터리")
    parser.add_argument("--scratch-model-path", type=str, default=None,
                        help="registry를 거치지 않고 직접 사용할 scratch .tflite 경로")
    parser.add_argument("--scratch-sequence-length", type=int, default=None,
                        help="scratch 모델 입력 window 길이")
    parser.add_argument("--scratch-feature-dims", type=int, choices=[2, 3, 4], default=None,
                        help="scratch 모델 joint feature dimension: 2=xy, 3=xyz, 4=xyzv")
    parser.add_argument("--scratch-input-layout", choices=["BTJC", "BJTC", "BTC"], default=None,
                        help="scratch TFLite 입력 layout")
    parser.add_argument("--scratch-candidate-stride", type=int, default=None,
                        help="정답 후보 window stride")
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
    config.setdefault("scratch", {})
    scratch_updates = {
        "model_name": args.scratch_model_name,
        "model_dir": args.scratch_model_dir,
        "model_path": args.scratch_model_path,
        "sequence_length": args.scratch_sequence_length,
        "feature_dims": args.scratch_feature_dims,
        "input_layout": args.scratch_input_layout,
        "candidate_stride": args.scratch_candidate_stride,
    }
    config["scratch"].update({
        key: value for key, value in scratch_updates.items()
        if value is not None
    })
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
