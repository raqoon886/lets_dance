"""
Let's Dance - Main Entry Point
Loads configuration and launches the game engine.
"""

import sys
import os

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
    config = load_config()
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
