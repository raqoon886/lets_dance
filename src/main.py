"""
Let's Dance - Main Entry Point
Loads configuration and launches the game engine.
"""

import yaml
from game.engine import GameEngine


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Load game configuration from YAML file."""
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
