"""
Game Modes - Different play modes with varying rules and difficulty.
"""

from abc import ABC, abstractmethod


class GameMode(ABC):
    """Base class for game modes."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def get_scoring_config(self) -> dict:
        """Return scoring configuration for this mode."""
        pass

    @abstractmethod
    def should_show_reference(self) -> bool:
        """Whether to display reference skeleton overlay."""
        pass

    @abstractmethod
    def get_time_limit(self) -> float:
        """Return time limit in seconds (0 = no limit)."""
        pass


class PracticeMode(GameMode):
    """
    Practice Mode - Learn the dance with reference overlay.
    No scoring pressure, reference skeleton always visible.
    Slower playback speed option available.
    """

    def __init__(self):
        super().__init__(
            name="Practice",
            description="Learn the dance moves at your own pace",
        )
        self.playback_speed = 1.0

    def get_scoring_config(self) -> dict:
        return {
            "enabled": True,
            "show_grade": True,
            "combo_enabled": False,
            "lenient_thresholds": True,
        }

    def should_show_reference(self) -> bool:
        return True

    def get_time_limit(self) -> float:
        return 0  # No time limit


class ChallengeMode(GameMode):
    """
    Challenge Mode - Full scoring with combos.
    Reference shown briefly then fades. Strict grading.
    """

    def __init__(self):
        super().__init__(
            name="Challenge",
            description="Test your moves! Score as high as you can",
        )

    def get_scoring_config(self) -> dict:
        return {
            "enabled": True,
            "show_grade": True,
            "combo_enabled": True,
            "lenient_thresholds": False,
        }

    def should_show_reference(self) -> bool:
        return False

    def get_time_limit(self) -> float:
        return 180  # 3 minutes max


class FreestyleMode(GameMode):
    """
    Freestyle Mode - Free dance with motion capture.
    Records the dance for later playback. No reference, no scoring.
    """

    def __init__(self):
        super().__init__(
            name="Freestyle",
            description="Dance freely and record your moves",
        )

    def get_scoring_config(self) -> dict:
        return {
            "enabled": False,
            "show_grade": False,
            "combo_enabled": False,
            "lenient_thresholds": False,
        }

    def should_show_reference(self) -> bool:
        return False

    def get_time_limit(self) -> float:
        return 300  # 5 minutes max


AVAILABLE_MODES = {
    "practice": PracticeMode,
    "challenge": ChallengeMode,
    "freestyle": FreestyleMode,
}
