"""
Feedback Generator - Produces real-time visual/audio feedback cues
based on scoring results.
"""


class FeedbackGenerator:
    """
    Generates feedback data (colors, text, sound triggers) based on
    the current dance evaluation result.
    """

    FEEDBACK_MAP = {
        "Perfect": {
            "color": (255, 215, 0),     # Gold
            "text": "PERFECT!",
            "sound": "perfect.wav",
            "particle_count": 20,
        },
        "Great": {
            "color": (0, 255, 128),     # Green
            "text": "GREAT!",
            "sound": "great.wav",
            "particle_count": 12,
        },
        "Good": {
            "color": (0, 200, 255),     # Cyan
            "text": "GOOD",
            "sound": "good.wav",
            "particle_count": 6,
        },
        "OK": {
            "color": (200, 200, 200),   # Gray
            "text": "OK",
            "sound": "ok.wav",
            "particle_count": 0,
        },
        "Miss": {
            "color": (128, 128, 128),   # Dark gray
            "text": "MISS",
            "sound": "miss.wav",
            "particle_count": 0,
        },
    }

    def __init__(self):
        self._current_feedback = None
        self._feedback_timer = 0
        self._display_duration = 30  # frames

    def generate(self, evaluation: dict) -> dict:
        """
        Generate feedback data from an evaluation result.

        Args:
            evaluation: Dict from DanceScorer.evaluate()

        Returns:
            Feedback data dict with color, text, sound, particles
        """
        grade = evaluation.get("grade", "Miss")
        feedback = dict(self.FEEDBACK_MAP.get(grade, self.FEEDBACK_MAP["Miss"]))
        feedback["combo"] = evaluation.get("combo", 0)
        feedback["points"] = evaluation.get("points", 0)

        self._current_feedback = feedback
        self._feedback_timer = self._display_duration
        return feedback

    def update(self) -> dict:
        """
        Update feedback state (call each frame). Returns current feedback
        or None if display timer expired.
        """
        if self._feedback_timer > 0:
            self._feedback_timer -= 1
            return self._current_feedback
        return None

    @property
    def is_active(self) -> bool:
        return self._feedback_timer > 0
