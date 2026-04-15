"""
Score Display Widget - Shows current score, combo counter, and grade popups.
"""


class ScoreDisplayWidget:
    """Animated score counter and combo display."""

    def __init__(self, position: tuple):
        """
        Args:
            position: (x, y) top-left position
        """
        self.position = position
        self._displayed_score = 0
        self._target_score = 0
        self._combo = 0
        self._grade_popup = None
        self._popup_timer = 0

    def set_score(self, score: float, combo: int):
        """Update the target score and combo for animated display."""
        self._target_score = score
        self._combo = combo

    def show_grade(self, grade: str, points: float):
        """Trigger a grade popup animation."""
        self._grade_popup = {"grade": grade, "points": points}
        self._popup_timer = 45  # frames

    def update(self):
        """Update animation state (call each frame)."""
        # Smooth score counter animation
        diff = self._target_score - self._displayed_score
        self._displayed_score += diff * 0.1

        if self._popup_timer > 0:
            self._popup_timer -= 1
        else:
            self._grade_popup = None

    def render(self, display):
        """
        Draw the score display.

        Renders:
            - Score counter with animated counting
            - Combo counter (with fire icon at high combos)
            - Grade popup text (PERFECT!, GREAT!, etc.) with fade
        """
        # TODO:
        # 1. Draw score number with formatting (e.g., "12,450")
        # 2. Draw combo counter with multiplier indicator
        # 3. Draw grade popup with scale/fade animation
        pass
