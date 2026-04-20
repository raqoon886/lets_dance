"""
Result Screen - Shows final score, grade breakdown, and stats after a dance.

Layout:
┌─────────────────────────────────────┐
│           DANCE COMPLETE!           │
│                                     │
│        ╔═══════════════╗            │
│        ║  TOTAL SCORE  ║            │
│        ║    12,450     ║            │
│        ║  Grade: A+    ║            │
│        ╚═══════════════╝            │
│                                     │
│   Perfect: 42   Great: 28          │
│   Good: 15      OK: 8   Miss: 3   │
│                                     │
│   Max Combo: 32    Avg: 82.5       │
│                                     │
│   [ ↻ RETRY ]  [ 🏠 MENU ]        │
│   [ 📊 DETAILS ]                   │
└─────────────────────────────────────┘
"""


class ResultScreen:
    """
    Displays the final results after a dance session.
    Shows score breakdown, grade distribution chart, and action buttons.
    """

    def __init__(self, app):
        self.app = app
        self._result_data = None
        self._animation_progress = 0.0
        self._grade_letter = ""

    def set_result(self, result: dict):
        """
        Load result data for display.

        Args:
            result: Dict from DanceScorer.get_final_result()
        """
        self._result_data = result
        self._animation_progress = 0.0
        self._grade_letter = self._compute_grade_letter(
            result.get("average_score", 0)
        )

    def render(self, display, game_state: dict):
        """
        Render the result screen with animated score reveal.

        Args:
            display: PyGame display surface
            game_state: Current state data
        """
        # TODO:
        # 1. Draw "DANCE COMPLETE!" header with confetti animation
        # 2. Animate score counter from 0 to final score
        # 3. Display grade letter (S/A/B/C/D) with glow
        # 4. Draw hit distribution bar chart
        # 5. Draw stats (max combo, average, total moves)
        # 6. Draw action buttons (retry, menu, details)
        self._animation_progress = min(self._animation_progress + 0.02, 1.0)
        pass

    def handle_input(self, events: list) -> str:
        """
        Handle result screen input.

        Returns:
            'retry', 'menu', 'details', or None
        """
        import pygame

        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.app.play_sfx("click")
                    return "retry"
                elif event.key in (pygame.K_ESCAPE, pygame.K_m):
                    self.app.play_sfx("click")
                    return "menu"
                elif event.key == pygame.K_d:
                    self.app.play_sfx("click")
                    return "details"
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # TODO: check event.pos against button rects when defined
                self.app.play_sfx("click")

        return None

    @staticmethod
    def _compute_grade_letter(average_score: float) -> str:
        """Convert average score to letter grade."""
        if average_score >= 95:
            return "S"
        elif average_score >= 85:
            return "A"
        elif average_score >= 70:
            return "B"
        elif average_score >= 55:
            return "C"
        return "D"
