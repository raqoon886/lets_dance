"""
Gameplay Screen - Main dance gameplay view with camera feed, scoring, and overlays.

Layout:
┌─────────────────────────────────────────────┐
│ Score: 12450   ★32 combo   ♪ Song Title     │
│┌───────────────────────────────┐ ┌────────┐ │
││                               │ │ Grade  │ │
││     Camera Feed               │ │PERFECT!│ │
││     + Skeleton Overlay        │ │        │ │
││     + Reference Ghost         │ │ +150   │ │
││                               │ │        │ │
│└───────────────────────────────┘ └────────┘ │
│ ╔═══════════════════════════════════════╗    │
│ ║  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░  ║    │
│ ╚═══════════════════════════════════════╝    │
│ 1:23 / 2:00                       [PAUSE]   │
└─────────────────────────────────────────────┘
"""

import numpy as np


class GameplayScreen:
    """
    Main gameplay screen that composites camera feed, skeleton overlays,
    score display, and progress bar into the game view.
    """

    def __init__(self, app):
        self.app = app
        self._camera_rect = None
        self._score_rect = None
        self._progress_rect = None
        self._feedback_opacity = 0

    def setup(self, display_size: tuple):
        """Calculate layout regions based on display size."""
        w, h = display_size
        # Camera feed takes ~75% width, full height minus header/footer
        self._camera_rect = (20, 60, int(w * 0.72), h - 140)
        # Score/feedback panel on the right
        self._score_rect = (int(w * 0.75), 60, int(w * 0.23), h - 140)
        # Progress bar at bottom
        self._progress_rect = (20, h - 60, w - 40, 30)

    def render(self, display, game_state: dict):
        """
        Render the full gameplay view.

        Args:
            display: PyGame display surface
            game_state: Dict with:
                - 'frame': current camera frame (np.ndarray)
                - 'landmarks': detected pose landmarks
                - 'reference_landmarks': reference pose for overlay
                - 'score': current score data
                - 'feedback': current feedback data
                - 'progress': song progress (0-1)
                - 'elapsed': elapsed time string
                - 'duration': total duration string
                - 'song_title': current song name
                - 'combo': current combo count
        """
        # TODO:
        # 1. Draw dark background
        # 2. Draw header bar (score, combo, song title)
        # 3. Convert camera frame to pygame surface and blit
        # 4. Draw skeleton overlay on camera feed
        # 5. Draw reference ghost skeleton (semi-transparent)
        # 6. Draw feedback panel (grade popup, points animation)
        # 7. Draw progress bar
        # 8. Draw time and pause button
        pass

    def _render_camera_feed(self, display, frame: np.ndarray):
        """Convert OpenCV frame to PyGame surface and draw."""
        # TODO: cv2 BGR -> RGB -> pygame.surfarray -> scale to camera_rect
        pass

    def _render_header(self, display, score: float, combo: int, title: str):
        """Draw the top status bar."""
        # TODO: Score on left, combo in center, song title on right
        pass

    def _render_feedback(self, display, feedback: dict):
        """Draw grade popup animation (PERFECT! etc.)."""
        # TODO: Animated text with color and scale effect
        pass

    def _render_progress_bar(self, display, progress: float,
                              elapsed: str, duration: str):
        """Draw song progress bar with time labels."""
        # TODO: Gradient filled bar, time text on each side
        pass

    def handle_input(self, events: list) -> str:
        """Handle gameplay input (pause, quit)."""
        # TODO: ESC to pause, SPACE to pause
        return None
