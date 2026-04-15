"""
Progress Bar Widget - Shows song playback progress with gradient fill.
"""


class ProgressBarWidget:
    """Horizontal progress bar for tracking song/dance progress."""

    def __init__(self, rect: tuple):
        """
        Args:
            rect: (x, y, width, height) display region
        """
        self.rect = rect
        self._progress = 0.0
        self._elapsed_text = "0:00"
        self._duration_text = "0:00"

    def set_progress(self, progress: float, elapsed: str, duration: str):
        """
        Update progress bar state.

        Args:
            progress: Progress fraction (0.0 to 1.0)
            elapsed: Elapsed time string (e.g., "1:23")
            duration: Total duration string (e.g., "2:00")
        """
        self._progress = max(0.0, min(1.0, progress))
        self._elapsed_text = elapsed
        self._duration_text = duration

    def render(self, display):
        """
        Draw the progress bar with gradient fill and time labels.

        Visual:
            0:45  ▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░  2:00
        """
        # TODO:
        # 1. Draw background track (dark rounded rect)
        # 2. Draw filled portion with gradient (green -> yellow -> red)
        # 3. Draw elapsed time on left
        # 4. Draw total duration on right
        # 5. Draw playhead marker
        pass
