"""
Countdown Widget - Animated countdown timer before dance starts.

Visual: Large centered numbers (3... 2... 1... GO!) with scale/fade animation.
"""


class CountdownWidget:
    """Full-screen countdown overlay before gameplay starts."""

    def __init__(self, duration_seconds=3):
        self.duration = duration_seconds
        self._current_count = duration_seconds
        self._frame_counter = 0
        self._fps = 30
        self._is_active = False
        self._scale = 1.0

    def start(self):
        """Start the countdown."""
        self._current_count = self.duration
        self._frame_counter = 0
        self._is_active = True

    def update(self) -> bool:
        """
        Update countdown state. Call each frame.

        Returns:
            True if countdown is still active, False when finished
        """
        if not self._is_active:
            return False

        self._frame_counter += 1
        elapsed_in_second = self._frame_counter % self._fps

        # Scale animation: start big, shrink during each second
        self._scale = 1.0 + (1.0 - elapsed_in_second / self._fps) * 0.5

        if self._frame_counter >= self._fps:
            self._frame_counter = 0
            self._current_count -= 1
            if self._current_count < 0:
                self._is_active = False
                return False

        return True

    def render(self, display):
        """
        Draw the countdown number centered on screen.

        Renders:
            - Large number (3, 2, 1) or "GO!" with scale animation
            - Semi-transparent dark overlay background
            - Pulse ring animation around the number
        """
        # TODO:
        # 1. Draw semi-transparent overlay
        # 2. Compute scaled font size based on self._scale
        # 3. Draw number or "GO!" text centered
        # 4. Draw expanding ring animation
        pass

    @property
    def display_text(self) -> str:
        if self._current_count > 0:
            return str(self._current_count)
        return "GO!"

    @property
    def is_active(self) -> bool:
        return self._is_active
