"""
Game Session - Manages a single dance play session from start to finish.
"""

import time
import numpy as np


class GameSession:
    """
    Encapsulates a single dance game session.
    Tracks timing, coordinates reference data playback, and collects results.
    """

    def __init__(self, song_id: str, reference_data: dict, mode: str = "challenge"):
        self.song_id = song_id
        self.reference_data = reference_data
        self.mode = mode
        self.start_time = None
        self.duration = reference_data.get("duration", 60.0)
        self._frame_index = 0
        self._embeddings_user = []
        self._embeddings_ref = []
        self._scores = []
        self.is_active = False

    def start(self):
        """Start the dance session."""
        self.start_time = time.time()
        self.is_active = True
        self._frame_index = 0

    def update(self, user_embedding: np.ndarray) -> dict:
        """
        Process one frame's embedding during gameplay.

        Args:
            user_embedding: User's dance embedding for current window

        Returns:
            Dict with current score data or None if no reference available
        """
        # TODO:
        # 1. Get corresponding reference embedding by frame index
        # 2. Compute similarity
        # 3. Return evaluation result
        self._frame_index += 1
        return {
            "similarity": 0.0,
            "frame_index": self._frame_index,
        }

    def get_reference_landmarks(self) -> np.ndarray:
        """
        Get reference dance landmarks for current frame (for overlay display).

        Returns:
            Reference landmarks for visualization, or None
        """
        # TODO: Index into reference_data by current frame
        return None

    @property
    def elapsed_time(self) -> float:
        """Seconds since session started."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    @property
    def progress(self) -> float:
        """Session progress as fraction (0.0 to 1.0)."""
        if self.duration <= 0:
            return 1.0
        return min(self.elapsed_time / self.duration, 1.0)

    @property
    def is_finished(self) -> bool:
        """Whether the session has reached its end."""
        return self.elapsed_time >= self.duration

    def finish(self) -> dict:
        """
        End the session and return summary data.

        Returns:
            Session summary dict
        """
        self.is_active = False
        return {
            "song_id": self.song_id,
            "mode": self.mode,
            "duration": self.elapsed_time,
            "total_frames": self._frame_index,
            "scores": self._scores,
        }
