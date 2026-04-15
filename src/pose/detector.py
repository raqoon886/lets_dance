"""
MediaPipe Pose Detector - Captures webcam frames and extracts body landmarks.
"""

import numpy as np


class PoseDetector:
    """Wraps MediaPipe Pose to detect body landmarks from webcam frames."""

    def __init__(self, model_complexity=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._pose = None

    def initialize(self):
        """Initialize MediaPipe Pose model."""
        # TODO: Initialize mp.solutions.pose.Pose with config params
        # import mediapipe as mp
        # self._pose = mp.solutions.pose.Pose(
        #     model_complexity=self.model_complexity,
        #     min_detection_confidence=self.min_detection_confidence,
        #     min_tracking_confidence=self.min_tracking_confidence,
        # )
        pass

    def detect(self, frame: np.ndarray) -> dict:
        """
        Detect pose landmarks from a single BGR frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)

        Returns:
            Dictionary with:
                - "landmarks": np.ndarray of shape (33, 4) [x, y, z, visibility]
                - "detected": bool indicating if pose was found
        """
        # TODO: Convert BGR -> RGB, run self._pose.process(), extract landmarks
        return {
            "landmarks": np.zeros((33, 4), dtype=np.float32),
            "detected": False,
        }

    def release(self):
        """Release MediaPipe resources."""
        if self._pose:
            self._pose.close()
            self._pose = None
