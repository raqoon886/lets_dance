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
        try:
            import mediapipe as mp
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        except ImportError:
            print("[WARN] mediapipe not installed, using dummy detector")
            self._pose = None

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
        landmarks = np.zeros((33, 4), dtype=np.float32)
        detected = False

        if self._pose is not None:
            import cv2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._pose.process(rgb)
            if results.pose_landmarks:
                detected = True
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    landmarks[i] = [lm.x, lm.y, lm.z, lm.visibility]

        return {
            "landmarks": landmarks,
            "detected": detected,
        }

    def release(self):
        """Release MediaPipe resources."""
        if self._pose:
            self._pose.close()
            self._pose = None
