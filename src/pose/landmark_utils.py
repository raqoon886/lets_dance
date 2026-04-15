"""
Landmark processing utilities - normalization, filtering, and conversion.
"""

import numpy as np


# MediaPipe Pose landmark indices for skeleton connections
SKELETON_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Upper body
    (11, 23), (12, 24), (23, 24),                        # Torso
    (23, 25), (25, 27), (24, 26), (26, 28),              # Lower body
    (15, 17), (15, 19), (16, 18), (16, 20),              # Hands
    (27, 29), (27, 31), (28, 30), (28, 32),              # Feet
]

# Key joint indices for dance evaluation
DANCE_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


class LandmarkProcessor:
    """Processes raw MediaPipe landmarks into normalized, usable formats."""

    def __init__(self, target_joints=None):
        self.target_joints = target_joints or DANCE_JOINTS

    def normalize(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks to be translation/scale invariant.
        Centers on hip midpoint and scales by torso length.

        Args:
            landmarks: Raw landmarks array (33, 4)

        Returns:
            Normalized landmarks (N_joints, 3) - x, y, z only
        """
        # TODO: Extract target joints, compute hip center, normalize by torso length
        n_joints = len(self.target_joints)
        return np.zeros((n_joints, 3), dtype=np.float32)

    def compute_joint_angles(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute angles between connected joints for pose comparison.

        Args:
            landmarks: Normalized landmarks (N_joints, 3)

        Returns:
            Array of joint angles in radians
        """
        # TODO: Calculate angles at each joint using connected limb vectors
        return np.zeros(len(self.target_joints), dtype=np.float32)

    def smooth(self, landmarks_buffer: list, window_size: int = 5) -> np.ndarray:
        """
        Apply temporal smoothing to reduce jitter using moving average.

        Args:
            landmarks_buffer: List of recent landmark arrays
            window_size: Number of frames to average

        Returns:
            Smoothed landmarks for current frame
        """
        # TODO: Weighted moving average over the buffer
        if not landmarks_buffer:
            return np.zeros((len(self.target_joints), 3), dtype=np.float32)
        return landmarks_buffer[-1]

    def to_skeleton_edges(self, landmarks: np.ndarray) -> list:
        """
        Convert landmarks into edge list for graph construction.

        Args:
            landmarks: Landmark positions (N_joints, 3)

        Returns:
            List of (src_idx, dst_idx) tuples for skeleton graph
        """
        # TODO: Map SKELETON_CONNECTIONS to target_joints indices
        edges = []
        return edges
