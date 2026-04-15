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
        # Extract target joints (x, y, z only)
        selected = landmarks[self.target_joints, :3].copy()

        # Hip center (left_hip=index 6, right_hip=index 7 in target_joints)
        # target_joints = [11,12,13,14,15,16,23,24,25,26,27,28]
        # index 6 -> joint 23 (left hip), index 7 -> joint 24 (right hip)
        hip_center = (selected[6] + selected[7]) / 2.0
        selected -= hip_center

        # Scale by torso length (hip center to shoulder center)
        shoulder_center = (selected[0] + selected[1]) / 2.0
        torso_length = np.linalg.norm(shoulder_center)
        if torso_length > 1e-6:
            selected /= torso_length

        return selected.astype(np.float32)

    def compute_joint_angles(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute angles between connected joints for pose comparison.

        Args:
            landmarks: Normalized landmarks (N_joints, 3)

        Returns:
            Array of joint angles in radians
        """
        angles = np.zeros(len(self.target_joints), dtype=np.float32)
        # Compute angle at each joint that has two connected neighbors
        adjacency = {
            0: [1, 2],    # left_shoulder -> right_shoulder, left_elbow
            1: [0, 3],    # right_shoulder -> left_shoulder, right_elbow
            2: [0, 4],    # left_elbow -> left_shoulder, left_wrist
            3: [1, 5],    # right_elbow -> right_shoulder, right_wrist
            6: [0, 8],    # left_hip -> left_shoulder, left_knee
            7: [1, 9],    # right_hip -> right_shoulder, right_knee
            8: [6, 10],   # left_knee -> left_hip, left_ankle
            9: [7, 11],   # right_knee -> right_hip, right_ankle
        }
        for joint_idx, (n1, n2) in adjacency.items():
            v1 = landmarks[n1] - landmarks[joint_idx]
            v2 = landmarks[n2] - landmarks[joint_idx]
            cos_angle = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
            )
            angles[joint_idx] = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return angles

    def smooth(self, landmarks_buffer: list, window_size: int = 5) -> np.ndarray:
        """
        Apply temporal smoothing to reduce jitter using moving average.

        Args:
            landmarks_buffer: List of recent landmark arrays
            window_size: Number of frames to average

        Returns:
            Smoothed landmarks for current frame
        """
        if not landmarks_buffer:
            return np.zeros((len(self.target_joints), 3), dtype=np.float32)
        recent = landmarks_buffer[-window_size:]
        return np.mean(recent, axis=0).astype(np.float32)

    def to_skeleton_edges(self, landmarks: np.ndarray) -> list:
        """
        Convert landmarks into edge list for graph construction.

        Args:
            landmarks: Landmark positions (N_joints, 3)

        Returns:
            List of (src_idx, dst_idx) tuples for skeleton graph
        """
        # Map SKELETON_CONNECTIONS (MediaPipe indices) to target_joints indices
        joint_to_idx = {j: i for i, j in enumerate(self.target_joints)}
        edges = []
        for src, dst in SKELETON_CONNECTIONS:
            if src in joint_to_idx and dst in joint_to_idx:
                edges.append((joint_to_idx[src], joint_to_idx[dst]))
        return edges
