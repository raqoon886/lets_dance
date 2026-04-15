"""
Skeleton Graph Builder - Constructs graph representation from pose landmarks.
Nodes = joints, Edges = bone connections, Features = positions + angles.
"""

import numpy as np


class SkeletonGraphBuilder:
    """Builds a graph data structure from skeleton landmarks for GNN input."""

    # Adjacency for 12 dance-relevant joints (indexed 0-11)
    # Maps to MediaPipe indices: [11,12,13,14,15,16,23,24,25,26,27,28]
    ADJACENCY = [
        (0, 1),   # left_shoulder - right_shoulder
        (0, 2),   # left_shoulder - left_elbow
        (2, 4),   # left_elbow - left_wrist
        (1, 3),   # right_shoulder - right_elbow
        (3, 5),   # right_elbow - right_wrist
        (0, 6),   # left_shoulder - left_hip
        (1, 7),   # right_shoulder - right_hip
        (6, 7),   # left_hip - right_hip
        (6, 8),   # left_hip - left_knee
        (8, 10),  # left_knee - left_ankle
        (7, 9),   # right_hip - right_knee
        (9, 11),  # right_knee - right_ankle
    ]

    def __init__(self, num_joints=12):
        self.num_joints = num_joints
        self._edge_index = self._build_edge_index()

    def _build_edge_index(self) -> np.ndarray:
        """
        Create bidirectional edge index array from adjacency list.

        Returns:
            Edge index array of shape (2, num_edges * 2)
        """
        # TODO: Convert ADJACENCY to bidirectional edge index for PyG
        edges = self.ADJACENCY + [(dst, src) for src, dst in self.ADJACENCY]
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        return np.array([src, dst], dtype=np.int64)

    def build_frame_graph(self, landmarks: np.ndarray) -> dict:
        """
        Build a single-frame graph from normalized landmarks.

        Args:
            landmarks: Normalized joint positions (num_joints, 3)

        Returns:
            Dict with 'node_features' (num_joints, feat_dim) and
            'edge_index' (2, num_edges)
        """
        # TODO: Combine position + velocity features as node features
        node_features = np.zeros((self.num_joints, 3), dtype=np.float32)
        return {
            "node_features": node_features,
            "edge_index": self._edge_index,
        }

    def build_sequence_graph(self, landmark_sequence: list) -> dict:
        """
        Build a spatio-temporal graph from a sequence of frames.
        Adds temporal edges connecting same joint across consecutive frames.

        Args:
            landmark_sequence: List of landmark arrays, each (num_joints, 3)

        Returns:
            Dict with 'node_features', 'edge_index', 'temporal_edge_index'
        """
        # TODO: Stack spatial graphs and add inter-frame temporal edges
        seq_len = len(landmark_sequence)
        total_nodes = seq_len * self.num_joints
        return {
            "node_features": np.zeros((total_nodes, 3), dtype=np.float32),
            "edge_index": self._edge_index,
            "temporal_edge_index": np.zeros((2, 0), dtype=np.int64),
            "sequence_length": seq_len,
        }
