"""
Embedding Extractor - Pipeline that takes raw frames and produces embeddings.
Orchestrates pose detection -> landmark processing -> graph building -> model inference.
"""

import numpy as np


class EmbeddingExtractor:
    """
    End-to-end pipeline for extracting dance embeddings from video frames.
    Maintains a sliding window buffer of landmarks for sequence-based extraction.
    """

    def __init__(self, pose_detector, landmark_processor, graph_builder,
                 embedding_model, sequence_length=30):
        self.pose_detector = pose_detector
        self.landmark_processor = landmark_processor
        self.graph_builder = graph_builder
        self.embedding_model = embedding_model
        self.sequence_length = sequence_length
        self._landmark_buffer = []

    def reset(self):
        """Clear the landmark buffer (e.g., at start of new song)."""
        self._landmark_buffer.clear()

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame and return embedding if buffer is full.

        Args:
            frame: BGR image from webcam (H, W, 3)

        Returns:
            Dict with:
                - "landmarks": current normalized landmarks or None
                - "embedding": embedding vector if sequence complete, else None
                - "detected": whether pose was detected
        """
        result = {
            "landmarks": None,
            "embedding": None,
            "detected": False,
        }

        # 1. Detect pose
        detection = self.pose_detector.detect(frame)
        result["detected"] = detection["detected"]

        if not detection["detected"]:
            return result

        # 2. Normalize landmarks
        normalized = self.landmark_processor.normalize(detection["landmarks"])
        result["landmarks"] = normalized

        # 3. Append to buffer and smooth
        self._landmark_buffer.append(normalized)
        if len(self._landmark_buffer) > self.sequence_length:
            self._landmark_buffer = self._landmark_buffer[-self.sequence_length:]

        # 4. If buffer full, build sequence graph and extract embedding
        if len(self._landmark_buffer) >= self.sequence_length:
            result["embedding"] = self.extract_from_sequence(self._landmark_buffer)

        return result

    def extract_from_sequence(self, landmark_sequence: list) -> np.ndarray:
        """
        Extract embedding from a complete landmark sequence.

        Args:
            landmark_sequence: List of normalized landmark arrays

        Returns:
            Embedding vector (embedding_dim,)
        """
        # TODO: Build sequence graph and run through model
        graph = self.graph_builder.build_sequence_graph(landmark_sequence)
        embedding = self.embedding_model.forward(graph)
        return embedding

    @property
    def buffer_progress(self) -> float:
        """Return how full the buffer is (0.0 to 1.0)."""
        return min(len(self._landmark_buffer) / self.sequence_length, 1.0)
