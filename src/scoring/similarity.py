"""
Similarity Calculator - Computes similarity between dance embeddings.
Supports cosine similarity, Euclidean distance, and DTW.
"""

import numpy as np


class SimilarityCalculator:
    """Computes similarity scores between user and reference dance embeddings."""

    def __init__(self, metric="cosine"):
        self.metric = metric
        self._metric_fn = {
            "cosine": self._cosine_similarity,
            "euclidean": self._euclidean_similarity,
            "dtw": self._dtw_similarity,
        }.get(metric, self._cosine_similarity)

    def compute(self, user_embedding: np.ndarray,
                reference_embedding: np.ndarray) -> float:
        """
        Compute similarity between two embeddings.

        Args:
            user_embedding: User's dance embedding vector
            reference_embedding: Reference dance embedding vector

        Returns:
            Similarity score in range [0.0, 1.0]
        """
        return self._metric_fn(user_embedding, reference_embedding)

    def compute_sequence(self, user_embeddings: list,
                         reference_embeddings: list) -> list:
        """
        Compute frame-by-frame similarity for a full dance sequence.

        Args:
            user_embeddings: List of user embedding vectors
            reference_embeddings: List of reference embedding vectors

        Returns:
            List of similarity scores
        """
        # TODO: Align sequences and compute per-window similarity
        min_len = min(len(user_embeddings), len(reference_embeddings))
        scores = []
        for i in range(min_len):
            score = self.compute(user_embeddings[i], reference_embeddings[i])
            scores.append(score)
        return scores

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        # TODO: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return 0.0

    @staticmethod
    def _euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Convert Euclidean distance to similarity score."""
        # TODO: 1 / (1 + np.linalg.norm(a - b))
        return 0.0

    @staticmethod
    def _dtw_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Dynamic Time Warping based similarity."""
        # TODO: Use scipy or fastdtw for DTW computation
        return 0.0
