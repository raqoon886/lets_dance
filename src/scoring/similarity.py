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
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Convert Euclidean distance to similarity score."""
        return float(1.0 / (1.0 + np.linalg.norm(a - b)))

    @staticmethod
    def _dtw_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Dynamic Time Warping based similarity (simplified for 1D vectors)."""
        try:
            from scipy.spatial.distance import euclidean
            n, m = len(a), len(b)
            dtw_matrix = np.full((n + 1, m + 1), np.inf)
            dtw_matrix[0, 0] = 0.0
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = abs(float(a[i - 1]) - float(b[j - 1]))
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i - 1, j],
                        dtw_matrix[i, j - 1],
                        dtw_matrix[i - 1, j - 1],
                    )
            distance = dtw_matrix[n, m]
            return float(1.0 / (1.0 + distance))
        except ImportError:
            return 0.0
