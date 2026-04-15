"""
Similarity Calculator - Computes similarity between dance embeddings.
Supports cosine similarity, Euclidean distance, DTW, and Sliding Window Cosine.
"""

import numpy as np
from collections import deque


class SimilarityCalculator:
    """Computes similarity scores between us er and reference dance embeddings."""

    def __init__(self, metric="cosine", window_size=15):
        """
        Args:
            metric: "cosine", "euclidean", "dtw", "sliding_window"
            window_size: sliding_window 모드에서 사용할 윈도우 크기 (프레임 수)
        """
        self.metric = metric
        self.window_size = window_size
        self._metric_fn = {
            "cosine": self._cosine_similarity,
            "euclidean": self._euclidean_similarity,
            "dtw": self._dtw_similarity,
            "sliding_window": self._sliding_window_cosine,
        }.get(metric, self._cosine_similarity)

        # Sliding Window용 버퍼
        self._user_buffer = deque(maxlen=window_size)
        self._ref_buffer = deque(maxlen=window_size)

    def reset(self):
        """슬라이딩 윈도우 버퍼 초기화 (새 곡 시작 시 호출)."""
        self._user_buffer.clear()
        self._ref_buffer.clear()

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
        if self.metric == "sliding_window":
            # 슬라이딩 윈도우: 버퍼에 쌓고 평균으로 비교
            self._user_buffer.append(user_embedding)
            self._ref_buffer.append(reference_embedding)
            return self._sliding_window_cosine()
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

    def _sliding_window_cosine(self) -> float:
        """
        Sliding Window Cosine Similarity.
        최근 k개 프레임의 임베딩 평균끼리 코사인 유사도를 계산합니다.

        수식:
            score = cos(mean(E_user[t-k:t]), mean(E_ref[t-k:t]))

        - 순간적인 관절 떨림(Jitter)을 평활화
        - 실시간 게이지/콤보 바에 적합
        """
        if len(self._user_buffer) == 0:
            return 0.0

        user_mean = np.mean(self._user_buffer, axis=0)
        ref_mean = np.mean(self._ref_buffer, axis=0)

        return self._cosine_similarity(user_mean, ref_mean)
