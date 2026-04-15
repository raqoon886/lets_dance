"""
Pose Similarity - 임베딩 없이 skeleton 키포인트 좌표만으로 코사인 유사도 계산
"""

import numpy as np


class PoseSimilarity:
    """
    두 포즈(skeleton keypoints) 간의 코사인 유사도를 직접 계산합니다.
    임베딩 모델 없이 동작하는 경량 버전입니다.
    """

    # 비교에 사용할 관절 인덱스 (MediaPipe 기준)
    # 어깨, 팔꿈치, 손목, 골반, 무릎, 발목
    KEY_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    def __init__(self, use_key_joints_only=True, normalize=True):
        """
        Args:
            use_key_joints_only: True면 핵심 관절만 사용, False면 전체 사용
            normalize: True면 골반 중심으로 정규화
        """
        self.use_key_joints_only = use_key_joints_only
        self.normalize = normalize

    def _extract_coords(self, landmarks: np.ndarray) -> np.ndarray:
        """
        landmarks에서 비교에 사용할 좌표를 추출합니다.

        Args:
            landmarks: (33, 4) [x, y, z, visibility] 또는 (17, 3) [y, x, conf]

        Returns:
            1D 벡터 (flatten된 x, y 좌표)
        """
        if self.use_key_joints_only:
            coords = landmarks[self.KEY_JOINTS, :2]  # x, y만
        else:
            coords = landmarks[:, :2]

        if self.normalize:
            coords = self._normalize_pose(coords)

        return coords.flatten()

    def _normalize_pose(self, coords: np.ndarray) -> np.ndarray:
        """
        골반 중심으로 이동 + 어깨 너비 기준으로 스케일 정규화

        Args:
            coords: (N, 2) 좌표

        Returns:
            정규화된 (N, 2) 좌표
        """
        coords = coords.copy()

        # KEY_JOINTS 기준 인덱스에서 골반 찾기
        if self.use_key_joints_only:
            # KEY_JOINTS에서 23(왼골반)=idx6, 24(오른골반)=idx7
            left_hip_idx = 6
            right_hip_idx = 7
            left_shoulder_idx = 0   # 11=idx0
            right_shoulder_idx = 1  # 12=idx1
        else:
            left_hip_idx = 23
            right_hip_idx = 24
            left_shoulder_idx = 11
            right_shoulder_idx = 12

        # 골반 중심으로 이동
        center = (coords[left_hip_idx] + coords[right_hip_idx]) / 2
        coords -= center

        # 어깨 너비로 스케일 정규화
        shoulder_dist = np.linalg.norm(
            coords[left_shoulder_idx] - coords[right_shoulder_idx]
        )
        if shoulder_dist > 1e-6:
            coords /= shoulder_dist

        return coords

    def cosine_similarity(self, pose_a: np.ndarray, pose_b: np.ndarray) -> float:
        """
        두 포즈 간 코사인 유사도 계산

        Args:
            pose_a: (33, 4) landmarks
            pose_b: (33, 4) landmarks

        Returns:
            유사도 (0.0 ~ 1.0)
        """
        vec_a = self._extract_coords(pose_a)
        vec_b = self._extract_coords(pose_b)

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0

        similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)
        # -1~1 범위를 0~1로 매핑
        return float(np.clip((similarity + 1) / 2, 0.0, 1.0))

    def compare_sequence(self, seq_a: list, seq_b: list) -> list:
        """
        두 포즈 시퀀스의 프레임별 유사도 계산

        Args:
            seq_a: 포즈 리스트 [(33,4), ...]
            seq_b: 포즈 리스트 [(33,4), ...]

        Returns:
            프레임별 유사도 리스트
        """
        min_len = min(len(seq_a), len(seq_b))
        return [
            self.cosine_similarity(seq_a[i], seq_b[i])
            for i in range(min_len)
        ]
