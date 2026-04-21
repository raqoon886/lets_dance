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
        return float(np.clip(similarity, 0.0, 1.0))

    def euclidean_similarity(self, pose_a: np.ndarray, pose_b: np.ndarray) -> float:
        """
        두 포즈 간 관절별 유클리드 거리 기반 유사도

        Args:
            pose_a: (33, 4) landmarks
            pose_b: (33, 4) landmarks

        Returns:
            유사도 (0.0 ~ 1.0)
        """
        if self.use_key_joints_only:
            coords_a = pose_a[self.KEY_JOINTS, :2].copy()
            coords_b = pose_b[self.KEY_JOINTS, :2].copy()
        else:
            coords_a = pose_a[:, :2].copy()
            coords_b = pose_b[:, :2].copy()

        if self.normalize:
            coords_a = self._normalize_pose(coords_a)
            coords_b = self._normalize_pose(coords_b)

        per_joint_dist = np.linalg.norm(coords_a - coords_b, axis=1)
        mean_dist = np.mean(per_joint_dist)
        return float(np.exp(-2.0 * mean_dist))

    def hybrid_similarity(self, pose_a: np.ndarray, pose_b: np.ndarray,
                          cosine_weight=0.3, euclidean_weight=0.7) -> float:
        """
        코사인 + 유클리드 혼합 유사도

        Args:
            pose_a: (33, 4) landmarks
            pose_b: (33, 4) landmarks
            cosine_weight: 코사인 비중 (기본 0.3)
            euclidean_weight: 유클리드 비중 (기본 0.7)

        Returns:
            유사도 (0.0 ~ 1.0)
        """
        cos_sim = self.cosine_similarity(pose_a, pose_b)
        euc_sim = self.euclidean_similarity(pose_a, pose_b)
        return float(np.clip(
            cosine_weight * cos_sim + euclidean_weight * euc_sim, 0.0, 1.0))

    # ── 관절 각도(Angle) 기반 유사도 ──

    # 각도를 계측할 관절 triplet: (A, B, C) → B 지점에서의 A→B→C 사이 각도
    # MediaPipe 33 keypoint 기준 인덱스
    ANGLE_JOINTS = [
        # 어깨: 골반 → 어깨 → 팔꿈치 (팔을 얼마나 올렸는지)
        (23, 11, 13),   # 왼쪽 어깨
        (24, 12, 14),   # 오른쪽 어깨
        # 팔꿈치: 어깨 → 팔꿈치 → 손목 (팔 굽힘)
        (11, 13, 15),   # 왼쪽 팔꿈치
        (12, 14, 16),   # 오른쪽 팔꿈치
        # 골반: 어깨 → 골반 → 무릎 (다리 들기)
        (11, 23, 25),   # 왼쪽 골반
        (12, 24, 26),   # 오른쪽 골반
        # 무릎: 골반 → 무릎 → 발목 (무릎 굽힘)
        (23, 25, 27),   # 왼쪽 무릎
        (24, 26, 28),   # 오른쪽 무릎
    ]

    @staticmethod
    def _calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        세 점 A, B, C에서 B 지점의 각도(라디안)를 계산.
        BA 벡터와 BC 벡터 사이의 각도.
        """
        ba = a - b
        bc = c - b
        cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return float(np.arccos(np.clip(cos_val, -1.0, 1.0)))

    def _extract_angles(self, landmarks: np.ndarray) -> np.ndarray:
        """
        포즈에서 8개 관절 각도를 추출.

        Args:
            landmarks: (33, 4) [x, y, z, visibility]

        Returns:
            (8,) 각도 배열 (라디안, 0~π)
        """
        coords = landmarks[:, :2]  # x, y만 사용
        angles = np.array([
            self._calc_angle(coords[a], coords[b], coords[c])
            for a, b, c in self.ANGLE_JOINTS
        ])
        return angles

    def angle_similarity(self, pose_a: np.ndarray, pose_b: np.ndarray) -> float:
        """
        관절 각도 기반 유사도. 각 관절의 각도 차이 평균으로 계산.
        위치/크기에 완전히 독립적.

        Args:
            pose_a: (33, 4) landmarks
            pose_b: (33, 4) landmarks

        Returns:
            유사도 (0.0 ~ 1.0)
        """
        angles_a = self._extract_angles(pose_a)
        angles_b = self._extract_angles(pose_b)

        # 각도 차이 (라디안). 최대 π (180도)
        diff = np.abs(angles_a - angles_b)
        
        # 단순히 평균을 내면 안 움직인 관절(오차0)이 섞여서 큰 오차가 희석됨. (가만히 서있기 꼼수 방지)
        # 제곱 평균(MSE)을 사용하여 틀린 관절이 하나라도 크게 어긋나면 치명적인 감점을 부여함.
        mean_sq_diff = np.mean(diff ** 2)

        # 지수 감쇠(Gaussian): 사람 몸의 자연스러운 전체적 오차(약 15도)는 허용하고,
        # 하나라도 45도 이상 기괴하게 꺾여 있다면 확 떨어지도록 설계.
        similarity = float(np.exp(-3.0 * mean_sq_diff))
        return float(np.clip(similarity, 0.0, 1.0))

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
