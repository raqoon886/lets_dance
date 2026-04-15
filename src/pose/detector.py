"""
Pose Detector - MoveNet 또는 MediaPipe 백엔드를 선택하여 포즈를 추출합니다.
두 백엔드 모두 (33, 4) [x, y, z, visibility] 형식으로 통일된 출력을 반환합니다.
"""

import numpy as np


# MoveNet 17 keypoints -> MediaPipe 33 landmarks 매핑
MOVENET_TO_MP = {
    0: 0,    # nose
    5: 11,   # left_shoulder
    6: 12,   # right_shoulder
    7: 13,   # left_elbow
    8: 14,   # right_elbow
    9: 15,   # left_wrist
    10: 16,  # right_wrist
    11: 23,  # left_hip
    12: 24,  # right_hip
    13: 25,  # left_knee
    14: 26,  # right_knee
    15: 27,  # left_ankle
    16: 28,  # right_ankle
}


class PoseDetector:
    """MoveNet 또는 MediaPipe 백엔드로 포즈를 감지합니다."""

    NUM_MP_LANDMARKS = 33
    BACKENDS = ("movenet", "mediapipe")

    def __init__(self, backend="mediapipe", model_complexity=1,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        if backend not in self.BACKENDS:
            raise ValueError(f"backend은 {self.BACKENDS} 중 하나여야 합니다: {backend}")
        self.backend = backend
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._movenet = None
        self._mp_pose = None

    def initialize(self):
        """선택된 백엔드 모델을 로드합니다."""
        if self.backend == "movenet":
            self._init_movenet()
        else:
            self._init_mediapipe()

    def _init_movenet(self):
        import tensorflow_hub as hub
        module = hub.load(
            'https://tfhub.dev/google/movenet/singlepose/lightning/4'
        )
        self._movenet = module.signatures['serving_default']

    def _init_mediapipe(self):
        import mediapipe as mp
        self._mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def detect(self, frame: np.ndarray) -> dict:
        """
        한 프레임에서 포즈 랜드마크를 감지합니다.

        Args:
            frame: BGR 이미지 (H, W, 3)

        Returns:
            dict:
                - "landmarks": (33, 4) [x, y, z, visibility]
                - "detected": bool
        """
        if self.backend == "movenet":
            return self._detect_movenet(frame)
        else:
            return self._detect_mediapipe(frame)

    def _detect_movenet(self, frame: np.ndarray) -> dict:
        import tensorflow as tf

        if self._movenet is None:
            return {"landmarks": np.zeros((self.NUM_MP_LANDMARKS, 4), dtype=np.float32), "detected": False}

        input_image = tf.cast(
            tf.image.resize_with_pad(tf.expand_dims(frame, 0), 192, 192),
            dtype=tf.int32
        )
        outputs = self._movenet(input=input_image)
        keypoints = outputs['output_0'].numpy()[0][0]  # (17, 3) [y, x, conf]

        detected = int(np.sum(keypoints[:, 2] > self.min_detection_confidence)) >= 5

        landmarks = np.zeros((self.NUM_MP_LANDMARKS, 4), dtype=np.float32)
        for mn_idx, mp_idx in MOVENET_TO_MP.items():
            y, x, conf = keypoints[mn_idx]
            landmarks[mp_idx] = [x, y, 0.0, conf]

        return {"landmarks": landmarks, "detected": detected}

    def _detect_mediapipe(self, frame: np.ndarray) -> dict:
        import cv2

        landmarks = np.zeros((self.NUM_MP_LANDMARKS, 4), dtype=np.float32)
        detected = False

        if self._mp_pose is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._mp_pose.process(rgb)
            if results.pose_landmarks:
                detected = True
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    landmarks[i] = [lm.x, lm.y, lm.z, lm.visibility]

        return {"landmarks": landmarks, "detected": detected}

    def release(self):
        """리소스 해제."""
        self._movenet = None
        if self._mp_pose is not None:
            self._mp_pose.close()
            self._mp_pose = None
