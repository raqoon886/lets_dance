"""Asynchronous camera and pose processing module."""

import threading
import cv2

class AsyncCameraPose:
    """Reads camera frames and processes pose detection in a background thread."""

    def __init__(self, camera_idx=0, width=640, height=480, pose_detector=None):
        self._camera = cv2.VideoCapture(camera_idx)
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self._pose_detector = pose_detector

        self._latest_frame = None
        self._latest_landmarks = None
        self._latest_detected = False

        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        """Start the background processing thread."""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._update, daemon=True, name="AsyncCameraPose")
            self._thread.start()

    def _update(self):
        """Continuous background loop for capturing and inferencing."""
        while self._running:
            ret, frame = self._camera.read()
            if not ret or frame is None:
                continue

            # (거울 모드) 사용자가 자연스럽게 보이도록 좌우 반전
            frame = cv2.flip(frame, 1)

            landmarks = None
            detected = False

            if self._pose_detector is not None:
                pose_res = self._pose_detector.detect(frame)
                landmarks = pose_res.get("landmarks")
                detected = pose_res.get("detected", False)

            with self._lock:
                self._latest_frame = frame
                # NumPy 배열이므로 얕은 복사 방지를 위해 copy 가능하나, 
                # detect()에서 매번 새로운 배열을 반환하므로 바로 덮어써도 안전합니다.
                self._latest_landmarks = landmarks
                self._latest_detected = detected

    def read(self):
        """
        Fetch the latest synchronized frame and pose data without blocking.
        Returns:
            (ret, frame, landmarks, detected)
        """
        with self._lock:
            if self._latest_frame is None:
                return False, None, None, False
            # 화면에 렌더링될 메인 스레드를 위해 frame 복제 (동시 접근 충돌 방지)
            return True, self._latest_frame.copy(), self._latest_landmarks, self._latest_detected

    def stop(self):
        """Gracefully stop the thread and release resources."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._camera.release()
        if self._pose_detector is not None:
            self._pose_detector.release()

