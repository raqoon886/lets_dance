"""Asynchronous camera and pose processing module."""

import threading
import cv2

class AsyncCameraPose:
    """Reads camera frames and processes pose detection in a background thread."""

    def __init__(self, camera_idx=0, width=640, height=480, pose_detector=None):
        self._camera = cv2.VideoCapture(camera_idx)
        
        # 💡 [초저지연 하드웨어 최적화] - v4l-utils 명령어와 동일한 효과
        # 웹캠 압축 포맷을 가장 빠르고 대역폭이 넓은 MJPG로 강제 지정하고 버퍼를 1로 줄입니다.
        # (YUYV 포맷 대비 USB 병목이 해소되어 극단적으로 딜레이가 줄어듦)
        self._camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._camera.set(cv2.CAP_PROP_FPS, 30)
        self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._pose_detector = pose_detector

        self._latest_frame = None
        self._latest_landmarks = None
        self._latest_detected = False

        self._lock = threading.Lock()
        self._raw_frame_lock = threading.Lock()
        self._latest_raw_frame = None
        
        self._running = False
        self._thread = None
        self._capture_thread = None

    def start(self):
        """Start the background processing threads."""
        if not self._running:
            self._running = True
            
            # 카메라 캡처 전용 스레드 (버퍼 딜레이 방지)
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraCapture")
            self._capture_thread.start()
            
            # 포즈 추론 전용 스레드
            self._thread = threading.Thread(target=self._update, daemon=True, name="AsyncCameraPose")
            self._thread.start()

    def _capture_loop(self):
        """Continuously reads from the camera to prevent buffer buildup (Zero-latency trick)."""
        while self._running:
            ret, frame = self._camera.read()
            if ret and frame is not None:
                with self._raw_frame_lock:
                    self._latest_raw_frame = frame

    def _update(self):
        """Continuous background loop for inferencing."""
        import time
        while self._running:
            frame = None
            with self._raw_frame_lock:
                if self._latest_raw_frame is not None:
                    frame = self._latest_raw_frame
                    # 처리할 프레임 가져온 후 소비
                    self._latest_raw_frame = None
            
            if frame is None:
                time.sleep(0.005)
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
            # 임계영역 최소화 및 충돌 방지를 위해 frame 복제
            return True, self._latest_frame.copy(), self._latest_landmarks, self._latest_detected

    def stop(self):
        """Gracefully stop the thread and release resources."""
        self._running = False
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._camera.release()
        if self._pose_detector is not None:
            self._pose_detector.release()

