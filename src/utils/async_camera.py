"""Asynchronous camera and pose processing module."""

import threading
import cv2

class AsyncCameraPose:
    """Reads camera frames and processes pose detection in a background thread."""

    def __init__(self, camera_idx=0, width=640, height=480, pose_detector=None):
        # 💡 [극단적 지연 시간 소거 1] V4L2 명시적 지정
        # 리눅스/라즈베리파이에서 GStreamer 등 무거운 백엔드가 기본값으로 잡히면서 발생하는
        # 0.5초 ~ 1초 가량의 막대한 파이프라인 지연을 강제로 우회하고 가장 날것의 드라이버를 씁니다.
        self._camera = cv2.VideoCapture(camera_idx, cv2.CAP_V4L2)
        
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
        self._raw_frame_seq = 0          # 캡처 프레임 시퀀스 번호
        self._infer_frame_seq = 0        # 추론 완료 프레임 시퀀스 번호
        
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
                # 캡처 직후 즉시 좌우 반전 처리 (UI 표시 및 추론용 통일)
                frame = cv2.flip(frame, 1)
                with self._raw_frame_lock:
                    self._raw_frame_seq += 1
                    self._latest_raw_frame = frame

    def _update(self):
        """Continuous background loop for inferencing."""
        import time
        last_processed_id = None
        while self._running:
            frame = None
            cur_seq = 0
            with self._raw_frame_lock:
                if self._latest_raw_frame is not None:
                    if id(self._latest_raw_frame) != last_processed_id:
                        frame = self._latest_raw_frame
                        last_processed_id = id(self._latest_raw_frame)
                        cur_seq = self._raw_frame_seq

            
            if frame is None:
                time.sleep(0.005)
                continue

            # (거울 모드 처리는 _capture_loop에서 이미 완료됨)

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
                self._infer_frame_seq = cur_seq

    def read(self):
        """
        Fetch the latest synchronized frame and pose data without blocking.
        Returns:
            (ret, frame, landmarks, detected)
        """
        # 💡 [극단적 지연 시간 소거 2] 강제 복사본 생성(copy) 제거
        # 이미 캡처 스레드에서 매번 새로운 배열이 할당되므로, 포인터만 던져줍니다. (3~5ms 즉시 단축)
        display_frame = None
        display_seq = 0
        with self._raw_frame_lock:
            if self._latest_raw_frame is not None:
                display_frame = self._latest_raw_frame
                display_seq = self._raw_frame_seq

        with self._lock:
            landmarks = self._latest_landmarks
            detected = self._latest_detected
            infer_seq = self._infer_frame_seq
            
            if display_frame is None and self._latest_frame is None:
                return False, None, None, False
                
            if display_frame is None:
                display_frame = self._latest_frame
                display_seq = infer_seq

            lag = display_seq - infer_seq
            if lag > 0:
                print(f"\r[CAM] display=#{display_seq} infer=#{infer_seq} lag={lag}frames", end="")

            return True, display_frame, landmarks, detected

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

