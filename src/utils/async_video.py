import threading
import time
import cv2
import numpy as np

class AsyncVideoPlayer:
    """
    백그라운드 스레드에서 레퍼런스 비디오를 디코딩하고 리사이즈하는 클래스.
    메인 루프의 UI 렌더링 블로킹(Frame Drop)을 방지하기 위해 사용됩니다.
    """
    def __init__(self, video_path: str, target_size: tuple = None, 
                 audio_latency_offset: float = 0.040, time_func=None):
        self.video_path = video_path
        self.target_size = target_size
        self.audio_latency_offset = audio_latency_offset
        self.time_func = time_func  # 현재 오디오 시간(초)을 반환하는 콜백 함수
        
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        self._latest_frame_rgb = None
        self._current_frame_idx = 0
        
        # 내부 fallback 타이머용
        self._fallback_start_time = 0.0
        self._use_fallback_timer = False

    def start(self):
        if self.running or not self.cap.isOpened():
            return
        self.running = True
        self._use_fallback_timer = (self.time_func is None)
        if self._use_fallback_timer:
            self._fallback_start_time = time.time()
            
        self.thread = threading.Thread(target=self._update, daemon=True, name="AsyncVideoPlayer")
        self.thread.start()

    def _get_current_time(self):
        if self._use_fallback_timer:
            return time.time() - self._fallback_start_time
        try:
            t = self.time_func()
            if t is not None and t >= 0:
                return t
            return 0.0
        except Exception:
            return 0.0

    def _update(self):
        while self.running:
            raw_time_sec = self._get_current_time()
            
            # 오디오 레이턴시 보정을 거친 최종 시스템 시간
            elapsed_sys = raw_time_sec - self.audio_latency_offset
            if elapsed_sys < 0:
                elapsed_sys = 0.0
                
            target_fi = int(elapsed_sys * self.fps)
            
            # 영상 끝 도달 시 대기
            if target_fi >= self.total_frames:
                time.sleep(0.05)
                continue
                
            # 타겟 프레임이 현재 처리된 인덱스보다 클 경우(즉, 화면을 갱신해야 할 때)
            if self._current_frame_idx <= target_fi:
                frames_to_skip = target_fi - self._current_frame_idx
                
                # 강제로 많이 뒤처진 경우 위치 리셋 (seek)
                if frames_to_skip > 10:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_fi)
                    self._current_frame_idx = target_fi
                else:
                    # 필요한 만큼 징검다리 스킵
                    for _ in range(frames_to_skip):
                        self.cap.grab()
                        self._current_frame_idx += 1
                        
                # 최종 화면 1장 디코딩 (I/O 부하)
                ret, frame = self.cap.retrieve()
                if ret:
                    # 여기서 메인 스레드가 해야 할 무거운 리사이즈와 색 변환을 몽땅 처리
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if self.target_size:
                        frame_rgb = cv2.resize(frame_rgb, self.target_size)
                        
                    # 스레드 안전하게 최신 프레임 갱신
                    with self.lock:
                        self._latest_frame_rgb = frame_rgb
                
                self._current_frame_idx += 1
            else:
                # 다음 프레임이 올 때까지 휴식 (CPU 절약)
                time.sleep(0.005)

    def get_latest_frame(self):
        """메인 스레드에서 안전하게 가져갈 수 있는 1개의 프레임 반환"""
        with self.lock:
            return self._latest_frame_rgb

    def reset_position(self):
        """음악을 처음부터 다시 시작할 때 위치를 동기화"""
        with self.lock:
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._current_frame_idx = 0
            if self._use_fallback_timer:
                self._fallback_start_time = time.time()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
