#!/usr/bin/env python3
"""
웹캠 영상 녹화 스크립트
정확도 비교 테스트용 영상을 녹화합니다.
"""
import cv2
import time
import os
import threading
import sys
import termios
import tty

class KeyboardReader:
    """터미널에서 non-blocking 키 입력 감지"""
    def __init__(self):
        self._key_buffer = []
        self._lock = threading.Lock()
        self._running = True
        self._old_settings = None
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)

    def start(self):
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self._thread.start()

    def stop(self):
        self._running = False
        if self._old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def _reader_loop(self):
        while self._running:
            try:
                ch = sys.stdin.read(1)
                if not ch:
                    continue
                with self._lock:
                    self._key_buffer.append(ch)
            except Exception:
                break

    def get_key(self):
        with self._lock:
            if self._key_buffer:
                return self._key_buffer.pop(0)
        return None

def record_webcam(output_path, resolution=(640, 480), duration=None):
    """
    웹캠 영상 녹화
    
    Args:
        output_path: 저장할 파일 경로
        resolution: (width, height)
        duration: 녹화 시간(초), None이면 무제한
    """
    global _stop_flag
    _stop_flag = False
    kb = KeyboardReader()
    kb.start()

    width, height = resolution
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (actual_w, actual_h))
    
    print(f"녹화 시작: {actual_w}x{actual_h} @ {fps:.0f}fps")
    print(f"저장 경로: {output_path}")
    if duration:
        print(f"녹화 시간: {duration}초")
    print(f"종료: 터미널에 q 입력 또는 화면 창에서 ESC 키\n")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            key_ch = kb.get_key()
            if key_ch == 'q':
                break
            if duration and time.time() - start_time >= duration:
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frame_count += 1
            
            # 화면 표시용은 거울모드 (좌우반전)
            display_frame = cv2.flip(frame, 1)
            
            elapsed = time.time() - start_time
            cv2.putText(
                display_frame,
                f"REC {elapsed:.1f}s  ({frame_count} frames)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            
            cv2.imshow("Recording", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            
            if frame_count % 30 == 0:
                print(f"녹화 중: {elapsed:.1f}초, {frame_count}프레임")
    
    finally:
        kb.stop()
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"\n{'='*50}")
    print(f"녹화 완료!")
    print(f"  총 프레임: {frame_count}")
    print(f"  녹화 시간: {total_time:.1f}초")
    print(f"  파일 크기: {file_size:.1f}MB")
    print(f"  저장 경로: {output_path}")
    print(f"{'='*50}")

if __name__ == "__main__":
    print("=" * 50)
    print("웹캠 영상 녹화 (정확도 비교용)")
    print("=" * 50)
    print("\n팁: 다양한 포즈를 취해보세요!")
    print("  - 팔 올리기/내리기")
    print("  - 옆으로 걷기")
    print("  - 점프")
    print("  - 제자리에서 가만히 서기 (안정성 테스트용)\n")
    
    save_dir = os.path.dirname(os.path.abspath(__file__))
    output = os.path.join(save_dir, "recorded.avi")
    
    record_webcam(output, resolution=(640, 480), duration=None)
