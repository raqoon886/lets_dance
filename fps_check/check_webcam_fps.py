#!/usr/bin/env python3
"""
웹캠 FPS 측정 스크립트
다양한 해상도에서 프레임 캡처 속도를 테스트합니다.
"""
import cv2
import time
import numpy as np
import threading
import sys

# 터미널에서 'q' 입력 감지용
_stop_flag = False
def _stdin_listener():
    global _stop_flag
    for line in sys.stdin:
        if line.strip().lower() == 'q':
            _stop_flag = True
            break

def test_webcam_fps(resolution=(640, 480), duration=10):
    """
    웹캠 FPS 측정
    
    Args:
        resolution: (width, height) 튜플
        duration: 테스트 지속 시간(초)
    """
    width, height = resolution
    
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # 실제 설정된 해상도 확인
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n테스트 해상도: {actual_width}x{actual_height}")
    print(f"테스트 지속시간: {duration}초")
    print("종료: 터미널에 q 입력\n")
    
    frame_count = 0
    start_time = time.time()
    frame_times = []
    
    try:
        # 초기 프레임은 버림 (카메라 워밍업)
        for _ in range(5):
            cap.read()
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            if _stop_flag:
                print("\n터미널에서 q 입력으로 종료")
                break
            frame_start = time.time()
            ret, frame = cap.read()
            frame_time = time.time() - frame_start
            
            if ret:
                frame_times.append(frame_time)
                frame_count += 1
                
                # 진행상황 표시
                if frame_count % 30 == 0:
                    current_fps = frame_count / (time.time() - start_time)
                    print(f"프레임: {frame_count}, 현재 FPS: {current_fps:.2f}")
            else:
                print("프레임 읽기 실패")
                break
    
    finally:
        cap.release()
    
    # 결과 분석
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    frame_times = np.array(frame_times)
    
    print(f"\n--- 측정 결과 ---")
    print(f"총 프레임: {frame_count}")
    print(f"평균 FPS: {avg_fps:.2f}")
    print(f"최대 FPS: {1.0/np.min(frame_times):.2f}")
    print(f"최소 FPS: {1.0/np.max(frame_times):.2f}")
    print(f"평균 프레임 시간: {np.mean(frame_times)*1000:.2f}ms")
    print(f"표준편차: {np.std(frame_times)*1000:.2f}ms\n")
    
    return avg_fps

if __name__ == "__main__":
    print("=" * 50)
    print("웹캠 FPS 측정 도구")
    print("=" * 50)
    
    # 다양한 해상도로 테스트
    resolutions = [
        (640, 480),    # VGA
        (320, 240),    # QVGA (더 가벼움)
        (1280, 720),   # HD (더 무거움)
    ]
    
    print("\n종료: 터미널에 q 입력\n")
    
    global _stop_flag
    _stop_flag = False
    listener = threading.Thread(target=_stdin_listener, daemon=True)
    listener.start()
    
    results = {}
    for res in resolutions:
        if _stop_flag:
            break
        try:
            fps = test_webcam_fps(resolution=res, duration=5)
            results[res] = fps
        except Exception as e:
            print(f"오류: {e}\n")
    
    print("=" * 50)
    print("최종 결과 요약")
    print("=" * 50)
    for res, fps in results.items():
        print(f"{res[0]}x{res[1]}: {fps:.2f} FPS")
