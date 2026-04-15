#!/usr/bin/env python3
"""
MediaPipe Pose Estimation 실시간 테스트 스크립트
웹캠에서 실시간으로 Pose Detection 성능을 측정합니다.
"""
import cv2
import mediapipe as mp
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

def test_mediapipe_pose(duration=None):
    """
    MediaPipe Pose 실시간 테스트
    
    Args:
        duration: 테스트 지속 시간(초), None이면 무제한
    """
    # MediaPipe Pose 초기화
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    print("MediaPipe Pose 모델 로딩 중...")
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # 0=light, 1=full, 2=heavy
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✓ 모델 로드 완료\n")
    
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    frame_times = []
    inference_times = []
    detection_count = 0
    
    print("테스트: 무제한")
    print("종료: 터미널에 q 입력 또는 화면 창에서 ESC 키\n")
    
    global _stop_flag
    _stop_flag = False
    listener = threading.Thread(target=_stdin_listener, daemon=True)
    listener.start()
    
    start_time = time.time()
    frame_count = 0
    
    try:
        # 워밍업 프레임 5개
        for _ in range(5):
            cap.read()
        
        while True:
            if duration is not None and time.time() - start_time >= duration:
                break
            
            frame_start = time.time()
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # RGB로 변환 (MediaPipe는 RGB 입력)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Pose 추론
            inf_start = time.time()
            results = pose.process(frame_rgb)
            inf_time = time.time() - inf_start
            inference_times.append(inf_time)
            
            # 결과 그리기
            if results.pose_landmarks:
                detection_count += 1
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
            
            total_frame_time = time.time() - frame_start
            frame_times.append(total_frame_time)
            frame_count += 1
            
            # 결과 표시
            fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else 0
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # 윈도우에 표시
            cv2.imshow("MediaPipe Pose Test", frame)
            
            # 터미널 q 입력 또는 화면 ESC 키로 종료
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27 or _stop_flag:
                break
            
            if frame_count % 30 == 0:
                print(f"프레임: {frame_count}, 감지율: {detection_count}/{frame_count} "
                      f"({100*detection_count/frame_count:.1f}%)")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
    
    # 결과 분석
    if frame_times:
        frame_times = np.array(frame_times)
        inference_times = np.array(inference_times)
        
        print("\n" + "=" * 60)
        print("MediaPipe Pose 성능 분석")
        print("=" * 60)
        print(f"\n총 프레임: {frame_count}")
        print(f"포즈 감지 성공: {detection_count}/{frame_count} ({100*detection_count/frame_count:.1f}%)")
        
        print(f"\n[전체 프레임 처리 시간]")
        print(f"  평균: {np.mean(frame_times)*1000:.2f}ms")
        print(f"  중앙값: {np.median(frame_times)*1000:.2f}ms")
        print(f"  최소: {np.min(frame_times)*1000:.2f}ms")
        print(f"  최대: {np.max(frame_times)*1000:.2f}ms")
        print(f"  표준편차: {np.std(frame_times)*1000:.2f}ms")
        print(f"  평균 FPS: {1.0/np.mean(frame_times):.2f}")
        
        print(f"\n[Pose 추론 시간만]")
        print(f"  평균: {np.mean(inference_times)*1000:.2f}ms")
        print(f"  중앙값: {np.median(inference_times)*1000:.2f}ms")
        print(f"  최소: {np.min(inference_times)*1000:.2f}ms")
        print(f"  최대: {np.max(inference_times)*1000:.2f}ms")
        print(f"  예상 FPS: {1.0/np.mean(inference_times):.2f}")
        
        print(f"\n[기타]")
        webcam_only = np.mean(frame_times) - np.mean(inference_times)
        print(f"  웹캠 캡처만: {webcam_only*1000:.2f}ms")
        print(f"  처리 오버헤드: {np.mean(inference_times)*1000:.2f}ms")
        
        print("=" * 60)
        print("\n✅ 평가:")
        fps = 1.0/np.mean(frame_times)
        if fps > 20:
            print("🟢 우수 - Just Dance 게임 구현 가능")
        elif fps > 10:
            print("🟡 보통 - 기본 제스처 인식 가능")
        else:
            print("🔴 낮음 - 모델 경량화나 프레임 스킵 필요")

if __name__ == "__main__":
    print("=" * 60)
    print("MediaPipe Pose Estimation 실시간 테스트")
    print("=" * 60)
    print("\n설명:")
    print("- 웹캠에서 실시간으로 Pose Detection 수행")
    print("- 스켈레톤 포인트가 화면에 표시됩니다")
    print("- 종료: 터미널에 q 입력 또는 화면 창에서 ESC 키\n")
    
    try:
        test_mediapipe_pose()
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
