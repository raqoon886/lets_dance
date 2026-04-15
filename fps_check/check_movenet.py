#!/usr/bin/env python3
"""
MoveNet Pose Estimation 실시간 테스트 스크립트
MediaPipe와 비교용
"""
import cv2
import numpy as np
import time
import tensorflow as tf
import tensorflow_hub as hub
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

def test_movenet_fps(model_type='lightning', duration=None):
    """
    MoveNet 실시간 FPS 테스트
    
    Args:
        model_type: 'lightning' (빠름) 또는 'thunder' (정확함)
        duration: 테스트 시간(초), None이면 무제한
    """
    print("=" * 70)
    print(f"MoveNet-{model_type.upper()} Pose Estimation 실시간 테스트")
    print("=" * 70)
    
    print(f"\n📥 모델 로딩 중...")
    
    # MoveNet 모델 로드
    model_urls = {
        'lightning': 'https://tfhub.dev/google/movenet/singlepose/lightning/4',
        'thunder': 'https://tfhub.dev/google/movenet/singlepose/thunder/4',
    }
    
    try:
        module = hub.load(model_urls[model_type])
        movenet = module.signatures['serving_default']
        print(f"✓ MoveNet-{model_type} 모델 로드 완료\n")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        print("\n다음 명령어를 먼저 실행하세요:")
        print("  pip install tensorflow tensorflow-hub")
        return
    
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"⏱️  테스트: 무제한")
    print(f"종료: 터미널에 q 입력 또는 화면 창에서 ESC 키")
    
    global _stop_flag
    _stop_flag = False
    listener = threading.Thread(target=_stdin_listener, daemon=True)
    listener.start()
    print(f"📷 웹캠: {actual_width}x{actual_height}\n")
    
    frame_times = []
    inference_times = []
    frame_count = 0
    
    start_time = time.time()
    
    try:
        # 워밍업 프레임
        for _ in range(3):
            cap.read()
        
        while True:
            if duration is not None and time.time() - start_time >= duration:
                break
            
            frame_start = time.time()
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # 입력 전처리 (Lightning: 192x192 int32)
            input_image = tf.cast(tf.image.resize_with_pad(
                tf.expand_dims(frame, 0),
                192, 192
            ), dtype=tf.int32)
            
            # MoveNet 추론
            inf_start = time.time()
            outputs = movenet(input=input_image)
            keypoints = outputs['output_0'].numpy()[0][0]
            inf_time = time.time() - inf_start
            inference_times.append(inf_time)
            
            # 키포인트 그리기
            frame_h, frame_w = frame.shape[:2]
            for kp in keypoints:
                y, x, conf = kp[0], kp[1], kp[2]
                if conf > 0.3:  # 신뢰도 필터
                    x_pixel = int(x * frame_w)
                    y_pixel = int(y * frame_h)
                    cv2.circle(frame, (x_pixel, y_pixel), 3, (0, 255, 0), -1)
            
            total_time = time.time() - frame_start
            frame_times.append(total_time)
            frame_count += 1
            
            # 결과 표시
            fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else 0
            cv2.putText(
                frame,
                f"MoveNet-{model_type.upper()} FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f"Inference: {inf_time*1000:.1f}ms",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            cv2.imshow("MoveNet Test", frame)
            
            # 터미널 q 입력 또는 화면 ESC 키로 종료
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27 or _stop_flag:
                break
            
            # 진행 상황 출력
            if frame_count % 50 == 0:
                avg_fps = 1.0 / np.mean(frame_times[-50:])
                avg_inf = np.mean(inference_times[-50:]) * 1000
                print(f"프레임: {frame_count:3d}, FPS: {avg_fps:6.2f}, 추론시간: {avg_inf:6.2f}ms")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # 최종 분석
    if frame_times and inference_times:
        frame_times = np.array(frame_times)
        inference_times = np.array(inference_times)
        
        print("\n" + "=" * 70)
        print(f"📊 MoveNet-{model_type.upper()} 성능 분석")
        print("=" * 70)
        
        print(f"\n총 프레임: {frame_count}")
        print(f"\n[전체 처리 시간]")
        print(f"  평균 FPS: {1.0/np.mean(frame_times):.2f}")
        print(f"  평균 시간: {np.mean(frame_times)*1000:.2f}ms")
        print(f"  최소 시간: {np.min(frame_times)*1000:.2f}ms")
        print(f"  최대 시간: {np.max(frame_times)*1000:.2f}ms")
        print(f"  표준편차: {np.std(frame_times)*1000:.2f}ms")
        
        print(f"\n[추론 시간만]")
        print(f"  평균 시간: {np.mean(inference_times)*1000:.2f}ms")
        print(f"  추론 FPS: {1.0/np.mean(inference_times):.2f}")
        
        print(f"\n[기타]")
        overhead = np.mean(frame_times) - np.mean(inference_times)
        print(f"  웹캠/전처리: {overhead*1000:.2f}ms")
        
        print("\n" + "=" * 70)
        print("✅ 평가:")
        fps = 1.0/np.mean(frame_times)
        if fps > 30:
            print("🟢 우수 - Just Dance 게임 완벽 구현 가능")
        elif fps > 20:
            print("🟢 좋음 - Just Dance 게임 구현 가능")
        elif fps > 10:
            print("🟡 보통 - 기본 제스처 인식 가능")
        else:
            print("🔴 낮음 - 모델 경량화 필요")
        print("=" * 70)

if __name__ == "__main__":
    print("\n설명:")
    print("- 웹캠에서 실시간으로 Pose Detection 수행")
    print("- 키포인트가 화면에 표시됩니다")
    print("- 종료: 터미널에 q 입력 또는 화면 창에서 ESC 키")
    
    print("\n모델 선택:")
    print("- lightning: 빠름 (추천)")
    print("- thunder: 정확함")
    print()
    
    try:
        test_movenet_fps(model_type='lightning', duration=None)
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        print("\n다음을 설치하세요:")
        print("  pip install tensorflow tensorflow-hub")
        import traceback
        traceback.print_exc()
