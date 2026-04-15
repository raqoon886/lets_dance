#!/usr/bin/env python3
"""
MediaPipe vs MoveNet 정확도 비교 스크립트
녹화된 동일 영상으로 두 모델을 비교합니다.

비교 항목:
1. 감지 성공률 (Detection Rate)
2. 키포인트 안정성 (Jitter - 표준편차)
3. 관절 각도 일관성
4. FPS
5. 시각적 비교 영상 저장
"""
import cv2
import numpy as np
import time
import os
import json

# ============================================================
# MediaPipe
# ============================================================
def run_mediapipe(video_path):
    """MediaPipe로 영상 분석"""
    import mediapipe as mp
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    results_data = {
        'keypoints_per_frame': [],
        'detection': [],
        'inference_times': [],
    }
    
    print(f"\n[MediaPipe] 분석 중... (총 {total_frames}프레임)")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        t0 = time.time()
        result = pose.process(frame_rgb)
        inf_time = time.time() - t0
        results_data['inference_times'].append(inf_time)
        
        if result.pose_landmarks:
            results_data['detection'].append(True)
            kps = []
            for lm in result.pose_landmarks.landmark:
                kps.append([lm.x, lm.y, lm.z, lm.visibility])
            results_data['keypoints_per_frame'].append(np.array(kps))
        else:
            results_data['detection'].append(False)
            results_data['keypoints_per_frame'].append(None)
        
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  {frame_idx}/{total_frames} 프레임 처리 완료")
    
    cap.release()
    pose.close()
    
    return results_data

# ============================================================
# MoveNet
# ============================================================
def run_movenet(video_path):
    """MoveNet으로 영상 분석"""
    import tensorflow as tf
    import tensorflow_hub as hub
    
    module = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
    movenet = module.signatures['serving_default']
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    results_data = {
        'keypoints_per_frame': [],
        'detection': [],
        'inference_times': [],
    }
    
    print(f"\n[MoveNet] 분석 중... (총 {total_frames}프레임)")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        input_image = tf.cast(tf.image.resize_with_pad(
            tf.expand_dims(frame, 0), 192, 192
        ), dtype=tf.int32)
        
        t0 = time.time()
        outputs = movenet(input=input_image)
        inf_time = time.time() - t0
        results_data['inference_times'].append(inf_time)
        
        keypoints = outputs['output_0'].numpy()[0][0]  # (17, 3)
        
        # 감지 여부: confidence > 0.3인 키포인트가 5개 이상
        detected = np.sum(keypoints[:, 2] > 0.3) >= 5
        results_data['detection'].append(detected)
        
        if detected:
            results_data['keypoints_per_frame'].append(keypoints)
        else:
            results_data['keypoints_per_frame'].append(None)
        
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  {frame_idx}/{total_frames} 프레임 처리 완료")
    
    cap.release()
    
    return results_data

# ============================================================
# 분석 함수들
# ============================================================
def calc_detection_rate(data):
    """감지 성공률 계산"""
    total = len(data['detection'])
    detected = sum(data['detection'])
    return detected / total * 100 if total > 0 else 0

def calc_jitter(data):
    """
    키포인트 안정성 (Jitter) 계산
    연속 프레임 간 키포인트 이동량의 표준편차
    낮을수록 안정적
    """
    keypoints = data['keypoints_per_frame']
    diffs = []
    
    for i in range(1, len(keypoints)):
        if keypoints[i] is not None and keypoints[i-1] is not None:
            kp_curr = keypoints[i][:, :2]  # x, y만
            kp_prev = keypoints[i-1][:, :2]
            diff = np.linalg.norm(kp_curr - kp_prev, axis=1)
            diffs.append(np.mean(diff))
    
    if diffs:
        return np.mean(diffs), np.std(diffs)
    return float('inf'), float('inf')

def calc_angle(p1, p2, p3):
    """세 점으로 각도 계산 (degree)"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.degrees(np.arccos(cos_angle))

def calc_angle_consistency(data, joint_indices):
    """
    관절 각도 일관성 측정
    같은 포즈에서 각도 변화의 표준편차
    낮을수록 일관적
    
    joint_indices: (p1, p2, p3) 인덱스 튜플
    """
    angles = []
    for kps in data['keypoints_per_frame']:
        if kps is not None:
            p1 = kps[joint_indices[0]][:2]
            p2 = kps[joint_indices[1]][:2]
            p3 = kps[joint_indices[2]][:2]
            angle = calc_angle(p1, p2, p3)
            angles.append(angle)
    
    if angles:
        return np.mean(angles), np.std(angles)
    return 0, float('inf')

def calc_fps(data):
    """FPS 계산"""
    times = data['inference_times']
    if times:
        return 1.0 / np.mean(times)
    return 0

# ============================================================
# 메인
# ============================================================
def compare_models(video_path):
    """두 모델 비교 분석"""
    
    if not os.path.exists(video_path):
        print(f"❌ 영상 파일을 찾을 수 없습니다: {video_path}")
        print("\n먼저 녹화하세요:")
        print("  python record_webcam.py")
        return
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print("=" * 70)
    print("MediaPipe vs MoveNet 정확도 비교")
    print("=" * 70)
    print(f"\n영상: {video_path}")
    print(f"프레임: {total_frames}, 해상도: {w}x{h}, FPS: {fps:.0f}")
    
    # 1. 두 모델 실행
    mp_data = run_mediapipe(video_path)
    mn_data = run_movenet(video_path)
    
    # 2. 감지 성공률
    mp_det = calc_detection_rate(mp_data)
    mn_det = calc_detection_rate(mn_data)
    
    # 3. 안정성 (Jitter)
    mp_jitter_mean, mp_jitter_std = calc_jitter(mp_data)
    mn_jitter_mean, mn_jitter_std = calc_jitter(mn_data)
    
    # 4. FPS
    mp_fps = calc_fps(mp_data)
    mn_fps = calc_fps(mn_data)
    
    # 5. 관절 각도 일관성 (공통 키포인트)
    # MediaPipe: 11=왼어깨, 13=왼팔꿈치, 15=왼손목
    # MoveNet:   5=왼어깨, 7=왼팔꿈치, 9=왼손목
    mp_angle_mean, mp_angle_std = calc_angle_consistency(mp_data, (11, 13, 15))
    mn_angle_mean, mn_angle_std = calc_angle_consistency(mn_data, (5, 7, 9))
    
    # 결과 출력
    print("\n" + "=" * 70)
    print("📊 비교 결과")
    print("=" * 70)
    
    print(f"\n{'항목':<25} {'MediaPipe':<20} {'MoveNet':<20} {'승자':<10}")
    print("-" * 75)
    
    # 감지 성공률
    det_winner = "MediaPipe" if mp_det > mn_det else "MoveNet" if mn_det > mp_det else "동일"
    print(f"{'감지 성공률':<25} {mp_det:.1f}%{'':<14} {mn_det:.1f}%{'':<14} {det_winner}")
    
    # FPS
    fps_winner = "MediaPipe" if mp_fps > mn_fps else "MoveNet" if mn_fps > mp_fps else "동일"
    print(f"{'추론 FPS':<25} {mp_fps:.1f}{'':<15} {mn_fps:.1f}{'':<15} {fps_winner}")
    
    # 추론 시간
    mp_inf = np.mean(mp_data['inference_times']) * 1000
    mn_inf = np.mean(mn_data['inference_times']) * 1000
    inf_winner = "MediaPipe" if mp_inf < mn_inf else "MoveNet" if mn_inf < mp_inf else "동일"
    print(f"{'평균 추론 시간':<25} {mp_inf:.1f}ms{'':<13} {mn_inf:.1f}ms{'':<13} {inf_winner}")
    
    # Jitter (낮을수록 좋음)
    jit_winner = "MediaPipe" if mp_jitter_mean < mn_jitter_mean else "MoveNet" if mn_jitter_mean < mp_jitter_mean else "동일"
    print(f"{'키포인트 안정성 (Jitter)':<25} {mp_jitter_mean:.4f}±{mp_jitter_std:.4f}  {mn_jitter_mean:.4f}±{mn_jitter_std:.4f}  {jit_winner}")
    
    # 각도 일관성 (std 낮을수록 좋음)
    ang_winner = "MediaPipe" if mp_angle_std < mn_angle_std else "MoveNet" if mn_angle_std < mp_angle_std else "동일"
    print(f"{'왼팔 각도 평균':<25} {mp_angle_mean:.1f}°±{mp_angle_std:.1f}°{'':<6} {mn_angle_mean:.1f}°±{mn_angle_std:.1f}°{'':<6} {ang_winner}")
    
    # 키포인트 수
    print(f"{'키포인트 수':<25} {'33개':<20} {'17개':<20} {'MediaPipe'}")
    
    print("\n" + "=" * 70)
    print("📌 종합 평가")
    print("=" * 70)
    
    scores = {'MediaPipe': 0, 'MoveNet': 0}
    for w in [det_winner, fps_winner, inf_winner, jit_winner, ang_winner]:
        if w in scores:
            scores[w] += 1
    
    print(f"\n  MediaPipe: {scores['MediaPipe']}점 / 5점")
    print(f"  MoveNet:   {scores['MoveNet']}점 / 5점")
    
    overall = max(scores, key=scores.get)
    print(f"\n  🏆 종합 우승: {overall}")
    
    if overall == "MediaPipe":
        print("  → MediaPipe가 라즈베리파이 Just Dance 게임에 더 적합합니다!")
    else:
        print("  → MoveNet이 라즈베리파이 Just Dance 게임에 더 적합합니다!")
    
    print("=" * 70)
    
    # 결과 저장
    save_dir = os.path.dirname(os.path.abspath(__file__))
    result = {
        'video': video_path,
        'total_frames': total_frames,
        'mediapipe': {
            'detection_rate': mp_det,
            'fps': mp_fps,
            'avg_inference_ms': mp_inf,
            'jitter_mean': float(mp_jitter_mean),
            'jitter_std': float(mp_jitter_std),
            'angle_mean': float(mp_angle_mean),
            'angle_std': float(mp_angle_std),
        },
        'movenet': {
            'detection_rate': mn_det,
            'fps': mn_fps,
            'avg_inference_ms': mn_inf,
            'jitter_mean': float(mn_jitter_mean),
            'jitter_std': float(mn_jitter_std),
            'angle_mean': float(mn_angle_mean),
            'angle_std': float(mn_angle_std),
        },
        'winner': overall,
    }
    
    result_path = os.path.join(save_dir, "comparison_result.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {result_path}")

if __name__ == "__main__":
    save_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(save_dir, "recorded.avi")
    
    try:
        compare_models(video_path)
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
