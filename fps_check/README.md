# FPS Benchmark Results

라즈베리파이에서 Pose Estimation 모델별 성능 측정 결과

## 테스트 환경
- **보드**: Raspberry Pi (aarch64)
- **OS**: Linux
- **Python**: 3.11
- **웹캠 해상도**: 320x240

---

## 1. 웹캠 FPS (순수 캡처 성능)

| 해상도 | 평균 FPS | 프레임 시간 | 표준편차 |
|--------|---------|------------|---------|
| 640x480 | 28.91 | 34.29ms | 2.18ms |
| 320x240 | 29.04 | 34.14ms | 2.00ms |
| 1280x720 | 29.05 | 34.14ms | 2.00ms |

> 모든 해상도에서 약 **30 FPS** 안정적

---

## 2. MediaPipe Pose

| 항목 | 결과 |
|------|------|
| 키포인트 수 | 33개 |
| 평균 FPS | |
| 추론 시간 | |
| 감지 성공률 | |
| 평가 | |

---

## 3. MoveNet Lightning

| 항목 | 결과 |
|------|------|
| 키포인트 수 | 17개 |
| 입력 해상도 | 192x192 |
| 평균 FPS | |
| 추론 시간 | |
| 평가 | |

---

## 비교 요약

| 모델 | FPS | 추론시간 | 키포인트 | 게임 적합성 |
|------|-----|---------|---------|-----------|
| 웹캠만 | 30 | - | - | - |
| MediaPipe | 약 17| | 33개 | |
| MoveNet | 약 19| | 17개 | |

---

## 테스트 스크립트

```bash
source ~/work/env/bin/activate
cd ~/work/lets_dance/fps_check

# 웹캠 FPS
python check_webcam_fps.py

# MediaPipe
python check_mediapipe_pose.py

# MoveNet
python check_movenet.py
```
