# FPS Benchmark Results

라즈베리파이에서 Pose Estimation 모델별 성능 측정 결과

## 테스트 환경
- **보드**: Raspberry Pi (aarch64)
- **OS**: Linux
- **Python**: 3.11
- **웹캠 해상도**: 640x480

---

## 1. 웹캠 FPS (순수 캡처 성능)

| 해상도 | 평균 FPS | 프레임 시간 | 표준편차 |
|--------|---------|------------|---------|
| 640x480 | 28.91 | 34.29ms | 2.18ms |
| 320x240 | 29.04 | 34.14ms | 2.00ms |
| 1280x720 | 29.05 | 34.14ms | 2.00ms |

> 모든 해상도에서 약 **30 FPS** 안정적

---

## 2. MediaPipe vs MoveNet 비교 (동일 영상 760프레임 기준)

| 항목 | MediaPipe | MoveNet Lightning | 승자 |
|------|-----------|-------------------|------|
| 평균 FPS | 18.09 | 20.86 | **MoveNet** |
| 추론 시간 | 55.3ms | 47.9ms | **MoveNet** |
| 감지 성공률 | 71.7% | 68.3% | MediaPipe |
| 키포인트 안정성 (Jitter) | 0.0138 | 0.0113 | **MoveNet** |
| 관절 각도 일관성 (std) | ±37.7° | ±37.1° | **MoveNet** |
| 키포인트 수 | 33개 | 17개 | MediaPipe |

### 종합 결과: **MoveNet 승리 (4:2)**

---

## 3. 분석

- **FPS**: MoveNet이 약간 빠름 (21 vs 18)
- **감지 성공률**: 비슷한 수준 (72% vs 68%)
- **안정성**: MoveNet의 키포인트 떨림이 더 적고 일관적
- **각도 일관성**: 비슷한 수준

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

# 영상 녹화 후 비교
python record_webcam.py
python compare_accuracy.py
```
