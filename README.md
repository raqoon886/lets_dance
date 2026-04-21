# Let's Dance - 온디바이스 AI 댄스 채점 게임

라즈베리 파이 및 엣지 디바이스에서 동작하는 온디바이스 AI 댄스 게임입니다. 웹캠으로 실시간 랜드마크를 감지하고, 경량화된 AI 모델(TFLite 기반 ST-GCN/MLP)로 동작 임베딩을 추출하여 레퍼런스 영상과 사용자의 댄스 유사도를 실시간으로 채점합니다.

## 실행 방법

최적화된 AI 모델(Scratch 모드)을 사용한 실행 커맨드는 다음과 같습니다:

```bash
python src/main.py -s scratch --scratch-model-path data/models/scratch/scratch_gcn_base_infonce_e64.tflite
```

### 주요 실행 옵션
- `-s <method>`: 채점 방식 설정 (`direct`, `embedding`, `scratch`)
  - `scratch`: TFLite 모델 추론 기반 채점 파이프라인 (권장)
  - `direct`: 별도 딥러닝 없이 랜드마크 스켈레톤 각도/거리를 직접 연산 (Fallback)
- `--scratch-model-path`: `scratch` 모드 실행 시 사용할 TFLite 모델 가중치 파일 경로
- `-S <metric>`: `direct` 모드 동작 시 사용할 유사도 측정 기준 (`angle`, `euclidean`, `cosine` 등)

## 아키텍처 및 성능 최적화

이 프로젝트는 라즈베리 파이와 같은 저사양 기기에서도 프레임 드랍 없이 부드러운 30FPS 게임 경험을 제공할 수 있도록 **렌더링과 추론을 분리**하고 극단적인 최적화를 적용했습니다.

### 🎮 비동기 렌더링 파이프라인
```text
비동기 웹캠 (Zero-latency 캡처) ──┐
                                  ├──> [프레임 시퀀스 캐시] 변경 시에만 렌더링 ──> 화면 출력 (PyGame)
백그라운드 디코딩 (레퍼런스 영상) ─┘   (cv2 연산, tobytes 비용 등 대폭 절감)
```

### 🧠 추론/채점 파이프라인 (비동기 2Hz 연산)
```text
포즈 감지 (MediaPipe/MoveNet) → 정규화 → Ring Buffer (O(1)) 누적 ─[15프레임마다]─> TFLite 추론 → 코사인 유사도
                                                                                        ↑
                                                                         레퍼런스 임베딩 캐시 (사전 연산 완료)
```

## 프로젝트 구조

```
lets_dance/
├── config/
│   └── settings.yaml              # 게임 설정 (카메라, 모델 경로, 각종 판정 임계치)
├── src/
│   ├── main.py                    # 애플리케이션 진입점 및 파라미터 핸들링
│   ├── pose/                      # 포즈 추정 모듈 (MediaPipe / MoveNet 등)
│   ├── scoring/                   # 채점 및 AI 추론 모듈
│   │   ├── scratch_similarity.py  # TFLite 모델 추론 및 임베딩 코사인 유사도 연산
│   │   ├── scorer.py              # 로우 유사도를 게임 점수, 콤보, 등급(Perfect/Great/Miss)으로 환산
│   │   └── feedback.py            # 인게임 실시간 피드백 생성
│   ├── game/                      # 게임 엔진 코어 모듈
│   │   └── engine.py              # PyGame 렌더링, 메인 상태 머신 (최적화 포함)
│   ├── utils/
│   │   ├── async_camera.py        # 백그라운드 스레드 기반 카메라 캡처 (지연 최소화)
│   │   └── async_video.py         # 백그라운드 영상 디코딩 (UI 프레임 방어)
│   └── ui/                        # 기타 PyGame UI 컴포넌트
├── assets/                        # 폰트, 이미지, 사운드 등 시스템 에셋
├── data/
│   ├── reference_dances/          # 사전 녹화된 레퍼런스 비디오 모음
│   └── models/                    # TFLite 경량화 AI 모델 가중치 모음
└── scripts/                       # 데이터 수집, 모델 컨버팅, 분석 도구
```

## 게임 흐름 (UI/UX)

```
┌──────────┐    ┌─────────────┐    ┌───────────┐    ┌─────────────┐    ┌──────────┐
│   홈     │───>│  곡 선택    │───>│ 카운트다운│───>│  게임 플레이│───>│  결과    │
│  메뉴    │    │   화면      │    │  3-2-1-GO │    │    화면     │    │   화면   │
└──────────┘    └─────────────┘    └───────────┘    └─────────────┘    └──────────┘
```

### 채점 시스템 (기본값)
| 등급 | 기준 | 피드백 |
|-------|-----------|----------|
| Perfect | ≥ 90% | 금빛 이펙트 (+ 콤보 유지) |
| Great | ≥ 75% | 초록색 텍스트 (+ 콤보 유지) |
| Good | ≥ 60% | 청록색 텍스트 (+ 콤보 유지) |
| OK | ≥ 40% | 회색 텍스트 (콤보 끊김) |
| Miss | < 40% | 빨간색 텍스트 (콤보 끊김) |
*(※ 포즈 프레임이 아예 없는 누락 상태, 또는 너무 타이밍이 빗나간 경우에도 감점 적용)*

## 하드웨어 요구사항
- **Raspberry Pi 4 / 5 (4GB 이상 권장)** 또는 일반 데스크탑 (Windows / Linux / macOS)
- USB 웹캠 (640x480 이상, 고프레임 권장 지원)
- 디스플레이 모니터 및 스피커

## 기술 스택
- **Machine Learning**: TensorFlow Lite (`ai-edge-litert`)
- **Pose Detection**: MediaPipe Pose / TensorFlow Lite MoveNet
- **Computer Vision**: OpenCV (`cv2`)
- **Game/UI Engine**: PyGame, NumPy
