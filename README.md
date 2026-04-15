# Let's Dance - 온디바이스 AI 댄스 채점 게임

라즈베리 파이에서 동작하는 온디바이스 AI 댄스 게임입니다. 웹캠으로 MediaPipe를 통해 자세를 감지하고, 시공간 그래프 합성곱 신경망(ST-GCN)으로 동작 임베딩을 추출하여 댄스 유사도를 실시간으로 채점합니다.

## 실행 방법
```bash
python src/main.py -S angle -s direct
```
옵션 설명

-S: direct compare 방식 설정인데, angle 기반이어서 저렇게 들어갔어요 (euclidian, cosine등 가능)

-s : 나중에 AI 모델로 비교하는거 들어오면 -s "모델명" 이런식으로 들어갈 예정이고 지금은 direct_compare의 준말

## 아키텍처 개요

```
웹캠 → MediaPipe 포즈 감지 → 랜드마크 정규화 → 스켈레톤 그래프 → ST-GCN 임베딩 → 코사인 유사도 → 점수
                                                                          ↑
                                                            레퍼런스 임베딩 (사전 녹화)
```

## 프로젝트 구조

```
lets_dance/
├── config/
│   └── settings.yaml              # 게임 설정 (카메라, 모델, UI, 채점)
├── src/
│   ├── main.py                    # 애플리케이션 진입점
│   ├── pose/                      # 포즈 추정 모듈
│   │   ├── detector.py            # MediaPipe 포즈 감지 래퍼
│   │   ├── landmark_utils.py      # 랜드마크 정규화, 스무딩, 관절 각도
│   │   └── visualizer.py          # 스켈레톤 오버레이 렌더링
│   ├── embedding/                 # 댄스 임베딩 모듈
│   │   ├── graph_builder.py       # 스켈레톤 → 그래프 자료구조 변환
│   │   ├── model.py               # ST-GCN 임베딩 모델 (PyTorch Geometric)
│   │   └── extractor.py           # 엔드-투-엔드 임베딩 추출 파이프라인
│   ├── scoring/                   # 채점 모듈
│   │   ├── similarity.py          # 코사인 / 유클리드 / DTW 유사도
│   │   ├── scorer.py              # 점수 계산, 콤보, 등급
│   │   └── feedback.py            # 실시간 피드백 생성
│   ├── game/                      # 게임 엔진 모듈
│   │   ├── engine.py              # 메인 게임 루프 및 상태 머신
│   │   ├── session.py             # 단일 댄스 플레이 세션
│   │   └── modes.py               # 연습 / 도전 / 자유 모드
│   └── ui/                        # UI 모듈 (PyGame)
│       ├── app.py                 # 메인 UI 애플리케이션
│       ├── screens/               # 전체 화면 뷰
│       │   ├── home.py            # 메인 메뉴
│       │   ├── song_select.py     # 곡/댄스 선택 화면
│       │   ├── gameplay.py        # 메인 게임 플레이 화면
│       │   ├── result.py          # 결과 화면
│       │   └── settings.py        # 설정 화면
│       └── components/            # 재사용 가능한 UI 위젯
│           ├── camera_feed.py     # 실시간 웹캠 화면
│           ├── score_display.py   # 애니메이션 점수 카운터
│           ├── progress_bar.py    # 곡 진행 바
│           ├── skeleton_overlay.py# 스켈레톤 시각화
│           └── countdown.py       # 3-2-1-GO 카운트다운
├── assets/                        # 정적 에셋
│   ├── fonts/                     # TTF 폰트 파일
│   ├── images/                    # 로고, 배경, 아이콘
│   └── sounds/                    # 효과음 (perfect.wav 등)
├── data/
│   ├── reference_dances/          # 사전 녹화된 레퍼런스 댄스 데이터
│   └── models/                    # 학습된 모델 가중치 (.pth)
├── scripts/
│   ├── collect_reference.py       # 레퍼런스 댄스 데이터 녹화
│   ├── train_embedding.py         # ST-GCN 임베딩 모델 학습
│   └── calibrate_camera.py        # 카메라 설정 및 캘리브레이션
├── tests/                         # 유닛 테스트
│   ├── test_pose.py
│   ├── test_embedding.py
│   ├── test_scoring.py
│   └── test_game.py
├── requirements.txt
├── setup.py
└── .gitignore
```

## 게임 흐름 (UI/UX)

```
┌──────────┐    ┌─────────────┐    ┌───────────┐    ┌─────────────┐    ┌──────────┐
│   홈     │───>│  곡 선택    │───>│ 카운트다운│───>│  게임 플레이│───>│  결과    │
│  메뉴    │    │   화면      │    │  3-2-1-GO │    │    화면     │    │   화면   │
└──────────┘    └─────────────┘    └───────────┘    └─────────────┘    └──────────┘
     │                                                                       │
     │<──────────────────────────────────────────────────────────────────────┘
     │                                                                  (다시하기 / 메뉴)
     v
┌──────────┐
│   설정   │
└──────────┘
```

### 게임 모드

| 모드 | 레퍼런스 오버레이 | 채점 | 콤보 | 설명 |
|------|:-:|:-:|:-:|---|
| **연습 (Practice)** | 있음 | 관대 | 없음 | 자신의 페이스로 동작을 익힌다 |
| **도전 (Challenge)** | 없음 | 엄격 | 있음 | 최고 점수를 목표로 도전한다 |
| **자유 (Freestyle)** | 없음 | 없음 | 없음 | 자유롭게 추고 동작을 녹화한다 |

### 채점 시스템

| 등급 | 기준 | 피드백 |
|-------|-----------|----------|
| Perfect | ≥ 90% | 금색 빛 효과 + 파티클 |
| Great | ≥ 75% | 초록색 플래시 |
| Good | ≥ 60% | 청록색 표시 |
| OK | ≥ 40% | 회색 텍스트 |
| Miss | < 40% | 빨간색 플래시 |

## 빠른 시작

```bash
# 1. 의존성 패키지 설치
pip install -r requirements.txt

# 2. 카메라 캘리브레이션
python scripts/calibrate_camera.py --camera 0

# 3. 레퍼런스 댄스 녹화
python scripts/collect_reference.py --song "my_dance" --duration 60

# 4. 임베딩 모델 학습
python scripts/train_embedding.py --data-dir data/reference_dances --epochs 100

# 5. 게임 실행
python src/main.py
```

## 하드웨어 요구사항

- 라즈베리 파이 4 (4GB 이상 권장)
- USB 웹캠 (720p 이상)
- 디스플레이 (HDMI, 1024x600 이상 권장)
- 스피커 (음악 및 효과음 출력용)

## 기술 스택

- **포즈 추정**:
- 1. MediaPipe Pose (33개 랜드마크)
  2. [movenet](https://www.kaggle.com/models/google/movenet/tensorFlow2/singlepose-lightning/4?tfhub-redirect=true)
- **임베딩 모델**: ST-GCN (시공간 그래프 합성곱 신경망)
- **ML 프레임워크**: PyTorch + PyTorch Geometric
- **게임 UI**: PyGame
- **카메라**: OpenCV
- **설정**: YAML
