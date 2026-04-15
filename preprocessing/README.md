# preprocessing

## 보드가 아닌 로컬 PC에서 실행 권장 !!!!
레퍼런스 댄스 영상을 게임에서 사용할 수 있는 형태로 변환하는 전처리 스크립트 모음입니다.


---

## 파일 목록

### 1. `download_youtube.py`

**역할**  
유튜브(YouTube / Shorts) URL에서 영상을 MP4 형식으로 다운로드합니다.  
`yt-dlp`를 사용하며, 최고 화질의 영상과 오디오를 자동으로 병합하여 저장합니다.  
출력 파일은 기본적으로 `assets/videos/` 디렉터리에 저장됩니다.

**실행 방법**

```bash
uv run --with yt-dlp python3 download_youtube.py <YouTube_URL>
```

**예시**

```bash
uv run --with yt-dlp python3 download_youtube.py "https://www.youtube.com/watch?v=XXXX"
```

**출력**

| 항목 | 내용 |
|------|------|
| 저장 경로 | `assets/videos/<제목>_<id>.mp4` |

---

### 2. `process_video.py`

**역할**  
입력 영상에서 사람 실루엣(회색)을 추출하고, 흰색 배경 위에 포즈 스켈레톤을 오버레이한 영상을 생성합니다.  
동시에 프레임별 관절 좌표를 NumPy 배열(`.npy`)로 저장합니다.

포즈 추출 모델로 두 가지를 지원합니다.

| 모드 | 모델 | 랜드마크 수 | 실루엣 추출 방식 |
|------|------|------------|----------------|
| `mediapipe` (기본값) | MediaPipe Pose | 33개 | 세그멘테이션 마스크 |
| `movenet` | TF Hub MoveNet Lightning | 17개 | rembg 배경 제거 |

**실행 방법**

```bash
# MediaPipe 모드 (기본)
uv run --python 3.10 \
  --with opencv-python --with numpy \
  --with "mediapipe==0.10.9" \
  python3 process_video.py <입력영상.mp4>

# MoveNet 모드
uv run \
  --with opencv-python --with numpy \
  --with tensorflow --with tensorflow-hub \
  python3 process_video.py <입력영상.mp4> --model movenet
```

**주요 옵션**

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `input_video` | (필수) | 입력 영상 경로 |
| `--model` | `mediapipe` | 포즈 추출 모델 (`mediapipe` / `movenet`) |
| `--output_video` | 자동 생성 | 출력 영상 경로 |
| `--crf` | `28` | ffmpeg 압축 품질 (낮을수록 고품질) |

**출력**

| 항목 | 경로 예시 | 설명 |
|------|-----------|------|
| 처리된 영상 | `input_mediapipe_processed.mp4` | 실루엣 + 스켈레톤 오버레이 영상 |
| 포즈 데이터 | `input_mediapipe_processed_pose.npy` | 프레임별 관절 좌표 `(N, 33, 3)` 또는 `(N, 17, 3)` |

> `.npy` 배열의 각 프레임 값은 `[x, y, visibility/score]` (정규화 좌표 0.0~1.0)

---

### 3. `make_silhouette.py`

**역할**  
그린스크린(크로마키) 배경으로 촬영된 영상에서 배경을 흰색으로, 사람 영역을 회색(128, 128, 128)으로 변환하여 실루엣 영상을 생성합니다.  
HSV 색공간 기반으로 녹색 배경을 감지하므로 크로마키 영상 전용입니다.

**실행 방법**

스크립트 내 경로를 직접 수정한 뒤 실행합니다.

```bash
# 스크립트 내 변수 수정 후 실행
python3 make_silhouette.py
```

또는 함수를 직접 호출하는 방식으로 사용합니다.

```python
from make_silhouette import process_video
process_video("assets/videos/input.mp4", "assets/videos/input_silhouette.mp4")
```

**기본 경로 설정 (스크립트 하단)**

```python
input_video  = "assets/videos/CJS1776217022.mp4"
output_video = "assets/videos/CJS1776217022_silhouette.mp4"
```

**의존성**

```
opencv-python, numpy
```

**출력**

| 항목 | 설명 |
|------|------|
| 실루엣 영상 | 흰 배경 + 회색 사람 영역의 MP4 영상 |

---

### 4. `add_skeleton.py`

**역할**  
기존 실루엣 영상 위에 원본 영상을 분석하여 MediaPipe Pose 스켈레톤을 오버레이합니다.  
`make_silhouette.py`로 생성된 실루엣 영상과 원본 영상이 모두 필요합니다.

**실행 방법**

스크립트 내 경로를 직접 수정한 뒤 실행합니다.

```bash
python3 add_skeleton.py
```

```python
from add_skeleton import overlay_skeleton
overlay_skeleton(
    orig_video_path="assets/videos/input.mp4",
    silhouette_video_path="assets/videos/input_silhouette.mp4",
    output_path="assets/videos/input_skeleton.mp4"
)
```

**기본 경로 설정 (스크립트 하단)**

```python
orig_video   = "assets/videos/CJS1776217022.mp4"
sil_video    = "assets/videos/CJS1776217022_silhouette.mp4"
output_video = "assets/videos/CJS1776217022_skeleton.mp4"
```

**의존성**

```
opencv-python, mediapipe, numpy
```

**출력**

| 항목 | 설명 |
|------|------|
| 스켈레톤 합성 영상 | 실루엣 배경 위에 MediaPipe 스켈레톤이 그려진 MP4 영상 |

---

## 전형적인 전처리 워크플로우

```
1. download_youtube.py     유튜브에서 원본 영상 다운로드
        ↓
2. process_video.py        원본 영상 → 실루엣 영상 + 포즈 .npy (권장)
        
   ── 또는 그린스크린 영상인 경우 ──
2a. make_silhouette.py     그린스크린 → 실루엣 영상
2b. add_skeleton.py        실루엣 영상 + 원본 → 스켈레톤 오버레이 영상
```

> 일반 영상(그린스크린 아님)에서 포즈 데이터까지 한 번에 추출하려면 **`process_video.py`** 사용을 권장합니다.
