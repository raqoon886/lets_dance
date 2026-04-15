# Scratch TFLite 모델 기반 채점 개발 내역

작성일: 2026-04-15

## 목표

기존 `direct` 비교는 유지하면서, `direct` 대신 직접 학습한 AI 모델로 skeleton 유사도를 계산하는 옵션을 추가했다. 새 옵션 이름은 `scratch`다.

서비스 실행 예:

```bash
python src/main.py -s scratch --scratch-model-name gcn_e64
```

`scratch`는 registry에 저장된 모델명을 받아 해당 `.tflite` encoder를 로드한다. 경로를 직접 지정할 수도 있다.

```bash
python src/main.py -s scratch --scratch-model-path data/models/scratch/gcn_e64.tflite
```

## Direct와 Scratch 차이

### direct

```text
현재 사용자 skeleton 1프레임
→ 정답 skeleton 후보 프레임
→ angle / cosine / euclidean / hybrid 직접 계산
→ similarity
→ DanceScorer
```

특징:

- 모델을 사용하지 않는다.
- 한 프레임 단위로 빠르게 점수를 낸다.
- 시간 흐름, 동작 방향, 박자 변화는 모델보다 약하게 반영된다.

### scratch

```text
최근 사용자 skeleton window
→ scratch TFLite encoder
→ user embedding

정답 skeleton 후보 window
→ 같은 scratch TFLite encoder
→ reference embedding

embedding cosine similarity
→ similarity
→ DanceScorer
```

특징:

- TCN 또는 GCN 모델을 사용한다.
- 기본 window는 30프레임이다.
- 30프레임이 쌓이기 전에는 점수를 보류한다.
- 한 순간의 pose가 아니라 짧은 동작 패턴을 비교한다.

## 추가된 런타임 기능

### `src/main.py`

추가된 CLI:

```bash
--score-method scratch
--scratch-model-name
--scratch-model-dir
--scratch-model-path
--scratch-sequence-length
--scratch-feature-dims
--scratch-input-layout
--scratch-candidate-stride
```

사용자는 모델명을 선택해서 실행할 수 있다.

```bash
python src/main.py -s scratch --scratch-model-name tcn_e32
python src/main.py -s scratch --scratch-model-name gcn_e64
```

### `src/game/engine.py`

변경 내용:

- `score_method == "scratch"` 분기 추가
- `ScratchPoseSimilarity` 초기화 추가
- gameplay scoring loop에서 scratch 모델 점수 계산 추가
- reference가 없으면 scratch scoring을 멈추고 warning 출력
- 새 게임 시작 시 scratch rolling window와 reference embedding cache reset

### `config/settings.yaml`

추가된 기본 설정:

```yaml
scratch:
  model_dir: "data/models/scratch"
  model_name: "gcn_e64"
  model_path: ""
  sequence_length: 30
  feature_dims: 2
  input_layout: "BTJC"
  top_k: 3
  candidate_stride: 3
```

`model_path`가 비어 있으면 `model_dir/model_registry.json`에서 `model_name`을 찾아 로드한다.

## 추가된 scoring 코드

### `src/scoring/scratch_features.py`

학습과 런타임이 공유하는 skeleton 전처리 코드다.

기능:

- MediaPipe 33개 landmark 중 춤 평가용 12개 joint 선택
- `feature_dims`에 따라 `xy`, `xyz`, `xyz+visibility` 선택
- 골반 중심 기준 translation normalization
- 어깨 너비 기준 scale normalization
- fixed-length pose window 생성

기본 모델 입력:

```text
(30, 12, 2)
```

TFLite 입력:

```text
(1, 30, 12, 2)
```

### `src/scoring/scratch_similarity.py`

서비스에서 `.tflite` 모델을 로드하고 점수를 계산하는 코드다.

기능:

- 모델명 또는 모델 경로로 TFLite encoder 로드
- `tflite_runtime` 우선 사용, 없으면 `tensorflow.lite.Interpreter` fallback
- 최근 사용자 skeleton window를 rolling buffer로 관리
- 정답 skeleton의 지연 허용 후보 window를 embedding으로 변환
- embedding cosine similarity를 `0~1` 점수로 변환
- NaN/Inf가 scoring으로 넘어가지 않도록 finite 값으로 처리

## 학습 데이터 구성

원본 데이터는 다음 구조를 기대한다.

```text
data/reference_dances/{dance_name}/reference.npy
```

현재 프로젝트에는 다음 reference가 있다.

```text
beginner_wave
cheerup_dance
freestyle_free
hiphop_move
kpop_basic
```

각 `reference.npy`는 pose estimation이 끝난 skeleton sequence다.

일반 shape:

```text
(frames, 33, 4)
```

마지막 차원:

```text
x, y, z, visibility
```

## 학습 pair 생성

학습 스크립트:

```text
scripts/train_scratch_similarity.py
```

별도의 수동 라벨 데이터셋은 만들지 않았다. reference skeleton sequence에서 positive/negative pair를 자동 구성한다.

### Positive pair

같은 dance에서 시간적으로 가까운 두 window를 뽑는다.

```text
same dance, close time
label = 1
```

기본값:

```text
positive_jitter = 6 frames
```

### Negative pair

다른 dance의 window 또는 같은 dance라도 충분히 멀리 떨어진 window를 뽑는다.

```text
different dance or far time
label = 0
```

기본값:

```text
negative_gap = 45 frames
```

### Augmentation

작은 Gaussian noise를 normalized skeleton에 추가한다.

```text
noise_std = 0.015
```

목적은 pose estimation jitter에 대한 robustness를 조금 높이는 것이다.

## 모델 구조

학습은 Siamese 구조다.

```text
user_window      → shared encoder → user_embedding
reference_window → shared encoder → reference_embedding

cosine(user_embedding, reference_embedding)
→ binary cross entropy(label)
```

서비스에 저장되는 것은 Siamese 전체가 아니라 shared encoder 하나다. 서비스에서는 user window와 reference window를 각각 encoder에 넣고 cosine similarity를 계산한다.

### TCN

```text
Input (T, 12, C)
→ flatten joints/features
→ Dense projection
→ dilated causal Conv1D blocks
→ GlobalAveragePooling1D
→ Dense
→ embedding
```

시간 흐름을 주로 학습한다.

### GCN

```text
Input (T, 12, C)
→ Dense projection
→ skeleton adjacency aggregation
→ spatial Dense
→ temporal Conv1D
→ residual blocks
→ GlobalAveragePooling2D
→ Dense
→ embedding
```

12개 joint의 연결 구조를 사용한다.

## 모델 학습 명령

단일 모델:

```bash
python scripts/train_scratch_similarity.py \
  --model-name gcn_custom \
  --model-type gcn \
  --sequence-length 30 \
  --feature-dims 2 \
  --embedding-dim 64 \
  --epochs 30
```

여러 모델:

```bash
python scripts/train_scratch_variants.py --profile quick
python scripts/train_scratch_variants.py --profile full --only gcn_e64
```

embedding dimension을 바꾼 grid:

```bash
python scripts/train_scratch_variants.py \
  --profile quick \
  --embedding-dims 16,32,64 \
  --model-types tcn,gcn
```

epoch만 바꿔 학습:

```bash
python scripts/train_scratch_variants.py \
  --profile quick \
  --epochs 5
```

## 저장 파일

기본 저장 위치:

```text
data/models/scratch/
```

각 모델마다 생성되는 파일:

```text
{model_name}.tflite
{model_name}_encoder.keras
{model_name}_meta.json
```

전체 registry:

```text
data/models/scratch/model_registry.json
```

registry에는 모델명, 모델 타입, 파라미터, metric, 서비스 실행 명령이 저장된다.

## 현재 학습된 모델

2026-04-15 20:51:53에 `quick` profile로 기본 4개 모델을 실제 학습하고 TFLite로 저장했다.

| 모델명 | 구조 | TFLite 경로 | val_binary_accuracy | val_loss |
|---|---|---|---:|---:|
| `tcn_e32` | TCN | `data/models/scratch/tcn_e32.tflite` | 0.6125 | 0.7089 |
| `tcn_e64` | TCN | `data/models/scratch/tcn_e64.tflite` | 0.7500 | 0.5811 |
| `gcn_e32` | GCN | `data/models/scratch/gcn_e32.tflite` | 0.7063 | 0.5383 |
| `gcn_e64` | GCN | `data/models/scratch/gcn_e64.tflite` | 0.7375 | 0.4982 |

동일 reference window와 다른 dance window를 runtime comparator로 넣어 finite score가 나오는 것도 확인했다.

```text
tcn_e32: same=1.000000, cross=0.877975
tcn_e64: same=1.000000, cross=0.804821
gcn_e32: same=1.000000, cross=0.455839
gcn_e64: same=1.000000, cross=0.014597
```

위 모델들은 빠른 확인용 `quick` 학습 결과다. 실제 서비스 후보는 더 많은 reference와 `full` profile로 다시 학습하는 것이 좋다.

## 학습 노트북

추가한 노트북:

```text
notebooks/scratch_tflite_training.ipynb
```

포함 내용:

- reference skeleton 데이터 확인
- skeleton 전처리 확인
- positive/negative pair 생성 확인
- TCN/GCN 모델 구조 확인
- 여러 dimension/epoch variant 학습 명령 실행
- registry 확인
- runtime smoke test
- 서비스 실행 명령 확인

노트북은 기본적으로 `RUN_TRAINING = False`로 되어 있다. 실제 학습하려면 해당 값을 `True`로 바꾸면 된다.

## 테스트

추가/수정된 테스트:

```text
tests/test_scoring.py
```

검증 내용:

- scratch comparator가 full window 전에는 `None`을 반환하는지
- full window 이후 finite similarity를 반환하는지
- registry에서 model name으로 `.tflite` 경로를 찾는지

## 주의 사항

- `scratch` 실행 전에 학습된 `.tflite` 모델이 있어야 한다.
- `--scratch-model-name`은 `data/models/scratch/model_registry.json`에 존재해야 한다.
- 학습 시 사용한 `sequence_length`, `feature_dims`, `input_layout`은 서비스 실행 설정과 맞아야 한다.
- `quick` profile은 기능 확인용이다. 서비스 후보 모델은 `full` profile로 다시 학습하는 것이 좋다.
- 현재 데이터는 reference dance 수가 제한적이므로 실제 사용자 다양성까지 잘 일반화하려면 데이터 확장이 필요하다.
