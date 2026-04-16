"""
process_video.py
----------------
배경을 흰색으로, 사람을 회색 실루엣으로 변환하고 포즈 스켈레톤을 오버레이합니다.
포즈 관절 위치 정보는 NumPy (.npy) 파일로도 저장됩니다.

Usage (MediaPipe, 기본):
    python process_video.py input.mp4 --model mediapipe

Usage (MoveNet):
    python process_video.py input.mp4 --model movenet

의존성:
    MediaPipe 모드: uv run --python 3.10 --with opencv-python --with numpy --with "mediapipe==0.10.9" python3 process_video.py ...
    MoveNet 모드:   uv run --with opencv-python --with numpy --with tensorflow --with tensorflow-hub python3 process_video.py --model movenet ...
"""

import cv2
import numpy as np
import os
import argparse
import subprocess

# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe 관련 상수
# ──────────────────────────────────────────────────────────────────────────────
MEDIAPIPE_NUM_LANDMARKS = 33

# ──────────────────────────────────────────────────────────────────────────────
# MoveNet 관련 상수 (Lightning 17 keypoints)
# ──────────────────────────────────────────────────────────────────────────────
MOVENET_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
MOVENET_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # face
    (5, 6),                                    # shoulders
    (5, 7), (7, 9),                            # left arm
    (6, 8), (8, 10),                           # right arm
    (5, 11), (6, 12),                          # torso sides
    (11, 12),                                  # hips
    (11, 13), (13, 15),                        # left leg
    (12, 14), (14, 16),                        # right leg
]
MOVENET_NUM_LANDMARKS = 17
MOVENET_INPUT_SIZE = 192  # Lightning 모델 입력 크기


# ══════════════════════════════════════════════════════════════════════════════
# MediaPipe 처리
# ══════════════════════════════════════════════════════════════════════════════

def run_mediapipe(input_path: str, tmp_output_path: str) -> np.ndarray:
    """
    MediaPipe Pose + Segmentation으로 실루엣 & 스켈레톤 영상을 생성하고,
    프레임별 랜드마크 배열 (N, 33, 3) [x, y, visibility]를 반환합니다.
    """
    import mediapipe as mp

    cap = cv2.VideoCapture(input_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tmp_output_path, fourcc, fps, (width, height))

    mp_pose          = mp.solutions.pose
    mp_drawing       = mp.solutions.drawing_utils
    mp_drawing_styles= mp.solutions.drawing_styles

    all_landmarks = []   # list of (33, 3) arrays

    print(f"[MediaPipe] Processing {frame_count} frames...")
    count = 0

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=True) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = pose.process(frame_rgb)

            # 흰색 배경
            result_frame = np.full(frame.shape, 255, dtype=np.uint8)

            # 세그멘테이션 마스크로 회색 실루엣 적용
            if results.segmentation_mask is not None:
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5
                gray      = np.full(frame.shape, 128, dtype=np.uint8)
                result_frame = np.where(condition, gray, result_frame)

            # 랜드마크 수집 (x, y, visibility) — 정규화 좌표 (0.0~1.0)
            if results.pose_landmarks:
                lm_array = np.array(
                    [[lm.x, lm.y, lm.visibility]
                     for lm in results.pose_landmarks.landmark],
                    dtype=np.float32,
                )
                mp_drawing.draw_landmarks(
                    result_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )
            else:
                lm_array = np.zeros((MEDIAPIPE_NUM_LANDMARKS, 3), dtype=np.float32)

            all_landmarks.append(lm_array)
            out.write(result_frame)

            count += 1
            if count % 100 == 0:
                print(f"  Processed {count}/{frame_count} frames...")

    cap.release()
    out.release()

    return np.stack(all_landmarks, axis=0)   # (N, 33, 3)


# ══════════════════════════════════════════════════════════════════════════════
# MoveNet 처리
# ══════════════════════════════════════════════════════════════════════════════

def _draw_movenet_skeleton(frame: np.ndarray, keypoints: np.ndarray,
                           threshold: float = 0.3):
    """
    keypoints: (17, 3) → [x, y, score] (정규화 좌표)
    """
    h, w = frame.shape[:2]
    points_px = []

    for x, y, score in keypoints:
        px = int(x * w)
        py = int(y * h)
        points_px.append((px, py, float(score)))
        if score > threshold:
            cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)

    for (i, j) in MOVENET_CONNECTIONS:
        _, _, si = points_px[i]
        _, _, sj = points_px[j]
        if si > threshold and sj > threshold:
            cv2.line(frame,
                     (points_px[i][0], points_px[i][1]),
                     (points_px[j][0], points_px[j][1]),
                     (0, 255, 0), 2)


def run_movenet(input_path: str, tmp_output_path: str) -> np.ndarray:
    """
    TF Hub MoveNet Lightning으로 포즈 추출 후 실루엣 & 스켈레톤 영상을 생성하고,
    프레임별 랜드마크 배열 (N, 17, 3) [x, y, score]를 반환합니다.
    """
    import tensorflow as tf
    import tensorflow_hub as hub
    from rembg import remove
    from PIL import Image
    import io

    print("[MoveNet] Loading model from TF Hub...")
    model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
    module    = hub.load(model_url)
    movenet   = module.signatures["serving_default"]

    cap = cv2.VideoCapture(input_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tmp_output_path, fourcc, fps, (width, height))

    all_landmarks = []
    print(f"[MoveNet] Processing {frame_count} frames...")
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ── 1. 배경 제거 (rembg) ──────────────────────────────────────────────
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(frame_rgb)
        buf       = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        removed   = remove(buf.read())
        pil_rgba  = Image.open(io.BytesIO(removed)).convert("RGBA")
        arr_rgba  = np.array(pil_rgba)

        alpha_mask   = arr_rgba[:, :, 3] > 10
        result_frame = np.full((height, width, 3), 255, dtype=np.uint8)
        result_frame[alpha_mask] = 128

        # ── 2. MoveNet 추론 ───────────────────────────────────────────────────
        img_tf  = tf.image.resize_with_pad(
                      tf.expand_dims(tf.cast(frame_rgb, tf.int32), axis=0),
                      MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE)
        img_tf  = tf.cast(img_tf, tf.int32)
        outputs = movenet(input=img_tf)
        
        # 모델 출력은 [y, x, score] 임. 이를 [x, y, score]로 스왑하여 저장.
        kps_raw = outputs["output_0"].numpy()[0, 0]   # (17, 3) [y, x, score]
        kps_swapped = kps_raw[:, [1, 0, 2]]           # (17, 3) [x, y, score]

        # ── 3. 스켈레톤 그리기 ────────────────────────────────────────────────
        _draw_movenet_skeleton(result_frame, kps_swapped)

        all_landmarks.append(kps_swapped.astype(np.float32))
        out.write(result_frame)

        count += 1
        if count % 100 == 0:
            print(f"  Processed {count}/{frame_count} frames...")

    cap.release()
    out.release()

    return np.stack(all_landmarks, axis=0)   # (N, 17, 3)


# ══════════════════════════════════════════════════════════════════════════════
# ffmpeg 후처리 압축
# ══════════════════════════════════════════════════════════════════════════════

def compress_video(tmp_path: str, final_path: str, orig_path: str, crf: int = 28):
    """
    ffmpeg으로 tmp_path → final_path 로 H.264/AAC 압축.
    원본 영상(orig_path)에서 오디오를 추출하여 결과물에 포함합니다.
    CRF 값이 클수록 압축률↑ (품질↓). 일반적으로 23~28 권장.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_path,
        "-i", orig_path,
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",  # AAC 코덱으로 오디오 인코딩 (또는 copy)
        "-map", "0:v:0",  # 첫 번째 입력(tmp_path)에서 비디오 가져오기
        "-map", "1:a:0?",  # 두 번째 입력(orig_path)에서 오디오 가져오기 (오디오가 없어도 에러 무시)
        final_path
    ]
    print(f"[ffmpeg] Compressing: {tmp_path} → {final_path} (CRF={crf}, Audio included)")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[ffmpeg] 압축 실패, 원본 tmp 파일을 그대로 사용합니다.")
        print(result.stderr)
        os.rename(tmp_path, final_path)
    else:
        os.remove(tmp_path)
        print(f"[ffmpeg] 압축 완료 → {final_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="실루엣 + 포즈 스켈레톤 영상 생성 및 NumPy 포즈 데이터 추출"
    )
    parser.add_argument("input_video", help="입력 영상 경로")
    parser.add_argument("--output_video", default=None,
                        help="출력 영상 경로 (미지정 시 자동 생성)")
    parser.add_argument("--model", choices=["mediapipe", "movenet"],
                        default="mediapipe",
                        help="포즈 추출 모델 선택 (기본값: mediapipe)")
    parser.add_argument("--crf", type=int, default=28,
                        help="ffmpeg CRF 압축 품질 (낮을수록 고품질, 기본: 28)")
    args = parser.parse_args()

    input_video = args.input_video
    base, _     = os.path.splitext(input_video)
    suffix      = f"_{args.model}_processed"

    output_video = args.output_video or f"{base}{suffix}.mp4"
    output_npy   = f"{base}{suffix}_pose.npy"
    tmp_video    = f"{base}{suffix}_tmp.mp4"

    out_dir = os.path.dirname(output_video)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"입력  : {input_video}")
    print(f"출력  : {output_video}")
    print(f"NumPy : {output_npy}")
    print(f"모델  : {args.model.upper()}")
    print("=" * 60)

    # ── 포즈 추출 & 실루엣 렌더링 ────────────────────────────────────────────
    if args.model == "mediapipe":
        landmarks = run_mediapipe(input_video, tmp_video)
    else:
        landmarks = run_movenet(input_video, tmp_video)

    # ── ffmpeg 압축 ──────────────────────────────────────────────────────────
    compress_video(tmp_video, output_video, input_video, crf=args.crf)

    # ── NumPy 저장 ───────────────────────────────────────────────────────────
    np.save(output_npy, landmarks)
    shape_str = " × ".join(map(str, landmarks.shape))
    print(f"[NumPy] Saved pose array ({shape_str}) → {output_npy}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
