#!/usr/bin/env python3
"""
collect_dance_skeletons.py
--------------------------
유튜브 검색어 기반으로 춤 영상을 수집하고 MediaPipe 스켈레톤(npy)을 추출하는 배치 파이프라인.

Usage:
    python collect_dance_skeletons.py --queries queries.txt --per-query 10 --max-duration 120

디렉토리 구조:
    data_collection/
    ├── queries.txt              # 검색어 리스트
    ├── collect_dance_skeletons.py
    ├── videos/                  # 다운로드된 원본 영상
    ├── cropped/                 # 9:16 크롭된 영상
    └── npy/                     # 추출된 스켈레톤 (N×33×3)
"""

import argparse
import os
import subprocess
import sys
import json
import glob

# ── 기본 경로 ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(SCRIPT_DIR, "videos")
CROPPED_DIR = os.path.join(SCRIPT_DIR, "cropped")
NPY_DIR = os.path.join(SCRIPT_DIR, "npy")


def ensure_dirs():
    for d in [VIDEO_DIR, CROPPED_DIR, NPY_DIR]:
        os.makedirs(d, exist_ok=True)


# ── 1단계: 검색 & 다운로드 ───────────────────────────────────────────────────
def search_and_download(queries_file: str, per_query: int, max_duration: int):
    """검색어/URL 파일에서 쿼리를 읽고 yt-dlp로 영상을 다운로드한다.
    
    queries.txt 형식:
        # 주석
        https://www.youtube.com/results?search_query=댄스+쇼츠    ← URL 직접 입력
        kpop dance shorts                                          ← 검색어 (ytsearch 사용)
    """
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = [
            line.strip() for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    if not queries:
        print("[ERROR] queries.txt에 검색어/URL이 없습니다.")
        sys.exit(1)

    print(f"[INFO] {len(queries)}개 항목, 각 최대 {per_query}개씩 수집 시작")

    downloaded = 0
    for query in queries:
        # URL인지 검색어인지 판별
        is_url = query.startswith("http://") or query.startswith("https://")
        if is_url:
            search_term = query
            label = query[:80]
        else:
            search_term = f"ytsearch{per_query}:{query}"
            label = query
        print(f"\n[SEARCH] '{label}' → 최대 {per_query}개 수집 중...")

        cmd = [
            sys.executable, "-m", "yt_dlp",
            search_term,
            "--match-filter", f"duration<={max_duration}",
            "--format", "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "--merge-output-format", "mp4",
            "-o", os.path.join(VIDEO_DIR, "%(id)s.mp4"),
            "--no-playlist",
            "--no-overwrites",
            "--write-info-json",
            "--restrict-filenames",
            "--quiet", "--progress",
        ]
        # URL일 때는 --playlist-end로 개수 제한
        if is_url:
            cmd.extend(["--playlist-end", str(per_query)])

        try:
            subprocess.run(cmd, check=False, timeout=600)
            # 다운로드된 파일 수 갱신
            new_count = len(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
            added = new_count - downloaded
            downloaded = new_count
            print(f"  → {added}개 다운로드 (총 {downloaded}개)")
        except subprocess.TimeoutExpired:
            print(f"  [WARN] '{label}' 타임아웃, 다음으로 넘어감")
        except Exception as e:
            print(f"  [WARN] '{query}' 실패: {e}")

    print(f"\n[INFO] 다운로드 완료: {downloaded}개 영상")
    return downloaded


# ── 2단계: 9:16 크롭 ─────────────────────────────────────────────────────────
def crop_videos():
    """다운로드된 영상을 중앙 9:16으로 크롭한다."""
    videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    print(f"\n[CROP] {len(videos)}개 영상 크롭 시작")

    for i, vpath in enumerate(videos, 1):
        vid_id = os.path.splitext(os.path.basename(vpath))[0]
        out_path = os.path.join(CROPPED_DIR, f"{vid_id}.mp4")

        if os.path.exists(out_path):
            continue

        # ffprobe로 해상도 확인
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries",
                 "stream=width,height", "-of", "json", vpath],
                capture_output=True, text=True, timeout=10
            )
            info = json.loads(probe.stdout)
            stream = info["streams"][0]
            w, h = int(stream["width"]), int(stream["height"])
        except Exception:
            print(f"  [{i}] {vid_id} → ffprobe 실패, 스킵")
            continue

        # 9:16 크롭 폭 계산
        target_w = int(h * 9 / 16)
        if target_w > w:
            # 이미 세로가 긴 영상 → 그냥 리사이즈만
            vf = f"scale=406:720"
        else:
            crop_x = (w - target_w) // 2
            vf = f"crop={target_w}:{h}:{crop_x}:0,scale=406:720"

        cmd = [
            "ffmpeg", "-y", "-i", vpath,
            "-vf", vf,
            "-c:v", "libx264", "-crf", "26",
            "-an",  # 오디오 불필요
            "-preset", "fast",
            out_path,
        ]

        try:
            subprocess.run(cmd, capture_output=True, timeout=120)
            print(f"  [{i}/{len(videos)}] {vid_id} → 크롭 완료 ({w}x{h} → 406x720)")
        except Exception as e:
            print(f"  [{i}] {vid_id} → 크롭 실패: {e}")

    cropped = len(glob.glob(os.path.join(CROPPED_DIR, "*.mp4")))
    print(f"[CROP] 완료: {cropped}개")
    return cropped


# ── 3단계: MediaPipe 스켈레톤 추출 ───────────────────────────────────────────
# 얼굴 랜드마크(0~10)는 불필요 → 저장 시 0으로 마스킹
FACE_INDICES = list(range(0, 11))
# 몸통 중심 판별 기준: 양 어깨(11,12) + 양 엉덩이(23,24)의 x 평균
TORSO_INDICES = [11, 12, 23, 24]
CENTER_X_MIN = 0.25  # 프레임 중앙 허용 범위
CENTER_X_MAX = 0.75

FULLBODY_INDICES = [11, 12, 23, 24, 27, 28]  # 어깨 + 엉덩이 + 발목

def _is_centered(landmarks_arr, check_frames=10):
    """처음 감지된 N프레임의 몸통 중심이 화면 중앙 부근인지 확인."""
    import numpy as np
    detected = [lm for lm in landmarks_arr[:check_frames * 3] if np.any(lm)]
    if len(detected) < check_frames:
        detected = [lm for lm in landmarks_arr if np.any(lm)]
    if not detected:
        return False
    # 처음 감지된 check_frames개 사용
    samples = detected[:check_frames]
    for lm in samples:
        torso_x = np.mean([lm[idx][0] for idx in TORSO_INDICES])
        if torso_x < CENTER_X_MIN or torso_x > CENTER_X_MAX:
            return False
    return True


def _is_fullbody(landmarks_arr, check_frames=10, min_ratio=0.8):
    """샘플 프레임 중 min_ratio 이상에서 전신(어깨+엉덩이+발목)이 감지되는지 확인."""
    import numpy as np
    detected = [lm for lm in landmarks_arr[:check_frames * 3] if np.any(lm)]
    if len(detected) < check_frames:
        detected = [lm for lm in landmarks_arr if np.any(lm)]
    if not detected:
        return False
    samples = detected[:check_frames]
    ok = 0
    for lm in samples:
        # 전신 키포인트가 모두 유효(0이 아닌 좌표)한지 확인
        if all(lm[idx][0] > 0 and lm[idx][1] > 0 for idx in FULLBODY_INDICES):
            ok += 1
    return ok / len(samples) >= min_ratio

def extract_skeletons():
    """크롭된 영상에서 MediaPipe로 포즈 랜드마크를 추출하여 npy로 저장한다."""
    import mediapipe as mp
    import numpy as np
    import cv2

    videos = sorted(glob.glob(os.path.join(CROPPED_DIR, "*.mp4")))
    print(f"\n[SKELETON] {len(videos)}개 영상 스켈레톤 추출 시작")

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    success = 0
    skipped_center = 0
    skipped_fullbody = 0
    for i, vpath in enumerate(videos, 1):
        vid_id = os.path.splitext(os.path.basename(vpath))[0]
        npy_path = os.path.join(NPY_DIR, f"{vid_id}.npy")

        if os.path.exists(npy_path):
            success += 1
            continue

        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"  [{i}] {vid_id} → 열기 실패, 스킵")
            continue

        landmarks_list = []
        frame_count = 0
        detected_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                arr = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)
                # 얼굴 랜드마크 제거 (0으로 마스킹)
                arr[FACE_INDICES] = 0.0
                landmarks_list.append(arr)
                detected_count += 1
            else:
                # 미감지 프레임은 0으로 채움
                landmarks_list.append(np.zeros((33, 3), dtype=np.float32))

        cap.release()

        if frame_count == 0:
            print(f"  [{i}] {vid_id} → 프레임 없음, 스킵")
            continue

        detect_rate = detected_count / frame_count * 100
        # 감지율이 너무 낮으면 품질 불량으로 스킵
        if detect_rate < 30:
            print(f"  [{i}/{len(videos)}] {vid_id} → 감지율 {detect_rate:.0f}% (너무 낮음, 스킵)")
            continue

        npy_data = np.array(landmarks_list, dtype=np.float32)  # (N, 33, 3)

        # 중앙 위치 필터: 몸통 중심이 화면 중앙 부근이 아니면 스킵
        if not _is_centered(npy_data):
            skipped_center += 1
            print(f"  [{i}/{len(videos)}] {vid_id} → 중앙 아님 (스킵)")
            continue

        # 전신 필터: 어깨+엉덩이+발목이 충분히 감지되지 않으면 스킵
        if not _is_fullbody(npy_data):
            skipped_fullbody += 1
            print(f"  [{i}/{len(videos)}] {vid_id} → 전신 아님 (스킵)")
            continue

        np.save(npy_path, npy_data)
        success += 1
        print(f"  [{i}/{len(videos)}] {vid_id} → {npy_data.shape} 감지율 {detect_rate:.0f}%")

    pose.close()
    print(f"\n[SKELETON] 완료: {success}개 저장, {skipped_center}개 중앙필터 스킵, {skipped_fullbody}개 전신필터 스킵 (경로: {NPY_DIR})")
    return success


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="유튜브 춤 영상 → 스켈레톤 npy 배치 수집 파이프라인")
    parser.add_argument("--queries", "-q", default=os.path.join(SCRIPT_DIR, "queries.txt"),
                        help="검색어 파일 경로 (기본: queries.txt)")
    parser.add_argument("--per-query", "-n", type=int, default=10,
                        help="검색어당 수집할 영상 수 (기본: 10)")
    parser.add_argument("--max-duration", "-d", type=int, default=120,
                        help="최대 영상 길이 초 (기본: 120)")
    parser.add_argument("--skip-download", action="store_true",
                        help="다운로드 건너뛰고 크롭+추출만")
    parser.add_argument("--skip-crop", action="store_true",
                        help="크롭 건너뛰고 추출만")
    parser.add_argument("--only-extract", action="store_true",
                        help="스켈레톤 추출만 (cropped/ 폴더에 영상 있어야 함)")

    args = parser.parse_args()
    ensure_dirs()

    if args.only_extract:
        extract_skeletons()
        return

    if not args.skip_download:
        search_and_download(args.queries, args.per_query, args.max_duration)

    if not args.skip_crop:
        crop_videos()

    extract_skeletons()

    # 결과 요약
    n_videos = len(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    n_cropped = len(glob.glob(os.path.join(CROPPED_DIR, "*.mp4")))
    n_npy = len(glob.glob(os.path.join(NPY_DIR, "*.npy")))
    print(f"\n{'='*50}")
    print(f"  수집 완료 요약")
    print(f"  원본 영상:    {n_videos}개")
    print(f"  크롭 영상:    {n_cropped}개")
    print(f"  스켈레톤 npy: {n_npy}개")
    print(f"  npy 경로:     {NPY_DIR}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
