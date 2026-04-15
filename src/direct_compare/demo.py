#!/usr/bin/env python3
"""
Direct Compare 데모
레퍼런스 영상에서 추출된 포즈 시퀀스와 웹캠 포즈를 실시간 비교합니다.

사용법:
  1단계 - 레퍼런스 추출:
    python demo.py extract reference_video.avi
    → reference_video.npy 로 포즈 시퀀스 저장

  2단계 - 실시간 비교:
    python demo.py play reference_video.npy
    → SPACE로 시작, 레퍼런스 타임라인에 맞춰 웹캠 포즈와 비교

  단축 (추출 + 바로 플레이):
    python demo.py reference_video.avi
"""

import sys
import os
import time
import threading
import termios
import tty
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from pose.detector import PoseDetector
from direct_compare.pose_similarity import PoseSimilarity


class KeyboardReader:
    def __init__(self):
        self._key_buffer = []
        self._lock = threading.Lock()
        self._running = True
        self._old_settings = None
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)

    def start(self):
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self._thread.start()

    def stop(self):
        self._running = False
        if self._old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def _reader_loop(self):
        while self._running:
            try:
                ch = sys.stdin.read(1)
                if ch:
                    with self._lock:
                        self._key_buffer.append(ch)
            except Exception:
                break

    def get_key(self):
        with self._lock:
            if self._key_buffer:
                return self._key_buffer.pop(0)
        return None


# ──────────────────────────────────────────────
#  1) 레퍼런스 영상에서 포즈 시퀀스 추출
# ──────────────────────────────────────────────
def extract_reference(video_path: str, out_path: str = None, model: str = "movenet"):
    """레퍼런스 영상의 모든 프레임에서 포즈를 추출하여 .npy로 저장"""
    if out_path is None:
        out_path = os.path.splitext(video_path)[0] + ".npy"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 영상을 열 수 없습니다: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"레퍼런스 영상: {video_path}")
    print(f"  FPS: {fps:.1f}, 총 프레임: {total}")

    print(f"{model} 모델 로딩 중...")
    detector = PoseDetector(backend=model)
    detector.initialize()

    poses = []       # (N, 33, 4)
    detected = []    # (N,) bool - 각 프레임 검출 여부
    count = 0

    print("포즈 추출 중...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect(frame)
        if result["detected"]:
            poses.append(result["landmarks"])
            detected.append(True)
        else:
            # 미검출 프레임은 빈 배열로 채움
            poses.append(np.zeros((33, 4), dtype=np.float32))
            detected.append(False)

        count += 1
        if count % 30 == 0:
            pct = count / total * 100 if total > 0 else 0
            print(f"  {count}/{total} ({pct:.0f}%)")

    cap.release()
    detector.release()

    # 메타데이터와 함께 저장
    data = {
        "poses": np.array(poses, dtype=np.float32),       # (N, 33, 4)
        "detected": np.array(detected, dtype=bool),        # (N,)
        "fps": fps,
        "source": os.path.basename(video_path),
        "model": model,
    }
    np.save(out_path, data, allow_pickle=True)
    print(f"\n저장 완료: {out_path}")
    print(f"  추출 프레임: {count}, 검출 성공: {sum(detected)}/{count}")
    return out_path


# ──────────────────────────────────────────────
#  2) 실시간 비교 플레이
# ──────────────────────────────────────────────
def play_compare(npy_path: str, model: str = None):
    """저장된 레퍼런스 포즈 시퀀스와 웹캠 포즈를 실시간 비교"""

    # 레퍼런스 로드
    data = np.load(npy_path, allow_pickle=True).item()
    ref_poses = data["poses"]         # (N, 33, 4)
    ref_detected = data["detected"]   # (N,)
    ref_fps = data["fps"]
    ref_source = data.get("source", "unknown")
    ref_model = data.get("model", "movenet")
    ref_total = len(ref_poses)

    # 모델 미지정이면 레퍼런스 추출에 사용한 모델과 동일하게
    if model is None:
        model = ref_model

    print("=" * 60)
    print("Direct Compare - 레퍼런스 영상 기반 실시간 비교")
    print("=" * 60)
    print(f"\n레퍼런스: {ref_source} (추출 모델: {ref_model})")
    print(f"  프레임: {ref_total}, FPS: {ref_fps:.1f}")
    print(f"  길이: {ref_total / ref_fps:.1f}초")
    print(f"웹캠 모델: {model}")
    print("\n조작:")
    print("  SPACE: 시작 / 재시작")
    print("  q: 종료\n")

    # 초기화
    print(f"{model} 모델 로딩 중...")
    detector = PoseDetector(backend=model)
    detector.initialize()
    comparator = PoseSimilarity(use_key_joints_only=True, normalize=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    kb = KeyboardReader()
    kb.start()

    CONNECTIONS = [
        (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 12), (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (24, 26), (26, 28),
    ]
    JOINT_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    playing = False
    start_time = 0.0
    scores = []         # 프레임별 유사도 기록
    current_sim = 0.0

    def draw_skeleton(img, lm, connections, joints, color, thickness=2, radius=4):
        """스켈레톤을 img 위에 그리기 (lm은 이미 거울모드 적용된 상태)"""
        h, w = img.shape[:2]
        for (i, j) in connections:
            x1, y1, _, v1 = lm[i]
            x2, y2, _, v2 = lm[j]
            if v1 > 0.3 and v2 > 0.3:
                pt1 = (int(x1 * w), int(y1 * h))
                pt2 = (int(x2 * w), int(y2 * h))
                cv2.line(img, pt1, pt2, color, thickness)
        for idx in joints:
            x, y, _, v = lm[idx]
            if v > 0.3:
                cv2.circle(img, (int(x * w), int(y * h)), radius, color, -1)

    print("준비 완료! SPACE를 눌러 시작하세요.\n")

    try:
        while True:
            key_ch = kb.get_key()
            if key_ch == 'q':
                break

            # SPACE: 시작 / 재시작
            if key_ch == ' ':
                playing = True
                start_time = time.time()
                scores = []
                current_sim = 0.0
                print("▶ 시작!")

            ret, frame = cap.read()
            if not ret:
                break

            result = detector.detect(frame)
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # 현재 레퍼런스 프레임 인덱스 계산
            if playing:
                elapsed = time.time() - start_time
                ref_idx = int(elapsed * ref_fps)

                if ref_idx >= ref_total:
                    # 끝남 → 결과 표시
                    playing = False
                    if scores:
                        avg = sum(scores) / len(scores)
                        print(f"\n■ 종료! 평균 유사도: {avg*100:.1f}%")
                        print(f"  비교 프레임: {len(scores)}")
                    ref_idx = ref_total - 1

            # ── 스켈레톤 & 유사도 ──
            if playing or (not playing and scores):
                ref_idx_show = ref_idx if playing else ref_total - 1

                # 레퍼런스 스켈레톤 (보라색)
                if ref_detected[ref_idx_show]:
                    ref_lm = ref_poses[ref_idx_show].copy()
                    ref_lm[:, 0] = 1.0 - ref_lm[:, 0]  # 거울모드
                    draw_skeleton(frame, ref_lm, CONNECTIONS, JOINT_INDICES,
                                  (255, 0, 255), thickness=3, radius=5)

                # 현재 포즈 (초록)
                if result["detected"]:
                    lm = result["landmarks"].copy()
                    lm[:, 0] = 1.0 - lm[:, 0]
                    draw_skeleton(frame, lm, CONNECTIONS, JOINT_INDICES,
                                  (0, 255, 128), thickness=2, radius=4)

                    # 유사도 계산
                    if playing and ref_detected[ref_idx]:
                        current_sim = comparator.cosine_similarity(
                            result["landmarks"], ref_poses[ref_idx])
                        scores.append(current_sim)

                # 유사도 바
                bar_w, bar_h = 250, 22
                bar_x, bar_y = 10, 65
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                fill = int(bar_w * current_sim)
                bar_color = (0, int(255 * current_sim), int(255 * (1 - current_sim)))
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + fill, bar_y + bar_h), bar_color, -1)
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
                cv2.putText(frame, f"Similarity: {current_sim*100:.1f}%",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)

                # 평균 유사도
                if scores:
                    avg = sum(scores) / len(scores)
                    cv2.putText(frame, f"Avg: {avg*100:.1f}%",
                                (bar_x + bar_w + 10, bar_y + 17),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                # 진행률
                if playing:
                    progress = (ref_idx + 1) / ref_total
                    pg_y = bar_y + bar_h + 10
                    cv2.rectangle(frame, (bar_x, pg_y),
                                  (bar_x + bar_w, pg_y + 6), (50, 50, 50), -1)
                    cv2.rectangle(frame, (bar_x, pg_y),
                                  (bar_x + int(bar_w * progress), pg_y + 6),
                                  (255, 200, 0), -1)
                    elapsed_s = time.time() - start_time
                    total_s = ref_total / ref_fps
                    cv2.putText(frame, f"{elapsed_s:.1f}s / {total_s:.1f}s",
                                (bar_x, pg_y + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            elif not playing and not scores:
                # 대기 상태: 포즈만 표시
                if result["detected"]:
                    lm = result["landmarks"].copy()
                    lm[:, 0] = 1.0 - lm[:, 0]
                    draw_skeleton(frame, lm, CONNECTIONS, JOINT_INDICES,
                                  (0, 255, 128), thickness=2, radius=4)
                cv2.putText(frame, "Press SPACE to start", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            # 결과 화면
            if not playing and scores:
                avg = sum(scores) / len(scores)
                grade = ("Perfect" if avg >= 0.9 else
                         "Great" if avg >= 0.75 else
                         "Good" if avg >= 0.6 else
                         "OK" if avg >= 0.4 else "Miss")
                cv2.putText(frame, grade, (w // 2 - 80, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 3)
                cv2.putText(frame, "SPACE: retry | q: quit",
                            (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            else:
                cv2.putText(frame, "SPACE: start | q: quit",
                            (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            # 범례
            cv2.putText(frame, "You", (w - 80, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 2)
            cv2.putText(frame, "Ref", (w - 80, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            cv2.imshow("Direct Compare", frame)
            cv2.waitKey(1)

    finally:
        kb.stop()
        cap.release()
        cv2.destroyAllWindows()
        detector.release()


# ──────────────────────────────────────────────
#  메인
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="레퍼런스 영상 기반 포즈 유사도 비교 데모")
    parser.add_argument("command", nargs="?", default=None,
                        help="extract / play / 또는 영상파일 경로(자동 추출+플레이)")
    parser.add_argument("file", nargs="?", default=None,
                        help="영상(.avi/.mp4) 또는 포즈 파일(.npy)")
    parser.add_argument("--model", "-m", choices=["movenet", "mediapipe"],
                        default="movenet",
                        help="포즈 추출 모델 선택 (기본: movenet)")
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print("\n사용 예:")
        print("  python demo.py extract dance.avi                # movenet으로 추출")
        print("  python demo.py extract dance.avi -m mediapipe   # mediapipe로 추출")
        print("  python demo.py play dance.npy                   # .npy로 실시간 비교")
        print("  python demo.py play dance.npy -m mediapipe      # 웹캠을 mediapipe로")
        print("  python demo.py dance.avi                        # 추출 + 바로 플레이")
        return

    # extract 모드
    if args.command == "extract":
        if not args.file:
            print("[ERROR] 영상 파일을 지정해주세요.")
            return
        extract_reference(args.file, model=args.model)
        return

    # play 모드
    if args.command == "play":
        if not args.file:
            print("[ERROR] .npy 파일을 지정해주세요.")
            return
        play_compare(args.file, model=args.model)
        return

    # 단축: 영상 파일 직접 지정 → 추출 + 플레이
    video_path = args.command
    if not os.path.isfile(video_path):
        print(f"[ERROR] 파일을 찾을 수 없습니다: {video_path}")
        return

    if video_path.endswith(".npy"):
        play_compare(video_path, model=args.model)
    else:
        npy_path = extract_reference(video_path, model=args.model)
        if npy_path:
            play_compare(npy_path)


if __name__ == "__main__":
    main()
