"""
Game Engine - Main game loop, state management, and component orchestration.
"""

import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class GameState:
    """Enumeration of game states."""
    MENU = "menu"
    SONG_SELECT = "song_select"
    READY = "ready"          # 준비 화면 (포즈 감지 + OK 사인 대기)
    COUNTDOWN = "countdown"
    PLAYING = "playing"
    PAUSED = "paused"
    RESULT = "result"
    SETTINGS = "settings"


class GameEngine:
    """
    Central game engine that manages the game loop, state transitions,
    and coordinates all subsystems (camera, pose, embedding, scoring, UI).
    """

    TARGET_FPS = 30

    def __init__(self, config: dict):
        self.config = config
        self.state = GameState.MENU
        self._prev_state = None
        self.running = False
        self._camera = None
        self._pose_detector = None
        self._embedding_extractor = None
        self._embedding_comparator = None
        self._scratch_comparator = None
        self._scratch_model_name = None
        self._embedding_model_name = None
        self._scratch_reference_cache_dir = "data/cache/reference_embeddings"
        self._embedding_reference_cache_dir = "data/cache/reference_embeddings"
        self._scratch_use_reference_cache = True
        self._embedding_use_reference_cache = True
        self._fallback_pose_comparator = None
        self._pose_hold_frames = 6
        self._last_valid_landmarks = None
        self._last_valid_landmarks_age_frames = 10**9
        self._score_hold_seconds = 0.3
        self._model_warmup_direct_fallback = True
        self._fallback_similarity_method = "angle"
        self._warned_scratch_no_ref = False
        self._scorer = None
        self._ui = None
        self._current_session = None
        self._clock = None
        self._display = None
        self._feedback_gen = None
        self._last_feedback = None
        self._countdown_timer = 0
        self._countdown_start = 0
        self._result_data = None
        # 터치/마우스 클릭용 버튼 rect 저장소
        self._btn_rects = {}
        # 곡 선택 관련
        self._songs: list = []
        self._selected_song_idx: int = 0
        self._current_mode: str = "practice"   # practice | challenge | freestyle
        self._current_song: dict = {}
        # 가이드 캐릭터용 참조 랜드마크
        self._ref_landmarks = None          # shape (N, 33, 4)
        self._ref_frame_landmarks = None    # shape (33, 4) — 현재 프레임
        self._ref_current_idx = 0           # 현재 참조 프레임 인덱스
        # 레퍼런스 영상 (mp4)
        self._ref_video_cap = None          # cv2.VideoCapture
        self._ref_video_fps: float = 30.0
        self._ref_video_frame = None        # 현재 프레임 (numpy BGR)
        self._ref_video_surf = None         # 캐시된 pygame Surface
        self._ref_video_size: tuple = (0, 0)  # 미리 계산된 리사이즈 크기
        self._ref_video_pos: tuple = (0, 0)   # 패널 내 배치 좌표
        # 피드백 이펙트 페이드 타이머 (초)
        self._feedback_timer: float = 0.0
        # 실루엣 렌더러 (사람 형태 캐릭터)
        self._user_silhouette = None
        self._guide_silhouette = None
        # READY 상태 관련
        self._ready_current_frame = None     # 웹캠 프레임
        self._ready_landmarks = None         # 포즈 랜드마크
        self._ready_pose_detected = False
        self._ready_full_body_start: float = 0.0  # 전신 감지 시작 시각
        self._ready_countdown: float = 0.0        # 전신 감지 후 카운트다운 (초)
        # 키보드/마우스 포커스 (메뉴 네비게이션)
        self._menu_focus_idx: int = 0        # 메뉴 화면 포커스 버튼 인덱스
        self._song_focus_idx: int = 0        # 곡 선택 화면 포커스 인덱스
        self._neon_tick: float = 0.0         # 네온 깜빡임 타이머
        # 터치 더블탭 지원: 첫 탭=포커스, 두 번째 탭=실행
        self._touch_focused_btn: str = ""    # 현재 터치-포커스된 버튼 이름
        self._generic_focus_idx: int = 0     # PAUSED/RESULT/SETTINGS 화면 포커스 인덱스
        self._last_event_type: str = "mouse" # 마지막 입력 이벤트 타입 ('mouse' | 'touch')
        # 파티클 이펙트 (피드백 폭발 효과)
        self._particles: list = []           # [{'x','y','vx','vy','life','max_life','color','size'}]
        # 피드백 등장 후 경과 시간 (판정 애니메이션 progress 계산용)
        self._feedback_age: float = 0.0      # 현재 피드백이 표시된 후 흐른 시간(초)

    def initialize(self):
        """
        Initialize all game subsystems.
        Called once at startup.
        """
        import pygame
        import cv2

        pygame.init()
        # 기본 버퍼(4096)는 ~100ms의 레이턴시를 유발하여 영상이 소리에 비해 먼저 나오는 느낌을 줍니다.
        # 지연을 최소화하기 위해 버퍼를 512로 줄여서 선제 초기화합니다.
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
        # 키보드 반복 입력: 200ms 후 첫 반복, 이후 80ms 간격
        pygame.key.set_repeat(200, 80)

        # Display setup
        ui_cfg = self.config.get("ui", {})
        w = ui_cfg.get("window_width", 1024)
        h = ui_cfg.get("window_height", 600)
        flags = pygame.FULLSCREEN if ui_cfg.get("fullscreen", False) else 0
        self._display = pygame.display.set_mode((w, h), flags)
        pygame.display.set_caption("Let's Dance!")
        self._clock = pygame.time.Clock()

        # Camera & Pose Detector (Async Thread)
        cam_cfg = self.config.get("camera", {})
        pose_cfg = self.config.get("pose", {})
        backend = self.config.get("pose_backend", "mediapipe")

        from pose.detector import PoseDetector
        pose_detector = PoseDetector(
            backend=backend,
            model_complexity=pose_cfg.get("model_complexity", 1),
            min_detection_confidence=pose_cfg.get("min_detection_confidence", 0.5),
            min_tracking_confidence=pose_cfg.get("min_tracking_confidence", 0.5),
        )
        pose_detector.initialize()

        # [MediaPipe 초기 렉 방지 워밍업]
        # MediaPipe는 첫 추론 시 그래프 컴파일을 위해 일시적으로 CPU를 최대치로 점유합니다.
        # 인게임(메뉴) 화면 진입 후 백그라운드 스레드에서 이것이 발생하면 화면 전체가 1~2초 멈추므로
        # 미리 더미 프레임으로 파이프라인을 뚫어두어 렉을 사전에 제거합니다.
        import numpy as np
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pose_detector.detect(dummy_frame)

        from utils.async_camera import AsyncCameraPose
        self._async_camera = AsyncCameraPose(
            camera_idx=cam_cfg.get("device_id", 0),
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            pose_detector=pose_detector,
        )
        self._async_camera.start()

        self._camera = None  # Deprecated
        self._pose_detector = None  # Deprecated

        # Scorer & Feedback
        from scoring.scorer import DanceScorer
        from scoring.feedback import FeedbackGenerator
        self._scorer = DanceScorer(
            **{k: v for k, v in self.config.get("scoring", {}).items()
               if k in ("score_scale", "combo_multiplier", "grade_thresholds")}
        )
        self._feedback_gen = FeedbackGenerator()

        smoothing_cfg = self.config.get("runtime_smoothing", {})
        self._pose_hold_frames = int(smoothing_cfg.get("pose_hold_frames", 6))
        self._score_hold_seconds = float(smoothing_cfg.get("score_hold_seconds", 0.3))
        self._model_warmup_direct_fallback = bool(
            smoothing_cfg.get("model_warmup_direct_fallback", True))
        self._fallback_similarity_method = smoothing_cfg.get(
            "fallback_similarity", self.config.get("similarity_method", "angle"))

        # Pose similarity comparator
        from direct_compare.pose_similarity import PoseSimilarity
        self._score_method = self.config.get("score_method", "direct")
        if self._score_method == "direct":
            self._pose_comparator = PoseSimilarity(use_key_joints_only=True, normalize=True)
            self._fallback_pose_comparator = self._pose_comparator
            self._similarity_method = self.config.get("similarity_method", "cosine")
            self._tolerance_delay = self.config.get("tolerance_delay", 1.0)
            self._embedding_extractor = None
            print(f"[INFO] Score method: direct ({self._similarity_method} similarity, tolerance {self._tolerance_delay}s)")
        elif self._score_method == "scratch":
            from scoring.scratch_similarity import (
                ScratchPoseSimilarity,
                resolve_scratch_model_path,
            )
            scratch_cfg = self.config.get("scratch", {})
            model_dir = scratch_cfg.get("model_dir", "data/models/scratch")
            model_name = scratch_cfg.get("model_name", "gcn_e64")
            self._scratch_model_name = model_name
            self._scratch_use_reference_cache = scratch_cfg.get("use_reference_cache", True)
            self._scratch_reference_cache_dir = scratch_cfg.get(
                "reference_cache_dir", "data/cache/reference_embeddings")
            model_path = resolve_scratch_model_path(
                model_name=model_name,
                model_dir=model_dir,
                model_path=scratch_cfg.get("model_path"),
            )
            self._scratch_comparator = ScratchPoseSimilarity(
                model_path=model_path,
                sequence_length=scratch_cfg.get("sequence_length", 30),
                feature_dims=scratch_cfg.get("feature_dims", 2),
                input_layout=scratch_cfg.get("input_layout", "BTJC"),
                top_k=scratch_cfg.get("top_k", 3),
                candidate_stride=scratch_cfg.get("candidate_stride", 3),
            )
            self._pose_comparator = None
            self._fallback_pose_comparator = PoseSimilarity(use_key_joints_only=True, normalize=True)
            self._embedding_extractor = None
            self._similarity_method = "scratch"
            self._tolerance_delay = self.config.get("tolerance_delay", 1.0)
            print(f"[INFO] Score method: scratch ({model_name}: {model_path}, tolerance {self._tolerance_delay}s)")
        else:
            # embedding: fine-tuned TFLite encoder + cosine similarity.
            from scoring.scratch_similarity import (
                ScratchPoseSimilarity,
                resolve_scratch_model_path,
            )
            embedding_cfg = self.config.get("embedding", {})
            model_dir = embedding_cfg.get("model_dir", "data/models/embedding")
            model_name = embedding_cfg.get("model_name", "dance_embedding")
            self._embedding_model_name = model_name
            self._embedding_use_reference_cache = embedding_cfg.get("use_reference_cache", True)
            self._embedding_reference_cache_dir = embedding_cfg.get(
                "reference_cache_dir", "data/cache/reference_embeddings")
            model_path = resolve_scratch_model_path(
                model_name=model_name,
                model_dir=model_dir,
                model_path=embedding_cfg.get("model_path"),
            )
            self._embedding_comparator = ScratchPoseSimilarity(
                model_path=model_path,
                sequence_length=embedding_cfg.get("sequence_length", 30),
                feature_dims=embedding_cfg.get("feature_dims", 2),
                input_layout=embedding_cfg.get("input_layout", "BTJC"),
                top_k=embedding_cfg.get("top_k", 3),
                candidate_stride=embedding_cfg.get("candidate_stride", 3),
            )
            self._pose_comparator = None
            self._fallback_pose_comparator = PoseSimilarity(use_key_joints_only=True, normalize=True)
            self._embedding_extractor = None
            self._similarity_method = "embedding"
            self._tolerance_delay = self.config.get("tolerance_delay", 1.0)
            print(f"[INFO] Score method: embedding ({model_name}: {model_path}, tolerance {self._tolerance_delay}s)")

        # ── 폰트 로드 (한국어 지원: NotoSansCJK → fallback SysFont) ──
        self._fonts = self._load_fonts(pygame)

        # ── 댄스 곡 목록 로드 ──
        self._songs = self._load_songs()

        # ── 실루엣 렌더러는 사용하지 않음 (스틱 피겨로 대체 — 성능 최적화) ──
        # 사용하지 않지만 호환성을 위해 None 유지
        self._user_silhouette = None
        self._guide_silhouette = None

        self.running = True

        # ── 터미널 stdin 키보드 → pygame 이벤트 브리지 ──────────────
        # pygame 창에 마우스/키보드 포커스가 없을 때,
        # 터미널에서 입력한 키를 pygame 이벤트 큐로 주입한다.
        # self.running = True 이후에 시작해야 루프가 즉시 종료되지 않는다.
        self._start_stdin_key_bridge()

    @staticmethod
    def _load_fonts(pygame):
        """폰트 로드.
        - 영문/숫자/ASCII: PressStart2P.ttf (레트로 픽셀 감성)
        - 한글 포함 텍스트: NotoSansKR.ttf (한글 지원)
        두 폰트 모두 assets/fonts/ 에 번들로 포함.
        """
        import os

        try:
            engine_dir   = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(engine_dir))
            fonts_dir    = os.path.join(project_root, "assets", "fonts")
        except Exception:
            fonts_dir = ""

        retro_path = os.path.join(fonts_dir, "PressStart2P.ttf") if fonts_dir else ""
        korean_path = os.path.join(fonts_dir, "NotoSansKR.ttf") if fonts_dir else ""

        # 시스템 한글 폰트 후보 (번들 폰트 없을 시 fallback)
        system_korean = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/Library/Fonts/NotoSansKR-Regular.otf",
            "C:/Windows/Fonts/malgun.ttf",
        ]

        if not os.path.exists(retro_path):
            retro_path = None
            print("[FONT] PressStart2P.ttf 없음 — SysFont fallback")
        else:
            print(f"[FONT] 레트로 폰트: {retro_path}")

        if not os.path.exists(korean_path):
            korean_path = None
            for p in system_korean:
                if os.path.exists(p):
                    korean_path = p
                    break
            if korean_path:
                print(f"[FONT] 한글 폰트: {korean_path}")
            else:
                print("[FONT] 한글 폰트를 찾지 못했습니다.")

        def make_retro(size):
            """영문/숫자용 레트로 픽셀 폰트."""
            if retro_path:
                try:
                    return pygame.font.Font(retro_path, size)
                except Exception as e:
                    print(f"[FONT] 레트로 폰트 로드 실패: {e}")
            return pygame.font.SysFont(None, size)

        def make_korean(size, bold=False):
            """한글 포함 텍스트용 폰트."""
            if korean_path:
                try:
                    return pygame.font.Font(korean_path, size)
                except Exception as e:
                    print(f"[FONT] 한글 폰트 로드 실패: {e}")
            for name in ["notosanscjkkr", "notosanscjk", "nanum gothic", "malgun gothic", "sans"]:
                try:
                    f = pygame.font.SysFont(name, size, bold=bold)
                    if f:
                        return f
                except Exception:
                    pass
            return pygame.font.SysFont(None, size)

        return {
            # 레트로 픽셀 (영문/숫자 전용): 타이틀, 점수, 피드백 등급
            "title":        make_retro(28),    # 메뉴 타이틀 "Let's Dance!"
            "score":        make_retro(28),    # 게임 중 점수
            "feedback":     make_retro(72),    # PERFECT / GREAT 등급 텍스트 (크게)
            "countdown":    make_retro(72),    # 카운트다운 숫자
            "result_big":   make_retro(22),    # 결과 화면 제목
            "small_retro":  make_retro(14),    # 소형 레이블 (ME / GUIDE / 버튼 등)
            "btn_retro":    make_retro(16),    # 헤더 버튼 (PAUSE / MENU)
            # 한글 포함: 메뉴 버튼, 안내 텍스트
            "menu":         make_korean(32),
            "body":         make_korean(22),
            "small":        make_korean(17),
        }

    def _start_stdin_key_bridge(self):
        """터미널 stdin → pygame 이벤트 브리지 (백그라운드 스레드).

        MobaXterm / VS Code SSH 터미널에서 키를 누르면 pygame.KEYDOWN 이벤트로
        변환해 큐에 주입한다.  pygame 창에 포커스가 없어도 키 제어 가능.

        지원 키 (raw 모드):
          방향키 (↑↓←→), Enter, ESC, W/A/S/D, P, Q, Space
        raw 모드 불가 시 줄 입력(line) 모드로 자동 전환:
          w/s/a/d/up/down/left/right/p/q/esc + Enter
        """
        import threading, sys, os as _os, termios, select

        def _post(key):
            import pygame
            try:
                pygame.event.post(pygame.event.Event(
                    pygame.KEYDOWN, key=key, mod=0, unicode='', scancode=0))
            except Exception:
                pass

        ARROW = {b'A': 'K_UP', b'B': 'K_DOWN', b'C': 'K_RIGHT', b'D': 'K_LEFT'}
        RAW_MAP = {
            b'\r':    'K_RETURN', b'\n': 'K_RETURN',
            b' ':     'K_SPACE',
            b'w':     'K_UP',    b'W': 'K_UP',
            b's':     'K_DOWN',  b'S': 'K_DOWN',
            b'a':     'K_LEFT',  b'A': 'K_LEFT',
            b'd':     'K_RIGHT', b'D': 'K_RIGHT',
            b'p':     'K_p',     b'P': 'K_p',
            b'q':     'K_q',     b'Q': 'K_q',
        }
        LINE_MAP = {
            '':       'K_RETURN',
            'w':      'K_UP',    'up':    'K_UP',
            's':      'K_DOWN',  'down':  'K_DOWN',
            'a':      'K_LEFT',  'left':  'K_LEFT',
            'd':      'K_RIGHT', 'right': 'K_RIGHT',
            'p':      'K_p',     'pause': 'K_p',
            'q':      'K_q',     'quit':  'K_q',
            'esc':    'K_ESCAPE','escape':'K_ESCAPE',
        }

        def _read_keys():
            import pygame
            fd = sys.stdin.fileno()

            # ── raw 모드 시도 ──────────────────────────────────────
            try:
                old_attr = termios.tcgetattr(fd)
            except Exception:
                return   # stdin이 TTY가 아니면 종료

            try:
                new_attr = termios.tcgetattr(fd)
                # IFLAG: 입력 처리 끄기
                new_attr[0] = 0
                # LFLAG: echo, canonical, 시그널 끄기
                new_attr[3] = new_attr[3] & ~(
                    termios.ECHO | termios.ICANON |
                    termios.IEXTEN | termios.ISIG)
                # cc: VMIN=1 (1바이트씩), VTIME=0 (즉시)
                new_attr[6][termios.VMIN]  = 1
                new_attr[6][termios.VTIME] = 0
                termios.tcsetattr(fd, termios.TCSANOW, new_attr)
                use_raw = True
            except Exception:
                use_raw = False

            try:
                if use_raw:
                    while self.running:
                        # 0.1초 타임아웃으로 select → running 변화 감지
                        r, _, _ = select.select([sys.stdin], [], [], 0.1)
                        if not r:
                            continue
                        ch = _os.read(fd, 1)
                        if not ch:
                            break

                        if ch == b'\x1b':
                            # ANSI 시퀀스 판별 (방향키)
                            r2, _, _ = select.select([sys.stdin], [], [], 0.05)
                            if r2:
                                ch2 = _os.read(fd, 1)
                                if ch2 == b'[':
                                    r3, _, _ = select.select([sys.stdin], [], [], 0.05)
                                    if r3:
                                        ch3 = _os.read(fd, 1)
                                        name = ARROW.get(ch3)
                                        if name:
                                            _post(getattr(pygame, name))
                                        continue
                                # ESC + 다른 문자 → ESC 처리
                            _post(pygame.K_ESCAPE)
                        else:
                            name = RAW_MAP.get(ch)
                            if name:
                                _post(getattr(pygame, name))
                else:
                    # ── 줄 입력 fallback ──────────────────────────
                    for line in sys.stdin:
                        if not self.running:
                            break
                        name = LINE_MAP.get(line.strip().lower())
                        if name:
                            _post(getattr(pygame, name))
            finally:
                if use_raw:
                    try:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
                    except Exception:
                        pass

        t = threading.Thread(target=_read_keys, daemon=True, name="stdin-key-bridge")
        t.start()

    def _load_songs(self) -> list:
        """data/reference_dances/ 폴더를 스캔해 곡 목록을 반환합니다."""
        import os, json

        # 경로 탐색 우선순위:
        #   1) engine.py 기준 3단계 위 (src/game → src → project)
        #   2) 현재 작업 디렉토리 기준
        #   3) config 경로 기준
        candidates = []

        try:
            engine_dir   = os.path.dirname(os.path.abspath(__file__))  # src/game
            src_dir      = os.path.dirname(engine_dir)                  # src
            project_root = os.path.dirname(src_dir)                     # project
            candidates.append(os.path.join(project_root, "data", "reference_dances"))
        except Exception:
            pass

        # cwd 기준도 후보에 추가
        candidates.append(os.path.join(os.getcwd(), "data", "reference_dances"))

        base = None
        for c in candidates:
            if os.path.isdir(c):
                base = c
                break

        if base is None:
            print(f"[WARN] reference_dances 폴더를 찾을 수 없습니다. 후보: {candidates}")
            return []

        songs = []
        for song_dir in sorted(os.listdir(base)):
            if song_dir.startswith('.'):
                continue
            meta_path = os.path.join(base, song_dir, "metadata.json")
            ref_path  = os.path.join(base, song_dir, "reference.npy")
            if not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["path"]          = os.path.join(base, song_dir)
                meta["has_reference"] = os.path.exists(ref_path)
                
                # --- 실제 비디오 파일 길이를 읽어 메타데이터 duration 보정 ---
                # 이 과정을 통해 스테이지 선택 화면에 표시되는 곡의 초수와 실제 게임 플레이 초수를 동기화합니다.
                video_rel = meta.get("video", "")
                if video_rel:
                    import cv2 as _cv2
                    project_root = os.path.dirname(os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))))
                    video_path = os.path.join(project_root, video_rel)
                    if not os.path.exists(video_path):
                        video_path = os.path.join(os.getcwd(), video_rel)
                    if os.path.exists(video_path):
                        cap = _cv2.VideoCapture(video_path)
                        fps = cap.get(_cv2.CAP_PROP_FPS)
                        total = cap.get(_cv2.CAP_PROP_FRAME_COUNT)
                        if fps > 0 and total > 0:
                            real_dur = total / fps
                            current_dur = float(meta.get("duration", real_dur))
                            meta["duration"] = min(current_dur, real_dur - 0.15)
                        cap.release()
                # -------------------------------------------------------------
                
                songs.append(meta)
            except Exception as e:
                print(f"[WARN] 곡 로드 실패 {song_dir}: {e}")

        print(f"[INFO] {len(songs)}곡 로드됨 (base: {base})")
        # 난이도 오름차순 정렬 (같은 난이도면 title 순)
        songs.sort(key=lambda s: (s.get("difficulty", 0), s.get("title", "")))
        return songs
        songs = []
        if not os.path.isdir(base):
            return songs
        for song_dir in sorted(os.listdir(base)):
            meta_path = os.path.join(base, song_dir, "metadata.json")
            ref_path  = os.path.join(base, song_dir, "reference.npy")
            if not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["path"] = os.path.join(base, song_dir)
                meta["has_reference"] = os.path.exists(ref_path)
                songs.append(meta)
            except Exception as e:
                print(f"[WARN] 곡 로드 실패 {song_dir}: {e}")
        return songs

    def _songs_for_mode(self, mode: str) -> list:
        """현재 모드에서 플레이 가능한 곡 목록을 반환합니다."""
        return [s for s in self._songs if mode in s.get("mode", [])]

    def run(self):
        """
        Main game loop.
        Handles: input -> update -> render cycle at TARGET_FPS.
        """
        self.running = True

        while self.running:
            # 1. Handle input events
            self._handle_input()

            # 2. Update game state
            self._update()

            # 3. Render frame
            self._render()

            # 4. Frame rate control
            self._clock.tick(self.TARGET_FPS)

    def _handle_input(self):
        """Process user input events (keyboard, mouse, touch)."""
        import pygame

        # ── 화면별 포커스 버튼 목록 정의 ──────────────────────────────
        FOCUS_LISTS = {
            GameState.MENU: [
                "btn_practice", "btn_challenge", "btn_freestyle",
                "btn_settings", "btn_quit",
            ],
            GameState.PAUSED:   ["btn_pause", "btn_gameplay_menu"],
            GameState.RESULT:   ["btn_retry", "btn_result_menu"],
            GameState.SETTINGS: ["btn_back"],
            GameState.READY: ["btn_ready_skip", "btn_ready_cancel"],
        }

        # SONG_SELECT: 좌우로 패널 전환 (0=곡목록, 1=상세/버튼)
        # _song_panel: 0=곡목록 패널, 1=버튼 패널(START=0, BACK=1)
        if not hasattr(self, '_song_panel'):
            self._song_panel = 0
        if not hasattr(self, '_song_btn_idx'):
            self._song_btn_idx = 0   # 0=START, 1=BACK

        def _get_focus_list():
            if self.state == GameState.SONG_SELECT:
                songs = self._songs_for_mode(self._current_mode)
                if self._song_panel == 0:
                    return [f"btn_song_{i}" for i in range(len(songs))]
                else:
                    return ["btn_song_start", "btn_song_back"]
            return FOCUS_LISTS.get(self.state, [])

        def _focus_count():
            return len(_get_focus_list())

        def _get_focus_idx():
            if self.state == GameState.MENU:
                return self._menu_focus_idx
            if self.state == GameState.SONG_SELECT:
                if self._song_panel == 0:
                    return self._song_focus_idx
                else:
                    return self._song_btn_idx
            return getattr(self, '_generic_focus_idx', 0)

        def _set_focus_idx(idx):
            fl = _get_focus_list()
            if not fl:
                return
            idx = idx % len(fl)
            if self.state == GameState.MENU:
                self._menu_focus_idx = idx
            elif self.state == GameState.SONG_SELECT:
                if self._song_panel == 0:
                    self._song_focus_idx = idx
                    self._selected_song_idx = idx
                else:
                    self._song_btn_idx = idx
            else:
                self._generic_focus_idx = idx

        def _press_focused():
            fl = _get_focus_list()
            if not fl:
                return
            idx = _get_focus_idx() % max(len(fl), 1)
            btn = fl[idx]
            self._on_button_press(btn)

        def _go_back():
            """B키 / ESC 뒤로가기."""
            if self.state in (GameState.SONG_SELECT, GameState.SETTINGS):
                self.transition_to(GameState.MENU)
            elif self.state in (GameState.READY, GameState.COUNTDOWN):
                self.transition_to(GameState.SONG_SELECT)
            elif self.state == GameState.PAUSED:
                self.transition_to(GameState.PLAYING)
            elif self.state in (GameState.PLAYING,):
                self.transition_to(GameState.PAUSED)
            elif self.state == GameState.RESULT:
                self.transition_to(GameState.MENU)
            elif self.state == GameState.MENU:
                self.running = False

        for event in pygame.event.get():
            # ── 창 닫기 ────────────────────────────────────────
            if event.type == pygame.QUIT:
                self.running = False
                return

            # ── 마우스 이동 → 포커스 이동 ──────────────────────
            elif event.type == pygame.MOUSEMOTION:
                fl = _get_focus_list()
                for i, k in enumerate(fl):
                    if k in self._btn_rects and self._btn_rects[k].collidepoint(event.pos):
                        _set_focus_idx(i)
                        break

            # ── 키보드 ──────────────────────────────────────────
            if event.type == pygame.KEYDOWN:

                # ESC → 뒤로가기
                if event.key == pygame.K_ESCAPE:
                    _go_back()

                # B → 뒤로가기
                elif event.key == pygame.K_b:
                    _go_back()

                # Q → 종료
                elif event.key == pygame.K_q:
                    self.running = False

                # ↓ : 다음 항목 (수직 이동)
                elif event.key == pygame.K_DOWN:
                    cnt = _focus_count()
                    if cnt:
                        _set_focus_idx(_get_focus_idx() + 1)

                # ↑ : 이전 항목 (수직 이동)
                elif event.key == pygame.K_UP:
                    cnt = _focus_count()
                    if cnt:
                        _set_focus_idx(_get_focus_idx() - 1)

                # → : SONG_SELECT=상세패널로, 그 외=다음 항목
                elif event.key == pygame.K_RIGHT:
                    if self.state == GameState.SONG_SELECT:
                        self._song_panel = 1
                        self._song_btn_idx = 0
                    else:
                        cnt = _focus_count()
                        if cnt:
                            _set_focus_idx(_get_focus_idx() + 1)

                # ← : SONG_SELECT=곡목록으로, 그 외=이전 항목
                elif event.key == pygame.K_LEFT:
                    if self.state == GameState.SONG_SELECT:
                        self._song_panel = 0
                    else:
                        cnt = _focus_count()
                        if cnt:
                            _set_focus_idx(_get_focus_idx() - 1)

                # Enter / Space → 선택 확정
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    if self.state == GameState.READY:
                        # READY 화면에서 Enter = SKIP (전신 감지 대기 없이 바로 시작)
                        self._on_button_press("btn_ready_skip")
                    else:
                        _press_focused()

                # P → 게임 중 일시정지 토글
                elif event.key == pygame.K_p:
                    if self.state == GameState.PLAYING:
                        self.transition_to(GameState.PAUSED)
                    elif self.state == GameState.PAUSED:
                        self.transition_to(GameState.PLAYING)

            # ── 마우스/터치 클릭 ────────────────────────────────
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if not getattr(self, '_finger_handled', False):
                    self._last_event_type = 'mouse'
                    self._handle_click(event.pos)
                self._finger_handled = False

            # ── 핑거(터치스크린 전용) 이벤트 ──────────────────────
            elif event.type == pygame.FINGERDOWN:
                w_d, h_d = self._display.get_size()
                touch_pos = (int(event.x * w_d), int(event.y * h_d))
                self._last_event_type = 'touch'
                self._finger_handled = True   # 뒤따라오는 MOUSEBUTTONDOWN 무시
                self._handle_click(touch_pos)

    def _handle_click(self, pos: tuple):
        """
        터치/마우스 클릭 처리.

        MENU / SONG_SELECT 상태에서는 더블탭 방식:
          - 1st click: 해당 버튼에 포커스만 이동 (하이라이트)
          - 2nd click (같은 버튼): 실제 액션 실행
        다른 상태(PLAYING, PAUSED 등)에서는 단일 클릭으로 즉시 실행.

        MOUSEMOTION으로 이미 hover된 버튼은 이미 focused 상태이므로
        마우스 클릭 시 즉시 실행된다.
        """
        double_tap_states = (GameState.MENU, GameState.SONG_SELECT)

        for btn_name, rect in self._btn_rects.items():
            if not rect.collidepoint(pos):
                continue

            if self.state in double_tap_states:
                if self._touch_focused_btn != btn_name:
                    # 1st tap → 포커스만
                    self._touch_focused_btn = btn_name
                    self._sync_focus_to_btn(btn_name)
                else:
                    # 2nd tap → 실행
                    self._touch_focused_btn = ""
                    self._on_button_press(btn_name)
            else:
                # 게임 중/일시정지/결과 등: 즉시 실행
                self._touch_focused_btn = ""
                self._on_button_press(btn_name)
            return

    def _sync_focus_to_btn(self, btn_name: str):
        """버튼 이름에 해당하는 포커스 인덱스를 갱신한다 (터치 첫 탭 시 시각적 하이라이트용)."""
        menu_keys = ["btn_practice", "btn_challenge", "btn_freestyle",
                     "btn_settings", "btn_quit"]
        if btn_name in menu_keys:
            self._menu_focus_idx = menu_keys.index(btn_name)
            return
        if btn_name.startswith("btn_song_"):
            try:
                idx = int(btn_name.split("_")[-1])
                self._song_focus_idx = idx
                self._selected_song_idx = idx
            except ValueError:
                pass

    def _on_button_press(self, btn_name: str):
        """버튼 이름에 따라 액션 실행."""
        # ── 메뉴 화면 버튼 ──
        if btn_name == "btn_practice":
            self._current_mode = "practice"
            self._selected_song_idx = 0
            self.transition_to(GameState.SONG_SELECT)
        elif btn_name == "btn_challenge":
            self._current_mode = "challenge"
            self._selected_song_idx = 0
            self.transition_to(GameState.SONG_SELECT)
        elif btn_name == "btn_freestyle":
            self._current_mode = "freestyle"
            self._selected_song_idx = 0
            self.transition_to(GameState.SONG_SELECT)
        elif btn_name == "btn_settings":
            self.transition_to(GameState.SETTINGS)
        elif btn_name == "btn_quit":
            self.running = False

        # ── 곡 선택 화면 버튼 ──
        elif btn_name == "btn_song_back":
            self.transition_to(GameState.MENU)
        elif btn_name == "btn_song_start":
            songs = self._songs_for_mode(self._current_mode)
            if songs:
                self._current_song = songs[self._selected_song_idx]
            self.transition_to(GameState.READY)
        elif btn_name.startswith("btn_song_"):
            try:
                idx = int(btn_name.split("_")[-1])
                songs = self._songs_for_mode(self._current_mode)
                if 0 <= idx < len(songs):
                    self._selected_song_idx = idx
            except ValueError:
                pass

        # ── 게임플레이 화면 버튼 ──
        elif btn_name == "btn_pause":
            if self.state == GameState.PLAYING:
                self.transition_to(GameState.PAUSED)
            elif self.state == GameState.PAUSED:
                self.transition_to(GameState.PLAYING)
        elif btn_name == "btn_gameplay_menu":
            self.transition_to(GameState.MENU)

        # ── 결과 화면 버튼 ──
        elif btn_name == "btn_retry":
            self.transition_to(GameState.READY)
        elif btn_name == "btn_result_menu":
            self.transition_to(GameState.MENU)

        # ── 설정/카운트다운/준비 화면 버튼 ──
        elif btn_name == "btn_back":
            self.transition_to(GameState.MENU)
        elif btn_name == "btn_countdown_cancel":
            self.transition_to(GameState.MENU)
        elif btn_name == "btn_ready_skip":
            self.transition_to(GameState.COUNTDOWN)
        elif btn_name == "btn_ready_cancel":
            self.transition_to(GameState.MENU)
        elif btn_name == "btn_ready_skip":
            # 전신 감지 대기 없이 바로 카운트다운으로 진입
            self.transition_to(GameState.COUNTDOWN)

    def _update(self):
        """Update game state based on current state."""
        # 네온 깜빡임 타이머 (항상 업데이트)
        self._neon_tick += 1.0 / self.TARGET_FPS

        # 피드백 나이 타이머 (등장 후 경과시간, 애니메이션 progress용)
        if self._last_feedback is not None:
            self._feedback_age += 1.0 / self.TARGET_FPS

        if self.state == GameState.READY:
            self._update_ready()

        elif self.state == GameState.COUNTDOWN:
            elapsed = time.time() - self._countdown_start
            remaining = 2 - int(elapsed)
            if remaining < 0:
                self.transition_to(GameState.PLAYING)
            else:
                self._countdown_timer = remaining

            # 카운트다운 중 카메라 프레임 워밍업
            if getattr(self, '_async_camera', None) is not None:
                ret, frame, _, _ = self._async_camera.read()
                if ret:
                    self._current_frame = frame

        elif self.state == GameState.PLAYING:
            self._update_gameplay()

        # PAUSED 상태에서는 카메라/포즈 업데이트 중단

    def _update_ready(self):
        """READY 상태: 비동기 카메라로부터 포즈 감지 및 카운트다운."""
        if getattr(self, '_async_camera', None) is not None:
            ret, frame, lm, detected = self._async_camera.read()
            if not ret:
                return

            self._ready_current_frame = frame
            self._ready_landmarks = lm
            self._ready_pose_detected = detected

        # 전신(발목) 감지 여부 확인
        full_body = False
        if self._ready_pose_detected and self._ready_landmarks is not None:
            lm = self._ready_landmarks
            L_ANKLE, R_ANKLE = 27, 28
            full_body = (lm[L_ANKLE][3] > 0.3 and lm[R_ANKLE][3] > 0.3)

        COUNTDOWN_SEC = 3.0  # 발목 감지 후 대기 시간

        if full_body:
            if self._ready_full_body_start == 0.0:
                self._ready_full_body_start = time.time()
            elapsed = time.time() - self._ready_full_body_start
            self._ready_countdown = max(0.0, COUNTDOWN_SEC - elapsed)
            if elapsed >= COUNTDOWN_SEC:
                # 3초 유지 완료 → 게임 시작
                self._ready_full_body_start = 0.0
                self._ready_countdown = 0.0
                self.transition_to(GameState.COUNTDOWN)
        else:
            # 전신 미감지 시 카운트다운 리셋
            self._ready_full_body_start = 0.0
            self._ready_countdown = 0.0

    def _mark_no_score_frame(self):
        """Keep the latest feedback visible briefly when no new score is produced."""
        if self._last_feedback is not None and self._feedback_timer <= 0 and self._score_hold_seconds > 0:
            self._feedback_timer = self._score_hold_seconds

    def _pose_similarity_with_method(self, comparator, user_landmarks, ref_landmarks, method):
        if method == "euclidean":
            return comparator.euclidean_similarity(user_landmarks, ref_landmarks)
        if method == "hybrid":
            return comparator.hybrid_similarity(user_landmarks, ref_landmarks)
        if method == "angle":
            return comparator.angle_similarity(user_landmarks, ref_landmarks)
        return comparator.cosine_similarity(user_landmarks, ref_landmarks)

    def _direct_window_similarity(self, user_landmarks, method=None, debug=False):
        """Direct pose similarity over the delay tolerance window."""
        comparator = self._pose_comparator or self._fallback_pose_comparator
        if comparator is None or user_landmarks is None or self._ref_landmarks is None:
            return None
        if self._ref_frame_landmarks is None:
            return None

        method = method or self._similarity_method
        tolerance_frames = int(self._tolerance_delay * self.TARGET_FPS)
        start_idx = max(0, self._ref_current_idx - tolerance_frames)
        end_idx = self._ref_current_idx + 1

        best_sims = []
        for ri in range(start_idx, end_idx):
            ref_lm = self._ref_landmarks[ri]
            best_sims.append(
                self._pose_similarity_with_method(comparator, user_landmarks, ref_lm, method)
            )
        if not best_sims:
            return None

        best_sims.sort(reverse=True)
        top_k = best_sims[:min(3, len(best_sims))]
        sim = sum(top_k) / len(top_k)

        if debug:
            sim_now = self._pose_similarity_with_method(
                comparator, user_landmarks, self._ref_frame_landmarks, method)
            print(
                f"\r[DBG] now={sim_now:.3f} top3={sim:.3f} "
                f"max={best_sims[0]:.3f} win={end_idx-start_idx}f",
                end="",
            )
        return sim

    def _update_gameplay(self):
        """Fetch async frame and compute score."""
        import cv2
        import numpy as np

        if getattr(self, '_async_camera', None) is None:
            return

        ret, frame, lm, detected = self._async_camera.read()
        if not ret:
            return

        self._current_frame = frame
        self._current_landmarks = lm
        self._pose_detected = detected

        scoring_landmarks = None
        if self._pose_detected and self._current_landmarks is not None:
            scoring_landmarks = self._current_landmarks
            self._last_valid_landmarks = np.asarray(self._current_landmarks, dtype=np.float32).copy()
            self._last_valid_landmarks_age_frames = 0
        elif (
            self._last_valid_landmarks is not None
            and self._last_valid_landmarks_age_frames < self._pose_hold_frames
        ):
            # 짧은 MediaPipe dropout은 마지막 정상 포즈로 채점해 UI 공백을 줄인다.
            self._last_valid_landmarks_age_frames += 1
            scoring_landmarks = self._last_valid_landmarks
        else:
            self._last_valid_landmarks_age_frames += 1

        # 참조 캐릭터 프레임 인덱싱 (30fps 기준)
        if self._ref_landmarks is not None and self._current_session:
            fi = min(
                int(self._current_session.elapsed_time * self.TARGET_FPS),
                len(self._ref_landmarks) - 1,
            )
            self._ref_frame_landmarks = self._ref_landmarks[fi]
            self._ref_current_idx = fi

        # 레퍼런스 영상 프레임 동기화 (영상 fps 기준)
        if self._ref_video_cap is not None and self._current_session:
            # pygame.mixer의 실제 오디오 재생 위치를 사용하여 정확도 극대화
            import pygame
            elapsed_sys = self._current_session.elapsed_time
            if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                pos_ms = pygame.mixer.music.get_pos()
                if pos_ms >= 0:
                    # 마이너스 지연(오프셋)을 주어 영상이 소리보다 살짝 늦게(느리게) 렌더링되게 보정
                    # 오디오 하드웨어 버퍼와 OS 전달 시간의 차이를 보정하는 리듬 게임 필수 로직
                    audio_latency = self.config.get("audio", {}).get("latency_offset", 0.040)
                    elapsed_sys = (pos_ms / 1000.0) - audio_latency
                    if elapsed_sys < 0: elapsed_sys = 0.0

            video_fi = int(elapsed_sys * self._ref_video_fps)
            total_video_frames = int(self._ref_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if video_fi < total_video_frames:
                current_pos = int(self._ref_video_cap.get(cv2.CAP_PROP_POS_FRAMES))

                # 오차(frames_to_skip)가 발생하면 동기화를 위해 스킵
                frames_to_skip = video_fi - current_pos
                if frames_to_skip < 0 or frames_to_skip > 10:
                    # 너무 멀면 seek (상당히 느림)
                    self._ref_video_cap.set(cv2.CAP_PROP_POS_FRAMES, video_fi)
                else:
                    # 목표 프레임(video_fi)의 바로 앞 컷까지 건너뜀 (grab이 빠름)
                    while current_pos < video_fi:
                        self._ref_video_cap.grab()
                        current_pos += 1

                vret, vframe = self._ref_video_cap.read()
                if vret:
                    self._ref_video_frame = vframe
                    tw, th = self._ref_video_size
                    if tw > 0 and th > 0:
                        small = cv2.resize(vframe, (tw, th),
                                           interpolation=cv2.INTER_NEAREST)
                        # BGR→RGB + pygame Surface (frombuffer가 swapaxes보다 빠름)
                        rgb = small[:, :, ::-1]
                        import pygame
                        self._ref_video_surf = pygame.image.frombuffer(
                            rgb.tobytes(), (tw, th), "RGB")
                    else:
                        self._ref_video_surf = None
            else:
                self._ref_video_frame = None
                self._ref_video_surf = None

        # Score based on pose similarity. scoring_landmarks may be a held pose
        # for a few frames when MediaPipe briefly drops detection.
        sim = None
        if scoring_landmarks is not None:
            if self._ref_frame_landmarks is not None:
                if self._score_method == "direct":
                    sim = self._direct_window_similarity(
                        scoring_landmarks,
                        method=self._similarity_method,
                        debug=True,
                    )
                elif self._score_method == "scratch":
                    tolerance_frames = int(self._tolerance_delay * self.TARGET_FPS)
                    sim = self._scratch_comparator.compute(
                        scoring_landmarks,
                        self._ref_landmarks,
                        self._ref_current_idx,
                        tolerance_frames=tolerance_frames,
                    )
                    if sim is None and self._model_warmup_direct_fallback:
                        sim = self._direct_window_similarity(
                            scoring_landmarks,
                            method=self._fallback_similarity_method,
                        )
                else:
                    tolerance_frames = int(self._tolerance_delay * self.TARGET_FPS)
                    sim = self._embedding_comparator.compute(
                        scoring_landmarks,
                        self._ref_landmarks,
                        self._ref_current_idx,
                        tolerance_frames=tolerance_frames,
                    )
                    if sim is None and self._model_warmup_direct_fallback:
                        sim = self._direct_window_similarity(
                            scoring_landmarks,
                            method=self._fallback_similarity_method,
                        )
            else:
                if self._score_method in ("scratch", "embedding"):
                    if not self._warned_scratch_no_ref:
                        print(f"[WARN] {self._score_method} scoring requires reference.npy; scoring paused.")
                        self._warned_scratch_no_ref = True
                else:
                    visibility = scoring_landmarks[:, 3]
                    visible = visibility[visibility > 0]
                    sim = min(float(np.mean(visible)), 1.0) if len(visible) else 0.0

        if sim is not None:
            evaluation = self._scorer.evaluate(sim)
            fb = self._feedback_gen.generate(evaluation)
            if fb:
                self._last_feedback = fb
                self._feedback_timer = 1.2  # 1.2초 동안 표시
                self._feedback_age   = 0.0  # 애니메이션 경과 시간 리셋
                # 파티클 폭발 효과 — fb["text"]로 등급 판단
                text = fb.get("text", "")
                particle_map = {"PERFECT!": 60, "GREAT!": 40, "GOOD": 25}
                particle_count = particle_map.get(text, 0)
                if particle_count > 0 and self._display is not None:
                    w_d, h_d = self._display.get_size()
                    color = fb.get("color", (255, 255, 255))
                    self._spawn_particles(w_d // 2, h_d // 2, color, particle_count)
        else:
            self._mark_no_score_frame()
            self._feedback_gen.update()

        # 피드백 타이머 감소
        if self._feedback_timer > 0:
            self._feedback_timer = max(0.0, self._feedback_timer - 1.0 / self.TARGET_FPS)
            if self._feedback_timer <= 0:
                self._last_feedback = None

        # Check session time
        if self._current_session:
            if self._current_session.is_finished:
                self._result_data = self._scorer.get_final_result()
                self.transition_to(GameState.RESULT)

    def _render(self):
        """Render current frame to display."""
        import pygame

        # 매 프레임 버튼 rect 초기화 (현재 화면의 버튼만 등록)
        self._btn_rects.clear()

        w, h = self._display.get_size()

        if self.state == GameState.MENU:
            self._render_menu(w, h)
        elif self.state == GameState.SONG_SELECT:
            self._render_song_select(w, h)
        elif self.state == GameState.READY:
            self._render_ready(w, h)
        elif self.state == GameState.COUNTDOWN:
            self._render_countdown(w, h)
        elif self.state == GameState.PLAYING:
            self._render_gameplay(w, h)
        elif self.state == GameState.PAUSED:
            self._render_gameplay(w, h)   # 동일 화면에 오버레이 추가
            self._render_pause_overlay(w, h)
        elif self.state == GameState.RESULT:
            self._render_result(w, h)
        elif self.state == GameState.SETTINGS:
            self._render_settings(w, h)

        pygame.display.flip()

    def _render_menu(self, w, h):
        """Render the main menu — retro-fancy neon style."""
        import pygame
        import math

        tick = self._neon_tick

        # ── 배경: 따뜻한 보라→핑크 그라데이션 ──────────────────
        for y in range(h):
            t = y / h
            r = int(28 + 20 * t)
            g = int(8  + 8  * t)
            b = int(55 + 25 * t)
            pygame.draw.line(self._display, (r, g, b), (0, y), (w, y))

        # 배경 격자선 (레트로 느낌)
        grid_color = (50, 20, 80)
        for gx in range(0, w, 60):
            pygame.draw.line(self._display, grid_color, (gx, 0), (gx, h))
        for gy in range(0, h, 60):
            pygame.draw.line(self._display, grid_color, (0, gy), (w, gy))

        # ── 레이아웃 ─────────────────────────────────────────────
        MARGIN_TOP    = 24
        BTN_H         = 54
        BTN_W         = min(440, w - 60)
        SMALL_BTN_H   = 42
        SMALL_BTN_W   = min(190, (BTN_W - 20) // 2)
        FOOTER_H      = 26
        MARGIN_BOTTOM = 36
        title_area_h  = 110

        mode_area_top    = MARGIN_TOP + title_area_h + 16
        bottom_area_h    = SMALL_BTN_H + FOOTER_H + 14
        mode_area_bottom = h - MARGIN_BOTTOM - bottom_area_h
        mode_area_h      = mode_area_bottom - mode_area_top
        num_btns         = 3
        gap              = max(14, (mode_area_h - num_btns * BTN_H) // (num_btns + 1))
        btn_start_y      = mode_area_top + (mode_area_h - (num_btns * BTN_H + gap * (num_btns - 1))) // 2
        btn_x            = w // 2 - BTN_W // 2

        # ── 타이틀 ───────────────────────────────────────────────
        neon_cyan = self._neon_color((80, 255, 220), tick)
        title_surf = self._fonts["title"].render("Let's Dance!", True, neon_cyan)
        # 타이틀 글로우 (살짝 번짐)
        glow_surf = self._fonts["title"].render("Let's Dance!", True, (30, 120, 100))
        for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
            self._display.blit(glow_surf, glow_surf.get_rect(
                center=(w // 2 + dx, MARGIN_TOP + 38 + dy)))
        self._display.blit(title_surf, title_surf.get_rect(center=(w // 2, MARGIN_TOP + 38)))

        sub_color = self._neon_color((230, 160, 255), tick, intensity=0.9)
        sub = self._fonts["body"].render("* AI DANCE SCORE GAME *", True, sub_color)
        self._display.blit(sub, sub.get_rect(center=(w // 2, MARGIN_TOP + 82)))

        # 구분선
        line_col = self._neon_color((180, 80, 255), tick, 0.7)
        pygame.draw.line(self._display, line_col,
                         (w//2 - BTN_W//2, MARGIN_TOP + 100),
                         (w//2 + BTN_W//2, MARGIN_TOP + 100), 1)

        # ── 모드 버튼 ────────────────────────────────────────────
        mouse_pos = pygame.mouse.get_pos()
        btn_defs = [
            ("btn_practice",  "PRACTICE",  (0, 220, 180),   (0, 80, 60)),
            ("btn_challenge", "CHALLENGE", (255, 190, 0),   (90, 60, 0)),
            ("btn_freestyle", "FREE STYLE",(200, 100, 255), (70, 20, 100)),
        ]

        for i, (btn_name, label, neon_col, fill_col) in enumerate(btn_defs):
            y_pos  = btn_start_y + i * (BTN_H + gap)
            rect   = pygame.Rect(btn_x, y_pos, BTN_W, BTN_H)
            self._btn_rects[btn_name] = rect

            focused = (i == self._menu_focus_idx)
            hover   = rect.collidepoint(mouse_pos)
            active  = focused or hover

            # 배경 채우기
            if active:
                bg_alpha_surf = pygame.Surface((BTN_W, BTN_H), pygame.SRCALPHA)
                bg_alpha_surf.fill((*fill_col, 180))
                self._display.blit(bg_alpha_surf, rect.topleft)
            else:
                bg_alpha_surf = pygame.Surface((BTN_W, BTN_H), pygame.SRCALPHA)
                bg_alpha_surf.fill((20, 10, 40, 160))
                self._display.blit(bg_alpha_surf, rect.topleft)

            # 네온 테두리 + 글로우
            border_col = self._neon_color(neon_col, tick) if active else \
                         tuple(c // 3 for c in neon_col)
            self._draw_neon_rect(self._display, rect, border_col,
                                 width=2, radius=14,
                                 glow_radius=10 if active else 3)

            # 포커스/호버 시 모서리 브래킷
            if active:
                bracket_col = self._neon_color(neon_col, tick * 2)
                self._draw_corner_brackets(self._display, rect, bracket_col,
                                           size=16, width=3)

            # 레이블
            txt_color = (255, 255, 255) if active else (180, 170, 200)
            lbl = self._fonts["body"].render(label, True, txt_color)
            self._display.blit(lbl, lbl.get_rect(center=rect.center))

        # ── 설정 / 종료 버튼 ────────────────────────────────────
        bottom_btn_y = h - MARGIN_BOTTOM - FOOTER_H - SMALL_BTN_H - 4
        btn_gap      = 20
        total_sw     = SMALL_BTN_W * 2 + btn_gap
        small_x      = w // 2 - total_sw // 2

        btn_s_rect = pygame.Rect(small_x, bottom_btn_y, SMALL_BTN_W, SMALL_BTN_H)
        btn_q_rect = pygame.Rect(small_x + SMALL_BTN_W + btn_gap,
                                 bottom_btn_y, SMALL_BTN_W, SMALL_BTN_H)
        self._btn_rects["btn_settings"] = btn_s_rect
        self._btn_rects["btn_quit"]     = btn_q_rect

        small_defs = [
            (btn_s_rect, "btn_settings", "SETTINGS", (100, 120, 255), 3),
            (btn_q_rect, "btn_quit",     "QUIT",     (255, 80,  80),  4),
        ]
        for rect, bname, label, ncol, focus_i in small_defs:
            focused = (self._menu_focus_idx == focus_i)
            hover   = rect.collidepoint(mouse_pos)
            active  = focused or hover

            bg_alpha = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            fill = tuple(c // 3 for c in ncol)
            bg_alpha.fill((*fill, 200 if active else 120))
            self._display.blit(bg_alpha, rect.topleft)

            bcol = self._neon_color(ncol, tick) if active else tuple(c // 2 for c in ncol)
            self._draw_neon_rect(self._display, rect, bcol,
                                 width=2, radius=12,
                                 glow_radius=8 if active else 2)
            if active:
                self._draw_corner_brackets(self._display, rect,
                                           self._neon_color(ncol, tick * 2), size=10, width=2)
            lbl = self._fonts["small"].render(label, True,
                                              (255, 255, 255) if active else (160, 155, 180))
            self._display.blit(lbl, lbl.get_rect(center=rect.center))

        # ── 푸터 ─────────────────────────────────────────────────
        footer_col = (140, 110, 180)
        footer = self._fonts["small"].render(
            "UP/DOWN: SELECT   ENTER: CONFIRM   ESC: QUIT", True, footer_col
        )
        self._display.blit(footer, footer.get_rect(
            center=(w // 2, h - MARGIN_BOTTOM // 2 - 2)))

        # ── 키보드 안내 박스 (타이틀 아래 우측) ──────────────────
        key_lines = [
            ("KEYBOARD CONTROLS", (180, 180, 255)),
            ("UP / DOWN   :  Move menu",       (200, 200, 220)),
            ("ENTER       :  Select / Start",  (200, 200, 220)),
            ("B / ESC     :  Back / Quit",     (200, 200, 220)),
            ("P           :  Pause game",      (200, 200, 220)),
            ("Q           :  Force quit",      (200, 200, 220)),
        ]
        kx = w - 14
        ky = MARGIN_TOP + 110
        line_h = 18
        box_w = 260
        box_h = len(key_lines) * line_h + 14
        kb_surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        kb_surf.fill((20, 10, 50, 160))
        self._display.blit(kb_surf, (kx - box_w, ky))
        pygame.draw.rect(self._display, (80, 60, 130),
                         pygame.Rect(kx - box_w, ky, box_w, box_h), 1, border_radius=6)
        for li, (txt, col) in enumerate(key_lines):
            f = self._fonts["small_retro"] if li == 0 else self._fonts["small"]
            s = f.render(txt, True, col)
            self._display.blit(s, (kx - box_w + 10, ky + 7 + li * line_h))

    def _render_song_select(self, w, h):
        """곡 선택 화면 — retro-fancy neon style."""
        import pygame
        import math

        tick = self._neon_tick

        MODE_LABELS = {"practice": "PRACTICE", "challenge": "CHALLENGE", "freestyle": "FREE STYLE"}
        DIFF_STARS  = {0: "FREE", 1: "★☆☆☆☆", 2: "★★☆☆☆", 3: "★★★☆☆", 4: "★★★★☆", 5: "★★★★★"}
        MODE_COLORS = {"practice": (0, 220, 180), "challenge": (255, 190, 0),
                       "freestyle": (200, 100, 255)}
        mode_col    = MODE_COLORS.get(self._current_mode, (180, 180, 255))

        HEADER_H      = 58
        FOOTER_H      = 34
        BOTTOM_MARGIN = 54
        BODY_TOP      = HEADER_H + 12
        BODY_BOTTOM   = h - BOTTOM_MARGIN - FOOTER_H

        # 배경 그라데이션
        for y in range(h):
            t = y / h
            pygame.draw.line(self._display, (int(22 + 16*t), int(8 + 8*t), int(50 + 20*t)),
                             (0, y), (w, y))
        # 배경 격자
        for gx in range(0, w, 60):
            pygame.draw.line(self._display, (45, 18, 70), (gx, 0), (gx, h))
        for gy in range(0, h, 60):
            pygame.draw.line(self._display, (45, 18, 70), (0, gy), (w, gy))

        # ── 헤더 ─────────────────────────────────────────────────
        hdr_surf = pygame.Surface((w, HEADER_H), pygame.SRCALPHA)
        hdr_surf.fill((15, 8, 40, 220))
        self._display.blit(hdr_surf, (0, 0))
        pygame.draw.line(self._display, self._neon_color(mode_col, tick, 0.8),
                         (0, HEADER_H - 1), (w, HEADER_H - 1), 2)

        mode_label = MODE_LABELS.get(self._current_mode, "")
        hdr_neon   = self._neon_color(mode_col, tick)
        hdr_txt    = self._fonts["menu"].render(f"{mode_label}  --  SELECT SONG", True, hdr_neon)
        self._display.blit(hdr_txt, hdr_txt.get_rect(midleft=(18, HEADER_H // 2)))

        songs     = self._songs_for_mode(self._current_mode)
        mouse_pos = pygame.mouse.get_pos()

        if not songs:
            msg = self._fonts["body"].render("No songs available for this mode.", True, (255, 120, 120))
            self._display.blit(msg, msg.get_rect(center=(w // 2, h // 2)))
        else:
            # ── 곡 카드 목록 (왼쪽 52%) ─────────────────────────
            card_area_w = int(w * 0.52)
            card_w      = card_area_w - 32
            card_x      = 16
            available_h = BODY_BOTTOM - BODY_TOP
            num_songs   = len(songs)
            card_h      = min(72, max(50, (available_h - 10 * num_songs) // max(num_songs, 1)))
            card_gap    = min(10, max(5,  (available_h - card_h * num_songs) // max(num_songs, 1)))

            for i, song in enumerate(songs):
                cy   = BODY_TOP + i * (card_h + card_gap)
                if cy + card_h > BODY_BOTTOM:
                    break
                rect = pygame.Rect(card_x, cy, card_w, card_h)
                self._btn_rects[f"btn_song_{i}"] = rect

                selected = (i == self._selected_song_idx)
                hover    = rect.collidepoint(mouse_pos) and not selected
                active   = selected or hover

                # 카드 배경
                bg_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
                if selected:
                    bg_surf.fill((*[c // 4 for c in mode_col], 210))
                elif hover:
                    bg_surf.fill((50, 30, 90, 180))
                else:
                    bg_surf.fill((20, 14, 48, 160))
                self._display.blit(bg_surf, rect.topleft)

                # 카드 테두리
                if selected:
                    self._draw_neon_rect(self._display, rect,
                                         self._neon_color(mode_col, tick),
                                         width=2, radius=12, glow_radius=10)
                    self._draw_corner_brackets(self._display, rect,
                                               self._neon_color(mode_col, tick * 2),
                                               size=14, width=3)
                elif hover:
                    self._draw_neon_rect(self._display, rect, (160, 130, 220),
                                         width=1, radius=12, glow_radius=4)
                else:
                    pygame.draw.rect(self._display, (60, 45, 100), rect, 1, border_radius=12)

                # 곡 제목
                title_col = (255, 255, 255) if active else (200, 195, 220)
                t_surf = self._fonts["body"].render(song.get("title", "?"), True, title_col)
                self._display.blit(t_surf, (rect.x + 14, rect.y + 8))

                # 곡 정보
                diff  = DIFF_STARS.get(song.get("difficulty", 0), "")
                info  = f"BPM {song.get('bpm',0)}  ·  {song.get('duration',0)}s  ·  {diff}"
                info_col = self._neon_color(mode_col, tick, 0.7) if selected else (130, 125, 160)
                i_surf = self._fonts["small"].render(info, True, info_col)
                self._display.blit(i_surf, (rect.x + 14, rect.y + card_h - 22))

            # ── 선택된 곡 상세 패널 (오른쪽) ─────────────────────
            if 0 <= self._selected_song_idx < len(songs):
                sel     = songs[self._selected_song_idx]
                px      = card_area_w + 12
                pw      = w - px - 14
                start_h = 50
                start_m = 14
                panel   = pygame.Rect(px, BODY_TOP, pw, BODY_BOTTOM - BODY_TOP)

                # 패널 배경
                p_surf = pygame.Surface((pw, panel.height), pygame.SRCALPHA)
                p_surf.fill((18, 10, 42, 210))
                self._display.blit(p_surf, panel.topleft)

                self._draw_neon_rect(self._display, panel,
                                     self._neon_color(mode_col, tick, 0.6),
                                     width=2, radius=14, glow_radius=6)
                self._draw_corner_brackets(self._display, panel,
                                           self._neon_color(mode_col, tick * 1.5),
                                           size=18, width=2)

                # 상세 정보
                detail_items = [
                    ("TITLE",    sel.get("title", "-")),
                    ("ARTIST",   sel.get("artist", "-")),
                    ("BPM",      str(sel.get("bpm", 0))),
                    ("LENGTH",   f"{sel.get('duration', 0)}s"),
                    ("LEVEL",    DIFF_STARS.get(sel.get("difficulty", 0), "-")),
                ]
                avail_h   = panel.height - start_h - start_m * 2 - 16
                item_h    = min(54, max(38, avail_h // max(len(detail_items), 1)))
                py_detail = panel.y + 14

                for lbl_txt, val_txt in detail_items:
                    lbl_s = self._fonts["small"].render(lbl_txt, True, (150, 140, 190))
                    val_s = self._fonts["body"].render(str(val_txt), True,
                                                        self._neon_color(mode_col, tick, 0.85))
                    self._display.blit(lbl_s, (panel.x + 16, py_detail))
                    self._display.blit(val_s, (panel.x + 16, py_detail + 17))
                    py_detail += item_h

                # 시작 버튼
                start_rect = pygame.Rect(panel.x + 16,
                                         panel.y + panel.height - start_h - start_m,
                                         pw - 32, start_h)
                self._btn_rects["btn_song_start"] = start_rect
                hover_s   = start_rect.collidepoint(mouse_pos)
                focused_s = (getattr(self, '_song_panel', 0) == 1 and
                             getattr(self, '_song_btn_idx', 0) == 0)

                btn_bg = pygame.Surface((start_rect.width, start_rect.height), pygame.SRCALPHA)
                btn_bg.fill((*[c // 3 for c in mode_col], 220 if (hover_s or focused_s) else 160))
                self._display.blit(btn_bg, start_rect.topleft)
                self._draw_neon_rect(self._display, start_rect,
                                     self._neon_color(mode_col, tick),
                                     width=2, radius=14, glow_radius=10 if (hover_s or focused_s) else 6)
                if hover_s or focused_s:
                    self._draw_corner_brackets(self._display, start_rect,
                                               self._neon_color(mode_col, tick * 2),
                                               size=14, width=3)
                go_s = self._fonts["menu"].render("START!", True, (255, 255, 255))
                self._display.blit(go_s, go_s.get_rect(center=start_rect.center))

        # ── 뒤로가기 버튼 ────────────────────────────────────────
        back_y    = h - BOTTOM_MARGIN - FOOTER_H + 6
        back_rect = pygame.Rect(14, back_y, 140, 38)
        self._btn_rects["btn_song_back"] = back_rect
        hover_b   = back_rect.collidepoint(mouse_pos)
        focused_b = (getattr(self, '_song_panel', 0) == 1 and
                     getattr(self, '_song_btn_idx', 0) == 1)

        bb_surf = pygame.Surface((140, 38), pygame.SRCALPHA)
        bb_surf.fill((40, 25, 70, 200 if (hover_b or focused_b) else 140))
        self._display.blit(bb_surf, back_rect.topleft)
        self._draw_neon_rect(self._display, back_rect,
                             self._neon_color((160, 140, 220), tick) if (hover_b or focused_b) else (80, 65, 120),
                             width=2, radius=10, glow_radius=6 if (hover_b or focused_b) else 2)
        back_s = self._fonts["body"].render("< BACK", True,
                                             (220, 215, 240) if (hover_b or focused_b) else (160, 155, 185))
        self._display.blit(back_s, back_s.get_rect(center=back_rect.center))

        # ── 패널 포커스 위치 안내 (우측 하단) ──────────────────────
        if getattr(self, '_song_panel', 0) == 0:
            nav_hint = "UP/DOWN: SELECT SONG   RIGHT: DETAILS   ESC: BACK"
        else:
            nav_hint = "UP/DOWN: START/BACK   LEFT: SONG LIST   ENTER: CONFIRM"
        hint = self._fonts["small"].render(nav_hint, True, (130, 110, 170))
        self._display.blit(hint, hint.get_rect(center=(w // 2, h - 14)))

    def _render_ready(self, w, h):
        """준비 화면: 좌=웹캠, 우=스켈레톤 + 전신 감지 안내 + OK 사인 대기."""
        import pygame
        import cv2

        MID_X = w // 2
        HEADER_H = 48
        FOOTER_H = 44

        # 배경
        self._display.fill((8, 6, 22))
        pygame.draw.line(self._display, (50, 45, 90),
                         (MID_X, HEADER_H), (MID_X, h - FOOTER_H), 2)

        # ── 헤더 ──────────────────────────────────────────────
        pygame.draw.rect(self._display, (18, 14, 45), pygame.Rect(0, 0, w, HEADER_H))
        song_title = (self._current_song or {}).get("title", "")
        hdr = self._fonts["body"].render(
            f"{'  ' + song_title + '  --  ' if song_title else ''}GET READY!",
            True, (180, 180, 255))
        self._display.blit(hdr, hdr.get_rect(center=(w // 2, HEADER_H // 2)))

        body_h = h - HEADER_H - FOOTER_H

        # ── 왼쪽: 웹캠 피드 ──────────────────────────────────
        pygame.draw.rect(self._display, (12, 10, 30),
                         pygame.Rect(0, HEADER_H, MID_X, body_h))
        lbl_cam = self._fonts["small"].render("MY CAM", True, (100, 160, 255))
        self._display.blit(lbl_cam, (12, HEADER_H + 8))

        if self._ready_current_frame is not None:
            cam_w = MID_X - 20
            cam_h = int(cam_w * 3 / 4)
            cam_x = 10
            cam_y = HEADER_H + (body_h - cam_h) // 2
            frame_small = cv2.resize(self._ready_current_frame, (cam_w, cam_h),
                                     interpolation=cv2.INTER_NEAREST)
            rgb = frame_small[:, :, ::-1]
            cam_surf = pygame.image.frombuffer(rgb.tobytes(), (cam_w, cam_h), "RGB")
            pygame.draw.rect(self._display, (40, 40, 70),
                             pygame.Rect(cam_x - 2, cam_y - 2, cam_w + 4, cam_h + 4),
                             border_radius=6)
            self._display.blit(cam_surf, (cam_x, cam_y))
        else:
            no_cam = self._fonts["body"].render("NO CAMERA", True, (80, 80, 110))
            self._display.blit(no_cam, no_cam.get_rect(center=(MID_X // 2, HEADER_H + body_h // 2)))

        # ── 오른쪽: 스켈레톤 ─────────────────────────────────
        pygame.draw.rect(self._display, (10, 8, 28),
                         pygame.Rect(MID_X, HEADER_H, w - MID_X, body_h))
        lbl_sk = self._fonts["small"].render("MY POSE", True, (255, 160, 80))
        self._display.blit(lbl_sk, (MID_X + 12, HEADER_H + 8))

        # 전신 감지 여부 판단
        full_body_detected = False
        if self._ready_pose_detected and self._ready_landmarks is not None:
            lm = self._ready_landmarks
            ANKLES = [27, 28]
            full_body_detected = all(lm[i][3] > 0.3 for i in ANKLES)

        if self._ready_pose_detected and self._ready_landmarks is not None:
            sk_rect = (MID_X, HEADER_H, w - MID_X, body_h)
            self._draw_stick_figure(
                self._display,
                self._ready_landmarks,
                sk_rect,
                line_color=(80, 220, 255) if full_body_detected else (200, 120, 60),
                joint_color=(200, 250, 255) if full_body_detected else (255, 200, 120),
                line_width=4,
                joint_radius=6,
            )
        else:
            no_pose = self._fonts["body"].render("DETECTING...", True, (80, 80, 120))
            self._display.blit(no_pose,
                               no_pose.get_rect(center=(MID_X + (w - MID_X) // 2,
                                                        HEADER_H + body_h // 2)))

        # ── 안내 문구 (OK 사인 진행 바 포함) ─────────────────
        fy = h - FOOTER_H
        pygame.draw.rect(self._display, (14, 12, 38), pygame.Rect(0, fy, w, FOOTER_H))
        pygame.draw.line(self._display, (50, 45, 90), (0, fy), (w, fy), 1)

        ok_progress = getattr(self, '_ready_countdown', 0.0)
        COUNTDOWN_SEC = 3.0
        full_body_start = getattr(self, '_ready_full_body_start', 0.0)
        is_counting = full_body_start > 0.0

        if not self._ready_pose_detected:
            msg = "STAND IN FRONT OF CAMERA!"
            msg_color = (200, 150, 80)
        elif not full_body_detected:
            msg = "STEP BACK UNTIL FULL BODY IS VISIBLE!"
            msg_color = (255, 160, 60)
        elif is_counting:
            remaining = max(0.0, ok_progress)
            msg = f"STARTING IN  {remaining:.1f}s..."
            msg_color = (0, 240, 150)
            # 진행 바 (3초 → 0초, 채워지는 방향)
            pct = 1.0 - remaining / COUNTDOWN_SEC
            bar_w = int((w - 40) * pct)
            pygame.draw.rect(self._display, (0, 50, 30),
                             pygame.Rect(20, fy + FOOTER_H - 8, w - 40, 5), border_radius=3)
            pygame.draw.rect(self._display, (0, 240, 150),
                             pygame.Rect(20, fy + FOOTER_H - 8, bar_w, 5), border_radius=3)
        else:
            msg = "FULL BODY DETECTED -- AUTO START IN 3s!"
            msg_color = (100, 220, 255)

        msg_surf = self._fonts["body"].render(msg, True, msg_color)
        self._display.blit(msg_surf, msg_surf.get_rect(center=(w // 2, fy + FOOTER_H // 2 - 2)))

        # 스킵 버튼
        skip_rect = pygame.Rect(w - 220, HEADER_H + 6, 100, 36)
        self._btn_rects["btn_ready_skip"] = skip_rect
        hover_s = skip_rect.collidepoint(pygame.mouse.get_pos())
        focused_s = (getattr(self, '_generic_focus_idx', 0) == 0)
        active_s = hover_s or focused_s
        pygame.draw.rect(self._display, (35, 100, 60) if active_s else (25, 70, 40),
                         skip_rect, border_radius=10)
        pygame.draw.rect(self._display, (255, 255, 100) if focused_s else (100, 200, 150),
                         skip_rect, 3 if focused_s else 2, border_radius=10)
        skip_lbl = self._fonts["small"].render("SKIP", True, (255, 255, 255) if active_s else (200, 255, 220))
        self._display.blit(skip_lbl, skip_lbl.get_rect(center=skip_rect.center))

        # 취소 버튼
        cancel_rect = pygame.Rect(w - 110, HEADER_H + 6, 100, 36)
        self._btn_rects["btn_ready_cancel"] = cancel_rect
        hover = cancel_rect.collidepoint(pygame.mouse.get_pos())
        focused_c = (getattr(self, '_generic_focus_idx', 0) == 1)
        active_c = hover or focused_c
        pygame.draw.rect(self._display, (100, 35, 35) if active_c else (70, 25, 25),
                         cancel_rect, border_radius=10)
        pygame.draw.rect(self._display, (255, 255, 100) if focused_c else (200, 100, 100),
                         cancel_rect, 3 if focused_c else 2, border_radius=10)
        cancel_lbl = self._fonts["small"].render("CANCEL", True, (255, 255, 255) if active_c else (255, 200, 200))
        self._display.blit(cancel_lbl, cancel_lbl.get_rect(center=cancel_rect.center))

    def _render_countdown(self, w, h):
        """Render countdown screen."""
        import pygame

        self._display.fill((10, 5, 30))

        count = max(self._countdown_timer, 0)

        txt_surface = self._fonts["countdown"].render(str(count + 1), True, (0, 255, 255))
        self._display.blit(txt_surface, txt_surface.get_rect(center=(w // 2, h // 2)))

        sub = self._fonts["body"].render("GET READY!", True, (180, 180, 220))
        self._display.blit(sub, sub.get_rect(center=(w // 2, h // 2 + 100)))

        # 취소 버튼 (터치로 메뉴로 복귀)
        cancel_rect = pygame.Rect(w // 2 - 100, h - 80, 200, 48)
        self._btn_rects["btn_countdown_cancel"] = cancel_rect
        hover = cancel_rect.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self._display, (120, 40, 40) if hover else (80, 30, 30),
                         cancel_rect, border_radius=12)
        pygame.draw.rect(self._display, (200, 100, 100), cancel_rect, 2, border_radius=12)
        lbl = self._fonts["body"].render("CANCEL", True, (255, 200, 200))
        self._display.blit(lbl, lbl.get_rect(center=cancel_rect.center))

    # ──────────────────────────────────────────────────────────
    #  공통 UI 헬퍼
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _neon_color(base_color: tuple, tick: float, intensity: float = 1.0) -> tuple:
        """시간에 따라 살짝 맥동하는 네온 색상을 반환합니다."""
        import math
        pulse = 0.75 + 0.25 * math.sin(tick * 3.5)
        pulse *= intensity
        return tuple(min(255, int(c * pulse)) for c in base_color)

    @staticmethod
    def _draw_neon_rect(surface, rect, color, width=2, radius=14, glow_radius=8):
        """네온 글로우 효과가 있는 둥근 사각형을 그립니다."""
        import pygame
        # 글로우 레이어 (바깥쪽부터 안쪽으로 점점 밝게)
        for i in range(glow_radius, 0, -1):
            alpha = int(80 * (1 - i / glow_radius))
            glow_surf = pygame.Surface((rect.width + i * 2, rect.height + i * 2),
                                       pygame.SRCALPHA)
            glow_color = (*color, alpha)
            pygame.draw.rect(glow_surf, glow_color,
                             pygame.Rect(0, 0, rect.width + i * 2, rect.height + i * 2),
                             border_radius=radius + i)
            surface.blit(glow_surf, (rect.x - i, rect.y - i))
        # 실제 테두리
        pygame.draw.rect(surface, color, rect, width, border_radius=radius)

    @staticmethod
    def _draw_corner_brackets(surface, rect, color, size=14, width=3):
        """선택된 항목의 네 모서리에 대괄호 형태의 하이라이트를 그립니다."""
        import pygame
        x, y, w, h = rect.x, rect.y, rect.width, rect.height
        corners = [
            # (시작점, 수평 끝, 수직 끝)
            ((x, y),         (x + size, y),       (x, y + size)),
            ((x + w, y),     (x + w - size, y),   (x + w, y + size)),
            ((x, y + h),     (x + size, y + h),   (x, y + h - size)),
            ((x + w, y + h), (x + w - size, y + h), (x + w, y + h - size)),
        ]
        for origin, h_end, v_end in corners:
            pygame.draw.line(surface, color, origin, h_end, width)
            pygame.draw.line(surface, color, origin, v_end, width)

    def _draw_stick_figure(self, surface, landmarks_33x4, panel_rect,
                           line_color=(0, 200, 255), joint_color=(255, 255, 255),
                           line_width=3, joint_radius=6, visibility_threshold=0.3):
        """
        panel_rect 영역 안에 스틱 피겨를 그립니다.
        좌표 범위가 0~1이 아닌 경우에도 자동으로 패널에 맞게 정규화합니다.

        Args:
            surface: pygame.Surface
            landmarks_33x4: numpy array (33, 4) — x, y, z, visibility
            panel_rect: (px, py, pw, ph) 패널 좌상단과 크기
            line_color: 뼈대 선 색상
            joint_color: 관절 점 색상
        """
        import pygame
        import numpy as np
        from pose.landmark_utils import SKELETON_CONNECTIONS, DANCE_JOINTS

        px, py, pw, ph = panel_rect

        # 패딩 적용 (캐릭터가 패널 가장자리에 붙지 않도록)
        pad_x = int(pw * 0.12)
        pad_y = int(ph * 0.08)
        draw_x = px + pad_x
        draw_y = py + pad_y
        draw_w = pw - pad_x * 2
        draw_h = ph - pad_y * 2

        # ── 가시성 임계값 이상인 랜드마크만 추출하여 좌표 범위 계산 ──
        visible_mask = landmarks_33x4[:, 3] > visibility_threshold
        if not np.any(visible_mask):
            return  # 보이는 랜드마크가 없으면 건너뜀

        vis_lm = landmarks_33x4[visible_mask]
        x_vals = vis_lm[:, 0]
        y_vals = vis_lm[:, 1]

        x_min, x_max = float(x_vals.min()), float(x_vals.max())
        y_min, y_max = float(y_vals.min()), float(y_vals.max())

        # 범위가 0이면 (한 점) 약간 여유를 줌
        x_range = x_max - x_min if (x_max - x_min) > 1e-6 else 1.0
        y_range = y_max - y_min if (y_max - y_min) > 1e-6 else 1.0

        # 종횡비를 유지하면서 패널에 맞추기
        scale = min(draw_w / x_range, draw_h / y_range)
        fig_w = x_range * scale
        fig_h = y_range * scale
        offset_x = draw_x + (draw_w - fig_w) / 2
        offset_y = draw_y + (draw_h - fig_h) / 2

        def to_px(lm):
            nx = (lm[0] - x_min) / x_range  # 0~1 정규화
            ny = (lm[1] - y_min) / y_range
            return (int(offset_x + nx * fig_w),
                    int(offset_y + ny * fig_h))

        # 뼈대 선
        for src, dst in SKELETON_CONNECTIONS:
            if (landmarks_33x4[src][3] > visibility_threshold and
                    landmarks_33x4[dst][3] > visibility_threshold):
                p1 = to_px(landmarks_33x4[src])
                p2 = to_px(landmarks_33x4[dst])
                pygame.draw.line(surface, line_color, p1, p2, line_width)

        # 관절 점
        for idx in DANCE_JOINTS:
            if landmarks_33x4[idx][3] > visibility_threshold:
                cx, cy = to_px(landmarks_33x4[idx])
                pygame.draw.circle(surface, joint_color, (cx, cy), joint_radius)
                # 내부 밝은 점
                pygame.draw.circle(surface, (255, 255, 255), (cx, cy), max(joint_radius - 3, 2))

    def _render_gameplay(self, w, h):
        """Render gameplay screen — dual panel layout.

        LEFT  : 웹캠 전체 화면 + 스켈레톤 오버레이  (파란 네온 테두리)
        RIGHT : 레퍼런스 영상 (또는 스틱피겨)        (노란/오렌지 네온 테두리)
        HEADER: 점수/콤보/시간/버튼
        FOOTER: 곡 정보/조작 안내
        """
        import pygame
        import cv2
        import numpy as np
        import math

        HEADER_H = 55
        FOOTER_H = 30
        BORDER   = 4          # 패널 테두리 두께
        MID_X    = w // 2
        BODY_Y   = HEADER_H
        BODY_H   = h - HEADER_H - FOOTER_H
        tick     = self._neon_tick

        # ── 배경 ──────────────────────────────────────────────────
        self._display.fill((6, 4, 18))

        # ── 패널 영역 ──────────────────────────────────────────────
        left_rect  = pygame.Rect(0,     BODY_Y, MID_X,     BODY_H)
        right_rect = pygame.Rect(MID_X, BODY_Y, w - MID_X, BODY_H)

        # ══════════════════════════════════════════════════════
        #  LEFT — 웹캠 + 스켈레톤 오버레이 (전체 패널 크기)
        # ══════════════════════════════════════════════════════
        pygame.draw.rect(self._display, (8, 8, 22), left_rect)

        if hasattr(self, '_current_frame') and self._current_frame is not None:
            cam_w_target = MID_X - BORDER * 2
            cam_h_target = BODY_H - BORDER * 2
            frame_bgr = cv2.resize(self._current_frame,
                                   (cam_w_target, cam_h_target),
                                   interpolation=cv2.INTER_NEAREST)

            # 스켈레톤을 카메라 프레임 위에 직접 그리기
            if hasattr(self, '_pose_detected') and self._pose_detected and \
                    self._current_landmarks is not None:
                from pose.landmark_utils import SKELETON_CONNECTIONS, DANCE_JOINTS
                lm = self._current_landmarks
                for src, dst in SKELETON_CONNECTIONS:
                    if src < len(lm) and dst < len(lm) and \
                       lm[src][3] > 0.3 and lm[dst][3] > 0.3:
                        x1 = int(lm[src][0] * cam_w_target)
                        y1 = int(lm[src][1] * cam_h_target)
                        x2 = int(lm[dst][0] * cam_w_target)
                        y2 = int(lm[dst][1] * cam_h_target)
                        cv2.line(frame_bgr, (x1, y1), (x2, y2), (0, 255, 180), 3)
                for idx in DANCE_JOINTS:
                    if idx < len(lm) and lm[idx][3] > 0.3:
                        cx_ = int(lm[idx][0] * cam_w_target)
                        cy_ = int(lm[idx][1] * cam_h_target)
                        cv2.circle(frame_bgr, (cx_, cy_), 5, (0, 255, 255), -1)
                        cv2.circle(frame_bgr, (cx_, cy_), 8, (0, 200, 150), 2)

            frame_rgb = frame_bgr[:, :, ::-1]
            cam_surf  = pygame.image.frombuffer(
                frame_rgb.tobytes(), (cam_w_target, cam_h_target), "RGB")
            self._display.blit(cam_surf, (BORDER, BODY_Y + BORDER))
        else:
            no_cam = self._fonts["body"].render("NO CAMERA", True, (80, 80, 110))
            self._display.blit(no_cam, no_cam.get_rect(center=left_rect.center))

        # 좌 패널 레이블은 나중에 테두리와 함께 그린다 (effects 위)

        # ══════════════════════════════════════════════════════
        #  RIGHT — 레퍼런스 영상 / 스틱피겨
        # ══════════════════════════════════════════════════════
        pygame.draw.rect(self._display, (10, 6, 22), right_rect)

        has_video = self._ref_video_cap is not None

        if has_video and self._ref_video_surf is not None:
            vx, vy = self._ref_video_pos
            self._display.blit(self._ref_video_surf, (vx, vy))
        elif not has_video and self._ref_frame_landmarks is not None:
            right_fig_rect = (right_rect.x, BODY_Y, right_rect.width, BODY_H)
            self._draw_stick_figure(
                self._display,
                self._ref_frame_landmarks,
                right_fig_rect,
                line_color=(255, 180, 60),
                joint_color=(255, 230, 120),
                line_width=4,
                joint_radius=6,
            )
        else:
            no_guide = self._fonts["body"].render("NO GUIDE", True, (80, 70, 60))
            self._display.blit(no_guide, no_guide.get_rect(center=right_rect.center))

        # 우 패널 레이블은 나중에 테두리와 함께 그린다 (effects 위)

        # ── 파티클 업데이트 & 렌더링 ────────────────────────────────
        self._update_and_draw_particles()

        # ── 피드백 이펙트 오버레이 ──────────────────────────────────
        if self._last_feedback and self._feedback_timer > 0:
            self._draw_feedback_effect(w, h, MID_X, BODY_Y, BODY_H)

        # ── 패널 풀 테두리 + 레이블 (effects 위에 그려 항상 보임) ─────
        left_border_col  = self._neon_color((60, 180, 255), tick, 0.9)
        right_border_col = self._neon_color((255, 160, 40), tick, 0.9)
        # glow_radius=0 → 안쪽으로 번지지 않는 단순 테두리
        self._draw_neon_rect(self._display, left_rect,  left_border_col,  width=3, radius=0, glow_radius=0)
        self._draw_neon_rect(self._display, right_rect, right_border_col, width=3, radius=0, glow_radius=0)

        lbl_me    = self._fonts["small_retro"].render("ME",    True, (120, 200, 255))
        lbl_guide = self._fonts["small_retro"].render("GUIDE", True, (255, 180, 80))
        self._display.blit(lbl_me,    (BORDER + 8,           BODY_Y + BORDER + 6))
        self._display.blit(lbl_guide, (MID_X + BORDER + 8,  BODY_Y + BORDER + 6))

        # ══════════════════════════════════════════════════════
        #  HEADER
        # ══════════════════════════════════════════════════════
        header_bg = pygame.Rect(0, 0, w, HEADER_H)
        pygame.draw.rect(self._display, (12, 8, 30), header_bg)
        # 헤더 하단 네온 라인
        hdr_line_col = self._neon_color((80, 60, 160), tick, 0.5)
        pygame.draw.line(self._display, hdr_line_col, (0, HEADER_H), (w, HEADER_H), 2)

        # 점수 (레트로 폰트)
        score_col = self._neon_color((0, 255, 200), tick)
        score_surf = self._fonts["score"].render(
            f"{int(self._scorer.total_score):06d}", True, score_col)
        self._display.blit(score_surf, score_surf.get_rect(midleft=(16, HEADER_H // 2)))

        # 콤보
        combo_val = self._scorer.combo
        if combo_val > 0:
            combo_col = self._neon_color((255, 230, 0), tick)
            combo_surf = self._fonts["score"].render(f"{combo_val}x COMBO", True, combo_col)
            self._display.blit(combo_surf, combo_surf.get_rect(center=(w // 2, HEADER_H // 2)))

        # 남은 시간
        elapsed  = self._current_session.elapsed_time if self._current_session else 0
        duration = (self._current_song or {}).get("duration", 60)
        remain   = max(0, duration - elapsed)
        time_col = (220, 200, 255) if remain > 10 else self._neon_color((255, 80, 80), tick)
        time_surf = self._fonts["score"].render(
            f"{int(remain // 60):02d}:{int(remain % 60):02d}", True, time_col)
        self._display.blit(time_surf, time_surf.get_rect(midright=(w - 230, HEADER_H // 2)))

        # 헤더 버튼
        mouse_pos = pygame.mouse.get_pos()
        pause_rect = pygame.Rect(w - 220, 8, 90, 38)
        menu_rect  = pygame.Rect(w - 120, 8, 90, 38)
        self._btn_rects["btn_pause"]         = pause_rect
        self._btn_rects["btn_gameplay_menu"] = menu_rect

        for rect, label, base_c in [
            (pause_rect, "PAUSE", (55, 55, 130)),
            (menu_rect,  "MENU",  (100, 38, 38)),
        ]:
            hover = rect.collidepoint(mouse_pos)
            color = tuple(min(c + 40, 255) for c in base_c) if hover else base_c
            pygame.draw.rect(self._display, color, rect, border_radius=8)
            pygame.draw.rect(self._display, (160, 160, 210), rect, 1, border_radius=8)
            lbl = self._fonts["btn_retro"].render(label, True, (240, 240, 240))
            self._display.blit(lbl, lbl.get_rect(center=rect.center))

        # ══════════════════════════════════════════════════════
        #  FOOTER
        # ══════════════════════════════════════════════════════
        fy = h - FOOTER_H
        pygame.draw.line(self._display, (40, 35, 70), (0, fy), (w, fy), 1)

        song_title  = (self._current_song or {}).get("title", "데모")
        mode_labels = {"practice": "PRACTICE", "challenge": "CHALLENGE", "freestyle": "FREE"}
        mode_label  = mode_labels.get(self._current_mode, "")
        footer_left = self._fonts["small"].render(
            f"[{mode_label}]  {song_title}", True, (120, 160, 220))
        self._display.blit(footer_left, (14, fy + 6))

        footer_right = self._fonts["small_retro"].render(
            "P: PAUSE  |  ESC: MENU", True, (80, 80, 110))
        self._display.blit(footer_right,
                           footer_right.get_rect(midright=(w - 10, fy + FOOTER_H // 2)))

    def _update_and_draw_particles(self):
        """파티클 업데이트 + 화면 그리기."""
        import pygame
        import math

        alive = []
        for p in self._particles:
            p['life'] -= 1
            if p['life'] <= 0:
                continue
            p['x']  += p['vx']
            p['y']  += p['vy']
            p['vy'] += 0.35   # 중력
            p['vx'] *= 0.97   # 공기 저항
            ratio = p['life'] / p['max_life']
            alpha = int(255 * ratio)
            size  = max(1, int(p['size'] * ratio))
            r, g, b = p['color']
            surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (r, g, b, alpha), (size, size), size)
            self._display.blit(surf, (int(p['x']) - size, int(p['y']) - size))
            alive.append(p)
        self._particles = alive

    def _spawn_particles(self, cx: int, cy: int, color: tuple, count: int = 30):
        """피드백 위치에서 파티클 폭발 생성."""
        import random, math
        for _ in range(count):
            angle = random.uniform(0, math.pi * 2)
            speed = random.uniform(5.0, 16.0)   # 빠른 속도
            life  = random.randint(10, 22)       # 짧은 수명 (빠르게 사라짐)
            size  = random.randint(4, 10)
            r = min(255, max(0, color[0] + random.randint(-40, 40)))
            g = min(255, max(0, color[1] + random.randint(-40, 40)))
            b = min(255, max(0, color[2] + random.randint(-40, 40)))
            self._particles.append({
                'x': cx, 'y': cy,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed - 3.0,
                'life': life, 'max_life': life,
                'color': (r, g, b),
                'size': size,
            })

    def _draw_feedback_effect(self, w, h, mid_x, body_y, body_h):
        """피드백 등급 텍스트 + 네온 테두리 + 바운스 애니메이션.

        _feedback_timer : 남은 표시 시간 (1.2→0, 매 프레임 리셋될 수 있음)
        _feedback_age   : 등장 후 경과 시간 (0→..., 리셋 시 0으로 초기화)
        """
        import pygame

        fb    = self._last_feedback
        timer = self._feedback_timer   # 남은 시간 — alpha fade-out에 사용
        age   = self._feedback_age     # 경과 시간 — 바운스 애니메이션에 사용

        color = fb.get("color", (255, 255, 255))
        text  = fb.get("text", "")
        pts   = fb.get("points", 0)
        combo = fb.get("combo", 0)

        # ── alpha: 마지막 0.35초에만 fade-out, 평소엔 완전 불투명 ──
        alpha = int(255 * min(timer / 0.35, 1.0))
        alpha = max(0, min(255, alpha))

        # ── scale: 등장 0.12초 동안 바운스 (age 기반) ──
        if age < 0.12:
            t     = age / 0.12
            scale = 0.5 + t * 0.7        # 0.5 → 1.2
        elif age < 0.18:
            t     = (age - 0.12) / 0.06
            scale = 1.2 - t * 0.2        # 1.2 → 1.0
        else:
            scale = 1.0

        cx = w // 2
        cy = body_y + int(body_h * 0.38)

        # ── 피드백 텍스트 ──
        base_surf = self._fonts["feedback"].render(text, True, color)
        if scale != 1.0:
            new_w = max(1, int(base_surf.get_width() * scale))
            new_h = max(1, int(base_surf.get_height() * scale))
            base_surf = pygame.transform.scale(base_surf, (new_w, new_h))
        base_surf.set_alpha(alpha)

        tw, th = base_surf.get_width(), base_surf.get_height()
        tx, ty = cx - tw // 2, cy - th // 2

        # 글로우
        if alpha > 60:
            gw = max(1, int(tw * 1.18))
            gh = max(1, int(th * 1.18))
            glow_base = self._fonts["feedback"].render(text, True,
                tuple(min(255, c + 60) for c in color))
            glow_surf = pygame.transform.scale(glow_base, (gw, gh))
            glow_surf.set_alpha(int(alpha * 0.35))
            self._display.blit(glow_surf, (cx - gw // 2, cy - gh // 2))

        # 네온 테두리 박스
        pad = 18
        box_rect = pygame.Rect(tx - pad, ty - pad // 2, tw + pad * 2, th + pad)
        if alpha > 60:
            box_surf = pygame.Surface((box_rect.width, box_rect.height), pygame.SRCALPHA)
            box_surf.fill((*tuple(c // 5 for c in color), int(alpha * 0.45)))
            self._display.blit(box_surf, box_rect.topleft)
            pygame.draw.rect(self._display, color,
                             pygame.Rect(box_rect.x - 2, box_rect.y - 2,
                                         box_rect.width + 4, box_rect.height + 4),
                             3, border_radius=10)
            pygame.draw.rect(self._display, tuple(min(255, c + 80) for c in color),
                             box_rect, 1, border_radius=8)

        self._display.blit(base_surf, (tx, ty))

        # +점수
        if pts > 0 and alpha > 30:
            pts_surf = self._fonts["feedback"].render(f"+{pts}", True, (255, 240, 80))
            pts_surf = pygame.transform.scale(pts_surf,
                (max(1, int(pts_surf.get_width() * 0.6)),
                 max(1, int(pts_surf.get_height() * 0.6))))
            pts_surf.set_alpha(alpha)
            self._display.blit(pts_surf,
                pts_surf.get_rect(midtop=(cx, ty + th + pad // 2 + 4)))

        # 콤보
        if combo >= 3 and alpha > 30:
            combo_col = self._neon_color((255, 200, 0), self._neon_tick)
            combo_surf = self._fonts["score"].render(f"{combo}x COMBO!!", True, combo_col)
            combo_surf.set_alpha(int(alpha * 0.9))
            self._display.blit(combo_surf,
                combo_surf.get_rect(midbottom=(cx, ty - pad // 2 - 4)))

    def _render_pause_overlay(self, w, h):
        """게임플레이 위에 반투명 일시정지 오버레이를 렌더링합니다."""
        import pygame

        # 반투명 어두운 오버레이
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self._display.blit(overlay, (0, 0))

        # 타이틀
        pause_txt = self._fonts["menu"].render("PAUSED", True, (255, 255, 255))
        self._display.blit(pause_txt, pause_txt.get_rect(center=(w // 2, h // 2 - 80)))

        mouse_pos = pygame.mouse.get_pos()

        btn_defs = [
            ("btn_pause",         "RESUME",    (0, 160, 100)),
            ("btn_gameplay_menu", "MAIN MENU", (140, 50, 50)),
        ]
        for i, (btn_name, label, color) in enumerate(btn_defs):
            rect = pygame.Rect(w // 2 - 160, h // 2 + i * 72, 320, 54)
            self._btn_rects[btn_name] = rect   # 오버레이 버튼으로 덮어쓰기
            hover = rect.collidepoint(mouse_pos)
            focused = (getattr(self, '_generic_focus_idx', 0) == i)
            draw_color = tuple(min(c + 50, 255) for c in color) if (hover or focused) else color
            pygame.draw.rect(self._display, draw_color, rect, border_radius=14)
            border_col = (255, 255, 100) if focused else (220, 220, 220)
            border_w   = 3 if focused else 2
            pygame.draw.rect(self._display, border_col, rect, border_w, border_radius=14)
            lbl = self._fonts["menu"].render(label, True, (255, 255, 255))
            self._display.blit(lbl, lbl.get_rect(center=rect.center))

        hint = self._fonts["small"].render("P / SPACE: RESUME  |  ESC: MENU", True, (160, 160, 180))
        self._display.blit(hint, hint.get_rect(center=(w // 2, h - 30)))

    def _render_result(self, w, h):
        """Render result screen."""
        import pygame

        tick = self._neon_tick

        # 배경 그라데이션
        for y_i in range(h):
            t = y_i / h
            pygame.draw.line(self._display,
                             (int(10 + 10*t), int(5 + 5*t), int(30 + 20*t)),
                             (0, y_i), (w, y_i))
        # 스캔라인 효과 (레트로)
        for y_i in range(0, h, 4):
            scan = pygame.Surface((w, 1), pygame.SRCALPHA)
            scan.fill((0, 0, 0, 40))
            self._display.blit(scan, (0, y_i))

        MARGIN_TOP = 30
        MARGIN_BOTTOM = 40
        FOOTER_H = 24
        BTN_H = 50
        BTN_W = min(200, (w - 60) // 2)

        # 타이틀 (레트로 폰트 + 네온 글로우)
        title_col = self._neon_color((255, 220, 50), tick)
        title = self._fonts["result_big"].render("DANCE COMPLETE!", True, title_col)
        # 글로우
        glow = self._fonts["result_big"].render("DANCE COMPLETE!", True, (100, 80, 0))
        for dx, dy in [(-3,0),(3,0),(0,-3),(0,3)]:
            self._display.blit(glow, glow.get_rect(center=(w // 2 + dx, MARGIN_TOP + 30 + dy)))
        title_rect = title.get_rect(center=(w // 2, MARGIN_TOP + 30))
        self._display.blit(title, title_rect)

        # 구분선 (네온)
        line_col = self._neon_color((200, 100, 255), tick, 0.7)
        pygame.draw.line(self._display, line_col,
                         (w // 4, MARGIN_TOP + 54), (w * 3 // 4, MARGIN_TOP + 54), 1)

        # 하단 버튼/푸터 영역 계산
        btn_area_y = h - MARGIN_BOTTOM - FOOTER_H - BTN_H - 10
        content_top = MARGIN_TOP + 70
        content_bottom = btn_area_y - 20

        if self._result_data:
            data = self._result_data
            items = [
                (f"SCORE:     {data.get('total_score', 0)}",    (0, 255, 200)),
                (f"MAX COMBO: {data.get('max_combo', 0)}",      (255, 220, 0)),
                (f"MOVES:     {data.get('total_moves', 0)}",    (200, 200, 220)),
                (f"AVG:       {data.get('average_score', 0):.1f}", (180, 180, 255)),
                (f"GRADE:     {data.get('final_grade', '-')}",  (255, 180, 0)),
            ]

            hits = data.get("hit_counts", {})
            total_items = len(items) + (1 if hits else 0)
            item_gap = min(48, max(30, (content_bottom - content_top) // max(total_items, 1)))

            y = content_top
            for text, color in items:
                surf = self._fonts["result_big"].render(text, True, color)
                self._display.blit(surf, surf.get_rect(center=(w // 2, y)))
                y += item_gap

            if hits:
                y += 4
                hit_text = "  |  ".join(f"{k}: {v}" for k, v in hits.items())
                hit_surf = self._fonts["small_retro"].render(hit_text, True, (160, 160, 180))
                self._display.blit(hit_surf, hit_surf.get_rect(center=(w // 2, y)))

        # 버튼: 다시하기 / 메뉴 (하단 고정, 중앙 정렬)
        mouse_pos = pygame.mouse.get_pos()
        btn_gap = 20
        total_w = BTN_W * 2 + btn_gap
        btn_x = w // 2 - total_w // 2

        btn_defs = [
            ("btn_retry",       "RETRY",  (0, 140, 90)),
            ("btn_result_menu", "MENU",   (100, 40, 120)),
        ]
        for i, (btn_name, label, color) in enumerate(btn_defs):
            rect = pygame.Rect(btn_x + i * (BTN_W + btn_gap), btn_area_y, BTN_W, BTN_H)
            self._btn_rects[btn_name] = rect
            hover = rect.collidepoint(mouse_pos)
            focused = (getattr(self, '_generic_focus_idx', 0) == i)
            draw_color = tuple(min(c + 50, 255) for c in color) if (hover or focused) else color
            pygame.draw.rect(self._display, draw_color, rect, border_radius=14)
            border_col = (255, 255, 100) if focused else (220, 220, 220)
            border_w   = 3 if focused else 2
            pygame.draw.rect(self._display, border_col, rect, border_w, border_radius=14)
            lbl = self._fonts["btn_retro"].render(label, True, (255, 255, 255))
            self._display.blit(lbl, lbl.get_rect(center=rect.center))

        hint = self._fonts["small_retro"].render(
            "ENTER: MENU  |  ESC: MENU", True, (100, 100, 130)
        )
        self._display.blit(hint, hint.get_rect(center=(w // 2, h - MARGIN_BOTTOM // 2)))

    def _render_settings(self, w, h):
        """Render settings screen."""
        import pygame

        self._display.fill((15, 10, 40))

        MARGIN_TOP = 30
        MARGIN_BOTTOM = 40
        FOOTER_H = 24
        BTN_H = 50

        title = self._fonts["menu"].render("SETTINGS", True, (200, 200, 255))
        self._display.blit(title, title.get_rect(center=(w // 2, MARGIN_TOP + 30)))

        settings_items = [
            f"CAMERA:    {self.config.get('camera', {}).get('device_id', 0)}",
            f"RESOLUTION: {self.config.get('camera', {}).get('width', 640)}"
            f"x{self.config.get('camera', {}).get('height', 480)}",
            f"UI THEME:  {self.config.get('ui', {}).get('theme', 'neon')}",
            f"FULLSCREEN: {'YES' if self.config.get('ui', {}).get('fullscreen', False) else 'NO'}",
            f"METRIC:    {self.config.get('scoring', {}).get('similarity_metric', 'cosine')}",
            f"TARGET FPS: {self.TARGET_FPS}",
        ]

        # 버튼 영역
        btn_y = h - MARGIN_BOTTOM - FOOTER_H - BTN_H - 10
        content_top = MARGIN_TOP + 80
        content_bottom = btn_y - 20

        # 동적 간격 계산
        num_items = len(settings_items)
        item_gap = min(46, max(28, (content_bottom - content_top) // max(num_items, 1)))
        start_y = content_top + ((content_bottom - content_top) - item_gap * num_items) // 2

        for i, text in enumerate(settings_items):
            surf = self._fonts["body"].render(text, True, (180, 180, 200))
            self._display.blit(surf, (w // 2 - 220, start_y + i * item_gap))

        # 뒤로가기 버튼 (하단 고정)
        mouse_pos = pygame.mouse.get_pos()
        back_rect = pygame.Rect(w // 2 - 110, btn_y, 220, BTN_H)
        self._btn_rects["btn_back"] = back_rect
        hover   = back_rect.collidepoint(mouse_pos)
        focused = (getattr(self, '_generic_focus_idx', 0) == 0)
        pygame.draw.rect(self._display, (70, 70, 150) if (hover or focused) else (50, 50, 110),
                         back_rect, border_radius=14)
        border_col = (255, 255, 100) if focused else (180, 180, 230)
        border_w   = 3 if focused else 2
        pygame.draw.rect(self._display, border_col, back_rect, border_w, border_radius=14)
        lbl = self._fonts["body"].render("< BACK", True, (255, 255, 255))
        self._display.blit(lbl, lbl.get_rect(center=back_rect.center))

        hint = self._fonts["small"].render("ESC: BACK", True, (100, 100, 130))
        self._display.blit(hint, hint.get_rect(center=(w // 2, h - MARGIN_BOTTOM // 2)))

    def transition_to(self, new_state: str):
        """Transition to a new game state."""
        old_state = self.state
        self._prev_state = old_state
        self.state = new_state
        self._on_state_enter(new_state)

    def _on_state_enter(self, state: str):
        """Handle setup when entering a new state."""
        # 화면 전환 시 터치 포커스 항상 초기화
        self._touch_focused_btn = ""
        # generic 포커스 리셋 (PAUSED / RESULT / SETTINGS)
        self._generic_focus_idx = 0
        # 피드백 age 리셋
        self._feedback_age = 0.0

        if state == GameState.MENU:
            self._menu_focus_idx = 0
            self._release_reference_assets()
        elif state == GameState.SONG_SELECT:
            self._selected_song_idx = 0
            self._song_focus_idx = 0
            self._song_panel = 0      # 항상 곡목록 패널부터 시작
            self._song_btn_idx = 0
            self._release_reference_assets()
        elif state == GameState.READY:
            # 준비 화면 진입 시 상태 초기화
            self._ready_current_frame = None
            self._ready_landmarks = None
            self._ready_pose_detected = False
            self._ready_full_body_start = 0.0
            self._ready_countdown = 0.0
            # READY에서 곡 선택이 아직 안 된 경우 첫 번째 곡 자동 선택
            if not self._current_song:
                songs = self._songs_for_mode(self._current_mode)
                if songs:
                    self._current_song = songs[self._selected_song_idx]
            
            # 여기서 리소스를 미리 로드하여 PLAYING 시작 시 렉(스파이크)을 완전히 제거
            self._load_reference_assets()
        elif state == GameState.COUNTDOWN:
            self._countdown_start = time.time()
            self._countdown_timer = 2  # 3초 카운트다운 (워밍업 시간 확보)
        elif state == GameState.PLAYING:
            # PAUSED→PLAYING 복귀인 경우에만 세션 유지
            resuming_from_pause = getattr(self, '_prev_state', None) == GameState.PAUSED
            if resuming_from_pause:
                import pygame
                pygame.mixer.music.unpause()
            else:
                # 새 게임 시작: 이전 세션 완전 정리
                if self._current_session is not None:
                    self._current_session.is_active = False
                    self._current_session = None
                import pygame
                pygame.mixer.music.stop()
                self._scorer.reset()
                self._last_feedback = None
                self._feedback_timer = 0.0
                self._current_frame = None
                self._pose_detected = False
                self._last_valid_landmarks = None
                self._last_valid_landmarks_age_frames = 10**9
                if self._scratch_comparator is not None:
                    self._scratch_comparator.reset()
                if self._embedding_comparator is not None:
                    self._embedding_comparator.reset()
                self._warned_scratch_no_ref = False
                
                self._load_reference_assets()
                
                # 재생 바를 0초로 돌리기
                self._ref_frame_landmarks = None
                self._ref_video_frame = None
                if getattr(self, '_ref_video_cap', None) is not None:
                    import cv2 as _cv2
                    self._ref_video_cap.set(_cv2.CAP_PROP_POS_FRAMES, 0)
                    
                if getattr(self, '_audio_path', None) and os.path.exists(self._audio_path):
                    try:
                        import pygame
                        pygame.mixer.music.play()
                    except Exception as e:
                        print(f"[WARN] 오디오 재생 실패: {e}")
                from game.session import GameSession
                # 선택된 곡 정보 사용 (없으면 기본값)
                song = self._current_song or {}
                duration = song.get("duration") or \
                    self.config.get("game", {}).get("max_song_duration", 60)
                self._current_session = GameSession(
                    song.get("id", "demo"),
                    {"duration": duration},
                    self._current_mode,
                )
                self._current_session.start()
        elif state == GameState.PAUSED:
            # 일시정지 — 현재 세션 타이머는 계속 흐름 (추후 개선 가능)
            import pygame
            pygame.mixer.music.pause()
        elif state == GameState.RESULT:
            self._result_data = self._scorer.get_final_result()

    def _release_reference_assets(self):
        """이전 곡의 리소스를 해제합니다."""
        self._ref_landmarks = None
        self._ref_frame_landmarks = None
        if getattr(self, '_ref_video_cap', None) is not None:
            self._ref_video_cap.release()
            self._ref_video_cap = None
        self._ref_video_frame = None
        self._audio_path = None

    def _load_reference_assets(self):
        """PLAYING 진입 시의 초기 렉을 없애기 위해 미리 무거운 리소스(영상, Numpy 배열 등)를 로드해둡니다."""
        if getattr(self, '_ref_landmarks', None) is not None:
            return  # 이미 로드됨
            
        import os
        import subprocess
        import pygame
        import numpy as np
        import cv2 as _cv2

        self._ref_video_fps = 30.0

        if self._current_song and self._current_song.get("has_reference"):
            ref_path = os.path.join(self._current_song["path"], "reference.npy")
            try:
                ref_data = np.load(ref_path)
                if ref_data.ndim == 3 and ref_data.shape[2] == 3:
                    vis = np.ones((*ref_data.shape[:2], 1), dtype=np.float32)
                    ref_data = np.concatenate([ref_data, vis], axis=2)
                self._ref_landmarks = ref_data.astype(np.float32)
                # Reference embedding cache가 있으면 디스크에서 로드하고,
                # 없으면 기존처럼 백그라운드 TFLite warmup으로 fallback한다.
                dance_name = os.path.basename(self._current_song["path"])
                project_root = os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))))
                
                if getattr(self, '_scratch_comparator', None):
                    loaded = False
                    if getattr(self, '_scratch_use_reference_cache', False):
                        cache_dir = getattr(self, '_scratch_reference_cache_dir', "")
                        if cache_dir and not os.path.isabs(cache_dir):
                            cache_dir = os.path.join(project_root, cache_dir)
                        if cache_dir:
                            loaded = self._scratch_comparator.load_reference_embedding_cache(
                                cache_dir=cache_dir,
                                dance_name=dance_name,
                                model_kind="scratch",
                                model_name=getattr(self, '_scratch_model_name', ""),
                                reference_path=ref_path,
                            )
                    if not loaded:
                        self._scratch_comparator.warmup_reference_embeddings(self._ref_landmarks)
                elif getattr(self, '_embedding_comparator', None):
                    loaded = False
                    if getattr(self, '_embedding_use_reference_cache', False):
                        cache_dir = getattr(self, '_embedding_reference_cache_dir', "")
                        if cache_dir and not os.path.isabs(cache_dir):
                            cache_dir = os.path.join(project_root, cache_dir)
                        if cache_dir:
                            loaded = self._embedding_comparator.load_reference_embedding_cache(
                                cache_dir=cache_dir,
                                dance_name=dance_name,
                                model_kind="embedding",
                                model_name=getattr(self, '_embedding_model_name', ""),
                                reference_path=ref_path,
                            )
                    if not loaded:
                        self._embedding_comparator.warmup_reference_embeddings(self._ref_landmarks)
            except Exception as e:
                print(f"[WARN] 참조 랜드마크 로드 실패: {e}")

            video_rel = self._current_song.get("video", "")
            if video_rel:
                project_root = os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))))
                video_path = os.path.join(project_root, video_rel)
                if not os.path.exists(video_path):
                    video_path = os.path.join(os.getcwd(), video_rel)
                
                if os.path.exists(video_path):
                    self._ref_video_cap = _cv2.VideoCapture(video_path)
                    self._ref_video_fps = self._current_song.get(
                        "video_fps",
                        self._ref_video_cap.get(_cv2.CAP_PROP_FPS) or 30.0,
                    )
                    
                    # 영상 리사이즈 크기 미리 계산 
                    ui_cfg = self.config.get("ui", {})
                    disp_w = ui_cfg.get("window_width", 1024)
                    disp_h = ui_cfg.get("window_height", 600)
                    HEADER_H = 55
                    FOOTER_H = 30
                    rp_w = disp_w - disp_w // 2
                    rp_h = disp_h - HEADER_H - FOOTER_H
                    
                    total_frames = self._ref_video_cap.get(_cv2.CAP_PROP_FRAME_COUNT)
                    vid_w = int(self._ref_video_cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
                    vid_h = int(self._ref_video_cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    if vid_w > 0 and vid_h > 0:
                        vscale = min(rp_w / vid_w, rp_h / vid_h)
                        tw, th = int(vid_w * vscale), int(vid_h * vscale)
                        self._ref_video_size = (tw, th)
                        self._ref_video_pos = (
                            disp_w // 2 + (rp_w - tw) // 2,
                            HEADER_H + (rp_h - th) // 2,
                        )

                    # [영상 길이 기반 duration 자동차단 패치]
                    # metadata.json의 시간보다 영상이 짧을 때 검은 화면이 나오는 현상을 막아줍니다.
                    if self._ref_video_fps > 0 and total_frames > 0:
                        real_dur = total_frames / self._ref_video_fps
                        current_dur = float(self._current_song.get("duration", real_dur))
                        # 원본 길이보다 더 길면 잘라냅니다. (안전하게 0.15초 마진)
                        self._current_song["duration"] = min(current_dur, real_dur - 0.15)

                    # 오디오 추출 및 프리로드 (Ogg Vorbis)
                    self._audio_path = video_path.rsplit('.', 1)[0] + ".ogg"
                    if not os.path.exists(self._audio_path):
                        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "libvorbis", "-q:a", "4", self._audio_path], capture_output=True)
                    
                    if os.path.exists(self._audio_path):
                        try:
                            pygame.mixer.music.load(self._audio_path)
                        except Exception:
                            pass
                else:
                    print(f"[WARN] 레퍼런스 영상 찾을 수 없음: {video_path}")

    def shutdown(self):
        """Clean up all resources."""
        self.running = False
        if getattr(self, '_async_camera', None) is not None:
            self._async_camera.stop()
        if self._ref_video_cap is not None:
            self._ref_video_cap.release()
            self._ref_video_cap = None
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass
