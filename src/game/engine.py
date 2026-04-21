"""
Game Engine - Main game loop, state management, and component orchestration.
"""

import time
import sys
import os
import json
import signal
import subprocess
from datetime import datetime

import pygame
import cv2
import numpy as np
import math

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pose.landmark_utils import SKELETON_CONNECTIONS, DANCE_JOINTS


class GameState:
    """Enumeration of game states."""
    MENU = "menu"
    SONG_SELECT = "song_select"
    READY = "ready"
    COUNTDOWN = "countdown"
    PLAYING = "playing"
    PAUSED = "paused"
    RESULT = "result"
    SETTINGS = "settings"
    LEADERBOARD = "leaderboard"
    WAITING = "waiting"    # 멀티플레이 상대방 탐색 중


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
        self._missing_pose_penalty_enabled = True
        self._missing_pose_similarity = 0.0
        self._missing_pose_grace_seconds = 1.0
        self._timing_offset_penalty_enabled = True
        self._timing_offset_max_penalty = 0.08
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
        self._leaderboard: list = []        # [{song_id, title, score, grade, mode, date}, ...]
        self._leaderboard_filter: str = ""  # "" = 전체, else song_id
        self._leaderboard_tab: str = "all"  # "all" | "practice" | "challenge" | "freestyle"
        self._lb_selected_row: int = 0      # 리더보드 컨텐츠 영역에서 선택된 행 인덱스
        self._lb_confirm_delete = None      # 삭제 확인 대기: "all" | int(행 인덱스) | None
        self._lb_scroll: int = 0            # 리더보드 스크롤 오프셋 (행 단위)
        self._lb_max_scroll: int = 0        # 리더보드 최대 스크롤 (render에서 갱신)
        # 이름 입력 오버레이 (결과 화면 진입 시 표시)
        self._name_input_active: bool = False  # 오버레이 표시 중 여부
        self._name_input_text: str = ""        # 현재 입력 텍스트
        self._player_name: str = ""            # 마지막 저장 이름 (다음 게임에 미리 채움)
        self._stdin_text_mode: bool = False    # True이면 stdin 브리지가 문자 그대로 전달
        self._name_btn_pending: str = ""       # 더블탭 대기 중인 버튼 이름 ("btn_name_save" | "btn_name_skip" | "")
        self._name_focus_idx: int = 0             # 이름 입력 오버레이 포커스 (0=텍스트, 1=SAVE, 2=SKIP)
        # 챌린지 모드 연속 MISS 카운터
        self._consecutive_miss: int = 0
        self._challenge_game_over: bool = False
        # 설정 화면 항목 포커스
        self._settings_focus_idx: int = 0
        # 볼륨 (0.0 ~ 1.0)
        self._bgm_volume: float = 0.8
        # 터치/마우스 클릭용 버튼 rect 저장소
        self._btn_rects = {}
        # 곡 선택 관련
        self._songs: list = []
        self._selected_song_idx: int = 0
        self._current_mode: str = "practice"   # practice | challenge | freestyle
        self._current_song: dict = {}
        # 가이드 캐릭터용 참조 랜드마크
        self._ref_landmarks = None          # shape (N, 33, 4) — 채점용 (x-flip 적용)
        self._ref_landmarks_display = None  # shape (N, 33, 4) — 렌더링용 (원본)
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
        # 점수 판정 주기 제어 — 15프레임(0.5초)마다 판정
        self._scoring_interval: int = 15     # 판정 간격 (프레임 수)
        self._scoring_frame_counter: int = 0 # 마지막 판정 후 경과 프레임
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
        self._song_focus_idx: int = 0        # 곡 선택 화면 포커스 인덱스 (depth1)
        self._song_depth: int = 1            # 곡 선택 뎁스: 1=목록+BACK, 2=START 버튼
        self._neon_tick: float = 0.0         # 네온 깜빡임 타이머
        # 터치 더블탭 지원: 첫 탭=포커스, 두 번째 탭=실행
        self._touch_focused_btn: str = ""    # 현재 터치-포커스된 버튼 이름
        self._generic_focus_idx: int = 0     # PAUSED/RESULT/SETTINGS 화면 포커스 인덱스
        self._last_event_type: str = "mouse" # 마지막 입력 이벤트 타입 ('mouse' | 'touch')
        # 파티클 이펙트 (피드백 폭발 효과)
        self._particles: list = []           # [{'x','y','vx','vy','life','max_life','color','size'}]
        # 피드백 등장 후 경과 시간 (판정 애니메이션 progress 계산용)
        self._feedback_age: float = 0.0      # 현재 피드백이 표시된 후 흐른 시간(초)
        # 점수 추적 로그(JSONL)
        self._score_trace_enabled: bool = True
        self._score_trace_dir: str = "data/logs/score_traces"
        self._score_trace_file = None
        self._score_trace_path: str = ""
        self._last_similarity_debug = None
        self._last_terminal_similarity = None
        # ── 렌더링 성능 캐시 ──
        self._ref_video_frame_seq = -1        # 레퍼런스 영상 변경 감지용 시퀀스
        self._rendered_cam_seq = -1           # 처리된 웹캠 프레임 시퀀스

        # ── 멀티플레이 ───────────────────────────────────────────────
        self._is_multi_mode: bool = False       # 현재 멀티 게임 중 여부
        self._multi_role: str = ""              # "host" | "client"
        self._multi_opponent_ip: str = ""
        self._multi_discovery = None            # Discovery 인스턴스
        self._multi_socket = None               # GameSocket 인스턴스
        self._multi_status_msg: str = ""        # WAITING 화면 상태 메시지
        self._multi_found: bool = False         # 탐색 성공 여부 (WAITING → SONG_SELECT)
        self._multi_timed_out: bool = False     # 탐색 실패 여부
        self._multi_game_start_received: bool = False  # CLIENT: HOST 시작 신호 수신 여부
        self._multi_my_pose_ready: bool = False        # 내 포즈 감지 3초 완료
        self._multi_opponent_pose_ready: bool = False  # 상대방 포즈 감지 3초 완료
        self._multi_mode_selected: bool = False        # 모드 선택 완료 여부
        self._multi_connected: bool = False            # Discovery 성공, 연결된 상태

    def initialize(self):
        """
        Initialize all game subsystems.
        Called once at startup.
        """
        # SDL이 SIGINT/SIGTERM을 가로채지 않도록 pygame.init() 전에 설정
        os.environ.setdefault("SDL_NO_SIGNAL_HANDLERS", "1")

        # mixer를 pygame.init()보다 먼저 초기화해야 buffer 설정이 적용됨
        # mono + 버퍼 64 → ~1.5ms 레이턴시 (44100Hz 기준)
        pygame.mixer.pre_init(44100, -16, 1, 64)
        pygame.init()

        # SFX 전용 채널 예약 (0=click, 1=select)
        pygame.mixer.set_num_channels(16)
        self._sfx_channel = pygame.mixer.Channel(0)
        self._sfx_channel_select = pygame.mixer.Channel(1)

        # SFX 로드 — MP3→PCM 사전 변환으로 재생 시 디코딩 오버헤드 제거
        self._sfx: dict = {}
        _sfx_volume = float(self.config.get("audio", {}).get("sfx_volume", 0.8))
        _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        for _sfx_name in ("click", "select"):
            _sfx_path = os.path.join(_project_root, "assets", "sounds", f"{_sfx_name}.mp3")
            if os.path.exists(_sfx_path):
                _raw = pygame.mixer.Sound(_sfx_path)
                _pcm_snd = pygame.mixer.Sound(buffer=_raw.get_raw())
                _pcm_snd.set_volume(_sfx_volume)
                self._sfx[_sfx_name] = _pcm_snd

        # 무음 재생으로 오디오 파이프라인 워밍업 (첫 재생 지연 방지)
        if self._sfx:
            _warmup_snd = next(iter(self._sfx.values()))
            self._sfx_channel.set_volume(0)
            self._sfx_channel.play(_warmup_snd)
            pygame.time.wait(10)
            self._sfx_channel.stop()
            self._sfx_channel.set_volume(1.0)

        # 키보드 반복 입력: 200ms 후 첫 반복, 이후 80ms 간격
        pygame.key.set_repeat(200, 80)

        # Display setup
        ui_cfg = self.config.get("ui", {})
        w = ui_cfg.get("window_width", 1024)
        h = ui_cfg.get("window_height", 600)
        flags = pygame.FULLSCREEN if ui_cfg.get("fullscreen", False) else 0
        self._display = pygame.display.set_mode((w, h), flags)
        pygame.display.set_caption("Let's Dance!")
        pygame.mouse.set_visible(False)
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
               if k in ("score_scale", "combo_multiplier", "grade_thresholds", "baseline")}
        )
        self._feedback_gen = FeedbackGenerator()

        scoring_cfg = self.config.get("scoring", {})
        missing_cfg = scoring_cfg.get("missing_pose_penalty", {})
        if isinstance(missing_cfg, bool):
            missing_cfg = {"enabled": missing_cfg}
        self._missing_pose_penalty_enabled = bool(missing_cfg.get("enabled", True))
        self._missing_pose_similarity = float(missing_cfg.get("similarity", 0.0))
        self._missing_pose_grace_seconds = float(missing_cfg.get("grace_seconds", 1.0))

        timing_cfg = scoring_cfg.get("timing_offset_penalty", {})
        if isinstance(timing_cfg, bool):
            timing_cfg = {"enabled": timing_cfg}
        self._timing_offset_penalty_enabled = bool(timing_cfg.get("enabled", True))
        self._timing_offset_max_penalty = float(timing_cfg.get("max_penalty", 0.08))

        smoothing_cfg = self.config.get("runtime_smoothing", {})
        self._pose_hold_frames = int(smoothing_cfg.get("pose_hold_frames", 6))
        self._score_hold_seconds = float(smoothing_cfg.get("score_hold_seconds", 0.3))
        self._model_warmup_direct_fallback = bool(
            smoothing_cfg.get("model_warmup_direct_fallback", True))
        self._fallback_similarity_method = smoothing_cfg.get(
            "fallback_similarity", self.config.get("similarity_method", "angle"))

        logging_cfg = self.config.get("logging", {})
        score_trace_cfg = logging_cfg.get("score_trace", {})
        if isinstance(score_trace_cfg, bool):
            score_trace_cfg = {"enabled": score_trace_cfg}
        self._score_trace_enabled = bool(score_trace_cfg.get("enabled", True))
        self._score_trace_dir = str(score_trace_cfg.get("dir", "data/logs/score_traces"))

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
                similarity_threshold=scratch_cfg.get("similarity_threshold", 0),
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
                similarity_threshold=embedding_cfg.get("similarity_threshold", 0),
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

        def _post(key, uni=''):
            try:
                pygame.event.post(pygame.event.Event(
                    pygame.KEYDOWN, key=key, mod=0, unicode=uni, scancode=0))
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
            b'm':     'K_s',     b'M': 'K_s',
            b'\x03':  'K_q',     # Ctrl+C → 게임 종료
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
            'm':      'K_s',     'screenshot': 'K_s',
        }

        def _read_keys():
            fd = sys.stdin.fileno()

            # ── raw 모드 시도 ──────────────────────────────────────
            try:
                old_attr = termios.tcgetattr(fd)
            except Exception:
                return   # stdin이 TTY가 아니면 종료

            def _restore():
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
                except Exception:
                    pass

            # SIGINT/SIGTERM 시 반드시 터미널 복구 (Ctrl+C로 강제 종료해도 안전)
            import signal as _signal
            _orig_sigint  = _signal.getsignal(_signal.SIGINT)
            _orig_sigterm = _signal.getsignal(_signal.SIGTERM)
            def _sig_handler(sig, frame):
                _restore()
                if sig == _signal.SIGINT  and callable(_orig_sigint):
                    _orig_sigint(sig, frame)
                if sig == _signal.SIGTERM and callable(_orig_sigterm):
                    _orig_sigterm(sig, frame)
                raise SystemExit(0)
            try:
                _signal.signal(_signal.SIGINT,  _sig_handler)
                _signal.signal(_signal.SIGTERM, _sig_handler)
            except Exception:
                pass  # 메인 스레드가 아닌 경우 signal 설정 불가

            try:
                new_attr = termios.tcgetattr(fd)
                # IFLAG: 입력 처리 끄기
                new_attr[0] = 0
                # LFLAG: echo, canonical 끄기. ISIG는 켜둠 → Ctrl+C가 SIGINT로 전달됨
                new_attr[3] = new_attr[3] & ~(
                    termios.ECHO | termios.ICANON | termios.IEXTEN)
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

                        # ── 텍스트 입력 모드 (이름 입력 등, 영어만 지원) ──
                        if getattr(self, '_stdin_text_mode', False):
                            if ch in (b'\x7f', b'\x08'):  # Backspace
                                sys.stdout.write('\b \b')
                                sys.stdout.flush()
                                _post(pygame.K_BACKSPACE)
                            elif ch in (b'\r', b'\n'):     # Enter
                                sys.stdout.write('\n')
                                sys.stdout.flush()
                                _post(pygame.K_RETURN)
                            elif ch == b'\x1b':            # ESC 또는 방향키
                                r2, _, _ = select.select([sys.stdin], [], [], 0.08)
                                if r2:
                                    ch2 = _os.read(fd, 1)
                                    if ch2 == b'[':
                                        r3, _, _ = select.select([sys.stdin], [], [], 0.05)
                                        if r3:
                                            ch3 = _os.read(fd, 1)
                                            arrow_name = ARROW.get(ch3)
                                            if arrow_name:
                                                _post(getattr(pygame, arrow_name))
                                            else:
                                                _post(pygame.K_ESCAPE)
                                        else:
                                            _post(pygame.K_ESCAPE)
                                    else:
                                        _post(pygame.K_ESCAPE)
                                else:
                                    _post(pygame.K_ESCAPE)
                            elif ch == b' ':               # Space
                                _post(pygame.K_SPACE)
                            elif 0x20 <= ch[0] <= 0x7E:   # ASCII 출력 가능 문자
                                sys.stdout.write(ch.decode('ascii'))
                                sys.stdout.flush()
                                _post(0, ch.decode('ascii'))
                            continue  # 일반 키맵 처리 건너뜀

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
                        # ── 텍스트 입력 모드 ──────────────────────────────
                        if getattr(self, '_stdin_text_mode', False):
                            text = line.rstrip('\r\n')
                            for c in text:
                                _post(0, c)
                            _post(pygame.K_RETURN)
                            continue
                        name = LINE_MAP.get(line.strip().lower())
                        if name:
                            _post(getattr(pygame, name))
            finally:
                if use_raw:
                    _restore()

        t = threading.Thread(target=_read_keys, daemon=True, name="stdin-key-bridge")
        t.start()

    def _load_songs(self) -> list:
        """data/reference_dances/ 폴더를 스캔해 곡 목록을 반환합니다."""

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
                # 앨범 커버 로드 (cover.png)
                cover_path = os.path.join(base, song_dir, "cover.png")
                if os.path.exists(cover_path):
                    try:
                        meta["_cover_surf"] = pygame.image.load(cover_path).convert_alpha()
                    except Exception:
                        meta["_cover_surf"] = None
                else:
                    meta["_cover_surf"] = None
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
        # SIGTERM 핸들러만 유지 (SIGINT는 Python이 KeyboardInterrupt로 변환하므로 제거)
        def _sigterm_handler(sig, frame):
            print("\n[INFO] SIGTERM — 게임을 종료합니다.")
            self.running = False

        signal.signal(signal.SIGTERM, _sigterm_handler)
        # SIGINT는 Python 기본 동작(KeyboardInterrupt 발생)으로 복원
        signal.signal(signal.SIGINT, signal.default_int_handler)

        self.running = True
        _frame_duration = 1.0 / self.TARGET_FPS

        try:
            while self.running:
                _t0 = time.perf_counter()
                self._handle_input()
                self._update()
                self._render()
                # pygame.clock.tick() 은 내부적으로 SDL_Delay(C레벨 블로킹)를 사용하므로
                # Python 시그널(KeyboardInterrupt)이 전달되지 않음.
                # time.sleep() 은 GIL을 해제하여 SIGINT → KeyboardInterrupt 즉시 처리.
                _elapsed = time.perf_counter() - _t0
                _sleep = _frame_duration - _elapsed
                if _sleep > 0:
                    time.sleep(_sleep)
        except KeyboardInterrupt:
            print("\n[INFO] Ctrl+C — 게임을 종료합니다.")
        finally:
            self.shutdown()

    def _handle_input(self):
        """Process user input events (keyboard, mouse, touch)."""

        # ── 화면별 포커스 버튼 목록 정의 ──────────────────────────────
        FOCUS_LISTS = {
            GameState.MENU:       ["btn_practice", "btn_challenge", "btn_freestyle",
                                   "btn_multi_play", "btn_leaderboard", "btn_settings", "btn_quit"],
            GameState.WAITING:    ["btn_waiting_cancel"],  # 동적으로 덮어씀 (_get_focus_list 참조)
            GameState.PAUSED:     ["btn_pause", "btn_gameplay_menu"],
            GameState.RESULT:     ["btn_retry", "btn_result_songs", "btn_result_menu"],
            GameState.SETTINGS:   ["btn_settings_vol_down", "btn_settings_vol_up", "btn_back"],
            GameState.READY:      ["btn_ready_skip", "btn_ready_cancel"],
            GameState.COUNTDOWN:  ["btn_countdown_cancel"],
            GameState.PLAYING:    ["btn_pause", "btn_gameplay_menu"],
            # 리더보드: index0=콘텐츠(탭영역), index1=DELETE ALL, index2=BACK
            GameState.LEADERBOARD: ["btn_lb_content", "btn_lb_delete_all", "btn_lb_back"],
        }

        # SONG_SELECT: depth1=[곡 목록+BACK], depth2=[START 버튼]
        # ↑/↓ 는 depth1만 순환, depth2는 Enter/Space로만 진입

        def _get_focus_list():
            if self.state == GameState.SONG_SELECT:
                songs = self._songs_for_mode(self._current_mode)
                return [f"btn_song_{i}" for i in range(len(songs))] + ["btn_song_back"]
            if self.state == GameState.WAITING and self._multi_connected and self._multi_role == "host" and not self._multi_mode_selected:
                return ["btn_multi_mode_practice", "btn_multi_mode_challenge",
                        "btn_multi_mode_freestyle", "btn_waiting_cancel"]
            return FOCUS_LISTS.get(self.state, [])

        def _focus_count():
            return len(_get_focus_list())

        def _get_focus_idx():
            if self.state == GameState.MENU:
                return self._menu_focus_idx
            if self.state == GameState.SONG_SELECT:
                return self._song_focus_idx
            return getattr(self, '_generic_focus_idx', 0)

        def _set_focus_idx(idx):
            fl = _get_focus_list()
            if not fl:
                return
            idx = idx % len(fl)
            # 네비게이션 효과음
            if "select" in self._sfx:
                self._sfx_channel_select.play(self._sfx["select"])
            if self.state == GameState.MENU:
                self._menu_focus_idx = idx
            elif self.state == GameState.SONG_SELECT:
                self._song_focus_idx = idx
                # 곡 카드 범위일 때만 selected_song_idx 갱신
                songs = self._songs_for_mode(self._current_mode)
                if idx < len(songs):
                    self._selected_song_idx = idx
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
            if self.state == GameState.WAITING:
                if self._multi_discovery:
                    self._multi_discovery.stop()
                self._is_multi_mode = False
                self.transition_to(GameState.MENU)
            elif self.state in (GameState.SONG_SELECT, GameState.SETTINGS):
                self.transition_to(GameState.MENU)
            elif self.state == GameState.LEADERBOARD:
                if self._lb_confirm_delete is not None:
                    # 삭제 확인 대기 중 → 취소
                    self._lb_confirm_delete = None
                else:
                    self.transition_to(GameState.MENU)
            elif self.state in (GameState.READY, GameState.COUNTDOWN):
                self.transition_to(GameState.SONG_SELECT)
            elif self.state == GameState.PAUSED:
                self.transition_to(GameState.PLAYING)
            elif self.state in (GameState.PLAYING,):
                self.transition_to(GameState.PAUSED)
            elif self.state == GameState.RESULT:
                if self._name_input_active:
                    # 이름 입력 오버레이 활성 중: ESC = 이전 이름으로 저장
                    self._name_input_active = False
                    self._stdin_text_mode = False
                    self._leaderboard_save_result(player=self._player_name)
                else:
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
                self._last_event_type = 'keyboard'

                # ── 이름 입력 오버레이 활성 중: 모든 키를 여기서 처리 ──
                if self._name_input_active:
                    if event.key in (pygame.K_LEFT, pygame.K_RIGHT,
                                     pygame.K_UP, pygame.K_DOWN):
                        # 방향키 → 포커스 순환 (0=텍스트, 1=SAVE, 2=SKIP)
                        if event.key in (pygame.K_DOWN, pygame.K_RIGHT):
                            self._name_focus_idx = (self._name_focus_idx + 1) % 3
                        else:
                            self._name_focus_idx = (self._name_focus_idx - 1) % 3
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER,
                                       pygame.K_SPACE):
                        if self._name_focus_idx <= 1:  # 텍스트(0) 또는 SAVE(1)
                            name = self._name_input_text.strip()
                            self._player_name = name
                            self._name_input_active = False
                            self._stdin_text_mode = False
                            self._leaderboard_save_result(player=name)
                            saved = name if name else "(이름 없음)"
                            print(f"[NAME] 저장됨: {saved}", flush=True)
                        else:  # SKIP(2)
                            self._name_input_active = False
                            self._stdin_text_mode = False
                            self._leaderboard_save_result(player=self._player_name)
                            saved = self._player_name if self._player_name else "(이름 없음)"
                            print(f"[NAME] 이전 이름으로 저장됨: {saved}", flush=True)
                    elif event.key == pygame.K_ESCAPE:
                        # ESC → 이전 이름(_player_name)으로 저장
                        self._name_input_active = False
                        self._stdin_text_mode = False
                        self._leaderboard_save_result(player=self._player_name)
                        saved = self._player_name if self._player_name else "(이름 없음)"
                        print(f"[NAME] 이전 이름으로 저장됨: {saved}", flush=True)
                    elif self._name_focus_idx == 0:
                        # 텍스트 포커스일 때만 문자 입력 허용
                        if event.key == pygame.K_BACKSPACE:
                            if self._name_input_text:
                                self._name_input_text = self._name_input_text[:-1]
                        else:
                            ch = event.unicode
                            if ch and ch.isprintable() and len(self._name_input_text) < 16:
                                self._name_input_text += ch
                    continue  # 이름 입력 중엔 다른 키 처리 건너뜀

                # ESC → 뒤로가기
                if event.key == pygame.K_ESCAPE:
                    if self.state == GameState.SONG_SELECT and self._song_depth == 2:
                        self._song_depth = 1   # depth2 → depth1로 돌아가기
                    else:
                        _go_back()

                # B → 뒤로가기
                elif event.key == pygame.K_b:
                    if self.state == GameState.SONG_SELECT and self._song_depth == 2:
                        self._song_depth = 1
                    else:
                        _go_back()

                # Q → 종료
                elif event.key == pygame.K_q:
                    self.running = False

                # Ctrl+C → 즉시 종료 (pygame 창이 키보드를 가져가므로 여기서 처리)
                elif event.key == pygame.K_c and (event.mod & pygame.KMOD_CTRL):
                    print("\n[INFO] Ctrl+C — 게임을 종료합니다.")
                    self.running = False
                    return

                # S → 스크린샷
                elif event.key == pygame.K_s:
                    self._take_screenshot()

                # ↓ : 다음 항목
                elif event.key == pygame.K_DOWN:
                    if self.state == GameState.LEADERBOARD:
                        if self._generic_focus_idx == 0:
                            if self._lb_scroll < self._lb_max_scroll:
                                self._lb_scroll += 1   # 목록 스크롤 아래로
                            else:
                                self._generic_focus_idx = 1  # 최하단 → BACK 포커스
                        # idx==1(BACK)에서 ↓는 아무것도 안 함
                    elif self.state == GameState.SONG_SELECT and self._song_depth == 2:
                        pass  # depth2에선 ↓ 무시
                    elif self.state == GameState.LEADERBOARD and self._generic_focus_idx == 0:
                        # 콘텐츠 포커스: 행 선택 이동, 마지막 행 넘으면 DELETE ALL로
                        filtered = self._lb_get_filtered_entries()
                        max_idx = max(0, len(filtered) - 1)
                        if self._lb_selected_row >= max_idx:
                            # 마지막 행 → DELETE ALL 버튼으로 포커스 이동
                            self._generic_focus_idx = 1
                        else:
                            self._lb_selected_row += 1
                        self._lb_confirm_delete = None
                    else:
                        cnt = _focus_count()
                        if cnt:
                            _set_focus_idx(_get_focus_idx() + 1)
                        if self.state == GameState.LEADERBOARD:
                            self._lb_confirm_delete = None

                # ↑ : 이전 항목
                elif event.key == pygame.K_UP:
                    if self.state == GameState.LEADERBOARD:
                        if self._generic_focus_idx == 1:
                            self._generic_focus_idx = 0  # BACK → 콘텐츠로 복귀
                        else:
                            self._lb_scroll = max(0, self._lb_scroll - 1)  # 목록 스크롤 위로
                    elif self.state == GameState.SONG_SELECT and self._song_depth == 2:
                        self._song_depth = 1   # ↑ → depth1으로 돌아가기
                    elif self.state == GameState.LEADERBOARD:
                        if self._generic_focus_idx > 0:
                            # 버튼 → 콘텐츠 복귀: 마지막 행 선택
                            self._generic_focus_idx = 0
                            filtered = self._lb_get_filtered_entries()
                            self._lb_selected_row = max(0, len(filtered) - 1)
                        else:
                            self._lb_selected_row = max(0, self._lb_selected_row - 1)
                        self._lb_confirm_delete = None
                    else:
                        cnt = _focus_count()
                        if cnt:
                            _set_focus_idx(_get_focus_idx() - 1)

                # → : LEADERBOARD=탭 이동(콘텐츠 포커스 시), SETTINGS=볼륨업, 그 외=↓와 동일
                elif event.key == pygame.K_RIGHT:
                    if self.state == GameState.LEADERBOARD:
                        if self._generic_focus_idx == 0:   # 콘텐츠 영역 포커스 중
                            tabs = ["all", "practice", "challenge", "freestyle"]
                            cur = tabs.index(self._leaderboard_tab) if self._leaderboard_tab in tabs else 0
                            self._leaderboard_tab = tabs[(cur + 1) % len(tabs)]
                        else:  # DELETE ALL(1) ↔ BACK(2)
                            self._generic_focus_idx = 1 if self._generic_focus_idx == 2 else 2
                            self._lb_confirm_delete = None
                            self._lb_scroll = 0
                    elif self.state == GameState.SETTINGS:
                        self._on_button_press("btn_settings_vol_up")
                    else:
                        cnt = _focus_count()
                        if cnt:
                            _set_focus_idx(_get_focus_idx() + 1)

                # ← : LEADERBOARD=탭 이동(콘텐츠 포커스 시), SETTINGS=볼륨다운, 그 외=↑와 동일
                elif event.key == pygame.K_LEFT:
                    if self.state == GameState.SONG_SELECT and self._song_depth == 2:
                        self._song_depth = 1   # ← → depth1으로 돌아가기
                    elif self.state == GameState.LEADERBOARD:
                        if self._generic_focus_idx == 0:
                            tabs = ["all", "practice", "challenge", "freestyle"]
                            cur = tabs.index(self._leaderboard_tab) if self._leaderboard_tab in tabs else 0
                            self._leaderboard_tab = tabs[(cur - 1) % len(tabs)]
                        else:  # DELETE ALL(1) ↔ BACK(2)
                            self._generic_focus_idx = 1 if self._generic_focus_idx == 2 else 2
                            self._lb_confirm_delete = None
                            self._lb_scroll = 0
                    elif self.state == GameState.SETTINGS:
                        self._on_button_press("btn_settings_vol_down")
                    else:
                        cnt = _focus_count()
                        if cnt:
                            _set_focus_idx(_get_focus_idx() - 1)

                # Enter / Space
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    if self.state == GameState.SONG_SELECT:
                        if self._song_depth == 2:
                            # depth2: START 버튼 실행 → READY 화면
                            self._on_button_press("btn_song_start")
                        else:
                            # depth1: BACK이면 뒤로, 곡이면 depth2로 이동
                            fl = _get_focus_list()
                            idx = _get_focus_idx() % max(len(fl), 1)
                            focused_btn = fl[idx]
                            if focused_btn == "btn_song_back":
                                self._on_button_press("btn_song_back")
                            else:
                                # 곡 선택 → depth2 (START 버튼으로 포커스 이동)
                                self._on_button_press(focused_btn)   # 곡 하이라이트
                                self._song_depth = 2
                    elif self.state == GameState.LEADERBOARD:
                        if self._generic_focus_idx == 0:  # 콘텐츠 포커스
                            self._lb_try_delete_selected()
                        elif self._generic_focus_idx == 1:  # DELETE ALL 포커스
                            self._lb_try_delete_all()
                        elif self._generic_focus_idx == 2:  # BACK 포커스
                            self._on_button_press("btn_lb_back")
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

            # ── 마우스 휠 스크롤 ────────────────────────────────
            elif event.type == pygame.MOUSEWHEEL:
                if self.state == GameState.LEADERBOARD:
                    self._lb_scroll = max(0, self._lb_scroll - event.y)

            # ── 핑거(터치스크린 전용) 이벤트 ──────────────────────
            elif event.type == pygame.FINGERDOWN:
                w_d, h_d = self._display.get_size()
                touch_pos = (int(event.x * w_d), int(event.y * h_d))
                self._last_event_type = 'touch'
                self._finger_handled = True   # 뒤따라오는 MOUSEBUTTONDOWN 무시
                self._handle_click(touch_pos)

            # ── 핑거 스와이프 (터치 스크롤) ────────────────────────
            elif event.type == pygame.FINGERMOTION:
                if self.state == GameState.LEADERBOARD:
                    h_d = self._display.get_height()
                    # dy > 0 → 아래→위 스와이프 → 목록 아래로 (scroll 증가)
                    # dy < 0 → 위→아래 스와이프 → 목록 위로 (scroll 감소)
                    dy_px = event.dy * h_d
                    if abs(dy_px) > 4:
                        self._lb_scroll = max(0, self._lb_scroll + (1 if dy_px < -8 else -1 if dy_px > 8 else 0))

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
        double_tap_states = (GameState.MENU, GameState.SONG_SELECT, GameState.WAITING)

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
        # 클릭 효과음 — 전용 채널로 즉시 재생 (이전 재생 즉시 중단)
        if "click" in self._sfx:
            self._sfx_channel.play(self._sfx["click"])

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
        elif btn_name == "btn_leaderboard":
            self.transition_to(GameState.LEADERBOARD)
        elif btn_name == "btn_multi_play":
            self._is_multi_mode = True
            self._multi_mode_selected = False
            self._multi_connected = False
            self._current_mode = ""
            self._multi_found = False
            self._multi_timed_out = False
            self._multi_status_msg = "상대방 탐색 중..."
            from network.discovery import Discovery
            self._multi_discovery = Discovery()
            self._multi_discovery.find_opponent(
                on_found=self._on_multi_found,
                on_timeout=self._on_multi_timeout,
                on_status=self._on_multi_status,
            )
            self.transition_to(GameState.WAITING)
        elif btn_name == "btn_quit":
            self.running = False

        # ── 곡 선택 화면 버튼 ──
        elif btn_name == "btn_song_back":
            self.transition_to(GameState.MENU)
        elif btn_name == "btn_song_start":
            songs = self._songs_for_mode(self._current_mode)
            if songs:
                self._current_song = songs[self._selected_song_idx]
            # 멀티플레이 HOST면 CLIENT에게 곡 정보 전송
            if self._is_multi_mode and self._multi_role == "host" and self._multi_socket:
                self._multi_socket.send_song(
                    self._current_song.get("id", ""),
                    mode=self._current_mode,
                )
            self.transition_to(GameState.READY)
        elif btn_name.startswith("btn_song_"):
            try:
                idx = int(btn_name.split("_")[-1])
                songs = self._songs_for_mode(self._current_mode)
                if 0 <= idx < len(songs):
                    self._selected_song_idx = idx
                    self._play_song_preview(songs[idx])
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

        # ── 결과 화면 버튼 (이름 입력 오버레이) ──
        elif btn_name in ("btn_name_save", "btn_name_skip"):
            if self._name_btn_pending == btn_name:
                # 두 번째 탭 → 실행
                self._name_btn_pending = ""
                if btn_name == "btn_name_save":
                    name = self._name_input_text.strip()
                    self._player_name = name
                    self._name_input_active = False
                    self._stdin_text_mode = False
                    self._leaderboard_save_result(player=name)
                    saved = name if name else "(이름 없음)"
                    print(f"[NAME] 저장됨: {saved}", flush=True)
                else:  # btn_name_skip
                    self._name_input_active = False
                    self._stdin_text_mode = False
                    self._leaderboard_save_result(player=self._player_name)
                    saved = self._player_name if self._player_name else "(이름 없음)"
                    print(f"[NAME] 이전 이름으로 저장됨: {saved}", flush=True)
            else:
                # 첫 번째 탭 → 하이라이트만
                self._name_btn_pending = btn_name
        elif btn_name == "btn_retry":
            self._name_btn_pending = ""
            self.transition_to(GameState.READY)
        elif btn_name == "btn_result_songs":
            self._name_btn_pending = ""
            self.transition_to(GameState.SONG_SELECT)
        elif btn_name == "btn_result_menu":
            self._name_btn_pending = ""
            self.transition_to(GameState.MENU)

        # ── 리더보드 화면 버튼 ──
        elif btn_name == "btn_lb_back":
            self.transition_to(GameState.MENU)
        elif btn_name.startswith("btn_lb_tab_"):
            tab = btn_name[len("btn_lb_tab_"):]
            self._leaderboard_tab = tab
            self._lb_scroll = 0   # 탭 변경 시 스크롤 초기화

        # ── 설정/카운트다운/준비 화면 버튼 ──
        elif btn_name == "btn_back":
            self.transition_to(GameState.MENU)
        elif btn_name in ("btn_multi_mode_practice", "btn_multi_mode_challenge", "btn_multi_mode_freestyle"):
            # HOST가 연결 후 모드를 선택 → 곡 선택 화면으로 이동
            self._current_mode = btn_name.replace("btn_multi_mode_", "")
            self._multi_mode_selected = True
            self._selected_song_idx = 0
            self.transition_to(GameState.SONG_SELECT)
        elif btn_name == "btn_waiting_cancel":
            if self._multi_discovery:
                self._multi_discovery.stop()
            self._is_multi_mode = False
            self._multi_mode_selected = False
            self._multi_connected = False
            self.transition_to(GameState.MENU)
        elif btn_name == "btn_settings_vol_down":
            self._bgm_volume = max(0.0, round(self._bgm_volume - 0.1, 1))
            pygame.mixer.music.set_volume(self._bgm_volume)
        elif btn_name == "btn_settings_vol_up":
            self._bgm_volume = min(1.0, round(self._bgm_volume + 0.1, 1))
            pygame.mixer.music.set_volume(self._bgm_volume)
        elif btn_name == "btn_countdown_cancel":
            self.transition_to(GameState.MENU)
        elif btn_name == "btn_ready_skip":
            self.transition_to(GameState.COUNTDOWN)
        elif btn_name == "btn_ready_cancel":
            self.transition_to(GameState.MENU)

    def _update(self):
        """Update game state based on current state."""
        # 네온 깜빡임 타이머 (항상 업데이트)
        self._neon_tick += 1.0 / self.TARGET_FPS

        # 피드백 나이 타이머 (등장 후 경과시간, 애니메이션 progress용)
        if self._last_feedback is not None:
            self._feedback_age += 1.0 / self.TARGET_FPS

        if self.state == GameState.WAITING:
            self._update_waiting()

        elif self.state == GameState.READY:
            if self._is_multi_mode:
                # HOST: 양쪽 모두 포즈 준비 완료 → send_start() + COUNTDOWN
                if (self._multi_role == "host"
                        and self._multi_my_pose_ready
                        and self._multi_opponent_pose_ready):
                    self._multi_my_pose_ready = False
                    self._multi_opponent_pose_ready = False
                    if self._multi_socket:
                        self._multi_socket.send_start()
                        print("[MULTI] 양쪽 포즈 준비 완료 → 시작 신호 전송", flush=True)
                    self.transition_to(GameState.COUNTDOWN)
                # CLIENT: 자신 준비 완료 + HOST 시작 신호 수신 → COUNTDOWN
                elif (self._multi_role == "client"
                        and self._multi_my_pose_ready
                        and self._multi_game_start_received):
                    self._multi_my_pose_ready = False
                    self._multi_game_start_received = False
                    self.transition_to(GameState.COUNTDOWN)
                # 아직 조건 미충족 → 포즈 감지 계속
                elif not self._multi_my_pose_ready:
                    self._update_ready()
                else:
                    # 내 포즈 준비 완료, 상대방 대기 중 → 카메라 프레임만 계속 업데이트
                    if getattr(self, '_async_camera', None) is not None:
                        ret, frame, lm, detected, _ = self._async_camera.read()
                        if ret:
                            self._ready_current_frame = frame
                            self._ready_landmarks = lm
                            self._ready_pose_detected = detected
            else:
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
                ret, frame, _, _, _ = self._async_camera.read()
                if ret:
                    self._current_frame = frame

        elif self.state == GameState.PLAYING:
            self._update_gameplay()

        # PAUSED 상태에서는 카메라/포즈 업데이트 중단

    def _update_ready(self):
        """READY 상태: 비동기 카메라로부터 포즈 감지 및 카운트다운."""
        if getattr(self, '_async_camera', None) is not None:
            ret, frame, lm, detected, seq = self._async_camera.read()
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
                # 3초 유지 완료
                self._ready_full_body_start = 0.0
                self._ready_countdown = 0.0
                if self._is_multi_mode:
                    # HOST/CLIENT 모두: 포즈 준비 완료 알림 전송, 상대방 신호 대기
                    if not self._multi_my_pose_ready:
                        self._multi_my_pose_ready = True
                        if self._multi_socket:
                            self._multi_socket.send_pose_ready()
                            print(f"[MULTI] 포즈 준비 완료 전송 ({self._multi_role})", flush=True)
                else:
                    self.transition_to(GameState.COUNTDOWN)
        else:
            # 전신 미감지 시 카운트다운 리셋
            self._ready_full_body_start = 0.0
            self._ready_countdown = 0.0

    def _mark_no_score_frame(self):
        """Keep the latest feedback visible briefly when no new score is produced."""
        if self._last_feedback is not None and self._feedback_timer <= 0 and self._score_hold_seconds > 0:
            self._feedback_timer = self._score_hold_seconds

    def _timing_penalty_value(self):
        if not self._timing_offset_penalty_enabled:
            return 0.0
        return max(0.0, float(self._timing_offset_max_penalty))

    @staticmethod
    def _sanitize_log_token(value):
        token = str(value or "unknown").strip()
        safe = []
        for ch in token:
            safe.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
        return "".join(safe).strip("_") or "unknown"

    def _resolve_project_root(self):
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    @staticmethod
    def _landmarks_to_list(landmarks):
        if landmarks is None:
            return None
        import numpy as np

        arr = np.asarray(landmarks, dtype=np.float32)
        return np.round(arr, 6).tolist()

    def _open_score_trace_log(self):
        if not self._score_trace_enabled:
            return
        self._close_score_trace_log()

        project_root = self._resolve_project_root()
        log_dir = self._score_trace_dir
        if not os.path.isabs(log_dir):
            log_dir = os.path.join(project_root, log_dir)
        os.makedirs(log_dir, exist_ok=True)

        song = self._current_song or {}
        song_id = self._sanitize_log_token(
            song.get("id") or song.get("title") or os.path.basename(song.get("path", "")) or "song"
        )
        mode = self._sanitize_log_token(self._current_mode or "unknown")
        started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._score_trace_path = os.path.join(
            log_dir, f"score_trace_{song_id}_{mode}_{started_at}.jsonl"
        )
        self._score_trace_file = open(self._score_trace_path, "a", encoding="utf-8")
        print(f"[INFO] Score trace logging -> {self._score_trace_path}")

    def _close_score_trace_log(self):
        if self._score_trace_file is None:
            return
        try:
            self._score_trace_file.flush()
            self._score_trace_file.close()
        except Exception:
            pass
        self._score_trace_file = None

    def _debug_info_for_score_source(self, score_source):
        if score_source in ("scratch", "scratch_cache"):
            return dict(getattr(self._scratch_comparator, "_last_debug_info", None) or {})
        if score_source in ("embedding", "embedding_cache"):
            return dict(getattr(self._embedding_comparator, "_last_debug_info", None) or {})
        if score_source in ("direct", "scratch_fallback_direct", "embedding_fallback_direct"):
            return dict(self._last_similarity_debug or {})
        if score_source == "missing_pose_penalty":
            return {"current_reference_index": int(self._ref_current_idx)}
        return {}

    def _write_score_trace(self, similarity, evaluation, user_landmarks, score_source):
        if not self._score_trace_enabled or self._score_trace_file is None:
            return

        debug_info = self._debug_info_for_score_source(score_source)
        similarity_metric = debug_info.get("method")
        if not similarity_metric:
            if score_source in ("scratch", "scratch_cache", "embedding", "embedding_cache"):
                similarity_metric = "cosine"
            elif score_source in ("scratch_fallback_direct", "embedding_fallback_direct"):
                similarity_metric = self._fallback_similarity_method
            elif score_source == "missing_pose_penalty":
                similarity_metric = "penalty"
            else:
                similarity_metric = self._similarity_method
        current_reference_index = debug_info.get("current_reference_index")
        if current_reference_index is None and self._ref_frame_landmarks is not None:
            current_reference_index = int(self._ref_current_idx)

        matched_reference_index = (
            debug_info.get("best_candidate_index")
            if debug_info.get("best_candidate_index") is not None
            else debug_info.get("best_reference_index")
        )

        matched_reference_landmarks = None
        if (
            matched_reference_index is not None
            and self._ref_landmarks is not None
            and 0 <= int(matched_reference_index) < len(self._ref_landmarks)
        ):
            matched_reference_landmarks = self._ref_landmarks[int(matched_reference_index)]

        record = {
            "event": "score_tick",
            "logged_at": datetime.now().isoformat(timespec="milliseconds"),
            "song_id": (self._current_song or {}).get("id"),
            "song_title": (self._current_song or {}).get("title"),
            "mode": self._current_mode,
            "state": self.state,
            "score_method": self._score_method,
            "score_source": score_source,
            "similarity_metric": similarity_metric,
            "similarity": float(similarity),
            "evaluation": evaluation,
            "elapsed_time_sec": round(
                float(getattr(self._current_session, "elapsed_time", 0.0) or 0.0), 3
            ),
            "reference_index_current": (
                int(current_reference_index) if current_reference_index is not None else None
            ),
            "reference_index_matched": (
                int(matched_reference_index) if matched_reference_index is not None else None
            ),
            "user_landmarks": self._landmarks_to_list(user_landmarks),
            "reference_landmarks_current": self._landmarks_to_list(self._ref_frame_landmarks),
            "reference_landmarks_matched": self._landmarks_to_list(matched_reference_landmarks),
            "similarity_debug": debug_info,
        }

        try:
            self._score_trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._score_trace_file.flush()
        except Exception as exc:
            print(f"[WARN] Score trace write failed: {exc}")

    def _print_terminal_similarity(self, similarity):
        try:
            self._last_terminal_similarity = float(similarity)
            print(f"\rSIM {self._last_terminal_similarity:.3f}", end="", flush=True)
        except Exception:
            pass

    def _should_apply_missing_pose_penalty(self):
        """Return True when a scoring tick should count as a Miss for no pose."""
        if not self._missing_pose_penalty_enabled:
            return False
        if self._current_session is None:
            return False
        if self._ref_frame_landmarks is None:
            return False
        if self._current_session.elapsed_time < self._missing_pose_grace_seconds:
            return False
        return True

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

        scored_candidates = []
        timing_penalty = self._timing_penalty_value()
        for ri in range(start_idx, end_idx):
            ref_lm = self._ref_landmarks[ri]
            sim = self._pose_similarity_with_method(comparator, user_landmarks, ref_lm, method)
            if timing_penalty > 0.0 and tolerance_frames > 0:
                offset = abs(self._ref_current_idx - ri)
                sim -= timing_penalty * min(offset / tolerance_frames, 1.0)
            scored_candidates.append((int(ri), float(max(0.0, min(1.0, sim)))))
        if not scored_candidates:
            self._last_similarity_debug = None
            return None

        scored_candidates.sort(key=lambda item: item[1], reverse=True)
        top_k = scored_candidates[:min(3, len(scored_candidates))]
        sim = sum(score for _, score in top_k) / len(top_k)
        self._last_similarity_debug = {
            "method": method,
            "current_reference_index": int(self._ref_current_idx),
            "best_reference_index": int(top_k[0][0]),
            "best_reference_similarity": float(top_k[0][1]),
            "top_candidates": [
                {"reference_index": int(idx), "similarity": float(score)}
                for idx, score in top_k
            ],
            "returned_similarity": float(sim),
        }

        if debug:
            sim_now = self._pose_similarity_with_method(
                comparator, user_landmarks, self._ref_frame_landmarks, method)
            print(
                f"\r[DBG] now={sim_now:.3f} top3={sim:.3f} "
                f"max={top_k[0][1]:.3f} win={end_idx-start_idx}f",
                end="",
            )
        return sim

    def _update_gameplay(self):
        """Fetch async frame and compute score."""

        if getattr(self, '_async_camera', None) is None:
            return

        ret, frame, lm, detected, seq = self._async_camera.read()
        if not ret:
            return

        self._current_frame = frame
        self._current_landmarks = lm
        self._pose_detected = detected
        self._current_frame_seq = seq

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
            display_src = getattr(self, '_ref_landmarks_display', self._ref_landmarks)
            self._ref_frame_landmarks = display_src[fi]
            self._ref_current_idx = fi

        # 레퍼런스 영상 프레임 — 변경 시에만 Surface 재생성 (tobytes 호출 최소화)
        if getattr(self, '_async_video_player', None) is not None and self._current_session:
            vframe_rgb, vframe_seq = self._async_video_player.get_latest_frame_with_seq()
            if vframe_rgb is not None:
                if vframe_seq != self._ref_video_frame_seq:
                    self._ref_video_frame_seq = vframe_seq
                    th, tw = vframe_rgb.shape[:2]
                    self._ref_video_surf = pygame.image.frombuffer(
                        vframe_rgb.tobytes(), (tw, th), "RGB")
            else:
                self._ref_video_surf = None

        # Score based on pose similarity. scoring_landmarks may be a held pose
        # for a few frames when MediaPipe briefly drops detection.
        #
        # 판정 주기 분리:
        #   매 프레임: buffer_frame()으로 포즈 버퍼 축적
        #   N프레임마다: compute_from_buffer()로 모델 추론 + 점수 산출
        sim = None
        score_source = None
        if self._current_mode != "freestyle":
            if scoring_landmarks is not None:
                # ── 매 프레임: 버퍼 축적 ──
                if self._score_method == "scratch" and hasattr(self, '_scratch_comparator') and self._scratch_comparator:
                    self._scratch_comparator.buffer_frame(scoring_landmarks)
                elif self._score_method == "embedding" and hasattr(self, '_embedding_comparator') and self._embedding_comparator:
                    self._embedding_comparator.buffer_frame(scoring_landmarks)

            # ── 판정 주기 도달 시: 유사도 계산 또는 missing-pose Miss 집계 ──
            self._scoring_frame_counter += 1
            if self._scoring_frame_counter >= self._scoring_interval:
                self._scoring_frame_counter = 0
                self._last_similarity_debug = None

                if self._ref_frame_landmarks is not None:
                    if scoring_landmarks is None:
                        if self._should_apply_missing_pose_penalty():
                            sim = self._missing_pose_similarity
                            score_source = "missing_pose_penalty"
                    elif self._score_method == "direct":
                        sim = self._direct_window_similarity(
                            scoring_landmarks,
                            method=self._similarity_method,
                            debug=True,
                        )
                        score_source = "direct"
                    elif self._score_method == "scratch":
                        tolerance_frames = int(self._tolerance_delay * self.TARGET_FPS)
                        sim = self._scratch_comparator.compute_from_buffer(
                            self._ref_landmarks,
                            self._ref_current_idx,
                            tolerance_frames=tolerance_frames,
                            timing_penalty=self._timing_penalty_value(),
                        )
                        if sim is not None:
                            score_source = "scratch"
                        if sim is None and self._model_warmup_direct_fallback:
                            sim = self._direct_window_similarity(
                                scoring_landmarks,
                                method=self._fallback_similarity_method,
                            )
                            if sim is not None:
                                score_source = "scratch_fallback_direct"
                    else:
                        tolerance_frames = int(self._tolerance_delay * self.TARGET_FPS)
                        sim = self._embedding_comparator.compute_from_buffer(
                            self._ref_landmarks,
                            self._ref_current_idx,
                            tolerance_frames=tolerance_frames,
                            timing_penalty=self._timing_penalty_value(),
                        )
                        if sim is not None:
                            score_source = "embedding"
                        if sim is None and self._model_warmup_direct_fallback:
                            sim = self._direct_window_similarity(
                                scoring_landmarks,
                                method=self._fallback_similarity_method,
                            )
                            if sim is not None:
                                score_source = "embedding_fallback_direct"
                else:
                    if scoring_landmarks is None:
                        pass
                    elif self._score_method in ("scratch", "embedding"):
                        if not self._warned_scratch_no_ref:
                            print(f"[WARN] {self._score_method} scoring requires reference.npy; scoring paused.")
                            self._warned_scratch_no_ref = True
                    else:
                        visibility = scoring_landmarks[:, 3]
                        visible = visibility[visibility > 0]
                        sim = min(float(np.mean(visible)), 1.0) if len(visible) else 0.0

        if sim is not None:
            if self._score_method != "direct":
                self._print_terminal_similarity(sim)
            evaluation = self._scorer.evaluate(sim)
            self._write_score_trace(sim, evaluation, scoring_landmarks, score_source or "unknown")
            fb = self._feedback_gen.generate(evaluation)
            if fb:
                self._last_feedback = fb
                self._feedback_timer = 1.2  # 1.2초 동안 표시
                self._feedback_age   = 0.0  # 애니메이션 경과 시간 리셋
                text = fb.get("text", "")
                # ── 챌린지 모드 연속 MISS 카운터 (비활성화 - 테스트 원활화) ──────────────
                # if self._current_mode == "challenge":
                #     if text == "MISS":
                #         self._consecutive_miss += 1
                #         if self._consecutive_miss >= 100:
                #             self._challenge_game_over = True
                #     else:
                #         self._consecutive_miss = 0
                # 파티클 폭발 효과 — fb["text"]로 등급 판단
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

        # Check session time / challenge game over
        if self._current_session:
            if self._challenge_game_over:
                self._result_data = self._scorer.get_final_result()
                self.transition_to(GameState.RESULT)
            elif self._current_session.is_finished:
                self._result_data = self._scorer.get_final_result()
                self.transition_to(GameState.RESULT)

        # ── 멀티플레이: 점수 전송 (판정 주기와 동기화) ────────────
        if self._is_multi_mode and self._multi_socket and self._current_session:
            if self._scoring_frame_counter == 0:   # 판정 직후
                self._multi_socket.send_score(
                    score=int(self._scorer.total_score),
                    combo=int(self._scorer.combo),
                    grade=str(self._last_feedback.get("text", "") if self._last_feedback else ""),
                )

    def _update_waiting(self):
        """WAITING 상태 처리 — HOST는 연결 후 모드/곡 선택, CLIENT는 곡 수신 후 READY 이동."""
        if self._multi_found:
            self._multi_found = False
            if self._multi_role == "host":
                # HOST: _multi_connected=True → 렌더링에서 모드 선택 UI 표시
                pass
            else:
                # CLIENT: 곡을 수신했으면 READY로, 아직이면 계속 대기
                if self._current_song:
                    self.transition_to(GameState.READY)

    def _render(self):
        """Render current frame to display."""

        # 매 프레임 버튼 rect 초기화 (현재 화면의 버튼만 등록)
        self._btn_rects.clear()

        w, h = self._display.get_size()

        if self.state == GameState.MENU:
            self._render_menu(w, h)
        elif self.state == GameState.WAITING:
            self._render_waiting(w, h)
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
        elif self.state == GameState.LEADERBOARD:
            self._render_leaderboard(w, h)

        # 멀티플레이 중 상태 배너 (WAITING 화면 제외 — 이미 타이틀이 있음)
        if self._is_multi_mode and self.state != GameState.WAITING:
            self._render_multi_status_banner(w)

        pygame.display.flip()

    def _render_multi_status_banner(self, w):
        """멀티플레이 중 화면 우상단에 상태 배너를 표시."""
        tick = self._neon_tick
        role_str = self._multi_role.upper() if self._multi_role else "?"
        opponent_ip = self._multi_opponent_ip or "..."
        label = f"MULTIPLAY  [{role_str}]  {opponent_ip}"
        col = self._neon_color((255, 80, 160), tick, 0.7)

        surf = self._fonts["small_retro"].render(label, True, col)
        bw = surf.get_width() + 16
        bh = surf.get_height() + 8
        bx = w - bw - 6
        by = 4
        bg_surf = pygame.Surface((bw, bh), pygame.SRCALPHA)
        bg_surf.fill((20, 8, 35, 180))
        self._display.blit(bg_surf, (bx, by))
        pygame.draw.rect(self._display, col, pygame.Rect(bx, by, bw, bh), 1, border_radius=6)
        self._display.blit(surf, (bx + 8, by + 4))

    def _render_waiting(self, w, h):
        """MULTI PLAY 상대방 탐색 중 화면."""
        import math
        tick = self._neon_tick

        # 배경
        self._display.fill((6, 4, 18))
        for y in range(h):
            t = y / h
            pygame.draw.line(self._display, (int(18+10*t), int(4+4*t), int(40+15*t)), (0, y), (w, y))

        # 타이틀
        title_col = self._neon_color((255, 80, 160), tick)
        title_surf = self._fonts["result_big"].render("MULTIPLAY", True, title_col)
        self._display.blit(title_surf, title_surf.get_rect(center=(w // 2, 44)))

        pygame.draw.line(self._display, self._neon_color((180, 40, 100), tick, 0.6),
                         (w // 4, 64), (w * 3 // 4, 64), 1)

        # ── 단계 1: 탐색 중 (아직 연결 안 됨) ─────────────────────
        if not self._multi_connected:
            cx, cy = w // 2, h // 2 - 20
            r_spin = 36
            num_dots = 10
            for i in range(num_dots):
                angle = math.radians(i * (360 / num_dots) + tick * 180)
                dx2 = int(cx + r_spin * math.cos(angle))
                dy2 = int(cy + r_spin * math.sin(angle))
                alpha = int(60 + 195 * (i / num_dots))
                col_d = (int(255 * alpha / 255), int(80 * alpha / 255), int(180 * alpha / 255))
                pygame.draw.circle(self._display, col_d, (dx2, dy2), 5)

            msg = self._multi_status_msg or "상대방 탐색 중..."
            msg_col = (255, 80, 80) if self._multi_timed_out else (220, 220, 255)
            msg_surf = self._fonts["body"].render(msg, True, msg_col)
            self._display.blit(msg_surf, msg_surf.get_rect(center=(w // 2, cy + 50)))

            cancel_rect = pygame.Rect(w // 2 - 100, h // 2 + 100, 200, 44)
            self._btn_rects["btn_waiting_cancel"] = cancel_rect
            hover = cancel_rect.collidepoint(pygame.mouse.get_pos())
            pygame.draw.rect(self._display, (80, 20, 20) if hover else (30, 10, 10),
                             cancel_rect, border_radius=10)
            self._draw_neon_rect(self._display, cancel_rect,
                                 self._neon_color((255, 80, 80), tick) if hover else (120, 40, 40),
                                 width=2, radius=10, glow_radius=6)
            cancel_lbl = self._fonts["small_retro"].render("CANCEL", True, (255, 255, 255))
            self._display.blit(cancel_lbl, cancel_lbl.get_rect(center=cancel_rect.center))
            self._display.blit(
                self._fonts["small_retro"].render("ESC: CANCEL", True, (80, 70, 100)),
                self._fonts["small_retro"].render("ESC: CANCEL", True, (80, 70, 100)).get_rect(center=(w // 2, h - 24))
            )
            return

        # ── 단계 2: HOST 연결됨 — 모드 선택 ──────────────────────
        if self._multi_role == "host" and not self._multi_mode_selected:
            # "-- SELECT MODE --" (타이틀 아래, 버튼 위에 고정)
            sm_col  = self._neon_color((200, 140, 255), tick)
            sm_glow = self._fonts["result_big"].render("-- SELECT MODE --", True, (60, 20, 80))
            sm_surf = self._fonts["result_big"].render("-- SELECT MODE --", True, sm_col)
            SM_Y = 84   # 항상 버튼 위에 고정
            for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
                self._display.blit(sm_glow, sm_glow.get_rect(center=(w//2+dx, SM_Y+dy)))
            self._display.blit(sm_surf, sm_surf.get_rect(center=(w // 2, SM_Y)))
            pygame.draw.line(self._display, self._neon_color((120, 40, 160), tick, 0.5),
                             (w // 4, SM_Y + 14), (w * 3 // 4, SM_Y + 14), 1)

            mode_btns = [
                ("btn_multi_mode_practice",  "PRACTICE",  "1",
                 (0, 220, 180),  (0, 55, 44)),
                ("btn_multi_mode_challenge", "CHALLENGE", "2",
                 (255, 190, 0),  (65, 48, 0)),
                ("btn_multi_mode_freestyle", "FREESTYLE", "3",
                 (200, 80, 255), (55, 14, 75)),
            ]
            BTN_W   = min(400, w - 40)   # 메인 메뉴와 유사한 크기
            BTN_H   = 60
            GAP     = 16
            START_Y = SM_Y + 28          # 구분선 아래 일정 여백 확보

            focus_list = ["btn_multi_mode_practice", "btn_multi_mode_challenge",
                          "btn_multi_mode_freestyle", "btn_waiting_cancel"]
            focus_idx  = self._generic_focus_idx % len(focus_list)

            # 모든 행 중 가장 넓은 row_w 계산 (정렬 통일용)
            max_row_w = 0
            for _, label, icon, _, _ in mode_btns:
                _ic = self._fonts["result_big"].render(f"[{icon}]", True, (255,255,255))
                _lb = self._fonts["result_big"].render(label, True, (255,255,255))
                max_row_w = max(max_row_w, _ic.get_width() + 16 + _lb.get_width())
            base_lx = w // 2 - max_row_w // 2   # 모든 행 공통 시작 X

            mouse_pos = pygame.mouse.get_pos()
            for i, (btn_id, label, icon, col, bg_base) in enumerate(mode_btns):
                bx = w // 2 - BTN_W // 2
                by = START_Y + i * (BTN_H + GAP)
                rect = pygame.Rect(bx, by, BTN_W, BTN_H)
                self._btn_rects[btn_id] = rect
                hover   = rect.collidepoint(mouse_pos)
                focused = (focus_list[focus_idx] == btn_id)
                active  = hover or focused

                # 배경
                bg = tuple(min(255, int(c * 2.8)) for c in bg_base) if active else bg_base
                pygame.draw.rect(self._display, bg, rect, border_radius=14)

                # 좌측 컬러 액센트 바
                accent_rect = pygame.Rect(bx + 4, by + 6, 6, BTN_H - 12)
                pygame.draw.rect(self._display,
                                 self._neon_color(col, tick) if active else tuple(c // 2 for c in col),
                                 accent_rect, border_radius=3)

                # 테두리 (포커스/호버 시 네온 글로우)
                if active:
                    self._draw_neon_rect(self._display, rect,
                                        self._neon_color(col, tick),
                                        width=3, radius=14, glow_radius=10)
                    self._draw_corner_brackets(self._display, rect,
                                              self._neon_color(col, tick * 2),
                                              size=14, width=3)
                else:
                    pygame.draw.rect(self._display, tuple(c // 2 for c in col),
                                     rect, 2, border_radius=14)

                # 아이콘 + 라벨 (레트로 폰트, 수직 중앙 정렬)
                # base_lx 기준 정렬 → 모든 행이 동일한 시작 X (PressStart2P 1 vs 2 너비 차이 해결)
                icon_surf = self._fonts["result_big"].render(
                    f"[{icon}]", True,
                    self._neon_color(col, tick) if active else tuple(min(255, c // 2 + 60) for c in col))
                lbl_surf  = self._fonts["result_big"].render(
                    label, True,
                    (255, 255, 255) if active else (190, 185, 210))

                cy_btn = by + BTN_H // 2
                self._display.blit(icon_surf, icon_surf.get_rect(midleft=(base_lx, cy_btn)))
                self._display.blit(lbl_surf,  lbl_surf.get_rect(midleft=(base_lx + icon_surf.get_width() + 16, cy_btn)))

            # CANCEL 버튼
            cancel_rect = pygame.Rect(w // 2 - 80, h - 52, 160, 34)
            self._btn_rects["btn_waiting_cancel"] = cancel_rect
            focused_c = (focus_list[focus_idx] == "btn_waiting_cancel")
            hover_c   = cancel_rect.collidepoint(pygame.mouse.get_pos())
            active_c  = hover_c or focused_c
            pygame.draw.rect(self._display, (80, 20, 20) if active_c else (30, 10, 10),
                             cancel_rect, border_radius=10)
            pygame.draw.rect(self._display,
                             self._neon_color((255, 80, 80), tick) if active_c else (120, 40, 40),
                             cancel_rect, 2 if not focused_c else 3, border_radius=10)
            cancel_lbl = self._fonts["small_retro"].render("< CANCEL", True,
                                                            (255, 200, 200) if active_c else (180, 120, 120))
            self._display.blit(cancel_lbl, cancel_lbl.get_rect(center=cancel_rect.center))
            return

        # ── 단계 3: CLIENT — HOST가 모드/곡 선택 중 ─────────────
        role_col = self._neon_color((80, 200, 255), tick)
        role_surf = self._fonts["result_big"].render("CLIENT", True, role_col)
        self._display.blit(role_surf, role_surf.get_rect(center=(w // 2, 90)))

        wait_col = self._neon_color((180, 160, 255), tick)
        wait_surf = self._fonts["body"].render("HOST가 모드/곡을 선택 중입니다...", True, wait_col)
        self._display.blit(wait_surf, wait_surf.get_rect(center=(w // 2, 125)))

        # 스피너
        cx, cy = w // 2, h // 2 - 10
        r_spin = 36
        num_dots = 10
        for i in range(num_dots):
            angle = math.radians(i * (360 / num_dots) + tick * 180)
            dx = int(cx + r_spin * math.cos(angle))
            dy = int(cy + r_spin * math.sin(angle))
            alpha = int(60 + 195 * (i / num_dots))
            col = (int(255 * alpha / 255), int(80 * alpha / 255), int(180 * alpha / 255))
            pygame.draw.circle(self._display, col, (dx, dy), 5)

        # 연결 IP 표시
        ip_surf = self._fonts["small_retro"].render(
            f"연결됨: {self._multi_opponent_ip}", True, (100, 180, 120))
        self._display.blit(ip_surf, ip_surf.get_rect(center=(w // 2, cy + 50)))

        # CANCEL 버튼
        cancel_rect = pygame.Rect(w // 2 - 100, h // 2 + 120, 200, 44)
        self._btn_rects["btn_waiting_cancel"] = cancel_rect
        hover = cancel_rect.collidepoint(pygame.mouse.get_pos())
        bg = (80, 20, 20) if hover else (30, 10, 10)
        pygame.draw.rect(self._display, bg, cancel_rect, border_radius=10)
        border_col = self._neon_color((255, 80, 80), tick) if hover else (120, 40, 40)
        self._draw_neon_rect(self._display, cancel_rect, border_col, width=2, radius=10, glow_radius=6)
        cancel_lbl = self._fonts["small_retro"].render("CANCEL", True, (255, 255, 255))
        self._display.blit(cancel_lbl, cancel_lbl.get_rect(center=cancel_rect.center))

        hint_col = (80, 70, 100)
        hint = self._fonts["small_retro"].render("ESC: CANCEL", True, hint_col)
        self._display.blit(hint, hint.get_rect(center=(w // 2, h - 24)))

    def _render_menu(self, w, h):
        """Render the main menu — retro-fancy neon style."""
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
        MARGIN_TOP    = 20
        BTN_H         = 44
        BTN_W         = min(400, w - 60)
        SMALL_BTN_H   = 36
        SMALL_BTN_W   = min(170, (BTN_W - 20) // 2)
        FOOTER_H      = 22
        MARGIN_BOTTOM = 28
        title_area_h  = 106

        mode_area_top    = MARGIN_TOP + title_area_h + 10
        bottom_area_h    = SMALL_BTN_H + FOOTER_H + 12
        mode_area_bottom = h - MARGIN_BOTTOM - bottom_area_h
        mode_area_h      = mode_area_bottom - mode_area_top
        num_btns         = 5
        gap              = max(6, (mode_area_h - num_btns * BTN_H) // (num_btns + 1))
        btn_start_y      = mode_area_top + (mode_area_h - (num_btns * BTN_H + gap * (num_btns - 1))) // 2
        btn_x            = w // 2 - BTN_W // 2

        # ── 타이틀 ───────────────────────────────────────────────
        neon_cyan = self._neon_color((80, 255, 220), tick)
        title_surf = self._fonts["title"].render("Let's Dance!", True, neon_cyan)
        # 타이틀 글로우 (살짝 번짐)
        glow_surf = self._fonts["title"].render("Let's Dance!", True, (30, 120, 100))
        for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
            self._display.blit(glow_surf, glow_surf.get_rect(
                center=(w // 2 + dx, MARGIN_TOP + 34 + dy)))
        self._display.blit(title_surf, title_surf.get_rect(center=(w // 2, MARGIN_TOP + 34)))

        sub_color = self._neon_color((230, 160, 255), tick, intensity=0.9)
        sub = self._fonts["body"].render("* AI DANCE SCORING GAME *", True, sub_color)
        self._display.blit(sub, sub.get_rect(center=(w // 2, MARGIN_TOP + 76)))

        # 구분선
        line_col = self._neon_color((180, 80, 255), tick, 0.7)
        pygame.draw.line(self._display, line_col,
                         (w//2 - BTN_W//2, MARGIN_TOP + 96),
                         (w//2 + BTN_W//2, MARGIN_TOP + 96), 1)

        # ── 모드 버튼 ────────────────────────────────────────────
        mouse_pos = pygame.mouse.get_pos()
        btn_defs = [
            ("btn_practice",    "PRACTICE",    (0, 220, 180),   (0, 80, 60)),
            ("btn_challenge",   "CHALLENGE",   (255, 190, 0),   (90, 60, 0)),
            ("btn_freestyle",   "FREE STYLE",  (200, 100, 255), (70, 20, 100)),
            ("btn_multi_play",  "MULTIPLAY",  (255, 80, 160),  (90, 15, 50)),
            ("btn_leaderboard", "LEADERBOARD", (80, 180, 255),  (10, 50, 90)),
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
            lbl = self._fonts["result_big"].render(label, True, txt_color)
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
            (btn_s_rect, "btn_settings", "SETTINGS", (100, 120, 255), 5),
            (btn_q_rect, "btn_quit",     "QUIT",     (255, 80,  80),  6),
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
            lbl = self._fonts["small_retro"].render(label, True,
                                              (255, 255, 255) if active else (160, 155, 180))
            self._display.blit(lbl, lbl.get_rect(center=rect.center))

        # ── 푸터 ─────────────────────────────────────────────────
        footer_col = (140, 110, 180)
        footer = self._fonts["small_retro"].render(
            "U/D: SELECT   ENTER: CONFIRM   ESC: QUIT", True, footer_col
        )
        self._display.blit(footer, footer.get_rect(
            center=(w // 2, h - MARGIN_BOTTOM + 6)))

        # ── 키보드 안내 박스 (타이틀 아래 우측) ──────────────────
        key_lines = [
            ("- KEYBOARD -",   (180, 180, 255)),
            ("U/D/L/R : MOVE", (200, 200, 220)),
            ("SPACE   : OK",   (200, 200, 220)),
            ("B/ESC   : BACK", (200, 200, 220)),
            ("P       : PAUSE",(200, 200, 220)),
            ("Q       : QUIT", (200, 200, 220)),
        ]
        kx = w - 10
        ky = MARGIN_TOP + 110
        line_h = 16
        pad_x, pad_y = 8, 6
        # 박스 너비를 폰트 실제 렌더 크기에 맞춤
        max_txt_w = max(
            self._fonts["small_retro"].render(txt, True, (0,0,0)).get_width()
            for txt, _ in key_lines
        )
        box_w = max_txt_w + pad_x * 2
        box_h = len(key_lines) * line_h + pad_y * 2
        # 오른쪽 경계 안쪽에 딱 맞게
        box_x = w - box_w - 6
        kb_surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        kb_surf.fill((20, 10, 50, 170))
        self._display.blit(kb_surf, (box_x, ky))
        pygame.draw.rect(self._display, (80, 60, 130),
                         pygame.Rect(box_x, ky, box_w, box_h), 1, border_radius=6)
        for li, (txt, col) in enumerate(key_lines):
            s = self._fonts["small_retro"].render(txt, True, col)
            self._display.blit(s, (box_x + pad_x, ky + pad_y + li * line_h))

    def _render_song_select(self, w, h):
        """곡 선택 화면 — retro-fancy neon style."""
        import math

        tick = self._neon_tick

        MODE_LABELS = {"practice": "PRACTICE", "challenge": "CHALLENGE", "freestyle": "FREE STYLE"}
        DIFF_STARS  = {0: "", 1: "★☆☆☆☆", 2: "★★☆☆☆", 3: "★★★☆☆", 4: "★★★★☆", 5: "★★★★★"}
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
                dur_raw = int(round(song.get('duration', 0)))
                dur_str = f"{dur_raw // 60}m {dur_raw % 60}s" if dur_raw >= 60 else f"{dur_raw}s"
                mode_list = song.get('mode', [])
                if 'practice' in mode_list or 'freestyle' in mode_list:
                    info = f"{dur_str}  ·  {diff}" if diff else dur_str
                else:
                    info = f"BPM {song.get('bpm',0)}  ·  {dur_str}  ·  {diff}"
                info_col = self._neon_color(mode_col, tick, 0.7) if selected else (130, 125, 160)
                i_surf = self._fonts["small_retro"].render(info, True, info_col)
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

                # 앨범 커버 표시 (정사각형, 패널 오른쪽 상단)
                cover_surf = sel.get("_cover_surf")
                cover_size = min(pw * 3 // 4, 240)
                cover_margin = 12
                has_cover = cover_surf is not None
                if has_cover:
                    cover_x = panel.right - cover_size - cover_margin
                    cover_y = panel.y + cover_margin
                    scaled = pygame.transform.smoothscale(cover_surf, (cover_size, cover_size))
                    self._display.blit(scaled, (cover_x, cover_y))
                    # 커버 테두리
                    cover_rect = pygame.Rect(cover_x, cover_y, cover_size, cover_size)
                    pygame.draw.rect(self._display, self._neon_color(mode_col, tick, 0.5),
                                     cover_rect, 2, border_radius=6)
                    # 정보 영역은 커버 왼쪽까지만
                    info_right = cover_x - 8
                else:
                    info_right = panel.right - 16

                # 상세 정보
                sel_dur = int(round(sel.get('duration', 0)))
                sel_dur_str = f"{sel_dur // 60}m {sel_dur % 60}s" if sel_dur >= 60 else f"{sel_dur}s"
                sel_modes = sel.get('mode', [])
                detail_items = [
                    ("TITLE",    sel.get("title", "-")),
                    ("ARTIST",   sel.get("artist", "-")),
                ]
                if 'practice' not in sel_modes and 'freestyle' not in sel_modes:
                    detail_items.append(("BPM", str(sel.get("bpm", 0))))
                detail_items.append(("LENGTH", sel_dur_str))
                diff_val = DIFF_STARS.get(sel.get("difficulty", 0), "")
                if diff_val:
                    detail_items.append(("LEVEL", diff_val))
                avail_h   = panel.height - start_h - start_m * 2 - 16
                item_h    = min(54, max(38, avail_h // max(len(detail_items), 1)))
                py_detail = panel.y + 14

                for lbl_txt, val_txt in detail_items:
                    lbl_s = self._fonts["small_retro"].render(lbl_txt, True, (150, 140, 190))
                    val_s = self._fonts["body"].render(str(val_txt), True,
                                                        self._neon_color(mode_col, tick, 0.85))
                    self._display.blit(lbl_s, (panel.x + 16, py_detail))
                    self._display.blit(val_s, (panel.x + 16, py_detail + 17))
                    py_detail += item_h

                # 시작 버튼 — depth2일 때 강하게 하이라이트
                start_rect = pygame.Rect(panel.x + 16,
                                         panel.y + panel.height - start_h - start_m,
                                         pw - 32, start_h)
                self._btn_rects["btn_song_start"] = start_rect
                hover_s    = start_rect.collidepoint(mouse_pos)
                focused_s  = (self._song_depth == 2)   # depth2면 START 포커스

                btn_bg = pygame.Surface((start_rect.width, start_rect.height), pygame.SRCALPHA)
                if focused_s:
                    btn_bg.fill((*mode_col, 210))
                elif hover_s:
                    btn_bg.fill((*[c // 2 for c in mode_col], 200))
                else:
                    btn_bg.fill((*[c // 4 for c in mode_col], 180))
                self._display.blit(btn_bg, start_rect.topleft)
                glow = 14 if focused_s else (10 if hover_s else 4)
                border_col = (255, 255, 100) if focused_s else self._neon_color(mode_col, tick)
                self._draw_neon_rect(self._display, start_rect, border_col,
                                     width=3 if focused_s else 2, radius=14, glow_radius=glow)
                if focused_s:
                    self._draw_corner_brackets(self._display, start_rect,
                                               self._neon_color((255, 220, 80), tick * 2),
                                               size=14, width=3)
                lbl_col = (30, 20, 50) if focused_s else (255, 255, 255)
                go_s = self._fonts["menu"].render("START!", True, lbl_col)
                self._display.blit(go_s, go_s.get_rect(center=start_rect.center))
                hint_key = "ENTER / SPACE" if focused_s else "ENTER → START"
                hint_s = self._fonts["small_retro"].render(hint_key, True,
                                                            self._neon_color(mode_col, tick, 0.7))
                self._display.blit(hint_s, (start_rect.x + 6, start_rect.bottom - 16))

        # ── 뒤로가기 버튼 ────────────────────────────────────────
        back_y    = h - BOTTOM_MARGIN - FOOTER_H + 6
        back_rect = pygame.Rect(14, back_y, 160, 42)
        self._btn_rects["btn_song_back"] = back_rect
        hover_b   = back_rect.collidepoint(mouse_pos)
        songs_len = len(self._songs_for_mode(self._current_mode))
        focused_b = (self._song_focus_idx == songs_len)   # 뎁스1: BACK = 마지막

        # 포커스/호버 배경
        bb_surf = pygame.Surface((back_rect.width, back_rect.height), pygame.SRCALPHA)
        bb_fill = (80, 50, 140) if focused_b else (40, 25, 70) if hover_b else (20, 12, 40)
        bb_surf.fill((*bb_fill, 230))
        self._display.blit(bb_surf, back_rect.topleft)
        # 테두리 — 포커스 시 밝은 노란색 두꺼운 테두리
        border_c = (255, 255, 100) if focused_b else self._neon_color((160, 140, 220), tick) if hover_b else (80, 65, 120)
        border_w = 3 if focused_b else 2
        pygame.draw.rect(self._display, border_c, back_rect, border_w, border_radius=12)
        if focused_b:
            self._draw_corner_brackets(self._display, back_rect,
                                       self._neon_color((255, 220, 80), tick * 2), size=12, width=3)
        back_s = self._fonts["btn_retro"].render("< BACK", True,
                                             (255, 255, 255) if (focused_b or hover_b) else (160, 155, 185))
        self._display.blit(back_s, back_s.get_rect(center=back_rect.center))

        # ── 푸터 안내 ──────────────────────────────────────────────
        nav_hint = "U/D: SELECT SONG   ENTER/→: START   ←/ESC: BACK"
        hint = self._fonts["small_retro"].render(nav_hint, True, (130, 110, 170))
        self._display.blit(hint, hint.get_rect(center=(w // 2, h - 22)))

    def _render_ready(self, w, h):
        """준비 화면: 좌=웹캠, 우=스켈레톤 + 전신 감지 안내 + OK 사인 대기."""

        MID_X = w // 2
        HEADER_H = 48
        FOOTER_H = 44

        # 배경
        self._display.fill((8, 6, 22))
        pygame.draw.line(self._display, (50, 45, 90),
                         (MID_X, HEADER_H), (MID_X, h - FOOTER_H), 2)

        # ── 헤더 ──────────────────────────────────────────────
        pygame.draw.rect(self._display, (18, 14, 45), pygame.Rect(0, 0, w, HEADER_H))
        song  = self._current_song or {}
        song_title = song.get("title", "")
        bpm_val    = song.get("bpm", 0)
        dur_val    = int(round(song.get("duration", 0)))
        dur_str_r  = f"{dur_val // 60}m {dur_val % 60}s" if dur_val >= 60 else f"{dur_val}s"
        diff_map   = {0:"FREE",1:"★☆☆☆☆",2:"★★☆☆☆",3:"★★★☆☆",4:"★★★★☆",5:"★★★★★"}
        diff_lbl   = diff_map.get(song.get("difficulty", 0), "")
        mode_color = {"practice":(0,220,180),"challenge":(255,190,0),"freestyle":(200,100,255)}
        mcol       = mode_color.get(self._current_mode, (180,180,255))
        hdr_txt    = f"{song_title}  --  GET READY!" if song_title else "GET READY!"
        hdr = self._fonts["body"].render(hdr_txt, True, (180, 180, 255))
        self._display.blit(hdr, hdr.get_rect(center=(w // 2, HEADER_H // 2)))
        # BPM / 길이 / 난이도 (오른쪽 상단)
        if bpm_val or dur_val:
            if self._current_mode in ("practice", "freestyle"):
                info_str = dur_str_r
            else:
                info_str = f"BPM {bpm_val}  |  {dur_str_r}  |  {diff_lbl}"
            info_surf = self._fonts["small_retro"].render(info_str, True, self._neon_color(mcol, self._neon_tick, 0.8))
            self._display.blit(info_surf, info_surf.get_rect(midright=(w - 12, HEADER_H // 2)))

        body_h = h - HEADER_H - FOOTER_H

        # ── 왼쪽: 웹캠 피드 ──────────────────────────────────
        pygame.draw.rect(self._display, (12, 10, 30),
                         pygame.Rect(0, HEADER_H, MID_X, body_h))
        lbl_cam = self._fonts["small_retro"].render("MY CAM", True, (100, 160, 255))
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
        lbl_sk = self._fonts["small_retro"].render("MY POSE", True, (255, 160, 80))
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

        # 멀티플레이: 내 포즈 준비 완료 후 상대방 대기 중 안내
        if self._is_multi_mode and self._multi_my_pose_ready:
            if self._multi_role == "host":
                wait_msg = "POSE READY!  WAITING FOR OPPONENT..."
            else:
                wait_msg = "POSE READY!  WAITING FOR HOST..."
            wait_surf = self._fonts["small_retro"].render(
                wait_msg, True,
                self._neon_color((80, 220, 255), self._neon_tick))
            self._display.blit(wait_surf, wait_surf.get_rect(center=(w // 2, fy + FOOTER_H // 2 + 18)))

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
        skip_lbl = self._fonts["small_retro"].render("SKIP", True, (255, 255, 255) if active_s else (200, 255, 220))
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
        cancel_lbl = self._fonts["small_retro"].render("CANCEL", True, (255, 255, 255) if active_c else (255, 200, 200))
        self._display.blit(cancel_lbl, cancel_lbl.get_rect(center=cancel_rect.center))

    def _render_countdown(self, w, h):
        """Render countdown screen."""

        self._display.fill((10, 5, 30))

        count = max(self._countdown_timer, 0)

        txt_surface = self._fonts["countdown"].render(str(count + 1), True, (0, 255, 255))
        self._display.blit(txt_surface, txt_surface.get_rect(center=(w // 2, h // 2)))

        sub = self._fonts["result_big"].render("GET READY!", True, (180, 180, 220))
        self._display.blit(sub, sub.get_rect(center=(w // 2, h // 2 + 100)))

        # 취소 버튼 (방향키/Enter 접근 가능)
        cancel_rect = pygame.Rect(w // 2 - 100, h - 80, 200, 48)
        self._btn_rects["btn_countdown_cancel"] = cancel_rect
        hover   = cancel_rect.collidepoint(pygame.mouse.get_pos())
        focused = (getattr(self, '_generic_focus_idx', 0) == 0)
        active  = hover or focused
        pygame.draw.rect(self._display, (120, 40, 40) if active else (80, 30, 30),
                         cancel_rect, border_radius=12)
        pygame.draw.rect(self._display, (255, 255, 100) if focused else (200, 100, 100),
                         cancel_rect, 3 if focused else 2, border_radius=12)
        lbl = self._fonts["btn_retro"].render("CANCEL", True, (255, 255, 255) if active else (255, 200, 200))
        self._display.blit(lbl, lbl.get_rect(center=cancel_rect.center))

        hint = self._fonts["small_retro"].render("ESC: CANCEL  |  ENTER: CANCEL", True, (100, 90, 130))
        self._display.blit(hint, hint.get_rect(center=(w // 2, h - 36)))

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

    def _get_skeleton_color_bgr(self):
        """현재 피드백 등급에 따라 스켈레톤 BGR 색상을 반환합니다."""
        if self._last_feedback and self._feedback_timer > 0:
            r, g, b = self._last_feedback.get("color", (0, 180, 255))
            line_bgr = (b, g, r)
            fill_bgr = (min(int(b * 1.3), 255), min(int(g * 1.3), 255), min(int(r * 1.3), 255))
            ring_bgr = (int(b * 0.7), int(g * 0.7), int(r * 0.7))
            return line_bgr, fill_bgr, ring_bgr
        return (0, 255, 180), (0, 255, 255), (0, 200, 150)

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
        HEADER_H = 55
        FOOTER_H = 44          # 여유 있는 하단 영역
        BORDER   = 4          # 패널 테두리 두께
        MID_X    = w // 2
        BODY_Y   = HEADER_H
        BODY_H   = h - HEADER_H - FOOTER_H
        tick     = self._neon_tick

        # ── 배경 ──────────────────────────────────────────────────
        self._display.fill((6, 4, 18))

        # ── 패널 영역 ──────────────────────────────────────────────
        is_freestyle = (self._current_mode == "freestyle")
        if is_freestyle:
            # freestyle: 유저 카메라만 전체 화면
            left_rect  = pygame.Rect(0, BODY_Y, w, BODY_H)
            right_rect = pygame.Rect(0, 0, 0, 0)  # 사용 안 함
        else:
            left_rect  = pygame.Rect(0,     BODY_Y, MID_X,     BODY_H)
            right_rect = pygame.Rect(MID_X, BODY_Y, w - MID_X, BODY_H)

        # ══════════════════════════════════════════════════════
        #  LEFT — 웹캠 + 스켈레톤 오버레이 (전체 패널 크기)
        # ══════════════════════════════════════════════════════
        pygame.draw.rect(self._display, (8, 8, 22), left_rect)

        if hasattr(self, '_current_frame') and self._current_frame is not None:
            panel_w = left_rect.width - BORDER * 2
            panel_h = BODY_H - BORDER * 2

            # 카메라 원본 비율 유지하면서 패널에 맞추기 (letterbox)
            src_h, src_w = self._current_frame.shape[:2]
            scale = min(panel_w / src_w, panel_h / src_h)
            cam_w_target = int(src_w * scale)
            cam_h_target = int(src_h * scale)
            cam_x_offset = left_rect.x + BORDER + (panel_w - cam_w_target) // 2
            cam_y_offset = BODY_Y + BORDER + (panel_h - cam_h_target) // 2

            cur_seq = getattr(self, '_current_frame_seq', -1)
            if cur_seq != self._rendered_cam_seq or not hasattr(self, '_cam_surf'):
                frame_bgr = cv2.resize(self._current_frame,
                                       (cam_w_target, cam_h_target),
                                       interpolation=cv2.INTER_NEAREST)

                # 스켈레톤을 카메라 프레임 위에 직접 그리기
                if hasattr(self, '_pose_detected') and self._pose_detected and \
                        self._current_landmarks is not None:
                    line_bgr, fill_bgr, ring_bgr = self._get_skeleton_color_bgr()
                    lm = self._current_landmarks

                    # 피드백 등급에 따라 스켈레톤 색상 결정 (BGR — feedback.py 색상 기준)
                    fb_text = (self._last_feedback or {}).get("text", "") if self._last_feedback else ""
                    _SK_COLORS = {
                        # RGB → BGR 변환: feedback.py 실제 색상과 동일
                        "PERFECT!": ((0, 215, 255), (0, 195, 240), (0, 160, 200)),  # Gold
                        "GREAT!":   ((128, 255, 0), (100, 240, 0), (80,  200, 0)),  # Green
                        "GOOD":     ((255, 200, 0), (240, 180, 0), (200, 150, 0)),  # Cyan
                        "OK":       ((200, 200, 200), (180, 180, 180), (140, 140, 140)),  # Gray
                        "MISS":     ((128, 128, 128), (100, 100, 100), (80,  80,  80)),   # Dark gray
                    }
                    sk_line, sk_fill, sk_ring = _SK_COLORS.get(
                        fb_text, ((0, 255, 180), (0, 255, 255), (0, 200, 150)))

                    for src_j, dst_j in SKELETON_CONNECTIONS:
                        if src_j < len(lm) and dst_j < len(lm) and \
                           lm[src_j][3] > 0.3 and lm[dst_j][3] > 0.3:
                            x1 = int(lm[src_j][0] * cam_w_target)
                            y1 = int(lm[src_j][1] * cam_h_target)
                            x2 = int(lm[dst_j][0] * cam_w_target)
                            y2 = int(lm[dst_j][1] * cam_h_target)
                            cv2.line(frame_bgr, (x1, y1), (x2, y2), sk_line, 3)
                    for idx in DANCE_JOINTS:
                        if idx < len(lm) and lm[idx][3] > 0.3:
                            cx_ = int(lm[idx][0] * cam_w_target)
                            cy_ = int(lm[idx][1] * cam_h_target)
                            cv2.circle(frame_bgr, (cx_, cy_), 5, sk_fill, -1)
                            cv2.circle(frame_bgr, (cx_, cy_), 8, sk_ring, 2)

                frame_rgb = frame_bgr[:, :, ::-1]
                self._cam_surf = pygame.image.frombuffer(
                    frame_rgb.tobytes(), (cam_w_target, cam_h_target), "RGB")
                self._rendered_cam_seq = cur_seq

            self._display.blit(self._cam_surf, (cam_x_offset, cam_y_offset))
        else:
            no_cam = self._fonts["body"].render("NO CAMERA", True, (80, 80, 110))
            self._display.blit(no_cam, no_cam.get_rect(center=left_rect.center))

        # 좌 패널 레이블은 나중에 테두리와 함께 그린다 (effects 위)

        # ══════════════════════════════════════════════════════
        #  RIGHT — 레퍼런스 영상 / 스틱피겨 (freestyle이면 건너뜀)
        # ══════════════════════════════════════════════════════
        if not is_freestyle:
            pygame.draw.rect(self._display, (10, 6, 22), right_rect)

            has_video = getattr(self, '_async_video_player', None) is not None

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


        # ── 피드백 이펙트 오버레이 (프리스타일에서는 숨김) ──────────────────
        if self._last_feedback and self._feedback_timer > 0 and not is_freestyle:
            self._draw_feedback_effect(w, h, MID_X, BODY_Y, BODY_H)

        # ── 패널 풀 테두리 + 레이블 (effects 위에 그려 항상 보임) ─────
        left_border_col  = self._neon_color((60, 180, 255), tick, 0.9)
        right_border_col = self._neon_color((255, 160, 40), tick, 0.9)
        # glow_radius=0 → 안쪽으로 번지지 않는 단순 테두리
        self._draw_neon_rect(self._display, left_rect,  left_border_col,  width=3, radius=0, glow_radius=0)
        if not is_freestyle:
            self._draw_neon_rect(self._display, right_rect, right_border_col, width=3, radius=0, glow_radius=0)

        lbl_me    = self._fonts["small_retro"].render("ME",    True, (120, 200, 255))
        lbl_guide = self._fonts["small_retro"].render("GUIDE", True, (255, 180, 80))
        self._display.blit(lbl_me,    (BORDER + 8,           BODY_Y + BORDER + 6))
        if not is_freestyle:
            self._display.blit(lbl_guide, (MID_X + BORDER + 8,  BODY_Y + BORDER + 6))

        # ══════════════════════════════════════════════════════
        #  HEADER
        # ══════════════════════════════════════════════════════
        header_bg = pygame.Rect(0, 0, w, HEADER_H)
        pygame.draw.rect(self._display, (12, 8, 30), header_bg)
        # 헤더 하단 네온 라인
        hdr_line_col = self._neon_color((80, 60, 160), tick, 0.5)
        pygame.draw.line(self._display, hdr_line_col, (0, HEADER_H), (w, HEADER_H), 2)

        # 점수 (레트로 폰트) — 프리스타일에서는 숨김
        if self._current_mode != "freestyle":
            score_col = self._neon_color((0, 255, 200), tick)
            score_surf = self._fonts["score"].render(
                f"{int(self._scorer.total_score):06d}", True, score_col)
            self._display.blit(score_surf, score_surf.get_rect(midleft=(16, HEADER_H // 2)))

        # 콤보 — 프리스타일에서는 숨김
        if self._current_mode != "freestyle":
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

        for i, (rect, label, base_c) in enumerate([
            (pause_rect, "PAUSE", (55, 55, 130)),
            (menu_rect,  "MENU",  (100, 38, 38)),
        ]):
            hover   = rect.collidepoint(mouse_pos)
            focused = (getattr(self, '_generic_focus_idx', 0) == i)
            active  = hover or focused
            color   = tuple(min(c + 40, 255) for c in base_c) if active else base_c
            pygame.draw.rect(self._display, color, rect, border_radius=8)
            border_col = (255, 255, 100) if focused else (160, 160, 210)
            pygame.draw.rect(self._display, border_col, rect, 2 if focused else 1, border_radius=8)
            lbl = self._fonts["btn_retro"].render(label, True, (240, 240, 240))
            self._display.blit(lbl, lbl.get_rect(center=rect.center))

        # ══════════════════════════════════════════════════════
        #  FOOTER  (2행 레이아웃)
        # ══════════════════════════════════════════════════════
        fy = h - FOOTER_H
        pygame.draw.rect(self._display, (10, 8, 26), pygame.Rect(0, fy, w, FOOTER_H))
        pygame.draw.line(self._display, (40, 35, 70), (0, fy), (w, fy), 1)

        song_title  = (self._current_song or {}).get("title", "데모")
        mode_labels = {"practice": "PRACTICE", "challenge": "CHALLENGE", "freestyle": "FREE"}
        mode_label  = mode_labels.get(self._current_mode, "")
        mode_colors = {"practice": (0, 220, 180), "challenge": (255, 180, 0), "freestyle": (200, 100, 255)}
        mode_col_ft = mode_colors.get(self._current_mode, (180, 180, 255))

        # 1행: 모드+타이틀 (왼) / 조작 안내 또는 MISS 경고 (오)
        ROW1_Y = fy + 6
        footer_left = self._fonts["small"].render(
            f"[{mode_label}]  {song_title}", True, mode_col_ft)
        self._display.blit(footer_left, (14, ROW1_Y))

        # (MISS 카운터 UI 삭제됨)
        footer_right = self._fonts["small_retro"].render(
            "P: PAUSE  |  ESC: MENU", True, (80, 80, 110))
        self._display.blit(footer_right,
                           footer_right.get_rect(midright=(w - 10, ROW1_Y + 8)))

        # 2행: 진행 바 + 남은 시간
        ROW2_Y = fy + FOOTER_H // 2 + 4
        elapsed  = self._current_session.elapsed_time if self._current_session else 0
        duration = (self._current_song or {}).get("duration", 60)
        pct      = min(1.0, elapsed / max(duration, 1))
        bar_margin = 80
        bar_x      = bar_margin
        bar_w_total = w - bar_margin * 2 - 50
        bar_h_px   = 8
        pygame.draw.rect(self._display, (40, 35, 70),
                         pygame.Rect(bar_x, ROW2_Y, bar_w_total, bar_h_px), border_radius=4)
        fill_w = int(bar_w_total * pct)
        bar_fill_col = self._neon_color(mode_col_ft, tick, 0.9)
        if fill_w > 0:
            pygame.draw.rect(self._display, bar_fill_col,
                             pygame.Rect(bar_x, ROW2_Y, fill_w, bar_h_px), border_radius=4)
        remain_s = max(0, duration - elapsed)
        time_lbl = self._fonts["small_retro"].render(
            f"{int(remain_s // 60):02d}:{int(remain_s % 60):02d}", True, (160, 160, 200))
        self._display.blit(time_lbl, time_lbl.get_rect(midleft=(bar_x + bar_w_total + 8, ROW2_Y + 4)))

        # ══════════════════════════════════════════════════════
        #  멀티플레이 상대방 점수 오버레이 (우측 하단 패널, 화면 미가림)
        # ══════════════════════════════════════════════════════
        if self._is_multi_mode and self._multi_socket:
            self._render_opponent_score_overlay(w, h, HEADER_H, fy)

    def _render_opponent_score_overlay(self, w, h, header_h, footer_y):
        """게임 화면 오른쪽 하단 구석에 상대방 점수를 작은 패널로 표시.

        패널 크기: 약 180×80px — 화면 콘텐츠(카메라/가이드)를 가리지 않는 위치.
        """
        sock = self._multi_socket
        if sock is None:
            return

        PAD = 8
        PANEL_W = 188
        PANEL_H = 78
        # 오른쪽 패널(가이드 영역) 안쪽 하단 구석
        px = w - PANEL_W - 10
        py = footer_y - PANEL_H - 8

        tick = self._neon_tick

        # 반투명 배경
        bg_surf = pygame.Surface((PANEL_W, PANEL_H), pygame.SRCALPHA)
        bg_surf.fill((8, 4, 22, 200))
        self._display.blit(bg_surf, (px, py))

        # 테두리 색: 연결 중=분홍, 연결 끊김=회색
        if sock.opponent_connected:
            border_col = self._neon_color((255, 80, 160), tick, 0.8)
        else:
            border_col = (80, 80, 80)
        panel_rect = pygame.Rect(px, py, PANEL_W, PANEL_H)
        self._draw_neon_rect(self._display, panel_rect, border_col,
                             width=2, radius=8, glow_radius=4)

        # "OPPONENT" 레이블
        lbl = self._fonts["small_retro"].render("OPPONENT", True, (200, 140, 200))
        self._display.blit(lbl, (px + PAD, py + PAD))

        # 연결 상태 표시 (점)
        dot_col = (0, 255, 120) if sock.opponent_connected else (120, 120, 120)
        pygame.draw.circle(self._display, dot_col, (px + PANEL_W - PAD - 5, py + PAD + 6), 5)

        # 점수
        score_col = self._neon_color((255, 120, 200), tick)
        score_txt = f"{int(sock.opponent_score):06d}"
        score_surf = self._fonts["score"].render(score_txt, True, score_col)
        # score 폰트가 클 수 있으니 스케일 다운 (PANEL_W - 2*PAD 기준)
        max_w = PANEL_W - PAD * 2
        if score_surf.get_width() > max_w:
            scale = max_w / score_surf.get_width()
            score_surf = pygame.transform.smoothscale(
                score_surf,
                (int(score_surf.get_width() * scale), int(score_surf.get_height() * scale))
            )
        self._display.blit(score_surf, (px + PAD, py + PAD + 18))

        # 콤보 + 최근 등급
        combo_val = sock.opponent_combo
        grade_txt = sock.opponent_grade
        detail_parts = []
        if combo_val > 0:
            detail_parts.append(f"{combo_val}x")
        if grade_txt:
            detail_parts.append(grade_txt)
        if detail_parts:
            detail_col = (200, 200, 255)
            detail_surf = self._fonts["small_retro"].render(" ".join(detail_parts), True, detail_col)
            self._display.blit(detail_surf, (px + PAD, py + PANEL_H - PAD - detail_surf.get_height()))

    def _update_and_draw_particles(self):
        """파티클 업데이트 + 화면 그리기."""
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

        # +점수 (0점이라도 표시)
        if pts >= 0 and alpha > 30:
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

        # 반투명 어두운 오버레이
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self._display.blit(overlay, (0, 0))

        # 타이틀
        pause_txt = self._fonts["result_big"].render("PAUSED", True, (255, 255, 255))
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
            lbl = self._fonts["btn_retro"].render(label, True, (255, 255, 255))
            self._display.blit(lbl, lbl.get_rect(center=rect.center))

        hint = self._fonts["small_retro"].render("P / SPACE: RESUME  |  ESC: MENU", True, (160, 160, 180))
        self._display.blit(hint, hint.get_rect(center=(w // 2, h - 38)))

    def _render_result(self, w, h):
        """Render result screen."""
        import math

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

        MARGIN_TOP    = 18
        MARGIN_BOTTOM = 36
        FOOTER_H      = 22
        BTN_H         = 46
        BTN_W         = min(160, (w - 80) // 3)
        btn_gap       = 14
        btn_area_y    = h - MARGIN_BOTTOM - FOOTER_H - BTN_H - 8

        mode_colors = {"practice": (0,220,180), "challenge": (255,190,0), "freestyle": (200,100,255)}
        mode_labels = {"practice": "PRACTICE", "challenge": "CHALLENGE", "freestyle": "FREE STYLE"}

        # ── 멀티플레이 레이아웃 ──────────────────────────────────
        if self._is_multi_mode and self._multi_socket:
            self._render_result_multi(w, h, tick, MARGIN_TOP, btn_area_y,
                                      mode_colors, mode_labels)
        else:
            # ── 싱글플레이 레이아웃 (기존) ──────────────────────
            self._render_result_single(w, h, tick, MARGIN_TOP, btn_area_y,
                                       mode_colors, mode_labels)

        # ── 하단 버튼 (공통) ─────────────────────────────────────
        mouse_pos = pygame.mouse.get_pos()
        total_w = BTN_W * 3 + btn_gap * 2
        btn_x   = w // 2 - total_w // 2
        btn_defs = [
            ("btn_retry",        "RETRY",  (0, 140, 90)),
            ("btn_result_songs", "SONGS",  (60, 100, 200)),
            ("btn_result_menu",  "MENU",   (100, 40, 120)),
        ]
        for i, (btn_name, label, color) in enumerate(btn_defs):
            rect = pygame.Rect(btn_x + i * (BTN_W + btn_gap), btn_area_y, BTN_W, BTN_H)
            self._btn_rects[btn_name] = rect
            hover   = rect.collidepoint(mouse_pos)
            focused = (getattr(self, '_generic_focus_idx', 0) == i)
            draw_color = tuple(min(c + 50, 255) for c in color) if (hover or focused) else color
            pygame.draw.rect(self._display, draw_color, rect, border_radius=14)
            border_col = (255, 255, 100) if focused else (220, 220, 220)
            border_w   = 3 if focused else 2
            pygame.draw.rect(self._display, border_col, rect, border_w, border_radius=14)
            lbl = self._fonts["btn_retro"].render(label, True, (255, 255, 255))
            self._display.blit(lbl, lbl.get_rect(center=rect.center))

        hint = self._fonts["small_retro"].render(
            "←/→: SELECT  ENTER: CONFIRM  S: SCREENSHOT  ESC: MENU", True, (100, 100, 130)
        )
        self._display.blit(hint, hint.get_rect(center=(w // 2, h - MARGIN_BOTTOM + 10)))

        # ── 이름 입력 오버레이 ───────────────────────────────────
        if self._name_input_active:
            self._render_name_input_overlay(w, h)

    # ── 결과 화면: 싱글플레이 ─────────────────────────────────────────

    def _render_result_single(self, w, h, tick, MARGIN_TOP, btn_area_y,
                              mode_colors, mode_labels):
        """싱글플레이 결과 — 기존 레이아웃."""
        # 타이틀
        if self._challenge_game_over:
            title_text, title_col, glow_col = "GAME OVER!", \
                self._neon_color((255,60,60), tick), (80,0,0)
        else:
            title_text, title_col, glow_col = "DANCE COMPLETE!", \
                self._neon_color((255,220,50), tick), (100,80,0)
        title = self._fonts["result_big"].render(title_text, True, title_col)
        glow  = self._fonts["result_big"].render(title_text, True, glow_col)
        for dx, dy in [(-3,0),(3,0),(0,-3),(0,3)]:
            self._display.blit(glow, glow.get_rect(center=(w//2+dx, MARGIN_TOP+30+dy)))
        self._display.blit(title, title.get_rect(center=(w//2, MARGIN_TOP+30)))

        mbadge = self._fonts["small_retro"].render(
            f"[ {mode_labels.get(self._current_mode,'')} ]", True,
            self._neon_color(mode_colors.get(self._current_mode,(180,180,255)), tick))
        self._display.blit(mbadge, mbadge.get_rect(midright=(w-16, MARGIN_TOP+30)))

        line_col = self._neon_color((200,100,255), tick, 0.7)
        pygame.draw.line(self._display, line_col,
                         (w//4, MARGIN_TOP+54), (w*3//4, MARGIN_TOP+54), 1)

        content_top    = MARGIN_TOP + 70
        content_bottom = btn_area_y - 20

        if self._result_data:
            data = self._result_data
            if self._current_mode == "freestyle":
                msg = self._fonts["result_big"].render(
                    "GREAT MOVES!", True, self._neon_color((200,100,255), tick))
                self._display.blit(msg, msg.get_rect(
                    center=(w//2, (content_top+content_bottom)//2)))
            else:
                items = [
                    (f"SCORE:     {data.get('total_score',0)}",       (0,255,200)),
                    (f"MAX COMBO: {data.get('max_combo',0)}",         (255,220,0)),
                    (f"MOVES:     {data.get('total_moves',0)}",       (200,200,220)),
                    (f"AVG:       {data.get('average_score',0):.1f}", (180,180,255)),
                    (f"GRADE:     {data.get('final_grade','-')}",     (255,180,0)),
                ]
                hits = data.get("hit_counts", {})
                total_items = len(items) + (1 if hits else 0)
                item_gap = min(48, max(28, (content_bottom - content_top) // max(total_items,1)))
                y = content_top
                for text, color in items:
                    surf = self._fonts["result_big"].render(text, True, color)
                    self._display.blit(surf, surf.get_rect(center=(w//2, y)))
                    y += item_gap
                if hits:
                    y += 4
                    hit_surf = self._fonts["small_retro"].render(
                        "  |  ".join(f"{k}: {v}" for k,v in hits.items()),
                        True, (160,160,180))
                    self._display.blit(hit_surf, hit_surf.get_rect(center=(w//2, y)))
                    y += item_gap
                self._leaderboard_load()
                mode_entries = sorted(
                    [e for e in self._leaderboard if e.get("mode","practice")==self._current_mode],
                    key=lambda e: e.get("score",0), reverse=True)
                cur_score = data.get("total_score", 0)
                rank  = sum(1 for e in mode_entries if e.get("score",0) > cur_score) + 1
                total = len(mode_entries)
                rank_col = (255,220,50) if rank==1 else (0,220,200) if rank<=3 else (180,180,220)
                rank_surf = self._fonts["small_retro"].render(
                    f"YOUR RANK:  #{rank}  of  {total}  [{mode_labels.get(self._current_mode,'')}]",
                    True, rank_col)
                self._display.blit(rank_surf, rank_surf.get_rect(center=(w//2, y+6)))

    # ── 결과 화면: 멀티플레이 ─────────────────────────────────────────

    def _render_result_multi(self, w, h, tick, MARGIN_TOP, btn_area_y,
                             mode_colors, mode_labels):
        """멀티플레이 결과 — 승부 결과 크게 + 싱글과 동일한 스탯 레이아웃."""
        import math

        data      = self._result_data or {}
        sock      = self._multi_socket
        my_score  = int(data.get("total_score", 0))
        opp_score = int(sock.opponent_final_score) if sock else 0
        finished  = sock.opponent_finished if sock else False

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #  상단 영역: 승부 결과 (feedback 폰트 = 72px 레트로, 펄스)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # feedback 폰트 높이 ~80px, 점수 행 ~28px, 여유 포함 → 약 120px
        VS_AREA_H = 118
        vs_cy     = MARGIN_TOP + VS_AREA_H // 2

        if not finished and opp_score == 0:
            wait_surf = self._fonts["result_big"].render(
                "Waiting opponent...", True, (160,160,200))
            self._display.blit(wait_surf, wait_surf.get_rect(center=(w//2, vs_cy)))
        else:
            if my_score > opp_score:
                vs_txt, vs_col, glow_col = "WIN!", (0, 255, 150), (0, 80, 40)
            elif my_score < opp_score:
                vs_txt, vs_col, glow_col = "LOSE", (255, 70, 70), (90, 0, 0)
            else:
                vs_txt, vs_col, glow_col = "DRAW", (255, 220, 0), (90, 70, 0)

            # 펄스 애니메이션 (±4%)
            pulse = 1.0 + 0.04 * math.sin(tick * 4)

            # 글로우 레이어 (feedback 폰트 72px)
            glow_surf = self._fonts["feedback"].render(vs_txt, True, glow_col)
            gs = pygame.transform.smoothscale(glow_surf, (
                int(glow_surf.get_width() * pulse * 1.06),
                int(glow_surf.get_height() * pulse * 1.06),
            ))
            for dx, dy in [(-5,0),(5,0),(0,-5),(0,5),(-4,-4),(4,4)]:
                self._display.blit(gs, gs.get_rect(center=(w//2+dx, vs_cy+dy)))

            # 메인 텍스트
            main_surf = self._fonts["feedback"].render(
                vs_txt, True, self._neon_color(vs_col, tick))
            ms = pygame.transform.smoothscale(main_surf, (
                int(main_surf.get_width() * pulse),
                int(main_surf.get_height() * pulse),
            ))
            self._display.blit(ms, ms.get_rect(center=(w//2, vs_cy)))

            # 점수 비교 행 (승부 결과 바로 아래, result_big 폰트)
            score_y = MARGIN_TOP + VS_AREA_H - 14
            my_col  = self._neon_color(vs_col if my_score >= opp_score else (180,180,200), tick)
            op_col  = self._neon_color((255,80,160), tick)

            me_surf  = self._fonts["result_big"].render(f"ME  {my_score:06d}", True, my_col)
            sep_surf = self._fonts["result_big"].render("  vs  ", True, (120,120,160))
            op_surf  = self._fonts["result_big"].render(f"{opp_score:06d}  OPP", True, op_col)

            total_row_w = me_surf.get_width() + sep_surf.get_width() + op_surf.get_width()
            max_w = w - 32
            if total_row_w > max_w:
                sc = max_w / total_row_w
                def _scale(s):
                    return pygame.transform.smoothscale(s,
                        (int(s.get_width()*sc), int(s.get_height()*sc)))
                me_surf  = _scale(me_surf)
                sep_surf = _scale(sep_surf)
                op_surf  = _scale(op_surf)
                total_row_w = me_surf.get_width() + sep_surf.get_width() + op_surf.get_width()

            rx = w // 2 - total_row_w // 2
            for surf in (me_surf, sep_surf, op_surf):
                self._display.blit(surf, surf.get_rect(midleft=(rx, score_y)))
                rx += surf.get_width()

        # 모드 배지
        mbadge = self._fonts["small_retro"].render(
            f"[ {mode_labels.get(self._current_mode,'')} ]", True,
            self._neon_color(mode_colors.get(self._current_mode,(180,180,255)), tick))
        self._display.blit(mbadge, mbadge.get_rect(midright=(w-16, MARGIN_TOP + 10)))

        # 구분선
        sep_y = MARGIN_TOP + VS_AREA_H + 2
        line_col = self._neon_color((200,100,255), tick, 0.7)
        pygame.draw.line(self._display, line_col, (w//4, sep_y), (w*3//4, sep_y), 1)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #  하단 영역: 내 스탯 — 싱글과 동일한 result_big 레이아웃
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        content_top    = sep_y + 28
        content_bottom = btn_area_y - 10

        if data and self._current_mode != "freestyle":
            items = [
                (f"SCORE:     {data.get('total_score',0)}",       (0,255,200)),
                (f"MAX COMBO: {data.get('max_combo',0)}",         (255,220,0)),
                (f"MOVES:     {data.get('total_moves',0)}",       (200,200,220)),
                (f"AVG:       {data.get('average_score',0):.1f}", (180,180,255)),
                (f"GRADE:     {data.get('final_grade','-')}",     (255,180,0)),
            ]
            hits = data.get("hit_counts", {})
            total_items = len(items) + (1 if hits else 0)
            item_gap = min(36, max(22, (content_bottom - content_top) // max(total_items, 1)))
            y = content_top
            for text, color in items:
                surf = self._fonts["result_big"].render(text, True, color)
                self._display.blit(surf, surf.get_rect(center=(w//2, y)))
                y += item_gap
            if hits:
                y += 4
                hit_surf = self._fonts["small_retro"].render(
                    "  |  ".join(f"{k}: {v}" for k, v in hits.items()),
                    True, (160, 160, 180))
                self._display.blit(hit_surf, hit_surf.get_rect(center=(w//2, y)))

    def _render_name_input_overlay(self, w, h):
        """결과 화면 위에 표시되는 이름 입력 반투명 오버레이."""
        # 반투명 배경
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        self._display.blit(overlay, (0, 0))

        BOX_W, BOX_H = min(480, w - 60), 220
        bx = w // 2 - BOX_W // 2
        by = h // 2 - BOX_H // 2

        # 박스 배경 + 테두리
        pygame.draw.rect(self._display, (15, 10, 35), (bx, by, BOX_W, BOX_H), border_radius=18)
        pygame.draw.rect(self._display, self._neon_color((160, 80, 255), self._neon_tick),
                         (bx, by, BOX_W, BOX_H), 2, border_radius=18)

        # 타이틀
        title_s = self._fonts["small_retro"].render("ENTER YOUR NAME", True, (220, 180, 255))
        self._display.blit(title_s, title_s.get_rect(center=(w // 2, by + 28)))

        sub_s = self._fonts["small_retro"].render(
            "ESC / SKIP = save previous name", True, (100, 90, 120))
        self._display.blit(sub_s, sub_s.get_rect(center=(w // 2, by + 50)))

        # 입력 박스 — small_retro(14pt) 사용, 박스 높이를 폰트에 맞게 설정
        font = self._fonts["small_retro"]
        INPUT_W = BOX_W - 40
        INPUT_H = font.get_height() + 14   # 상하 패딩 7px
        ix = w // 2 - INPUT_W // 2
        iy = by + 70
        text_focused = (getattr(self, '_name_focus_idx', 0) == 0)
        input_bg = (40, 28, 70) if text_focused else (30, 20, 55)
        input_border = (0, 255, 200) if text_focused else (180, 130, 255)
        input_border_w = 3 if text_focused else 2
        pygame.draw.rect(self._display, input_bg, (ix, iy, INPUT_W, INPUT_H), border_radius=8)
        pygame.draw.rect(self._display, input_border, (ix, iy, INPUT_W, INPUT_H), input_border_w, border_radius=8)

        # 커서 깜빡임 (텍스트 포커스일 때만 표시)
        display_text = self._name_input_text
        if text_focused and int(self._neon_tick * 2) % 2 == 0:
            display_text += "|"

        # 텍스트 너비가 박스를 넘으면 오른쪽 끝을 보여주도록 클리핑
        text_s = font.render(display_text, True, (255, 255, 255))
        TEXT_MARGIN = 8
        clip_rect = pygame.Rect(ix + TEXT_MARGIN, iy, INPUT_W - TEXT_MARGIN * 2, INPUT_H)
        tx = ix + TEXT_MARGIN
        # 텍스트가 박스보다 넓으면 오른쪽 정렬
        if text_s.get_width() > INPUT_W - TEXT_MARGIN * 2:
            tx = ix + INPUT_W - TEXT_MARGIN - text_s.get_width()
        ty = iy + (INPUT_H - text_s.get_height()) // 2
        self._display.set_clip(clip_rect)
        self._display.blit(text_s, (tx, ty))
        self._display.set_clip(None)

        # SAVE / SKIP 버튼 (더블탭: 첫 탭=하이라이트, 두 번째 탭=실행)
        mouse_pos = pygame.mouse.get_pos()
        BTN_W2, BTN_H2 = 130, 38
        gap = 20
        total_bw = BTN_W2 * 2 + gap
        btn_y = iy + INPUT_H + 18
        save_rect = pygame.Rect(w // 2 - total_bw // 2,              btn_y, BTN_W2, BTN_H2)
        skip_rect = pygame.Rect(w // 2 - total_bw // 2 + BTN_W2 + gap, btn_y, BTN_W2, BTN_H2)
        self._btn_rects["btn_name_save"] = save_rect
        self._btn_rects["btn_name_skip"] = skip_rect

        pending = getattr(self, '_name_btn_pending', '')
        focus_idx = getattr(self, '_name_focus_idx', 0)
        for i, (rect, btn_id, label, base_col) in enumerate([
            (save_rect, "btn_name_save", "SAVE",  (0, 160, 100)),
            (skip_rect, "btn_name_skip", "SKIP",  (80, 80, 110)),
        ]):
            btn_focus_i = i + 1  # 0=텍스트, 1=SAVE, 2=SKIP
            is_pending = (pending == btn_id)
            is_hover   = rect.collidepoint(mouse_pos)
            is_focused = (focus_idx == btn_focus_i)
            if is_pending:
                # 첫 탭 후: 밝게 + 네온 테두리 + "한 번 더" 안내
                col = tuple(min(c + 80, 255) for c in base_col)
                border_col = (255, 255, 80)
                border_w = 3
            elif is_focused:
                col = tuple(min(c + 50, 255) for c in base_col)
                border_col = (0, 255, 200)
                border_w = 3
            elif is_hover:
                col = tuple(min(c + 40, 255) for c in base_col)
                border_col = (200, 200, 220)
                border_w = 2
            else:
                col = base_col
                border_col = (200, 200, 220)
                border_w = 2
            pygame.draw.rect(self._display, col, rect, border_radius=10)
            pygame.draw.rect(self._display, border_col, rect, border_w, border_radius=10)
            lbl_txt = f"[{label}]" if is_pending else label
            # 포커스 표시: 선택 표시자
            if is_focused and not is_pending:
                lbl_txt = f"> {label}"
            lbl = self._fonts["btn_retro"].render(lbl_txt, True, (255, 255, 255))
            self._display.blit(lbl, lbl.get_rect(center=rect.center))

        # 안내 힌트
        if pending:
            hint_s = self._fonts["small_retro"].render(
                "press again to confirm", True, (255, 220, 80))
            self._display.blit(hint_s, hint_s.get_rect(
                center=(w // 2, btn_y + BTN_H2 + 14)))
        else:
            nav_hint = self._fonts["small_retro"].render(
                "arrows: MOVE  SPACE/ENTER: SELECT", True, (100, 90, 130))
            self._display.blit(nav_hint, nav_hint.get_rect(
                center=(w // 2, btn_y + BTN_H2 + 14)))

    @staticmethod
    def _fmt_lb_date(date_str: str) -> str:
        """리더보드 날짜 포매팅: 'YYYY-MM-DD HH:MM' → 'MM/DD HH:MM' (11자)."""
        if len(date_str) >= 16:
            return date_str[5:10].replace("-", "/") + " " + date_str[11:16]
        return date_str[:10]

    def _render_leaderboard(self, w, h):
        """리더보드 화면 렌더링."""

        tick = self._neon_tick

        # 배경 그라데이션
        for y_i in range(h):
            t = y_i / h
            pygame.draw.line(self._display,
                             (int(8 + 8*t), int(5 + 5*t), int(35 + 25*t)),
                             (0, y_i), (w, y_i))
        # 스캔라인
        for y_i in range(0, h, 4):
            scan = pygame.Surface((w, 1), pygame.SRCALPHA)
            scan.fill((0, 0, 0, 35))
            self._display.blit(scan, (0, y_i))

        MARGIN_TOP = 30
        MARGIN_BOTTOM = 50
        BTN_H = 44
        BTN_W = 120

        # 타이틀
        title_col = self._neon_color((80, 200, 255), tick)
        title_surf = self._fonts["result_big"].render("LEADERBOARD", True, title_col)
        glow = self._fonts["result_big"].render("LEADERBOARD", True, (10, 60, 100))
        for dx, dy in [(-3,0),(3,0),(0,-3),(0,3)]:
            self._display.blit(glow, glow.get_rect(center=(w//2+dx, MARGIN_TOP+28+dy)))
        self._display.blit(title_surf, title_surf.get_rect(center=(w//2, MARGIN_TOP+28)))

        line_col = self._neon_color((100, 160, 255), tick, 0.7)
        pygame.draw.line(self._display, line_col,
                         (w//4, MARGIN_TOP+50), (w*3//4, MARGIN_TOP+50), 1)

        # ── 모드 탭 ──────────────────────────────────────────
        tabs = [("all", "ALL"), ("practice", "PRACTICE"), ("challenge", "CHALLENGE"), ("freestyle", "FREE")]
        tab_colors = {"all": (180,180,255), "practice": (0,220,180), "challenge": (255,190,0), "freestyle": (200,100,255)}
        tab_w = min(120, (w - 40) // len(tabs))
        tab_h = 24
        TAB_TOP = MARGIN_TOP + 56
        mouse_pos = pygame.mouse.get_pos()
        for ti, (tab_key, tab_lbl) in enumerate(tabs):
            tx = 20 + ti * (tab_w + 8)
            tab_rect = pygame.Rect(tx, TAB_TOP, tab_w, tab_h)
            self._btn_rects[f"btn_lb_tab_{tab_key}"] = tab_rect
            active_tab = (self._leaderboard_tab == tab_key)
            tcol = tab_colors[tab_key]
            bg_alpha = 200 if active_tab else 80
            bg = pygame.Surface((tab_w, tab_h), pygame.SRCALPHA)
            bg.fill((*[c//3 for c in tcol], bg_alpha))
            self._display.blit(bg, tab_rect.topleft)
            border_col = self._neon_color(tcol, tick) if active_tab else tuple(c//2 for c in tcol)
            pygame.draw.rect(self._display, border_col, tab_rect, 2 if active_tab else 1, border_radius=6)
            ts = self._fonts["small_retro"].render(tab_lbl, True, (255,255,255) if active_tab else (160,155,180))
            self._display.blit(ts, ts.get_rect(center=tab_rect.center))

        TABLE_TOP = TAB_TOP + tab_h + 10

        # 리더보드 데이터 (탭 필터 적용)
        entries = self._lb_get_filtered_entries()

        # 선택 행 범위 클램프
        if entries:
            self._lb_selected_row = min(self._lb_selected_row, len(entries) - 1)
        else:
            self._lb_selected_row = 0

        # 테이블 헤더
        COL_W = max(50, (w - 40) // 6)
        headers = ["RANK", "SONG", "PLAYER", "SCORE", "GRADE", "DATE"]
        header_xs = [20 + i * COL_W for i in range(6)]
        header_col = (180, 180, 255)
        for hi, (hdr, hx) in enumerate(zip(headers, header_xs)):
            hs = self._fonts["small_retro"].render(hdr, True, header_col)
            self._display.blit(hs, (hx, TABLE_TOP))

        pygame.draw.line(self._display, (80, 60, 130),
                         (16, TABLE_TOP+16), (w-16, TABLE_TOP+16), 1)

        # 엔트리 목록
        ROW_H = 22
        max_rows = max(1, (h - MARGIN_BOTTOM - BTN_H - 20 - TABLE_TOP - 24) // ROW_H)
        content_focused = (self._generic_focus_idx == 0)

        # 스크롤 범위 clamp
        total_entries = len(entries)
        max_scroll = max(0, total_entries - max_rows)
        self._lb_max_scroll = max_scroll
        self._lb_scroll = max(0, min(self._lb_scroll, max_scroll))

        if not entries:
            empty = self._fonts["small_retro"].render("No records yet!", True, (120, 120, 160))
            self._display.blit(empty, empty.get_rect(center=(w//2, TABLE_TOP + 50)))
        else:
            visible = entries[self._lb_scroll : self._lb_scroll + max_rows]
            for ri, entry in enumerate(visible):
                abs_rank = self._lb_scroll + ri   # 0-based
                ry = TABLE_TOP + 24 + ri * ROW_H
                is_selected = (content_focused and ri == self._lb_selected_row)
                is_confirm = (self._lb_confirm_delete == ri)

                # 선택된 행 배경 하이라이트
                if is_confirm:
                    row_bg = pygame.Surface((w - 32, ROW_H), pygame.SRCALPHA)
                    row_bg.fill((180, 40, 40, 120))
                    self._display.blit(row_bg, (16, ry - 2))
                elif is_selected:
                    row_bg = pygame.Surface((w - 32, ROW_H), pygame.SRCALPHA)
                    row_bg.fill((60, 40, 120, 140))
                    self._display.blit(row_bg, (16, ry - 2))

                if is_confirm:
                    row_col = (255, 100, 100)
                elif ri == 0:
                    row_col = (255, 220, 50)
                elif is_selected:
                    row_col = (180, 220, 255)
                else:
                    row_col = (200, 200, 220)
                row_col = (255, 220, 50) if abs_rank == 0 else (200, 200, 220)
                player = entry.get("player", "") or "-"
                vals = [
                    f"#{abs_rank+1}",
                    entry.get("title", entry.get("song_id", "?"))[:10],
                    player[:10],
                    str(entry.get("score", 0)),
                    entry.get("grade", "-"),
                    self._fmt_lb_date(entry.get("date", "")),
                ]
                for vi, (val, vx) in enumerate(zip(vals, header_xs)):
                    # 확인 대기 중이면 DATE 열 대신 "DEL?" 표시
                    if is_confirm and vi == 5:  # DATE 열
                        ds = self._fonts["small_retro"].render("DEL?", True, (255, 80, 80))
                        self._display.blit(ds, (vx, ry))
                    else:
                        vs = self._fonts["small_retro"].render(val, True, row_col)
                        self._display.blit(vs, (vx, ry))

        # ── 하단 버튼: DELETE ALL / BACK ──────────────────────────
            # 스크롤 인디케이터 (우측 사이드바)
            if total_entries > max_rows:
                bar_x = w - 10
                bar_top = TABLE_TOP + 24
                bar_bot = TABLE_TOP + 24 + max_rows * ROW_H
                bar_h = bar_bot - bar_top
                pygame.draw.line(self._display, (60, 50, 100), (bar_x, bar_top), (bar_x, bar_bot), 2)
                # 썸 위치
                thumb_h = max(16, bar_h * max_rows // max(total_entries, 1))
                thumb_y = bar_top + (bar_h - thumb_h) * self._lb_scroll // max(max_scroll, 1)
                pygame.draw.rect(self._display, (160, 100, 255),
                                 (bar_x - 3, thumb_y, 6, thumb_h), border_radius=3)
                # ▲▼ 힌트
                if self._lb_scroll > 0:
                    up_s = self._fonts["small_retro"].render("▲", True, (160, 140, 220))
                    self._display.blit(up_s, up_s.get_rect(midright=(w - 14, bar_top - 6)))
                if self._lb_scroll < max_scroll:
                    dn_s = self._fonts["small_retro"].render("▼", True, (160, 140, 220))
                    self._display.blit(dn_s, dn_s.get_rect(midright=(w - 14, bar_bot + 8)))
                # 페이지 카운터
                pg_s = self._fonts["small_retro"].render(
                    f"{self._lb_scroll+1}-{min(self._lb_scroll+max_rows, total_entries)}/{total_entries}",
                    True, (120, 110, 160))
                self._display.blit(pg_s, pg_s.get_rect(midright=(w - 16, bar_bot + 22)))

        # BACK 버튼 — _generic_focus_idx==1 일 때만 강조
        mouse_pos = pygame.mouse.get_pos()
        DEL_BTN_W = 160
        btn_gap = 20
        total_btn_w = DEL_BTN_W + btn_gap + BTN_W
        btn_start_x = w // 2 - total_btn_w // 2
        btn_y = h - MARGIN_BOTTOM - BTN_H + 6

        # DELETE ALL 버튼 — _generic_focus_idx==1
        del_all_rect = pygame.Rect(btn_start_x, btn_y, DEL_BTN_W, BTN_H)
        self._btn_rects["btn_lb_delete_all"] = del_all_rect
        del_hover = del_all_rect.collidepoint(mouse_pos)
        del_focused = (self._generic_focus_idx == 1)
        del_confirming = (self._lb_confirm_delete == "all")
        if del_confirming:
            del_bg = (180, 30, 30)
            del_border = self._neon_color((255, 80, 80), tick)
        elif del_focused:
            del_bg = (120, 30, 60)
            del_border = (255, 100, 100)
        else:
            del_bg = (60, 20, 35)
            del_border = (130, 60, 80)
        pygame.draw.rect(self._display, del_bg, del_all_rect, border_radius=12)
        if del_focused or del_hover or del_confirming:
            neon_col = self._neon_color((255, 80, 80), tick) if del_confirming else (255, 100, 100)
            self._draw_neon_rect(self._display, del_all_rect, neon_col,
                                 width=3, radius=12, glow_radius=8)
            bracket_col = self._neon_color((255, 80, 80), tick * 2) if del_confirming else self._neon_color((255, 120, 120), tick * 2)
            self._draw_corner_brackets(self._display, del_all_rect,
                                       bracket_col, size=12, width=3)
        else:
            pygame.draw.rect(self._display, del_border, del_all_rect, 2, border_radius=12)
        del_lbl_text = "[DELETE ALL?]" if del_confirming else "DELETE ALL"
        del_lbl_col = (255, 255, 255) if (del_focused or del_hover or del_confirming) else (180, 120, 140)
        del_lbl = self._fonts["btn_retro"].render(del_lbl_text, True, del_lbl_col)
        self._display.blit(del_lbl, del_lbl.get_rect(center=del_all_rect.center))

        # BACK 버튼 — _generic_focus_idx==2
        back_rect = pygame.Rect(btn_start_x + DEL_BTN_W + btn_gap, btn_y, BTN_W, BTN_H)
        self._btn_rects["btn_lb_back"] = back_rect
        hover    = back_rect.collidepoint(mouse_pos)
        focused  = (self._generic_focus_idx == 2)   # BACK이 index2
        # 배경
        bg_col = (120, 50, 180) if focused else (50, 30, 80)
        pygame.draw.rect(self._display, bg_col, back_rect, border_radius=12)
        # 테두리
        if focused or hover:
            self._draw_neon_rect(self._display, back_rect, (255, 255, 100),
                                 width=3, radius=12, glow_radius=8)
            self._draw_corner_brackets(self._display, back_rect,
                                       self._neon_color((255, 220, 80), tick * 2), size=12, width=3)
        else:
            pygame.draw.rect(self._display, (90, 70, 130), back_rect, width=2, border_radius=12)
        lbl_col = (255, 255, 255) if (focused or hover) else (170, 150, 200)
        lbl = self._fonts["btn_retro"].render("BACK", True, lbl_col)
        self._display.blit(lbl, lbl.get_rect(center=back_rect.center))

        hint = self._fonts["small_retro"].render(
            "←/→: TAB  ↑/↓: SELECT  ENTER: DELETE/CONFIRM  ESC: BACK", True, (120, 110, 160))
        self._display.blit(hint, hint.get_rect(center=(w//2, h - 22)))

    def _render_settings(self, w, h):
        """Render settings screen — 볼륨 슬라이더 + 설정 정보."""

        tick = self._neon_tick
        for y_i in range(h):
            t = y_i / h
            pygame.draw.line(self._display,
                             (int(10+10*t), int(6+6*t), int(38+22*t)),
                             (0, y_i), (w, y_i))

        MARGIN_TOP    = 30
        MARGIN_BOTTOM = 46
        BTN_H         = 46
        FOOTER_H      = 24

        # 타이틀
        title_col = self._neon_color((160, 140, 255), tick)
        title = self._fonts["result_big"].render("SETTINGS", True, title_col)
        glow  = self._fonts["result_big"].render("SETTINGS", True, (40, 30, 80))
        for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
            self._display.blit(glow, glow.get_rect(center=(w//2+dx, MARGIN_TOP+28+dy)))
        self._display.blit(title, title.get_rect(center=(w//2, MARGIN_TOP+28)))
        line_col = self._neon_color((120, 100, 220), tick, 0.7)
        pygame.draw.line(self._display, line_col,
                         (w//4, MARGIN_TOP+50), (w*3//4, MARGIN_TOP+50), 1)

        btn_y        = h - MARGIN_BOTTOM - FOOTER_H - BTN_H - 10
        content_top  = MARGIN_TOP + 60
        content_bot  = btn_y - 16
        mouse_pos    = pygame.mouse.get_pos()
        CX           = w // 2

        # ── 읽기 전용 설정값 표시 ────────────────────────────
        info_items = [
            ("CAMERA",      str(self.config.get("camera", {}).get("device_id", 0))),
            ("RESOLUTION",  f"{self.config.get('camera',{}).get('width',640)}x"
                            f"{self.config.get('camera',{}).get('height',480)}"),
            ("SCORE METHOD", str(self.config.get("scoring", {}).get("similarity_metric", "cosine"))),
            ("TARGET FPS",  str(self.TARGET_FPS)),
        ]
        num_info = len(info_items)
        row_h    = min(34, (content_bot - content_top - 80) // max(num_info, 1))
        y_info   = content_top

        for lbl_txt, val_txt in info_items:
            lbl_s = self._fonts["small_retro"].render(lbl_txt, True, (130, 125, 180))
            val_s = self._fonts["small_retro"].render(val_txt, True, (200, 220, 255))
            self._display.blit(lbl_s, (CX - 220, y_info))
            self._display.blit(val_s, (CX + 20,  y_info))
            y_info += row_h

        # ── 볼륨 슬라이더 ────────────────────────────────────
        vol_y       = y_info + 14
        VOL_LBL_W   = 140
        SLIDER_W    = min(260, w - VOL_LBL_W - 120)
        SLIDER_H    = 12
        ARROW_BTN_W = 36
        ARROW_BTN_H = 36
        slider_x    = CX - SLIDER_W // 2
        slider_y    = vol_y + 36   # 레이블 아래에 슬라이더 배치

        vol_lbl = self._fonts["small_retro"].render("BGM VOLUME", True,
                                                     self._neon_color((180, 160, 255), tick))
        self._display.blit(vol_lbl, vol_lbl.get_rect(center=(CX, vol_y + 14)))

        # 슬라이더 트랙
        pygame.draw.rect(self._display, (50, 45, 90),
                         pygame.Rect(slider_x, slider_y, SLIDER_W, SLIDER_H), border_radius=6)
        fill_w = int(SLIDER_W * self._bgm_volume)
        if fill_w > 0:
            pygame.draw.rect(self._display,
                             self._neon_color((120, 100, 255), tick),
                             pygame.Rect(slider_x, slider_y, fill_w, SLIDER_H), border_radius=6)
        # 슬라이더 핸들
        handle_x = slider_x + fill_w
        pygame.draw.circle(self._display, (200, 180, 255), (handle_x, slider_y + SLIDER_H//2), 9)
        pygame.draw.circle(self._display, (255, 255, 255), (handle_x, slider_y + SLIDER_H//2), 5)

        # 퍼센트 표시
        pct_s = self._fonts["small_retro"].render(f"{int(self._bgm_volume*100)}%", True, (220, 220, 255))
        self._display.blit(pct_s, pct_s.get_rect(midleft=(slider_x + SLIDER_W + 12, slider_y + SLIDER_H//2)))

        # ◀ 버튼
        vd_rect = pygame.Rect(slider_x - ARROW_BTN_W - 52, slider_y - (ARROW_BTN_H - SLIDER_H)//2,
                              ARROW_BTN_W, ARROW_BTN_H)
        self._btn_rects["btn_settings_vol_down"] = vd_rect
        foc_vd  = (getattr(self, '_generic_focus_idx', 0) == 0)
        hov_vd  = vd_rect.collidepoint(mouse_pos)
        act_vd  = foc_vd or hov_vd
        pygame.draw.rect(self._display, (80,50,140) if act_vd else (50,35,90), vd_rect, border_radius=8)
        pygame.draw.rect(self._display, (255,255,100) if foc_vd else (160,140,220), vd_rect, 2 if foc_vd else 1, border_radius=8)
        self._display.blit(self._fonts["btn_retro"].render("◀", True, (255,255,255)),
                           self._fonts["btn_retro"].render("◀", True,(0,0,0)).get_rect(center=vd_rect.center).move(vd_rect.topleft[0]-
                           self._fonts["btn_retro"].render("◀",True,(0,0,0)).get_rect().x, 0))
        lv = self._fonts["btn_retro"].render("-", True, (255,255,255))
        self._display.blit(lv, lv.get_rect(center=vd_rect.center))

        # ▶ 버튼
        vu_rect = pygame.Rect(slider_x + SLIDER_W + 52, slider_y - (ARROW_BTN_H - SLIDER_H)//2,
                              ARROW_BTN_W, ARROW_BTN_H)
        self._btn_rects["btn_settings_vol_up"] = vu_rect
        foc_vu  = (getattr(self, '_generic_focus_idx', 0) == 1)
        hov_vu  = vu_rect.collidepoint(mouse_pos)
        act_vu  = foc_vu or hov_vu
        pygame.draw.rect(self._display, (80,50,140) if act_vu else (50,35,90), vu_rect, border_radius=8)
        pygame.draw.rect(self._display, (255,255,100) if foc_vu else (160,140,220), vu_rect, 2 if foc_vu else 1, border_radius=8)
        lv2 = self._fonts["btn_retro"].render("+", True, (255,255,255))
        self._display.blit(lv2, lv2.get_rect(center=vu_rect.center))

        # ── BACK 버튼 ────────────────────────────────────────
        back_rect = pygame.Rect(w//2 - 110, btn_y, 220, BTN_H)
        self._btn_rects["btn_back"] = back_rect
        foc_bk  = (getattr(self, '_generic_focus_idx', 0) == 2)
        hov_bk  = back_rect.collidepoint(mouse_pos)
        pygame.draw.rect(self._display, (70,70,150) if (foc_bk or hov_bk) else (50,50,110),
                         back_rect, border_radius=14)
        pygame.draw.rect(self._display, (255,255,100) if foc_bk else (180,180,230),
                         back_rect, 3 if foc_bk else 2, border_radius=14)
        lbl = self._fonts["btn_retro"].render("< BACK", True, (255,255,255))
        self._display.blit(lbl, lbl.get_rect(center=back_rect.center))

        hint = self._fonts["small_retro"].render(
            "←/→: VOL  U/D: SELECT  ENTER: OK  ESC: BACK", True, (100,100,130))
        self._display.blit(hint, hint.get_rect(center=(w//2, h - MARGIN_BOTTOM + 10)))

    # ── 멀티플레이 Discovery 콜백 (백그라운드 스레드에서 호출됨) ──────────────

    def _on_multi_found(self, role: str, opponent_ip: str):
        """상대방 탐색 성공 — 백그라운드 스레드에서 호출."""
        self._multi_role = role
        self._multi_opponent_ip = opponent_ip
        self._multi_discovery = None

        # GameSocket 생성 + 시작
        from network.game_socket import GameSocket
        sock = GameSocket(opponent_ip)
        sock.on_disconnect = self._on_multi_disconnect
        sock.on_opponent_finish = lambda: None  # 결과 화면에서 처리
        # 역할 무관하게 상대방 포즈 준비 알림 콜백 등록
        sock.on_opponent_pose_ready = self._on_multi_opponent_pose_ready
        if role == "client":
            # CLIENT: HOST의 곡 선택 수신 + 시작 신호 수신
            sock.on_song_select = self._on_multi_song_received
            sock.on_game_start  = self._on_multi_game_start
        sock.start()
        self._multi_socket = sock

        self._multi_found = True
        self._multi_connected = True
        self._multi_status_msg = f"연결됨! ({role.upper()}) — {opponent_ip}"
        print(f"[MULTI] 상대방 발견: role={role} ip={opponent_ip}", flush=True)

        # HOST: WAITING 화면에서 모드 선택 UI를 표시 (메인 스레드에서 _multi_connected 감지)
        # CLIENT: WAITING 화면에서 HOST의 곡 선택 대기

    def _on_multi_timeout(self):
        """탐색 시간 초과 — 백그라운드 스레드에서 호출."""
        self._multi_timed_out = True
        self._multi_status_msg = "탐색 시간 초과. ESC로 돌아가세요."
        print("[MULTI] 탐색 시간 초과", flush=True)

    def _on_multi_status(self, msg: str):
        """탐색 상태 메시지 업데이트."""
        self._multi_status_msg = msg

    def _on_multi_disconnect(self):
        """게임 중 연결 끊김."""
        print("[MULTI] 상대방 연결 끊김", flush=True)

    def _on_multi_song_received(self, song_id: str, mode: str = "practice"):
        """CLIENT: HOST가 선택한 곡+모드를 수신 — 백그라운드 스레드에서 호출."""
        self._current_mode = mode
        self._multi_mode_selected = True
        for song in self._songs:
            if song.get("id") == song_id:
                self._current_song = song
                break
        self._multi_found = True  # WAITING 화면 루프에서 진행 트리거
        print(f"[MULTI] HOST 곡/모드 수신: song={song_id} mode={mode}", flush=True)

    def _on_multi_game_start(self):
        """CLIENT: HOST의 카운트다운 시작 신호 수신 — 백그라운드 스레드에서 호출."""
        self._multi_game_start_received = True
        print("[MULTI] 시작 신호 수신 → COUNTDOWN", flush=True)

    def _on_multi_opponent_pose_ready(self):
        """상대방 포즈 감지 3초 완료 알림 수신 — 백그라운드 스레드에서 호출."""
        self._multi_opponent_pose_ready = True
        print(f"[MULTI] 상대방 포즈 준비 완료 ({self._multi_role})", flush=True)

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
            self._close_score_trace_log()
            self._menu_focus_idx = 0
            # 게임 중 메뉴로 돌아오면 음악 정지
            if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            self._release_reference_assets()
            # 멀티플레이 정리
            if self._multi_socket:
                self._multi_socket.stop()
                self._multi_socket = None
            if self._multi_discovery:
                self._multi_discovery.stop()
                self._multi_discovery = None
            self._is_multi_mode = False
            self._multi_role = ""
            self._multi_opponent_ip = ""
            self._multi_my_pose_ready = False
            self._multi_opponent_pose_ready = False
            self._multi_game_start_received = False
        elif state == GameState.SONG_SELECT:
            self._close_score_trace_log()
            self._selected_song_idx = 0
            self._song_focus_idx = 0
            self._song_depth = 1            # 곡 선택 창 진입 시 항상 depth1로 초기화
            self._release_reference_assets()
        elif state == GameState.READY:
            self._stop_preview()  # 미리듣기 중지
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
                pygame.mixer.music.unpause()
            else:
                if self._current_mode != "freestyle":
                    self._open_score_trace_log()
                # 새 게임 시작: 이전 세션 완전 정리
                if self._current_session is not None:
                    self._current_session.is_active = False
                    self._current_session = None
                pygame.mixer.music.stop()
                self._scorer.reset()
                self._scoring_frame_counter = 0
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

                # ── 모드별 판정 오차 조정 ──────────────────────
                base_tolerance = self.config.get("tolerance_delay", 1.0)
                if self._current_mode == "practice":
                    self._tolerance_delay = base_tolerance * 1   # 관대한 판정
                elif self._current_mode == "challenge":
                    self._tolerance_delay = base_tolerance * 1   # 엄격한 판정
                else:  # freestyle
                    self._tolerance_delay = base_tolerance

                # 챌린지 MISS 카운터 초기화
                self._consecutive_miss = 0
                self._challenge_game_over = False
                
                # 재생 바를 0초로 돌리기
                self._ref_frame_landmarks = None
                self._ref_video_frame = None
                if getattr(self, '_async_video_player', None) is not None:
                    self._async_video_player.reset_position()
                    self._async_video_player.start()
                    
                if getattr(self, '_audio_path', None) and os.path.exists(self._audio_path):
                    try:
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

                # 멀티플레이: 소켓이 이미 있으면 상대 점수 리셋
                if self._is_multi_mode and self._multi_socket:
                    self._multi_socket.opponent_score = 0
                    self._multi_socket.opponent_combo = 0
                    self._multi_socket.opponent_grade = ""
                    self._multi_socket.opponent_finished = False

        elif state == GameState.PAUSED:
            # 일시정지 — 현재 세션 타이머는 계속 흐름 (추후 개선 가능)
            pygame.mixer.music.pause()
        elif state == GameState.RESULT:
            self._close_score_trace_log()
            self._result_data = self._scorer.get_final_result()
            # 멀티플레이: 최종 점수 전송
            if self._is_multi_mode and self._multi_socket:
                self._multi_socket.send_end(
                    int(self._result_data.get("total_score", 0))
                )
            # 프리스타일은 저장 안 함 / 그 외는 이름 입력 오버레이 표시
            if self._current_mode != "freestyle":
                self._name_input_active = True
                self._stdin_text_mode = True
                self._name_input_text = self._player_name  # 이전 이름 미리 채움
                print(f"\n[NAME] Enter your name + Enter to save  |  Empty + Enter = save without name  |  ESC = save with previous name ({self._player_name or 'none'}): ",
                      end="", flush=True)
            else:
                self._name_input_active = False
                self._stdin_text_mode = False
        elif state == GameState.LEADERBOARD:
            self._leaderboard_load()
            self._generic_focus_idx = 0
            self._lb_selected_row = 0
            self._lb_confirm_delete = None
        elif state == GameState.WAITING:
            self._generic_focus_idx = 0   # 모드 선택 첫 항목에 포커스
            self._multi_found = False
            self._multi_timed_out = False
            self._multi_status_msg = ""
            # 모드가 이미 선택된 경우에만 Discovery 시작 (미선택이면 모드 선택 화면 표시)
            if self._multi_mode_selected:
                from network.discovery import Discovery
                self._multi_discovery = Discovery()
                self._multi_discovery.find_opponent(
                    on_found=self._on_multi_found,
                    on_timeout=self._on_multi_timeout,
                    on_status=self._on_multi_status,
                )

    def _play_song_preview(self, song: dict):
        """곡 선택 시 오디오 미리듣기."""
        import subprocess
        if not pygame.mixer.get_init():
            return

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        audio_path = None

        video_rel = song.get("video", "")
        if video_rel:
            # video가 있는 경우 (practice / challenge 모드)
            video_path = os.path.join(project_root, video_rel)
            if not os.path.exists(video_path):
                return
            audio_path = video_path.rsplit(".", 1)[0] + ".ogg"
            if not os.path.exists(audio_path):
                subprocess.run(
                    ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "libvorbis", "-q:a", "4", audio_path],
                    capture_output=True,
                )
        else:
            # video 없이 audio만 있는 경우 (프리스타일 모드)
            audio_rel = song.get("audio", "")
            if not audio_rel:
                return
            audio_path = os.path.join(project_root, audio_rel)
            if not os.path.exists(audio_path):
                audio_path = os.path.join(os.getcwd(), audio_rel)

        if not audio_path or not os.path.exists(audio_path):
            return

        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.set_volume(self._bgm_volume * 0.6)  # 미리듣기는 약간 낮게
            pygame.mixer.music.play(start=5.0)   # 5초 지점부터 재생
        except Exception as e:
            print(f"[WARN] 미리듣기 실패: {e}")

    def _stop_preview(self):
        """미리듣기 중지."""
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except Exception:
            pass

    def _take_screenshot(self):
        """현재 화면을 screenshots/ 폴더에 저장합니다."""
        from datetime import datetime
        screenshots_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "screenshots"
        )
        os.makedirs(screenshots_dir, exist_ok=True)
        fname = datetime.now().strftime("screenshot_%Y%m%d_%H%M%S.png")
        path = os.path.join(screenshots_dir, fname)
        try:
            pygame.image.save(self._display, path)
            print(f"[INFO] 스크린샷 저장: {path}")
        except Exception as e:
            print(f"[WARN] 스크린샷 실패: {e}")

    def _leaderboard_load(self):
        """data/leaderboard.json 파일에서 리더보드를 로드한다."""
        lb_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "leaderboard.json")
        lb_path = os.path.normpath(lb_path)
        try:
            with open(lb_path, "r", encoding="utf-8") as f:
                self._leaderboard = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._leaderboard = []

    def _leaderboard_save_result(self, player: str = ""):
        """현재 게임 결과를 리더보드 파일에 추가로 저장한다."""
        from datetime import datetime
        if not self._result_data:
            return
        song = self._current_song or {}
        entry = {
            "song_id": song.get("id", "unknown"),
            "title":   song.get("title", song.get("id", "Unknown")),
            "player":  player,
            "score":   self._result_data.get("total_score", 0),
            "grade":   self._result_data.get("final_grade", "-"),
            "mode":    self._current_mode,
            "date":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        lb_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "leaderboard.json")
        lb_path = os.path.normpath(lb_path)
        try:
            with open(lb_path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            records = []
        records.append(entry)
        # 점수 높은 순으로 정렬 후 최대 200개 유지
        records = sorted(records, key=lambda e: e.get("score", 0), reverse=True)[:200]
        try:
            with open(lb_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] 리더보드 저장 실패: {e}")

    def _lb_get_filtered_entries(self) -> list:
        """현재 탭/필터에 맞는 리더보드 엔트리를 점수 내림차순으로 반환."""
        entries = list(self._leaderboard)
        if self._leaderboard_tab != "all":
            entries = [e for e in entries if e.get("mode", "practice") == self._leaderboard_tab]
        if self._leaderboard_filter:
            entries = [e for e in entries if e.get("song_id") == self._leaderboard_filter]
        return sorted(entries, key=lambda e: e.get("score", 0), reverse=True)

    def _lb_try_delete_selected(self):
        """콘텐츠 포커스에서 Enter/Space → 선택된 행 삭제 (확인 절차)."""
        entries = self._lb_get_filtered_entries()
        if not entries:
            return
        row = min(self._lb_selected_row, len(entries) - 1)
        if self._lb_confirm_delete == row:
            # 두 번째 누름 → 실제 삭제
            target = entries[row]
            try:
                self._leaderboard.remove(target)
            except ValueError:
                pass
            self._leaderboard_save_all()
            self._lb_confirm_delete = None
            if self._lb_selected_row >= len(self._lb_get_filtered_entries()):
                self._lb_selected_row = max(0, self._lb_selected_row - 1)
        else:
            self._lb_confirm_delete = row

    def _lb_try_delete_all(self):
        """DELETE ALL 버튼 → 전체 삭제 (확인 절차)."""
        if self._lb_confirm_delete == "all":
            # 두 번째 누름 → 실제 전체 삭제
            if self._leaderboard_tab == "all":
                self._leaderboard.clear()
            else:
                self._leaderboard = [e for e in self._leaderboard
                                     if e.get("mode", "practice") != self._leaderboard_tab]
            self._leaderboard_save_all()
            self._lb_confirm_delete = None
            self._lb_selected_row = 0
        else:
            self._lb_confirm_delete = "all"

    def _leaderboard_save_all(self):
        """현재 self._leaderboard 전체를 파일에 저장."""
        lb_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "leaderboard.json")
        lb_path = os.path.normpath(lb_path)
        records = sorted(self._leaderboard, key=lambda e: e.get("score", 0), reverse=True)[:200]
        try:
            with open(lb_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] 리더보드 저장 실패: {e}")

    def _release_reference_assets(self):
        """이전 곡의 리소스를 해제합니다."""
        self._ref_landmarks = None
        self._ref_landmarks_display = None
        self._ref_frame_landmarks = None
        if getattr(self, '_async_video_player', None) is not None:
            self._async_video_player.stop()
            self._async_video_player = None
        self._ref_video_surf = None
        self._ref_video_size = None
        self._audio_path = None

    def _load_reference_assets(self):
        """PLAYING 진입 시의 초기 렉을 없애기 위해 미리 무거운 리소스(영상, Numpy 배열 등)를 로드해둡니다."""
        if getattr(self, '_ref_landmarks', None) is not None:
            return  # 이미 로드됨
            
        import subprocess
        import numpy as np
        _cv2 = cv2

        self._ref_video_fps = 30.0

        if self._current_song and self._current_song.get("has_reference"):
            ref_path = os.path.join(self._current_song["path"], "reference.npy")
            try:
                ref_data = np.load(ref_path)
                if ref_data.ndim == 3 and ref_data.shape[2] == 3:
                    vis = np.ones((*ref_data.shape[:2], 1), dtype=np.float32)
                    ref_data = np.concatenate([ref_data, vis], axis=2)
                ref_data = ref_data.astype(np.float32)
                # 렌더링용은 원본 좌표 그대로 유지
                self._ref_landmarks_display = ref_data.copy()
                # 서비스는 cv2.flip(frame, 1) 후 MediaPipe를 실행하므로 (거울 모드),
                # reference.npy는 원본 영상(flip 없음)에서 추출됐기 때문에 x축이 반대.
                # 채점용은 x를 반전해서 유저 포즈와 좌표계를 일치시킨다.
                ref_data_scoring = ref_data.copy()
                ref_data_scoring[:, :, 0] = 1.0 - ref_data_scoring[:, :, 0]
                self._ref_landmarks = ref_data_scoring
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
                    _cv2 = cv2
                    cap = _cv2.VideoCapture(video_path)
                    
                    self._ref_video_fps = self._current_song.get(
                        "video_fps",
                        cap.get(_cv2.CAP_PROP_FPS) or 30.0,
                    )
                    
                    # 영상 리사이즈 크기 미리 계산 
                    ui_cfg = self.config.get("ui", {})
                    disp_w = ui_cfg.get("window_width", 1024)
                    disp_h = ui_cfg.get("window_height", 600)
                    HEADER_H = 55
                    FOOTER_H = 44
                    rp_w = disp_w - disp_w // 2
                    rp_h = disp_h - HEADER_H - FOOTER_H

                    total_frames = cap.get(_cv2.CAP_PROP_FRAME_COUNT)
                    vid_w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
                    vid_h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))

                    cap.release()

                    if vid_w > 0 and vid_h > 0:
                        vscale = min(rp_w / vid_w, rp_h / vid_h)
                        tw, th = int(vid_w * vscale), int(vid_h * vscale)
                        self._ref_video_size = (tw, th)
                        self._ref_video_pos = (
                            disp_w // 2 + (rp_w - tw) // 2,
                            HEADER_H + (rp_h - th) // 2,
                        )
                        
                        # [AsyncVideoPlayer 초기화 및 적용]
                        from utils.async_video import AsyncVideoPlayer
                        
                        def _get_audio_time():
                            if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                                pm = pygame.mixer.music.get_pos()
                                return (pm / 1000.0) if pm >= 0 else 0.0
                            return None
                            
                        audio_latency = self.config.get("audio", {}).get("latency_offset", 0.040)
                        self._async_video_player = AsyncVideoPlayer(
                            video_path=video_path,
                            target_size=self._ref_video_size,
                            audio_latency_offset=audio_latency,
                            time_func=_get_audio_time
                        )

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

            # 영상 없이 오디오만 있는 경우 (프리스타일 음악 전용)
            if not video_rel or not os.path.exists(video_path if video_rel else ""):
                audio_rel = self._current_song.get("audio", "")
                if audio_rel:
                    project_root = os.path.dirname(os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))))
                    audio_path = os.path.join(project_root, audio_rel)
                    if not os.path.exists(audio_path):
                        audio_path = os.path.join(os.getcwd(), audio_rel)
                    if os.path.exists(audio_path):
                        self._audio_path = audio_path
                        try:
                            pygame.mixer.music.load(audio_path)
                        except Exception:
                            pass

    def shutdown(self):
        """Clean up all resources. 중복 호출에도 안전합니다."""
        if getattr(self, '_shutdown_done', False):
            return
        self._shutdown_done = True
        self.running = False
        self._close_score_trace_log()
        if getattr(self, '_async_camera', None) is not None:
            self._async_camera.stop()
        if getattr(self, '_async_video_player', None):
            self._async_video_player.stop()
            self._async_video_player = None
        if getattr(self, '_camera', None) is not None:
            try:
                self._camera.release()
            except Exception:
                pass
        try:
            pygame.quit()
        except Exception:
            pass
        # pygame/SDL이 터미널 설정을 망가뜨릴 수 있으므로 복원
        try:
            if sys.stdin.isatty():
                subprocess.run(["stty", "sane"], stdin=sys.stdin, timeout=2)
        except Exception:
            pass
