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
        self._scratch_comparator = None
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

    def initialize(self):
        """
        Initialize all game subsystems.
        Called once at startup.
        """
        import pygame
        import cv2

        pygame.init()
        pygame.mixer.init()

        # Display setup
        ui_cfg = self.config.get("ui", {})
        w = ui_cfg.get("window_width", 1024)
        h = ui_cfg.get("window_height", 600)
        flags = pygame.FULLSCREEN if ui_cfg.get("fullscreen", False) else 0
        self._display = pygame.display.set_mode((w, h), flags)
        pygame.display.set_caption("Let's Dance!")
        self._clock = pygame.time.Clock()

        # Camera
        cam_cfg = self.config.get("camera", {})
        self._camera = cv2.VideoCapture(cam_cfg.get("device_id", 0))
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 640))
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 480))

        # Pose detector
        from pose.detector import PoseDetector
        pose_cfg = self.config.get("pose", {})
        backend = self.config.get("pose_backend", "mediapipe")
        self._pose_detector = PoseDetector(
            backend=backend,
            model_complexity=pose_cfg.get("model_complexity", 1),
            min_detection_confidence=pose_cfg.get("min_detection_confidence", 0.5),
            min_tracking_confidence=pose_cfg.get("min_tracking_confidence", 0.5),
        )
        self._pose_detector.initialize()

        # Scorer & Feedback
        from scoring.scorer import DanceScorer
        from scoring.feedback import FeedbackGenerator
        self._scorer = DanceScorer(
            **{k: v for k, v in self.config.get("scoring", {}).items()
               if k in ("score_scale", "combo_multiplier", "grade_thresholds")}
        )
        self._feedback_gen = FeedbackGenerator()

        # Pose similarity comparator
        self._score_method = self.config.get("score_method", "direct")
        if self._score_method == "direct":
            from direct_compare.pose_similarity import PoseSimilarity
            self._pose_comparator = PoseSimilarity(use_key_joints_only=True, normalize=True)
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
            self._embedding_extractor = None
            self._similarity_method = "scratch"
            self._tolerance_delay = self.config.get("tolerance_delay", 1.0)
            print(f"[INFO] Score method: scratch ({model_name}: {model_path}, tolerance {self._tolerance_delay}s)")
        else:
            # embedding 모드: ST-GCN 임베딩 + SimilarityCalculator
            from scoring.similarity import SimilarityCalculator
            self._similarity_calc = SimilarityCalculator(metric="sliding_window", window_size=15)
            self._pose_comparator = None
            # TODO: EmbeddingExtractor 초기화 (모델 학습 완료 후)
            self._embedding_extractor = None
            print(f"[INFO] Score method: embedding (ST-GCN + sliding window cosine)")

        # ── 폰트 로드 (한국어 지원: NotoSansCJK → fallback SysFont) ──
        self._fonts = self._load_fonts(pygame)

        # ── 댄스 곡 목록 로드 ──
        self._songs = self._load_songs()

        # ── 실루엣 렌더러는 사용하지 않음 (스틱 피겨로 대체 — 성능 최적화) ──
        # 사용하지 않지만 호환성을 위해 None 유지
        self._user_silhouette = None
        self._guide_silhouette = None

        self.running = True

    @staticmethod
    def _load_fonts(pygame):
        """한국어 지원 폰트를 로드합니다.
        우선순위: ① 프로젝트 번들 폰트 → ② 시스템 경로 → ③ SysFont fallback
        번들 폰트(assets/fonts/NotoSansKR.ttf)를 git에 포함시켜
        어떤 OS/환경에서도 한글이 깨지지 않도록 합니다.
        """
        import os

        # ① 프로젝트 번들 폰트 (engine.py → src/game → src → project/assets/fonts)
        try:
            engine_dir   = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(engine_dir))
            bundle_font  = os.path.join(project_root, "assets", "fonts", "NotoSansKR.ttf")
        except Exception:
            bundle_font = ""

        # ② 시스템 경로 후보 (라즈베리파이 / 우분투 / macOS / Windows)
        system_candidates = [
            # Linux (Raspberry Pi, Ubuntu)
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            # macOS
            "/Library/Fonts/NotoSansKR-Regular.otf",
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
            # Windows
            "C:/Windows/Fonts/malgun.ttf",       # 맑은 고딕
            "C:/Windows/Fonts/gulim.ttc",         # 굴림
        ]

        font_path = None
        # 번들 폰트 우선
        if os.path.exists(bundle_font):
            font_path = bundle_font
            print(f"[FONT] 번들 폰트 사용: {bundle_font}")
        else:
            for p in system_candidates:
                if os.path.exists(p):
                    font_path = p
                    print(f"[FONT] 시스템 폰트 사용: {p}")
                    break

        if font_path is None:
            print("[FONT] 한글 폰트를 찾지 못했습니다. 한글이 깨질 수 있습니다.")

        def make(size, bold=False):
            if font_path:
                try:
                    return pygame.font.Font(font_path, size)
                except Exception as e:
                    print(f"[FONT] 폰트 로드 실패 ({font_path}): {e}")
            # ③ SysFont fallback
            for name in ["notosanscjkkr", "notosanscjk", "malgun gothic",
                         "applegothic", "nanum gothic", "sans"]:
                try:
                    f = pygame.font.SysFont(name, size, bold=bold)
                    if f:
                        return f
                except Exception:
                    pass
            return pygame.font.SysFont(None, size)

        return {
            "title":     make(50, bold=True),
            "menu":      make(34),
            "score":     make(46, bold=True),
            "body":      make(24),
            "feedback":  make(68, bold=True),
            "countdown": make(110, bold=True),
            "small":     make(19),
        }

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
                songs.append(meta)
            except Exception as e:
                print(f"[WARN] 곡 로드 실패 {song_dir}: {e}")

        print(f"[INFO] {len(songs)}곡 로드됨 (base: {base})")
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

        # 메뉴 버튼 수 (MENU: 5개 = 연습/도전/자유/설정/종료, SONG_SELECT: 곡수)
        MENU_BTN_COUNT = 5

        for event in pygame.event.get():
            # ── 창 닫기 ──────────────────────────────────────
            if event.type == pygame.QUIT:
                self.running = False
                return

            # ── 마우스 이동 → 포커스 이동 ───────────────────
            elif event.type == pygame.MOUSEMOTION:
                if self.state == GameState.MENU:
                    menu_keys = ["btn_practice", "btn_challenge", "btn_freestyle",
                                 "btn_settings", "btn_quit"]
                    for i, k in enumerate(menu_keys):
                        if k in self._btn_rects and self._btn_rects[k].collidepoint(event.pos):
                            self._menu_focus_idx = i
                            break
                elif self.state == GameState.SONG_SELECT:
                    songs = self._songs_for_mode(self._current_mode)
                    for i in range(len(songs)):
                        k = f"btn_song_{i}"
                        if k in self._btn_rects and self._btn_rects[k].collidepoint(event.pos):
                            self._song_focus_idx = i
                            self._selected_song_idx = i
                            break

            # ── 키보드 ────────────────────────────────────────
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.state in (GameState.PLAYING, GameState.PAUSED,
                                      GameState.SETTINGS, GameState.RESULT,
                                      GameState.COUNTDOWN, GameState.READY):
                        self.transition_to(GameState.MENU)
                    else:
                        self.running = False

                # ── 방향키 네비게이션 ──────────────────────────
                elif event.key in (pygame.K_DOWN, pygame.K_RIGHT):
                    if self.state == GameState.MENU:
                        self._menu_focus_idx = (self._menu_focus_idx + 1) % MENU_BTN_COUNT
                    elif self.state == GameState.SONG_SELECT:
                        songs = self._songs_for_mode(self._current_mode)
                        if songs:
                            self._song_focus_idx = (self._song_focus_idx + 1) % len(songs)
                            self._selected_song_idx = self._song_focus_idx

                elif event.key in (pygame.K_UP, pygame.K_LEFT):
                    if self.state == GameState.MENU:
                        self._menu_focus_idx = (self._menu_focus_idx - 1) % MENU_BTN_COUNT
                    elif self.state == GameState.SONG_SELECT:
                        songs = self._songs_for_mode(self._current_mode)
                        if songs:
                            self._song_focus_idx = (self._song_focus_idx - 1) % len(songs)
                            self._selected_song_idx = self._song_focus_idx

                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    if self.state == GameState.MENU:
                        # 포커스된 메뉴 항목 실행
                        menu_actions = [
                            "btn_practice", "btn_challenge", "btn_freestyle",
                            "btn_settings", "btn_quit"
                        ]
                        self._on_button_press(menu_actions[self._menu_focus_idx])
                    elif self.state == GameState.SONG_SELECT:
                        songs = self._songs_for_mode(self._current_mode)
                        if songs:
                            self._current_song = songs[self._song_focus_idx]
                        self.transition_to(GameState.READY)
                    elif self.state == GameState.PAUSED:
                        self.transition_to(GameState.PLAYING)
                    elif self.state == GameState.RESULT:
                        self.transition_to(GameState.MENU)

                elif event.key == pygame.K_p:
                    if self.state == GameState.PLAYING:
                        self.transition_to(GameState.PAUSED)
                    elif self.state == GameState.PAUSED:
                        self.transition_to(GameState.PLAYING)

                elif event.key == pygame.K_s:
                    if self.state == GameState.MENU:
                        self.transition_to(GameState.SETTINGS)
                elif event.key == pygame.K_q:
                    self.running = False

            # ── 마우스/터치 클릭 (MOUSEBUTTONDOWN = 터치 포함) ──
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_click(event.pos)

            # ── 핑거(멀티터치) 이벤트 – SDL2 터치스크린 ──────────
            elif event.type == pygame.FINGERDOWN:
                w, h = self._display.get_size()
                touch_pos = (int(event.x * w), int(event.y * h))
                self._handle_click(touch_pos)

    def _handle_click(self, pos: tuple):
        """
        터치/마우스 클릭 좌표를 받아 현재 화면의 버튼과 충돌 검사 후 처리.

        Args:
            pos: (x, y) 클릭/터치 좌표
        """
        import pygame

        for btn_name, rect in self._btn_rects.items():
            if rect.collidepoint(pos):
                self._on_button_press(btn_name)
                return

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
        elif btn_name == "btn_ready_cancel":
            self.transition_to(GameState.MENU)

    def _update(self):
        """Update game state based on current state."""
        # 네온 깜빡임 타이머 (항상 업데이트)
        self._neon_tick += 1.0 / self.TARGET_FPS

        if self.state == GameState.READY:
            self._update_ready()

        elif self.state == GameState.COUNTDOWN:
            elapsed = time.time() - self._countdown_start
            remaining = 2 - int(elapsed)
            if remaining < 0:
                self.transition_to(GameState.PLAYING)
            else:
                self._countdown_timer = remaining

            # 카운트다운 중 카메라+포즈 워밍업 (렉 방지)
            import cv2
            if self._camera is not None:
                ret, frame = self._camera.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    self._current_frame = frame
                    self._pose_detector.detect(frame)  # 모델 워밍업

        elif self.state == GameState.PLAYING:
            self._update_gameplay()

        # PAUSED 상태에서는 카메라/포즈 업데이트 중단

    def _update_ready(self):
        """READY 상태: 웹캠 + 포즈 감지. 발목까지 감지되면 3초 카운트다운 후 시작."""
        import cv2

        if self._camera is None:
            return

        ret, frame = self._camera.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        self._ready_current_frame = frame

        result = self._pose_detector.detect(frame)
        self._ready_landmarks = result["landmarks"]
        self._ready_pose_detected = result["detected"]

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

    def _update_gameplay(self):
        """Capture frame, detect pose, compute score."""
        import cv2
        import numpy as np

        if self._camera is None:
            return

        ret, frame = self._camera.read()
        if not ret:
            return

        # 좌우반전 (거울 모드 — 사용자가 자연스럽게 보이도록)
        frame = cv2.flip(frame, 1)
        self._current_frame = frame

        # Detect pose
        result = self._pose_detector.detect(frame)
        self._current_landmarks = result["landmarks"]
        self._pose_detected = result["detected"]

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
            video_fi = int(self._current_session.elapsed_time * self._ref_video_fps)
            total_video_frames = int(self._ref_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if video_fi < total_video_frames:
                current_pos = int(self._ref_video_cap.get(cv2.CAP_PROP_POS_FRAMES))

                # 영상fps > 게임fps이면 불필요한 프레임은 grab()으로 건너뜀
                frames_to_skip = video_fi - current_pos
                if frames_to_skip < 0 or frames_to_skip > 10:
                    # 너무 멀면 seek
                    self._ref_video_cap.set(cv2.CAP_PROP_POS_FRAMES, video_fi)
                    frames_to_skip = 0
                elif frames_to_skip > 1:
                    # 중간 프레임은 grab만 (디코딩 안 함 — retrieve보다 훨씬 빠름)
                    for _ in range(frames_to_skip - 1):
                        self._ref_video_cap.grab()

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

        # Score based on pose similarity
        if self._pose_detected:
            if self._ref_frame_landmarks is not None:
                if self._score_method == "direct":
                    # direct: 반응 딜레이 윈도우 내 최대 유사도
                    tolerance_frames = int(self._tolerance_delay * self.TARGET_FPS)
                    start_idx = max(0, self._ref_current_idx - tolerance_frames)
                    end_idx = self._ref_current_idx + 1  # 현재 프레임 포함

                    best_sims = []
                    for ri in range(start_idx, end_idx):
                        ref_lm = self._ref_landmarks[ri]
                        if self._similarity_method == "euclidean":
                            s = self._pose_comparator.euclidean_similarity(
                                self._current_landmarks, ref_lm)
                        elif self._similarity_method == "hybrid":
                            s = self._pose_comparator.hybrid_similarity(
                                self._current_landmarks, ref_lm)
                        elif self._similarity_method == "angle":
                            s = self._pose_comparator.angle_similarity(
                                self._current_landmarks, ref_lm)
                        else:
                            s = self._pose_comparator.cosine_similarity(
                                self._current_landmarks, ref_lm)
                        best_sims.append(s)
                    # 상위 3개 평균 (윈도우 내 순간 최대가 아닌 안정적 매칭)
                    best_sims.sort(reverse=True)
                    top_k = best_sims[:min(3, len(best_sims))]
                    sim = sum(top_k) / len(top_k)

                    # 디버그: 현재 프레임과만 비교한 값 vs 윈도우 최대값
                    if self._similarity_method == "angle":
                        sim_now = self._pose_comparator.angle_similarity(
                            self._current_landmarks, self._ref_frame_landmarks)
                    elif self._similarity_method == "euclidean":
                        sim_now = self._pose_comparator.euclidean_similarity(
                            self._current_landmarks, self._ref_frame_landmarks)
                    elif self._similarity_method == "hybrid":
                        sim_now = self._pose_comparator.hybrid_similarity(
                            self._current_landmarks, self._ref_frame_landmarks)
                    else:
                        sim_now = self._pose_comparator.cosine_similarity(
                            self._current_landmarks, self._ref_frame_landmarks)
                    print(f"\r[DBG] now={sim_now:.3f} top3={sim:.3f} max={best_sims[0]:.3f} win={end_idx-start_idx}f", end="")
                elif self._score_method == "scratch":
                    # scratch: compare recent user motion window with reference
                    # motion windows through a scratch-trained TFLite encoder.
                    tolerance_frames = int(self._tolerance_delay * self.TARGET_FPS)
                    sim = self._scratch_comparator.compute(
                        self._current_landmarks,
                        self._ref_landmarks,
                        self._ref_current_idx,
                        tolerance_frames=tolerance_frames,
                    )
                    if sim is None:
                        return
                else:
                    # embedding: ST-GCN 임베딩 비교 (TODO: 구현 후 연결)
                    # 현재는 fallback으로 detection confidence 사용
                    visibility = self._current_landmarks[:, 3]
                    mean_vis = float(np.mean(visibility[visibility > 0]))
                    sim = min(mean_vis, 1.0)
            else:
                if self._score_method == "scratch":
                    if not self._warned_scratch_no_ref:
                        print("[WARN] scratch scoring requires reference.npy; scoring paused.")
                        self._warned_scratch_no_ref = True
                    return
                # 레퍼런스 없으면 detection confidence로 대체
                visibility = self._current_landmarks[:, 3]
                mean_vis = float(np.mean(visibility[visibility > 0]))
                sim = min(mean_vis, 1.0)
            evaluation = self._scorer.evaluate(sim)
            fb = self._feedback_gen.generate(evaluation)
            if fb:
                self._last_feedback = fb
                self._feedback_timer = 1.2  # 1.2초 동안 표시
        else:
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
        sub = self._fonts["body"].render("✦  AI 댄스 채점 게임  ✦", True, sub_color)
        self._display.blit(sub, sub.get_rect(center=(w // 2, MARGIN_TOP + 82)))

        # 구분선
        line_col = self._neon_color((180, 80, 255), tick, 0.7)
        pygame.draw.line(self._display, line_col,
                         (w//2 - BTN_W//2, MARGIN_TOP + 100),
                         (w//2 + BTN_W//2, MARGIN_TOP + 100), 1)

        # ── 모드 버튼 ────────────────────────────────────────────
        mouse_pos = pygame.mouse.get_pos()
        btn_defs = [
            ("btn_practice",  "▶  연습 모드",   (0, 220, 180),   (0, 80, 60)),
            ("btn_challenge", "▶  도전 모드",   (255, 190, 0),   (90, 60, 0)),
            ("btn_freestyle", "▶  자유 모드",   (200, 100, 255), (70, 20, 100)),
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
            (btn_s_rect, "btn_settings", "⚙  설정",  (100, 120, 255), 3),
            (btn_q_rect, "btn_quit",     "✕  종료",  (255, 80,  80),  4),
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
            "↑↓ / 마우스: 선택   Enter/Space: 확인   ESC: 종료", True, footer_col
        )
        self._display.blit(footer, footer.get_rect(
            center=(w // 2, h - MARGIN_BOTTOM // 2 - 2)))

    def _render_song_select(self, w, h):
        """곡 선택 화면 — retro-fancy neon style."""
        import pygame
        import math

        tick = self._neon_tick

        MODE_LABELS = {"practice": "연습 모드", "challenge": "도전 모드", "freestyle": "자유 모드"}
        DIFF_STARS  = {0: "FREE", 1: "★☆☆", 2: "★★☆", 3: "★★★"}
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
        hdr_txt    = self._fonts["menu"].render(f"♪  {mode_label}  —  곡 선택", True, hdr_neon)
        self._display.blit(hdr_txt, hdr_txt.get_rect(midleft=(18, HEADER_H // 2)))

        songs     = self._songs_for_mode(self._current_mode)
        mouse_pos = pygame.mouse.get_pos()

        if not songs:
            msg = self._fonts["body"].render("이 모드에서 플레이 가능한 곡이 없습니다.", True, (255, 120, 120))
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
                    ("곡 제목",  sel.get("title", "-")),
                    ("아티스트", sel.get("artist", "-")),
                    ("BPM",     str(sel.get("bpm", 0))),
                    ("길이",    f"{sel.get('duration', 0)}s"),
                    ("난이도",  DIFF_STARS.get(sel.get("difficulty", 0), "-")),
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
                hover_s = start_rect.collidepoint(mouse_pos)

                btn_bg = pygame.Surface((start_rect.width, start_rect.height), pygame.SRCALPHA)
                btn_bg.fill((*[c // 3 for c in mode_col], 220 if hover_s else 160))
                self._display.blit(btn_bg, start_rect.topleft)
                self._draw_neon_rect(self._display, start_rect,
                                     self._neon_color(mode_col, tick),
                                     width=2, radius=14, glow_radius=10 if hover_s else 6)
                if hover_s:
                    self._draw_corner_brackets(self._display, start_rect,
                                               self._neon_color(mode_col, tick * 2),
                                               size=14, width=3)
                go_s = self._fonts["menu"].render("▶  시작하기", True, (255, 255, 255))
                self._display.blit(go_s, go_s.get_rect(center=start_rect.center))

        # ── 뒤로가기 버튼 ────────────────────────────────────────
        back_y    = h - BOTTOM_MARGIN - FOOTER_H + 6
        back_rect = pygame.Rect(14, back_y, 140, 38)
        self._btn_rects["btn_song_back"] = back_rect
        hover_b   = back_rect.collidepoint(mouse_pos)

        bb_surf = pygame.Surface((140, 38), pygame.SRCALPHA)
        bb_surf.fill((40, 25, 70, 200 if hover_b else 140))
        self._display.blit(bb_surf, back_rect.topleft)
        self._draw_neon_rect(self._display, back_rect,
                             self._neon_color((160, 140, 220), tick) if hover_b else (80, 65, 120),
                             width=2, radius=10, glow_radius=6 if hover_b else 2)
        back_s = self._fonts["body"].render("← 뒤로", True,
                                             (220, 215, 240) if hover_b else (160, 155, 185))
        self._display.blit(back_s, back_s.get_rect(center=back_rect.center))

        # ── 푸터 ─────────────────────────────────────────────────
        hint = self._fonts["small"].render(
            "↑↓: 이동   Enter: 시작   ESC: 뒤로", True, (130, 110, 170))
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
            f"{'  ' + song_title + '  —  ' if song_title else ''}포즈를 준비해주세요",
            True, (180, 180, 255))
        self._display.blit(hdr, hdr.get_rect(center=(w // 2, HEADER_H // 2)))

        body_h = h - HEADER_H - FOOTER_H

        # ── 왼쪽: 웹캠 피드 ──────────────────────────────────
        pygame.draw.rect(self._display, (12, 10, 30),
                         pygame.Rect(0, HEADER_H, MID_X, body_h))
        lbl_cam = self._fonts["small"].render("내 화면", True, (100, 160, 255))
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
            no_cam = self._fonts["body"].render("카메라 없음", True, (80, 80, 110))
            self._display.blit(no_cam, no_cam.get_rect(center=(MID_X // 2, HEADER_H + body_h // 2)))

        # ── 오른쪽: 스켈레톤 ─────────────────────────────────
        pygame.draw.rect(self._display, (10, 8, 28),
                         pygame.Rect(MID_X, HEADER_H, w - MID_X, body_h))
        lbl_sk = self._fonts["small"].render("내 스켈레톤", True, (255, 160, 80))
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
            no_pose = self._fonts["body"].render("포즈 감지 중...", True, (80, 80, 120))
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
            msg = "카메라 앞에 서 주세요!"
            msg_color = (200, 150, 80)
        elif not full_body_detected:
            msg = "몸 전체가 보일 때까지 뒤로 가주세요!"
            msg_color = (255, 160, 60)
        elif is_counting:
            remaining = max(0.0, ok_progress)
            msg = f"게임 시작까지  {remaining:.1f}초..."
            msg_color = (0, 240, 150)
            # 진행 바 (3초 → 0초, 채워지는 방향)
            pct = 1.0 - remaining / COUNTDOWN_SEC
            bar_w = int((w - 40) * pct)
            pygame.draw.rect(self._display, (0, 50, 30),
                             pygame.Rect(20, fy + FOOTER_H - 8, w - 40, 5), border_radius=3)
            pygame.draw.rect(self._display, (0, 240, 150),
                             pygame.Rect(20, fy + FOOTER_H - 8, bar_w, 5), border_radius=3)
        else:
            msg = "전신이 보이면 3초 후 자동으로 시작합니다!"
            msg_color = (100, 220, 255)

        msg_surf = self._fonts["body"].render(msg, True, msg_color)
        self._display.blit(msg_surf, msg_surf.get_rect(center=(w // 2, fy + FOOTER_H // 2 - 2)))

        # ── 취소 버튼 ──────────────────────────────────────────
        cancel_rect = pygame.Rect(w - 110, HEADER_H + 6, 100, 36)
        self._btn_rects["btn_ready_cancel"] = cancel_rect
        hover = cancel_rect.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self._display, (100, 35, 35) if hover else (70, 25, 25),
                         cancel_rect, border_radius=10)
        pygame.draw.rect(self._display, (200, 100, 100), cancel_rect, 2, border_radius=10)
        cancel_lbl = self._fonts["small"].render("취소", True, (255, 200, 200))
        self._display.blit(cancel_lbl, cancel_lbl.get_rect(center=cancel_rect.center))

    def _render_countdown(self, w, h):
        """Render countdown screen."""
        import pygame

        self._display.fill((10, 5, 30))

        count = max(self._countdown_timer, 0)

        txt_surface = self._fonts["countdown"].render(str(count + 1), True, (0, 255, 255))
        self._display.blit(txt_surface, txt_surface.get_rect(center=(w // 2, h // 2)))

        sub = self._fonts["body"].render("준비하세요!", True, (180, 180, 220))
        self._display.blit(sub, sub.get_rect(center=(w // 2, h // 2 + 100)))

        # 취소 버튼 (터치로 메뉴로 복귀)
        cancel_rect = pygame.Rect(w // 2 - 100, h - 80, 200, 48)
        self._btn_rects["btn_countdown_cancel"] = cancel_rect
        hover = cancel_rect.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self._display, (120, 40, 40) if hover else (80, 30, 30),
                         cancel_rect, border_radius=12)
        pygame.draw.rect(self._display, (200, 100, 100), cancel_rect, 2, border_radius=12)
        lbl = self._fonts["body"].render("취소", True, (255, 200, 200))
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
        """Render gameplay screen — dual character layout.

        Layout:
          ┌──────────────────────┬──────────────────────────────┐
          │   [헤더: 점수/콤보/버튼]                             │  50px
          ├──────────────────────┼──────────────────────────────┤
          │  내 캐릭터(스틱피겨)  │  가이드 캐릭터(참조 스틱피겨) │
          │  + 카메라 피드(작게)  │  + 피드백 이펙트 오버레이    │
          └──────────────────────┴──────────────────────────────┘
          [푸터: 조작 안내]                                       30px
        """
        import pygame
        import cv2
        import numpy as np

        HEADER_H = 55
        FOOTER_H = 30
        MID_X = w // 2
        BODY_Y = HEADER_H
        BODY_H = h - HEADER_H - FOOTER_H

        # ── 배경 ──────────────────────────────────────────────────
        self._display.fill((8, 6, 22))

        # 좌/우 패널 구분선
        pygame.draw.line(self._display, (50, 45, 90),
                         (MID_X, BODY_Y), (MID_X, BODY_Y + BODY_H), 2)

        # ── 패널 영역 정의 ─────────────────────────────────────────
        left_panel  = (0,     BODY_Y, MID_X,     BODY_H)   # (x, y, w, h)
        right_panel = (MID_X, BODY_Y, w - MID_X, BODY_H)

        # ══════════════════════════════════════════════════════
        #  LEFT — 내 캐릭터 + 카메라 피드
        # ══════════════════════════════════════════════════════

        # 좌 패널 배경
        pygame.draw.rect(self._display, (12, 10, 30),
                         pygame.Rect(left_panel[0], left_panel[1],
                                     left_panel[2], left_panel[3]))

        # 패널 레이블
        lbl_me = self._fonts["small"].render("나", True, (100, 160, 255))
        self._display.blit(lbl_me, (left_panel[0] + 12, left_panel[1] + 8))

        # 카메라 피드 크기 (좌 하단, 작게)
        CAM_W = min(200, MID_X - 20)
        CAM_H = int(CAM_W * 3 / 4)  # 4:3 비율
        cam_x = left_panel[0] + 10
        cam_y = BODY_Y + BODY_H - CAM_H - 10

        # 스틱 피겨 영역: 패널 전체에서 카메라 높이 제외
        fig_area_h = BODY_H - CAM_H - 20
        left_fig_rect = (left_panel[0], BODY_Y, left_panel[2], fig_area_h)

        if hasattr(self, '_pose_detected') and self._pose_detected and \
                self._current_landmarks is not None:
            # 스틱 피겨로 그리기 (실루엣보다 훨씬 가벼움)
            self._draw_stick_figure(
                self._display,
                self._current_landmarks,
                left_fig_rect,
                line_color=(80, 220, 255),
                joint_color=(200, 250, 255),
                line_width=4,
                joint_radius=6,
            )
        else:
            # 포즈 미감지 안내
            no_pose = self._fonts["body"].render("포즈 감지 중...", True, (80, 80, 120))
            fp_rect = pygame.Rect(*left_fig_rect)
            self._display.blit(no_pose, no_pose.get_rect(center=fp_rect.center))

        # 카메라 피드 (작게, 좌 하단)
        if hasattr(self, '_current_frame') and self._current_frame is not None:
            frame_small = cv2.resize(self._current_frame, (CAM_W, CAM_H),
                                     interpolation=cv2.INTER_NEAREST)

            # ── 카메라 피드 위에 스켈레톤 오버레이 그리기 ──
            if hasattr(self, '_pose_detected') and self._pose_detected and \
                    self._current_landmarks is not None:
                from pose.landmark_utils import SKELETON_CONNECTIONS, DANCE_JOINTS
                lm = self._current_landmarks
                cam_h_orig, cam_w_orig = self._current_frame.shape[:2]
                for src, dst in SKELETON_CONNECTIONS:
                    if src < len(lm) and dst < len(lm) and \
                       lm[src][3] > 0.3 and lm[dst][3] > 0.3:
                        x1 = int(lm[src][0] * CAM_W)
                        y1 = int(lm[src][1] * CAM_H)
                        x2 = int(lm[dst][0] * CAM_W)
                        y2 = int(lm[dst][1] * CAM_H)
                        cv2.line(frame_small, (x1, y1), (x2, y2), (0, 255, 200), 2)
                for idx in DANCE_JOINTS:
                    if idx < len(lm) and lm[idx][3] > 0.3:
                        cx = int(lm[idx][0] * CAM_W)
                        cy = int(lm[idx][1] * CAM_H)
                        cv2.circle(frame_small, (cx, cy), 3, (0, 255, 255), -1)

            # BGR→RGB + Surface 변환 (frombuffer가 swapaxes보다 빠름)
            frame_rgb = frame_small[:, :, ::-1]
            cam_surf = pygame.image.frombuffer(
                frame_rgb.tobytes(), (CAM_W, CAM_H), "RGB")
            # 카메라 피드 테두리
            pygame.draw.rect(self._display, (40, 40, 70),
                             pygame.Rect(cam_x - 2, cam_y - 2, CAM_W + 4, CAM_H + 4),
                             border_radius=6)
            self._display.blit(cam_surf, (cam_x, cam_y))
            # "카메라" 레이블
            cam_lbl = self._fonts["small"].render("카메라", True, (80, 120, 180))
            self._display.blit(cam_lbl, (cam_x + 4, cam_y - 20))
        else:
            # 카메라 없음 플레이스홀더
            pygame.draw.rect(self._display, (25, 25, 45),
                             pygame.Rect(cam_x, cam_y, CAM_W, CAM_H),
                             border_radius=6)
            no_cam = self._fonts["small"].render("카메라 없음", True, (80, 80, 110))
            self._display.blit(no_cam,
                               no_cam.get_rect(center=(cam_x + CAM_W // 2, cam_y + CAM_H // 2)))

        # ══════════════════════════════════════════════════════
        #  RIGHT — 가이드: 영상이 있으면 mp4만 재생, 없으면 실루엣
        # ══════════════════════════════════════════════════════

        # 우 패널 배경
        pygame.draw.rect(self._display, (10, 8, 28),
                         pygame.Rect(right_panel[0], right_panel[1],
                                     right_panel[2], right_panel[3]))

        # 패널 레이블
        lbl_guide = self._fonts["small"].render("가이드", True, (255, 160, 80))
        self._display.blit(lbl_guide, (right_panel[0] + 12, right_panel[1] + 8))

        has_video = self._ref_video_cap is not None

        if has_video and self._ref_video_surf is not None:
            # ── 영상이 있는 곡: mp4를 패널에 직접 표시 (실루엣 렌더링 불필요) ──
            vx, vy = self._ref_video_pos
            self._display.blit(self._ref_video_surf, (vx, vy))
        elif not has_video and self._ref_frame_landmarks is not None:
            # ── 영상이 없는 곡: 스틱 피겨 (실루엣보다 가벼움) ──
            right_fig_rect = (right_panel[0], BODY_Y, right_panel[2], BODY_H)
            self._draw_stick_figure(
                self._display,
                self._ref_frame_landmarks,
                right_fig_rect,
                line_color=(255, 170, 80),
                joint_color=(255, 220, 140),
                line_width=4,
                joint_radius=6,
            )
        else:
            # 가이드 없음 안내
            rp_rect = pygame.Rect(right_panel[0], BODY_Y, right_panel[2], BODY_H)
            no_guide = self._fonts["body"].render("가이드 없음", True, (80, 70, 60))
            self._display.blit(no_guide, no_guide.get_rect(center=rp_rect.center))

        # ── 피드백 이펙트 (가이드 캐릭터 위에 오버레이) ─────────────
        if self._last_feedback and self._feedback_timer > 0:
            fb = self._last_feedback
            alpha_ratio = min(self._feedback_timer / 0.4, 1.0)  # 마지막 0.4초에 페이드
            color = fb.get("color", (255, 255, 255))

            # 피드백 텍스트 (가이드 패널 상단 중앙)
            grade_surf = self._fonts["feedback"].render(fb["text"], True, color)
            gx = right_panel[0] + right_panel[2] // 2 - grade_surf.get_width() // 2
            gy = BODY_Y + int(BODY_H * 0.12)

            # 알파 페이드 적용
            if alpha_ratio < 1.0:
                grade_surf.set_alpha(int(255 * alpha_ratio))
            self._display.blit(grade_surf, (gx, gy))

            # 점수 표시 (피드백 아래)
            pts = fb.get("points", 0)
            if pts > 0:
                pts_surf = self._fonts["menu"].render(f"+{pts}", True, (255, 240, 100))
                if alpha_ratio < 1.0:
                    pts_surf.set_alpha(int(255 * alpha_ratio))
                px2 = right_panel[0] + right_panel[2] // 2 - pts_surf.get_width() // 2
                py2 = gy + grade_surf.get_height() + 4
                self._display.blit(pts_surf, (px2, py2))

            # 콤보 표시 (Perfect/Great 이상일 때)
            combo = fb.get("combo", 0)
            if combo >= 2:
                combo_surf = self._fonts["body"].render(f"{combo} COMBO!", True, (255, 220, 0))
                if alpha_ratio < 1.0:
                    combo_surf.set_alpha(int(200 * alpha_ratio))
                cx2 = right_panel[0] + right_panel[2] // 2 - combo_surf.get_width() // 2
                cy2 = gy - combo_surf.get_height() - 6
                self._display.blit(combo_surf, (cx2, cy2))

        # ══════════════════════════════════════════════════════
        #  HEADER — 점수 / 콤보 / 버튼
        # ══════════════════════════════════════════════════════
        header_bg = pygame.Rect(0, 0, w, HEADER_H)
        pygame.draw.rect(self._display, (18, 14, 45), header_bg)
        pygame.draw.line(self._display, (60, 50, 100), (0, HEADER_H), (w, HEADER_H), 1)

        score_surf = self._fonts["score"].render(
            f"{int(self._scorer.total_score)}", True, (0, 240, 200)
        )
        self._display.blit(score_surf, score_surf.get_rect(midleft=(16, HEADER_H // 2)))

        combo_surf = self._fonts["body"].render(
            f"★ {self._scorer.combo} 콤보", True, (255, 220, 0)
        )
        self._display.blit(combo_surf, combo_surf.get_rect(center=(w // 2, HEADER_H // 2)))

        # 진행 시간
        elapsed = self._current_session.elapsed_time if self._current_session else 0
        duration = (self._current_song or {}).get("duration", 60)
        remain = max(0, duration - elapsed)
        time_surf = self._fonts["body"].render(
            f"{int(remain // 60):02d}:{int(remain % 60):02d}", True, (180, 180, 220)
        )
        self._display.blit(time_surf, time_surf.get_rect(midright=(w - 230, HEADER_H // 2)))

        # 헤더 우측 버튼 (일시정지 / 메뉴)
        mouse_pos = pygame.mouse.get_pos()
        pause_rect = pygame.Rect(w - 220, 8, 90, 38)
        menu_rect  = pygame.Rect(w - 120, 8, 90, 38)
        self._btn_rects["btn_pause"]         = pause_rect
        self._btn_rects["btn_gameplay_menu"] = menu_rect

        for rect, label, base_c in [
            (pause_rect, "일시정지", (55, 55, 130)),
            (menu_rect,  "메뉴",    (100, 38, 38)),
        ]:
            hover = rect.collidepoint(mouse_pos)
            color = tuple(min(c + 40, 255) for c in base_c) if hover else base_c
            pygame.draw.rect(self._display, color, rect, border_radius=8)
            pygame.draw.rect(self._display, (160, 160, 210), rect, 1, border_radius=8)
            lbl = self._fonts["small"].render(label, True, (240, 240, 240))
            self._display.blit(lbl, lbl.get_rect(center=rect.center))

        # ══════════════════════════════════════════════════════
        #  FOOTER — 곡 정보 / 조작 안내
        # ══════════════════════════════════════════════════════
        fy = h - FOOTER_H
        pygame.draw.line(self._display, (40, 35, 70), (0, fy), (w, fy), 1)

        song_title  = (self._current_song or {}).get("title", "데모")
        mode_labels = {"practice": "연습", "challenge": "도전", "freestyle": "자유"}
        mode_label  = mode_labels.get(self._current_mode, "")
        footer_left = self._fonts["small"].render(
            f"[{mode_label}]  {song_title}", True, (120, 160, 220)
        )
        self._display.blit(footer_left, (14, fy + 6))

        footer_right = self._fonts["small"].render(
            "P: 일시정지  |  ESC: 메뉴", True, (80, 80, 110)
        )
        self._display.blit(footer_right, footer_right.get_rect(midright=(w - 10, fy + FOOTER_H // 2)))

    def _render_pause_overlay(self, w, h):
        """게임플레이 위에 반투명 일시정지 오버레이를 렌더링합니다."""
        import pygame

        # 반투명 어두운 오버레이
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self._display.blit(overlay, (0, 0))

        # 타이틀
        pause_txt = self._fonts["title"].render("일시정지", True, (255, 255, 255))
        self._display.blit(pause_txt, pause_txt.get_rect(center=(w // 2, h // 2 - 80)))

        mouse_pos = pygame.mouse.get_pos()

        btn_defs = [
            ("btn_pause",         "▶  계속하기",  (0, 160, 100)),
            ("btn_gameplay_menu", "메뉴로 돌아가기", (140, 50, 50)),
        ]
        for i, (btn_name, label, color) in enumerate(btn_defs):
            rect = pygame.Rect(w // 2 - 160, h // 2 + i * 72, 320, 54)
            self._btn_rects[btn_name] = rect   # 오버레이 버튼으로 덮어쓰기
            hover = rect.collidepoint(mouse_pos)
            draw_color = tuple(min(c + 50, 255) for c in color) if hover else color
            pygame.draw.rect(self._display, draw_color, rect, border_radius=14)
            pygame.draw.rect(self._display, (220, 220, 220), rect, 2, border_radius=14)
            lbl = self._fonts["menu"].render(label, True, (255, 255, 255))
            self._display.blit(lbl, lbl.get_rect(center=rect.center))

        hint = self._fonts["small"].render("P / Space: 계속  |  ESC: 메뉴", True, (160, 160, 180))
        self._display.blit(hint, hint.get_rect(center=(w // 2, h - 30)))

    def _render_result(self, w, h):
        """Render result screen."""
        import pygame

        self._display.fill((15, 10, 40))

        MARGIN_TOP = 30
        MARGIN_BOTTOM = 40
        FOOTER_H = 24
        BTN_H = 50
        BTN_W = min(200, (w - 60) // 2)

        # 타이틀
        title = self._fonts["title"].render("DANCE COMPLETE!", True, (255, 220, 50))
        title_rect = title.get_rect(center=(w // 2, MARGIN_TOP + 30))
        self._display.blit(title, title_rect)

        # 하단 버튼/푸터 영역 계산
        btn_area_y = h - MARGIN_BOTTOM - FOOTER_H - BTN_H - 10
        content_top = MARGIN_TOP + 80
        content_bottom = btn_area_y - 20

        if self._result_data:
            data = self._result_data
            items = [
                (f"총점: {data.get('total_score', 0)}",       (0, 255, 200)),
                (f"최대 콤보: {data.get('max_combo', 0)}",    (255, 220, 0)),
                (f"총 동작: {data.get('total_moves', 0)}",    (200, 200, 220)),
                (f"평균: {data.get('average_score', 0):.1f}", (180, 180, 255)),
                (f"등급: {data.get('final_grade', '-')}",     (255, 180, 0)),
            ]

            hits = data.get("hit_counts", {})
            total_items = len(items) + (1 if hits else 0)
            item_gap = min(48, max(30, (content_bottom - content_top) // max(total_items, 1)))

            y = content_top
            for text, color in items:
                surf = self._fonts["menu"].render(text, True, color)
                self._display.blit(surf, surf.get_rect(center=(w // 2, y)))
                y += item_gap

            if hits:
                y += 4
                hit_text = "  |  ".join(f"{k}: {v}" for k, v in hits.items())
                hit_surf = self._fonts["body"].render(hit_text, True, (160, 160, 180))
                self._display.blit(hit_surf, hit_surf.get_rect(center=(w // 2, y)))

        # 버튼: 다시하기 / 메뉴 (하단 고정, 중앙 정렬)
        mouse_pos = pygame.mouse.get_pos()
        btn_gap = 20
        total_w = BTN_W * 2 + btn_gap
        btn_x = w // 2 - total_w // 2

        btn_defs = [
            ("btn_retry",       "↻  다시하기", (0, 140, 90)),
            ("btn_result_menu", "홈 화면",    (100, 40, 120)),
        ]
        for i, (btn_name, label, color) in enumerate(btn_defs):
            rect = pygame.Rect(btn_x + i * (BTN_W + btn_gap), btn_area_y, BTN_W, BTN_H)
            self._btn_rects[btn_name] = rect
            hover = rect.collidepoint(mouse_pos)
            draw_color = tuple(min(c + 50, 255) for c in color) if hover else color
            pygame.draw.rect(self._display, draw_color, rect, border_radius=14)
            pygame.draw.rect(self._display, (220, 220, 220), rect, 2, border_radius=14)
            lbl = self._fonts["body"].render(label, True, (255, 255, 255))
            self._display.blit(lbl, lbl.get_rect(center=rect.center))

        hint = self._fonts["small"].render(
            "Enter: 메뉴  |  ESC: 메뉴", True, (100, 100, 130)
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

        title = self._fonts["title"].render("설정", True, (200, 200, 255))
        self._display.blit(title, title.get_rect(center=(w // 2, MARGIN_TOP + 30)))

        settings_items = [
            f"카메라 장치: {self.config.get('camera', {}).get('device_id', 0)}",
            f"해상도: {self.config.get('camera', {}).get('width', 640)}"
            f"x{self.config.get('camera', {}).get('height', 480)}",
            f"UI 테마: {self.config.get('ui', {}).get('theme', 'neon')}",
            f"전체화면: {'예' if self.config.get('ui', {}).get('fullscreen', False) else '아니오'}",
            f"유사도 메트릭: {self.config.get('scoring', {}).get('similarity_metric', 'cosine')}",
            f"목표 FPS: {self.TARGET_FPS}",
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
        hover = back_rect.collidepoint(mouse_pos)
        pygame.draw.rect(self._display, (70, 70, 150) if hover else (50, 50, 110),
                         back_rect, border_radius=14)
        pygame.draw.rect(self._display, (180, 180, 230), back_rect, 2, border_radius=14)
        lbl = self._fonts["body"].render("← 뒤로", True, (255, 255, 255))
        self._display.blit(lbl, lbl.get_rect(center=back_rect.center))

        hint = self._fonts["small"].render("ESC: 뒤로", True, (100, 100, 130))
        self._display.blit(hint, hint.get_rect(center=(w // 2, h - MARGIN_BOTTOM // 2)))

    def transition_to(self, new_state: str):
        """Transition to a new game state."""
        old_state = self.state
        self._prev_state = old_state
        self.state = new_state
        self._on_state_enter(new_state)

    def _on_state_enter(self, state: str):
        """Handle setup when entering a new state."""
        if state == GameState.MENU:
            self._menu_focus_idx = 0
        elif state == GameState.SONG_SELECT:
            self._selected_song_idx = 0
            self._song_focus_idx = 0
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
        elif state == GameState.COUNTDOWN:
            self._countdown_start = time.time()
            self._countdown_timer = 2  # 3초 카운트다운 (워밍업 시간 확보)
        elif state == GameState.PLAYING:
            # PAUSED→PLAYING 복귀인 경우에만 세션 유지
            resuming_from_pause = getattr(self, '_prev_state', None) == GameState.PAUSED
            if not resuming_from_pause:
                # 새 게임 시작: 이전 세션 완전 정리
                if self._current_session is not None:
                    self._current_session.is_active = False
                    self._current_session = None
                self._scorer.reset()
                self._last_feedback = None
                self._feedback_timer = 0.0
                self._current_frame = None
                self._pose_detected = False
                if self._scratch_comparator is not None:
                    self._scratch_comparator.reset()
                self._warned_scratch_no_ref = False
                # 참조 랜드마크 로드
                self._ref_landmarks = None
                self._ref_frame_landmarks = None
                # 레퍼런스 영상 초기화
                if self._ref_video_cap is not None:
                    self._ref_video_cap.release()
                    self._ref_video_cap = None
                self._ref_video_frame = None
                self._ref_video_fps = 30.0

                if self._current_song and self._current_song.get("has_reference"):
                    import numpy as np
                    import cv2 as _cv2
                    ref_path = os.path.join(self._current_song["path"], "reference.npy")
                    try:
                        ref_data = np.load(ref_path)
                        # (N, 33, 3) → (N, 33, 4): visibility 채널 추가
                        if ref_data.ndim == 3 and ref_data.shape[2] == 3:
                            vis = np.ones((*ref_data.shape[:2], 1), dtype=np.float32)
                            ref_data = np.concatenate([ref_data, vis], axis=2)
                        self._ref_landmarks = ref_data.astype(np.float32)
                        print(f"[INFO] 참조 랜드마크 로드: {self._ref_landmarks.shape}")
                    except Exception as e:
                        print(f"[WARN] 참조 랜드마크 로드 실패: {e}")

                    # 레퍼런스 mp4 영상 로드 (metadata에 video 경로가 있는 경우)
                    video_rel = self._current_song.get("video", "")
                    if video_rel:
                        # 절대 경로 구성
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
                            # 영상 리사이즈 크기 미리 계산 (렌더링에서 반복 안 함)
                            ui_cfg = self.config.get("ui", {})
                            disp_w = ui_cfg.get("window_width", 1024)
                            disp_h = ui_cfg.get("window_height", 600)
                            HEADER_H = 55
                            FOOTER_H = 30
                            rp_w = disp_w - disp_w // 2  # 우 패널 너비
                            rp_h = disp_h - HEADER_H - FOOTER_H
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
                            print(f"[INFO] 레퍼런스 영상 로드: {video_path} "
                                  f"({self._ref_video_fps:.1f}fps, "
                                  f"resize→{self._ref_video_size})")
                        else:
                            print(f"[WARN] 레퍼런스 영상 없음: {video_path}")
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
            pass
        elif state == GameState.RESULT:
            self._result_data = self._scorer.get_final_result()

    def shutdown(self):
        """Clean up all resources."""
        self.running = False
        if self._pose_detector:
            self._pose_detector.release()
        if self._camera:
            self._camera.release()
        if self._ref_video_cap is not None:
            self._ref_video_cap.release()
            self._ref_video_cap = None
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass
