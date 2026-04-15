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
        self.running = False
        self._camera = None
        self._pose_detector = None
        self._embedding_extractor = None
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
        # 피드백 이펙트 페이드 타이머 (초)
        self._feedback_timer: float = 0.0

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
        pygame.display.set_caption("Let's Dance! 🎵")
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

        # ── 폰트 로드 (한국어 지원: NotoSansCJK → fallback SysFont) ──
        self._fonts = self._load_fonts(pygame)

        # ── 댄스 곡 목록 로드 ──
        self._songs = self._load_songs()

        self.running = True

    @staticmethod
    def _load_fonts(pygame):
        """한국어 지원 폰트를 로드합니다. 파일 경로 → SysFont 순으로 fallback."""
        import os

        # 우선순위 폰트 파일 목록 (라즈베리파이 Noto CJK 경로)
        candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        ]
        font_path = None
        for p in candidates:
            if os.path.exists(p):
                font_path = p
                break

        def make(size, bold=False):
            if font_path:
                try:
                    return pygame.font.Font(font_path, size)
                except Exception:
                    pass
            # SysFont fallback
            for name in ["notosanscjkkr", "notosanscjkjp", "notosanscjksc", "sans"]:
                f = pygame.font.SysFont(name, size, bold=bold)
                if f:
                    return f
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

        for event in pygame.event.get():
            # ── 창 닫기 ──────────────────────────────────────
            if event.type == pygame.QUIT:
                self.running = False
                return

            # ── 키보드 ────────────────────────────────────────
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.state in (GameState.PLAYING, GameState.PAUSED,
                                      GameState.SETTINGS, GameState.RESULT,
                                      GameState.COUNTDOWN):
                        self.transition_to(GameState.MENU)
                    else:
                        self.running = False

                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    if self.state == GameState.MENU:
                        self.transition_to(GameState.COUNTDOWN)
                    elif self.state == GameState.SONG_SELECT:
                        self.transition_to(GameState.COUNTDOWN)
                    elif self.state == GameState.PAUSED:
                        self.transition_to(GameState.PLAYING)
                    elif self.state == GameState.RESULT:
                        self.transition_to(GameState.MENU)

                elif event.key == pygame.K_p:
                    if self.state == GameState.PLAYING:
                        self.transition_to(GameState.PAUSED)
                    elif self.state == GameState.PAUSED:
                        self.transition_to(GameState.PLAYING)

                elif event.key == pygame.K_1:
                    if self.state == GameState.MENU:
                        self.transition_to(GameState.COUNTDOWN)
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
            self.transition_to(GameState.COUNTDOWN)
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
            self.transition_to(GameState.COUNTDOWN)
        elif btn_name == "btn_result_menu":
            self.transition_to(GameState.MENU)

        # ── 설정/카운트다운 화면 버튼 ──
        elif btn_name == "btn_back":
            self.transition_to(GameState.MENU)
        elif btn_name == "btn_countdown_cancel":
            self.transition_to(GameState.MENU)

    def _update(self):
        """Update game state based on current state."""
        if self.state == GameState.COUNTDOWN:
            elapsed = time.time() - self._countdown_start
            remaining = 3 - int(elapsed)
            if remaining < 0:
                self.transition_to(GameState.PLAYING)
            else:
                self._countdown_timer = remaining

        elif self.state == GameState.PLAYING:
            self._update_gameplay()

        # PAUSED 상태에서는 카메라/포즈 업데이트 중단

    def _update_gameplay(self):
        """Capture frame, detect pose, compute score."""
        import cv2
        import numpy as np

        if self._camera is None:
            return

        ret, frame = self._camera.read()
        if not ret:
            return

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

        # Score based on pose detection (simulated reference matching)
        if self._pose_detected:
            # In a full implementation, compare with reference.
            # For demo: score based on detection confidence (mean visibility)
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
        """Render the main menu."""
        import pygame

        # Gradient background
        self._display.fill((15, 10, 40))
        for y in range(h):
            alpha = y / h
            color = (int(15 + 20 * alpha), int(10 + 10 * alpha), int(40 + 30 * alpha))
            pygame.draw.line(self._display, color, (0, y), (w, y))

        # Title
        title = self._fonts["title"].render("Let's Dance!", True, (0, 255, 200))
        title_rect = title.get_rect(center=(w // 2, h // 5))
        self._display.blit(title, title_rect)

        # Subtitle
        sub = self._fonts["body"].render("AI 댄스 채점 게임", True, (180, 180, 220))
        sub_rect = sub.get_rect(center=(w // 2, h // 5 + 60))
        self._display.blit(sub, sub_rect)

        # Menu 버튼 정의 (btn_name, label, color)
        btn_defs = [
            ("btn_practice",  "1. 연습 모드 (Practice)",  (0, 180, 130)),
            ("btn_challenge", "2. 도전 모드 (Challenge)", (200, 140, 0)),
            ("btn_freestyle", "3. 자유 모드 (Freestyle)", (140, 80, 210)),
        ]

        mouse_pos = pygame.mouse.get_pos()

        for i, (btn_name, label, color) in enumerate(btn_defs):
            y_pos = h // 2 + i * 75 - 40
            rect = pygame.Rect(w // 2 - 210, y_pos, 420, 58)
            self._btn_rects[btn_name] = rect

            # 호버/터치 시 밝게
            hover = rect.collidepoint(mouse_pos)
            draw_color = tuple(min(c + 50, 255) for c in color) if hover else color
            pygame.draw.rect(self._display, draw_color, rect, border_radius=14)
            pygame.draw.rect(self._display, (255, 255, 255), rect, 2, border_radius=14)
            lbl_surf = self._fonts["body"].render(label, True, (255, 255, 255))
            self._display.blit(lbl_surf, lbl_surf.get_rect(center=rect.center))

        # 설정 / 종료 버튼
        btn_s_rect = pygame.Rect(w // 2 - 210, h - 100, 195, 48)
        btn_q_rect = pygame.Rect(w // 2 + 15,  h - 100, 195, 48)
        self._btn_rects["btn_settings"] = btn_s_rect
        self._btn_rects["btn_quit"]     = btn_q_rect

        for rect, label, color in [
            (btn_s_rect, "⚙  설정",  (80, 80, 160)),
            (btn_q_rect, "✕  종료",  (160, 60, 60)),
        ]:
            hover = rect.collidepoint(mouse_pos)
            draw_color = tuple(min(c + 40, 255) for c in color) if hover else color
            pygame.draw.rect(self._display, draw_color, rect, border_radius=12)
            pygame.draw.rect(self._display, (200, 200, 200), rect, 2, border_radius=12)
            lbl_surf = self._fonts["body"].render(label, True, (255, 255, 255))
            self._display.blit(lbl_surf, lbl_surf.get_rect(center=rect.center))

        # 조작 안내 (화면 하단)
        footer = self._fonts["small"].render(
            "Enter/Space: 시작  |  P: 일시정지  |  ESC: 종료", True, (100, 100, 130)
        )
        self._display.blit(footer, footer.get_rect(center=(w // 2, h - 20)))

    def _render_song_select(self, w, h):
        """곡 선택 화면을 렌더링합니다."""
        import pygame

        MODE_LABELS = {
            "practice":  "연습 모드",
            "challenge": "도전 모드",
            "freestyle": "자유 모드",
        }
        DIFF_STARS = {0: "자유", 1: "★☆☆", 2: "★★☆", 3: "★★★"}

        self._display.fill((12, 8, 35))

        # 헤더
        header_bg = pygame.Rect(0, 0, w, 56)
        pygame.draw.rect(self._display, (25, 18, 60), header_bg)
        title_txt = self._fonts["menu"].render(
            f"{MODE_LABELS.get(self._current_mode, '')}  —  곡 선택", True, (200, 200, 255)
        )
        self._display.blit(title_txt, title_txt.get_rect(midleft=(20, 28)))

        songs = self._songs_for_mode(self._current_mode)
        mouse_pos = pygame.mouse.get_pos()

        if not songs:
            msg = self._fonts["body"].render("이 모드에서 플레이 가능한 곡이 없습니다.", True, (180, 100, 100))
            self._display.blit(msg, msg.get_rect(center=(w // 2, h // 2)))
        else:
            # ── 곡 카드 목록 (왼쪽 55%) ──
            card_w, card_h = int(w * 0.52), 68
            card_x = 20
            card_start_y = 72

            for i, song in enumerate(songs):
                cy = card_start_y + i * (card_h + 10)
                rect = pygame.Rect(card_x, cy, card_w, card_h)
                self._btn_rects[f"btn_song_{i}"] = rect

                selected = (i == self._selected_song_idx)
                hover    = rect.collidepoint(mouse_pos) and not selected

                if selected:
                    bg = (0, 140, 100)
                    border = (0, 255, 180)
                elif hover:
                    bg = (40, 35, 80)
                    border = (120, 100, 200)
                else:
                    bg = (28, 22, 55)
                    border = (60, 55, 100)

                pygame.draw.rect(self._display, bg,     rect, border_radius=12)
                pygame.draw.rect(self._display, border, rect, 2, border_radius=12)

                # 곡 제목
                title_s = self._fonts["body"].render(song.get("title", "?"), True, (255, 255, 255))
                self._display.blit(title_s, (rect.x + 14, rect.y + 10))

                # BPM / 난이도
                diff  = DIFF_STARS.get(song.get("difficulty", 0), "")
                bpm   = song.get("bpm", 0)
                dur   = song.get("duration", 0)
                info  = f"BPM {bpm}  |  {dur}초  |  {diff}"
                info_s = self._fonts["small"].render(info, True, (160, 160, 190))
                self._display.blit(info_s, (rect.x + 14, rect.y + 38))

            # ── 선택된 곡 상세 패널 (오른쪽 40%) ──
            if 0 <= self._selected_song_idx < len(songs):
                sel = songs[self._selected_song_idx]
                px = int(w * 0.58)
                panel = pygame.Rect(px, 70, w - px - 16, h - 160)
                pygame.draw.rect(self._display, (22, 18, 52), panel, border_radius=14)
                pygame.draw.rect(self._display, (80, 70, 140), panel, 2, border_radius=14)

                py = panel.y + 20
                for label, val in [
                    ("곡 제목",  sel.get("title", "-")),
                    ("아티스트", sel.get("artist", "-")),
                    ("BPM",     str(sel.get("bpm", 0))),
                    ("길이",    f"{sel.get('duration', 0)}초"),
                    ("난이도",  DIFF_STARS.get(sel.get("difficulty", 0), "-")),
                    ("레퍼런스", "있음" if sel.get("has_reference") else "없음"),
                ]:
                    lbl_s = self._fonts["small"].render(label, True, (140, 140, 180))
                    val_s = self._fonts["body"].render(str(val), True, (220, 220, 255))
                    self._display.blit(lbl_s, (panel.x + 16, py))
                    self._display.blit(val_s, (panel.x + 16, py + 20))
                    py += 58

                # 시작 버튼
                start_rect = pygame.Rect(panel.x + 20, h - 145, panel.width - 40, 52)
                self._btn_rects["btn_song_start"] = start_rect
                hover_s = start_rect.collidepoint(mouse_pos)
                pygame.draw.rect(self._display,
                                 (0, 190, 120) if hover_s else (0, 150, 90),
                                 start_rect, border_radius=14)
                pygame.draw.rect(self._display, (200, 255, 220), start_rect, 2, border_radius=14)
                go_s = self._fonts["menu"].render("▶  시작하기", True, (255, 255, 255))
                self._display.blit(go_s, go_s.get_rect(center=start_rect.center))

        # 뒤로가기 버튼
        back_rect = pygame.Rect(16, h - 72, 150, 44)
        self._btn_rects["btn_song_back"] = back_rect
        hover_b = back_rect.collidepoint(mouse_pos)
        pygame.draw.rect(self._display, (70, 55, 110) if hover_b else (50, 40, 80),
                         back_rect, border_radius=10)
        pygame.draw.rect(self._display, (160, 140, 200), back_rect, 2, border_radius=10)
        back_s = self._fonts["body"].render("← 뒤로", True, (220, 220, 240))
        self._display.blit(back_s, back_s.get_rect(center=back_rect.center))

        # 조작 안내
        hint = self._fonts["small"].render(
            "터치/클릭으로 곡 선택  |  ESC: 뒤로", True, (90, 90, 120)
        )
        self._display.blit(hint, hint.get_rect(center=(w // 2, h - 18)))

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
            self._draw_stick_figure(
                self._display,
                self._current_landmarks,
                left_fig_rect,
                line_color=(0, 200, 255),
                joint_color=(0, 255, 200),
            )
        else:
            # 포즈 미감지 안내
            no_pose = self._fonts["body"].render("포즈 감지 중...", True, (80, 80, 120))
            fp_rect = pygame.Rect(*left_fig_rect)
            self._display.blit(no_pose, no_pose.get_rect(center=fp_rect.center))

        # 카메라 피드 (작게, 좌 하단)
        if hasattr(self, '_current_frame') and self._current_frame is not None:
            frame_rgb = cv2.cvtColor(self._current_frame, cv2.COLOR_BGR2RGB)
            frame_small = cv2.resize(frame_rgb, (CAM_W, CAM_H))
            cam_surf = pygame.surfarray.make_surface(frame_small.swapaxes(0, 1))
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
        #  RIGHT — 가이드 캐릭터 + 피드백 이펙트
        # ══════════════════════════════════════════════════════

        # 우 패널 배경
        pygame.draw.rect(self._display, (10, 8, 28),
                         pygame.Rect(right_panel[0], right_panel[1],
                                     right_panel[2], right_panel[3]))

        # 패널 레이블
        lbl_guide = self._fonts["small"].render("가이드", True, (255, 160, 80))
        self._display.blit(lbl_guide, (right_panel[0] + 12, right_panel[1] + 8))

        right_fig_rect = (right_panel[0], BODY_Y, right_panel[2], BODY_H)

        if self._ref_frame_landmarks is not None:
            self._draw_stick_figure(
                self._display,
                self._ref_frame_landmarks,
                right_fig_rect,
                line_color=(255, 140, 0),
                joint_color=(255, 200, 80),
            )
        else:
            # 가이드 없음 안내
            no_guide = self._fonts["body"].render("가이드 캐릭터 없음", True, (80, 70, 60))
            rp_rect = pygame.Rect(*right_fig_rect)
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

        title = self._fonts["title"].render("DANCE COMPLETE!", True, (255, 220, 50))
        title_rect = title.get_rect(center=(w // 2, 55))
        self._display.blit(title, title_rect)

        if self._result_data:
            data = self._result_data
            y = 130
            items = [
                (f"총점: {data.get('total_score', 0)}",       (0, 255, 200)),
                (f"최대 콤보: {data.get('max_combo', 0)}",    (255, 220, 0)),
                (f"총 동작: {data.get('total_moves', 0)}",    (200, 200, 220)),
                (f"평균: {data.get('average_score', 0):.1f}", (180, 180, 255)),
                (f"등급: {data.get('final_grade', '-')}",     (255, 180, 0)),
            ]
            for text, color in items:
                surf = self._fonts["menu"].render(text, True, color)
                self._display.blit(surf, surf.get_rect(center=(w // 2, y)))
                y += 55

            hits = data.get("hit_counts", {})
            if hits:
                y += 5
                hit_text = "  |  ".join(f"{k}: {v}" for k, v in hits.items())
                hit_surf = self._fonts["body"].render(hit_text, True, (160, 160, 180))
                self._display.blit(hit_surf, hit_surf.get_rect(center=(w // 2, y)))

        # 버튼: 다시하기 / 메뉴
        mouse_pos = pygame.mouse.get_pos()
        btn_defs = [
            ("btn_retry",       "↻  다시하기", (0, 140, 90)),
            ("btn_result_menu", "홈 화면",    (100, 40, 120)),
        ]
        for i, (btn_name, label, color) in enumerate(btn_defs):
            rect = pygame.Rect(w // 2 - 215 + i * 230, h - 90, 210, 54)
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
        self._display.blit(hint, hint.get_rect(center=(w // 2, h - 18)))

    def _render_settings(self, w, h):
        """Render settings screen."""
        import pygame

        self._display.fill((15, 10, 40))

        title = self._fonts["title"].render("설정", True, (200, 200, 255))
        self._display.blit(title, title.get_rect(center=(w // 2, 55)))

        settings_items = [
            f"카메라 장치: {self.config.get('camera', {}).get('device_id', 0)}",
            f"해상도: {self.config.get('camera', {}).get('width', 640)}"
            f"x{self.config.get('camera', {}).get('height', 480)}",
            f"UI 테마: {self.config.get('ui', {}).get('theme', 'neon')}",
            f"전체화면: {'예' if self.config.get('ui', {}).get('fullscreen', False) else '아니오'}",
            f"유사도 메트릭: {self.config.get('scoring', {}).get('similarity_metric', 'cosine')}",
            f"목표 FPS: {self.TARGET_FPS}",
        ]
        for i, text in enumerate(settings_items):
            surf = self._fonts["body"].render(text, True, (180, 180, 200))
            self._display.blit(surf, (w // 2 - 220, 130 + i * 50))

        # 뒤로가기 버튼
        mouse_pos = pygame.mouse.get_pos()
        back_rect = pygame.Rect(w // 2 - 110, h - 90, 220, 54)
        self._btn_rects["btn_back"] = back_rect
        hover = back_rect.collidepoint(mouse_pos)
        pygame.draw.rect(self._display, (70, 70, 150) if hover else (50, 50, 110),
                         back_rect, border_radius=14)
        pygame.draw.rect(self._display, (180, 180, 230), back_rect, 2, border_radius=14)
        lbl = self._fonts["body"].render("← 뒤로", True, (255, 255, 255))
        self._display.blit(lbl, lbl.get_rect(center=back_rect.center))

        hint = self._fonts["small"].render("ESC: 뒤로", True, (100, 100, 130))
        self._display.blit(hint, hint.get_rect(center=(w // 2, h - 18)))

    def transition_to(self, new_state: str):
        """Transition to a new game state."""
        old_state = self.state
        self.state = new_state
        self._on_state_enter(new_state)

    def _on_state_enter(self, state: str):
        """Handle setup when entering a new state."""
        if state == GameState.SONG_SELECT:
            self._selected_song_idx = 0
        elif state == GameState.COUNTDOWN:
            self._countdown_start = time.time()
            self._countdown_timer = 3
        elif state == GameState.PLAYING:
            # PAUSED에서 복귀하는 경우 세션 유지
            if not (self._current_session and self._current_session.is_active):
                self._scorer.reset()
                self._last_feedback = None
                self._feedback_timer = 0.0
                self._current_frame = None
                self._pose_detected = False
                # 참조 랜드마크 로드
                self._ref_landmarks = None
                self._ref_frame_landmarks = None
                if self._current_song and self._current_song.get("has_reference"):
                    import numpy as np
                    ref_path = os.path.join(self._current_song["path"], "reference.npy")
                    try:
                        self._ref_landmarks = np.load(ref_path)  # (N, 33, 4)
                        print(f"[INFO] 참조 랜드마크 로드: {self._ref_landmarks.shape}")
                    except Exception as e:
                        print(f"[WARN] 참조 랜드마크 로드 실패: {e}")
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
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass
