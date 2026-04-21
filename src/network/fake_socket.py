"""
FakeGameSocket — 단일 보드 멀티플레이 UI 테스트용 가짜 소켓.

실제 네트워크 통신 없이 가짜 상대방 점수를 주기적으로 생성합니다.
테스트가 끝나면 이 파일을 삭제하거나 무시하면 됩니다.
"""
import threading
import time
import random


class FakeGameSocket:
    """GameSocket과 동일한 인터페이스를 가지는 가짜 소켓."""

    def __init__(self, song_duration: float = 60.0):
        self._song_duration = song_duration   # 곡 길이(초) — 이 시간 후 자동 종료

        # GameSocket과 동일한 공개 속성
        self.opponent_score: int   = 0
        self.opponent_combo: int   = 0
        self.opponent_grade: str   = ""
        self.opponent_connected: bool  = True
        self.opponent_finished: bool   = False
        self.opponent_final_score: int = 0

        # 콜백 (실제 소켓과 동일한 인터페이스)
        self.on_disconnect      = None
        self.on_opponent_finish = None
        self.on_song_select     = None
        self.on_game_start      = None

        self._running = False
        self._sim_started = False  # 실제 점수 시뮬레이션 시작 여부
        self._thread: threading.Thread | None = None

    # ── GameSocket과 동일한 공개 메서드 ──────────────────────────────

    def start(self):
        """소켓 시작 — 점수 시뮬레이션은 COUNTDOWN 시작(send_start) 이후부터."""
        self._running = True
        self._thread = threading.Thread(target=self._simulate_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def send_score(self, score: int, combo: int, grade: str):
        pass   # 가짜 소켓은 전송을 무시

    def send_end(self, final_score: int):
        """실제 게임이 끝났을 때 호출됨 — 상대방도 즉시 종료 처리."""
        if not self.opponent_finished:
            # 내 최종 점수의 80~120% 수준으로 상대 점수 결정
            ratio = random.uniform(0.75, 1.25)
            self.opponent_final_score = int(final_score * ratio)
            self.opponent_score       = self.opponent_final_score
            self.opponent_finished    = True
            if self.on_opponent_finish:
                self.on_opponent_finish()

    def send_song(self, song_id: str):
        pass

    def send_start(self):
        """HOST가 카운트다운 시작 신호를 보낼 때 호출 — 이 시점부터 점수 시뮬레이션 시작."""
        self._sim_started = True

    # ── 가짜 점수 시뮬레이션 루프 ──────────────────────────────────

    def _simulate_loop(self):
        """send_start() 호출 후부터 0.5초마다 상대방 점수를 올립니다."""
        grades  = ["GOOD", "GOOD", "GREAT!", "GREAT!", "PERFECT!", "MISS"]
        weights = [25, 25, 25, 15, 5, 5]

        # send_start() 신호 대기 (최대 120초)
        wait_start = time.time()
        while self._running and not self._sim_started:
            if time.time() - wait_start > 120:
                return
            time.sleep(0.1)

        start = time.time()

        while self._running:
            # 곡 시간이 지나면 자동 종료
            if time.time() - start >= self._song_duration:
                if not self.opponent_finished:
                    self.opponent_final_score = self.opponent_score
                    self.opponent_finished    = True
                    if self.on_opponent_finish:
                        self.on_opponent_finish()
                break

            grade = random.choices(grades, weights=weights)[0]
            pts   = {"PERFECT!": 300, "GREAT!": 200, "GOOD": 100, "MISS": 0}.get(grade, 0)
            pts   = int(pts * random.uniform(0.8, 1.2))

            if grade != "MISS":
                self.opponent_combo += 1
            else:
                self.opponent_combo = 0

            self.opponent_score += pts
            self.opponent_grade  = grade

            time.sleep(0.5)
