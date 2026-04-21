"""
UDP Game Socket — real-time score exchange during multi-player gameplay.
"""
import socket
import threading
import time

from .protocol import (
    make_msg, parse_msg,
    MSG_SCORE_UPDATE, MSG_GAME_END, MSG_DISCONNECT, MSG_HEARTBEAT,
    MSG_SONG_SELECT, MSG_GAME_START,
    GAME_PORT,
)

HEARTBEAT_INTERVAL = 1.0   # seconds between heartbeats
TIMEOUT_SEC        = 5.0   # seconds of silence before marking disconnected


class GameSocket:
    """
    Send / receive score updates via UDP during gameplay.

    Thread-safe for read access to opponent_* attributes.
    """

    def __init__(self, opponent_ip: str):
        self.opponent_ip = opponent_ip

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("", GAME_PORT))
        self._sock.settimeout(0.05)

        self._running = False
        self._thread: threading.Thread | None = None
        self._last_recv: float = 0.0
        self._last_heartbeat: float = 0.0

        # ── opponent state (updated by recv thread, read by main thread) ──
        self.opponent_score: int    = 0
        self.opponent_combo: int    = 0
        self.opponent_grade: str    = ""
        self.opponent_connected: bool = False
        self.opponent_finished: bool  = False
        self.opponent_final_score: int = 0

        # optional callbacks (called from recv thread)
        self.on_disconnect = None
        self.on_opponent_finish = None
        self.on_song_select = None    # (song_id: str)
        self.on_game_start  = None    # () — CLIENT가 HOST의 시작 신호 수신 시

    # ─── public API ──────────────────────────────────────────────────────────

    def start(self):
        """Start background receive loop."""
        self._running = True
        self.opponent_connected = True
        self._last_recv = time.time()
        self._last_heartbeat = time.time()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Gracefully close socket and stop receive loop."""
        self._running = False
        try:
            self._sock.sendto(
                make_msg(MSG_DISCONNECT),
                (self.opponent_ip, GAME_PORT),
            )
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass

    def send_song(self, song_id: str, mode: str = "practice"):
        """HOST가 선택한 곡 ID와 모드를 CLIENT에 전송."""
        self._send(make_msg(MSG_SONG_SELECT, song_id=song_id, mode=mode))

    def send_start(self):
        """HOST가 카운트다운 시작 신호를 CLIENT에 전송."""
        self._send(make_msg(MSG_GAME_START))

    def send_score(self, score: int, combo: int, grade: str):
        """Send current score snapshot to opponent."""
        self._send(make_msg(MSG_SCORE_UPDATE, score=score, combo=combo, grade=grade))

    def send_end(self, final_score: int):
        """Notify opponent that this side has finished."""
        self._send(make_msg(MSG_GAME_END, final_score=final_score))

    # ─── internal ────────────────────────────────────────────────────────────

    def _send(self, data: bytes):
        try:
            self._sock.sendto(data, (self.opponent_ip, GAME_PORT))
        except OSError:
            pass

    def _recv_loop(self):
        while self._running:
            now = time.time()

            # periodic heartbeat
            if now - self._last_heartbeat >= HEARTBEAT_INTERVAL:
                self._send(make_msg(MSG_HEARTBEAT))
                self._last_heartbeat = now

            # check timeout
            if now - self._last_recv > TIMEOUT_SEC:
                if self.opponent_connected:
                    self.opponent_connected = False
                    if self.on_disconnect:
                        self.on_disconnect()

            try:
                data, _ = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break

            self._last_recv = time.time()
            if not self.opponent_connected:
                self.opponent_connected = True  # reconnected

            msg = parse_msg(data)
            mtype = msg.get("type")

            if mtype == MSG_SCORE_UPDATE:
                self.opponent_score = int(msg.get("score", 0))
                self.opponent_combo = int(msg.get("combo", 0))
                self.opponent_grade = str(msg.get("grade", ""))

            elif mtype == MSG_SONG_SELECT:
                song_id = str(msg.get("song_id", ""))
                mode    = str(msg.get("mode", "practice"))
                if song_id and self.on_song_select:
                    self.on_song_select(song_id, mode)

            elif mtype == MSG_GAME_START:
                if self.on_game_start:
                    self.on_game_start()

            elif mtype == MSG_GAME_END:
                self.opponent_final_score = int(msg.get("final_score", 0))
                self.opponent_score = self.opponent_final_score
                self.opponent_finished = True
                if self.on_opponent_finish:
                    self.on_opponent_finish()

            elif mtype == MSG_DISCONNECT:
                self.opponent_connected = False
                if self.on_disconnect:
                    self.on_disconnect()

            # MSG_HEARTBEAT — just updates _last_recv (already done above)
