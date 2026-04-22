"""
UDP Game Socket — real-time score exchange during multi-player gameplay.
"""
import socket
import threading
import time

from .protocol import (
    make_msg, parse_msg,
    MSG_SCORE_UPDATE, MSG_GAME_END, MSG_DISCONNECT, MSG_HEARTBEAT,
    MSG_SONG_SELECT, MSG_GAME_START, MSG_POSE_READY, MSG_SKELETON_UPDATE,
    GAME_PORT,
)

HEARTBEAT_INTERVAL = 1.0   # seconds between heartbeats
TIMEOUT_SEC        = 5.0   # seconds of silence before marking disconnected


class GameSocket:
    """
    Send / receive score updates via UDP during gameplay.

    Thread-safe for read access to opponent_* attributes.
    """

    def __init__(self, opponent_ip: str, role: str = "player", host_ip: str = "", client_ip: str = ""):
        self.opponent_ip = opponent_ip
        self.role = role
        self.host_ip = host_ip
        self.client_ip = client_ip

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
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

        # Spectator states
        self.players_state = {
            self.host_ip: {"score": 0, "combo": 0, "grade": "", "pose": None},
            self.client_ip: {"score": 0, "combo": 0, "grade": "", "pose": None}
        }

        # optional callbacks (called from recv thread)
        self.on_disconnect = None
        self.on_opponent_finish = None
        self.on_song_select = None         # (song_id: str, mode: str)
        self.on_game_start  = None         # () — CLIENT가 HOST의 시작 신호 수신 시
        self.on_opponent_pose_ready = None # () — 상대방 포즈 감지 3초 완료 알림

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

    def send_pose_ready(self):
        """포즈 감지 3초 완료 알림을 상대방에게 전송 (HOST/CLIENT 모두 사용)."""
        self._send(make_msg(MSG_POSE_READY))

    def send_start(self):
        """HOST가 카운트다운 시작 신호를 CLIENT에 전송."""
        self._send(make_msg(MSG_GAME_START))

    def send_score(self, score: int, combo: int, grade: str):
        """Send current score snapshot to opponent."""
        self._send(make_msg(MSG_SCORE_UPDATE, score=score, combo=combo, grade=grade))

    def send_skeleton(self, pose_data: list):
        """Send skeleton metadata to spectators."""
        self._send(make_msg(MSG_SKELETON_UPDATE, pose=pose_data))

    def send_end(self, final_score: int):
        """Notify opponent that this side has finished."""
        self._send(make_msg(MSG_GAME_END, final_score=final_score))

    # ─── internal ────────────────────────────────────────────────────────────

    def _send(self, data: bytes):
        try:
            if self.opponent_ip:
                self._sock.sendto(data, (self.opponent_ip, GAME_PORT))
        except OSError:
            pass
        # Also broadcast for spectators
        try:
            self._sock.sendto(data, ("255.255.255.255", GAME_PORT))
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
                data, addr = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break

            self._last_recv = time.time()
            if not self.opponent_connected:
                self.opponent_connected = True  # reconnected

            msg = parse_msg(data)
            mtype = msg.get("type")
            sender_ip = addr[0].strip()

            if self.role == "spectator":
                # Check if this IP is one of our tracked players
                if sender_ip in self.players_state:
                    p = self.players_state[sender_ip]
                    if mtype == MSG_SCORE_UPDATE:
                        p["score"] = int(msg.get("score", 0))
                        p["combo"] = int(msg.get("combo", 0))
                        p["grade"] = str(msg.get("grade", ""))
                    elif mtype == MSG_SKELETON_UPDATE:
                        p["pose"] = msg.get("pose")
                else:
                    if mtype in (MSG_SCORE_UPDATE, MSG_SKELETON_UPDATE):
                        print(f"[SPECTATOR] 무시: sender={sender_ip} not in players_state keys={list(self.players_state.keys())}", flush=True)
                # MSG_SONG_SELECT는 발신자 무관하게 처리 (관전자도 곡 정보 수신 필요)
                if mtype == MSG_SONG_SELECT:
                    song_id = str(msg.get("song_id", ""))
                    mode    = str(msg.get("mode", "practice"))
                    if song_id and self.on_song_select:
                        self.on_song_select(song_id, mode)
                # MSG_GAME_START: 양쪽 플레이어 준비 완료 → 관전자 영상 재생 시작
                elif mtype == MSG_GAME_START:
                    if self.on_game_start:
                        self.on_game_start()
                continue

            if sender_ip != self.opponent_ip:
                continue

            if mtype == MSG_SCORE_UPDATE:
                self.opponent_score = int(msg.get("score", 0))
                self.opponent_combo = int(msg.get("combo", 0))
                self.opponent_grade = str(msg.get("grade", ""))

            elif mtype == MSG_SONG_SELECT:
                song_id = str(msg.get("song_id", ""))
                mode    = str(msg.get("mode", "practice"))
                if song_id and self.on_song_select:
                    self.on_song_select(song_id, mode)

            elif mtype == MSG_POSE_READY:
                if self.on_opponent_pose_ready:
                    self.on_opponent_pose_ready()

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
