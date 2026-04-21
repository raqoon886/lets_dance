"""
UDP Broadcast Discovery — finds an opponent on the local network.

Flow:
  Both Pi's send HELLO broadcasts.
  Whichever RECEIVES a HELLO first → becomes HOST, replies ACK.
  The sender of that HELLO becomes CLIENT upon receiving ACK.
"""
import socket
import threading
import time

from .protocol import make_msg, parse_msg, MSG_HELLO, MSG_ACK, DISCOVERY_PORT

BROADCAST_ADDR   = "255.255.255.255"
DISCOVERY_TIMEOUT = 30.0   # seconds before giving up
HELLO_INTERVAL    = 1.0    # seconds between HELLO broadcasts


class Discovery:
    """Discover an opponent via UDP broadcast."""

    def __init__(self):
        self.role: str = ""            # "host" | "client"
        self.opponent_ip: str = ""
        self._stop = threading.Event()

    def find_opponent(self, on_found, on_timeout, on_status=None):
        """
        Start discovery in a background thread.

        Callbacks (called from background thread):
          on_found(role: str, opponent_ip: str)
          on_timeout()
          on_status(message: str)   — optional progress updates
        """
        self._stop.clear()
        t = threading.Thread(
            target=self._run,
            args=(on_found, on_timeout, on_status),
            daemon=True,
        )
        t.start()

    def stop(self):
        """Cancel ongoing discovery."""
        self._stop.set()

    # ─── internal ────────────────────────────────────────────────────────────

    def _run(self, on_found, on_timeout, on_status):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", DISCOVERY_PORT))
        except OSError as e:
            if on_status:
                on_status(f"포트 바인드 실패: {e}")
            on_timeout()
            return

        sock.settimeout(0.5)
        start = time.time()
        last_hello = 0.0

        while not self._stop.is_set():
            elapsed = time.time() - start
            if elapsed > DISCOVERY_TIMEOUT:
                sock.close()
                on_timeout()
                return

            # ── broadcast HELLO periodically ──────────────────────────────
            if time.time() - last_hello >= HELLO_INTERVAL:
                try:
                    sock.sendto(make_msg(MSG_HELLO), (BROADCAST_ADDR, DISCOVERY_PORT))
                    last_hello = time.time()
                    remain = int(DISCOVERY_TIMEOUT - elapsed)
                    if on_status:
                        on_status(f"상대방 탐색 중... ({remain}s)")
                except OSError:
                    pass

            # ── listen for incoming messages ───────────────────────────────
            try:
                data, addr = sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break

            msg = parse_msg(data)
            mtype = msg.get("type")

            if mtype == MSG_HELLO:
                # A peer sent HELLO → we are HOST, they are CLIENT
                opponent_ip = addr[0]
                # Don't reply to our own broadcast
                own_ips = self._get_own_ips()
                if opponent_ip in own_ips:
                    continue

                self.role = "host"
                self.opponent_ip = opponent_ip
                try:
                    sock.sendto(make_msg(MSG_ACK, role="client"), addr)
                except OSError:
                    pass
                sock.close()
                on_found("host", opponent_ip)
                return

            elif mtype == MSG_ACK:
                # Host confirmed us → we are CLIENT
                self.role = "client"
                self.opponent_ip = addr[0]
                sock.close()
                on_found("client", addr[0])
                return

        sock.close()

    @staticmethod
    def _get_own_ips() -> set:
        """Return all local IP addresses to avoid self-matching."""
        ips = {"127.0.0.1", "::1"}
        try:
            import subprocess
            out = subprocess.check_output(
                ["hostname", "-I"], text=True, timeout=1
            )
            for ip in out.strip().split():
                ips.add(ip)
        except Exception:
            pass
        return ips
