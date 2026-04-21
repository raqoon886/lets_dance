"""
Network protocol — message format definitions.
"""
import json
import time

# Message types
MSG_HELLO        = "HELLO"
MSG_ACK          = "ACK"
MSG_SONG_SELECT  = "SONG_SELECT"
MSG_GAME_START   = "GAME_START"    # HOST → CLIENT: 동시 카운트다운 시작 신호
MSG_SCORE_UPDATE = "SCORE_UPDATE"
MSG_GAME_END     = "GAME_END"
MSG_HEARTBEAT    = "HEARTBEAT"
MSG_DISCONNECT   = "DISCONNECT"

DISCOVERY_PORT = 5555
GAME_PORT      = 5556


def make_msg(msg_type: str, **kwargs) -> bytes:
    """Encode a message to bytes."""
    return json.dumps({"type": msg_type, "ts": time.time(), **kwargs}).encode("utf-8")


def parse_msg(data: bytes) -> dict:
    """Decode a message from bytes. Returns {} on error."""
    try:
        return json.loads(data.decode("utf-8"))
    except Exception:
        return {}
