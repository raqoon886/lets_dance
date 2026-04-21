"""
멀티플레이 loopback 테스트 스크립트
====================================
보드 1개에서 터미널 2개로 멀티플레이 통신을 테스트합니다.

사용법:
  터미널 1:  python test_multi_loopback.py host
  터미널 2:  python test_multi_loopback.py client

테스트 항목:
  1. Discovery (HELLO/ACK) 연결
  2. 곡 선택 동기화 (MSG_SONG_SELECT)
  3. 점수 교환 (MSG_SCORE_UPDATE) — 5회
  4. 게임 종료 (MSG_GAME_END)
  5. 연결 해제 (MSG_DISCONNECT)
"""

import sys
import socket
import time
import threading

# ── 경로 설정 ──────────────────────────────────────────────────────────
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from network.protocol import (
    make_msg, parse_msg,
    MSG_HELLO, MSG_ACK, MSG_SONG_SELECT,
    MSG_SCORE_UPDATE, MSG_GAME_END, MSG_DISCONNECT, MSG_HEARTBEAT,
    DISCOVERY_PORT, GAME_PORT,
)

LOOPBACK = "127.0.0.1"

# ── 색상 출력 헬퍼 ─────────────────────────────────────────────────────
def green(s):  return f"\033[92m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"
def red(s):    return f"\033[91m{s}\033[0m"
def cyan(s):   return f"\033[96m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"

def log(role, msg, color=None):
    ts = time.strftime("%H:%M:%S")
    prefix = bold(f"[{role.upper():6s}]")
    line = f"{ts} {prefix} {msg}"
    print(color(line) if color else line, flush=True)

# ══════════════════════════════════════════════════════════════════════
#  HOST
# ══════════════════════════════════════════════════════════════════════

def run_host():
    role = "host"
    log(role, "시작 — CLIENT 연결 대기 중...", cyan)

    # ── 1. Discovery: CLIENT의 HELLO 수신 ────────────────────────────
    disc_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    disc_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    disc_sock.bind((LOOPBACK, DISCOVERY_PORT))
    disc_sock.settimeout(20.0)

    log(role, f"포트 {DISCOVERY_PORT} 수신 대기 중...")

    try:
        data, addr = disc_sock.recvfrom(1024)
    except socket.timeout:
        log(role, "타임아웃 — CLIENT가 연결하지 않았습니다.", red)
        disc_sock.close()
        return

    msg = parse_msg(data)
    if msg.get("type") != MSG_HELLO:
        log(role, f"예상치 못한 메시지: {msg}", red)
        disc_sock.close()
        return

    client_ip = addr[0]
    client_port = addr[1]
    log(role, f"CLIENT HELLO 수신: {client_ip}:{client_port}", green)

    # ACK 전송 → CLIENT에게 role=client 알림
    disc_sock.sendto(make_msg(MSG_ACK, role="client"), addr)
    log(role, "ACK 전송 완료", green)
    disc_sock.close()

    time.sleep(0.3)  # CLIENT가 game_sock을 열 시간

    # ── 2. Game Socket 열기 ────────────────────────────────────────────
    game_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    game_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    game_sock.bind((LOOPBACK, GAME_PORT))
    game_sock.settimeout(5.0)

    opponent_addr = (client_ip, GAME_PORT + 1)  # CLIENT는 GAME_PORT+1 사용 (충돌 회피)

    log(role, "Game Socket 열림")

    # ── 3. 곡 선택 전송 ────────────────────────────────────────────────
    time.sleep(0.5)
    song_id = "404_dance"
    game_sock.sendto(make_msg(MSG_SONG_SELECT, song_id=song_id), opponent_addr)
    log(role, f"곡 선택 전송: {yellow(song_id)}")

    # ── 4. 점수 교환 (5회) ────────────────────────────────────────────
    log(role, "점수 교환 시작...")
    for i in range(1, 6):
        my_score = i * 1000
        combo    = i * 2
        grades   = ["GOOD", "GREAT!", "PERFECT!", "GOOD", "GREAT!"]
        grade    = grades[i - 1]

        # 전송
        game_sock.sendto(
            make_msg(MSG_SCORE_UPDATE, score=my_score, combo=combo, grade=grade),
            opponent_addr,
        )
        log(role, f"  점수 전송 #{i}: score={my_score} combo={combo} grade={grade}")

        # 수신
        try:
            data, _ = game_sock.recvfrom(2048)
            rmsg = parse_msg(data)
            if rmsg.get("type") == MSG_SCORE_UPDATE:
                log(role,
                    f"  상대 점수 수신 #{i}: score={rmsg.get('score')} "
                    f"combo={rmsg.get('combo')} grade={rmsg.get('grade')}",
                    green)
            elif rmsg.get("type") == MSG_HEARTBEAT:
                log(role, f"  HEARTBEAT 수신", yellow)
        except socket.timeout:
            log(role, f"  수신 타임아웃 #{i}", red)

        time.sleep(0.8)

    # ── 5. 게임 종료 ────────────────────────────────────────────────
    final = 5000
    game_sock.sendto(make_msg(MSG_GAME_END, final_score=final), opponent_addr)
    log(role, f"게임 종료 전송: final_score={final}", yellow)

    # 상대 종료 수신 대기
    try:
        data, _ = game_sock.recvfrom(2048)
        rmsg = parse_msg(data)
        if rmsg.get("type") == MSG_GAME_END:
            log(role, f"상대 게임 종료 수신: final_score={rmsg.get('final_score')}", green)
    except socket.timeout:
        log(role, "상대 게임 종료 수신 타임아웃", red)

    # ── 6. 연결 해제 ────────────────────────────────────────────────
    game_sock.sendto(make_msg(MSG_DISCONNECT), opponent_addr)
    log(role, "연결 해제 전송", yellow)
    game_sock.close()

    log(role, bold("=== HOST 테스트 완료 ==="), green)
    _print_result_summary(role, my_score=5000, opp_score=5000)


# ══════════════════════════════════════════════════════════════════════
#  CLIENT
# ══════════════════════════════════════════════════════════════════════

def run_client():
    role = "client"
    log(role, "시작 — HOST에게 HELLO 전송...", cyan)

    # ── 1. Discovery: HOST에게 HELLO 전송 ────────────────────────────
    disc_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    disc_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    disc_sock.settimeout(20.0)

    # HELLO 전송 (HOST의 DISCOVERY_PORT로)
    disc_sock.sendto(make_msg(MSG_HELLO), (LOOPBACK, DISCOVERY_PORT))
    log(role, f"HELLO 전송 → {LOOPBACK}:{DISCOVERY_PORT}")

    # ACK 수신
    try:
        data, addr = disc_sock.recvfrom(1024)
    except socket.timeout:
        log(role, "타임아웃 — HOST가 응답하지 않았습니다.", red)
        disc_sock.close()
        return

    msg = parse_msg(data)
    if msg.get("type") != MSG_ACK:
        log(role, f"예상치 못한 메시지: {msg}", red)
        disc_sock.close()
        return

    host_ip = addr[0]
    log(role, f"HOST ACK 수신: {host_ip} → CLIENT 확정", green)
    disc_sock.close()

    # ── 2. Game Socket 열기 (GAME_PORT+1 사용 — HOST와 포트 충돌 방지) ──
    game_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    game_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    game_sock.bind((LOOPBACK, GAME_PORT + 1))
    game_sock.settimeout(5.0)

    opponent_addr = (host_ip, GAME_PORT)
    log(role, "Game Socket 열림")

    # ── 3. 곡 선택 수신 대기 ──────────────────────────────────────────
    log(role, "HOST의 곡 선택 대기 중...")
    song_id = None
    for _ in range(20):  # 최대 10초 대기
        try:
            data, _ = game_sock.recvfrom(2048)
            rmsg = parse_msg(data)
            if rmsg.get("type") == MSG_SONG_SELECT:
                song_id = rmsg.get("song_id")
                log(role, f"곡 선택 수신: {yellow(song_id)}", green)
                break
        except socket.timeout:
            log(role, "  곡 선택 대기 중...", yellow)

    if not song_id:
        log(role, "곡 선택 수신 실패", red)

    # ── 4. 점수 교환 (5회) ────────────────────────────────────────────
    log(role, "점수 교환 시작...")
    opp_final = 0
    for i in range(1, 6):
        my_score = i * 800
        combo    = i
        grades   = ["MISS", "GOOD", "GREAT!", "PERFECT!", "GREAT!"]
        grade    = grades[i - 1]

        # 전송
        game_sock.sendto(
            make_msg(MSG_SCORE_UPDATE, score=my_score, combo=combo, grade=grade),
            opponent_addr,
        )
        log(role, f"  점수 전송 #{i}: score={my_score} combo={combo} grade={grade}")

        # 수신
        try:
            data, _ = game_sock.recvfrom(2048)
            rmsg = parse_msg(data)
            if rmsg.get("type") == MSG_SCORE_UPDATE:
                log(role,
                    f"  상대 점수 수신 #{i}: score={rmsg.get('score')} "
                    f"combo={rmsg.get('combo')} grade={rmsg.get('grade')}",
                    green)
            elif rmsg.get("type") == MSG_GAME_END:
                opp_final = int(rmsg.get("final_score", 0))
                log(role, f"  상대 게임 종료 수신: final_score={opp_final}", yellow)
                break
        except socket.timeout:
            log(role, f"  수신 타임아웃 #{i}", red)

        time.sleep(0.8)

    # ── 5. 게임 종료 ────────────────────────────────────────────────
    final = 4000
    game_sock.sendto(make_msg(MSG_GAME_END, final_score=final), opponent_addr)
    log(role, f"게임 종료 전송: final_score={final}", yellow)

    # 상대 종료 & 연결 해제 수신 대기
    try:
        data, _ = game_sock.recvfrom(2048)
        rmsg = parse_msg(data)
        mtype = rmsg.get("type")
        if mtype == MSG_GAME_END:
            opp_final = int(rmsg.get("final_score", 0))
            log(role, f"상대 게임 종료 수신: final_score={opp_final}", green)
        elif mtype == MSG_DISCONNECT:
            log(role, "상대 연결 해제 수신", yellow)
    except socket.timeout:
        pass

    game_sock.close()

    log(role, bold("=== CLIENT 테스트 완료 ==="), green)
    _print_result_summary(role, my_score=final, opp_score=opp_final if opp_final else 5000)


# ── 결과 요약 ──────────────────────────────────────────────────────────

def _print_result_summary(role, my_score, opp_score):
    print()
    print(bold("─" * 40))
    print(bold(f"  결과 요약 ({role.upper()})"))
    print(bold("─" * 40))
    print(f"  내 점수       : {green(str(my_score))}")
    print(f"  상대방 점수   : {yellow(str(opp_score))}")
    if my_score > opp_score:
        print(f"  결과          : {green('WIN! 🎉')}")
    elif my_score < opp_score:
        print(f"  결과          : {red('LOSE...')}")
    else:
        print(f"  결과          : {yellow('DRAW')}")
    print(bold("─" * 40))
    print()


# ── 진입점 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("host", "client"):
        print(f"""
{bold('사용법:')}
  터미널 1:  {green('python test_multi_loopback.py host')}
  터미널 2:  {green('python test_multi_loopback.py client')}

{bold('순서:')}
  1. 터미널 1에서 host 먼저 실행 (수신 대기 상태)
  2. 터미널 2에서 client 실행
  3. 자동으로 연결 → 곡 선택 → 점수 교환 → 종료
""")
        sys.exit(1)

    mode = sys.argv[1]
    try:
        if mode == "host":
            run_host()
        else:
            run_client()
    except KeyboardInterrupt:
        print(red("\n중단됨 (Ctrl+C)"))
