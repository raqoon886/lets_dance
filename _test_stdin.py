"""한글/영어 멀티바이트 stdin 텍스트 입력 테스트
사용법: python3 _test_stdin.py
→ 프롬프트가 뜨면 한글/영어 이름 입력 후 Enter
"""
import pygame, sys, os, threading, termios, select, time
os.environ['SDL_NO_SIGNAL_HANDLERS'] = '1'
os.environ['SDL_VIDEODRIVER'] = 'dummy'
pygame.init()

text_mode = {'on': False}

def _post(key, uni=''):
    pygame.event.post(pygame.event.Event(
        pygame.KEYDOWN, key=key, mod=0, unicode=uni, scancode=0))

def bridge():
    import os as _os, signal as _signal
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    def _restore():
        try: termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except: pass

    # Ctrl+C 시 터미널 복구 후 종료
    _orig = _signal.getsignal(_signal.SIGINT)
    def _sig(sig, frame):
        _restore()
        sys.exit(0)
    _signal.signal(_signal.SIGINT, _sig)

    new = termios.tcgetattr(fd)
    # ISIG 켜둠 → Ctrl+C 가 SIGINT 로 전달됨
    new[3] = new[3] & ~(termios.ECHO | termios.ICANON | termios.IEXTEN)
    new[6][termios.VMIN] = 1
    new[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, new)
    _utf8_buf = b''
    try:
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not r: continue
            ch = _os.read(fd, 1)
            if not ch: break
            if text_mode['on']:
                if ch in (b'\x7f', b'\x08'):          # Backspace
                    _utf8_buf = b''
                    sys.stdout.write('\b \b'); sys.stdout.flush()
                    _post(pygame.K_BACKSPACE)
                elif ch in (b'\r', b'\n'):             # Enter
                    _utf8_buf = b''
                    sys.stdout.write('\n'); sys.stdout.flush()
                    _post(pygame.K_RETURN)
                elif ch == b'\x1b':                    # ESC
                    _utf8_buf = b''
                    _post(pygame.K_ESCAPE)
                elif ch[0] >= 0x20:                    # 출력 가능 문자 (ASCII or 멀티바이트)
                    _utf8_buf += ch
                    first = _utf8_buf[0]
                    need = (1 if first < 0x80 else
                            2 if first < 0xE0 else
                            3 if first < 0xF0 else 4)
                    if len(_utf8_buf) >= need:
                        try:
                            uni = _utf8_buf[:need].decode('utf-8')
                            sys.stdout.buffer.write(_utf8_buf[:need])
                            sys.stdout.buffer.flush()
                            _post(0, uni)
                        except Exception:
                            pass
                        _utf8_buf = _utf8_buf[need:]
    finally:
        _restore()

t = threading.Thread(target=bridge, daemon=True)
t.start()

time.sleep(0.3)
text_mode['on'] = True
print('[TEST] 이름 입력 (한글/영어) 후 Enter: ', end='', flush=True)

name_buf = []
deadline = time.time() + 30
while time.time() < deadline:
    for ev in pygame.event.get():
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_RETURN:
                print()
                print('[TEST] ✓ 최종 이름: [' + ''.join(name_buf) + ']')
                sys.exit(0)
            elif ev.key == pygame.K_BACKSPACE:
                if name_buf:
                    removed = name_buf.pop()
                    cols = 2 if ord(removed) > 0x7F else 1
                    sys.stdout.write(('\b \b') * cols); sys.stdout.flush()
            elif ev.unicode:
                name_buf.append(ev.unicode)
    time.sleep(0.01)

print()
print('[TEST] 타임아웃. 입력된 값: [' + ''.join(name_buf) + ']')
