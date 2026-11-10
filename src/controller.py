"""Gửi lệnh chuột/phím vào game.

- pydirectinput (SendInput + scan code): game trình duyệt/DirectX nhận được, kể cả khi khóa con trỏ.
- Nút/phím: so Desired với trạng thái thực tế, chỉ gửi phần chênh lệch (nhấn/nhả) - gửi NGAY.
- Con trỏ: camera chỉ cho 30 vị trí/giây -> nếu nhảy thẳng tới từng vị trí, mắt thấy con trỏ giật nấc.
  Một LUỒNG 120 Hz trượt con trỏ dần tới vị trí đích (lọc mũ theo thời gian, tau ~ 25 ms) và
  DỰ ĐOÁN thêm ~nửa frame theo vận tốc để bù trễ. Có VÙNG CHẾT vài pixel để con trỏ đứng yên hẳn khi tay đứng yên.
- Chuột tương đối: phần dịch chuyển được dồn vào "kho" và xả dần mỗi tick; có GIA TỐC
  (tay đi chậm -> chính xác, vẩy nhanh -> đi xa).
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import math
import threading
import time

from .profiles.base import Desired


class _CURSORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wt.DWORD), ("flags", wt.DWORD), ("hCursor", wt.HANDLE), ("ptScreenPos", wt.POINT)]


class DirectInputBackend:
    def __init__(self):
        import pydirectinput

        pydirectinput.PAUSE = 0          # mặc định thư viện ngủ 0.1 s sau mỗi lệnh -> phải tắt
        pydirectinput.FAILSAFE = False
        self.pdi = pydirectinput

    def screen_size(self) -> tuple[int, int]:
        u = ctypes.windll.user32
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)

    def pointer_locked(self) -> bool:
        """Game đang KHÓA con trỏ? (trình duyệt Pointer Lock / game 3D ẩn con trỏ hoặc nhốt nó trong 1 vùng nhỏ)
        - GetCursorInfo: cờ CURSOR_SHOWING = 0 -> con trỏ bị ẩn
        - GetClipCursor: vùng giới hạn con trỏ nhỏ hơn màn hình -> con trỏ bị nhốt"""
        u = ctypes.windll.user32
        ci = _CURSORINFO()
        ci.cbSize = ctypes.sizeof(_CURSORINFO)
        if u.GetCursorInfo(ctypes.byref(ci)) and not (ci.flags & 1):
            return True
        rc = wt.RECT()
        if u.GetClipCursor(ctypes.byref(rc)):
            w, h = self.screen_size()
            if (rc.right - rc.left) < w * 0.9 or (rc.bottom - rc.top) < h * 0.9:
                return True
        return False

    def move_abs(self, x: int, y: int) -> None:
        self.pdi.moveTo(x, y, _pause=False)

    def move_rel(self, dx: int, dy: int) -> None:
        self.pdi.moveRel(dx, dy, relative=True, _pause=False)

    def button(self, name: str, down: bool) -> None:
        (self.pdi.mouseDown if down else self.pdi.mouseUp)(button=name, _pause=False)

    def key(self, name: str, down: bool) -> None:
        (self.pdi.keyDown if down else self.pdi.keyUp)(name, _pause=False)


class Controller:
    def __init__(self, backend, cfg: dict):
        self.backend = backend
        self.sensitivity = float(cfg.get("relative_sensitivity", 1800))
        self.accel = float(cfg.get("relative_accel", 0.6))
        self.tau = float(cfg.get("cursor_tau_ms", 25)) / 1000.0
        self.lead = float(cfg.get("cursor_lead_ms", 16)) / 1000.0
        self.deadzone = float(cfg.get("cursor_deadzone_px", 2))
        self.max_lead_px = float(cfg.get("cursor_max_lead_px", 40))
        self.hz = float(cfg.get("cursor_hz", 120))
        self.buttons: set[str] = set()
        self.keys: set[str] = set()
        self.enabled = True
        self._size: tuple[int, int] | None = None
        self._lock = threading.Lock()
        # trạng thái con trỏ tuyệt đối (pixel)
        self._target: tuple[float, float] | None = None
        self._target_t = 0.0
        self._vel = (0.0, 0.0)
        self._cur: tuple[float, float] | None = None
        self._sent: tuple[int, int] | None = None
        # chuột tương đối: kho dịch chuyển chưa gửi (pixel)
        self._rel_pending = [0.0, 0.0]
        self._rel_frac = [0.0, 0.0]
        # chuột tương đối theo VẬN TỐC (kiểu cần gạt): vận tốc đích nhận ở nhịp camera, vận tốc thực được làm mượt
        # và cộng dồn mỗi tick 120 Hz -> quay đều như cần analog
        self._vel_target = (0.0, 0.0)
        self._vel_now = (0.0, 0.0)
        self._vel_t = -1.0
        self.vel_tau = float(cfg.get("rotate_smooth_ms", 60)) / 1000.0
        self._running = False
        self._thread: threading.Thread | None = None
        self.auto_relative = False          # V2: đang dùng chuột tương đối vì game khóa con trỏ
        self._lock_checked = -1.0

    # ---------- luồng 120 Hz ----------
    def start(self) -> "Controller":
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="cursor-120hz", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        period = 1.0 / self.hz
        last = time.perf_counter()
        while self._running:
            now = time.perf_counter()
            self.tick(now - last, now)
            last = now
            time.sleep(max(0.0, period - (time.perf_counter() - now)))

    def _screen(self) -> tuple[int, int]:
        if self._size is None:
            self._size = self.backend.screen_size()
        return self._size

    def tick(self, dt: float, now: float) -> None:
        """Một bước nội suy (gọi bởi luồng 120 Hz, hoặc trực tiếp trong test)."""
        with self._lock:
            if not self.enabled:
                return
            if self._target is not None:
                tx, ty = self._target
                # dự đoán: đích + vận tốc x lead, giới hạn độ dài; frame cũ quá 50 ms thì thôi dự đoán
                px, py = tx, ty
                if now - self._target_t < 0.05:
                    ox, oy = self._vel[0] * self.lead, self._vel[1] * self.lead
                    n = math.hypot(ox, oy)
                    if n > self.max_lead_px:
                        ox, oy = ox * self.max_lead_px / n, oy * self.max_lead_px / n
                    px, py = tx + ox, ty + oy
                if self._cur is None:
                    self._cur = (tx, ty)
                a = 1.0 - math.exp(-max(dt, 1e-4) / self.tau)
                cx = self._cur[0] + (px - self._cur[0]) * a
                cy = self._cur[1] + (py - self._cur[1]) * a
                self._cur = (cx, cy)
                xi, yi = int(round(cx)), int(round(cy))
                if self._sent is None or abs(xi - self._sent[0]) + abs(yi - self._sent[1]) >= self.deadzone:
                    self.backend.move_abs(xi, yi)
                    self._sent = (xi, yi)
            # vận tốc: làm mượt về vận tốc đích, mất tín hiệu > 0.15 s (mất tay) -> về 0
            target = self._vel_target if now - self._vel_t < 0.15 else (0.0, 0.0)
            a = 1.0 - math.exp(-max(dt, 1e-4) / self.vel_tau)
            self._vel_now = (self._vel_now[0] + (target[0] - self._vel_now[0]) * a,
                             self._vel_now[1] + (target[1] - self._vel_now[1]) * a)
            if abs(self._vel_now[0]) > 1e-6 or abs(self._vel_now[1]) > 1e-6:
                self._rel_pending[0] += self._vel_now[0] * dt * self.sensitivity
                self._rel_pending[1] += self._vel_now[1] * dt * self.sensitivity
            if self._rel_pending[0] or self._rel_pending[1]:
                part = 0.5                                         # xả 50% kho mỗi tick
                fx = self._rel_pending[0] * part + self._rel_frac[0]
                fy = self._rel_pending[1] * part + self._rel_frac[1]
                dx, dy = int(fx), int(fy)
                self._rel_frac = [fx - dx, fy - dy]
                self._rel_pending[0] *= (1 - part)
                self._rel_pending[1] *= (1 - part)
                if abs(self._rel_pending[0]) < 0.5 and abs(self._rel_pending[1]) < 0.5:
                    self._rel_pending = [0.0, 0.0]
                if dx or dy:
                    self.backend.move_rel(dx, dy)

    def _flush_cursor(self) -> None:
        """Đưa con trỏ tới đích ngay (trước khi nhấn nút để click rơi đúng chỗ)."""
        if self._target is not None:
            xi, yi = int(round(self._target[0])), int(round(self._target[1]))
            if self._sent != (xi, yi):
                self.backend.move_abs(xi, yi)
            self._cur, self._sent = (float(xi), float(yi)), (xi, yi)
        dx, dy = int(self._rel_pending[0]), int(self._rel_pending[1])
        if dx or dy:
            self.backend.move_rel(dx, dy)
        self._rel_pending = [0.0, 0.0]

    def _pointer_locked(self, now: float) -> bool:
        """Kiểm tra tối đa 5 lần/giây (gọi API Windows)."""
        if now - self._lock_checked >= 0.2:
            self._lock_checked = now
            fn = getattr(self.backend, "pointer_locked", None)
            self.auto_relative = bool(fn()) if fn else False
        return self.auto_relative

    # ---------- nhận Desired từ luồng vision ----------
    def apply(self, d: Desired, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        with self._lock:
            if not self.enabled:
                d = Desired()
            if d.cursor_auto:                 # V2: chọn 1 trong 2 theo trạng thái khóa con trỏ
                if self._pointer_locked(now):
                    d.cursor_abs = None
                else:
                    d.cursor_rel = d.cursor_vel = None
            if d.cursor_vel is not None:
                self._vel_target, self._vel_t = d.cursor_vel, now
            else:
                self._vel_target = (0.0, 0.0)
            if d.cursor_abs is not None:
                w, h = self._screen()
                tx, ty = d.cursor_abs[0] * (w - 1), d.cursor_abs[1] * (h - 1)
                if self._target is not None:
                    dt = max(now - self._target_t, 1e-3)
                    vx, vy = (tx - self._target[0]) / dt, (ty - self._target[1]) / dt
                    self._vel = (0.5 * self._vel[0] + 0.5 * vx, 0.5 * self._vel[1] + 0.5 * vy)
                self._target, self._target_t = (tx, ty), now
            else:
                self._target, self._vel, self._cur = None, (0.0, 0.0), None
            if d.cursor_rel is not None:
                dx, dy = d.cursor_rel
                speed = math.hypot(dx, dy) * 30.0                  # ~ đơn vị màn hình / giây
                gain = self.sensitivity * (1.0 + self.accel * min(speed, 3.0))
                self._rel_pending[0] += dx * gain
                self._rel_pending[1] += dy * gain

            pressing = (d.buttons - self.buttons) or (d.keys - self.keys)
            if pressing:
                self._flush_cursor()
            for b in sorted(self.buttons - d.buttons):
                self.backend.button(b, False)
            for k in sorted(self.keys - d.keys):
                self.backend.key(k, False)
            for b in sorted(d.buttons - self.buttons):
                self.backend.button(b, True)
            for k in sorted(d.keys - self.keys):
                self.backend.key(k, True)
            self.buttons, self.keys = set(d.buttons), set(d.keys)
            for k in d.taps:
                self.backend.key(k, True)
                self.backend.key(k, False)

    def release_all(self) -> None:
        self.apply(Desired())

    def set_enabled(self, on: bool) -> None:
        with self._lock:
            self.enabled = on
        if not on:
            self.release_all()     # enabled=False -> apply() dùng Desired rỗng -> nhả hết
