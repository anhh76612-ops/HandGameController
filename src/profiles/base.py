"""Phần dùng chung cho các profile.

Mỗi frame, profile nhận các bàn tay và trả về `Desired` = TRẠNG THÁI MONG MUỐN (nút chuột/phím đang giữ,
vị trí con trỏ, tên cử chỉ của từng tay). Controller chỉ gửi phần chênh lệch so với thực tế.
-> "chạm là giữ, tách là nhả"; mất tay / cử chỉ lạ -> Desired rỗng -> mọi nút tự nhả, không kẹt phím.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..gestures import (
    CLICK_PHAI, CLICK_TRAI, CON_TRO, GIU_PHAI, GIU_TRAI, INDEX_ONLY, KEO_CHEM, KHONG_RO, OPEN, POINTER, TAM_DUNG, THREE,
    TWO_TOGETHER, Hand,
)


@dataclass
class Desired:
    buttons: set[str] = field(default_factory=set)          # {"left", "right"}
    keys: set[str] = field(default_factory=set)             # tên phím pydirectinput
    cursor_abs: tuple[float, float] | None = None           # vị trí màn hình chuẩn hóa 0..1
    cursor_rel: tuple[float, float] | None = None           # dịch chuyển chuẩn hóa
    cursor_vel: tuple[float, float] | None = None           # VẬN TỐC xoay (đơn vị màn hình / giây) - luồng 120 Hz tự cộng dồn
    taps: list[str] = field(default_factory=list)           # phím nhấn-nhả một lần
    gestures: dict[str, str] = field(default_factory=dict)  # {"Left"/"Right": tên cử chỉ}
    joystick: dict | None = None                            # V2: {"vec": (x, y), "keys": set, "active": bool}
    cursor_auto: bool = False                               # có cả abs + rel: controller tự chọn theo khóa con trỏ
    aim_stick: dict | None = None                           # V2: tâm xoay của tay phải (kiểu cần gạt)

    @property
    def label(self) -> str:
        return " | ".join(f"{'Phải' if k == 'Right' else 'Trái'}: {v}" for k, v in sorted(self.gestures.items()))


@dataclass
class PointerOut:
    cursor: tuple[float, float] | None = None
    buttons: set[str] = field(default_factory=set)
    name: str = ""
    taps: list[str] = field(default_factory=list)


class StickyCursor:
    """Con trỏ ỔN ĐỊNH cho chuột tuyệt đối (con trỏ = đầu ngón trỏ).

    - DÍNH: tay gần như đứng yên (tốc độ < still_speed trong still_ms) -> giữ con trỏ đứng im; chỉ chạy
      tiếp khi tay rời điểm giữ quá release_dist -> tay hơi xê dịch không làm con trỏ rung.
    - KHÓA: khi ngón cái sắp chạm / đang click -> con trỏ đứng tại vị trí của rollback_ms trước
      (trước lúc tay kịp lệch vì động tác click).
    """

    def __init__(self, cfg: dict):
        self.still_speed = float(cfg.get("still_speed", 0.08))
        self.still_sec = float(cfg.get("still_ms", 80)) / 1000
        self.release = float(cfg.get("still_release", 0.008))
        self.rollback = float(cfg.get("rollback_ms", 100)) / 1000
        self.reset()

    def reset(self) -> None:
        self.hist: deque[tuple[float, np.ndarray]] = deque(maxlen=30)
        self.hold: np.ndarray | None = None
        self.slow_since: float | None = None
        self.lock: np.ndarray | None = None

    def _at(self, t: float) -> np.ndarray:
        for ts, p in reversed(self.hist):
            if ts <= t:
                return p
        return self.hist[0][1]

    def update(self, pos, t: float, lock: bool = False) -> tuple[float, float]:
        pos = np.asarray(pos, dtype=float)
        self.hist.append((t, pos))
        if lock:
            if self.lock is None:
                self.lock = self.hold if self.hold is not None else self._at(t - self.rollback)
            return float(self.lock[0]), float(self.lock[1])
        if self.lock is not None:                # vừa mở khóa: giữ điểm đó làm điểm "dính"
            self.hold, self.lock = self.lock, None
        if self.hold is not None:
            if float(np.linalg.norm(pos - self.hold)) <= self.release:
                return float(self.hold[0]), float(self.hold[1])
            self.hold, self.slow_since = None, None
        old = self._at(t - 0.066)
        speed = float(np.linalg.norm(pos - old)) / 0.066
        if speed < self.still_speed and len(self.hist) > 2:
            if self.slow_since is None:
                self.slow_since = t
            elif t - self.slow_since >= self.still_sec:
                self.hold = pos.copy()
        else:
            self.slow_since = None
        return float(pos[0]), float(pos[1])


class CommonPointer:
    """Cử chỉ CHUNG cho tay phải ở mọi chế độ (máy trạng thái 1 lượt, KHÔNG đệ quy).

      CON TRỎ    : giơ ngón trỏ (có thể kèm ngón giữa tách ra) -> con trỏ = đầu ngón trỏ
      CLICK TRÁI : đầu ngón cái CHẠM đường ngón trỏ (chỗ nào cũng được) -> giữ chuột trái, tách ra -> nhả
      CLICK PHẢI : NGÓN GIỮA DUỖI NỬA CHỪNG (dài ~1/2 ngón trỏ) -> giữ chuột phải, duỗi thẳng/gập hẳn -> nhả

    CLICK NHANH và GIỮ ĐỂ KÉO được phân biệt theo thời gian giữ:
      - nhả trước click_hold_ms (mặc định 300 ms): con trỏ KHÓA suốt lần click -> click đúng chỗ, không lệch;
      - giữ lâu hơn: chuyển sang GIỮ CHUỘT -> con trỏ đi theo tay để kéo-thả / bôi đen.
      KÉO / CHÉM : khép ngón trỏ + giữa -> giữ chuột trái, con trỏ theo đầu ngón trỏ (allow_drag)
      TẠM DỪNG   : xòe cả bàn tay giữ 1 giây -> Esc

    Một trạng thái chỉ được vào khi TƯ THẾ (đã bỏ phiếu) và TRẠNG THÁI NGÓN hiện tại cùng đồng ý,
    nên khi hạ tay (tư thế cũ còn lưu nhưng ngón đã cụp) sẽ về KHÔNG RÕ chứ không lặp vô hạn.
    """

    def __init__(self, cfg: dict, allow_drag: bool = True, allow_pause: bool = True, open_is_pointer: bool = False):
        self.cfg = cfg
        self.allow_drag = allow_drag
        self.allow_pause = allow_pause            # chế độ bắn: không gửi Esc giữa trận
        self.open_is_pointer = open_is_pointer    # V2: xòe cả bàn tay cũng điều khiển con trỏ / xoay
        self.hold_after = float(cfg.get("click_hold_ms", 300)) / 1000.0
        self.cursor = StickyCursor(cfg)
        self.reset()

    def reset(self) -> None:
        self.state = "idle"
        self.click_start = 0.0
        self.open_since: float | None = None
        self.pause_sent = False
        self.cursor.reset()

    def _next_state(self, h: Hand) -> str:
        two = h.index and h.middle and not h.ring and not h.pinky
        pointing = h.index and (self.open_is_pointer or (not h.ring and not h.pinky))
        clicking_ok = pointing and self.state in ("pointer", "click_left", "click_right")
        if clicking_ok and h.touch == "index":
            return "click_left"
        if clicking_ok and h.middle_half:
            return "click_right"
        if self.allow_drag and two and h.together and (h.pose == TWO_TOGETHER or self.state == "drag"):
            return "drag"
        poses = (POINTER, INDEX_ONLY) + ((OPEN, THREE, TWO_TOGETHER) if self.open_is_pointer else ())
        if pointing and (h.pose in poses or self.state in ("pointer", "click_left", "click_right", "drag")):
            return "pointer"
        if self.allow_pause and h.pose == OPEN and h.index and h.middle and h.ring and h.pinky:
            return "open"
        return "idle"

    def update(self, h: Hand | None, t: float) -> PointerOut:
        if h is None:
            self.reset()
            return PointerOut()
        prev, self.state = self.state, self._next_state(h)
        if self.state in ("click_left", "click_right") and prev != self.state:
            self.click_start = t
        if self.state != "open":
            self.open_since, self.pause_sent = None, False

        if self.state in ("click_left", "click_right"):
            holding = t - self.click_start >= self.hold_after       # giữ lâu -> kéo theo tay
            pos = self.cursor.update(h.tip, t, lock=not holding)
            if self.state == "click_left":
                return PointerOut(pos, {"left"}, GIU_TRAI if holding else CLICK_TRAI)
            return PointerOut(pos, {"right"}, GIU_PHAI if holding else CLICK_PHAI)
        if self.state == "drag":
            pos = self.cursor.update(h.tip, t)
            return PointerOut(pos, {"left"}, KEO_CHEM)
        if self.state == "pointer":
            pos = self.cursor.update(h.tip, t, lock=h.approach)
            return PointerOut(pos, set(), CON_TRO)
        if self.state == "open":
            out = PointerOut(None, set(), TAM_DUNG)
            if self.open_since is None:
                self.open_since = t
            if not self.pause_sent and t - self.open_since >= 1.0:
                out.taps.append(self.cfg.get("pause_key", "esc"))
                self.pause_sent = True
            return out
        self.cursor.reset()
        return PointerOut(None, set(), KHONG_RO)


class RelativeTracker:
    """Vị trí tuyệt đối -> dịch chuyển giữa 2 frame (cho chuột tương đối)."""

    def __init__(self):
        self.prev: np.ndarray | None = None

    def reset(self) -> None:
        self.prev = None

    def delta(self, pos) -> tuple[float, float]:
        pos = np.asarray(pos, dtype=np.float64)
        if self.prev is None:
            self.prev = pos
            return 0.0, 0.0
        d = pos - self.prev
        self.prev = pos
        return float(d[0]), float(d[1])


def fine_aim(dx: float, dy: float, dt: float, cfg: dict) -> tuple[float, float]:
    """Đường cong độ nhạy cho chuột tương đối: chuyển động chậm bị giảm (ngắm tinh, tay rung không làm
    màn hình rung), chuyển động nhanh được tăng (xoay nhanh)."""
    speed = math.hypot(dx, dy) / max(dt, 1e-3)               # đơn vị màn hình / giây
    lo, hi = float(cfg.get("fine_speed", 0.25)), float(cfg.get("fast_speed", 1.5))
    g_lo, g_hi = float(cfg.get("fine_gain", 0.3)), float(cfg.get("fast_gain", 1.6))
    if speed <= lo:
        g = g_lo
    elif speed >= hi:
        g = g_hi
    else:
        g = g_lo + (g_hi - g_lo) * (speed - lo) / (hi - lo)
    return dx * g, dy * g


def apply_pointer(d: Desired, out: PointerOut, mouse_mode: str, rel: RelativeTracker) -> None:
    """Ghép kết quả CommonPointer vào Desired (tuyệt đối hoặc tương đối)."""
    d.buttons |= out.buttons
    d.taps += out.taps
    if out.cursor is None:
        rel.reset()
    elif mouse_mode == "relative":
        d.cursor_rel = rel.delta(out.cursor)
    else:
        d.cursor_abs = out.cursor


class Profile:
    name = "base"
    title = ""

    def __init__(self, cfg: dict, mouse_mode: str = "absolute"):
        self.cfg = cfg
        self.mouse_mode = mouse_mode

    def reset(self) -> None:  # pragma: no cover
        pass

    def update(self, hands: dict[str, Hand], t: float) -> Desired:  # pragma: no cover
        raise NotImplementedError
