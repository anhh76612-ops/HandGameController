"""Luồng chính (Qt) - Lớp phủ trong suốt toàn màn hình.

- Cửa sổ không viền, luôn nằm trên cùng, nền trong suốt, CLICK XUYÊN QUA (WindowTransparentForInput +
  WS_EX_TRANSPARENT) -> mọi click giả lập rơi xuống game bên dưới.
- Mỗi tay CHỈ 1 vệt neon ở đầu ngón trỏ (tay phải xanh, tay trái hồng) + TÊN CỬ CHỈ cạnh vệt.
  Vị trí vẽ được nội suy mỗi 16 ms (lọc mũ theo thời gian) nên vệt trôi mượt dù camera chỉ 30 FPS.
- Thông báo nhỏ ở giữa phía trên mỗi khi cử chỉ đổi ("Tay phải: CLICK TRÁI").
- BẮN V2: cần điều khiển ảo ở góc dưới bên trái (núm theo tay trái, hướng W/A/S/D đang giữ sáng lên).
- Khung camera ở góc dưới bên phải (F6), HUD góc trên trái, gợi ý/cảnh báo ở giữa phía dưới.
- Hiệu ứng neon: mỗi đoạn vệt vẽ 3 lớp (quầng rộng mờ -> thân màu -> lõi trắng), mờ dần theo tuổi.
"""
from __future__ import annotations

import ctypes
import math
import time
from collections import deque

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .gestures import KHONG_RO
from .pipeline import AppState

GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000

MODE_TITLE = {"slice": "CHÉM", "shoot_v1": "BẮN V1", "shoot_v2": "BẮN V2"}
HAND_VI = {"Right": "Tay phải", "Left": "Tay trái"}


class Overlay(QWidget):
    def __init__(self, state: AppState, cfg: dict, on_tick=None, keys_cfg: dict | None = None):
        super().__init__()
        self.state, self.cfg, self.on_tick = state, cfg, on_tick
        self.keys_cfg = keys_cfg or {"up": "w", "down": "s", "left": "a", "right": "d"}
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                            | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setGeometry(QGuiApplication.primaryScreen().geometry())
        self.trails: dict[str, deque] = {}               # label -> deque[(t, x, y)]
        self.disp: dict[str, tuple[float, float]] = {}   # vị trí vẽ đã nội suy
        self.names: dict[str, str] = {}
        self.reasons: dict[str, str] = {}
        self.toast, self.toast_until = "", 0.0
        self.colors = {"Right": QColor(*cfg["color_right"]), "Left": QColor(*cfg["color_left"])}
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)
        self._last_tick = time.perf_counter()
        self._frames, self._t0, self.fps = 0, time.perf_counter(), 0.0

    def showEvent(self, event):  # noqa: N802 - tên hàm của Qt
        super().showEvent(event)
        # Không tự thêm WS_EX_LAYERED: cửa sổ layered mà không gọi SetLayeredWindowAttributes sẽ bị ẩn đi.
        hwnd = int(self.winId())
        u = ctypes.windll.user32
        style = u.GetWindowLongW(hwnd, GWL_EXSTYLE)
        u.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
        self.exstyle = u.GetWindowLongW(hwnd, GWL_EXSTYLE)

    # ---------- cập nhật 60 lần/giây ----------
    def _tick(self) -> None:
        if self.on_tick:
            self.on_tick()
        now = time.perf_counter()
        dt, self._last_tick = now - self._last_tick, now
        a = 1.0 - math.exp(-max(dt, 1e-3) / 0.03)
        life = self.cfg["trail_sec"]
        w, h = self.width(), self.height()
        with self.state.lock:
            hands = list(self.state.hands)
        seen = set()
        for hv in hands:
            seen.add(hv.label)
            tx, ty = float(hv.tip[0]) * w, float(hv.tip[1]) * h
            px, py = self.disp.get(hv.label, (tx, ty))
            pos = (px + (tx - px) * a, py + (ty - py) * a)
            self.disp[hv.label] = pos
            self.trails.setdefault(hv.label, deque()).append((now, pos[0], pos[1]))
            name = hv.name
            if name and name != KHONG_RO and name != self.names.get(hv.label):
                self.toast, self.toast_until = f"{HAND_VI.get(hv.label, hv.label)}: {name}", now + 1.2
            self.names[hv.label] = name
            self.reasons[hv.label] = hv.reason
        for label in list(self.disp):
            if label not in seen:
                self.disp.pop(label, None)
                self.names.pop(label, None)
                self.reasons.pop(label, None)
        for label, dq in list(self.trails.items()):
            while dq and now - dq[0][0] > life:
                dq.popleft()
            if not dq and label not in seen:
                del self.trails[label]
        self._frames += 1
        if now - self._t0 >= 1.0:
            self.fps, self._frames, self._t0 = self._frames / (now - self._t0), 0, now
        self.update()

    # ---------- vẽ ----------
    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        now = time.perf_counter()
        self._trails(p, now)
        self._joystick(p)
        self._aim_stick(p)
        self._preview(p)
        if self.cfg.get("show_hud", True):
            self._hud(p, now)
        p.end()

    def _trails(self, p: QPainter, now: float) -> None:
        life = self.cfg["trail_sec"]
        for label, dq in self.trails.items():
            base = self.colors.get(label, QColor(0, 255, 240))
            pts = list(dq)
            for width, alpha_scale, color in ((22, 0.18, base), (10, 0.55, base), (3, 1.0, QColor(255, 255, 255))):
                for (_t1, x1, y1), (t2, x2, y2) in zip(pts, pts[1:]):
                    age = max(0.0, 1.0 - (now - t2) / life)
                    c = QColor(color)
                    c.setAlphaF(min(1.0, alpha_scale * age))
                    pen = QPen(c, width * (0.4 + 0.6 * age))
                    pen.setCapStyle(Qt.RoundCap)
                    p.setPen(pen)
                    p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
            if label in self.disp:      # chấm sáng ở đầu ngón + tên cử chỉ
                x, y = self.disp[label]
                glow = QColor(base)
                glow.setAlphaF(0.35)
                p.setPen(Qt.NoPen)
                p.setBrush(glow)
                p.drawEllipse(QPointF(x, y), 14, 14)
                p.setBrush(QColor(255, 255, 255))
                p.drawEllipse(QPointF(x, y), 4, 4)
                name = self.names.get(label, "")
                if name:
                    self._tag(p, x + 20, y - 18, name, base if name != KHONG_RO else QColor(160, 160, 160))

    def _tag(self, p: QPainter, x: float, y: float, text: str, color: QColor) -> None:
        font = QFont("Segoe UI", 11, QFont.Bold)
        p.setFont(font)
        fm = QFontMetrics(font)
        rect = QRectF(x, y - fm.ascent(), fm.horizontalAdvance(text) + 14, fm.height() + 4)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 150))
        p.drawRoundedRect(rect, 6, 6)
        p.setPen(color)
        p.drawText(rect, Qt.AlignCenter, text)

    def _joystick(self, p: QPainter) -> None:
        with self.state.lock:
            js = self.state.joystick if self.state.mode == "shoot_v2" else None
        if js is None:                    # chỉ hiện khi tay trái đang DI CHUYỂN / ĐỨNG LẠI
            return
        avail = QGuiApplication.primaryScreen().availableGeometry()
        geo = self.geometry()
        r = float(js.get("radius", 0.08)) * self.height()     # cùng bán kính với lúc tính hướng đi
        cx = avail.left() - geo.left() + 30 + r
        cy = avail.bottom() - geo.top() - 30 - r
        if js.get("center") is not None:          # tâm = vị trí chấm sáng tay trái lúc bắt đầu
            cx, cy = js["center"][0] * self.width(), js["center"][1] * self.height()
        active = bool(js.get("active", True))
        keys = js["keys"]
        p.setPen(QPen(QColor(255, 60, 220, 200 if active else 90), 3))
        p.setBrush(QColor(0, 0, 0, 110))
        p.drawEllipse(QPointF(cx, cy), r, r)
        k = self.keys_cfg
        p.setFont(QFont("Segoe UI", 11, QFont.Bold))
        for key, (dx, dy) in ((k["up"], (0, -1)), (k["down"], (0, 1)), (k["left"], (-1, 0)), (k["right"], (1, 0))):
            on = key in keys
            p.setPen(QColor(255, 120, 230) if on else QColor(200, 200, 200, 140))
            p.drawText(QRectF(cx + dx * (r - 16) - 12, cy + dy * (r - 16) - 12, 24, 24), Qt.AlignCenter, key.upper())
        vx, vy = js["vec"]
        kx, ky = cx + vx * r, cy + vy * r          # núm trùng chấm sáng của ngón tay (giữ trong vòng tròn)
        knob = QColor(255, 60, 220, 230 if active else 100)
        p.setPen(Qt.NoPen)
        p.setBrush(knob)
        p.drawEllipse(QPointF(kx, ky), 20, 20)
        p.setPen(QColor(230, 230, 230))
        p.setFont(QFont("Segoe UI", 9))
        label = ("Đặt tâm: đưa tay tới chỗ thoải mái rồi khép 2 ngón" if js.get("setting")
                 else "Mất tay - đi tiếp hướng cũ" if js.get("lost")
                 else "Di chuyển (tay trái)" if active else "Đứng lại")
        p.drawText(QPointF(cx - r, cy - r - 8), label)

    def _aim_stick(self, p: QPainter) -> None:
        """V2: vòng tròn mờ ở TÂM XOAY của tay phải; vạch chỉ hướng + độ mạnh đang xoay."""
        with self.state.lock:
            st = self.state.aim_stick if self.state.mode == "shoot_v2" else None
        if st is None:
            return
        cx, cy = st["center"][0] * self.width(), st["center"][1] * self.height()
        r = st["radius"] * self.height()
        r2 = st.get("point_radius", st["radius"]) * self.height()
        rdz = st["deadzone"] * self.height()
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(0, 255, 240, 110 if st["active"] else 50), 2))      # vòng 3: xoay liên tục
        p.drawEllipse(QPointF(cx, cy), r, r)
        p.setPen(QPen(QColor(0, 255, 240, 70 if st["active"] else 35), 1))       # vòng 2: chỉ đâu nhìn đó
        p.drawEllipse(QPointF(cx, cy), r2, r2)
        p.setPen(QPen(QColor(0, 255, 240, 90), 1, Qt.DashLine))                  # vòng 1: không xoay
        p.drawEllipse(QPointF(cx, cy), rdz, rdz)
        vx, vy = st["vec"]
        if st["active"] and (vx or vy):
            p.setPen(QPen(QColor(0, 255, 240, 200), 4))
            p.drawLine(QPointF(cx, cy), QPointF(cx + vx * r, cy + vy * r))
        p.setPen(QColor(200, 255, 250, 160))
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(QPointF(cx - r, cy - r - 6), "Đặt tâm xoay: đưa tay tới chỗ thoải mái rồi mở tay"
                   if st.get("setting") else "Mất tay - xoay tiếp hướng cũ" if st.get("lost")
                   else "Xoay: vòng giữa = chỉ đâu nhìn đó, vòng ngoài = xoay liên tục")

    def _preview(self, p: QPainter) -> None:
        """Ảnh camera (đã vẽ khớp ngón) ở một góc màn hình, tránh thanh taskbar."""
        with self.state.lock:
            img = self.state.preview if self.state.show_preview else None
        if img is None:
            return
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, img.strides[0], QImage.Format_RGB888)
        avail = QGuiApplication.primaryScreen().availableGeometry()
        geo = self.geometry()
        margin = 16
        corner = self.cfg.get("preview_corner", "bottom-right")
        x = avail.right() - w - margin if "right" in corner else avail.left() + margin
        y = avail.bottom() - h - margin if "bottom" in corner else avail.top() + margin + 130
        x, y = x - geo.left(), y - geo.top()
        target = QRectF(x, y, w, h)
        p.setOpacity(float(self.cfg.get("preview_opacity", 0.92)))
        p.drawImage(target, qimg)
        p.setOpacity(1.0)
        p.setPen(QPen(QColor(0, 255, 220, 200), 2))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(target.adjusted(-1, -1, 1, 1), 4, 4)
        p.setPen(QColor(220, 255, 250))
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(QPointF(x + 4, y - 6), "Camera + khớp ngón tay (F6 ẩn/hiện)")

    def _hud(self, p: QPainter, now: float) -> None:
        with self.state.lock:
            s = self.state
            if s.mouse_mode == "auto":
                mm = "chuột tự động: " + ("TƯƠNG ĐỐI (game khóa con trỏ)" if s.auto_relative else "TUYỆT ĐỐI")
            else:
                mm = "chuột tương đối" if s.mouse_mode == "relative" else "chuột tuyệt đối"
            lines = [
                f"Chế độ: {MODE_TITLE.get(s.mode, s.mode)}  ({mm})"
                + ("" if s.enabled else "   [ĐÃ TẮT GỬI LỆNH - F9]"),
                f"Game: {s.game_name or '(chưa nhận diện - vẽ vòng tròn bằng ngón trỏ phải)'}",
                "Cử chỉ: " + (s.action_label or "-"),
                "Phím đang giữ (app gửi vào game): " + (s.held or "-")
                + ("" if s.enabled else "  [đang tắt gửi lệnh]"),
                f"Camera {s.cam_fps:.0f} FPS | Xử lý {s.vision_fps:.0f} FPS | Overlay {self.fps:.0f} FPS | "
                f"Trễ {s.latency_ms:.0f} ms | Sáng {s.brightness:.0f}{' +CLAHE' if s.clahe else ''}",
                "F5 hiệu chỉnh chạm ngón · F6 camera · F7 kiểu chuột · F8 chế độ · F9 bật/tắt · F10 thoát",
            ]
            msg = s.message if now < s.message_until else ""
            warning = s.warning
        p.setFont(QFont("Segoe UI", 10))
        x, y, lh = 16, 16, 20
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 140))
        p.drawRoundedRect(x - 8, y - 6, 640, lh * len(lines) + 12, 8, 8)
        p.setPen(QColor(220, 255, 250))
        for i, line in enumerate(lines):
            p.drawText(x, y + lh * (i + 1) - 5, line)
        # thông báo đổi cử chỉ: giữa phía trên
        if now < self.toast_until and self.toast:
            p.setFont(QFont("Segoe UI", 14, QFont.Bold))
            fm = QFontMetrics(p.font())
            tw = fm.horizontalAdvance(self.toast) + 30
            rect = QRectF((self.width() - tw) / 2, 24, tw, fm.height() + 12)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 160))
            p.drawRoundedRect(rect, 10, 10)
            p.setPen(QColor(0, 255, 200))
            p.drawText(rect, Qt.AlignCenter, self.toast)
        # cảnh báo + thông báo hệ thống: giữa phía dưới
        bottom = self.rect().adjusted(0, 0, 0, -80)
        if msg:
            p.setFont(QFont("Segoe UI", 18, QFont.Bold))
            p.setPen(QColor(0, 255, 200))
            p.drawText(bottom, Qt.AlignHCenter | Qt.AlignBottom, msg)
            bottom = bottom.adjusted(0, 0, 0, -40)
        if warning:
            p.setFont(QFont("Segoe UI", 14, QFont.Bold))
            p.setPen(QColor(255, 170, 60))
            p.drawText(bottom, Qt.AlignHCenter | Qt.AlignBottom, "⚠ " + warning)
