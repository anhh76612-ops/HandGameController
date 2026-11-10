"""Chế độ BẮN V2 - bắn có di chuyển nhân vật (vd Bodycamera Shooter).

CHUỘT TỰ CHỌN: con trỏ Windows đang hiện (menu, sảnh) -> CHUỘT TUYỆT ĐỐI, con trỏ nằm đúng chỗ ngón tay;
game đã khóa/ẩn con trỏ (đang trong trận) -> CHUỘT TƯƠNG ĐỐI để xoay màn hình (tâm ngắm ở giữa).
Tay PHẢI (không vẽ vệt sáng trên màn hình ở chế độ này):
  XOAY      : giơ ngón trỏ. Trong trận - 3 VÒNG quanh tâm tay (docs/04): (1) vòng nhỏ: không xoay;
              (2) vòng giữa: chỉ đâu nhìn đó; (3) vòng ngoài: xoay liên tục về hướng tay.
              Mất tay khi đang xoay ở vòng ngoài -> xoay tiếp theo hướng cũ tối đa lost_keep_sec giây.
              (aim_style: delta -> kiểu chuột thật + đẩy mép như bản cũ)
  NẠP ĐẠN   : xòe cả bàn tay phải -> nhấn R (1 lần mỗi lần xòe)
  ĐẶT TÂM   : nắm cả bàn tay phải -> xóa tâm xoay cũ, không xoay; mở ra giơ ngón trỏ -> chấm khi đó thành tâm mới
  NGẮM      : KHÉP ngón trỏ + giữa -> giữ chuột phải (vẫn xoay được trong lúc ngắm)
  BẮN       : KHÉP ngón cái lại (chạm / tới gần ngón trỏ, hoặc gập) -> chuột trái. Phải thấy ngón cái mở ra 1 lần
              trước. Vẫn xoay được trong lúc bắn.
Không có TẠM DỪNG (Esc) ở chế độ bắn.
Tay TRÁI:
  DI CHUYỂN : KHÉP ngón trỏ + giữa. Lần đầu làm cử chỉ -> LƯU VỊ TRÍ GỐC của tay (tâm lòng bàn tay) và vẽ
              cần điều khiển ngay tại đó: TÂM = vị trí chấm sáng (đầu ngón trỏ trái) lúc bắt đầu, NÚM = chấm sáng
              hiện tại. Hướng đi = hướng từ tâm tới chấm (lên = W, xuống = S, trái = A, ...); dịch tay hay nghiêng
              ngón đều làm chấm đi. Chấm còn gần tâm -> mặc định đi thẳng (W).
              8 hướng, có vùng trễ ±8° để không nhảy qua lại. Gốc giữ nguyên tới khi mất tay trái quá anchor_sec.
  ĐỨNG LẠI  : XÒE cả bàn tay trái -> nhả W/A/S/D (giữ nguyên tâm)
  ĐẶT TÂM   : nắm tay -> nhả W/A/S/D VÀ ĐẶT LẠI TÂM (vòng tròn đi theo bàn tay; mở ra khép 2 ngón -> tâm mới)
  NHẢY      : đang DI CHUYỂN thì khép ngón cái lại (chạm / tới gần ngón trỏ, hoặc gập) -> giữ Space
  không thấy tay trái khi đang đi -> đi tiếp theo hướng lần cuối tối đa lost_keep_sec giây, rồi đứng yên
"""
from __future__ import annotations

import math

import numpy as np

from ..gestures import (
    BAN, DAT_TAM, DI_CHUYEN, DUNG_LAI, FIST, KHONG_RO, NAP_DAN, NGAM, NHAY, OPEN, POINTER, TWO_TOGETHER, XOAY, Hand,
)
from .base import Desired, Profile, RelativeTracker, StickyCursor, fine_aim



# (góc bắt đầu, góc kết thúc) -> hướng; góc đo từ "thẳng lên" (= hướng gốc), dương = sang phải
SECTORS = [
    (-22.5, 22.5, ("up",)), (22.5, 67.5, ("up", "right")), (67.5, 112.5, ("right",)),
    (112.5, 157.5, ("down", "right")), (157.5, 202.5, ("down",)), (-157.5, -112.5, ("down", "left")),
    (-112.5, -67.5, ("left",)), (-67.5, -22.5, ("up", "left")),
]


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def sector_of(angle: float, current: int | None, hyst: float) -> int:
    """Chọn 1 trong 8 hướng; giữ hướng hiện tại nếu góc còn trong vùng đó nới rộng thêm `hyst` độ."""
    if current is not None:
        lo, hi, _ = SECTORS[current]
        if _wrap(angle - (lo + hi) / 2) ** 2 <= ((hi - lo) / 2 + hyst) ** 2:
            return current
    for i, (lo, hi, _) in enumerate(SECTORS):
        if _wrap(angle - (lo + hi) / 2) ** 2 <= ((hi - lo) / 2) ** 2:
            return i
    return 0


class ShootV2Profile(Profile):
    name = "shoot_v2"
    title = "BẮN V2"

    def __init__(self, cfg: dict, mouse_mode: str = "auto"):
        super().__init__(cfg, "auto")                       # V2: tự chọn tuyệt đối / tương đối
        self.rel = RelativeTracker()
        self.cursor = StickyCursor(cfg)
        self.hold_after = float(cfg.get("click_hold_ms", 300)) / 1000.0
        self.reset()

    def reset(self) -> None:
        self.rel.reset()
        self.cursor.reset()
        self.r_thumb_was_open = False
        self.r_open = False
        self.right_seen = -1e9
        self.aim_anchor: np.ndarray | None = None       # tâm xoay của tay phải (kiểu cần gạt)
        self._aim_vec = np.zeros(2)
        self.fire_start: float | None = None
        self.last_t: float | None = None
        self.left_state = "none"           # none | move | stop | reload
        self.anchor: np.ndarray | None = None          # vị trí gốc của tay trái (pixel ảnh)
        self.anchor_screen: tuple[float, float] | None = None
        self.left_seen = -1e9
        self.ref: np.ndarray | None = None              # hướng ngón lúc bắt đầu
        self.dir_vec: np.ndarray | None = None
        self.sector: int | None = None
        self.thumb_was_open = False
        self.p_prev: np.ndarray | None = None          # vòng 2: vị trí "chỉ đâu nhìn đó" ở frame trước
        self._settle_prev: tuple[float, np.ndarray] | None = None
        self._settle_since: float | None = None
        self.last_vel = (0.0, 0.0)                      # vận tốc xoay lần cuối (để xoay tiếp khi mất tay)
        self.last_move_keys: set[str] = set()           # phím di chuyển lần cuối (để đi tiếp khi mất tay)
        self.last_knob = (0.0, 0.0)

    # ---------- tay trái ----------
    def _direction(self, h: Hand) -> tuple[float, tuple[float, float]]:
        """Hướng đi theo CHẤM SÁNG của đầu ngón trỏ trái so với TÂM cần điều khiển (vị trí chấm lúc bắt đầu).

        Đo trên màn hình, đổi về đơn vị "chiều cao màn hình" (nhân x với tỉ lệ khung) để trái/phải và lên/xuống
        cùng thang. Núm = đúng vị trí chấm (bị giữ trong vòng tròn) -> núm và ngón tay luôn đi cùng hướng.
        Nghiêng ngón cũng làm đầu ngón dịch đi nên vẫn đổi hướng được. Chấm còn trong vùng chết -> đi thẳng (W).
        """
        aspect = float(self.cfg.get("screen_aspect", 16 / 9))
        radius = float(self.cfg.get("joystick_radius", 0.08))
        k = self._ctrl(h)
        off = np.array([(float(k[0]) - self.anchor[0]) * aspect, float(k[1]) - self.anchor[1]])
        n = float(np.linalg.norm(off))
        vec = off / radius
        if float(np.linalg.norm(vec)) > 1.0:
            vec = vec / float(np.linalg.norm(vec))
        if n <= float(self.cfg["joystick_deadzone"]) * radius:
            return 0.0, (float(vec[0]), float(vec[1]))
        return math.degrees(math.atan2(float(off[0]), float(-off[1]))), (float(vec[0]), float(vec[1]))

    def _left(self, h: Hand | None, t: float, d: Desired) -> None:
        keys = self.cfg["keys"]
        if h is None:
            # MẤT TAY khi đang đi -> đi tiếp theo hướng lần cuối (tối đa lost_keep_sec giây)
            if (self.left_state == "move" and self.last_move_keys
                    and t - self.left_seen <= float(self.cfg.get("lost_keep_sec", 3.0))):
                d.keys |= self.last_move_keys
                d.gestures["Left"] = "ĐI TIẾP (mất tay)"
                self._show_joystick(d, True, self.last_knob, self.last_move_keys, lost=True)
                return
            if t - self.left_seen > float(self.cfg.get("anchor_sec", 3.0)):
                self.anchor, self.anchor_screen, self.ref = None, None, None
            self.left_state, self.sector, self.thumb_was_open = "none", None, False
            self.last_move_keys = set()
            return
        self.left_seen = t

        # XÒE CẢ BÀN TAY = ĐỨNG LẠI (giữ nguyên tâm cần điều khiển)
        if h.pose == OPEN and h.index and h.middle and h.ring and h.pinky:
            self.left_state, self.sector = "stop", None
            d.gestures["Left"] = DUNG_LAI
            self._show_joystick(d, None, (0.0, 0.0), set())
            return
        # NẮM TAY = ĐỨNG LẠI + ĐẶT TÂM: tâm đi theo ĐỐT NGÓN trong lúc nắm; mở ra khép 2 ngón thì tâm GIỮ NGUYÊN
        # ở chỗ đó (đốt ngón gần như không đổi khi nắm/mở -> nút di chuyển không nhảy)
        if h.pose == FIST and not (h.index or h.middle):
            self.left_state, self.sector = "stop", None
            k = self._ctrl(h)
            self.anchor = np.array([float(k[0]), float(k[1])])
            self.anchor_screen = (float(k[0]), float(k[1]))
            d.gestures["Left"] = f"{DUNG_LAI} + {DAT_TAM}"
            d.joystick = {"vec": (0.0, 0.0), "keys": set(), "active": False, "setting": True,
                          "center": self.anchor_screen, "radius": float(self.cfg.get("joystick_radius", 0.08))}
            return

        moving = self.left_state == "move"
        two = h.index and h.middle and not h.ring and not h.pinky
        # bắt đầu đi khi 2 ngón KHÉP hoặc HƠI TÁCH (trước đây bắt buộc khép sát -> hơi hở là đứng im)
        if not (two and (h.pose in (TWO_TOGETHER, POINTER) or moving)):
            self.left_state, self.sector = "none", None
            d.gestures["Left"] = KHONG_RO
            self._show_joystick(d, None, (0.0, 0.0), set())
            return

        if self.anchor is None:                                   # LƯU VỊ TRÍ GỐC = đốt ngón lúc bắt đầu
            k = self._ctrl(h)
            self.anchor = np.array([float(k[0]), float(k[1])])
            self.anchor_screen = (float(k[0]), float(k[1]))
        if not moving:
            self.ref, self.dir_vec, self.sector = None, None, None
            self.thumb_was_open = False
        self.left_state = "move"
        ang, knob = self._direction(h)
        self.sector = sector_of(ang, self.sector, float(self.cfg.get("dir_hyst_deg", 8)))
        move = {keys[k] for k in SECTORS[self.sector][2]}
        d.keys |= move
        self.last_move_keys, self.last_knob = set(move), knob
        d.gestures["Left"] = DI_CHUYEN
        # NHẢY: khép ngón cái lại (phải thấy ngón cái mở ra trước)
        closed = h.touch == "index" or h.approach or not h.thumb
        if not closed:
            self.thumb_was_open = True
        if closed and self.thumb_was_open:
            d.keys.add(keys["jump"])
            d.gestures["Left"] = NHAY
        self._show_joystick(d, True, knob, move)

    def _show_joystick(self, d: Desired, active, knob, keys: set, lost: bool = False) -> None:
        """Cần điều khiển chỉ hiện khi đã có vị trí gốc; vẽ tại vị trí gốc của tay trên màn hình."""
        if self.anchor_screen is None:
            return
        d.joystick = {"vec": knob, "keys": set(keys), "active": bool(active), "center": self.anchor_screen,
                      "radius": float(self.cfg.get("joystick_radius", 0.08)), "lost": lost}

    # ---------- tay phải ----------
    def _right(self, h: Hand | None, t: float, dt: float, d: Desired) -> None:
        keys = self.cfg["keys"]
        if h is None:
            self.p_prev = None                                   # tay xuất hiện lại -> đồng bộ, không giật
            self.rel.reset()
            self.cursor.reset()
            self.r_thumb_was_open, self.fire_start, self.r_open = False, None, False
            # MẤT TAY khi đang xoay ở vòng ngoài -> xoay tiếp theo hướng lần cuối (tối đa lost_keep_sec giây)
            if (self.last_vel != (0.0, 0.0) and t - self.right_seen <= float(self.cfg.get("lost_keep_sec", 3.0))):
                d.cursor_vel, d.cursor_auto = self.last_vel, True
                d.gestures["Right"] = "XOAY TIẾP (mất tay)"
                self._show_aim(d, True, lost=True)
                return
            self.last_vel = (0.0, 0.0)
            if t - self.right_seen > float(self.cfg.get("anchor_sec", 3.0)):
                self.aim_anchor = None
            return
        self.right_seen = t
        # NẠP ĐẠN: xòe cả bàn tay phải (gửi R 1 lần mỗi lần xòe), không xoay
        if h.pose == OPEN and h.index and h.middle and h.ring and h.pinky:
            if not self.r_open:
                d.taps.append(keys["reload"])
            self.r_open = True
            d.gestures["Right"] = NAP_DAN
            self.rel.reset()
            self.p_prev, self.last_vel = None, (0.0, 0.0)
            self._show_aim(d, None)
            return
        self.r_open = False
        if h.pose == FIST and not (h.index or h.middle):
            # NẮM TAY = ĐẶT TÂM XOAY: chỉ để chọn chỗ đặt tay thoải mái. Tâm (theo ĐỐT NGÓN) đi theo tay trong lúc
            # nắm và giữ nguyên khi mở ra; vòng tròn trên màn hình vẫn GHIM Ở GIỮA (chỗ tâm ngắm). Không xoay.
            k = self._ctrl(h)
            self.aim_anchor = np.array([float(k[0]), float(k[1])])
            self._aim_vec = np.zeros(2)
            self.p_prev, self.last_vel = None, (0.0, 0.0)
            self.rel.reset()
            self.cursor.reset()
            self.r_thumb_was_open, self.fire_start = False, None
            d.gestures["Right"] = DAT_TAM
            self._show_aim(d, None, setting=True)
            return
        if not h.index:
            self.rel.reset()
            self.cursor.reset()
            self.r_thumb_was_open, self.fire_start = False, None
            self.p_prev, self.last_vel = None, (0.0, 0.0)
            d.gestures["Right"] = KHONG_RO
            self._show_aim(d, None)
            return
        aim = h.middle and h.together                        # KHÉP trỏ + giữa -> NGẮM (chuột phải)
        closed = h.touch == "index" or h.approach or not h.thumb
        if not closed:
            self.r_thumb_was_open = True
        fire = closed and self.r_thumb_was_open              # KHÉP ngón cái -> BẮN (chuột trái)
        if fire and self.fire_start is None:
            self.fire_start = t
        elif not fire:
            self.fire_start = None
        fire_lock = False                                    # vẫn xoay được trong lúc bắn

        names = []
        if aim:
            d.buttons.add("right")
            names.append(NGAM)
        if fire:
            d.buttons.add("left")
            names.append(BAN)
        d.gestures["Right"] = " + ".join(names) if names else XOAY

        # Gửi CẢ 2: vị trí tuyệt đối (menu, con trỏ hiện) và dịch chuyển tương đối (trong trận, game khóa
        # con trỏ). Controller tự chọn theo việc con trỏ Windows đang bị ẩn/khóa hay không (cursor_auto).
        d.cursor_abs = self.cursor.update(h.tip, t, lock=fire_lock)
        d.cursor_auto = True
        vscale = float(self.cfg.get("aim_vertical_scale", 0.1))  # xoay LÊN/XUỐNG chậm hơn hẳn chiều ngang
        if self.cfg.get("aim_style", "rate") == "delta":
            dx, dy = self._delta_aim(h, dt, fire_lock)
            d.cursor_rel = (dx, dy * vscale)
        else:
            # 3 vòng: (1) vùng chết, (2) "chỉ đâu nhìn đó" -> dịch chuyển tương đối, (3) xoay liên tục -> VẬN TỐC
            # (vận tốc được luồng 120 Hz cộng dồn đều mỗi ~8 ms -> quay đều, không giật theo nhịp camera)
            (rx, ry), (vx, vy) = self._zone_aim(h, fire_lock, t)
            d.cursor_rel = (rx, ry * vscale)
            d.cursor_vel = (vx, vy * vscale)
            self.last_vel = d.cursor_vel

    def _zone_aim(self, h: Hand, frozen: bool, t: float = 0.0) -> tuple[tuple[float, float], tuple[float, float]]:
        """XOAY 3 VÒNG quanh tâm tay phải (đo theo đốt ngón, đơn vị = chiều cao màn hình):

        (1) |lệch| < aim_deadzone            : KHÔNG xoay (tay dao động nhẹ, chống giật).
        (2) aim_deadzone .. aim_point_radius : "CHỈ ĐÂU NHÌN ĐÓ" - mỗi vị trí tay ứng với một điểm trên màn hình
            (tâm màn hình + độ lệch); màn hình xoay để đưa điểm đó về tâm ngắm. Gửi DỊCH CHUYỂN = thay đổi vị trí
            tay x aim_point_gain -> tay đứng yên thì màn hình dừng, đưa tay về thì màn hình quay về.
            Khi màn hình đã xoay tới chỗ tay chỉ và TAY ĐỨNG YÊN (aim_settle_ms), chỗ tay đang đặt tự thành TÂM MỚI
            (giống nắm tay đặt tâm) -> vạch chỉ hướng biến mất, lần sau tính từ đây.
        (3) > aim_point_radius               : XOAY LIÊN TỤC về hướng tay (vòng này không tự đặt lại tâm); tốc độ tăng dần tới aim_max_speed ở
            aim_radius (vòng ngoài cùng): tốc_độ = max * ((lệch - r2) / (r3 - r2)) ^ aim_curve.
        Trả (dịch chuyển tương đối, vận tốc) theo đơn vị màn hình.
        """
        # tỉ lệ khung của VÙNG ĐIỀU KHIỂN trên camera (không phải màn hình): cùng quãng tay thật -> cùng độ lệch
        # ở mọi hướng -> xoay ngang / dọc / chéo nhanh như nhau
        aspect = float(self.cfg.get("control_aspect", self.cfg.get("screen_aspect", 16 / 9)))
        k = self._ctrl(h)
        if self.aim_anchor is None:
            self.aim_anchor = np.array([float(k[0]), float(k[1])])
        off = np.array([(float(k[0]) - self.aim_anchor[0]) * aspect, float(k[1]) - self.aim_anchor[1]])
        dist = float(np.linalg.norm(off))
        direction = off / dist if dist > 1e-9 else np.zeros(2)
        dz = float(self.cfg.get("aim_deadzone", 0.02))
        r2 = float(self.cfg.get("aim_point_radius", 0.20))
        r3 = float(self.cfg.get("aim_radius", 0.32))
        self._aim_vec = direction * min(dist / r3, 1.0)
        # vòng 2: vị trí "chỉ đâu nhìn đó" (bỏ phần vùng chết, giới hạn ở r2)
        p = direction * max(0.0, min(dist, r2) - dz)
        if self.p_prev is None or frozen:
            delta = np.zeros(2)                               # đồng bộ (vừa xuất hiện / đang bắn) -> không nhảy
        else:
            delta = p - self.p_prev
        self.p_prev = p
        gain = float(self.cfg.get("aim_point_gain", 1.0))
        rel = (float(delta[0] * gain), float(delta[1] * gain))
        # vòng 2: tay đã đứng yên -> chỗ đặt tay thành TÂM MỚI
        cur = np.array([float(k[0]) * aspect, float(k[1])])
        speed_hand = 0.0
        if self._settle_prev is not None and t > self._settle_prev[0]:
            speed_hand = float(np.linalg.norm(cur - self._settle_prev[1])) / (t - self._settle_prev[0])
        self._settle_prev = (t, cur)
        if dz < dist <= r2 and speed_hand < float(self.cfg.get("aim_settle_speed", 0.15)):
            if self._settle_since is None:
                self._settle_since = t
            elif t - self._settle_since >= float(self.cfg.get("aim_settle_ms", 250)) / 1000.0:
                self.aim_anchor = np.array([float(k[0]), float(k[1])])
                self.p_prev = np.zeros(2)
                self._aim_vec = np.zeros(2)
                self._settle_since = None
        else:
            self._settle_since = None
        # vòng 3: xoay liên tục
        vel = (0.0, 0.0)
        if dist > r2 and not frozen:
            kk = min((dist - r2) / max(r3 - r2, 1e-6), 1.0) ** float(self.cfg.get("aim_curve", 1.7))
            speed = float(self.cfg.get("aim_max_speed", 0.8)) * kk
            vel = (float(direction[0] * speed), float(direction[1] * speed))
        return rel, vel

    def _delta_aim(self, h: Hand, dt: float, frozen: bool) -> tuple[float, float]:
        """KIỂU CHUỘT THẬT + ĐẨY MÉP (docs/04, cách 1 + 3)."""
        dx, dy = self.rel.delta(h.tip)
        if frozen:
            return 0.0, 0.0
        dx, dy = fine_aim(dx, dy, dt, self.cfg)
        zone, speed = float(self.cfg.get("edge_zone", 0.08)), float(self.cfg.get("edge_speed", 0.5))
        x, y = float(h.tip[0]), float(h.tip[1])
        if x < zone:
            dx -= speed * dt * (zone - x) / zone
        elif x > 1 - zone:
            dx += speed * dt * (x - (1 - zone)) / zone
        if y < zone:
            dy -= speed * dt * (zone - y) / zone
        elif y > 1 - zone:
            dy += speed * dt * (y - (1 - zone)) / zone
        return dx, dy

    @staticmethod
    def _ctrl(h: Hand) -> np.ndarray:
        """Điểm điều khiển của V2: ĐỐT NGÓN (khớp gốc trỏ + giữa) - không nhảy khi nắm/mở tay."""
        return h.knuckle_screen if h.knuckle_screen is not None else h.tip

    def _show_aim(self, d: Desired, active, setting: bool = False, lost: bool = False) -> None:
        """3 vòng xoay GHIM GIỮA MÀN HÌNH (chỗ tâm ngắm); vạch = hướng + độ lệch của tay so với tâm."""
        if self.aim_anchor is None or self.cfg.get("aim_style", "rate") == "delta":
            return
        d.aim_stick = {"center": (0.5, 0.5), "setting": setting, "lost": lost,
                       "vec": tuple(float(v) for v in self._aim_vec) if active else (0.0, 0.0),
                       "radius": float(self.cfg.get("aim_radius", 0.32)),
                       "point_radius": float(self.cfg.get("aim_point_radius", 0.20)),
                       "deadzone": float(self.cfg.get("aim_deadzone", 0.02)), "active": bool(active)}

    def update(self, hands: dict[str, Hand], t: float) -> Desired:
        d = Desired()
        dt = 1 / 30 if self.last_t is None else min(max(t - self.last_t, 1e-3), 0.1)
        self.last_t = t
        self._left(hands.get("Left"), t, d)
        self._right(hands.get("Right"), t, dt, d)
        if "Right" in hands and d.gestures.get("Right") not in (None, KHONG_RO, NAP_DAN, DAT_TAM):
            self._show_aim(d, True)
        return d
