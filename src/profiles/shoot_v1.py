"""Chế độ BẮN V1 - bắn đứng yên (vd Rattlesnake Reflex).

Tay TRÁI:
  SÚNG    : bàn tay DỌC, ngón trỏ + giữa duỗi khép sát chĩa lên. Tâm ngắm nằm ĐÚNG CHỖ đầu ngón trỏ
            (cùng điểm với vệt sáng trên màn hình), có "dính" để tay hơi rung không làm tâm rung.
  BÓP CÒ  : ĐẦU NGÓN CÁI trái GẬP XUỐNG Ở KHỚP THỨ 2 -> bắn, còn gập thì bắn liên tục, duỗi ra thì ngừng;
            (tuỳ chọn v1_fire_on_close: ngón cái KHÉP VÀO / chạm ngón trỏ cũng bắn - mặc định TẮT vì đo trên
            video thật gây ~22% bắn nhầm, còn gập khớp thứ 2 thì 0%).
            Bắn nhanh: tâm ngắm khóa tại chỗ; giữ lâu hơn click_hold_ms: tâm đi theo tay (vừa bắn vừa kéo tâm).
Tay PHẢI (khi tay trái đang SÚNG):
  CHỈ LÊN : nắm tay, chỉ ngón trỏ thẳng lên
  MỞ NGẮM : từ CHỈ LÊN cụp ngón trỏ -> giữ chuột phải, giơ lại -> nhả
Tay PHẢI (khi tay trái không SÚNG): cử chỉ chung để bấm menu (KHÔNG có KÉO / CHÉM ở chế độ này).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..finger_state import joint_angle
from ..gestures import BOP_CO, CHI_LEN, INDEX_ONLY, KHONG_RO, MO_NGAM, SUNG, TWO_TOGETHER, Hand
from .base import CommonPointer, Desired, Profile, RelativeTracker, StickyCursor, apply_pointer


class ShootV1Profile(Profile):
    name = "shoot_v1"
    title = "BẮN V1"

    def __init__(self, cfg: dict, mouse_mode: str = "absolute"):
        super().__init__(cfg, mouse_mode)
        self.pointer = CommonPointer(cfg, allow_drag=False, allow_pause=False)
        self.aim = StickyCursor(cfg)
        self.rel = RelativeTracker()
        self.lock_after = float(cfg.get("click_hold_ms", 300)) / 1000.0   # bắn nhanh: khóa tâm; giữ lâu: tâm theo tay
        self.reset()

    def reset(self) -> None:
        self.pointer.reset()
        self.aim.reset()
        self.rel.reset()
        self.left_state = "none"         # none | gun | fire
        self.fire_start = 0.0
        self.thumb_was_open = False
        self.gun_start = 0.0
        self.thumb_hist: deque[tuple[float, float]] = deque(maxlen=30)
        self.bent = False
        self.right_state = "none"        # none | ready | scope

    # ---------- tay trái ----------
    def _left(self, h: Hand | None, t: float, d: Desired) -> bool:
        if h is None:
            self.left_state, self.thumb_was_open, self.bent = "none", False, False
            self.thumb_hist.clear()
            self.aim.reset()
            return False
        two = h.index and h.middle and not h.ring and not h.pinky
        if self.left_state in ("gun", "fire") and two:
            pass                                             # đang SÚNG: giữ nguyên cả khi tay hơi nghiêng
        elif h.pose == TWO_TOGETHER and h.vertical:
            self.left_state = "gun"
            self.gun_start = t
            self.thumb_hist.clear()
            self.bent = False
        else:
            self.left_state, self.thumb_was_open, self.bent = "none", False, False
            self.thumb_hist.clear()
            self.aim.reset()
            d.gestures["Left"] = KHONG_RO
            return False

        # BÓP CÒ (1): ngón cái KHÉP VÀO (chạm / tới gần ngón trỏ, hoặc gập) -> giữ bắn. Phải thấy ngón cái mở ra
        # ít nhất 1 lần trước -> giơ súng với ngón cái khép sẵn không tự bắn.
        # NGÓN CÁI GẬP VÀO LÒNG BÀN TAY (nét đỏ) hoặc CHẠM ngón trỏ -> bắn. Không dùng "sắp chạm" (h.approach): ngưỡng
        # đó lỏng, gây phần lớn số lần bắn nhầm khi đo trên video.
        closed = bool(self.cfg.get("v1_fire_on_close", True)) and (h.touch == "index" or not h.thumb)
        if not closed:
            self.thumb_was_open = True
        # BÓP CÒ (2): ĐẦU NGÓN CÁI GẬP XUỐNG Ở KHỚP THỨ 2 -> bắn, còn gập thì bắn liên tục (xem _thumb_bend)
        firing = (closed and self.thumb_was_open) or self._thumb_bend(h, t)
        if firing and self.left_state != "fire":
            self.fire_start = t
        self.left_state = "fire" if firing else "gun"
        lock = firing and t - self.fire_start < self.lock_after
        pos = self.aim.update(h.tip, t, lock=lock)
        if self.mouse_mode == "relative":
            d.cursor_rel = self.rel.delta(pos)
        else:
            d.cursor_abs = pos
        if firing:
            d.buttons.add("left")
            d.gestures["Left"] = BOP_CO
        else:
            d.gestures["Left"] = SUNG
        return True

    def _thumb_bend(self, h: Hand, t: float) -> bool:
        """ĐẦU NGÓN CÁI GẬP XUỐNG Ở KHỚP THỨ 2 (khớp IP, điểm 3)?

        Góc khớp IP = góc (điểm 2 -> 3 -> 4) trên toạ độ 3D thật -> không đổi khi nghiêng / xoay / dời cả bàn tay.
        Bắn khi góc GIẢM ít nhất v1_bend_drop độ so với lúc thẳng nhất trong 0.5 s gần đây VÀ xuống dưới
        v1_bend_below độ; giữ bắn khi còn gập, duỗi ra quá v1_bend_below + 10 độ thì nhả.
        (Đo trên video thật lúc KHÔNG bắn: 35 độ / 130 độ -> tự bắn nhầm ~1.8% frame khi tay di chuyển mạnh.)
        """
        src = h.world if h.world is not None else h.cam
        if src is None or len(src) < 21:
            return False
        p = np.asarray(src, dtype=float)
        a = joint_angle(p[2], p[3], p[4])
        if self.thumb_hist:
            a = 0.5 * a + 0.5 * self.thumb_hist[-1][1]          # làm mượt nhẹ
        self.thumb_hist.append((t, a))
        while self.thumb_hist and t - self.thumb_hist[0][0] > 0.5:
            self.thumb_hist.popleft()
        below = float(self.cfg.get("v1_bend_below", 130))
        if self.bent:
            self.bent = a < below + 10
        elif t - self.gun_start >= float(self.cfg.get("v1_twitch_warmup_ms", 300)) / 1000.0:
            straight = max(x[1] for x in self.thumb_hist)
            self.bent = straight - a >= float(self.cfg.get("v1_bend_drop", 35)) and a < below
        return self.bent

    # ---------- tay phải khi đang ngắm ----------
    def _right_scope(self, h: Hand | None, d: Desired) -> None:
        if h is None:
            self.right_state = "none"
            return
        if self.right_state == "scope":
            if not h.index:
                d.buttons.add("right")
                d.gestures["Right"] = MO_NGAM
                return
            self.right_state = "ready"
        if self.right_state == "ready" and not h.index and not h.middle:     # cụp ngón trỏ -> MỞ NGẮM
            self.right_state = "scope"
            d.buttons.add("right")
            d.gestures["Right"] = MO_NGAM
        elif (h.pose == INDEX_ONLY and h.index) or (self.right_state == "ready" and h.index and not h.middle):
            self.right_state = "ready"
            d.gestures["Right"] = CHI_LEN
        else:
            self.right_state = "none"
            d.gestures["Right"] = KHONG_RO

    def update(self, hands: dict[str, Hand], t: float) -> Desired:
        d = Desired()
        aiming = self._left(hands.get("Left"), t, d)
        right = hands.get("Right")
        if aiming:
            self.pointer.reset()
            self._right_scope(right, d)
        else:
            self.right_state = "none"
            out = self.pointer.update(right, t)
            apply_pointer(d, out, self.mouse_mode, self.rel)
            if right is not None:
                d.gestures["Right"] = out.name
        return d
