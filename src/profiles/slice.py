"""Chế độ CHÉM - chỉ đọc tay PHẢI, dùng toàn bộ cử chỉ chung.

  CON TRỎ / CLICK TRÁI / CLICK PHẢI / TẠM DỪNG như cử chỉ chung
  KÉO / CHÉM : khép ngón trỏ + giữa -> giữ chuột trái và di chuyển = chém
"""
from __future__ import annotations

from ..gestures import Hand
from .base import CommonPointer, Desired, Profile, RelativeTracker, apply_pointer


class SliceProfile(Profile):
    name = "slice"
    title = "CHÉM"

    def __init__(self, cfg: dict, mouse_mode: str = "absolute"):
        super().__init__(cfg, mouse_mode)
        self.pointer = CommonPointer(cfg)
        self.rel = RelativeTracker()

    def reset(self) -> None:
        self.pointer.reset()
        self.rel.reset()

    def update(self, hands: dict[str, Hand], t: float) -> Desired:
        d = Desired()
        r = hands.get("Right")
        out = self.pointer.update(r, t)
        apply_pointer(d, out, self.mouse_mode, self.rel)
        if r is not None:
            d.gestures["Right"] = out.name
        return d
