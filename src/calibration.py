"""Hiệu chỉnh ngưỡng CHẠM NGÓN CÁI theo bàn tay người dùng (phím F5).

Bước 1 (2.5 s): tay PHẢI giơ ngón trỏ, TÁCH ngón cái ra xa.
Bước 2 (2.5 s): giữ ngón trỏ, CHẠM đầu ngón cái vào ngón trỏ.
Lấy trung vị khoảng cách (đầu ngón cái -> đường ngón trỏ) trên ảnh 2D và 3D của mỗi bước rồi đặt
(3D chỉ là điều kiện lỏng vì độ sâu của webcam 1 mắt nhiễu):
  ngưỡng CHẠM  = chạm + 40% khoảng cách (tách - chạm)
  ngưỡng NHẢ   = chạm + 60%            (giữa 2 ngưỡng = hysteresis)
  ngưỡng SẮP CHẠM (khóa con trỏ) = chạm + 85%
Kết quả lưu vào calibration.json (không sửa config.yaml để giữ nguyên chú thích), tự nạp ở lần chạy sau.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .finger_state import FingerMeasure

STEPS = [
    ("away", "Hiệu chỉnh 1/2: tay PHẢI giơ ngón trỏ, TÁCH ngón cái ra xa", 2.5),
    ("touch", "Hiệu chỉnh 2/2: CHẠM đầu ngón cái vào ngón trỏ và giữ yên", 2.5),
]
KEYS = ("touch_on_2d", "touch_on_3d", "touch_off_2d", "touch_off_3d", "approach_2d")


def load_calibration(path: Path, touch_cfg: dict) -> bool:
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    if not all(k in data for k in KEYS):          # file của phiên bản cũ (ngưỡng gập ngón) -> bỏ qua
        return False
    for k in KEYS:
        touch_cfg[k] = float(data[k])
    return True


class Calibrator:
    def __init__(self, path: Path, touch_cfg: dict, notify):
        self.path, self.cfg, self.notify = path, touch_cfg, notify
        self.active = False

    def start(self, t: float) -> None:
        self.active, self.step, self.step_start = True, 0, t
        self.samples: dict[str, list[tuple[float, float]]] = {"away": [], "touch": []}
        self.notify(STEPS[0][1], seconds=STEPS[0][2] + 0.5)

    def update(self, m: FingerMeasure | None, t: float) -> None:
        if not self.active:
            return
        key, _msg, dur = STEPS[self.step]
        if m is not None and t - self.step_start > 0.6:          # bỏ 0.6 s đầu để người dùng kịp làm
            self.samples[key].append((m.touch_index_2d, m.touch_index_3d))
        if t - self.step_start >= dur:
            self.step += 1
            self.step_start = t
            if self.step < len(STEPS):
                self.notify(STEPS[self.step][1], seconds=STEPS[self.step][2] + 0.5)
            else:
                self.active = False
                self._finish()

    @staticmethod
    def compute(away: list, touch: list) -> dict | None:
        if len(away) < 5 or len(touch) < 5:
            return None
        a, c = np.median(np.array(away), axis=0), np.median(np.array(touch), axis=0)
        if a[0] - c[0] < 0.1:                                   # 2D khác biệt quá nhỏ -> làm sai bước
            return None
        r = lambda x: round(float(x), 3)  # noqa: E731
        # 3D (độ sâu) nhiễu -> chỉ dùng làm điều kiện lỏng: ít nhất "lúc chạm + 0.15"
        on3 = max(c[1] + 0.8 * max(a[1] - c[1], 0.0), c[1] + 0.15)
        return {
            "touch_on_2d": r(c[0] + 0.40 * (a[0] - c[0])), "touch_on_3d": r(on3),
            "touch_off_2d": r(c[0] + 0.60 * (a[0] - c[0])), "touch_off_3d": r(on3 + 0.1),
            "approach_2d": r(c[0] + 0.85 * (a[0] - c[0])),
        }

    def _finish(self) -> None:
        res = self.compute(self.samples["away"], self.samples["touch"])
        if res is None:
            self.notify("Hiệu chỉnh thất bại - thấy tay chưa rõ hoặc làm chưa đúng bước, bấm F5 thử lại", 5)
            return
        self.cfg.update(res)
        self.path.write_text(json.dumps(res, indent=1), encoding="utf-8")
        self.notify(f"Đã hiệu chỉnh chạm ngón: chạm < {res['touch_on_2d']:.2f}, nhả > {res['touch_off_2d']:.2f}", 5)
