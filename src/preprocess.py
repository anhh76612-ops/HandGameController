"""Tiền xử lý ảnh trước khi đưa vào MediaPipe.

Các bước (mỗi bước đều có thể minh họa bằng tools/debug_view.py):
1. Lật gương (cv2.flip): người dùng nhìn màn hình như nhìn gương -> tay phải di chuyển sang phải.
2. Đo độ sáng: chuyển BGR -> LAB, lấy trung bình kênh L (độ sáng cảm nhận, tách khỏi màu).
3. Nếu tối: CLAHE (cân bằng histogram thích nghi có giới hạn tương phản) trên kênh L,
   giữ nguyên kênh màu a, b -> tăng tương phản cục bộ mà không làm lệch màu da.
4. BGR -> RGB vì MediaPipe nhận ảnh RGB.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    bgr: np.ndarray          # ảnh sau lật (+ CLAHE nếu có), dùng để hiển thị
    rgb: np.ndarray          # ảnh đưa vào MediaPipe
    brightness: float        # độ sáng trung bình kênh L trước khi xử lý
    clahe_applied: bool


class Preprocessor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.clahe = cv2.createCLAHE(
            clipLimit=cfg["clahe_clip_limit"],
            tileGridSize=(cfg["clahe_grid"], cfg["clahe_grid"]),
        )

    def __call__(self, frame: np.ndarray) -> PreprocessResult:
        img = cv2.flip(frame, 1) if self.cfg.get("flip", True) else frame
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        brightness = float(l_ch.mean())
        applied = False
        if self.cfg.get("clahe_auto", True) and brightness < self.cfg["clahe_brightness_threshold"]:
            l_ch = self.clahe.apply(l_ch)
            img = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
            applied = True
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return PreprocessResult(img, rgb, brightness, applied)
