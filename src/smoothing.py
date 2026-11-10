"""Làm mượt tọa độ và ánh xạ camera -> màn hình.

One Euro filter (Casiez et al., 2012): bộ lọc thông thấp có tần số cắt THAY ĐỔI theo tốc độ.
- Tay đứng yên: tần số cắt thấp -> lọc mạnh -> hết rung.
- Tay di chuyển nhanh: tần số cắt tăng -> lọc nhẹ -> ít trễ (quan trọng khi chém/ngắm).
  cutoff = min_cutoff + beta * |tốc độ|
  alpha  = 1 / (1 + tau/dt), tau = 1 / (2*pi*cutoff)
  x_hat  = alpha * x + (1 - alpha) * x_hat_prev
"""
from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Lọc đồng thời cả một mảng (vd 21 điểm x 2 tọa độ)."""

    def __init__(self, min_cutoff=1.5, beta=8.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None
        self.t_prev: float | None = None

    def reset(self) -> None:
        self.x_prev = self.dx_prev = self.t_prev = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.x_prev is None or self.t_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        dx_hat = self.dx_prev + _alpha(self.d_cutoff, dt) * (dx - self.dx_prev)
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = self.x_prev + a * (x - self.x_prev)
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class ScreenMapper:
    """Ánh xạ tuyến tính vùng hoạt động trong khung camera -> toàn màn hình (chuẩn hóa 0..1).

    Đầu ra chuẩn hóa để overlay (pixel logic của Qt) và controller (pixel vật lý của Windows)
    tự nhân với kích thước màn hình riêng của mình -> không lệch khi Windows đặt scale 125%/150%.
    """

    def __init__(self, region):
        self.x0, self.y0, self.x1, self.y1 = region

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        out = np.empty((len(pts), 2), dtype=np.float64)
        out[:, 0] = (pts[:, 0] - self.x0) / (self.x1 - self.x0)
        out[:, 1] = (pts[:, 1] - self.y0) / (self.y1 - self.y0)
        return np.clip(out, 0.0, 1.0)
