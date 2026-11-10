"""Đo trạng thái từng ngón DUỖI / CỤP trên tọa độ 3D thật (world landmarks, đơn vị mét).

Vì sao dùng world landmarks: tọa độ ảnh (x, y) bị méo khi bàn tay nghiêng/xoay và thay đổi theo khoảng
cách tới camera; tọa độ world là 3D thật quanh bàn tay nên góc khớp và tỉ lệ khoảng cách ổn định hơn.

Ngón trỏ/giữa/áp út/út:  góc tại khớp PIP = góc giữa (PIP->MCP) và (PIP->TIP). Thẳng ~180°, gập < 120°.
Ngón cái: r = |đầu ngón cái - gốc ngón giữa| / lòng bàn tay. Dựng/chìa ra: r lớn; gập vào lòng: r nhỏ.
Lòng bàn tay = |cổ tay - gốc ngón giữa|  -> mọi tỉ lệ không phụ thuộc kích thước tay.

HYSTERESIS: chỉ chuyển sang DUỖI khi vượt ngưỡng trên, chỉ chuyển sang CỤP khi dưới ngưỡng dưới.
DEBOUNCE: trạng thái mới phải lặp lại N frame liên tiếp.
"BẮT ĐẦU GẬP" (leaving): ngón vẫn đang DUỖI nhưng số đo đã rơi khỏi vùng duỗi -> dùng để khóa con trỏ
trước khi click, tránh con trỏ trượt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

THUMB, INDEX, MIDDLE, RING, PINKY = range(5)
FINGER_NAMES = ["cai", "tro", "giua", "ap_ut", "ut"]
# (MCP, PIP, TIP) cho 4 ngón dài
JOINTS = {INDEX: (5, 6, 8), MIDDLE: (9, 10, 12), RING: (13, 14, 16), PINKY: (17, 18, 20)}
PALM_POINTS = [0, 5, 9, 13, 17]


def to_metric(lm: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Tọa độ ảnh chuẩn hóa -> cùng thang pixel cho x, y, z."""
    return np.stack([lm[:, 0] * img_w, lm[:, 1] * img_h, lm[:, 2] * img_w], axis=1)


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Góc ABC (độ) tại đỉnh B."""
    v1, v2 = a - b, c - b
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def palm_size(p: np.ndarray) -> float:
    return float(np.linalg.norm(p[9] - p[0])) + 1e-9


def seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Khoảng cách từ điểm p tới đoạn thẳng ab."""
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def finger_line(p: np.ndarray, mcp: int) -> list[np.ndarray]:
    """Đường gấp khúc CẢ ngón, từ khớp gốc tới đầu ngón (đúng đoạn xanh lá trong khung camera):
    ngón cái chạm vào CHỖ NÀO trên đoạn này cũng tính là chạm."""
    return [p[mcp], p[mcp + 1], p[mcp + 2], p[mcp + 3]]


def polyline_dist(pt: np.ndarray, line: list[np.ndarray]) -> float:
    return min(seg_dist(pt, a, b) for a, b in zip(line, line[1:]))


@dataclass
class FingerMeasure:
    angles: list[float]       # [tỉ lệ ngón cái r, góc PIP trỏ, giữa, áp út, út]
    thumb_ratio: float
    palm: float               # lòng bàn tay (mét)
    together: float           # khoảng cách CẢ NGÓN trỏ-giữa (trung bình cặp 7-11, 8-12) / lòng bàn tay
    pinch: float              # |đầu cái - đầu trỏ| / lòng bàn tay
    foreshorten: float        # độ dài ngón trỏ trên ẢNH / độ dài thật (world). Nhỏ = ngón chĩa vào camera
    up_angle: float           # góc giữa hướng ngón trỏ (trên ảnh) và hướng thẳng lên (độ)
    # đầu ngón cái -> đường ngón trỏ / ngón giữa, trên ảnh 2D (chia lòng bàn tay 2D) và 3D thật
    touch_index_2d: float = 9.0
    touch_index_3d: float = 9.0
    touch_middle_2d: float = 9.0
    touch_middle_3d: float = 9.0
    middle_ratio: float = 1.0  # độ dài ngón giữa / ngón trỏ TRÊN ẢNH (gốc -> đầu ngón). Duỗi ~1, nửa chừng ~0.3-0.7


def measure(world: np.ndarray, img_metric: np.ndarray | None = None) -> FingerMeasure:
    """world: (21,3) mét. img_metric: (21,3) tọa độ ảnh đã đổi sang pixel (để kiểm tra hướng/chĩa)."""
    palm = palm_size(world)
    angles = [0.0] * 5
    for f, (mcp, pip, tip) in JOINTS.items():
        angles[f] = joint_angle(world[mcp], world[pip], world[tip])
    thumb_ratio = float(np.linalg.norm(world[4] - world[9])) / palm
    angles[THUMB] = thumb_ratio
    together = (float(np.linalg.norm(world[7] - world[11])) + float(np.linalg.norm(world[8] - world[12]))) / 2 / palm
    pinch = float(np.linalg.norm(world[4] - world[8])) / palm
    ti3 = polyline_dist(world[4], finger_line(world, 5)) / palm
    tm3 = polyline_dist(world[4], finger_line(world, 9)) / palm
    ti2, tm2 = ti3, tm3
    middle_ratio = float(np.linalg.norm(world[12] - world[9])) / (float(np.linalg.norm(world[8] - world[5])) + 1e-9)
    foreshorten, up_angle = 1.0, 0.0
    if img_metric is not None:
        img2 = img_metric[:, :2]
        palm2 = float(np.linalg.norm(img2[9] - img2[0])) + 1e-9
        ti2 = polyline_dist(img2[4], finger_line(img2, 5)) / palm2
        tm2 = polyline_dist(img2[4], finger_line(img2, 9)) / palm2
        middle_ratio = float(np.linalg.norm(img2[12] - img2[9])) / (float(np.linalg.norm(img2[8] - img2[5])) + 1e-9)
        len_img = float(np.linalg.norm(img_metric[8, :2] - img_metric[5, :2])) / (
            float(np.linalg.norm(img_metric[9, :2] - img_metric[0, :2])) + 1e-9)
        len_world = float(np.linalg.norm(world[8] - world[5])) / palm
        foreshorten = len_img / (len_world + 1e-9)
        d = img_metric[8, :2] - img_metric[5, :2]
        up_angle = float(np.degrees(np.arctan2(abs(d[0]), -d[1]))) if np.linalg.norm(d) > 1e-6 else 180.0
    return FingerMeasure(angles, thumb_ratio, palm, together, pinch, foreshorten, up_angle, ti2, ti3, tm2, tm3,
                         middle_ratio)


class TouchTracker:
    """Phát hiện ĐẦU NGÓN CÁI CHẠM đường ngón trỏ (-> "index") = CLICK TRÁI.
    (Click phải dùng NGÓN GIỮA NỬA CHỪNG - xem MiddleHalfTracker - nên không xét chạm ngón giữa.)

    Chạm khi CẢ khoảng cách 2D (trên ảnh) và 3D (thật) nhỏ hơn ngưỡng chạm; nhả khi một trong hai vượt
    ngưỡng nhả (hysteresis). Trạng thái mới phải ổn định `debounce` frame. Gần cả hai ngón -> chọn ngón gần hơn.
    `approach`: ngón cái đã tới gần (sắp chạm) -> dùng để khóa con trỏ trước khi click.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.stable: str | None = None
        self._cand: str | None = None
        self._count = 0
        self.approach = False

    def _is_on(self, d2: float, d3: float, currently: bool) -> bool:
        c = self.cfg
        if currently:
            return d2 <= c["touch_off_2d"] and d3 <= c["touch_off_3d"]
        return d2 < c["touch_on_2d"] and d3 < c["touch_on_3d"]

    def update(self, m: FingerMeasure) -> str | None:
        on_i = self._is_on(m.touch_index_2d, m.touch_index_3d, self.stable == "index")
        raw = "index" if on_i else None
        self.approach = m.touch_index_2d < self.cfg["approach_2d"]
        n = max(1, int(self.cfg.get("touch_debounce", 2)))
        if raw == self.stable:
            self._count = 0
        else:
            if raw == self._cand:
                self._count += 1
            else:
                self._cand, self._count = raw, 1
            if self._count >= n:
                self.stable, self._count = raw, 0
        return self.stable


class MiddleHalfTracker:
    """CLICK PHẢI = NGÓN GIỮA DUỖI NỬA CHỪNG (dài khoảng một nửa ngón trỏ, không gập hẳn) khi ngón trỏ đang duỗi.

    Đo r = độ dài ngón giữa / ngón trỏ trên ảnh. Vào trạng thái khi r nằm trong [half_lo, half_hi] liên tiếp
    `frames` frame VÀ r gần như không đổi (|Δr| < steady) -> lúc gập/duỗi nhanh, r đi XUYÊN QUA vùng nửa chừng
    nhưng không dừng lại nên không bị tính là click. Ra khi r < off_lo (gập hẳn) hoặc r > off_hi (duỗi thẳng).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.count = 0
        self.prev: float | None = None

    def update(self, ratio: float, index_extended: bool) -> bool:
        c = self.cfg
        steady = self.prev is None or abs(ratio - self.prev) < c["steady"]
        self.prev = ratio
        if not index_extended:
            self.active, self.count = False, 0
            return False
        if self.active:
            if ratio < c["off_lo"] or ratio > c["off_hi"]:
                self.active, self.count = False, 0
            return self.active
        if c["half_lo"] <= ratio <= c["half_hi"] and steady:
            self.count += 1
            if self.count >= c["frames"]:
                self.active = True
        else:
            self.count = 0
        return self.active


def finger_gap_ratio(bgr: np.ndarray, lm: np.ndarray) -> float | None:
    """XỬ LÝ ẢNH: giữa ngón trỏ và ngón giữa có lộ NỀN không?

    1. Lấy mẫu màu da trên chính 2 ngón (ô 5x5 quanh điểm 6, 7, 10, 11) trong không gian YCrCb
       (Cr, Cb mô tả màu da khá ổn định khi độ sáng thay đổi).
    2. Lấy mẫu các pixel trên những đoạn nối ngón trỏ <-> ngón giữa ở 3 mức (khớp 7-11, giữa 7-8/11-12,
       đầu 8-12), chỉ phần giữa đoạn (30%..70%) để tránh chính thân ngón.
    3. Pixel "không giống da" nếu Cr hoặc Cb lệch khỏi màu mẫu quá 2.5 độ lệch chuẩn (+6).
    Trả về tỉ lệ pixel không giống da (lớn = có khe nền giữa 2 ngón = đang TÁCH), hoặc None nếu không đo được.
    """
    h, w = bgr.shape[:2]
    pts = np.stack([lm[:, 0] * w, lm[:, 1] * h], axis=1)
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    samples = []
    for i in (6, 7, 10, 11):
        x, y = int(round(pts[i, 0])), int(round(pts[i, 1]))
        if 2 <= x < w - 2 and 2 <= y < h - 2:
            samples.append(ycc[y - 2:y + 3, x - 2:x + 3, 1:].reshape(-1, 2))
    if len(samples) < 2:
        return None
    ref = np.concatenate(samples).astype(np.float32)
    mean, std = ref.mean(axis=0), ref.std(axis=0)
    tol = 2.5 * std + 6.0
    pairs = [(pts[7], pts[11]), ((pts[7] + pts[8]) / 2, (pts[11] + pts[12]) / 2), (pts[8], pts[12])]
    total = bad = 0
    for a, b in pairs:
        if np.linalg.norm(b - a) < 4:            # 2 ngón đã sát nhau trên ảnh -> không có chỗ cho khe
            continue
        for t in np.linspace(0.3, 0.7, 7):
            x, y = (a + t * (b - a)).round().astype(int)
            if 0 <= x < w and 0 <= y < h:
                total += 1
                bad += int(np.any(np.abs(ycc[y, x, 1:].astype(np.float32) - mean) > tol))
    if total == 0:
        return 0.0
    return bad / total


class Hyst:
    """Cờ đúng/sai có ngưỡng kép. high_is_true: giá trị lớn -> True."""

    def __init__(self, hi: float, lo: float, init: bool, high_is_true: bool = True):
        self.hi, self.lo, self.value, self.high_is_true = hi, lo, init, high_is_true

    def update(self, x: float) -> bool:
        if x > self.hi:
            self.value = self.high_is_true
        elif x < self.lo:
            self.value = not self.high_is_true
        return self.value


@dataclass
class FingerTracker:
    """Trạng thái ổn định (hysteresis + debounce) của 5 ngón cho MỘT bàn tay."""

    cfg: dict
    stable: list[bool] = field(default_factory=lambda: [True] * 5)     # True = DUỖI
    leaving: list[bool] = field(default_factory=lambda: [False] * 5)   # đang duỗi nhưng bắt đầu gập
    _cand: list[bool] = field(default_factory=lambda: [True] * 5)
    _count: list[int] = field(default_factory=lambda: [0] * 5)

    def reset(self) -> None:
        self.stable, self.leaving = [True] * 5, [False] * 5
        self._cand, self._count = [True] * 5, [0] * 5

    def _thresholds(self, f: int) -> tuple[float, float]:
        c = self.cfg
        if f == THUMB:
            return c["thumb_extend_ratio"], c["thumb_fold_ratio"]
        return c["extend_angle"], c["fold_angle"]

    def update(self, m: FingerMeasure) -> list[bool]:
        n = max(1, int(self.cfg["debounce_frames"]))
        for f in range(5):
            hi, lo = self._thresholds(f)
            v = m.angles[f]
            raw = True if v > hi else False if v < lo else self.stable[f]
            self.leaving[f] = self.stable[f] and v < hi
            if raw == self.stable[f]:
                self._count[f] = 0
                continue
            if raw == self._cand[f]:
                self._count[f] += 1
            else:
                self._cand[f], self._count[f] = raw, 1
            if self._count[f] >= n:
                self.stable[f], self._count[f] = raw, 0
        return list(self.stable)
