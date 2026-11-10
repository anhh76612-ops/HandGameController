"""Tư thế tay (pose), tên cử chỉ hiển thị, lọc cử chỉ lạ và nhận diện vòng tròn.

Hai tầng:
1. TƯ THẾ TĨNH (pose) - xác định từ trạng thái 5 ngón + khép/tách trỏ-giữa, có kiểm tra chéo với nhãn
   cử chỉ của mô hình Google và BỎ PHIẾU qua vài frame (PoseStabilizer). Không khớp -> NONE.
2. CỬ CHỈ CÓ TÊN (CON TRỎ, CLICK TRÁI, ...) - do máy trạng thái trong từng profile sinh ra từ tư thế
   và các CHUYỂN TRẠNG THÁI ngón (vd đang CON TRỎ thì cụp ngón cái = CLICK TRÁI).
   Profile chỉ chấp nhận các cử chỉ trong danh sách của chế độ; còn lại là KHÔNG RÕ và bị bỏ qua.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass

import cv2
import numpy as np

from .finger_state import INDEX, MIDDLE, PINKY, RING, THUMB  # noqa: F401

# ---------------- tư thế tĩnh ----------------
FIST, OPEN, THREE, POINTER, TWO_TOGETHER, INDEX_ONLY, NONE = (
    "FIST", "OPEN", "THREE", "POINTER", "TWO_TOGETHER", "INDEX_ONLY", "NONE")

# Nhãn của mô hình Google mâu thuẫn với tư thế -> loại (khi điểm tin cậy của mô hình đủ cao)
INCOMPATIBLE = {
    FIST: {"Open_Palm", "Victory", "Pointing_Up"},
    OPEN: {"Closed_Fist", "Pointing_Up"},
    THREE: {"Closed_Fist", "Pointing_Up"},
    POINTER: {"Closed_Fist", "Open_Palm"},
    TWO_TOGETHER: {"Closed_Fist", "Open_Palm"},
    INDEX_ONLY: {"Open_Palm", "Victory"},
}

# ---------------- tên cử chỉ hiển thị ----------------
CON_TRO = "CON TRỎ"
CLICK_TRAI = "CLICK TRÁI"
CLICK_PHAI = "CLICK PHẢI"
GIU_TRAI = "GIỮ CHUỘT TRÁI"
GIU_PHAI = "GIỮ CHUỘT PHẢI"
KEO_CHEM = "KÉO / CHÉM"
TAM_DUNG = "TẠM DỪNG"
NHAN_DIEN_GAME = "NHẬN DIỆN GAME"
SUNG = "SÚNG"
BOP_CO = "BÓP CÒ"
CHI_LEN = "CHỈ LÊN"
MO_NGAM = "MỞ NGẮM"
XOAY = "XOAY"
BAN = "BẮN"
NGAM = "NGẮM"
DI_CHUYEN = "DI CHUYỂN"
DUNG_LAI = "ĐỨNG LẠI"
DAT_TAM = "ĐẶT TÂM"
NHAY = "NHẢY"
NAP_DAN = "NẠP ĐẠN"
KHONG_RO = "KHÔNG RÕ"


def classify_pose(states: list[bool], together: bool, spread: bool | None = None) -> str:
    """Tư thế từ 4 ngón dài (ngón cái không xét - ngón cái dùng để CHẠM/click)."""
    i, m, r, p = states[INDEX], states[MIDDLE], states[RING], states[PINKY]
    if not (i or m or r or p):
        return FIST
    if i and m and r and p:
        return OPEN
    if i and m and r and not p:
        return THREE
    if i and m and not r and not p:
        return TWO_TOGETHER if together else POINTER
    if i and not m and not r and not p:
        return INDEX_ONLY
    return NONE


def model_conflict(pose: str, model_gesture: str, model_score: float, min_score: float) -> bool:
    return model_score >= min_score and model_gesture in INCOMPATIBLE.get(pose, set())


class PoseStabilizer:
    """Tư thế chỉ được đổi khi đa số (need/len) các frame gần nhất đồng ý."""

    def __init__(self, frames: int = 4, need: int = 3):
        self.hist: deque[str] = deque(maxlen=frames)
        self.need = need
        self.stable = NONE

    def reset(self) -> None:
        self.hist.clear()
        self.stable = NONE

    def update(self, pose: str) -> str:
        self.hist.append(pose)
        best, count = Counter(self.hist).most_common(1)[0]
        if count >= self.need:
            self.stable = best
        return self.stable


@dataclass
class Hand:
    """Toàn bộ thông tin một bàn tay ở một frame - đầu vào của các profile."""

    label: str                       # "Left" | "Right" (đã qua hand_identity)
    states: list[bool]               # 5 ngón, True = duỗi (ổn định)
    leaving: list[bool]              # ngón bắt đầu gập (để khóa con trỏ)
    pose: str                        # tư thế ổn định
    together: bool                   # trỏ + giữa khép sát
    vertical: bool                   # ngón trỏ chĩa lên trên (bàn tay dọc)
    tip: np.ndarray                  # (2,) đầu ngón trỏ trên màn hình, chuẩn hóa 0..1, đã lọc One Euro
    palm_center: np.ndarray          # (2,) tâm lòng bàn tay trên ảnh, pixel
    palm_px: float                   # kích thước lòng bàn tay trên ảnh, pixel
    cam: np.ndarray | None = None    # (21, 3) điểm mốc thô (để vẽ)
    world: np.ndarray | None = None  # (21, 3) điểm mốc 3D thật (mét) - đo cử động ngón cái trong hệ trục bàn tay
    source: str = ""                 # nguồn nhãn tay
    model_gesture: str = "None"
    touch: str | None = None         # ngón cái đang chạm đường ngón trỏ: "index" | None
    middle_half: bool = False        # ngón giữa đang duỗi NỬA CHỪNG (click phải)
    approach: bool = False           # ngón cái đã tới gần ngón trỏ/giữa (sắp chạm)
    fingers_px: np.ndarray | None = None   # trung điểm đầu ngón trỏ + giữa trên ảnh (pixel) - hướng đi V2
    palm_screen: np.ndarray | None = None  # tâm lòng bàn tay trên màn hình (chuẩn hóa)
    knuckle_screen: np.ndarray | None = None   # trung điểm khớp gốc ngón trỏ + giữa (điểm 5, 9) trên màn hình -
                                               # gần như KHÔNG ĐỔI khi nắm/mở tay -> điểm điều khiển của V2

    @property
    def thumb(self) -> bool:
        return self.states[THUMB]

    @property
    def index(self) -> bool:
        return self.states[INDEX]

    @property
    def middle(self) -> bool:
        return self.states[MIDDLE]

    @property
    def ring(self) -> bool:
        return self.states[RING]

    @property
    def pinky(self) -> bool:
        return self.states[PINKY]


class CircleDetector:
    """Nhận diện ngón trỏ vẽ một vòng tròn (xem docs/01, bước 11)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.pts: deque[tuple[float, float, float]] = deque()
        self.last_fire = -1e9

    def reset(self) -> None:
        self.pts.clear()

    def update(self, x: float, y: float, t: float) -> bool:
        c = self.cfg
        self.pts.append((t, x, y))
        while self.pts and t - self.pts[0][0] > c["window_sec"]:
            self.pts.popleft()
        if t - self.last_fire < c["cooldown_sec"] or len(self.pts) < c["min_points"]:
            return False
        xy = np.array([(p[1], p[2]) for p in self.pts], dtype=np.float32)
        for start in range(0, len(xy) - c["min_points"] + 1, 3):
            if self._is_circle(xy[start:]):
                self.last_fire = t
                self.pts.clear()
                return True
        return False

    def _is_circle(self, xy: np.ndarray) -> bool:
        c = self.cfg
        (_cx, _cy), r_enc = cv2.minEnclosingCircle(xy)
        if r_enc < c["min_radius"]:
            return False
        center = xy.mean(axis=0)
        d = np.linalg.norm(xy - center, axis=1)
        r_mean = float(d.mean())
        if r_mean < c["min_radius"] * 0.7 or float(d.std()) / (r_mean + 1e-9) > c["max_radius_cv"]:
            return False
        ang = np.unwrap(np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0]))
        if math.degrees(abs(float(ang[-1] - ang[0]))) < c["min_sweep_deg"]:
            return False
        return float(np.linalg.norm(xy[-1] - xy[0])) < c["close_ratio"] * r_mean * 2
