"""Vẽ khung xương bàn tay lên ảnh camera để thấy chương trình xác định khớp ngón tay thế nào.

- Mỗi ngón một màu; ngón DUỖI vẽ nét sáng liền, ngón CỤP vẽ nét đỏ.
- Khớp PIP của 4 ngón dài ghi góc (độ) - chính là góc dùng để quyết định duỗi/cụp.
- Ngón cái ghi tỉ lệ r = |đầu ngón cái - gốc ngón giữa| / lòng bàn tay (tọa độ 3D).
- Trên bàn tay: nhãn tay + nguồn nhãn (vị trí/bám/MediaPipe), tên cử chỉ, nhãn của mô hình Google;
  nếu frame bị loại thì ghi lý do (chữ đỏ).
OpenCV không vẽ được chữ có dấu nên chữ trong khung camera được bỏ dấu (ascii_vi).
- Đầu ngón (TIP) chấm to, khớp MCP/PIP/DIP chấm nhỏ, cổ tay chấm vuông.
"""
from __future__ import annotations

import unicodedata

import cv2
import numpy as np

from .finger_state import JOINTS, FingerMeasure

# (các đoạn xương của từng ngón), theo thứ tự: cái, trỏ, giữa, áp út, út
FINGER_BONES = [
    [(0, 1), (1, 2), (2, 3), (3, 4)],
    [(5, 6), (6, 7), (7, 8)],
    [(9, 10), (10, 11), (11, 12)],
    [(13, 14), (14, 15), (15, 16)],
    [(17, 18), (18, 19), (19, 20)],
]
PALM_BONES = [(0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]
# màu BGR từng ngón khi DUỖI
FINGER_COLORS = [(0, 200, 255), (0, 255, 120), (255, 220, 0), (255, 120, 255), (120, 160, 255)]
FOLDED = (40, 40, 230)
LABEL_VI = {"Right": "PHAI", "Left": "TRAI"}


def ascii_vi(text: str) -> str:
    """Bỏ dấu tiếng Việt (cv2.putText chỉ vẽ được ASCII)."""
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def _text(img, txt, org, fs, color, thick=1):
    txt = ascii_vi(txt)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, color, thick, cv2.LINE_AA)


def draw_hand(img: np.ndarray, lm: np.ndarray, states: list[bool] | None = None,
              m: FingerMeasure | None = None, label: str = "", gesture: str = "",
              source: str = "", reason: str = "", model_gesture: str = "", touch: str | None = None,
              gap: float | None = None, click_info: str = "") -> None:
    """lm: (21, >=2) tọa độ chuẩn hóa 0..1 theo khung ảnh `img` (vẽ trực tiếp lên img)."""
    h, w = img.shape[:2]
    scale = max(0.6, w / 640)
    pts = [(int(p[0] * w), int(p[1] * h)) for p in lm]
    th = max(1, int(round(2 * scale)))

    for a, b in PALM_BONES:
        cv2.line(img, pts[a], pts[b], (200, 200, 200), th, cv2.LINE_AA)
    for f, bones in enumerate(FINGER_BONES):
        ext = True if states is None else states[f]
        color = FINGER_COLORS[f] if ext else FOLDED
        for a, b in bones:
            cv2.line(img, pts[a], pts[b], color, th + (1 if ext else 0), cv2.LINE_AA)

    # ngón đang bị ngón cái chạm: tô trắng đậm + vẽ đoạn nối từ đầu ngón cái
    if touch == "index":
        f = 1
        for a, b in FINGER_BONES[f]:
            cv2.line(img, pts[a], pts[b], (255, 255, 255), th + 3, cv2.LINE_AA)
        cv2.circle(img, pts[4], int(9 * scale), (255, 255, 255), 2, cv2.LINE_AA)

    for i, pt in enumerate(pts):
        if i == 0:
            r = int(4 * scale)
            cv2.rectangle(img, (pt[0] - r, pt[1] - r), (pt[0] + r, pt[1] + r), (255, 255, 255), -1)
        elif i in (4, 8, 12, 16, 20):
            cv2.circle(img, pt, int(5 * scale), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, pt, int(5 * scale), (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.circle(img, pt, int(3 * scale), (255, 255, 255), -1, cv2.LINE_AA)

    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale
    if m is not None:
        # góc tại khớp PIP của 4 ngón dài
        for f, (_mcp, pip, _tip) in JOINTS.items():
            ext = True if states is None else states[f]
            txt = f"{m.angles[f]:.0f}"
            org = (pts[pip][0] + int(6 * scale), pts[pip][1])
            cv2.putText(img, txt, org, font, fs, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, org, font, fs, FINGER_COLORS[f] if ext else FOLDED, 1, cv2.LINE_AA)
        txt = f"cham {m.touch_index_2d:.2f}/{m.touch_index_3d:.2f}"
        org = (max(2, pts[4][0] - int(120 * scale)), pts[4][1] + int(18 * scale))
        cv2.putText(img, txt, org, font, fs, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, org, font, fs, FINGER_COLORS[0] if (states is None or states[0]) else FOLDED, 1,
                    cv2.LINE_AA)

    if label:
        top = min(p[1] for p in pts)
        x = max(2, min(p[0] for p in pts))
        lines = [(f"{LABEL_VI.get(label, label)}" + (f" ({source})" if source else ""), (255, 255, 255))]
        if reason:
            lines.append((reason, (60, 60, 255)))
        else:
            lines.append((gesture or "-", (0, 255, 200)))
            extra = []
            if model_gesture and model_gesture != "None":
                extra.append(f"mo hinh: {model_gesture}")
            if m is not None:
                extra.append(f"khep {m.together:.2f}" + (f" khe {gap:.2f}" if gap is not None else ""))
            if extra:
                lines.append((" | ".join(extra), (200, 200, 200)))
            if click_info:
                lines.append((click_info, (120, 255, 255)))
        lh = int(14 * scale)
        y0 = max(lh, top - int(8 * scale) - lh * (len(lines) - 1))
        for i, (txt, color) in enumerate(lines):
            _text(img, txt, (x, y0 + i * lh), 0.42 * scale, color)


def draw_region(img: np.ndarray, region) -> None:
    """Khung vùng hoạt động (phần ảnh được kéo giãn ra toàn màn hình)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = region
    cv2.rectangle(img, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (90, 90, 90), 1, cv2.LINE_AA)


def legend(img: np.ndarray, extend_angle: float, fold_angle: float) -> None:
    h = img.shape[0]
    txt = f"so = goc khop PIP (>{extend_angle:.0f} duoi, <{fold_angle:.0f} cup) | net do = ngon CUP"
    for color, thick in (((0, 0, 0), 3), ((230, 230, 230), 1)):
        cv2.putText(img, txt, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, thick, cv2.LINE_AA)
