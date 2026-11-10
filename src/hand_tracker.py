"""Phát hiện bàn tay bằng MediaPipe Gesture Recognizer (mô hình chính thức của Google, Apache 2.0).

Một lần chạy trả về cho mỗi bàn tay:
  - 21 điểm mốc chuẩn hóa theo khung ảnh (x, y trong 0..1, z tương đối)
  - 21 điểm mốc "world": tọa độ 3D tính bằng MÉT, gốc ở giữa bàn tay -> không phụ thuộc tay gần/xa camera,
    ít méo khi bàn tay xoay -> dùng để tính góc khớp, khoảng cách giữa các ngón
  - nhãn tay Left/Right kèm điểm tin cậy
  - 1 trong 8 cử chỉ có sẵn (Closed_Fist, Open_Palm, Pointing_Up, Victory, Thumb_Up, Thumb_Down,
    ILoveYou, None) kèm điểm tin cậy -> dùng để kiểm tra chéo luật hình học

Chỉ số điểm mốc:
  0 cổ tay | ngón cái 1-4 | trỏ 5-8 | giữa 9-12 | áp út 13-16 | út 17-20
"""
from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)


def ensure_model(path: Path, url: str = MODEL_URL) -> Path:
    """Tải file mô hình chính thức của Google nếu chưa có (~8 MB, chỉ lần đầu)."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[tracker] Đang tải mô hình MediaPipe về {path} ...")
        urllib.request.urlretrieve(url, path)
    return path


@dataclass
class RawHand:
    label: str               # nhãn MediaPipe ("Left" | "Right"), CHƯA qua hand_identity
    score: float             # độ tin cậy nhãn tay
    landmarks: np.ndarray    # (21, 3) chuẩn hóa theo khung ảnh
    world: np.ndarray        # (21, 3) mét
    model_gesture: str = "None"
    model_score: float = 0.0


class HandTracker:
    def __init__(self, cfg: dict, model_path: Path):
        ensure_model(model_path)
        options = vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg["num_hands"],
            min_hand_detection_confidence=cfg["min_detection_confidence"],
            min_hand_presence_confidence=cfg["min_presence_confidence"],
            min_tracking_confidence=cfg["min_tracking_confidence"],
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        self.swap = cfg.get("swap_handedness", False)
        self._last_ts = 0

    def detect(self, rgb: np.ndarray) -> list[RawHand]:
        ts = int(time.perf_counter() * 1000)
        if ts <= self._last_ts:          # chế độ VIDEO yêu cầu timestamp tăng dần
            ts = self._last_ts + 1
        self._last_ts = ts
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.recognizer.recognize_for_video(image, ts)
        hands: list[RawHand] = []
        for i, lms in enumerate(res.hand_landmarks):
            handed = res.handedness[i][0] if i < len(res.handedness) and res.handedness[i] else None
            label = handed.category_name if handed else "Right"
            if self.swap:
                label = "Left" if label == "Right" else "Right"
            world = res.hand_world_landmarks[i] if i < len(res.hand_world_landmarks) else lms
            g = res.gestures[i][0] if i < len(res.gestures) and res.gestures[i] else None
            hands.append(RawHand(
                label=label,
                score=float(handed.score) if handed else 0.0,
                landmarks=np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32),
                world=np.array([[p.x, p.y, p.z] for p in world], dtype=np.float32),
                model_gesture=g.category_name if g else "None",
                model_score=float(g.score) if g else 0.0,
            ))
        return hands

    def close(self) -> None:
        self.recognizer.close()
