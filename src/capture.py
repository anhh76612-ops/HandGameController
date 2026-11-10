"""Luồng 1 - Đọc webcam.

Mục tiêu chính là GIẢM TRỄ:
- CAP_DSHOW + MJPG + xin 60 FPS.
- CAP_PROP_BUFFERSIZE = 1 để driver không giữ nhiều frame cũ.
- Một luồng riêng đọc liên tục và chỉ giữ frame MỚI NHẤT; luồng xử lý luôn lấy frame mới nhất,
  frame cũ bị bỏ qua (không xếp hàng) -> độ trễ không tích lũy khi xử lý chậm hơn camera.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class Camera:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._seq = 0
        self._cond = threading.Condition()
        self._running = False
        self._thread: threading.Thread | None = None
        self.fps = 0.0

    def open(self) -> None:
        c = self.cfg
        cap = cv2.VideoCapture(c["index"], cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(c["index"])
        if not cap.isOpened():
            raise RuntimeError(f"Không mở được camera {c['index']}")
        if c.get("mjpg", True):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
        cap.set(cv2.CAP_PROP_FPS, c["fps"])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap

    def start(self, timeout: float = 15.0) -> "Camera":
        """Mở camera NGAY TRÊN luồng đọc: DirectShow (COM) gắn với luồng đã mở nó,
        mở ở luồng này rồi đọc ở luồng khác sẽ không nhận được frame / lỗi khi giải phóng."""
        self._running = True
        self._opened = threading.Event()
        self._open_error: Exception | None = None
        self._thread = threading.Thread(target=self._loop, name="capture", daemon=True)
        self._thread.start()
        if not self._opened.wait(timeout):
            raise RuntimeError("Mở camera quá lâu")
        if self._open_error is not None:
            raise self._open_error
        return self

    def _loop(self) -> None:
        try:
            self.open()
        except Exception as e:  # noqa: BLE001
            self._open_error = e
            self._running = False
            return
        finally:
            self._opened.set()
        n, t0 = 0, time.perf_counter()
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            now = time.perf_counter()
            with self._cond:
                self._frame, self._stamp = frame, now
                self._seq += 1
                self._cond.notify_all()
            n += 1
            if now - t0 >= 1.0:
                self.fps, n, t0 = n / (now - t0), 0, now
        self.cap.release()        # giải phóng trên chính luồng đã mở

    def latest(self, after_seq: int = 0, timeout: float = 1.0):
        """Chờ frame có seq > after_seq. Trả (seq, frame, thời điểm chụp) hoặc None."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > after_seq or not self._running, timeout):
                return None
            if self._frame is None:
                return None
            return self._seq, self._frame, self._stamp

    def stop(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._thread:
            self._thread.join(timeout=2.0)
