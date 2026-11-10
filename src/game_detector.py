"""Luồng 4 - Nhận diện game đang chơi.

Kích hoạt khi tay phải vẽ vòng tròn:
  chụp màn hình (mss) + lấy tiêu đề cửa sổ đang active
  -> tra GameDB (so khớp cục bộ bằng OpenCV)
  -> không có thì gọi Claude API (ảnh -> loại game) và lưu mẫu lại.
Chạy trên luồng riêng nên không làm chậm vẽ neon hay điều khiển.
"""
from __future__ import annotations

import base64
import ctypes
import json
import os
import threading
from dataclasses import dataclass
from typing import Callable

import cv2
import mss
import numpy as np

from .game_db import GameDB, GameEntry

MODES = ["slice", "shoot_v1", "shoot_v2", "other"]

PROMPT = """Ảnh là ảnh chụp toàn màn hình máy tính của người dùng đang mở một game trên trình duyệt.
Tiêu đề cửa sổ: "{title}".
Hãy xác định game và chọn chế độ điều khiển bằng cử chỉ tay phù hợp:
- "slice": game chém/cắt vật thể bằng cách vuốt chuột (ví dụ chém hoa quả).
- "shoot_v1": game bắn súng mà người chơi đứng yên, chỉ ngắm và bắn (thường bắn bia, bắn mục tiêu, phản xạ).
- "shoot_v2": game bắn súng góc nhìn thứ nhất/thứ ba mà người chơi phải di chuyển nhân vật (WASD) và xoay camera.
- "other": không thuộc các loại trên hoặc không thấy game.
"mouse_mode": "relative" nếu game kiểu góc nhìn thứ nhất xoay camera bằng chuột (khóa con trỏ), ngược lại "absolute".
"name": tên ngắn gọn của game."""

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "mode": {"type": "string", "enum": MODES},
        "mouse_mode": {"type": "string", "enum": ["absolute", "relative"]},
    },
    "required": ["name", "mode", "mouse_mode"],
    "additionalProperties": False,
}


def foreground_title() -> str:
    u = ctypes.windll.user32
    hwnd = u.GetForegroundWindow()
    n = u.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    u.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def grab_screen() -> np.ndarray:
    factory = getattr(mss, "MSS", None) or mss.mss
    with factory() as sct:
        shot = np.array(sct.grab(sct.monitors[1]))      # BGRA, màn hình chính
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


@dataclass
class DetectResult:
    entry: GameEntry | None
    source: str        # "cache-title" | "cache-image" | "claude" | "error"
    message: str


def classify_with_claude(bgr: np.ndarray, title: str, model: str, max_width: int) -> dict:
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Chưa có ANTHROPIC_API_KEY trong file .env")
    h, w = bgr.shape[:2]
    if w > max_width:
        bgr = cv2.resize(bgr, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("Không mã hóa được ảnh chụp màn hình")
    data = base64.standard_b64encode(jpg.tobytes()).decode("ascii")

    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=model,
        max_tokens=1024,
        betas=["server-side-fallback-2026-07-01"],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
                {"type": "text", "text": PROMPT.format(title=title)},
            ],
        }],
        extra_body={"fallbacks": "default"},
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude từ chối phân loại ảnh này")
    text = "".join(b.text for b in response.content if b.type == "text")
    result = json.loads(text)
    if result.get("mode") not in MODES:
        raise RuntimeError(f"Kết quả không hợp lệ: {result}")
    return result


class GameDetector:
    def __init__(self, db: GameDB, cfg: dict, on_result: Callable[[DetectResult], None],
                 classifier: Callable[..., dict] | None = None):
        self.db, self.cfg, self.on_result = db, cfg, on_result
        self.classifier = classifier or classify_with_claude
        self._event = threading.Event()
        self._running = False
        self.busy = False

    def trigger(self) -> bool:
        if self.busy:
            return False
        self._event.set()
        return True

    def start(self) -> "GameDetector":
        self._running = True
        threading.Thread(target=self._loop, name="game-detector", daemon=True).start()
        return self

    def stop(self) -> None:
        self._running = False
        self._event.set()

    def _loop(self) -> None:
        while self._running:
            self._event.wait()
            self._event.clear()
            if not self._running:
                break
            self.busy = True
            try:
                self.on_result(self.detect(grab_screen(), foreground_title()))
            except Exception as e:  # noqa: BLE001 - mọi lỗi đều báo lên overlay, không làm sập app
                self.on_result(DetectResult(None, "error", f"Lỗi nhận diện: {e}"))
            finally:
                self.busy = False

    def detect(self, bgr: np.ndarray, title: str) -> DetectResult:
        match = self.db.find(bgr, title)
        if match is not None:
            e = match.entry
            return DetectResult(e, f"cache-{match.by}", f"Đã nhận (bộ nhớ): {e.name}")
        result = self.classifier(bgr, title, self.cfg["model"], self.cfg["screenshot_max_width"])
        entry = self.db.add(bgr, title, result["name"], result["mode"], result["mouse_mode"])
        return DetectResult(entry, "claude", f"Đã nhận (Claude): {entry.name}")
