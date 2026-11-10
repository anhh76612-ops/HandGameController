"""Lưu và tra cứu mẫu game đã nhận diện (để lần sau không cần gọi API).

Đặc trưng ảnh của một ảnh chụp màn hình:
1. dHash 64-bit (difference hash):
   grayscale -> resize 9x8 (INTER_AREA = lấy trung bình vùng, khử chi tiết nhỏ)
   -> so sánh mỗi pixel với pixel bên phải -> 8x8 = 64 bit.
   Hai ảnh giống nhau về bố cục sáng/tối có khoảng cách Hamming nhỏ (ít bit khác nhau).
2. Histogram màu HSV 2 chiều (H: 30 bin, S: 32 bin) trên ảnh đã thu nhỏ + làm mờ Gauss, chuẩn hóa -> so bằng hệ số tương quan
   (cv2.compareHist, HISTCMP_CORREL). Đặc trưng cho "bảng màu" của game, không phụ thuộc vị trí.
Khớp nếu: tiêu đề cửa sổ trùng, HOẶC (Hamming <= ngưỡng VÀ tương quan histogram >= ngưỡng).
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np


def dhash(bgr: np.ndarray) -> int:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def hsv_hist(bgr: np.ndarray) -> np.ndarray:
    # Thu nhỏ bằng INTER_AREA trước (lấy trung bình vùng) để khử nhiễu pixel, nén ảnh, chữ nhỏ
    small = cv2.resize(bgr, (256, max(1, int(256 * bgr.shape[0] / bgr.shape[1]))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.astype(np.float32)


def hist_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_CORREL))


def clean_title(title: str) -> str:
    """Bỏ phần đuôi tên trình duyệt để tiêu đề tab ổn định hơn."""
    for suffix in (" - Google Chrome", " - Microsoft Edge", " — Mozilla Firefox",
                   " - Mozilla Firefox", " - Opera", " - Cốc Cốc", " - Brave"):
        title = title.replace(suffix, "")
    return title.strip()


@dataclass
class GameEntry:
    name: str
    mode: str                      # slice | shoot_v1 | shoot_v2 | other
    title: str
    dhash: str                     # hex
    hist: list = field(default_factory=list)
    mouse_mode: str = "absolute"   # absolute | relative (cho V1)
    thumb: str = ""
    created: float = 0.0


@dataclass
class Match:
    entry: GameEntry
    by: str                        # "title" | "image"
    distance: int
    correlation: float


class GameDB:
    def __init__(self, path: Path, hash_max: int, hist_min: float):
        self.path = Path(path)
        self.hash_max, self.hist_min = hash_max, hist_min
        self.entries: list[GameEntry] = []
        self._lock = threading.Lock()
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = [GameEntry(**e) for e in data.get("games", [])]

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {"games": [asdict(e) for e in self.entries]}
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    def find(self, bgr: np.ndarray, title: str) -> Match | None:
        title = clean_title(title)
        h, hist = dhash(bgr), hsv_hist(bgr)
        best: Match | None = None
        for e in self.entries:
            dist = hamming(h, int(e.dhash, 16))
            corr = hist_corr(hist, np.array(e.hist, dtype=np.float32).reshape(30, 32)) if e.hist else -1.0
            if title and e.title and title == e.title:
                return Match(e, "title", dist, corr)
            if dist <= self.hash_max and corr >= self.hist_min:
                if best is None or dist < best.distance:
                    best = Match(e, "image", dist, corr)
        return best

    def add(self, bgr: np.ndarray, title: str, name: str, mode: str, mouse_mode: str) -> GameEntry:
        title = clean_title(title)
        entry = GameEntry(
            name=name, mode=mode, title=title, dhash=format(dhash(bgr), "016x"),
            hist=hsv_hist(bgr).flatten().round(6).tolist(), mouse_mode=mouse_mode, created=time.time(),
        )
        thumb_dir = self.path.parent / "thumbs"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        thumb = thumb_dir / f"{int(entry.created)}.jpg"
        small = cv2.resize(bgr, (320, int(320 * bgr.shape[0] / bgr.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(thumb), small)
        entry.thumb = str(thumb.relative_to(self.path.parent))
        with self._lock:
            self.entries.append(entry)
        self.save()
        return entry

    def set_mouse_mode(self, entry: GameEntry, mouse_mode: str) -> None:
        entry.mouse_mode = mouse_mode
        self.save()
