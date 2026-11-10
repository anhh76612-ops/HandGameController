"""Xác định tay TRÁI / PHẢI ổn định (không tin tuyệt đối vào nhãn MediaPipe ở từng frame).

MediaPipe đôi khi đảo nhãn Left/Right trong vài frame (nhất là khi tay nghiêng, chĩa vào camera),
hoặc đảo cả hệ thống tùy cách lật ảnh. Cách xử lý:
1. Thấy 2 tay  -> theo VỊ TRÍ: cổ tay nằm bên trái ảnh (ảnh đã lật gương) là tay Trái.
2. Thấy 1 tay đang được bám -> giữ nhãn cũ nếu cổ tay gần vị trí cũ (BÁM).
3. Tay mới xuất hiện một mình:
   - cổ tay ở 40% bên trái ảnh -> Trái, 40% bên phải -> Phải (VỊ TRÍ);
   - ở vùng giữa -> dùng nhãn MediaPipe đã sửa theo PHIẾU: mỗi lần thấy 2 tay, so nhãn MediaPipe với luật
     vị trí để biết MediaPipe có đang đảo nhãn không; ~5 frame đầu chọn theo đa số.
Đo trên video thật: với ảnh đã lật gương, MediaPipe gán tay trái là "Right" ở ~95% frame -> config đặt
swap_handedness: true để đảo lại ngay từ đầu.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from .hand_tracker import RawHand

FLIP = {"Left": "Right", "Right": "Left"}


@dataclass
class Track:
    label: str
    wrist: np.ndarray
    last_t: float
    age: int = 0


@dataclass
class Identified:
    label: str
    raw: RawHand
    source: str          # "vị trí" | "bám" | "MediaPipe"


class HandIdentity:
    def __init__(self, track_dist: float = 0.15, track_timeout: float = 0.5, vote_frames: int = 5,
                 side_band: float = 0.4):
        self.track_dist, self.track_timeout, self.vote_frames = track_dist, track_timeout, vote_frames
        self.side_band = side_band
        self.dup_dist = 0.08
        self.tracks: dict[str, Track] = {}
        self.swap_votes = 0                 # > 0: MediaPipe đang đảo nhãn so với vị trí
        self._new_votes: deque[str] = deque(maxlen=vote_frames)

    def _corrected(self, raw_label: str) -> str:
        return FLIP[raw_label] if self.swap_votes > 3 else raw_label

    def assign(self, hands: list[RawHand], t: float) -> list[Identified]:
        # bỏ các track quá cũ
        for lb in [lb for lb, tr in self.tracks.items() if t - tr.last_t > self.track_timeout]:
            del self.tracks[lb]
        if not hands:
            self._new_votes.clear()
            return []
        # MediaPipe đôi khi phát hiện CÙNG một bàn tay 2 lần -> bỏ bản trùng (giữ bản có điểm tin cậy cao hơn)
        if len(hands) >= 2 and float(np.linalg.norm(hands[0].landmarks[0, :2] - hands[1].landmarks[0, :2])) < self.dup_dist:
            hands = [max(hands[:2], key=lambda h: h.score)]

        if len(hands) >= 2:
            two = sorted(hands[:2], key=lambda h: float(h.landmarks[0, 0]))
            out = [Identified("Left", two[0], "vị trí"), Identified("Right", two[1], "vị trí")]
            for o in out:
                self.swap_votes += 1 if o.raw.label != o.label else -1
            self.swap_votes = max(-20, min(20, self.swap_votes))
            self._new_votes.clear()
        else:
            h = hands[0]
            wrist = h.landmarks[0, :2]
            near = [(float(np.linalg.norm(tr.wrist - wrist)), lb) for lb, tr in self.tracks.items()]
            near = [x for x in near if x[0] < self.track_dist]
            corrected = self._corrected(h.label)
            if near:
                label = min(near)[1]
                tr = self.tracks[label]
                # track còn non: cho phép sửa nhãn theo đa số các frame đầu
                if tr.age < self.vote_frames:
                    self._new_votes.append(corrected)
                    label = Counter(self._new_votes).most_common(1)[0][0]
                    if label != tr.label:
                        self.tracks.pop(tr.label, None)
                out = [Identified(label, h, "bám")]
            else:
                self._new_votes.clear()
                x = float(wrist[0])
                if x < self.side_band:                     # tay mới ở hẳn bên trái ảnh (đã lật gương)
                    out = [Identified("Left", h, "vị trí")]
                elif x > 1.0 - self.side_band:
                    out = [Identified("Right", h, "vị trí")]
                else:
                    self._new_votes.append(corrected)
                    out = [Identified(corrected, h, "MediaPipe")]

        for o in out:
            prev = self.tracks.get(o.label)
            self.tracks[o.label] = Track(o.label, o.raw.landmarks[0, :2].copy(), t, (prev.age + 1) if prev else 0)
        return out
