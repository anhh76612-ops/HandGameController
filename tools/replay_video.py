"""Chạy lại ĐÚNG pipeline của main.py trên một file video (không gửi chuột/phím) để kiểm tra cử chỉ.

Ví dụ:
  .venv\\Scripts\\python.exe tools\\replay_video.py "C:\\...\\video.mp4" --mode shoot_v2
  .venv\\Scripts\\python.exe tools\\replay_video.py video.mp4 --out ket_qua.mp4     (ghi video có vẽ khớp ngón)

In theo thời gian: nhãn tay (+ nguồn nhãn), tư thế, tên cử chỉ, phím/nút mong muốn; cuối cùng in thống kê.
Video quay bằng app Camera của Windows thường KHÔNG lật gương -> mặc định lật giống camera trực tiếp.
"""
from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.calibration import load_calibration  # noqa: E402
from src.config import load_config, resolve  # noqa: E402
from src.controller import Controller  # noqa: E402
from src.hand_tracker import HandTracker  # noqa: E402
from src.pipeline import MODE_ORDER, AppState, VisionPipeline  # noqa: E402
from src.visualize import ascii_vi  # noqa: E402


class NullBackend:
    def screen_size(self):
        return 1920, 1080

    def move_abs(self, x, y):
        pass

    def move_rel(self, dx, dy):
        pass

    def button(self, b, down):
        pass

    def key(self, k, down):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--mode", choices=MODE_ORDER, default="shoot_v2")
    ap.add_argument("--every", type=int, default=6, help="in 1 dòng mỗi N frame")
    ap.add_argument("--out", help="ghi video kết quả (mp4)")
    ap.add_argument("--no-flip", action="store_true", help="video đã lật gương sẵn")
    args = ap.parse_args()

    cfg = load_config()
    load_calibration(resolve("calibration.json"), cfg["touch"])
    if args.no_flip:
        cfg["preprocess"]["flip"] = False
    cfg["overlay"]["preview_width"] = 640
    tracker = HandTracker(cfg["tracker"], resolve(cfg["tracker"]["model_path"]))
    state = AppState(mode=args.mode)
    pipe = VisionPipeline(cfg, state, None, tracker, Controller(NullBackend(), cfg["profiles"]), lambda: None)
    pipe.set_mode(args.mode)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    stats = {k: collections.Counter() for k in ("label_side", "source", "gesture", "keys", "reason")}
    i, t_proc = 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
        t = i / fps
        t0 = time.perf_counter()
        hands = pipe.process(frame, t)
        d = pipe.step(hands, t)
        t_proc += time.perf_counter() - t0
        for dt in pipe._details:
            side = "trai_anh" if dt.landmarks[0, 0] < 0.5 else "phai_anh"
            stats["label_side"][f"{dt.label}@{side}"] += 1
            stats["source"][dt.source] += 1
        for lb, name in d.gestures.items():
            stats["gesture"][f"{lb}:{name}"] += 1
        stats["keys"]["+".join(sorted(d.keys | d.buttons)) or "-"] += 1
        for r in pipe._reasons.values():
            stats["reason"][r] += 1
        if i % args.every == 0:
            hs = " ".join(f"[{lb} {h.source} {h.pose} -> {d.gestures.get(lb, '')}]" for lb, h in hands.items())
            print(f"t={t:5.1f}s {hs or '(không thấy tay)'} keys={sorted(d.keys | d.buttons) or '-'}"
                  + (f" taps={d.taps}" if d.taps else "") + (f" loai={list(pipe._reasons.values())}" if pipe._reasons else ""))
        if args.out:
            img = cv2.cvtColor(pipe.render_preview(d.gestures), cv2.COLOR_RGB2BGR)
            cv2.putText(img, ascii_vi(f"{t:5.1f}s {d.label} {sorted(d.keys | d.buttons)}"), (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if writer is None:
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
            writer.write(img)
        i += 1
    tracker.close()
    if writer is not None:
        writer.release()
    print(f"\n== {i} frame, xử lý trung bình {t_proc / max(i, 1) * 1000:.1f} ms/frame ==")
    for k, c in stats.items():
        print(f"{k}: {dict(c.most_common(12))}")


if __name__ == "__main__":
    main()
