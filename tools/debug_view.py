"""Cửa sổ minh họa các bước xử lý ảnh + cử chỉ (dùng cho báo cáo, vấn đáp và tinh chỉnh ngưỡng).

Chạy CHÍNH XÁC cùng pipeline với main.py (xác định tay, đo ngón 3D, lọc cử chỉ lạ, profile) nhưng
KHÔNG gửi chuột/phím - an toàn để tập cử chỉ.
Phím: m = đổi chế độ | k = hiệu chỉnh (như F5) | s = lưu ảnh từng bước vào docs/images/ | q / Esc = thoát
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.calibration import load_calibration  # noqa: E402
from src.capture import Camera  # noqa: E402
from src.config import load_config, resolve  # noqa: E402
from src.controller import Controller  # noqa: E402
from src.hand_tracker import HandTracker  # noqa: E402
from src.pipeline import MODE_ORDER, AppState, VisionPipeline  # noqa: E402
from src.visualize import ascii_vi  # noqa: E402


class NullBackend:
    """Backend giả: không gửi gì ra hệ điều hành."""

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
    cfg = load_config()
    load_calibration(resolve("calibration.json"), cfg["touch"])
    cfg["overlay"]["preview_width"] = 900
    cam = Camera(cfg["camera"]).start()
    tracker = HandTracker(cfg["tracker"], resolve(cfg["tracker"]["model_path"]))
    state = AppState(mode=cfg["profiles"]["start_mode"])
    pipe = VisionPipeline(cfg, state, cam, tracker, Controller(NullBackend(), cfg["profiles"]),
                          on_circle=lambda: state.notify("VE VONG TRON -> nhan dien game"))
    out_dir = ROOT / "docs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    seq = 0
    try:
        while True:
            got = cam.latest(seq)
            if got is None:
                continue
            seq, frame, stamp = got
            t = time.perf_counter()
            hands = pipe.process(frame, t)
            infer_ms = (time.perf_counter() - t) * 1000
            desired = pipe.step(hands, t)
            view = cv2.cvtColor(pipe.render_preview(desired.gestures), cv2.COLOR_RGB2BGR)
            hh = view.shape[0]
            info = (f"che do: {state.mode} | cam {cam.fps:.0f}fps | xu ly {infer_ms:.0f}ms | "
                    f"tre {(time.perf_counter() - stamp) * 1000:.0f}ms | sang {state.brightness:.0f}"
                    f"{' CLAHE' if state.clahe else ''}")
            for color, th in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                cv2.putText(view, info, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, th, cv2.LINE_AA)
                if desired.label:
                    cv2.putText(view, ascii_vi(desired.label), (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 200) if th == 1 else color, th + 1, cv2.LINE_AA)
                if time.perf_counter() < state.message_until:
                    cv2.putText(view, ascii_vi(state.message), (10, hh - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0) if th == 1 else color, th + 1, cv2.LINE_AA)

            # Các bước xử lý ảnh để minh họa
            pre = pipe.pre(frame)
            lab = cv2.cvtColor(pre.bgr, cv2.COLOR_BGR2LAB)
            raw_small = cv2.resize(frame, (320, 240))
            l_small = cv2.resize(cv2.cvtColor(lab[:, :, 0], cv2.COLOR_GRAY2BGR), (320, 240))
            cv2.putText(raw_small, "1. anh goc", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(l_small, "2. kenh L (LAB)", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            side = np.vstack([raw_small, l_small])
            side = cv2.resize(side, (int(side.shape[1] * hh / side.shape[0]), hh))
            cv2.imshow("HandGameController - debug (m: che do, k: hieu chinh, s: luu anh, q: thoat)",
                       np.hstack([view, side]))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                pipe.set_mode(MODE_ORDER[(MODE_ORDER.index(state.mode) + 1) % len(MODE_ORDER)])
            if key == ord("k"):
                pipe.start_calibration()
            if key == ord("s"):
                st = time.strftime("%H%M%S")
                cv2.imwrite(str(out_dir / f"{st}_1_goc.png"), frame)
                cv2.imwrite(str(out_dir / f"{st}_2_kenh_L.png"), lab[:, :, 0])
                cv2.imwrite(str(out_dir / f"{st}_3_tien_xu_ly.png"), pre.bgr)
                cv2.imwrite(str(out_dir / f"{st}_4_khop_ngon_cu_chi.png"), view)
                print(f"Đã lưu ảnh {st}_*.png vào {out_dir}")
    finally:
        cam.stop()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
