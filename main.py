"""HandGameController - điều khiển game web bằng cử chỉ tay.

Chạy:  .venv\\Scripts\\python.exe main.py
       .venv\\Scripts\\python.exe main.py --dry-run   (chỉ vẽ neon, KHÔNG gửi chuột/phím)
Thoát: F10 (luôn nhả mọi phím / nút chuột trước khi thoát)
"""
from __future__ import annotations

import argparse
import sys
import threading
import time

from PySide6.QtWidgets import QApplication
from pynput import keyboard

from src.calibration import load_calibration
from src.capture import Camera
from src.config import load_config, resolve
from src.controller import Controller, DirectInputBackend
from src.game_db import GameDB
from src.game_detector import DetectResult, GameDetector
from src.hand_tracker import HandTracker
from src.overlay import MODE_TITLE, Overlay
from src.pipeline import DEFAULT_MOUSE, MODE_ORDER, AppState, VisionPipeline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="không gửi chuột/phím (thử cử chỉ an toàn)")
    ap.add_argument("--seconds", type=float, default=0, help="tự thoát sau N giây (dùng khi kiểm thử)")
    ap.add_argument("--mode", choices=MODE_ORDER, help="chế độ khi khởi động (mặc định theo config.yaml)")
    args = ap.parse_args()
    cfg = load_config()
    app = QApplication(sys.argv[:1])        # tạo trước để Windows bật chế độ DPI-aware cho cả tiến trình
    geo = app.primaryScreen().geometry()
    cfg["profiles"]["screen_aspect"] = geo.width() / max(geo.height(), 1)

    if load_calibration(resolve("calibration.json"), cfg["touch"]):
        print("[app] Đã nạp ngưỡng hiệu chỉnh từ calibration.json")
    start_mode = args.mode or cfg["profiles"]["start_mode"]
    state = AppState(mode=start_mode, mouse_mode=DEFAULT_MOUSE.get(start_mode, "absolute"),
                     show_preview=cfg["overlay"].get("camera_preview", True))
    controller = Controller(DirectInputBackend(), cfg["profiles"]).start()
    if args.dry_run:
        controller.set_enabled(False)
        state.enabled = False
    gd = cfg["game_detect"]
    db = GameDB(resolve(gd["db_path"]), gd["hash_max_distance"], gd["hist_min_correlation"])
    current_game = {"entry": None}
    # Các đối tượng nặng (camera ~3 s, MediaPipe ~7 s) được tạo ở luồng nền để overlay hiện ngay
    rt: dict = {"camera": None, "tracker": None, "pipeline": None}

    def on_detect(res: DetectResult) -> None:
        if res.entry is None:
            state.notify(res.message)
            return
        e = res.entry
        current_game["entry"] = e
        with state.lock:
            state.game_name = e.name
        if e.mode in MODE_ORDER and rt["pipeline"] is not None:
            rt["pipeline"].set_mode(e.mode, e.mouse_mode)
            state.notify(f"{res.message} → {MODE_TITLE[e.mode]}")
        else:
            state.notify(f"{res.message} → không có chế độ phù hợp, giữ nguyên")

    detector = GameDetector(db, gd, on_detect).start()

    def on_circle() -> None:
        if detector.trigger():
            state.notify("Đang nhận diện game...", seconds=10)

    def boot() -> None:
        try:
            state.notify("Đang mở camera...", seconds=60)
            rt["camera"] = Camera(cfg["camera"]).start()
            state.notify("Đang tải MediaPipe (lần đầu có thể mất ~10 giây)...", seconds=60)
            rt["tracker"] = HandTracker(cfg["tracker"], resolve(cfg["tracker"]["model_path"]))
            if state.quit:
                return
            rt["pipeline"] = VisionPipeline(cfg, state, rt["camera"], rt["tracker"], controller, on_circle).start()
            state.notify("Sẵn sàng! Vẽ vòng tròn bằng ngón trỏ tay phải để nhận diện game.", seconds=5)
        except Exception as e:  # noqa: BLE001
            state.notify(f"Lỗi khởi động: {e}  (F10 để thoát)", seconds=3600)

    # ---------- phím tắt toàn cục ----------
    hk = cfg["hotkeys"]
    keymap = {getattr(keyboard.Key, hk[name]): name for name in hk}

    def on_press(key) -> None:
        action = keymap.get(key)
        pipeline = rt["pipeline"]
        if action == "quit":
            controller.set_enabled(False)
            state.quit = True
        elif action == "calibrate":
            if pipeline is not None:
                pipeline.start_calibration()
        elif action == "toggle_preview":
            with state.lock:
                state.show_preview = not state.show_preview
        elif action == "toggle_enabled":
            on = not controller.enabled
            controller.set_enabled(on)
            with state.lock:
                state.enabled = on
            state.notify("BẬT gửi lệnh" if on else "TẮT gửi lệnh")
        elif pipeline is None:
            return
        elif action == "cycle_mode":
            mode = MODE_ORDER[(MODE_ORDER.index(state.mode) + 1) % len(MODE_ORDER)]
            pipeline.set_mode(mode)
            state.notify(f"Chế độ: {MODE_TITLE[mode]}")
        elif action == "toggle_mouse_mode":
            if state.mode == "shoot_v2":
                state.notify("BẮN V2 tự chọn: tuyệt đối khi con trỏ hiện, tương đối khi game khóa con trỏ")
                return
            mm = "relative" if state.mouse_mode == "absolute" else "absolute"
            pipeline.set_mode(state.mode, mm)
            if current_game["entry"] is not None:
                db.set_mouse_mode(current_game["entry"], mm)
            state.notify("Chuột TƯƠNG ĐỐI" if mm == "relative" else "Chuột TUYỆT ĐỐI")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    deadline = time.perf_counter() + args.seconds if args.seconds > 0 else None

    def on_tick() -> None:
        if state.quit or (deadline is not None and time.perf_counter() > deadline):
            app.quit()

    overlay = Overlay(state, cfg["overlay"], on_tick, cfg["profiles"]["keys"])
    overlay.show()
    boot_thread = threading.Thread(target=boot, name="boot", daemon=True)
    boot_thread.start()

    try:
        code = app.exec()
    finally:
        state.quit = True
        boot_thread.join(timeout=15)
        if rt["pipeline"] is not None:
            rt["pipeline"].stop()
        controller.set_enabled(False)
        controller.stop()
        detector.stop()
        listener.stop()
        if rt["camera"] is not None:
            rt["camera"].stop()
        if rt["tracker"] is not None:
            rt["tracker"].close()
    return code


if __name__ == "__main__":
    sys.exit(main())
