"""Luồng 2 - Vision: frame -> tiền xử lý -> MediaPipe Gesture Recognizer -> xác định tay trái/phải
-> đo ngón (3D) -> tư thế (bỏ phiếu, kiểm tra chéo với mô hình) -> profile (cử chỉ có tên) -> controller.

Frame bất thường bị loại (cổ tay ra ngoài khung, ngón chĩa vào camera, điểm mốc nhảy vọt) - khi đó dùng lại
tay của frame trước trong lost_timeout giây để không nhả nút ngoài ý muốn.
Đồng thời ghi vị trí đầu ngón trỏ + tên cử chỉ vào AppState để overlay vẽ (overlay KHÔNG chờ controller).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibration import Calibrator
from .capture import Camera
from .config import resolve
from .controller import Controller
from .finger_state import (
    INDEX, PALM_POINTS, FingerMeasure, FingerTracker, MiddleHalfTracker, TouchTracker, finger_gap_ratio, measure,
    to_metric,
)
from .gestures import (
    INDEX_ONLY, NONE, POINTER, TWO_TOGETHER, CircleDetector, Hand, PoseStabilizer, classify_pose, model_conflict,
)
from .hand_identity import HandIdentity
from .hand_tracker import HandTracker
from .preprocess import Preprocessor
from .profiles.base import Profile
from .profiles.shoot_v1 import ShootV1Profile
from .profiles.shoot_v2 import ShootV2Profile
from .profiles.slice import SliceProfile
from .smoothing import OneEuroFilter, ScreenMapper
from .visualize import draw_hand, draw_region, legend

MODE_ORDER = ["slice", "shoot_v1", "shoot_v2"]
DEFAULT_MOUSE = {"slice": "absolute", "shoot_v1": "absolute", "shoot_v2": "auto"}
LABELS = ("Left", "Right")


@dataclass
class HandView:
    label: str
    name: str                 # tên cử chỉ đang nhận (rỗng nếu chưa có)
    tip: np.ndarray           # (2,) đầu ngón trỏ, chuẩn hóa màn hình
    reason: str = ""          # lý do frame bị loại (nếu có)


@dataclass
class Detail:
    """Thông tin để vẽ khung camera."""
    label: str
    landmarks: np.ndarray
    m: FingerMeasure
    states: list[bool]
    source: str
    reason: str
    model_gesture: str
    touch: str | None = None
    gap: float | None = None
    click_info: str = ""


@dataclass
class AppState:
    """Trạng thái dùng chung giữa các luồng (đọc/ghi qua lock)."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    hands: list[HandView] = field(default_factory=list)
    hands_time: float = 0.0
    mode: str = "slice"
    mouse_mode: str = "absolute"
    game_name: str = ""
    action_label: str = ""
    enabled: bool = True
    message: str = ""
    message_until: float = 0.0
    cam_fps: float = 0.0
    vision_fps: float = 0.0
    latency_ms: float = 0.0
    brightness: float = 0.0
    clahe: bool = False
    quit: bool = False
    show_preview: bool = True
    preview: np.ndarray | None = None      # ảnh camera thu nhỏ (RGB) đã vẽ khớp ngón, cho overlay
    joystick: dict | None = None           # V2: cần điều khiển {"vec": (x, y), "keys": set}
    aim_stick: dict | None = None          # V2: vòng tâm xoay của tay phải
    warning: str = ""                      # gợi ý cho người dùng (vd đừng chĩa ngón vào camera)
    auto_relative: bool = False            # V2: game đang khóa con trỏ -> đang dùng chuột tương đối
    held: str = ""                         # phím / nút chuột controller ĐANG GIỮ (chẩn đoán trên HUD)

    def notify(self, msg: str, seconds: float = 3.0) -> None:
        with self.lock:
            self.message, self.message_until = msg, time.perf_counter() + seconds
        print("[app]", msg)


def make_profile(mode: str, cfg: dict, mouse_mode: str) -> Profile:
    if mode == "shoot_v1":
        return ShootV1Profile(cfg, mouse_mode)
    if mode == "shoot_v2":
        return ShootV2Profile(cfg, mouse_mode)
    return SliceProfile(cfg, mouse_mode)


class VisionPipeline:
    def __init__(self, cfg: dict, state: AppState, camera: Camera | None, tracker: HandTracker | None,
                 controller: Controller, on_circle):
        self.cfg, self.state, self.camera, self.tracker = cfg, state, camera, tracker
        self.controller, self.on_circle = controller, on_circle
        self.pre = Preprocessor(cfg["preprocess"])
        self.region = self._region_for(state.mode)
        self.mapper = ScreenMapper(self.region)
        sm, fc, gc = cfg["smoothing"], cfg["fingers"], cfg["gestures"]
        self.filters = {lb: OneEuroFilter(sm["min_cutoff"], sm["beta"], sm["d_cutoff"]) for lb in LABELS}
        self.knuckle_filters = {lb: OneEuroFilter(sm["min_cutoff"], sm["beta"], sm["d_cutoff"]) for lb in LABELS}
        self.fingers = {lb: FingerTracker(fc) for lb in LABELS}
        self.touch = {lb: TouchTracker(cfg["touch"]) for lb in LABELS}
        self.middle_half = {lb: MiddleHalfTracker(cfg["middle_half"]) for lb in LABELS}
        self.together: dict[str, bool] = {lb: False for lb in LABELS}
        self.stabilizers = {lb: PoseStabilizer(gc["stable_frames"], gc["stable_need"]) for lb in LABELS}
        self.identity = HandIdentity()
        self.last_seen: dict[str, tuple[float, Hand]] = {}
        self.prev_wrist: dict[str, tuple[np.ndarray, bool]] = {}
        self.circle = CircleDetector(cfg["circle"])
        self.calibrator = Calibrator(resolve("calibration.json"), cfg["touch"], state.notify)
        cfg["profiles"]["control_aspect"] = self._control_aspect(self._region_for(state.mode))
        cam = cfg.get("camera", {})
        cfg["profiles"]["camera_aspect"] = float(cam.get("width", 640)) / float(cam.get("height", 480))
        self.profile: Profile = make_profile(state.mode, cfg["profiles"], state.mouse_mode)
        self._profile_lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_bgr: np.ndarray | None = None
        self._details: list[Detail] = []
        self._reasons: dict[str, str] = {}

    def _control_aspect(self, region) -> float:
        """Tỉ lệ rộng/cao THẬT (pixel camera) của vùng điều khiển: dùng để đo độ lệch tay như nhau ở mọi hướng."""
        cam = self.cfg.get("camera", {})
        w, h = float(cam.get("width", 640)), float(cam.get("height", 480))
        x0, y0, x1, y1 = region
        return (x1 - x0) * w / max((y1 - y0) * h, 1e-6)

    def _region_for(self, mode: str) -> list[float]:
        """Vùng hoạt động (phần khung camera được kéo giãn ra toàn màn hình) riêng cho từng chế độ."""
        mp = self.cfg["mapping"]
        return list(mp.get("region_by_mode", {}).get(mode, mp["region"]))

    # ---------- đổi chế độ (phím tắt hoặc game detector) ----------
    def set_mode(self, mode: str, mouse_mode: str | None = None) -> None:
        with self._profile_lock:
            self.controller.release_all()
            mm = mouse_mode or DEFAULT_MOUSE.get(mode, "absolute")
            if mode == "shoot_v2":
                mm = "auto"              # V2: tuyệt đối khi con trỏ hiện, tương đối khi game khóa con trỏ
            with self.state.lock:
                self.state.mode, self.state.mouse_mode, self.state.joystick = mode, mm, None
            self.cfg["profiles"]["control_aspect"] = self._control_aspect(self._region_for(mode))
            self.profile = make_profile(mode, self.cfg["profiles"], mm)
            self.region = self._region_for(mode)
            self.mapper = ScreenMapper(self.region)
            for f in self.filters.values():          # vị trí cũ theo vùng cũ -> làm mới bộ lọc tránh con trỏ giật
                f.reset()

    def start_calibration(self) -> None:
        self.calibrator.start(time.perf_counter())

    def start(self) -> "VisionPipeline":
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="vision", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)     # chờ frame đang xử lý xong rồi mới đóng camera/MediaPipe

    # ---------- kiểm tra frame bất thường ----------
    def _reject_reason(self, label: str, lm: np.ndarray, m: FingerMeasure, palm_px: float, w: int, h: int) -> str:
        gc = self.cfg["gestures"]
        e = gc["edge_margin"]
        wx, wy = float(lm[0, 0]), float(lm[0, 1])
        mcp = lm[[5, 9, 13, 17], :2]
        out = int(np.sum((mcp < -e).any(axis=1) | (mcp > 1 + e).any(axis=1)))
        if out >= 2 or wy > gc["wrist_y_max"]:      # cổ tay chỉ sát mép dưới thì vẫn nhận
            return "Bàn tay ra ngoài khung - đưa tay lên cao hơn"
        if m.angles[INDEX] > self.cfg["fingers"]["extend_angle"] and m.foreshorten < gc["foreshorten_min"]:
            return "Giơ tay dọc, đừng chĩa ngón vào camera"
        wrist_px = np.array([wx * w, wy * h])
        prev = self.prev_wrist.get(label)
        jumped = prev is not None and float(np.linalg.norm(wrist_px - prev[0])) > gc["jump_ratio"] * palm_px
        self.prev_wrist[label] = (wrist_px, jumped)
        if jumped and not prev[1]:      # nhảy 1 frame -> bỏ; nhảy 2 frame liên tiếp -> chấp nhận vị trí mới
            return "Điểm mốc nhảy bất thường"
        return ""

    # ---------- xử lý một frame ----------
    def process(self, frame: np.ndarray, t: float) -> dict[str, Hand]:
        pre = self.pre(frame)
        h_img, w_img = pre.rgb.shape[:2]
        raw_hands = self.tracker.detect(pre.rgb)
        return self.process_raw(raw_hands, t, w_img, h_img, pre)

    def _together(self, label: str, m: FingerMeasure, gap: float | None) -> bool:
        """Khép trỏ + giữa: khoảng cách cả ngón nhỏ, HOẶC khoảng cách trung bình mà ảnh không lộ khe nền."""
        fc = self.cfg["fingers"]
        cur = self.together[label]
        d = m.together
        if d < fc["together_ratio"]:
            cur = True
        elif d > fc["spread_ratio"]:
            cur = False
        elif gap is not None:
            if gap < fc["gap_max"]:
                cur = True
            elif gap > fc["gap_max"] * 1.6:
                cur = False
        self.together[label] = cur
        return cur

    def _click_info(self, m: FingerMeasure, states: list[bool], touch, mid_half: bool) -> str:
        """Dòng chẩn đoán trong khung camera: vì sao CÓ / KHÔNG nhận click."""
        tc, mc = self.cfg["touch"], self.cfg["middle_half"]
        if not states[INDEX]:
            return "click: ngon tro dang cup"
        ok2 = m.touch_index_2d < tc["touch_on_2d"]
        ok3 = m.touch_index_3d < tc["touch_on_3d"]
        if touch:
            left = "TRAI: dang cham"
        else:
            left = (f"trai: 2D {m.touch_index_2d:.2f}{'<' if ok2 else '>'}{tc['touch_on_2d']}"
                    f" 3D {m.touch_index_3d:.2f}{'<' if ok3 else '>'}{tc['touch_on_3d']}")
        if mid_half:
            right = "PHAI: giua nua chung"
        else:
            right = f"phai: giua/tro {m.middle_ratio:.2f} (can {mc['half_lo']}-{mc['half_hi']})"
        return f"{left} | {right}"

    def process_raw(self, raw_hands, t: float, w_img: int, h_img: int, pre=None) -> dict[str, Hand]:
        fc, gc = self.cfg["fingers"], self.cfg["gestures"]
        hands: dict[str, Hand] = {}
        self._details, self._reasons = [], {}
        right_measure = None
        for ident in self.identity.assign(raw_hands, t):
            rh, label = ident.raw, ident.label
            img_metric = to_metric(rh.landmarks, w_img, h_img)
            m = measure(rh.world, img_metric)
            palm_px = float(np.linalg.norm(img_metric[9, :2] - img_metric[0, :2])) + 1e-6
            reason = self._reject_reason(label, rh.landmarks, m, palm_px, w_img, h_img)
            tracker = self.fingers[label]
            if reason:
                self._reasons[label] = reason
                self._details.append(Detail(label, rh.landmarks, m, list(tracker.stable), ident.source, reason,
                                            rh.model_gesture))
                continue
            if label == "Right":
                right_measure = m
            states = tracker.update(m)
            touch = self.touch[label].update(m)
            mid_half = self.middle_half[label].update(m.middle_ratio, states[INDEX])
            click_info = self._click_info(m, states, touch, mid_half)
            gap = finger_gap_ratio(pre.bgr, rh.landmarks) if pre is not None and states[1] and states[2] else None
            together = self._together(label, m, gap)
            pose = classify_pose(states, together)
            if model_conflict(pose, rh.model_gesture, rh.model_score, gc["model_min_score"]):
                pose = NONE
            pose = self.stabilizers[label].update(pose)
            tip = self.filters[label](self.mapper(rh.landmarks[[8], :2])[0], t)
            hand = Hand(
                label=label, states=states, leaving=list(tracker.leaving), pose=pose, together=together,
                vertical=m.up_angle < gc["vertical_max_deg"], tip=tip,
                palm_center=img_metric[PALM_POINTS, :2].mean(axis=0), palm_px=palm_px,
                cam=rh.landmarks, world=rh.world, source=ident.source, model_gesture=rh.model_gesture,
                touch=touch, approach=self.touch[label].approach, middle_half=mid_half,
                fingers_px=(img_metric[8, :2] + img_metric[12, :2]) / 2,
                palm_screen=self.mapper(rh.landmarks[PALM_POINTS, :2].mean(axis=0, keepdims=True))[0],
                knuckle_screen=self.knuckle_filters[label](
                    self.mapper(rh.landmarks[[5, 9], :2].mean(axis=0, keepdims=True))[0], t),
            )
            hands[label] = hand
            self.last_seen[label] = (t, hand)
            self._details.append(Detail(label, rh.landmarks, m, states, ident.source, "", rh.model_gesture,
                                        touch, gap, click_info))
        self.calibrator.update(right_measure, t)
        # Mất tay / frame bị loại trong thời gian ngắn -> dùng lại tay cũ, tránh nhả nút ngoài ý muốn
        for label, (seen, hand) in list(self.last_seen.items()):
            if label in hands:
                continue
            if t - seen <= fc["lost_timeout"]:
                hands[label] = hand
            else:
                del self.last_seen[label]
                self.filters[label].reset()
                self.knuckle_filters[label].reset()
                self.fingers[label].reset()
                self.touch[label].reset()
                self.middle_half[label].reset()
                self.together[label] = False
                self.stabilizers[label].reset()
                self.prev_wrist.pop(label, None)
        if pre is not None:
            self._last_bgr = pre.bgr
            with self.state.lock:
                self.state.brightness, self.state.clahe = pre.brightness, pre.clahe_applied
        return hands

    def step(self, hands: dict[str, Hand], t: float):
        """Vòng tròn + profile + controller cho một frame. Trả về Desired."""
        right = hands.get("Right")
        if right is not None and right.index and right.pose in (POINTER, INDEX_ONLY, TWO_TOGETHER)                 and self.state.mode != "shoot_v2":
            tip = right.cam[8]
            if self.circle.update(float(tip[0]), float(tip[1]), t):
                self.on_circle()
        else:
            self.circle.reset()
        with self._profile_lock:
            desired = self.profile.update(hands, t)
            self.controller.apply(desired)
        return desired

    def _loop(self) -> None:
        seq, n, t0 = 0, 0, time.perf_counter()
        while self._running:
            got = self.camera.latest(seq, timeout=1.0)
            if got is None:
                continue
            seq, frame, stamp = got
            t = time.perf_counter()
            hands = self.process(frame, t)
            desired = self.step(hands, t)
            done = time.perf_counter()
            # Ảnh xem trước vẽ SAU khi đã gửi lệnh -> không cộng thêm độ trễ điều khiển
            preview = self.render_preview(desired.gestures) if self.state.show_preview else None
            warning = next(iter(self._reasons.values()), "")
            n += 1
            with self.state.lock:
                self.state.hands = self.hand_views(hands, desired.gestures)
                self.state.hands_time = done
                self.state.action_label = desired.label
                self.state.joystick = desired.joystick
                self.state.aim_stick = desired.aim_stick
                self.state.warning = warning
                self.state.latency_ms = (done - stamp) * 1000
                self.state.cam_fps = self.camera.fps
                self.state.auto_relative = self.controller.auto_relative
                held = sorted(k.upper() for k in self.controller.keys) + \
                    [{"left": "Chuột trái", "right": "Chuột phải"}.get(b, b) for b in sorted(self.controller.buttons)]
                self.state.held = ", ".join(held)
                self.state.preview = preview
                if done - t0 >= 1.0:
                    self.state.vision_fps, n, t0 = n / (done - t0), 0, done

    def hand_views(self, hands: dict[str, Hand], names: dict[str, str]) -> list[HandView]:
        """Các tay được vẽ vệt sáng trên overlay.
        BẮN V2: tay phải chỉ xoay màn hình -> không vẽ; tay trái vẽ ở ĐỐT NGÓN (điểm điều khiển cần di chuyển)."""
        v2 = self.state.mode == "shoot_v2"
        out = []
        for lb, h in hands.items():
            if v2 and lb == "Right":
                continue
            pos = h.knuckle_screen if (v2 and h.knuckle_screen is not None) else h.tip
            out.append(HandView(lb, names.get(lb, ""), pos, self._reasons.get(lb, "")))
        return out

    def render_preview(self, names: dict[str, str]) -> np.ndarray | None:
        """Ảnh camera thu nhỏ + khung xương bàn tay + góc khớp + tên cử chỉ (RGB) cho góc màn hình."""
        if self._last_bgr is None:
            return None
        width = int(self.cfg["overlay"].get("preview_width", 360))
        h, w = self._last_bgr.shape[:2]
        img = cv2.resize(self._last_bgr, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        draw_region(img, self.region)
        for dt in self._details:
            draw_hand(img, dt.landmarks, dt.states, dt.m, dt.label, names.get(dt.label, ""),
                      source=dt.source, reason=dt.reason, model_gesture=dt.model_gesture, touch=dt.touch, gap=dt.gap,
                      click_info=dt.click_info)
        fc = self.cfg["fingers"]
        legend(img, fc["extend_angle"], fc["fold_angle"])
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
