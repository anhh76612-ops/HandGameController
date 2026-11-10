"""Kiểm thử logic không cần camera: đo ngón 3D, chạm ngón cái, khe hở giữa ngón, tư thế, xác định tay,
con trỏ chung (không đệ quy, dính, khóa khi click), 3 profile, controller 120 Hz, hiệu chỉnh, vòng tròn, cache game."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from src.calibration import Calibrator
from src.config import load_config
from src.controller import Controller
from src.finger_state import FingerTracker, MiddleHalfTracker, TouchTracker, finger_gap_ratio, measure, to_metric
from src.game_db import GameDB, dhash, hamming
from src.game_detector import GameDetector
from src.gestures import (
    BOP_CO, CHI_LEN, CLICK_PHAI, CLICK_TRAI, CON_TRO, DI_CHUYEN, DUNG_LAI, FIST, GIU_TRAI, INDEX_ONLY,
    KEO_CHEM, KHONG_RO, MO_NGAM, NAP_DAN, NHAY, NONE, OPEN, POINTER, SUNG, TAM_DUNG, THREE, TWO_TOGETHER,
    CircleDetector, Hand, PoseStabilizer, classify_pose, model_conflict,
)
from src.hand_identity import HandIdentity
from src.hand_tracker import RawHand
from src.pipeline import AppState, VisionPipeline
from src.profiles.base import CommonPointer, Desired, StickyCursor
from src.profiles.shoot_v1 import ShootV1Profile
from src.profiles.shoot_v2 import ShootV2Profile, sector_of
from src.profiles.slice import SliceProfile
from src.smoothing import OneEuroFilter

CFG = load_config()
FCFG = {**CFG["fingers"], "debounce_frames": 1}
TCFG = {**CFG["touch"], "touch_debounce": 1}
PCFG = CFG["profiles"]
W, H = 640, 480
T, F = True, False

# ---------------- bàn tay tổng hợp ----------------
BASE_X = {1: -0.03, 2: 0.0, 3: 0.03, 4: 0.06}          # gốc ngón trỏ, giữa, áp út, út (so với cổ tay)


def synth_hand(ext: list[bool], wrist=(0.5, 0.75), together=False, touch: str | None = None) -> np.ndarray:
    """21 điểm mốc chuẩn hóa. ext = [cái, trỏ, giữa, áp út, út]. touch: đặt đầu ngón cái lên ngón trỏ/giữa."""
    lm = np.zeros((21, 3), dtype=np.float32)
    wx, wy = wrist
    lm[0] = (wx, wy, 0)
    thumb = ([(-0.04, -0.03), (-0.08, -0.05), (-0.11, -0.07), (-0.14, -0.09)] if ext[0]
             else [(-0.03, -0.03), (-0.03, -0.06), (-0.015, -0.085), (0.0, -0.095)])
    for i, (dx, dy) in enumerate(thumb):
        lm[1 + i] = (wx + dx, wy + dy, 0)
    for f in range(1, 5):
        mcp = 1 + 4 * f
        bx, by = BASE_X[f], -0.10
        lm[mcp] = (wx + bx, wy + by, 0)
        lean = (0.16 if together else -0.25) if f == 1 else (-0.16 if together else 0.25) if f == 2 else 0.0
        segs = [(0, -0.04), (0, -0.07), (0, -0.095)] if ext[f] else [(0, -0.035), (0, -0.02), (0, -0.005)]
        for k, (_dx, dy) in enumerate(segs):
            dx = lean * abs(dy) if ext[f] else 0.0
            lm[mcp + 1 + k] = (wx + bx + dx, wy + by + dy, -0.03 * k if not ext[f] else 0)
    if touch:
        f0 = 5 if touch == "index" else 9
        target = (lm[f0 + 1] + lm[f0 + 2]) / 2
        lm[4] = target + np.array([-0.004, 0.0, 0.0], dtype=np.float32)
        lm[3] = (lm[2] + lm[4]) / 2
    return lm


def world_of(lm: np.ndarray) -> np.ndarray:
    return (to_metric(lm, W, H) / 1000.0).astype(np.float32)


def meas(lm):
    return measure(world_of(lm), to_metric(lm, W, H))


def pose_of(ext, together=False):
    lm = synth_hand(ext, together=together)
    m = meas(lm)
    states = FingerTracker(FCFG).update(m)
    tog = m.together < FCFG["together_ratio"]
    return classify_pose(states, tog), states, m


P_TWO = [T, T, T, F, F]          # cái + trỏ + giữa (tư thế CON TRỎ / DI CHUYỂN)
P_INDEX = [F, T, F, F, F]
P_THREE = [T, T, T, T, F]
P_OPEN = [T] * 5
P_FIST = [F] * 5


@pytest.mark.parametrize("ext,together,expected", [
    (P_TWO, False, POINTER),
    (P_TWO, True, TWO_TOGETHER),
    (P_INDEX, False, INDEX_ONLY),
    (P_THREE, False, THREE),
    (P_OPEN, False, OPEN),
    (P_FIST, False, FIST),
    ([T, T, F, T, T], False, NONE),
])
def test_pose_from_synthetic_hand(ext, together, expected):
    pose, states, _ = pose_of(ext, together)
    assert states[1:] == ext[1:]
    assert pose == expected


def test_together_uses_whole_finger_distance():
    assert meas(synth_hand(P_TWO, together=True)).together < FCFG["together_ratio"]
    assert meas(synth_hand(P_TWO, together=False)).together > FCFG["spread_ratio"]


# ---------------- chạm ngón cái ----------------
def test_touch_index_anywhere_on_the_line():
    tt = TouchTracker(TCFG)
    assert tt.update(meas(synth_hand(P_TWO))) is None
    assert tt.update(meas(synth_hand(P_TWO, touch="index"))) == "index"
    assert tt.update(meas(synth_hand(P_TWO))) is None
    # chạm phần SÁT GỐC ngón trỏ (đoạn 5-6) cũng là click trái
    lm = synth_hand(P_TWO)
    lm[4] = lm[5] + 0.3 * (lm[6] - lm[5]) + np.array([-0.004, 0, 0], dtype=np.float32)
    lm[3] = (lm[2] + lm[4]) / 2
    assert tt.update(meas(lm)) == "index"
    # chạm ngón giữa không còn là click (click phải = ngón giữa nửa chừng)
    assert TouchTracker(TCFG).update(meas(synth_hand(P_TWO, touch="middle"))) is None


def test_middle_half_tracker():
    mh = MiddleHalfTracker(CFG["middle_half"])
    assert not any(mh.update(1.0, True) for _ in range(5))          # duỗi thẳng
    # frame đầu nhảy vào vùng nửa chừng chưa "đứng yên" -> cần thêm 3 frame ổn định
    assert [mh.update(0.5, True) for _ in range(4)] == [False, False, False, True]
    assert mh.update(0.6, True)                                      # vẫn nửa chừng
    assert not mh.update(1.0, True)                                  # duỗi thẳng -> nhả
    # gập nhanh đi XUYÊN qua vùng nửa chừng -> không tính
    mh.reset()
    assert not any(mh.update(r, True) for r in (1.0, 0.65, 0.4, 0.15, 0.12, 0.12))
    assert not mh.update(0.5, False)                                 # ngón trỏ cụp -> không click


def test_touch_debounce_and_approach():
    tt = TouchTracker({**CFG["touch"], "touch_debounce": 2})
    m_touch = meas(synth_hand(P_TWO, touch="index"))
    assert tt.update(m_touch) is None and tt.approach
    assert tt.update(m_touch) == "index"


# ---------------- khe hở giữa 2 ngón (xử lý ảnh) ----------------
def _finger_image(gap: bool) -> tuple[np.ndarray, np.ndarray]:
    img = np.full((480, 640, 3), (40, 90, 30), dtype=np.uint8)          # nền xanh
    lm = synth_hand(P_TWO, together=False)
    pts = (lm[:, :2] * [640, 480]).astype(int)
    skin = (120, 150, 210)                                              # màu da (BGR)
    thick = 10 if gap else 40
    import cv2
    for a, b in ((5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12)):
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), skin, thick)
    if not gap:
        cv2.fillPoly(img, [pts[[5, 6, 7, 8, 12, 11, 10, 9]]], skin)
    return img, lm


def test_gap_ratio_sees_background_between_fingers():
    img_gap, lm = _finger_image(gap=True)
    img_closed, _ = _finger_image(gap=False)
    assert finger_gap_ratio(img_gap, lm) > 0.5
    assert finger_gap_ratio(img_closed, lm) < 0.1


def test_model_conflict_and_stabilizer():
    assert model_conflict(POINTER, "Closed_Fist", 0.9, 0.6)
    assert not model_conflict(POINTER, "Closed_Fist", 0.3, 0.6)
    st = PoseStabilizer(4, 3)
    assert [st.update(p) for p in (POINTER, POINTER, FIST, POINTER)] == [NONE, NONE, NONE, POINTER]
    assert st.update(FIST) == POINTER


# ---------------- xác định tay trái/phải ----------------
def raw(label, wrist_x, score=0.9):
    lm = synth_hand(P_TWO, wrist=(wrist_x, 0.75))
    return RawHand(label, score, lm, world_of(lm))


def test_identity_two_hands_by_position():
    out = HandIdentity().assign([raw("Left", 0.8), raw("Right", 0.2)], 0.0)
    labels = {o.label: float(o.raw.landmarks[0, 0]) for o in out}
    assert labels["Left"] < labels["Right"]


def test_identity_single_hand_uses_side_of_image():
    assert HandIdentity().assign([raw("Right", 0.2)], 0.0)[0].label == "Left"
    assert HandIdentity().assign([raw("Left", 0.8)], 0.0)[0].label == "Right"


def test_identity_learns_swap_for_center_hand():
    ident = HandIdentity()
    for i in range(6):
        ident.assign([raw("Left", 0.8), raw("Right", 0.2)], i * 0.03)
    out = ident.assign([raw("Left", 0.5)], 2.0)
    assert out[0].label == "Right" and out[0].source == "MediaPipe"


def test_identity_tracking_and_duplicate_detection():
    ident = HandIdentity()
    labels = [ident.assign([raw(lb, 0.5 + 0.005 * i)], i * 0.03)[0].label
              for i, lb in enumerate(["Right"] * 8 + ["Left"] + ["Right"] * 3)]
    assert set(labels) == {"Right"}
    out = HandIdentity().assign([raw("Right", 0.20, 0.9), raw("Right", 0.22, 0.6)], 0.0)
    assert len(out) == 1


# ---------------- Hand giả cho profile ----------------
def make_hand(label, states, pose, tip=(0.5, 0.5), together=False, vertical=True, touch=None, approach=False,
              palm_center=(320.0, 300.0), fingers_px=None, palm_px=50.0, middle_half=False, palm_screen=(0.3, 0.7),
              knuckle=None):
    fp = np.array(fingers_px if fingers_px is not None else (palm_center[0], palm_center[1] - 80), dtype=float)
    return Hand(label, list(states), [False] * 5, pose, together, vertical, np.array(tip, dtype=float),
                np.array(palm_center, dtype=float), palm_px, cam=np.zeros((21, 3)), touch=touch,
                approach=approach, fingers_px=fp, middle_half=middle_half,
                palm_screen=np.array(palm_screen, dtype=float),
                knuckle_screen=np.array(knuckle if knuckle is not None else tip, dtype=float))


# ---------------- con trỏ chung ----------------
def test_common_pointer_click_with_thumb_touch():
    cp = CommonPointer(PCFG)
    out = cp.update(make_hand("Right", P_TWO, POINTER, tip=(0.2, 0.3)), 0.0)
    assert out.name == CON_TRO and out.cursor == pytest.approx((0.2, 0.3)) and not out.buttons
    out = cp.update(make_hand("Right", P_TWO, POINTER, tip=(0.2, 0.3)), 0.2)
    # ngón cái sắp chạm -> khóa; tay lệch đi nhưng con trỏ đứng yên
    out = cp.update(make_hand("Right", P_TWO, POINTER, tip=(0.25, 0.35), approach=True), 0.23)
    assert out.cursor == pytest.approx((0.2, 0.3))
    out = cp.update(make_hand("Right", P_TWO, POINTER, tip=(0.26, 0.36), touch="index", approach=True), 0.26)
    assert out.name == CLICK_TRAI and out.buttons == {"left"} and out.cursor == pytest.approx((0.2, 0.3))
    out = cp.update(make_hand("Right", P_TWO, POINTER, tip=(0.2, 0.3)), 0.5)
    assert out.name == CON_TRO and not out.buttons


def test_common_pointer_right_click_with_half_middle():
    cp = CommonPointer(PCFG)
    cp.update(make_hand("Right", P_TWO, POINTER), 0.0)
    out = cp.update(make_hand("Right", P_TWO, POINTER, middle_half=True), 0.05)
    assert out.name == CLICK_PHAI and out.buttons == {"right"}
    assert not cp.update(make_hand("Right", P_TWO, POINTER), 0.1).buttons


def test_quick_click_keeps_cursor_but_long_hold_drags():
    cp = CommonPointer(PCFG)
    for i in range(5):
        cp.update(make_hand("Right", P_TWO, POINTER, tip=(0.3, 0.3)), i * 0.033)
    # click nhanh: tay lệch trong lúc chạm nhưng con trỏ đứng yên, nhả xong vẫn ở chỗ cũ
    out = cp.update(make_hand("Right", P_TWO, POINTER, tip=(0.33, 0.31), touch="index"), 0.2)
    assert out.name == CLICK_TRAI and out.cursor == pytest.approx((0.3, 0.3))
    out = cp.update(make_hand("Right", P_TWO, POINTER, tip=(0.302, 0.301)), 0.3)
    assert not out.buttons and out.cursor == pytest.approx((0.3, 0.3))
    # giữ lâu: sau click_hold_ms con trỏ đi theo tay để kéo / bôi đen
    cp.update(make_hand("Right", P_TWO, POINTER, tip=(0.3, 0.3), touch="index"), 1.0)
    out = cp.update(make_hand("Right", P_TWO, POINTER, tip=(0.3, 0.3), touch="index"), 1.2)
    assert out.name == CLICK_TRAI
    out = cp.update(make_hand("Right", P_TWO, POINTER, tip=(0.5, 0.45), touch="index"), 1.4)
    assert out.name == GIU_TRAI and out.buttons == {"left"} and out.cursor == pytest.approx((0.5, 0.45))


def test_recursion_bug_fixed_when_lowering_hand_from_drag():
    """Tái hiện lỗi cũ: tư thế (đã bỏ phiếu) vẫn là TWO_TOGETHER nhưng các ngón đã cụp khi hạ tay."""
    cp = CommonPointer(PCFG)
    assert cp.update(make_hand("Right", P_TWO, TWO_TOGETHER, together=True), 0.0).name == KEO_CHEM
    out = cp.update(make_hand("Right", P_FIST, TWO_TOGETHER, together=True), 0.03)
    assert out.name == KHONG_RO and not out.buttons
    assert cp.update(None, 0.06).buttons == set()


def test_drag_then_spread_back_to_pointer():
    cp = CommonPointer(PCFG)
    out = cp.update(make_hand("Right", P_TWO, TWO_TOGETHER, tip=(0.6, 0.6), together=True), 0.0)
    assert out.name == KEO_CHEM and out.buttons == {"left"}
    out = cp.update(make_hand("Right", P_TWO, POINTER, tip=(0.6, 0.6)), 0.05)
    assert out.name == CON_TRO and not out.buttons


def test_unknown_pose_and_pause():
    cp = CommonPointer(PCFG)
    out = cp.update(make_hand("Right", [F, T, F, T, T], NONE), 0.0)
    assert out.name == KHONG_RO and out.cursor is None
    taps = []
    for i in range(40):
        out = cp.update(make_hand("Right", P_OPEN, OPEN), i * 0.05)
        taps += out.taps
    assert out.name == TAM_DUNG and taps == ["esc"]


def test_sticky_cursor_ignores_small_jitter():
    sc = StickyCursor(PCFG)
    rng = np.random.default_rng(1)
    outs = [sc.update((0.5 + rng.normal(0, 0.001), 0.5 + rng.normal(0, 0.001)), i / 30) for i in range(40)]
    assert len({(round(x, 6), round(y, 6)) for x, y in outs[10:]}) == 1          # đứng im hẳn
    x, _ = sc.update((0.56, 0.5), 41 / 30)                                       # tay đi thật -> đi theo
    assert x == pytest.approx(0.56)


def test_slice_uses_right_hand_only():
    prof = SliceProfile(PCFG)
    assert not prof.update({"Left": make_hand("Left", P_TWO, POINTER)}, 0.0).gestures
    d = prof.update({"Right": make_hand("Right", P_TWO, TWO_TOGETHER, together=True)}, 0.0)
    assert d.buttons == {"left"} and d.gestures == {"Right": KEO_CHEM}


# ---------------- BẮN V1 ----------------
def test_v1_gun_cursor_at_finger_and_fire_with_thumb_touch():
    prof = ShootV1Profile({**PCFG, "v1_fire_on_close": True})
    d = prof.update({"Left": make_hand("Left", P_TWO, TWO_TOGETHER, tip=(0.3, 0.3), together=True)}, 0.0)
    assert d.gestures["Left"] == SUNG and d.cursor_abs == pytest.approx((0.3, 0.3))
    d = prof.update({"Left": make_hand("Left", P_TWO, TWO_TOGETHER, tip=(0.4, 0.2), together=True)}, 0.03)
    assert d.cursor_abs == pytest.approx((0.4, 0.2))                            # đúng chỗ đầu ngón
    d = prof.update({"Left": make_hand("Left", P_TWO, TWO_TOGETHER, tip=(0.42, 0.22), together=True,
                                       touch="index")}, 0.06)
    assert d.gestures["Left"] == BOP_CO and d.buttons == {"left"}


def test_v1_never_drag_on_right_hand():
    prof = ShootV1Profile(PCFG)
    d = prof.update({"Right": make_hand("Right", P_TWO, TWO_TOGETHER, together=True)}, 0.0)
    assert d.gestures["Right"] != KEO_CHEM and "left" not in d.buttons


def test_v1_scope_with_right_index():
    prof = ShootV1Profile(PCFG)
    gun = make_hand("Left", P_TWO, TWO_TOGETHER, together=True)
    d = prof.update({"Left": gun, "Right": make_hand("Right", P_INDEX, INDEX_ONLY)}, 0.0)
    assert d.gestures["Right"] == CHI_LEN
    d = prof.update({"Left": gun, "Right": make_hand("Right", P_FIST, INDEX_ONLY)}, 0.03)
    assert d.gestures["Right"] == MO_NGAM and "right" in d.buttons


# ---------------- BẮN V2 ----------------
def test_v2_sectors_with_hysteresis():
    assert sector_of(0, None, 8) == 0
    assert sector_of(90, None, 8) == 2 and sector_of(-90, None, 8) == 6 and sector_of(180, None, 8) == 4
    assert sector_of(27, 0, 8) == 0 and sector_of(32, 0, 8) == 1


GUN2 = dict(together=True)          # tay trái V2: khép trỏ + giữa


def _left2(tip):
    return {"Left": make_hand("Left", P_TWO, TWO_TOGETHER, tip=tip, **GUN2)}


def test_v2_left_anchor_default_forward_and_knob_is_the_dot():
    cfg = {**PCFG, "screen_aspect": 1.6, "joystick_radius": 0.08}
    prof = ShootV2Profile(cfg)
    d = prof.update(_left2((0.25, 0.6)), 0.0)
    assert d.gestures["Left"] == DI_CHUYEN and d.keys == {"w"}               # mặc định đi thẳng
    assert d.joystick["center"] == pytest.approx((0.25, 0.6)) and d.joystick["active"]
    # chấm đi sang phải 0.03 bề rộng = 0.048 chiều cao -> núm ở đúng chỗ chấm (0.048/0.08 = 0.6 bán kính)
    d = prof.update(_left2((0.28, 0.6)), 0.03)
    assert d.joystick["center"] == pytest.approx((0.25, 0.6))               # tâm vẫn ở vị trí gốc
    assert d.joystick["vec"] == pytest.approx((0.6, 0.0)) and d.keys == {"d"}
    # chấm ra ngoài vòng tròn -> núm giữ ở mép, cùng hướng với ngón tay
    d = prof.update(_left2((0.25, 0.4)), 0.06)
    assert d.joystick["vec"] == pytest.approx((0.0, -1.0)) and d.keys == {"w"}


@pytest.mark.parametrize("dx,dy,keys", [(0, -0.05, {"w"}), (0, 0.05, {"s"}), (-0.04, 0, {"a"}), (0.04, 0, {"d"}),
                                        (0.03, -0.045, {"w", "d"}), (-0.03, 0.045, {"s", "a"})])
def test_v2_left_direction_follows_the_dot(dx, dy, keys):
    prof = ShootV2Profile({**PCFG, "screen_aspect": 1.6})
    prof.update(_left2((0.3, 0.6)), 0.0)
    d = prof.update(_left2((0.3 + dx, 0.6 + dy)), 0.03)
    assert d.keys == keys
    vx, vy = d.joystick["vec"]
    assert vx * dx >= 0 and vy * dy >= 0                                    # núm cùng hướng với ngón tay


def test_v2_left_small_jitter_stays_forward():
    prof = ShootV2Profile({**PCFG, "screen_aspect": 1.6})
    prof.update(_left2((0.3, 0.6)), 0.0)
    d = prof.update(_left2((0.305, 0.603)), 0.03)                          # trong vùng chết
    assert d.keys == {"w"}


def test_v2_left_stop_jump():
    prof = ShootV2Profile(PCFG)
    hand = dict(tip=(0.3, 0.6), **GUN2)
    prof.update({"Left": make_hand("Left", P_TWO, TWO_TOGETHER, **hand)}, 0.0)
    d = prof.update({"Left": make_hand("Left", [F, T, T, F, F], TWO_TOGETHER, **hand)}, 0.03)
    assert d.gestures["Left"] == NHAY and d.keys == {"w", "space"}             # khép ngón cái -> nhảy
    d = prof.update({"Left": make_hand("Left", P_FIST, FIST)}, 0.1)
    assert d.gestures["Left"].startswith(DUNG_LAI) and not d.keys              # nắm tay -> dừng
    d = prof.update({"Left": make_hand("Left", P_OPEN, OPEN)}, 0.2)              # xòe tay trái = ĐỨNG LẠI
    assert d.gestures["Left"] == DUNG_LAI and not d.taps and not d.keys
    # đang đi rồi xòe tay -> đứng lại nhưng GIỮ NGUYÊN tâm (khác với nắm tay là đặt lại tâm)
    prof2 = ShootV2Profile(PCFG)
    prof2.update({"Left": make_hand("Left", P_TWO, TWO_TOGETHER, **hand)}, 0.0)
    d = prof2.update({"Left": make_hand("Left", P_OPEN, OPEN)}, 0.05)
    assert d.gestures["Left"] == DUNG_LAI and not d.keys and d.joystick["center"] == pytest.approx((0.3, 0.6))
    d = prof.update({}, 0.4)                                                    # mất tay -> đứng yên
    assert not d.keys


def test_v2_right_open_palm_reloads_once_and_no_pause():
    prof = ShootV2Profile(PCFG)
    taps = []
    for i in range(45):
        d = prof.update({"Right": make_hand("Right", P_OPEN, OPEN, tip=(0.5 + i * 0.01, 0.5))}, i * 0.033)
        taps += d.taps
    assert taps == ["r"] and d.gestures["Right"] == NAP_DAN                    # chỉ R một lần, không Esc
    assert d.cursor_rel is None and d.cursor_vel is None                        # xòe tay không xoay


def test_v1_no_pause_in_shooting_mode():
    prof = ShootV1Profile(PCFG)
    taps = []
    for i in range(45):
        taps += prof.update({"Right": make_hand("Right", P_OPEN, OPEN)}, i * 0.033).taps
    assert not taps


def test_v2_right_sends_abs_and_rel_and_clicks():
    prof = ShootV2Profile({**PCFG, "screen_aspect": 1.6, "aim_deadzone": 0.02, "aim_point_radius": 0.20, "aim_radius": 0.32, "aim_point_gain": 1.0})
    prof.update({"Right": make_hand("Right", P_TWO, POINTER, tip=(0.5, 0.5))}, 0.0)
    d = prof.update({"Right": make_hand("Right", P_TWO, POINTER, tip=(0.55, 0.5))}, 0.033)   # vòng 2
    assert d.cursor_auto and d.cursor_abs == pytest.approx((0.55, 0.5)) and d.cursor_rel[0] > 0
    d = prof.update({"Right": make_hand("Right", P_TWO, POINTER, tip=(0.56, 0.5), touch="index")}, 0.066)
    assert d.buttons == {"left"} and d.gestures["Right"] == "BẮN" and d.cursor_rel[0] > 0   # bắn vẫn xoay
    d = prof.update({"Right": make_hand("Right", P_TWO, POINTER, tip=(0.56, 0.5))}, 0.1)
    assert not d.buttons and d.gestures["Right"] == "XOAY"

def test_v2_right_index_rotates_together_aims_thumb_fires():
    prof = ShootV2Profile({**PCFG, "screen_aspect": 1.6, "aim_deadzone": 0.02, "aim_point_radius": 0.20, "aim_radius": 0.32, "aim_point_gain": 1.0})
    one = [T, T, F, F, F]
    prof.update({"Right": make_hand("Right", one, INDEX_ONLY, tip=(0.5, 0.5))}, 0.0)
    d = prof.update({"Right": make_hand("Right", one, INDEX_ONLY, tip=(0.55, 0.5))}, 0.033)
    assert d.gestures["Right"] == "XOAY" and not d.buttons and d.cursor_rel[0] > 0
    # khép trỏ + giữa -> ngắm (chuột phải), vẫn xoay được
    d = prof.update({"Right": make_hand("Right", P_TWO, TWO_TOGETHER, tip=(0.58, 0.5), together=True)}, 0.066)
    assert d.buttons == {"right"} and d.gestures["Right"] == "NGẮM" and d.cursor_rel[0] > 0
    # khép ngón cái khi đang ngắm -> vừa ngắm vừa bắn, vẫn xoay được
    d = prof.update({"Right": make_hand("Right", [F, T, T, F, F], TWO_TOGETHER, tip=(0.6, 0.5), together=True)}, 0.1)
    assert d.buttons == {"left", "right"} and d.gestures["Right"] == "NGẮM + BẮN"
    assert d.cursor_rel[0] > 0
    prof2 = ShootV2Profile(PCFG)
    assert "left" not in prof2.update({"Right": make_hand("Right", [F, T, F, F, F], INDEX_ONLY)}, 0.0).buttons

def test_v2_hides_right_hand_trail():
    pipe = make_pipeline()
    pipe.set_mode("shoot_v2")
    lm = synth_hand(P_TWO, wrist=(0.7, 0.75))
    hands = pipe.process_raw([RawHand("Right", 0.9, lm, world_of(lm))], 0.0, W, H)
    assert "Right" in hands                                     # vẫn điều khiển ...
    assert pipe.hand_views(hands, {}) == []                     # ... nhưng không vẽ vệt sáng
    lm_l = synth_hand(P_TWO, wrist=(0.3, 0.75))
    hands = pipe.process_raw([RawHand("Left", 0.9, lm_l, world_of(lm_l))], 0.04, W, H)
    view = pipe.hand_views(hands, {})[0]                        # tay trái V2: chấm vẽ ở ĐỐT NGÓN = núm
    assert np.allclose(view.tip, hands["Left"].knuckle_screen)
    pipe.set_mode("slice")
    assert "Right" in [v.label for v in pipe.hand_views(hands, {})]   # chế độ khác vẫn vẽ tay phải


def test_controller_auto_picks_abs_or_rel_by_pointer_lock():
    fb = FakeBackend()
    c = Controller(fb, {**PCFG, "relative_accel": 0})
    fb.locked = False
    c.apply(Desired(cursor_abs=(0.5, 0.5), cursor_rel=(0.01, 0), cursor_auto=True), now=0.0)
    for i in range(10):
        c.tick(1 / 120, i / 120)
    assert any(e[0] == "abs" for e in fb.log) and not any(e[0] == "rel" for e in fb.log)
    fb.log.clear()
    fb.locked = True                                            # game khóa con trỏ -> tương đối
    c.apply(Desired(cursor_abs=(0.6, 0.5), cursor_rel=(0.01, 0), cursor_auto=True), now=1.0)
    for i in range(10):
        c.tick(1 / 120, 1.0 + i / 120)
    assert c.auto_relative and any(e[0] == "rel" for e in fb.log) and not any(e[0] == "abs" for e in fb.log)


def test_v2_three_zones():
    """Vòng 1: không xoay. Vòng 2: chỉ đâu nhìn đó (dịch chuyển, tay đứng yên thì dừng). Vòng 3: xoay liên tục."""
    prof = ShootV2Profile({**PCFG, "screen_aspect": 1.6, "aim_deadzone": 0.02, "aim_point_radius": 0.20, "aim_radius": 0.32, "aim_point_gain": 1.0})
    prof.update({"Right": make_hand("Right", P_TWO, POINTER, tip=(0.5, 0.5))}, 0.0)          # lưu tâm

    def at(x, y, t):
        return prof.update({"Right": make_hand("Right", P_TWO, POINTER, tip=(x, y))}, t)
    # vòng 1: lệch 0.008 bề rộng = 0.0128 chiều cao < 0.02
    d = at(0.508, 0.5, 0.033)
    assert d.cursor_rel == (0.0, 0.0) and d.cursor_vel == (0.0, 0.0)
    # vòng 2: lệch 0.1 chiều cao -> màn hình xoay đúng 0.1 - 0.02 = 0.08 (x aim_point_gain), không xoay liên tục
    d = at(0.5625, 0.5, 0.066)
    assert d.cursor_rel[0] == pytest.approx(0.08, abs=1e-6) and d.cursor_vel == (0.0, 0.0)
    d = at(0.5625, 0.5, 0.1)                                                # tay đứng yên -> dừng
    assert d.cursor_rel == pytest.approx((0.0, 0.0)) and d.cursor_vel == (0.0, 0.0)
    d = at(0.5, 0.5, 0.133)                                                 # đưa tay về -> màn hình quay về
    assert d.cursor_rel[0] == pytest.approx(-0.08, abs=1e-6)
    # vòng 3: lệch 0.26 chiều cao (> 0.20) -> xoay liên tục, giữ yên vẫn xoay đều
    vels = [at(0.6625, 0.5, 0.2 + i * 0.033).cursor_vel for i in range(5)]
    assert all(v[0] > 0 and v[1] == 0 for v in vels) and max(v[0] for v in vels) == pytest.approx(min(v[0] for v in vels))
    # xoay dọc chậm hơn ngang (aim_vertical_scale)
    up = at(0.5, 0.24, 0.5).cursor_vel[1]
    side = at(0.6625, 0.5, 0.533).cursor_vel[0]
    assert up < 0 and abs(up) == pytest.approx(side * PCFG["aim_vertical_scale"], rel=1e-6)

def test_v2_outer_zone_speed_curve():
    prof = ShootV2Profile({**PCFG, "screen_aspect": 1.6, "aim_deadzone": 0.02, "aim_point_radius": 0.20, "aim_radius": 0.32, "aim_point_gain": 1.0})
    prof.update({"Right": make_hand("Right", P_TWO, POINTER, tip=(0.5, 0.5))}, 0.0)

    def speed(dx):
        return prof.update({"Right": make_hand("Right", P_TWO, POINTER, tip=(0.5 + dx / 1.6, 0.5))}, 0.033).cursor_vel[0]
    assert speed(0.15) == 0.0                                               # còn trong vòng 2
    assert 0 < speed(0.23) < speed(0.27) < speed(0.31)                      # vòng 3: càng xa càng nhanh
    assert speed(0.4) == pytest.approx(speed(0.6))                          # ngoài vòng 3 -> tối đa

def test_v2_delta_style_still_available_with_edge_push():
    prof = ShootV2Profile({**PCFG, "aim_style": "delta"})
    rels = [prof.update({"Right": make_hand("Right", P_TWO, POINTER, tip=(0.99, 0.5))}, i * 0.033).cursor_rel[0]
            for i in range(5)]
    assert all(r > 0 for r in rels[1:])                        # kiểu cũ: tay đứng ở rìa phải -> vẫn xoay phải


def test_v2_fine_aim_damps_small_moves():
    prof = ShootV2Profile({**PCFG, "aim_style": "delta"})
    prof.update({"Right": make_hand("Right", P_TWO, POINTER, tip=(0.5, 0.5))}, 0.0)
    small = prof.update({"Right": make_hand("Right", P_TWO, POINTER, tip=(0.502, 0.5))}, 0.033).cursor_rel[0]
    assert small < 0.002 * 0.5


# ---------------- controller 120 Hz ----------------
class FakeBackend:
    def __init__(self):
        self.log = []
        self.locked = False

    def pointer_locked(self):
        return self.locked

    def screen_size(self):
        return 1920, 1080

    def move_abs(self, x, y):
        self.log.append(("abs", x, y))

    def move_rel(self, dx, dy):
        self.log.append(("rel", dx, dy))

    def button(self, b, down):
        self.log.append(("btn", b, down))

    def key(self, k, down):
        self.log.append(("key", k, down))


def test_controller_interpolates_toward_target():
    fb = FakeBackend()
    c = Controller(fb, PCFG)
    c.apply(Desired(cursor_abs=(0.0, 0.0)), now=0.0)
    c.tick(1 / 120, 0.0)
    c.apply(Desired(cursor_abs=(0.5, 0.5)), now=0.033)
    xs = []
    for i in range(1, 30):
        c.tick(1 / 120, 0.033 + i / 120)
        xs.append(fb.log[-1][1])
    assert xs[0] < 900 and abs(xs[-1] - 959) < 40
    assert all(b >= a - 1 for a, b in zip(xs, xs[1:]))


def test_controller_flushes_cursor_before_click_and_releases():
    fb = FakeBackend()
    c = Controller(fb, PCFG)
    c.apply(Desired(cursor_abs=(0.5, 0.5), buttons={"left"}, keys={"w"}), now=0.0)
    assert fb.log[:3] == [("abs", 960, 540), ("btn", "left", True), ("key", "w", True)]
    fb.log.clear()
    c.set_enabled(False)
    assert ("btn", "left", False) in fb.log and ("key", "w", False) in fb.log


def test_controller_relative_drains_smoothly():
    fb = FakeBackend()
    c = Controller(fb, {**PCFG, "relative_accel": 0})
    c.apply(Desired(cursor_rel=(0.01, 0.0)), now=0.0)
    for i in range(12):
        c.tick(1 / 120, i / 120)
    moves = [e[1] for e in fb.log if e[0] == "rel"]
    assert len(moves) > 2 and sum(moves) in (17, 18)


# ---------------- pipeline: lọc frame ----------------
def make_pipeline():
    cfg = copy.deepcopy(CFG)
    return VisionPipeline(cfg, AppState(), None, None, Controller(FakeBackend(), cfg["profiles"]), lambda: None)


def test_pipeline_accepts_wrist_at_bottom_edge_but_rejects_hand_outside():
    pipe = make_pipeline()
    lm = synth_hand(P_TWO, wrist=(0.5, 0.995))                  # cổ tay sát mép dưới -> vẫn nhận
    for i in range(5):
        hands = pipe.process_raw([RawHand("Right", 0.9, lm, world_of(lm))], i * 0.03, W, H)
    assert "Right" not in pipe._reasons and hands["Right"].pose == POINTER
    lm2 = synth_hand(P_TWO, wrist=(0.5, 0.75))
    lm2[[5, 9, 13, 17], 0] += 0.6                              # khớp gốc ngón ra ngoài khung
    pipe.process_raw([RawHand("Right", 0.9, lm2, world_of(lm2))], 0.2, W, H)
    assert "ngoài khung" in pipe._reasons["Right"]


# ---------------- hiệu chỉnh ----------------
def test_calibration_compute_touch_thresholds():
    res = Calibrator.compute([(0.5, 0.6)] * 10, [(0.1, 0.15)] * 10)
    assert res["touch_on_2d"] == pytest.approx(0.26) and res["touch_off_2d"] == pytest.approx(0.34)
    assert res["touch_on_3d"] == pytest.approx(0.51) and res["approach_2d"] == pytest.approx(0.44)
    # độ sâu nhiễu: 3D lúc chạm còn LỚN hơn lúc tách -> vẫn hiệu chỉnh được theo 2D
    res = Calibrator.compute([(0.5, 0.30)] * 10, [(0.05, 0.35)] * 10)
    assert res is not None and res["touch_on_3d"] == pytest.approx(0.5)
    assert Calibrator.compute([(0.3, 0.3)] * 10, [(0.28, 0.29)] * 10) is None


def test_touch_counts_when_2d_touches_but_3d_depth_is_noisy():
    lm = synth_hand(P_TWO, touch="index")
    m = meas(lm)
    m.touch_index_3d = 0.35                                    # độ sâu đoán sai như ảnh người dùng gửi
    assert TouchTracker(TCFG).update(m) == "index"


# ---------------- vòng tròn, One Euro, cache game ----------------
def test_circle_detected_and_line_rejected():
    det = CircleDetector(CFG["circle"])
    hit = False
    for i in range(40):
        a = 2 * math.pi * i / 36
        hit |= det.update(0.5 + 0.1 * math.cos(a), 0.5 + 0.1 * math.sin(a), i * 0.03)
    assert hit
    det2 = CircleDetector(CFG["circle"])
    assert not any(det2.update(0.2 + i * 0.01, 0.5, i * 0.03) for i in range(40))


def test_one_euro_smooths_jitter_and_follows_motion():
    f = OneEuroFilter(1.0, 8.0, 1.0)
    rng = np.random.default_rng(0)
    out = [f(np.array([0.5 + rng.normal(0, 0.005)]), i / 30)[0] for i in range(60)]
    assert np.std(out[10:]) < 0.004
    f.reset()
    for i in range(30):
        y = f(np.array([i * 0.02]), i / 30)[0]
    assert abs(y - 29 * 0.02) < 0.06


def _fake_screen(seed, color):
    rng = np.random.default_rng(seed)
    img = np.full((360, 640, 3), color, dtype=np.uint8)
    for _ in range(25):
        x, y = rng.integers(0, 600), rng.integers(0, 320)
        img[y:y + 40, x:x + 40] = rng.integers(0, 255, 3)
    return img


def test_game_cache_calls_claude_once(tmp_path):
    db = GameDB(tmp_path / "db.json", 10, 0.7)
    calls = []

    def fake_claude(bgr, title, model, max_w):
        calls.append(title)
        return {"name": "Fruit Slicer", "mode": "slice", "mouse_mode": "absolute"}

    det = GameDetector(db, CFG["game_detect"], lambda r: None, classifier=fake_claude)
    img = _fake_screen(1, (30, 120, 40))
    assert det.detect(img, "Fruit Slicer - Google Chrome").source == "claude"
    assert det.detect(img, "Fruit Slicer - Google Chrome").source == "cache-title"
    assert len(calls) == 1


def test_dhash_differs_for_different_screens():
    a, b = _fake_screen(1, (30, 120, 40)), _fake_screen(2, (200, 30, 30))
    assert hamming(dhash(a), dhash(b)) > 10


@pytest.mark.parametrize("profile_cls", [ShootV1Profile, ShootV2Profile])
def test_shooting_modes_never_drag(profile_cls):
    """Ở chế độ bắn, khép trỏ + giữa (tay phải hoặc tay trái) không bao giờ thành KÉO / CHÉM."""
    prof = profile_cls(PCFG)
    for i in range(10):
        two = make_hand("Right", P_TWO, TWO_TOGETHER, together=True)
        d = prof.update({"Right": two}, i * 0.033)
        assert KEO_CHEM not in d.gestures.values() and "left" not in d.buttons


def test_clicks_keep_working_after_many_mode_switches():
    """Người dùng báo: dùng một lúc / đổi chế độ thì không còn click được. Mô phỏng phiên dài: đổi chế độ
    nhiều lần, tay ra vào khung hình; mỗi lần quay lại CHÉM, chạm ngón cái vẫn phải ra CLICK TRÁI."""
    pipe = make_pipeline()
    lm_point = synth_hand(P_TWO, wrist=(0.6, 0.75))
    lm_touch = synth_hand(P_TWO, wrist=(0.6, 0.75), touch="index")
    t = 0.0
    for cycle in range(6):
        for mode in ("shoot_v1", "shoot_v2", "slice"):
            pipe.set_mode(mode)
            for _ in range(8):
                t += 0.033
                pipe.step(pipe.process_raw([RawHand("Right", 0.9, lm_point, world_of(lm_point))], t, W, H), t)
            t += 1.0                                            # bỏ tay ra một lúc
            pipe.step(pipe.process_raw([], t, W, H), t)
        names = []
        for _ in range(8):
            t += 0.033
            pipe.step(pipe.process_raw([RawHand("Right", 0.9, lm_point, world_of(lm_point))], t, W, H), t)
        for _ in range(4):
            t += 0.033
            d = pipe.step(pipe.process_raw([RawHand("Right", 0.9, lm_touch, world_of(lm_touch))], t, W, H), t)
            names.append(d.gestures.get("Right"))
        assert CLICK_TRAI in names, (cycle, names)


def test_v1_fire_when_thumb_closes_in_any_way():
    """BÓP CÒ: ngón cái chạm, tới gần, hoặc gập xuống đều bắn; ngón cái khép sẵn từ đầu thì không tự bắn."""
    for kw in ({"touch": "index"}, {"states": [F, T, T, F, F]}):
        prof = ShootV1Profile({**PCFG, "v1_fire_on_close": True})          # tuỳ chọn, mặc định tắt
        open_gun = make_hand("Left", P_TWO, TWO_TOGETHER, together=True)
        assert not prof.update({"Left": open_gun}, 0.0).buttons
        states = kw.pop("states", P_TWO)
        d = prof.update({"Left": make_hand("Left", states, TWO_TOGETHER, together=True, **kw)}, 0.05)
        assert d.buttons == {"left"} and d.gestures["Left"] == BOP_CO
    prof = ShootV1Profile({**PCFG, "v1_fire_on_close": True})
    closed = make_hand("Left", [F, T, T, F, F], TWO_TOGETHER, together=True)
    assert not prof.update({"Left": closed}, 0.0).buttons                 # khép sẵn -> chưa bắn


def test_v1_uses_lower_region():
    """BẮN V1: vùng hoạt động hạ thấp -> tay ở ngang tầm cánh tay (y ~ 0.9 khung camera) vẫn tới đáy màn hình."""
    pipe = make_pipeline()
    pipe.set_mode("shoot_v1")
    assert pipe.region == CFG["mapping"]["region_by_mode"]["shoot_v1"]
    lm = synth_hand(P_TWO, wrist=(0.5, 0.99))                   # đầu ngón trỏ ở y ~ 0.79
    for i in range(5):
        hands = pipe.process_raw([RawHand("Left", 0.9, lm, world_of(lm))], i * 0.033, W, H)
    tip_y = float(next(iter(hands.values())).tip[1])
    assert 0.6 < tip_y < 1.0                                    # vùng cũ (đáy 0.68) sẽ kẹt ở 1.0
    pipe.set_mode("slice")
    assert pipe.region == CFG["mapping"]["region"]


def test_v2_keeps_rotating_while_firing():
    """Đang bắn vẫn xoay được (không còn đứng im)."""
    prof = ShootV2Profile({**PCFG, "screen_aspect": 1.6, "aim_deadzone": 0.02, "aim_point_radius": 0.20, "aim_radius": 0.32, "aim_point_gain": 1.0})
    prof.update({"Right": make_hand("Right", P_TWO, POINTER, tip=(0.5, 0.5))}, 0.0)
    firing = [F, T, T, F, F]
    d = prof.update({"Right": make_hand("Right", P_TWO, POINTER, tip=(0.5, 0.5))}, 0.03)
    for i in range(10):
        d = prof.update({"Right": make_hand("Right", firing, POINTER, tip=(0.7, 0.5))}, 0.06 + 0.033 * i)
        assert "left" in d.buttons and d.cursor_vel[0] > 0                    # vòng 3: vừa bắn vừa xoay


def test_v2_equal_speed_in_all_directions():
    """Cùng quãng tay thật (pixel camera) theo mọi hướng -> cùng tốc độ xoay."""
    cfg = {**PCFG, "screen_aspect": 1.6, "control_aspect": 640 / 480, "aim_vertical_scale": 1.0,
           "aim_point_radius": 0.20, "aim_radius": 0.32}
    speeds = []
    for ang in range(0, 360, 45):
        prof = ShootV2Profile(cfg)
        prof.update({"Right": make_hand("Right", P_TWO, POINTER, tip=(0.5, 0.5))}, 0.0)
        r = 0.26                                                            # trong vòng 3
        dx = math.sin(math.radians(ang)) * r / cfg["control_aspect"]           # đổi ra toạ độ chuẩn hoá ngang
        dy = -math.cos(math.radians(ang)) * r
        v = prof.update({"Right": make_hand("Right", P_TWO, POINTER, tip=(0.5 + dx, 0.5 + dy))}, 0.033).cursor_vel
        speeds.append(math.hypot(*v))
    assert max(speeds) == pytest.approx(min(speeds), rel=1e-6) and min(speeds) > 0

def test_v2_fist_recenters_joystick_and_aim():
    """Nắm tay = ĐẶT TÂM theo ĐỐT NGÓN: tâm đi theo tay lúc nắm và GIỮ NGUYÊN khi mở ra (nút không nhảy).
    Vòng xoay của tay phải luôn ghim ở giữa màn hình."""
    prof = ShootV2Profile({**PCFG, "screen_aspect": 1.6, "aim_deadzone": 0.02, "aim_point_radius": 0.20, "aim_radius": 0.32, "aim_point_gain": 1.0})
    prof.update({"Left": make_hand("Left", P_TWO, TWO_TOGETHER, knuckle=(0.3, 0.6), together=True)}, 0.0)
    d = prof.update({"Left": make_hand("Left", P_FIST, FIST, knuckle=(0.5, 0.7))}, 0.1)
    assert d.joystick["setting"] and d.joystick["center"] == pytest.approx((0.5, 0.7)) and not d.keys
    d = prof.update({"Left": make_hand("Left", P_FIST, FIST, knuckle=(0.52, 0.71))}, 0.13)
    assert d.joystick["center"] == pytest.approx((0.52, 0.71))
    d = prof.update({"Left": make_hand("Left", P_TWO, TWO_TOGETHER, knuckle=(0.521, 0.711), tip=(0.53, 0.55),
                                       together=True)}, 0.2)
    assert d.joystick["center"] == pytest.approx((0.52, 0.71)) and d.keys == {"w"}
    d = prof.update({"Right": make_hand("Right", P_FIST, FIST, knuckle=(0.8, 0.8))}, 0.1)
    assert d.gestures["Right"] == "ĐẶT TÂM" and d.aim_stick["setting"] and d.aim_stick["center"] == (0.5, 0.5)
    assert d.cursor_vel is None and d.cursor_rel is None
    d = prof.update({"Right": make_hand("Right", P_TWO, POINTER, knuckle=(0.801, 0.8), tip=(0.8, 0.65))}, 0.2)
    assert d.cursor_rel == (0.0, 0.0) and d.cursor_vel == (0.0, 0.0)            # mở tay tại chỗ -> chưa xoay
    d = prof.update({"Right": make_hand("Right", P_TWO, POINTER, knuckle=(0.86, 0.8), tip=(0.86, 0.65))}, 0.25)
    assert d.cursor_rel[0] > 0                                                    # lệch phải -> xoay phải


def test_v2_lost_hand_keeps_last_direction():
    """Mất tay khi đang xoay ở vòng ngoài / đang đi -> tiếp tục hướng cũ, tối đa lost_keep_sec giây."""
    prof = ShootV2Profile({**PCFG, "screen_aspect": 1.6, "aim_deadzone": 0.02, "aim_point_radius": 0.20, "aim_radius": 0.32, "aim_point_gain": 1.0})
    prof.update({"Left": make_hand("Left", P_TWO, TWO_TOGETHER, knuckle=(0.3, 0.6), together=True),
                 "Right": make_hand("Right", P_TWO, POINTER, tip=(0.5, 0.5))}, 0.0)
    d = prof.update({"Left": make_hand("Left", P_TWO, TWO_TOGETHER, knuckle=(0.35, 0.6), together=True),
                     "Right": make_hand("Right", P_TWO, POINTER, tip=(0.7, 0.5))}, 0.033)
    keys, vel = set(d.keys), d.cursor_vel
    assert keys == {"d"} and vel[0] > 0
    d = prof.update({}, 1.0)                                                  # mất cả 2 tay
    assert d.keys == keys and d.cursor_vel == vel
    assert d.gestures["Left"].startswith("ĐI TIẾP") and d.gestures["Right"].startswith("XOAY TIẾP")
    d = prof.update({}, 3.5)                                                  # quá lost_keep_sec -> dừng
    assert not d.keys and d.cursor_vel is None
    # đang ở vòng 2 (không xoay liên tục) mà mất tay -> không xoay tiếp
    prof2 = ShootV2Profile({**PCFG, "screen_aspect": 1.6, "aim_deadzone": 0.02, "aim_point_radius": 0.20, "aim_radius": 0.32, "aim_point_gain": 1.0})
    prof2.update({"Right": make_hand("Right", P_TWO, POINTER, tip=(0.5, 0.5))}, 0.0)
    prof2.update({"Right": make_hand("Right", P_TWO, POINTER, tip=(0.55, 0.5))}, 0.033)
    assert prof2.update({}, 0.5).cursor_vel is None

def test_controller_velocity_turns_smoothly_at_120hz():
    """Vận tốc xoay nhận 30 lần/giây nhưng chuột được gửi đều mỗi tick 120 Hz (không dồn cục theo nhịp camera)."""
    fb = FakeBackend()
    fb.locked = True
    c = Controller(fb, {**PCFG, "relative_sensitivity": 1000, "rotate_smooth_ms": 1})
    t = 0.0
    moves = []
    for frame in range(30):                                   # 1 giây, camera 30 FPS
        c.apply(Desired(cursor_vel=(0.5, 0.0), cursor_abs=(0.5, 0.5), cursor_auto=True), now=t)
        for _ in range(4):                                    # 4 tick 120 Hz giữa 2 frame
            t += 1 / 120
            before = len(fb.log)
            c.tick(1 / 120, t)
            moves.append(sum(e[1] for e in fb.log[before:] if e[0] == "rel"))
    assert abs(sum(moves) - 500) <= 10                         # 0.5 màn hình/giây x 1000 px x 1 s
    steady = moves[8:]
    assert max(steady) - min(steady) <= 2                      # đều đặn từng tick, không giật theo nhịp 30 Hz
    c.apply(Desired(cursor_abs=(0.5, 0.5), cursor_auto=True), now=t)   # tay về tâm -> vận tốc 0
    for _ in range(20):
        t += 1 / 120
        c.tick(1 / 120, t)
    before = len(fb.log)
    c.tick(1 / 120, t + 1 / 120)
    assert not [e for e in fb.log[before:] if e[0] == "rel"]


def test_v2_left_moves_when_two_fingers_slightly_apart():
    """Trước đây phải khép SÁT 2 ngón mới đi -> hơi hở là đứng im. Giờ khép hoặc hơi tách đều đi."""
    prof = ShootV2Profile(PCFG)
    d = prof.update({"Left": make_hand("Left", P_TWO, POINTER, knuckle=(0.3, 0.6), together=False)}, 0.0)
    assert d.gestures["Left"] == DI_CHUYEN and d.keys == {"w"}


def _gun_world(ip_angle=175.0, shift=(0.0, 0.0, 0.0), rot_deg=0.0):
    """Tay súng 3D; đầu ngón cái gập ở khớp IP tới góc ip_angle; có thể dời / xoay cả bàn tay."""
    w = world_of(synth_hand(P_TWO, together=True)).astype(float)
    a, b = w[2], w[3]
    seg = np.linalg.norm(w[4] - w[3])
    d = (b - a) / np.linalg.norm(b - a)
    n = np.array([0.0, 0.0, 1.0])
    bend = math.radians(180.0 - ip_angle)
    perp = np.cross(n, d)
    perp /= np.linalg.norm(perp)
    w[4] = b + seg * (math.cos(bend) * d + math.sin(bend) * perp)
    r = math.radians(rot_deg)                                   # xoay cả bàn tay quanh trục dọc
    rot = np.array([[math.cos(r), 0, math.sin(r)], [0, 1, 0], [-math.sin(r), 0, math.cos(r)]])
    return w @ rot.T + np.array(shift)


def _gun(world, tip=(0.4, 0.4)):
    h = make_hand("Left", P_TWO, TWO_TOGETHER, tip=tip, together=True)
    h.world = world
    return h


def test_v1_fire_when_thumb_tip_bends_at_ip_joint():
    prof = ShootV1Profile(PCFG)
    t = 0.0
    for _ in range(12):                                          # giơ súng, ngón cái thẳng
        d = prof.update({"Left": _gun(_gun_world(175))}, t)
        t += 0.033
        assert "left" not in d.buttons
    for ang in (150, 115, 100):                                  # gập đầu ngón cái xuống
        d = prof.update({"Left": _gun(_gun_world(ang))}, t)
        t += 0.033
    assert "left" in d.buttons and d.gestures["Left"] == BOP_CO
    for _ in range(3):                                           # còn gập -> bắn liên tục
        d = prof.update({"Left": _gun(_gun_world(100))}, t)
        t += 0.033
        assert "left" in d.buttons
    for _ in range(3):                                           # duỗi ra -> ngừng
        d = prof.update({"Left": _gun(_gun_world(175))}, t)
        t += 0.033
    assert "left" not in d.buttons


def test_v1_no_fire_when_whole_hand_moves_or_rotates():
    prof = ShootV1Profile(PCFG)
    t = 0.0
    for i in range(40):
        w = _gun_world(170, shift=(0.003 * i, -0.002 * i, 0.0), rot_deg=1.5 * i)
        d = prof.update({"Left": _gun(w)}, t)
        t += 0.033
        assert "left" not in d.buttons


def test_v1_no_fire_during_warmup():
    prof = ShootV1Profile(PCFG)
    prof.update({"Left": _gun(_gun_world(175))}, 0.0)
    for i, ang in enumerate((140, 110, 100)):                    # gập ngay lúc vừa giơ súng
        d = prof.update({"Left": _gun(_gun_world(ang))}, 0.033 * (i + 1))
        assert "left" not in d.buttons


def test_v1_thumb_folded_fires_by_default_but_not_mere_approach():
    """Mặc định: ngón cái GẬP vào lòng bàn tay / CHẠM ngón trỏ là bắn; chỉ "tới gần" thì không (tránh bắn nhầm)."""
    prof = ShootV1Profile(PCFG)
    prof.update({"Left": make_hand("Left", P_TWO, TWO_TOGETHER, together=True)}, 0.0)
    d = prof.update({"Left": make_hand("Left", P_TWO, TWO_TOGETHER, together=True, approach=True)}, 0.5)
    assert "left" not in d.buttons
    d = prof.update({"Left": make_hand("Left", [F, T, T, F, F], TWO_TOGETHER, together=True)}, 0.6)
    assert "left" in d.buttons and d.gestures["Left"] == BOP_CO
    d = prof.update({"Left": make_hand("Left", P_TWO, TWO_TOGETHER, together=True)}, 0.7)
    assert "left" not in d.buttons                                # mở ngón cái ra -> ngừng bắn
def test_v2_zone2_hand_position_becomes_new_center_after_settling():
    """Vòng 2: xoay tới chỗ tay chỉ, tay đứng yên -> chỗ đặt tay thành TÂM MỚI (vạch xanh biến mất)."""
    cfg = {**PCFG, "screen_aspect": 1.6, "control_aspect": 1.6, "aim_deadzone": 0.02, "aim_point_radius": 0.20,
           "aim_radius": 0.32, "aim_settle_ms": 250, "aim_settle_speed": 0.15}
    prof = ShootV2Profile(cfg)

    def at(x, t):
        return prof.update({"Right": make_hand("Right", P_TWO, POINTER, tip=(x, 0.5))}, t)
    at(0.5, 0.0)
    d = at(0.5625, 0.033)                                       # lệch 0.1 -> xoay 0.08
    assert d.cursor_rel[0] == pytest.approx(0.08, abs=1e-6) and d.aim_stick["vec"][0] > 0
    t = 0.066
    for _ in range(12):                                         # giữ yên ~0.4 s
        d = at(0.5625, t)
        t += 0.033
    assert d.aim_stick["vec"] == pytest.approx((0.0, 0.0))     # vạch biến mất: tâm mới = chỗ đặt tay
    assert prof.aim_anchor[0] == pytest.approx(0.5625)
    d = at(0.5625 + 0.05 / 1.6 * 1.6 / 1.6, t)                  # dịch tiếp 0.05 từ tâm mới -> xoay 0.05 - 0.02
    assert d.cursor_rel[0] == pytest.approx(0.03, abs=1e-6)


def test_v2_zone3_does_not_recenter():
    cfg = {**PCFG, "screen_aspect": 1.6, "control_aspect": 1.6, "aim_point_radius": 0.20, "aim_radius": 0.32}
    prof = ShootV2Profile(cfg)
    prof.update({"Right": make_hand("Right", P_TWO, POINTER, tip=(0.5, 0.5))}, 0.0)
    for i in range(20):                                         # giữ yên ở vòng 3
        d = prof.update({"Right": make_hand("Right", P_TWO, POINTER, tip=(0.6625, 0.5))}, 0.033 * (i + 1))
    assert d.cursor_vel[0] > 0 and prof.aim_anchor[0] == pytest.approx(0.5)
