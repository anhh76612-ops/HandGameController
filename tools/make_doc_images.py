"""Sinh ảnh minh hoạ TRƯỚC / SAU cho từng bước xử lý ảnh trong docs/01_cac_buoc_xu_ly_anh.md.

Dùng chính các hàm của dự án (preprocess, MediaPipe, finger_state, smoothing, controller, gestures, game_db)
chạy trên video thật của người dùng, nên ảnh minh hoạ đúng với những gì chương trình làm.

  .venv\\Scripts\\python.exe tools\\make_doc_images.py video.mp4 [--screenshot anh_game.png] [--out docs/images]

Ảnh game cho bước 12 (dHash + histogram): truyền --screenshot; nếu không có thì dùng một frame của video.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve  # noqa: E402
from src.controller import Controller  # noqa: E402
from src.finger_state import (  # noqa: E402
    INDEX, MIDDLE, FingerTracker, finger_gap_ratio, finger_line, joint_angle, measure, seg_dist, to_metric,
)
from src.game_db import dhash, hsv_hist  # noqa: E402
from src.gestures import CircleDetector  # noqa: E402
from src.hand_tracker import HandTracker  # noqa: E402
from src.preprocess import Preprocessor  # noqa: E402
from src.profiles.base import Desired  # noqa: E402
from src.smoothing import OneEuroFilter  # noqa: E402
from src.visualize import draw_hand  # noqa: E402

W, H = 640, 360
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 110})
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)
    print("  ->", out / name)


def hand_box(lm, w, h, pad=0.06):
    x0, y0 = lm[:, 0].min() - pad, lm[:, 1].min() - pad
    x1, y1 = lm[:, 0].max() + pad, lm[:, 1].max() + pad
    return max(0, int(x0 * w)), max(0, int(y0 * h)), min(w, int(x1 * w)), min(h, int(y1 * h))


# ---------------------------------------------------------------- quét video, chọn frame
def scan(video: str, cfg: dict):
    tr = HandTracker(cfg["tracker"], resolve(cfg["tracker"]["model_path"]))
    pre = Preprocessor({**cfg["preprocess"], "clahe_auto": False})
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (W, H))
        p = pre(f)
        hs = tr.detect(p.rgb)
        rec = {"i": i, "t": i / fps, "raw": f, "flip": p.bgr, "hand": None}
        if hs:
            hnd = max(hs, key=lambda x: x.score)
            m = measure(hnd.world, to_metric(hnd.landmarks, W, H))
            rec.update(hand=hnd, m=m)
        frames.append(rec)
        i += 1
    tr.close()
    return frames, fps


def pick(frames):
    good = [r for r in frames if r["hand"] is not None]
    two = [r for r in good if r["m"].angles[INDEX] > 150 and r["m"].angles[MIDDLE] > 150
           and 0.1 < r["hand"].landmarks[0, 1] < 0.97]
    main = max(two or good, key=lambda r: r["hand"].score + 0.002 * r["m"].angles[INDEX])
    for r in two:
        r["gap"] = finger_gap_ratio(r["flip"], r["hand"].landmarks)
    measured = [r for r in two if r["gap"] is not None]
    spread_c = [r for r in measured if r["m"].together > 0.45] or measured or good
    spread = max(spread_c, key=lambda r: (r.get("gap") or 0.0))
    together_c = [r for r in measured if r["m"].together < 0.30] or measured or good
    together = min(together_c, key=lambda r: ((r.get("gap") or 0.0), r["m"].together))
    touch = min(good, key=lambda r: r["m"].touch_index_2d)
    # đoạn liên tục dài nhất có thấy tay (cho biểu đồ làm mượt)
    best, cur = [], []
    for r in frames:
        if r["hand"] is not None:
            cur.append(r)
            if len(cur) > len(best):
                best = list(cur)
        else:
            cur = []
    return main, spread, together, touch, best[:150]


# ---------------------------------------------------------------- các hình
def fig_flip(r, out):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].imshow(rgb(r["raw"]))
    ax[0].set_title("TRƯỚC: ảnh gốc từ webcam (tay trái nằm bên PHẢI ảnh)")
    ax[1].imshow(rgb(r["flip"]))
    ax[1].set_title("SAU: cv2.flip(frame, 1) - lật gương (tay trái về bên TRÁI)")
    for a in ax:
        a.axis("off")
    save(fig, out, "b02_lat_guong.png")


def fig_lab(r, out):
    lab = cv2.cvtColor(r["flip"], cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.2), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    ax[0].imshow(rgb(r["flip"]))
    ax[0].set_title("TRƯỚC: ảnh màu BGR")
    ax[1].imshow(L, cmap="gray", vmin=0, vmax=255)
    ax[1].set_title("SAU: kênh L của LAB (độ sáng, tách khỏi màu)")
    ax[2].hist(L.ravel(), bins=64, range=(0, 255), color="#888")
    ax[2].axvline(L.mean(), color="r", label=f"trung bình = {L.mean():.0f}")
    ax[2].axvline(90, color="b", ls="--", label="ngưỡng bật CLAHE = 90")
    ax[2].set_title("Histogram kênh L -> đo độ sáng")
    ax[2].legend(fontsize=8)
    for a in ax[:2]:
        a.axis("off")
    save(fig, out, "b03_kenh_L.png")


def fig_clahe(r, out):
    dark = cv2.convertScaleAbs(r["flip"], alpha=0.4, beta=0)          # giả lập phòng tối
    pre = Preprocessor({"flip": False, "clahe_auto": True, "clahe_brightness_threshold": 256,
                        "clahe_clip_limit": 2.0, "clahe_grid": 8})
    res = pre(dark)
    Ld = cv2.cvtColor(dark, cv2.COLOR_BGR2LAB)[:, :, 0]
    Lc = cv2.cvtColor(res.bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.4))
    ax[0, 0].imshow(rgb(dark))
    ax[0, 0].set_title(f"TRƯỚC: phòng tối (độ sáng L = {Ld.mean():.0f})")
    ax[0, 1].imshow(rgb(res.bgr))
    ax[0, 1].set_title(f"SAU: CLAHE trên kênh L (L = {Lc.mean():.0f}), màu da giữ nguyên")
    ax[1, 0].hist(Ld.ravel(), bins=64, range=(0, 255), color="#555")
    ax[1, 0].set_title("Histogram L trước: dồn về vùng tối")
    ax[1, 1].hist(Lc.ravel(), bins=64, range=(0, 255), color="#2a7")
    ax[1, 1].set_title("Histogram L sau: trải rộng hơn (tăng tương phản cục bộ)")
    for a in (ax[0, 0], ax[0, 1]):
        a.axis("off")
    save(fig, out, "b04_clahe.png")


def fig_rgb(r, out):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].imshow(r["flip"])                                           # cố tình hiển thị mảng BGR như RGB
    ax[0].set_title("Nếu đưa mảng BGR cho chương trình đọc RGB: da bị xanh")
    ax[1].imshow(rgb(r["flip"]))
    ax[1].set_title("SAU cv2.cvtColor(BGR2RGB): đúng màu cho MediaPipe")
    for a in ax:
        a.axis("off")
    save(fig, out, "b05_bgr_rgb.png")


def fig_landmarks(r, out):
    hnd = r["hand"]
    img = r["flip"].copy()
    pts = [(int(p[0] * W), int(p[1] * H)) for p in hnd.landmarks]
    for a, b in BONES:
        cv2.line(img, pts[a], pts[b], (0, 255, 200), 2, cv2.LINE_AA)
    for k, pt in enumerate(pts):
        cv2.circle(img, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(img, str(k), (pt[0] + 3, pt[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 2)
        cv2.putText(img, str(k), (pt[0] + 3, pt[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 0), 1)
    x0, y0, x1, y1 = hand_box(hnd.landmarks, W, H, 0.08)
    fig = plt.figure(figsize=(13, 4.2))
    a0 = fig.add_subplot(1, 3, 1)
    a0.imshow(rgb(r["flip"]))
    a0.set_title("TRƯỚC: ảnh RGB đưa vào mô hình")
    a0.axis("off")
    a1 = fig.add_subplot(1, 3, 2)
    a1.imshow(rgb(img[y0:y1, x0:x1]))
    a1.set_title(f"SAU: 21 điểm mốc ảnh (x, y)\nnhãn MediaPipe gốc: {hnd.label} ({hnd.score:.2f}), cử chỉ mô hình: {hnd.model_gesture}")
    a1.axis("off")
    a2 = fig.add_subplot(1, 3, 3, projection="3d")
    w3 = hnd.world * 100
    for a, b in BONES:
        a2.plot([w3[a, 0], w3[b, 0]], [w3[a, 2], w3[b, 2]], [-w3[a, 1], -w3[b, 1]], color="#0a8")
    a2.scatter(w3[:, 0], w3[:, 2], -w3[:, 1], s=10, c="k")
    a2.set_title("SAU: 21 điểm mốc 3D thật (cm)")
    a2.set_xlabel("x")
    a2.set_ylabel("z (sâu)")
    a2.set_zlabel("y")
    save(fig, out, "b06_diem_moc.png")


def fig_angles(r, out):
    hnd, m = r["hand"], r["m"]
    states = FingerTracker(load_config()["fingers"]).update(m)
    k = 3                                                              # phóng to để chữ không chồng nhau
    big = cv2.resize(r["flip"], (W * k, H * k), interpolation=cv2.INTER_CUBIC)
    draw_hand(big, hnd.landmarks, states, m, "Left", "", source="", reason="")
    x0, y0, x1, y1 = hand_box(hnd.landmarks, W * k, H * k, 0.06)
    p = to_metric(hnd.landmarks, W, H)[:, :2]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
    ax[0].imshow(rgb(big[y0:y1, x0:x1]))
    ax[0].set_title("SAU: số = góc khớp PIP (độ); nét màu = DUỖI, nét đỏ = CỤP")
    ax[0].axis("off")
    mcp, pip, tip = p[5], p[6], p[8]
    ax[1].plot(*zip(pip, mcp), "o-", color="#0a8", lw=3, label="PIP -> MCP (gốc ngón)")
    ax[1].plot(*zip(pip, tip), "o-", color="#e70", lw=3, label="PIP -> TIP (đầu ngón)")
    for name, q in (("MCP 5", mcp), ("PIP 6", pip), ("TIP 8", tip)):
        ax[1].annotate(name, q, xytext=(4, -10), textcoords="offset points")
    ang = joint_angle(hnd.world[5], hnd.world[6], hnd.world[8])
    pad = 25
    xs, ys = [mcp[0], pip[0], tip[0]], [mcp[1], pip[1], tip[1]]
    ax[1].set_xlim(min(xs) - pad, max(xs) + pad)
    ax[1].set_ylim(max(ys) + pad, min(ys) - pad)
    ax[1].set_aspect("equal", adjustable="datalim")
    ax[1].set_title(f"Ngón trỏ: θ = góc giữa 2 vector tại PIP = {ang:.0f}° (tính trên 3D)\n"
                    "θ > 140° -> DUỖI,  θ < 118° -> CỤP")
    ax[1].legend(loc="lower left", fontsize=8)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    save(fig, out, "b07_goc_khop.png")


def touch_panel(ax, r, title):
    hnd = r["hand"]
    img2 = to_metric(hnd.landmarks, W, H)[:, :2]
    palm2 = float(np.linalg.norm(img2[9] - img2[0]))
    line = finger_line(img2, 5)
    best = None
    for a, b in zip(line, line[1:]):
        ab = b - a
        t = float(np.clip(np.dot(img2[4] - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1))
        dd = seg_dist(img2[4], a, b)
        if best is None or dd < best[0]:
            best = (dd, a + t * ab)
    k = 3
    img = cv2.resize(r["flip"], (W * k, H * k), interpolation=cv2.INTER_CUBIC)
    for a, b in zip(line, line[1:]):
        cv2.line(img, tuple(int(v * k) for v in a), tuple(int(v * k) for v in b), (0, 255, 0), 6, cv2.LINE_AA)
    tp = tuple(int(v * k) for v in img2[4])
    q = tuple(int(v * k) for v in best[1])
    cv2.line(img, tp, q, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.circle(img, tp, 12, (0, 200, 255), -1, cv2.LINE_AA)
    x0, y0, x1, y1 = hand_box(hnd.landmarks, W * k, H * k, 0.06)
    dist = best[0] / palm2
    ax.imshow(rgb(img[y0:y1, x0:x1]))
    ax.set_title(f"{title}\nkhoảng cách = {dist:.2f} lòng bàn tay -> {'CHẠM' if dist < 0.15 else 'chưa chạm'}")
    ax.axis("off")


def fig_touch(r, r_touch, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    touch_panel(ax[0], r, "Ngón cái tách ra")
    touch_panel(ax[1], r_touch, "Ngón cái chạm đường ngón trỏ")
    fig.suptitle("Xanh lá: đường ngón trỏ (khớp 5 -> 8). Đỏ: khoảng cách ngắn nhất từ đầu ngón cái (cam) "
                 "tới đường đó. Chạm khi < 0.15", y=1.02)
    save(fig, out, "b07b_cham_ngon.png")


def gap_panel(ax, r, title):
    hnd = r["hand"]
    bgr, lm = r["flip"], hnd.landmarks
    ratio = finger_gap_ratio(bgr, lm)
    pts = np.stack([lm[:, 0] * W, lm[:, 1] * H], axis=1)
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    samples = []
    for i in (6, 7, 10, 11):
        x, y = int(round(pts[i, 0])), int(round(pts[i, 1]))
        samples.append(ycc[max(0, y - 2):y + 3, max(0, x - 2):x + 3, 1:].reshape(-1, 2))
    ref = np.concatenate(samples).astype(np.float32)
    mean, tol = ref.mean(axis=0), 2.5 * ref.std(axis=0) + 6
    img = bgr.copy()
    for a, b in ((pts[7], pts[11]), ((pts[7] + pts[8]) / 2, (pts[11] + pts[12]) / 2), (pts[8], pts[12])):
        for t in np.linspace(0.3, 0.7, 7):
            x, y = (a + t * (b - a)).round().astype(int)
            if 0 <= x < W and 0 <= y < H:
                bad = np.any(np.abs(ycc[y, x, 1:].astype(np.float32) - mean) > tol)
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255) if bad else (0, 255, 0), -1)
    xs, ys = pts[[5, 6, 7, 8, 9, 10, 11, 12], 0], pts[[5, 6, 7, 8, 9, 10, 11, 12], 1]
    x0, x1 = int(max(0, xs.min() - 25)), int(min(W, xs.max() + 25))
    y0, y1 = int(max(0, ys.min() - 25)), int(min(H, ys.max() + 25))
    ax.imshow(rgb(cv2.resize(img[y0:y1, x0:x1], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)))
    rtxt = "không đo được" if ratio is None else f"{ratio:.2f}"
    ax.set_title(f"{title}\nkhoảng cách cả ngón = {r['m'].together:.2f}, tỉ lệ pixel không giống da = {rtxt}")
    ax.axis("off")


def fig_gap(spread, together, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    gap_panel(ax[0], spread, "Trỏ + giữa TÁCH: nhiều chấm đỏ (lộ nền)")
    gap_panel(ax[1], together, "Trỏ + giữa KHÉP: chấm xanh (toàn da)")
    fig.suptitle("Kiểm tra khe hở giữa 2 ngón trên ảnh (YCrCb): xanh = giống màu da, đỏ = nền", y=1.02)
    save(fig, out, "b08_khe_ho.png")


def fig_mapping(r, cfg, out):
    region = cfg["mapping"]["region"]
    img = r["flip"].copy()
    x0, y0, x1, y1 = region
    cv2.rectangle(img, (int(x0 * W), int(y0 * H)), (int(x1 * W), int(y1 * H)), (0, 255, 255), 2)
    tip = r["hand"].landmarks[8, :2]
    cv2.circle(img, (int(tip[0] * W), int(tip[1] * H)), 6, (0, 0, 255), -1)
    u = np.clip((tip[0] - x0) / (x1 - x0), 0, 1)
    v = np.clip((tip[1] - y0) / (y1 - y0), 0, 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].imshow(rgb(img))
    ax[0].set_title("TRƯỚC: đầu ngón trỏ trong VÙNG HOẠT ĐỘNG (khung vàng) của camera")
    ax[0].axis("off")
    ax[1].add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=2))
    ax[1].plot([u], [v], "ro", ms=10)
    ax[1].set_xlim(-0.05, 1.05)
    ax[1].set_ylim(1.05, -0.05)
    ax[1].set_aspect(10 / 16)
    ax[1].set_title(f"SAU: toạ độ màn hình u={u:.2f}, v={v:.2f}\nu=(x-x0)/(x1-x0), v=(y-y0)/(y1-y0)")
    save(fig, out, "b09_anh_xa.png")


def fig_one_euro(seq, cfg, out):
    sm = cfg["smoothing"]
    f = OneEuroFilter(sm["min_cutoff"], sm["beta"], sm["d_cutoff"])
    t = np.array([r["t"] for r in seq])
    raw = np.array([r["hand"].landmarks[8, 0] for r in seq])
    filt = np.array([f(np.array([x]), tt)[0] for x, tt in zip(raw, t)])
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
    ax[0].plot(t - t[0], raw, color="#bbb", lw=1, label="thô (MediaPipe)")
    ax[0].plot(t - t[0], filt, color="#d40", lw=2, label="sau One Euro")
    ax[0].set_xlabel("giây")
    ax[0].set_ylabel("x đầu ngón trỏ (chuẩn hoá)")
    ax[0].set_title("Toàn đoạn: lọc mạnh khi chậm, bám sát khi nhanh")
    ax[0].legend()
    k = int(np.argmin(np.abs(np.diff(raw, prepend=raw[0])))) if len(raw) > 40 else 0
    k = max(0, min(len(raw) - 30, k - 15))
    ax[1].plot(t[k:k + 30] - t[0], raw[k:k + 30], "o-", color="#bbb", ms=3, label="thô")
    ax[1].plot(t[k:k + 30] - t[0], filt[k:k + 30], color="#d40", lw=2, label="One Euro")
    ax[1].set_title(f"Phóng to 1 giây: rung giảm (độ lệch chuẩn của sai phân "
                    f"{np.std(np.diff(raw[k:k + 30])):.4f} -> {np.std(np.diff(filt[k:k + 30])):.4f})")
    ax[1].set_xlabel("giây")
    ax[1].legend()
    save(fig, out, "b09b_one_euro.png")
    return t, filt


class _Log:
    def __init__(self):
        self.xs = []

    def screen_size(self):
        return 1920, 1080

    def move_abs(self, x, y):
        self.xs.append(x)

    def move_rel(self, *a):
        pass

    def button(self, *a):
        pass

    def key(self, *a):
        pass


def fig_120hz(t, filt, cfg, out):
    log = _Log()
    c = Controller(log, cfg["profiles"])
    now, ticks_t, ticks_x = 0.0, [], []
    tt = t - t[0]
    for x, ft in zip(filt[:45], tt[:45]):
        c.apply(Desired(cursor_abs=(float(x), 0.5)), now=ft)
        while now < ft + 1 / 30:
            now += 1 / 120
            c.tick(1 / 120, now)
            ticks_t.append(now)
            ticks_x.append(log.xs[-1] if log.xs else np.nan)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.step(tt[:45], filt[:45] * 1919, where="post", color="#999", label="nhảy thẳng theo frame camera (30 lần/giây)")
    ax.plot(ticks_t, ticks_x, color="#06c", lw=2, label="con trỏ thật: nội suy 120 lần/giây")
    ax.set_xlabel("giây")
    ax.set_ylabel("x con trỏ (pixel)")
    ax.set_title("Luồng 120 Hz: con trỏ trôi mượt giữa các frame thay vì nhảy nấc")
    ax.legend()
    save(fig, out, "b09c_120hz.png")


def fig_circle(cfg, out):
    rng = np.random.default_rng(2)
    tt = np.arange(45) * 0.033
    ang = np.linspace(0.3, 0.3 + 2 * math.pi * 0.95, 45)
    xy = np.stack([0.5 + 0.1 * np.cos(ang), 0.5 + 0.09 * np.sin(ang)], axis=1) + rng.normal(0, 0.004, (45, 2))
    det = CircleDetector(cfg["circle"])
    hit = [det.update(float(x), float(y), float(t0)) for (x, y), t0 in zip(xy, tt)]
    (cx, cy), r = cv2.minEnclosingCircle(xy.astype(np.float32))
    c = xy.mean(axis=0)
    d = np.linalg.norm(xy - c, axis=1)
    sweep = np.degrees(np.unwrap(np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0])))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(xy[:, 0], xy[:, 1], "o-", ms=3, color="#06c", label="quỹ đạo đầu ngón trỏ")
    ax[0].add_patch(plt.Circle((cx, cy), r, fill=False, ls="--", color="#d40", label="minEnclosingCircle"))
    ax[0].plot(*c, "k+", ms=12, label="trọng tâm")
    ax[0].set_aspect("equal")
    ax[0].invert_yaxis()
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"Bán kính đều: độ lệch chuẩn / trung bình = {d.std() / d.mean():.2f} (< 0.35)")
    ax[1].plot(tt, sweep - sweep[0], color="#06c")
    ax[1].axhline(300, color="r", ls="--", label="ngưỡng 300°")
    ax[1].set_xlabel("giây")
    ax[1].set_ylabel("góc quét cộng dồn (độ)")
    ax[1].set_title(f"np.unwrap(atan2) -> quét {abs(sweep[-1] - sweep[0]):.0f}°; nhận vòng tròn: {any(hit)}")
    ax[1].legend()
    save(fig, out, "b11_vong_tron.png")


def fig_game(shot, out):
    gray = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).astype(int)
    hist = hsv_hist(shot)
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), gridspec_kw={"width_ratios": [1.6, 0.9, 0.8, 1.2]})
    ax[0].imshow(rgb(shot))
    ax[0].set_title("TRƯỚC: ảnh chụp màn hình game")
    ax[1].imshow(small, cmap="gray", interpolation="nearest")
    ax[1].set_title("Xám + thu nhỏ 9x8 (INTER_AREA)")
    ax[2].imshow(bits, cmap="gray", interpolation="nearest")
    ax[2].set_title(f"dHash 64 bit\n0x{dhash(shot):016x}")
    from matplotlib.colors import LogNorm
    im = ax[3].imshow(hist + 1e-5, aspect="auto", cmap="magma", origin="lower", norm=LogNorm(vmin=1e-5, vmax=hist.max()))
    ax[3].set_xlabel("S (32 ngăn)")
    ax[3].set_ylabel("H (30 ngăn)")
    ax[3].set_title("Histogram HSV 2 chiều (thang log)\n= 'bảng màu' của game")
    fig.colorbar(im, ax=ax[3], fraction=0.05)
    for a in ax[:3]:
        a.axis("off")
    save(fig, out, "b12_dhash_hist.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--screenshot", help="ảnh chụp màn hình game cho bước 12")
    ap.add_argument("--out", default=str(ROOT / "docs" / "images"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    print("Đang quét video ...")
    frames, _fps = scan(args.video, cfg)
    main_f, spread, together, touch_f, seq = pick(frames)
    print(f"frame chính: {main_f['t']:.1f}s, tách: {spread['t']:.1f}s, khép: {together['t']:.1f}s, "
          f"đoạn liên tục: {len(seq)} frame")
    cv2.imwrite(str(out / "b01_anh_goc.png"), main_f["raw"])
    fig_flip(main_f, out)
    fig_lab(main_f, out)
    fig_clahe(main_f, out)
    fig_rgb(main_f, out)
    fig_landmarks(main_f, out)
    fig_angles(main_f, out)
    fig_touch(main_f, touch_f, out)
    fig_gap(spread, together, out)
    fig_mapping(main_f, cfg, out)
    t, filt = fig_one_euro(seq, cfg, out)
    fig_120hz(t, filt, cfg, out)
    fig_circle(cfg, out)
    shot = cv2.imread(args.screenshot) if args.screenshot else main_f["raw"]
    fig_game(shot, out)
    print("Xong.")


if __name__ == "__main__":
    main()
