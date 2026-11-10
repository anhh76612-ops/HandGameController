# HandGameController

Ứng dụng điều khiển game trên trình duyệt bằng cử chỉ tay qua webcam. Đây là bài tập môn Xử lý ảnh số.

- Một lớp phủ trong suốt phủ toàn màn hình vẽ **vệt neon** theo ngón tay. Song song với đó, ứng dụng gửi **chuột/phím ảo** vào game.
- Có 3 chế độ: **CHÉM** (game chém hoa quả), **BẮN V1** (bắn đứng yên), **BẮN V2** (bắn có di chuyển).
- Dùng ngón trỏ tay phải **vẽ vòng tròn** để nhận diện game. Ứng dụng so với các mẫu đã lưu bằng OpenCV; nếu chưa có thì hỏi Claude API rồi lưu mẫu lại.
- Kỹ thuật: MediaPipe **Gesture Recognizer** lấy 21 điểm mốc (ảnh + 3D world) và nhãn cử chỉ cho mỗi bàn tay. OpenCV và NumPy đảm nhận tiền xử lý (LAB, CLAHE), xác định duỗi/cụp bằng góc khớp (có hysteresis và debounce), bộ lọc One Euro, nhận diện vòng tròn, dHash và histogram HSV.

## Chạy nhanh
```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env      # rồi điền ANTHROPIC_API_KEY
.venv\Scripts\python.exe tools\debug_view.py    # tập cử chỉ (không điều khiển chuột)
.venv\Scripts\python.exe main.py                # chạy thật; F10 để thoát
```

## Tài liệu
- [docs/01_cac_buoc_xu_ly_anh.md](docs/01_cac_buoc_xu_ly_anh.md): các bước xử lý ảnh, công thức và số đo độ trễ
- [docs/02_kien_truc_va_luong.md](docs/02_kien_truc_va_luong.md): các luồng, sơ đồ Mermaid, máy trạng thái
- [docs/03_cai_dat_va_chay.md](docs/03_cai_dat_va_chay.md): cài đặt, bảng cử chỉ, phím tắt, tinh chỉnh, lỗi thường gặp
- [docs/04_cac_cach_xoay_man_hinh.md](docs/04_cac_cach_xoay_man_hinh.md): 5 cách xoay màn hình bằng tay, so sánh và cách đang dùng

## Cấu trúc
```
main.py              khởi động các luồng + overlay + phím tắt
config.yaml          mọi ngưỡng / phím có thể chỉnh
src/capture.py       luồng đọc webcam (chỉ giữ frame mới nhất)
src/preprocess.py    lật gương, LAB, CLAHE, RGB
src/hand_tracker.py  MediaPipe Gesture Recognizer
src/hand_identity.py xác định tay trái/phải (vị trí, bám, bỏ bản trùng, tự hiệu chỉnh)
src/calibration.py   hiệu chỉnh ngưỡng chạm ngón cái theo tay người dùng (F5)
src/finger_state.py  duỗi/cụp (3D), chạm ngón cái, khe hở giữa 2 ngón (YCrCb)
src/gestures.py      tư thế, tên cử chỉ, lọc cử chỉ lạ, vòng tròn
src/smoothing.py     ánh xạ camera→màn hình + One Euro
src/profiles/        CHÉM, BẮN V1, BẮN V2
src/controller.py    gửi chuột/phím, luồng 120 Hz nội suy con trỏ
src/overlay.py       lớp phủ neon + khung camera ở góc (PySide6)
src/visualize.py     vẽ khung xương bàn tay + góc khớp
src/game_detector.py chụp màn hình + Claude API
src/game_db.py       lưu mẫu game (dHash + histogram)
tools/debug_view.py  cửa sổ minh họa các bước xử lý ảnh
tools/replay_video.py chạy lại pipeline trên video đã quay (kiểm tra không cần camera)
tools/make_doc_images.py sinh ảnh minh hoạ trước/sau từng bước xử lý ảnh cho docs/01 (từ video)
tests/               75 bài kiểm thử logic
```
