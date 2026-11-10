# 3. Cài đặt và chạy

## 3.1 Yêu cầu
- Windows 10/11, có webcam.
- **Python 3.12.** Nên dùng bản 3.12 vì MediaPipe thường hỗ trợ Python mới chậm. Kiểm tra bằng `py -0p`.
- Internet để: cài thư viện, tải mô hình MediaPipe ở lần chạy đầu (khoảng 7,5 MB), và gọi Claude khi gặp game mới.

## 3.2 Cài đặt
```powershell
cd D:\HandGameController
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### API key cho Claude (phần nhận diện game)
1. Tạo key tại https://console.anthropic.com (mục API Keys).
2. Sao chép `.env.example` thành `.env` rồi điền key:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```
Không có key thì ứng dụng vẫn chạy bình thường, chỉ không tự nhận diện được game mới. Khi đó dùng **F8** để đổi chế độ bằng tay.

## 3.3 Chạy
| Lệnh | Mục đích |
|---|---|
| `.venv\Scripts\python.exe tools\debug_view.py` | Xem từng bước xử lý ảnh, trạng thái ngón và cử chỉ. **Không** điều khiển chuột. Dùng để tập cử chỉ, chỉnh ngưỡng và chụp ảnh cho báo cáo (phím `s`) |
| `.venv\Scripts\python.exe main.py --dry-run` | Chạy lớp phủ neon nhưng **không gửi** chuột/phím (bấm F9 để bật) |
| `.venv\Scripts\python.exe main.py` | Chạy đầy đủ |
| `.venv\Scripts\python.exe tools\replay_video.py video.mp4 --mode shoot_v2` | Chạy lại pipeline trên một video đã quay (không cần camera), in cử chỉ/phím theo thời gian; `--out ket_qua.mp4` để ghi video có vẽ khớp ngón |
| `.venv\Scripts\python.exe tools\make_doc_images.py video.mp4 --screenshot anh_game.png` | Sinh lại ảnh minh hoạ trước/sau từng bước cho `docs/01` (lưu vào `docs/images/`) |
| `.venv\Scripts\python.exe main.py --mode shoot_v2` | Mở thẳng một chế độ (`slice`, `shoot_v1`, `shoot_v2`) |
| `.venv\Scripts\python.exe -m pytest -q tests` | Chạy 75 bài kiểm thử logic (không cần camera) |

## 3.4 Cách chơi
1. Mở game trên trình duyệt và để cửa sổ trình duyệt được chọn (focus).
2. Chạy `main.py`.
3. Ngồi cách webcam khoảng 50–80 cm. Giơ tay **cao**, sao cho cả **cổ tay** nằm trong khung hình. Giữ **bàn tay dọc, lòng hoặc cạnh bàn tay hướng về camera**; **đừng chĩa ngón tay thẳng vào ống kính** (camera chỉ có một mắt nên không đo được độ sâu).
4. Lần đầu dùng, bấm **F5** và làm theo 2 bước trên màn hình (tách ngón cái ra → chạm ngón cái vào ngón trỏ) để chương trình đo ngưỡng **chạm ngón** theo bàn tay bạn.
5. **Dùng ngón trỏ tay phải vẽ một vòng tròn** để ứng dụng nhận diện game. Lần đầu sẽ gọi Claude (khoảng 1–5 s), các lần sau nhận ngay từ bộ nhớ.

### Bảng cử chỉ
Tên ở cột đầu là tên hiện trên màn hình khi cử chỉ được nhận. Cử chỉ không có trong bảng của chế độ đang chạy thì hiện **KHÔNG RÕ** và bị bỏ qua.

**Click:**
- **Click trái** = đầu ngón cái **chạm vào chỗ nào cũng được trên đường ngón trỏ** (đường xanh lá trong khung camera, từ gốc tới đầu ngón).
- **Click phải** = **ngón giữa duỗi nửa chừng** (dài khoảng một nửa ngón trỏ, không gập hẳn). Phải giữ yên khoảng 0,1 giây; gập hay duỗi nhanh đi qua vùng này thì không tính.
- **Click nhanh** (nhả trong vòng 0,3 giây): con trỏ đứng yên tại chỗ suốt lần click, nên click không lệch.
- **Giữ lâu** (quá 0,3 giây): tên đổi thành GIỮ CHUỘT TRÁI/PHẢI và con trỏ đi theo tay, dùng để kéo-thả hoặc bôi đen.

**Cử chỉ chung – tay phải – CHÉM và menu**
| Tên | Cử chỉ | Hành động |
|---|---|---|
| CON TRỎ | Giơ ngón trỏ (có thể kèm ngón giữa) | Con trỏ nằm đúng chỗ đầu ngón trỏ (trùng vệt sáng); tay hơi rung thì con trỏ đứng im |
| CLICK TRÁI / GIỮ CHUỘT TRÁI | Ngón cái chạm đường ngón trỏ | Chuột trái |
| CLICK PHẢI / GIỮ CHUỘT PHẢI | Ngón giữa duỗi nửa chừng | Chuột phải |
| KÉO / CHÉM (chỉ chế độ CHÉM) | Khép ngón trỏ và ngón giữa sát nhau | Giữ chuột trái, con trỏ theo đầu ngón trỏ |
| TẠM DỪNG | Xòe cả bàn tay 1 giây | Esc |
| (nhận diện game) | Ngón trỏ vẽ một vòng tròn | Chụp màn hình → chọn chế độ |

**BẮN V1** (đứng yên, không có KÉO / CHÉM)
| Tay | Tên | Cử chỉ | Hành động |
|---|---|---|---|
| Trái | SÚNG | Bàn tay dọc, trỏ + giữa duỗi khép sát chĩa lên | Tâm ngắm nằm đúng chỗ đầu ngón trỏ |
| Trái | BÓP CÒ | **Gập ngón cái vào lòng bàn tay** (hoặc chạm ngón trỏ), **hoặc gập đầu ngón cái xuống ở khớp thứ 2** | Bắn; còn gập thì bắn liên tục, mở ra thì ngừng. Phải mở ngón cái ra một lần trước, để giơ súng với ngón cái khép sẵn thì không tự bắn |
| Phải | CHỈ LÊN → MỞ NGẮM | Chỉ ngón trỏ lên, rồi cụp ngón trỏ xuống | Giữ chuột phải |
| Phải | (chung) | Khi tay trái không làm SÚNG | Bấm menu |

**BẮN V2** (có di chuyển, không có KÉO / CHÉM, không có TẠM DỪNG). Vùng điều khiển gần trọn khung hình. Tay phải **không vẽ vệt sáng** trên màn hình.
| Tay | Tên | Cử chỉ | Hành động |
|---|---|---|---|
| Phải | XOAY | Giơ ngón trỏ rồi đưa tay lệch khỏi tâm | Ở menu (con trỏ hiện): con trỏ nằm đúng chỗ ngón tay. Trong trận: **3 vòng** ghim giữa màn hình, tính theo đốt ngón. **Vòng 1** (nhỏ): tay dao động trong đây thì không xoay. **Vòng 2** (rộng nhất): *chỉ đâu nhìn đó*, màn hình xoay để chỗ tay chỉ về tâm ngắm; tay đứng yên thì dừng, và sau khoảng 0,25 giây chỗ đặt tay thành **tâm mới** (vạch xanh biến mất). **Vòng 3** (ngoài cùng): xoay liên tục về hướng tay, càng ra ngoài càng nhanh. **Mất tay** khi đang ở vòng 3 thì xoay tiếp hướng cũ tối đa 3 giây. Xem [04_cac_cach_xoay_man_hinh.md](04_cac_cach_xoay_man_hinh.md) |
| Phải | NẠP ĐẠN | **Xòe cả bàn tay phải** | R (1 lần mỗi lần xòe) |
| Phải | ĐẶT TÂM | **Nắm cả bàn tay phải** | Chỉ để chọn chỗ đặt tay cho thoải mái: lúc nắm, tâm đi theo tay; mở tay ra thì tâm giữ nguyên ở đó. Không xoay. Vòng xoay vẫn ở giữa màn hình |
| Phải | NGẮM | **Khép ngón trỏ + ngón giữa** | Giữ chuột phải; vẫn xoay được trong lúc ngắm |
| Phải | BẮN | **Khép ngón cái lại** (chạm/tới gần ngón trỏ, hoặc gập) | Chuột trái, vẫn xoay được trong lúc bắn. Phải mở ngón cái ra một lần trước |
| Trái | DI CHUYỂN | **Khép ngón trỏ + ngón giữa** | Điểm điều khiển là **đốt ngón** (khớp gốc ngón trỏ + giữa), chấm sáng tay trái cũng vẽ ở đó nên **núm luôn trùng chấm**. Lần đầu làm cử chỉ thì chỗ đó thành tâm. Đưa chấm đi hướng nào thì nhân vật đi hướng đó (8 hướng); chấm gần tâm thì đi thẳng (W) **Mất tay** khi đang đi thì đi tiếp hướng cũ tối đa 3 giây |
| Trái | ĐỨNG LẠI | **Xòe cả bàn tay trái** | Nhả W/A/S/D, giữ nguyên tâm |
| Trái | ĐỨNG LẠI + ĐẶT TÂM | **Nắm cả bàn tay** | Nhả W/A/S/D. Lúc nắm, nút di chuyển đi theo tay; mở ra khép 2 ngón thì nút **giữ nguyên tại chỗ** (không nhảy) |
| Trái | NHẢY | Đang di chuyển thì khép ngón cái lại | Space |
| Trái | (không thấy tay) | | Đang đi thì đi tiếp hướng cũ tối đa `lost_keep_sec` (3 giây), sau đó đứng yên. Mất tay quá 3 giây thì lần sau lưu lại vị trí gốc mới |

**Khi click không ăn:** khung camera có một dòng chẩn đoán màu xanh nhạt, ví dụ `trai: 2D 0.31>0.15 3D 0.40<0.45 | phai: giua/tro 0.92 (can 0.25-0.7)`. Dòng này cho biết số đo nào đang vượt ngưỡng, từ đó biết cần chạm gần hơn, bấm F5 hay chỉnh `touch.*` / `middle_half.*`.

### Phím tắt
| Phím | Tác dụng |
|---|---|
| **F5** | Hiệu chỉnh ngưỡng **chạm ngón** theo tay bạn (lưu `calibration.json`) |
| **F6** | Ẩn/hiện khung camera ở góc màn hình (khung xương bàn tay + góc khớp) |
| **F7** | Đổi chuột tuyệt đối / tương đối (dùng khi game khóa con trỏ); được lưu theo từng game |
| **F8** | Đổi chế độ: CHÉM → BẮN V1 → BẮN V2 |
| **F9** | Bật/tắt gửi lệnh. Khi tắt, mọi phím và nút chuột đều được nhả |
| **F10** | Thoát ngay |

### Game đã thử
- Chém: https://naptechgames.com/game/fruit-slicer, https://www.coolmathgames.com/0-fruit-ninja?variant=next
- Bắn V1: https://www.crazygames.com/vn/game/rattlesnake-reflex
- Bắn V2: https://www.crazygames.com/vn/game/bodycamera-shooter

### Khung camera ở góc màn hình
Ở góc dưới bên phải màn hình có ảnh camera thu nhỏ, cho thấy chương trình đang xác định khớp ngón tay ra sao:
- **Nét màu** là ngón đang DUỖI, mỗi ngón một màu. **Nét đỏ** là ngón đang CỤP.
- **Con số cạnh khớp giữa (PIP)** của 4 ngón dài là góc khớp tính bằng độ. Góc trên 150° thì tính là duỗi, dưới 120° thì tính là cụp.
- **r=...** cạnh ngón cái là tỉ lệ khoảng cách từ đầu ngón cái đến gốc ngón trỏ, chia cho kích thước lòng bàn tay.
- Phía trên mỗi bàn tay có nhãn tay (PHAI/TRAI) kèm nguồn của nhãn (vị trí / bám / MediaPipe), tên cử chỉ và nhãn của mô hình Google. Nếu frame bị loại thì ghi lý do bằng chữ đỏ. Chữ trong khung camera không có dấu vì OpenCV không vẽ được chữ có dấu. Khung xám là vùng hoạt động được kéo giãn ra toàn màn hình.

Đổi vị trí và kích thước bằng `overlay.preview_corner` và `overlay.preview_width` trong `config.yaml`, hoặc tắt hẳn bằng `overlay.camera_preview: false`.

## 3.5 Tinh chỉnh (`config.yaml`)
| Hiện tượng | Chỉnh |
|---|---|
| Phải với tay quá xa mới tới mép màn hình | Thu hẹp `mapping.region` |
| Phải giơ tay quá cao, tay che màn hình | Hạ vùng hoạt động: `mapping.region_by_mode` cho từng chế độ (BẮN V1 mặc định `[0.05, 0.35, 0.75, 0.97]`: thấp ngang cánh tay, lệch trái cho tay trái bắn; BẮN V2 `[0.02, 0.02, 0.98, 0.98]`: gần trọn khung hình) |
| Con trỏ rung | Giảm `smoothing.min_cutoff` (ví dụ 1.0) |
| Con trỏ trễ khi vẩy nhanh | Tăng `smoothing.beta` |
| Click trái không ăn / tự click | Bấm **F5**; xem dòng chẩn đoán "trai: ..." trong khung camera rồi chỉnh `touch.*` |
| Click phải khó / tự click phải | Xem "giua/tro" trong dòng chẩn đoán; chỉnh `middle_half.half_lo`, `half_hi`, `frames` |
| Click nhanh bị thành kéo (hoặc ngược lại) | Chỉnh `profiles.click_hold_ms` |
| Ngón nhận sai duỗi/cụp | Xem góc khớp trong khung camera rồi chỉnh `fingers.extend_angle` / `fold_angle` |
| Khép trỏ + giữa không nhận | Xem số "khep .. khe .." trong khung camera; chỉnh `fingers.together_ratio`, `spread_ratio`, `gap_max` |
| Con trỏ "dính" quá / chưa đủ | Chỉnh `profiles.still_speed`, `still_release` |
| V2 xoay quá nhanh / quá chậm | Vòng 2 xoay chưa tới hoặc quá chỗ tay chỉ: chỉnh `aim_point_gain`. Kích thước 3 vòng: `aim_deadzone`, `aim_point_radius`, `aim_radius`. Tốc độ xoay liên tục ở vòng 3: `aim_max_speed`, `aim_curve`. Thời gian đi/xoay tiếp khi mất tay: `lost_keep_sec` (0 = tắt). Muốn quay về kiểu chuột thật: `aim_style: delta`. Xoay lên/xuống: `aim_vertical_scale`. Xoay còn giật: tăng `rotate_smooth_ms` |
| Nút di chuyển to/nhỏ | `profiles.joystick_radius` (mặc định 0,12 chiều cao màn hình) |
| V2 hướng đi nhảy qua lại | Giảm `profiles.dir_smooth` hoặc tăng `dir_hyst_deg` |
| Hay bị KHÔNG RÕ | Giảm `gestures.stable_need` xuống 2, hoặc tăng `gestures.model_min_score` |
| Báo "đừng chĩa ngón vào camera" dù tay đang dọc | Giảm `gestures.foreshorten_min` |
| Báo "bàn tay ra ngoài khung" | Đưa tay lên cao; hoặc tăng `gestures.edge_margin` |
| Con trỏ còn rung / còn trễ | Tăng / giảm `profiles.cursor_tau_ms` |
| Cử chỉ phản ứng chậm (game phản xạ) | `fingers.debounce_frames: 1` |
| SÚNG tay trái không nhận | Giữ bàn tay dọc; tăng `fingers.together_ratio` hoặc `gestures.vertical_max_deg` |
| Nhân vật V2 đi lung tung | Tăng `profiles.joystick_deadzone` |
| Xoay màn hình V2 quá nhanh/chậm | Chỉnh `profiles.relative_sensitivity` |
| Nhãn trái/phải bị ngược | Mặc định đã đảo (`tracker.swap_handedness: true`, đo trên video thật). Nếu webcam của bạn cho kết quả ngược lại thì đặt `false` |

## 3.6 Lỗi thường gặp
| Lỗi | Cách xử lý |
|---|---|
| `Không mở được camera 0` | Đóng các ứng dụng khác đang dùng camera (Zoom, Teams...) hoặc đổi `camera.index` sang 1 |
| HUD báo Camera < 25 FPS | Phòng tối khiến camera tăng thời gian phơi sáng. Hãy bật đèn chiếu vào mặt và tay |
| Đang ra cử chỉ di chuyển mà nhân vật đứng im | Nhìn dòng **"Phím đang giữ"** trên HUD. **Không có W/A/S/D:** app chưa nhận cử chỉ (xem tên cử chỉ tay trái; khép hoặc hơi tách 2 ngón đều được). **Có W/A/S/D mà nhân vật vẫn đứng:** do phần mềm khác hoặc do game. Thường gặp nhất là **UniKey** ở chế độ tiếng Việt (Telex dùng W, S, D để gõ dấu): chuyển UniKey sang **E** (tiếng Anh) bằng Ctrl+Shift, Alt+Z, hoặc bấm vào biểu tượng UniKey trên khay hệ thống. Ngoài ra kiểm tra cửa sổ game có đang được chọn (focus) không, và nhân vật có đang bị tường/vật cản chặn không |
| Game không nhận click/phím | Bấm vào cửa sổ game để game có focus. Nếu game chạy với quyền Administrator thì `main.py` cũng phải chạy với quyền Administrator |
| Tâm ngắm không xoay trong game 3D | Bấm F7 để chuyển sang chuột tương đối |
| `Chưa có ANTHROPIC_API_KEY` | Tạo file `.env` (xem 3.2), hoặc đổi chế độ bằng F8 |
| Lần đầu chạy bị chậm | Ứng dụng đang tải mô hình MediaPipe về `models/` (chỉ một lần) |
