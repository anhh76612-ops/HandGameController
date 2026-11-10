# 4. Các cách xoay màn hình (camera) bằng tay

Trong game bắn súng góc nhìn thứ nhất, game **khóa con trỏ** (Pointer Lock) và chỉ nhận **chuyển động chuột tương đối**: chuột đi bao nhiêu thì camera xoay bấy nhiêu. Vì vậy mọi cách điều khiển bằng tay cuối cùng đều phải đổi chuyển động hoặc tư thế của tay thành một dòng **Δx, Δy gửi mỗi frame**. Dưới đây là các cách phổ biến, kèm ưu nhược điểm.

| # | Tên | Tay làm gì | Màn hình làm gì | Công thức (mỗi frame) |
|---|---|---|---|---|
| 1 | **Kiểu chuột thật** (position / delta) | Di chuyển tay | Tay đi bao nhiêu xoay bấy nhiêu; tay đứng yên thì dừng | Δ = (vị trí_bây_giờ − vị trí_frame_trước) × độ nhạy |
| 2 | **Kiểu cần gạt** (rate control / joystick) | Đưa tay **lệch** khỏi tâm rồi **giữ** ở đó | Xoay **liên tục** về hướng lệch; lệch càng xa xoay càng nhanh; đưa tay về tâm thì dừng | Δ = hướng_lệch × tốc_độ(khoảng_lệch) × Δt |
| 3 | **Lai** (delta + đẩy mép) | Như kiểu 1, nhưng khi tay ra sát mép vùng điều khiển | Giữa vùng: như chuột thật. Sát mép: xoay tiếp liên tục | kiểu 1 ở giữa, kiểu 2 ở vùng rìa |
| 4 | **Góc tuyệt đối** | Vị trí tay = góc nhìn | Tay ở đâu, camera nhìn về đó | cần biết chính xác độ nhạy của game → khó làm với game web |
| 5 | **Xoay cổ tay** (orientation) | Nghiêng/xoay bàn tay như cầm tay cầm | Như kiểu 2 nhưng dùng góc nghiêng thay cho vị trí | Δ = góc_nghiêng × hệ số × Δt |

## So sánh chi tiết

### 1. Kiểu chuột thật (delta)
- **Ưu:** chính xác nhất; dễ ngắm tinh vào một điểm; giống cảm giác dùng chuột.
- **Nhược:** muốn xoay nhiều (ví dụ quay 180°) phải **đưa tay qua lại nhiều lần**, vì tay sẽ chạm mép khung hình. Tay rung là camera rung theo.
- **Kỹ thuật chống rung:** đường cong độ nhạy (chậm nhân 0,3, nhanh nhân 1,6), vùng chết vài pixel.
- Dự án dùng kiểu này cho BẮN V2 ở phiên bản trước.

### 2. Kiểu cần gạt (rate control) — **phiên bản hiện tại dùng cách này**
- Giống cần analog phải của tay cầm chơi game: **tâm** là chỗ tay bắt đầu, độ lệch so với tâm quyết định **tốc độ xoay** (không phải góc xoay).
- **Vùng chết** quanh tâm: tay hơi rung hay xê dịch nhẹ không làm camera xoay. Đây là ưu điểm lớn nhất so với kiểu 1, vì tay trong không khí rất khó giữ yên.
- **Đường cong tốc độ:** `tốc_độ = tốc_độ_max × ((lệch − vùng_chết) / (bán_kính − vùng_chết))^γ`, với γ ≈ 1,5–2. Lệch ít thì xoay chậm để ngắm tinh, lệch nhiều thì xoay nhanh để quay người.
- **Ưu:** quay bao nhiêu cũng được mà không phải đưa tay qua lại; ít mỏi tay; không sợ chạm mép khung hình.
- **Nhược:** ngắm tinh khó hơn chuột thật (phải canh để camera dừng đúng chỗ); cần tập vài phút để quen tốc độ.

### 3. Kiểu lai (delta + đẩy mép)
- Giữa vùng điều khiển: như chuột thật (chính xác). Khi tay ra **8% sát mép**: xoay tiếp liên tục như cần gạt.
- Giải quyết nhược điểm "hết chỗ" của kiểu 1, nhưng vẫn rung theo tay ở giữa vùng.
- Dự án đã thử kiểu này ở phiên bản trước (`edge_zone`, `edge_speed`).

### 4. Góc tuyệt đối
- Vị trí tay ánh xạ thẳng thành góc nhìn, như ánh xạ con trỏ tuyệt đối ở chế độ CHÉM.
- Phải biết chính xác "1 pixel chuột = bao nhiêu độ" của từng game, và game không được làm mượt hay tăng tốc chuột. Gần như không làm được với game trên trình duyệt.

### 5. Xoay cổ tay
- Dùng góc nghiêng của bàn tay (tính từ các điểm mốc 3D) thay cho vị trí. Tay có thể đứng yên một chỗ.
- Nhược: góc nghiêng ước lượng từ webcam một mắt nhiễu hơn vị trí, và xoay cổ tay lâu dễ mỏi.

## Cài đặt trong dự án (BẮN V2, tay phải)
Khi game khóa con trỏ (đang trong trận), tay phải dùng **kiểu 2 – cần gạt**:
1. Lần đầu giơ ngón trỏ, vị trí chấm đầu ngón được lưu làm **tâm xoay**. Overlay vẽ một vòng tròn mờ ở đó để bạn biết tâm nằm đâu.
2. Chấm cách tâm ít hơn vùng chết (`aim_deadzone`) thì không xoay.
3. Ra ngoài vùng chết thì xoay liên tục về hướng lệch, với tốc độ theo đường cong ở trên (`aim_max_speed`, `aim_curve`).
4. Trong lúc bắn vẫn xoay được.
5. Ở menu (con trỏ hiện), con trỏ vẫn nằm đúng chỗ ngón tay (tuyệt đối) như trước.
6. **Nắm cả bàn tay** để đặt lại tâm xoay: vòng tròn đi theo bàn tay; mở ra giơ ngón trỏ thì chấm đầu ngón khi đó là tâm mới. Không lấy tâm ngay lúc nắm, vì khi đó đầu ngón trỏ đang gập trong lòng bàn tay; vừa duỗi ra, chấm sẽ nhảy lên làm màn hình tự xoay. Tay trái (cần điều khiển di chuyển) cũng đặt lại tâm theo cách này.

Muốn quay lại kiểu 1 hoặc kiểu 3, đặt `profiles.aim_style` trong `config.yaml` thành `delta` (kiểu 1, có đẩy mép như kiểu 3). Giá trị mặc định là `rate` (kiểu 2).

**Xoay lên/xuống chậm hơn chiều ngang:** game bắn súng hầu như chỉ cần xoay ngang, còn ngước lên/cúi xuống dễ làm mất tâm. Vì vậy tốc độ dọc được nhân thêm `aim_vertical_scale` (mặc định 0,1 = 10%; đặt 0 để tắt hẳn xoay dọc).

## Vì sao xoay bằng tay không mượt bằng chuột?
| | Chuột | Tay qua webcam |
|---|---|---|
| Số lần cập nhật vị trí mỗi giây | 125–1000 | 30 (khoảng 20 khi phòng tối) |
| Độ trễ | khoảng 10 ms | khoảng 100–150 ms (camera + nhận diện + làm mượt) |
| Độ chính xác | từng pixel, không rung | điểm mốc rung vài pixel, phải lọc nên thêm trễ |
| Cách điều khiển | vị trí (chính xác) | cần gạt: điều khiển *tốc độ*, khó dừng đúng chỗ hơn |

**Đã cải thiện trong dự án:**
- **Gửi vận tốc thay cho "cục" dịch chuyển:** bản trước, mỗi frame camera (30 lần/giây) gửi một quãng dịch chuyển và controller xả quãng đó trong vài ms, nên màn hình quay theo từng nhịp giật. Bây giờ profile gửi **vận tốc xoay**, và luồng 120 Hz tự cộng `vận_tốc × Δt` mỗi khoảng 8 ms, nên màn hình quay đều như cần analog. Khi vận tốc đổi, nó chuyển dần sang giá trị mới trong khoảng `rotate_smooth_ms` (60 ms) để không giật. Mất tay quá 0,15 giây thì vận tốc tự về 0.
- Vùng chết gần tâm, đường cong tốc độ, xoay dọc chậm hơn xoay ngang, và đứng im khi bắn.

**Bạn có thể làm thêm:**
- **Bật đủ đèn** để camera đạt 30 FPS thay vì khoảng 20 FPS; đây là yếu tố ảnh hưởng nhiều nhất.
- Trong game, đặt **độ nhạy chuột thấp hơn** và tăng `aim_max_speed` tương ứng. Bước xoay nhỏ hơn thì nhìn mượt hơn.
- Tăng `rotate_smooth_ms` (ví dụ 100) để mượt hơn, đổi lại phản ứng hơi chậm hơn.

## Điểm điều khiển: đốt ngón thay cho đầu ngón
Ở BẮN V2, cả nút di chuyển (tay trái) và xoay màn hình (tay phải) đều tính theo **đốt ngón**: trung điểm khớp gốc ngón trỏ và ngón giữa (điểm 5 và 9). Đầu ngón trỏ di chuyển rất nhiều khi nắm hoặc mở tay, nên trước đây vừa thả nắm tay là nút nhảy theo. Đốt ngón thì gần như đứng yên khi nắm hay mở, nên:
- nắm tay để dời tâm tới chỗ thoải mái, mở ra thì tâm nằm đúng chỗ đó, không nhảy;
- chấm sáng tay trái được vẽ ở đốt ngón, nên luôn trùng với núm.

Vòng xoay của tay phải được **ghim ở giữa màn hình** (chỗ tâm ngắm của game), chỉ để hiển thị hướng và độ mạnh đang xoay. Tâm thật của tay phải là chỗ bạn đặt tay (chọn bằng cách nắm tay).

## Phiên bản hiện tại: 3 vòng (kết hợp cách 1 và cách 2)
Vòng tròn vẽ ghim ở giữa màn hình. Độ lệch là khoảng cách từ đốt ngón tới tâm tay, tính theo chiều cao màn hình:

| Vòng | Bán kính | Hành vi | Loại điều khiển |
|---|---|---|---|
| 1 – trong cùng | < `aim_deadzone` (0,02) | Không xoay: tay dao động nhẹ, chống giật | – |
| 2 – giữa, rộng nhất | 0,02 – `aim_point_radius` (0,30) | **Chỉ đâu nhìn đó:** mỗi vị trí tay ứng với một điểm trên màn hình, và màn hình xoay để đưa điểm đó về tâm ngắm. Tay đứng yên thì dừng; đưa tay về thì màn hình quay về | Vị trí, kiểu 1 (gửi dịch chuyển = thay đổi vị trí × `aim_point_gain`) |
| 3 – ngoài cùng | 0,30 – `aim_radius` (0,42) | **Xoay liên tục** về hướng tay, nhanh dần tới `aim_max_speed` ở mép ngoài | Tốc độ, kiểu 2 (gửi vận tốc, luồng 120 Hz cộng dồn) |

- **Mất tay** khi đang ở vòng 3: màn hình xoay tiếp theo hướng lần cuối, tối đa `lost_keep_sec` giây (mặc định 3). Ở vòng 2 thì không xoay tiếp, vì kiểu "chỉ đâu nhìn đó" không có vận tốc. Tay trái cũng vậy: đang đi mà mất tay thì đi tiếp hướng cũ. Có giới hạn thời gian để nhân vật không chạy mãi khi camera mất dấu tay lâu.
- Khi tay xuất hiện lại, vị trí ở vòng 2 được đồng bộ ở frame đầu tiên, nên màn hình không bị giật bù.
- **Vòng 2 tự đặt lại tâm:** sau khi màn hình xoay tới chỗ tay chỉ, nếu tay đứng yên khoảng 0,25 giây (`aim_settle_ms`) thì chỗ tay đang đặt trở thành tâm mới, giống như vừa nắm tay đặt tâm. Vạch chỉ hướng biến mất và lần xoay sau tính từ đây. Vòng 3 không tự đặt lại tâm, nếu không màn hình sẽ ngừng xoay liên tục. Muốn đưa tay về chỗ cũ thì nắm tay rồi dời tay.
- Vẫn xoay được trong lúc bắn.
- **Tốc độ như nhau mọi hướng:** độ lệch của tay tính theo tỉ lệ khung *camera* (4:3), không theo màn hình (16:10), nên cùng một quãng tay thật thì xoay ngang, dọc hay chéo đều nhanh như nhau. `aim_vertical_scale` mặc định là 1.
