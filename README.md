# Bài tập lớn – Phương pháp số

Gồm 2 ứng dụng GUI độc lập, viết bằng **Python + PySide6 (Qt)**, đồ thị bằng **matplotlib**.

```
BaiTapLon/
├── CauA_GaussSeidel/main.py     ← Câu 10A: giải mạch điện bằng Gauss–Seidel
├── CauB_MonteCarlo/main.py      ← Câu 6B: định tuổi phóng xạ bằng Monte-Carlo
├── sample_data/                 ← File dữ liệu mẫu để test nút Import/Đọc dữ liệu
├── requirements.txt
├── build_windows.bat            ← chạy trên Windows để tạo .exe
├── build_macos.sh               ← chạy trên macOS để tạo .dmg
└── .github/workflows/build.yml  ← tự build cả .exe lẫn .dmg trên GitHub (khuyên dùng)
```

## 1. Chạy thử ngay (không cần đóng gói)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

python CauA_GaussSeidel/main.py
python CauB_MonteCarlo/main.py
```

## 2. Nội dung từng ứng dụng

### Câu 10A — Gauss–Seidel (mạch điện nhiều nút)
- Nhập ma trận hệ số A và vector b bằng bảng, hoặc bấm **"Ví dụ mẫu (3 nút)"**.
- Bấm **"Import ma trận từ file…"** và chọn `sample_data/mach_dien_4nut.csv` để thử chức năng nhập từ file ngoài (định dạng: mỗi hàng là 1 phương trình, n hệ số + 1 giá trị b, phân tách bằng dấu phẩy).
- Bấm **"GIẢI HỆ"**: chương trình lặp Gauss–Seidel, vẽ đồ thị hội tụ sai số theo số vòng lặp (trục log), in nghiệm + số vòng lặp cần thiết, đồng thời so sánh với nghiệm giải trực tiếp (`numpy.linalg.solve`).
- Lưu kết quả (`.txt`) và đồ thị (`.png`) bằng 2 nút phía dưới bảng nhập liệu.

### Câu 6B — Monte-Carlo (định tuổi phóng xạ)
Có 2 tab:
- **"Mô phỏng chuẩn"**: nhập N₀ (số hạt nhân ban đầu), N (đo hiện tại), T₁/₂ (chu kỳ bán rã) cùng sai số đo của từng đại lượng → chương trình chạy Monte-Carlo (mặc định 100.000 lần thử) và tính đồng thời **tuổi mẫu vật t**, **hằng số phân rã λ**, **chu kỳ bán rã T₁/₂** kèm khoảng tin cậy 95%. Bấm **"Ví dụ mẫu (C-14)"** hoặc **"Đọc dữ liệu đã lưu…"** rồi chọn `sample_data/du_lieu_dinh_tuoi_C14.json`.
- **"Phương trình tùy ý"**: gõ bất kỳ công thức nào (vd `ln(N0/N)/lam`), khai báo các biến với giá trị trung bình/độ lệch chuẩn/phân bố → chương trình tự parse (sympy) và chạy Monte-Carlo, cho ra kết quả + khoảng tin cậy 95%. Có thể nạp sẵn `sample_data/phuong_trinh_mau.json`.
- Nút **"Lưu hình ảnh đồ thị"** và **"Lưu kết quả tính toán"** ở cuối cửa sổ áp dụng cho tab đang mở.

## 3. Đóng gói thành ứng dụng độc lập (.exe / .dmg) — để lấy điểm khuyến khích

Mã nguồn đã được kiểm thử với PyInstaller (build thành công, không thiếu module). Có 3 cách:

### Cách A — Tự động, KHÔNG cần máy Windows/Mac (khuyên dùng)
1. Tạo 1 repo GitHub mới, đẩy (push) toàn bộ thư mục `BaiTapLon` này lên.
2. Vào tab **Actions** của repo → chọn workflow **"Build EXE (Windows) & DMG (macOS)"** → bấm **Run workflow**.
3. Đợi ~5 phút, vào mục **Artifacts** của lần chạy để tải về:
   - `CauA_GaussSeidel-windows-exe.zip`, `CauB_MonteCarlo-windows-exe.zip`
   - `CauA_GaussSeidel-macos-dmg.zip`, `CauB_MonteCarlo-macos-dmg.zip`

GitHub cấp máy ảo Windows và macOS miễn phí để build, bạn không cần sở hữu máy Mac.

### Cách B — Build thủ công trên máy Windows
```
build_windows.bat
```
→ ra `CauA_GaussSeidel\dist\CauA_GaussSeidel.exe` và `CauB_MonteCarlo\dist\CauB_MonteCarlo.exe`

### Cách C — Build thủ công trên máy macOS
```bash
chmod +x build_macos.sh
./build_macos.sh
```
→ ra `CauA_GaussSeidel/dist/CauA_GaussSeidel.dmg` và `CauB_MonteCarlo/dist/CauB_MonteCarlo.dmg`

## 4. Bảng đối chiếu với yêu cầu đề bài

| Yêu cầu đề bài | Nơi đáp ứng |
|---|---|
| **10A** – nhập ma trận/vector, giải Gauss–Seidel (4đ) | Bảng nhập liệu + nút "GIẢI HỆ" |
| **10A** – đồ thị hội tụ sai số, in nghiệm + số vòng lặp (4đ) | Đồ thị bên phải + ô kết quả |
| **10A** – import file, so sánh trực tiếp, lưu kết quả (2đ) | Nút Import, so sánh tự động, nút Lưu |
| **6B** – đọc dữ liệu đã lưu, đồ thị, tính t/λ/T₁/₂ bằng MC (4đ) | Tab "Mô phỏng chuẩn" |
| **6B** – nhập phương trình bất kỳ (3đ) + CI (khuyến khích) | Tab "Phương trình tùy ý" (đều có CI 95%) |
| **6B** – lưu hình ảnh + kết quả (3đ) + đóng gói app (khuyến khích) | 2 nút Lưu + hướng dẫn build .exe/.dmg ở trên |

## 5. Gợi ý khi báo cáo/thuyết trình
- Giải thích ngắn gọn nguyên lý Monte-Carlo: lan truyền sai số đo bằng cách lấy mẫu ngẫu nhiên nhiều lần theo phân bố của biến đầu vào, từ đó suy ra phân bố (và độ bất định) của đại lượng cần tìm.
- Với Gauss–Seidel: nhắc điều kiện hội tụ (ma trận chéo trội hàng là điều kiện đủ) — chương trình có tự kiểm tra và cảnh báo.
- Nên chụp màn hình đồ thị + kết quả từ chính chương trình để chèn vào báo cáo (dùng luôn nút "Lưu hình ảnh").
