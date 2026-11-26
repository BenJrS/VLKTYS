# Bài tập lớn: Câu 10A (Gauss–Seidel) và Câu 6B (Monte-Carlo)

Hai ứng dụng desktop viết bằng **Python**: NumPy, SymPy, Matplotlib cho phần tính toán và vẽ đồ thị, **PySide6 (Qt)** cho giao diện.

```
CauA_GaussSeidel/main.py   Câu 10A: giải hệ phương trình nút mạch điện bằng Gauss–Seidel
CauB_MonteCarlo/main.py    Câu 6B: định tuổi phóng xạ bằng mô phỏng Monte-Carlo
shared/theme.py            giao diện dùng chung (màu, font, kiểu đồ thị)
shared/fonts/              font IBM Plex (giấy phép SIL OFL), đóng gói kèm ứng dụng
shared/icons/              biểu tượng ứng dụng và script tạo ra chúng
sample_data/               file mẫu để thử chức năng mở file
*.spec                     cấu hình đóng gói PyInstaller
```

## Chạy từ mã nguồn

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python CauA_GaussSeidel/main.py
python CauB_MonteCarlo/main.py
```

## Câu 10A: Gauss–Seidel

Mô hình phân tích điện áp nút **G·V = I**: G là ma trận điện dẫn, V là điện áp các nút cần tìm, I là dòng nguồn bơm vào từng nút.

| Yêu cầu đề bài | Cách thực hiện trong chương trình |
|---|---|
| Nhập ma trận hệ số và vector vế phải, giải bằng Gauss–Seidel (4đ) | Bảng ma trận mở rộng [A \| b] có thể sửa trực tiếp; chọn số nút, sai số dừng `tol`, số vòng lặp tối đa; nút **Giải hệ** |
| Đồ thị hội tụ của sai số theo số vòng lặp, ghi chú nghiệm và số vòng lặp (4đ) | Đồ thị ‖x(k) − x(k−1)‖∞ (trục log) có đường `tol`. Khung chú thích **nằm ngay trong hình** ghi số vòng lặp và toàn bộ nghiệm V1…Vn, nên ảnh lưu ra vẫn đủ thông tin |
| Import ma trận từ file, so sánh với nghịch đảo ma trận, lưu kết quả (2đ) | **Mở ma trận từ file…** (.csv/.txt, phân tách bằng phẩy, chấm phẩy hoặc khoảng trắng). Nghiệm được so sánh với **x = A⁻¹·b** (`numpy.linalg.inv`) theo từng ẩn. **Lưu kết quả…** ra .txt (báo cáo đầy đủ kèm lịch sử sai số) hoặc .csv; **Lưu đồ thị…** ra PNG/SVG/PDF |

Chương trình kiểm tra trước khi giải:
- **Chéo trội hàng**: ô đường chéo hiện màu xanh nếu hàng đó chéo trội chặt, màu đỏ nếu không.
- **Bán kính phổ** ρ(T) của ma trận lặp T = −(D+L)⁻¹U. Gauss–Seidel hội tụ với mọi điểm xuất phát khi và chỉ khi ρ(T) < 1. Điều kiện này chặt hơn chéo trội, vốn chỉ là điều kiện đủ.
- Phát hiện trường hợp phân kỳ (sai số tăng dần) và báo rõ.

File mẫu: `sample_data/mach_dien_4nut.csv` (4 nút, hội tụ sau 24 vòng với tol = 1e-10).

## Câu 6B: Monte-Carlo định tuổi phóng xạ

Mô hình phân rã bậc nhất: N(t) = N₀·e^(−λt), T½ = ln2/λ, tuổi mẫu t = ln(N₀/N)/λ.

**Tab "Dữ liệu đo"**. Chương trình đọc file chứa số đếm phân rã N(tᵢ) của một mẫu chuẩn cùng kết quả đo mẫu vật (N₀ ± σ, N ± σ). Mỗi lần thử Monte-Carlo gồm 3 bước:
1. Sinh số đếm giả lập Nᵢ* ~ Poisson(Nᵢ), vì số đếm phóng xạ tuân theo phân bố Poisson.
2. Khớp ln Nᵢ* theo tᵢ bằng bình phương tối thiểu có trọng số, suy ra **λ\*** và **T½\* = ln2/λ\***.
3. Lấy mẫu N₀*, N* theo sai số đo, suy ra **tuổi t\* = ln(N₀\*/N\*)/λ\***.

Sau 100.000 lần thử, chương trình báo trung bình, độ lệch chuẩn và **khoảng tin cậy** (90/95/99%) cho λ, T½ và tuổi. Đồ thị gồm đường phân rã kèm dải tin cậy và 3 histogram, vùng nằm trong khoảng tin cậy được tô đậm.

Như vậy λ và T½ đều được **ước lượng từ dữ liệu**. Ở phiên bản trước, T½ là đầu vào rồi lại được in ra như kết quả.

**Tab "Phương trình tự nhập"**. Người dùng gõ bất kỳ phương trình nào và khai báo các biến đo được (giá trị, σ, phân bố chuẩn hoặc đều):
- dạng tường minh: `t = ln(N0/N)/lam`
- dạng phương trình ẩn: `N = N0*exp(-ln(2)*t/T)`. Ký hiệu duy nhất chưa có trong bảng biến là ẩn cần tính, và chương trình tự giải theo ẩn đó bằng SymPy.

Ngay khi gõ, chương trình hiện "Sẽ tính: t = …", hoặc báo lỗi cụ thể nếu thiếu biến hay sai cú pháp. Có sẵn 4 phương trình mẫu. Kết quả có khoảng tin cậy.

| Yêu cầu đề bài | Cách thực hiện |
|---|---|
| Đọc dữ liệu/phương trình đã lưu, đồ thị trên GUI, tính tuổi, λ, T½ bằng Monte-Carlo (4đ) | **Mở dữ liệu…** (.json, hoặc .csv hai cột t, N); tab "Dữ liệu đo" |
| Nhập phương trình bất kỳ (3đ) + khoảng tin cậy (khuyến khích) | Tab "Phương trình tự nhập", có CI ở cả hai tab |
| Lưu hình ảnh và kết quả (3đ) + đóng gói chạy độc lập (khuyến khích) | **Lưu đồ thị…** (PNG/SVG/PDF), **Lưu kết quả…** (.txt/.csv), **Lưu dữ liệu…** (.json); bản .exe/.app đã build |

Mỗi lần chạy dùng một **seed** cố định (mặc định 2026), nên chạy lại cho đúng cùng kết quả và người chấm kiểm tra lại được.

File mẫu: `sample_data/du_lieu_dinh_tuoi_C14.json`, `sample_data/phuong_trinh_mau.json`.

## Đóng gói ứng dụng chạy độc lập

- **Windows (.exe)**: push repo lên GitHub, workflow `.github/workflows/build.yml` tự build. Tải file ở mục *Artifacts* của lần chạy. Nếu có máy Windows thì chạy `build_windows.bat`.
- **macOS (.app/.dmg)**: `./build_macos.sh`.

Font IBM Plex và icon được đóng gói kèm, nên giao diện hiển thị giống nhau trên mọi máy.
