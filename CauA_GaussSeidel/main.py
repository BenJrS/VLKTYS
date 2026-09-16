# -*- coding: utf-8 -*-
"""
CÂU 10A: Giải hệ phương trình tuyến tính của mạch điện nhiều nút
bằng phương pháp lặp Gauss-Seidel.

Yêu cầu:
  1. Nhập ma trận hệ số A và vector vế phải b (định luật Kirchhoff dòng nút),
     giải bằng Gauss-Seidel.                                        (4 điểm)
  2. Vẽ đồ thị hội tụ sai số theo số lần lặp. In nghiệm và số vòng lặp. (4 điểm)
  3. Import ma trận từ file ngoài, so sánh với phương pháp trực tiếp
     (nghịch đảo ma trận), lưu kết quả.                              (2 điểm)

Tác giả: (điền tên bạn)
"""

import sys
import os
import datetime
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFileDialog, QMessageBox, QTextEdit, QSplitter, QHeaderView,
    QTabWidget, QFormLayout
)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# --------------------------------------------------------------------------
# PHẦN LÕI TÍNH TOÁN (tách riêng khỏi GUI để dễ kiểm thử / tái sử dụng)
# --------------------------------------------------------------------------

def gauss_seidel(A: np.ndarray, b: np.ndarray, tol: float = 1e-6,
                  max_iter: int = 500, x0: np.ndarray = None):
    """
    Giải hệ Ax = b bằng phương pháp lặp Gauss-Seidel.

    Trả về:
        x            : nghiệm xấp xỉ cuối cùng (np.ndarray)
        error_hist   : list sai số (chuẩn vô cùng ||x_new - x_old||_inf) theo mỗi vòng lặp
        n_iter       : số vòng lặp thực hiện
        converged    : True nếu hội tụ trước max_iter
    """
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)
    x = np.zeros(n) if x0 is None else x0.astype(float).copy()

    diag = np.abs(np.diag(A))
    if np.any(diag == 0):
        raise ValueError("Phần tử trên đường chéo chính bằng 0, không thể chia. "
                          "Hãy hoán vị các phương trình (pivoting) trước khi giải.")

    error_hist = []
    converged = False
    n_iter = 0
    for k in range(1, max_iter + 1):
        x_new = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])          # đã cập nhật trong vòng lặp này
            s2 = np.dot(A[i, i + 1:], x[i + 1:])        # chưa cập nhật
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        err = np.linalg.norm(x_new - x, ord=np.inf)
        error_hist.append(err)
        x = x_new
        n_iter = k
        if err < tol:
            converged = True
            break

    return x, error_hist, n_iter, converged


def solve_direct(A: np.ndarray, b: np.ndarray):
    """Giải hệ bằng phương pháp trực tiếp (nghịch đảo ma trận)."""
    return np.linalg.solve(A, b)


def is_diagonally_dominant(A: np.ndarray) -> bool:
    """Kiểm tra ma trận có chéo trội hàng hay không (điều kiện đủ để Gauss-Seidel hội tụ)."""
    A = np.abs(A)
    diag = np.diag(A)
    off_diag_sum = A.sum(axis=1) - diag
    return bool(np.all(diag >= off_diag_sum))


# --------------------------------------------------------------------------
# WIDGET: bảng nhập ma trận A | b  (dùng chung cho nhập tay và import file)
# --------------------------------------------------------------------------

class MatrixTable(QTableWidget):
    """Bảng n hàng x (n+1) cột: n cột đầu là ma trận A, cột cuối là vector b."""

    def __init__(self, n=3, parent=None):
        super().__init__(parent)
        self.set_size(n)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def set_size(self, n: int, keep_data=False):
        old_A, old_b = None, None
        if keep_data:
            try:
                old_A, old_b = self.get_matrix()
            except Exception:
                pass
        self.setRowCount(n)
        self.setColumnCount(n + 1)
        headers = [f"A[{i+1}]" for i in range(n)] + ["b"]
        self.setHorizontalHeaderLabels(headers)
        self.setVerticalHeaderLabels([f"PT {i+1}" for i in range(n)])
        for i in range(n):
            for j in range(n + 1):
                val = 0.0
                if old_A is not None and i < old_A.shape[0]:
                    if j < old_A.shape[1] and j < n:
                        val = old_A[i, j] if j < old_A.shape[1] else 0.0
                    elif j == n:
                        val = old_b[i] if i < len(old_b) else 0.0
                item = QTableWidgetItem(f"{val:g}")
                item.setTextAlignment(Qt.AlignCenter)
                self.setItem(i, j, item)

    def fill_from_arrays(self, A: np.ndarray, b: np.ndarray):
        n = A.shape[0]
        self.set_size(n)
        for i in range(n):
            for j in range(n):
                self.setItem(i, j, QTableWidgetItem(f"{A[i, j]:g}"))
            self.setItem(i, n, QTableWidgetItem(f"{b[i]:g}"))
            for j in range(n + 1):
                self.item(i, j).setTextAlignment(Qt.AlignCenter)

    def get_matrix(self):
        n = self.rowCount()
        A = np.zeros((n, n))
        b = np.zeros(n)
        for i in range(n):
            for j in range(n):
                item = self.item(i, j)
                text = item.text().strip() if item else ""
                if text == "":
                    raise ValueError(f"Ô A[{i+1}][{j+1}] đang trống.")
                A[i, j] = float(text)
            item_b = self.item(i, n)
            text_b = item_b.text().strip() if item_b else ""
            if text_b == "":
                raise ValueError(f"Ô b[{i+1}] đang trống.")
            b[i] = float(text_b)
        return A, b


# --------------------------------------------------------------------------
# CỬA SỔ CHÍNH
# --------------------------------------------------------------------------

class GaussSeidelWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Câu 10A – Giải mạch điện nhiều nút bằng Gauss–Seidel")
        self.resize(1180, 720)

        self.last_result = None  # lưu kết quả gần nhất để xuất file

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ---------------- Cột trái: nhập liệu & điều khiển ----------------
        left = QVBoxLayout()

        info = QLabel(
            "<b>Hệ phương trình nút mạch điện: A·V = I</b><br>"
            "Nhập ma trận hệ số A (theo định luật Kirchhoff dòng tại từng nút) "
            "và vector vế phải b, hoặc import từ file."
        )
        info.setWordWrap(True)
        left.addWidget(info)

        size_box = QHBoxLayout()
        size_box.addWidget(QLabel("Số nút (n):"))
        self.spin_n = QSpinBox()
        self.spin_n.setRange(2, 20)
        self.spin_n.setValue(3)
        self.spin_n.valueChanged.connect(self.on_size_changed)
        size_box.addWidget(self.spin_n)
        size_box.addStretch()
        left.addLayout(size_box)

        self.table = MatrixTable(3)
        left.addWidget(self.table, stretch=1)

        # tham số Gauss-Seidel
        params_group = QGroupBox("Tham số Gauss–Seidel")
        params_form = QFormLayout()
        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setDecimals(10)
        self.spin_tol.setRange(1e-10, 1.0)
        self.spin_tol.setValue(1e-6)
        self.spin_tol.setSingleStep(1e-6)
        params_form.addRow("Sai số dừng (tol):", self.spin_tol)

        self.spin_maxit = QSpinBox()
        self.spin_maxit.setRange(1, 100000)
        self.spin_maxit.setValue(500)
        params_form.addRow("Số vòng lặp tối đa:", self.spin_maxit)
        params_group.setLayout(params_form)
        left.addWidget(params_group)

        # nút chức năng
        btn_row1 = QHBoxLayout()
        self.btn_example = QPushButton("Ví dụ mẫu (3 nút)")
        self.btn_example.clicked.connect(self.load_example)
        self.btn_import = QPushButton("Import ma trận từ file…")
        self.btn_import.clicked.connect(self.import_matrix)
        btn_row1.addWidget(self.btn_example)
        btn_row1.addWidget(self.btn_import)
        left.addLayout(btn_row1)

        self.btn_solve = QPushButton("▶ GIẢI HỆ (Gauss–Seidel)")
        self.btn_solve.setStyleSheet("font-weight:bold; padding:8px;")
        self.btn_solve.clicked.connect(self.solve)
        left.addWidget(self.btn_solve)

        btn_row2 = QHBoxLayout()
        self.btn_save_results = QPushButton("Lưu kết quả (.txt)")
        self.btn_save_results.clicked.connect(self.save_results)
        self.btn_save_plot = QPushButton("Lưu đồ thị (.png)")
        self.btn_save_plot.clicked.connect(self.save_plot)
        btn_row2.addWidget(self.btn_save_results)
        btn_row2.addWidget(self.btn_save_plot)
        left.addLayout(btn_row2)

        main_layout.addLayout(left, stretch=5)

        # ---------------- Cột phải: kết quả & đồ thị ----------------
        right = QVBoxLayout()

        self.fig = Figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self._init_plot()
        right.addWidget(self.canvas, stretch=3)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(220)
        self.result_text.setFontFamily("Consolas")
        right.addWidget(self.result_text, stretch=2)

        main_layout.addLayout(right, stretch=6)

    # ---------------------------------------------------------------
    def _init_plot(self):
        self.ax.clear()
        self.ax.set_title("Hội tụ sai số theo số vòng lặp")
        self.ax.set_xlabel("Vòng lặp k")
        self.ax.set_ylabel("Sai số ||x(k) - x(k-1)||∞")
        self.ax.set_yscale("log")
        self.ax.grid(True, which="both", linestyle="--", alpha=0.5)
        self.canvas.draw()

    def on_size_changed(self, n):
        self.table.set_size(n, keep_data=True)

    def load_example(self):
        """
        Ví dụ mạch điện 3 nút (định luật Kirchhoff dòng nút, dẫn nạp G, dòng nguồn I):
            10*V1 - 2*V2 - 3*V3 = 6
            -2*V1 + 8*V2 - 1*V3 = 4
            -3*V1 - 1*V2 + 9*V3 = 5
        Ma trận chéo trội -> Gauss-Seidel hội tụ tốt.
        """
        A = np.array([
            [10.0, -2.0, -3.0],
            [-2.0, 8.0, -1.0],
            [-3.0, -1.0, 9.0],
        ])
        b = np.array([6.0, 4.0, 5.0])
        self.spin_n.blockSignals(True)
        self.spin_n.setValue(3)
        self.spin_n.blockSignals(False)
        self.table.fill_from_arrays(A, b)
        QMessageBox.information(self, "Đã tải ví dụ",
                                 "Đã nạp ví dụ mạch điện 3 nút mẫu (ma trận chéo trội).")

    def import_matrix(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn file ma trận", "",
            "Text/CSV (*.txt *.csv);;Tất cả file (*)"
        )
        if not path:
            return
        try:
            data = np.loadtxt(path, delimiter=",")
        except Exception:
            try:
                data = np.loadtxt(path)
            except Exception as e:
                QMessageBox.critical(self, "Lỗi đọc file",
                                      f"Không đọc được file:\n{e}\n\n"
                                      "Định dạng mong đợi: mỗi hàng là một phương trình, "
                                      "gồm n hệ số + 1 giá trị b (phân tách bằng dấu phẩy hoặc khoảng trắng).")
                return
        if data.ndim != 2 or data.shape[1] != data.shape[0] + 1:
            QMessageBox.critical(self, "Lỗi định dạng",
                                  f"File cần có dạng n hàng x (n+1) cột (A | b).\n"
                                  f"Đã đọc được kích thước: {data.shape}")
            return
        n = data.shape[0]
        A = data[:, :n]
        b = data[:, n]
        self.spin_n.blockSignals(True)
        self.spin_n.setValue(n)
        self.spin_n.blockSignals(False)
        self.table.fill_from_arrays(A, b)
        QMessageBox.information(self, "Import thành công",
                                 f"Đã import ma trận {n}x{n} từ:\n{path}")

    def solve(self):
        try:
            A, b = self.table.get_matrix()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi nhập liệu", str(e))
            return

        tol = self.spin_tol.value()
        max_iter = self.spin_maxit.value()

        dom = is_diagonally_dominant(A)
        try:
            x_gs, err_hist, n_iter, converged = gauss_seidel(A, b, tol=tol, max_iter=max_iter)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi khi giải", str(e))
            return

        try:
            x_direct = solve_direct(A, b)
            diff = np.linalg.norm(x_gs - x_direct, ord=np.inf)
            direct_ok = True
        except np.linalg.LinAlgError:
            x_direct = None
            diff = None
            direct_ok = False

        # vẽ đồ thị hội tụ
        self.ax.clear()
        self.ax.plot(range(1, len(err_hist) + 1), err_hist, marker="o", markersize=3,
                     color="#1f77b4", linewidth=1.5)
        self.ax.set_title("Hội tụ sai số theo số vòng lặp (Gauss–Seidel)")
        self.ax.set_xlabel("Vòng lặp k")
        self.ax.set_ylabel("Sai số ||x(k) - x(k-1)||∞")
        self.ax.set_yscale("log")
        self.ax.grid(True, which="both", linestyle="--", alpha=0.5)
        self.canvas.draw()

        # soạn báo cáo kết quả
        lines = []
        lines.append("=" * 60)
        lines.append("KẾT QUẢ GIẢI HỆ PHƯƠNG TRÌNH NÚT MẠCH ĐIỆN (Gauss–Seidel)")
        lines.append("=" * 60)
        lines.append(f"Số ẩn (số nút): {A.shape[0]}")
        lines.append(f"Ma trận A chéo trội hàng? {'CÓ' if dom else 'KHÔNG (có thể không hội tụ)'}")
        lines.append(f"Sai số dừng (tol): {tol:g}    Số vòng lặp tối đa: {max_iter}")
        lines.append("-" * 60)
        lines.append(f"Hội tụ: {'CÓ' if converged else 'KHÔNG (đạt số vòng lặp tối đa)'}")
        lines.append(f"Số vòng lặp cần thiết: {n_iter}")
        lines.append(f"Sai số cuối cùng: {err_hist[-1]:.3e}")
        lines.append("")
        lines.append("Nghiệm (Gauss-Seidel):")
        for i, v in enumerate(x_gs):
            lines.append(f"   x{i+1} = {v: .8f}")
        lines.append("")
        if direct_ok:
            lines.append("Nghiệm (phương pháp trực tiếp - nghịch đảo ma trận):")
            for i, v in enumerate(x_direct):
                lines.append(f"   x{i+1} = {v: .8f}")
            lines.append("")
            lines.append(f"Sai khác lớn nhất |x_GS - x_direct| = {diff:.3e}")
        else:
            lines.append("Không thể giải bằng phương pháp trực tiếp (ma trận suy biến).")
        lines.append("=" * 60)

        self.result_text.setPlainText("\n".join(lines))

        self.last_result = dict(
            A=A, b=b, x_gs=x_gs, x_direct=x_direct, diff=diff,
            err_hist=err_hist, n_iter=n_iter, converged=converged,
            tol=tol, max_iter=max_iter, dom=dom,
            report="\n".join(lines),
        )

    def save_results(self):
        if self.last_result is None:
            QMessageBox.warning(self, "Chưa có kết quả", "Hãy bấm 'Giải hệ' trước khi lưu.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Lưu kết quả", "ket_qua_gauss_seidel.txt", "Text (*.txt)"
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.last_result["report"])
            f.write(f"\n\nXuất lúc: {datetime.datetime.now().isoformat()}\n")
        QMessageBox.information(self, "Đã lưu", f"Đã lưu kết quả vào:\n{path}")

    def save_plot(self):
        if self.last_result is None:
            QMessageBox.warning(self, "Chưa có đồ thị", "Hãy bấm 'Giải hệ' trước khi lưu ảnh.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Lưu đồ thị", "do_thi_hoi_tu.png", "PNG (*.png)"
        )
        if not path:
            return
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
        QMessageBox.information(self, "Đã lưu", f"Đã lưu đồ thị vào:\n{path}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = GaussSeidelWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
