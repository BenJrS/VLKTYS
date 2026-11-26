# -*- coding: utf-8 -*-
"""
CÂU 10A: Giải hệ phương trình tuyến tính của mạch điện nhiều nút
bằng phương pháp lặp Gauss–Seidel.

Yêu cầu của đề và nơi đáp ứng trong chương trình:
  1. Nhập ma trận hệ số A và vector vế phải b mô tả hệ phương trình nút
     (định luật Kirchhoff), giải bằng Gauss–Seidel.            -> bảng nhập + nút "Giải hệ"
  2. Vẽ đồ thị hội tụ của sai số theo số lần lặp; ghi chú đầy đủ
     nghiệm tìm được và số vòng lặp cần thiết.                   -> đồ thị + khung chú thích
  3. Import ma trận từ file ngoài, so sánh nghiệm với phương pháp
     trực tiếp (nghịch đảo ma trận), lưu lại kết quả.            -> "Mở ma trận từ file", bảng so sánh, "Lưu kết quả"

Mô hình: phân tích điện áp nút  G·V = I
  G (n×n) : ma trận điện dẫn (S), G[i][i] = tổng điện dẫn nối vào nút i, G[i][j] = −điện dẫn giữa nút i và j
  V (n)   : điện áp các nút so với nút gốc (V)  — ẩn cần tìm
  I (n)   : tổng dòng nguồn bơm vào mỗi nút (A)

Tác giả: (điền họ tên – MSSV)
"""

import sys
import datetime
from pathlib import Path

import numpy as np

if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
import theme  # noqa: E402

from PySide6.QtCore import Qt, QLocale, QTimer  # noqa: E402
from PySide6.QtGui import QColor, QPen, QDoubleValidator, QKeySequence, QShortcut  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QSpinBox, QLineEdit,
    QFileDialog, QMessageBox, QHeaderView, QStyledItemDelegate, QAbstractItemView,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


# ==========================================================================
# PHẦN LÕI TÍNH TOÁN (tách riêng khỏi giao diện để dễ kiểm thử)
# ==========================================================================

def gauss_seidel(A, b, tol=1e-6, max_iter=500, x0=None):
    """
    Giải A·x = b bằng phương pháp lặp Gauss–Seidel:
        x_i^(k) = ( b_i − Σ_{j<i} a_ij·x_j^(k) − Σ_{j>i} a_ij·x_j^(k−1) ) / a_ii
    Điều kiện dừng: ‖x^(k) − x^(k−1)‖∞ < tol.

    Trả về dict: x, errors (sai số từng vòng), n_iter, converged, diverged.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    if np.any(np.diag(A) == 0):
        raise ValueError("Có phần tử đường chéo a_ii = 0 nên không chia được.\n"
                         "Hãy đổi thứ tự các phương trình để đường chéo khác 0.")
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()

    errors = []
    converged = diverged = False
    for _ in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s = A[i, :i] @ x[:i] + A[i, i + 1:] @ x_old[i + 1:]
            x[i] = (b[i] - s) / A[i, i]
        err = float(np.max(np.abs(x - x_old)))
        if not np.isfinite(err) or err > 1e100:
            diverged = True
            break
        errors.append(err)
        if err < tol:
            converged = True
            break
    if not converged and len(errors) > 1 and errors[-1] > errors[0]:
        diverged = True          # sai số tăng dần qua các vòng lặp
    return {"x": x, "errors": errors, "n_iter": len(errors),
            "converged": converged, "diverged": diverged}


def solve_inverse(A, b):
    """Phương pháp trực tiếp theo đề: x = A⁻¹·b (nghịch đảo ma trận)."""
    return np.linalg.inv(np.asarray(A, float)) @ np.asarray(b, float)


def row_dominance(A):
    """Mảng bool: hàng i có |a_ii| > Σ_{j≠i} |a_ij| (chéo trội chặt) hay không."""
    M = np.abs(np.asarray(A, float))
    d = np.diag(M)
    return d > (M.sum(axis=1) - d)


def gs_spectral_radius(A):
    """
    Bán kính phổ của ma trận lặp Gauss–Seidel T = −(D+L)⁻¹·U.
    Gauss–Seidel hội tụ với mọi x0  ⇔  ρ(T) < 1.
    """
    A = np.asarray(A, float)
    T = -np.linalg.solve(np.tril(A), np.triu(A, 1))
    return float(np.max(np.abs(np.linalg.eigvals(T))))


def load_matrix_file(path):
    """
    Đọc file ma trận mở rộng [A | b]: n hàng, mỗi hàng n hệ số rồi đến b.
    Chấp nhận dấu phân cách: phẩy, chấm phẩy, tab hoặc khoảng trắng; dòng bắt đầu bằng # là chú thích.
    """
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.replace(";", " ").replace(",", " ").replace("\t", " ").split()
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                raise ValueError(f"Dòng {line_no} có giá trị không phải số:\n{line}")
    if not rows:
        raise ValueError("File không có dữ liệu.")
    n = len(rows)
    if any(len(r) != n + 1 for r in rows):
        lens = sorted({len(r) for r in rows})
        raise ValueError(f"File có {n} hàng nên mỗi hàng cần {n + 1} số (n hệ số + b), "
                         f"nhưng đọc được hàng có {', '.join(map(str, lens))} số.")
    data = np.array(rows)
    return data[:, :n], data[:, n]


def build_report(r):
    A, b = r["A"], r["b"]
    n = len(b)
    L = []
    L.append("KẾT QUẢ GIẢI HỆ PHƯƠNG TRÌNH NÚT MẠCH ĐIỆN BẰNG GAUSS–SEIDEL")
    L.append(f"Thời điểm: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    if r.get("source"):
        L.append(f"Nguồn dữ liệu: {r['source']}")
    L.append("")
    L.append(f"Ma trận mở rộng [A | b]  ({n} nút):")
    for i in range(n):
        L.append("  " + " ".join(f"{v:>11.6g}" for v in A[i]) + "  | " + f"{b[i]:>11.6g}")
    L.append("")
    L.append(f"Sai số dừng tol         : {r['tol']:g}")
    L.append(f"Số vòng lặp tối đa      : {r['max_iter']}")
    L.append(f"Chéo trội hàng (chặt)   : {int(r['dom'].sum())}/{n} hàng")
    L.append(f"Bán kính phổ ρ(T_GS)    : {r['rho']:.6f}  ({'< 1, hội tụ' if r['rho'] < 1 else '≥ 1, không đảm bảo hội tụ'})")
    L.append("")
    gs = r["gs"]
    status = "hội tụ" if gs["converged"] else ("phân kỳ" if gs["diverged"] else "chưa hội tụ (đạt số vòng tối đa)")
    L.append(f"Trạng thái              : {status}")
    L.append(f"Số vòng lặp cần thiết   : {gs['n_iter']}")
    if gs["errors"]:
        L.append(f"Sai số vòng cuối        : {gs['errors'][-1]:.3e}")
    L.append(f"Phần dư ‖A·x − b‖∞      : {r['residual']:.3e}")
    L.append("")
    L.append(f"{'Ẩn':<6}{'Gauss–Seidel':>20}{'Nghịch đảo A⁻¹b':>20}{'|Chênh lệch|':>16}")
    for i in range(n):
        xi = gs["x"][i]
        if r["x_inv"] is not None:
            L.append(f"V{i + 1:<5}{xi:>20.10f}{r['x_inv'][i]:>20.10f}{abs(xi - r['x_inv'][i]):>16.3e}")
        else:
            L.append(f"V{i + 1:<5}{xi:>20.10f}{'—':>20}{'—':>16}")
    if r["x_inv"] is not None:
        L.append(f"Chênh lệch lớn nhất ‖x_GS − A⁻¹b‖∞ = {r['diff']:.3e}")
    else:
        L.append("Ma trận A suy biến: không tính được A⁻¹.")
    L.append("")
    L.append("Lịch sử sai số theo vòng lặp:")
    L.append(f"{'k':>6}  {'‖x(k) − x(k−1)‖∞':>18}")
    for k, e in enumerate(gs["errors"], 1):
        L.append(f"{k:>6}  {e:>18.6e}")
    return "\n".join(L)


def num(x):
    """Số hiển thị trong bảng nghiệm: 10 chữ số thập phân, dạng khoa học nếu quá lớn."""
    return f"{x:.10f}" if abs(x) < 1e6 else f"{x:.6e}"


# ==========================================================================
# BẢNG NHẬP MA TRẬN [A | b]
# ==========================================================================

class AugmentedDelegate(QStyledItemDelegate):
    """Vẽ vạch đứng ngăn cách A và b (giống ký hiệu ma trận mở rộng)."""

    def __init__(self, table):
        super().__init__(table)
        self.table = table

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        if index.column() == self.table.columnCount() - 1:
            painter.save()
            painter.setPen(QPen(QColor(theme.INK), 2))
            r = option.rect
            painter.drawLine(r.left() + 1, r.top(), r.left() + 1, r.bottom() + 1)
            painter.restore()


class MatrixTable(QTableWidget):
    """n hàng × (n+1) cột: n cột đầu là ma trận A, cột cuối là vector b."""

    def __init__(self, n=3):
        super().__init__()
        self.setItemDelegate(AugmentedDelegate(self))
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setDefaultSectionSize(30)
        self.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.setSelectionMode(QAbstractItemView.ContiguousSelection)
        self.setFont(theme.mono_font(10.5))
        self.set_size(n)

    def _item(self, val):
        it = QTableWidgetItem("" if val is None else f"{val:.12g}")
        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return it

    def set_size(self, n, keep=True):
        old = self.raw_values() if keep and self.rowCount() else None
        old_n = self.rowCount()
        self.blockSignals(True)
        self.setRowCount(n)
        self.setColumnCount(n + 1)
        self.setHorizontalHeaderLabels([f"V{j + 1}" for j in range(n)] + ["b (I)"])
        self.setVerticalHeaderLabels([f"Nút {i + 1}" for i in range(n)])
        for i in range(n):
            for j in range(n + 1):
                val = 0.0
                if old is not None and i < old_n:
                    if j < n and j < old_n:
                        val = old[i][j]
                    elif j == n:
                        val = old[i][old_n]
                self.setItem(i, j, self._item(val))
        self.blockSignals(False)
        self.itemChanged.emit(self.item(0, 0))

    def fill(self, A, b):
        n = len(b)
        self.set_size(n, keep=False)
        self.blockSignals(True)
        for i in range(n):
            for j in range(n):
                self.setItem(i, j, self._item(A[i, j]))
            self.setItem(i, n, self._item(b[i]))
        self.blockSignals(False)
        self.itemChanged.emit(self.item(0, 0))

    def raw_values(self):
        """Giá trị hiện có (None với ô không đọc được) — dùng khi đổi kích thước."""
        out = []
        for i in range(self.rowCount()):
            row = []
            for j in range(self.columnCount()):
                try:
                    row.append(self._parse(i, j))
                except ValueError:
                    row.append(0.0)
            out.append(row)
        return out

    def _parse(self, i, j):
        it = self.item(i, j)
        text = (it.text() if it else "").strip().replace(",", ".")
        if not text:
            raise ValueError("trống")
        return float(text)

    def get_system(self):
        n = self.rowCount()
        A = np.zeros((n, n))
        b = np.zeros(n)
        for i in range(n):
            for j in range(n + 1):
                try:
                    v = self._parse(i, j)
                except ValueError:
                    where = f"b[{i + 1}]" if j == n else f"A[{i + 1}][{j + 1}]"
                    raise ValueError(f"Ô {where} đang trống hoặc không phải số. "
                                     "Hãy nhập một số thực, ví dụ −2.5 hoặc 1e-3.")
                if j == n:
                    b[i] = v
                else:
                    A[i, j] = v
        return A, b

    def mark_diagonal(self, dominant):
        """Tô màu ô đường chéo: xanh nếu hàng chéo trội chặt, đỏ nhạt nếu không."""
        acc = theme.ACCENTS["circuit"]
        self.blockSignals(True)
        for i in range(self.rowCount()):
            it = self.item(i, i)
            if it is None:
                continue
            if dominant is None:
                it.setBackground(QColor(theme.SURFACE))
                it.setForeground(QColor(theme.INK))
            elif dominant[i]:
                it.setBackground(QColor(acc["soft"]))
                it.setForeground(QColor(acc["main"]))
            else:
                it.setBackground(QColor(theme.WARN_SOFT))
                it.setForeground(QColor(theme.WARN))
        self.blockSignals(False)


# ==========================================================================
# CỬA SỔ CHÍNH
# ==========================================================================

EXAMPLE_A = np.array([[10.0, -2.0, -3.0],
                      [-2.0, 8.0, -1.0],
                      [-3.0, -1.0, 9.0]])
EXAMPLE_B = np.array([6.0, 4.0, 5.0])


class GaussSeidelWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Câu 10A — Phân tích nút mạch điện bằng Gauss–Seidel")
        self.resize(1320, 820)
        self.last = None
        self.source = None

        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(20, 16, 20, 20)
        outer.setSpacing(14)

        self.btn_example = QPushButton("Nạp ví dụ 3 nút")
        self.btn_example.clicked.connect(self.load_example)
        self.btn_import = QPushButton("Mở ma trận từ file…")
        self.btn_import.setToolTip("File .csv hoặc .txt: mỗi hàng gồm n hệ số rồi đến vế phải b,\n"
                                   "phân tách bằng dấu phẩy, chấm phẩy hoặc khoảng trắng.")
        self.btn_import.clicked.connect(self.import_matrix)
        self.btn_save_results = QPushButton("Lưu kết quả…")
        self.btn_save_results.clicked.connect(self.save_results)
        self.btn_save_plot = QPushButton("Lưu đồ thị…")
        self.btn_save_plot.clicked.connect(self.save_plot)
        for btn in (self.btn_save_results, self.btn_save_plot):
            btn.setEnabled(False)

        outer.addWidget(theme.header(
            "Phân tích nút mạch điện",
            "Giải hệ G·V = I bằng phương pháp lặp Gauss–Seidel và đối chiếu với nghịch đảo ma trận",
            [self.btn_example, self.btn_import, None, self.btn_save_results, self.btn_save_plot],
        ))

        body = QHBoxLayout()
        body.setSpacing(14)
        outer.addLayout(body, 1)
        body.addWidget(self._build_input_panel(), 5)

        right = QVBoxLayout()
        right.setSpacing(14)
        right.addWidget(self._build_plot_panel(), 3)
        right.addWidget(self._build_result_panel(), 2)
        body.addLayout(right, 7)

        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.solve)
        QShortcut(QKeySequence.Open, self, activated=self.import_matrix)
        QShortcut(QKeySequence.Save, self, activated=self.save_results)

        self.load_example()
        self.update_diagnostics()
        self.table.setFocus()

    # ------------------------------------------------------------- dựng giao diện
    def _build_input_panel(self):
        frame, lay = theme.panel()
        lay.addWidget(theme.label("Hệ phương trình nút", "Section"))
        lay.addWidget(theme.label(
            "Mỗi hàng là định luật Kirchhoff về dòng tại một nút: tổng G·V bằng dòng nguồn I bơm vào nút. "
            "Ô đường chéo hiện màu xanh khi hàng đó chéo trội.", "Hint", wrap=True))

        params = QGridLayout()
        params.setHorizontalSpacing(10)
        params.setVerticalSpacing(4)
        self.spin_n = QSpinBox()
        self.spin_n.setRange(2, 30)
        self.spin_n.setValue(3)
        self.spin_n.valueChanged.connect(lambda n: self.table.set_size(n))
        self.edit_tol = QLineEdit("1e-6")
        v = QDoubleValidator(1e-15, 1.0, 15, self)
        v.setNotation(QDoubleValidator.ScientificNotation)
        v.setLocale(QLocale.c())
        self.edit_tol.setValidator(v)
        self.spin_maxit = QSpinBox()
        self.spin_maxit.setRange(1, 100000)
        self.spin_maxit.setValue(500)
        for col, (name, w) in enumerate([("Số nút n", self.spin_n),
                                         ("Sai số dừng tol", self.edit_tol),
                                         ("Số vòng lặp tối đa", self.spin_maxit)]):
            params.addWidget(theme.label(name, "FactName"), 0, col)
            params.addWidget(w, 1, col)
        lay.addLayout(params)

        self.table = MatrixTable(3)
        self.table.itemChanged.connect(lambda *_: self._diag_timer.start())
        lay.addWidget(self.table, 1)

        self._diag_timer = QTimer(self, singleShot=True, interval=120)
        self._diag_timer.timeout.connect(self.update_diagnostics)

        diag = QHBoxLayout()
        self.lbl_dom = theme.label("", "Badge")
        self.lbl_rho = theme.label("", "Badge")
        diag.addWidget(self.lbl_dom)
        diag.addWidget(self.lbl_rho)
        diag.addStretch()
        lay.addLayout(diag)
        self.lbl_source = theme.label("", "Status", wrap=True)
        lay.addWidget(self.lbl_source)

        self.btn_solve = QPushButton("Giải hệ")
        self.btn_solve.setObjectName("Primary")
        self.btn_solve.setToolTip("Ctrl+Enter")
        self.btn_solve.clicked.connect(self.solve)
        lay.addWidget(self.btn_solve)
        return frame

    def _build_plot_panel(self):
        frame, lay = theme.panel()
        self.fig = Figure(figsize=(7, 4))
        self.canvas = FigureCanvas(self.fig)
        lay.addWidget(self.canvas)
        self._draw_empty()
        return frame

    def _build_result_panel(self):
        frame, lay = theme.panel()
        top = QHBoxLayout()
        top.addWidget(theme.label("Nghiệm và so sánh với A⁻¹b", "Section"))
        top.addStretch()
        self.lbl_status = theme.label("Chưa giải", "Badge")
        theme.set_badge(self.lbl_status, "Chưa giải", "idle")
        top.addWidget(self.lbl_status)
        lay.addLayout(top)

        row = QHBoxLayout()
        row.setSpacing(18)
        facts = QGridLayout()
        facts.setVerticalSpacing(6)
        facts.setHorizontalSpacing(14)
        self.facts = {}
        for r, (key, name) in enumerate([("iter", "Số vòng lặp"),
                                         ("err", "Sai số vòng cuối"),
                                         ("res", "Phần dư ‖A·x − b‖∞"),
                                         ("diff", "Chênh lệch với A⁻¹b")]):
            facts.addWidget(theme.label(name, "FactName"), r, 0)
            val = theme.label("—", "Fact")
            facts.addWidget(val, r, 1)
            self.facts[key] = val
        facts.setRowStretch(4, 1)
        row.addLayout(facts, 0)

        self.result_table = QTableWidget(0, 4)
        self.result_table.setHorizontalHeaderLabels(["Ẩn", "Gauss–Seidel", "Nghịch đảo A⁻¹b", "|Chênh lệch|"])
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_table.setFont(theme.mono_font(10.5))
        row.addWidget(self.result_table, 1)
        lay.addLayout(row, 1)
        return frame

    # ------------------------------------------------------------- đồ thị
    def _draw_empty(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_title("Hội tụ của sai số theo số vòng lặp")
        ax.set_xlabel("Vòng lặp k")
        ax.set_ylabel("‖x(k) − x(k−1)‖∞")
        ax.text(0.5, 0.5, "Nhấn \"Giải hệ\" để vẽ đồ thị hội tụ", transform=ax.transAxes,
                ha="center", va="center", color=theme.MUTED)
        ax.set_xticks([])
        ax.set_yticks([])
        self.fig.tight_layout()
        self.canvas.draw()

    def _draw_convergence(self, r):
        gs = r["gs"]
        errs = np.array(gs["errors"])
        k = np.arange(1, len(errs) + 1)
        acc = theme.ACCENTS["circuit"]["main"]
        n = len(r["b"])

        self.fig.clear()
        gsp = self.fig.add_gridspec(1, 2, width_ratios=[3.2, 1.25 if n <= 12 else 2.1], wspace=0.04)
        ax = self.fig.add_subplot(gsp[0])
        info = self.fig.add_subplot(gsp[1])
        info.axis("off")

        if len(errs):
            positive = np.where(errs > 0, errs, np.nan)
            ax.semilogy(k, positive, color=acc, lw=1.8, marker="o", ms=3.2, zorder=3)
            ax.axhline(r["tol"], color=theme.MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
            ax.annotate(f"tol = {r['tol']:g}", xy=(0, r["tol"]), xycoords=("axes fraction", "data"),
                        xytext=(4, 4), textcoords="offset points", ha="left", va="bottom",
                        color=theme.MUTED, fontsize=8.5)
            if np.isfinite(positive[-1]):
                ax.plot(k[-1], positive[-1], "o", ms=7, mfc="white", mec=acc, mew=1.8, zorder=4)
                ax.annotate(f"k = {gs['n_iter']}", xy=(k[-1], positive[-1]), xytext=(-9, 7),
                            textcoords="offset points", ha="right", va="bottom", color=acc,
                            fontsize=9, fontweight="medium")
        ax.set_title("Hội tụ của sai số Gauss–Seidel theo số vòng lặp")
        ax.set_xlabel("Vòng lặp k")
        ax.set_ylabel("Sai số ‖x(k) − x(k−1)‖∞ (thang log)")
        ax.margins(x=0.04)
        ax.grid(True, which="major")
        if len(k) <= 25:
            ax.set_xticks(k if len(k) <= 12 else k[::2])

        # Khung chú thích: nghiệm + số vòng lặp (đưa vào chính hình để ảnh lưu ra có đủ thông tin)
        if gs["converged"]:
            head = f"Hội tụ sau {gs['n_iter']} vòng lặp"
        elif gs["diverged"]:
            head = f"Phân kỳ sau {gs['n_iter']} vòng lặp"
        else:
            head = f"Chưa hội tụ sau {gs['n_iter']} vòng"
        info.text(0.06, 1.0, head, transform=info.transAxes, va="top", fontsize=10,
                  fontweight="semibold", color=acc if gs["converged"] else theme.WARN)
        info.text(0.06, 0.92, f"tol = {r['tol']:g},  ρ(T) = {r['rho']:.3f}", transform=info.transAxes,
                  va="top", fontsize=8.5, color=theme.MUTED)
        cols = 1 if n <= 12 else 2
        per_col = int(np.ceil(n / cols))
        step = min(0.065, 0.78 / max(per_col, 1))
        for i, xi in enumerate(gs["x"]):
            c, rr = divmod(i, per_col)
            info.text(0.06 + c * 0.5, 0.80 - rr * step, f"V{i + 1} = {xi:.6g}",
                      transform=info.transAxes, va="top", fontsize=9 if cols == 1 else 8.2,
                      color=theme.INK, family=[theme.SANS, "DejaVu Sans"])
        info.text(0.06, 0.80 - per_col * step - 0.02, "đơn vị: vôn (V)",
                  transform=info.transAxes, va="top", fontsize=7.5, color=theme.MUTED)

        self.fig.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.13)
        self.canvas.draw()

    # ------------------------------------------------------------- hành động
    def update_diagnostics(self):
        try:
            A, _ = self.table.get_system()
        except ValueError:
            self.table.mark_diagonal(None)
            theme.set_badge(self.lbl_dom, "Ma trận chưa đầy đủ", "idle")
            theme.set_badge(self.lbl_rho, "ρ(T) —", "idle")
            return
        dom = row_dominance(A)
        self.table.mark_diagonal(dom)
        theme.set_badge(self.lbl_dom, f"Chéo trội {int(dom.sum())}/{len(dom)} hàng",
                        "ok" if dom.all() else "bad")
        try:
            rho = gs_spectral_radius(A)
            if rho < 1:
                theme.set_badge(self.lbl_rho, f"ρ(T) = {rho:.3f} < 1: sẽ hội tụ", "ok")
            else:
                theme.set_badge(self.lbl_rho, f"ρ(T) = {rho:.3f} ≥ 1: không hội tụ", "bad")
        except np.linalg.LinAlgError:
            theme.set_badge(self.lbl_rho, "ρ(T) —  đường chéo có số 0", "bad")

    def load_example(self):
        self.spin_n.blockSignals(True)
        self.spin_n.setValue(3)
        self.spin_n.blockSignals(False)
        self.table.fill(EXAMPLE_A, EXAMPLE_B)
        self.source = "Ví dụ mạch 3 nút có sẵn"
        self.lbl_source.setText("Đang dùng ví dụ mạch 3 nút có sẵn.")

    def import_matrix(self):
        path, _ = QFileDialog.getOpenFileName(self, "Mở ma trận [A | b]", "",
                                              "Ma trận (*.csv *.txt);;Tất cả file (*)")
        if not path:
            return
        try:
            A, b = load_matrix_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Không đọc được file", str(e))
            return
        n = len(b)
        if n > self.spin_n.maximum():
            QMessageBox.critical(self, "Ma trận quá lớn",
                                 f"File có {n} nút; chương trình hỗ trợ tối đa {self.spin_n.maximum()} nút.")
            return
        self.spin_n.blockSignals(True)
        self.spin_n.setValue(n)
        self.spin_n.blockSignals(False)
        self.table.fill(A, b)
        self.source = path
        self.lbl_source.setText(f"Đã nạp {Path(path).name}: {n} nút. Nhấn \"Giải hệ\" để giải và so sánh.")

    def solve(self):
        try:
            A, b = self.table.get_system()
        except ValueError as e:
            QMessageBox.warning(self, "Thiếu dữ liệu", str(e))
            return
        try:
            tol = float(self.edit_tol.text().replace(",", "."))
            if not (0 < tol < 1):
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Sai số dừng không hợp lệ",
                                "Nhập tol là số dương nhỏ hơn 1, ví dụ 1e-6.")
            return
        max_iter = self.spin_maxit.value()

        try:
            gs = gauss_seidel(A, b, tol=tol, max_iter=max_iter)
        except ValueError as e:
            QMessageBox.critical(self, "Không giải được", str(e))
            return
        try:
            x_inv = solve_inverse(A, b)
            diff = float(np.max(np.abs(gs["x"] - x_inv)))
        except np.linalg.LinAlgError:
            x_inv, diff = None, None
        try:
            rho = gs_spectral_radius(A)
        except np.linalg.LinAlgError:
            rho = float("nan")

        r = dict(A=A, b=b, tol=tol, max_iter=max_iter, gs=gs, x_inv=x_inv, diff=diff,
                 rho=rho, dom=row_dominance(A), source=self.source,
                 residual=float(np.max(np.abs(A @ gs["x"] - b))))
        self.last = r
        self._draw_convergence(r)
        self._show_results(r)
        self.btn_save_results.setEnabled(True)
        self.btn_save_plot.setEnabled(True)

    def _show_results(self, r):
        gs = r["gs"]
        if gs["converged"]:
            theme.set_badge(self.lbl_status, f"Hội tụ sau {gs['n_iter']} vòng", "ok")
        elif gs["diverged"]:
            theme.set_badge(self.lbl_status, "Phân kỳ", "bad")
        else:
            theme.set_badge(self.lbl_status, f"Chưa hội tụ sau {gs['n_iter']} vòng", "bad")
        self.facts["iter"].setText(str(gs["n_iter"]))
        self.facts["err"].setText(f"{gs['errors'][-1]:.3e}" if gs["errors"] else "—")
        self.facts["res"].setText(f"{r['residual']:.3e}")
        self.facts["diff"].setText(f"{r['diff']:.3e}" if r["diff"] is not None else "A suy biến")

        n = len(r["b"])
        t = self.result_table
        t.setRowCount(n)
        for i in range(n):
            vals = [f"V{i + 1}", num(gs["x"][i])]
            if r["x_inv"] is not None:
                vals += [num(r["x_inv"][i]), f"{abs(gs['x'][i] - r['x_inv'][i]):.2e}"]
            else:
                vals += ["—", "—"]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setTextAlignment((Qt.AlignLeft if j == 0 else Qt.AlignRight) | Qt.AlignVCenter)
                t.setItem(i, j, it)

    def save_results(self):
        if self.last is None:
            return
        path, flt = QFileDialog.getSaveFileName(
            self, "Lưu kết quả", "ket_qua_gauss_seidel.txt",
            "Báo cáo văn bản (*.txt);;Bảng nghiệm CSV (*.csv)")
        if not path:
            return
        r = self.last
        try:
            if path.lower().endswith(".csv"):
                with open(path, "w", encoding="utf-8-sig") as f:
                    f.write("an,gauss_seidel,nghich_dao,chenh_lech\n")
                    for i, xi in enumerate(r["gs"]["x"]):
                        xv = r["x_inv"][i] if r["x_inv"] is not None else float("nan")
                        f.write(f"V{i + 1},{xi:.12g},{xv:.12g},{abs(xi - xv):.6g}\n")
                    f.write(f"\nso_vong_lap,{r['gs']['n_iter']}\nhoi_tu,{r['gs']['converged']}\n")
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(build_report(r))
        except OSError as e:
            QMessageBox.critical(self, "Không lưu được", str(e))
            return
        self.lbl_source.setText(f"Đã lưu kết quả: {path}")

    def save_plot(self):
        if self.last is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Lưu đồ thị", "do_thi_hoi_tu.png",
                                              "Ảnh PNG (*.png);;Ảnh vector SVG (*.svg);;PDF (*.pdf)")
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=200)
        except OSError as e:
            QMessageBox.critical(self, "Không lưu được", str(e))
            return
        self.lbl_source.setText(f"Đã lưu đồ thị: {path}")


def main():
    app = QApplication(sys.argv)
    theme.apply(app, "circuit")
    win = GaussSeidelWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
