# -*- coding: utf-8 -*-
"""
CÂU 6B: Tính tuổi mẫu vật và hằng số phân rã phóng xạ
bằng phương pháp mô phỏng Monte-Carlo.

Yêu cầu:
  1. Đọc dữ liệu/phương trình đã lưu sẵn, hiển thị đồ thị lên GUI.
     Tính tuổi mẫu vật, hằng số phân rã, chu kỳ bán rã bằng Monte-Carlo. (4 điểm)
  2. Cho phép người dùng nhập phương trình bất kỳ để tính toán.          (3 điểm)
     (+ khuyến khích: hiển thị khoảng tin cậy của kết quả ước lượng)
  3. Cho phép lưu lại hình ảnh đã vẽ và kết quả đã tính.                 (3 điểm)
     (+ khuyến khích: đóng gói ứng dụng chạy độc lập)

Nguyên lý vật lý (mô hình mặc định - phân rã phóng xạ bậc 1):
    N(t) = N0 * exp(-λt)   =>   t = ln(N0/N) / λ
    λ = ln(2) / T_1/2
Mỗi đại lượng đo được (N0, N, T_1/2, ...) có sai số đo (độ lệch chuẩn).
Monte-Carlo: lấy mẫu ngẫu nhiên nhiều lần theo phân bố của từng đại lượng đầu vào,
tính ra phân bố của đại lượng cần tìm (tuổi t, hằng số phân rã λ...),
từ đó suy ra giá trị trung bình, độ lệch chuẩn và khoảng tin cậy (CI).

Tác giả: (điền tên bạn)
"""

import sys
import os
import json
import datetime
import numpy as np
import sympy as sp

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, QSpinBox,
    QGroupBox, QFileDialog, QMessageBox, QTextEdit, QHeaderView,
    QTabWidget, QFormLayout, QLineEdit, QDoubleSpinBox, QComboBox,
    QSizePolicy
)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ==========================================================================
# PHẦN LÕI TÍNH TOÁN MONTE-CARLO (tách riêng để dễ kiểm thử)
# ==========================================================================

N_TRIALS_DEFAULT = 100_000


def sample_variable(mean: float, std: float, dist: str, n: int, rng: np.random.Generator):
    """Lấy n mẫu ngẫu nhiên cho một biến đầu vào theo phân bố chọn trước."""
    if std <= 0:
        return np.full(n, mean)
    if dist == "normal":
        return rng.normal(loc=mean, scale=std, size=n)
    elif dist == "uniform":
        half = std * np.sqrt(3)  # std cho phân bố đều [-half,half] quanh mean
        return rng.uniform(low=mean - half, high=mean + half, size=n)
    else:
        raise ValueError(f"Phân bố không hỗ trợ: {dist}")


def monte_carlo_standard(N0_mean, N0_std, N_mean, N_std, Thalf_mean, Thalf_std,
                          n_trials=N_TRIALS_DEFAULT, seed=None):
    """
    Mô phỏng Monte-Carlo chuẩn cho bài toán định tuổi phóng xạ.
    Đầu vào: N0 (số hạt nhân ban đầu), N (số hạt nhân còn lại đo được),
             T_1/2 (chu kỳ bán rã tham khảo) - mỗi đại lượng có (mean, std).
    Đầu ra: mảng mẫu Monte-Carlo cho t (tuổi), lambda (hằng số phân rã), T_1/2.
    """
    rng = np.random.default_rng(seed)
    N0 = sample_variable(N0_mean, N0_std, "normal", n_trials, rng)
    N = sample_variable(N_mean, N_std, "normal", n_trials, rng)
    Thalf = sample_variable(Thalf_mean, Thalf_std, "normal", n_trials, rng)

    # loại các mẫu vật lý phi lý (N<=0, N0<=0, N>=N0, Thalf<=0) để tránh log âm/chia 0
    valid = (N0 > 0) & (N > 0) & (N < N0) & (Thalf > 0)

    lam = np.full(n_trials, np.nan)
    t = np.full(n_trials, np.nan)
    lam[valid] = np.log(2.0) / Thalf[valid]
    t[valid] = np.log(N0[valid] / N[valid]) / lam[valid]

    n_invalid = n_trials - np.count_nonzero(valid)
    return {
        "t": t[valid],
        "lambda": lam[valid],
        "T_half": Thalf[valid],
        "n_invalid": n_invalid,
        "n_trials": n_trials,
    }


def summarize(samples: np.ndarray, ci=95):
    """Trả về (mean, std, ci_low, ci_high) cho một mảng kết quả Monte-Carlo."""
    samples = samples[~np.isnan(samples)]
    if len(samples) == 0:
        raise ValueError("Không có mẫu hợp lệ nào (kiểm tra lại tham số đầu vào).")
    mean = np.mean(samples)
    std = np.std(samples, ddof=1)
    alpha = (100 - ci) / 2
    lo = np.percentile(samples, alpha)
    hi = np.percentile(samples, 100 - alpha)
    return mean, std, lo, hi


ALLOWED_FUNCS = {
    "ln": sp.log, "log": sp.log, "log10": lambda x: sp.log(x, 10),
    "exp": sp.exp, "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "pi": sp.pi, "E": sp.E,
}


def monte_carlo_custom(formula: str, variables: dict, n_trials=N_TRIALS_DEFAULT, seed=None):
    """
    Mô phỏng Monte-Carlo cho một phương trình bất kỳ do người dùng nhập.

    formula   : chuỗi biểu thức, ví dụ "ln(N0/N)/lam"  (không cần vế trái)
    variables : dict {ten_bien: (mean, std, dist)}
    Trả về mảng kết quả Monte-Carlo (đã loại NaN/inf).
    """
    if not variables:
        raise ValueError("Chưa khai báo biến nào.")
    # Khai báo tường minh Symbol cho từng biến người dùng đặt tên, để tránh xung đột
    # với các ký hiệu dựng sẵn của sympy (vd: N, E, I, S, O là hàm/hằng số có sẵn).
    local_dict = dict(ALLOWED_FUNCS)
    for name in variables:
        local_dict[name] = sp.Symbol(name)

    expr = sp.sympify(formula, locals=local_dict)
    free_syms = sorted(expr.free_symbols, key=lambda s: s.name)
    missing = [s.name for s in free_syms if s.name not in variables]
    if missing:
        raise ValueError(f"Thiếu giá trị cho biến: {', '.join(missing)}")

    f = sp.lambdify([s.name for s in free_syms], expr, modules=["numpy"])

    rng = np.random.default_rng(seed)
    sampled = {}
    for s in free_syms:
        mean, std, dist = variables[s.name]
        sampled[s.name] = sample_variable(mean, std, dist, n_trials, rng)

    args = [sampled[s.name] for s in free_syms]
    with np.errstate(all="ignore"):
        result = f(*args)
    result = np.asarray(result, dtype=float)
    result = result[np.isfinite(result)]
    return result, [s.name for s in free_syms]


# ==========================================================================
# TAB 1: MÔ PHỎNG CHUẨN (tuổi mẫu vật / hằng số phân rã / chu kỳ bán rã)
# ==========================================================================

class StandardTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.last_data = None  # {'t':.., 'lambda':.., 'T_half':.., ...}

        layout = QHBoxLayout(self)

        # ---- cột trái: nhập liệu ----
        left = QVBoxLayout()
        desc = QLabel(
            "<b>Mô hình phân rã bậc 1:</b> N(t) = N₀·e<sup>-λt</sup><br>"
            "t = ln(N₀/N) / λ,&nbsp;&nbsp; λ = ln(2) / T<sub>1/2</sub><br>"
            "Nhập giá trị đo được và sai số (độ lệch chuẩn) cho từng đại lượng."
        )
        desc.setWordWrap(True)
        left.addWidget(desc)

        form_group = QGroupBox("Dữ liệu đo được")
        form = QFormLayout()

        self.N0_mean = QDoubleSpinBox(); self._cfg(self.N0_mean, 1000, 0, 1e12)
        self.N0_std = QDoubleSpinBox(); self._cfg(self.N0_std, 20, 0, 1e12)
        form.addRow("N₀ (số hạt nhân ban đầu) — trung bình:", self.N0_mean)
        form.addRow("N₀ — độ lệch chuẩn (σ):", self.N0_std)

        self.N_mean = QDoubleSpinBox(); self._cfg(self.N_mean, 250, 0, 1e12)
        self.N_std = QDoubleSpinBox(); self._cfg(self.N_std, 10, 0, 1e12)
        form.addRow("N (số hạt nhân đo hiện tại) — trung bình:", self.N_mean)
        form.addRow("N — độ lệch chuẩn (σ):", self.N_std)

        self.Th_mean = QDoubleSpinBox(); self._cfg(self.Th_mean, 5730, 0, 1e12)
        self.Th_std = QDoubleSpinBox(); self._cfg(self.Th_std, 40, 0, 1e12)
        form.addRow("T₁/₂ (chu kỳ bán rã, năm) — trung bình:", self.Th_mean)
        form.addRow("T₁/₂ — độ lệch chuẩn (σ):", self.Th_std)

        form_group.setLayout(form)
        left.addWidget(form_group)

        sim_group = QGroupBox("Mô phỏng")
        sim_form = QFormLayout()
        self.spin_trials = QSpinBox()
        self.spin_trials.setRange(1000, 5_000_000)
        self.spin_trials.setSingleStep(10000)
        self.spin_trials.setValue(N_TRIALS_DEFAULT)
        sim_form.addRow("Số lần thử Monte-Carlo:", self.spin_trials)
        self.combo_target = QComboBox()
        self.combo_target.addItems(["Tuổi mẫu vật (t)", "Hằng số phân rã (λ)", "Chu kỳ bán rã (T₁/₂)"])
        sim_form.addRow("Hiển thị histogram cho:", self.combo_target)
        sim_group.setLayout(sim_form)
        left.addWidget(sim_group)

        btn_row = QHBoxLayout()
        self.btn_example = QPushButton("Ví dụ mẫu (C-14)")
        self.btn_example.clicked.connect(self.load_example)
        self.btn_load = QPushButton("Đọc dữ liệu đã lưu…")
        self.btn_load.clicked.connect(self.load_config)
        self.btn_save_cfg = QPushButton("Lưu dữ liệu…")
        self.btn_save_cfg.clicked.connect(self.save_config)
        btn_row.addWidget(self.btn_example)
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_save_cfg)
        left.addLayout(btn_row)

        self.btn_run = QPushButton("▶ CHẠY MÔ PHỎNG MONTE-CARLO")
        self.btn_run.setStyleSheet("font-weight:bold; padding:8px;")
        self.btn_run.clicked.connect(self.run_simulation)
        left.addWidget(self.btn_run)

        self.combo_target.currentIndexChanged.connect(self.refresh_plot_only)

        left.addStretch()
        layout.addLayout(left, stretch=5)

        # ---- cột phải: đồ thị + kết quả ----
        right = QVBoxLayout()
        self.fig = Figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        right.addWidget(self.canvas, stretch=3)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFontFamily("Consolas")
        self.result_text.setMinimumHeight(200)
        right.addWidget(self.result_text, stretch=2)

        layout.addLayout(right, stretch=6)

    @staticmethod
    def _cfg(spin, val, lo, hi):
        spin.setRange(lo, hi)
        spin.setDecimals(4)
        spin.setValue(val)

    def load_example(self):
        """Ví dụ định tuổi bằng carbon phóng xạ C-14 (T1/2 ~ 5730 năm)."""
        self.N0_mean.setValue(1000); self.N0_std.setValue(20)
        self.N_mean.setValue(250); self.N_std.setValue(10)
        self.Th_mean.setValue(5730); self.Th_std.setValue(40)
        QMessageBox.information(self, "Đã tải ví dụ",
                                 "Đã nạp ví dụ định tuổi bằng đồng vị Carbon-14 "
                                 "(T₁/₂ ≈ 5730 năm).")

    def get_config(self):
        return {
            "N0_mean": self.N0_mean.value(), "N0_std": self.N0_std.value(),
            "N_mean": self.N_mean.value(), "N_std": self.N_std.value(),
            "Th_mean": self.Th_mean.value(), "Th_std": self.Th_std.value(),
            "n_trials": self.spin_trials.value(),
        }

    def set_config(self, cfg):
        self.N0_mean.setValue(cfg.get("N0_mean", 1000))
        self.N0_std.setValue(cfg.get("N0_std", 20))
        self.N_mean.setValue(cfg.get("N_mean", 250))
        self.N_std.setValue(cfg.get("N_std", 10))
        self.Th_mean.setValue(cfg.get("Th_mean", 5730))
        self.Th_std.setValue(cfg.get("Th_std", 40))
        self.spin_trials.setValue(cfg.get("n_trials", N_TRIALS_DEFAULT))

    def save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Lưu dữ liệu/phương trình",
                                               "du_lieu_monte_carlo.json", "JSON (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"mode": "standard", **self.get_config()}, f, ensure_ascii=False, indent=2)
        QMessageBox.information(self, "Đã lưu", f"Đã lưu dữ liệu vào:\n{path}")

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Đọc dữ liệu đã lưu", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.set_config(cfg)
            QMessageBox.information(self, "Đã đọc dữ liệu", f"Đã nạp dữ liệu từ:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi đọc file", str(e))

    def run_simulation(self):
        cfg = self.get_config()
        try:
            data = monte_carlo_standard(
                cfg["N0_mean"], cfg["N0_std"], cfg["N_mean"], cfg["N_std"],
                cfg["Th_mean"], cfg["Th_std"], n_trials=cfg["n_trials"]
            )
        except Exception as e:
            QMessageBox.critical(self, "Lỗi mô phỏng", str(e))
            return

        self.last_data = data
        self.refresh_plot_only()

    def _target_key(self):
        idx = self.combo_target.currentIndex()
        return ["t", "lambda", "T_half"][idx]

    def _target_label(self):
        idx = self.combo_target.currentIndex()
        return ["Tuổi mẫu vật t (năm)", "Hằng số phân rã λ (1/năm)", "Chu kỳ bán rã T₁/₂ (năm)"][idx]

    def refresh_plot_only(self):
        if self.last_data is None:
            return
        data = self.last_data
        key = self._target_key()
        samples = data[key]

        mean, std, lo, hi = summarize(samples, ci=95)

        self.ax.clear()
        self.ax.hist(samples, bins=80, color="#4C72B0", alpha=0.85, density=True)
        self.ax.axvline(mean, color="red", linestyle="-", linewidth=1.5, label=f"Trung bình = {mean:.4g}")
        self.ax.axvline(lo, color="green", linestyle="--", linewidth=1.2, label=f"CI 95%: [{lo:.4g}, {hi:.4g}]")
        self.ax.axvline(hi, color="green", linestyle="--", linewidth=1.2)
        self.ax.set_title(f"Phân bố Monte-Carlo: {self._target_label()}")
        self.ax.set_xlabel(self._target_label())
        self.ax.set_ylabel("Mật độ xác suất")
        self.ax.legend(fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()

        # tổng hợp cả 3 đại lượng để in ra báo cáo
        m_t, s_t, lo_t, hi_t = summarize(data["t"])
        m_l, s_l, lo_l, hi_l = summarize(data["lambda"])
        m_T, s_T, lo_T, hi_T = summarize(data["T_half"])

        lines = []
        lines.append("=" * 62)
        lines.append("KẾT QUẢ MÔ PHỎNG MONTE-CARLO (Định tuổi phóng xạ)")
        lines.append("=" * 62)
        lines.append(f"Số lần thử: {data['n_trials']:,}   |   Số mẫu không hợp lệ (loại bỏ): {data['n_invalid']:,}")
        lines.append("-" * 62)
        lines.append(f"Tuổi mẫu vật t          = {m_t:.6g} ± {s_t:.4g}   (95% CI: [{lo_t:.6g}, {hi_t:.6g}])")
        lines.append(f"Hằng số phân rã λ       = {m_l:.6g} ± {s_l:.4g}   (95% CI: [{lo_l:.6g}, {hi_l:.6g}])")
        lines.append(f"Chu kỳ bán rã T₁/₂      = {m_T:.6g} ± {s_T:.4g}   (95% CI: [{lo_T:.6g}, {hi_T:.6g}])")
        lines.append("=" * 62)
        self.result_text.setPlainText("\n".join(lines))

        self.last_report = "\n".join(lines)


# ==========================================================================
# TAB 2: PHƯƠNG TRÌNH TÙY Ý
# ==========================================================================

class VarTable(QTableWidget):
    """Bảng khai báo biến: Tên | Trung bình | Độ lệch chuẩn | Phân bố"""

    def __init__(self):
        super().__init__(0, 4)
        self.setHorizontalHeaderLabels(["Tên biến", "Trung bình", "Độ lệch chuẩn (σ)", "Phân bố"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def add_row(self, name="", mean=0.0, std=0.0, dist="normal"):
        r = self.rowCount()
        self.insertRow(r)
        self.setItem(r, 0, QTableWidgetItem(name))
        self.setItem(r, 1, QTableWidgetItem(str(mean)))
        self.setItem(r, 2, QTableWidgetItem(str(std)))
        combo = QComboBox()
        combo.addItems(["normal", "uniform"])
        combo.setCurrentText(dist)
        self.setCellWidget(r, 3, combo)

    def remove_selected(self):
        rows = sorted({idx.row() for idx in self.selectedIndexes()}, reverse=True)
        for r in rows:
            self.removeRow(r)

    def get_variables(self):
        variables = {}
        for r in range(self.rowCount()):
            name_item = self.item(r, 0)
            mean_item = self.item(r, 1)
            std_item = self.item(r, 2)
            combo = self.cellWidget(r, 3)
            if not name_item or not name_item.text().strip():
                continue
            name = name_item.text().strip()
            mean = float(mean_item.text()) if mean_item and mean_item.text().strip() else 0.0
            std = float(std_item.text()) if std_item and std_item.text().strip() else 0.0
            dist = combo.currentText() if combo else "normal"
            variables[name] = (mean, std, dist)
        return variables

    def set_variables(self, variables: dict):
        self.setRowCount(0)
        for name, (mean, std, dist) in variables.items():
            self.add_row(name, mean, std, dist)


class CustomTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.last_samples = None
        self.last_report = None

        layout = QHBoxLayout(self)
        left = QVBoxLayout()

        desc = QLabel(
            "Nhập <b>bất kỳ</b> phương trình nào (dùng ln, log, exp, sqrt, sin, cos, ...).\n"
            "Ví dụ: <code>ln(N0/N)/lam</code>  hoặc  <code>ln(2)/lam</code>"
        )
        desc.setWordWrap(True)
        left.addWidget(desc)

        left.addWidget(QLabel("Phương trình (vế phải, theo tên biến bạn khai báo bên dưới):"))
        self.formula_edit = QLineEdit("ln(N0/N)/lam")
        left.addWidget(self.formula_edit)

        left.addWidget(QLabel("Khai báo các biến trong phương trình:"))
        self.var_table = VarTable()
        self.var_table.add_row("N0", 1000, 20, "normal")
        self.var_table.add_row("N", 250, 10, "normal")
        self.var_table.add_row("lam", 1.209e-4, 8e-6, "normal")
        left.addWidget(self.var_table, stretch=1)

        row_btn = QHBoxLayout()
        self.btn_add_row = QPushButton("+ Thêm biến")
        self.btn_add_row.clicked.connect(lambda: self.var_table.add_row())
        self.btn_del_row = QPushButton("- Xóa biến đã chọn")
        self.btn_del_row.clicked.connect(self.var_table.remove_selected)
        row_btn.addWidget(self.btn_add_row)
        row_btn.addWidget(self.btn_del_row)
        left.addLayout(row_btn)

        trial_row = QHBoxLayout()
        trial_row.addWidget(QLabel("Số lần thử Monte-Carlo:"))
        self.spin_trials = QSpinBox()
        self.spin_trials.setRange(1000, 5_000_000)
        self.spin_trials.setValue(N_TRIALS_DEFAULT)
        self.spin_trials.setSingleStep(10000)
        trial_row.addWidget(self.spin_trials)
        left.addLayout(trial_row)

        cfg_row = QHBoxLayout()
        self.btn_load = QPushButton("Đọc dữ liệu/phương trình đã lưu…")
        self.btn_load.clicked.connect(self.load_config)
        self.btn_save = QPushButton("Lưu dữ liệu/phương trình…")
        self.btn_save.clicked.connect(self.save_config)
        cfg_row.addWidget(self.btn_load)
        cfg_row.addWidget(self.btn_save)
        left.addLayout(cfg_row)

        self.btn_run = QPushButton("▶ TÍNH TOÁN (Monte-Carlo)")
        self.btn_run.setStyleSheet("font-weight:bold; padding:8px;")
        self.btn_run.clicked.connect(self.run_simulation)
        left.addWidget(self.btn_run)
        left.addStretch()

        layout.addLayout(left, stretch=5)

        right = QVBoxLayout()
        self.fig = Figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        right.addWidget(self.canvas, stretch=3)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFontFamily("Consolas")
        self.result_text.setMinimumHeight(200)
        right.addWidget(self.result_text, stretch=2)

        layout.addLayout(right, stretch=6)

    def get_config(self):
        return {
            "formula": self.formula_edit.text(),
            "variables": self.var_table.get_variables(),
            "n_trials": self.spin_trials.value(),
        }

    def save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Lưu phương trình tùy ý",
                                               "phuong_trinh_tuy_y.json", "JSON (*.json)")
        if not path:
            return
        cfg = self.get_config()
        cfg["mode"] = "custom"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        QMessageBox.information(self, "Đã lưu", f"Đã lưu vào:\n{path}")

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Đọc dữ liệu/phương trình", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.formula_edit.setText(cfg.get("formula", ""))
            variables = {k: tuple(v) for k, v in cfg.get("variables", {}).items()}
            self.var_table.set_variables(variables)
            self.spin_trials.setValue(cfg.get("n_trials", N_TRIALS_DEFAULT))
            QMessageBox.information(self, "Đã đọc", f"Đã nạp dữ liệu/phương trình từ:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi đọc file", str(e))

    def run_simulation(self):
        formula = self.formula_edit.text().strip()
        if not formula:
            QMessageBox.warning(self, "Thiếu phương trình", "Hãy nhập phương trình cần tính.")
            return
        try:
            variables = self.var_table.get_variables()
            samples, used_vars = monte_carlo_custom(
                formula, variables, n_trials=self.spin_trials.value()
            )
        except Exception as e:
            QMessageBox.critical(self, "Lỗi tính toán", f"{e}")
            return

        if len(samples) == 0:
            QMessageBox.warning(self, "Không có kết quả",
                                 "Tất cả mẫu đều không hợp lệ (NaN/vô cực). Kiểm tra lại phương trình/tham số.")
            return

        mean, std, lo, hi = summarize(samples, ci=95)

        self.ax.clear()
        self.ax.hist(samples, bins=80, color="#DD8452", alpha=0.85, density=True)
        self.ax.axvline(mean, color="red", linewidth=1.5, label=f"Trung bình = {mean:.4g}")
        self.ax.axvline(lo, color="green", linestyle="--", linewidth=1.2, label=f"CI 95%: [{lo:.4g}, {hi:.4g}]")
        self.ax.axvline(hi, color="green", linestyle="--", linewidth=1.2)
        self.ax.set_title(f"Phân bố Monte-Carlo của: {formula}")
        self.ax.set_xlabel("Giá trị kết quả")
        self.ax.set_ylabel("Mật độ xác suất")
        self.ax.legend(fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()

        lines = []
        lines.append("=" * 62)
        lines.append("KẾT QUẢ MÔ PHỎNG MONTE-CARLO (Phương trình tùy ý)")
        lines.append("=" * 62)
        lines.append(f"Phương trình: result = {formula}")
        lines.append(f"Biến sử dụng: {', '.join(used_vars)}")
        lines.append(f"Số lần thử: {self.spin_trials.value():,}   |   Số mẫu hợp lệ: {len(samples):,}")
        lines.append("-" * 62)
        lines.append(f"Kết quả = {mean:.6g} ± {std:.4g}")
        lines.append(f"Khoảng tin cậy 95%: [{lo:.6g}, {hi:.6g}]")
        lines.append("=" * 62)
        self.result_text.setPlainText("\n".join(lines))

        self.last_samples = samples
        self.last_report = "\n".join(lines)


# ==========================================================================
# CỬA SỔ CHÍNH
# ==========================================================================

class MonteCarloWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Câu 6B – Định tuổi phóng xạ bằng mô phỏng Monte-Carlo")
        self.resize(1200, 760)

        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)

        self.tabs = QTabWidget()
        self.tab_standard = StandardTab(self)
        self.tab_custom = CustomTab(self)
        self.tabs.addTab(self.tab_standard, "Mô phỏng chuẩn (tuổi / λ / T₁/₂)")
        self.tabs.addTab(self.tab_custom, "Phương trình tùy ý")
        v.addWidget(self.tabs, stretch=1)

        save_row = QHBoxLayout()
        self.btn_save_image = QPushButton("💾 Lưu hình ảnh đồ thị (.png)")
        self.btn_save_image.clicked.connect(self.save_image)
        self.btn_save_report = QPushButton("💾 Lưu kết quả tính toán (.txt)")
        self.btn_save_report.clicked.connect(self.save_report)
        save_row.addWidget(self.btn_save_image)
        save_row.addWidget(self.btn_save_report)
        v.addLayout(save_row)

    def _active_tab(self):
        return self.tab_standard if self.tabs.currentIndex() == 0 else self.tab_custom

    def save_image(self):
        tab = self._active_tab()
        path, _ = QFileDialog.getSaveFileName(self, "Lưu hình ảnh", "do_thi_monte_carlo.png", "PNG (*.png)")
        if not path:
            return
        tab.fig.savefig(path, dpi=150, bbox_inches="tight")
        QMessageBox.information(self, "Đã lưu", f"Đã lưu hình ảnh vào:\n{path}")

    def save_report(self):
        tab = self._active_tab()
        report = getattr(tab, "last_report", None)
        if not report:
            QMessageBox.warning(self, "Chưa có kết quả", "Hãy chạy mô phỏng trước khi lưu.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Lưu kết quả", "ket_qua_monte_carlo.txt", "Text (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
            f.write(f"\n\nXuất lúc: {datetime.datetime.now().isoformat()}\n")
        QMessageBox.information(self, "Đã lưu", f"Đã lưu kết quả vào:\n{path}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MonteCarloWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
