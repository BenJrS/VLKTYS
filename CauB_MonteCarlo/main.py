# -*- coding: utf-8 -*-
"""
CÂU 6B: Tính tuổi mẫu vật và hằng số phân rã phóng xạ bằng phương pháp mô phỏng Monte-Carlo.

Yêu cầu của đề và nơi đáp ứng trong chương trình:
  1. Đọc dữ liệu / phương trình đã lưu sẵn, hiển thị đồ thị lên GUI; tính tuổi mẫu vật,
     hằng số phân rã, chu kỳ bán rã bằng Monte-Carlo.          -> tab "Dữ liệu đo" + "Mở dữ liệu…"
  2. Cho phép nhập phương trình bất kỳ để tính toán
     (+ khuyến khích: khoảng tin cậy của kết quả ước lượng).     -> tab "Phương trình tự nhập" (có CI)
  3. Lưu hình ảnh đã vẽ và kết quả đã tính
     (+ khuyến khích: đóng gói ứng dụng chạy độc lập).          -> "Lưu đồ thị…", "Lưu kết quả…", bản build

Mô hình vật lý — phân rã bậc nhất:
    N(t) = N0·e^(−λt)       =>   ln N = ln N0 − λ·t
    T½ = ln 2 / λ,           tuổi mẫu  t = ln(N0/N) / λ

Thuật toán Monte-Carlo (tab "Dữ liệu đo"), lặp M lần:
  a) Tạo bộ số đếm giả lập N_i* ~ Poisson(N_i)  (số đếm phóng xạ tuân theo phân bố Poisson).
  b) Khớp đường thẳng ln N_i* theo t_i bằng bình phương tối thiểu có trọng số (w_i = N_i*)
     -> λ* = −độ dốc,  T½* = ln2/λ*.
  c) Lấy mẫu kết quả đo mẫu vật N0* ~ Chuẩn(N0, σ0), N* ~ Chuẩn(N, σN)  -> tuổi t* = ln(N0*/N*)/λ*.
  Từ M giá trị (λ*, T½*, t*) suy ra trung bình, độ lệch chuẩn và khoảng tin cậy (phân vị).

Tác giả: (điền họ tên – MSSV)
"""

import re
import sys
import json
import keyword
import datetime
from tokenize import TokenError
from pathlib import Path

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor

if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
import theme  # noqa: E402

from PySide6.QtCore import Qt, QLocale, QTimer, Signal  # noqa: E402
from PySide6.QtGui import QDoubleValidator, QKeySequence, QShortcut  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QSpinBox, QLineEdit, QComboBox,
    QFileDialog, QMessageBox, QHeaderView, QAbstractItemView, QTabWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


# ==========================================================================
# PHẦN LÕI TÍNH TOÁN (tách riêng khỏi giao diện để dễ kiểm thử)
# ==========================================================================

LN2 = np.log(2.0)


def summarize(samples, ci=95):
    """Trung bình, độ lệch chuẩn, trung vị và khoảng tin cậy (phân vị) của mẫu Monte-Carlo."""
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    if s.size < 2:
        raise ValueError("Không đủ mẫu hợp lệ để thống kê. Kiểm tra lại dữ liệu đầu vào.")
    a = (100 - ci) / 2
    lo, med, hi = np.percentile(s, [a, 50, 100 - a])
    return {"mean": float(s.mean()), "std": float(s.std(ddof=1)), "median": float(med),
            "lo": float(lo), "hi": float(hi), "n": int(s.size), "ci": ci}


def sample_variable(mean, std, dist, n, rng):
    """Lấy n mẫu cho một đại lượng đo: phân bố chuẩn, hoặc đều có cùng độ lệch chuẩn."""
    if std <= 0:
        return np.full(n, float(mean))
    if dist == "normal":
        return rng.normal(mean, std, n)
    if dist == "uniform":
        h = std * np.sqrt(3.0)          # phân bố đều [m−h, m+h] có độ lệch chuẩn = h/√3
        return rng.uniform(mean - h, mean + h, n)
    raise ValueError(f"Phân bố không hỗ trợ: {dist}")


def weighted_decay_fit(t, counts):
    """
    Khớp ln N = a − λ·t cho từng hàng của `counts` (M×m) bằng bình phương tối thiểu có trọng số w = N.
    Trả về (a, λ) dạng mảng độ dài M.
    """
    c = np.asarray(counts, float)
    w = c
    y = np.log(np.where(c > 0, c, np.nan))
    sw = w.sum(axis=-1)
    st = (w * t).sum(axis=-1)
    stt = (w * t * t).sum(axis=-1)
    sy = (w * y).sum(axis=-1)
    sty = (w * t * y).sum(axis=-1)
    den = sw * stt - st ** 2
    slope = (sw * sty - st * sy) / den
    a = (sy - slope * st) / sw
    return a, -slope


def monte_carlo_decay(t, counts, N0, N0_sigma, N, N_sigma, n_trials=100_000, seed=None):
    """Mô phỏng Monte-Carlo: ước lượng λ, T½ từ số đếm phân rã và tuổi của mẫu vật."""
    t = np.asarray(t, float)
    counts = np.asarray(counts, float)
    if t.size < 3:
        raise ValueError("Cần ít nhất 3 điểm đo (t, N) để khớp đường phân rã.")
    if np.any(counts <= 0):
        raise ValueError("Số đếm N phải dương ở mọi thời điểm.")
    if np.ptp(t) == 0:
        raise ValueError("Các thời điểm t phải khác nhau.")
    if N0 <= 0 or N <= 0 or N >= N0:
        raise ValueError("Cần 0 < N < N0 cho mẫu vật (lượng còn lại phải nhỏ hơn ban đầu).")

    rng = np.random.default_rng(seed)
    sim = rng.poisson(counts, size=(n_trials, counts.size)).astype(float)
    a, lam = weighted_decay_fit(t, sim)
    N0s = rng.normal(N0, N0_sigma, n_trials) if N0_sigma > 0 else np.full(n_trials, N0)
    Ns = rng.normal(N, N_sigma, n_trials) if N_sigma > 0 else np.full(n_trials, N)

    valid = np.isfinite(lam) & (lam > 0) & (N0s > 0) & (Ns > 0) & (Ns < N0s)
    lam_v, a_v = lam[valid], a[valid]
    age = np.log(N0s[valid] / Ns[valid]) / lam_v

    # dải tin cậy cho đường khớp (dùng tối đa 4000 lần thử để vẽ nhanh)
    a0, lam0 = weighted_decay_fit(t, counts)
    grid = np.linspace(t.min(), t.max(), 160)
    sub = slice(0, min(4000, lam_v.size))
    curves = np.exp(a_v[sub, None] - lam_v[sub, None] * grid[None, :])

    return {"lambda": lam_v, "T_half": LN2 / lam_v, "age": age,
            "n_trials": n_trials, "n_valid": int(valid.sum()),
            "t": t, "counts": counts, "fit": (float(a0), float(lam0)),
            "grid": grid, "curves": curves}


# ---------------------------------------------------------------- phương trình tự nhập

FUNCS = {"ln": sp.log, "log": sp.log, "log10": lambda x: sp.log(x, 10), "exp": sp.exp,
         "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "abs": sp.Abs,
         "pi": sp.pi, "e": sp.E}
TRANSFORMS = standard_transformations + (convert_xor,)


def check_name(name):
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"Tên biến \"{name}\" không hợp lệ. Dùng chữ cái, số, dấu _ và "
                         "không trùng từ khóa Python (ví dụ dùng lam thay cho lambda).")
    if name in FUNCS:
        raise ValueError(f"\"{name}\" là tên hàm/hằng có sẵn, hãy đặt tên biến khác.")


def parse_equation(text, var_names, means=None):
    """
    Phân tích phương trình người dùng nhập.
      - Dạng biểu thức:  ln(N0/N)/lam          -> tính trực tiếp, mọi ký hiệu phải có giá trị.
      - Dạng phương trình:  t = ln(N0/N)/lam   hoặc   N = N0*exp(-lam*t)
        -> đúng một ký hiệu chưa có giá trị là ẩn cần tính; chương trình tự giải theo ẩn đó.
    Trả về (tên ẩn, biểu thức sympy theo các biến đã khai báo).
    """
    text = text.strip()
    if not text:
        raise ValueError("Hãy nhập phương trình.")
    for nm in var_names:
        check_name(nm)
    # Mọi tên trong phương trình (trừ hàm có sẵn) đều là ký hiệu của người dùng. Khai báo tường minh để
    # tránh trùng với tên dựng sẵn của sympy (N, E, S, I, O, Q, beta, gamma…).
    names = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", text)) - set(FUNCS)
    for nm in names:
        check_name(nm)
    local = dict(FUNCS)
    local.update({nm: sp.Symbol(nm) for nm in names | set(var_names)})
    known = set(var_names)

    def parse(s):
        s = s.strip()
        if not s:
            raise ValueError("Một vế của phương trình đang trống.")
        return parse_expr(s, local_dict=local, transformations=TRANSFORMS, evaluate=True)

    def collect(expr):
        # ký hiệu chưa khai báo sẽ trở thành Symbol mới (parse_expr tự tạo)
        return {s.name for s in expr.free_symbols}

    if text.count("=") > 1 or "==" in text:
        raise ValueError("Chỉ dùng một dấu = trong phương trình.")
    try:
        if "=" in text:
            lhs_s, rhs_s = text.split("=")
            lhs, rhs = parse(lhs_s), parse(rhs_s)
            unknown = (collect(lhs) | collect(rhs)) - known
        else:
            lhs, rhs = None, parse(text)
            unknown = collect(rhs) - known
    except (SyntaxError, TypeError, sp.SympifyError, TokenError):
        raise ValueError("Phương trình viết chưa đúng cú pháp. Ví dụ hợp lệ: N = N0*exp(-lam*t)")

    if lhs is None:
        if unknown:
            raise ValueError(f"Chưa có giá trị cho: {', '.join(sorted(unknown))}. "
                             "Thêm vào bảng biến, hoặc viết dạng \"ẩn = …\".")
        return "kết quả", rhs

    if not unknown:
        raise ValueError("Mọi ký hiệu đều đã có giá trị nên không còn ẩn để tính. "
                         "Xóa đại lượng cần tính khỏi bảng biến.")
    if len(unknown) > 1:
        raise ValueError(f"Còn {len(unknown)} ký hiệu chưa có giá trị: {', '.join(sorted(unknown))}. "
                         "Chỉ để trống đúng một ẩn cần tính.")
    target = unknown.pop()
    sym = sp.Symbol(target)
    if lhs == sym and sym not in rhs.free_symbols:
        return target, rhs
    try:
        sols = sp.solve(sp.Eq(lhs, rhs), sym)
    except (NotImplementedError, ValueError):
        sols = []
    if not sols:
        raise ValueError(f"Không giải được phương trình theo {target} dưới dạng tường minh. "
                         f"Hãy viết lại dạng \"{target} = …\".")
    if len(sols) == 1 or not means:
        return target, sols[0]
    # nhiều nghiệm: chọn nghiệm thực, ưu tiên dương, tại giá trị trung bình của các biến
    subs = {sp.Symbol(k): v for k, v in means.items()}
    scored = []
    for s in sols:
        try:
            v = complex(s.evalf(subs=subs))
        except (TypeError, ValueError):
            continue
        if abs(v.imag) < 1e-12 * max(1.0, abs(v.real)):
            scored.append((0 if v.real > 0 else 1, s))
    return target, (sorted(scored, key=lambda p: p[0])[0][1] if scored else sols[0])


def monte_carlo_custom(formula, variables, n_trials=100_000, seed=None):
    """variables: {tên: (trung bình, độ lệch chuẩn, phân bố)}. Trả về (tên ẩn, biểu thức, mẫu)."""
    target, expr = parse_equation(formula, list(variables),
                                  {k: v[0] for k, v in variables.items()})
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    f = sp.lambdify(syms, expr, modules="numpy")
    rng = np.random.default_rng(seed)
    args = [sample_variable(*variables[s.name], n_trials, rng) for s in syms]
    with np.errstate(all="ignore"):
        out = np.broadcast_to(np.asarray(f(*args)), (n_trials,))
    if np.iscomplexobj(out):
        out = np.where(np.abs(out.imag) < 1e-12, out.real, np.nan)
    out = np.asarray(out, float)
    return target, expr, out[np.isfinite(out)]


# ==========================================================================
# THÀNH PHẦN GIAO DIỆN
# ==========================================================================

ACC = theme.ACCENTS["decay"]


class NumberEdit(QLineEdit):
    """Ô nhập số thực (chấp nhận dạng 1.2e-4 và dấu phẩy thập phân)."""

    def __init__(self, value=0.0):
        super().__init__()
        v = QDoubleValidator(self)
        v.setNotation(QDoubleValidator.ScientificNotation)
        v.setLocale(QLocale.c())
        self.setFont(theme.mono_font(10.5))
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setValue(value)

    def value(self):
        try:
            return float(self.text().strip().replace(",", "."))
        except ValueError:
            raise ValueError(f"\"{self.text()}\" không phải là số.")

    def setValue(self, v):
        self.setText(f"{float(v):.10g}")


def num_item(v, align=Qt.AlignRight):
    it = QTableWidgetItem(v if isinstance(v, str) else f"{v:.10g}")
    it.setTextAlignment(align | Qt.AlignVCenter)
    return it


def make_table(headers, mono=True):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(28)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    if mono:
        t.setFont(theme.mono_font(10.5))
    return t


def ci_combo():
    c = QComboBox()
    c.addItems(["90%", "95%", "99%"])
    c.setCurrentIndex(1)
    return c


def draw_hist(ax, samples, st, title, xlabel):
    """Histogram mẫu Monte-Carlo; phần nằm trong khoảng tin cậy được tô đậm."""
    lo_cut, hi_cut = np.percentile(samples, [0.1, 99.9])
    s = samples[(samples >= lo_cut) & (samples <= hi_cut)]
    h, edges = np.histogram(s, bins=70, density=True)
    mid = (edges[:-1] + edges[1:]) / 2
    inside = (mid >= st["lo"]) & (mid <= st["hi"])
    ax.bar(mid, h, width=np.diff(edges), linewidth=0,
           color=np.where(inside, ACC["main"], ACC["soft"]), alpha=0.8)
    ax.axvline(st["mean"], color=theme.INK, lw=1.3)
    for x in (st["lo"], st["hi"]):
        ax.axvline(x, color=ACC["main"], lw=1, ls=(0, (3, 2)))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 5), useMathText=True)


def results_to_csv(rows, path):
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write("dai_luong,trung_binh,do_lech_chuan,trung_vi,ci_duoi,ci_tren,muc_tin_cay,don_vi\n")
        for name, st, unit in rows:
            f.write(f"{name},{st['mean']:.10g},{st['std']:.6g},{st['median']:.10g},"
                    f"{st['lo']:.10g},{st['hi']:.10g},{st['ci']},{unit}\n")


# ==========================================================================
# TAB 1: DỮ LIỆU ĐO — tuổi mẫu vật, λ, T½
# ==========================================================================

class DecayTab(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.result = None
        self.isotope = "C-14"
        self.unit = "năm"
        self.source = None

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 12, 0, 0)
        lay.setSpacing(14)
        lay.addWidget(self._build_inputs(), 4)

        right = QVBoxLayout()
        right.setSpacing(14)
        pf, pl = theme.panel()
        self.fig = Figure(figsize=(8, 5.4))
        self.canvas = FigureCanvas(self.fig)
        pl.addWidget(self.canvas)
        right.addWidget(pf, 1)
        right.addWidget(self._build_results(), 0)
        lay.addLayout(right, 7)
        self._draw_empty()

    # ------------------------------------------------------------------ giao diện
    def _build_inputs(self):
        frame, lay = theme.panel()
        lay.addWidget(theme.label("Số đếm phân rã của mẫu chuẩn", "Section"))
        self.lbl_source = theme.label("", "Hint", wrap=True)
        lay.addWidget(self.lbl_source)
        self.table = make_table(["t", "N (số đếm)"])
        lay.addWidget(self.table, 1)
        rows = QHBoxLayout()
        b_add = QPushButton("Thêm hàng")
        b_add.setObjectName("Quiet")
        b_add.clicked.connect(lambda: self._append_row(None, None))
        b_del = QPushButton("Xóa hàng đã chọn")
        b_del.setObjectName("Quiet")
        b_del.clicked.connect(self._remove_rows)
        rows.addWidget(b_add)
        rows.addWidget(b_del)
        rows.addStretch()
        lay.addLayout(rows)

        lay.addWidget(theme.rule())
        lay.addWidget(theme.label("Mẫu vật cần định tuổi", "Section"))
        g = QGridLayout()
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(4)
        self.N0, self.N0s = NumberEdit(1000), NumberEdit(20)
        self.N, self.Ns = NumberEdit(250), NumberEdit(10)
        g.addWidget(theme.label("Ban đầu N₀", "FactName"), 0, 0)
        g.addWidget(theme.label("σ của N₀", "FactName"), 0, 1)
        g.addWidget(self.N0, 1, 0)
        g.addWidget(self.N0s, 1, 1)
        g.addWidget(theme.label("Hiện tại N", "FactName"), 2, 0)
        g.addWidget(theme.label("σ của N", "FactName"), 2, 1)
        g.addWidget(self.N, 3, 0)
        g.addWidget(self.Ns, 3, 1)
        lay.addLayout(g)

        lay.addWidget(theme.rule())
        sim = QGridLayout()
        sim.setHorizontalSpacing(10)
        sim.setVerticalSpacing(4)
        self.spin_trials = QSpinBox()
        self.spin_trials.setRange(1000, 2_000_000)
        self.spin_trials.setSingleStep(50_000)
        self.spin_trials.setValue(100_000)
        self.spin_trials.setGroupSeparatorShown(True)
        self.spin_trials.setLocale(QLocale(QLocale.Vietnamese, QLocale.Vietnam))
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999_999)
        self.spin_seed.setValue(2026)
        self.spin_seed.setToolTip("Cùng seed cho cùng kết quả, giúp kiểm tra lại được.")
        self.combo_ci = ci_combo()
        for c, (name, w) in enumerate([("Số lần thử", self.spin_trials), ("Seed", self.spin_seed),
                                       ("Độ tin cậy", self.combo_ci)]):
            sim.addWidget(theme.label(name, "FactName"), 0, c)
            sim.addWidget(w, 1, c)
        sim.setColumnStretch(0, 2)
        lay.addLayout(sim)

        self.btn_run = QPushButton("Chạy mô phỏng")
        self.btn_run.setObjectName("Primary")
        self.btn_run.setToolTip("Ctrl+Enter")
        self.btn_run.clicked.connect(self.run)
        lay.addWidget(self.btn_run)
        return frame

    def _build_results(self):
        frame, lay = theme.panel()
        top = QHBoxLayout()
        top.addWidget(theme.label("Kết quả ước lượng", "Section"))
        top.addStretch()
        self.lbl_meta = theme.label("", "Status")
        top.addWidget(self.lbl_meta)
        lay.addLayout(top)
        self.res_table = make_table(["Đại lượng", "Trung bình", "Độ lệch chuẩn", "CI dưới", "CI trên", "Đơn vị"])
        self.res_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.res_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.res_table.setFixedHeight(28 * 3 + 32)
        self.res_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        lay.addWidget(self.res_table)
        return frame

    def _append_row(self, t, n):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, num_item("" if t is None else t))
        self.table.setItem(r, 1, num_item("" if n is None else n))

    def _remove_rows(self):
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)

    # ------------------------------------------------------------------ dữ liệu
    def set_data(self, cfg, source=None):
        d = cfg.get("decay_data", {})
        t, n = d.get("t", []), d.get("N", [])
        if len(t) != len(n):
            raise ValueError("decay_data.t và decay_data.N phải có cùng số phần tử.")
        s = cfg.get("sample", {})
        self.isotope = cfg.get("isotope", "mẫu")
        self.unit = cfg.get("time_unit", "năm")
        self.table.setRowCount(0)
        for ti, ni in zip(t, n):
            self._append_row(float(ti), float(ni))
        self.table.setHorizontalHeaderLabels([f"t ({self.unit})", "N (số đếm)"])
        self.N0.setValue(s.get("N0", 1000))
        self.N0s.setValue(s.get("N0_sigma", 0))
        self.N.setValue(s.get("N", 250))
        self.Ns.setValue(s.get("N_sigma", 0))
        self.spin_trials.setValue(int(cfg.get("n_trials", 100_000)))
        self.spin_seed.setValue(int(cfg.get("seed", 2026)))
        self.combo_ci.setCurrentText(f"{int(cfg.get('ci', 95))}%")
        self.source = source
        where = Path(source).name if source else "ví dụ có sẵn"
        self.lbl_source.setText(f"Đồng vị {self.isotope}, nguồn: {where}. "
                                "Mỗi hàng là số đếm N ghi tại thời điểm t.")

    def load_csv(self, path):
        """CSV/TXT hai cột t, N (có thể có dòng tiêu đề)."""
        rows = []
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                parts = line.replace(";", ",").replace("\t", ",").split(",")
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except (ValueError, IndexError):
                    continue
        if len(rows) < 3:
            raise ValueError("File cần ít nhất 3 hàng số dạng: t, N")
        self.table.setRowCount(0)
        for t, n in rows:
            self._append_row(t, n)
        self.source = path
        self.lbl_source.setText(f"Đã nạp {len(rows)} điểm đo từ {Path(path).name}.")

    def get_data(self):
        t, n = [], []
        for r in range(self.table.rowCount()):
            cells = [self.table.item(r, c) for c in (0, 1)]
            texts = [(c.text() if c else "").strip().replace(",", ".") for c in cells]
            if not any(texts):
                continue
            try:
                t.append(float(texts[0]))
                n.append(float(texts[1]))
            except ValueError:
                raise ValueError(f"Hàng {r + 1} của bảng số đếm có ô trống hoặc không phải số.")
        return {
            "mode": "decay", "isotope": self.isotope, "time_unit": self.unit,
            "decay_data": {"t": t, "N": n},
            "sample": {"N0": self.N0.value(), "N0_sigma": self.N0s.value(),
                       "N": self.N.value(), "N_sigma": self.Ns.value()},
            "n_trials": self.spin_trials.value(), "seed": self.spin_seed.value(),
            "ci": int(self.combo_ci.currentText().rstrip("%")),
        }

    # ------------------------------------------------------------------ tính toán
    def run(self):
        try:
            cfg = self.get_data()
            s = cfg["sample"]
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                res = monte_carlo_decay(cfg["decay_data"]["t"], cfg["decay_data"]["N"],
                                        s["N0"], s["N0_sigma"], s["N"], s["N_sigma"],
                                        n_trials=cfg["n_trials"], seed=cfg["seed"])
            finally:
                QApplication.restoreOverrideCursor()
            ci = cfg["ci"]
            stats = {k: summarize(res[k], ci) for k in ("lambda", "T_half", "age")}
        except ValueError as e:
            QMessageBox.warning(self, "Chưa chạy được mô phỏng", str(e))
            return
        self.result = {"cfg": cfg, "res": res, "stats": stats}
        self._draw(res, stats, ci)
        self._fill_results(res, stats)
        self.window.on_result()

    def rows(self):
        st, u = self.result["stats"], self.unit
        return [("Hằng số phân rã λ", st["lambda"], f"1/{u}"),
                ("Chu kỳ bán rã T½", st["T_half"], u),
                ("Tuổi mẫu vật t", st["age"], u)]

    def _fill_results(self, res, stats):
        self.res_table.setRowCount(3)
        for i, (name, st, unit) in enumerate(self.rows()):
            vals = [name, theme.fmt(st["mean"]), theme.fmt(st["std"], 3),
                    theme.fmt(st["lo"]), theme.fmt(st["hi"]), unit]
            for j, v in enumerate(vals):
                it = num_item(v, Qt.AlignLeft if j in (0, 5) else Qt.AlignRight)
                if j in (0, 5):
                    it.setFont(self.font())
                self.res_table.setItem(i, j, it)
        ci = stats["age"]["ci"]
        self.res_table.setHorizontalHeaderLabels(["Đại lượng", "Trung bình", "Độ lệch chuẩn",
                                                  f"CI {ci}% dưới", f"CI {ci}% trên", "Đơn vị"])
        self.lbl_meta.setText(f"{res['n_valid']:,} / {res['n_trials']:,} lần thử hợp lệ".replace(",", "."))

    def _draw_empty(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Nhấn \"Chạy mô phỏng\" để xem đường phân rã và phân bố Monte-Carlo",
                ha="center", va="center", color=theme.MUTED, transform=ax.transAxes)
        self.canvas.draw()

    def _draw(self, res, st, ci):
        u = self.unit
        self.fig.clear()
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.55, wspace=0.12)

        ax = self.fig.add_subplot(gs[0, :])
        lo, hi = np.percentile(res["curves"], [(100 - ci) / 2, 100 - (100 - ci) / 2], axis=0)
        ax.fill_between(res["grid"], lo, hi, color=ACC["soft"], lw=0, label=f"Dải tin cậy {ci}% của đường khớp")
        a0, lam0 = res["fit"]
        ax.plot(res["grid"], np.exp(a0 - lam0 * res["grid"]), color=ACC["main"], lw=1.8,
                label=f"Khớp N₀·e^(−λt), λ = {theme.fmt(lam0, 4)} /{u}")
        ax.errorbar(res["t"], res["counts"], yerr=np.sqrt(res["counts"]), fmt="o", ms=4,
                    color=theme.INK, mfc="white", mew=1.2, elinewidth=1, capsize=0, label="Số đếm đo (±√N)")
        ax.set_title(f"Đường phân rã của {self.isotope}")
        ax.set_xlabel(f"Thời gian t ({u})")
        ax.set_ylabel("Số đếm N")
        ax.legend(loc="upper right")

        specs = [("lambda", f"λ = {theme.fmt(st['lambda']['mean'], 4)} /{u}", f"λ (1/{u})"),
                 ("T_half", f"T½ = {theme.fmt(st['T_half']['mean'], 5)} {u}", f"T½ ({u})"),
                 ("age", f"Tuổi t = {theme.fmt(st['age']['mean'], 5)} {u}", f"t ({u})")]
        for c, (key, title, xl) in enumerate(specs):
            draw_hist(self.fig.add_subplot(gs[1, c]), res[key], st[key], title, xl)
        self.fig.text(0.99, 0.008, f"Vùng tô đậm: khoảng tin cậy {ci}%   |   vạch đen: trung bình   |   "
                                   f"{res['n_trials']:,} lần thử Monte-Carlo".replace(",", "."),
                      ha="right", va="bottom", fontsize=7.5, color=theme.MUTED)
        self.fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.12)
        self.canvas.draw()

    def report(self):
        r = self.result
        cfg, res = r["cfg"], r["res"]
        s = cfg["sample"]
        L = ["KẾT QUẢ MÔ PHỎNG MONTE-CARLO: ĐỊNH TUỔI PHÓNG XẠ",
             f"Thời điểm: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
             f"Nguồn dữ liệu: {self.source or 'ví dụ có sẵn'}", "",
             f"Đồng vị: {self.isotope}    Đơn vị thời gian: {self.unit}",
             f"Số lần thử: {res['n_trials']}    Hợp lệ: {res['n_valid']}    Seed: {cfg['seed']}",
             f"Mẫu vật: N0 = {s['N0']:g} ± {s['N0_sigma']:g},  N = {s['N']:g} ± {s['N_sigma']:g}", "",
             "Số đếm phân rã của mẫu chuẩn:", f"{'t':>10}  {'N':>10}"]
        L += [f"{t:>10g}  {n:>10g}" for t, n in zip(res["t"], res["counts"])]
        L += ["", "Phương pháp: N_i* ~ Poisson(N_i); khớp ln N* = a − λt (bình phương tối thiểu có trọng số);",
              "T½ = ln2/λ; N0*, N* ~ Chuẩn; tuổi t = ln(N0*/N*)/λ.", ""]
        ci = r["stats"]["age"]["ci"]
        L.append(f"{'Đại lượng':<20}{'Trung bình':>14}{'Độ lệch chuẩn':>16}{'Trung vị':>14}"
                 f"{f'CI {ci}% dưới':>14}{f'CI {ci}% trên':>14}  Đơn vị")
        for name, st, unit in self.rows():
            L.append(f"{name:<20}{st['mean']:>14.6g}{st['std']:>16.4g}{st['median']:>14.6g}"
                     f"{st['lo']:>14.6g}{st['hi']:>14.6g}  {unit}")
        return "\n".join(L)


# ==========================================================================
# TAB 2: PHƯƠNG TRÌNH TỰ NHẬP
# ==========================================================================

PRESETS = [
    ("Tuổi từ định luật phân rã (tự giải theo t)", "N = N0*exp(-ln(2)*t/T)",
     {"N0": (1000, 20, "normal"), "N": (250, 10, "normal"), "T": (5730, 40, "normal")}),
    ("Tuổi từ tỉ số N₀/N và λ", "t = ln(N0/N)/lam",
     {"N0": (1000, 20, "normal"), "N": (250, 10, "normal"), "lam": (1.2097e-4, 8.4e-7, "normal")}),
    ("Chu kỳ bán rã từ hai lần đếm", "T = ln(2)*dt/ln(N1/N2)",
     {"N1": (5000, 71, "normal"), "N2": (4430, 67, "normal"), "dt": (1000, 0, "normal")}),
    ("Hằng số phân rã từ chu kỳ bán rã", "lam = ln(2)/T",
     {"T": (5730, 40, "normal")}),
]
DISTS = [("Chuẩn", "normal"), ("Đều", "uniform")]


class VarTable(QTableWidget):
    changed = Signal()

    def __init__(self):
        super().__init__(0, 4)
        self.itemChanged.connect(lambda *_: self.changed.emit())
        self.setHorizontalHeaderLabels(["Tên biến", "Giá trị đo", "σ", "Phân bố"])
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(30)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.setFont(theme.mono_font(10.5))

    def add(self, name="", mean=0.0, std=0.0, dist="normal"):
        r = self.rowCount()
        self.insertRow(r)
        self.setItem(r, 0, num_item(name, Qt.AlignLeft))
        self.setItem(r, 1, num_item(mean))
        self.setItem(r, 2, num_item(std))
        combo = QComboBox()
        for label, key in DISTS:
            combo.addItem(label, key)
        combo.setCurrentIndex(max(combo.findData(dist), 0))
        combo.currentIndexChanged.connect(lambda *_: self.changed.emit())
        self.setCellWidget(r, 3, combo)

    def remove_selected(self):
        for r in sorted({i.row() for i in self.selectedIndexes()}, reverse=True):
            self.removeRow(r)
        self.changed.emit()

    def names(self):
        out = []
        for r in range(self.rowCount()):
            it = self.item(r, 0)
            if it and it.text().strip():
                out.append(it.text().strip())
        return out

    def variables(self):
        v = {}
        for r in range(self.rowCount()):
            it = self.item(r, 0)
            name = it.text().strip() if it else ""
            if not name:
                continue
            check_name(name)
            if name in v:
                raise ValueError(f"Biến \"{name}\" bị khai báo hai lần.")
            try:
                mean = float(self.item(r, 1).text().replace(",", "."))
                std = float((self.item(r, 2).text() or "0").replace(",", "."))
            except (ValueError, AttributeError):
                raise ValueError(f"Giá trị hoặc σ của biến \"{name}\" không phải số.")
            if std < 0:
                raise ValueError(f"σ của biến \"{name}\" phải ≥ 0.")
            v[name] = (mean, std, self.cellWidget(r, 3).currentData())
        return v

    def set_variables(self, variables):
        self.blockSignals(True)
        self.setRowCount(0)
        for name, val in variables.items():
            mean, std, dist = (list(val) + [0.0, "normal"])[:3]
            self.add(name, float(mean), float(std), dist)
        self.blockSignals(False)
        self.changed.emit()


class EquationTab(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.result = None
        self.source = None
        self._parse_timer = QTimer(self, singleShot=True, interval=250)
        self._parse_timer.timeout.connect(self.preview)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 12, 0, 0)
        lay.setSpacing(14)
        lay.addWidget(self._build_inputs(), 4)

        right = QVBoxLayout()
        right.setSpacing(14)
        pf, pl = theme.panel()
        self.fig = Figure(figsize=(7, 4.4))
        self.canvas = FigureCanvas(self.fig)
        pl.addWidget(self.canvas)
        right.addWidget(pf, 1)
        right.addWidget(self._build_results(), 0)
        lay.addLayout(right, 7)

        self.apply_preset(0)
        self._draw_empty()

    def _build_inputs(self):
        frame, lay = theme.panel()
        head = QHBoxLayout()
        head.addWidget(theme.label("Phương trình", "Section"))
        head.addStretch()
        self.combo_preset = QComboBox()
        self.combo_preset.addItem("Chọn phương trình mẫu…")
        for name, _, _ in PRESETS:
            self.combo_preset.addItem(name)
        self.combo_preset.activated.connect(lambda i: i and self.apply_preset(i - 1))
        head.addWidget(self.combo_preset)
        lay.addLayout(head)

        self.edit_eq = QLineEdit()
        self.edit_eq.setObjectName("Equation")
        self.edit_eq.setPlaceholderText("ví dụ: N = N0*exp(-lam*t)")
        self.edit_eq.textChanged.connect(lambda *_: self._parse_timer.start())
        lay.addWidget(self.edit_eq)
        self.lbl_solved = theme.label("", "Formula", wrap=True)
        lay.addWidget(self.lbl_solved)
        lay.addWidget(theme.label(
            "Viết \"ẩn = biểu thức\" hoặc một phương trình bất kỳ. Ký hiệu duy nhất không có trong bảng biến "
            "là đại lượng cần tính; chương trình tự giải theo nó. Hàm dùng được: ln, log10, exp, sqrt, "
            "sin, cos, tan, abs; hằng pi, e; lũy thừa ** hoặc ^.", "Hint", wrap=True))

        lay.addWidget(theme.rule())
        lay.addWidget(theme.label("Biến đo và sai số", "Section"))
        self.vars = VarTable()
        self.vars.changed.connect(self._parse_timer.start)
        lay.addWidget(self.vars, 1)
        rows = QHBoxLayout()
        b_add = QPushButton("Thêm biến")
        b_add.setObjectName("Quiet")
        b_add.clicked.connect(lambda: self.vars.add())
        b_del = QPushButton("Xóa biến đã chọn")
        b_del.setObjectName("Quiet")
        b_del.clicked.connect(self.vars.remove_selected)
        rows.addWidget(b_add)
        rows.addWidget(b_del)
        rows.addStretch()
        lay.addLayout(rows)

        lay.addWidget(theme.rule())
        sim = QGridLayout()
        sim.setHorizontalSpacing(10)
        sim.setVerticalSpacing(4)
        self.spin_trials = QSpinBox()
        self.spin_trials.setRange(1000, 2_000_000)
        self.spin_trials.setSingleStep(50_000)
        self.spin_trials.setValue(100_000)
        self.spin_trials.setGroupSeparatorShown(True)
        self.spin_trials.setLocale(QLocale(QLocale.Vietnamese, QLocale.Vietnam))
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999_999)
        self.spin_seed.setValue(2026)
        self.combo_ci = ci_combo()
        for c, (name, w) in enumerate([("Số lần thử", self.spin_trials), ("Seed", self.spin_seed),
                                       ("Độ tin cậy", self.combo_ci)]):
            sim.addWidget(theme.label(name, "FactName"), 0, c)
            sim.addWidget(w, 1, c)
        sim.setColumnStretch(0, 2)
        lay.addLayout(sim)

        self.btn_run = QPushButton("Tính bằng Monte-Carlo")
        self.btn_run.setObjectName("Primary")
        self.btn_run.setToolTip("Ctrl+Enter")
        self.btn_run.clicked.connect(self.run)
        lay.addWidget(self.btn_run)
        return frame

    def _build_results(self):
        frame, lay = theme.panel()
        top = QHBoxLayout()
        top.addWidget(theme.label("Kết quả ước lượng", "Section"))
        top.addStretch()
        self.lbl_meta = theme.label("", "Status")
        top.addWidget(self.lbl_meta)
        lay.addLayout(top)
        self.res_table = make_table(["Đại lượng", "Trung bình", "Độ lệch chuẩn", "Trung vị", "CI dưới", "CI trên"])
        self.res_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.res_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.res_table.setFixedHeight(28 + 32)
        self.res_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        lay.addWidget(self.res_table)
        return frame

    # ------------------------------------------------------------------ dữ liệu
    def apply_preset(self, i):
        _, formula, variables = PRESETS[i]
        self.edit_eq.setText(formula)
        self.vars.set_variables(variables)
        self.combo_preset.setCurrentIndex(0)
        self.preview()

    def set_data(self, cfg, source=None):
        self.edit_eq.setText(cfg.get("formula", ""))
        self.vars.set_variables(cfg.get("variables", {}))
        self.spin_trials.setValue(int(cfg.get("n_trials", 100_000)))
        self.spin_seed.setValue(int(cfg.get("seed", 2026)))
        self.combo_ci.setCurrentText(f"{int(cfg.get('ci', 95))}%")
        self.source = source
        self.preview()

    def get_data(self):
        return {"mode": "custom", "formula": self.edit_eq.text().strip(),
                "variables": {k: list(v) for k, v in self.vars.variables().items()},
                "n_trials": self.spin_trials.value(), "seed": self.spin_seed.value(),
                "ci": int(self.combo_ci.currentText().rstrip("%"))}

    def preview(self):
        """Phân tích phương trình ngay khi gõ và cho biết sẽ tính đại lượng nào."""
        try:
            variables = self.vars.variables()
            target, expr = parse_equation(self.edit_eq.text(), list(variables),
                                          {k: v[0] for k, v in variables.items()})
            self.lbl_solved.setText(f"Sẽ tính:  {target} = {sp.sstr(expr)}")
            self.lbl_solved.setStyleSheet(f"color: {ACC['main']};")
            self.edit_eq.setProperty("invalid", False)
        except ValueError as e:
            self.lbl_solved.setText(str(e))
            self.lbl_solved.setStyleSheet(f"color: {theme.WARN}; font-family: '{theme.SANS}';")
            self.edit_eq.setProperty("invalid", bool(self.edit_eq.text().strip()))
        self.edit_eq.style().unpolish(self.edit_eq)
        self.edit_eq.style().polish(self.edit_eq)

    # ------------------------------------------------------------------ tính toán
    def run(self):
        try:
            cfg = self.get_data()
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                target, expr, samples = monte_carlo_custom(
                    cfg["formula"], {k: tuple(v) for k, v in cfg["variables"].items()},
                    n_trials=cfg["n_trials"], seed=cfg["seed"])
            finally:
                QApplication.restoreOverrideCursor()
            if samples.size < 2:
                raise ValueError("Hầu hết các lần thử cho giá trị không xác định (chia 0, log số âm…). "
                                 "Kiểm tra lại giá trị và σ của các biến.")
            st = summarize(samples, cfg["ci"])
        except ValueError as e:
            QMessageBox.warning(self, "Chưa tính được", str(e))
            return
        self.result = {"cfg": cfg, "target": target, "expr": expr, "samples": samples, "stats": st}
        self._draw()
        vals = [target, theme.fmt(st["mean"]), theme.fmt(st["std"], 3), theme.fmt(st["median"]),
                theme.fmt(st["lo"]), theme.fmt(st["hi"])]
        self.res_table.setRowCount(1)
        for j, v in enumerate(vals):
            self.res_table.setItem(0, j, num_item(v, Qt.AlignLeft if j == 0 else Qt.AlignRight))
        self.res_table.setHorizontalHeaderLabels(["Đại lượng", "Trung bình", "Độ lệch chuẩn", "Trung vị",
                                                  f"CI {st['ci']}% dưới", f"CI {st['ci']}% trên"])
        self.lbl_meta.setText(f"{st['n']:,} / {cfg['n_trials']:,} lần thử hợp lệ".replace(",", "."))
        self.window.on_result()

    def _draw_empty(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Nhập phương trình và nhấn \"Tính bằng Monte-Carlo\"",
                ha="center", va="center", color=theme.MUTED, transform=ax.transAxes)
        self.canvas.draw()

    def _draw(self):
        r = self.result
        st = r["stats"]
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        draw_hist(ax, r["samples"], st,
                  f"Phân bố Monte-Carlo của {r['target']}:  {r['cfg']['formula']}", r["target"])
        ax.text(0.99, 0.97,
                f"trung bình = {theme.fmt(st['mean'])}\n"
                f"độ lệch chuẩn = {theme.fmt(st['std'], 3)}\n"
                f"CI {st['ci']}%: [{theme.fmt(st['lo'])}, {theme.fmt(st['hi'])}]",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=theme.INK,
                linespacing=1.5)
        self.fig.text(0.99, 0.01, f"Vùng tô đậm: khoảng tin cậy {st['ci']}%   |   vạch đen: trung bình   |   "
                                  f"{r['cfg']['n_trials']:,} lần thử".replace(",", "."),
                      ha="right", va="bottom", fontsize=7.5, color=theme.MUTED)
        self.fig.subplots_adjust(left=0.04, right=0.985, top=0.9, bottom=0.16)
        self.canvas.draw()

    def rows(self):
        return [(self.result["target"], self.result["stats"], "")]

    def report(self):
        r = self.result
        cfg, st = r["cfg"], r["stats"]
        L = ["KẾT QUẢ MÔ PHỎNG MONTE-CARLO: PHƯƠNG TRÌNH TỰ NHẬP",
             f"Thời điểm: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
             f"Nguồn dữ liệu: {self.source or 'nhập trực tiếp'}", "",
             f"Phương trình nhập: {cfg['formula']}",
             f"Đã giải theo     : {r['target']} = {sp.sstr(r['expr'])}", "",
             "Biến đầu vào:", f"{'Tên':<10}{'Giá trị':>14}{'σ':>12}  Phân bố"]
        L += [f"{k:<10}{v[0]:>14.6g}{v[1]:>12.4g}  {v[2]}" for k, v in cfg["variables"].items()]
        L += ["", f"Số lần thử: {cfg['n_trials']}    Hợp lệ: {st['n']}    Seed: {cfg['seed']}", "",
              f"{r['target']} trung bình    = {st['mean']:.8g}",
              f"Độ lệch chuẩn     = {st['std']:.6g}",
              f"Trung vị          = {st['median']:.8g}",
              f"Khoảng tin cậy {st['ci']}% = [{st['lo']:.8g}, {st['hi']:.8g}]"]
        return "\n".join(L)


# ==========================================================================
# CỬA SỔ CHÍNH
# ==========================================================================

EXAMPLE = {
    "isotope": "C-14", "time_unit": "năm",
    "decay_data": {"t": [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000],
                   "N": [20153, 17801, 15753, 14008, 12318, 11001, 9720, 8549, 7526, 6724, 5977, 5322,
                         4604, 4077]},
    "sample": {"N0": 1000, "N0_sigma": 20, "N": 250, "N_sigma": 10},
    "n_trials": 100000, "seed": 2026, "ci": 95,
}


class MonteCarloWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Câu 6B — Định tuổi phóng xạ bằng mô phỏng Monte-Carlo")
        self.resize(1360, 860)

        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(20, 16, 20, 20)
        outer.setSpacing(6)

        self.btn_open = QPushButton("Mở dữ liệu…")
        self.btn_open.setToolTip("File .json đã lưu từ chương trình (dữ liệu đo hoặc phương trình),\n"
                                 "hoặc file .csv hai cột t, N cho bảng số đếm.")
        self.btn_open.clicked.connect(self.open_data)
        self.btn_example = QPushButton("Nạp ví dụ C-14")
        self.btn_example.clicked.connect(self.load_example)
        self.btn_save_data = QPushButton("Lưu dữ liệu…")
        self.btn_save_data.clicked.connect(self.save_data)
        self.btn_save_report = QPushButton("Lưu kết quả…")
        self.btn_save_report.clicked.connect(self.save_report)
        self.btn_save_image = QPushButton("Lưu đồ thị…")
        self.btn_save_image.clicked.connect(self.save_image)

        outer.addWidget(theme.header(
            "Định tuổi phóng xạ",
            "Ước lượng hằng số phân rã, chu kỳ bán rã và tuổi mẫu vật bằng mô phỏng Monte-Carlo",
            [self.btn_example, self.btn_open, self.btn_save_data, None, self.btn_save_report, self.btn_save_image],
        ))

        self.tabs = QTabWidget()
        self.tab_decay = DecayTab(self)
        self.tab_eq = EquationTab(self)
        self.tabs.addTab(self.tab_decay, "Dữ liệu đo")
        self.tabs.addTab(self.tab_eq, "Phương trình tự nhập")
        self.tabs.currentChanged.connect(lambda *_: self.on_result())
        outer.addWidget(self.tabs, 1)
        self.lbl_status = theme.label("", "Status")
        outer.addWidget(self.lbl_status)

        QShortcut(QKeySequence("Ctrl+Return"), self, activated=lambda: self.active().run())
        QShortcut(QKeySequence.Open, self, activated=self.open_data)

        self.tab_decay.set_data(EXAMPLE)
        self.on_result()
        self.tab_decay.btn_run.setFocus()

    def active(self):
        return self.tabs.currentWidget()

    def on_result(self):
        has = self.active().result is not None
        self.btn_save_report.setEnabled(has)
        self.btn_save_image.setEnabled(has)

    def load_example(self):
        self.tab_decay.set_data(EXAMPLE)
        self.tabs.setCurrentWidget(self.tab_decay)
        self.lbl_status.setText("Đã nạp ví dụ C-14. Nhấn \"Chạy mô phỏng\".")

    def open_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Mở dữ liệu / phương trình đã lưu", "",
                                              "Dữ liệu (*.json *.csv *.txt);;Tất cả file (*)")
        if not path:
            return
        try:
            if path.lower().endswith((".csv", ".txt")):
                self.tab_decay.load_csv(path)
                self.tabs.setCurrentWidget(self.tab_decay)
            else:
                with open(path, encoding="utf-8") as f:
                    cfg = json.load(f)
                if cfg.get("mode") == "custom" or "formula" in cfg:
                    self.tab_eq.set_data(cfg, path)
                    self.tabs.setCurrentWidget(self.tab_eq)
                elif "decay_data" in cfg:
                    self.tab_decay.set_data(cfg, path)
                    self.tabs.setCurrentWidget(self.tab_decay)
                else:
                    raise ValueError("File JSON không có \"decay_data\" (dữ liệu đo) hay \"formula\" (phương trình).")
        except (OSError, ValueError, TypeError, KeyError) as e:
            QMessageBox.critical(self, "Không đọc được file", str(e))
            return
        self.lbl_status.setText(f"Đã nạp {Path(path).name}.")

    def save_data(self):
        tab = self.active()
        try:
            cfg = tab.get_data()
        except ValueError as e:
            QMessageBox.warning(self, "Dữ liệu chưa hợp lệ", str(e))
            return
        name = "du_lieu_dinh_tuoi.json" if tab is self.tab_decay else "phuong_trinh.json"
        path, _ = QFileDialog.getSaveFileName(self, "Lưu dữ liệu / phương trình", name, "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except OSError as e:
            QMessageBox.critical(self, "Không lưu được", str(e))
            return
        tab.source = path
        self.lbl_status.setText(f"Đã lưu dữ liệu: {path}")

    def save_report(self):
        tab = self.active()
        if tab.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Lưu kết quả", "ket_qua_monte_carlo.txt",
                                              "Báo cáo văn bản (*.txt);;Bảng kết quả CSV (*.csv)")
        if not path:
            return
        try:
            if path.lower().endswith(".csv"):
                results_to_csv(tab.rows(), path)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(tab.report())
        except OSError as e:
            QMessageBox.critical(self, "Không lưu được", str(e))
            return
        self.lbl_status.setText(f"Đã lưu kết quả: {path}")

    def save_image(self):
        tab = self.active()
        if tab.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Lưu đồ thị", "do_thi_monte_carlo.png",
                                              "Ảnh PNG (*.png);;Ảnh vector SVG (*.svg);;PDF (*.pdf)")
        if not path:
            return
        try:
            tab.fig.savefig(path, dpi=200)
        except OSError as e:
            QMessageBox.critical(self, "Không lưu được", str(e))
            return
        self.lbl_status.setText(f"Đã lưu đồ thị: {path}")


def main():
    app = QApplication(sys.argv)
    theme.apply(app, "decay")
    win = MonteCarloWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
