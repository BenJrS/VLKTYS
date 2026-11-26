# -*- coding: utf-8 -*-
"""
Giao diện dùng chung cho 2 ứng dụng (Câu 10A và Câu 6B).

- Nạp font IBM Plex Sans / Plex Mono (giấy phép SIL OFL, đi kèm trong thư mục fonts/).
- Bảng màu + stylesheet Qt (QSS); mỗi ứng dụng truyền vào một màu nhấn riêng.
- Kiểu đồ thị matplotlib đồng bộ với giao diện.
"""

import sys
from pathlib import Path

from PySide6.QtGui import QFontDatabase, QFont, QIcon
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

# ---------------------------------------------------------------- màu nền chung
PAPER = "#EEF1EF"      # nền cửa sổ (xám lạnh)
SURFACE = "#FFFFFF"    # nền khung nội dung
FIELD = "#F7F9F8"      # nền ô nhập
INK = "#1B2420"        # chữ chính
MUTED = "#5E6B66"      # chữ phụ
RULE = "#D3DAD6"       # đường kẻ, viền
GRID = "#E4E9E6"       # lưới đồ thị
WARN = "#B4261A"
WARN_SOFT = "#FBE9E7"

# ---------------------------------------------------------------- màu nhấn theo ứng dụng
ACCENTS = {
    # 10A: xanh lục của lớp phủ mạch in (solder mask)
    "circuit": {"main": "#17664C", "hover": "#10513C", "soft": "#DDEEE6"},
    # 6B: xanh của bức xạ Cherenkov
    "decay": {"main": "#2446C9", "hover": "#1B37A3", "soft": "#E2E7FA"},
}

SANS = "IBM Plex Sans"
MONO = "IBM Plex Mono"


def resource_dir() -> Path:
    """Thư mục chứa tài nguyên (font) — chạy từ mã nguồn hoặc từ bản đóng gói PyInstaller."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def load_fonts():
    for f in sorted((resource_dir() / "fonts").glob("*.otf")):
        QFontDatabase.addApplicationFont(str(f))


def mono_font(size=10) -> QFont:
    f = QFont(MONO)
    f.setPointSizeF(size)
    f.setStyleHint(QFont.Monospace)
    return f


def stylesheet(accent: str) -> str:
    a = ACCENTS[accent]
    return f"""
    QWidget {{
        color: {INK};
        font-family: "{SANS}";
        font-size: 13px;
    }}
    QMainWindow, QWidget#Root {{ background: {PAPER}; }}
    QFrame#Panel {{
        background: {SURFACE};
        border: 1px solid {RULE};
        border-radius: 6px;
    }}
    QFrame#Panel QLabel {{ background: transparent; }}
    QFrame#Rule {{ background: {RULE}; max-height: 1px; min-height: 1px; border: none; }}

    QLabel#Title {{ font-size: 18px; font-weight: 600; }}
    QLabel#Subtitle {{ color: {MUTED}; font-size: 13px; }}
    QLabel#Section {{ font-size: 14px; font-weight: 600; padding-top: 2px; }}
    QLabel#Hint {{ color: {MUTED}; font-size: 12px; }}
    QLabel#Status {{ color: {MUTED}; font-size: 12px; }}
    QLabel#Formula {{ font-family: "{MONO}"; font-size: 12px; color: {INK}; }}
    QLabel#Badge {{
        border-radius: 3px; padding: 2px 8px; font-size: 12px; font-weight: 500;
        background: {a['soft']}; color: {a['main']};
    }}
    QLabel#Badge[state="bad"] {{ background: {WARN_SOFT}; color: {WARN}; }}
    QLabel#Badge[state="idle"] {{ background: {PAPER}; color: {MUTED}; }}
    QLabel#Fact {{ font-family: "{MONO}"; font-size: 13px; }}
    QLabel#FactName {{ color: {MUTED}; font-size: 12px; }}

    QPushButton {{
        background: {SURFACE};
        border: 1px solid {RULE};
        border-radius: 4px;
        padding: 6px 12px;
    }}
    QPushButton:hover {{ border-color: {MUTED}; }}
    QPushButton:pressed {{ background: {FIELD}; }}
    QPushButton:focus {{ border: 1px solid {a['main']}; }}
    QPushButton:disabled {{ color: #A3ADA9; border-color: {GRID}; }}
    QPushButton#Primary {{
        background: {a['main']};
        color: white;
        border: 1px solid {a['main']};
        padding: 9px 16px;
        font-weight: 600;
    }}
    QPushButton#Primary:hover {{ background: {a['hover']}; }}
    QPushButton#Primary:focus {{ border: 2px solid {INK}; }}
    QPushButton#Quiet {{ border: none; color: {a['main']}; padding: 4px 6px; }}
    QPushButton#Quiet:hover {{ text-decoration: underline; }}

    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background: {FIELD};
        border: 1px solid {RULE};
        border-radius: 4px;
        padding: 4px 6px;
        min-height: 20px;
        selection-background-color: {a['soft']};
        selection-color: {INK};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border: 1px solid {a['main']};
        background: {SURFACE};
    }}
    QLineEdit#Equation {{ font-family: "{MONO}"; font-size: 15px; padding: 8px 10px; }}
    QLineEdit[invalid="true"] {{ border: 1px solid {WARN}; background: {WARN_SOFT}; }}
    QComboBox::drop-down {{ border: none; width: 18px; }}
    QComboBox QAbstractItemView {{
        background: {SURFACE}; border: 1px solid {RULE};
        selection-background-color: {a['soft']}; selection-color: {INK};
    }}

    QTableWidget {{
        background: {SURFACE};
        border: 1px solid {RULE};
        border-radius: 4px;
        gridline-color: {GRID};
        selection-background-color: {a['soft']};
        selection-color: {INK};
    }}
    QTableWidget:focus {{ border: 1px solid {a['main']}; }}
    QHeaderView::section {{
        background: {PAPER};
        color: {MUTED};
        border: none;
        border-right: 1px solid {GRID};
        border-bottom: 1px solid {RULE};
        padding: 4px 6px;
        font-size: 12px;
    }}
    QHeaderView {{ background: {SURFACE}; }}
    QTableCornerButton::section {{ background: {PAPER}; border: none; border-bottom: 1px solid {RULE}; }}

    QTabWidget::pane {{ border: none; }}
    QTabBar {{ qproperty-drawBase: 0; }}
    QTabBar::tab {{
        background: transparent;
        color: {MUTED};
        padding: 8px 2px;
        margin-right: 22px;
        border: none;
        border-bottom: 2px solid transparent;
        font-size: 14px;
    }}
    QTabBar::tab:selected {{ color: {INK}; border-bottom: 2px solid {a['main']}; font-weight: 600; }}
    QTabBar::tab:hover:!selected {{ color: {INK}; }}

    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {RULE}; border-radius: 3px; min-height: 24px; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle:horizontal {{ background: {RULE}; border-radius: 3px; min-width: 24px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QScrollArea {{ border: none; background: transparent; }}

    QSplitter::handle {{ background: transparent; }}
    QToolTip {{ background: {INK}; color: white; border: none; padding: 4px 6px; }}
    QMessageBox QLabel {{ font-size: 13px; }}
    """


def apply(app, accent: str):
    load_fonts()
    font = QFont(SANS)
    font.setPointSizeF(10)
    app.setFont(font)
    app.setStyle("Fusion")
    app.setStyleSheet(stylesheet(accent))
    icon = resource_dir() / "icons" / f"{accent}.png"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    setup_matplotlib(accent)


def setup_matplotlib(accent: str):
    import matplotlib
    from matplotlib import font_manager

    for f in (resource_dir() / "fonts").glob("IBMPlexSans-*.otf"):
        font_manager.fontManager.addfont(str(f))
    matplotlib.rcParams.update({
        "font.family": [SANS, "DejaVu Sans"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "medium",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9.5,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": "#AEB8B3",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.prop_cycle": matplotlib.cycler(color=[ACCENTS[accent]["main"], INK, MUTED]),
        "axes.unicode_minus": True,
        "mathtext.fontset": "custom",
        "mathtext.rm": SANS,
        "mathtext.it": SANS,
    })


# ---------------------------------------------------------------- khối giao diện nhỏ

def panel() -> tuple:
    """Khung nền trắng có viền. Trả về (frame, layout dọc bên trong)."""
    frame = QFrame()
    frame.setObjectName("Panel")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(16, 14, 16, 16)
    lay.setSpacing(10)
    return frame, lay


def label(text: str, kind: str = "", wrap=False) -> QLabel:
    lb = QLabel(text)
    if kind:
        lb.setObjectName(kind)
    lb.setWordWrap(wrap)
    return lb


def rule() -> QFrame:
    r = QFrame()
    r.setObjectName("Rule")
    return r


def set_badge(lb: QLabel, text: str, state: str = "ok"):
    lb.setText(text)
    lb.setProperty("state", state)
    lb.style().unpolish(lb)
    lb.style().polish(lb)


def header(title: str, subtitle: str, actions: list) -> QWidget:
    """Thanh tiêu đề: tên ứng dụng + mô tả bên trái, các nút thao tác file bên phải."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(4, 0, 4, 4)
    text = QVBoxLayout()
    text.setSpacing(2)
    text.addWidget(label(title, "Title"))
    text.addWidget(label(subtitle, "Subtitle"))
    lay.addLayout(text)
    lay.addStretch()
    for a in actions:
        if a is None:
            lay.addSpacing(14)
        else:
            lay.addWidget(a)
    return w


def fmt(x: float, sig: int = 6) -> str:
    """Định dạng số: dạng thường nếu vừa phải, dạng khoa học nếu quá lớn/nhỏ."""
    if x == 0 or (1e-3 <= abs(x) < 1e6):
        return f"{x:.{sig}g}"
    return f"{x:.{max(sig - 2, 2)}e}"
