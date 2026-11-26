# -*- mode: python ; coding: utf-8 -*-
# Build: chạy "pyinstaller --noconfirm CauB_MonteCarlo.spec" trong thư mục này.
# macOS -> dist/CauB_MonteCarlo.app (thư mục)  |  Windows -> dist/CauB_MonteCarlo.exe (một file duy nhất)
import sys
from pathlib import Path

NAME = "CauB_MonteCarlo"
ICON = "decay"
SHARED = Path(SPECPATH).parent / "shared"

# Các phần của Qt / thư viện không dùng tới -> bỏ để giảm dung lượng bản đóng gói
DROP = ("QtQuick", "QtQml", "QtVirtualKeyboard", "qtvirtualkeyboard", "QtPdf", "qpdf", "QtNetwork",
        "QtWebEngine", "Qt3D", "QtMultimedia", "QtCharts", "QtDataVisualization", "QtSql", "QtTest")

a = Analysis(
    ["main.py"],
    pathex=[str(SHARED)],
    datas=[(str(SHARED / "fonts"), "fonts"), (str(SHARED / "icons" / f"{ICON}.png"), "icons")],
    hiddenimports=["theme"],
    excludes=["tkinter", "IPython", "pytest", "scipy", "pandas", "PyQt5", "PyQt6", "PySide2"],
    noarchive=False,
)
a.binaries = [b for b in a.binaries if not any(k in b[0] for k in DROP)]
a.datas = [d for d in a.datas if not any(k in d[0] for k in DROP)]
pyz = PYZ(a.pure)

if sys.platform == "darwin":
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=NAME, console=False,
              icon=str(SHARED / "icons" / f"{ICON}.icns"))
    coll = COLLECT(exe, a.binaries, a.datas, name=NAME)
    app = BUNDLE(coll, name=f"{NAME}.app", icon=str(SHARED / "icons" / f"{ICON}.icns"),
                 bundle_identifier=f"vn.baitaplon.{NAME.lower()}",
                 info_plist={"NSHighResolutionCapable": True, "CFBundleShortVersionString": "2.0"})
else:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name=NAME, console=False, upx=False,
              icon=str(SHARED / "icons" / f"{ICON}.ico"))
