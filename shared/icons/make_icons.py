# -*- coding: utf-8 -*-
"""Tạo biểu tượng ứng dụng (.png, .ico, .icns) cho 2 ứng dụng. Chạy một lần trước khi build."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

HERE = Path(__file__).resolve().parent


def base(color):
    fig = plt.figure(figsize=(10.24, 10.24), dpi=100)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.1, 0.1), 0.8, 0.8, boxstyle="round,pad=0,rounding_size=0.18",
                                fc=color, ec="none"))
    return fig, ax


def circuit():
    fig, ax = base("#17664C")
    pts = np.array([[0.3, 0.32], [0.7, 0.32], [0.5, 0.68]])
    for i in range(3):
        for j in range(i + 1, 3):
            ax.plot(*zip(pts[i], pts[j]), color="white", lw=22, solid_capstyle="round")
    ax.scatter(*pts.T, s=5200, color="white", zorder=3)
    ax.scatter(*pts.T, s=1500, color="#17664C", zorder=4)
    return fig


def decay():
    fig, ax = base("#2446C9")
    t = np.linspace(0.25, 0.78, 200)
    ax.plot(t, 0.24 + 0.5 * np.exp(-(t - 0.25) * 5.2), color="white", lw=24, solid_capstyle="round")
    tt = np.array([0.31, 0.45, 0.59, 0.73])
    ax.scatter(tt, 0.24 + 0.5 * np.exp(-(tt - 0.25) * 5.2) + np.array([0.035, -0.03, 0.03, -0.025]),
               s=1700, color="white", zorder=3)
    return fig


def export(fig, name):
    png = HERE / f"{name}.png"
    fig.savefig(png, transparent=True)
    plt.close(fig)
    img = Image.open(png).convert("RGBA")
    img.save(HERE / f"{name}.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    if sys.platform == "darwin":
        iconset = HERE / f"{name}.iconset"
        iconset.mkdir(exist_ok=True)
        for s in (16, 32, 128, 256, 512):
            img.resize((s, s), Image.LANCZOS).save(iconset / f"icon_{s}x{s}.png")
            img.resize((2 * s, 2 * s), Image.LANCZOS).save(iconset / f"icon_{s}x{s}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(HERE / f"{name}.icns")], check=True)
        for f in iconset.iterdir():
            f.unlink()
        iconset.rmdir()


if __name__ == "__main__":
    export(circuit(), "circuit")
    export(decay(), "decay")
    print("Đã tạo icon trong", HERE)
