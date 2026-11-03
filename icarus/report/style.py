"""House style for every ICARUS graphic (Viktor Berg's data-story design system).

Warm off-white canvas, Inter, hairline solid grids, and color that carries
meaning: dark red = risk / bad news, green = healthy, gray = context, blue = the
AI-era comparison series. The palette passed the dataviz colorblind validator
(worst adjacent deutan dE 11.1, all >= 3:1 contrast on the surface).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from .. import config  # noqa: E402

SURFACE = "#fbfaf7"
INK = "#0b0b0b"
INK2 = "#45443f"
MUTED = "#7d7b74"
GRID = "#e4e2da"
AXIS = "#b9b7ae"
GOOD = "#12a150"
BAD = "#b3261e"
BLUE = "#2a78d6"
NEUTRAL = "#c9c6bc"
NEUTRAL_DK = "#8f8c83"
AUTHOR = "Viktor Berg"
DPI = 150

_FONTS = config.ROOT / "assets" / "fonts"
FONT = "DejaVu Sans"
if (_FONTS / "Inter-400.ttf").exists():
    for w in (400, 500, 600, 700):
        fm.fontManager.addfont(str(_FONTS / f"Inter-{w}.ttf"))
    FONT = "Inter"
plt.rcParams.update({"font.family": FONT, "axes.unicode_minus": False,
                     "text.parse_math": False})  # "$1.5T" is money, not LaTeX


def mix(hex_c: str, t: float) -> str:
    """Blend a color toward white by t (0..1)."""
    c = np.array([int(hex_c[i:i + 2], 16) for i in (1, 3, 5)])
    m = c + (255 - c) * t
    return "#" + "".join(f"{int(round(v)):02x}" for v in m)


BAD_TINT = mix(BAD, 0.86)
BLUE_TINT = mix(BLUE, 0.86)


class Frame:
    """A figure laid out in pixels (y down), with header, footer and plot slots."""

    def __init__(self, w: int = 1600, h: int = 1000, margin: int = 64):
        self.W, self.H, self.M = w, h, margin
        self.fig = plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI, facecolor=SURFACE)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)
        self.ax.axis("off")

    # px helpers
    def pt(self, px: float) -> float:
        return px * 72 / DPI

    def fx(self, px: float) -> float:
        return px / self.W

    def fy(self, px: float) -> float:
        return 1 - px / self.H

    def text(self, x, y, s, size=24, weight=400, color=INK, ha="left", va="top", linespacing=1.25, **kw):
        return self.ax.text(x, y, s, fontsize=self.pt(size), fontweight=weight, color=color,
                            ha=ha, va=va, linespacing=linespacing, **kw)

    def header(self, kicker: str, title: str, subtitle: str, title_size: int = 50, title_lines: int = 1) -> int:
        M = self.M
        self.text(M, 50, kicker.upper(), 20, 600, MUTED)
        self.text(M, 84, title, title_size, 700, INK)
        sub_top = 84 + title_lines * int(title_size * 1.12) + 18
        self.text(M, sub_top, subtitle, 25, 400, INK2)
        return sub_top + 25 * (subtitle.count("\n") + 1) * 1.3

    def footer(self, source: str) -> None:
        y = self.H - 64
        self.ax.plot([self.M, self.W - self.M], [y, y], color=GRID, lw=1)
        self.text(self.M, y + 16, source, 18, 400, MUTED)
        self.text(self.W - self.M, y + 14, AUTHOR, 21, 700, INK2, ha="right")

    def plot_axes(self, left: float, top: float, width: float, height: float):
        a = self.fig.add_axes([self.fx(left), self.fy(top + height), width / self.W, height / self.H])
        style_axes(a, self)
        return a

    def save(self, path, scale: float = 1.0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=DPI * scale, facecolor=SURFACE)
        plt.close(self.fig)
        Image.open(path).convert("RGB").save(path, optimize=True)


def style_axes(a, frame: Frame | None = None, size: int = 19) -> None:
    pt = (lambda px: px * 72 / DPI)
    a.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        a.spines[side].set_visible(False)
    a.spines["bottom"].set_color(AXIS)
    a.spines["bottom"].set_linewidth(1)
    a.grid(axis="y", color=GRID, lw=1, ls="-")
    a.set_axisbelow(True)
    a.tick_params(colors=MUTED, labelsize=pt(size), length=0, pad=8)
    a.tick_params(which="minor", length=0)
    for lab in a.get_xticklabels() + a.get_yticklabels():
        lab.set_fontfamily(FONT)


def legend_row(frame: Frame, y: float, items, x: float | None = None, gap: int = 40, size: int = 21) -> None:
    """items: [(color, label, 'line'|'box'|'dot'), ...] in one row."""
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    x = frame.M if x is None else x
    fp = FontProperties(family=FONT, size=frame.pt(size))
    for col, lab, kind in items:
        if kind == "line":
            frame.ax.plot([x, x + 28], [y, y], color=col, lw=3.5, solid_capstyle="round")
        elif kind == "dot":
            frame.ax.scatter([x + 14], [y], s=90, color=col, edgecolor=SURFACE, linewidth=2, zorder=5)
        else:
            frame.ax.add_patch(FancyBboxPatch((x + 3, y - 10), 20, 20, boxstyle="round,pad=0,rounding_size=4",
                                              fc=col, ec="none"))
        frame.text(x + 38, y, lab, size, 400, INK2, va="center")
        x += 38 + TextPath((0, 0), lab, prop=fp).get_extents().width * DPI / 72 + gap


def hbar(ax, x0, yc, length, thick, color, r=5, z=3):
    """Horizontal bar: rounded data end, square at the baseline; negative grows left."""
    if abs(length) < 1e-9:
        return
    s = 1 if length > 0 else -1
    L = abs(length)
    left = x0 if s > 0 else x0 - L
    if L <= 2 * r:
        ax.add_patch(Rectangle((left, yc - thick / 2), L, thick, color=color, lw=0, zorder=z))
        return
    ax.add_patch(FancyBboxPatch((left, yc - thick / 2), L, thick, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=color, ec="none", zorder=z))
    ax.add_patch(Rectangle((x0 if s > 0 else x0 - 2 * r, yc - thick / 2), 2 * r, thick, color=color, lw=0, zorder=z))


def vbar(ax, xc, y0, height, width, color, r=6, z=3):
    """Column from baseline y0 (px, y down) upward by height px, rounded top."""
    if height <= 0:
        return
    ax.add_patch(FancyBboxPatch((xc - width / 2, y0 - height), width, height,
                                boxstyle=f"round,pad=0,rounding_size={min(r, height / 2)}", fc=color, ec="none", zorder=z))
    ax.add_patch(Rectangle((xc - width / 2, y0 - min(2 * r, height)), width, min(2 * r, height), color=color, lw=0, zorder=z))


def arrow(ax, xy_from, xy_to, rad=0.0, color=INK):
    ax.annotate("", xy=xy_to, xytext=xy_from, zorder=8,
                arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.32", color=color, lw=2.2,
                                shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))


def pct(x: float | None, digits: int = 0) -> str:
    """Percent label that never rounds a probability to a false certainty."""
    if x is None or not np.isfinite(x):
        return "n/a"
    if digits == 0 and 0.995 <= x < 1:
        return ">99%"
    if digits == 0 and 0 < x < 0.005:
        return "<1%"
    return f"{100 * x:.{digits}f}%"
