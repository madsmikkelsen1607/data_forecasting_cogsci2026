"""Shared matplotlib style for paper figures: one consistent categorical
palette (fixed hue order, never cycled per-chart) and light, print-friendly
chrome. Hex values are the validated default palette from the dataviz
skill (references/palette.md) - swap here if the palette ever changes.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

METHOD_ORDER = ["seasonal_naive", "sarima", "lag_ml"]
METHOD_COLORS = dict(zip(METHOD_ORDER, CATEGORICAL))
METHOD_LABELS = {
    "seasonal_naive": "Seasonal naive",
    "sarima": "SARIMA",
    "lag_ml": "Lag-feature GBRT",
}

WEEKDAY_LABELS = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}

# Sequential single-hue ramp (blue, light -> dark) for magnitude/heatmap encodings.
SEQUENTIAL_BLUE_STEPS = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK_PRIMARY,
            "axes.titlecolor": INK_PRIMARY,
            "text.color": INK_PRIMARY,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "grid.color": GRIDLINE,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "figure.dpi": 120,
            "savefig.facecolor": SURFACE,
        }
    )


def store_color(i: int) -> str:
    return CATEGORICAL[i % len(CATEGORICAL)]


def sequential_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_BLUE_STEPS)


def comma_axis(ax, which: str = "y") -> None:
    fmt = FuncFormatter(lambda x, _pos: f"{x:,.0f}")
    if which in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)
    if which in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
