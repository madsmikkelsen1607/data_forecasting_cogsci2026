"""Shared matplotlib style for paper figures: one consistent categorical
palette (fixed hue order, never cycled per-chart) and light, print-friendly
chrome. Hex values are the validated default palette from the dataviz
skill (references/palette.md) - swap here if the palette ever changes.
"""

import matplotlib.pyplot as plt
import pandas as pd
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


def dataframe_to_image(
    df: pd.DataFrame,
    path,
    col_labels: list | None = None,
    title: str | None = None,
    fontsize: int = 11,
) -> None:
    """Render a small, already-formatted DataFrame as a clean booktabs-style
    table image - ready to paste straight into a document. Round/rename
    columns and drop anything not paper-ready before calling this; it just
    renders whatever strings it's given."""
    set_paper_style()
    labels = col_labels if col_labels is not None else [str(c) for c in df.columns]
    n_rows, n_cols = df.shape

    fig_width = max(5.0, 1.4 * n_cols)
    fig_height = 0.42 * (n_rows + 1) + (0.35 if title else 0.1)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=fontsize + 2, loc="left", pad=8, weight="bold")

    table = ax.table(
        cellText=df.astype(str).values,
        colLabels=labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.8)
    table.auto_set_column_width(col=list(range(n_cols)))

    last_row = n_rows  # header is row 0, data rows are 1..n_rows
    for (row, _col), cell in table.get_celld().items():
        cell.set_facecolor(SURFACE)
        cell.set_edgecolor(INK_PRIMARY)
        cell.visible_edges = ""
        if row == 0:
            cell.set_text_props(weight="bold", color=INK_PRIMARY)
            cell.visible_edges = "TB"
            cell.set_linewidth(1.3)
        else:
            cell.set_text_props(color=INK_PRIMARY)
            if row == last_row:
                cell.visible_edges = "B"
                cell.set_linewidth(1.0)

    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.15, facecolor=SURFACE)
    plt.close(fig)
