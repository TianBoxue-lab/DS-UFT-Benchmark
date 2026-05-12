import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib.ticker as ticker

# ===== Unified style configuration =====
FONT_FAMILY = 'Arial'
FONT_SIZE_TICKS_X = 7
FONT_SIZE_TICKS_Y = 7
FONT_SIZE_LABELS = 7
FONT_SIZE_TITLE = 7
FONT_SIZE_LEGEND = 6
AXIS_COLOR = 'black'
AXIS_WIDTH = 0.8  # Border line width
DPI = 600
CM_TO_INCH = 0.393701  # Conversion factor from centimeters to inches

# ===== Fixed figure and plot area sizes (centimeters) =====
FIGURE_SIZE_CM = (5, 6)  # Figure size (width, height) in centimeters
FIGURE_SIZE = (FIGURE_SIZE_CM[0] * CM_TO_INCH, FIGURE_SIZE_CM[1] * CM_TO_INCH)  # Convert to inches

# Fixed plot area size (centimeters)
AXIS_SIZE_CM = (3.5, 3.5)  # Plot area width and height (width, height) in centimeters
AXIS_SIZE = (AXIS_SIZE_CM[0] * CM_TO_INCH, AXIS_SIZE_CM[1] * CM_TO_INCH)  # Convert to inches

# Fixed margins (centimeters)
MARGIN_LEFT_CM = 1.0
MARGIN_RIGHT_CM = 0.5
MARGIN_BOTTOM_CM = 1.0
MARGIN_TOP_CM = 0.5

# Convert to inches
MARGIN_LEFT = MARGIN_LEFT_CM * CM_TO_INCH
MARGIN_RIGHT = MARGIN_RIGHT_CM * CM_TO_INCH
MARGIN_BOTTOM = MARGIN_BOTTOM_CM * CM_TO_INCH
MARGIN_TOP = MARGIN_TOP_CM * CM_TO_INCH

# Scatter plot settings
SCATTER_MARKERSIZE = 5  # Point size
SCATTER_MARKEREDGE_WIDTH = 0  # Point edge circle thickness (set to 0 for no border)
SCATTER_ALPHA = 0.6  # Transparency

# Reference line settings
REFERENCE_LINE_WIDTH = 0.8  # x=y line thickness

# Configure PDF font embedding and ensure all text is pure black
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['text.color'] = 'black'
rcParams['axes.labelcolor'] = 'black'
rcParams['xtick.color'] = 'black'
rcParams['ytick.color'] = 'black'
rcParams['axes.edgecolor'] = 'black'

# ====== 1. Read data ======
files = [
    "../data/perplexity_pfam-ESM2.csv",
    "../data/perplexity_pfam-DS-UFT.csv",
    "../data/perplexity_clan-ESM2.csv",
    "../data/perplexity_clan-DS-UFT.csv",
    "../data/perplexity_combined-ESM2.csv",
    "../data/perplexity_combined-DS-UFT.csv"
]

dfs = []
for f in files:
    df = pd.read_csv(f)
    df = df.rename(columns=lambda x: x.strip().replace("/", ""))  # Fix column names

    if "pfam" in f:
        group = "Pfam"
    elif "clans" in f:
        group = "Clans"
    elif "combine" in f:
        group = "Combine"
    else:
        group = "Unknown"

    if "ori" in f:
        weight = "Ori"
    elif "fine-tune" in f:
        weight = "Fine-tune"
    else:
        weight = "Unknown"

    df["Group"] = group
    df["Weight"] = weight
    dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)

# Nature style light colors (only differentiate 0.5 / 0.9)
suffix_colors = {
    "0.5": "#9DD79D",  # Light green
    "0.9": "#9ABBF3",  # Light blue
}

for group in all_data["Group"].unique():
    fig = plt.figure(figsize=FIGURE_SIZE)
    fig_width, fig_height = FIGURE_SIZE
    axis_width, axis_height = AXIS_SIZE
    left = MARGIN_LEFT / fig_width
    bottom = MARGIN_BOTTOM / fig_height
    width = axis_width / fig_width
    height = axis_height / fig_height
    ax = fig.add_axes([left, bottom, width, height])

    # Turn off grid
    ax.grid(False)

    for suffix, color in suffix_colors.items():
        # Extract subset for this group + Pfam_id suffix
        subset = all_data[all_data["Group"] == group].copy()
        subset = subset[subset["Pfam_id"].astype(str).str.endswith(suffix)]

        if subset.empty:
            continue

        ori = subset[subset["Weight"] == "Ori"]["perplexity"].reset_index(drop=True)
        ft = subset[subset["Weight"] == "Fine-tune"]["perplexity"].reset_index(drop=True)
        n = min(len(ori), len(ft))
        if n == 0:
            continue

        ax.scatter(
            ori[:n],
            ft[:n],
            s=SCATTER_MARKERSIZE,            # Point size
            alpha=SCATTER_ALPHA,
            color=color,
            edgecolors='none',  # Remove point edge circles
            linewidths=SCATTER_MARKEREDGE_WIDTH,  # Point edge circle thickness
            label=f"Sequence Identity Threshold {suffix}"
        )

    # Automatically calculate axis range
    xlims = [0, 10]
    ylims = [1, 10000]

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    # x-axis ticks 0, 2, 4, 6, 8, 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    # Use log scale for y-axis
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.FixedLocator([1, 10, 100, 1000, 10000]))
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    # x=y reference line
    ax.plot(xlims, xlims, 'k--', linewidth=REFERENCE_LINE_WIDTH)

    # Reference line
    line_x = np.linspace(0, 10, 100)
    ax.plot(line_x, np.maximum(line_x, 1), 'k--', linewidth=REFERENCE_LINE_WIDTH)

    ax.set_xlabel("ESM-2 650M Perplexity", fontsize=FONT_SIZE_LABELS, fontname=FONT_FAMILY)
    ax.set_ylabel("DS-UFT Perplexity", fontsize=FONT_SIZE_LABELS, fontname=FONT_FAMILY)

    # Set tick labels
    for label in ax.get_xticklabels():
        label.set_fontname(FONT_FAMILY)
        label.set_fontsize(FONT_SIZE_TICKS_X)
        label.set_color(AXIS_COLOR)

    for label in ax.get_yticklabels():
        label.set_fontname(FONT_FAMILY)
        label.set_fontsize(FONT_SIZE_TICKS_Y)
        label.set_color(AXIS_COLOR)

    # Set border
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_edgecolor(AXIS_COLOR)
        spine.set_linewidth(AXIS_WIDTH)

    ax.tick_params(axis='both', direction='in', length=3)

    ax.legend(
        loc="upper left",
        frameon=False,
        prop={'family': FONT_FAMILY, 'size': FONT_SIZE_LEGEND}
    )

    # Save in four formats, keep transparent
    # fig.savefig(f"perplexity_scatter_{group}_suffix.svg", format='svg', dpi=DPI, transparent=True)
    # fig.savefig(f"perplexity_scatter_{group}_suffix.pdf", format='pdf', dpi=DPI, transparent=True)
    fig.savefig(f"perplexity_scatter_{group}_suffix.png", format='png', dpi=DPI, transparent=True)
    # fig.savefig(f"perplexity_scatter_{group}_suffix.tiff", format='tiff', dpi=DPI, transparent=True)
    plt.close()

print("Figure saved!")
