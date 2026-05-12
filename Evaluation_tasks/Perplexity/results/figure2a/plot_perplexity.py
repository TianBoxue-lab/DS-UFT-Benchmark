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
FIGURE_SIZE_CM = (6, 5)  # Figure size (width, height) in centimeters
FIGURE_SIZE = (FIGURE_SIZE_CM[0] * CM_TO_INCH, FIGURE_SIZE_CM[1] * CM_TO_INCH)  # Convert to inches

# Fixed plot area size (centimeters)
AXIS_SIZE_CM = (4.2, 3.5)  # Plot area width and height (width, height) in centimeters
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

# Box plot line width settings
BOX_EDGE_WIDTH = 0.4  # Box edge thickness
WHISKER_LINEWIDTH = 0.4  # Whisker line thickness
MEDIAN_LINEWIDTH = 0.4  # Median line thickness
CAP_LINEWIDTH = 0.4  # Cap line thickness
FLIER_LINEWIDTH = 0.2  # Outlier (circle) border thickness
FLIER_MARKERSIZE = 2  # Outlier (circle) size

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
    df = pd.read_csv(f)  # Change to sep="," if comma-delimited
    df = df.rename(columns=lambda x: x.strip().replace("/", ""))  # Fix column names

    if "pfam" in f:
        group = "Pfam"
    elif "clans" in f:
        group = "Clan"
    elif "combine" in f:
        group = "Combined"
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

# ====== 2. Statistical metrics ======
summary = (
    all_data
    .groupby(["Group", "Weight"])["perplexity"]
    .describe()  # count, mean, std, min, 25%, 50%, 75%, max
    .reset_index()
)

print("\n=== Summary statistics (per group + weight) ===")
print(summary)

# Save as CSV
summary.to_csv("perplexity_summary.csv", index=False)



# -------- Overall box plot --------
fig = plt.figure(figsize=FIGURE_SIZE)
fig_width, fig_height = FIGURE_SIZE
axis_width, axis_height = AXIS_SIZE
left = MARGIN_LEFT / fig_width
bottom = MARGIN_BOTTOM / fig_height
width = axis_width / fig_width
height = axis_height / fig_height
ax = fig.add_axes([left, bottom, width, height])

sns.boxplot(
    data=all_data, x="Group", y="perplexity", hue="Weight",
    palette=["#9DD79D", "#9ABBF3"],
    linewidth=BOX_EDGE_WIDTH,
    flierprops={'mew': FLIER_LINEWIDTH, 'markersize': FLIER_MARKERSIZE},
    whiskerprops={'linewidth': WHISKER_LINEWIDTH},
    medianprops={'linewidth': MEDIAN_LINEWIDTH},
    capprops={'linewidth': CAP_LINEWIDTH},
    ax=ax
)
ax.set_yscale("log")
# ax.set_title("Perplexity Distribution by Group and Weight (Boxplot, log-scale)")
ax.set_ylabel("Perplexity", fontsize=FONT_SIZE_LABELS, fontname=FONT_FAMILY)
ax.set_xlabel("", fontsize=FONT_SIZE_LABELS, fontname=FONT_FAMILY)

# Modify legend labels
handles, labels = ax.get_legend_handles_labels()
labels = ['ESM-2 650M', 'DS-UFT']
ax.legend(handles=handles, labels=labels, frameon=False, prop={'family': FONT_FAMILY, 'size': FONT_SIZE_LEGEND},
          loc='upper right', bbox_to_anchor=(0.98, 0.98))

# Set y-axis range and ticks (10^0, 10^1, 10^2, 10^3, 10^4)
ax.set_ylim(1, 10000)
ax.set_yscale("log")
ax.yaxis.set_major_locator(ticker.FixedLocator([1, 10, 100, 1000, 10000]))
ax.yaxis.set_minor_locator(ticker.NullLocator())  # Do not display minor tick lines

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

# Save in four formats, keep transparent
# fig.savefig(f"perplexity_boxplot_overall.svg", format='svg', dpi=DPI, transparent=True)
# fig.savefig(f"perplexity_boxplot_overall.pdf", format='pdf', dpi=DPI, transparent=True)
fig.savefig(f"perplexity_boxplot_overall.png", format='png', dpi=DPI, transparent=True)
# fig.savefig(f"perplexity_boxplot_overall.tiff", format='tiff', dpi=DPI, transparent=True)
plt.close()
