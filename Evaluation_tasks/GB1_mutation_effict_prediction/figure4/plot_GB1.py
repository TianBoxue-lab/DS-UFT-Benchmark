import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib.ticker as ticker

# ===== Unified Style Configuration =====
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

# ===== 固定图片和绘图框尺寸 (centimeters) =====
FIGURE_SIZE_CM = (9, 6)  # Image size (width, height) in centimeters
FIGURE_SIZE = (FIGURE_SIZE_CM[0] * CM_TO_INCH, FIGURE_SIZE_CM[1] * CM_TO_INCH)  # Convert to inches

# Fixed axis size (centimeters)
AXIS_SIZE_CM = (7.5, 4.5)  # Axis width and height (width, height) in centimeters
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

# Configure PDF to embed fonts, ensure all text is pure black
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['text.color'] = 'black'
rcParams['axes.labelcolor'] = 'black'
rcParams['xtick.color'] = 'black'
rcParams['ytick.color'] = 'black'
rcParams['axes.edgecolor'] = 'black'

# =========================
# 1. Data
# =========================
data = {
    "Metric": ["Spearman", "Pearson", r"R$^2$", "RMSE"],
    "ESM-2 650M": ["0.859±0.006", "0.843±0.007", "0.696±0.007", "0.703±0.008"],
    "DS-UFT-CL0041": ["0.861±0.004", "0.825±0.006", "0.653±0.015", "0.751±0.017"],
    "ESM-2 650M + SFT": ["0.897±0.003", "0.893±0.005", "0.796±0.009", "0.576±0.013"],
    "DS-UFT-CL0041 + SFT": ["0.887±0.006", "0.867±0.014", "0.733±0.028", "0.659±0.034"],
}

df = pd.DataFrame(data)

# =========================
# 2. Split Mean ± Error
# =========================
methods = df.columns[1:]
values, errors = {}, {}

for method in methods:
    vals, errs = [], []
    for entry in df[method]:
        mean, err = entry.split("±")
        vals.append(float(mean))
        errs.append(float(err))
    values[method] = vals
    errors[method] = errs

# =========================
# 3. Colors (Colorblind-friendly & consistent with original style)
# =========================
colors = [
    "#8ECFC9",  # Light cyan
    "#82B0D2",  # Light blue
    "#BEB8DC",  # Light purple
    "#F2C99D",  # Light orange
]

# Bar border width
BAR_EDGE_WIDTH = 0.4
ERROR_BAR_LINEWIDTH = 0.4  # Error bar line width
ERROR_BAR_CAP_THICK = 0.4  # Error bar cap thickness

# =========================
# 3. Plot Grouped Bar Chart
# =========================
x = np.arange(len(df["Metric"]))
bar_width = 0.18

# Create figure with fixed axis size
fig = plt.figure(figsize=FIGURE_SIZE)
fig_width, fig_height = FIGURE_SIZE
axis_width, axis_height = AXIS_SIZE
left = MARGIN_LEFT / fig_width
bottom = MARGIN_BOTTOM / fig_height
width = axis_width / fig_width
height = axis_height / fig_height
ax = fig.add_axes([left, bottom, width, height])

for i, method in enumerate(methods):
    ax.bar(
        x + i * bar_width,
        values[method],
        bar_width,
        yerr=errors[method],
        capsize=1,
        label=method,
        color=colors[i],
        edgecolor="black",
        linewidth=BAR_EDGE_WIDTH,
        alpha=0.85,
        error_kw=dict(ecolor="#444444", lw=ERROR_BAR_LINEWIDTH, capthick=ERROR_BAR_CAP_THICK)
    )

# =========================
# 4. Figure Detail Settings
# =========================
ax.set_xticks(x + bar_width * (len(methods) - 1) / 2)
ax.set_xticklabels(df["Metric"], fontsize=FONT_SIZE_TICKS_X, fontname=FONT_FAMILY)
ax.set_ylabel("Performance", fontsize=FONT_SIZE_LABELS, fontname=FONT_FAMILY)
ax.set_ylim(0.50, 0.95)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))

# ax.set_title("Model Performance Comparison (SFT vs Un-SFT)", fontsize=FONT_SIZE_TITLE, fontname=FONT_FAMILY)

ax.legend(
    loc="upper right",
    frameon=False,
    prop={'family': FONT_FAMILY, 'size': FONT_SIZE_TICKS_Y},
    bbox_to_anchor=(0.98, 0.98)
)

# Set tick label font and color
for label in ax.get_xticklabels():
    label.set_fontname(FONT_FAMILY)
    label.set_fontsize(FONT_SIZE_TICKS_X)
    label.set_color(AXIS_COLOR)

for label in ax.get_yticklabels():
    label.set_fontname(FONT_FAMILY)
    label.set_fontsize(FONT_SIZE_TICKS_Y)
    label.set_color(AXIS_COLOR)

# Set borders
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_edgecolor(AXIS_COLOR)
    spine.set_linewidth(AXIS_WIDTH)

ax.tick_params(axis='both', direction='in', length=3)

# Save in four formats with transparency
fig.savefig("plot_GB1.svg", format='svg', dpi=DPI, transparent=True)
fig.savefig("plot_GB1.pdf", format='pdf', dpi=DPI, transparent=True)
fig.savefig("plot_GB1.png", format='png', dpi=DPI, transparent=True)
fig.savefig("plot_GB1.tiff", format='tiff', dpi=DPI, transparent=True)
plt.close()

print("Figure saved:")
print("  - plot_GB1.svg")
print("  - plot_GB1.pdf")
print("  - plot_GB1.png")
print("  - plot_GB1.tiff")
