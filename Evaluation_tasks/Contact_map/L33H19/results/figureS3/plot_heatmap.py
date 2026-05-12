#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams

# ===== Unified style configuration =====
FONT_FAMILY = 'Arial'
FONT_SIZE_TICKS = 6
FONT_SIZE_LABELS = 6
FONT_SIZE_TITLE = 6
FONT_SIZE_ANNOT = 4  # Heatmap internal number font size
AXIS_COLOR = 'black'
AXIS_WIDTH = 0.8
DPI = 600
CM_TO_INCH = 0.393701  # Conversion factor from centimeters to inches

# ===== Colorbar configuration =====
COLORBAR_WIDTH_CM = 0.2  # Colorbar width (centimeters)
COLORBAR_WIDTH = COLORBAR_WIDTH_CM * CM_TO_INCH  # Convert to inches
COLORBAR_TICK_START = 0.2  # Colorbar start value
COLORBAR_TICK_STEP = 0.1  # Colorbar tick interval
COLORBAR_LABEL = 'Mean Score'  # Colorbar label

# ===== Fixed figure and heatmap sizes (centimeters) =====
# Total figure size (width, height) in centimeters
FIGURE_SIZE_CM = (12, 6)
FIGURE_SIZE = (FIGURE_SIZE_CM[0] * CM_TO_INCH, FIGURE_SIZE_CM[1] * CM_TO_INCH)  # Convert to inches

# Heatmap length and width (excluding axis labels, colorbar, etc.)
HEATMAP_SIZE_CM = (6.0, 3)  # Heatmap length and width (width, height) in centimeters
HEATMAP_SIZE = (HEATMAP_SIZE_CM[0] * CM_TO_INCH, HEATMAP_SIZE_CM[1] * CM_TO_INCH)  # Convert to inches

# Fixed margins (centimeters)
MARGIN_LEFT_CM = 3.0
MARGIN_RIGHT_CM = 1.0
MARGIN_BOTTOM_CM = 2.0
MARGIN_TOP_CM = 0.5

# Convert to inches
MARGIN_LEFT = MARGIN_LEFT_CM * CM_TO_INCH
MARGIN_RIGHT = MARGIN_RIGHT_CM * CM_TO_INCH
MARGIN_BOTTOM = MARGIN_BOTTOM_CM * CM_TO_INCH
MARGIN_TOP = MARGIN_TOP_CM * CM_TO_INCH

# Configure PDF font embedding and ensure all text is pure black
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['text.color'] = 'black'
rcParams['axes.labelcolor'] = 'black'
rcParams['xtick.color'] = 'black'
rcParams['ytick.color'] = 'black'
rcParams['axes.edgecolor'] = 'black'

# ===== Configuration =====
csv_files = [
    "../data/contact_map_pfam_ori.csv",
    "../data/contact_map_pfam_fine-tune.csv",
    "../data/contact_map_clans_ori.csv",
    "../data/contact_map_clans_fine-tune.csv",
    "../data/contact_map_combine_ori.csv",
    "../data/contact_map_combine_fine-tune.csv",
]

# Output directory
out_dir = "."  # Current directory

# ===== Read & Statistics =====
stats = {}

for file in csv_files:
    df = pd.read_csv(file, sep="\t|,", engine="python")  # Automatically support tab or comma delimiter
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats[file] = {
        "mean": df[numeric_cols].mean(),
        "std": df[numeric_cols].std()
    }

# Convert to DataFrame
mean_df = pd.DataFrame({f: stats[f]["mean"] for f in stats}).T
std_df = pd.DataFrame({f: stats[f]["std"] for f in stats}).T

# Clean index names: remove prefixes and suffixes
def format_index_name(f):
    name = os.path.basename(f).replace("contact_map_", "").replace(".csv", "")
    if "ori" in name:
        name = name.replace("ori", "ESM-2")
    elif "fine-tune" in name:
        name = name.replace("fine-tune", "DS-UFT")
    parts = name.split("_")
    if len(parts) == 2:
        # Adjust order: Pre-trained first, data type after, connected with underscore
        data_type = parts[0].capitalize()
        if data_type == "Clans":
            data_type = "Clan"  # Specifically handle Clans -> Clan
        elif data_type == "Combine":
            data_type = "Combined"  
        name = f"{parts[1]} {data_type}"
    return name

# Set labels in specified order
desired_order = [
    "ESM-2 Pfam",
    "DS-UFT Pfam",
    "ESM-2 Clan",
    "DS-UFT Clan",
    "ESM-2 Combined",
    "DS-UFT Combined"
]

# Reindex and sort
mean_df.index = [format_index_name(f) for f in mean_df.index]
mean_df = mean_df.reindex(desired_order)
std_df.index = mean_df.index

# Control whether to show colorbar: True=show, False=hide
SHOW_COLORBAR = True

palette = sns.color_palette("pastel")  # Light color scheme

heatmap_cmap = LinearSegmentedColormap.from_list(
    "soft_blue_orange",
    ["#9DD79D", "#9ABBF3"]
)

# ===== Plot heatmap (mean) =====
# Create figure
fig = plt.figure(figsize=FIGURE_SIZE)
fig_width, fig_height = FIGURE_SIZE

# Calculate heatmap axes position
left = MARGIN_LEFT / fig_width
bottom = MARGIN_BOTTOM / fig_height
width = HEATMAP_SIZE[0] / fig_width
height = HEATMAP_SIZE[1] / fig_height

# Create heatmap axes
ax = fig.add_axes([left, bottom, width, height])

# Plot heatmap
if SHOW_COLORBAR:
    # Calculate colorbar axes position
    cbar_left = left + width + 0.02 / fig_width
    cbar_ax = fig.add_axes([cbar_left, bottom, COLORBAR_WIDTH / fig_width, height])
    
    sns.heatmap(
        mean_df,
        annot=True,
        fmt=".2f",
        cmap=heatmap_cmap,
        linewidths=0,  # Remove grid lines
        ax=ax,
        cbar_ax=cbar_ax,
        annot_kws={"size": FONT_SIZE_ANNOT, "color": "black", "fontname": FONT_FAMILY}
    )
    cbar_ax.set_ylabel(COLORBAR_LABEL, fontsize=FONT_SIZE_LABELS, fontname=FONT_FAMILY)
    # Set colorbar ticks
    vmin, vmax = ax.collections[0].get_clim()
    cbar_ticks = np.arange(COLORBAR_TICK_START, vmax, COLORBAR_TICK_STEP)
    cbar_ax.set_yticks(cbar_ticks)
    for label in cbar_ax.get_yticklabels():
        label.set_fontname(FONT_FAMILY)
        label.set_fontsize(FONT_SIZE_TICKS)
        label.set_color(AXIS_COLOR)
else:
    sns.heatmap(
        mean_df,
        annot=True,
        fmt=".2f",
        cmap=heatmap_cmap,
        linewidths=0,  # Remove grid lines
        ax=ax,
        cbar=False,  # Hide colorbar
        annot_kws={"size": FONT_SIZE_ANNOT, "color": "black", "fontname": FONT_FAMILY}
    )

# Set x-axis labels (rotated)
for label in ax.get_xticklabels():
    label.set_fontname(FONT_FAMILY)
    label.set_fontsize(FONT_SIZE_TICKS)
    label.set_color(AXIS_COLOR)
    label.set_rotation(45)
    label.set_ha('right')  # Right align to align label end to tick position
    label.set_rotation_mode('anchor')  # Rotate around anchor point

# Set y-axis labels
for label in ax.get_yticklabels():
    label.set_fontname(FONT_FAMILY)
    label.set_fontsize(FONT_SIZE_TICKS)
    label.set_color(AXIS_COLOR)

# Hide border
for _, spine in ax.spines.items():
    spine.set_visible(False)

heatmap_path = os.path.join(out_dir, "heatmap_mean")

# Save in four formats, keep transparent
fig.savefig(f"{heatmap_path}.svg", format='svg', dpi=DPI, transparent=True)
fig.savefig(f"{heatmap_path}.pdf", format='pdf', dpi=DPI, transparent=True)
fig.savefig(f"{heatmap_path}.png", format='png', dpi=DPI, transparent=True)
fig.savefig(f"{heatmap_path}.tiff", format='tiff', dpi=DPI, transparent=True)
plt.close()
