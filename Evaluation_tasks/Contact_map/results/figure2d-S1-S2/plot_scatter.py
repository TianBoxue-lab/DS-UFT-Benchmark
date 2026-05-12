import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams

# ===== Unified style configuration =====
FONT_FAMILY = 'Arial'
FONT_SIZE_TICKS_X = 7
FONT_SIZE_TICKS_Y = 7
FONT_SIZE_LABELS = 7
FONT_SIZE_TITLE = 7
FONT_SIZE_TEXT = 6
AXIS_COLOR = 'black'
AXIS_WIDTH = 0.8  # Border line width
DPI = 600
CM_TO_INCH = 0.393701  # Conversion factor from centimeters to inches

# ===== Fixed figure and plot area sizes (centimeters) =====
FIGURE_SIZE_CM = (7, 6)  # Figure size (width, height) in centimeters
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

# x=y reference line settings
REFERENCE_LINE_WIDTH = 0.8  # x=y line thickness

# Configure PDF font embedding and ensure all text is pure black
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['text.color'] = 'black'
rcParams['axes.labelcolor'] = 'black'
rcParams['xtick.color'] = 'black'
rcParams['ytick.color'] = 'black'
rcParams['axes.edgecolor'] = 'black'

# Read protein information table and calculate length
protein_info = pd.read_csv('../human_proteome_info_with_select_pdb_chain_change_sequence.csv')
# Keep only rows where seq_st_full is not empty
protein_info = protein_info.dropna(subset=['seq_st_full'])
# Calculate length
protein_info['length'] = protein_info['seq_st_full'].str.len()

# 读取两个CSV
system = 'pfam' # pfam clans combine
ori = pd.read_csv(f'../data/contact_map_{system}_ori.csv')
fine_tune = pd.read_csv(f'../data/contact_map_{system}_fine-tune.csv')

# Link uniprot with uniprot_id and add length information
# Here uniprot corresponds to uniprot_id in protein_info
ori = ori.merge(protein_info[['uniprot_id', 'length']], left_on='uniprot', right_on='uniprot_id', how='left')
fine_tune = fine_tune.merge(protein_info[['uniprot_id', 'length']], left_on='uniprot', right_on='uniprot_id', how='left')

# 不需要的列
drop_cols = ['Model_name', 'Pfam_id', 'uniprot', 'uniprot_id', 'pdb_chain', 'length'] # 注意这里暂时不把length画成param

# Extract parameter columns
param_cols = [col for col in ori.columns if col not in drop_cols]

# Create folder for saving figures
os.makedirs(f'{system}', exist_ok=True)

heatmap_cmap = LinearSegmentedColormap.from_list(
    "soft_blue_orange",
    ["#9DD79D", "#9ABBF3"]
)

# Plot one figure for each parameter
for param in param_cols:
    # Filter out rows without length information
    mask = ori['length'].notna() & fine_tune['length'].notna()
    ori_valid = ori[mask]
    fine_tune_valid = fine_tune[mask]

    # 使用固定绘图框尺寸创建图形
    fig = plt.figure(figsize=FIGURE_SIZE)
    fig_width, fig_height = FIGURE_SIZE
    axis_width, axis_height = AXIS_SIZE
    left = MARGIN_LEFT / fig_width
    bottom = MARGIN_BOTTOM / fig_height
    width = axis_width / fig_width
    height = axis_height / fig_height
    ax = fig.add_axes([left, bottom, width, height])

    # Draw scatter plot
    sc = ax.scatter(
        ori_valid[param], fine_tune_valid[param],
        c=ori_valid['length'],   # Color by ori length
        cmap=heatmap_cmap,
        s=SCATTER_MARKERSIZE,  # Point size
        alpha=SCATTER_ALPHA,
        edgecolors='none',  # Remove point edge circles
        linewidths=SCATTER_MARKEREDGE_WIDTH,  # Point edge circle thickness
    )

    # 画 y = x 的黑色虚线
    min_val = min(ori_valid[param].min(), fine_tune_valid[param].min())
    max_val = max(ori_valid[param].max(), fine_tune_valid[param].max())
    ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=REFERENCE_LINE_WIDTH)

    # Calculate number of points above/below
    ori_better = sum(ori_valid[param] > fine_tune_valid[param])
    fine_tune_better = sum(fine_tune_valid[param] > ori_valid[param])

    # Annotate point counts
    ax.text(0.05, 0.95, f'ESM-2 650M better: {ori_better}', transform=ax.transAxes, fontsize=FONT_SIZE_TEXT, color='black', fontname=FONT_FAMILY)
    ax.text(0.05, 0.90, f'DS-UFT better: {fine_tune_better}', transform=ax.transAxes, fontsize=FONT_SIZE_TEXT, color='black', fontname=FONT_FAMILY)

    # Set labels
    param_title = param.replace('_', ' ')
    ax.set_xlabel(f'ESM-2 650M ({param_title})', fontsize=FONT_SIZE_LABELS, fontname=FONT_FAMILY)
    ax.set_ylabel(f'DS-UFT ({param_title})', fontsize=FONT_SIZE_LABELS, fontname=FONT_FAMILY)
    ax.set_title(param_title, fontsize=FONT_SIZE_TITLE, fontname=FONT_FAMILY)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # x and y axis ticks 0, 0.2, 0.4, 0.6, 0.8, 1.0
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

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
    ax.grid(False)

    # Set colorbar (use separate axes to avoid occupying main plot space)
    cbar_ax = fig.add_axes([left + width + 0.02, bottom, 0.03, height])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.locator = ticker.MultipleLocator(200)
    cbar.set_label('Sequence Length', fontsize=FONT_SIZE_LABELS, fontname=FONT_FAMILY)
    for label in cbar.ax.yaxis.get_ticklabels():
        label.set_fontname(FONT_FAMILY)
        label.set_fontsize(FONT_SIZE_TICKS_Y)
        label.set_color(AXIS_COLOR)

    # Save in four formats, keep transparent
    fig.savefig(f"{system}/{system}_{param}_scatter.svg", format='svg', dpi=DPI, transparent=True)
    fig.savefig(f"{system}/{system}_{param}_scatter.pdf", format='pdf', dpi=DPI, transparent=True)
    fig.savefig(f"{system}/{system}_{param}_scatter.png", format='png', dpi=DPI, transparent=True)
    fig.savefig(f"{system}/{system}_{param}_scatter.tiff", format='tiff', dpi=DPI, transparent=True)
    plt.close()

print(f'All plots saved in "{system}" folder.')
