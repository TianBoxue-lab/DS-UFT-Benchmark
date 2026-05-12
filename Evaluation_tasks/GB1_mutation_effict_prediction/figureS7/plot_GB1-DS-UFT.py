import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib.ticker as ticker

# ===== 统一样式配置 =====
FONT_FAMILY = 'Arial'
FONT_SIZE_TICKS_X = 7
FONT_SIZE_TICKS_Y = 7
FONT_SIZE_LABELS = 7
FONT_SIZE_TITLE = 7
FONT_SIZE_LEGEND = 6
AXIS_COLOR = 'black'
AXIS_WIDTH = 0.8  # 边框线宽
DPI = 600
CM_TO_INCH = 0.393701  # 厘米转英寸的换算系数

# ===== 固定图片和绘图框尺寸 (centimeters) =====
FIGURE_SIZE_CM = (9, 6)  # 图片大小 (width, height) in centimeters
FIGURE_SIZE = (FIGURE_SIZE_CM[0] * CM_TO_INCH, FIGURE_SIZE_CM[1] * CM_TO_INCH)  # 转换为英寸

# 固定绘图框尺寸 (centimeters)
AXIS_SIZE_CM = (7.5, 4.5)  # 绘图框的宽和高 (width, height) in centimeters
AXIS_SIZE = (AXIS_SIZE_CM[0] * CM_TO_INCH, AXIS_SIZE_CM[1] * CM_TO_INCH)  # 转换为英寸

# 固定边距 (centimeters)
MARGIN_LEFT_CM = 1.0
MARGIN_RIGHT_CM = 0.5
MARGIN_BOTTOM_CM = 1.0
MARGIN_TOP_CM = 0.5

# 转换为英寸
MARGIN_LEFT = MARGIN_LEFT_CM * CM_TO_INCH
MARGIN_RIGHT = MARGIN_RIGHT_CM * CM_TO_INCH
MARGIN_BOTTOM = MARGIN_BOTTOM_CM * CM_TO_INCH
MARGIN_TOP = MARGIN_TOP_CM * CM_TO_INCH

# 配置PDF保存时嵌入字体, 确保所有文本颜色都是纯黑
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['text.color'] = 'black'
rcParams['axes.labelcolor'] = 'black'
rcParams['xtick.color'] = 'black'
rcParams['ytick.color'] = 'black'
rcParams['axes.edgecolor'] = 'black'

# =========================
# 1. 数据
# =========================
data = {
    "Metric": ["Spearman", "Pearson", r"R$^2$", "RMSE"],
    "ESM-2 650M": ["0.859±0.006", "0.843±0.007", "0.696±0.007", "0.703±0.008"],
    "DS-UFT-CL0041_09": ["0.861±0.004", "0.825±0.006", "0.653±0.015", "0.751±0.017"],
    "DS-UFT-PF00001_09": ["0.854±0.003", "0.813±0.005", "0.628±0.005", "0.778±0.005"],
    "DS-UFT-PF00076_05": ["0.865±0.004", "0.836±0.008", "0.679±0.017", "0.722±0.019"],
    "DS-UFT-PF00076_09": ["0.860±0.004", "0.836±0.003", "0.695±0.009", "0.704±0.010"],
    "DS-UFT-Combined_05": ["0.872±0.004", "0.865±0.004", "0.732±0.013", "0.660±0.016"],
}

df = pd.DataFrame(data)

# =========================
# 2. 拆分均值 ± 误差
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
# 3. 颜色（色盲友好 & 与你原图风格一致）
# =========================
colors = [
    "#8ECFC9",  # 浅青
    "#82B0D2",  # 浅蓝
    "#BEB8DC",  # 浅紫
    "#F2C99D",  # 浅橙
    "#C4A5DE",  # 淡紫
    "#9FD0A1",  # 灰绿（色盲友好）
]

# 柱状图边框粗细
BAR_EDGE_WIDTH = 0.4
ERROR_BAR_LINEWIDTH = 0.4  # error bar 线粗细
ERROR_BAR_CAP_THICK = 0.4  # error bar 帽子粗细

# =========================
# 3. 绘制分组柱状图
# =========================
x = np.arange(len(df["Metric"]))
bar_width = 0.13

# 使用固定绘图框尺寸创建图形
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
# 4. 图形细节设置
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

# 设置刻度标签字体和颜色
for label in ax.get_xticklabels():
    label.set_fontname(FONT_FAMILY)
    label.set_fontsize(FONT_SIZE_TICKS_X)
    label.set_color(AXIS_COLOR)

for label in ax.get_yticklabels():
    label.set_fontname(FONT_FAMILY)
    label.set_fontsize(FONT_SIZE_TICKS_Y)
    label.set_color(AXIS_COLOR)

# 设置边框
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_edgecolor(AXIS_COLOR)
    spine.set_linewidth(AXIS_WIDTH)

ax.tick_params(axis='both', direction='in', length=3)

# 保存为四种格式，保持透明
fig.savefig("plot_GB1-DS-UFT.svg", format='svg', dpi=DPI, transparent=True)
fig.savefig("plot_GB1-DS-UFT.pdf", format='pdf', dpi=DPI, transparent=True)
fig.savefig("plot_GB1-DS-UFT.png", format='png', dpi=DPI, transparent=True)
fig.savefig("plot_GB1-DS-UFT.tiff", format='tiff', dpi=DPI, transparent=True)
plt.close()

print("图已保存:")
print("  - plot_GB1.svg")
print("  - plot_GB1.pdf")
print("  - plot_GB1.png")
print("  - plot_GB1.tiff")
