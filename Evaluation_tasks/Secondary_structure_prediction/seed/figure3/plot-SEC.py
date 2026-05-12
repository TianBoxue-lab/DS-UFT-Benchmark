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

# 柱状图边框粗细
BAR_EDGE_WIDTH = 0.4
ERROR_BAR_LINEWIDTH = 0.4  # error bar 线粗细
ERROR_BAR_CAP_THICK = 0.4  # error bar 帽子粗细
ERROR_BAR_CAP_SIZE = 1  # error bar 帽子尺寸

# 配置PDF保存时嵌入字体, 确保所有文本颜色都是纯黑
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['text.color'] = 'black'
rcParams['axes.labelcolor'] = 'black'
rcParams['xtick.color'] = 'black'
rcParams['ytick.color'] = 'black'
rcParams['axes.edgecolor'] = 'black'

# =========================
# 1. 构建数据
# =========================
# data = {
#     "Metric": ["Q3_Accuracy", "SOV", "Macro_F1-score", "Weighted_F1-score"],
#     "ESM-2 650M All": ["0.830±0.0004","0.825±0.0006","0.828±0.0003","0.830±0.0004"],
#     "DS-UFT All": ["0.818±0.0001","0.810±0.0003","0.815±0.0001","0.818±0.0001"],
#     "ESM-2 650M Part": ["0.846±0.0002","0.844±0.0015","0.840±0.0002","0.846±0.0002"],
#     "DS-UFT Part": ["0.818±0.0001"," 0.806±0.0014","0.812±0.0002","0.818±0.0001"]
# }

data = {
    "Metric": ["Q3 Accuracy", "SOV", "Macro F1-score", "Weighted F1-score"],
    "ESM-2 650M": ["0.846±0.0002","0.844±0.0015","0.840±0.0002","0.846±0.0002"],
    "DS-UFT": ["0.818±0.0001","0.806±0.0014","0.812±0.0002","0.818±0.0001"]
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
# 3. 浅色 + 色盲友好配色
# =========================
colors = [
    "#8ECFC9",  # 浅青
    "#82B0D2",  # 浅蓝
    "#BEB8DC",  # 浅紫
    "#F2C99D",  # 浅橙
]

# =========================
# 4. 绘制分组柱状图
# =========================
x = np.arange(len(df["Metric"]))
bar_width = 0.18

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
        capsize=ERROR_BAR_CAP_SIZE,
        label=method,
        color=colors[i % len(colors)],
        edgecolor="black",
        linewidth=BAR_EDGE_WIDTH,
        alpha=0.85,
        error_kw=dict(ecolor="#444444", lw=ERROR_BAR_LINEWIDTH, capthick=ERROR_BAR_CAP_THICK)
    )

# =========================
# 5. 坐标轴 & 图例
# =========================
ax.set_xticks(x + bar_width * (len(methods) - 1) / 2)
ax.set_xticklabels(df["Metric"], ha="center", fontsize=FONT_SIZE_TICKS_X, fontname=FONT_FAMILY)
ax.set_ylabel("Performance", fontsize=FONT_SIZE_LABELS, fontname=FONT_FAMILY)
ax.set_ylim(0.7, 0.90)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))

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

# =========================
# 6. 论文风格细节
# =========================
# 设置边框
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor(AXIS_COLOR)
    spine.set_linewidth(AXIS_WIDTH)

ax.tick_params(axis='both', direction='in', length=3)

# 保存图片
fig.savefig("plot-SEC.svg", format='svg', dpi=DPI, transparent=True, pad_inches=0)
fig.savefig("plot-SEC.pdf", format='pdf', dpi=DPI, transparent=True, pad_inches=0)
fig.savefig("plot-SEC.png", format='png', dpi=DPI, transparent=True, pad_inches=0)
fig.savefig("plot-SEC.tiff", format='tiff', dpi=DPI, transparent=True, pad_inches=0)
plt.close()

print("图已保存:")
print("  - plot-SEC.svg")
print("  - plot-SEC.pdf")
print("  - plot-SEC.png")
print("  - plot-SEC.tiff")
