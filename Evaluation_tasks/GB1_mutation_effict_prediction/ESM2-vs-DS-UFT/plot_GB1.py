import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = {
    "Metric": ["Spearman", "Pearson", "MSE", "RMSE", "MAE", "R2"],
    "ESM-2 650M": ["0.859±0.006", "0.843±0.007", "0.494±0.011", "0.703±0.008", "0.491±0.008", "0.696±0.007"],
    "DS-UFT-CL0041_09": ["0.861±0.004", "0.825±0.006", "0.564±0.025", "0.751±0.017", "0.513±0.011", "0.653±0.015"],
    "DS-UFT-PF00001_09": ["0.854±0.003", "0.813±0.005", "0.605±0.008", "0.778±0.005", "0.514±0.004", "0.628±0.005"],
    "DS-UFT-PF00076_05": ["0.865±0.004", "0.836±0.008", "0.522±0.028", "0.722±0.019", "0.506±0.017", "0.679±0.017"],
    "DS-UFT-PF00076_09": ["0.860±0.004", "0.836±0.003", "0.496±0.015", "0.704±0.010", "0.487±0.008", "0.695±0.009"],
    "DS-UFT-combine_05": ["0.872±0.004", "0.865±0.004", "0.436±0.021", "0.660±0.016", "0.459±0.012", "0.732±0.013"],
}

df = pd.DataFrame(data)
methods = df.columns[1:]
values = {}
errors = {}

for method in methods:
    vals, errs = [], []
    for entry in df[method]:
        mean, err = entry.split("±")
        vals.append(float(mean))
        errs.append(float(err))
    values[method] = vals
    errors[method] = errs


colors = [
    "#8ECFC9",  # Light cyan
    "#82B0D2",  # Light blue
    "#BEB8DC",  # Light purple
    "#F2C99D",  # Light orange
    "#C4A5DE",  # Pale purple
    "#9FD0A1",  # Gray-green (colorblind-friendly)
]

# =========================
# 4. Plot grouped bar chart
# =========================
x = np.arange(len(df["Metric"]))
width = 0.13

fig, ax = plt.subplots(figsize=(12, 6))

for i, method in enumerate(methods):
    ax.bar(
        x + i * width,
        values[method],
        width,
        yerr=errors[method],
        capsize=3,
        label=method,
        color=colors[i],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
        error_kw=dict(ecolor="#444444", lw=1)
    )

ax.set_xticks(x + width * (len(methods) - 1) / 2)
ax.set_xticklabels(df["Metric"], fontsize=12, fontname='Arial')
ax.set_ylabel("Score", fontsize=12, fontname='Arial')
ax.set_ylim(bottom=0.3) 
ax.set_title("Model Performance Comparison", fontsize=14, fontname='Arial')

ax.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    frameon=False,
    fontsize=10
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"GB1.pdf", format='pdf', dpi=600, transparent=True)
plt.savefig(f"GB1.png", format='png', dpi=600, transparent=True)

