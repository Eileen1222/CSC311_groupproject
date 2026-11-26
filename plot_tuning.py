"""Plot grouped-CV macro-F1 against the boundary hyperparameters.

Usage: python plot_tuning.py <experiment_dir> <output.png>
The experiment directory is the one written by train.py (needs cv_summary.csv).
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'

run, out = Path(sys.argv[1]), Path(sys.argv[2])
cv = pd.read_csv(run / 'cv_summary.csv')
cv = cv[cv.eligible]
fam, df = cv['config.family'], cv['config.df']

# kNN: the selected representation (min_df=3, text weight 2), all k.
knn = cv[(fam == 'knn') & (df == 3) & (cv['config.text_weight'] == 2)].sort_values('config.k')

# Softmax: the selected representation (min_df=5, rate 0.1), by penalty and update cap.
soft = cv[(fam == 'softmax') & (df == 5) & (cv['config.rate'] == 0.1)].copy()
soft['cap'] = soft['config.max_updates'].fillna(5000).astype(int)
order = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
tick = {p: i for i, p in enumerate(order)}
soft['x'] = soft['config.penalty'].map(tick)

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)


def style(ax, title, xlabel):
    ax.set_title(title, fontsize=9.5, color=INK, loc='left')
    ax.set_xlabel(xlabel)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


ax = axes[0]
x, y, sd = knn['config.k'], 100 * knn.macro_f1_mean, 100 * knn.macro_f1_sd
ax.errorbar(x, y, yerr=sd, color=BLUE, linewidth=2, marker='o', markersize=5,
            capsize=2.5, elinewidth=1)
ax.axvspan(15.5, 47, color=GRID, alpha=0.45, linewidth=0)
ax.text(31, 94.6, 'boundary check', ha='center', color=MUTED, fontsize=8)
best = knn.loc[knn.macro_f1_mean.idxmax()]
ax.scatter([best['config.k']], [100 * best.macro_f1_mean], s=110, facecolors='none',
           edgecolors=INK, linewidths=1.2, zorder=5)
ax.text(best['config.k'], 81.2, f"selected k={int(best['config.k'])}\n({100*best.macro_f1_mean:.1f})",
        ha='center', color=INK, fontsize=8)
ax.set_xticks(x)
ax.set_xlim(-1, 47)
ax.set_ylabel('CV macro-F1 (%)')
style(ax, 'kNN (min_df=3, text weight 2)', 'k (neighbours)')

ax = axes[1]
for cap, color, offset in ((5000, BLUE, -0.08), (20000, ORANGE, 0.08)):
    part = soft[soft.cap == cap].sort_values('x')
    ax.errorbar(part.x + offset, 100 * part.macro_f1_mean, yerr=100 * part.macro_f1_sd,
                color=color, linewidth=2, marker='o', markersize=5, capsize=2.5,
                elinewidth=1, label=f'{cap:,} updates')
ax.axvspan(-0.5, 1.5, color=GRID, alpha=0.45, linewidth=0)
ax.text(0.5, 94.6, 'boundary check', ha='center', color=MUTED, fontsize=8)
chosen = soft[(soft.cap == 5000) & (soft['config.penalty'] == 1e-4)].iloc[0]
ax.scatter([chosen.x - 0.08], [100 * chosen.macro_f1_mean], s=110, facecolors='none',
           edgecolors=INK, linewidths=1.2, zorder=5)
ax.text(2.0, 81.2, f"selected λ=1e-4\n({100*chosen.macro_f1_mean:.1f})",
        ha='center', color=INK, fontsize=8)
ax.set_xticks(range(len(order)), ['0', '1e-5', '1e-4', '1e-3', '1e-2', '1e-1'])
ax.set_xlim(-0.5, len(order) - 0.5)
ax.legend(frameon=False, fontsize=8, loc='lower right', labelcolor=INK)
style(ax, 'Softmax (min_df=5, learning rate 0.1)', 'L2 penalty λ')

axes[0].set_ylim(78, 96)
fig.tight_layout()
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200)
print(f'Wrote {out}')
