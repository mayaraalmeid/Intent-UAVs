#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PLOT_DIR = Path('results/plots')
PLOT_DIR.mkdir(exist_ok=True, parents=True)

DATA = [
    ('Reactive SSF\n(baseline)', 1.53, 141.3, 'baseline'),
    ('Qwen2.5 14B\n(table)',      1.96, 106.9, 'table'),
    ('Qwen2.5 14B\n(policy)',     1.26, 107.6, 'policy'),
    ('LLaMA 3.1 8B\n(table)',     0.68,  87.5, 'table'),
    ('LLaMA 3.1 8B\n(policy)',    0.92,  93.0, 'policy'),
    ('Ministral 8B\n(table)',     1.43,  98.1, 'table'),
    ('Ministral 8B\n(policy)',    0.89,  63.9, 'policy'),
    ('DeepSeek-Flash\n(table)',   0.86,  84.7, 'table'),
    ('DeepSeek-Flash\n(policy)',  0.77,  95.4, 'policy'),
]

COLORS = {'baseline': '#c0392b', 'table': '#4a7ab5', 'policy': '#2f9e6e'}

labels   = [d[0] for d in DATA]
losses   = [d[1] for d in DATA]
lats     = [d[2] for d in DATA]
colors   = [COLORS[d[3]] for d in DATA]

def make_bar(values, ylabel, fname, fmt='{:.2f}'):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.65)
    for rect, v in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(),
                 fmt.format(v), ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(values) * 1.15)

    handles = [plt.Rectangle((0,0),1,1, color=COLORS[k]) for k in ['baseline','table','policy']]
    ax.legend(handles, ['Reactive SSF baseline', 'Classify-then-map', 'Policy agent'],
              loc='upper right', fontsize=9, frameon=False)

    fig.tight_layout()
    out = PLOT_DIR / fname
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', out)

make_bar(losses, 'Packet Loss (%)', 'fig7_packet_loss.png')
make_bar(lats, 'Latency (ms)', 'fig7_latency.png', fmt='{:.1f}')
