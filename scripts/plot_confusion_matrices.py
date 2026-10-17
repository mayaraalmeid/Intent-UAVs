#!/usr/bin/env python3

import json, glob, os, re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path('results')
PLOT_DIR = RESULTS_DIR / 'plots'
PLOT_DIR.mkdir(exist_ok=True, parents=True)

LABELS = ['I0', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8']
LABEL_NAMES = {
    'I0': 'Neutral', 'I1': 'I1 Takeoff req.', 'I2': 'I2 Takeoff auth.',
    'I3': 'I3 In-flight', 'I4': 'I4 Landing req.', 'I5': 'I5 Landing auth.',
    'I6': 'I6 Ground conf.', 'I7': 'I7 Conflict', 'I8': 'I8 Pause',
}

CONFIGS = [
    ('Qwen2.5 14B',    'qwen2.5:14b',         'qwen2.5_14b_llm_metrics_*.json'),
    ('Ministral 8B',   'ministral-8b-latest', 'ministral-8b-latest_llm_metrics_*.json'),
    ('LLaMA 3.1 8B',   'llama3.1:8b',         'llama3.1_8b_llm_metrics_*.json'),
    ('DeepSeek-Flash', 'deepseek-flash',      'deepseek-flash_llm_metrics_*.json'),
]


def epoch_from_name(fn):
    m = re.search(r'(\d{9,})', fn)
    return int(m.group(1)) if m else None


def load_confusion(approach, glob_pat, state):
    reps = state['valid'][approach]
    ne = sorted(epoch_from_name(r['file']) for r in reps)[-1]
    files = {epoch_from_name(os.path.basename(f)): f
             for f in glob.glob(str(RESULTS_DIR / glob_pat))}
    _, best = min((abs(le - ne), le) for le in files if abs(le - ne) < 3000)
    d = json.load(open(files[best]))
    preds = d['predictions']
    acc = d.get('accuracy_pct')

    idx = {l: i for i, l in enumerate(LABELS)}
    mat = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for p in preds:
        gt, pred = p['gt'], p['pred']
        if gt in idx and pred in idx:
            mat[idx[gt], idx[pred]] += 1
    return mat, acc


def plot_confusion(mat, title, out_path):
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = mat / row_sums
    n = len(LABELS)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    for i in range(n):
        for j in range(n):
            color = 'white' if norm[i, j] > 0.6 else 'black'
            ax.text(j, i, str(int(mat[i, j])), ha='center', va='center',
                    color=color, fontsize=9)

    short_labels = [l if l != 'I0' else 'Neutral' for l in LABELS]
    ax.set_xticks(range(n)); ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    state = json.load(open(RESULTS_DIR / 'clean_state.json'))

    names = ['a_qwen2_5_14b', 'b_ministral_8b', 'c_llama3_1_8b', 'd_deepseek_flash']
    for name, (label, approach, glob_pat) in zip(names, CONFIGS):
        mat, acc = load_confusion(approach, glob_pat, state)
        out = PLOT_DIR / f'fig6_{name}.png'
        plot_confusion(mat, f'{label} ({acc:.2f}%)', out)
        print(f'Saved: {out}')


if __name__ == '__main__':
    main()
