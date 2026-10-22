import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BG = '#faf9f6'
COLORS = ['#147d86', '#cc713b', '#7d62ab']


def canvas():
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), facecolor=BG)
    fig.subplots_adjust(left=.015, right=.985, bottom=.035, top=.965, wspace=.09)
    for ax in axes:
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(which='both', length=0, labelbottom=False, labelleft=False)
        ax.yaxis.get_offset_text().set_visible(False)
        ax.xaxis.get_offset_text().set_visible(False)
        ax.set_xlim(0, 20000)
        ax.grid(axis='y', color='#dce1e2', alpha=.7, linewidth=.7)
    return fig, axes


def save(fig, destination, name):
    fig.savefig(destination / f'{name}.png', dpi=200)
    fig.savefig(destination / f'{name}.svg')
    svg = destination / f'{name}.svg'
    svg.write_text(re.sub(r'<!--.*?-->', '', svg.read_text(encoding='utf-8'), flags=re.S), encoding='utf-8', newline='\n')
    plt.close(fig)


def make_figures(root):
    root = Path(root)
    destination = root / 'figures'
    destination.mkdir(exist_ok=True)
    runs = []
    for seed in range(3):
        for wd in (1, 0):
            path = root / 'runs' / f'wd{wd}-seed{seed}' / 'metrics.csv'
            with path.open() as stream:
                rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
            runs.append((seed, wd, rows))
    fig, axes = canvas()
    for seed, wd, rows in runs:
        ax = axes[0 if wd else 1]
        steps = [r['step'] for r in rows]
        ax.plot(steps, [r['train_acc']*100 for r in rows], color=COLORS[seed], linestyle=(0, (4, 4)), alpha=.4, linewidth=1.1)
        ax.plot(steps, [r['heldout_acc']*100 for r in rows], color=COLORS[seed], linewidth=2.5)
    for ax in axes:
        ax.set_ylim(-2, 102)
        ax.set_yticks([0, 25, 50, 75, 100])
    save(fig, destination, 'learning_curves')
    fig, axes = canvas()
    for seed, wd, rows in runs:
        steps = [r['step'] for r in rows]
        style = '-' if wd else (0, (4, 4))
        axes[0].plot(steps, [r['heldout_loss'] for r in rows], color=COLORS[seed], linestyle=style, linewidth=1.6)
        axes[1].plot(steps, [r['embedding_top5_fourier_fraction']*100 for r in rows], color=COLORS[seed], linestyle=style, linewidth=1.8)
    axes[0].set_yscale('log')
    axes[1].set_ylim(0, 100)
    axes[1].set_yticks([0, 25, 50, 75, 100])
    save(fig, destination, 'diagnostics')


if __name__ == '__main__':
    import sys
    make_figures(sys.argv[1] if len(sys.argv) > 1 else '.')
