import base64
import csv
import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'code'))
from plot import make_figures


def main(root):
    root = Path(root)
    entries = []
    for wd in (0, 1):
        for seed in range(3):
            path = root / 'runs' / f'wd{wd}-seed{seed}'
            summary = json.loads((path / 'summary.json').read_text())
            with (path / 'metrics.csv').open() as stream:
                final = {k: float(v) for k, v in list(csv.DictReader(stream))[-1].items()}
            assert summary['status'] == 'complete' and final['step'] == 20000
            assert abs(final['heldout_acc'] - summary['final']['heldout_acc']) < 1e-9
            entries.append({'run': path.name, 'weight_decay': float(wd), 'seed': seed,
                            'train_99_step': summary['first_train_99'],
                            'heldout_99_step': summary['first_heldout_99_sustained_5_evaluations'],
                            'gap_steps': summary['gap_steps'],
                            'final_train_accuracy': final['train_acc'],
                            'final_heldout_accuracy': final['heldout_acc'], 'seconds': final['seconds']})
    (root / 'results.json').write_text(json.dumps(entries, indent=2))
    make_figures(root)
    def inline(value):
        value = html.escape(value)
        value = re.sub(r'\[([^\]]+)\]\((https://[^\s)]+)\)', r'<a href="\2">\1</a>', value)
        return re.sub(r'`([^`]+)`', r'<code>\1</code>', value)
    sections = []
    for block in (root / 'EXPERIMENT.md').read_text(encoding='utf-8').split('\n\n'):
        if block.startswith('!['):
            name = block.split('](', 1)[1].rstrip(')')
            image = base64.b64encode((root / name).read_bytes()).decode()
            sections.append(f'<img src="data:image/png;base64,{image}" alt="график">')
        elif block.startswith('```'):
            sections.append('<pre>' + html.escape('\n'.join(block.splitlines()[1:-1])) + '</pre>')
        elif block.startswith('|'):
            rows = block.splitlines()
            table = ''.join('<tr>' + ''.join('<td>' + html.escape(cell.strip()) + '</td>' for cell in row.strip('|').split('|')) + '</tr>' for row in [rows[0], *rows[2:]])
            sections.append('<table>' + table + '</table>')
        elif block.strip():
            sections.append('<p>' + inline(block) + '</p>')
    report = '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>grokking</title><style>body{max-width:1000px;margin:32px auto;padding:0 20px;background:#faf9f6;color:#202b35;font:16px/1.7 system-ui}img{width:100%;height:auto}pre{overflow:auto}table{border-collapse:collapse;width:100%}td{padding:8px;text-align:left;border-bottom:1px solid #ddd}</style>' + ''.join(sections) + '</html>'
    (root / 'report.html').write_text(report, encoding='utf-8')
    print(json.dumps({'complete_runs': len(entries)}))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
