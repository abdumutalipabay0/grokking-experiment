import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
from train import Transformer, dataset


root = Path(sys.argv[1])
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
records = []
source_hash = hashlib.sha256((root / 'code' / 'train.py').read_bytes()).hexdigest()
accepted_source_hashes = {source_hash}
source_metadata = root / 'source.json'
if source_metadata.exists():
    metadata = json.loads(source_metadata.read_text())
    assert metadata['current_sha256'] == source_hash
    accepted_source_hashes.add(metadata['original_sha256'])
for seed in range(3):
    initial = []
    hashes = []
    for wd in (1, 0):
        run = root / 'runs' / f'wd{wd}-seed{seed}'
        config = json.loads((run / 'config.json').read_text())
        summary = json.loads((run / 'summary.json').read_text())
        assert config['source_sha256'] in accepted_source_hashes
        hashes.append(config['split_sha256'])
        with (run / 'metrics.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        initial.append({k: v for k, v in rows[0].items() if k != 'seconds'})
        assert len(rows) == 201 and int(rows[-1]['step']) == 20000
        x, y, train, heldout = dataset(config['p'], config['fraction'], config['split_seed'])
        saved = torch.load(run / 'split.pt', weights_only=True)
        assert torch.equal(saved['train'], train) and torch.equal(saved['heldout'], heldout)
        checkpoint = torch.load(run / 'final.pt', weights_only=False, map_location='cuda')
        assert checkpoint['step'] == 20000
        model = Transformer(config['p']).cuda().eval()
        model.load_state_dict(checkpoint['model'])
        with torch.no_grad():
            predicted = torch.cat([model(t.cuda()).argmax(1).cpu() for t in x.split(2048)])
        train_acc = (predicted[train] == y[train]).float().mean().item()
        heldout_acc = (predicted[heldout] == y[heldout]).float().mean().item()
        assert abs(train_acc - summary['final']['train_acc']) < 1e-7
        assert abs(heldout_acc - summary['final']['heldout_acc']) < 1e-7
        records.append({'run': run.name, 'train_accuracy': train_acc, 'heldout_accuracy': heldout_acc})
        del model, checkpoint
    assert initial[0] == initial[1], 'Paired runs must have identical initial metrics'
    assert hashes[0] == hashes[1]
result = {'status': 'passed', 'checks': ['reloaded accuracy', 'split indices', 'paired initial metrics', 'source hashes', '201 evaluations per run'], 'runs': records}
(root / 'verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
