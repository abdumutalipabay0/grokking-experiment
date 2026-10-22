import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class Transformer(nn.Module):


    def __init__(self, p=113, width=128, heads=4, hidden=512):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.embed = nn.Embedding(p + 1, width)
        self.pos = nn.Parameter(torch.empty(3, width))
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.mlp_in = nn.Linear(width, hidden, bias=False)
        self.mlp_out = nn.Linear(hidden, width, bias=False)
        self.unembed = nn.Linear(width, p, bias=False)
        for parameter in self.parameters():
            nn.init.normal_(parameter, std=0.8 / math.sqrt(width))

    def forward(self, tokens, full_reference=False):
        x = self.embed(tokens) + self.pos
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = [a.transpose(1, 2) for a in (q, k, v)]
        if full_reference:
            z = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            z = z.transpose(1, 2).reshape(b, n, d)
            x = x + self.out(z)
            x = x + self.mlp_out(F.relu(self.mlp_in(x)))
            return self.unembed(x[:, -1])
        z = F.scaled_dot_product_attention(q[:, :, -1:], k, v)
        x = x[:, -1] + self.out(z.transpose(1, 2).reshape(b, d))
        x = x + self.mlp_out(F.relu(self.mlp_in(x)))
        return self.unembed(x)


def dataset(p, fraction, split_seed):
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing='ij')
    x = torch.stack([a.flatten(), b.flatten(), torch.full((p*p,), p)], dim=1)
    y = (a.flatten() + b.flatten()) % p
    order = torch.randperm(p*p, generator=torch.Generator().manual_seed(split_seed))
    n = int(p*p*fraction)
    return x, y, order[:n], order[n:]


def validate():
    torch.manual_seed(0)
    x, y, train, test = dataset(113, .3, 2026)
    assert len(train) == 3830 and len(test) == 8939
    assert len(set(train.tolist()) & set(test.tolist())) == 0
    assert len(torch.unique(torch.cat([train, test]))) == 113**2
    assert torch.equal(y, (x[:, 0] + x[:, 1]) % 113)
    model = Transformer()
    fast = model(x[:17])
    reference = model(x[:17], full_reference=True)
    torch.testing.assert_close(fast, reference, atol=1e-6, rtol=1e-5)
    g1 = torch.autograd.grad(fast.square().sum(), tuple(model.parameters()))
    g2 = torch.autograd.grad(reference.square().sum(), tuple(model.parameters()))
    for a, b in zip(g1, g2):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    initial = F.cross_entropy(model(x[:32]), y[:32]).item()
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x[:32]), y[:32])
        loss.backward()
        optimizer.step()
    assert loss.item() < initial / 10
    print(json.dumps({'validation': 'passed', 'initial_loss': initial, 'final_loss': loss.item()}))


def main(args):
    if args.validate:
        torch.set_num_threads(2)
        validate()
        return
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.20)
    run = Path(args.output)
    run.mkdir(parents=True, exist_ok=False)
    x, y, train, test = dataset(args.p, args.fraction, args.split_seed)
    split_hash = hashlib.sha256(train.numpy().tobytes()).hexdigest()
    torch.save({'train': train, 'heldout': test}, run / 'split.pt')
    x, y, train, test = [t.to(device) for t in (x, y, train, test)]
    tx, ty = x[train], y[train]
    model = Transformer(args.p).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                 betas=(.9, .98), weight_decay=args.wd, fused=device.type == 'cuda')
    config = vars(args) | {'split_sha256': split_hash, 'parameters': sum(p.numel() for p in model.parameters()),
                          'torch': torch.__version__, 'python': platform.python_version(),
                          'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
                          'dtype': 'float32', 'tf32': False, 'train_n': len(train), 'heldout_n': len(test),
                          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          'architecture': '1L causal transformer; d_model=128; heads=4; d_mlp=512; ReLU; no LN, bias or dropout; init std=0.8/sqrt(128)'}
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    evaluate_model = model
    
    train_model = torch.compile(model) if args.compile else model
    start = time.monotonic()
    first_memorized = first_generalized = None
    streak = 0

    def checkpoint(name, step):
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'step': step, 'config': config, 'rng': torch.get_rng_state()}, run / f'{name}.pt')

    fields = ['step', 'seconds', 'train_loss', 'heldout_loss', 'train_acc', 'heldout_acc', 'weight_norm', 'embedding_top5_fourier_fraction']
    with (run / 'metrics.csv').open('w', newline='', buffering=1) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for step in range(args.steps + 1):
            if step % args.eval_every == 0 or step == args.steps:
                model.eval()
                with torch.no_grad():
                    logits = torch.cat([evaluate_model(chunk) for chunk in x.split(2048)])
                    power = torch.fft.rfft(model.embed.weight[:args.p], dim=0).abs().square().sum(1)[1:]
                    row = {'step': step, 'seconds': round(time.monotonic()-start, 3),
                           'train_loss': F.cross_entropy(logits[train], y[train]).item(),
                           'heldout_loss': F.cross_entropy(logits[test], y[test]).item(),
                           'train_acc': (logits[train].argmax(1) == y[train]).float().mean().item(),
                           'heldout_acc': (logits[test].argmax(1) == y[test]).float().mean().item(),
                           'weight_norm': sum(t.square().sum() for t in model.parameters()).sqrt().item(),
                           'embedding_top5_fourier_fraction': (power.topk(5).values.sum()/power.sum()).item()}
                writer.writerow(row)
                (run / 'latest.json').write_text(json.dumps(row, indent=2))
                if first_memorized is None and row['train_acc'] >= .99:
                    first_memorized = step
                    checkpoint('memorized', step)
                if row['heldout_acc'] >= .99:
                    streak += 1
                    if streak == 5 and first_generalized is None:
                        first_generalized = step - 4 * args.eval_every
                        checkpoint('generalized', step)
                else:
                    streak = 0
                if step % 500 == 0 or step == args.steps:
                    print(json.dumps(row), flush=True)
                checkpoint('latest', step)
            if step == args.steps:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(train_model(tx), ty)
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite training loss at step {step}')
            loss.backward()
            optimizer.step()
    checkpoint('final', args.steps)
    summary = {'status': 'complete', 'first_train_99': first_memorized,
               'first_heldout_99_sustained_5_evaluations': first_generalized,
               'gap_steps': first_generalized-first_memorized if first_generalized is not None and first_memorized is not None else None,
               'final': row, 'max_cuda_memory_mb': torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else None}
    (run / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--p', type=int, default=113)
    parser.add_argument('--fraction', type=float, default=.3)
    parser.add_argument('--split-seed', type=int, default=2026)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wd', type=float, default=1.)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--output', default='runs/wd1-seed0')
    main(parser.parse_args())
