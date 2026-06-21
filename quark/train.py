import random
import click
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from .enc import CircuitEncoder
from .feat import encode_batch
from .pairs import make_pair, random_circuit, rewrite_chain, corrupt


class TripletDS(Dataset):
    def __init__(self, n_samples=1500, qmax=4, depth_range=(8, 24), k=4, seed=0, rewrites=None):
        self.items = []
        rng = random.Random(seed)
        for s in range(n_samples):
            n = rng.randint(2, qmax)
            d = rng.randint(*depth_range)
            a, p = make_pair(n, d, k=k, seed=seed * 1000 + s, rewrites=rewrites)
            n_neg = rng.randint(2, qmax)
            d_neg = rng.randint(*depth_range)
            neg = random_circuit(n_neg, d_neg, seed=seed * 7777 + s)
            twin = corrupt(a, random.Random(seed * 555 + s)) or neg
            self.items.append((a, p, neg, twin))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _collate(batch):
    a, p, n, t = zip(*batch)
    return (encode_batch(list(a)), encode_batch(list(p)),
            encode_batch(list(n)), encode_batch(list(t)))


def triplet(a, p, n, m=0.2):
    pd = (a - p).pow(2).sum(-1)
    nd = (a - n).pow(2).sum(-1)
    return torch.clamp(pd - nd + m, min=0).mean()


def info_nce(a, p, temp=0.07, extra_neg=None):
    labels = torch.arange(a.shape[0], device=a.device)
    pos = (a @ p.t()) / temp
    if extra_neg is not None and extra_neg.numel():
        logits = torch.cat([pos, (a @ extra_neg.t()) / temp], dim=1)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(pos.t(), labels)) / 2
    return (F.cross_entropy(pos, labels) + F.cross_entropy(pos.t(), labels)) / 2


def train(samples=1500, epochs=12, batch=32, lr=3e-4, qmax=4,
          depth_range=(8, 24), k=4, save_to=None, device='cpu', seed=0,
          rewrites=None, loss='triplet', hard_negatives=False):
    ds = TripletDS(n_samples=samples, qmax=qmax, depth_range=depth_range, k=k, seed=seed, rewrites=rewrites)
    n_val = max(50, len(ds) // 10)
    n_tr = len(ds) - n_val
    tr, val = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(seed))
    dl = DataLoader(tr, batch_size=batch, collate_fn=_collate, shuffle=True)
    vdl = DataLoader(val, batch_size=batch, collate_fn=_collate)

    enc = CircuitEncoder().to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    mine = hard_negatives and loss == 'infonce'
    from . import _style as ui
    click.echo()
    ui.rule(f'training · loss={loss} · hard_negatives={mine}')
    ui.kv('params', f'{enc.n_params():,}')
    ui.kv('schedule', f'{epochs} epochs', f'· batch {batch} · lr {lr:g}')

    def _step(batch):
        A, P, N, T = batch
        ea = enc(*[t.to(device) for t in A]); ep_ = enc(*[t.to(device) for t in P])
        if loss == 'infonce':
            extra = enc(*[t.to(device) for t in T]) if mine else None
            return info_nce(ea, ep_, extra_neg=extra)
        en = enc(*[t.to(device) for t in N])
        return triplet(ea, ep_, en)

    for ep in range(epochs):
        enc.train()
        ttot = 0; tn = 0
        for batch in dl:
            l = _step(batch)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()
            ttot += l.item(); tn += 1

        enc.train(False)
        vtot = 0; vn = 0
        with torch.no_grad():
            for batch in vdl:
                vtot += _step(batch).item(); vn += 1
        sched.step()
        click.echo('  ' + ui.paint(f'ep {ep:>2}', fg=ui.DIM)
                   + '   ' + ui.paint('train ', fg=ui.DIM) + f'{ttot / max(tn, 1):.4f}'
                   + ui.paint('    val ', fg=ui.DIM) + f'{vtot / max(vn, 1):.4f}')

    if save_to:
        torch.save(enc.state_dict(), save_to)
        ui.status(True, f'saved → {save_to}')
    return enc
