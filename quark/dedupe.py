import os
import json
import torch
from collections import defaultdict

from .enc import CircuitEncoder, embed
from .adapters import load


def _walk(root):
    for dp, _, files in os.walk(root):
        for f in files:
            if f.endswith('.qasm'):
                yield os.path.join(dp, f)


def cluster(paths, weights, threshold=0.9, device='cpu', verify=False):
    enc = CircuitEncoder().to(device)
    sd = torch.load(weights, map_location=device, weights_only=True)
    enc.load_state_dict(sd)

    qcs = []
    kept_paths = []
    failed = []
    for p in paths:
        try:
            qcs.append(load(p))
            kept_paths.append(p)
        except Exception as e:
            failed.append((p, str(e)))

    if not qcs:
        return {'groups': [], 'failed': failed}

    embs = embed(enc, qcs, device=device)
    sims = embs @ embs.t()

    n = len(qcs)
    visited = [False] * n
    groups = []
    for i in range(n):
        if visited[i]:
            continue
        group = [i]
        visited[i] = True
        for j in range(i + 1, n):
            if visited[j]:
                continue
            if sims[i, j].item() >= threshold:
                if verify:
                    from .verify import equiv
                    if qcs[i].num_qubits == qcs[j].num_qubits and qcs[i].num_qubits <= 8:
                        if not equiv(qcs[i], qcs[j]):
                            continue
                group.append(j)
                visited[j] = True
        if len(group) > 1:
            groups.append([kept_paths[k] for k in group])

    return {'groups': groups, 'failed': failed, 'n_files': n}


def fmt_report(result):
    from . import _style as ui
    out = []
    g = result['groups']
    n = result.get('n_files', 0)
    if not g:
        out.append('  ' + ui.paint(f'no duplicate groups in {n} files', fg=ui.DIM))
    else:
        out.append('  ' + ui.paint(f'{len(g)} duplicate group(s)', fg=ui.ACCENT, bold=True)
                   + ui.paint(f' in {n} files', fg=ui.DIM))
        out.append('')
        for i, group in enumerate(g):
            out.append('  ' + ui.paint(f'group {i + 1}', bold=True)
                       + ui.paint(f'  ·  {len(group)} files', fg=ui.DIM))
            for j, p in enumerate(group):
                branch = '└─' if j == len(group) - 1 else '├─'
                out.append('    ' + ui.paint(branch, fg=ui.DIM) + ' ' + p)
            out.append('')
    if result['failed']:
        out.append('  ' + ui.paint(f"{len(result['failed'])} file(s) failed to parse:", fg=ui.WARN))
        for p, e in result['failed'][:10]:
            out.append('    ' + ui.paint('!', fg=ui.WARN) + f' {p}: ' + ui.paint(e, fg=ui.DIM))
    return "\n".join(out)
