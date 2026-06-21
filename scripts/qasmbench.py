import argparse
import json
import os
import torch
from glob import glob

from quark.enc import CircuitEncoder, embed
from quark.adapters import load
from quark.search import hash_repr, sorted_repr, count_vec, collect_vocab, cosine_topk, hash_topk
from quark.eval_ import recall_at_k


def collect(root):
    files = sorted(glob(os.path.join(root, '*', '*.qasm')))
    pairs = {}
    for p in files:
        name = os.path.basename(p)
        base = name.replace('_transpiled.qasm', '.qasm').replace('.qasm', '')
        is_t = '_transpiled' in name
        if base not in pairs:
            pairs[base] = {}
        try:
            pairs[base]['transpiled' if is_t else 'orig'] = load(p)
        except Exception:
            pass
    out = []
    for k, v in pairs.items():
        if 'orig' in v and 'transpiled' in v:
            if v['orig'].num_qubits <= 8 and len(v['orig'].data) <= 128 and len(v['transpiled'].data) <= 128:
                out.append((k, v['orig'], v['transpiled']))
    return out


def drop_self(retrieved, self_idx):
    # each query's own identical copy is also in the library; it is never a valid
    # hit, so strip it before scoring (otherwise it ranks first and crowds out the
    # transpiled target, mechanically forcing recall@1 to zero).
    return [[j for j in row if j != s] for row, s in zip(retrieved, self_idx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/tmp/QASMBench/small')
    ap.add_argument('--weights', default='quark.pt')
    ap.add_argument('--out', default='qasmbench_results.json')
    args = ap.parse_args()

    triples = collect(args.root)
    print(f"loaded {len(triples)} (orig, transpiled) pairs from QASMBench/small")
    if not triples:
        print("nothing to bench. either file paths are wrong or all circuits exceed 128 gates / 8 qubits")
        return

    n = len(triples)
    queries = [orig for _, orig, _ in triples]
    library = [tp for _, _, tp in triples]   # transpiled forms: the retrieval targets
    truth = [[i] for i in range(n)]           # query i's target is library[i]
    self_idx = []                             # each query's own original also lives in the library
    for _, orig, _ in triples:
        self_idx.append(len(library))
        library.append(orig)

    enc = CircuitEncoder()
    enc.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True))
    q_emb = embed(enc, queries)
    l_emb = embed(enc, library)
    quark_top = drop_self(cosine_topk(q_emb, l_emb, k=20), self_idx)

    q_h = [hash_repr(c) for c in queries]
    l_h = [hash_repr(c) for c in library]
    hash_top = drop_self(hash_topk(q_h, l_h, k=20), self_idx)

    q_s = [sorted_repr(c) for c in queries]
    l_s = [sorted_repr(c) for c in library]
    sorted_top = drop_self(hash_topk(q_s, l_s, k=20), self_idx)

    vocab = collect_vocab(library + queries)
    l_cv = torch.tensor([count_vec(c, vocab) for c in library], dtype=torch.float32)
    q_cv = torch.tensor([count_vec(c, vocab) for c in queries], dtype=torch.float32)
    cv_top = drop_self(cosine_topk(q_cv, l_cv, k=20), self_idx)

    res = {}
    for K in (1, 5, 10):
        res[f'quark_r@{K}'] = recall_at_k(quark_top, truth, K)
        res[f'count_r@{K}'] = recall_at_k(cv_top, truth, K)
        res[f'hash_r@{K}'] = recall_at_k(hash_top, truth, K)
        res[f'sorted_r@{K}'] = recall_at_k(sorted_top, truth, K)

    print()
    print(f"  task: given original circuit, find its transpiled version in a library of {len(library)} circuits")
    print()
    print(f"  method   |  R@1   |  R@5   |  R@10")
    print(f"  ---------|--------|--------|-------")
    for m in ('hash', 'sorted', 'count', 'quark'):
        r1, r5, r10 = res[f'{m}_r@1'], res[f'{m}_r@5'], res[f'{m}_r@10']
        print(f"  {m:8s} | {r1:.3f}  | {r5:.3f}  | {r10:.3f}")
    print()

    with open(args.out, 'w') as f:
        json.dump({'n_pairs': len(triples), 'results': res}, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    main()
