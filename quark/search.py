import hashlib
from collections import Counter
import torch


def hash_repr(qc):
    parts = []
    for inst in qc.data:
        qs = ','.join(str(qc.find_bit(q).index) for q in inst.qubits)
        parts.append(f"{inst.operation.name}({qs})")
    return hashlib.md5(';'.join(parts).encode()).hexdigest()


def sorted_repr(qc):
    items = []
    for inst in qc.data:
        qs = '_'.join(str(qc.find_bit(q).index) for q in inst.qubits)
        items.append(f"{inst.operation.name}_{qs}")
    items.sort()
    return hashlib.md5(';'.join(items).encode()).hexdigest()


def count_vec(qc, vocab):
    c = Counter(inst.operation.name for inst in qc.data)
    return [c.get(k, 0) for k in vocab]


def collect_vocab(circuits):
    v = set()
    for qc in circuits:
        for inst in qc.data:
            v.add(inst.operation.name)
    return sorted(v)


def cosine_topk(query_vec, lib_mat, k):
    q = query_vec
    L = lib_mat
    qn = q / (q.norm(dim=-1, keepdim=True) + 1e-9)
    Ln = L / (L.norm(dim=-1, keepdim=True) + 1e-9)
    sims = qn @ Ln.t()
    return sims.topk(min(k, L.shape[0]), dim=-1).indices.tolist()


def hash_topk(q_hashes, lib_hashes, k):
    out = []
    for h in q_hashes:
        m = [i for i, x in enumerate(lib_hashes) if x == h]
        out.append(m[:k])
    return out
