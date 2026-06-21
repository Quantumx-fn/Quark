import torch
from quark.search import hash_repr, sorted_repr, count_vec, cosine_topk, hash_topk, collect_vocab
from quark.pairs import random_circuit


def test_hash_str():
    qc = random_circuit(2, 5, seed=0)
    h = hash_repr(qc)
    assert isinstance(h, str) and len(h) == 32


def test_sorted_invariant_to_disjoint_reorder():
    from qiskit import QuantumCircuit
    a = QuantumCircuit(3); a.h(0); a.x(1)
    b = QuantumCircuit(3); b.x(1); b.h(0)
    assert sorted_repr(a) == sorted_repr(b)


def test_count_vec():
    qc = random_circuit(3, 10, seed=0)
    v = count_vec(qc, vocab=['h', 'x', 'cx', 'rz'])
    assert len(v) == 4


def test_cosine_topk():
    lib = torch.eye(5)
    q = torch.tensor([[1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0]])
    top = cosine_topk(q, lib, k=1)
    assert top[0][0] == 0
    assert top[1][0] == 1
