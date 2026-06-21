import random
from qiskit import QuantumCircuit
from quark.pairs import (make_pair, random_circuit, rewrite_chain, insert_id,
                         cancel_consecutive, name_to_rotation, clifford_conjugation,
                         gate_algebra, push_pauli_cnot)
from quark.verify import equiv


def _names(qc):
    return tuple(inst.operation.name for inst in qc.data)


def test_random_circuit_qubits():
    qc = random_circuit(3, 12, seed=0)
    assert qc.num_qubits == 3


def test_pair_same_qubits():
    a, b = make_pair(3, 10, k=3, seed=1)
    assert a.num_qubits == b.num_qubits


def test_pair_equivalent_small():
    failures = []
    for s in range(8):
        a, b = make_pair(2, 8, k=4, seed=s)
        if not equiv(a, b):
            failures.append(s)
    assert not failures, f"rewrite changed semantics on seeds: {failures}"


def test_insert_id_lengthens():
    a = random_circuit(2, 6, seed=0)
    rng = random.Random(0)
    b = insert_id(a, rng)
    assert len(b.data) == len(a.data) + 2


def test_cancel_does_not_break():
    a = random_circuit(3, 10, seed=2)
    rng = random.Random(0)
    b = cancel_consecutive(a, rng)
    assert b.num_qubits == a.num_qubits


def _trigger_cases():
    c1 = QuantumCircuit(2); c1.s(0); c1.h(1); c1.cx(0, 1)      # name_to_rotation
    c2 = QuantumCircuit(2); c2.x(0); c2.h(1); c2.cx(0, 1)      # clifford_conjugation
    c3 = QuantumCircuit(2); c3.s(0); c3.swap(0, 1); c3.z(1)    # gate_algebra
    c4 = QuantumCircuit(2); c4.x(0); c4.cx(0, 1)               # push_pauli_cnot
    return [(name_to_rotation, c1), (clifford_conjugation, c2),
            (gate_algebra, c3), (push_pauli_cnot, c4)]


def test_new_rewrites_preserve_unitary_and_fire():
    for fn, qc in _trigger_cases():
        for s in range(8):
            b = fn(qc, random.Random(s))
            assert equiv(qc, b), f"{fn.__name__} broke equivalence (seed {s})"
        fired = any(_names(fn(qc, random.Random(s))) != _names(qc) for s in range(8))
        assert fired, f"{fn.__name__} never fired on its trigger circuit"


def test_full_pool_chain_preserves_unitary():
    failures = []
    for s in range(20):
        nq = 2 + (s % 2)
        a = random_circuit(nq, 10, seed=s)
        b = rewrite_chain(a, 6, random.Random(s))
        if not equiv(a, b):
            failures.append(s)
    assert not failures, f"rewrite chain broke equivalence on seeds: {failures}"
