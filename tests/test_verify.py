from qiskit import QuantumCircuit
from quark.verify import equiv


def test_self():
    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1)
    assert equiv(qc, qc)


def test_different():
    a = QuantumCircuit(2); a.h(0); a.cx(0, 1)
    b = QuantumCircuit(2); b.x(0); b.cx(0, 1)
    assert not equiv(a, b)


def test_hh_is_identity():
    a = QuantumCircuit(2)
    a.h(0); a.cx(0, 1)
    b = QuantumCircuit(2)
    b.h(0); b.h(0); b.h(0); b.cx(0, 1)
    assert equiv(a, b)


def test_disjoint_commute():
    a = QuantumCircuit(2); a.h(0); a.x(1)
    b = QuantumCircuit(2); b.x(1); b.h(0)
    assert equiv(a, b)
