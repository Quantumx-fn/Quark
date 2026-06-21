from qiskit import QuantumCircuit
from quark.feat import encode, encode_batch, GATE_VOCAB, MAX_LEN, PAD


def test_basic_shape():
    qc = QuantumCircuit(3)
    qc.h(0); qc.cx(0, 1); qc.rz(0.5, 2)
    g, q1, q2, p = encode(qc)
    assert g.shape == (MAX_LEN,)
    assert g[0].item() == GATE_VOCAB['h']
    assert g[1].item() == GATE_VOCAB['cx']
    assert g[2].item() == GATE_VOCAB['rz']


def test_padding():
    qc = QuantumCircuit(2)
    qc.h(0)
    g, q1, q2, p = encode(qc)
    assert g[1].item() == PAD
    assert q1[1].item() == 0


def test_qubit_offset():
    qc = QuantumCircuit(2)
    qc.h(0)
    g, q1, q2, p = encode(qc)
    assert q1[0].item() == 1


def test_batch():
    qc1 = QuantumCircuit(2); qc1.h(0)
    qc2 = QuantumCircuit(3); qc2.cx(0, 1)
    g, q1, q2, p = encode_batch([qc1, qc2])
    assert g.shape == (2, MAX_LEN)
