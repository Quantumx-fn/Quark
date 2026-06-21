import os
import tempfile
import pytest
from qiskit.qasm2 import dumps
from quark.pairs import random_circuit
from quark.adapters import from_qasm, load, from_pennylane, from_cirq


def test_from_qasm_roundtrip():
    qc = random_circuit(2, 8, seed=0)
    qasm = dumps(qc)
    qc2 = from_qasm(qasm)
    assert qc.num_qubits == qc2.num_qubits
    assert len(qc.data) == len(qc2.data)


def test_load_from_file():
    qc = random_circuit(3, 10, seed=1)
    with tempfile.NamedTemporaryFile('w', suffix='.qasm', delete=False) as f:
        f.write(dumps(qc))
        path = f.name
    qc2 = load(path)
    os.unlink(path)
    assert qc2.num_qubits == 3


def test_from_pennylane_known_gates():
    qml = pytest.importorskip("pennylane")
    tape = qml.tape.QuantumScript([qml.Hadamard(0), qml.CNOT(wires=[0, 1])])
    qc = from_pennylane(tape)
    assert qc.num_qubits == 2
    assert [inst.operation.name for inst in qc.data] == ['h', 'cx']


def test_from_pennylane_raises_on_unknown_gate():
    qml = pytest.importorskip("pennylane")
    tape = qml.tape.QuantumScript([qml.Toffoli(wires=[0, 1, 2])])
    with pytest.raises(ValueError):
        from_pennylane(tape)


def test_from_cirq_known_gates():
    cirq = pytest.importorskip("cirq")
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit([cirq.H(q[0]), cirq.CNOT(q[0], q[1])])
    qc = from_cirq(c)
    assert qc.num_qubits == 2
    assert [inst.operation.name for inst in qc.data] == ['h', 'cx']


def test_from_cirq_raises_on_unknown_gate():
    cirq = pytest.importorskip("cirq")
    q = cirq.LineQubit.range(3)
    c = cirq.Circuit([cirq.TOFFOLI(q[0], q[1], q[2])])
    with pytest.raises(ValueError):
        from_cirq(c)
