import math


def _pennylane_wire_map(tape):
    # Honour the tape's own wire ordering when present, else fall back to
    # first-seen order. Works for integer, string, or arbitrary wire labels.
    wires = list(getattr(tape, 'wires', []) or [])
    if not wires:
        for op in tape.operations:
            for w in op.wires:
                if w not in wires:
                    wires.append(w)
    return {w: i for i, w in enumerate(wires)}


def from_pennylane(tape):
    from qiskit import QuantumCircuit
    widx = _pennylane_wire_map(tape)
    qc = QuantumCircuit(max(len(widx), 1))
    name_map = {
        'Hadamard': 'h', 'PauliX': 'x', 'PauliY': 'y', 'PauliZ': 'z',
        'S': 's', 'T': 't', 'SX': 'sx',
        'RX': 'rx', 'RY': 'ry', 'RZ': 'rz',
        'CNOT': 'cx', 'CZ': 'cz', 'SWAP': 'swap',
    }
    for op in tape.operations:
        nm = name_map.get(op.name)
        ws = [widx[w] for w in op.wires]
        if nm == 'h': qc.h(ws[0])
        elif nm == 'x': qc.x(ws[0])
        elif nm == 'y': qc.y(ws[0])
        elif nm == 'z': qc.z(ws[0])
        elif nm == 's': qc.s(ws[0])
        elif nm == 't': qc.t(ws[0])
        elif nm == 'sx': qc.sx(ws[0])
        elif nm == 'rx': qc.rx(float(op.parameters[0]), ws[0])
        elif nm == 'ry': qc.ry(float(op.parameters[0]), ws[0])
        elif nm == 'rz': qc.rz(float(op.parameters[0]), ws[0])
        elif nm == 'cx': qc.cx(ws[0], ws[1])
        elif nm == 'cz': qc.cz(ws[0], ws[1])
        elif nm == 'swap': qc.swap(ws[0], ws[1])
        else:
            raise ValueError(f"from_pennylane: unrecognized gate {op.name!r}; "
                             "export to OpenQASM and load with from_qasm() for full gate coverage")
    return qc


def _cirq_qubit_key(q):
    if hasattr(q, 'row') and hasattr(q, 'col'):
        return (0, q.row, q.col)
    if hasattr(q, 'x'):
        return (1, q.x, 0)
    return (2, str(q), '')


def from_cirq(circuit):
    from qiskit import QuantumCircuit
    qubits = sorted(circuit.all_qubits(), key=_cirq_qubit_key)
    qmap = {q: i for i, q in enumerate(qubits)}
    qc = QuantumCircuit(max(len(qubits), 1))
    for moment in circuit:
        for op in moment.operations:
            g = op.gate
            ws = [qmap[q] for q in op.qubits]
            nm = type(g).__name__
            if nm == 'HPowGate' and abs(getattr(g, 'exponent', 1) - 1) < 1e-9: qc.h(ws[0])
            elif nm == 'XPowGate': qc.rx(float(g.exponent) * math.pi, ws[0])
            elif nm == 'YPowGate': qc.ry(float(g.exponent) * math.pi, ws[0])
            elif nm == 'ZPowGate': qc.rz(float(g.exponent) * math.pi, ws[0])
            elif nm in ('CNotPowGate', 'CXPowGate'): qc.cx(ws[0], ws[1])
            elif nm == 'CZPowGate': qc.cz(ws[0], ws[1])
            elif nm == 'SwapPowGate': qc.swap(ws[0], ws[1])
            else:
                raise ValueError(f"from_cirq: unrecognized gate {nm!r}; "
                                 "export to OpenQASM and load with from_qasm() for full gate coverage")
    return qc


def from_qasm(text):
    # LEGACY_CUSTOM_INSTRUCTIONS makes the parser accept the full historical
    # qelib1 set (swap, sx, cy, ...), which QASMBench's transpiled circuits use.
    from qiskit.qasm2 import loads, LEGACY_CUSTOM_INSTRUCTIONS
    return loads(text, custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)


def load(path):
    with open(path) as f:
        return from_qasm(f.read())
