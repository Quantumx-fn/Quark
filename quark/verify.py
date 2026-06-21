import numpy as np
from qiskit.quantum_info import Operator


def equiv(qc1, qc2, atol=1e-8):
    if qc1.num_qubits != qc2.num_qubits:
        return False
    u1 = Operator(qc1).data
    u2 = Operator(qc2).data
    phase = None
    for i in range(u1.shape[0]):
        for j in range(u1.shape[1]):
            if abs(u2[i, j]) > 1e-12 and abs(u1[i, j]) > 1e-12:
                phase = u1[i, j] / u2[i, j]
                break
        if phase is not None:
            break
    if phase is None:
        return np.allclose(u1, u2, atol=atol)
    return np.allclose(u1, phase * u2, atol=atol)
