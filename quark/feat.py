import math
import warnings
import torch

GATE_VOCAB = {
    'h': 1, 'x': 2, 'y': 3, 'z': 4,
    's': 5, 'sdg': 6,
    't': 7, 'tdg': 8,
    'sx': 9, 'sxdg': 10,
    'rx': 11, 'ry': 12, 'rz': 13,
    'cx': 14, 'cy': 15, 'cz': 16,
    'swap': 17,
}
PAD = 0
OTHER_1Q = 18
OTHER_2Q = 19
VOCAB = 20
MAX_QUBITS = 64
MAX_LEN = 128


def encode(qc, max_len=MAX_LEN):
    g = []
    q1 = []
    q2 = []
    p = []
    for inst in qc.data:
        op = inst.operation
        gid = GATE_VOCAB.get(op.name)
        if gid is None:
            gid = OTHER_2Q if len(inst.qubits) == 2 else OTHER_1Q
        if len(inst.qubits) == 2:
            a = qc.find_bit(inst.qubits[0]).index
            b = qc.find_bit(inst.qubits[1]).index
            q1.append(min(a + 1, MAX_QUBITS))
            q2.append(min(b + 1, MAX_QUBITS))
        else:
            a = qc.find_bit(inst.qubits[0]).index
            q1.append(min(a + 1, MAX_QUBITS))
            q2.append(0)
        try:
            theta = float(op.params[0]) if op.params else 0.0
            p.append([math.cos(theta), math.sin(theta)])
        except (TypeError, ValueError, IndexError):
            p.append([1.0, 0.0])
        g.append(gid)

    if len(g) > max_len:
        warnings.warn(
            f"circuit exceeds the {max_len}-gate context window and was truncated; "
            "embeddings lose information for circuits this long.",
            stacklevel=2,
        )
    g = g[:max_len]; q1 = q1[:max_len]; q2 = q2[:max_len]; p = p[:max_len]
    while len(g) < max_len:
        g.append(PAD); q1.append(0); q2.append(0); p.append([0.0, 0.0])

    return (
        torch.tensor(g, dtype=torch.long),
        torch.tensor(q1, dtype=torch.long),
        torch.tensor(q2, dtype=torch.long),
        torch.tensor(p, dtype=torch.float32),
    )


def encode_batch(qcs, max_len=MAX_LEN):
    G, Q1, Q2, P = [], [], [], []
    for qc in qcs:
        g, q1, q2, p = encode(qc, max_len)
        G.append(g); Q1.append(q1); Q2.append(q2); P.append(p)
    return torch.stack(G), torch.stack(Q1), torch.stack(Q2), torch.stack(P)


def pad_mask(g):
    return g == PAD
