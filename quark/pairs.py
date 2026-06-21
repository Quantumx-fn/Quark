import math
import random
from qiskit import QuantumCircuit

from .verify import equiv

_INVERSE = {
    's': 'sdg', 'sdg': 's', 't': 'tdg', 'tdg': 't',
    'sx': 'sxdg', 'sxdg': 'sx', 'cx': 'cy', 'cy': 'cx', 'x': 'y', 'y': 'x',
}


ONE_Q = ['h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg', 'sx']
ROT = ['rx', 'ry', 'rz']
TWO_Q = ['cx', 'cy', 'cz', 'swap']


def random_circuit(n, depth, seed=None):
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for _ in range(depth):
        r = rng.random()
        if n >= 2 and r < 0.4:
            a, b = rng.sample(range(n), 2)
            getattr(qc, rng.choice(TWO_Q))(a, b)
        elif r < 0.55:
            getattr(qc, rng.choice(ROT))(rng.uniform(0, 2 * math.pi), rng.randrange(n))
        else:
            getattr(qc, rng.choice(ONE_Q))(rng.randrange(n))
    return qc


def _qidx(qc, q):
    return qc.find_bit(q).index


def insert_id(qc, rng):
    n = qc.num_qubits
    pos = rng.randrange(len(qc.data) + 1) if qc.data else 0
    out = QuantumCircuit(n)

    def emit_pair():
        kind = rng.choice(['h', 'x', 'cx']) if n >= 2 else rng.choice(['h', 'x'])
        if kind == 'h':
            q = rng.randrange(n)
            out.h(q); out.h(q)
        elif kind == 'x':
            q = rng.randrange(n)
            out.x(q); out.x(q)
        else:
            a, b = rng.sample(range(n), 2)
            out.cx(a, b); out.cx(a, b)

    for i, inst in enumerate(qc.data):
        if i == pos:
            emit_pair()
        out.append(inst.operation, inst.qubits, inst.clbits)
    if pos == len(qc.data):
        emit_pair()
    return out


def cancel_consecutive(qc, rng):
    n = qc.num_qubits
    out = QuantumCircuit(n)
    instrs = list(qc.data)
    skip = -1
    for i, inst in enumerate(instrs):
        if i == skip:
            continue
        if i + 1 < len(instrs):
            nxt = instrs[i + 1]
            if (inst.operation.name == nxt.operation.name
                    and inst.operation.name in ('h', 'x', 'y', 'z', 'cx', 'cz')
                    and tuple(inst.qubits) == tuple(nxt.qubits)):
                skip = i + 1
                continue
        out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def commute_swap(qc, rng):
    n = qc.num_qubits
    instrs = list(qc.data)
    if len(instrs) < 2:
        return qc.copy()
    cands = []
    for i in range(len(instrs) - 1):
        a = instrs[i]; b = instrs[i + 1]
        qa = {_qidx(qc, q) for q in a.qubits}
        qb = {_qidx(qc, q) for q in b.qubits}
        if not (qa & qb):
            cands.append(i)
    if not cands:
        return qc.copy()
    i = rng.choice(cands)
    instrs[i], instrs[i + 1] = instrs[i + 1], instrs[i]
    out = QuantumCircuit(n)
    for inst in instrs:
        out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def merge_rz(qc, rng):
    n = qc.num_qubits
    out = QuantumCircuit(n)
    instrs = list(qc.data)
    i = 0
    while i < len(instrs):
        cur = instrs[i]
        if (i + 1 < len(instrs)
                and cur.operation.name == 'rz'
                and instrs[i + 1].operation.name == 'rz'
                and tuple(cur.qubits) == tuple(instrs[i + 1].qubits)):
            theta = float(cur.operation.params[0]) + float(instrs[i + 1].operation.params[0])
            out.rz(theta, _qidx(qc, cur.qubits[0]))
            i += 2
        else:
            out.append(cur.operation, cur.qubits, cur.clbits)
            i += 1
    return out


def split_rz(qc, rng):
    instrs = list(qc.data)
    rz_idx = [i for i, inst in enumerate(instrs) if inst.operation.name == 'rz']
    if not rz_idx:
        return qc.copy()
    pick = rng.choice(rz_idx)
    n = qc.num_qubits
    out = QuantumCircuit(n)
    for j, inst in enumerate(instrs):
        if j == pick:
            theta = float(inst.operation.params[0])
            q = _qidx(qc, inst.qubits[0])
            out.rz(theta / 2, q)
            out.rz(theta / 2, q)
        else:
            out.append(inst.operation, inst.qubits, inst.clbits)
    return out


_NAME_TO_ROT = {
    'z': ('rz', math.pi),
    's': ('rz', math.pi / 2),
    'sdg': ('rz', -math.pi / 2),
    't': ('rz', math.pi / 4),
    'tdg': ('rz', -math.pi / 4),
    'x': ('rx', math.pi),
    'y': ('ry', math.pi),
}


def name_to_rotation(qc, rng):
    # Z = Rz(pi), S = Rz(pi/2), T = Rz(pi/4), X = Rx(pi), Y = Ry(pi), ...
    # (each up to global phase). Rewrites a named gate to its rotation form.
    instrs = list(qc.data)
    idxs = [i for i, inst in enumerate(instrs)
            if inst.operation.name in _NAME_TO_ROT and len(inst.qubits) == 1]
    if not idxs:
        return qc.copy()
    pick = rng.choice(idxs)
    out = QuantumCircuit(qc.num_qubits)
    for j, inst in enumerate(instrs):
        if j == pick:
            gate, angle = _NAME_TO_ROT[inst.operation.name]
            getattr(out, gate)(angle, _qidx(qc, inst.qubits[0]))
        else:
            out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def clifford_conjugation(qc, rng):
    # X = H Z H and Z = H X H. Expands a single X or Z into its conjugated form.
    instrs = list(qc.data)
    idxs = [i for i, inst in enumerate(instrs)
            if inst.operation.name in ('x', 'z') and len(inst.qubits) == 1]
    if not idxs:
        return qc.copy()
    pick = rng.choice(idxs)
    out = QuantumCircuit(qc.num_qubits)
    for j, inst in enumerate(instrs):
        if j == pick:
            q = _qidx(qc, inst.qubits[0])
            mid = 'z' if inst.operation.name == 'x' else 'x'
            out.h(q); getattr(out, mid)(q); out.h(q)
        else:
            out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def gate_algebra(qc, rng):
    # S = T*T, Z = S*S, SWAP = CX(a,b) CX(b,a) CX(a,b).
    instrs = list(qc.data)
    idxs = [i for i, inst in enumerate(instrs) if inst.operation.name in ('s', 'z', 'swap')]
    if not idxs:
        return qc.copy()
    pick = rng.choice(idxs)
    out = QuantumCircuit(qc.num_qubits)
    for j, inst in enumerate(instrs):
        if j == pick:
            nm = inst.operation.name
            if nm == 's':
                q = _qidx(qc, inst.qubits[0]); out.t(q); out.t(q)
            elif nm == 'z':
                q = _qidx(qc, inst.qubits[0]); out.s(q); out.s(q)
            else:
                a = _qidx(qc, inst.qubits[0]); b = _qidx(qc, inst.qubits[1])
                out.cx(a, b); out.cx(b, a); out.cx(a, b)
        else:
            out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def push_pauli_cnot(qc, rng):
    # CX_{c,t} (X_c) = (X_c X_t) CX_{c,t}: an X on the control before a CNOT
    # equals an X on both qubits after it. Rewrites [X_c, CX] -> [CX, X_c, X_t].
    instrs = list(qc.data)
    cands = []
    for i in range(len(instrs) - 1):
        a, b = instrs[i], instrs[i + 1]
        if (a.operation.name == 'x' and len(a.qubits) == 1
                and b.operation.name == 'cx'
                and _qidx(qc, a.qubits[0]) == _qidx(qc, b.qubits[0])):
            cands.append(i)
    if not cands:
        return qc.copy()
    pick = rng.choice(cands)
    out = QuantumCircuit(qc.num_qubits)
    j = 0
    while j < len(instrs):
        if j == pick:
            cx = instrs[j + 1]
            c = _qidx(qc, cx.qubits[0]); t = _qidx(qc, cx.qubits[1])
            out.cx(c, t); out.x(c); out.x(t)
            j += 2
        else:
            out.append(instrs[j].operation, instrs[j].qubits, instrs[j].clbits)
            j += 1
    return out


REWRITES = [
    insert_id, cancel_consecutive, commute_swap, merge_rz, split_rz,
    name_to_rotation, clifford_conjugation, gate_algebra, push_pauli_cnot,
]
REWRITE_NAMES = {f.__name__: f for f in REWRITES}


def rewrite_chain(qc, k, rng, rewrites=None):
    cur = qc
    pool = rewrites if rewrites is not None else REWRITES
    for _ in range(k):
        op = rng.choice(pool)
        cur = op(cur, rng)
    return cur


def make_pair(n, depth, k=4, seed=None, rewrites=None):
    rng = random.Random(seed)
    a = random_circuit(n, depth, seed=seed)
    b = rewrite_chain(a, k, rng, rewrites=rewrites)
    return a, b


def corrupt(qc, rng, max_tries=8):
    # A "hard negative": a circuit structurally near-identical to qc but NOT
    # equivalent -- one gate swapped for a confusable sibling (S<->Sdg, CX<->CY,
    # X<->Y) or a rotation shifted by pi. Verified non-equivalent before return.
    instrs = list(qc.data)
    if not instrs:
        return None
    for _ in range(max_tries):
        i = rng.randrange(len(instrs))
        nm = instrs[i].operation.name
        if nm not in _INVERSE and nm not in ('rz', 'rx', 'ry'):
            continue
        out = QuantumCircuit(qc.num_qubits)
        for j, ins in enumerate(instrs):
            if j == i:
                qs = [_qidx(qc, q) for q in ins.qubits]
                if nm in _INVERSE:
                    getattr(out, _INVERSE[nm])(*qs)
                else:
                    getattr(out, nm)(float(ins.operation.params[0]) + math.pi, *qs)
            else:
                out.append(ins.operation, ins.qubits, ins.clbits)
        if not equiv(qc, out):
            return out
    return None
