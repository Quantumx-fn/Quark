import os
import tempfile
import random
from qiskit.qasm2 import dumps
from quark.pairs import random_circuit, rewrite_chain
from quark.dedupe import cluster, _walk


def test_walk_finds_qasm():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, 'a.qasm'), 'w') as f:
            f.write('OPENQASM 2.0; include "qelib1.inc"; qreg q[1];')
        with open(os.path.join(d, 'b.txt'), 'w') as f:
            f.write('not qasm')
        files = list(_walk(d))
        assert len(files) == 1
        assert files[0].endswith('a.qasm')


def test_cluster_groups_equivalents(tmp_path):
    rng = random.Random(0)
    qc = random_circuit(2, 8, seed=42)
    eq = rewrite_chain(qc, k=3, rng=rng)

    p1 = tmp_path / 'a.qasm'
    p2 = tmp_path / 'b.qasm'
    p3 = tmp_path / 'c.qasm'
    p1.write_text(dumps(qc))
    p2.write_text(dumps(eq))
    other = random_circuit(2, 8, seed=999)
    p3.write_text(dumps(other))

    res = cluster([str(p1), str(p2), str(p3)], weights='quark.pt', threshold=0.5)
    assert res['n_files'] == 3
