import json
import random
import click
import torch

from .enc import CircuitEncoder, embed
from .pairs import random_circuit, rewrite_chain
from .search import hash_repr, sorted_repr, count_vec, collect_vocab, cosine_topk, hash_topk


def build_pool(n_queries, lib_size, equiv_per_query, qmax=4, depth_range=(8, 24), k=4, seed=42, rewrites=None):
    rng = random.Random(seed)
    queries = []
    library = []
    truth = []
    fillers = lib_size - n_queries * equiv_per_query

    for i in range(n_queries):
        n = rng.randint(2, qmax)
        d = rng.randint(*depth_range)
        q = random_circuit(n, d, seed=seed * 999 + i)
        queries.append(q)
        idxs = []
        for j in range(equiv_per_query):
            sub_rng = random.Random(seed * 13 + i * 100 + j)
            qe = rewrite_chain(q, k, sub_rng, rewrites=rewrites)
            idxs.append(len(library))
            library.append(qe)
        truth.append(idxs)

    for i in range(fillers):
        n = rng.randint(2, qmax)
        d = rng.randint(*depth_range)
        library.append(random_circuit(n, d, seed=seed * 11 + i + 99999))

    perm = list(range(len(library)))
    rng.shuffle(perm)
    inv = [0] * len(perm)
    for new, old in enumerate(perm):
        inv[old] = new
    library = [library[old] for old in perm]
    truth = [[inv[idx] for idx in t] for t in truth]

    return queries, library, truth


def recall_at_k(retrieved, truth, k):
    hits = 0
    total = 0
    for r, t in zip(retrieved, truth):
        rk = r[:k]
        for g in t:
            total += 1
            if g in rk:
                hits += 1
    return hits / max(total, 1)


_DISCRIM = [('s', 'sdg'), ('t', 'tdg'), ('sx', 'sxdg'), ('cx', 'cy')]


def discrimination(enc, n=60, seed=7, device='cpu'):
    # pairs differing ONLY by a gate vs a confusable sibling (its inverse, or cx/cy):
    # genuinely NOT equivalent, so a good model gives them a LOW cosine.
    # returns (mean_cos, separated_frac).
    rng = random.Random(seed)
    A, B = [], []
    for i in range(n):
        g1, g2 = rng.choice(_DISCRIM)
        nq = rng.randint(2, 4)
        base = random_circuit(nq, rng.randint(6, 16), seed=seed * 131 + i)
        a = base.copy(); b = base.copy()
        if g1 in ('cx', 'cy'):
            x, y = rng.sample(range(nq), 2)
            getattr(a, g1)(x, y); getattr(b, g2)(x, y)
        else:
            q = rng.randrange(nq)
            getattr(a, g1)(q); getattr(b, g2)(q)
        A.append(a); B.append(b)
    ea = embed(enc, A, device=device); eb = embed(enc, B, device=device)
    sims = (ea * eb).sum(dim=1)
    return float(sims.mean()), float((sims < 0.9).float().mean())


def run(weights=None, n_queries=100, lib_size=500, equiv_per_query=2,
        qmax=4, depth_range=(8, 24), k=4, seed=42, out=None, device='cpu', rewrites=None):
    queries, library, truth = build_pool(n_queries, lib_size, equiv_per_query, qmax, depth_range, k, seed, rewrites=rewrites)

    enc = CircuitEncoder().to(device)
    if weights:
        sd = torch.load(weights, map_location=device, weights_only=True)
        enc.load_state_dict(sd)
    q_emb = embed(enc, queries, device=device)
    l_emb = embed(enc, library, device=device)
    quark_top = cosine_topk(q_emb, l_emb, k=20)

    lib_h = [hash_repr(c) for c in library]
    q_h = [hash_repr(c) for c in queries]
    hash_top = hash_topk(q_h, lib_h, k=20)

    lib_s = [sorted_repr(c) for c in library]
    q_s = [sorted_repr(c) for c in queries]
    sorted_top = hash_topk(q_s, lib_s, k=20)

    vocab = collect_vocab(library + queries)
    lib_cv = torch.tensor([count_vec(c, vocab) for c in library], dtype=torch.float32)
    q_cv = torch.tensor([count_vec(c, vocab) for c in queries], dtype=torch.float32)
    cv_top = cosine_topk(q_cv, lib_cv, k=20)

    res = {}
    for K in (1, 5, 10):
        res[f'quark_r@{K}'] = recall_at_k(quark_top, truth, K)
        res[f'count_r@{K}'] = recall_at_k(cv_top, truth, K)
        res[f'hash_r@{K}'] = recall_at_k(hash_top, truth, K)
        res[f'sorted_r@{K}'] = recall_at_k(sorted_top, truth, K)

    from . import _style as ui
    click.echo()
    ui.rule(f'retrieval · {n_queries} queries · {lib_size} library · {equiv_per_query} equiv/query')
    rows = [[m] + [f'{res[f"{m}_r@{K}"]:.3f}' for K in (1, 5, 10)]
            for m in ('hash', 'sorted', 'count', 'quark')]
    ui.table(['method', 'R@1', 'R@5', 'R@10'], rows, highlight='quark')

    d_sim, d_sep = discrimination(enc, device=device)
    res['discrim_mean_sim'] = d_sim
    res['discrim_separated'] = d_sep
    click.echo()
    ui.rule('discrimination · gate vs confusable sibling — NOT equivalent (lower is better)')
    ui.kv('mean cosine', f'{d_sim:.3f}')
    ui.kv('separated', f'{d_sep:.0%}')
    click.echo()

    if out:
        with open(out, 'w') as f:
            json.dump(res, f, indent=2)

    return res


def smoke_bench(n_circuits=50, qmax=8, seed=0, weights=None, device='cpu'):
    # fast end-to-end sanity check (generate -> encode -> embed); the hard
    # guarantee is unit-norm embeddings. used by CI and runs without weights.
    import time

    if n_circuits < 1:
        raise ValueError("n_circuits must be >= 1")

    rng = random.Random(seed)
    circuits = [
        random_circuit(rng.randint(2, max(2, qmax)), rng.randint(8, 24), seed=seed * 1000 + i)
        for i in range(n_circuits)
    ]

    enc = CircuitEncoder().to(device)
    tag = 'random init'
    if weights:
        enc.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        tag = weights

    t0 = time.perf_counter()
    emb = embed(enc, circuits, device=device)
    dt = time.perf_counter() - t0

    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "embeddings are not unit-norm"

    eq_sims, rand_sims = [], []
    for i in range(min(n_circuits, 20)):
        eqc = rewrite_chain(circuits[i], 4, random.Random(seed * 7 + i))
        pair = embed(enc, [circuits[i], eqc], device=device)
        eq_sims.append(float((pair[0] * pair[1]).sum()))
        if n_circuits > 1:
            rand_sims.append(float((emb[i] * emb[(i + 1) % n_circuits]).sum()))

    eq_mean = sum(eq_sims) / len(eq_sims)
    rand_mean = sum(rand_sims) / len(rand_sims) if rand_sims else None

    from . import _style as ui
    click.echo()
    ui.rule('smoke bench')
    ui.kv('weights', tag)
    ui.kv('circuits', f'{n_circuits}', f'(up to {qmax} qubits, seed {seed})')
    ui.kv('embed time', f'{dt * 1000:.1f} ms', f'({n_circuits / dt:.0f} circuits/s)')
    ui.kv('equivalent sim', f'{eq_mean:.3f}')
    if rand_mean is not None:
        ui.kv('random sim', f'{rand_mean:.3f}')
        ui.kv('separation', f'{eq_mean - rand_mean:+.3f}')
    else:
        ui.kv('random sim', 'n/a', '(needs >= 2 circuits)')
    click.echo()

    return {'embed_s': dt, 'equiv_sim': eq_mean, 'random_sim': rand_mean}
