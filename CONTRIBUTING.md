# contributing

quark is small. that's the point. PRs welcome but please keep additions narrowly-scoped.

## scope

things that fit:
- new equivalence-preserving rewrites (with a unit test using `quark.verify.equiv`)
- new framework adapters (`quark.adapters.from_X`)
- bugfixes
- benchmark numbers on real corpora
- improvements to encoder, loss, training

things that don't fit:
- hardware integrations (this is deliberately framework-only)
- a UI / website (the streamlit demo is enough)
- alternative ML methods unless they materially beat current numbers

## dev setup

```
git clone https://github.com/Quantumx-fn/Quark.git
cd quark
pip install -e ".[dev]"
pytest tests/
```

## testing

every new rewrite must come with a test that calls `quark.verify.equiv` on at least 5 random small circuits to confirm unitary-equivalence is preserved. this is non-negotiable — a broken rewrite poisons all training signal silently.

```python
from quark.pairs import my_new_rewrite, random_circuit
from quark.verify import equiv
import random

def test_my_new_rewrite_preserves_unitary():
    for s in range(8):
        a = random_circuit(2, 8, seed=s)
        b = my_new_rewrite(a, random.Random(s))
        assert equiv(a, b), f"rewrite broke semantics on seed {s}"
```

## reporting numbers

if you add a new dataset, eval, or model: include the exact reproduction command and the random seed in the README. numbers without seeds are unfalsifiable.

## style

- no docstrings, no comments unless something non-obvious is happening
- short variable names where unambiguous (`qc`, `enc`, `e`)
- direct imports, no lazy import unless there's a real circular-import reason
- prefer adding to existing files over creating new ones unless the new concept is genuinely separate
