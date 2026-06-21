# changelog

## v0.4.0 (06-05-2026)

Quantum-core accuracy pass.

- **Fixed a false-positive bug.** The tokeniser mapped a gate and its inverse (`s`/`sdg`, `t`/`tdg`, `sx`/`sxdg`) and `cx`/`cy` to the same id, so circuits differing only by a gate versus its inverse embedded identically (cosine 1.0000). They now get distinct ids (vocabulary 16 → 20) and are trained apart.
- **Four new equivalence-preserving rewrites** (each with a unitary-equivalence test): named-gate ↔ rotation (Z=Rz(π), S=Rz(π/2), T=Rz(π/4), X=Rx(π), Y=Ry(π)), Clifford conjugation (HXH=Z, HZH=X), gate algebra (S=T·T, Z=S·S, SWAP=3·CNOT), and Pauli propagation through CNOT. Nine rewrites in total.
- **Corrupted-twin hard negatives.** Each anchor trains against a near-identical but verified non-equivalent twin (one gate swapped for its inverse), which is what teaches the model to separate them.
- **New discrimination benchmark** reported by `quark eval`: gate-vs-inverse mean cosine 1.00 → 0.77 (67% correctly separated).
- Widened the synthetic training gate set so the full vocabulary is exercised.
- Result changes: in-distribution R@10 = 1.000 (unchanged); held-out rewrite R@10 = 0.985 vs 0.550 baseline (up from 0.935); QASMBench R@10 = 0.172, down from 0.345 and now level with the baseline — a deliberate trade-off (see README).
- 27 tests, all passing.

## v0.3.0 (06-05-2026)

Initial public release.

- 647k-parameter transformer encoder, contrastive (InfoNCE) training on synthetic equivalence pairs
- Five equivalence-preserving rewrites: insert/cancel HH-XX-CNOT pairs, commute disjoint gates, split/merge Rz angles
- Three retrieval evaluations: in-distribution synthetic (R@10 = 1.000), held-out rewrite (R@10 = 0.935 vs 0.290 baseline), QASMBench out-of-distribution (R@10 = 0.345 vs 0.172)
- `quark dedupe <directory>` CLI for grouping equivalent QASM files (default cosine threshold 0.9)
- `quark bench` for a fast end-to-end smoke benchmark, also run in CI
- PennyLane and Cirq adapters, with arbitrary wire labels supported and unrecognised gates reported instead of silently dropped
- Pretrained weights loaded with `weights_only=True`; a warning is emitted when a circuit is truncated to the 128-gate context window
- 24 tests, all passing
- arXiv-ready preprint scaffold in `paper/`
- Streamlit demo in `demo/`
- Quickstart notebook in `examples/`
