# Security Policy

## Supported versions

quark is pre-1.0; security fixes are applied to the latest released version on
the `main` branch only.

| Version | Supported |
|---------|-----------|
| 0.4.x   | ✅        |
| < 0.4   | ❌        |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security problems.

Report privately by either:

- Using GitHub's [private vulnerability reporting](https://github.com/Quantumx-fn/Quark/security/advisories/new), or
- Emailing **hi@quantumx.foundation**

Include a description, reproduction steps, and the affected version/commit.

We aim to acknowledge reports within **72 hours** and to provide a remediation
timeline after triage. Please give us a reasonable window to release a fix
before any public disclosure.

## Scope

quark is a CPU-side machine-learning library that loads bundled PyTorch weights
(`weights_only=True`) and parses user-supplied QASM via Qiskit. Reports
involving untrusted-input parsing, deserialization, or dependency
vulnerabilities are in scope. quark does not touch quantum hardware,
networks, or credentials.
