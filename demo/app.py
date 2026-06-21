import streamlit as st
import torch
from qiskit import QuantumCircuit
from quark.enc import CircuitEncoder, embed
from quark.adapters import from_qasm
from quark.verify import equiv

st.set_page_config(page_title="quark: circuit similarity", layout="wide")
st.title("quark — circuit similarity")
st.caption("paste two openqasm 2.0 circuits. quark embeds them with a small transformer trained on equivalence-preserving rewrites and reports cosine similarity. for ground truth, use the verify button (computes full unitary, only feasible for ≤10 qubits).")

@st.cache_resource
def load_model():
    enc = CircuitEncoder()
    enc.load_state_dict(torch.load("quark.pt", map_location="cpu", weights_only=True))
    enc.train(False)
    return enc


DEFAULT_A = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[0];
cx q[0],q[1];
cx q[1],q[2];"""

DEFAULT_B = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[0];
h q[0];
h q[0];
cx q[0],q[1];
cx q[1],q[2];"""

col1, col2 = st.columns(2)
with col1:
    a_text = st.text_area("circuit a (qasm 2.0)", value=DEFAULT_A, height=200)
with col2:
    b_text = st.text_area("circuit b (qasm 2.0)", value=DEFAULT_B, height=200)

if st.button("compare"):
    try:
        a = from_qasm(a_text)
        b = from_qasm(b_text)
        enc = load_model()
        ea, eb = embed(enc, [a, b])
        sim = float((ea * eb).sum().item())
        st.metric("cosine similarity", f"{sim:.4f}")
        st.write(f"a: {a.num_qubits} qubits, {len(a.data)} gates")
        st.write(f"b: {b.num_qubits} qubits, {len(b.data)} gates")
        if sim > 0.90:
            st.success("very similar embeddings — likely equivalent under the rewrites quark trained on")
        elif sim > 0.80:
            st.info("similar embeddings — possibly related, run verify for the ground truth")
        else:
            st.warning("dissimilar embeddings — probably not equivalent")
    except Exception as e:
        st.error(f"parse failed: {e}")

if st.button("verify (ground truth, ≤10 qubits)"):
    try:
        a = from_qasm(a_text)
        b = from_qasm(b_text)
        if max(a.num_qubits, b.num_qubits) > 10:
            st.error("too large for ground-truth verification (>10 qubits)")
        else:
            ok = equiv(a, b)
            if ok:
                st.success("circuits are equivalent up to global phase (verified by full unitary)")
            else:
                st.error("circuits are NOT equivalent (full unitary differs)")
    except Exception as e:
        st.error(f"verify failed: {e}")

st.markdown("---")
st.caption("trained on synthetic random circuits (2-4 qubits, depth 8-24). real-world circuits may be out-of-distribution.")
