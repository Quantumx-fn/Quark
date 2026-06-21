import os
import torch
import pytest
from quark.enc import CircuitEncoder
from quark.eval_ import discrimination

WEIGHTS = os.path.join(os.path.dirname(__file__), '..', 'quark.pt')


@pytest.mark.skipif(not os.path.exists(WEIGHTS), reason="pretrained weights not present")
def test_trained_model_separates_gate_from_inverse():
    enc = CircuitEncoder()
    enc.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True))
    mean_sim, separated = discrimination(enc)
    # circuits that differ only by a gate vs its inverse are NOT equivalent; the
    # trained model must not call them identical (the pre-fix model returned ~1.0).
    assert mean_sim < 0.9, f"gate/inverse pairs too similar (mean cosine {mean_sim:.3f})"
    assert separated > 0.5, f"too few pairs separated ({separated:.0%})"
