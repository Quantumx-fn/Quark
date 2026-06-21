import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .feat import VOCAB, MAX_QUBITS, MAX_LEN, encode_batch, pad_mask


class CircuitEncoder(nn.Module):
    def __init__(self, d=128, h=4, layers=3, max_len=MAX_LEN):
        super().__init__()
        self.d = d
        self.gate = nn.Embedding(VOCAB, d, padding_idx=0)
        self.q1 = nn.Embedding(MAX_QUBITS + 1, d, padding_idx=0)
        self.q2 = nn.Embedding(MAX_QUBITS + 1, d, padding_idx=0)
        self.par = nn.Linear(2, d)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, max_len + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, h, dim_feedforward=4 * d, batch_first=True, activation='gelu')
        self.tx = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, d)

    def forward(self, g, q1, q2, p):
        b = g.shape[0]
        x = self.gate(g) + self.q1(q1) + self.q2(q2) + self.par(p)
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, :x.shape[1]]
        m = pad_mask(g)
        m = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=m.device), m], dim=1)
        h = self.tx(x, src_key_padding_mask=m)
        e = self.head(h[:, 0])
        return F.normalize(e, dim=-1)

    def n_params(self):
        return sum(x.numel() for x in self.parameters())


@torch.no_grad()
def embed(model, qcs, batch=32, device='cpu'):
    model.train(False)
    out = []
    for i in range(0, len(qcs), batch):
        chunk = qcs[i:i + batch]
        g, q1, q2, p = encode_batch(chunk)
        e = model(g.to(device), q1.to(device), q2.to(device), p.to(device))
        out.append(e.cpu())
    return torch.cat(out, dim=0)


def default_weights():
    # pretrained weights ship inside the package, so pip-installed copies
    # (with no repo checkout) resolve them without --weights.
    return os.path.join(os.path.dirname(__file__), 'quark.pt')


def load_pretrained(weights=None, device='cpu'):
    enc = CircuitEncoder().to(device)
    sd = torch.load(weights or default_weights(), map_location=device, weights_only=True)
    enc.load_state_dict(sd)
    enc.train(False)
    return enc
