"""Compare current model, simulator, loss and gradients with the saved Colab source."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sir_hybrid_vae as H

torch.set_num_threads(1)
nb = json.loads((ROOT / 'archive/pre_colab_alignment/Hybrid_SIR_VAE_report.ipynb').read_text())
ref = {'np': np, 'torch': torch, 'nn': nn, 'EPOCHS': 120}
for i in [5, 7, 11, 15, 17]:
    exec(''.join(nb['cells'][i]['source']), ref)
for level in [0.0, 1.0]:
    y, _, theta, mod = H.make_data(6, seed=7, misspec=level)
    ry, rt, rm = ref['simulate'](6, seed=7, intervention=level)
    torch.testing.assert_close(y, ry, rtol=0, atol=0)
    torch.testing.assert_close(theta, rt, rtol=0, atol=0)
    if mod is not None: torch.testing.assert_close(mod, rm)

for dim in [0, 4]:
    torch.manual_seed(3)
    actual = H.HybridVAE(z_res_dim=dim)
    torch.manual_seed(3)
    expected = ref['HybridVAE'](d_res=dim)
    # A nonzero residual checks where the correction enters the SIR, not just the zero initialization.
    if dim:
        with torch.no_grad():
            actual.res.net[-1].weight.fill_(0.03)
            expected.res.net[-1].weight.fill_(0.03)
    x = (torch.log1p(y) - torch.log1p(y).mean()) / torch.log1p(y).std()
    zp, zr = torch.randn(6, 3), torch.randn(6, dim)
    torch.testing.assert_close(actual.decode(zp, zr), expected.decode(zp, zr))
    for fb in [0.0, 0.5]:
        actual.zero_grad(); expected.zero_grad()
        torch.manual_seed(8)
        loss, _ = H.objective(actual, x, y, 1.0, 1, fb)
        torch.manual_seed(8)
        reference = ref['loss_fn'](expected, x, y, fb)
        torch.testing.assert_close(loss, reference)
        loss.backward(); reference.backward()
        for ap, ep in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(ap.grad, ep.grad)
print('PASS: Colab simulator, active beta decoder, ELBO and all parameter gradients; pure and hybrid modes.')

# Compare a complete short training trajectory, including validation RNG and scheduling.
from types import SimpleNamespace
args = SimpleNamespace(lr=2e-3, epochs=2, beta_kl=1.0, kl_warmup=0,
                       batch=3, iwae=1, free_bits=0.0, patience=0)
torch.manual_seed(12)
a = H.HybridVAE()
torch.manual_seed(12)
b = ref['HybridVAE']()
torch.manual_seed(19)
H.train(a, x, y, x[:2], y[:2], args, 'cpu')
torch.manual_seed(19)
ref['train'](b, x, y, x[:2], y[:2], epochs=2, batch=3, verbose=True)
for ap, bp in zip(a.parameters(), b.parameters()):
    torch.testing.assert_close(ap, bp)
print('PASS: two-epoch training trajectory agrees with Colab.')
