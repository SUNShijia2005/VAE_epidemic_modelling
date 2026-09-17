"""Check mixture ELBO weighting and finite gradients without training labels."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,math
import train_mixture_elbo as M
import experiment_alpha as E
import sir_hybrid_vae as H

torch.set_num_threads(1);y=E.make_data(4,7733)[0];m=M.Model();m.enc.fit_scaler(y)
with torch.no_grad():
 m.enc.head.weight.zero_();m.enc.head.bias.zero_()
 m.enc.head.bias[10:12]=torch.tensor([math.log(.2),math.log(.8)])
torch.manual_seed(123);actual=M.objective(m,y,4)
torch.manual_seed(123)
ep=torch.randn(2,2,4,2);ep=torch.cat([ep,-ep]);er=torch.randn(2,2,4,4);er=torch.cat([er,-er])
# Identical standard-normal components => physical KL exactly zero for every sample.
lam=m.decode(ep.reshape(-1,2),er.reshape(-1,4))
ll=H.nb_log_prob(y.repeat(8,1),lam,m.log_k.exp()).sum(1).reshape(4,2,4)
expected=-(ll.mean(0)*torch.tensor([.2,.8])[:,None]).sum(0).mean()
torch.testing.assert_close(actual,expected)
actual.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
print('PASS: mixture ELBO equals weighted likelihood when q equals the prior; gradients are finite.')
from infer_elbo_only import predict
from unittest.mock import patch
import numpy as np
calls=[];hook=m.enc.register_forward_hook(lambda *a:calls.append(1))
with patch.object(H,'sir_incidence',side_effect=AssertionError('No SIR during inference')):
 pred=predict(m,y)
hook.remove();assert len(calls)==1
# Two identical N(0,I) components reproduce the prior lognormal marginal summaries.
s=torch.tensor([H.LOG_BETA_S,H.LOG_ALPHA_S]);c=torch.tensor([H.LOG_BETA_M,H.LOG_ALPHA_M])
expected=(c+.5*s.square()).exp().numpy()
np.testing.assert_allclose(pred['mean'][0,:2],expected,rtol=1e-6)
np.testing.assert_allclose(pred['lo90'][0,:2],(c-1.6448536269514722*s).exp().numpy(),rtol=1e-6)
np.testing.assert_allclose(pred['hi90'][0,:2],(c+1.6448536269514722*s).exp().numpy(),rtol=1e-6)
print('PASS: mixture inference uses one encoder call and analytic marginal summaries; identical components reproduce lognormal quantiles.')
