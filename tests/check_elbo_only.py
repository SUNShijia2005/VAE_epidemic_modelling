"""Observation-only objective and one-pass posterior inference invariants."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unittest.mock import patch
import torch
import train_elbo_only as T
import experiment_alpha as E
import sir_hybrid_vae as H

torch.set_num_threads(1)
y,_,z,mean=E.make_data(4,7711)
for scale in [0.,.5]:
    m=T.Model(scale);m.enc.fit_scaler(y)
    torch.testing.assert_close(m.decode(z,torch.zeros(4,m.z_res_dim)),mean.clamp(min=1e-6))
    mu,L,mr,vr=m.enc(y)
    q=torch.distributions.MultivariateNormal(mu,scale_tril=L)
    prior=torch.distributions.MultivariateNormal(torch.zeros_like(mu),torch.eye(2).expand(len(y),2,2))
    torch.testing.assert_close(H.kl_mvn_std(mu,L),torch.distributions.kl_divergence(q,prior))
    torch.manual_seed(34);loss=T.elbo(m,y,4)
    torch.manual_seed(34)
    ep=torch.randn(2,4,2);ep=torch.cat([ep,-ep])
    er=torch.randn(2,4,m.z_res_dim);er=torch.cat([er,-er])
    zp=mu[None]+torch.einsum('bij,sbj->sbi',L,ep)
    zr=mr[None]+(.5*vr).exp()[None]*er
    ll=[]
    for j in range(4):ll.append(H.nb_log_prob(y,m.decode(zp[j],zr[j]),m.log_k.exp()).sum(1))
    expected=(-torch.stack(ll).mean(0)+torch.distributions.kl_divergence(q,prior)+H.kl_std_normal(mr,vr)).mean()
    torch.testing.assert_close(loss,expected)
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    old={k:v.clone() for k,v in m.state_dict().items()};calls=[]
    hook=m.enc.register_forward_hook(lambda *a:calls.append(1))
    with patch.object(H,'sir_incidence',side_effect=AssertionError('No SIR at test inference')):
        pd=E.draws(m,y,100)
    hook.remove();assert len(calls)==1 and pd.shape==(100,4,2)
    for k,v in m.state_dict().items():torch.testing.assert_close(v,old[k],rtol=0,atol=0)
# Directional derivative through the mechanistic decoder agrees with finite difference.
m=T.Model(0);z=torch.tensor([[.2,-.3]],requires_grad=True)
v=torch.tensor([[.4,-.2]])
f=m.decode(z,torch.zeros(1,0)).log().mean();grad=torch.autograd.grad(f,z)[0]
eps=.001
finite=(m.decode(z.detach()+eps*v,torch.zeros(1,0)).log().mean()-m.decode(z.detach()-eps*v,torch.zeros(1,0)).log().mean())/(2*eps)
torch.testing.assert_close((grad*v).sum(),finite,rtol=.02,atol=.002)
print('PASS: matched means, exact Gaussian KL, ordinary multi-sample ELBO (not IWAE), finite gradients, finite-difference SIR derivative, one-pass unchanged test inference.')
from infer_elbo_only import predict
m=T.Model(.5);m.enc.fit_scaler(y);calls=[]
hook=m.enc.register_forward_hook(lambda *a:calls.append(1))
with patch.object(H,'sir_incidence',side_effect=AssertionError('No decoder at inference')):
    prediction=predict(m,y)
hook.remove();assert len(calls)==1
assert prediction['mean'].shape==(4,3)
assert (prediction['lo90']<prediction['hi90']).all()
assert (prediction['lo90'][:,0]>0).all()
# Identity: R0 log variance includes the beta-alpha covariance.
mu=torch.tensor(prediction['latent_mean']);L=torch.tensor(prediction['latent_cholesky'])
w=torch.tensor([H.LOG_BETA_S,-H.LOG_ALPHA_S])
var=torch.einsum('i,bij,j->b',w,L@L.transpose(1,2),w)
expected=torch.exp(H.LOG_BETA_M-H.LOG_ALPHA_M+mu@w+.5*var).detach().numpy()
import numpy as np
np.testing.assert_allclose(prediction['mean'][:,2],expected,rtol=1e-6)
for invalid in [torch.zeros(119),torch.full((120,),-1.),torch.full((120,),float('nan'))]:
    try:predict(m,invalid)
    except ValueError:pass
    else:raise AssertionError('Invalid observations should be rejected')
print('PASS: public inference returns exact lognormal summaries in one encoder pass, including correlated R0 uncertainty, and validates inputs.')
