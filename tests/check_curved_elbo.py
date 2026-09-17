"""Check nonlinear posterior invertibility, Jacobian, KL and ELBO gradients."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from unittest.mock import patch
import train_curved_elbo as C
import train_elbo_only as T
import experiment_alpha as E
import sir_hybrid_vae as H

torch.set_num_threads(1);torch.manual_seed(41)
mu=torch.tensor([[.2,-.3]],dtype=torch.float64)
L=torch.tensor([[[.5,0.],[.2,.7]]],dtype=torch.float64);c=torch.tensor([.17],dtype=torch.float64)
x=torch.tensor([.4,-.1],dtype=torch.float64,requires_grad=True)
jac=torch.autograd.functional.jacobian(lambda v:C.transform(v[None,None],mu,L,c)[0,0],x)
torch.testing.assert_close(torch.det(jac),torch.tensor(1.,dtype=torch.float64))
base=torch.distributions.MultivariateNormal(mu,scale_tril=L).sample((300000,))
z=C.transform(base,mu,L,c);var=L[:,1,:].square().sum(1)
recovered=z.clone();recovered[:,:,0]-=c*((z[:,:,1]-mu[:,1])**2-var)
torch.testing.assert_close(base,recovered)
logq=torch.distributions.MultivariateNormal(mu,scale_tril=L).log_prob(base)
logp=torch.distributions.MultivariateNormal(torch.zeros_like(mu),torch.eye(2,dtype=mu.dtype)[None]).log_prob(z)
torch.testing.assert_close((logq-logp).mean(0),C.kl(mu,L,c),atol=.012,rtol=0)
y=E.make_data(4,7722)[0]
a=T.Model(.5);a.enc.fit_scaler(y);b=C.Model(.5)
b.load_state_dict(a.state_dict(),strict=False)
torch.manual_seed(11);la=T.elbo(a,y,8)
torch.manual_seed(11);lb=C.elbo(b,y,8)
torch.testing.assert_close(la,lb)
lb.backward();assert all(torch.isfinite(p.grad).all() for p in b.parameters() if p.grad is not None)
assert b.enc.curve_head.bias.grad.abs().max()>0
print('PASS: invertible unit-Jacobian transform; analytic KL matches independent density estimate; zero curvature reproduces ordinary ELBO; finite learning gradients.')
from infer_elbo_only import predict
with torch.no_grad():b.enc.curve_head.bias.fill_(.3)
old={k:v.clone() for k,v in b.state_dict().items()};calls=[]
hook=b.enc.register_forward_hook(lambda *a:calls.append(1))
with patch.object(H,'sir_incidence',side_effect=AssertionError('No SIR allowed in test inference')):
    prediction=predict(b,y)
hook.remove();assert len(calls)==1 and prediction['mean'].shape==(4,3)
assert (prediction['curvature']!=0).all()
for k,v in b.state_dict().items():torch.testing.assert_close(v,old[k],rtol=0,atol=0)
print('PASS: public curved-posterior inference uses one encoder pass and leaves every parameter unchanged.')
