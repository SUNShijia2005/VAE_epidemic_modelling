"""Two-parameter diagnostic: known rho/k, matched SIR and hybrid on identical data.
This is a separate experiment; the aligned three-parameter Colab stays unchanged.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import sir_hybrid_vae as H


class TwoParameterEncoder(H.Encoder):
    def __init__(self, d_res):
        super().__init__(d_res)
        self.head_phys = nn.Linear(128, 5)  # two means, two log variances, one covariance

    def forward(self, x):
        h = self.mlp(self.conv(x.unsqueeze(1)).flatten(1))
        o = self.head_phys(h)
        L = torch.diag_embed(torch.exp(0.5 * o[:, 2:4].clamp(-8, 4)))
        L = L.clone()
        L[:, 1, 0] = o[:, 4]
        if self.z_res_dim:
            r = self.head_res(h)
            mr, vr = r[:, :self.z_res_dim], r[:, self.z_res_dim:].clamp(-8, 4)
        else:
            mr = vr = h.new_zeros(len(h), 0)
        return o[:, :2], L, mr, vr


class TwoParameterVAE(H.HybridVAE):
    def __init__(self, scale=0.0):
        super().__init__(nn_scale=scale)
        self.enc = TwoParameterEncoder(self.z_res_dim)
        self.log_k.requires_grad_(False)

    def decode(self, z_phys, z_res):
        # z_rho=0 fixes rho at 0.10, exactly as in the diagnostic data generator.
        return super().decode(torch.cat([z_phys, z_phys.new_zeros(len(z_phys), 1)], 1), z_res)


def make_data(n, seed):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, 2, generator=g)
    full = torch.cat([z, torch.zeros(n, 1)], 1)
    beta, alpha, rho = H.phys_from_z(full)
    mean = H.sir_incidence(beta, alpha) * rho[:, None]
    torch.manual_seed(seed + 999)
    y = torch.distributions.NegativeBinomial(total_count=H.NB_K,
            logits=mean.clamp(min=1e-6).log() - np.log(H.NB_K)).sample()
    return y, torch.stack([beta, alpha], 1), z, mean


def metrics(draws, truth):
    rows = {}
    for j, name in enumerate(['beta', 'alpha', 'R0']):
        d = draws[:, :, j] if j < 2 else draws[:, :, 0] / draws[:, :, 1]
        t = truth[:, j] if j < 2 else truth[:, 0] / truth[:, 1]
        pm = d.mean(0); lo, hi = np.percentile(d, [5, 95], axis=0)
        rows[name] = dict(corr=float(np.corrcoef(pm,t)[0,1]), rmse=float(np.sqrt(np.mean((pm-t)**2))),
                          bias=float(np.mean(pm-t)), cover90=float(np.mean((lo<=t)&(t<=hi))))
    return rows


@torch.no_grad()
def draws(model, x, count=2000):
    mu,L,_,_ = model.enc(x)
    z = model.q_phys(mu,L).sample((count,))
    return torch.exp(z * z.new_tensor([H.LOG_BETA_S,H.LOG_ALPHA_S]) +
                     z.new_tensor([H.LOG_BETA_M,H.LOG_ALPHA_M])).numpy()


@torch.no_grad()
def grid_reference(y, truth, size, span):
    axis = torch.linspace(-span,span,size)
    z = torch.stack(torch.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)
    theta = torch.exp(z * torch.tensor([H.LOG_BETA_S,H.LOG_ALPHA_S]) +
                      torch.tensor([H.LOG_BETA_M,H.LOG_ALPHA_M]))
    logweights=[]
    for idx in torch.arange(len(z)).split(4096):
        mean=H.sir_incidence(theta[idx,0],theta[idx,1])*0.1
        mean=mean.clamp(min=1e-6).double(); k=H.NB_K
        # Observation-only constants cancel in posterior normalization.
        ll=y.double() @ (mean.log()-torch.log(mean+k)).T - k*torch.log(mean+k).sum(1)[None,:]
        logweights.append(ll-0.5*z[idx].double().square().sum(1)[None,:])
    w=torch.softmax(torch.cat(logweights,1),1)
    edge=(z.abs().max(1).values>=span-2*span/(size-1))
    summaries={}; means=[]
    for j,name in enumerate(['beta','alpha','R0']):
        values=theta[:,j] if j<2 else theta[:,0]/theta[:,1]
        values=values.double(); order=values.argsort(); v=values[order]
        pm=w@values; sd=((w@values.square())-pm.square()).clamp(min=0).sqrt()
        c=w[:,order].cumsum(1)
        lo=v[(c<.05).sum(1).clamp(max=len(v)-1)];hi=v[(c<.95).sum(1).clamp(max=len(v)-1)]
        t=truth[:,j] if j<2 else truth[:,0]/truth[:,1]
        summaries[name]=dict(corr=float(np.corrcoef(pm.numpy(),t.numpy())[0,1]),
            rmse=float(((pm-t)**2).mean().sqrt()),bias=float((pm-t).mean()),
            cover90=float(((lo<=t)&(t<=hi)).double().mean()),mean_sd=float(sd.mean()))
        means.append(pm)
    return summaries, torch.stack(means,1),float(w[:,edge].sum(1).max())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--n_train',type=int,default=2000)
    ap.add_argument('--epochs',type=int,default=60)
    ap.add_argument('--n_test',type=int,default=200)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--out',default='results_alpha/diagnostic')
    args=ap.parse_args();torch.set_num_threads(1)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    nval=300
    y,theta,z,mean=make_data(args.n_train+nval+args.n_test,args.seed)
    yt,yv,yr=y[:args.n_test],y[args.n_test:args.n_test+nval],y[args.n_test+nval:]
    m,s=torch.log1p(yr).mean(),torch.log1p(yr).std()
    xt,xv,xr=[(torch.log1p(v)-m)/s for v in [yt,yv,yr]]
    truth=theta[:args.n_test]
    cfg=SimpleNamespace(lr=.002,epochs=args.epochs,beta_kl=1.,kl_warmup=0,batch=128,iwae=1,free_bits=0.,patience=0)
    result={'config':vars(args),'fixed_rho':.1,'fixed_k':10,'n_val':nval,'metrics':{}}
    for tag,scale in [('matched',0.),('hybrid',.5)]:
        torch.manual_seed(args.seed);model=TwoParameterVAE(scale)
        if tag=='matched':
            torch.testing.assert_close(model.decode(z,torch.zeros(len(z),0)),mean.clamp(min=1e-6))
            assert model.enc.head_phys.out_features==5 and not model.log_k.requires_grad
        print('\nTRAIN',tag,flush=True)
        H.train(model,xr,yr,xv,yv,cfg,'cpu')
        torch.manual_seed(2026);pd=draws(model,xt)
        result['metrics'][tag]=metrics(pd,truth.numpy())
        torch.save({'model':model.state_dict(),'scale':scale,'config':vars(args),'fixed_rho':.1,'fixed_k':10,
                    'x_scaler':(float(m),float(s))},str(out)+'_'+tag+'.pt')
        np.savez(str(out)+'_'+tag+'_evaluation.npz',draws=pd,truth=truth.numpy(),y=yt.numpy())
        print(tag,json.dumps(result['metrics'][tag]),flush=True)
        Path(str(out)+'.json').write_text(json.dumps(result,indent=2))
    previous=None
    for size,span in [(201,4.),(401,4.),(501,5.),(751,5.),(901,6.)]:
        stats,pm,edge=grid_reference(yt,truth,size,span)
        row={'size':size,'span':span,'metrics':stats,'max_boundary_mass':edge}
        if previous is not None:row['max_mean_change']=float((pm-previous).abs().max())
        result.setdefault('grid',[]).append(row);previous=pm
        print('GRID',json.dumps(row),flush=True)
        Path(str(out)+'.json').write_text(json.dumps(result,indent=2))
    torch.save({'mean':previous,'note':'Constant-SIR reference; fixed rho=.1, k=10'},str(out)+'_grid_reference.pt')

if __name__=='__main__':main()
