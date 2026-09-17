"""One encoder pass for beta/alpha posterior summaries, without decoder/optimization.
Default checkpoint is validation-selected; interval calibration remains imperfect."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
import train_elbo_only as T
import sir_hybrid_vae as H


def load_model(checkpoint='results_elbo/curved_hybrid/model.pt'):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if 'mixture' in ck.get('objective',''):
        from train_mixture_elbo import Model
        model=Model()
    elif 'enc.curve_head.weight' in ck['model']:
        from train_curved_elbo import Model
        model=Model(ck['scale'],ck['config'].get('cumulative',False))
    else:
        model=T.Model(ck['config']['scale'],ck['config'].get('encoder','mlp'))
    model.load_state_dict(ck['model']);model.eval()
    return model


@torch.no_grad()
def predict(model,counts):
    y=torch.as_tensor(counts,dtype=torch.float32)
    if y.ndim==1:y=y[None,:]
    if y.ndim!=2 or y.shape[1]!=H.T_DAYS:raise ValueError('Expected one or more curves of 120 daily counts.')
    if not torch.isfinite(y).all() or (y<0).any():raise ValueError('Counts must be finite and nonnegative.')
    outputs=model.enc(y)  # exactly one encoder pass
    if outputs[0].ndim==3:
        mu,L,logw,mr,_=outputs
        s=y.new_tensor([H.LOG_BETA_S,H.LOG_ALPHA_S]);m=y.new_tensor([H.LOG_BETA_M,H.LOG_ALPHA_M])
        means=m+s*mu
        cov=(L@L.transpose(-1,-2))*s[None,None,:,None]*s[None,None,None,:]
        means=torch.cat([means,(means[:,:,0]-means[:,:,1])[:,:,None]],2)
        variance=torch.cat([cov.diagonal(dim1=-1,dim2=-2),(cov[:,:,0,0]+cov[:,:,1,1]-2*cov[:,:,0,1])[:,:,None]],2).clamp(min=1e-12)
        sd=variance.sqrt();weights=logw.exp()
        pm=(weights[:,:,None]*torch.exp(means+.5*variance)).sum(1)
        normal=torch.distributions.Normal(0.,1.)
        def quantile(prob):
            lo=(means-12*sd).min(1).values;hi=(means+12*sd).max(1).values
            for _ in range(48):
                mid=(lo+hi)/2
                cdf=(weights[:,:,None]*normal.cdf((mid[:,None]-means)/sd)).sum(1)
                lo=torch.where(cdf<prob,mid,lo);hi=torch.where(cdf>=prob,mid,hi)
            return torch.exp((lo+hi)/2)
        return {'names':['beta','alpha','R0'],'mean':pm.numpy(),'lo90':quantile(.05).numpy(),
                'hi90':quantile(.95).numpy(),'mixture_weights':weights.numpy(),
                'latent_mean':mu.numpy(),'latent_cholesky':L.numpy(),'residual_mean':mr.numpy()}
    mu,L,mr,_=outputs[:4]
    curvature=outputs[4] if len(outputs)==5 else y.new_zeros(len(y))
    s=y.new_tensor([H.LOG_BETA_S,H.LOG_ALPHA_S]);m=y.new_tensor([H.LOG_BETA_M,H.LOG_ALPHA_M])
    logmean=m+s*mu
    logcov=(L@L.transpose(1,2))*s[None,:,None]*s[None,None,:]
    means=torch.cat([logmean,(logmean[:,0]-logmean[:,1])[:,None]],1)
    variances=torch.cat([logcov.diagonal(dim1=1,dim2=2),
        (logcov[:,0,0]+logcov[:,1,1]-2*logcov[:,0,1])[:,None]],1).clamp(min=0)
    sd=variances.sqrt();q=1.6448536269514722
    pm=torch.exp(means+.5*variances);lo=torch.exp(means-q*sd);hi=torch.exp(means+q*sd)
    if torch.any(curvature!=0):
        from train_curved_elbo import transform
        g=torch.Generator().manual_seed(10104)
        eps=torch.randn(6000,len(y),2,generator=g)
        base=mu[None]+torch.einsum('bij,sbj->sbi',L,eps)
        z=transform(base,mu,L,curvature)
        physical=torch.exp(m+s*z)
        values=torch.cat([physical,(physical[:,:,0]/physical[:,:,1])[:,:,None]],2)
        for j in [0,2]:
            pm[:,j]=values[:,:,j].mean(0)
            lo[:,j]=torch.quantile(values[:,:,j],.05,dim=0)
            hi[:,j]=torch.quantile(values[:,:,j],.95,dim=0)
    return {'names':['beta','alpha','R0'],'mean':pm.numpy(),
            'lo90':lo.numpy(),'hi90':hi.numpy(),'curvature':curvature.numpy(),
            'residual_mean':mr.numpy(),'latent_mean':mu.numpy(),'latent_cholesky':L.numpy()}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('counts',help='.npy array, shape (120,) or (N,120)')
    ap.add_argument('--checkpoint',default='results_elbo/curved_hybrid/model.pt');a=ap.parse_args()
    torch.set_num_threads(1);result=predict(load_model(a.checkpoint),np.load(a.counts))
    print(json.dumps({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items()},indent=2))
