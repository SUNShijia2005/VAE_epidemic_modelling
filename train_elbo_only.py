"""Observation-only ELBO experiments. No teacher, true-parameter targets or test adaptation.
Known rho=.1/k=10; two physical coordinates; optional neural beta(t) correction.
"""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import experiment_alpha as E
import sir_hybrid_vae as H

class SequenceEncoder(nn.Module):
    def __init__(self,d_res=0):
        super().__init__();self.d_res=d_res
        self.register_buffer('center',torch.zeros(240));self.register_buffer('scale',torch.ones(240))
        self.net=nn.Sequential(nn.Linear(240,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU())
        self.head=nn.Linear(128,5+2*d_res)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        with torch.no_grad():self.head.bias[2:4].fill_(-1.5)
    def features(self,y):return torch.cat([torch.log1p(y),torch.sqrt(y)],1)
    def fit_scaler(self,y):
        f=self.features(y);self.center.copy_(f.mean(0));self.scale.copy_(f.std(0).clamp(min=.1))
    def forward(self,y):
        o=self.head(self.net((self.features(y)-self.center)/self.scale))
        L=torch.diag_embed(o[:,2:4].clamp(-6,2).exp());L=L.clone();L[:,1,0]=o[:,4]
        return o[:,:2],L,o[:,5:5+self.d_res],o[:,5+self.d_res:].clamp(-8,4)

class CNNEncoder(E.TwoParameterEncoder):
    def __init__(self,d_res):
        super().__init__(d_res)
        self.register_buffer('center',torch.tensor(0.))
        self.register_buffer('scale',torch.tensor(1.))
    def fit_scaler(self,y):
        self.center.copy_(torch.log1p(y).mean());self.scale.copy_(torch.log1p(y).std())
    def forward(self,y):return super().forward((torch.log1p(y)-self.center)/self.scale)

class Model(E.TwoParameterVAE):
    def __init__(self,scale=0.,encoder='mlp'):
        super().__init__(scale)
        self.enc=SequenceEncoder(self.z_res_dim) if encoder=='mlp' else CNNEncoder(self.z_res_dim)


def elbo(model,y,draws=4):
    mu,L,mr,vr=model.enc(y);n=len(y)
    # Antithetic samples reduce Monte Carlo noise; this is a mean of log-likelihoods,
    # not log-mean importance weights. KL weight remains exactly 1.
    ep=torch.randn(draws//2,n,2);ep=torch.cat([ep,-ep],0)
    er=torch.randn(draws//2,n,model.z_res_dim);er=torch.cat([er,-er],0)
    zp=mu[None]+torch.einsum('bij,sbj->sbi',L,ep)
    zr=mr[None]+(.5*vr).exp()[None]*er
    lam=model.decode(zp.reshape(-1,2),zr.reshape(draws*n,model.z_res_dim))
    loglik=H.nb_log_prob(y.repeat(draws,1),lam,model.log_k.exp()).sum(1).reshape(draws,n).mean(0)
    return (-loglik+H.kl_mvn_std(mu,L)+H.kl_std_normal(mr,vr)).mean()

@torch.no_grad()
def validation_loss(model,y):
    model.eval()
    with torch.random.fork_rng():
        torch.manual_seed(8201)
        return sum(elbo(model,b,16).item()*len(b) for b in y.split(128))/len(y)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scale',type=float,default=0.)
    ap.add_argument('--encoder',choices=['mlp','cnn'],default='mlp')
    ap.add_argument('--n_train',type=int,default=8000);ap.add_argument('--epochs',type=int,default=160)
    ap.add_argument('--draws',type=int,default=4);ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--out',default='results_elbo/matched');a=ap.parse_args()
    if a.draws<2 or a.draws%2:ap.error('draws must be a positive even number')
    torch.set_num_threads(1);p=Path(a.out);p.mkdir(parents=True,exist_ok=True)
    # Deliberately discard all ground-truth labels before training/selection.
    yr=E.make_data(a.n_train,7101)[0];yv=E.make_data(500,7102)[0]
    torch.manual_seed(a.seed);model=Model(a.scale,a.encoder);model.enc.fit_scaler(yr)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=.00001)
    best=float('inf');history=[];start=time.time()
    for ep in range(a.epochs):
        model.train();total=0
        for ix in torch.randperm(len(yr)).split(128):
            loss=elbo(model,yr[ix],a.draws)
            if not torch.isfinite(loss):raise RuntimeError('non-finite loss')
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            total+=loss.item()*len(ix)
        sched.step()
        if ep%5==0 or ep==a.epochs-1:
            val=validation_loss(model,yv)
            row={'epoch':ep+1,'train_elbo_loss':total/len(yr),'val_elbo_loss':val,'seconds':time.time()-start}
            history.append(row);print(row,flush=True)
            if val<best:
                best=val
                torch.save({'model':model.state_dict(),'config':vars(a),'train_seed':7101,'validation_seed':7102,
                            'selected_epoch':ep+1,'val_loss':val,'objective':'ordinary ELBO with antithetic MC; no labels or teacher'},p/'model.pt')
            (p/'history.json').write_text(json.dumps(history,indent=2))
    print('TRAIN COMPLETE. No test data evaluated.',flush=True)
if __name__=='__main__':main()
