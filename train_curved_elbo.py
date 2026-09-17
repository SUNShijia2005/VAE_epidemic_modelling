"""Observation-only ELBO with a triangular quadratic posterior transformation.
Decoder unchanged. Unit-Jacobian transformation allows exact analytic KL.
No parameter labels, numerical posterior targets, or test-time optimization.
"""
import argparse,json,time
from pathlib import Path
import torch
from torch import nn
import train_elbo_only as T
import experiment_alpha as E
import sir_hybrid_vae as H

class CurvedEncoder(T.SequenceEncoder):
    def __init__(self,d_res=0,cumulative=False):
        super().__init__(d_res)
        self.cumulative=cumulative
        if cumulative:
            self.center=torch.zeros(360);self.scale=torch.ones(360)
            self.net[0]=nn.Linear(360,256)
        self.curve_head=nn.Linear(128,1)
        nn.init.zeros_(self.curve_head.weight);nn.init.zeros_(self.curve_head.bias)
    def features(self,y):
        basic=super().features(y)
        return torch.cat([basic,y.cumsum(1)/(.1*H.N_POP)],1) if self.cumulative else basic
    def forward(self,y):
        h=self.net((self.features(y)-self.center)/self.scale);o=self.head(h)
        L=torch.diag_embed(o[:,2:4].clamp(-6,2).exp());L=L.clone();L[:,1,0]=o[:,4]
        var=L[:,1,:].square().sum(1)
        c=.25*torch.tanh(self.curve_head(h)[:,0])/(1+var)
        return o[:,:2],L,o[:,5:5+self.d_res],o[:,5+self.d_res:].clamp(-8,4),c

class Model(E.TwoParameterVAE):
    def __init__(self,scale=.5,cumulative=False):
        super().__init__(scale);self.enc=CurvedEncoder(self.z_res_dim,cumulative)


def transform(base,mu,L,c):
    var=L[:,1,:].square().sum(1)
    z=base.clone()
    z[:,:,0]=base[:,:,0]+c[None]*((base[:,:,1]-mu[None,:,1]).square()-var[None])
    return z


def kl(mu,L,c):
    # E[z0^2] gains 2*c^2*Var(base1)^2; entropy is unchanged (Jacobian determinant 1).
    return H.kl_mvn_std(mu,L)+c.square()*L[:,1,:].square().sum(1).square()


def elbo(model,y,draws=8):
    mu,L,mr,vr,c=model.enc(y);n=len(y)
    ep=torch.randn(draws//2,n,2);ep=torch.cat([ep,-ep])
    er=torch.randn(draws//2,n,model.z_res_dim);er=torch.cat([er,-er])
    base=mu[None]+torch.einsum('bij,sbj->sbi',L,ep)
    zp=transform(base,mu,L,c)
    zr=mr[None]+(.5*vr).exp()[None]*er
    lam=model.decode(zp.reshape(-1,2),zr.reshape(draws*n,model.z_res_dim))
    ll=H.nb_log_prob(y.repeat(draws,1),lam,model.log_k.exp()).sum(1).reshape(draws,n).mean(0)
    return (-ll+kl(mu,L,c)+H.kl_std_normal(mr,vr)).mean()

@torch.no_grad()
def validation(model,y):
    model.eval()
    with torch.random.fork_rng():
        torch.manual_seed(8201)
        return sum(elbo(model,b,32).item()*len(b) for b in y.split(128))/len(y)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default='results_elbo/hybrid/model.pt')
    ap.add_argument('--out',default='results_elbo/curved_hybrid');ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--cumulative',action='store_true',help='Add cumulative observed cases as input; no labels')
    ap.add_argument('--freeze_curve',action='store_true',help='Same training budget, retain Gaussian posterior')
    ap.add_argument('--seed',type=int,default=27);a=ap.parse_args();torch.set_num_threads(1)
    ck=torch.load(a.checkpoint,weights_only=False);torch.manual_seed(a.seed)
    model=Model(ck['config']['scale'],a.cumulative)
    state=ck['model'].copy()
    if a.cumulative:
        state['enc.center']=torch.cat([state['enc.center'],torch.zeros(120)])
        state['enc.scale']=torch.cat([state['enc.scale'],torch.ones(120)])
        state['enc.net.0.weight']=torch.cat([state['enc.net.0.weight'],torch.zeros(256,120)],1)
    missing,unexpected=model.load_state_dict(state,strict=False)
    assert set(missing)=={'enc.curve_head.weight','enc.curve_head.bias'} and not unexpected
    if a.freeze_curve:
        for param in model.enc.curve_head.parameters():param.requires_grad_(False)
    yr=E.make_data(8000,7101)[0];yv=E.make_data(500,7102)[0]
    if a.cumulative:
        f=model.enc.features(yr)
        model.enc.center[240:]=f[:,240:].mean(0)
        model.enc.scale[240:]=f[:,240:].std(0).clamp(min=1e-4)
    opt=torch.optim.Adam(model.parameters(),lr=.00025)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=.000005)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    best=validation(model,yv);hist=[{'epoch':0,'val':best}]
    def save(epoch):
        torch.save({'model':model.state_dict(),'config':vars(a),'scale':model.nn_scale,'selected_epoch':epoch,
                    'val_loss':best,'train_seed':7101,'validation_seed':7102,
                    'objective':'ordinary ELBO, analytic unit-Jacobian curved posterior KL; no labels/teacher'},out/'model.pt')
    save(0);print('initial validation',best,flush=True);start=time.time()
    for ep in range(a.epochs):
        model.train();total=0
        for ix in torch.randperm(len(yr)).split(128):
            loss=elbo(model,yr[ix],8)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            total+=loss.item()*len(ix)
        sch.step()
        if ep%5==0 or ep==a.epochs-1:
            val=validation(model,yv);row={'epoch':ep+1,'train':total/len(yr),'val':val,'seconds':time.time()-start};hist.append(row)
            if val<best:best=val;save(ep+1)
            print(row,flush=True);(out/'history.json').write_text(json.dumps(hist,indent=2))
    print('COMPLETE; test not evaluated',flush=True)
if __name__=='__main__':main()
