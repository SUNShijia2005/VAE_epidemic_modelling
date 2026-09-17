"""Two-component physical posterior; ordinary observation-only ELBO.
Stratified reparameterization sums over both components, without label targets.
"""
import argparse,json,math,time
from pathlib import Path
import torch
from torch import nn
import train_elbo_only as T
import experiment_alpha as E
import sir_hybrid_vae as H

class Encoder(T.SequenceEncoder):
    def __init__(self,d_res):
        super().__init__(d_res);self.head=nn.Linear(128,12+2*d_res)
    def forward(self,y):
        o=self.head(self.net((self.features(y)-self.center)/self.scale))
        p=o[:,:10].reshape(-1,2,5)
        L=torch.diag_embed(p[:,:,2:4].clamp(-6,2).exp());L=L.clone();L[:,:,1,0]=p[:,:,4]
        return p[:,:,:2],L,o[:,10:12].log_softmax(1),o[:,12:12+self.d_res],o[:,12+self.d_res:].clamp(-8,4)

class Model(E.TwoParameterVAE):
    def __init__(self):super().__init__(.5);self.enc=Encoder(self.z_res_dim)


def objective(model,y,draws=8):
    mu,L,logw,mr,vr=model.enc(y);n=len(y)
    ep=torch.randn(draws//2,2,n,2);ep=torch.cat([ep,-ep])
    er=torch.randn(draws//2,2,n,model.z_res_dim);er=torch.cat([er,-er])
    z=mu.permute(1,0,2)[None]+torch.einsum('bkij,skbj->skbi',L,ep)
    zr=mr[None,None]+(.5*vr).exp()[None,None]*er
    lam=model.decode(z.reshape(-1,2),zr.reshape(draws*2*n,model.z_res_dim))
    ll=H.nb_log_prob(y.repeat(draws*2,1),lam,model.log_k.exp()).sum(1).reshape(draws,2,n)
    q=torch.distributions.MultivariateNormal(mu,scale_tril=L)
    logq=torch.logsumexp(q.log_prob(z[:,:,:,None,:])+logw[None,None],-1)
    logp=-.5*(z.square().sum(-1)+2*math.log(2*math.pi))
    loss=((logq-logp-ll).mean(0)*logw.exp().T).sum(0)+H.kl_std_normal(mr,vr)
    return loss.mean()

@torch.no_grad()
def validation(model,y):
    model.eval()
    with torch.random.fork_rng():
        torch.manual_seed(8201)
        return sum(objective(model,b,32).item()*len(b) for b in y.split(128))/len(y)


def initialize():
    ck=torch.load('results_elbo/hybrid/model.pt',weights_only=False);model=Model()
    state=ck['model'].copy();w=state.pop('enc.head.weight');b=state.pop('enc.head.bias')
    model.load_state_dict(state,strict=False)
    with torch.no_grad():
        model.enc.head.weight.zero_();model.enc.head.bias.zero_()
        for k,sign in [(0,-1),(1,1)]:
            model.enc.head.weight[k*5:k*5+5]=w[:5];model.enc.head.bias[k*5:k*5+5]=b[:5]
            model.enc.head.bias[k*5:k*5+2]+=sign*torch.tensor([.025,.05])
        model.enc.head.weight[12:]=w[5:];model.enc.head.bias[12:]=b[5:]
    return model


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--out',default='results_elbo/mixture_hybrid');a=ap.parse_args()
    torch.set_num_threads(1);torch.manual_seed(37);model=initialize()
    yr=E.make_data(8000,7101)[0];yv=E.make_data(500,7102)[0]
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    opt=torch.optim.Adam(model.parameters(),lr=.00025)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=.000005)
    best=float('inf');hist=[];start=time.time()
    for ep in range(a.epochs):
        model.train();total=0
        for ix in torch.randperm(len(yr)).split(128):
            loss=objective(model,yr[ix],8)
            if not torch.isfinite(loss):raise RuntimeError('non-finite loss')
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=loss.item()*len(ix)
        sch.step()
        if ep%5==0 or ep==a.epochs-1:
            val=validation(model,yv);row={'epoch':ep+1,'train':total/len(yr),'val':val,'seconds':time.time()-start};hist.append(row)
            if val<best:
                best=val;torch.save({'model':model.state_dict(),'selected_epoch':ep+1,'val_loss':val,'config':vars(a),
                    'objective':'ordinary ELBO, stratified two-component mixture; no labels/teacher',
                    'train_seed':7101,'validation_seed':7102},out/'model.pt')
            print(row,flush=True);(out/'history.json').write_text(json.dumps(hist,indent=2))
    print('COMPLETE; no test evaluated',flush=True)
if __name__=='__main__':main()
