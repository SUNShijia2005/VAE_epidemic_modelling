"""Only vary ELBO sample count; same observations, batches and model family."""
import argparse,json,time
from pathlib import Path
import torch
from torch import nn
import train_curved_elbo as C
from train_fresh_elbo import observations
from infer_elbo_only import load_model


@torch.no_grad()
def validation(model,y):
    model.eval()
    with torch.random.fork_rng():
        torch.manual_seed(8201)
        return sum(C.elbo(model,b,64).item()*len(b) for b in y.split(128))/len(y)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--draws',type=int,choices=[8,32],required=True)
    ap.add_argument('--epochs',type=int,default=40);a=ap.parse_args();torch.set_num_threads(1)
    initial='results_elbo/fresh_ablation/hybrid_fresh/model.pt'
    model=load_model(initial);yv=observations(500,7102)
    torch.manual_seed(227)
    opt=torch.optim.Adam(model.parameters(),lr=.0001)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=.000003)
    out=Path(f'results_elbo/sampling_ablation/mc{a.draws}');out.mkdir(parents=True,exist_ok=True)
    best=validation(model,yv);history=[{'epoch':0,'val':best}]
    def save(epoch):
        torch.save({'model':model.state_dict(),'config':vars(a),'scale':model.nn_scale,
                    'initial_checkpoint':initial,'selected_epoch':epoch,'val_loss':best,
                    'objective':'ordinary ELBO; curved posterior; only observed curves; MC sample-count ablation'},out/'model.pt')
    save(0);start=time.time();print('initial',best,flush=True)
    for ep in range(a.epochs):
        yr=observations(8000,30000+ep)
        # Independent shuffle RNG keeps all batch contents identical across arms.
        order=torch.randperm(len(yr),generator=torch.Generator().manual_seed(40000+ep))
        model.train();total=0
        for ix in order.split(128):
            loss=C.elbo(model,yr[ix],a.draws)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite ELBO')
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            total+=loss.item()*len(ix)
        sch.step()
        if ep%5==0 or ep==a.epochs-1:
            val=validation(model,yv);row={'epoch':ep+1,'train':total/len(yr),'val':val,'seconds':time.time()-start};history.append(row)
            if val<best:best=val;save(ep+1)
            (out/'history.json').write_text(json.dumps(history,indent=2));print(row,flush=True)
    print('Complete; no test observations or parameter labels used.',flush=True)


if __name__=='__main__':main()
