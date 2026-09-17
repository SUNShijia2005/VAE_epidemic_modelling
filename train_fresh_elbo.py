"""Paired fixed-vs-fresh curves ablation. Same model, optimizer and ELBO.
Only observed curves enter training. No numerical reference or theta targets.
"""
import argparse,json,time
from pathlib import Path
import torch
from torch import nn
import train_curved_elbo as C
import experiment_alpha as E
from infer_elbo_only import load_model


def observations(n,seed):
    # Simulator RNG must not alter the paired optimizer/sampling RNG stream.
    with torch.random.fork_rng():
        return E.make_data(n,seed)[0]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--kind',choices=['matched','hybrid'],required=True)
    ap.add_argument('--data',choices=['fixed','fresh'],required=True)
    ap.add_argument('--epochs',type=int,default=60)
    a=ap.parse_args();torch.set_num_threads(1)
    checkpoint=f'results_elbo/curved_{a.kind}/model.pt'
    model=load_model(checkpoint)
    yr=observations(8000,7101);yv=observations(500,7102)
    # Keep the original training-only input normalization for both arms.
    torch.manual_seed(127)
    opt=torch.optim.Adam(model.parameters(),lr=.00015)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=.000005)
    out=Path(f'results_elbo/fresh_ablation/{a.kind}_{a.data}');out.mkdir(parents=True,exist_ok=True)
    best=C.validation(model,yv);history=[{'epoch':0,'val':best}]
    def save(epoch):
        torch.save({'model':model.state_dict(),'config':vars(a),'scale':model.nn_scale,
                    'selected_epoch':epoch,'val_loss':best,'initial_checkpoint':checkpoint,
                    'objective':'ordinary ELBO with curved posterior; observed curves only; paired fixed/fresh ablation'},out/'model.pt')
    save(0);start=time.time()
    print(a.kind,a.data,'initial',best,flush=True)
    for ep in range(a.epochs):
        if a.data=='fresh':yr=observations(8000,20000+ep)
        model.train();total=0
        for ix in torch.randperm(len(yr)).split(128):
            loss=C.elbo(model,yr[ix],8)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite ELBO')
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            total+=loss.item()*len(ix)
        sch.step()
        if ep%5==0 or ep==a.epochs-1:
            val=C.validation(model,yv)
            row={'epoch':ep+1,'train':total/len(yr),'val':val,'seconds':time.time()-start};history.append(row)
            if val<best:best=val;save(ep+1)
            (out/'history.json').write_text(json.dumps(history,indent=2))
            print(a.kind,a.data,row,flush=True)
    print('Complete; no test data evaluated.',flush=True)


if __name__=='__main__':main()
