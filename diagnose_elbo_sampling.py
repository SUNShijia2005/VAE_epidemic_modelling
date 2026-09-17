"""Training-observation-only sampling noise diagnostic; no truth/reference."""
import json
from pathlib import Path
import numpy as np
import torch
import train_curved_elbo as C
from train_fresh_elbo import observations
from infer_elbo_only import load_model


def main():
    torch.set_num_threads(1)
    out=Path('results_elbo/sampling_ablation');out.mkdir(parents=True,exist_ok=True)
    model=load_model('results_elbo/fresh_ablation/hybrid_fresh/model.pt')
    y=observations(8000,7101)
    result={}
    for batch in [0,1]:
        rows={}
        for draws in [8,32]:
            grads=[];losses=[]
            for seed in range(64):
                torch.manual_seed(50000+seed)
                loss=C.elbo(model,y[batch*128:(batch+1)*128],draws)
                # Row 1 controls the latent alpha mean (not physical truth).
                w,b=torch.autograd.grad(loss,[model.enc.head.weight,model.enc.head.bias])
                grads.append(torch.cat([w[1],b[1:2]]).numpy());losses.append(loss.item())
            g=np.stack(grads)
            rows[str(draws)]={'loss_mean':float(np.mean(losses)), 'loss_sd':float(np.std(losses,ddof=1)),
                'alpha_mean_gradient_noise_rms':float(np.sqrt(g.var(0,ddof=1).sum())),
                'mean_alpha_gradient_norm':float(np.linalg.norm(g.mean(0)))}
        result[str(batch)]=rows
        print(batch,json.dumps(rows),flush=True)
    (out/'sampling_noise.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
