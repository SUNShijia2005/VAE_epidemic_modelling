"""Predeclared validation selection, then seed 12103 holdout evaluation."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import experiment_alpha as E
import train_curved_elbo as C
from infer_elbo_only import load_model,predict
from diagnose_alpha_posterior import reference


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    out=Path('results_elbo/fresh_ablation/evaluation');out.mkdir(parents=True,exist_ok=True)
    paths={'hybrid_original':'results_elbo/curved_hybrid/model.pt',
           'matched_original':'results_elbo/curved_matched/model.pt'}
    for kind in ['matched','hybrid']:
        for data in ['fixed','fresh']:
            paths[f'{kind}_{data}']=f'results_elbo/fresh_ablation/{kind}_{data}/model.pt'
    models={k:load_model(v) for k,v in paths.items()}
    yv=E.make_data(500,7102)[0];scores={}
    for name,model in models.items():
        vals=[]
        for seed in range(8801,8806):
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                vals.append(sum(C.elbo(model,b,32).item()*len(b) for b in yv.split(128))/len(yv))
        scores[name]={'mean':float(np.mean(vals)),'replicates':vals}
    selected=min((n for n in models if n.startswith('hybrid')),key=lambda n:scores[n]['mean'])
    result={'selection':{'selected':selected,'scores':scores,'checkpoint':paths[selected],
                        'sha256':hashlib.sha256(Path(paths[selected]).read_bytes()).hexdigest(),
                        'test_seed':12103,'n_test':2000,
                        'criterion':'five-stream mean validation ELBO; selection precedes test generation'},'models':{}}
    (out/'selection.json').write_text(json.dumps(result['selection'],indent=2))
    print('SELECTION',json.dumps(result['selection']),flush=True)
    y,theta,_,_=E.make_data(2000,12103)
    truth=np.column_stack([theta.numpy(),(theta[:,0]/theta[:,1]).numpy()])
    for name,model in models.items():
        pred=predict(model,y);stats={}
        for j,par in enumerate(pred['names']):
            m=pred['mean'][:,j];t=truth[:,j]
            stats[par]={'corr':float(np.corrcoef(m,t)[0,1]),'rmse':float(np.sqrt(np.mean((m-t)**2))),
                        'bias':float(np.mean(m-t)),
                        'cover90':float(np.mean((pred['lo90'][:,j]<=t)&(t<=pred['hi90'][:,j])))}
        result['models'][name]=stats
        np.savez(out/(name+'.npz'),truth=truth,y=y.numpy(),mean=pred['mean'],lo=pred['lo90'],hi=pred['hi90'])
        print(name,json.dumps(stats),flush=True)
        (out/'evaluation.json').write_text(json.dumps(result,indent=2))
    ref=reference(y)
    np.savez(out/'alpha_reference.npz',**ref)
    t=truth[:,1];m=ref['mean']
    result['pure_sir_alpha_reference']={'corr':float(np.corrcoef(m,t)[0,1]),
                'rmse':float(np.sqrt(np.mean((m-t)**2))),
                'cover90':float(np.mean((ref['lo']<=t)&(t<=ref['hi']))),
                'max_boundary_mass':float(ref['edge'].max())}
    assert ref['edge'].max()<1e-4
    print('REFERENCE',json.dumps(result['pure_sir_alpha_reference']),flush=True)
    # Paired bootstrap is descriptive uncertainty, never used for selection.
    rng=np.random.default_rng(12104)
    for kind in ['matched','hybrid']:
        fixed=np.load(out/f'{kind}_fixed.npz');fresh=np.load(out/f'{kind}_fresh.npz')
        df=(fresh['mean'][:,1]-t)**2-(fixed['mean'][:,1]-t)**2
        dc=((fresh['lo'][:,1]<=t)&(t<=fresh['hi'][:,1])).astype(float)-((fixed['lo'][:,1]<=t)&(t<=fixed['hi'][:,1]))
        ix=rng.integers(0,len(t),size=(4000,len(t)))
        result.setdefault('fresh_minus_fixed',{})[kind]={
            'mse_difference':float(df.mean()),'mse_difference_bootstrap95':np.quantile(df[ix].mean(1),[.025,.975]).tolist(),
            'coverage_difference':float(dc.mean()),'coverage_difference_bootstrap95':np.quantile(dc[ix].mean(1),[.025,.975]).tolist()}
    (out/'evaluation.json').write_text(json.dumps(result,indent=2))
    print('COMPLETE',json.dumps(result['fresh_minus_fixed']),flush=True)


if __name__=='__main__':main()
