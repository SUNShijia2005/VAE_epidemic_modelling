"""Validation-only selection followed by a new 2,000-curve holdout."""
import json,hashlib
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
    out=Path('results_elbo/sampling_ablation/evaluation');out.mkdir(parents=True,exist_ok=True)
    paths={'default':'results_elbo/curved_hybrid/model.pt',
           'fresh':'results_elbo/fresh_ablation/hybrid_fresh/model.pt',
           'mc8':'results_elbo/sampling_ablation/mc8/model.pt',
           'mc32':'results_elbo/sampling_ablation/mc32/model.pt'}
    models={n:load_model(p) for n,p in paths.items()}
    yv=E.make_data(500,7102)[0];scores={}
    for name,model in models.items():
        vals=[]
        for seed in range(8901,8906):
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                vals.append(sum(C.elbo(model,b,128).item()*len(b) for b in yv.split(128))/len(yv))
        scores[name]={'mean':float(np.mean(vals)),'replicates':vals}
        print('VALIDATION',name,scores[name],flush=True)
    selected=min(scores,key=lambda n:scores[n]['mean'])
    selection={'selected':selected,'checkpoint':paths[selected],
               'sha256':hashlib.sha256(Path(paths[selected]).read_bytes()).hexdigest(),
               'scores':scores,'test_seed':13103,'n_test':2000,
               'criterion':'five-stream observation-only validation ELBO, 128 draws; before test generation'}
    (out/'selection.json').write_text(json.dumps(selection,indent=2))
    print('SELECTION',selected,flush=True)
    y,theta,_,_=E.make_data(2000,13103)
    truth=np.column_stack([theta.numpy(),(theta[:,0]/theta[:,1]).numpy()])
    result={'selection':selection,'models':{}}
    for name,model in models.items():
        pred=predict(model,y);stats={}
        for j,par in enumerate(pred['names']):
            m=pred['mean'][:,j];t=truth[:,j]
            stats[par]={'corr':float(np.corrcoef(m,t)[0,1]),'rmse':float(np.sqrt(np.mean((m-t)**2))),
                        'bias':float(np.mean(m-t)),
                        'cover90':float(np.mean((pred['lo90'][:,j]<=t)&(t<=pred['hi90'][:,j])))}
        result['models'][name]=stats
        np.savez(out/(name+'.npz'),truth=truth,y=y.numpy(),mean=pred['mean'],lo=pred['lo90'],hi=pred['hi90'])
        (out/'evaluation.json').write_text(json.dumps(result,indent=2))
        print('TEST',name,json.dumps(stats),flush=True)
    ref=reference(y);np.savez(out/'alpha_reference.npz',**ref)
    coarse=reference(y[:32],size=501,span=5.)
    t=truth[:,1]
    result['pure_sir_alpha_reference']={'corr':float(np.corrcoef(ref['mean'],t)[0,1]),
        'rmse':float(np.sqrt(np.mean((ref['mean']-t)**2))),
        'cover90':float(np.mean((ref['lo']<=t)&(t<=ref['hi']))),
        'max_boundary_mass':float(ref['edge'].max()),
        'first32_max_mean_change_vs_coarse':float(np.max(np.abs(ref['mean'][:32]-coarse['mean'])))}
    assert ref['edge'].max()<1e-4
    assert result['pure_sir_alpha_reference']['first32_max_mean_change_vs_coarse']<5e-4
    rng=np.random.default_rng(13104);ix=rng.integers(0,len(t),size=(4000,len(t)))
    for new,old in [('mc32','mc8'),(selected,'default')]:
        p=np.load(out/(new+'.npz'));q=np.load(out/(old+'.npz'))
        mse=(p['mean'][:,1]-t)**2-(q['mean'][:,1]-t)**2
        cover=((p['lo'][:,1]<=t)&(t<=p['hi'][:,1])).astype(float)-((q['lo'][:,1]<=t)&(t<=q['hi'][:,1]))
        result.setdefault('paired_comparisons',{})[new+'_minus_'+old]={
            'mse_difference':float(mse.mean()),'mse_bootstrap95':np.quantile(mse[ix].mean(1),[.025,.975]).tolist(),
            'coverage_difference':float(cover.mean()),'coverage_bootstrap95':np.quantile(cover[ix].mean(1),[.025,.975]).tolist()}
    for name in models:
        p=np.load(out/(name+'.npz'))
        result.setdefault('alpha_center_error_vs_pure_sir',{})[name]=float(np.sqrt(np.mean(((p['mean'][:,1]-ref['mean'])/ref['sd'])**2)))
    (out/'evaluation.json').write_text(json.dumps(result,indent=2))
    print('COMPLETE',json.dumps(result['paired_comparisons']),flush=True)


if __name__=='__main__':main()
