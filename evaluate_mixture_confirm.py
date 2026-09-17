"""Validation-only selection and untouched third holdout for the final candidates."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
import train_curved_elbo as C
import train_mixture_elbo as M
import experiment_alpha as E
from infer_elbo_only import load_model,predict

@torch.no_grad()
def main():
    torch.set_num_threads(1)
    paths={'curved':'results_elbo/curved_hybrid/model.pt','mixture':'results_elbo/mixture_hybrid/model.pt'}
    models={k:load_model(v) for k,v in paths.items()}
    yv=E.make_data(500,7102)[0];scores={}
    for name,model in models.items():
        losses=[]
        for seed in [8801,8802,8803,8804,8805]:
            with torch.random.fork_rng():
                torch.manual_seed(seed);fn=M.objective if name=='mixture' else C.elbo
                draws=16 if name=='mixture' else 32
                losses.append(sum(fn(model,b,draws).item()*len(b) for b in yv.split(128))/len(yv))
        scores[name]={'mean_validation_elbo':float(np.mean(losses)),'replicates':losses}
    selected=min(scores,key=lambda n:scores[n]['mean_validation_elbo'])
    out=Path('results_elbo/final_holdout');out.mkdir(parents=True,exist_ok=True)
    selection={'checkpoint_sha256':hashlib.sha256(Path(paths[selected]).read_bytes()).hexdigest(),'torch_version':str(torch.__version__),'selected':selected,'scores':scores,'checkpoint':paths[selected],'test_seed':11103,'n_test':1000,
               'criterion':'Mean observation-only validation ELBO over five fixed random streams; 32 total likelihood samples each.'}
    (out/'selection.json').write_text(json.dumps(selection,indent=2));print('SELECTION',selection,flush=True)
    y,truth,_,_=E.make_data(1000,11103)
    cases={'cnn':load_model('results_elbo/cnn_control/model.pt'),'gaussian':load_model('results_elbo/hybrid/model.pt'),**models}
    result={'selection':selection,'models':{}}
    true=np.column_stack([truth.numpy(),(truth[:,0]/truth[:,1]).numpy()])
    for name,model in cases.items():
        pred=predict(model,y);stats={}
        for j,par in enumerate(pred['names']):
            pm=pred['mean'][:,j];t=true[:,j]
            stats[par]={'corr':float(np.corrcoef(pm,t)[0,1]),'rmse':float(np.sqrt(np.mean((pm-t)**2))),
                        'bias':float(np.mean(pm-t)),'cover90':float(np.mean((pred['lo90'][:,j]<=t)&(t<=pred['hi90'][:,j])))}
        result['models'][name]={'metrics':stats}
        np.savez(out/(name+'.npz'),truth=true,y=y.numpy(),mean=pred['mean'],lo=pred['lo90'],hi=pred['hi90'])
        print(name,json.dumps(stats),flush=True);(out/'evaluation.json').write_text(json.dumps(result,indent=2))
    for size,span in [(501,5.),(751,6.)]:
        means=[];parts=[];edges=[]
        for start in range(0,len(y),50):
            stats,pm,edge=E.grid_reference(y[start:start+50],truth[start:start+50],size,span)
            means.append(pm.numpy());parts.append(stats);edges.append(edge)
        pm=np.concatenate(means);stats={}
        for j,par in enumerate(['beta','alpha','R0']):
            stats[par]={'corr':float(np.corrcoef(pm[:,j],true[:,j])[0,1]),'rmse':float(np.sqrt(np.mean((pm[:,j]-true[:,j])**2))),
                        'bias':float(np.mean(pm[:,j]-true[:,j])),'cover90':float(np.mean([s[par]['cover90'] for s in parts]))}
        row={'size':size,'span':span,'metrics':stats,'max_boundary_mass':max(edges)}
        if 'old' in locals():row['max_mean_change']=float(np.max(np.abs(pm-old)))
        old=pm;result.setdefault('constant_sir_reference',[]).append(row)
        print('GRID',json.dumps(row),flush=True);(out/'evaluation.json').write_text(json.dumps(result,indent=2));np.save(out/'grid_mean.npy',pm)
if __name__=='__main__':main()
