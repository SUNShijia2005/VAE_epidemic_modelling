"""Select by observation-only validation ELBO, then evaluate on a NEW holdout."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
import train_curved_elbo as C
import experiment_alpha as E
import sir_hybrid_vae as H
from infer_elbo_only import load_model,predict



def main():
    torch.set_num_threads(1)
    candidates=['curved_hybrid','continued_gaussian','cumulative_gaussian']
    checkpoints={n:torch.load(Path('results_elbo')/n/'model.pt',weights_only=False) for n in candidates}
    selected=min(candidates,key=lambda n:checkpoints[n]['val_loss'])
    out=Path('results_elbo/confirmation');out.mkdir(parents=True,exist_ok=True)
    selection={'criterion':'validation ELBO only, no true labels or posterior reference',
               'selected':selected,'candidates':{n:{'val_loss':v['val_loss'],'epoch':v['selected_epoch']} for n,v in checkpoints.items()},
               'test_seed':10103,'n_test':1000}
    (out/'selection.json').write_text(json.dumps(selection,indent=2));print('SELECTION',selection,flush=True)
    # Do not generate the new test set until selection is recorded.
    y,truth,_,_=E.make_data(1000,10103)
    result={'selection':selection,'models':{}}
    cases=[('previous_gaussian',None),('selected_hybrid',checkpoints[selected]),
           ('curved_matched',torch.load('results_elbo/curved_matched/model.pt',weights_only=False))]
    for tag,ck in cases:
        if ck is None:
            model=load_model('results_elbo/hybrid/model.pt');pred=predict(model,y);curvature=np.zeros(len(y))
        else:
            model=C.Model(ck['scale'],ck['config'].get('cumulative',False));model.load_state_dict(ck['model']);model.eval()
            pred=predict(model,y);curvature=pred['curvature']
        values=np.column_stack([truth.numpy(),(truth[:,0]/truth[:,1]).numpy()]);stats={}
        for j,n in enumerate(pred['names']):
            pm=pred['mean'][:,j];t=values[:,j]
            stats[n]={'corr':float(np.corrcoef(pm,t)[0,1]),'rmse':float(np.sqrt(np.mean((pm-t)**2))),
                      'bias':float(np.mean(pm-t)),'cover90':float(np.mean((pred['lo90'][:,j]<=t)&(t<=pred['hi90'][:,j])))}
        mu=torch.from_numpy(pred['latent_mean']);L=torch.from_numpy(pred['latent_cholesky'])
        ztrue=(truth.log()-torch.tensor([H.LOG_BETA_M,H.LOG_ALPHA_M]))/torch.tensor([H.LOG_BETA_S,H.LOG_ALPHA_S])
        var=L[:,1,:].square().sum(1);base=ztrue.clone()
        base[:,0]-=torch.from_numpy(curvature)*((ztrue[:,1]-mu[:,1]).square()-var)
        white=torch.linalg.solve_triangular(L,(base-mu)[:,:,None],upper=False).squeeze(-1)
        result['models'][tag]={'metrics':stats,'joint_cover90':float((white.square().sum(1)<=4.605170185988092).float().mean()),
                             'mean_abs_curvature':float(np.abs(curvature).mean()),
                             'checkpoint':{k:v for k,v in ck.items() if k!='model'} if ck else 'results_elbo/hybrid/model.pt'}
        np.savez(out/(tag+'.npz'),truth=truth.numpy(),y=y.numpy(),mean=pred['mean'],lo=pred['lo90'],hi=pred['hi90'])
        print(tag,json.dumps(result['models'][tag]),flush=True)
        (out/'evaluation.json').write_text(json.dumps(result,indent=2))
    for size,span in [(501,5.),(751,6.)]:
        pm=[];parts=[];edges=[]
        for start in range(0,len(y),50):
            stats,means,edge=E.grid_reference(y[start:start+50],truth[start:start+50],size,span)
            pm.append(means.numpy());parts.append(stats);edges.append(edge)
        pm=np.concatenate(pm);tr=np.column_stack([truth.numpy(),(truth[:,0]/truth[:,1]).numpy()]);stats={}
        for j,n in enumerate(['beta','alpha','R0']):
            stats[n]={'corr':float(np.corrcoef(pm[:,j],tr[:,j])[0,1]),'rmse':float(np.sqrt(np.mean((pm[:,j]-tr[:,j])**2))),
                      'bias':float(np.mean(pm[:,j]-tr[:,j])),'cover90':float(np.mean([s[n]['cover90'] for s in parts]))}
        row={'size':size,'span':span,'metrics':stats,'max_boundary_mass':max(edges)}
        if 'old' in locals():row['max_mean_change']=float(np.max(np.abs(old-pm)))
        old=pm;result.setdefault('constant_sir_grid',[]).append(row)
        (out/'evaluation.json').write_text(json.dumps(result,indent=2));np.save(out/'grid_mean.npy',pm)
        print('GRID',json.dumps(row),flush=True)
if __name__=='__main__':main()
