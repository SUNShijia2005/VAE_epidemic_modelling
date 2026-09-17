"""Final held-out evaluation after observation-only validation selection.
No optimization or numerical-reference use in encoder predictions.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
import train_elbo_only as T
import experiment_alpha as E
import sir_hybrid_vae as H
from infer_elbo_only import predict

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',default=['results_elbo/matched','results_elbo/hybrid','results_elbo/cnn_control','results_elbo/hybrid_seed17'])
    ap.add_argument('--seed',type=int,default=9103);ap.add_argument('--n_test',type=int,default=1000)
    ap.add_argument('--out',default='results_elbo/final');a=ap.parse_args();torch.set_num_threads(1)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    y,truth,_,_=E.make_data(a.n_test,a.seed)
    result={'test_seed':a.seed,'n_test':a.n_test,'inference':'One encoder pass, exact lognormal marginal summaries; no optimization, sampling or labels.',
            'fixed_rho':.1,'fixed_k':10.,'models':{}}
    for name in a.models:
        p=Path(name);ck=torch.load(p/'model.pt',weights_only=False)
        model=T.Model(ck['config']['scale'],ck['config'].get('encoder','mlp'))
        model.load_state_dict(ck['model']);model.eval()
        prediction=predict(model,y)
        values=np.column_stack([truth.numpy(),(truth[:,0]/truth[:,1]).numpy()])
        rows={}
        for j,par in enumerate(prediction['names']):
            mean=prediction['mean'][:,j];true=values[:,j]
            rows[par]={'corr':float(np.corrcoef(mean,true)[0,1]),'rmse':float(np.sqrt(np.mean((mean-true)**2))),
                       'bias':float(np.mean(mean-true)),
                       'cover90':float(np.mean((prediction['lo90'][:,j]<=true)&(true<=prediction['hi90'][:,j])))}
        mu=torch.from_numpy(prediction['latent_mean']);L=torch.from_numpy(prediction['latent_cholesky'])
        ztrue=(truth.log()-torch.tensor([H.LOG_BETA_M,H.LOG_ALPHA_M]))/torch.tensor([H.LOG_BETA_S,H.LOG_ALPHA_S])
        whitened=torch.linalg.solve_triangular(L,(ztrue-mu).unsqueeze(-1),upper=False).squeeze(-1)
        joint_cover=float((whitened.square().sum(1)<=4.605170185988092).float().mean())
        alpha_sd=(L@L.transpose(1,2))[:,1,1].sqrt()
        alpha_error=(ztrue[:,1]-mu[:,1])/alpha_sd
        calibration={}
        for nominal in [.5,.8,.9,.95]:
            quantile=torch.distributions.Normal(0.,1.).icdf(torch.tensor((1+nominal)/2))
            calibration[str(nominal)]=float((alpha_error.abs()<=quantile).float().mean())
        mr=torch.from_numpy(prediction['residual_mean'])
        amplitude=float((model.res(mr)*model.nn_scale).abs().mean()) if model.res is not None else 0.
        result['models'][p.name]={'metrics':rows,'checkpoint':{k:v for k,v in ck.items() if k!='model'},
                                'checkpoint_sha256':hashlib.sha256((p/'model.pt').read_bytes()).hexdigest(),
                                'mean_abs_log_beta_correction':amplitude,'joint_beta_alpha_cover90':joint_cover,
                                'alpha_calibration':calibration}
        np.savez(out/(p.name+'.npz'),truth=truth.numpy(),y=y.numpy(),mean=prediction['mean'],
                 lo=prediction['lo90'],hi=prediction['hi90'],mu_z=prediction['latent_mean'],L=prediction['latent_cholesky'])
        (out/'evaluation.json').write_text(json.dumps(result,indent=2));print(p.name,json.dumps(rows),flush=True)
    # Numerical reference only after every encoder has produced predictions.
    # It represents the pure SIR control, NOT the neural-residual hybrid posterior.
    for size,span in [(501,5.),(751,6.)]:
        sums={};means=[];edges=[]
        for start in range(0,len(y),50):
            end=min(start+50,len(y))
            stats,pm,edge=E.grid_reference(y[start:end],truth[start:end],size,span)
            means.append(pm);edges.append(edge)
            for par in ['beta','alpha','R0']:
                sums.setdefault(par,[]).append((stats[par],end-start))
        pm=torch.cat(means).numpy();tr=np.column_stack([truth.numpy(),(truth[:,0]/truth[:,1]).numpy()])
        metrics={}
        for j,par in enumerate(['beta','alpha','R0']):
            metrics[par]={'corr':float(np.corrcoef(pm[:,j],tr[:,j])[0,1]),'rmse':float(np.sqrt(np.mean((pm[:,j]-tr[:,j])**2))),
                          'bias':float(np.mean(pm[:,j]-tr[:,j])),
                          'cover90':sum(m['cover90']*n for m,n in sums[par])/len(y)}
        row={'size':size,'span':span,'metrics':metrics,'max_boundary_mass':max(edges)}
        if 'old_mean' in locals():row['max_mean_change']=float(np.max(np.abs(pm-old_mean)))
        old_mean=pm
        result.setdefault('constant_sir_grid',[]).append(row)
        np.save(out/'grid_mean.npy',pm)
        (out/'evaluation.json').write_text(json.dumps(result,indent=2));print('GRID',json.dumps(row),flush=True)

if __name__=='__main__':main()
