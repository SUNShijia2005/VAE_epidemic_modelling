"""Frozen-model diagnosis, never used by training or public inference.

Reuses seed 11103 as development data. Numerical reference is exact up to
grid discretization ONLY for the pure SIR controls, not the Hybrid model.
Counterfactual intervals below are diagnostic decompositions, not predictions.
"""
import json
from pathlib import Path
import numpy as np
import torch
import sir_hybrid_vae as H
from infer_elbo_only import load_model, predict


@torch.no_grad()
def reference(y, size=751, span=6.):
    axis = torch.linspace(-span, span, size)
    z = torch.stack(torch.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)
    theta = torch.exp(z * torch.tensor([H.LOG_BETA_S, H.LOG_ALPHA_S]) +
                      torch.tensor([H.LOG_BETA_M, H.LOG_ALPHA_M]))
    kernels, offsets = [], []
    for ix in torch.arange(len(z)).split(4096):
        mean = (H.sir_incidence(theta[ix, 0], theta[ix, 1]) * .1).clamp(min=1e-6).double()
        kernels.append((mean.log() - (mean + H.NB_K).log()).T)
        offsets.append(-H.NB_K * (mean + H.NB_K).log().sum(1) - .5 * z[ix].double().square().sum(1))
    kernel, offset = torch.cat(kernels, 1), torch.cat(offsets)
    logalpha = H.LOG_ALPHA_M + H.LOG_ALPHA_S * axis.double()
    alpha = logalpha.exp()
    rows = []
    for start in range(0, len(y), 32):
        w = torch.softmax(y[start:start+32].double() @ kernel + offset, 1).reshape(-1, size, size)
        marginal = w.sum(1)
        lm = marginal @ logalpha
        lv = (marginal @ logalpha.square() - lm.square()).clamp(min=0)
        am = marginal @ alpha
        av = (marginal @ alpha.square() - am.square()).clamp(min=0)
        cdf = marginal.cumsum(1)
        lo = alpha[(cdf < .05).sum(1).clamp(max=size-1)]
        hi = alpha[(cdf < .95).sum(1).clamp(max=size-1)]
        edge = w[:, :2, :].sum((1,2)) + w[:, -2:, :].sum((1,2)) + w[:, 2:-2, :2].sum((1,2)) + w[:, 2:-2, -2:].sum((1,2))
        rows.append(torch.stack([am, av.sqrt(), lo, hi, lm, lv.sqrt(), edge], 1).numpy())
    return dict(zip(['mean','sd','lo','hi','logmean','logsd','edge'], np.concatenate(rows).T))


def summarize(pred, ref, truth, mask):
    p = {k:v[mask] for k,v in pred.items()}
    r = {k:v[mask] for k,v in ref.items()}
    t = truth[mask]
    cover = lambda lo, hi: float(np.mean((lo <= t) & (t <= hi)))
    # Swapping centers or widths uses numerical reference ONLY for diagnosis.
    center_fixed_lo = np.exp(r['logmean'] - 1.64485362695*p['logsd'])
    center_fixed_hi = np.exp(r['logmean'] + 1.64485362695*p['logsd'])
    shift = p['logmean'] - r['logmean']
    return {'n':int(mask.sum()), 'rmse':float(np.sqrt(np.mean((p['mean']-t)**2))),
            'coverage':cover(p['lo'],p['hi']), 'reference_coverage':cover(r['lo'],r['hi']),
            'median_width_ratio':float(np.median((p['hi']-p['lo'])/(r['hi']-r['lo']))),
            'center_error_in_reference_sd':float(np.sqrt(np.mean(((p['logmean']-r['logmean'])/r['logsd'])**2))),
            'center_only_counterfactual_coverage':cover(center_fixed_lo,center_fixed_hi),
            'width_shape_only_counterfactual_coverage':cover(r['lo']*np.exp(shift),r['hi']*np.exp(shift))}


def main():
    torch.set_num_threads(1)
    out = Path('results_elbo/diagnosis');out.mkdir(exist_ok=True)
    data = np.load('results_elbo/final_holdout/curved.npz')
    y = torch.tensor(data['y']);truth=data['truth'][:,1]
    paths = {'matched_gaussian':'results_elbo/matched/model.pt',
             'matched_curved':'results_elbo/curved_matched/model.pt',
             'hybrid_curved':'results_elbo/curved_hybrid/model.pt'}
    preds={}
    for name,path in paths.items():
        p=predict(load_model(path),y)
        preds[name]={'mean':p['mean'][:,1], 'lo':p['lo90'][:,1], 'hi':p['hi90'][:,1],
                     'logmean':H.LOG_ALPHA_M+H.LOG_ALPHA_S*p['latent_mean'][:,1],
                     'logsd':H.LOG_ALPHA_S*np.sqrt(np.square(p['latent_cholesky'][:,1,:]).sum(1))}
    print('Frozen one-pass predictions completed.',flush=True)
    ref=reference(y)
    np.savez(out/'reference.npz',**ref)
    # Cross-check against independently saved earlier full-grid summaries.
    old=np.load('results_elbo/final_holdout/grid_mean.npy')[:,1]
    assert np.max(np.abs(ref['mean']-old)) < 1e-6
    assert ref['edge'].max() < 1e-4
    counts=y.numpy().sum(1)
    thresholds=np.quantile(counts,[.25,.5,.75])
    groups={'all':np.ones(len(y),dtype=bool)}
    bins=np.digitize(counts,thresholds)
    for i in range(4):groups[f'case_count_quartile_{i+1}']=bins==i
    late=y.numpy()[:,90:].sum(1)/np.maximum(counts,1)
    groups['late_quarter_over_10pct']=late>.1
    groups['late_quarter_at_most_10pct']=late<=.1
    result={'seed':11103,'status':'development diagnosis; no model changes',
            'reference_scope':'constant SIR only; Hybrid comparison is not a same-model posterior comparison',
            'case_count_quartile_boundaries':thresholds.tolist(),'max_grid_boundary_mass':float(ref['edge'].max()),
            'max_reference_mean_difference':float(np.max(np.abs(ref['mean']-old))), 'models':{}}
    for name,p in preds.items():
        result['models'][name]={g:summarize(p,ref,truth,m) for g,m in groups.items() if m.any()}
        np.savez(out/(name+'.npz'),**p)
        print(name,json.dumps(result['models'][name]),flush=True)
    (out/'diagnosis.json').write_text(json.dumps(result,indent=2))
    print('Diagnosis saved; no weights or inference changed.',flush=True)


if __name__=='__main__':main()
