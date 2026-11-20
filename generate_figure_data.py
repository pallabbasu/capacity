"""Compute the capacity curves for Janus and the cosine-potential family.

Dependencies: numpy, mpmath. Example:
  python3 generate_figure_data.py

All central-charge columns are normalized by c. The Janus solver uses an
angular finite-difference spectrum and radial tridiagonal solve.
The cosine-potential solver evaluates a gamma-function quadrature.
"""
from pathlib import Path
import argparse,csv,json,math,sys,time
import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'reproduction'))
from janus_exact_kernel import Kernel
from potential_beta_spectral import evaluate

ASYMPTOTIC_OFFSET=-0.28753607636500674679483642185741346

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:writer.writerow({k:format(float(v),'.12g') for k,v in row.items()})

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--h',type=float,default=.0125)
    ap.add_argument('--mumax',type=float,default=10.)
    ap.add_argument('--vmax',type=float,default=10.)
    ap.add_argument('--janus-points',type=int,default=91)
    ap.add_argument('--beta-points',type=int,default=80)
    ap.add_argument('--dps',type=int,default=25)
    ap.add_argument('--output-dir', type=Path, default=HERE / 'results')
    ap.add_argument('--skip-validation',action='store_true')
    args=ap.parse_args();started=time.time()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    problem=Kernel(args.h,args.mumax,args.vmax,quad=256)
    # Include the independently checked minimum, crossing, and ratio minimum.
    deltas=np.unique(np.r_[np.linspace(0,8,args.janus_points),1.3475,2.1040,3.1307])
    janus=[]
    for i,delta in enumerate(deltas):
        kappa=1/math.cosh(delta/math.sqrt(2));result=problem.solve(kappa)
        entropy=math.sqrt((1+kappa)/2)
        transmission=1. if delta==0 else math.tanh(delta/math.sqrt(2))/(delta/math.sqrt(2))
        janus.append(dict(delta=delta,capacity=result['capacity'],entropy=entropy,
                          transmission=transmission,upper=entropy**2/transmission,
                          lowerJanus=(1+entropy**2)/2,
                          asymptote=delta/(2*math.sqrt(2))+ASYMPTOTIC_OFFSET))
        if i%20==0:print(f'Janus {i+1}/{len(deltas)}',flush=True)
    write_csv(output_dir/'figure_janus.csv',janus)

    betas=np.linspace(.02,1,args.beta_points)
    betas[np.argmin(abs(betas-.567380375))]=.567380375
    betas.sort();beta=[]
    for i,b in enumerate(betas):
        value=evaluate(float(b),args.dps)
        beta.append(dict(beta=b,capacity=float(value['ccap_over_c']),entropy=b,
                         upper=1.,lower=1-4*b*(1-b*b)/9))
        if i%20==0:print(f'Beta {i+1}/{len(betas)}',flush=True)
    write_csv(output_dir/'figure_beta.csv',beta)

    checks={}
    if not args.skip_validation:
        fine=Kernel(args.h/2,args.mumax,args.vmax,quad=256)
        selected=[0.,1.3475,4.,8.]
        comparisons=[]
        for delta in selected:
            kappa=1/math.cosh(delta/math.sqrt(2))
            coarse=problem.solve(kappa)['capacity'];refined=fine.solve(kappa)['capacity']
            comparisons.append(dict(delta=delta,coarse=coarse,refined=refined,
                                    absolute_difference=abs(coarse-refined)))
        checks['janus_half_mesh']=comparisons
        checks['max_selected_mesh_difference']=max(r['absolute_difference'] for r in comparisons)
        checks['beta_precision_checks']=[]
        for b in [.02,.567380375,.9]:
            coarse=float(evaluate(b,args.dps)['ccap_over_c'])
            refined=float(evaluate(b,args.dps+10)['ccap_over_c'])
            checks['beta_precision_checks'].append(dict(beta=b,coarse=coarse,refined=refined,
                                                        absolute_difference=abs(coarse-refined)))
    checks['janus_capacity_above_lower']=all(r['capacity']>=r['lowerJanus']-1e-10 for r in janus)
    checks['janus_capacity_below_upper']=all(r['capacity']<=r['upper']+1e-10 for r in janus)
    checks['beta_capacity_above_lower']=all(r['capacity']>=r['lower']-1e-10 for r in beta)
    checks['beta_capacity_below_upper']=all(r['capacity']<=r['upper']+1e-10 for r in beta)
    parameters = {key: value for key, value in vars(args).items() if key != 'output_dir'}
    manifest=dict(parameters=parameters,janus_rows=len(janus),beta_rows=len(beta),
                  normalization='All central-charge quantities divided by c',
                  janus_method='Independent angular finite differences plus radial tridiagonal resolvent',
                  beta_method='mpmath quadrature of independently verified gamma-function Green function',
                  asymptotic_offset=ASYMPTOTIC_OFFSET,checks=checks,seconds=time.time()-started,
                  caveat='Mesh/precision checks are numerical comparisons, not certified continuum error bounds.')
    (output_dir/'figure_data_validation.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
