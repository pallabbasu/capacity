"""Exact one-dimensional quadrature for the BF kink family replica response.
The formula assumes a fixed interface boundary condition and the constrained
replica branch derived in Supplemental Sections S2, S3 and S6; it is not a generic
Einstein-scalar capacity formula.
"""
import mpmath as mp, json, argparse

def spectral_measure(p):
    return p*mp.tanh(mp.pi*p)/(2*mp.pi)*abs(mp.gamma(mp.mpf('.25')+1j*p/2)/mp.gamma(mp.mpf('.75')+1j*p/2))**2

def radial_green_tip(p,beta):
    q=p*p+mp.mpf('2.25');rho=mp.sqrt(q/beta**2-mp.mpf('.25'))
    return abs(mp.gamma(mp.mpf('.75')+1j*rho/2)/mp.gamma(mp.mpf('1.25')+1j*rho/2))**2/4

def evaluate(beta,dps=35):
    mp.mp.dps=dps;beta=mp.mpf(str(beta))
    I=mp.quad(lambda p:spectral_measure(p)*radial_green_tip(p,beta)/(p*p+mp.mpf('2.25')),[0,1,3,10,30,mp.inf])
    cap=1-2*mp.pi*(1-beta**2)*I
    lower=1-mp.mpf(4)/9*beta*(1-beta**2)
    return dict(beta=str(beta),I=mp.nstr(I,dps-5),ccap_over_c=mp.nstr(cap,dps-5),ceff_over_c=str(beta),proven_lower_bound=mp.nstr(lower,dps-5),capacity_entropy_ratio=mp.nstr(cap/beta,dps-5))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--beta',nargs='+',default=['1','.9','.7','.5','.3','.1']);p.add_argument('--dps',type=int,default=35);p.add_argument('--output');a=p.parse_args()
    out=[evaluate(b,a.dps) for b in a.beta]
    payload=json.dumps(out,indent=2)
    if a.output:
        with open(a.output,'w') as f:f.write(payload+'\n')
    print(payload)
