"""Exploratory radial-gauge Janus perturbation; coefficients require convergence.
Polar Einstein evolution with tip normalization, boundary angular period response.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import argparse,json,time

def cumulative(v,h):
    out=np.zeros_like(v)
    out[1:]=np.cumsum((v[1:]+v[:-1])*(h/2),axis=0)
    return out

def thomas(lower,diag,upper,rhs):
    d=diag.copy();z=rhs.copy()
    for i in range(1,len(d)):
        q=lower[i]/d[i-1]
        d[i]-=q*upper[i-1];z[i]-=q*z[i-1]
    out=np.empty_like(z);out[-1]=z[-1]/d[-1]
    for i in range(len(d)-2,-1,-1):out[i]=(z[i]-upper[i]*out[i+1])/d[i]
    return out

def solve(n=1.0,nr=1600,nt=1024,rmax=9.0,physical_source=True,smoothing='hard',modes=None):
    r=np.linspace(0,rmax,nr+1);dr=r[1]
    k=np.arange(nt//2+1)
    rp=r[1:-1]+dr/2;rm=r[1:-1]-dr/2
    wp=np.sinh(rp)*np.cosh(rp)/dr**2
    wm=np.sinh(rm)*np.cosh(rm)/dr**2
    wr=np.sinh(r[1:-1])*np.cosh(r[1:-1])
    wt=n*n/np.tanh(r[1:-1])
    low=wm[:,None]*np.ones((1,len(k)))
    upp=wp[:,None]*np.ones((1,len(k)))
    diag=-(wp+wm)[:,None]-wt[:,None]*k[None,:]**2
    # Fit the known regular near-tip power rho^(n m) for low modes. This
    # removes an O(dr) error in n derivatives from a singular Frobenius point.
    ri=r[1:-1]
    for m in [1,3,5,7]:
        power=n*m
        exact=wr*(power*(power-1)/ri**2+2*power/(ri*np.tanh(2*ri))-power**2/np.sinh(ri)**2)
        discrete=wp*((ri+dr)/ri)**power+wm*((ri-dr)/ri)**power-(wp+wm)-wt*m*m
        diag[:,m]+=exact-discrete
    if modes is None:modes=nt//4 if smoothing=='hard' else nt//16
    bc=np.zeros(len(k),complex)
    odds=np.arange(1,nt//2,2)
    bc[odds]=nt*2/np.pi*(-1.)**((odds-1)//2)/odds
    if smoothing=='hard':bc[k>=modes]=0
    elif smoothing=='gaussian':bc*=np.exp(-(k/modes)**2)
    elif smoothing=='tanh':bc=np.fft.rfft(np.tanh(modes*np.cos(2*np.pi*np.arange(nt)/nt)))
    else:raise ValueError(smoothing)
    rhs=np.zeros(diag.shape,complex);rhs[-1]-=wp[-1]*bc
    fc=np.zeros((nr+1,len(k)),complex);fc[-1]=bc
    fc[1:-1]=thomas(low,diag,upp,rhs)
    def ir(v):return np.fft.irfft(v,n=nt,axis=1)
    def deriv(v):return ir(1j*k[None,:]*np.fft.rfft(v,axis=1))
    def radial_deriv(vc):
        dc=np.gradient(vc,dr,axis=0,edge_order=2)
        for m in [1,3,5,7]:
            power=n*m
            hh=np.zeros(nr+1,complex);hh[1:]=vc[1:,m]/r[1:]**power
            hh[0]=(4*hh[1]-hh[2])/3
            hhprime=np.gradient(hh,dr,edge_order=2)
            idx=(r>0)&(r<.12)
            dc[idx,m]=power*vc[idx,m]/r[idx]+r[idx]**power*hhprime[idx]
        dc[0]=dc[1]
        return ir(dc)
    f=ir(fc);fr=radial_deriv(fc)
    ft=ir(1j*k[None,:]*fc);ftt=ir(-k[None,:]**2*fc)
    sr=np.sinh(r)[:,None];cr=np.cosh(r)[:,None]
    weight=np.zeros_like(sr);weight[1:]=n*n/sr[1:]**2
    angular=weight*ft*ft
    # Limiting gradients at tip; for n>1 both vanish, n<1 integrable singular.
    angular[0]=angular[1]
    bp=np.zeros_like(f);hp=np.zeros_like(f)
    bb=sr*sr*(fr*fr+angular);hh=cr*cr*(fr*fr-angular)
    bb[0]=bb[1]*(2/(2*n+1)-1)
    hh[0]=hh[1]*(2/(2*n-1)-1)
    bp[1:]=-.5*cumulative(bb,dr)[1:]/sr[1:]**2
    hp=-.5*cumulative(hh,dr)/(cr*cr)
    b=cumulative(bp,dr);h=cumulative(hp,dr)
    w2=float(np.mean(b[-1]-h[-1]))
    source=2*weight*b*ftt-(bp+hp)*fr-weight*(deriv(h)-deriv(b))*ft
    sc=np.fft.rfft(source,axis=1)
    gc=np.zeros_like(fc)
    grhs=wr[:,None]*sc[1:-1]
    qcoeff=np.fft.rfft(b[-1]-h[-1]-w2)
    qcoeff[1:]/=1j*k[1:];qcoeff[0]=0
    qboundary=np.fft.irfft(qcoeff,n=nt)
    if physical_source:
        gc[-1]=np.fft.rfft(qboundary*ft[-1])
        grhs[-1]-=wp[-1]*gc[-1]
    gc[1:-1]=thomas(low,diag,upp,grhs)
    g=ir(gc);gr=radial_deriv(gc);gt=ir(1j*k[None,:]*gc)
    integ=np.exp(-r)[:,None]*(cr*hp*hp-sr*bp*bp)+np.exp(-2*r)[:,None]*fr*gr-weight*ft*gt+weight*b*ft*ft
    # The angular mean of Fr Gr - n² csch²(r) Ft Gt cancels at leading
    # r^(2n-2). Integrate the remaining r^(2n-1) first-cell mean exactly.
    integ[0]=np.mean(integ[1])*(1/n-1)
    w4=float(np.mean(cumulative(integ,dr)[-1]+.5*(b[-1]-h[-1])**2))
    # Integrated quadratic expression independent of b/h discretization.
    w2_check=float(.5*np.mean(cumulative(np.exp(-2*r)[:,None]*fr*fr-angular,dr)[-1]))
    coth=np.zeros_like(sr);coth[1:]=cr[1:]/sr[1:]
    cross=deriv(hp)+(np.tanh(r)[:,None]-coth)*deriv(h)+fr*ft
    rr=-coth*hp-np.tanh(r)[:,None]*bp-weight*deriv(deriv(h))+.5*(fr*fr-angular)
    dsource=sr*sr*(-bp*bp-fr*gr-weight*ft*gt+weight*b*ft*ft)
    ksource=cr*cr*(-hp*hp-fr*gr+weight*ft*gt-weight*b*ft*ft)
    dsource[0]=dsource[1]*(2/(2*n+1)-1)
    ksource[0]=ksource[1]*(2/(2*n-1)-1)
    dp=np.zeros_like(f);dp[1:]=cumulative(dsource,dr)[1:]/sr[1:]**2
    kp=cumulative(ksource,dr)/(cr*cr)
    dd=cumulative(dp,dr);kk=cumulative(kp,dr)
    cross4=deriv(kp)+(np.tanh(r)[:,None]-coth)*deriv(kk)+(hp-bp)*deriv(h)+fr*gt+gr*ft
    rr4=-coth*kp-np.tanh(r)[:,None]*dp-bp*hp-weight*(deriv(deriv(kk))+deriv(h)**2-deriv(b)*deriv(h)-2*b*deriv(deriv(h)))+fr*gr-weight*ft*gt+weight*b*ft*ft
    mask=(r>.2)&(r<4)
    return dict(n=n,nr=nr,nt=nt,rmax=rmax,physical_source=physical_source,smoothing=smoothing,modes=modes,w2=w2,w2_check=w2_check,w4=w4,w4_check=float(np.mean(dd[-1]-kk[-1]+.5*(b[-1]-h[-1])**2)),source_max=float(np.max(abs(source))),f_max=float(np.max(abs(f))),qboundary_max=float(np.max(abs(qboundary))),gboundary_max=float(np.max(abs(qboundary*ft[-1]))),constraint_cross=float(np.sqrt(np.mean(cross[mask]**2))),constraint_rr=float(np.sqrt(np.mean(rr[mask]**2))),constraint_cross4=float(np.sqrt(np.mean(cross4[mask]**2))),constraint_rr4=float(np.sqrt(np.mean(rr4[mask]**2))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--n',type=float,nargs='+',default=[1]);p.add_argument('--nr',type=int,default=1600);p.add_argument('--nt',type=int,default=1024);p.add_argument('--rmax',type=float,default=9);p.add_argument('--unfixed-source',action='store_true');p.add_argument('--smoothing',choices=['hard','gaussian','tanh'],default='hard');p.add_argument('--modes',type=int)
    args=p.parse_args()
    for n in args.n:
        t=time.time();out=solve(n,args.nr,args.nt,args.rmax,not args.unfixed_source,args.smoothing,args.modes);out['seconds']=time.time()-t;print(json.dumps(out),flush=True)
