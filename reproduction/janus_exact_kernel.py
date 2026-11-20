"""Independent numpy-only check of the proposed exact Janus capacity kernel.

After multiplying U by sqrt(cosh(v)), the v operator is the Schrodinger
operator -d_v²+1/4+sech²(v)/4. Diagonalize its finite-difference matrix,
then solve the tridiagonal mu equations directly.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import argparse,json,time

def solve_tri(diag,rhs,off):
    d=diag.copy();z=rhs.copy()
    for i in range(1,len(d)):
        q=off/d[i-1];d[i]-=q*off;z[i]-=q*z[i-1]
    ans=np.empty_like(z);ans[-1]=z[-1]/d[-1]
    for i in range(len(d)-2,-1,-1):ans[i]=(z[i]-off*ans[i+1])/d[i]
    return ans

class Kernel:
    def __init__(self,h=.025,mumax=10,vmax=10,quad=256):
        self.h=h;self.mumax=mumax;self.vmax=vmax
        self.mu=np.arange(-round(mumax/h)+1,round(mumax/h))*h
        self.v=np.arange(-round(vmax/h)+1,round(vmax/h))*h
        self.i0=len(self.mu)//2;self.j0=len(self.v)//2
        nv=len(self.v)
        hv=np.diag(2/h**2+.25+.25/np.cosh(self.v)**2)
        hv+=np.diag(np.full(nv-1,-1/h**2),1)+np.diag(np.full(nv-1,-1/h**2),-1)
        self.lam,self.vec=np.linalg.eigh(hv)
        self.weights=self.vec[self.j0]**2/h
        self.nodes,self.qweights=np.polynomial.legendre.leggauss(quad)
    def xi(self,mu,kappa):
        mu=np.abs(np.asarray(mu))
        if kappa==1:
            t=np.tanh(mu)
            return (1-t)**2*(t+2)/3
        upper=self.mumax+6
        x=(upper-mu[:,None])/2*self.nodes[None,:]+(upper+mu[:,None])/2
        f=(1+kappa*np.cosh(2*x))/2
        return (upper-mu)/2*np.sum(self.qweights[None,:]/f**2,axis=1)
    def solve(self,kappa):
        f=(1+kappa*np.cosh(2*self.mu))/2
        f0=(1+kappa)/2;gamma2=(1-kappa*kappa)/2
        xx=self.xi(self.mu,kappa)
        diag=2/self.h**2+4+self.lam[None,:]/f[:,None]
        u=solve_tri(diag,np.repeat(xx[:,None],len(self.lam),axis=1),-1/self.h**2)
        value=float(np.dot(self.weights,u[self.i0]))
        cap=f0+np.pi*gamma2*value
        # The throat comparison: same v spectrum, constant f=f0; integrate
        # its one-dimensional Green function analytically in mu except for
        # ordinary Gauss-Legendre quadrature of the smooth source.
        x=(self.nodes+1)*self.mumax/2
        xi=self.xi(x,kappa)
        kk=np.sqrt(4+self.lam/f0)
        throat_modes=self.mumax/2*np.sum(self.qweights[:,None]*xi[:,None]*np.exp(-x[:,None]*kk[None,:]),axis=0)/kk
        throat=float(np.dot(self.weights,throat_modes))
        delta=np.sqrt(2)*np.arctanh(np.sqrt(1-kappa*kappa)) if kappa<1 else 0
        clr=2*np.sqrt(gamma2)/delta if kappa<1 else 1
        out=dict(kappa=kappa,h=self.h,mumax=self.mumax,vmax=self.vmax,quadrature_order=len(self.nodes),U00=value,capacity=cap,entropy=np.sqrt(f0),transmission=clr,upper=f0/clr,throat_U=throat,lower_ratio=2*np.pi*np.sqrt(f0)*(1+np.sqrt(f0))*throat)
        if kappa==1:
            t=np.tanh(abs(self.mu))
            xi1=(1-t)**2*(t**3+2*t*t+3*t+4)/5
            a1=np.cosh(2*self.mu)/(2*np.cosh(self.mu)**4)
            u1=solve_tri(diag,xi1[:,None]-self.lam[None,:]*a1[:,None]*u,-1/self.h**2)
            U1=float(np.dot(self.weights,u1[self.i0]))
            out.update(pure_U_exact=1/(3*np.pi),pure_error=value-1/(3*np.pi),U_gamma2=U1,capacity_gamma4=np.pi*U1-.25)
        return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kappa',type=float,nargs='+',default=[1,.5,.2]);p.add_argument('--h',type=float,default=.025);p.add_argument('--mumax',type=float,default=10);p.add_argument('--vmax',type=float,default=10);p.add_argument('--out');args=p.parse_args()
    start=time.time();problem=Kernel(args.h,args.mumax,args.vmax);out=[]
    for kappa in args.kappa:
        r=problem.solve(kappa);out.append(r);print(json.dumps(r),flush=True)
    if args.out:
        with open(args.out,'w') as f:json.dump(out,f,indent=2)
    print(json.dumps(dict(seconds=time.time()-start)),flush=True)
