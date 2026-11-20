"""Independent symbolic and numerical checks of the exact sharp trial.

Requires sympy and mpmath. No numerical optimization enters the proof.
"""
import json
import sympy as sp
import mpmath as mp

y,s,t=sp.symbols('y s t',positive=True)
F=t*t/(y*y+t*t)
AF=2*F-sp.diff((1+y*y)*sp.diff(F,y),y)
Q=2*t**3*(1-t*t)/(y*y+t*t)**2
assert sp.simplify(AF+sp.diff(Q,t))==0

# Residue-evaluated rational integral of F_t/(y^2+s^2)^2 over R.
inner=sp.pi*t*(t+2*s)/(2*s**3*(t+s)**2)
spatial=sp.simplify(2*s**3*(1-s*s)*sp.integrate(inner,(t,s,1)))
temporal=sp.pi*s*(1-s*s)/2
assert sp.simplify(spatial-sp.pi*(1-sp.Rational(3,2)*s+s**3/2))==0
assert sp.simplify(spatial+temporal-sp.pi*(1-s))==0

# Check the comparison-weight Euler equation on y>0, away from the cusp.
w=1-s-y*(sp.atan(1/y)-sp.atan(s/y))
Aw=2*w-sp.diff((1+y*y)*sp.diff(w,y),y)
assert sp.simplify(Aw-2*s**3*(1-s*s)/(y*y+s*s)**2)==0
wxx=(1-s*s)*sp.diff(w,s,2)-s*sp.diff(w,s)
Wx_over_W=(-2+3*s*s)/(s*sp.sqrt(1-s*s))
wx=sp.sqrt(1-s*s)*sp.diff(w,s)
assert sp.simplify(wxx+Wx_over_W*wx-Aw)==0

mp.mp.dps=45
rows=[]
for val in ['0.001','0.1','0.5','0.9','0.999']:
    a=mp.mpf(val)
    def wt(z):
        if z==0:return 1-a
        return 1-a-z*(mp.atan(1/z)-mp.atan(a/z))
    # Direct energy by w A w, separately from the closed polynomial.
    Esp=2*mp.quad(lambda z: wt(z)*2*a**3*(1-a*a)/(z*z+a*a)**2,[0,a,1,mp.inf])
    Et=2*mp.quad(lambda z: a**4*(1-a*a)/(z*z+a*a)**2,[0,a,1,mp.inf])
    error=(Esp+Et)/mp.pi-(1-a)
    assert abs(error)<mp.mpf('1e-38')
    rows.append({'s':val,'K':mp.nstr((Esp+Et)/mp.pi,20),'target':mp.nstr(1-a,20),'error':mp.nstr(error,6)})
print(json.dumps({'symbolic_checks':'PASS','direct_quadrature':rows},indent=2))
