"""Exact unbounded canonical Hamiltonian in a nondegenerate 4th-time-order test sector.

Mechanical witness L=Q2^2/2-m2*Q1^2/2-K*a^2/2, Q2=dot Q1,
a=dot Q2. Its nonzero acceleration Hessian -K is a structural
obstruction; NOT a field-theoretic proof of all stability/instability.
"""
import json
import sympy as s


def check():
    Q1,Q2,a,m2,K,p1,p2=s.symbols("Q1 Q2 a m2 K p1 p2",real=True)
    lag=Q2**2/2-m2*Q1**2/2-K*a*a/2
    momentum2=s.diff(lag,a)
    accel=-p2/K
    ham=s.expand(p1*Q2+p2*accel-lag.subs(a,accel))
    expected=p1*Q2-p2*p2/(2*K)-Q2*Q2/2+m2*Q1*Q1/2
    passed=s.simplify(momentum2+K*a)==0 and s.simplify(ham-expected)==0
    passed=passed and s.diff(lag,a,2)==-K and s.diff(ham,p1)==Q2
    return {"passed":bool(passed),"lagrangian":str(lag),"hamiltonian":str(ham),
            "acceleration_hessian":"-K; nondegenerate if K>0",
            "linear_canonical_momentum":"p1*Q2; arbitrarily signed at any fixed Q2 != 0",
            "scope":"nondegenerate homogeneous test; no ghost-free completion asserted"}


if __name__=="__main__":
    d=check()
    print(json.dumps(d,indent=2))
    if not d["passed"]: raise SystemExit(1)
