"""Independent exact algebra checks for foundational geometry issues #16/#18.

SymPy identities, not numerical dispersion evidence or a dynamical PDE proof.
"""
import json
import sympy as s


def verify():
    K,G,T,m,q,z=s.symbols("K G T m q z",real=True)
    B=4*K*T-G**2
    D=(z-q)-m-K*(z-q)**2+G*q*(z-q)-T*q**2
    disc=(1+G*q)**2-4*K*(m+T*q**2)
    checks={}
    def check(name,expr):
        remainder=s.simplify(s.factor(s.together(expr)))
        assert remainder==0,(name,remainder)
        checks[name]=True
    check("Q6_completed_square",4*K*D-(disc-(2*K*(z-q)-(1+G*q))**2))
    check("discriminant_B",disc-(1-4*K*m+2*G*q-B*q*q))
    check("static_stiffness",D.subs(z,0)+m+q+(K+G+T)*q*q)
    check("static_stiffness_identity",4*K*(K+G+T)-((2*K+G)**2+B))
    root=s.symbols("root",real=True)
    zlo=q+(1+G*q-root)/(2*K)
    zhi=q+(1+G*q+root)/(2*K)
    check("low_residue",s.diff(D,z).subs(z,zlo)-root)
    check("high_residue",s.diff(D,z).subs(z,zhi)+root)
    check("low_pole_exact",s.rem(s.together(4*K*D.subs(z,zlo)),root**2-disc,root))
    check("high_pole_exact",s.rem(s.together(4*K*D.subs(z,zhi)),root**2-disc,root))

    omega,Omega,u,F,Fp,Fpp,kappa,theta,gamma,Vp,Vpp=s.symbols(
        "omega Omega u F Fp Fpp kappa theta gamma Vp Vpp",real=True)
    X=z-q+Omega**2
    Q=s.Matrix([[X,2*s.I*Omega*omega],[-2*s.I*Omega*omega,X]])
    P=-q*s.eye(2)
    E=s.diag(1,0)
    J=kappa*Omega**2*Q+gamma*Omega**2*P/2
    M=-Q-Vp*s.eye(2)-2*u*Vpp*E+F*(kappa*Q**2+theta*P**2+gamma*(Q*P+P*Q)/2)+2*u*Fp*(E*J+J*E)+u*kappa*Omega**4*Fp*s.eye(2)+2*u**2*kappa*Omega**4*Fpp*E
    C=-2+4*F*kappa*X-2*F*gamma*q+4*u*Fp*kappa*Omega**2
    Cdiag=-X-Vp+F*(kappa*(X**2+4*Omega**2*z)+theta*q**2-gamma*q*X)+u*kappa*Omega**4*Fp
    Adiag=Cdiag-2*u*Vpp+4*u*Fp*(kappa*Omega**2*X-gamma*Omega**2*q/2)+2*u**2*kappa*Omega**4*Fpp
    expected=s.Matrix([[Adiag,s.I*Omega*omega*C],[-s.I*Omega*omega*C,Cdiag]])
    for i in range(2):
        for j in range(2):
            check(f"rotating_Q4_Q5_{i}{j}",(M[i,j]-expected[i,j]).subs(z,omega**2))
    check("rotating_Q5_det",s.det(expected)-(Adiag*Cdiag-Omega**2*omega**2*C**2))
    I=s.eye(2)
    PB=s.BlockMatrix([[s.zeros(2),-I],[I,s.zeros(2)]]).as_explicit()
    assert PB.det()==1
    checks["two_field_dirac_second_class"]=True
    a1,a2,W=s.symbols("a1 a2 W",real=True)
    Hess=s.hessian(W*kappa*(a1*a1+a2*a2),(a1,a2))
    check("two_field_acceleration_hessian",Hess.det()-4*(W*kappa)**2)
    a,b,c,d=s.symbols("a b c d",real=True)
    check("invertible_point_transform_hessian",
        (W*kappa)**2*((a*a+c*c)*(b*b+d*d)-(a*b+c*d)**2-(a*d-b*c)**2))
    mu,aa,bb=s.symbols("mu aa bb",real=True)
    freq=mu-aa*q+bb*q*q
    check("independent_rail_positive_square",
        4*bb*freq-((2*bb*q-aa)**2+4*bb*mu-aa**2))
    check("independent_rail_minimum",s.diff(freq,q).subs(q,aa/(2*bb)))
    check("independent_rail_convexity",s.diff(freq,q,2)-2*bb)
    # Exact full real-pole trichotomy and conditional F2 extremum algebra.
    A=2*K+G
    N=G**2+B*(1-4*K*m)
    check("full_real_pole_vertex",
          B*disc-(N-(B*q-G)**2))
    check("tangent_shift_identity",
          disc-(1-4*K*m+2*G*q-B*q**2))
    center=s.symbols("center", real=True)
    qstar=(G-A*center)/B
    candidate_residual=s.together(B*disc.subs(q,qstar)-B*center**2)
    check("exact_F2_minimizer_square_condition",
          s.rem(candidate_residual,center**2-N/(A**2+B),center))
    zlo_exact=q+(1+G*q-s.sqrt(disc))/(2*K)
    zhi_exact=q+(1+G*q+s.sqrt(disc))/(2*K)
    check("lower_branch_strict_convexity_numerator",
          2*K*disc**s.Rational(3,2)*s.diff(zlo_exact,q,2) -
          (B*(1-4*K*m)+G**2))
    check("upper_branch_strict_concavity_numerator",
          2*K*disc**s.Rational(3,2)*s.diff(zhi_exact,q,2) +
          (B*(1-4*K*m)+G**2))
    delta=s.symbols("delta",real=True)
    root_star=s.symbols("root_star",real=True)
    tangent=(root_star+A*(q-delta))**2-disc-(A**2+B)*(q-delta)**2
    condition=G-B*delta-A*root_star
    check("tangent_square_supporting_identity",
          s.rem(s.together(tangent+2*(q-delta)*condition),
                root_star**2-disc.subs(q,delta),root_star))

    # New admissibility witnesses: source S004 and independent spatial rail
    # are not globally the same static quadratic polynomial.
    static=lambda qq: m+qq+(K+G+T)*qq**2
    rail=lambda qq: mu-aa*qq+bb*qq**2
    check("parent_static_odd_coefficient",static(1)-static(-1)-2)
    check("alternative_static_odd_coefficient",rail(1)-rail(-1)+2*aa)
    check("parent_vs_spatial_odd_difference",
          (static(1)-static(-1))-(rail(1)-rail(-1))-2*(1+aa))
    check("phase_kernel_gapless",D.subs({m:0,z:0,q:0}))
    check("amplitude_phase_kernel_mismatch",
          D.subs({z:0,q:0})-D.subs({m:0,z:0,q:0})+m)
    # V''' and V'''' are independent data: cubic perturbations preserve
    # the full quadratic potential jet at the original background.
    tt,j0,j1,j2,lam=s.symbols("tt j0 j1 j2 lam",real=True)
    base=j0+j1*tt+j2*tt**2
    altered=base+lam*tt**3
    for order in [0,1,2]:
        check(f"cubic_shift_preserves_jet_{order}",
              s.diff(altered-base,tt,order).subs(tt,0))
    check("cubic_shift_third_derivative",s.diff(altered-base,tt,3)-6*lam)
    return checks


if __name__=="__main__":
    checks=verify()
    print(json.dumps({"status":"PASS","checks":len(checks),"names":list(checks)},indent=2))
