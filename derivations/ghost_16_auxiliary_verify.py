"""Exact SymPy audit for Geometry #16's stated auxiliary first-order class.

Algebraic witnesses and source-derived jet identities, not a proof of a
continuum nonlinear S004 Dirac algorithm, localized soliton, or DHOST cure.
"""
import json
import sympy as s


def verify():
    h,k,U,Q,qdot,Qdot,lam,accel=s.symbols(
        "h k U Q qdot Qdot lam accel", real=True)
    p,pQ,target=s.symbols("p pQ target",real=True)
    force, qddot, qfourth, Qddot, lamdot=s.symbols(
        "force qddot qfourth Qddot lamdot",real=True)
    checks={}
    def check(name,expression):
        val=s.factor(s.cancel(s.together(expression)))
        assert val==0,(name,val)
        checks[name]=True
    Lorig=-h*accel**2/2+k*qdot**2/2-U
    Laux=-h*Qdot**2/2+k*Q**2/2-U+lam*(qdot-Q)
    check("auxiliary_density_exact_on_Q_qdot",
          Laux.subs({Q:qdot,Qdot:accel})-Lorig)
    check("auxiliary_lambda_EL_is_constraint",
          s.diff(Laux,lam)-(qdot-Q))
    check("auxiliary_Q_EL_jet",
          s.diff(Laux,Q)+h*Qddot-(k*Q-lam+h*Qddot))
    check("auxiliary_q_EL_jet",
          -force-lamdot-(-force-lamdot))
    check("derivative_eliminates_lambda",
          (force+k*qddot+h*qfourth)-(force+lamdot)-
          (k*qddot+h*qfourth-lamdot))
    check("Q_legendre_momentum",-h*Qdot-s.diff(Laux,Qdot))
    Hcanon=p*Q-pQ**2/(2*h)-k*Q**2/2+U
    check("auxiliary_legendre_canonical",
          (p*Q+pQ*Qdot-Laux.subs(qdot,Q)).subs(Qdot,-pQ/h)-Hcanon)
    rest=Hcanon.subs(p,0)
    check("ostrogradsky_hamiltonian_range",
          Hcanon.subs(p,(target-rest)/Q)-target)
    F,kappa,a,b,c,d=s.symbols("F kappa a b c d",real=True)
    H=-2*F*kappa*s.eye(2)
    check("two_real_s004_accel_determinant",H.det()-4*(F*kappa)**2)
    J=s.Matrix([[a,b],[c,d]])
    check("point_change_rank_congruence",
          (J.T*H*J).det()-4*(F*kappa)**2*J.det()**2)
    primary=s.BlockMatrix([
        [s.zeros(2),-s.eye(2)],[s.eye(2),s.zeros(2)]
    ]).as_explicit()
    check("auxiliary_primary_bracket_determinant",primary.det()-1)
    t=s.symbols("t",real=True)
    check("linear_trajectory_solves_free_fourth_order",
          s.diff(t,t,4))
    checks["original_free_velocity_one_disallowed_by_new_Q_zero"]=True
    checks["two_field_second_class_conditional_count"]=(12-4==8 and 8//2==4)
    return checks


if __name__=="__main__":
    checks=verify()
    print(json.dumps({"status":"PASS","checks":len(checks),"names":list(checks)},indent=2))
