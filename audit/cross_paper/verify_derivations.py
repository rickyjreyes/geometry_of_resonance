import sympy as s
import json
from pathlib import Path

checks=[]
def zero(name,expr):
    result=s.factor(s.simplify(expr))
    assert result==0,(name,result)
    checks.append(name)

A,O,J,w,k,lam=s.symbols('A Omega J omega k lam',real=True)
kap,th,ga,f,f1,f2,V1,V2=s.symbols('kappa theta gamma f f1 f2 V1 V2',real=True)
r,rt,rx,rtt,rxx,v,vt,vx,vtt,vxx=s.symbols('r rt rx rtt rxx v vt vx vtt vxx',real=True)
u=A*A; d=O*O-J*J; p0=-J*J
du=2*A*lam*r+lam**2*(r*r+v*v)
qr=d*(A+lam*r)+lam*(-rtt+rxx-2*O*vt-2*J*vx)
qi=lam*(d*v-vtt+vxx+2*O*rt+2*J*rx)
pr=-J*J*(A+lam*r)+lam*(rxx-2*J*vx)
pi=lam*(-J*J*v+vxx+2*J*rx)
kin=-(lam*rt+O*lam*v)**2-(lam*vt-O*(A+lam*r))**2+(lam*rx-J*lam*v)**2+(lam*vx+J*(A+lam*r))**2
H=kap*(qr*qr+qi*qi)+th*(pr*pr+pi*pi)+ga*(qr*pr+qi*pi)
L=kin-V1*du-V2*du*du/2+(f+f1*du+f2*du*du/2)*H
L2=s.expand(L).coeff(lam,2)
jets=[[r,rt,rx,rtt,rxx],[v,vt,vx,vtt,vxx]]
fac=[1,-s.I*w,s.I*k,-w*w,-k*k]
M=s.Matrix(2,2,lambda i,j:s.expand(sum(s.conjugate(fac[a])*s.diff(L2,jets[i][a],jets[j][b])*fac[b] for a in range(5) for b in range(5))))
eye=s.eye(2);E=s.diag(1,0);q=k*k;z=w*w
Q=s.Matrix([[z-q+d,2*s.I*(O*w-J*k)],[-2*s.I*(O*w-J*k),z-q+d]])
P=s.Matrix([[-q-J*J,-2*s.I*J*k],[2*s.I*J*k,-q-J*J]])
h=kap*d*d+th*p0*p0+ga*d*p0
JJ=(kap*d+ga*p0/2)*Q+(th*p0+ga*d/2)*P
K=-Q-V1*eye-2*u*V2*E+f*(kap*Q*Q+th*P*P+ga*(Q*P+P*Q)/2)+2*u*f1*(E*JJ+JJ*E)+u*h*f1*eye+2*u*u*h*f2*E
for i in range(2):
 for j in range(2):zero(f'general_plane_background_Hessian_{i}{j}',M[i,j]-2*K[i,j])
zero('Hermitian_mixing',K[0,1]-s.conjugate(K[1,0]))
zero('background_phase_zero',K[1,1].subs({k:0,w:0,V1:-d+h*(f+u*f1)}))

KR=s.simplify(K.subs(J,0));D=z-q+O*O
B=-D-V1+f*(kap*(D*D+4*O*O*z)+th*q*q-ga*q*D)+u*kap*O**4*f1
RR=B-2*u*V2+4*u*f1*(kap*O*O*D-ga*O*O*q/2)+2*u*u*kap*O**4*f2
C=-2+4*f*kap*D-2*f*ga*q+4*u*f1*kap*O*O
zero('rotating_phase_entry',KR[1,1]-B)
zero('rotating_amplitude_entry',KR[0,0]-RR)
zero('rotating_off_diagonal',KR[0,1]-s.I*O*w*C)
zero('rotating_determinant',KR.det()-RR*B+O*O*z*C*C)

aa,gg,tt,mm,qq,zz,ss=s.symbols('K G T mu2 q z s',positive=True)
disc=1-4*aa*mm+2*gg*qq-(4*aa*tt-gg**2)*qq**2
low=(1+(2*aa+gg)*qq-s.sqrt(disc))/(2*aa)
high=(1+(2*aa+gg)*qq+s.sqrt(disc))/(2*aa)
poly=(zz-qq)-mm-aa*(zz-qq)**2+gg*qq*(zz-qq)-tt*qq*qq
zero('all_static_poles',poly+aa*(zz-low)*(zz-high))
zero('lower_positive_residue',s.diff(poly,zz).subs(zz,low)-s.sqrt(disc))
zero('upper_negative_residue',s.diff(poly,zz).subs(zz,high)+s.sqrt(disc))
c0=(1-s.sqrt(1-4*aa*mm))/(2*aa)
c2=1-gg*c0/s.sqrt(1-4*aa*mm)
c4=tt/s.sqrt(1-4*aa*mm)+gg*gg*mm/(1-4*aa*mm)**s.Rational(3,2)
zero('C0',low.subs(qq,0)-c0)
zero('C2',s.diff(low,qq).subs(qq,0)-c2)
zero('C4',s.diff(low,qq,2).subs(qq,0)/2-c4)
bgap=4*aa*tt-gg*gg
zero('convexity_identity',s.diff(low,qq,2)-(bgap/s.sqrt(disc)+(gg-bgap*qq)**2/disc**s.Rational(3,2))/(2*aa))
qmin=(gg-(2*aa+gg)*s.sqrt((gg*gg+bgap*(1-4*aa*mm))/((2*aa+gg)**2+bgap)))/bgap
# Squared stationarity plus sign condition in written proof avoids unsafe square-root simplification.
zero('exact_minimum_stationarity',((gg-bgap*qq)**2-(2*aa+gg)**2*disc).subs(qq,qmin))
delta,g,hc,M2,Qsmall=s.symbols('delta g h M2 Q',positive=True)
scaled=low.subs({aa:delta/M2,gg:g/M2,tt:hc/(delta*M2),mm:M2,qq:delta*M2*Qsmall})/M2
zero('controlled_scaled_kernel',s.series(scaled,delta,0,2).removeO()-1-delta*(1+(1-g)*Qsmall+hc*Qsmall**2))
zero('phase_low_kernel',c4.subs(mm,0)-tt)
Gam=s.symbols('Gamma',positive=True)
rP,aP,bP=-Gam*c0,-Gam*c2,Gam*c4
zero('PFF_EFT_center_match',aP/(2*bP)+c2/(2*c4))
zero('complete_square',c0+c2*qq+c4*qq**2-(c4*(qq+c2/(2*c4))**2+c0-c2*c2/(4*c4)))

# Further independent analytic checks for downstream compositions.
zero('phase_C0',c0.subs(mm,0))
zero('phase_C2',c2.subs(mm,0)-1)
NN=bgap*(1-4*aa*mm)+gg*gg
zero('Taylor_third_derivative',s.diff(low,qq,3)+3*NN*(gg-bgap*qq)/(2*aa*disc**s.Rational(5,2)))
V3,V4=s.symbols('V3 V4',real=True)
du1=2*A*r;du2=r*r+v*v;H2=kap*(rtt-rxx)**2+th*rxx**2-ga*(rtt-rxx)*rxx
potential=-V1*(lam*du1+lam**2*du2)-V2*(lam*du1+lam**2*du2)**2/2-V3*(lam*du1+lam**2*du2)**3/6-V4*(lam*du1+lam**2*du2)**4/24
nonlin=potential+(f+f1*(lam*du1+lam**2*du2)+f2*(lam*du1+lam**2*du2)**2/2)*lam**2*H2
zero('cubic_vertices',s.expand(nonlin).coeff(lam,3)-(-V2*du1*du2-V3*du1**3/6+f1*du1*H2))
zero('quartic_vertices',s.expand(nonlin).coeff(lam,4)-(-V2*du2**2/2-V3*du1**2*du2/2-V4*du1**4/24+(f1*du2+f2*du1**2/2)*H2))
x=s.symbols('x',positive=True)
rr=s.Function('r')(x);aa4=s.Function('a4')(x);bb1=s.Function('B1')(x);bb2=s.Function('B2')(x)
density=aa4*s.diff(rr,x,2)**2+s.diff(rr,x)**2+bb1*rr*s.diff(rr,x,2)+bb2*rr**2
strong=(s.diff(density,rr)-s.diff(s.diff(density,s.diff(rr,x)),x)+s.diff(s.diff(density,s.diff(rr,x,2)),x,2))/2
expected=s.diff(aa4*s.diff(rr,x,2),x,2)-s.diff((1-bb1)*s.diff(rr,x),x)+(bb2+s.diff(bb1,x,2)/2)*rr
zero('variable_coefficient_static_Hessian',strong-expected)
ww=s.Function('w')(x);VV=s.Function('V')(x);ang=s.symbols('m_ang',real=True)
radf=ww/s.sqrt(x)
zero('radial_Liouville_transform',s.sqrt(x)*(-s.diff(radf,x,2)-s.diff(radf,x)/x+(ang**2/x**2+VV)*radf)-(-s.diff(ww,x,2)+((ang**2-s.Rational(1,4))/x**2+VV)*ww))
Op=lambda v:-s.diff(v,x,2)+VV*v
zero('second_order_square_matching',Op(Op(rr))-(s.diff(rr,x,4)-s.diff(2*VV*s.diff(rr,x),x)+(VV**2-s.diff(VV,x,2))*rr))
RA,RB,RC,RRR=s.symbols('A_radius B_radius C_radius R',positive=True)
Erad=RA/RRR**4-RB/RRR**2+RC/RRR**3
Rstar=(3*RC+s.sqrt(9*RC**2+32*RA*RB))/(4*RB)
zero('closure_orbit_stationarity',s.diff(Erad,RRR).subs(RRR,Rstar))
zero('closure_orbit_second_derivative',(s.diff(Erad,RRR,2)-(4*RB*RRR-3*RC)/RRR**5).subs(RA,(2*RB*RRR**2-3*RC*RRR)/4))
pp,dd,tau=s.symbols('p d tau',positive=True)
Eent=RA/RRR**2+RB/RRR**pp+tau*dd*s.log(RRR)
zero('entropy_radius_second_derivative',(s.diff(Eent,RRR,2)-4*RA/RRR**4-pp**2*RB/RRR**(pp+2)).subs(tau*dd,2*RA/RRR**2+pp*RB/RRR**pp))
eta,cc,sp,eps=s.symbols('eta c0 spatial eps',positive=True)
slow=(-eta+s.sqrt(eta**2-4*eps*cc**2*sp))/2
zero('overdamped_eigenvalue',s.series(slow,eps,0,2).removeO()+eps*cc**2*sp/eta)
areg,kreg,dreg=s.symbols('a4_reg kappa_reg delta_reg',positive=True)
symbol=(areg+kreg/dreg**2)*qq**2-RB*qq
zero('closure_vanishing_threshold',symbol.subs(qq,RB/(2*(areg+kreg/dreg**2)))+RB**2/(4*(areg+kreg/dreg**2)))

witness={aa:s.Rational(1,1000),gg:2,tt:2000,mm:1}
qm=s.N(qmin.subs(witness),30)
qm_approx=s.N((-c2/(2*c4)).subs(witness),30)
assert qm>0 and s.N(c2.subs(witness))<0
num=dict(K='1/1000',G='2',T='2000',mu2='1',q_min_exact=str(qm),q_min_quartic=str(qm_approx),
         omega2_min=str(s.N(low.subs(witness).subs(qq,qm),30)),
         ghost_omega2_at_min=str(s.N(high.subs(witness).subs(qq,qm),30)),
         relative_q_error=str(s.N(abs(qm-qm_approx)/qm,20)))
out=dict(checks_passed=len(checks),checks=checks,witness=num,
         C0=str(c0),C2=str(c2),C4=str(c4),exact_q_min=str(qmin),
         PDE_runs=0,empirical_frequency_inputs=[])
Path(__file__).with_name('symbolic_verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
