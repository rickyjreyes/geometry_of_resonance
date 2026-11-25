"""Exact algebra and counterexamples for the physical-domain audit.

No empirical inputs, field arrays, optimization scans, or PDE time integration.
Requires Python 3.12 and SymPy 1.14.0 (mpmath 1.3.0).
"""
from pathlib import Path
import json
import platform
import sympy as sp


def verify():
    checks = []

    def zero(identifier, expression, scope):
        residual = sp.simplify(sp.expand_complex(expression))
        assert residual == 0, (identifier, residual)
        checks.append(dict(id=identifier, passed=True, residual=str(residual), scope=scope))

    x, t = sp.symbols('x t', real=True)
    a, eps, rho, beta, gamma, radius, length = sp.symbols(
        'alpha epsilon rho beta gamma R L', positive=True)
    # Exact polar identity on a node-free chart, not a phase definition for Theta.
    amp = sp.Function('A', real=True, positive=True)(x)
    phase = sp.Function('phi', real=True)(x)
    field = amp * sp.exp(sp.I * phase)
    polar = (sp.diff(amp, x, 2)/amp - sp.diff(phase, x)**2
             + sp.I*(2*sp.diff(amp, x)*sp.diff(phase, x)/amp + sp.diff(phase, x, 2)))
    zero('V01_POLAR_LAPLACIAN', sp.diff(field, x, 2)/field-polar,
         'Exact in one dimension; componentwise sum gives the flat multidimensional identity, A>0.')

    den = rho + eps**2*sp.exp(-2*a*rho)
    weight = rho/den**2
    expected = (eps**2*sp.exp(-2*a*rho)-rho+4*a*rho*eps**2*sp.exp(-2*a*rho))/den**3
    zero('V02_CORRECTED_WEIGHT_DERIVATIVE', sp.diff(weight, rho)-expected,
         'Corrected complex-safe action, not the legacy additive quotient.')
    z = sp.symbols('z', real=True)
    hd = weight.subs(rho, z**2*rho)*z**2
    zero('V03_NO_QUADRATIC_CURVATURE_AT_ZERO', sp.diff(hd, z, 2).subs(z, 0),
         'Curvature energy F(|psi|^2)|D2 psi|^2 begins at fourth amplitude order at the zero background.')
    zero('V04_CORRECTED_QUARTIC_LEADING_TERM', sp.limit(hd/z**4, z, 0)-rho/eps**4,
         'Amplitude expansion only; no nonlinear well-posedness conclusion.')
    y = sp.symbols('y', positive=True)
    zero('V05_LEGACY_ROOT_MONOTONICITY',
         sp.diff(y*sp.exp(a*y*y), y)-sp.exp(a*y*y)*(1+2*a*y*y),
         'Positive derivative plus endpoint limits proves a unique negative-real denominator zero.')

    r = sp.symbols('r', real=True)
    rate = 2*rho*(r-beta*rho)
    assert rate.subs({rho:1,r:2,beta:1}) == 2
    checks.append(dict(id='V06_PFF_PHYSICAL_CLOCK_OBSTRUCTION', passed=True,
                       witness='uniform rho=1, r=2, beta=1: rho_t=2, div(S)=0',
                       scope='If u=|A|^2 and the gradient rail uses the same closed-system physical time, PFF continuity fails for this datum.'))
    energy = -r*rho+beta*rho*rho/2
    zero('V07_COMPLEX_GRADIENT_ENERGY_FACTOR', sp.diff(energy,rho)*rate+2*rho*(r-beta*rho)**2,
         'Standard Wirtinger convention gives dE/dt=-2||A_t||^2; the sign survives an energy/time normalization.')
    aa, bb, k = sp.symbols('a b k', positive=True)
    symbol = r+aa*k*k-bb*k**4
    zero('V08_FINITE_BAND_MAXIMUM', sp.diff(symbol,k).subs(k,sp.sqrt(aa/(2*bb))),
         'A spatial wavenumber selected by independently supplied dimensional coefficients, not a log-observable domain.')
    zero('V09_FINITE_BAND_CURVATURE', sp.diff(symbol,k,2).subs(k,sp.sqrt(aa/(2*bb)))+4*aa,
         'The selected shell has negative symbol curvature when a>0.')

    q, k0, amplitude = sp.symbols('q k0 amplitude', real=True)
    omega = (k0*k0-q*q)**2-beta*amplitude**2+gamma*amplitude**4
    plane = amplitude*sp.exp(sp.I*(q*x-omega*t))
    rhs = sp.diff(plane,x,4)+2*k0*k0*sp.diff(plane,x,2)+k0**4*plane \
          -beta*amplitude**2*plane+gamma*amplitude**4*plane
    zero('V10_HAMILTONIAN_PLANE_WAVE_FAMILY', sp.I*sp.diff(plane,t)-rhs,
         'Exact finite-periodic-domain solutions for q=2pi*n/L; does not assert localization or stability.')
    n = sp.symbols('n', integer=True)
    zero('V11_PLANE_WAVE_WINDING', (2*sp.pi*n/length)*length/(2*sp.pi)-n,
         'Every supplied integer sector exists in this exact family; no integer is selected by existence alone.')

    J, Iw, hol, transverse = sp.symbols('J Iw hol m', real=True)
    mismatch = 2*sp.pi*n-transverse*hol-J
    C = mismatch/Iw
    zero('V12_LOCKING_CLOSURE', J+C*Iw+transverse*hol-2*sp.pi*n,
         'Iw=integral 1/w >0; loop geometry and winding supplied.')
    zero('V13_LOCKING_MINIMUM_COST', C*C*Iw-mismatch*mismatch/Iw,
         'Cauchy-Schwarz supplies the minimization proof; this check verifies its exact value.')
    zz = sp.symbols('zeta', real=True)
    cost = (2*sp.pi)**2*(n-zz)**2/Iw
    zero('V14_ADJACENT_SECTOR_COST', cost.subs(n,n+1)-cost-(2*sp.pi)**2*(2*n+1-2*zz)/Iw,
         'With Iw>0, minimizers are nearest integers to zeta; dynamical crossing of sectors is an extra premise.')
    zero('V15_CIRCLE_LOCK_ALL_RADII', (1/radius)*(2*sp.pi*radius)-2*sp.pi,
         'Planar circle, sigma=1/R, n=1: perfect locking for every R>0. Radius not selected.')
    zero('V16_CIRCLE_COST', (2*sp.pi*(n-1))**2/(2*sp.pi*radius)-2*sp.pi*(n-1)**2/radius,
         'Uniform unit weight; sector minimization selects n=1 but leaves radius continuously free.')

    L1,L2,n1,n2,c,p = sp.symbols('L1 L2 n1 n2 c p', real=True, nonzero=True)
    k1=2*sp.pi*n1/L1
    zero('V17_WINDING_WIDTH_TRANSFER', k1*L1/L2*n2/n1-2*sp.pi*n2/L2,
         'L1,L2>0 and n1 !=0; zero source winding must use the direct phase-increment formula.')
    zero('V18_COORDINATE_PRODUCT', (k/p)*(p*length)-k*length,
         'Oriented widths for p!=0; ordinary positive width requires p>0 or endpoint reordering.')
    zero('V19_TRANSLATION_WIDTH', ((length+c)-c)-length,
         'Common endpoint translation; sliding a crop on a fixed nonlinear phase is a different operation.')
    s = sp.symbols('s', positive=True)
    zero('V20_ROOT_MASS_VS_MASS', (sp.sqrt(6)*sp.sqrt(s))**2-6*s,
         'A hypothetical root-mass dilation sqrt(6) implies mass dilation 6, not sqrt(6).')

    # Counterexample: same exact endpoint closure, biased fitted cosine frequency.
    theta=2*sp.pi*x
    h=x*(1-x)*(x-sp.Rational(1,2))
    nuisance=sp.Matrix([sp.cos(theta),sp.sin(theta)])
    score=-x*sp.sin(theta)
    inner=lambda f,g: sp.integrate(sp.expand_trig(f*g),(x,0,1))
    gram=sp.Matrix(2,2,lambda i,j:inner(nuisance[i],nuisance[j]))
    proj=sp.Matrix([inner(nuisance[i],score) for i in range(2)])
    residual_score=sp.simplify(score-(nuisance.T*gram.inv()*proj)[0])
    bias=sp.simplify(inner(residual_score,-h*sp.sin(theta))/inner(residual_score,residual_score))
    bias_expected=(16*sp.pi**4+60*sp.pi**2-225)/(160*sp.pi**4-360*sp.pi**2)
    zero('V21_FIT_BIAS_COEFFICIENT', bias-bias_expected,
         'First derivative of local nonlinear least-squares frequency at epsilon=0 with free amplitude and phase, known zero background.')
    zero('V22_EXACT_MEAN_PHASE_CLOSURE', h.subs(x,1)-h.subs(x,0),
         'phi=2pi*x+epsilon*h(x) has mean phase gradient exactly 2pi for every epsilon.')
    assert float(bias)>0
    checks.append(dict(id='V23_NONZERO_FIT_BIAS',passed=True,exact=str(bias),decimal=str(sp.N(bias,17)),
                       scope='Therefore k_fit=2pi+0.1600447295...*epsilon+O(epsilon^2), despite exact unchanged winding.'))

    phi=sp.symbols('phi',real=True)
    cosines=sp.Matrix([sp.cos(phi+2*sp.pi*j/3) for j in range(3)])
    zero('V24_Z3_ZERO_SUM',sum(cosines),'Exact normalized finite-dimensional geometry.')
    zero('V25_Z3_NORM',sp.trigsimp((cosines.T*cosines)[0])-sp.Rational(3,2),
         'Cosine-vector norm sqrt(3/2), not a physical dilation eigenvalue.')
    ones=sp.ones(3,1)
    lift=ones+sp.sqrt(2)*cosines
    zero('V26_RECTANGULAR_LIFT_GAIN',sp.trigsimp((lift.T*lift)[0])-6,
         'T:R->R^3 has singular value sqrt(6); it is not a square physical evolution operator.')
    cartan=sp.Matrix([[2,-1],[-1,2]])
    assert cartan.eigenvals()=={1:1,3:1}
    checks.append(dict(id='V27_A2_CARTAN_EIGENVALUES',passed=True,eigenvalues=[1,3],
                       scope='Exact matrix eigenvalues do not establish observable scaling dynamics.'))
    eta=sp.symbols('eta',positive=True)
    willmore=sp.pi**2/(eta*sp.sqrt(1-eta**2))
    zero('V28_WILLMORE_MINIMUM',sp.diff(willmore,eta).subs(eta,1/sp.sqrt(2)),
         'Exact stationary shape ratio in the torus-of-revolution Willmore problem.')
    zero('V29_WILLMORE_VALUE',willmore.subs(eta,1/sp.sqrt(2))-2*sp.pi**2,
         'The positive divergent endpoints make this stationary point the unique ratio minimum.')
    rr, minor=sp.symbols('major minor',positive=True)
    zero('V30_WILLMORE_SCALE_DEGENERACY',(s*minor)/(s*rr)-minor/rr,
         'Homothety leaves eta and Willmore energy unchanged; size not selected.')
    return dict(schema='WCT_PHYSICAL_DOMAIN_SYMBOLIC_CHECKS_V1',python=platform.python_version(),
                sympy=sp.__version__,checks_run=len(checks),checks_passed=len(checks),checks=checks,
                empirical_frequency_inputs=[],new_PDE_simulations=0,collider_evaluations=0,
                fit_counterexample=dict(domain=[0,1],phase='2*pi*x+epsilon*x*(1-x)*(x-1/2)',
                                        mean_phase_gradient='2*pi',fit_bias_exact=str(bias),
                                        fit_bias_decimal=str(sp.N(bias,17))),
                proof_scope='Symbolic checks support the written proofs and explicit counterexamples; not full PDE existence/stability or a Lean kernel proof.')


if __name__=='__main__':
    result=verify()
    destination=Path(__file__).with_name('symbolic_results.json')
    destination.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('checks_run','checks_passed','empirical_frequency_inputs','new_PDE_simulations')}))
