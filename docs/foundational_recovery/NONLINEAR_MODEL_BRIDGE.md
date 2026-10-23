# Nonlinear model bridge and mass dependency audit

Date: 2026-10-20. **MODEL_FAMILY_NOT_YET_UNIFIED; MASS_BRIDGE_OBSTRUCTED.**
Route A produces no accepted healthy parent. Route B is independently defined.
An exact statement within one model is not a derivation between models.

## C1. A finite Fourier window is not an invariant nonlinear sector

**Proposition.** On a flat torus, beta,gamma>0, no Fourier coordinate subspace
spanned by a finite set S of at least two modes is invariant under the
unprojected cubic–quintic Hamiltonian equation. Single plane-wave subspaces
are exceptions. This statement is about a coordinate truncation, not every
possible nonlinear invariant manifold or a controlled approximate envelope.

**Proof.** Choose a real linear functional on the finite frequency set with
unique maximizing k and minimizing l. Put u=a exp(i k.x)+b exp(i l.x), with
a,b nonzero. The quintic vertex u^3 bar u^2 has at frequency 3k-2l the unique
coefficient a^3 bar b^2. This frequency has strictly larger functional value
than every member of S. The cubic vertex has maximal value at 2k-l, smaller
than 3k-2l, and the linear operator is diagonal; neither can cancel it.
Thus the vector field outside S is -i gamma a^3 bar b^2!=0. Smooth local
evolution has that excluded component equal to this derivative times t+o(t).
The amplitude product has the expected fifth-order smallness, but is not zero.

In particular u=a e^(ikx)+b e^(-ikx) immediately generates 5k. A cosine
carrier also generates 3k through the cubic term. An initial frequency cutoff
is not a conserved constraint. This exact support calculation strengthens
the previous illustrative harmonic example; it does not assert that every
initial datum generates every mode, or prove S004's nonlinear norm inflation.

For S004 the unspecified potential and its derivative nonlinearities can
also generate out-of-window content. A proposed reduction must bound those
terms and the resulting unstable/high-pole response. Rejecting a coordinate
truncation does not rule out a physically justified nonlinear reduction with
slaved excluded modes and a finite-time error estimate.

## Explicit obligation map

| Object | S004 / candidate parent | Independent Hamiltonian | Current bridge |
|---|---|---|---|
| Fields | Complex scalar with four real regular configuration degrees; published scalar–tensor adds a metric | One complex first-order field, one canonical pair | OPEN; projection/preparation not specified |
| Time | tau=c0 t; preferred foliation in S004 | Declared Schrödinger time with quartic spatial coefficient 1 | OPEN physical clock and unit map |
| Domain | Fixed flat background specialization; boundary terms required | T_42^2 for D4, separate H^2(R^d) mathematical problem | OPEN physical boundary and domain-removal map |
| Symmetries | Global phase; no scalar local gauge; covariant candidate has actual diffeomorphisms | Global phase, translations, rotations compatible with domain, time reversal | Shared symmetries insufficient for derivation |
| Charge | Q=integral(p0.Jx+p1.Jv), profile/frequency dependent for standing waves | -M in the stated phase-generator convention | OPEN normalization and excluded-mode corrections |
| Energy | Ostrogradsky H, or indefinite Einstein-frame scalar kinetic energy | H=||Au||^2-beta||u||4^4/2+gamma||u||6^6/3 | OPEN; no full energy-preserving equivalence |
| Nonlinearities | V unspecified, nonlinear regulator and derivative vertices | Focusing cubic and defocusing quintic explicitly postulated in this model | OPEN coefficient/vertex derivation |
| Modes | Extra higher-time modes; static phase is gapless | Both real amplitude and phase evolve | OPEN preparation and mode elimination |
| Cutoff | No independently selected physical UV cutoff | Projection-free equation; C1 proves finite support generally leaks | OPEN physical justification and leakage bound |
| Error | Controlled one-real quadratic Taylor remainder exists | No nonlinear parent approximation theorem | OPEN norm, interval and excluded-mode error |

Changing higher derivatives of V while fixing V'(rho0), V''(rho0) leaves the
static quadratic poles unchanged but changes nonlinear vertices. Therefore a
quadratic dispersion match cannot determine beta,gamma. The massless phase
pole is not an identical copy of the gapped amplitude pole and cannot be
integrated out using an assumed positive gap. A derivation must carry it.

The existing one-real-component lower-pole reduction is retained as
**CONTROLLED_ASYMPTOTIC_REDUCTION at quadratic order** in its explicitly
positive-discriminant spectral window. It is not upgraded to a nonlinear EFT.
The exact S004-to-EL arrow and Hamiltonian-to-stationary-equation arrow survive.
The full S004-to-Hamiltonian arrow remains **OPEN**. The A1 scalar–tensor
candidate is itself ghostly, so it supplies no alternative healthy parent.

## C2. PFF and reversibility

At an equilibrium, an invertible differentiable same-time autonomous field
change conjugates linear generators by similarity. A Hamiltonian oscillator
has exponents in opposite imaginary pairs; a strictly dissipative relaxation
mode has a negative real exponent. They cannot be the same full local flow
under such a transformation. Global phase space dimensions are also different
for a real scalar relaxation equation and a complex Hamiltonian pair.

An independently supplied bath, radiation boundary or noninvertible
coarse-graining could produce an effective dissipative equation. Its physical
mechanism, fluctuation/energy accounting, limit and error must be specified.
The earlier damped one-real quadratic slow-mode calculation remains
**CONDITIONAL**. No damping term or growth band was inserted in this work.

## Physical mass and inertia: seven dependencies

For a specified product parent wave equation Box_(4+d)Psi=0, self-adjoint
nonnegative internal operator and normalizable separated mode, the conditional
dispersion gives m_n=(hbar/c)sqrt(lambda_n) after identifying E=hbar omega
and p=hbar k. This theorem and M1's equality criterion are preserved.

| Required physical input | Route A result | Route B result |
|---|---|---|
| Independently selected confinement operator | No healthy selected parent | A^2 and stationary Hessians are mathematically declared, not derived as a physical internal Laplacian |
| Physical domain/boundary | Unselected | Torus imposed; whole-space binding open |
| Length scale | Unselected | k_* and L fix mathematical scales in chosen units, not their physical calibration |
| Action/excitation normalization | Unfixed; scalar kinetic indefiniteness persists | Symplectic normalization declared; multiplying the whole action changes Noether energy without changing the PDE |
| Mode/particle assignment | Unselected | No particle correspondence derived |
| Finite conserved rest energy | Healthy localized parent absent | Finite H and M on H^2, plus finite periodic state energy; physical rest frame/normalization unestablished |
| Inertial response | Not derived | Translation momentum exists, but no physical energy–momentum/force law derived |

The Hamiltonian's preferred finite-k shell and a standing-wave frequency do
not provide a relativistic mass shell. No mass value was used to set a free
normalization. In B3, “mass” means the PDE conserved L2 norm, including the
historical D4 prescribed norm; it is not a particle rest mass. No electron,
muon, tau or other measured particle parameter enters this work.

R1–R3, A1 and C1 are mathematical deductions with stated boundaries. None is
evidence of a physical particle or an experimentally selected governing law.
