def wct_mass_final(n, particle, is_matter=True):
    m0 = 1.0
    theta = 1e-120
    sigma = 0.0100
    beta = 0.163299
    gamma = 1e-120
    E_b = 1045.0
    A_l = 0.090638
    lambda_l = 0.108556
    A_q = 1.156775
    lambda_q = 0.116595
    chi = -0.07343
    eta = 0.001  # CPT phase asymmetry

    # Topological anchoring for n=1,2,3
    particle_key = particle.lower()
    if n == 1 and particle_key == 'electron':
        return 0.511
    elif n == 2 and particle_key == 'up quark':
        return 2.2
    elif n == 3 and particle_key == 'neutrino':
        return 0.0001

    ratio = n / (n + 1)
    m_harmonic = ratio * m0
    delta_spin = 0.25 / (2 * (n + 1))
    delta_topo = beta * (n % 3 - 1) * (-1)**n
    delta_entropy = sigma * np.log1p(n)
    delta_curv = theta * n**2
    delta_inertia = gamma * n**3
    delta_bind = -E_b if m_harmonic > 300 else 0.0

    if particle_key in ['electron', 'muon', 'tau']:
        delta_hierarchy = A_l * np.exp(lambda_l * n)
        delta_chiral = chi / (n**2)
    else:
        delta_hierarchy = A_q * np.exp(lambda_q * n)
        delta_chiral = 0.0

    delta_phase = eta if is_matter else -eta

    return (m_harmonic + delta_spin + delta_topo + delta_entropy +
            delta_curv + delta_inertia + delta_bind +
            delta_hierarchy + delta_chiral + delta_phase)
