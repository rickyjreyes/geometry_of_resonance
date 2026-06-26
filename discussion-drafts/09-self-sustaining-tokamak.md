# A Tokamak Control Loop Powered by a Bounded Fraction of Alpha Heating

**Author:** Richard J. Reyes  
**Published:** April 14, 2026  
**DOI:** https://doi.org/10.5281/zenodo.19578185  
**Category:** Fusion and Control

## Main result

I designed a burning-plasma control architecture in which a bounded fraction of alpha heating powers diagnostics, computation, and electromagnetic stabilization after handoff.

The plasma remains fuel-driven, but the control system no longer requires continuous external wall power after the defined confinement, stability, and power-balance conditions are reached.

The design includes turbulence and shear proxies, actuator limits, an explicit control-power budget, a handoff gate, and a latched self-sustaining state.

## What this adds to the corpus

This paper applies resonance and bounded-feedback ideas to a concrete control problem. It connects WCT-inspired stabilization with conventional plasma diagnostics, power accounting, actuator constraints, and tokamak operating windows.

## Direct tests

- Implement the control law in a validated burning-plasma simulation.
- Include actuator delay, saturation, sensor noise, and disruption precursors.
- Verify that the handoff margin remains positive under parameter uncertainty.
- Test helium ash, impurity accumulation, current-profile evolution, and edge stability.
- Require sustained zero external control power for a defined number of confinement times.

## Citation

R. J. Reyes, “Nuclear Fusion Tokamak with Self Sustaining Resonance,” Zenodo, Apr. 14, 2026. doi: 10.5281/zenodo.19578185.

## Corpus

Publications: https://rickyjreyes.github.io/publications/  
Research repository: https://github.com/rickyjreyes/geometry_of_resonance
