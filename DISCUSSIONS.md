# GitHub Discussions Guide

GitHub Discussions is the main forum for open technical conversation around Wave Confinement Theory, including derivations, numerical tests, formalization, empirical proposals, failed tests, and interpretation of results.

## Recommended categories

- **Announcements**: Releases, frozen research snapshots, canonical-status changes, and maintainer notices.
- **General**: Broad discussion about the research program and repository organization.
- **Q&A**: Focused questions tied to a specific equation, claim, file, test, figure, or result.
- **Ideas**: Testable research proposals with dependencies, deliverables, and explicit failure conditions.
- **Show and tell**: Derivations, proof attempts, reproductions, simulations, counterexamples, empirical tests, and research tools.
- **Polls**: Community priorities only. Polls are not scientific evidence.

## Required separation of evidence layers

Every substantive discussion should distinguish among:

1. **Mathematical status**: definition, assumption, conjecture, derivation, theorem, proof, counterexample, or unresolved step.
2. **Computational status**: executable, tested, reproduced, independently implemented, converged, unstable, failed, or incomplete.
3. **Empirical status**: untested, exploratory, preregistered, controlled, replicated, falsified, or independently confirmed.
4. **Interpretive status**: conventional interpretation, WCT-motivated hypothesis, speculative extension, or established consequence.

A passing test establishes only the claim encoded by that test. Software reproducibility does not by itself validate the underlying physics.

## Minimum standard for strong technical posts

Include an exact target claim or equation, commit SHA or release, assumptions, units, domain and boundary conditions, method, commands or proof files, machine-readable outputs, uncertainty or residuals, comparison baseline, unfavorable results, and a narrow statement of what the result does and does not establish.

Use **PASS**, **FAIL**, and **INCOMPLETE** only when the acceptance gate is explicit. Otherwise use more descriptive language such as *numerically supported*, *symbolically checked*, *counterexample found*, or *still speculative*.

When a discussion becomes a concrete implementation task, move the scoped work to an issue while keeping the scientific reasoning and interpretation in Discussions.
