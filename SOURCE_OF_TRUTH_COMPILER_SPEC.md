# Build a Verified WCT Source-of-Truth Compiler and Obsidian Synchronization Pipeline

Update the existing repositories:

```text
rickyjreyes/geometry_of_resonance
rickyjreyes/wct-sympy
rickyjreyes/wct-lean
rickyjreyes/obsidian
```

Do not create another Obsidian plugin.

Do not redesign the existing WCT Graph Engine UI. The user likes the present graph, tabs, inspector, hover behavior, equation rendering, timeline, and navigation.

Do not work on priority scoring in this task.

The problem to solve is source integrity:

> Every definition, concept, equation, derivation, Lean status, SymPy status, prediction, experiment, paper relationship, and reference displayed in Obsidian must be traceable to the current GitHub repositories, the actual papers, and their actual references.

Obsidian must become a verified projection of the research corpus. It must not invent, infer, or silently preserve stale research state.

---

# 1. Canonical Architecture

Use this data flow:

```text
GitHub repositories
+ canonical paper PDFs
+ canonical bibliography
        ↓
verified source manifest
        ↓
source checksums and repository commit lock
        ↓
research compiler
        ↓
canonical research registry
        ↓
human-readable source-of-truth Markdown
        ↓
verified Obsidian synchronization
        ↓
existing WCT Graph Engine UI
```

Obsidian is not the canonical source.

The source of truth must live in:

```text
rickyjreyes/geometry_of_resonance
```

The other repositories contribute specific validation information, but they do not replace the canonical research content.

---

# 2. Create the Canonical Source Files

Create these files in `geometry_of_resonance`:

```text
WCT_RESEARCH_SOURCE_OF_TRUTH.md
registry/WCT_SOURCE_MANIFEST.yaml
registry/wct-research-registry.json
registry/wct-source-lock.json
registry/WCT_REFERENCES.bib
registry/WCT_BUILD_REPORT.md
```

## 2.1 `WCT_RESEARCH_SOURCE_OF_TRUTH.md`

This is the human-readable, auditable current state of the corpus.

It must be generated from the same compiled registry used by Obsidian.

It must not be separately handwritten.

It must include:

```text
Snapshot ID
Generated timestamp
Compiler version
geometry_of_resonance commit SHA
wct-sympy commit SHA
wct-lean commit SHA
paper count
paper PDF hashes
reference count
definition count
concept count
equation count
derivation count
prediction count
experiment count
symbolic-status totals
formal-status totals
source conflicts
unresolved review candidates
```

The document must then contain indexed sections for:

```text
Papers
Definitions
Concepts
Symbols
Equations
Derivations
Claims
Theorems
Predictions
Experiments
References
Repositories
Validation conflicts
```

Every object must show its provenance.

## 2.2 `WCT_SOURCE_MANIFEST.yaml`

This is the only manually maintained source inventory.

It must identify exactly which files are authoritative and what role each source has.

Example structure:

```yaml
schema_version: 1.0.0

repositories:
  geometry_of_resonance:
    url: https://github.com/rickyjreyes/geometry_of_resonance
    branch: main
    role: canonical-research-content

  wct-sympy:
    url: https://github.com/rickyjreyes/wct-sympy
    branch: main
    role: symbolic-validation

  wct-lean:
    url: https://github.com/rickyjreyes/wct-lean
    branch: main
    role: formal-validation

canonical_files:
  equations:
    - WCT_FULL_EQUATION_LIST_CORRECTED.md
    - WCT_MASTER_EQUATIONS_UPDATED.md

  papers:
    directory: Papers/

  references:
    - registry/WCT_REFERENCES.bib

  symbolic_validation:
    - repository: wct-sympy
      path: compiled-registry.json

  formal_validation:
    - repository: wct-lean
      path: WCTLean/Main.lean
```

Every paper must have a manifest entry:

```yaml
papers:
  - paper_id: PAPER-GEOMETRY-OF-RESONANCE
    title: The Geometry of Resonance
    repository_path: Papers/...
    zenodo_doi: ...
    sha256: ...
    status: canonical
```

Do not discover authoritative papers from arbitrary local Obsidian `pdf_url` fields.

## 2.3 `wct-source-lock.json`

Record the exact source snapshot used for a successful build:

```json
{
  "snapshot_id": "...",
  "generated_at": "...",
  "repositories": {
    "geometry_of_resonance": {
      "branch": "main",
      "commit": "..."
    },
    "wct-sympy": {
      "branch": "main",
      "commit": "..."
    },
    "wct-lean": {
      "branch": "main",
      "commit": "..."
    }
  },
  "files": {
    "WCT_FULL_EQUATION_LIST_CORRECTED.md": {
      "sha256": "..."
    }
  },
  "papers": {
    "PAPER-GEOMETRY-OF-RESONANCE": {
      "sha256": "..."
    }
  }
}
```

The Markdown, JSON registry, lock file, and build report must all contain the same `snapshot_id`.

---

# 3. Source Precedence

Apply explicit source roles.

## 3.1 Canonical scientific content

Use `geometry_of_resonance` for:

* equation identity;
* canonical equation name;
* canonical LaTeX;
* symbols;
* definitions;
* assumptions;
* paper identity;
* paper ordering;
* claims;
* predictions;
* experiment descriptions;
* protocols;
* repository relationships;
* current research narrative.

## 3.2 Symbolic state

Use `wct-sympy/compiled-registry.json` for:

* symbolic status;
* baseline symbolic status;
* effective symbolic status;
* verification kind;
* checker;
* assumptions used by the checker;
* symbolic limitations;
* symbolic counterexamples;
* symbolic source file.

It must not supply:

* formal-proof status;
* physical validity;
* experimental validation;
* canonical paper definitions unless explicitly sourced back to a paper or canonical equation file.

## 3.3 Formal state

Use `wct-lean` for:

* Lean declaration names;
* compiled declarations;
* formal status;
* formal assumptions;
* formal limitations;
* formal source path;
* formal repository commit.

It must not supply:

* symbolic status;
* empirical status;
* physical validity.

## 3.4 Papers

Use the exact PDFs listed in `WCT_SOURCE_MANIFEST.yaml`.

Prefer PDFs committed under:

```text
geometry_of_resonance/Papers/
```

Zenodo may be used as the immutable archival source only when:

* the DOI or record is listed in the manifest;
* the downloaded PDF hash matches the recorded hash;
* the paper identity matches the manifest.

Do not trust arbitrary URLs discovered in Obsidian notes.

## 3.5 References

Use:

```text
registry/WCT_REFERENCES.bib
```

as the canonical normalized bibliography.

The compiler must also inspect each paper’s references section and report:

* references present in the paper but missing from the BibTeX registry;
* BibTeX entries not linked to any paper;
* duplicate DOI entries;
* duplicate title entries;
* malformed citation keys;
* references that could not be resolved.

Missing references become review candidates. They must not silently become canonical references.

---

# 4. Build One Provenance-Bearing Research Registry

Create:

```text
registry/wct-research-registry.json
```

This must contain the complete current state used by Obsidian.

Every object must have:

```json
{
  "stable_id": "...",
  "object_type": "...",
  "title": "...",
  "aliases": [],
  "current_state": "...",
  "source_assertions": []
}
```

Every populated scientific field must have source assertions.

Example:

```json
{
  "field": "canonical_definition",
  "value": "...",
  "source": {
    "repository": "rickyjreyes/geometry_of_resonance",
    "path": "Papers/...",
    "commit": "...",
    "pages": "12-13",
    "sha256": "...",
    "extraction_method": "pdf-text",
    "human_verified": true
  }
}
```

For Markdown sources, use:

```json
{
  "repository": "...",
  "path": "...",
  "commit": "...",
  "heading": "M2 — Nonsingular Curvature Operator",
  "line_range": "..."
}
```

No scientific field may appear in Obsidian without at least one source assertion.

If a value is inferred rather than directly stated, mark it:

```text
assertion_status: inferred
```

Allowed assertion states:

```text
direct
compiled
inferred
extracted-unreviewed
human-reviewed
conflicted
superseded
```

---

# 5. Canonical Definition and Glossary Model

The glossary must no longer be generated from Zotero descriptions, keywords, filenames, headings, or arbitrary snippets.

Create canonical definition objects.

Each definition must include:

```yaml
stable_id:
term:
display_term:
aliases:
plain_definition:
technical_definition:
scope:
introduced_in:
paper_appearances:
source_pages:
related_definitions:
related_concepts:
related_symbols:
related_equations:
related_derivations:
related_predictions:
related_experiments:
references:
source_assertions:
review_state:
```

## Related definitions

`related_definitions` must use typed relationships:

```yaml
related_definitions:
  - stable_id: DEF-CURVATURE-LOCKING
    relation: depends-on
  - stable_id: DEF-PHASE-LOCKING
    relation: closely-related
  - stable_id: DEF-SPECTRAL-LOCKING
    relation: narrower
  - stable_id: DEF-DECOHERENCE
    relation: contrasts-with
```

Allowed relationship types should include:

```text
broader
narrower
depends-on
defines
used-by
closely-related
contrasts-with
special-case-of
generalization-of
measured-by
implemented-by
```

The Obsidian Definition tab must show:

1. plain definition;
2. technical definition;
3. source paper and page;
4. aliases;
5. related definitions with their definitions;
6. related concepts;
7. related equations;
8. papers using the term;
9. references supporting or contextualizing the term.

Do not display a paper abstract fragment as a definition.

Do not create glossary terms such as:

```text
Cited by
References
Abstract / record description
02 Concepts/
Experiments
```

---

# 6. Concept Objects

Definitions and concepts are related but not identical.

A concept object represents a scientific idea spanning multiple terms or objects.

Each concept must include:

```yaml
stable_id:
name:
summary:
canonical_definition_ids:
key_equations:
key_derivations:
claims:
predictions:
experiments:
papers:
references:
broader_concepts:
narrower_concepts:
related_concepts:
source_assertions:
```

The graph must use these explicit relationships rather than guessing connections from shared words alone.

---

# 7. Equation Objects

For every registered equation, include:

```yaml
stable_id:
canonical_name:
canonical_latex:
plain_explanation:
defined_symbols:
assumptions:
domain:
units:
derived_from:
derives:
paper_appearances:
source_pages:
related_definitions:
related_concepts:
related_derivations:
related_claims:
predictions:
experiments:
sympy:
lean:
physical_status:
experimental_status:
source_assertions:
```

## SymPy state

Include:

```yaml
sympy:
  baseline_status:
  effective_status:
  verification_kind:
  checker:
  assumptions:
  limitations:
  source_repository:
  source_path:
  source_commit:
```

## Lean state

Include:

```yaml
lean:
  status:
  declarations:
  assumptions:
  limitations:
  source_repository:
  source_path:
  source_commit:
```

SymPy and Lean statuses must always come from their current repository snapshots.

Never copy symbolic PASS into formal, physical, or experimental status.

Never copy Lean proof status into empirical status.

---

# 8. Derivation Objects

Derivations must come from:

1. explicit canonical Markdown derivations in GitHub;
2. exact paper sections and page ranges;
3. verified links to canonical equations.

Each derivation must contain:

```yaml
stable_id:
title:
statement:
assumptions:
inputs:
steps:
result:
equations_used:
equations_derived:
paper:
source_pages:
source_hash:
human_verified:
canonical_latex_verified:
sympy_status:
lean_status:
limitations:
source_assertions:
```

Raw PDF extraction must never automatically become a canonical derivation.

PDF extraction may create only:

```text
review-candidate
```

A candidate must remain outside the canonical graph until:

```yaml
human_verified: true
canonical_latex_verified: true
canonical_equations_resolved: true
```

---

# 9. Predictions

Predictions must come from explicit predictive statements in papers or canonical GitHub documents.

Each prediction must contain:

```yaml
stable_id:
statement:
model_or_equations:
assumptions:
predicted_observable:
predicted_range_or_pattern:
test_method:
falsifier:
paper_sources:
source_pages:
related_experiments:
current_status:
evidence:
source_assertions:
```

Do not create predictions from:

* headings;
* keywords;
* bibliography text;
* general discussion;
* repository filenames.

---

# 10. Experiments

Experiments must come from explicit paper sections, protocol documents, or maintained GitHub experiment records.

Each experiment must include:

```yaml
stable_id:
title:
research_question:
linked_predictions:
apparatus:
materials:
protocol:
controls:
measured_observables:
expected_result:
falsifier:
data_location:
analysis_code:
paper_sources:
source_pages:
current_status:
replication_status:
limitations:
source_assertions:
```

Distinguish:

```text
proposed
planned
in-progress
completed
analyzed
replicated
failed
inconclusive
```

A protocol ledger is not itself an experiment unless it contains one complete experiment record.

An apparatus component is not an experiment.

---

# 11. Current-State Conflict Detection

The compiler must fail when authoritative files disagree.

For example, if:

```text
WCT_FULL_EQUATION_LIST_CORRECTED.md
```

reports one status total and:

```text
WCT_MASTER_EQUATIONS_UPDATED.md
```

reports another, the compiler must:

1. identify the conflicting files;
2. identify the conflicting values;
3. apply declared source precedence only when permitted;
4. preserve the losing value as superseded provenance;
5. fail the release if a required canonical document remains stale.

Do not silently choose one value and continue.

Required conflict checks:

```text
equation ID mismatch
equation formula mismatch
equation status mismatch
symbol definition mismatch
paper title mismatch
paper hash mismatch
reference DOI collision
Lean declaration missing
SymPy object missing
stale repository commit
duplicate stable ID
missing source assertion
```

---

# 12. Reliable Build Command

Create one deterministic command in `geometry_of_resonance`:

```bash
python scripts/build_research_source_of_truth.py
```

Required modes:

```bash
python scripts/build_research_source_of_truth.py --check
python scripts/build_research_source_of_truth.py --build
python scripts/build_research_source_of_truth.py --diff
```

## `--check`

Must:

* fetch or inspect current repository heads;
* validate manifest sources;
* verify PDF hashes;
* verify bibliography;
* compare SymPy and Lean snapshots;
* detect stale generated files;
* detect conflicts;
* write no canonical output.

## `--build`

Must:

* run all checks;
* stop on blocking errors;
* compile the registry;
* write the Markdown source of truth;
* write the JSON registry;
* write the source lock;
* write the build report;
* produce deterministic output.

## `--diff`

Must show what changed since the previous compiled snapshot:

```text
new objects
removed objects
changed definitions
changed equations
changed statuses
changed Lean declarations
changed SymPy checks
changed derivations
changed predictions
changed experiments
changed references
source commit changes
```

---

# 13. GitHub Automation

Add a GitHub Actions workflow in `geometry_of_resonance`.

It must run:

```text
on changes to canonical Markdown
on changes to Papers/
on changes to the bibliography
on manual workflow dispatch
on repository dispatch from wct-sympy
on repository dispatch from wct-lean
```

The workflow must:

1. build the verified registry;
2. fail on source conflicts;
3. fail on stale generated output;
4. publish the build report;
5. commit or open a pull request for regenerated files;
6. never silently rewrite canonical scientific text.

The generated snapshot must identify the exact commits of all contributing repositories.

---

# 14. Replace the Untrusted Obsidian Sync Behavior

The Obsidian plugin must no longer build the research corpus from arbitrary local files, cached registry data, or PDF heuristics.

Obsidian must pull only these compiled artifacts from `geometry_of_resonance`:

```text
WCT_RESEARCH_SOURCE_OF_TRUTH.md
registry/wct-research-registry.json
registry/wct-source-lock.json
```

## Verified synchronization process

Implement two phases.

### Phase 1 — Check source state

The existing sync control should first:

1. fetch the three compiled artifacts;
2. confirm all three share the same `snapshot_id`;
3. verify the registry checksum against the lock file;
4. show repository commits;
5. show generated timestamp;
6. show whether the snapshot is current or stale;
7. show object counts;
8. show conflicts or warnings;
9. generate a preview diff;
10. make no changes to the vault.

Label this action:

```text
Check Source State
```

### Phase 2 — Apply verified snapshot

Enable this only after Phase 1 succeeds.

Label it:

```text
Apply Verified Sync
```

The apply step must:

1. stage all changes;
2. validate stable IDs;
3. validate links;
4. validate object counts;
5. validate snapshot ID;
6. apply updates atomically;
7. preserve user-authored content outside managed regions;
8. write a sync report;
9. update the graph only after the entire sync succeeds.

If any step fails, preserve the existing Obsidian state.

Do not partially update the vault.

---

# 15. Managed Obsidian Content

Mark compiler-controlled content with:

```yaml
managed_by: wct-research-source-sync
source_snapshot_id:
source_repository:
source_commit:
source_path:
source_pages:
source_hash:
last_verified_sync:
```

Where existing notes contain user-authored material, update only bounded sections:

```markdown
<!-- WCT-SYNC:BEGIN -->
Generated canonical content
<!-- WCT-SYNC:END -->
```

Do not overwrite content outside those markers.

The following folders should receive verified canonical projections:

```text
Research/01 Literature Notes
Research/02 Concepts
Research/03 Glossary
Research/04 Equations
Research/08 Derivations
Research/09 Predictions
Research/10 Experiments
Research/05 References
Research/06 Repositories
```

Use the existing folder structure where it already exists. Do not duplicate the vault into another parallel tree.

---

# 16. Remove Canonical Authority from the PDF Button

The local Obsidian PDF button must not create canonical definitions, equations, derivations, predictions, claims, or experiments.

Either remove it from the primary UI or rename it:

```text
Review PDF Candidates
```

It may only display review candidates already produced by the verified compiler.

If local extraction remains available:

* it must use only manifest-listed PDFs;
* it must verify the PDF SHA-256 first;
* it must show exact source pages;
* it must write only to a review folder;
* it must never update canonical objects;
* it must never change validation status;
* it must never generate stable canonical IDs;
* it must clearly display `UNVERIFIED EXTRACTION`.

The reliable ingestion process belongs in the `geometry_of_resonance` compiler, not in an Obsidian button.

---

# 17. Preserve the Existing UI

Keep the existing:

```text
graph
tabs
hover cards
inspector
equation rendering
timeline
search
filters
paper views
repository views
backlinks
navigation
```

Change the data beneath the UI, not the visual design.

Add a compact source-state indicator:

```text
Verified snapshot
Snapshot ID
Generated time
geometry commit
SymPy commit
Lean commit
Last successful sync
Current / stale / conflict
```

Each displayed object must have a visible `Source` section containing clickable links to:

* repository;
* file;
* commit;
* paper;
* page range;
* DOI or Zenodo record where applicable.

---

# 18. Required Acceptance Tests

## Source verification

1. Every canonical object has at least one source assertion.
2. Every source assertion contains a repository commit or immutable paper hash.
3. All compiled files share the same snapshot ID.
4. A PDF hash mismatch aborts the build.
5. A stale repository snapshot is visibly reported.
6. A missing canonical source aborts the build.
7. Conflicting canonical status totals abort the build.

## Glossary

8. Definitions are not generated from Zotero descriptions alone.
9. Every definition has a source paper or canonical GitHub source.
10. Related definitions use typed relationships.
11. Related definitions display their actual definitions.
12. Headings such as `References` and `Cited by` do not become glossary terms.
13. Aliases resolve to one canonical definition.

## Concepts

14. Concepts connect explicitly to definitions, equations, papers, predictions, experiments, and references.
15. Concept relationships come from compiled relationships, not word-overlap guessing.

## Equations

16. All 142 registered equation objects resolve to canonical IDs.
17. Canonical LaTeX comes from `geometry_of_resonance`.
18. SymPy status comes only from the locked `wct-sympy` snapshot.
19. Lean status comes only from the locked `wct-lean` snapshot.
20. Symbolic PASS does not imply formal or empirical PASS.
21. Formula conflicts are visible and block release.

## Derivations

22. Canonical derivations include source pages and assumptions.
23. Raw PDF extraction cannot become canonical automatically.
24. Unreviewed extraction remains outside the canonical graph.
25. Canonical equation links must resolve before promotion.

## Predictions and experiments

26. Predictions originate in explicit source statements.
27. Every prediction has a measurable observable and falsifier.
28. Experiments originate in explicit protocols or experiment records.
29. Every experiment links to predictions and source pages.
30. Proposed and completed experiments remain distinguishable.

## References

31. References are normalized by DOI, citation key, and title.
32. Duplicate DOI entries are reported.
33. Paper bibliography entries missing from the canonical BibTeX file are reported.
34. References are linked to the papers and objects that cite them.

## Obsidian sync

35. Checking source state writes no canonical changes.
36. Applying a sync requires a verified snapshot.
37. Snapshot mismatch aborts the sync.
38. Failed sync leaves the existing vault unchanged.
39. Manual content outside managed blocks is preserved.
40. The graph rebuilds only after successful atomic synchronization.
41. The sync report lists every changed object.
42. The UI displays the exact source snapshot and repository commits.

## UI preservation

43. Existing graph navigation still works.
44. Existing tabs still work.
45. Equation rendering remains readable.
46. Hover cards and inspector remain readable.
47. Timeline and filters remain functional.
48. No second Obsidian plugin is created.

---

# 19. Completion Standard

This task is complete only when the user can select any object in Obsidian and answer:

```text
What is it?
What is its canonical definition?
Where did that definition come from?
Which paper and pages contain it?
Which related definitions explain it?
Which concepts contain it?
Which equations use it?
Where are those equations derived?
What is the current SymPy state?
What is the current Lean state?
Which predictions follow from it?
Which experiments test it?
Which references support or contextualize it?
Which repository commits produced the displayed state?
Is the displayed information current or stale?
```

Do not report success because more nodes were generated.

Do not report success because the sync button completed.

Do not report success because PDF extraction produced files.

Success means the complete Obsidian research state is reproducibly compiled from verified GitHub repositories, canonical papers, and canonical references, with field-level provenance and an auditable current-source snapshot.
