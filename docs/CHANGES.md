# Changes relative to the original notebook

The original notebook is `PDB_structure_pipeline.ipynb`, the version with the
entrance-lid classifier.

## Milestone 1 — refactor (v0.1.0)

**Verification.** The original script and the new package were run side by
side on identical synthetic inputs, without coordinates. Every value in every
original sheet matched except for the intentional changes listed below.
`compare_workbooks()` repeats this check on real data in notebook 01,
section D.

### Unchanged
- All 14 original sheets, with the same names, in the same order.
- All original column names.
- Cache file names, so an existing `pdb_cache/` folder can be reused by
  pointing `paths.cache` at it.
- Scientific defaults: cut-offs, ligand classes, active-site residues, lid
  thresholds and calibration structures, and the ranking criteria and weights.

### Values that can differ (intended)

| Where | Change | Why |
|---|---|---|
| BindingSites → `Waters in contact` | Counts **unique** water molecules. The old count (ligand atom–water pairs) moved to `Ligand atom–water pairs`. | Bug: a ligand with 30 atoms near one water was reported as having 30 waters. |
| Summary → `Warnings` | Adds `no SIFTS mapping to the target accession` when relevant. | No silent failure. |
| Summary → `Active-site mean normalised B` | Excludes residues whose model amino acid differs from UniProt (for example E424A mutants). | Brief rule: residues that disagree with UniProt are excluded. |
| Assembly choice when **only software-defined** assemblies exist | Picks the lowest-numbered software-defined assembly. The old code took the lowest RCSB ID by string sort, so `-10` sorted before `-2`. | Deterministic, numeric order, as in the brief. Identical to the old result whenever an author-defined assembly exists. |
| Notes | More rows (assembly reason, schema variant, missing-value rule). | Documentation. |

### Added columns
- **Summary:**
  - `Assembly choice reason`, `Assembly copies source`, `RCSB assembly chain count`
  - `Polymer entities`, `Model analysed`
  - `Active-site residues differing from UniProt`
- **Chains:** `Model`, `Calculation status`, `Warning`.
- **Ligands:** `Chains (label)`.
- **BindingSites:**
  - `Model`, `Chain (label)`, `Entity`
  - `Ligand atom–water pairs`, `Residue/UniProt mismatches`
  - `Calculation status`, `Warning`
- **Glycosylation:** `Model`, `Chain (label)`, `Entity`, `Root sugar chain (label)`, `Calculation status`, `Warning`.
- **ActiveSite:**
  - `Model`, `Chain (label)`, `Entity`
  - `Author residue number`, `Label seq`
  - `Calculation status`, `Warning`
- **Validation:** `Calculation status`, `Warning`.
- **Variability, EntranceLid:** `Model`.
- **Assemblies:** `Selection reason`.
- **TemplateRanking:** `Ranking mode`, `component: <criterion>` columns, `Criteria used`, `Score explanation`.

### Added sheets
`SanityChecks`, `ValidationSet` (notebook runs), `ActiveSiteDefinition`,
`RunIssues`, `Software`, `Reproducibility`, `DataDictionary`.

### Added files
- `<ACC>_manifest.json`: the reproducibility manifest.
- `survey_tables/*.csv` and `*.parquet`: every sheet as a machine-readable table.
