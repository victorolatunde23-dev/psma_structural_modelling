# Project plan and inspection of the original pipeline

This document answers the ten points requested before implementation
(brief §22). It is written against `PDB_structure_pipeline.ipynb`, the version
with the entrance-lid classifier, which the user ran successfully in Colab
(gemmi included).

The live progress checklist is in [`CHECKLIST.md`](CHECKLIST.md).

---

## 1. Map of the original notebook

### Global configuration variables (notebook cells)

| Variable | Value | Becomes |
|---|---|---|
| `ACCESSION` | `"Q04609"` | `default_config.yaml: target.accession` |
| `LIMIT` | `None` | `run.limit` |
| `USE_COORDINATES` | `True` | `run.use_coordinates` |
| `CACHE` | `Path("pdb_cache")` | `paths.cache` (default `cache/`) |
| `OUT_XLSX` | `f"{ACCESSION}_structures.xlsx"` | `paths.outputs` + `outputs.survey_workbook` |

### Module-level constants

| Constant | Purpose | Becomes |
|---|---|---|
| `RCSB_SEARCH`, `RCSB_GRAPHQL`, `RCSB_CIF`, `PDBE_SIFTS`, `PDBE_BEST`, `UNIPROT_JSON`, `PDBE_VAL_PERCENTILES`, `PDBE_VAL_OUTLIERS` | API endpoints | `default_config.yaml: endpoints` |
| `IONS_OF_INTEREST`, `GLYCANS`, `ADDITIVES`, `METALS` | ligand classes | `ligand_classes.yaml` |
| `CUTOFF_METAL`, `CUTOFF_ANION`, `CUTOFF_LIGAND`, `BACKBONE` | contact cut-offs | `default_config.yaml: contacts` |
| `LITERATURE_SITES` | Q04609 active-site roles | `psma_active_site.yaml` |
| `FLEXIBLE_REGIONS`, `LID_STATE_CONFIG` | Q04609 regions, lid calibration | `psma_regions.yaml` |
| `HELIX_CA_I3_MAX`, `HELIX_FRACTION_MIN`, `MIN_LID_MODELLED`, `ASHMAN_D_MIN`, `LOW_CONFIDENCE_RES` | lid classifier thresholds | `psma_regions.yaml: entrance_lid.classifier` |
| `GRAPHQL_QUERY`, `RCSB_VALIDATION_QUERIES` | GraphQL text | kept in `rcsb.py` / `validation.py` |

### Functions and their dependencies

| Function | Inputs | Depends on | Network | gemmi |
|---|---|---|---|---|
| `make_session` | – | requests, urllib3 | – | – |
| `cached_json` | session, url, cache file | – | yes | – |
| `rcsb_search` | session, accession | – | yes | – |
| `rcsb_metadata` | session, ids, cache | `cached_json` | yes | – |
| `entity_is_target` | entity dict, accession | – | – | – |
| `sifts_segments` | SIFTS json | – | – | – |
| `uniprot_info` | session, accession | `cached_json` | yes | – |
| `to_ranges`, `label_to_unp`, `oligomer_label` | primitives | – | – | – |
| `choose_assembly`, `global_symmetry` | RCSB assembly dicts | – | – | – |
| `load_structure` | session, PDB ID | requests | yes | yes |
| `coordinate_analysis` | gemmi Structure, target entity ids | – | – | yes |
| `construct_differences` | entity seq, SIFTS segments, UniProt seq | – | – | – |
| `process_entry` | entry metadata, SIFTS, … | most of the above | via `load_structure` | yes (optional) |
| `cross_check` | id lists, metadata, PDBe, UniProt | – | – | – |
| `rcsb_validation`, `pdbe_validation`, `_metric_key` | session, ids | `cached_json` | yes | – |
| `active_site_residues` | accession, sequence, cached UniProt json | `LITERATURE_SITES` | – | – |
| `binding_site_analysis` | pdb, ctx, outliers | `_neighbor_search`, `_label`, `classify_component` | – | yes |
| `active_site_analysis` | pdb, ctx, sites, sequence, outliers | `_unp_maps` | – | yes |
| `ca_coordinates` | pdb, ctx | `_unp_maps` | – | yes |
| `_kabsch`, `structural_variability` | CA dicts | numpy | – | – |
| `template_ranking` | Summary, Chains, ActiveSite tables | – | – | – |
| `analyse_entry_extended` | per-entry context | validation + gemmi analyses | yes | yes |
| `merge_into_summary` | result tables | – | – | – |
| `_helicity`, `_two_means_1d`, `construct_ranges`, `lid_conformation`, `lid_entry_summary`, `lid_ligand_association` | CA dicts, tables | numpy | – | – |
| `write_excel` | dict of DataFrames | openpyxl | – | – |

### Data flow

```
rcsb_search ─► ids ─► rcsb_metadata ─► meta ─┐
uniprot_info ─► useq, xrefs ──────────────────┤
PDBe best_structures ─► best ─────────────────┤
rcsb_validation ─► rcsb_val ──────────────────┤
active_site_residues ─► sites ────────────────┤
                                              ▼
        for each PDB: SIFTS ─► process_entry ─► summary / chains / ligands / assemblies + ctx
                                   ctx ─► analyse_entry_extended ─► validation / binding / glyco / active / CA
                                              ▼
merge_into_summary ─► template_ranking ─► structural_variability ─► lid_conformation ─► lid tables
                                              ▼
                                cross_check ─► write_excel
```

---

## 2. Functions moved unchanged (logic identical; only constants become parameters)

`make_session`, `cached_json`, `rcsb_search`, `rcsb_metadata`, `entity_is_target`,
`sifts_segments`, `uniprot_info`, `to_ranges`, `label_to_unp`, `oligomer_label`,
`global_symmetry`, `load_structure`, `coordinate_analysis`, `construct_differences`,
`cross_check`, `rcsb_validation`, `pdbe_validation`, `_metric_key`, `ca_coordinates`,
`_kabsch`, `structural_variability`, `merge_into_summary`, `_helicity`,
`_two_means_1d`, `construct_ranges`, `lid_entry_summary`, `lid_ligand_association`,
`active_site_analysis` (identifier columns added), `template_ranking` (criteria
now read from config; the defaults reproduce the old ranking exactly),
`lid_conformation` (thresholds read from config; same defaults).

Changes to these functions are limited to:

- module-level constants are passed in from the configuration;
- `print(..., file=sys.stderr)` becomes `logging`;
- identifier columns are added (model number, label asym ID, entity ID,
  author residue number, calculation status, warning).

---

## 3. Functions that need correction or redesign

| Function | Issue | Milestone |
|---|---|---|
| `binding_site_analysis` | **Bug:** `Waters in contact` counted atom–water pairs rather than unique waters, which inflated the number for multi-atom ligands. | **M1 — fixed.** The old count is kept as `Ligand atom–water pairs` so results stay comparable. |
| `binding_site_analysis` | Residue labels combined the model's amino acid with the UniProt number without checking they agree. A mismatch (a mutation, or a SIFTS error) could produce labels such as `A377`. | **M1 — added** a `Residue/UniProt mismatches` column. Mismatched residues are kept, but flagged. |
| `binding_site_analysis` | Glycosylation detection happened inside the binding-site loop. | **M1 — split** into `glycans.find_glycosylation_sites`; the rows are identical. |
| `binding_site_analysis` | Symmetry contacts use the crystal lattice, so a biological dimer partner generated by symmetry looks the same as a crystal-packing contact. | M2: contacts are calculated in the built biological assembly (`gemmi.make_assembly`) and lattice-only contacts are labelled separately. |
| `choose_assembly` | No reason was recorded. With no author-defined assembly, it fell straight back to the lowest assembly ID, skipping the "software-defined next" step. | **M1 — redesigned** to author → software → assembly 1, returning the reason. This selects the same assembly as before whenever an author-defined one exists. |
| `coordinate_analysis` | Counts copies from the assembly generators but never builds the coordinates, which interfaces will need. Always uses `st[0]`. | M1: model index added as a parameter. M2: `build_assembly()` added. |
| `process_entry` | Model 1 is hard-coded. | M1: a `model_policy` setting. `first` is implemented. `all` is accepted but **recorded as a warning and falls back to model 1** until M2, because the CA and lid keys must gain a model index first. |
| `active_site_residues` | Printed warnings to stderr only. | **M1:** returns the dropped residues so they appear in a table (no silent failure). |
| `active_site_analysis` | When two altlocs have different residue names (microheterogeneity), one of them is silently kept. | M2: record microheterogeneity explicitly. |
| `template_ranking` | One hard-coded ranking. | M1: driven by config (`global` mode). M2: ten ranking modes with filters, weights and missing-value policy. |
| `cached_json` | A 404 (for example, a validation report not yet released) is cached as `{}` forever. | **M1:** cache entries now record status and fetch time; `refresh_404_after_days` re-fetches stale 404s. |
| `write_excel` | No data dictionary, units, status colouring or missing-value convention. | **M1:** all added; existing sheets and column names preserved. |

---

## 4. Dependency and installation plan (Google Colab)

| Tier | Packages | How in Colab | Needed from |
|---|---|---|---|
| Core | numpy, pandas, scipy, requests, openpyxl, pyyaml, gemmi, matplotlib, seaborn, scikit-learn, networkx, pytest | `pip install -e .` (from `pyproject.toml`) | M1 |
| Structure extras | biopython, freesasa | `pip install -e ".[structure]"` | M2 (BSA, with a Shrake–Rupley fallback if freesasa is unavailable) |
| Chemistry | rdkit | `pip install -e ".[chem]"` | M3 (ligand descriptors) |
| AlphaFold | tmtools (TM-score) | `pip install -e ".[alphafold]"` | M4 |
| Preparation | pdbfixer, openmm, propka, pdb2pqr | `pip install -e ".[prep]"` (pdbfixer from GitHub) | M5 |
| Conda-only | ambertools, openbabel, fpocket, dssp | `condacolab`, then `mamba install -c conda-forge …` (restarts the runtime once) | M5–M6 (optional) |
| Licensed | MODELLER | needs a licence key: `KEY_MODELLER=… mamba install -c salilab modeller` | M5 (optional) |
| Trajectory | MDAnalysis | `pip install -e ".[md]"` | M7 |

Colab recipe (the first cell of every notebook):

```python
!git clone -q https://github.com/victorolatunde23-dev/psma_structural_modelling.git
%cd psma_structural_modelling
!pip -q install -e .
```

To persist the cache between Colab sessions, mount Google Drive and set
`paths.cache` to a Drive folder.

---

## 5. File tree

```
psma_structural_modelling/
├── notebooks/
│   ├── 01_structure_survey.ipynb              M1  ✔
│   ├── 02_interface_pocket_validation.ipynb   M2–M3
│   ├── 03_alphafold_comparison.ipynb          M4
│   ├── 04_model_preparation.ipynb             M5
│   ├── 05_md_system_setup.ipynb               M6
│   └── 06_md_trajectory_analysis.ipynb        M7
├── src/psma_pipeline/
│   ├── __init__.py          package version, logging setup
│   ├── config.py            YAML loading, merging, validation against sequence   M1
│   ├── status.py            Status values, IssueLog (warnings recorded in tables) M1
│   ├── cache.py             HTTP session, JSON/file cache, hashing               M1
│   ├── rcsb.py              search, GraphQL metadata, mmCIF download             M1
│   ├── pdbe.py              SIFTS, best_structures                               M1
│   ├── uniprot.py           sequence, xrefs, features                            M1
│   ├── numbering.py         label_seq ↔ UniProt ↔ author numbering, ranges        M1 (new)
│   ├── assemblies.py        assembly choice, generator expansion, coordinates    M1 (+M2 build)
│   ├── ligands.py           ligand classification                                M1 (new)
│   ├── pockets.py           binding sites, metal coordination                    M1 (+M3 fingerprints)
│   ├── glycans.py           glycosylation sites                                  M1 (+M5)
│   ├── active_site.py       active-site definition, B/occupancy                  M1 (new)
│   ├── conformations.py     superposition, variability, entrance lid             M1 (new)
│   ├── validation.py        RCSB/PDBe validation (+M2 wwPDB XML)                 M1
│   ├── ranking.py           config-driven rankings                               M1 (+M2 modes)
│   ├── survey.py            per-entry orchestration, cross-check, summary        M1 (new)
│   ├── sanity.py            sanity checks and validation-set selection           M1 (new)
│   ├── manifest.py          reproducibility manifest, software detection         M1 (new)
│   ├── excel_output.py      workbook writer, data dictionary, comparison         M1
│   ├── interfaces.py        M2
│   ├── waters.py            M3 (conserved waters; new)
│   ├── variants.py          M3
│   ├── alphafold.py         M4
│   ├── metal_sites.py       M5
│   ├── model_preparation.py M5 (repairs, protonation, construct context)
│   ├── md_readiness.py      M6
│   └── trajectory_analysis.py M7
├── config/   default_config.yaml, psma_active_site.yaml, psma_regions.yaml,
│             ligand_classes.yaml (M1); md_protocol.yaml (M6)
├── tests/    test_numbering, test_assemblies, test_conformations, test_ranking,
│             test_config, test_excel_output, test_regression (M1); others later
├── scripts/  run_survey.py (command-line equivalent of notebook 01)
├── data/  cache/  outputs/   (contents git-ignored)
├── pyproject.toml  environment.yml  requirements.txt  README.md
```

Departures from the suggested tree, and why:

- **`numbering.py` added.** Numbering is the most error-prone part, and `tests/test_numbering.py` needs one obvious place to test.
- **`survey.py`, `sanity.py`, `manifest.py` and `status.py` added.** These hold orchestration and cross-cutting concerns that would otherwise end up in notebooks.
- **`conformations.py` holds variability and the lid classifier.** Both share superposition code and will be reused for AlphaFold models (M4) and trajectory frames (M7).
- **`active_site.py` and `ligands.py` are split out of `pockets.py`.** The active site is also used by ranking, variants and MD readiness.
- **`waters.py` planned.** Conserved waters (brief §14) don't fit naturally in pockets.

---

## 6. Milestone plan

| M | Scope | Notebook | Exit criteria |
|---|---|---|---|
| 1 | Refactor into modules, config, logging, tests, reproducibility manifest, data dictionary, sanity checks, validation-set selection | 01 | Tests pass in Colab; validation-set sanity checks have no critical FAIL; `compare_workbooks` against the last good workbook shows only documented changes |
| 2 | Biological-assembly building, dimer interfaces (contacts, H-bonds, salt bridges, BSA), interface comparison and clustering; wwPDB validation XML (RSCC/RSRZ/Q-score, EM map fit); ranking modes | 02 | BSA test on a known complex; interface found in every dimer; ranking modes auditable |
| 3 | Ligand interaction fingerprints, sub-pockets, ligand clustering, ligand-conditioned ranking; conserved waters; variant mapping | 02 | Fingerprint tests; sub-pocket residues validated; variant reference residues validated |
| 4 | AlphaFold/ColabFold import, PAE/pLDDT, monomer and multimer comparison, AF rankings | 03 | PAE indexing and chain-boundary tests; confident-dimer rule |
| 5 | Repair assessment and rebuilding, protonation, metal sites, ligand and glycan preparation, construct/membrane context | 04 | Never overwrite originals; provenance chain; alternative microstates |
| 6 | MD readiness validator, engine-agnostic setup, OpenMM inputs, minimisation checks | 05 | No READY status with any critical FAIL; Zn strategy mandatory |
| 7 | Trajectory analysis, replica convergence, lid dynamics | 06 | Convergence checked before pooling |

At the end of every milestone: run the validation set, display the summaries,
save the outputs, list warnings and failures, compare with the previous
milestone, and stop if a critical numbering or assembly check fails.

---

## 7. Backwards compatibility of the Excel output

1. **All 14 original sheets keep their names and their columns.** No column is renamed or removed.
2. **New information is added as new columns or new sheets only.** Every
   addition is listed in `docs/CHANGES.md`.
3. **One value changes meaning:** `BindingSites → Waters in contact` now counts
   unique waters (the bug fix). The old atom-pair count is kept in a new column,
   `Ligand atom–water pairs`.
4. `excel_output.compare_workbooks(old, new)` reports, per sheet:
   - added and removed columns;
   - row-count changes;
   - per-cell differences on shared keys, with a numeric tolerance.
   Notebook 01 runs it against the user's previous workbook.
5. Later milestones write their own workbooks (`modelling_assessment.xlsx`,
   `model_preparation.xlsx`, `md_readiness.xlsx`, `trajectory_summary.xlsx`)
   rather than growing the survey workbook past usability. The survey workbook
   stays the canonical record of experimental structures.
6. Every table is also written as CSV. Parquet is written when `pyarrow` is
   installed (it is in Colab).

---

## 8. Analyses that need optional external software

| Analysis | Preferred | Fallback when missing |
|---|---|---|
| SASA / buried surface area | FreeSASA | numpy Shrake–Rupley (documented; method recorded per row) |
| Secondary structure | DSSP (mkdssp) | CA-geometry helicity (already used for the lid) |
| Ligand descriptors, chemical similarity | RDKit | values recorded as missing, status `NOT RUN` |
| Interaction typing | PLIP | in-house geometric rules (distance + atom type) |
| Pocket volume | fpocket | `NOT RUN` |
| pKa / protonation | PROPKA, PDB2PQR | `NOT RUN`; no state assigned automatically |
| Missing atoms / loops | PDBFixer, MODELLER | assessment only; no rebuilding |
| Ligand parameters | AmberTools (antechamber), Open Babel | MD readiness = FAIL for that ligand |
| Geometry validation | MolProbity | clash and Ramachandran estimates from gemmi |
| MD | OpenMM (GROMACS/Amber via manifests) | setup manifest only |
| Electrostatics | APBS | `NOT RUN` |
| Visualisation | ChimeraX | matplotlib only |
| TM-score | tmtools | CA RMSD only |
| Trajectories | MDAnalysis or MDTraj | – (required for notebook 06) |

`manifest.detect_software()` records availability and version at the start of
every run. Each analysis records `NOT RUN (tool unavailable)` rather than an
empty or estimated value.

---

## 9. Small validation dataset and expected sanity checks

The validation set is assembled by `sanity.select_validation_set()`. It combines
fixed IDs from the config with entries chosen automatically from the survey
metadata, so it adapts as new structures are released:

| Category | How it is chosen | Expected sanity checks |
|---|---|---|
| Calibration, closed lid | `3BHX` (fixed) | lid = closed |
| Calibration, open lid | `2OOT` (fixed) | lid = open |
| Top-ranked template | `9S7U` (fixed; skipped with a warning if absent) | ranks near the top; active site complete |
| High-resolution dimer, 2 chains in ASU | best-resolution X-ray entry with 2 target chains in the ASU and an assembly of 2 | assembly copies = 2 = ASU copies |
| ASU monomer → dimer assembly | best-resolution entry with 1 target chain and an author assembly of 2 | gemmi assembly copies = 2; note says "generated by symmetry" |
| Ligand-bound | best-resolution entry with a ligand of interest | binding site non-empty; ligand contacts include Zn-site or S1′ residues |
| Apo | best-resolution entry with no ligand of interest | no ligand rows |
| Cryo-EM | best-resolution EM entry | RSRZ blank; EM flag on normalised B |
| Lid disordered | from the previous workbook's `EntranceLid` sheet (`disordered`), or `validation_set.extra_ids` | lid = disordered, never open |

Automatic checks (`sanity.run_sanity_checks`), each PASS / WARNING / FAIL:

- **[critical]** gemmi assembly copies agree with RCSB `oligomeric_count` for homomeric entries;
- **[critical]** no target chain has more than 10% of its SIFTS-mapped residues differing from UniProt (that would indicate a numbering offset). A missing SIFTS mapping or an inconsistent segment is a WARNING, because nothing is mis-numbered;
- **[critical]** every active-site residue modelled with UniProt numbering matches the canonical amino acid (otherwise excluded and flagged);
- **[critical]** calibration structures receive their literature lid state;
- Zn count per protomer in the assembly is 0 or 2 (WARNING otherwise);
- Zn coordination distances lie within the configured range;
- no "open" lid call has fewer than 6/8 lid residues modelled;
- B-factor ratios are positive; resolutions are positive;
- every entry has at least one target chain.

---

## 10. Scientific assumptions that need explicit configuration

All of these live in `config/` and are recorded in the reproducibility manifest:

| Assumption | Key | Default |
|---|---|---|
| Target accession | `target.accession` | Q04609 |
| Model policy (NMR) | `structure.model_policy` | `first` |
| Assembly preference | `structure.assembly_preference` | author → software → first |
| Contact cut-offs | `contacts.*` | metal 2.8 Å, anion 3.6 Å, ligand 4.0 Å, glycan C1–ND2 2.0 Å |
| Ion classes | `ligand_classes.yaml` | Zn/Ca/Cl of interest; additive list |
| Active-site residues and roles | `psma_active_site.yaml` | 14 literature residues + UniProt features |
| Flexible regions | `psma_regions.yaml: flexible_regions` | lid 541–548, glutarate sensor 692–704 |
| Lid definition and calibration | `psma_regions.yaml: entrance_lid` | 3BHX closed, 2OOT open; helicity 6.0 Å / 0.6; Ashman D 2.0; ≥ 6/8 modelled |
| Low-confidence resolution | `entrance_lid.classifier.low_confidence_resolution` | 3.5 Å |
| Expected Zn per protomer | `psma_active_site.yaml: metal_site.expected_zn_per_protomer` | 2 |
| Zn coordination range | `metal_site.zn_ligand_distance_range` | 1.8–2.6 Å |
| Ranking criteria, directions, weights | `default_config.yaml: ranking.modes` | global mode = original equal-weight ranking |
| Random seed | `run.random_seed` | 20261021 |
| Sub-pockets, topology, MD protocol, Zn MD treatment | added in M3, M5 and M6 | – (the Zn treatment must be chosen by the user) |
