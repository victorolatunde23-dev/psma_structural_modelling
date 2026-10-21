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

### Bug fix found by the gemmi tests in Colab (affects the original notebook too)

`binding_site_analysis` and glycosylation detection read a neighbour-search
hit's position as `mark.x/.y/.z`. Those attributes don't exist in current
gemmi, so the step raised `AttributeError`. The pipeline caught the error and
wrote it into `Summary → Warnings` as `binding sites: ...`, but the affected
outputs were left empty:

- the **BindingSites** and **Glycosylation** sheets;
- **Max ligand B ratio**, **Glycosylated Asn**, **Arene-site contact** and **Largest ligand** in Summary;
- **LidVsLigand**.

The lid states, active-site B-factors, variability and ranking were **not**
affected, because they use CA coordinates only.

A second, subtler error was fixed at the same time. `mark.pos` is wrapped into
the unit cell, so the distance to a symmetry or lattice copy could be wrong for
atoms near a cell edge, and pure lattice-translation contacts were not flagged
as symmetry contacts. Distances now use the nearest periodic image
(`UnitCell.find_nearest_pbc_image`). A regression test covers a contact
across a cell edge.

### Validation-set findings in Colab (first real run)

- **Lid calibration.** 2OOT has an unmodelled lid, so the classifier correctly
  calls it *disordered*. It can't serve as the "open" calibration structure.
  - The open calibration is now **2XEG**: ARM-P4 sits in the arene-binding site, which requires an open lid.
  - 2OOT stays in the validation set as the disordered-lid example.
  - The lid-calibration check now returns FAIL only when the classifier calls the *opposite* state. An unusable calibration structure gives a WARNING.
- **Zn coordination.** `Coordination (atom: Å)` lists every atom within 2.8 Å, including carbons such as Asp CG of a bidentate carboxylate. Two new columns list potential ligand atoms only, and the Zn distance check uses them:
  - `Donor coordination number`;
  - `Donor atoms (N/O/S: Å)`, with alternate locations marked.

### First full run in Colab (90 entries): comparison with the original notebook

All differences were traced to these causes:

1. **The `mark.x` bug was fixed.** BindingSites (474 rows), Glycosylation (642) and LidVsLigand are now filled, and the bug message is gone from Summary → Warnings in all 90 entries.
2. **SIFTS mappings came back for 3BXM, 9HLW and 9HVL.** The original run cached an empty PDBe answer for these entries; refetching recovered the mappings. Five chains gain UniProt numbering, active-site, lid and variability rows, and 3BXM's E424A mutation is now detected. Ranking scores shift slightly as a knock-on effect, because ranks are relative.
3. **2XEG is now the open-lid calibration.** It supplies the reference vote that was missing before: 5 lids change from ambiguous to open, and lid confidences and votes change accordingly. PC1 scores change because the PCA is refitted with the 5 extra chains.
4. **E424 mutants** are excluded from the active-site mean normalised B (15 entries).
5. **List order is now deterministic.** Summary lists (ligands, glycans, additives) are sorted alphabetically; RCSB had returned entities in a different order between runs. `compare_workbooks` now ignores list order and matches repeated keys by occurrence.
6. **Zn check.** Donors between 2.6 Å and the 2.8 Å cut-off are reported as long contacts (PASS), typically the second oxygen of bidentate D453. Distances below 1.8 Å are flagged as implausible.

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

## Milestone 2 — interfaces, local model fit, ranking modes (v0.2.0)

**New output:** `outputs/<ACC>_modelling_assessment.xlsx`, written by notebook 02 or `scripts/run_assessment.py`. Its sheets are:
- AssessmentSummary
- Interfaces, CrystalContacts, InterfaceContacts, InterfaceComparison, CanonicalInterface
- LocalModelFit, LigandModelFit, EMMapFit
- TemplateRankings, RankingTop
- AssessmentIssues, Software, Reproducibility, DataDictionary

The CSV/Parquet tables and the interface Jaccard matrix are also written. The survey workbook from notebook 01 is unchanged.

**Design decisions:**
- **Assemblies are built from gemmi's assembly operators applied to an atom table,** not with `gemmi.make_assembly`. Author chain, label_asym and UniProt numbering therefore survive every copy, and the geometry is testable without gemmi.
- **SASA:** FreeSASA when installed, otherwise a numpy Shrake–Rupley implementation. The method used is recorded on every interface row. Tested against the analytic two-sphere complex (within 1%).
- **The canonical interface is a consensus:** residues present in ≥ 50% of author-defined dimer interfaces. Noncanonical means a Jaccard below 0.3, or a BSA below half the canonical median.
- **Crystal-lattice search:** every space-group operation × lattice shift of each target chain. This finds the canonical dimer in entries whose assembly is annotated as a monomer.
- **Local fit uses method-specific rules:**
  - X-ray: RSCC/RSRZ.
  - Cryo-EM: Q-score relative to the resolution-expected value (Pintilie et al. 2020).
  - Categories are combined only after these rules. A ligand without a metric is always "unsupported or unavailable".
- **Ranking modes:** global (unchanged), active_site, ligand_bound, apo, open_lid, closed_lid, dimer_interface, md_start, glycosylated, cryoem_reference.

**Bug fixed during development:** in a ranking mode, exclusion rules with several conditions removed only rows matching *all* of them. They now remove rows matching *any* (regression test added).


## Milestone 3 — pockets, conserved waters, variants (v0.3.0)

**New output:** `outputs/<ACC>_pockets_waters_variants.xlsx`, written by notebook 02 part 2 or `scripts/run_pockets.py`. A separate workbook keeps part 1 and part 2 independently re-runnable. Its sheets are:
- SubpocketDefinition, PocketFingerprints, LigandInteractions, SubpocketOccupancy
- PocketSimilarity, LigandClusters, ClusterDiagnostics, LigandChemistry
- ConservedWaters, WaterMembers, WaterSites
- Variants, VariantStructureMapping, VariantSummary
- LigandConditionedRanking (only when a reference ligand is given)
- PocketChecks, PocketIssues, Software, Reproducibility, DataDictionary

**Design decisions:**
- **Interactions are typed by distance only** and labelled "possible" where geometry would be needed (H-bond, salt bridge, π-stacking, cation–π). Ligand rings are found from a distance-derived bond graph (planar 5/6-membered cycles).
- **Sub-pockets** are defined in `config/psma_active_site.yaml` and validated against the canonical sequence, like every configured residue.
- **One representative per (entry, ligand)** – the copy with most contacts – is used for similarity and clustering; every copy is kept in PocketFingerprints.
- **Clustering** on 1 − interaction Tanimoto: hierarchical (average linkage) at several thresholds with silhouette diagnostics, plus DBSCAN. No fixed number of clusters.
- **Ligand-conditioned ranking** (`rank_for_ligand`) reuses the stored fingerprints, so a new reference ligand costs seconds. Apo entries get missing similarity (= worst).
- **Conserved waters:** X-ray ≤ 2.2 Å and clashscore ≤ 10 only; superposition on active-site CA; complete linkage at 1.2 Å; Zn-bridging waters are flagged for the Milestone 5 metal-model decision.
- **Variants:** reference residue checked against the canonical sequence (mismatches are FAIL and not mapped); structural context from the top `md_start` template's assembly; text is always a hypothesis, never a prediction.

**Changed in part 1:** the LocalModelFit "conserved waters" row is now NOT APPLICABLE with a pointer to the new workbook (was NOT RUN, "Milestone 3").
