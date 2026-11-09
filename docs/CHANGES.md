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

### Milestone 3 — fixes after the first full Colab run (90 entries, 8/8 pocket checks PASS)

- **Partner-protomer flag (bug).** Every ligand was flagged as touching a partner protomer, because the ligand's own label (its non-polymer chain) was compared with protein chains. The ligand's "home" protomer is now the protein protomer it contacts most. On the real data only 6H7Y TAM truly reaches the other protomer.
- **Catalytic pocket definition.** The survey's active-site set includes UniProt Ca/Cl binding sites, so the water region and the variants' "distance to active site" covered a much larger area (e.g. V627L at 1.3 Å from the "active site", 39 Å from Zn). Waters and variants now use the *catalytic pocket*: literature residues + Zn ligands + sub-pockets. The column is renamed `Distance to catalytic pocket (Å)`. Superposition still uses the broader set.
- **DBSCAN.** eps 0.4 merged 79/84 ligands into one cluster. Default eps is now 0.25, and the diagnostics report eps 0.2–0.4.
- **TAM** (a tris analogue, buffer) is now an additive. This also removes it from the ligands of interest in notebook 01.


## Milestone 4 — AlphaFold comparison and membrane context (v0.4.0)

**New output:** `outputs/<ACC>_alphafold_membrane.xlsx` (notebook 03 or `scripts/run_alphafold.py`), plus
`alphafold_tables/` and `md_membrane_inputs/<model>/`. AlphaFold2 results are written to `cache/alphafold/colabfold/<job>/`.
The sheets are:
- AFModels, AFChains, AFResidueConfidence, AFRegionConfidence, AFDomainPAE
- AFvsExperimental, AFRegionRMSD, AFEntranceLid
- AFInterfaces, AFInterfaceContacts, AFRankings, AFRankingTop
- MembraneTopology, TopologyComparison, MembranePlacement, ConstructContext, MembraneMDReadiness
- VariantAlphaFold (when the pockets workbook exists)
- AlphaFoldChecks, AlphaFoldIssues, Software, Reproducibility, DataDictionary

The full design is in `docs/M4_ALPHAFOLD_MEMBRANE.md`.

**Design decisions:**
- **AlphaFold DB is the default model source.** It is the current release, fetched by accession. Both the current API field names and the names retired in June 2026 are read.
- **The whole job stays in Colab.** AlphaFold2-multimer v3 runs on the Colab GPU through ColabFold (`colabfold_run.py`) for the full-length dimer, and it is resumable. AlphaFold 3 is not used: its weights are not freely available, and AlphaFold Server is outside Colab. AlphaFold2 does not model Zn²⁺ or glycans; both are Milestone 5 blockers in the readiness table.
- **Predicted models are numbered by local alignment to UniProt** (`numbering.align_to_uniprot`), because they have no SIFTS mapping. The critical `numbering_identity` check enforces it.
- **PAE is indexed by `(chain, residue)` key.** The matrix size and chain blocks must match the model. This is enforced by the critical `pae_indexing` check, and the chain-boundary tests are the exit criterion.
- **Superposition uses the confident core** (pLDDT ≥ 70). RMSD is reported over the core and over all common residues, with pLDDT–deviation calibration.
- **The confident-dimer rule** requires all three of: ipTM ≥ 0.8, interface PAE ≤ 5 Å, and Jaccard ≥ 0.5 to the canonical interface. A missing criterion means *uncertain*.
- **Membrane placement uses the sequence topology.** For dimers, the normal is constrained to the C2 axis. The TM–ectodomain PAE decides whether the model *determines* the ectodomain's orientation at all.
- **Membrane-ready inputs are written for the top-ranked model.** A dimer is used only when its call is *confident*. The readiness table is never READY, and the lipid composition is a placeholder recorded as an assumption.

**Changed in earlier stages:**
- `VariantStructureMapping` AlphaFold columns can now be filled (`variant_alphafold_columns`). They are written to the new VariantAlphaFold sheet; the pockets workbook is not rewritten.
- `pyproject.toml`: new optional extra `membrane` (tmfold, py3Dmol).

**Bugs found during development:**
- **`tmfold` alignment.** A semi-global alignment cannot skip *both* an expression tag and the unmodelled start of UniProt, so a tag on a mid-sequence construct was forced onto the preceding residues as mismatches. `align_to_uniprot` and `tmfold` now use a local alignment, with regression tests in both repositories.
- **`tmfold` membrane thickness.** The loop penalty was a fraction of *all* loop residues, so PSMA's ~700-residue ectodomain diluted it. Ties between offsets also resolved to the first offset rather than the thinnest slab, so the fitted slab grew to its 40 Å limit. The penalty is now per residue relative to the TM count, and ties go to the thinnest slab (36 Å for a 24-residue helix). Regression tests were added in both repositories.

### Milestone 4 — fixes after the first Colab run (AlphaFold DB + 3 AlphaFold2-multimer dimer models)

- **Construct context (bug).** Coverage used the SIFTS construct range, which includes expressed but unresolved residues, so full-length constructs counted as an "experimentally observed" membrane anchor. ConstructContext now reports both *construct* coverage (`Covers …`) and *modelled* coverage (`Modelled …` = construct minus missing residues), and the note is based on the modelled residues.
- **Membrane rankings.** Rank-based scoring let a dimer with a poorly predicted TM helix (pLDDT 26.5, 68° tilt) tie with good ones. `membrane_anchor` and `membrane_dimer` now exclude models with TM pLDDT ≤ 50 and use TM helix tilt as a criterion.


## Milestone 5 — model preparation (v0.5.0)

**New output:** `outputs/<ACC>_model_preparation.xlsx` (notebook 04 or `scripts/run_preparation.py`), plus
`preparation_tables/` and `model_preparation/<name>/`. The folder holds the oriented and template-frame PDBs, the protonation microstates, the membrane planes, `provenance.json` and a README. The sheets are:
- PrepSummary, HybridSegments, ResidueSources, Junctions, Repairs, TemplateHetero
- Glycans, SequonObservations, ZnSite, ZnContacts, MetalTreatmentOptions
- Protonation, Microstates, Clashes, HybridMembrane, PrepReadiness, PrepChecks
- PrepIssues, Software, Reproducibility, DataDictionary

The full design is in `docs/M5_MODEL_PREPARATION.md`.

**User decisions:**
- the starting model is a hybrid of the 9S7U ectodomain dimer and the AlphaFold2 membrane anchor;
- the Zn treatment is decided later;
- glycans are modelled as the core GlcNAc where observed;
- protonation is assigned at pH 7.4.

**Design decisions:**
- **Experimental first.** Every residue, ion and water that 9S7U resolves is used as is. AlphaFold2 fills only the membrane anchor (in the global dimer frame), the gaps (fitted on their flanks), missing side-chain atoms and reverted substitutions. Each atom records its source.
- **The N-terminal graft point is searched** to minimise the peptide C–N distance. Junctions and clashes are reported, not fixed: minimisation belongs to Milestone 6.
- **Zn ions are experimental.** Their coordination is checked, and the MD treatment is tabulated but never chosen by the pipeline.
- **Only the core GlcNAc is modelled.** Crystallographic glycans are fragments. Transplants are clash-checked, and a rejected transplant falls through to the next donor.
- **Protonation is written as residue names**, with heavy atoms only. Zn-ligand residues are fixed. Alternatives cover residues within 1 pH unit of their pKa plus the catalytic E424, one residue per microstate.
- **Originals are hashed before and after the run** (critical), and every output appears in the provenance.

### Milestone 5 — fixes after the first Colab run (9S7U + CF:dimer:rank_001)

- **Graft point (bug).** The N-terminal graft point minimised the peptide C–N distance, so it picked overlapping atoms (0.75 Å). That passed, because only bonds above 2.0 Å were flagged. The graft point is now the one closest to the ideal 1.33 Å, and a junction is a WARNING outside 1.1–2.0 Å (`junction_min_cn`, `junction_max_cn`).
- **Zn geometry check.** For each ligand residue, only the closest donor atom is range-checked. The second oxygen of a monodentate Asp (D453 OD1, 2.76 Å) is still listed in ZnContacts, but no longer flagged.
- **pandas warning.** `rank_models` builds the top table from records.


## Milestone 6, part A — bonded Zn parameters (v0.6.0)

**New output:** `outputs/<ACC>_md_setup.xlsx` (notebook 05 part A), `md_tables/` and `md_setup/zinc_mcpb/`. The sheets are:
- AmberTools, SiteGeometry, QMSummary
- FrcmodBond, FrcmodAngl, FrcmodNonb
- SiteResidues, ZnBonds, TleapCheck, ZincChecks
- ZincIssues, Software, Reproducibility, DataDictionary

The full design is in `docs/M6_ZINC_PARAMETERS.md`.

**User decisions:**
- the Zn treatment is the bonded model from MCPB.py;
- the bridging species is a hydroxide, with Glu424 protonated (GCPII QM/MM literature);
- the QM runs in PySCF inside Colab.

**Design decisions:**
- **`qm_bridge` runs MCPB.py's own Gaussian inputs with PySCF**, using Gaussian's B3LYP (`b3lypg`) and an UltraFine grid. It writes the `.fchk`/`.log` fields MCPB.py reads, so MCPB.py still builds the models and fits the parameters.
- **One site is parameterised (chain A) and reused for chain B.**
- **Metal–ligand cut-off 2.5 Å**, from the 9S7U geometry.
- **Frequency scale factor 0.9614.**
- **ff14SB/TIP3P**, for Lipid21 compatibility in part B.
- **The whole metal site must have an integer charge**, not each residue (RESP spreads charge).
- **A water self-test** (frequencies against the literature, plus pymsmt read-back) gates the metal-site QM.

**Changed in Milestone 5:** `preparation.zn_md_treatment` is now `bonded_mcpb` and the new `preparation.zn_bridge` is `hydroxide`. The readiness item "bridging water / hydroxide" passes when a species is chosen and Glu424's default state is consistent with it.


## Milestone 6, part B — membrane system build and minimisation (v0.7.0)

**New output:** `outputs/<ACC>_md_setup_system.xlsx` (notebook 05 part B), `md_system_tables/` and `md_setup/system/` (`system.prmtop`/`.inpcrd`, `system.xml`, `system_minimised.pdb`, smoke-run state). The sheets are:
- SoluteSummary, ZnTemplateMap, GlycanLinks
- Composition, MergeSummary, TleapBuild
- Minimisation, SystemSummary, SystemChecks, MDReadiness
- SystemIssues, Software, Reproducibility, DataDictionary

The full design is in `docs/M6_SYSTEM_BUILD.md`.

**User decision:** proceed with the placeholder membrane (POPC:CHL1 7:3) and core GlcNAc at all observed sites; a CHARMM-GUI build will be compared later. Both are listed as WARNING assumptions in MDReadiness.

**Design decisions:**
- **packmol-memgen packs around the protein only**; the full solute (both Zn sites, hydroxides, glycans, ions) is superposed back and clashing molecules are removed whole.
- **Chain B's Zn site reuses chain A's MCPB.py templates**, matched by ligand residues; part A's residue numbers (tleap's 1…n) are translated to UniProt numbering.
- **Readiness can be READY** for the first time: no FAIL, with the assumptions as WARNINGs.
- **Zn geometry check on the minimised structure** (0.3 Å); the smoke run allows twice that for thermal motion.

**New config:** `md.system` (lipids, ratio, salt, margins, clash cut-off, force fields) and `md.openmm` (cut-off, restraints, tolerances, smoke-run lengths, temperature).

**Notebook 05:** part B sections F–J; setup installs packmol and OpenMM from conda-forge.

### Milestone 6 — Colab install fix (v0.7.1)

- **Bug (first Colab run):** `mamba install ambertools … openmm` failed. Colab's Python is now 3.13, which condacolab pins, and AmberTools has no conda build for it.
- **Fix:** the MD tools no longer go into Colab's Python. Notebook 05 builds a **separate micromamba environment with Python 3.11** (`psma_pipeline.md_env`, ~5–10 min, no runtime restart). The QM self-test, zinc parameters and system build run there through `scripts/run_md_setup.py`; results come back as JSON tables for display. condacolab is no longer used.

### Milestone 6 — QM resume and GPU diagnosis (v0.7.2)

- **The geometry optimisation now saves its geometry after every step** (`*.opt_checkpoint.npz` on Drive). A disconnected Colab run resumes from the last step instead of starting over.
- **The reason PySCF is not on the GPU is reported:** `qm_bridge.gpu_status()`, shown in setup cell 0b ("PySCF on GPU"), the self-test and the QM summary. In the first Colab run the runtime had a GPU, but PySCF ran on the CPU (`gpu: false`).

### Notebook 05 on your own GPU (v0.7.3)

- **Notebook 05 runs on your own machine** (Linux, or Windows via WSL2, with an NVIDIA GPU), in Jupyter or with Colab connected to a local runtime. Guide: `docs/LOCAL_GPU.md`.
- `md_env`:
  - environment in `~/psma_mdenv` (or `PSMA_MDENV`), built once;
  - micromamba for the right platform, or an existing conda/mamba;
  - GPU packages only when `nvidia-smi` finds a card;
  - a clear message on native Windows.
- `colab.local_overrides(DATA_DIR)` points the run at the synced Google Drive folder, so the results land with the Colab ones.

### Colab GPU for PySCF (v0.7.4)

- Setup no longer assumes gpu4pyscf works once pip has installed it. It asks the MD environment whether PySCF can see the GPU. If not, it adds the CUDA libraries as pip wheels (`cupy-cuda12x[ctk]`, `nvidia-*-cu12`) and checks again, then prints the outcome and the reason.
