# Milestone 4 and membrane context: design

This covers checklist item 7 (AlphaFold / ColabFold import and comparison, Milestone 4) and
the membrane part of item 15 (membrane and construct context), with a bridge to
membrane MD (Milestones 5–6).

The project brief was not available when this was written. It follows the plan's Milestone 4
scope and exit criteria (PLAN.md §6), the repository's rules (nothing fails silently, original
files are never overwritten, every assumption lives in `config/`, every column is in the data
dictionary), and the additions listed in §6 below.

Code: `src/psma_pipeline/alphafold.py`, `src/psma_pipeline/membrane.py`.
Notebook: `notebooks/03_alphafold_comparison.ipynb`. AlphaFold2 runner: `src/psma_pipeline/colabfold_run.py`. Script: `scripts/run_alphafold.py`.
Workbook: `outputs/<ACC>_alphafold_membrane.xlsx`.

---

## 1. Model sources (everything runs in Colab)

| Source | How it is obtained | Confidence data |
|---|---|---|
| **AlphaFold DB** (current release; v6 at the time of writing) — monomer | `GET /api/prediction/{accession}`. The entry whose sequence equals the UniProt canonical sequence is preferred. Both the current field names (`modelEntityId`, `sequenceStart`, `sequence`, …) and the names retired in June 2026 (`entryId`, `uniprotStart`, `uniprotSequence`, …) are accepted. | pLDDT (B-factor column) and PAE (`paeDocUrl`) |
| **AlphaFold2-multimer v3, run in Colab** (ColabFold) — full-length dimer by default | `colabfold_run.run_colabfold_jobs`: installs ColabFold into the session, builds the MSA with the ColabFold MMseqs2 server, and runs `colabfold_batch` on the Colab GPU. Jobs are configured in `alphafold.colabfold.jobs` (dimer on; full-length monomer and 44–750 ectodomain dimer available). | `*_scores_rank_NNN_*.json`: `plddt`, `pae`, `ptm`, `iptm` |
| Other ColabFold results / any PDB or mmCIF model | Added from disk or uploaded (optional) | pLDDT if the B-factor column holds it (declared by the user) |

**Why AlphaFold2 and not AlphaFold 3.** The whole job stays in Colab. AlphaFold 3 weights are not freely downloadable, and AlphaFold Server is a separate web service. AlphaFold2-multimer runs on a Colab GPU and predicts the full-length dimer, including both TM helices, which the C2-constrained membrane placement needs.

**What that costs.** AlphaFold2 does not model Zn²⁺ or glycans. Both were going to be added in Milestone 5 anyway, and both are listed as blockers in the readiness table.

**Running it.**
- The defaults are 3 of the 5 AlphaFold2 models, 6 recycles, no templates and no Amber relaxation. Relaxation is structure preparation, which belongs to Milestone 5.
- On a T4, expect roughly 15–40 min per dimer model.
- Results are written to the cache (Google Drive in Colab), so a disconnected run resumes. Finished jobs are skipped, and the ColabFold command line is recorded.
- Result folders are read in place, never copied.

## 2. Numbering and PAE indexing (exit criterion)

* **Numbering.** Every model chain is aligned to the UniProt canonical sequence:
  - identical sequences and exact substrings (for example the 44–750 ectodomain dimer) map
    directly;
  - anything else uses a local alignment, which skips tags and tolerates mutations and gaps.

  All downstream tables use UniProt numbering, the same as the survey. Chains with less than
  95 % identity on aligned residues are flagged. A chain below 50 % identity is not treated as
  the target.
* **PAE.** AlphaFold2 PAE is residue-indexed, with chains concatenated in input order. Rows are
  mapped to `(chain, UniProt number)` from the model's own residue order, and residues without a
  UniProt mapping (tags) are dropped.
* **Critical check `pae_indexing`.** It fails when:
  - the matrix is not square;
  - its size disagrees with the model's polymer residues;
  - a chain block is not contiguous, or not in UniProt order;
  - the diagonal is not near its minimum (a sign of a misaligned matrix).

## 3. Comparison with experimental structures

For each AlphaFold model chain, compared against the reference experimental chains (the top
`alphafold.compare.top_n` entries of the survey `TemplateRanking`, plus the lid calibration
structures):

| Measurement | Notes |
|---|---|
| Global CA RMSD | Over all common residues, **and** over the confident core (pLDDT ≥ 70), after Kabsch superposition on that core |
| TM-score | Via `tmtools` when installed; otherwise `NOT RUN` |
| GDT-TS-like fraction | Fraction of residues within 1, 2, 4 and 8 Å after the core fit |
| Per-region RMSD | Configured domains and flexible regions, after the global fit |
| Active-site geometry | Local superposition on the active-site CA atoms; side-chain heavy-atom RMSD of the Zn ligands and the other literature residues |
| pLDDT calibration | Spearman correlation between pLDDT and per-residue deviation. A good model has negative correlation: low pLDDT where it deviates. |
| Entrance lid | The AlphaFold model is classified with the existing three-vote classifier, alongside the calibration structures |

## 4. Multimers and the confident-dimer rule (exit criterion)

For multimer models (AlphaFold2-multimer), every target–target chain pair is
analysed with the existing interface code (`interfaces.analyse_interface`):
- contacts and typed interactions;
- buried surface area;
- Jaccard similarity to the **canonical interface** from the modelling-assessment workbook.

Interface PAE is the mean of the PAE over interface residue pairs, in both directions.

A dimer is called **confident** only when **all three** of these hold (`alphafold.dimer`):
- ipTM ≥ 0.8;
- mean interface PAE ≤ 5 Å;
- residue Jaccard to the canonical interface ≥ 0.5.

It is **low confidence** when ipTM < 0.6 or interface PAE > 10 Å. Everything else is
**uncertain**. The rule's components are reported separately, so the call is auditable.
Without an assessment workbook, the canonical-interface condition is `NOT RUN` and the call is
at most `uncertain`. Confidence without agreement with experiment is not enough.

## 5. Membrane context (item 15, membrane part)

PSMA is a type II membrane protein:
- a short cytoplasmic tail (1–19);
- one signal-anchor transmembrane helix (20–43);
- a large extracellular ectodomain.

Every experimental structure is an ectodomain construct, so the membrane anchor exists only in
predicted models.

1. **Topology.** Topology comes from the UniProt features (Topological domain, Transmembrane,
   Intramembrane) and, optionally, from DeepTMHMM (BioLib, through the `tmfold` package). The two
   are compared residue by residue. `membrane.topology_source` chooses `uniprot`, `deeptmhmm` or
   `consensus` (residues where both agree; otherwise UniProt).
2. **Placement.** The membrane slab is fitted to the model with `tmfold.geometry.fit_membrane`:
   - TM residues should be inside the slab;
   - cytoplasmic and extracellular residues should be on their own side;
   - a mild prior keeps the normal close to the TM helix axis.

   **For homodimers, the membrane normal is additionally constrained to the dimer's two-fold (C2)
   axis.** A symmetric homodimer anchored in a flat membrane must have its symmetry axis
   parallel to the membrane normal. The angle between the TM-fitted normal and the C2 axis is
   reported. Above `membrane.c2_max_angle_deg` it is a WARNING, because the model's
   TM–ectodomain geometry is then physically inconsistent.
3. **Membrane QC (`MembranePlacement`).** For each model:
   - TM fraction inside the slab and loop-side agreement;
   - TM helix tilt and fitted thickness;
   - **PAE between the TM helix and the ectodomain**: when this is high, the model does not
     determine how the ectodomain sits on the membrane;
   - **ectodomain clearance**: lowest ectodomain heavy atom above the outer plane, and the count
     of ectodomain atoms inside the slab;
   - **glycosylation sequons on the wrong side** (N-glycans must be extracellular);
   - active-site height above the membrane;
   - C2-axis angle (dimers).

   An overall `Orientation reliability` call (`supported` / `weak` / `unsupported`) combines
   these.
4. **Construct context (`ConstructContext`).** Each experimental chain's UniProt range is set
   against the topology: which regions are present, the fraction of the full-length protein,
   and what a full-length membrane model would need from prediction.

## 6. Bridge to membrane MD (membrane-ready inputs, not MD-ready)

For the model chosen by the `membrane_anchor` (monomer) or `membrane_dimer` ranking, the stage
writes `outputs/md_membrane_inputs/<model>/`:

* `model_membrane_oriented.pdb`: the membrane normal along +z, the mid-plane at z = 0, and the
  cytoplasm at z < 0. This orientation is accepted directly by CHARMM-GUI Membrane Builder ("use
  PDB orientation") and by `packmol-memgen --preoriented`.
* `membrane_planes_dum.pdb`: OPM-style dummy atoms on both planes, for visual checking only.
* `packmol_memgen_command.sh`: lipid composition, salt and margins from `membrane.builder`.
* `charmm_gui_settings.json`: the same settings in a form that maps onto the CHARMM-GUI fields.
* `MembraneMDReadiness`: one row per blocker. **The overall status is never READY in this
  milestone.** Detected blockers include:
  - no Zn²⁺ in the model (two per protomer required, so the Zn treatment must be chosen in M5);
  - glycans absent on the N-glycosylation sites;
  - protonation states not assigned;
  - low-pLDDT segments that need a decision (trim, keep or remodel);
  - termini capping not decided;
  - the lipid composition marked as an assumption to confirm.

  This follows the plan's M6 rule: no READY status with any critical FAIL.

The default lipid composition is a **placeholder** (symmetric POPC:CHOL 7:3). It is recorded as
an assumption in the manifest and in the readiness table. Choose a composition appropriate to
your system, for example an asymmetric plasma membrane, before building.

## 7. AlphaFold rankings

`AFRankings` reuses `ranking.rank_templates` (the same engine, filters, missing-value policy and
explanations as the template rankings), with the model ID as the key. Modes, in
`alphafold.ranking.modes`:

| Mode | Ranks models by |
|---|---|
| `ectodomain_template` | ectodomain pLDDT, core RMSD to the best experimental template, TM-score |
| `active_site` | active-site pLDDT, Zn-ligand side-chain RMSD |
| `dimer` | ipTM, interface PAE, canonical-interface Jaccard (multimers only) |
| `membrane_anchor` | TM pLDDT, TM-in-slab, TM–ectodomain PAE (lower is better), ectodomain clearance |
| `membrane_dimer` | `membrane_anchor` criteria plus the C2-axis angle |

## 8. Other outputs

* `AFResidueConfidence`: per model, chain and UniProt residue:
  - pLDDT and confidence band;
  - region labels;
  - depth relative to the membrane;
  - whether any experimental chain models the residue.
* `AFRegionConfidence`: mean and minimum pLDDT and the intra-region PAE per region, plus
  `AFDomainPAE`, the mean PAE between every pair of domains.
* The Milestone 3 `VariantStructureMapping` placeholders `AlphaFold pLDDT` / `AlphaFold status`
  are filled by `alphafold.variant_alphafold_columns`.
* `AlphaFoldChecks`: sanity checks, with `pae_indexing` and `numbering_identity` critical.

## 9. Improvements over a plain AlphaFold comparison

1. The complete prediction runs in Colab: AlphaFold DB plus an AlphaFold2-multimer dimer, run on
   the Colab GPU and resumable. PAE indexing and chain blocks are enforced by a critical check.
2. A core-fitted RMSD plus pLDDT-calibration statistics, so a confident but wrong region is
   visible.
3. The confident-dimer rule requires agreement with the experimental canonical interface, not
   only ipTM.
4. Membrane placement uses the sequence topology and, for dimers, the C2 symmetry constraint. It
   reports whether the model even determines the ectodomain's orientation (TM–ectodomain PAE).
5. Construct context shows explicitly that the membrane anchor is prediction-only.
6. Membrane-builder inputs come with a blocker list that prevents premature MD.
