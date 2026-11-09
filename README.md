# PSMA structural modelling

A reproducible structural-bioinformatics pipeline for **prostate-specific
membrane antigen** (PSMA / GCPII / FOLH1, UniProt **Q04609**). The pipeline
runs from a survey of every experimental structure, through modelling
assessment, to MD system preparation and trajectory analysis.

Q04609 is the default target. Everything target-specific lives in `config/`,
so the core workflow can be reused for another UniProt accession.

## Status

**Milestones 1–5 of 7 are complete and run in Colab; Milestone 6 (bonded Zn parameters; membrane system build and minimisation) is built** (Colab runs pending). Milestone 1: the original survey notebook has been
refactored into an importable package, with configuration, tests, sanity
checks and a reproducibility manifest. See [`docs/CHECKLIST.md`](docs/CHECKLIST.md)
for every component and [`docs/PLAN.md`](docs/PLAN.md) for the full plan.

| Notebook | Purpose | Milestone |
|---|---|---|
| `01_structure_survey.ipynb` | PDB survey, cross-check, assemblies, ligands, validation, lid states, template ranking | **M1 ✔** |
| `02_interface_pocket_validation.ipynb` | part 1: dimer interfaces, crystal contacts, local map-model fit, ranking modes (**M2 ✔**); part 2: pocket fingerprints, ligand clustering, ligand-conditioned ranking, conserved waters, variants (**M3 ✔**) | M2–M3 |
| `03_alphafold_comparison.ipynb` | AlphaFold DB model + AlphaFold2-multimer dimer **run in Colab** (ColabFold); confidence, comparison with experiment, confident-dimer rule, AF rankings; membrane topology, placement and QC; membrane-ready MD inputs ([design](docs/M4_ALPHAFOLD_MEMBRANE.md)) | **M4 ✔** (Colab run pending) |
| `04_model_preparation.ipynb` | hybrid model (9S7U ectodomain dimer + AlphaFold2 membrane anchor), repairs, Zn site + treatment options, core GlcNAc, PROPKA protonation + microstates, membrane-oriented model, provenance ([design](docs/M5_MODEL_PREPARATION.md)) | **M5 ✔** (Colab run pending) |
| `05_md_system_setup.ipynb` | part A: bonded Zn parameters with MCPB.py, QM in PySCF inside Colab ([design](docs/M6_ZINC_PARAMETERS.md)); part B: membrane system build, minimisation, MD readiness | **M6 parts A and B ✔** (Colab runs pending) |
| `06_md_trajectory_analysis.ipynb` | stability, interfaces, Zn geometry, lid dynamics, convergence | M7 |

## Quick start

### Google Colab
Open `notebooks/01_structure_survey.ipynb` in Colab and run the cells in
order. The first cell clones this repository and installs the package.

If the repository is private, clone with a personal access token:
`https://<token>@github.com/victorolatunde23-dev/psma_structural_modelling.git`.

### Local
```bash
git clone https://github.com/victorolatunde23-dev/psma_structural_modelling.git
cd psma_structural_modelling
pip install -e .                 # or: conda env create -f environment.yml
pytest -q                        # offline unit tests
python scripts/run_survey.py     # validation set → sanity gate → full survey
```

Outputs are written to `outputs/`:
- `Q04609_structures.xlsx`: the 14 original sheets, plus SanityChecks, RunIssues, Reproducibility, DataDictionary and others;
- `Q04609_manifest.json`;
- `survey_tables/*.csv` and `*.parquet`.

To reuse a cache from the original notebook, set `paths.cache: /path/to/pdb_cache`.

## Layout

```
config/       default_config.yaml, psma_active_site.yaml, psma_regions.yaml, ligand_classes.yaml
src/psma_pipeline/
  config.py   status.py   cache.py   rcsb.py   pdbe.py   uniprot.py   numbering.py
  assemblies.py   ligands.py   pockets.py   glycans.py   active_site.py   conformations.py
  validation.py   ranking.py   survey.py   sanity.py   manifest.py   excel_output.py
  structure_atoms.py   sasa.py   interfaces.py   local_fit.py   assessment.py   (M2)
  fingerprints.py   ligand_chem.py   waters.py   variants.py   pocket_analysis.py   (M3)
  alphafold.py   colabfold_run.py   membrane.py   (M4 + membrane context)
  model_preparation.py   metal_sites.py   (M5)
  zinc_params.py   qm_bridge.py   system_build.py   (M6)
notebooks/    one notebook per stage (explanation + plots; no calculations)
scripts/      run_survey.py, run_assessment.py, run_pockets.py, run_alphafold.py, run_preparation.py (command-line equivalents of notebooks 01–04)
tests/        pytest suite (offline; gemmi tests skip without gemmi; network tests opt-in)
docs/         PLAN.md, CHECKLIST.md, CHANGES.md, M4_ALPHAFOLD_MEMBRANE.md, M5_MODEL_PREPARATION.md, M6_ZINC_PARAMETERS.md, M6_SYSTEM_BUILD.md, LOCAL_GPU.md
data/ cache/ outputs/   (contents not tracked)
```

## Scientific rules enforced in code

- **Biological assembly vs ASU.** The oligomeric state comes from the biological assembly's generators, never from the number of chains in the asymmetric unit.
- **UniProt numbering only through SIFTS** for experimental structures. Author, label_seq and UniProt numbering are kept in separately labelled columns. Predicted models have no SIFTS mapping, so they are numbered by sequence alignment to UniProt, and the critical `numbering_identity` check enforces it.
- **PAE is indexed by residue key, never by position.** The matrix size and chain blocks must match the model, enforced by the critical `pae_indexing` check.
- **A confident dimer must also be the experimental dimer.** ipTM and interface PAE are not enough: the confident-dimer rule also requires agreement with the canonical interface.
- **Membrane-ready is not MD-ready.** Only the Milestone 6 `MDReadiness` table can say READY, and only when nothing FAILs; assumptions (lipids, glycans) stay listed as WARNING.
- **Originals are never modified.** Milestone 5 hashes every input before and after the run (critical check) and records each step in `provenance.json`.
- **The Zn treatment is the user's choice.** `preparation.zn_md_treatment` must be set before MD; the pipeline only tabulates the options.
- **Configured residues are checked.** Every configured residue is validated against the canonical sequence. Mismatches are dropped and reported.
- **A missing lid is not an open lid.** A poorly modelled lid is reported as *disordered*, never as open.
- **The B-factor ratio is an indirect indicator.** It is not used as a measure of map-model fit.
- **Failures are recorded.** Every failure goes into a `Calculation status` / `Warning` column and the `RunIssues` sheet; none are silent.
- **Originals are never overwritten.** Downloaded files are immutable.
- **Variants are hypotheses.** Variant tables describe where a substitution sits; they never predict its effect.

## Data sources

All are public and cached on first use:

- RCSB PDB Search and Data APIs, and mmCIF files
- PDBe SIFTS and validation APIs
- UniProt REST
