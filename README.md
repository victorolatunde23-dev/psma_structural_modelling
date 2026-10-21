# PSMA structural modelling

A reproducible structural-bioinformatics pipeline for **prostate-specific
membrane antigen** (PSMA / GCPII / FOLH1, UniProt **Q04609**). The pipeline
runs from a survey of every experimental structure, through modelling
assessment, to MD system preparation and trajectory analysis.

Q04609 is the default target. Everything target-specific lives in `config/`,
so the core workflow can be reused for another UniProt accession.

## Status

**Milestones 1–2 of 7 are complete.** Milestone 1: the original survey notebook has been
refactored into an importable package, with configuration, tests, sanity
checks and a reproducibility manifest. See [`docs/CHECKLIST.md`](docs/CHECKLIST.md)
for every component and [`docs/PLAN.md`](docs/PLAN.md) for the full plan.

| Notebook | Purpose | Milestone |
|---|---|---|
| `01_structure_survey.ipynb` | PDB survey, cross-check, assemblies, ligands, validation, lid states, template ranking | **M1 ✔** |
| `02_interface_pocket_validation.ipynb` | dimer interfaces, crystal contacts, local map-model fit, ranking modes (**M2 ✔**); pocket fingerprints, conserved waters, variants (M3) | M2–M3 |
| `03_alphafold_comparison.ipynb` | AlphaFold / ColabFold monomer and multimer assessment | M4 |
| `04_model_preparation.ipynb` | repairs, protonation, Zn site, ligands, glycans, construct context | M5 |
| `05_md_system_setup.ipynb` | force fields, parameter provenance, solvation, MD readiness | M6 |
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
notebooks/    one notebook per stage (explanation + plots; no calculations)
scripts/      run_survey.py, run_assessment.py (command-line equivalents of notebooks 01 and 02)
tests/        pytest suite (offline; gemmi tests skip without gemmi; network tests opt-in)
docs/         PLAN.md, CHECKLIST.md, CHANGES.md
data/ cache/ outputs/   (contents not tracked)
```

## Scientific rules enforced in code

- **Biological assembly vs ASU.** The oligomeric state comes from the biological assembly's generators, never from the number of chains in the asymmetric unit.
- **UniProt numbering only through SIFTS.** Author, label_seq and UniProt numbering are kept in separately labelled columns.
- **Configured residues are checked.** Every configured residue is validated against the canonical sequence. Mismatches are dropped and reported.
- **A missing lid is not an open lid.** A poorly modelled lid is reported as *disordered*, never as open.
- **The B-factor ratio is an indirect indicator.** It is not used as a measure of map-model fit.
- **Failures are recorded.** Every failure goes into a `Calculation status` / `Warning` column and the `RunIssues` sheet; none are silent.
- **Originals are never overwritten.** Downloaded files are immutable.

## Data sources

All are public and cached on first use:

- RCSB PDB Search and Data APIs, and mmCIF files
- PDBe SIFTS and validation APIs
- UniProt REST
