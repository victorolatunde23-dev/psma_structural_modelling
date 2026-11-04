# Milestone 6, part B: membrane system build, minimisation and MD readiness

Code: `src/psma_pipeline/system_build.py`.
Notebook: `notebooks/05_md_system_setup.ipynb` (part B, sections F–J).
Inputs:
- the Milestone 5 default microstate (membrane-oriented hybrid dimer);
- the part A MCPB.py outputs in `outputs/md_setup/zinc_mcpb/`.

Outputs:
- `outputs/<ACC>_md_setup_system.xlsx` and `outputs/md_system_tables/`;
- `outputs/md_setup/system/`, containing:
  - `system.prmtop` / `system.inpcrd` (Amber), `system.xml` (OpenMM);
  - `system_minimised.pdb` and `system_smoke_state.xml` (positions and velocities after the smoke run);
  - `build.in`, the logs and `provenance.json`.

## Decisions

| Decision | Choice | Basis |
|---|---|---|
| Lipids | **POPC : CHL1 7 : 3**, Lipid21 | **Assumption** (placeholder). The user will compare with a CHARMM-GUI build later. |
| Glycans | Core GlcNAc only, GLYCAM06j (`0YB` linked to `NLN`), at the sites observed glycosylated (including 153 and 336) | **Assumption**: full glycoforms are not modelled |
| Water / salt | TIP3P, 0.15 M NaCl, neutralised | Physiological; TIP3P is the water of the part A parameters |
| Packing | packmol-memgen, protein only, `--preoriented --notprotonate`, 15 Å lipid margin, 17.5 Å water layer | packmol-memgen never touches the solute's ligands or protonation; the solute is put back afterwards |
| Clash removal | Whole molecules (lipid, water, ion) with a heavy atom within 1.6 Å of the solute | A partial lipid would break tleap |
| Second zinc site | Chain B reuses chain A's MCPB.py templates. Each Zn is matched to a template by its ligand residues (not by order). | The two sites are chemically identical |
| Minimisation | 10 kcal/mol/Å² restraints on protein heavy atoms, then none; tolerance 10 kJ/mol/nm | |
| Smoke run | 20 ps NVT heating to 310 K + 20 ps NPT, `MonteCarloMembraneBarostat` (xy isotropic, z free, zero surface tension), restraints kept | Proves the system is stable enough to start equilibration. GPU only. |

## How the solute is built

The prepared model is heavy-atom only; tleap adds hydrogens from the templates.

1. **Residue numbering.** tleap renumbered the part A site 1…n. The site map translates those numbers back to UniProt numbers through chain A's residue order (a different residue count stops the build: re-run part A).
2. **Protein.** Every residue on both chains gets its MCPB.py name (for example `HD1` for a Zn-bound His) where part A renamed it.
3. **Zn.** Each Zn ion gets the template whose ligand residues overlap most with its own (atoms within 2.6 Å). If no template overlaps, the build stops.
4. **Hydroxide.** Each chain's bridging water becomes the MCPB.py hydroxide residue, with its H placed as in part A.
5. **Glycans.** `NAG` becomes `0YB` (C7→C2N, O7→O2N, C8→CME), and the Asn it is attached to becomes `NLN`.
6. **Bonds.** Written as tleap `bond` commands:
   - every bond from part A's tleap input, for both chains (the Zn–ligand bonds and the peptide links of renamed residues);
   - disulfides (CYX SG–SG < 2.5 Å);
   - ND2–C1 glycan links.

## Checks (`SystemChecks`)

| Check | Critical | Fails when |
|---|---|---|
| `originals_unchanged` | yes | the prepared model or a part A file changed |
| `tleap_build` | yes | tleap errors, missing parameters or unknown residues |
| `neutral` | yes | the net charge after neutralising is not 0 |
| `minimisation_finite` | yes | any energy is NaN / infinite (NOT RUN without OpenMM) |
| `minimised_forces` | yes | the largest force after unrestrained minimisation exceeds 1000 kJ/mol/nm |
| `zn_geometry_kept` | yes | a Zn–ligand distance moved by more than 0.3 Å during minimisation |
| `lipid_ratio` | no | the cholesterol fraction after clash removal is more than 0.05 off the requested one |
| `smoke_run` | no | temperature more than 15 K off target, a NaN, or a Zn–ligand change above 0.6 Å (NOT RUN without a GPU) |

## MD readiness (`MDReadiness`)

OVERALL is **READY** only when no item FAILs or is NOT RUN. Exception: the smoke run needs a GPU, so when it is NOT RUN it counts as a WARNING. The lipid and glycan assumptions are always WARNING rows, so they stay visible in every report.

## Comparing with a CHARMM-GUI build

These are measured the same way on both builds:
- composition and box size;
- minimised Zn–ligand distances;
- in equilibration: area per lipid, bilayer thickness, protein tilt and insertion depth.

CHARMM-GUI uses CHARMM36m and cannot carry the MCPB.py zinc model. Differences in the zinc site are therefore expected; they are part of what the comparison tests.

## Validation

- **Offline (tests).** Fake packmol-memgen and tleap executables are used. The fake tleap checks that every `bond` command names an atom that exists in the residue it addresses. The tests also cover:
  - the site map and its translation to UniProt numbering;
  - Zn template matching on both chains;
  - the GLYCAM renaming;
  - molecule grouping without TER records;
  - superposition and whole-molecule clash removal;
  - checks and readiness logic.
- **Not tested locally:** OpenMM (`run_openmm`) runs for the first time in Colab.
