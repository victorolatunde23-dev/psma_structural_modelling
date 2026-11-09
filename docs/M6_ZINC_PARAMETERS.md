# Milestone 6, part A: bonded parameters for the binuclear Zn site

Code: `src/psma_pipeline/zinc_params.py`, `src/psma_pipeline/qm_bridge.py`.
Notebook: `notebooks/05_md_system_setup.ipynb` (part A). The stages run in a separate Python 3.11 environment (`psma_pipeline.md_env`, `scripts/run_md_setup.py`), because AmberTools has no build for Colab's Python 3.13.
Outputs:
- `outputs/<ACC>_md_setup.xlsx`
- `outputs/md_tables/`
- `outputs/md_setup/zinc_mcpb/`, containing:
  - the MCPB.py inputs and outputs and the PySCF result caches;
  - the logs;
  - `provenance.json`.

## Decisions

| Decision | Choice | Basis |
|---|---|---|
| Zn MD treatment | Bonded model, MCPB.py | User decision: keeps the coordination geometry |
| Bridging species | Hydroxide; Glu424 protonated (GLH) | QM/MM of GCPII: "a hydroxide anion, not a water molecule bridges the two zinc ions", with the proton most likely on Glu424 (Plechanovová *et al.* 2011, *J Med Chem* 54:7535). This agrees with the PROPKA default state of Glu424 in Milestone 5. |
| QM program | PySCF (Colab, GPU via gpu4pyscf when supported) | User decision: stay in Colab. MCPB.py reads only Gaussian/GAMESS output, so `qm_bridge` writes those formats. |
| QM level | B3LYP/6-31G*, Gaussian's B3LYP (VWN3, PySCF `b3lypg`), UltraFine-equivalent grid (99, 590) | MCPB.py's default level. This reproduces what MCPB.py would get from Gaussian. |
| Protein force field / water | ff14SB / TIP3P | The validated combination with Lipid21 and GLYCAM06j for the membrane system (part B) |
| Metal–ligand cut-off | 2.5 Å (MCPB default 2.8) | From the 9S7U site in Milestone 5: Zn2–E425 OE1 2.45 Å (bidentate, included); D453 OD1 2.76 Å (second O of a monodentate Asp, excluded); an extra water at 2.52 Å from Zn2 (weak, excluded, kept as ordinary water) |
| Small-model optimisation | **Hydrogens only**; heavy atoms held at the 9S7U positions (`md.mcpb.qm_opt: hydrogens`) | User decision after the first Colab run. A full gas-phase optimisation (65 steps, not converged) moved the carboxylates without the protein around them: D453 became bidentate on Zn1 (2.76 → 2.22 Å) and E425 OE1 left Zn2 (2.45 → 2.73 Å). MCPB.py would then take those lengths as r0 and pull the MD site away from the experiment. With heavy atoms held, r0 and θ0 are the crystal values. The QM still gives the Seminario force constants (from the Hessian at the experimental geometry, which is not a QM minimum, so some imaginary frequencies are expected and reported) and the RESP charges. |
| Frequency scale factor | 0.9614 | B3LYP/6-31G* (Scott & Radom 1996). MCPB.py's default is 1.0. |
| Ion parameter set | CM (12-6, TIP3P) | For the Zn van der Waals terms and the MK radius |
| Sites | Chain A parameterised; chain B reuses the residue templates and bonds | The two sites are chemically identical (symmetric dimer). Putting both in one QM cluster would double the cost for no information. |

## Workflow

1. **Site PDB.** Chain A of the default microstate supplies:
   - the protein, with hydrogens added by tleap using ff14SB templates, so HID/HIE/GLH/… names are respected;
   - its two Zn;
   - the bridging water (O within 2.6 Å of both Zn), converted to residue `OHX`, with H placed 0.97 Å from O pointing away from the Zn–Zn midpoint.

   `OHX.mol2` uses GAFF types `oh`/`ho`, and `OHX.frcmod` comes from parmchk2. The starting charges are placeholders, because RESP refits them.
2. **MCPB.py step 1** builds the small model (side chains, for Seminario) and the large model (with backbone, for RESP), and writes `*_small_opt.com`, `*_small_fc.com` and `*_large_mk.com`.
3. **QM (`qm_bridge`)** parses MCPB.py's own `.com` files: route, charge, multiplicity, coordinates, frozen flags and `ReadRadii`. It then runs:
   - **opt:** geomeTRIC optimisation; convergence is critical;
   - **freq:** analytic Hessian at the optimised geometry; imaginary frequencies below −50 cm⁻¹ are flagged;
   - **MK:** SCF, then the electrostatic potential on Merz–Kollman points. These are four shells at 1.4/1.6/1.8/2.0 × the MK radii (H 1.20, C/N 1.50, O 1.40, S 1.75, Zn from `ReadRadii`), at `IOp(6/42=n)` points per Å², on Fibonacci spheres rather than Gaussian's own point set (RESP is insensitive to the choice).

   It writes `*_small_opt.fchk` (`Current cartesian coordinates` in Bohr and the lower-triangle `Cartesian Force Constants` in Hartree/Bohr², 5E16.8) and `*_large_mk.log` (the `Electrostatic Properties Using The SCF Density` and `Electrostatic Properties (Atomic Units)` sections, with coordinates in columns 33–62). These are the fields and layouts read by `pymsmt.mol.gauio` (`get_crds_from_fchk`, `get_matrix_from_fchk`, `get_esp_from_gau`). Every QM result is cached (`*.npz`), so an interrupted run resumes.
4. **MCPB.py steps 2–4:** Seminario force constants (`*_mcpbpy.frcmod`), RESP charges (renamed residue `.mol2` templates) and the tleap input (`*_tleap.in`, with Zn–ligand bond commands).
5. **Validation build.** tleap loads the parameterised chain (MCPB.py's input without solvation), runs `check` and `charge`, and saves a prmtop.

## Checks (`ZincChecks`)

| Check | Critical | Fails when |
|---|---|---|
| `originals_unchanged` | yes | the prepared model changed during the run |
| `qm_converged` | yes | the small-model optimisation did not converge |
| `tleap_build` | yes | tleap reports errors or missing parameters |
| `site_charge_integer` | yes | the RESP charges of the metal-site residues do not sum to an integer. RESP spreads charge over the Zn ions and their ligands, so only the whole site is integer, not each residue. |
| `zn_bonds_parameterised` | yes | a Zn bond in the tleap input has no r0/k |
| `qm_minimum` | no | imaginary frequencies below −50 cm⁻¹ |
| `zn_r0_vs_structure` | no | a QM equilibrium length differs from the experimental distance by more than 0.15 Å |
| `chain_total_charge_integer` | no | the parameterised chain's total charge is not an integer |

## Validation of the bridge

- **Offline (tests):**
  - the writers reproduce MCPB.py's parsing rules (fixed-count blocks, column positions), checked by round trips;
  - in development, the files were also read back with the pymsmt parsers themselves;
  - the whole orchestration runs against fake AmberTools executables.
- **In Colab (notebook section B), before the metal-site QM:** water at B3LYP/6-31G* through the full bridge. Its frequencies must match the literature (1713/3727/3849 cm⁻¹) within 25 cm⁻¹, and the written files must be read back by AmberTools' installed pymsmt.
