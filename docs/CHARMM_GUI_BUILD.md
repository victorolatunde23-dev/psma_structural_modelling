# CHARMM-GUI comparison build (Membrane Builder)

**CHARMM-GUI job ID: 9494567473** (built 2026-11-17). Reopen the job from charmm-gui.org with this ID to download `charmm-gui.tgz` again.

## Input
- `charmm_gui_input.pdb`: notebook 04 model (`microstate_00_default.pdb`), heavy atoms only.
  - NAG sugars and their LINK records removed (4 of the 18 NAG were not attached: links to Asn153/Asn336 of 3.1–3.6 Å).
  - Amber protonation names replaced by standard names (HID/HIE → HIS, GLH → GLU, ASH → ASP, LYN → LYS).
  - Kept: 4 Zn, 2 Ca, 2 Cl, 4 crystal waters (including the Zn-bridging water).
- Segments: PROA, PROB, Zn/Ca/Cl hetero segments, WATA/WATB.

## Protonation patches (from notebook 04 `Protonation.csv`, pH 7.4)
| | PROA | PROB |
|---|---|---|
| GLUP | 424, 433 | 424, 433, 444 |
| ASPP | 520 | 520 |
| LSN | 213 | 213 |
| HSE | 6, 56, 82, 112, 124, 295, 345, 347, 396, 475, 573, 618, 689 | same |
| HSD | 377, 553, 697 | same |

## Known differences from the Amber (notebook 05) system
- **No glycans.** CHARMM-GUI's N-glycosylation (bDGlcNAc on Asn 76, 121, 140, 195, 459, 476, 638) gave a non-integer total charge (−0.16 per Asn–GlcNAc link, missing angle parameters at each link). Reading the NAG from the PDB treated them as ligands (O6 undefined). Built without the 14 core GlcNAc.
- **Zn site:** CHARMM36m nonbonded Zn²⁺. The bridging species is a water, not a hydroxide. Glu424 is GLUP, as in Amber.
- **Ions:** NaCl 0.15 M, neutralised (CHARMM-GUI estimate ~401 Na⁺ / 403 Cl⁻; Amber: 356 / 353).

## System
| | CHARMM-GUI | Amber (notebook 05) |
|---|---|---|
| Orientation | PPM 2.0 | packmol-memgen, pre-oriented |
| Atoms | 560,138 | 534,300 |
| Box (Å) | 171.9 × 171.9 × 200.4 | 171.8 × 171.7 × 206.4 |
| Lipids per leaflet | 341 POPC + 147 CHL1 (488) | 341 POPC + 147 CHL (488) |
| Waters | 140,878 | 132,164 |
| Water thickness | 17.5 Å | 17.5 Å |
| Lipid checks | no protein-surface or ring penetration | 0 cholesterol ring piercings |

## Input generation settings
- Force field CHARMM36m; TIP3P; hydrogen mass repartitioning on.
- Outputs: OpenMM (+ AMBER).
- NPT; 310 K; PME grid automatic.

## Equilibration protocol (shared)
Notebook 06 (v0.7.24) equilibrates the Amber system with CHARMM-GUI's restraint protocol: backbone, side chains, lipid-head z, lipid dihedrals, water out of the bilayer core, and ions in step 1, with E = k·d². When this job's archive is unpacked in `outputs/charmm_gui/` on Drive, the exact values are read from its `openmm/step6.N_equilibration.inp` files.
