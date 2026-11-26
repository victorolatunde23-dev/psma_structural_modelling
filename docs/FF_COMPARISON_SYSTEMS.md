# Force-field comparison: the three PSMA membrane systems

| | A: Amber (main run) | B: CHARMM36m | C: OPLS4 |
|---|---|---|---|
| Built with | this pipeline (notebooks 04–05): packmol-memgen + tleap | CHARMM-GUI Membrane Builder (job 9494567473) | Maestro: Protein Preparation Workflow, then Desmond System Builder |
| Force field | ff14SB / Lipid21 / GLYCAM_06j-1 / TIP3P + MCPB.py bonded Zn (QM: PySCF B3LYP/6-31G*, empirical force constants, r0 correction) | CHARMM36m / TIP3P; nonbonded Zn²⁺ | OPLS4; Zn with zero-order bonds |
| Zn bridge | hydroxide | water | hydroxide (set by hand) |
| Glycans | 14 core GlcNAc | none (glycan link failed in CHARMM-GUI) | as kept in Maestro preparation |
| Membrane | POPC:CHOL 7:3, 488 lipids per leaflet | POPC:CHL1 7:3, 488 per leaflet | **pure POPC, no cholesterol** (differs from A and B) |
| Salt | 0.15 M NaCl | 0.15 M NaCl | **7 counter-ions only** (build of 2026-11-26; 0.15 M NaCl not added) |
| **Atoms** | **534,300** | **560,138** | **359,771** (rebuilt 2026-11-26; first build 245,820 with a 10 Å buffer) |
| Box (Å) | 171.8 × 171.7 × 206.4 (after equilibration 160.1 × 160.0 × 205.8) | 171.9 × 171.9 × 200.4 | buffer 10 × 10 × 40 Å (box dimensions to record); TIP3P |
| Engine | OpenMM (Colab L4) | OpenMM (Colab) or Desmond (CHARMM-GUI Desmond files, workstation) | Desmond (workstation) |
| Equilibration | CHARMM-GUI six-step protocol (step6.1–6.6), done 2026-11-18, all checks PASS | CHARMM-GUI six-step (OpenMM) / Desmond 6.0–6.2 | Desmond default membrane relaxation |
| Production | 200 ns NPT, 310 K, 4 fs (HMR); running (8.75 ns on 2026-11-26) | not started | not started |

Common to all three: the notebook 04 model (microstate 00, pH 7.4; Glu424 neutral; His377/His553 HID); 310 K; tensionless membrane barostat. The internal crystal waters were not carried over (see CHARMM_GUI_BUILD.md).
