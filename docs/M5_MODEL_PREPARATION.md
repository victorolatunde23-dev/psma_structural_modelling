# Milestone 5: model preparation — design

Checklist items: 9 (repair assessment and rebuilding), 10 (protonation), 11 (binuclear Zn site and MD
metadata), 12 (ligands), 13 (glycans), and the construct part of 15.

Code: `src/psma_pipeline/model_preparation.py`, `src/psma_pipeline/metal_sites.py`.
Notebook: `notebooks/04_model_preparation.ipynb`. Script: `scripts/run_preparation.py`.
Outputs:
- `outputs/<ACC>_model_preparation.xlsx`
- `outputs/preparation_tables/`
- `outputs/model_preparation/<name>/`, containing:
  - the oriented and template-frame PDBs;
  - `microstates/`;
  - the membrane planes;
  - `provenance.json` and `README.md`.

The plan's exit criteria are: **never overwrite originals; a provenance chain; alternative microstates.**

## User decisions (recorded in `config/default_config.yaml: preparation`)

| Decision | Choice | Key |
|---|---|---|
| Starting model | **Hybrid**: 9S7U ectodomain dimer + AlphaFold2 membrane anchor | `template`, `anchor_model` |
| Zn MD treatment | **Decide later**; the readiness table stays FAIL until chosen | `zn_md_treatment` |
| Glycans | **Core GlcNAc where observed** | `glycans.strategy` |
| pH | **7.4** | `protonation.ph` |

## 1. Hybrid model

1. **Template.** The template is 9S7U's biological assembly, built from its operators with the Milestone 2 code, so it is the experimental dimer.
2. **Global dimer fit.** The AlphaFold2 dimer (`CF:dimer:rank_001`) is superposed on the template dimer:
   - both chains are fitted at once, and both chain pairings are tried;
   - the fit uses ectodomain residues ≥ 57 with pLDDT ≥ 70;
   - one rigid transform keeps the predicted TM-helix pair intact.
3. **Per residue.** Each template residue is used as follows:

   | Template residue | What is used |
   |---|---|
   | Matches UniProt, backbone complete | **experimental** |
   | Matches UniProt, side-chain atoms missing | experimental atoms, plus the missing atoms from the AlphaFold2 residue fitted on that residue's common atoms |
   | Differs from UniProt (a substitution) | **reverted to UniProt**: the AlphaFold2 residue fitted on the template backbone (N, CA, C) |
   | Not modelled | AlphaFold2 (see 4) |

4. **AlphaFold2 segments.**
   - **N-terminal membrane anchor** (cytoplasmic tail, TM helix, stalk): placed in the global dimer frame. The graft point is searched over the first `junction_search` template residues, minimising the peptide C–N distance at the junction.
   - **Internal gaps:** locally fitted on up to `local_flank` template residues on each side.
   - **C-terminal tail:** fitted on its left flank.
5. **Junctions.** Every graft point is reported (*Junctions*). A C–N distance above `junction_max_cn` (2.0 Å) is a WARNING. Small strain is resolved by minimisation in Milestone 6.
6. **Hetero groups.**
   - **Kept:** Zn²⁺, Ca²⁺, Cl⁻; waters within 3 Å of a Zn (the catalytic water or hydroxide); and root sugars whose C1 is within 2.0 Å of an Asn ND2.
   - **Removed and listed:** ligands, additives, outer glycan residues, bulk water and non-target polymer chains (*TemplateHetero*).
7. **Clashes.** Heavy-atom pairs below 2.2 Å are listed. Same-residue and adjacent-residue pairs, the glycosidic link, and ions and waters are excluded.

## 2. Binuclear Zn site (`metal_sites.py`)

The Zn coordination is measured on the hybrid:
- each Zn is assigned to Zn1 or Zn2 by the configured ligands (Zn1: H377, D387, D453; Zn2: D387, E425, H553);
- ligand distances are checked against `sanity.zn_ligand_distance_range`;
- the Zn–Zn distance and the bridging water/hydroxide are reported.

The critical check `zn_site_complete` requires two Zn per protomer, with every expected ligand coordinating.

**MD treatment options** (*MetalTreatmentOptions*):
- `nonbonded_12_6_4`;
- `bonded_mcpb` (MCPB.py; needs AmberTools and a QM program);
- `cationic_dummy`.

The pipeline never picks one. The bridging water versus hydroxide is a separate decision, made together with the treatment.

## 3. N-glycans: core GlcNAc where observed

*SequonObservations* counts, per N-X-S/T sequon, the PDB entries with a glycan on it, using the survey's *Glycosylation* sheet. For each sequon observed glycosylated in at least `min_observed_structures` entries:
1. **The template's root sugar is kept** if it has one.
2. **Otherwise it is transplanted** from a donor entry (examples first, then TemplateRanking order, up to `donors_max`). The donor is fitted locally on the backbone of residues u−2 to u+2 plus the Asn side chain.
3. **The transplant is rejected** if any sugar heavy atom comes within 2.0 Å of the protein. The next donor is then tried.

Sequons never observed glycosylated are left unglycosylated. Crystallographic glycans are fragments, so only the first GlcNAc is modelled. The PDB files carry `LINK` records for the C1–ND2 bonds.

## 4. Protonation and microstates

- **PROPKA 3** is run on the protein, at `ph` 7.4. If it is unavailable, default pKa values are used, with a WARNING.
- **Fixed states:**
  - Zn-ligand His take the tautomer that leaves the coordinating N unprotonated (HID when NE2 coordinates);
  - Zn-ligand Asp/Glu are deprotonated;
  - Cys with SG–SG < 2.5 Å are CYX.
- **Other His:** HIP if pKa > pH. Otherwise the tautomer follows the H-bond acceptors within 3.2 Å of ND1 or NE2 (HIE by default).
- **Alternatives** are written for every residue with |pKa − pH| ≤ `window` (1.0) and for `always_alternative` (E424, the catalytic acid/base).
  - Microstate 00 is the default.
  - Each further microstate changes one residue, most ambiguous first, up to `max_microstates`.
- **States are residue names** (AMBER: HID/HIE/HIP, ASH, GLH, LYN, CYX/CYM). Heavy atoms only; hydrogens are added by the Milestone 6 builder.

## 5. Membrane

The hybrid is placed with the Milestone 4 membrane code (`tmfold` score, C2-axis constrained for the dimer), using the membrane topology from notebook 03. The membrane-oriented model has the normal along +z, the mid-plane at z = 0 and the cytoplasm at z < 0.

## 6. Originals, provenance, readiness

- **Originals.** The template mmCIF, the AlphaFold2 PDB and scores, and the donor mmCIFs are hashed before the run and again after it (critical check `originals_unchanged`). Downloads are only ever added to the cache; an existing file is never rewritten.
- **Provenance.** `provenance.json` lists the original hashes and every step: name, tool, time, inputs and outputs with SHA-256, and parameters. Critical check `provenance_complete`: every written file appears as a step output.
- **Readiness.** *PrepReadiness* is never READY in this milestone. The Zn treatment, the bridging water/hydroxide, the lipid composition and all parameters remain open for Milestone 6.

## Critical checks

| Check | Fails when |
|---|---|
| `numbering_identity` | a hybrid chain's sequence or numbering differs from UniProt 1–750 |
| `originals_unchanged` | any original file's hash changed during the run |
| `zn_site_complete` | a protomer lacks two Zn, or an expected ligand does not coordinate |
| `provenance_complete` | a written file is missing from the provenance |
