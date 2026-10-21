# Implementation checklist

Status key:
- ✅ **done and tested**: unit tests pass offline.
- 🟡 **done, Colab test pending**: needs gemmi or network; tests are written but skipped where those are unavailable.
- 🟠 **partial**.
- ⬜ **pending**.

| Brief § | Component | Module | Status |
|---|---|---|---|
| 1 | Package layout, notebooks call modules | `src/psma_pipeline` | ✅ |
| 1 | YAML configuration, Q04609 as default, reusable for another accession | `config.py`, `config/*.yaml` | ✅ |
| 1 | Docstrings, type hints, controlled exceptions, warnings recorded in tables | all M1 modules, `status.py` | ✅ |
| 2 | Caching of every response / download; 404 refresh | `cache.py` | ✅ |
| 2 | Identifier columns (PDB, model, author/label chain, entity, numbering, status, warning) | survey tables | ✅ |
| 2 | NMR `model_policy: all` | `assemblies.py`, `survey.py` | 🟠 falls back to model 1 with a recorded warning; Q04609 has no NMR entries, so this stays deferred |
| 2 | Reproducibility manifest | `manifest.py` | ✅ |
| 2 | Optional-software detection | `manifest.detect_software` | ✅ |
| 19 | Original 14 sheets preserved | `survey.py`, `excel_output.py` | ✅ |
| 19 | Frozen headers, filters, widths, status colours, data dictionary, CSV/Parquet | `excel_output.py` | ✅ |
| 20 | Tests: SIFTS mapping, author/label chains, ranking missing values, lid classification, residue validation | `tests/` | ✅ |
| 20 | Tests: assembly expansion, symmetry dimer counting, contacts, Zn distances, glycosylation, cell-edge contacts | `tests/test_assemblies.py`, `tests/test_pockets.py` | ✅ passed in Colab (gemmi 0.7.5) |
| 20 | Validation-set selection and sanity checks | `sanity.py` | ✅ validation set and full run (90 entries) in Colab; no critical FAIL |
| 21 | Milestone regression: compare workbooks | `excel_output.compare_workbooks` | ✅ M1 vs original: all differences explained (docs/CHANGES.md) |
| 3 | Dimer-interface analysis: assembly from gemmi operators, contacts, typed interactions, BSA, composition, B-factors, outliers, nearby glycans/ligands | `structure_atoms.py`, `sasa.py`, `interfaces.py` | ✅ offline tests; ✅ Colab: validation set and full run (90 entries, 93/93 checks PASS, no issues recorded) |
| 3 | Interface comparison: Jaccard, contact pairs, recovery, interface RMSD, orientation, clustering, noncanonical flag; crystal-lattice contacts | `interfaces.py`, `assessment.py` | ✅ offline; 🟡 real data |
| 3 | Interface conservation | `variants.py` | ✅ variants categorised "dimer interface" in VariantStructureMapping; 🟡 real data |
| 5 | wwPDB validation XML (schema recorded), LocalModelFit, LigandModelFit, EMMapFit, method-specific categories | `validation.py`, `local_fit.py` | ✅ offline (synthetic X-ray and EM XML); 🟡 real reports in Colab |
| 8 | Ten ranking modes with filters, weights, missing policy, explanations; ligand-conditioned ranking | `ranking.py`, `assessment.py`, `pocket_analysis.rank_for_ligand`, config | ✅ offline; 🟡 ligand-conditioned on real data |
| 4 | Pocket fingerprints (binary, weighted, typed), sub-pocket occupancy, similarity, clustering with diagnostics, ligand chemistry | `fingerprints.py`, `ligand_chem.py`, `pocket_analysis.py` | ✅ offline (synthetic geometry, end-to-end with stubbed gemmi); 🟡 real data, RDKit in Colab |
| 14 | Conserved waters (superposition, complete-linkage clustering, Zn bridging, retain/consider/remove) | `waters.py` | ✅ offline; 🟡 real data |
| 6 | Variant mapping (UniProt, optional EBI Proteins API; reference check; RSA, distances, RMSF, sequons, disulfides; hypotheses only) | `variants.py` | ✅ offline; 🟡 real data; AlphaFold columns M4 |
| 7 | AlphaFold / ColabFold import and comparison | `alphafold.py` | ⬜ M4 |
| 9 | Structure-repair assessment and rebuilding | `model_preparation.py` | ⬜ M5 |
| 10 | Protonation and H-bond network | `model_preparation.py` | ⬜ M5 |
| 11 | Binuclear Zn site and MD parameterisation metadata | `metal_sites.py` | ⬜ M5 |
| 12 | Ligand preparation and parameter provenance | `model_preparation.py` | ⬜ M5 |
| 13 | Glycosylation (composition, strategies) | `glycans.py` | 🟠 site detection done; M5 for the rest |
| 15 | Membrane and construct context | `model_preparation.py` | ⬜ M5 |
| 16 | MD-readiness validator | `md_readiness.py` | ⬜ M6 |
| 17 | MD system setup (engine-agnostic, OpenMM) | `md_readiness.py`, `config/md_protocol.yaml` | ⬜ M6 |
| 18 | Trajectory analysis | `trajectory_analysis.py` | ⬜ M7 |
