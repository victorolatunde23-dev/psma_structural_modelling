"""Sanity checks: each must flag the suspicious case it is designed for."""

from __future__ import annotations

import pandas as pd

from psma_pipeline.sanity import (check_active_site_sequence, check_assembly_copies, check_lid_calibration,
                                  check_open_lid_modelled, check_zn_count, check_zn_distances,
                                  critical_failures, run_sanity_checks, select_validation_set)

import fixtures as F


def statuses(rows):
    return [r["Status"] for r in rows]


def test_assembly_copies_disagreement_fails():
    s = pd.DataFrame([
        {"PDB ID": "A", "Assembly copies source": "gemmi generators", "Polymer entities": 1,
         "Target copies in assembly": 2, "RCSB assembly chain count": 2},
        {"PDB ID": "B", "Assembly copies source": "gemmi generators", "Polymer entities": 1,
         "Target copies in assembly": 1, "RCSB assembly chain count": 2},
        {"PDB ID": "C", "Assembly copies source": "gemmi generators", "Polymer entities": 3,
         "Target copies in assembly": 2, "RCSB assembly chain count": 6},
    ])
    assert statuses(check_assembly_copies(s)) == ["PASS", "FAIL", "NOT APPLICABLE"]


def test_unexplained_active_site_mismatch_is_a_numbering_failure():
    active = pd.DataFrame([
        {"PDB ID": "M", "Chain": "A", "UniProt residue": "E424", "Modelled": True, "Matches UniProt": False,
         "Residue in model": "ALA424"},
        {"PDB ID": "N", "Chain": "A", "UniProt residue": "E424", "Modelled": True, "Matches UniProt": False,
         "Residue in model": "LYS430"},
    ])
    chains = pd.DataFrame([{"PDB ID": "M", "Chain (auth)": "A", "Substitutions vs UniProt": "E424A"},
                           {"PDB ID": "N", "Chain (auth)": "A", "Substitutions vs UniProt": ""}])
    assert statuses(check_active_site_sequence(active, chains)) == ["WARNING", "FAIL"]


def test_lid_calibration_outcomes():
    lid = pd.DataFrame([{"PDB ID": "3BHX", "Chain": "A", "Lid state": "open"},
                        {"PDB ID": "2XEG", "Chain": "A", "Lid state": "open"},
                        {"PDB ID": "2OOT", "Chain": "A", "Lid state": "disordered (not modelled)"}])
    out = check_lid_calibration(lid, {"3BHX": "closed", "2XEG": "open", "2OOT": "open", "9XXX": "open"})
    # opposite state = classifier wrong (FAIL); unmodelled lid = unusable calibration (WARNING)
    assert statuses(out) == ["FAIL", "PASS", "WARNING", "NOT RUN"]


def test_open_lid_with_missing_residues_fails():
    lid = pd.DataFrame([{"PDB ID": "X", "Chain": "A", "Lid state": "open", "Lid residues modelled": "3/8"}])
    assert statuses(check_open_lid_modelled(lid, 6)) == ["FAIL"]


def test_zn_distance_outside_range_is_flagged():
    b = pd.DataFrame([
        {"PDB ID": "A", "Component": "ZN", "Chain": "A", "Residue no.": "801",
         "Coordination (atom: Å)": "H377/A NE2: 2.05; D387/A OD1: 2.10; HOH5 O: 2.15"},
        {"PDB ID": "B", "Component": "ZN", "Chain": "A", "Residue no.": "801",
         "Coordination (atom: Å)": "H377/A NE2: 2.75"},
    ])
    assert statuses(check_zn_distances(b, (1.8, 2.6))) == ["PASS", "WARNING"]


def test_zn_check_ignores_carbon_atoms_inside_the_cutoff():
    # Asp CG of a bidentate carboxylate sits ~2.6-2.8 Å from Zn but is not a ligand
    b = pd.DataFrame([{"PDB ID": "A", "Component": "ZN", "Chain": "A", "Residue no.": "801",
                       "Coordination (atom: Å)": "D453/A OD1: 2.05; D453/A CG: 2.70; H553/A NE2: 2.10",
                       "Donor atoms (N/O/S: Å)": "D453/A OD1: 2.05; H553/A NE2: 2.10"}])
    assert statuses(check_zn_distances(b, (1.8, 2.6))) == ["PASS"]


def test_zn_count_per_protomer():
    s = pd.DataFrame([{"PDB ID": "A", "Zn assembly": 4, "Target copies in assembly": 2},
                      {"PDB ID": "B", "Zn assembly": 3, "Target copies in assembly": 2},
                      {"PDB ID": "C", "Zn assembly": 0, "Target copies in assembly": 2}])
    assert statuses(check_zn_count(s, 2)) == ["PASS", "WARNING", "PASS"]


def test_numbering_offset_is_a_critical_failure():
    from psma_pipeline.sanity import check_sifts_integrity
    chains = pd.DataFrame([
        {"PDB ID": "OK", "Chain (auth)": "A", "UniProt range (SIFTS)": "44-750", "UniProt coverage": 0.94,
         "Substitutions vs UniProt": "E424A", "Internal gaps / insertions": ""},
        {"PDB ID": "OFF", "Chain (auth)": "A", "UniProt range (SIFTS)": "44-143", "UniProt coverage": 0.13,
         "Substitutions vs UniProt": ", ".join(f"A{n}G" for n in range(44, 84)), "Internal gaps / insertions": ""},
        {"PDB ID": "NONE", "Chain (auth)": "A", "UniProt range (SIFTS)": "no SIFTS mapping",
         "UniProt coverage": None, "Substitutions vs UniProt": "", "Internal gaps / insertions": ""},
    ])
    assert statuses(check_sifts_integrity(chains)) == ["PASS", "FAIL", "WARNING"]


def test_full_sanity_table_on_offline_survey(offline_survey, cfg):
    df = run_sanity_checks(offline_survey.tables, cfg)
    assert {"Check", "Critical", "Status", "PDB ID", "Chain", "Detail"} == set(df.columns)
    # 3CCC has no SIFTS mapping: reported, but not a critical failure (nothing is mis-numbered)
    rows = df[(df["Check"] == "sifts_segment_integrity") & (df["PDB ID"] == "3CCC")]
    assert list(rows["Status"]) == ["WARNING"]
    assert critical_failures(df).empty


def test_validation_set_selection(cfg):
    sel = select_validation_set(F.metadata(), cfg).set_index("Category")
    assert sel.loc["ASU monomer, dimer assembly", "PDB ID"] == "2BBB"
    assert sel.loc["high-resolution dimer (2 chains in ASU)", "PDB ID"] == "1AAA"
    assert sel.loc["cryo-EM", "PDB ID"] == "4DDD"
    fixed = sel.loc["fixed (config)"]
    assert set(fixed["Status"]) == {"NOT RUN"}          # 3BHX etc. are not in the synthetic set
    assert sel.loc["lid disordered", "Status"] == "NOT RUN"
