"""wwPDB validation XML parsing and method-specific local-fit categories."""

from __future__ import annotations

import gzip

import pandas as pd
import pytest

from psma_pipeline.config import load_config
from psma_pipeline.local_fit import (ACCEPTABLE, NONE, STRONG, WEAK, categorise, em_entry_row, expected_q,
                                     group_fit, ligand_fit_rows, residue_fit_table)
from psma_pipeline.validation import parse_validation_xml, validation_xml_url

XRAY_XML = """<?xml version="1.0" ?>
<wwPDB-validation-information xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:noNamespaceSchemaLocation="http://wwpdb.org/validation/schema/wwpdb_validation_v6.06.xsd">
  <Entry pdbid="1abc" PDB-resolution="1.60" clashscore="2.1"/>
  <ModelledSubgroup model="1" chain="A" resnum="377" icode=" " resname="HIS" said="A" ent="1" seq="334"
    altcode=" " rscc="0.95" rsrz="0.4" avgoccu="1.00" rama="Favored" rota="m-70"/>
  <ModelledSubgroup model="1" chain="A" resnum="387" icode=" " resname="ASP" said="A" ent="1" seq="344"
    altcode=" " rscc="0.84" rsrz="2.5" avgoccu="1.00">
    <clash atom="OD1" cid="1" clashmag="0.45" dist="2.1"/>
  </ModelledSubgroup>
  <ModelledSubgroup model="1" chain="A" resnum="541" icode=" " resname="TRP" said="A" ent="1" seq="498"
    altcode=" " rscc="0.62" rsrz="4.1" avgoccu="0.60"/>
  <ModelledSubgroup model="1" chain="A" resnum="801" icode=" " resname="XYZ" said="C" ent="3" seq="."
    altcode=" " rscc="0.71" avgoccu="0.80" mogul_bonds_rmsz="1.9">
    <mog-bond-outlier atoms="C1,C2"/>
    <chiral-outlier atom="C3"/>
  </ModelledSubgroup>
  <ModelledSubgroup model="1" chain="A" resnum="901" icode=" " resname="HOH" said="D" ent="4" seq="."
    altcode=" " rscc="0.9"/>
</wwPDB-validation-information>
"""

EM_XML = """<?xml version="1.0" ?>
<wwPDB-validation-information xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:noNamespaceSchemaLocation="http://wwpdb.org/validation/schema/wwpdb_validation_v6.06.xsd">
  <Entry pdbid="9xyz" emdb_id="EMD-12345" EMDB-resolution="3.0" Q_score="0.55"
    atom_inclusion_backbone="0.85" atom_inclusion_all_atoms="0.75"/>
  <ModelledSubgroup model="1" chain="A" resnum="377" icode=" " resname="HIS" said="A" ent="1" seq="334"
    altcode=" " Q_score="0.62" residue_inclusion="1.0"/>
  <ModelledSubgroup model="1" chain="A" resnum="541" icode=" " resname="TRP" said="A" ent="1" seq="498"
    altcode=" " Q_score="0.40"/>
</wwPDB-validation-information>
"""


@pytest.fixture
def lf():
    return load_config().require("local_fit")


def test_xml_url_template():
    assert validation_xml_url("3BHX", "x/{mid}/{pdb}/{pdb}.gz") == "x/bh/3bhx/3bhx.gz"


def test_parse_xray_xml_including_gzip(tmp_path):
    p = tmp_path / "x.xml.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(XRAY_XML)
    d = parse_validation_xml(p)
    assert d["schema"] == "wwpdb_validation_v6.06.xsd" and d["entry"]["pdbid"] == "1abc"
    r = d["residues"].set_index("resnum")
    assert r.loc[377, "rscc"] == 0.95 and r.loc[377, "label_seq"] == 334 and r.loc[377, "icode"] == ""
    assert r.loc[387, "n_clashes"] == 1
    assert pd.isna(r.loc[801, "label_seq"]) and r.loc[801, "n_mogul_bond_outliers"] == 1
    assert r.loc[801, "n_chiral_outliers"] == 1 and r.loc[801, "mogul_bonds_rmsz"] == 1.9


def test_xray_categories(lf):
    assert categorise("xray", 0.95, 0.4, cfg=lf) == STRONG
    assert categorise("xray", 0.84, 2.5, cfg=lf) == ACCEPTABLE
    assert categorise("xray", 0.62, 4.1, cfg=lf) == WEAK
    assert categorise("xray", 0.92, None, cfg=lf) == STRONG       # ligands often have no RSRZ
    assert categorise("xray", None, None, cfg=lf) == NONE


def test_em_categories_use_resolution_dependent_expectation(lf):
    assert expected_q(3.0, lf) == pytest.approx(-0.1775 * 3 + 1.1192)
    qe = expected_q(3.0, lf)
    assert categorise("em", q=qe + 0.01, resolution=3.0, cfg=lf) == STRONG
    assert categorise("em", q=qe - 0.10, resolution=3.0, cfg=lf) == ACCEPTABLE
    assert categorise("em", q=qe - 0.30, resolution=3.0, cfg=lf) == WEAK
    assert categorise("em", q=0.6, resolution=None, cfg=lf) == NONE  # no resolution: never guessed


def test_rscc_and_q_are_never_mixed(lf):
    # an RSCC-like number passed as Q for an X-ray entry is ignored, and vice versa
    assert categorise("xray", rscc=None, q=0.95, cfg=lf) == NONE
    assert categorise("em", rscc=0.95, q=None, resolution=3.0, cfg=lf) == NONE


def test_group_fit_and_lowest_residue(tmp_path, lf):
    p = tmp_path / "x.xml"
    p.write_text(XRAY_XML)
    fit = residue_fit_table(parse_validation_xml(p)["residues"], "xray", 1.6, lf)
    g = group_fit("1ABC", "active site", [("A", 377, ""), ("A", 387, ""), ("A", 500, "")], fit, "xray", 1.6, lf)
    assert g["Residues in group"] == 3 and g["Residues with metric"] == 2
    assert g["Group support"] == ACCEPTABLE                 # median RSCC 0.895, median RSRZ 1.45
    assert "387" in g["Lowest-supported residue"] and g["Calculation status"] == "WARNING"
    lid = group_fit("1ABC", "lid", [("A", 541, "")], fit, "xray", 1.6, lf)
    assert lid["Group support"] == WEAK
    empty = group_fit("1ABC", "x", [], fit, "xray", 1.6, lf)
    assert empty["Calculation status"] == "NOT APPLICABLE"


def test_ligand_rows_never_call_unsupported_ligand_well_supported(tmp_path, lf):
    p = tmp_path / "x.xml"
    p.write_text(XRAY_XML.replace('rscc="0.71" avgoccu="0.80"', 'avgoccu="0.80"'))
    fit = residue_fit_table(parse_validation_xml(p)["residues"], "xray", 1.6, lf)
    rows = ligand_fit_rows("1ABC", fit, "xray", 1.6, lf, lambda c: "ligand of interest")
    assert len(rows) == 1                                    # water excluded
    r = rows[0]
    assert r["Local support"] == NONE and "cannot be called well supported" in r["Warning"]
    assert r["Chirality outliers"] == 1 and r["Geometry outliers"] >= 2


def test_em_entry_row(tmp_path, lf):
    p = tmp_path / "e.xml"
    p.write_text(EM_XML)
    d = parse_validation_xml(p)
    fit = residue_fit_table(d["residues"], "em", 3.0, lf)
    row = em_entry_row("9XYZ", d["entry"], fit, d["schema"])
    assert row["emdb_id"] == "EMD-12345" and row["EMDB-resolution"] == 3.0 and row["Q_score"] == 0.55
    assert row["Per-chain mean Q-score"] == "A: 0.510"
    assert row["Calculation status"] == "OK"
    assert set(fit["Support"]) == {STRONG, WEAK}
