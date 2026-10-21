"""End-to-end modelling assessment on synthetic entries (no network, gemmi stubbed).

Entries
    DIM1  X-ray, one chain in the ASU, author assembly of two copies (symmetry dimer)
    DIM2  X-ray, two chains in the ASU forming the dimer, with an E424A substitution
    MONO  X-ray, software-only monomeric assembly; the dimer exists only in the lattice
    EM1   cryo-EM dimer with Q-scores
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from psma_pipeline import assessment as A
from psma_pipeline.excel_output import data_dictionary

import fixtures as F

IDS = ["DIM1", "DIM2", "MONO", "EM1"]


def _atoms(pdb):
    a = F.toy_chain_atoms()
    if pdb == "DIM2":
        b = F.toy_chain_atoms(chain="B", label="B")
        b[["x", "y", "z"]] = F.twofold().apply(b[["x", "y", "z"]].to_numpy())
        a = pd.concat([a, b], ignore_index=True)
    return a


def _gens(pdb):
    if pdb == "DIM2":
        return [(["A", "B"], [F.identity_op()])]
    if pdb == "MONO":
        return [(["A"], [F.identity_op()])]
    return [(["A"], [F.identity_op(), F.twofold()])]


def _xml(pdb, tmp_path):
    em = pdb == "EM1"
    head = ('<?xml version="1.0" ?><wwPDB-validation-information xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xsi:noNamespaceSchemaLocation="http://wwpdb.org/validation/schema/wwpdb_validation_v6.06.xsd">')
    entry = ('<Entry pdbid="em1" emdb_id="EMD-1" EMDB-resolution="3.0" Q_score="0.6" atom_inclusion_all_atoms="0.8"/>'
             if em else f'<Entry pdbid="{pdb.lower()}" PDB-resolution="2.0"/>')
    chains = ["A", "B"] if pdb == "DIM2" else ["A"]
    body = []
    for c in chains:
        for n in range(44, 144):
            metric = f'Q_score="{0.7 if n != 541 else 0.2}"' if em else f'rscc="{0.95 if n % 10 else 0.7}" rsrz="0.5"'
            body.append(f'<ModelledSubgroup model="1" chain="{c}" resnum="{n}" icode=" " resname="ALA" said="{c}" '
                        f'ent="1" seq="{n - 43}" altcode=" " {metric}/>')
    lig_metric = 'Q_score="0.5"' if em else 'rscc="0.93"'
    body.append('<ModelledSubgroup model="1" chain="A" resnum="801" icode=" " resname="XYZ" said="Z" ent="3" seq="." '
                f'altcode=" " {lig_metric}/>')
    p = tmp_path / f"{pdb}.xml"
    p.write_text(head + entry + "".join(body) + "</wwPDB-validation-information>")
    return p


def _tables():
    s = pd.DataFrame({
        "PDB ID": IDS, "Method": ["X-RAY DIFFRACTION"] * 3 + ["ELECTRON MICROSCOPY"],
        "Resolution (Å)": [1.5, 2.0, 2.5, 3.0], "Assembly used": ["1 (author_defined_assembly)"] * 2
        + ["1 (software_defined_assembly)", "1 (author_defined_assembly)"],
        "Assembly choice reason": ["author-defined (x)", "author-defined (x)", "software-defined only (x)",
                                   "author-defined (x)"],
        "Monomer / dimer": ["dimer", "dimer", "monomer", "dimer"], "Other polymer entities": ["", "", "", "Fab"],
        "Zn assembly": [4, 4, 2, 4], "Target copies in assembly": [2, 2, 1, 2],
        "Ligands of interest": ["XYZ(1)", "", "", "XYZ(1)"], "Entrance lid state": ["closed", "open", "open", "closed"],
        "Clashscore": [2.0, 3.0, 4.0, 6.0], "Ramachandran outliers (%)": [0.1, 0.2, 0.3, 0.5],
        "Active-site residues differing from UniProt": [None, "E424/A", None, None],
        "Max ligand B ratio": [1.1, None, None, 1.5], "Active-site mean normalised B": [-0.5, -0.2, 0.1, 0.0],
    })
    chains = pd.DataFrame({"PDB ID": ["DIM1", "DIM2", "DIM2", "MONO", "EM1"], "Entity": ["1"] * 5,
                           "Chain (auth)": ["A", "A", "B", "A", "A"],
                           "Substitutions vs UniProt": ["", "E424A", "E424A", "", ""],
                           "UniProt coverage": [0.94] * 5, "N missing internal": [0, 2, 2, 5, 1]})
    active = pd.DataFrame({"PDB ID": IDS, "Chain": ["A"] * 4, "Modelled": [True] * 4})
    site_def = pd.DataFrame({"UniProt number": [61, 71, 81], "Status": ["OK"] * 3})
    bind = pd.DataFrame({"PDB ID": ["DIM1"], "Component": ["ZN"], "Class": ["ion of interest"], "Chain": ["A"],
                         "Residue no.": ["801"], "Donor atoms (N/O/S: Å)": ["D45/A OD1: 2.05; K47/A NZ: 2.1"],
                         "Protein residues (UniProt/chain)": ["D45/A, K47/A"], "B ratio ligand/environment": [1.0]})
    glyco = pd.DataFrame({"PDB ID": ["DIM1"], "Chain": ["A"], "Asn (author)": ["50"], "Asn (UniProt)": [50]})
    rank = pd.DataFrame({"PDB ID": IDS, "Rank": [1, 2, 3, 4]})
    return {"Summary": s, "Chains": chains, "ActiveSite": active, "ActiveSiteDefinition": site_def,
            "BindingSites": bind, "Glycosylation": glyco, "TemplateRanking": rank}


@pytest.fixture
def assessed(cfg, monkeypatch, tmp_path):
    if "gemmi" not in sys.modules:
        try:
            import gemmi  # noqa: F401
        except ImportError:
            monkeypatch.setitem(sys.modules, "gemmi", types.ModuleType("gemmi"))
    monkeypatch.setattr(A, "load_structure", lambda s, pdb, d, u, **k: pdb)
    monkeypatch.setattr(A, "fetch_sifts", lambda *a, **k: {})
    monkeypatch.setattr(A, "atoms_from_gemmi", lambda st, m, segs, ents, alt="first": _atoms(st))
    monkeypatch.setattr(A, "assembly_generators", lambda st, aid: _gens(st))
    monkeypatch.setattr(A, "lattice_operators", lambda st, n=1: [F.identity_op("1_555"), F.twofold(9.0, "2_555")])
    monkeypatch.setattr(A, "pdbe_validation", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(A, "fetch_validation_xml", lambda s, pdb, *a, **k: _xml(pdb, tmp_path))
    monkeypatch.setattr(A, "_emdb_maps", lambda s, e, c: {"Primary map": "available", "Half maps": "2 available"})
    cfg = cfg.with_overrides({"interfaces": {"sasa": {"method": "shrake_rupley", "n_points": 40}}})
    return cfg, A.run_assessment(cfg, _tables())


def test_interfaces_and_lattice(assessed):
    cfg, res = assessed
    t = res.tables
    by = t["Interfaces"].groupby("PDB ID").size().to_dict()
    assert by == {"DIM1": 1, "DIM2": 1, "EM1": 1}                      # MONO has a single protomer
    s = t["AssessmentSummary"].set_index("PDB ID")
    assert s.loc["MONO", "Canonical dimer in crystal lattice"] == "2_555"
    assert s.loc["DIM1", "Canonical interface similarity"] == pytest.approx(1.0)
    assert bool(s.loc["DIM1", "Interface complete"]) is True
    assert len(res.canonical["canonical residues"]) > 0
    assert res.jaccard_matrix.shape == (3, 3)


def test_local_fit_tables(assessed):
    _, res = assessed
    t = res.tables
    lf = t["LocalModelFit"]
    groups = set(lf[lf["PDB ID"] == "DIM1"]["Group"])
    assert {"active site", "metal-coordinating residues", "ligand-contact residues", "entrance lid 541-548",
            "dimer-interface residues", "glycosylation sites", "ligands", "conserved waters"} <= groups
    assert lf[(lf["PDB ID"] == "DIM1") & (lf["Group"] == "conserved waters")]["Calculation status"].iloc[0] == "NOT RUN"
    em = t["EMMapFit"]
    assert list(em["PDB ID"]) == ["EM1"] and em["Half maps"].iloc[0] == "2 available"
    lig = t["LigandModelFit"]
    assert set(lig["PDB ID"]) == set(IDS)
    s = t["AssessmentSummary"].set_index("PDB ID")
    assert s.loc["DIM1", "Active-site local support"] == "strong local support"
    assert s.loc["EM1", "EM Q-score"] == pytest.approx(0.6)


def test_ranking_modes(assessed, cfg):
    _, res = assessed
    r = res.tables["TemplateRankings"]
    assert set(r["RankingMode"]) == set(cfg.get("ranking.modes"))
    md = r[r["RankingMode"] == "md_start"]["PDB ID"].tolist()
    assert "DIM2" not in md and "EM1" not in md and "MONO" not in md   # substitution, Fab chains, monomer
    assert r[r["RankingMode"] == "cryoem_reference"]["PDB ID"].tolist() == ["EM1"]
    assert r[r["RankingMode"] == "apo"]["PDB ID"].tolist() == ["DIM2", "MONO"] or \
        set(r[r["RankingMode"] == "apo"]["PDB ID"]) == {"DIM2", "MONO"}
    assert "DIM2" not in r[r["RankingMode"] == "active_site"]["PDB ID"].tolist()
    assert len(res.tables["RankingTop"]) > 0


def test_assessment_workbook_is_fully_documented(assessed, tmp_path):
    cfg, res = assessed
    dd = data_dictionary(res.tables)
    und = dd[dd["Description"] == "UNDOCUMENTED"]
    assert und.empty, und[["Sheet", "Column"]].drop_duplicates().values.tolist()
    paths = A.save_assessment(cfg.with_overrides({"paths": {"outputs": str(tmp_path / "out")}}), res)
    wb = pd.read_excel(paths["workbook"], sheet_name=None)
    for sheet in ("Interfaces", "InterfaceContacts", "InterfaceComparison", "LocalModelFit", "LigandModelFit",
                  "EMMapFit", "TemplateRankings", "AssessmentSummary", "Reproducibility", "DataDictionary"):
        assert sheet in wb
    assert np.isfinite(np.loadtxt(paths["tables_dir"] / "interface_jaccard_matrix.csv", delimiter=",", skiprows=1)).all()


def test_assessment_sanity_checks(assessed):
    from psma_pipeline.sanity import critical_failures, run_assessment_checks
    _, res = assessed
    df = run_assessment_checks(res.tables, _tables()["Summary"])
    assert critical_failures(df).empty, df.to_string()
    assert set(df["Check"]) >= {"dimer_interface_present", "bsa_positive", "canonical_interface_defined",
                                "unsupported_ligand_not_strong", "em_metrics_only_for_em"}
    # a dimer whose interface has no contacts must fail
    t = dict(res.tables)
    t["Interfaces"] = t["Interfaces"].assign(**{"Residue contacts": 0})
    bad = run_assessment_checks(t, _tables()["Summary"])
    assert (bad[bad["Check"] == "dimer_interface_present"]["Status"] == "FAIL").all()


def test_assessment_validation_set_always_has_dimers(cfg):
    from psma_pipeline.sanity import assessment_validation_ids
    t = _tables()                                            # no ValidationSet sheet, as after an automatic survey
    sel = assessment_validation_ids(t, cfg)
    ids = list(sel["PDB ID"])
    assert {"DIM1", "DIM2"} <= set(ids)                      # author-defined dimers define the canonical interface
    assert "MONO" in ids and "EM1" in ids
