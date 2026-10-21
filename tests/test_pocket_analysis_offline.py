"""End-to-end pocket / water / variant orchestration on synthetic structures (gemmi and network stubbed)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd

import fixtures as F
from psma_pipeline import pocket_analysis as PA
from psma_pipeline.excel_output import data_dictionary
from psma_pipeline.status import Status

SITE = list(range(205, 216))


def _seq():
    s = list(F.USEQ)
    s[208] = "F"                       # F209
    return "".join(s)


def _atoms(R: np.ndarray, t: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []

    def add(resname, resnum, name, el, xyz, polymer=True, water=False, chain="A", unp=True):
        p = np.asarray(xyz, float) @ R.T + t
        rows.append({"chain": chain, "label_asym": chain, "entity": "1" if polymer else "2", "resname": resname,
                     "resnum": resnum, "icode": "", "label_seq": resnum if polymer else None,
                     "unp": float(resnum) if polymer and unp else np.nan, "atom": name, "element": el, "altloc": "",
                     "x": p[0], "y": p[1], "z": p[2], "b": 20.0, "occ": 1.0, "polymer": polymer, "water": water,
                     "target": polymer})

    for n in range(195, 260):
        if n in (209, 210, 212):
            continue
        add("ALA", n, "CA", "C", rng.normal(0, 3, 3) + np.array([15.0 + (n - 195) * 0.5, 15.0, 0.0]))
    ang = np.radians(np.arange(6) * 60)
    add("PHE", 209, "CA", "C", (10, -3, 0))
    for nm, a in zip(["CG", "CD1", "CE1", "CZ", "CE2", "CD2"], ang):
        add("PHE", 209, nm, "C", (10 + 1.39 * np.cos(a), 1.39 * np.sin(a), 0))
    add("ARG", 210, "CA", "C", (0, -3, 0))
    add("ARG", 210, "CZ", "C", (0, -1.3, 0))
    add("ARG", 210, "NH1", "N", (0, 0, 0))
    add("ASP", 212, "CA", "C", (0, 4.0, -2.0))
    add("ASP", 212, "OD1", "O", (0, 3.65, 0.9))          # bridges both Zn (2.6 Å)
    for nm, a in zip(["C1", "C2", "C3", "C4", "C5", "C6"], ang):
        add("LIG", 701, nm, "C", (10 + 1.39 * np.cos(a), 1.39 * np.sin(a), 3.7), polymer=False, chain="L")
    add("LIG", 701, "O1", "O", (0, 0, 2.9), polymer=False, chain="L")
    add("ZN", 801, "ZN", "ZN", (0, 2.0, 2.9), polymer=False, chain="C")
    add("ZN", 802, "ZN", "ZN", (0, 5.3, 2.9), polymer=False, chain="C")
    add("HOH", 901, "O", "O", (0, 3.65, 2.9), polymer=False, water=True, chain="W")
    add("HOH", 902, "O", "O", (-3.0, 0, 1.5), polymer=False, water=True, chain="W")
    df = pd.DataFrame(rows)
    df["unp"] = pd.to_numeric(df["unp"])
    return df


def _rot(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


STRUCT = {"1AAA": _atoms(np.eye(3), np.zeros(3)), "2BBB": _atoms(_rot(55), np.array([4.0, -7.0, 3.0]))}


def _tables():
    summary = pd.DataFrame([
        {"PDB ID": "1AAA", "Assembly used": "none", "Resolution (Å)": 1.6, "Clashscore": 3.0,
         "Method": "X-RAY DIFFRACTION", "Entrance lid state": "closed", "Ramachandran outliers (%)": 0.0},
        {"PDB ID": "2BBB", "Assembly used": "none", "Resolution (Å)": 2.0, "Clashscore": 5.0,
         "Method": "X-RAY DIFFRACTION", "Entrance lid state": "open", "Ramachandran outliers (%)": 0.1}])
    chains = pd.DataFrame([{"PDB ID": p, "Entity": "1", "Chain": "A", "UniProt coverage": 0.9,
                            "N missing internal": 0, "Substitutions vs UniProt": ""} for p in ("1AAA", "2BBB")])
    site_def = pd.DataFrame([{"UniProt number": n, "Status": "OK"} for n in SITE])
    rank = pd.DataFrame([{"PDB ID": "1AAA", "Rank": 1}, {"PDB ID": "2BBB", "Rank": 2}])
    return {"Summary": summary, "Chains": chains, "ActiveSiteDefinition": site_def, "TemplateRanking": rank}


def _stub(monkeypatch):
    monkeypatch.setitem(sys.modules, "gemmi", types.ModuleType("gemmi"))
    entry = {"sequence": {"value": _seq()}, "features": [
        {"type": "Natural variant", "featureId": "VAR_X", "location": {"start": {"value": 210}, "end": {"value": 210}},
         "alternativeSequence": {"originalSequence": "R", "alternativeSequences": ["H"]}, "description": "test"},
        {"type": "Natural variant", "featureId": "VAR_BAD", "location": {"start": {"value": 210}, "end": {"value": 210}},
         "alternativeSequence": {"originalSequence": "W", "alternativeSequences": ["H"]}, "description": "isoform"}]}
    monkeypatch.setattr(PA, "uniprot_entry", lambda *a, **k: entry)
    monkeypatch.setattr(PA, "load_structure", lambda s, pdb, *a, **k: pdb)
    monkeypatch.setattr(PA, "fetch_sifts", lambda *a, **k: {})
    monkeypatch.setattr(PA, "sifts_segments", lambda *a, **k: {})
    monkeypatch.setattr(PA, "atoms_from_gemmi", lambda st, *a, **k: STRUCT[st].copy())
    monkeypatch.setattr(PA, "assembly_generators", lambda *a, **k: None)
    monkeypatch.setattr(PA, "fetch_chem_comps", lambda s, comps, *a, **k: {
        "LIG": {"name": "benzoate-like", "formula": "C7 O", "weight": 100.0, "smiles": "OC(=O)c1ccccc1",
                "smiles_stereo": "OC(=O)c1ccccc1"}})


def test_pocket_analysis_end_to_end(cfg, monkeypatch):
    _stub(monkeypatch)
    cfg = cfg.with_overrides({"interfaces": {"sasa": {"method": "shrake_rupley", "n_points": 30}}})
    res = PA.run_pocket_analysis(cfg, _tables())
    t = res.tables
    fp = t["PocketFingerprints"]
    assert len(fp) == 2 and fp["Representative"].all()
    assert set(fp["Zn-coordinating atoms"]) == {1}
    assert all("S1' glutamate pocket" in s for s in fp["Occupied sub-pockets"])
    assert (t["PocketSimilarity"]["Residue Jaccard"] == 1.0).all()
    assert len(t["LigandClusters"]) == 2 and t["LigandClusters"]["Hierarchical cluster"].nunique() == 1
    # waters: the Zn-bridging water is found in both superposed sites
    cw = t["ConservedWaters"]
    assert cw.iloc[0]["Sites with water"] == 2 and cw.iloc[0]["Zn-bridging fraction"] == 1.0
    assert cw.iloc[0]["Recommendation"] == "retain"
    # variants: R210H mapped, the W210H mismatch is a FAIL and not mapped
    v = t["Variants"].set_index("Variant ID")
    assert v.loc["VAR_BAD", "Status"] == Status.FAIL.value
    m = t["VariantStructureMapping"]
    assert list(m["Substitution"]) == ["R210H"]
    assert "sub-pocket" in m.iloc[0]["Categories"] and m.iloc[0]["Hypothesis"].startswith("Hypothesis")
    assert m.iloc[0]["AlphaFold status"].startswith(Status.NOT_RUN.value)
    assert "LigandConditionedRanking" not in t            # no reference ligand configured
    # ligand-conditioned ranking on demand
    r = PA.rank_for_ligand(res, cfg, comp="LIG", required_subpockets=["S1' glutamate pocket"], lid_state="closed")
    assert list(r["PDB ID"])[0] == "1AAA"                  # same pocket, better resolution, lid matches
    assert r["Reference instance"].iloc[0].startswith("1AAA:LIG")
    assert (r["Pocket-contact similarity"] == 1.0).all()
    # every new column is documented
    dd = data_dictionary({**t, "LigandConditionedRanking": r})
    undocumented = dd[dd["Description"] == "UNDOCUMENTED"]
    assert undocumented.empty, undocumented[["Sheet", "Column"]].to_string()
    from psma_pipeline.sanity import critical_failures, run_pocket_checks
    checks = run_pocket_checks(t, _tables(), ids=["1AAA", "2BBB"]).set_index("Check")
    assert critical_failures(checks.reset_index()).empty
    assert checks.loc["variant_numbering", "Status"] == Status.WARNING.value          # 1 of 2 fails
    assert checks.loc["catalytic_water_found", "Status"] == Status.NOT_APPLICABLE.value  # only 2 sites
    paths = PA.save_pocket_analysis(cfg, res)
    assert paths["workbook"].exists() and paths["manifest"].exists()


def test_rank_for_ligand_rejects_unknown_inputs(cfg, monkeypatch):
    _stub(monkeypatch)
    cfg = cfg.with_overrides({"interfaces": {"sasa": {"method": "shrake_rupley", "n_points": 30}}})
    res = PA.run_pocket_analysis(cfg, _tables())
    for kw in ({"comp": "NOPE"}, {"comp": "LIG", "required_subpockets": ["no such pocket"]}):
        try:
            PA.rank_for_ligand(res, cfg, **kw)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kw}")
