"""Milestone 5 (model preparation): offline tests on the synthetic hybrid system.

Exit criteria covered: originals never overwritten (hash check), a provenance chain for every
output, and alternative microstates; plus the hybrid construction, Zn site, glycans and protonation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import prep_fixtures as P
from psma_pipeline import model_preparation as MP
from psma_pipeline.status import IssueLog, Status

pytest.importorskip("tmfold", reason="membrane placement needs tmfold (pip install -e '.[membrane]')")


@pytest.fixture(scope="module")
def pcfg(tmp_path_factory):
    from psma_pipeline import load_config
    tmp = tmp_path_factory.mktemp("prep")
    return load_config(overrides={"paths": {"cache": str(tmp / "cache"), "outputs": str(tmp / "outputs")},
                                  "run": {"use_coordinates": False}, "membrane": {"deeptmhmm": {"enabled": False}}})


@pytest.fixture(scope="module")
def inputs(pcfg, tmp_path_factory):
    return {"anchor": P.af_model(tmp_path_factory.mktemp("af"), pcfg), "asm": P.template_assembly(),
            "donors": {"DON1": P.donor_atoms()}}


@pytest.fixture(scope="module")
def result(pcfg, inputs):
    return MP.run_preparation(pcfg, P.survey_tables(), None, sess=object(), template_asm=inputs["asm"],
                              anchor=inputs["anchor"], donors=inputs["donors"], uniprot=P.uniprot_entry(), run_pka=False)


def test_global_fit_recovers_frame(pcfg, inputs):
    """The synthetic dimer is exactly C2-symmetric, so both chain pairings fit equally well (a tie);
    the direct pairing must win regardless of floating-point rounding."""
    fit = MP.global_dimer_fit(inputs["asm"], inputs["anchor"], pcfg)
    assert fit["rmsd"] < 0.3 and fit["pairs"] == {"A|1": "A", "B|1": "B"}
    # swapped chains in the AF2 model: the direct pairing (template A -> AF2 'A' label) is now the swapped one
    swapped = inputs["anchor"]
    m = MP.AFModel(swapped.model_id, swapped.source, swapped.kind,
                   swapped.atoms.assign(chain=swapped.atoms["chain"].map({"A": "B", "B": "A"})),
                   chain_info=[{**c, "Chain": {"A": "B", "B": "A"}[c["Chain"]]} for c in swapped.chain_info])
    fit2 = MP.global_dimer_fit(inputs["asm"], m, pcfg)
    assert fit2["rmsd"] < 0.3 and list(fit2["pairs"].values()) == m.target_chains[:2]


def test_hybrid_residue_sources(result):
    src = result.tables["ResidueSources"].set_index(["Chain", "UniProt number"])["Source"]
    assert src[("A", 30)].startswith("AF2") and "membrane anchor" in src[("A", 30)]
    assert src[("A", 302)].startswith("AF2") and "internal gap" in src[("A", 302)]
    assert "reverted to UniProt" in src[("A", 200)]
    assert src[("A", 257)].endswith("+ AF2 side-chain atoms")
    assert src[("A", 400)] == "experimental 9S7U"
    rep = result.tables["Repairs"]
    assert {"substitution reverted to UniProt", "missing side-chain atoms from AF2"} <= set(rep["Repair"])
    atoms = result.atoms
    res200 = atoms[(atoms["chain"] == "A") & (atoms["resnum"] == 200)]
    assert res200["resname"].iloc[0] == "ALA" and "CB" in set(res200["atom"])


def test_junctions_and_numbering(result, inputs):
    """Each graft must reproduce the AF2 model's own C–N distance (the synthetic ectodomain is a point
    cloud, not a chain, so the intrinsic values vary; the fits must not add error)."""
    j = result.tables["Junctions"]
    assert {"N-terminal anchor → template", "internal gap: template → AF2", "internal gap: AF2 → template"} <= set(j["Type"])
    af = inputs["anchor"].atoms
    for _, r in j.iterrows():
        a, b = map(int, r["Junction"].split("-"))
        c = af[(af["chain"] == r["Chain"]) & (af["resnum"] == a) & (af["atom"] == "C")][["x", "y", "z"]].to_numpy()
        n = af[(af["chain"] == r["Chain"]) & (af["resnum"] == b) & (af["atom"] == "N")][["x", "y", "z"]].to_numpy()
        assert abs(r["Peptide C–N (Å)"] - float(np.linalg.norm(c - n))) < 0.3, r.to_dict()
    checks = result.tables["PrepChecks"]
    assert (checks.query("Check == 'numbering_identity'")["Status"] == "PASS").all()


def test_graft_point_prefers_ideal_peptide_bond():
    """Regression (first Colab run): the shortest C–N (0.75 Å, overlapping atoms) was chosen and passed."""
    best = None
    for d in (2.4, 0.75, 1.45, 1.9):
        if MP.better_graft(d, best):
            best = d
    assert best == 1.45
    assert MP.junction_status(0.75) is Status.WARNING and MP.junction_status(2.4) is Status.WARNING
    assert MP.junction_status(1.33) is Status.PASS and MP.junction_status(None) is Status.WARNING


def test_monodentate_carboxylate_not_flagged(pcfg):
    """Regression: D453 binds Zn1 through OD2 (2.0 Å); its OD1 at 2.76 Å must not count as a bad bond."""
    from psma_pipeline.metal_sites import zn_coordination
    rows = [("A", "ZN", 3001, "ZN", "ZN", 0.0, 0.0, 0.0, "ion"),
            ("A", "HIS", 377, "NE2", "N", 2.1, 0.0, 0.0, "protein"),
            ("A", "ASP", 387, "OD1", "O", 0.0, 1.95, 0.0, "protein"),
            ("A", "ASP", 453, "OD2", "O", 0.0, 0.0, 2.0, "protein"),
            ("A", "ASP", 453, "OD1", "O", -1.5, 0.0, 2.3, "protein")]
    atoms = pd.DataFrame(rows, columns=["chain", "resname", "resnum", "atom", "element", "x", "y", "z", "kind"])
    tab, contacts = zn_coordination(atoms, pcfg)
    assert tab.iloc[0]["Site"] == "Zn1" and tab.iloc[0]["Status"] == "PASS", tab.iloc[0]["Warning"]
    assert len(contacts) == 4                                   # OD1 still listed in ZnContacts


def test_hetero_selection(result):
    het = result.tables["TemplateHetero"]
    kept = set(het.loc[het["Decision"] == "kept", "Component"])
    assert kept == {"ZN", "HOH", "CA", "CL", "NAG"}
    removed = het[het["Decision"] == "removed"]
    assert "JB7" in set(removed["Component"]) and "outer glycan" in " ".join(removed["Note"].astype(str))
    assert int((result.atoms["kind"] == "water").sum()) == 2       # bridging water per chain, bulk dropped


def test_zn_site(result, pcfg):
    zn = result.tables["ZnSite"]
    assert len(zn) == 4 and set(zn["Site"]) == {"Zn1", "Zn2"}
    assert (zn["Status"] == "PASS").all()
    assert zn["Zn–Zn distance (Å)"].round(1).eq(3.3).all() and (zn["Bridging water/hydroxide"] >= 1).all()
    opt = result.tables["MetalTreatmentOptions"]
    assert opt.set_index("Treatment")["Chosen"].to_dict() == {"nonbonded_12_6_4": False, "bonded_mcpb": True,
                                                              "cationic_dummy": False}
    r = result.tables["PrepReadiness"].set_index("Item")
    assert r.loc["zinc MD treatment", "Status"] == "PASS" and r.loc["zinc ions placed (experimental)", "Status"] == "PASS"
    assert r.loc["bridging water / hydroxide", "Status"] in ("PASS", "WARNING")   # hydroxide chosen in the config


def test_zn_treatment_unchosen_blocks(pcfg, inputs):
    cfg2 = pcfg.with_overrides({"preparation": {"zn_md_treatment": None, "zn_bridge": None}})
    res = MP.run_preparation(cfg2, P.survey_tables(), None, sess=object(), template_asm=inputs["asm"],
                             anchor=inputs["anchor"], donors=inputs["donors"], uniprot=P.uniprot_entry(), run_pka=False)
    r = res.tables["PrepReadiness"].set_index("Item")
    assert r.loc["zinc MD treatment", "Status"] == "FAIL" and r.loc["bridging water / hydroxide", "Status"] == "WARNING"
    assert not res.tables["MetalTreatmentOptions"]["Chosen"].any()


def test_zn_treatment_choice_unblocks_item(pcfg, inputs):
    cfg2 = pcfg.with_overrides({"preparation": {"zn_md_treatment": "nonbonded_12_6_4"}})
    res = MP.run_preparation(cfg2, P.survey_tables(), None, sess=object(), template_asm=inputs["asm"],
                             anchor=inputs["anchor"], donors=inputs["donors"], uniprot=P.uniprot_entry(), run_pka=False)
    r = res.tables["PrepReadiness"].set_index("Item")
    assert r.loc["zinc MD treatment", "Status"] == "PASS" and r.loc["OVERALL", "Status"] == "FAIL"
    assert res.tables["MetalTreatmentOptions"].set_index("Treatment").loc["nonbonded_12_6_4", "Chosen"]


def test_glycans_core_glcnac(result):
    g = result.tables["Glycans"].set_index(["Chain", "Asn (UniProt)"])
    assert g.loc[("A", 140), "Decision"] == "core GlcNAc" and g.loc[("A", 140), "Source"] == "template 9S7U"
    assert g.loc[("B", 140), "Decision"] == "core GlcNAc" and g.loc[("B", 140), "Source"].startswith("transplanted from DON1")
    assert g.loc[("A", 459), "Decision"] == "not glycosylated"
    text = result.files["membrane_oriented"].read_text()
    assert text.count("LINK") == 2 and " ND2 ASN A 140" in text


def test_protonation_rules(result):
    p = result.tables["Protonation"].set_index(["Chain", "UniProt number"])
    assert p.loc[("A", 377), "State"] == "HID" and "Zn ligand via NE2" in p.loc[("A", 377), "Reason"]
    assert p.loc[("A", 387), "State"] == "ASP" and p.loc[("A", 387), "pKa source"] == "fixed (Zn ligand)"
    assert p.loc[("A", 424), "Alternative state"] == "GLH"          # always_alternative: catalytic E424
    ms = result.tables["Microstates"]
    assert ms["Microstate"].iloc[0] == "microstate_00_default" and any("424" in m for m in ms["Microstate"])
    text = (result.files["microstates"] / f"{ms['Microstate'].iloc[1]}.pdb").read_text()
    assert "GLH" in text or "HIP" in text or "ASH" in text or "LYN" in text


def test_parse_propka_summary():
    txt = """
SUMMARY OF THIS PREDICTION
       Group      pKa  model-pKa   ligand atom-type
   ASP 387 A     2.11       3.80
   GLU 424 A     6.95       4.50
   HIS 377 B     5.10       6.50
   N+    1 A     7.80       8.00
--------------------------------------------------------------------------------------------------------
"""
    df = MP.parse_propka(txt)
    assert df["Group"].tolist() == ["ASP", "GLU", "HIS", "N+"]
    assert df.loc[1, "pKa"] == pytest.approx(6.95) and df.loc[2, "Chain"] == "B"


def test_membrane_and_outputs(result):
    hm = result.tables["HybridMembrane"].iloc[0]
    assert hm["Placement method"].startswith("C2-axis constrained") and hm["TM in slab"] >= 0.9
    from psma_pipeline.alphafold import _atoms_from_pdb_text
    o = _atoms_from_pdb_text(result.files["membrane_oriented"].read_text())
    ca = o[(o["atom"] == "CA") & (o["chain"] == "A") & o["polymer"]].set_index("resnum")
    assert abs(ca.loc[20:43, "z"].mean()) < 5 and ca.loc[1:19, "z"].max() < 0 and ca.loc[57:, "z"].min() > 0


def test_exit_criteria(result, inputs):
    c = result.tables["PrepChecks"]
    crit = c[c["Critical"]]
    assert (crit["Status"] == "PASS").all(), crit.to_string()
    assert set(crit["Check"]) == {"numbering_identity", "originals_unchanged", "zn_site_complete", "provenance_complete"}
    import json
    prov = json.loads(result.files["provenance"].read_text())
    assert prov["originals"] and [s["name"] for s in prov["steps"]][:3] == ["hybrid model", "N-glycans (core GlcNAc)", "protonation"]
    assert result.tables["PrepReadiness"].set_index("Item").loc["OVERALL", "Status"] == "FAIL"


def test_originals_tampering_detected(pcfg, inputs, monkeypatch):
    """If an original changes during the run, the critical check fails."""
    orig = MP._hash_all
    calls = {"n": 0}

    def fake(paths):
        calls["n"] += 1
        h = orig(paths)
        return {k: (v if calls["n"] == 1 else "changed") for k, v in h.items()}
    monkeypatch.setattr(MP, "_hash_all", fake)
    res = MP.run_preparation(pcfg, P.survey_tables(), None, sess=object(), template_asm=inputs["asm"],
                             anchor=inputs["anchor"], donors=inputs["donors"], uniprot=P.uniprot_entry(), run_pka=False)
    c = res.tables["PrepChecks"].set_index("Check")
    assert c.loc["originals_unchanged", "Status"] == "FAIL"


def test_save_and_dictionary(pcfg, result):
    from psma_pipeline.excel_output import data_dictionary
    paths = MP.save_preparation(pcfg, result, tag="test")
    assert paths["workbook"].exists()
    dd = data_dictionary({**result.tables, "PrepIssues": result.issues.to_frame()})
    und = dd[dd["Description"] == "UNDOCUMENTED"]
    assert und.empty, und[["Sheet", "Column"]].to_string()
