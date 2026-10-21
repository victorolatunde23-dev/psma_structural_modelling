"""Conserved waters, variant mapping and ligand-conditioned ranking (offline)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from psma_pipeline.status import Status
from psma_pipeline.variants import (ca_rmsf, categorise, sequon_effect, sequons, uniprot_variants,
                                    validate_variants, residue_environment)
from psma_pipeline.waters import WaterSite, conserved_presence, conserved_waters

WCFG = {"superpose_on": "active_site", "cluster_distance": 1.2, "hbond_max": 3.5, "zn_bridge_max": 2.5,
        "retain_min_fraction": 0.7, "retain_min_partners": 2}


def _rot(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _site(pdb, R, t, waters):
    rng = np.random.default_rng(0)
    ca0 = {n: rng.normal(0, 6, 3) for n in range(1, 9)}
    ca = {n: v @ R.T + t for n, v in ca0.items()}
    zn = np.array([[0.0, 0.0, 0.0], [3.3, 0.0, 0.0]]) @ R.T + t
    polar0 = [(1, "ARG", "NH1", (1.65, 3.0, 0.0)), (2, "TYR", "OH", (1.65, -3.0, 0.0))]
    polar = pd.DataFrame([{"unp": u, "resname": rn, "atom": an,
                           **dict(zip("xyz", np.array(p) @ R.T + t))} for u, rn, an, p in polar0])
    w = pd.DataFrame([{"resnum": 900 + i, **dict(zip("xyz", np.array(p) @ R.T + t)), "b": 15.0, "occ": 1.0}
                      for i, p in enumerate(waters)])
    return WaterSite(pdb, "A", ca, w, polar, zn, zn)


def test_conserved_waters_clusters_across_superposed_sites():
    bridging = (1.65, 0.0, 0.0)       # between both Zn, ~1.65 Å from each
    second = (1.65, 0.5, 1.6)         # near both polar atoms? (distance check below)
    lone = (8.0, 8.0, 8.0)
    s1 = _site("1AAA", np.eye(3), np.zeros(3), [bridging, second, lone])
    s2 = _site("2BBB", _rot(40), np.array([5.0, -3.0, 2.0]), [bridging, second])
    s3 = _site("3CCC", _rot(-70), np.array([-4.0, 1.0, 9.0]), [bridging])
    cl, mem, per = conserved_waters([s1, s2, s3], WCFG, list(range(1, 9)))
    assert (per["Status"] == Status.OK.value).all()
    assert per["Superposition RMSD (Å)"].max() < 1e-6
    top = cl.iloc[0]
    assert top["Water cluster"] == "W1" and top["Sites with water"] == 3 and top["Conservation fraction"] == 1.0
    assert top["Zn-bridging fraction"] == 1.0 and top["Recommendation"] == "retain"
    assert set(cl["Sites with water"]) == {3, 2, 1}
    assert cl.set_index("Sites with water").loc[1, "Recommendation"] == "remove"
    pres = conserved_presence(mem, cl)
    assert pres["3CCC"] == 1.0


def test_conserved_waters_empty():
    cl, mem, per = conserved_waters([], WCFG, [1])
    assert cl.empty and mem.empty and per.empty


SEQ = "MANATKNPSQ" + "A" * 20


def test_sequons_and_effects():
    assert sequons(SEQ) == [3]                       # N3-A-T; N7-P-S is not a sequon
    assert sequon_effect(SEQ, 5, "A") == "removes sequon N3"
    assert sequon_effect(SEQ, 8, "A") == "creates sequon N7"
    assert sequon_effect(SEQ, 1, "L") == ""


def test_validate_variants_reference_check():
    rows = [{"Source": "UniProt", "Source type": "natural variant", "Variant ID": "V1", "Start": 2, "End": 2,
             "Reference aa": "A", "Variant aa": "V"},
            {"Source": "UniProt", "Source type": "natural variant", "Variant ID": "V2", "Start": 2, "End": 2,
             "Reference aa": "G", "Variant aa": "V"},
            {"Source": "UniProt", "Source type": "natural variant", "Variant ID": "V3", "Start": 2, "End": 4,
             "Reference aa": "ANA", "Variant aa": "A"},
            {"Source": "EBI Proteins API", "Source type": "missense", "Variant ID": "rs1", "Start": 5, "End": 5,
             "Reference aa": "Thr", "Variant aa": "Met"}]
    df = validate_variants(rows, SEQ).set_index("Variant ID")
    assert df.loc["V1", "Status"] == Status.OK.value
    assert df.loc["V2", "Status"] == Status.FAIL.value and "does not match" in df.loc["V2", "Status reason"]
    assert df.loc["V3", "Status"] == Status.NOT_APPLICABLE.value
    assert df.loc["rs1", "Substitution"] == "T5M" and df.loc["rs1", "Status"] == Status.OK.value


def test_uniprot_variant_parsing():
    entry = {"features": [
        {"type": "Natural variant", "featureId": "VAR_1", "location": {"start": {"value": 5}, "end": {"value": 5}},
         "alternativeSequence": {"originalSequence": "T", "alternativeSequences": ["M", "K"]},
         "description": "in dbSNP:rs123", "evidences": [{}]},
        {"type": "Mutagenesis", "location": {"start": {"value": 2}, "end": {"value": 2}},
         "alternativeSequence": {"originalSequence": "A", "alternativeSequences": ["G"]}, "description": "Loss"},
        {"type": "Chain", "location": {"start": {"value": 1}, "end": {"value": 30}}}]}
    v = uniprot_variants(entry)
    assert len(v) == 3
    assert {r["Source type"] for r in v} == {"natural variant", "mutagenesis"}
    assert "rs123" in v[0]["Cross-references"]


def test_categorise_is_hypothesis_only():
    ctx = {"site_numbers": {10}, "zn_ligands": {10}, "subpockets": {"S1": {10, 11}}, "rmsf": {10: 0.3},
           "regions": {"lid": (9, 12)}, "disulfide_cys": set(), "sequon": ""}
    env = {"RSA": 0.05, "Distance to ligand (Å)": 3.0, "Distance to Zn (Å)": 2.1,
           "Distance to partner protomer (Å)": 20.0}
    cats, hyp = categorise(10, "H", "Y", env, ctx, {"near_distance": 6.0, "buried_rsa_max": 0.2})
    assert "zinc ligand" in cats and "buried" in cats and "region: lid" in cats
    assert hyp.startswith("Hypothesis (not a prediction)")
    cats2, hyp2 = categorise(10, "H", "Y", None, ctx, {})
    assert "not modelled in reference structure" in cats2


def test_ca_rmsf_identical_and_moving():
    rng = np.random.default_rng(1)
    base = {n: rng.normal(0, 10, 3) for n in range(1, 61)}
    ca = {("P", str(i)): {n: v @ np.eye(3) + i for n, v in base.items()} for i in range(4)}
    rmsf, cnt = ca_rmsf(ca)
    assert max(rmsf.values()) < 1e-6 and cnt[1] == 4
    moved = dict(ca[("P", "1")])
    moved[60] = moved[60] + np.array([4.0, 0, 0])
    ca[("P", "1")] = moved
    rmsf, _ = ca_rmsf(ca)
    assert rmsf[60] > 1.0 and rmsf[1] < 0.5


def test_residue_environment_columns():
    def a(prot, unp, resname, name, el, xyz, polymer=True):
        return {"chain": prot[0], "label_asym": prot[0], "resname": resname, "resnum": unp, "icode": "",
                "unp": unp if polymer else None, "atom": name, "element": el, "x": xyz[0], "y": xyz[1],
                "z": xyz[2], "b": 20.0 + unp, "occ": 1.0, "polymer": polymer, "water": False, "target": polymer,
                "copy": "1", "protomer": prot}
    rows = [a("A|1", 1, "ALA", "CA", "C", (0, 0, 0)), a("A|1", 1, "ALA", "CB", "C", (1.5, 0, 0)),
            a("A|1", 2, "HIS", "CA", "C", (3.8, 0, 0)), a("A|1", 2, "HIS", "NE2", "N", (5.0, 1.0, 0)),
            a("B|1", 3, "LEU", "CA", "C", (5.0, 4.0, 0)),
            a("Z|1", 900, "ZN", "ZN", "ZN", (6.5, 1.5, 0), polymer=False)]
    asm = pd.DataFrame(rows)
    env = residue_environment(asm, "A|1", [2], {}, lambda c: "ion of interest", {"method": "shrake_rupley",
                                                                                   "n_points": 50})
    assert list(env.index) == [1, 2]
    assert env.loc[2, "Distance to active site (Å)"] == 0.0
    assert env.loc[2, "Distance to Zn (Å)"] == pytest.approx(np.hypot(1.5, 0.5), abs=0.01)
    assert env.loc[1, "Distance to partner protomer (Å)"] > env.loc[2, "Distance to partner protomer (Å)"]
    assert 0 < env.loc[1, "RSA"] <= 1.5
