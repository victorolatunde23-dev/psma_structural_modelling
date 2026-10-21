"""Ligand interaction fingerprints, similarity and clustering (synthetic geometry, no gemmi)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from psma_pipeline.fingerprints import (cluster_ligands, cosine, ligand_fingerprint, ligand_rings,
                                        similarity_tables, tanimoto, zinc_similarity)

PCFG = {"contact_max": 4.5, "hbond_max": 3.5, "ionic_max": 4.0, "hydrophobic_max": 4.5,
        "aromatic_centroid_max": 6.0, "cation_pi_max": 6.0, "metal_max": 2.8, "water_bridge_max": 3.5,
        "weight_floor": 2.5}


def atom(resname, resnum, name, el, xyz, unp=None, polymer=True, water=False, chain="A", protomer="A|1"):
    return {"chain": chain, "label_asym": chain, "entity": "1", "resname": resname, "resnum": resnum, "icode": "",
            "label_seq": resnum, "unp": unp, "atom": name, "element": el, "altloc": "",
            "x": xyz[0], "y": xyz[1], "z": xyz[2], "b": 20.0, "occ": 1.0, "polymer": polymer, "water": water,
            "target": polymer, "copy": "1", "protomer": protomer}


def hexagon(centre, names, z=0.0, r=1.39):
    ang = np.radians(np.arange(6) * 60)
    return [(n, (centre[0] + r * np.cos(a), centre[1] + r * np.sin(a), z)) for n, a in zip(names, ang)]


def toy_site():
    prot = [atom("PHE", 209, n, "C", p, unp=209) for n, p in hexagon((10, 0), ["CG", "CD1", "CE1", "CZ", "CE2", "CD2"])]
    prot += [atom("ARG", 210, "NH1", "N", (0, 0, 0), unp=210), atom("ARG", 210, "CZ", "C", (0, -1.3, 0), unp=210)]
    prot += [atom("ALA", 500, "CB", "C", (40, 40, 40), unp=500)]
    het = [atom("ZN", 801, "ZN", "ZN", (0, 2.0, 2.9), polymer=False, chain="C"),
           atom("HOH", 901, "O", "O", (-3.0, 0, 1.5), polymer=False, water=True, chain="W")]
    lig = [atom("LIG", 701, n, "C", p, polymer=False, chain="L")
           for n, p in hexagon((10, 0), ["C1", "C2", "C3", "C4", "C5", "C6"], z=3.7)]
    lig += [atom("LIG", 701, "O1", "O", (0, 0, 2.9), polymer=False, chain="L")]
    return pd.DataFrame(lig), pd.DataFrame(prot + het)


def test_ligand_rings_detects_benzene_only():
    lig, _ = toy_site()
    rings = ligand_rings(lig)
    assert len(rings) == 1
    assert np.allclose(rings[0], [10, 0, 3.7], atol=1e-6)


def test_fingerprint_interaction_types_and_subpockets():
    lig, env = toy_site()
    sp = {"S1' glutamate pocket": {209, 210}, "other": {500}}
    tab, fp = ligand_fingerprint("TEST", lig, env, PCFG, sp, {"ZN"})
    types = set(tab["Interaction type"])
    for t in ("contact", "hydrogen bond (possible)", "ionic (possible salt bridge)", "hydrophobic",
              "aromatic (π-stacking, possible)", "metal coordination", "water-mediated"):
        assert t in types, t
    assert fp["residues"] == {209, 210}
    assert fp["subpockets"] == {"S1' glutamate pocket": True, "other": False}
    assert "210:ionic (possible salt bridge)" in fp["typed"]
    assert any("ZN" in m for m in fp["metal_atoms"])
    assert fp["weighted"][210] == pytest.approx((4.5 - 2.9) / 2.0, abs=1e-3)
    assert fp["partner"] is False


def test_partner_protomer_is_relative_to_the_most_contacted_protein_chain():
    lig, env = toy_site()
    lig = lig.assign(protomer="L|1")                     # ligand chain label differs from every protein chain
    partner = pd.DataFrame([atom("LEU", 680, "CD1", "C", (10, 0, 7.5), unp=680, chain="B", protomer="B|2")])
    tab, fp = ligand_fingerprint("TEST", lig, pd.concat([env, partner], ignore_index=True), PCFG, {"x": {209}}, {"ZN"})
    prot = tab.dropna(subset=["UniProt number"])
    assert set(prot.loc[prot["Partner protomer"], "Protomer"]) == {"B|2"}
    assert not prot.loc[prot["Protomer"] == "A|1", "Partner protomer"].any()
    assert fp["partner"] is True
    _, fp2 = ligand_fingerprint("TEST", lig, env, PCFG, {"x": {209}}, {"ZN"})
    assert fp2["partner"] is False


def test_fingerprint_empty_environment():
    lig, _ = toy_site()
    env = pd.DataFrame([atom("ALA", 1, "CB", "C", (100, 100, 100), unp=1)])
    tab, fp = ligand_fingerprint("TEST", lig, env, PCFG, {"x": {1}}, {"ZN"})
    assert tab.empty and fp["residues"] == set()


def test_similarity_metrics():
    assert tanimoto({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    assert tanimoto(set(), set()) is None
    assert cosine({1: 1.0}, {1: 2.0}) == pytest.approx(1.0)
    assert cosine({1: 1.0}, {2: 1.0}) == pytest.approx(0.0)
    assert zinc_similarity(["O1–ZN801", "O2–ZN802"], ["O1–ZN801"]) == pytest.approx(0.5)
    assert zinc_similarity([], []) == 1.0


def _fp(res, typed):
    return {"residues": set(res), "typed": set(typed), "weighted": {r: 1.0 for r in res},
            "subpockets": {"a": 1 in res}, "metal_atoms": []}


def test_similarity_table_and_clustering_deterministic():
    fps = {"A": _fp([1, 2, 3], ["1:h", "2:h"]), "B": _fp([1, 2, 3], ["1:h", "2:h"]),
           "C": _fp([1, 2, 3], ["1:h", "2:h", "3:h"]), "D": _fp([7, 8], ["7:h"])}
    tab, T, ids = similarity_tables(fps)
    assert len(tab) == 6 and ids == ["A", "B", "C", "D"]
    assert T[0, 1] == 1.0 and T[0, 3] == 0.0
    cfg = {"distance_thresholds": [0.3, 0.5], "chosen_threshold": 0.5, "dbscan_eps": 0.4,
           "dbscan_eps_values": [0.2, 0.4], "dbscan_min_samples": 2}
    cl, diag = cluster_ligands(T, ids, cfg)
    lab = dict(zip(cl["Ligand instance"], cl["Hierarchical cluster"]))
    assert lab["A"] == lab["B"] == lab["C"] != lab["D"]
    assert set(diag["Method"]) == {"hierarchical", "DBSCAN"}
    assert len(diag[diag["Method"] == "DBSCAN"]) == 2 and diag.loc[diag["Method"] == "DBSCAN", "Chosen"].sum() == 1
    cl2, _ = cluster_ligands(T, ids, cfg)
    assert cl.equals(cl2)
