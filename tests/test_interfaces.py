"""Interface contacts, BSA (analytic test complex), comparison metrics and clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from psma_pipeline.interfaces import (analyse_interface, canonical_sets, cluster_interfaces, compare_interfaces,
                                      contact_pairs, interface_geometry, jaccard, lattice_interfaces, residue_set,
                                      rotation_angle)
from psma_pipeline.sasa import buried_surface, sasa_shrake_rupley
from psma_pipeline.structure_atoms import Operator, build_assembly, orthogonalization_matrix, target_protomers

import fixtures as F

CFG = {"contact_cutoff": 4.0, "hbond_max": 3.5, "salt_bridge_max": 4.0, "hydrophobic_max": 4.5,
       "aromatic_centroid_max": 6.5, "neighbourhood": 8.0, "_glycans": ["NAG"],
       "sasa": {"method": "shrake_rupley", "n_points": 60},
       "canonical": {"min_fraction": 0.5, "noncanonical_jaccard": 0.3, "min_bsa_fraction": 0.5},
       "clustering": {"method": "average", "distance_threshold": 0.5},
       "lattice": {"ca_prefilter": 12.0}}


def dimer(offset=9.0):
    a = F.toy_chain_atoms()
    return build_assembly(a, [(["A"], [F.identity_op(), F.twofold(offset)])])


# ---- buried surface area ------------------------------------------------------
def test_bsa_two_spheres_matches_analytic_value():
    """Known test complex: two carbon atoms 4 Å apart. Each expanded sphere
    (R = 1.7 + 1.4 Å) loses a cap of area 2πR(R − d/2)."""
    R, d = 3.1, 4.0
    res = buried_surface(np.array([[0, 0, 0.0]]), ["C"], np.array([[d, 0, 0.0]]), ["C"],
                         "shrake_rupley", 1.4, 4000)
    assert res["bsa"] == pytest.approx(2 * 2 * np.pi * R * (R - d / 2), rel=0.01)
    assert res["sasa_a"] == pytest.approx(4 * np.pi * R ** 2, rel=1e-6)
    assert res["method"] == "shrake_rupley"


def test_bsa_is_zero_for_separated_partners_and_never_negative():
    res = buried_surface(np.array([[0, 0, 0.0]]), ["C"], np.array([[20, 0, 0.0]]), ["C"], "shrake_rupley")
    assert res["bsa"] == 0
    with pytest.raises(ValueError):
        buried_surface(np.zeros((0, 3)), [], np.array([[0, 0, 0.0]]), ["C"], "shrake_rupley")


def test_isolated_atom_sasa():
    assert sasa_shrake_rupley(np.zeros((1, 3)), np.array([1.52]))[0] == pytest.approx(4 * np.pi * 2.92 ** 2)


# ---- contacts and interaction types --------------------------------------------
def _atom(chain, resname, num, atom, el, x, y=0.0, z=0.0):
    return dict(chain=chain, label_asym=chain, entity="1", resname=resname, resnum=num, icode="", label_seq=num,
                unp=float(num), atom=atom, element=el, altloc="", x=x, y=y, z=z, b=20.0, occ=1.0,
                polymer=True, water=False, target=True, copy="1", protomer=f"{chain}|1")


def test_interaction_typing():
    rows = [_atom("A", "ASP", 100, "OD1", "O", 0), _atom("B", "LYS", 200, "NZ", "N", 2.9),
            _atom("A", "LEU", 101, "CD1", "C", 0, 10), _atom("B", "ILE", 201, "CD1", "C", 3.8, 10),
            _atom("A", "LEU", 102, "N", "N", 0, 20), _atom("B", "ALA", 202, "N", "N", 3.0, 20)]
    df = pd.DataFrame(rows)
    c = contact_pairs(df[df.chain == "A"], df[df.chain == "B"], CFG).set_index("res_a")
    assert c.loc["D100", "salt_bridges"] == 1 and c.loc["D100", "hbonds"] == 1
    assert c.loc["L101", "hydrophobic"] == 1 and c.loc["L101", "hbonds"] == 0
    assert c.loc["L102", "hbonds"] == 0              # backbone N–N: two donors, not an H-bond


def test_aromatic_ring_contact():
    ring = [("CG", 0, 0), ("CD1", 1.2, 0.7), ("CD2", -1.2, 0.7), ("CE1", 1.2, 2.1), ("CE2", -1.2, 2.1), ("CZ", 0, 2.8)]
    rows = [_atom("A", "PHE", 10, n, "C", x, y) for n, x, y in ring]
    rows += [_atom("B", "TYR", 20, n, "C", x, y, 3.8) for n, x, y in ring]
    df = pd.DataFrame(rows)
    c = contact_pairs(df[df.chain == "A"], df[df.chain == "B"], CFG)
    assert c["aromatic"].sum() == 1


def test_dimer_interface_from_assembly_operators():
    asm = dimer()
    p = target_protomers(asm)
    assert p == ["A|1", "A|2"]
    row, con, ca = analyse_interface("TOY", asm, p[0], p[1], CFG, {})
    assert row["Residue contacts"] > 0 and row["BSA total (Å²)"] > 0
    assert row["Dimer interface area (Å²)"] == pytest.approx(row["BSA total (Å²)"] / 2, abs=0.1)
    # symmetric homodimer: both sides bury the same area (within Shrake–Rupley sampling noise,
    # because surface points do not rotate with the copy) and use the same residues
    assert row["BSA protomer 1 (Å²)"] == pytest.approx(row["BSA protomer 2 (Å²)"], rel=0.02)
    assert row["Interface residues 1 (UniProt)"] == row["Interface residues 2 (UniProt)"]
    assert 0 < row["Fraction buried 1"] < 1
    assert set(ca) == {"p1", "p2"} and len(ca["p1"]) == 100


def test_zero_contact_dimer_is_flagged():
    row, con, ca = analyse_interface("FAR", dimer(offset=200.0), "A|1", "A|2", CFG, {})
    assert row["Residue contacts"] == 0 and row["Calculation status"] == "WARNING"
    assert "no inter-protomer contacts" in row["Warning"]


# ---- comparison --------------------------------------------------------------------
def test_jaccard_and_canonical_sets():
    assert jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(0.5)
    assert jaccard(set(), set()) is None
    rows = [{"Interface ID": i, "Interface residues 1 (UniProt)": s, "Interface residues 2 (UniProt)": ""}
            for i, s in (("a", "1, 2, 3"), ("b", "2, 3, 4"), ("c", "3, 9"))]
    res, pairs, n = canonical_sets(rows, {"a": {(1, 2)}, "b": {(1, 2)}, "c": set()}, 0.5)
    assert res == {2, 3} and pairs == {(1, 2)} and n == 3
    assert residue_set({"Interface residues 1 (UniProt)": "1, 2", "Interface residues 2 (UniProt)": "2, 5"}) == {1, 2, 5}


def test_interface_geometry_detects_rotation_and_is_assignment_invariant():
    asm = dimer()
    _, _, ref = analyse_interface("R", asm, "A|1", "A|2", CFG, {})
    swapped = {"p1": ref["p2"], "p2": ref["p1"]}               # same dimer, protomers listed the other way
    g0 = interface_geometry(swapped, ref, set(range(44, 144)))
    assert g0["Interface RMSD (Å)"] == pytest.approx(0, abs=1e-6)
    # rotate protomer 2 by 20° about its own axis: orientation change must be ~20°
    th = np.radians(20)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    centre = np.array([9.0, 0, 0])
    moved = {"p1": ref["p1"], "p2": {n: tuple((np.array(v) - centre) @ Rz.T + centre) for n, v in ref["p2"].items()}}
    g = interface_geometry(moved, ref, set(range(44, 144)))
    assert g["Protomer-2 rotation (°)"] == pytest.approx(20, abs=0.5)
    assert g["Interface RMSD (Å)"] > 0.5
    assert rotation_angle(np.eye(3)) == pytest.approx(0)


def test_compare_flags_noncanonical_interface():
    asm = dimer()
    row, con, ca = analyse_interface("A1", asm, "A|1", "A|2", CFG, {})
    from psma_pipeline.interfaces import pair_set
    rows, pairs, cas = [], {}, {}
    for pid in ("A1", "A2", "A3"):
        r = dict(row, **{"Interface ID": pid})
        rows.append(r)
        pairs[pid], cas[pid] = pair_set(con), ca
    odd = dict(row, **{"Interface ID": "ODD", "Interface residues 1 (UniProt)": "300, 301",
                       "Interface residues 2 (UniProt)": "302", "BSA total (Å²)": 300.0})
    rows.append(odd)
    pairs["ODD"] = {(300, 302)}
    per, pair_tab, J, ids, info = compare_interfaces(rows, pairs, cas, CFG, ["A1", "A2", "A3"], "A1")
    per = per.set_index("Interface ID")
    assert per.loc["A2", "Canonical interface similarity"] == 1.0
    assert bool(per.loc["A2", "Noncanonical interface flag"]) is False
    assert bool(per.loc["ODD", "Noncanonical interface flag"]) is True
    assert per.loc["A1", "Interface cluster"] != per.loc["ODD", "Interface cluster"]
    assert J.shape == (4, 4) and len(pair_tab) == 6


def test_clustering_is_deterministic():
    J = np.array([[1, .9, .1], [.9, 1, .2], [.1, .2, 1.0]])
    a = cluster_interfaces(J, {"method": "average", "distance_threshold": 0.5})
    b = cluster_interfaces(J.copy(), {"method": "average", "distance_threshold": 0.5})
    assert a == b and a[0] == a[1] != a[2]


# ---- crystal lattice ---------------------------------------------------------------
def test_lattice_search_finds_the_dimer_in_a_monomeric_asu():
    atoms = F.toy_chain_atoms()
    ops = [F.identity_op("1_555"), F.twofold(9.0, "2_555"), F.twofold(300.0, "3_555")]
    rows, tabs, _ = lattice_interfaces("MONO", atoms, ops, CFG)
    assert [r["Lattice operator"] for r in rows] == ["2_555"]     # far image ignored, identity skipped
    assert rows[0]["Source"] == "crystal lattice"


def test_orthogonalization_matrix_orthorhombic():
    M = orthogonalization_matrix(10, 20, 30, 90, 90, 90)
    assert np.allclose(M, np.diag([10, 20, 30]))


def test_operator_application():
    op = Operator("x", np.diag([-1.0, -1, 1]), np.array([9.0, 0, 0]))
    assert np.allclose(op.apply(np.array([[1.0, 2, 3]])), [[8, -2, 3]])
    assert F.identity_op().is_identity and not op.is_identity
