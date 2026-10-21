"""Superposition, structural variability and the entrance-lid classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from psma_pipeline.conformations import (helicity, kabsch, lid_conformation, lid_entry_summary, rmsd,
                                         structural_variability, two_means_1d)

LID_CFG = {"range": (541, 548), "arene_site": ["W541", "R463", "R511"],
           "known_states": {"3BHX": "closed", "2OOT": "open"},
           "classifier": {"helix_ca_i3_max": 6.0, "helix_fraction_min": 0.6, "min_lid_modelled": 6,
                          "ashman_d_min": 2.0, "min_complete_lids_for_clustering": 4,
                          "cluster_helicity_gap": 0.2, "superposition_exclusion_margin": 2}}


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def make_chains(seed=1):
    rng = np.random.default_rng(seed)
    body = {n: tuple(rng.normal(size=3) * 12) for n in list(range(44, 539)) + list(range(551, 751))}
    helix = {541 + i: (2.3 * np.cos(np.radians(100 * i)), 2.3 * np.sin(np.radians(100 * i)), 1.5 * i)
             for i in range(8)}
    ext = {541 + i: (5 + 3.6 * i, 5, 6) for i in range(8)}

    def chain(lid, angle=0.0):
        R = rot(angle)
        return {n: tuple(np.array(v) @ R.T + 3 + rng.normal(size=3) * 0.3) for n, v in {**body, **lid}.items()}

    ca = {(f"C{i}AA", "A"): chain(helix, 0.3 * i) for i in range(5)}
    ca.update({(f"O{i}AA", "A"): chain(ext, 0.3 * i) for i in range(5)})
    ca[("3BHX", "A")] = chain(helix)
    ca[("2OOT", "A")] = chain(ext)
    dis = chain(helix)
    for n in range(541, 546):
        dis.pop(n)
    ca[("DISO", "A")] = dis
    ca[("NOLD", "A")] = {n: v for n, v in chain(helix).items() if n < 500}
    return ca


def test_kabsch_recovers_rigid_motion():
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(50, 3)) * 10
    P = Q @ rot(1.1).T + np.array([4.0, -2.0, 7.0])
    R, pm, qm = kabsch(P, Q)
    assert rmsd((P - pm) @ R.T + qm, Q) == pytest.approx(0, abs=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_helicity_values():
    helix = {541 + i: (2.3 * np.cos(np.radians(100 * i)), 2.3 * np.sin(np.radians(100 * i)), 1.5 * i)
             for i in range(8)}
    ext = {541 + i: (3.6 * i, 0, 0) for i in range(8)}
    assert helicity(helix, (541, 548), 6.0) == 1.0
    assert helicity(ext, (541, 548), 6.0) == 0.0
    assert helicity({}, (541, 548), 6.0) is None


def test_two_means_is_deterministic_and_bimodal():
    x = np.array([0.1, 0.0, -0.2, 10.0, 10.3, 9.9])
    l1, d1 = two_means_1d(x)
    l2, d2 = two_means_1d(x.copy())
    assert (l1 == l2).all() and d1 == d2 and d1 > 2
    assert len(set(l1[:3])) == 1 and l1[0] != l1[3]


def test_lid_classifier_states_and_rules():
    ca = make_chains()
    df, info = lid_conformation(ca, ("C0AA", "A"), LID_CFG, {("NOLD", "A"): (44, 499)})
    st = df.set_index("PDB ID")["Lid state"]
    assert all(st[f"C{i}AA"] == "closed" for i in range(5))
    assert all(st[f"O{i}AA"] == "open" for i in range(5))
    assert st["DISO"] == "disordered (not modelled)"      # never "open"
    assert st["NOLD"] == "not in construct"
    assert info["ashman_D"] > 2
    cal = df.set_index("PDB ID")
    assert cal.loc["3BHX", "Calibration label (literature)"] == "closed"
    assert pd.isna(cal.loc["3BHX"].get("Vote: reference"))  # calibration never votes for itself
    assert cal.loc["3BHX", "Lid state"] == "closed" and cal.loc["2OOT", "Lid state"] == "open"


def test_lid_entry_summary_mixed_and_low_resolution():
    df_lid = pd.DataFrame([
        {"PDB ID": "X", "Chain": "A", "Lid state": "open", "Lid confidence": "high"},
        {"PDB ID": "X", "Chain": "B", "Lid state": "closed", "Lid confidence": "high"},
        {"PDB ID": "Y", "Chain": "A", "Lid state": "closed", "Lid confidence": "high"},
    ])
    summ = pd.DataFrame({"PDB ID": ["X", "Y"], "Resolution (Å)": [2.0, 4.0]})
    out = lid_entry_summary(df_lid, summ, 3.5).set_index("PDB ID")
    assert out.loc["X", "Entrance lid state"].startswith("mixed")
    assert out.loc["Y", "Entrance lid confidence"].startswith("low (resolution 4.0")


def test_structural_variability_detects_lid_movement():
    ca = make_chains()
    regions = {"entrance lid 541-548": (541, 548)}
    df = structural_variability({k: ca[k] for k in [("C0AA", "A"), ("C1AA", "A"), ("O0AA", "A")]},
                                ("C0AA", "A"), regions, [377, 424], 50).set_index("PDB ID")
    assert df.loc["C0AA", "Global CA RMSD (Å)"] == 0
    assert df.loc["O0AA", "entrance lid 541-548 CA RMSD (Å)"] > 3 > df.loc["C1AA", "entrance lid 541-548 CA RMSD (Å)"]
    assert structural_variability(ca, ("NONE", "A"), regions, []).empty
