"""Binuclear Zn site: coordination geometry and MD-treatment options (Milestone 5, checklist item 11).

The Zn ions come from the experimental template (never predicted). This module
checks their coordination against the configured ligands (``psma_active_site.yaml:
metal_site.zn_ligands``) and distance range (``sanity.zn_ligand_distance_range``),
finds the bridging water/hydroxide, and tabulates the MD treatments the user must
choose between (``preparation.zn_md_treatment``). Nothing here picks a treatment.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .status import Status

DONORS = {"N", "O", "S"}

TREATMENTS = [
    {"Treatment": "nonbonded_12_6_4",
     "Model": "Non-bonded Zn²⁺ with the 12-6-4 Lennard-Jones model (ion–induced dipole term)",
     "Needs": "AMBER force field with 12-6-4 ion parameters and the matching water model; per-atom C4 terms set in tleap/ParmEd",
     "Keeps geometry": "not enforced: ligands can exchange; check the site during equilibration",
     "Bridging water / hydroxide": "explicit water or hydroxide, chosen separately",
     "Effort": "low"},
    {"Treatment": "bonded_mcpb",
     "Model": "Bonded model built with MCPB.py (AmberTools): Zn–ligand bonds, angles and RESP charges",
     "Needs": "AmberTools (conda in Colab) and a QM program for the force constants and charges of the cluster",
     "Keeps geometry": "yes: coordination fixed by bonded terms",
     "Bridging water / hydroxide": "included in the QM cluster (protonation state must be decided first)",
     "Effort": "high"},
    {"Treatment": "cationic_dummy",
     "Model": "Cationic dummy-atom Zn (charge distributed over dummy sites around the ion)",
     "Needs": "published dummy-atom parameters for Zn²⁺ compatible with the chosen force field",
     "Keeps geometry": "partly: favours the parameterised coordination geometry",
     "Bridging water / hydroxide": "explicit, chosen separately",
     "Effort": "medium"},
]


def _code_num(code: str) -> int:
    return int(re.sub(r"\D", "", code))


def zn_coordination(atoms: pd.DataFrame, cfg: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per Zn (site assignment, ligands, distances, status) and one per Zn–donor contact.

    ``atoms`` is a prepared-model atom table: ``chain``, ``resname``, ``resnum`` (UniProt for
    protein), ``atom``, ``element``, ``x/y/z``, ``kind`` (protein | ion | water | glycan).
    """
    ms = cfg.active_site.get("metal_site", {}) or {}
    expected = {name: {_code_num(c) for c in codes} for name, codes in (ms.get("zn_ligands") or {}).items()}
    cutoff = float(cfg.get("contacts.metal", 2.8))
    lo, hi = cfg.get("sanity.zn_ligand_distance_range", [1.8, 2.6])
    bridge_max = float(cfg.get("waters.zn_bridge_max", 2.5))
    zn = atoms[atoms["resname"].str.upper() == "ZN"].reset_index(drop=True)
    don = atoms[atoms["element"].isin(DONORS) & atoms["kind"].isin(["protein", "water"])]
    D = don[["x", "y", "z"]].to_numpy()
    rows, contacts = [], []
    for i, z in zn.iterrows():
        p = np.array([z["x"], z["y"], z["z"]])
        d = np.linalg.norm(D - p, axis=1)
        near = don.assign(dist=d)[d <= cutoff].sort_values("dist")
        prot = near[near["kind"] == "protein"]
        lig_res = {int(r) for r in prot["resnum"]}
        site = max(expected, key=lambda k: len(expected[k] & lig_res)) if expected else ""
        for _, a in near.iterrows():
            contacts.append({"Zn": f"{z['chain']}/ZN{int(z['resnum'])}", "Site": site, "Chain": a["chain"],
                             "Residue": f"{a['resname']}{int(a['resnum'])}" if a["kind"] == "protein" else "HOH",
                             "Atom": a["atom"], "Kind": a["kind"], "Distance (Å)": round(float(a["dist"]), 3),
                             "In range": bool(lo <= a["dist"] <= hi)})
        missing = sorted(expected.get(site, set()) - lig_res)
        # one donor per residue (the closest): the second carboxylate O of a monodentate Asp/Glu is not a bond
        closest = prot.sort_values("dist").drop_duplicates(["chain", "resnum"])
        bad = closest[(closest["dist"] < lo) | (closest["dist"] > hi)]
        st, warn = Status.PASS, []
        if missing:
            st = Status.FAIL
            warn.append("expected ligand(s) not coordinating: " + ", ".join(map(str, missing)))
        if len(bad):
            st = Status.WARNING if st is Status.PASS else st
            warn.append(f"{len(bad)} distance(s) outside {lo}–{hi} Å")
        rows.append({"Zn": f"{z['chain']}/ZN{int(z['resnum'])}", "Chain": z["chain"], "Site": site,
                     "Protein ligands": ", ".join(f"{r.resname}{int(r.resnum)}:{r.atom} {r.dist:.2f}" for r in prot.itertuples()),
                     "Expected ligands": ", ".join(map(str, sorted(expected.get(site, set())))),
                     "Coordination number": int(len(near)), "Waters coordinating": int((near["kind"] == "water").sum()),
                     "Mean ligand distance (Å)": round(float(prot["dist"].mean()), 3) if len(prot) else None,
                     "Status": st.value, "Warning": "; ".join(warn), "_xyz": p})
    tab = pd.DataFrame(rows)
    if len(tab):
        # Zn–Zn distance and bridging water/hydroxide per chain
        zz, bridge = [], []
        wat = atoms[(atoms["kind"] == "water") & (atoms["element"] == "O")]
        W = wat[["x", "y", "z"]].to_numpy()
        for _, r in tab.iterrows():
            same = tab[(tab["Chain"] == r["Chain"]) & (tab["Zn"] != r["Zn"])]
            zz.append(round(float(min(np.linalg.norm(np.stack(same["_xyz"]) - r["_xyz"], axis=1))), 3) if len(same) else None)
            if len(same) and len(W):
                other = np.stack(same["_xyz"])[0]
                ok = (np.linalg.norm(W - r["_xyz"], axis=1) <= bridge_max) & (np.linalg.norm(W - other, axis=1) <= bridge_max)
                bridge.append(int(ok.sum()))
            else:
                bridge.append(0)
        tab["Zn–Zn distance (Å)"] = zz
        tab["Bridging water/hydroxide"] = bridge
        tab = tab.drop(columns=["_xyz"])
    return tab, pd.DataFrame(contacts)


def treatment_options(cfg: Any) -> pd.DataFrame:
    """The MD treatments to choose between; the chosen one (``preparation.zn_md_treatment``) is marked."""
    chosen = cfg.get("preparation.zn_md_treatment")
    df = pd.DataFrame(TREATMENTS)
    df["Chosen"] = df["Treatment"] == chosen
    return df


def treatment_status(cfg: Any) -> tuple[Status, str]:
    chosen = cfg.get("preparation.zn_md_treatment")
    valid = {t["Treatment"] for t in TREATMENTS}
    if chosen is None:
        return Status.FAIL, ("not chosen: set preparation.zn_md_treatment to one of "
                             + ", ".join(sorted(valid)) + " (see MetalTreatmentOptions)")
    if chosen not in valid:
        return Status.FAIL, f"unknown treatment {chosen!r}; choose one of {', '.join(sorted(valid))}"
    return Status.PASS, f"{chosen} chosen; parameters are built in Milestone 6"
