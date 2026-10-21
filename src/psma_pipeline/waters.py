"""Conserved structural waters near the active site (Milestone 3).

Method
------
1. **Eligible sites**: target chains of X-ray entries with resolution
   ≤ ``waters.max_resolution`` and clashscore ≤ ``waters.max_clashscore``
   (cryo-EM and poorer X-ray structures are listed as NOT APPLICABLE –
   absent waters there say nothing about conservation).
2. **Superposition**: every site is superposed (Kabsch) on a reference site
   using the CA atoms of the active-site residues (``superpose_on:
   active_site``) or of all common residues (``core``).
3. **Region**: waters within ``region_radius`` Å of the site's Zn ions,
   active-site atoms or ligand atoms are collected.
4. **Clustering**: water oxygens of all sites (in the reference frame) are
   clustered by complete linkage at ``cluster_distance`` Å, so every pair in
   a cluster is within that distance. A cluster counts a site at most once.
5. **Per cluster**: fraction of eligible sites containing it, median number
   of protein N/O partners ≤ ``hbond_max`` Å (measured in each site's own
   coordinates), the residues most often contacted (UniProt numbering), Zn
   coordination (≤ ``zn_bridge_max`` Å; "bridging" when both Zn), mean B and
   occupancy.
6. **Recommendation**: *retain for modelling/MD* when the fraction ≥
   ``retain_min_fraction`` and the water is either bound to Zn or has ≥
   ``retain_min_partners`` protein partners; *consider* when the fraction is
   ≥ half that; *remove* otherwise. This is a starting point, not a rule:
   a Zn-bound water/hydroxide is part of the catalytic site and needs a
   metal parameterisation decision (Milestone 5).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .conformations import kabsch
from .interfaces import THREE_TO_ONE
from .status import Status


@dataclass
class WaterSite:
    """Waters and reference atoms of one target chain (deposited coordinates)."""

    pdb: str
    chain: str
    ca: dict[int, np.ndarray]                 # UniProt number -> CA position
    waters: pd.DataFrame                      # resnum, x, y, z, b, occ
    polar: pd.DataFrame                       # protein N/O near the site: unp, resname, atom, x, y, z
    zn: np.ndarray                            # (n, 3) Zn positions of this site
    centre_atoms: np.ndarray                  # active-site / ligand / Zn atom positions defining the region


def site_from_atoms(pdb: str, chain: str, atoms: pd.DataFrame, site_numbers: list[int], radius: float,
                    ligand_xyz: np.ndarray | None = None) -> WaterSite | None:
    """Build a :class:`WaterSite` for one author chain from an ASU atom table.

    Waters and Zn of the whole model are considered and kept when they lie
    within ``radius`` Å of this chain's active-site atoms (so a water
    deposited under another chain ID is not lost).
    """
    ch = atoms[(atoms["chain"] == chain) & atoms["polymer"] & atoms["target"]]
    if ch.empty:
        return None
    ca = {int(r.unp): np.array([r.x, r.y, r.z]) for r in ch[(ch["atom"] == "CA") & ch["unp"].notna()].itertuples()}
    act = ch[ch["unp"].isin(site_numbers)]
    if act.empty:
        return None
    centre = act[["x", "y", "z"]].to_numpy()
    zn_all = atoms[(atoms["resname"].str.upper() == "ZN") & ~atoms["polymer"]][["x", "y", "z"]].to_numpy()
    if len(zn_all):
        d = cKDTree(centre).query(zn_all)[0]
        zn = zn_all[d <= 3.5]
    else:
        zn = np.empty((0, 3))
    parts = [centre, zn] + ([ligand_xyz] if ligand_xyz is not None and len(ligand_xyz) else [])
    centre_all = np.vstack([p for p in parts if len(p)])
    w = atoms[atoms["water"] & (atoms["element"] == "O")]
    if len(w):
        d = cKDTree(centre_all).query(w[["x", "y", "z"]].to_numpy())[0]
        w = w[d <= radius]
    polar = ch[ch["element"].isin(["N", "O"])]
    if len(polar):
        d = cKDTree(centre_all).query(polar[["x", "y", "z"]].to_numpy())[0]
        polar = polar[d <= radius + 4]
    return WaterSite(pdb, chain, ca, w[["resnum", "x", "y", "z", "b", "occ"]].reset_index(drop=True),
                     polar[["unp", "resname", "atom", "x", "y", "z"]].reset_index(drop=True), zn, centre_all)


def _superpose(site: WaterSite, ref: WaterSite, numbers: list[int] | None, min_common: int = 5):
    keys = sorted((set(site.ca) & set(ref.ca)) if numbers is None else set(site.ca) & set(ref.ca) & set(numbers))
    if len(keys) < min_common:
        return None, len(keys), None
    P = np.array([site.ca[k] for k in keys])
    Q = np.array([ref.ca[k] for k in keys])
    R, pm, qm = kabsch(P, Q)
    fit = (P - pm) @ R.T + qm
    rmsd = float(np.sqrt(((fit - Q) ** 2).sum(1).mean()))
    return (lambda X: (X - pm) @ R.T + qm), len(keys), rmsd


def conserved_waters(sites: list[WaterSite], cfg: dict, site_numbers: list[int],
                     reference: tuple[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cluster waters of eligible sites.

    Returns
    -------
    (clusters, members, per_site)
        ``clusters`` – one row per water cluster (ConservedWaters sheet);
        ``members`` – one row per water assigned to a cluster;
        ``per_site`` – superposition and water counts per site.
    """
    if not sites:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    ref = next((s for s in sites if (s.pdb, s.chain) == tuple(reference or ())), sites[0])
    numbers = site_numbers if cfg.get("superpose_on", "active_site") == "active_site" else None
    hb, znb = float(cfg.get("hbond_max", 3.5)), float(cfg.get("zn_bridge_max", 2.5))
    recs, per_site = [], []
    for s in sites:
        tf, n_common, rmsd = _superpose(s, ref, numbers)
        row = {"PDB ID": s.pdb, "Chain": s.chain, "Reference": f"{ref.pdb}/{ref.chain}",
               "Superposition CA": n_common, "Superposition RMSD (Å)": None if rmsd is None else round(rmsd, 2),
               "Waters in region": len(s.waters)}
        if tf is None:
            row["Status"] = Status.WARNING.value
            row["Note"] = "too few common CA atoms for superposition; site skipped"
            per_site.append(row)
            continue
        row["Status"] = Status.OK.value
        per_site.append(row)
        if s.waters.empty:
            continue
        wx = s.waters[["x", "y", "z"]].to_numpy()
        wref = tf(wx)
        ptree = cKDTree(s.polar[["x", "y", "z"]].to_numpy()) if len(s.polar) else None
        for i, w in s.waters.iterrows():
            partners = ptree.query_ball_point(wx[i], hb) if ptree is not None else []
            pres = s.polar.iloc[partners] if partners else s.polar.iloc[0:0]
            zd = np.linalg.norm(s.zn - wx[i], axis=1) if len(s.zn) else np.array([])
            recs.append({"PDB ID": s.pdb, "Chain": s.chain, "Water residue": int(w["resnum"]),
                         "x_ref": wref[i, 0], "y_ref": wref[i, 1], "z_ref": wref[i, 2],
                         "B": w["b"], "Occupancy": w["occ"], "Partners": len(partners),
                         "Partner residues": sorted({f"{THREE_TO_ONE.get(r.resname, 'X')}{int(r.unp)}" for r in pres.itertuples()
                                                     if pd.notna(r.unp)}),
                         "Zn contacts": int((zd <= znb).sum()),
                         "Nearest Zn (Å)": round(float(zd.min()), 2) if len(zd) else None})
    members = pd.DataFrame(recs)
    per_site_df = pd.DataFrame(per_site)
    n_sites = int((per_site_df["Status"] == Status.OK.value).sum()) if len(per_site_df) else 0
    if members.empty:
        return pd.DataFrame(), members, per_site_df
    X = members[["x_ref", "y_ref", "z_ref"]].to_numpy()
    if len(X) == 1:
        labels = np.array([1])
    else:
        from scipy.cluster.hierarchy import fcluster, linkage
        labels = fcluster(linkage(X, method="complete"), t=float(cfg.get("cluster_distance", 1.2)),
                          criterion="distance")
    members["Cluster"] = labels
    rows = []
    rmin, pmin = float(cfg.get("retain_min_fraction", 0.7)), int(cfg.get("retain_min_partners", 2))
    for c, g in members.groupby("Cluster"):
        site_keys = set(zip(g["PDB ID"], g["Chain"]))
        frac = len(site_keys) / n_sites if n_sites else 0.0
        cen = g[["x_ref", "y_ref", "z_ref"]].mean().to_numpy()
        res_counts = Counter(r for lst in g["Partner residues"] for r in lst)
        zn_frac = float((g["Zn contacts"] > 0).mean())
        bridging = float((g["Zn contacts"] >= 2).mean())
        med_p = float(g["Partners"].median())
        zn_bound = zn_frac >= 0.5
        if frac >= rmin and (zn_bound or med_p >= pmin):
            rec = "retain"
        elif frac >= rmin / 2:
            rec = "consider"
        else:
            rec = "remove"
        reason = (f"present in {len(site_keys)}/{n_sites} sites ({frac:.0%}); median {med_p:g} protein partners"
                  + ("; Zn-bound in most sites" if zn_bound else "")
                  + ("; bridges both Zn (catalytic water/hydroxide – needs a metal-model decision)"
                     if bridging >= 0.5 else ""))
        rows.append({"Water cluster": int(c), "Sites with water": len(site_keys), "Eligible sites": n_sites,
                     "Conservation fraction": round(frac, 3),
                     "Centroid x (ref)": round(cen[0], 2), "Centroid y (ref)": round(cen[1], 2),
                     "Centroid z (ref)": round(cen[2], 2),
                     "Median protein partners": med_p,
                     "Frequent partner residues (UniProt)": ", ".join(f"{k} ({v})" for k, v in res_counts.most_common(5)),
                     "Zn-bound fraction": round(zn_frac, 3), "Zn-bridging fraction": round(bridging, 3),
                     "Mean B (Å²)": round(float(pd.to_numeric(g["B"]).mean()), 1),
                     "Mean occupancy": round(float(pd.to_numeric(g["Occupancy"]).mean()), 2),
                     "Recommendation": rec, "Reason": reason,
                     "Entries": ", ".join(sorted({f"{p}/{ch}" for p, ch in site_keys}))})
    clusters = (pd.DataFrame(rows).sort_values(
        ["Conservation fraction", "Zn-bound fraction", "Median protein partners", "Centroid x (ref)",
         "Centroid y (ref)", "Centroid z (ref)"], ascending=[False, False, False, True, True, True])
        .reset_index(drop=True))
    # renumber clusters by conservation (W1 = most conserved), deterministic
    order = {old: f"W{i + 1}" for i, old in enumerate(clusters["Water cluster"])}
    clusters["Water cluster"] = clusters["Water cluster"].map(order)
    members["Cluster"] = members["Cluster"].map(order)
    members["Partner residues"] = members["Partner residues"].map(", ".join)
    return clusters, members, per_site_df


def conserved_presence(members: pd.DataFrame, clusters: pd.DataFrame) -> dict[str, float]:
    """Per entry: fraction of 'retain' clusters present in at least one of its chains."""
    if members.empty or clusters.empty:
        return {}
    keep = set(clusters.loc[clusters["Recommendation"] == "retain", "Water cluster"])
    if not keep:
        return {}
    out = {}
    for pdb, g in members.groupby("PDB ID"):
        out[pdb] = round(len(set(g["Cluster"]) & keep) / len(keep), 3)
    return out
