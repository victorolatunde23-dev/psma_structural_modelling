"""Superposition, structural variability and the entrance-lid classifier.

The CA dictionaries used here are ``{(PDB ID, author chain): {UniProt number:
(x, y, z)}}``, so every comparison is made in UniProt numbering. The same
functions will be reused for AlphaFold models (Milestone 4) and trajectory
frames (Milestone 7).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from .gemmi_utils import unp_maps

CADict = dict[tuple[str, str], dict[int, tuple[float, float, float]]]


# ---------------------------------------------------------------------------
# Superposition and CA extraction
# ---------------------------------------------------------------------------
def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotation ``R`` and centroids ``(pm, qm)`` superposing P onto Q (N×3 each).

    Apply as ``(P - pm) @ R.T + qm``. Reflections are excluded.
    """
    pm, qm = P.mean(0), Q.mean(0)
    H = (P - pm).T @ (Q - qm)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, pm, qm


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square deviation between two N×3 arrays (no fitting)."""
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def ca_coordinates(pdb: str, ctx: dict) -> CADict:
    """``{(pdb, author chain): {UniProt number: (x, y, z)}}`` for target CA atoms.

    The first alternate location of each CA is used.
    """
    coords = ctx["coords"]
    model = coords["model"]
    l2u = unp_maps(ctx)
    out: CADict = {}
    for sub in coords["target_subs"]:
        auth = coords["auth_of_sub"].get(sub)
        if auth is None:
            continue
        d = {}
        for r in model[auth]:
            if r.subchain != sub or r.label_seq not in l2u.get(sub, {}):
                continue
            ca = r.find_atom("CA", "*")
            if ca is not None:
                d[l2u[sub][r.label_seq]] = (ca.pos.x, ca.pos.y, ca.pos.z)
        if d:
            out[(pdb, auth)] = d
    return out


def structural_variability(ca: CADict, reference: tuple[str, str], regions: dict,
                           site_numbers: list[int], min_common: int = 50) -> pd.DataFrame:
    """Global and per-region CA RMSD of every chain against ``reference``.

    Each chain is superposed on the reference using all common CA atoms;
    region RMSDs are measured after that global fit, so they report local
    movement relative to the rest of the protein. Chains sharing fewer than
    ``min_common`` CA atoms with the reference are skipped.
    """
    ref = ca.get(reference)
    if ref is None:
        return pd.DataFrame()
    regions = dict(regions)
    if site_numbers:
        regions["active-site residues"] = sorted(site_numbers)
    rows = []
    for key, d in ca.items():
        common = sorted(set(d) & set(ref))
        if len(common) < min_common:
            continue
        P = np.array([d[n] for n in common])
        Q = np.array([ref[n] for n in common])
        R, pm, qm = kabsch(P, Q)
        fit = (P - pm) @ R.T + qm
        dev = np.linalg.norm(fit - Q, axis=1)
        dev_by_n = dict(zip(common, dev))
        row = {"PDB ID": key[0], "Chain": key[1], "Model": 1,
               "Reference": f"{reference[0]}/{reference[1]}",
               "Common CA": len(common),
               "Global CA RMSD (Å)": round(float(np.sqrt((dev ** 2).mean())), 2),
               "Largest deviation": f"{common[int(dev.argmax())]} ({dev.max():.1f} Å)"}
        for name, span in regions.items():
            nums = range(span[0], span[1] + 1) if isinstance(span, tuple) else span
            ds = [dev_by_n[n] for n in nums if n in dev_by_n]
            row[f"{name} CA RMSD (Å)"] = round(float(np.sqrt(np.mean(np.square(ds)))), 2) if ds else None
            row[f"{name} residues compared"] = len(ds)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Global CA RMSD (Å)")


# ---------------------------------------------------------------------------
# Entrance-lid classifier
# ---------------------------------------------------------------------------
def helicity(d: dict, lid: tuple[int, int], helix_ca_i3_max: float) -> float | None:
    """Fraction of CA(i)–CA(i+3) distances in the lid below ``helix_ca_i3_max`` Å."""
    pairs = [(n, n + 3) for n in range(lid[0], lid[1] - 2) if n in d and n + 3 in d]
    if not pairs:
        return None
    dist = [float(np.linalg.norm(np.subtract(d[a], d[b]))) for a, b in pairs]
    return sum(x < helix_ca_i3_max for x in dist) / len(dist)


def two_means_1d(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Optimal two-cluster split of 1-D data and Ashman's D (D > 2 = clean bimodality).

    Deterministic (exhaustive over sorted split points), so no seed is needed.
    """
    order = np.argsort(x, kind="stable")
    xs = x[order]
    best = None
    for k in range(1, len(xs)):
        a, b = xs[:k], xs[k:]
        ss = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        if best is None or ss < best[0]:
            best = (ss, k)
    labels = np.zeros(len(x), dtype=int)
    labels[order[best[1]:]] = 1
    a, b = x[labels == 0], x[labels == 1]
    var = a.var() + b.var()
    d = float(np.sqrt(2) * abs(a.mean() - b.mean()) / np.sqrt(var)) if var > 0 else float("inf")
    return labels, d


def construct_ranges(df_chains: pd.DataFrame) -> dict[tuple[str, str], tuple[int, int]]:
    """``(PDB ID, author chain) -> (first, last)`` UniProt residue of each construct."""
    out: dict[tuple[str, str], tuple[int, int]] = {}
    if not len(df_chains):
        return out
    for _, r in df_chains.iterrows():
        nums = [int(n) for n in re.findall(r"\d+", str(r.get("UniProt range (SIFTS)", "")))]
        if nums:
            out[(r["PDB ID"], r["Chain (auth)"])] = (min(nums), max(nums))
    return out


def lid_conformation(ca: CADict, reference: tuple[str, str] | None, lid_cfg: dict,
                     ranges: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Classify the entrance lid of every chain as open / closed / disordered.

    Three independent votes; the call is their majority:

    1. **Helicity** – CA(i)–CA(i+3) helicity of the lid (closed = helical).
    2. **Clustering** – lid CA coordinates after superposing each chain on the
       reference using non-lid residues; PCA, then a two-means split on PC1.
       Votes only when the split is clean (Ashman's D ≥ ``ashman_d_min``).
       Clusters are named by mean helicity (or by calibration structures).
    3. **Reference** – nearest calibration structure (lid CA RMSD).
       Calibration structures never vote for themselves.

    Non-votes: ``not in construct``, ``disordered (not modelled)`` (fewer than
    ``min_lid_modelled`` lid residues), ``ambiguous`` (tied votes),
    ``undetermined`` (no vote available).

    Parameters
    ----------
    lid_cfg:
        ``Config.lid`` (``range``, ``known_states``, ``classifier``).

    Returns
    -------
    (table, diagnostics)
    """
    if not lid_cfg or not ca:
        return pd.DataFrame(), {}
    c = lid_cfg.get("classifier", {})
    h_max = float(c.get("helix_ca_i3_max", 6.0))
    h_min = float(c.get("helix_fraction_min", 0.6))
    min_mod = int(c.get("min_lid_modelled", 6))
    d_min = float(c.get("ashman_d_min", 2.0))
    min_complete = int(c.get("min_complete_lids_for_clustering", 4))
    gap = float(c.get("cluster_helicity_gap", 0.2))
    margin = int(c.get("superposition_exclusion_margin", 2))
    known_states = lid_cfg.get("known_states", {})

    lid = tuple(lid_cfg["range"])
    lid_nums = list(range(lid[0], lid[1] + 1))
    ref = ca.get(reference) if reference else None
    rows: dict = {}
    fitted: dict = {}
    vecs: list = []
    complete_keys: list = []

    for key, d in ca.items():
        row: dict[str, Any] = {"PDB ID": key[0], "Chain": key[1], "Model": 1}
        present = [n for n in lid_nums if n in d]
        row["Lid residues modelled"] = f"{len(present)}/{len(lid_nums)}"
        rng = (ranges or {}).get(key)
        if rng and (rng[0] > lid[1] or rng[1] < lid[0]):
            row["Lid state"] = "not in construct"
            rows[key] = row
            continue
        if len(present) < min_mod:
            row["Lid state"] = "disordered (not modelled)"
            rows[key] = row
            continue
        h = helicity(d, lid, h_max)
        row["Lid helicity (CA i,i+3)"] = round(h, 2) if h is not None else None
        row["Vote: helicity"] = None if h is None else ("closed" if h >= h_min else "open")
        if ref:
            common = [n for n in d if n in ref and not (lid[0] - margin <= n <= lid[1] + margin)]
            if len(common) >= 50:
                P = np.array([d[n] for n in common])
                Q = np.array([ref[n] for n in common])
                R, pm, qm = kabsch(P, Q)
                xyz = (np.array([d[n] for n in present]) - pm) @ R.T + qm
                fitted[key] = dict(zip(present, xyz))
                rl = [n for n in present if n in ref]
                if rl:
                    dev = [np.linalg.norm(fitted[key][n] - np.array(ref[n])) for n in rl]
                    row["Lid RMSD to superposition reference (Å)"] = round(
                        float(np.sqrt(np.mean(np.square(dev)))), 2)
                if len(present) == len(lid_nums):
                    vecs.append(xyz.ravel())
                    complete_keys.append(key)
        rows[key] = row

    # ---- vote 2: clustering ------------------------------------------------
    info: dict[str, Any] = {"n_complete_lids": len(vecs), "ashman_D": None, "pc1_variance": None,
                            "reference": f"{reference[0]}/{reference[1]}" if reference else None}
    if len(vecs) >= min_complete:
        X = np.array(vecs)
        Xc = X - X.mean(0)
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        pc1 = Xc @ Vt[0]
        labels, D = two_means_1d(pc1)
        info.update(ashman_D=round(D, 2), pc1_variance=round(float(S[0] ** 2 / (S ** 2).sum()), 2))
        hel = [[rows[k].get("Lid helicity (CA i,i+3)") for k, l in zip(complete_keys, labels) if l == cl]
               for cl in (0, 1)]
        hmean = [np.nanmean([v for v in hh if v is not None]) if any(v is not None for v in hh)
                 else np.nan for hh in hel]
        named_by_helicity = not np.isnan(hmean).any() and abs(hmean[0] - hmean[1]) >= gap
        closed_cluster = int(np.argmax(hmean)) if named_by_helicity else None
        if closed_cluster is None:
            for cl in (0, 1):
                ks = [k for k, l in zip(complete_keys, labels) if l == cl]
                if any(known_states.get(k[0]) == "closed" for k in ks):
                    closed_cluster = cl
        info["clusters_named_by"] = ("helicity" if named_by_helicity else
                                     "calibration structures" if closed_cluster is not None else "unnamed")
        for k, p, l in zip(complete_keys, pc1, labels):
            rows[k]["Lid PC1 score"] = round(float(p), 2)
            if closed_cluster is not None:
                name = "closed" if l == closed_cluster else "open"
                rows[k]["Cluster"] = name
                if D >= d_min:
                    rows[k]["Vote: clustering"] = name

    # ---- vote 3: nearest calibration structure -------------------------------
    known = [(k, known_states[k[0]]) for k in fitted if k[0] in known_states]
    info["calibration chains found"] = ", ".join(f"{k[0]}/{k[1]}={s}" for k, s in known) or "none"
    for key, xyz in fitted.items():
        if key[0] in known_states:
            rows[key]["Calibration label (literature)"] = known_states[key[0]]
            continue
        best = None
        for kk, state in known:
            if kk[0] == key[0]:
                continue
            common = [n for n in xyz if n in fitted[kk]]
            if len(common) < min_mod:
                continue
            r = float(np.sqrt(np.mean([np.sum((xyz[n] - fitted[kk][n]) ** 2) for n in common])))
            rows[key][f"Lid RMSD to {kk[0]}/{kk[1]} ({state}) (Å)"] = round(r, 2)
            if best is None or r < best[0]:
                best = (r, state)
        if best:
            rows[key]["Vote: reference"] = best[1]

    # ---- consensus -------------------------------------------------------------
    for row in rows.values():
        if "Lid state" in row:
            row["Lid confidence"] = ""
            continue
        votes = [row.get(v) for v in ("Vote: helicity", "Vote: clustering", "Vote: reference")]
        votes = [v for v in votes if v]
        if not votes:
            row["Lid state"], row["Lid confidence"] = "undetermined", ""
            continue
        cnt = Counter(votes)
        top, n = cnt.most_common(1)[0]
        if len(cnt) > 1 and cnt.most_common(2)[1][1] == n:
            row["Lid state"], row["Lid confidence"] = "ambiguous", "votes split"
            continue
        row["Lid state"] = top
        row["Lid confidence"] = ("high" if n == len(votes) and len(votes) >= 2 else
                                 "medium" if n > len(votes) / 2 and len(votes) >= 2 else "low (one signal)")
        row["Lid votes"] = f"{n}/{len(votes)}"
    df = pd.DataFrame(rows.values())
    lead = ["PDB ID", "Chain", "Model", "Lid state", "Lid confidence", "Lid votes", "Lid residues modelled",
            "Lid helicity (CA i,i+3)", "Vote: helicity", "Lid PC1 score", "Cluster",
            "Vote: clustering", "Vote: reference", "Calibration label (literature)"]
    df = df[[col for col in lead if col in df] + [col for col in df if col not in lead]]
    return df, info


def lid_entry_summary(df_lid: pd.DataFrame, df_summary: pd.DataFrame,
                      low_confidence_resolution: float = 3.5) -> pd.DataFrame:
    """One lid call per entry; disagreeing chains give ``mixed (...)``; calls
    from structures worse than ``low_confidence_resolution`` Å are downgraded."""
    if not len(df_lid):
        return pd.DataFrame(columns=["PDB ID"])
    res = df_summary.set_index("PDB ID")["Resolution (Å)"] if "Resolution (Å)" in df_summary else {}
    out = []
    for pid, g in df_lid.groupby("PDB ID"):
        states = dict(zip(g["Chain"], g["Lid state"]))
        uniq = set(states.values())
        call = uniq.pop() if len(uniq) == 1 else "mixed (" + "; ".join(
            f"{c}: {s}" for c, s in sorted(states.items())) + ")"
        conf = ", ".join(sorted(set(g["Lid confidence"].dropna()) - {""}))
        r = res.get(pid) if hasattr(res, "get") else None
        if r is not None and pd.notna(r) and r > low_confidence_resolution and call in ("open", "closed"):
            conf = f"low (resolution {r} Å)"
        out.append({"PDB ID": pid, "Entrance lid state": call, "Entrance lid confidence": conf})
    return pd.DataFrame(out)


def lid_ligand_association(df_summary: pd.DataFrame, df_binding: pd.DataFrame, lid_cfg: dict,
                           useq: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add arene-site contact and largest-ligand columns to Summary and
    cross-tabulate lid state against ligand size (tests the literature claim)."""
    arene = [r for r in lid_cfg.get("arene_site", [])
             if int(r[1:]) <= len(useq) and useq[int(r[1:]) - 1] == r[0]]
    c = lid_cfg.get("classifier", {})
    bins = c.get("ligand_size_bins", [-1, 0, 20, 35, 1e9])
    labels = c.get("ligand_size_labels", ["none", "small (≤20)", "medium (21-35)", "large (>35)"])
    df = df_summary.copy()
    if len(df_binding):
        lig = df_binding[df_binding["Class"] == "ligand of interest"].copy()
        if arene and len(lig):
            pat = r"\b(?:" + "|".join(arene) + r")/"
            lig["Arene-site contact"] = lig["Protein residues (UniProt/chain)"].str.contains(pat, regex=True)
            df["Arene-site contact (" + "/".join(arene) + ")"] = df["PDB ID"].map(
                lig.groupby("PDB ID")["Arene-site contact"].any())
        if len(lig):
            df["Largest ligand (atoms)"] = df["PDB ID"].map(lig.groupby("PDB ID")["Atoms"].max())
    tab = pd.DataFrame()
    if "Entrance lid state" in df:
        d = df[df["Entrance lid state"].isin(["open", "closed"])].copy()
        if "Largest ligand (atoms)" in d:
            d["Ligand size"] = pd.cut(d["Largest ligand (atoms)"].fillna(0), bins, labels=labels)
            tab = pd.crosstab(d["Ligand size"], d["Entrance lid state"])
    return df, tab
