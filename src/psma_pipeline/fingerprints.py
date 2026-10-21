"""Ligand interaction fingerprints, sub-pocket mapping, similarity and clustering.

Interaction rules (``pockets`` section of the config; all distance-based)

* **hydrogen bond (possible)** – ligand N/O to a protein donor/acceptor ≤ 3.5 Å
  (no hydrogens or angles; ligand donor/acceptor roles are not assigned).
* **ionic (possible salt bridge)** – ligand O to Lys NZ / Arg NE, NH1, NH2, or
  ligand N to Asp/Glu carboxylate O, ≤ 4.0 Å (ligand charges are not assigned).
* **hydrophobic** – ligand C to a side-chain C of A, V, L, I, M, F, W, P, Y, C ≤ 4.5 Å.
* **aromatic (π-stacking, possible)** – ligand ring centroid to F/Y/W/H ring
  centroid ≤ 6.0 Å. Ligand rings are planar 5/6-membered cycles of the
  distance-derived bond graph.
* **cation–π (possible)** – Lys NZ / Arg CZ to a ligand ring centroid, or a
  ligand N to a protein ring centroid, ≤ 6.0 Å.
* **metal coordination** – ligand N/O/S to a metal ion ≤ 2.8 Å.
* **water-mediated** – a water O within 3.5 Å of both a ligand N/O and a
  protein N/O.
* **contact** – any ligand–protein heavy-atom pair ≤ 4.5 Å.

Fingerprints (per ligand instance, UniProt numbering)

1. binary residue-contact set;
2. distance-weighted residue vector (weight 1 at ≤ ``weight_floor`` Å,
   falling linearly to 0 at ``contact_max``; the maximum over atom pairs);
3. interaction-type set ``"<UniProt>:<type>"``;
4. sub-pocket occupancy (sub-pockets with at least one contacting residue).

Similarity: Jaccard (residue sets), Tanimoto (binary fingerprints), cosine
(weighted vectors). Clustering: hierarchical (several distance thresholds
reported with silhouette diagnostics; no fixed cluster number) and DBSCAN.
Both are deterministic.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .interfaces import ACCEPTORS, ACIDIC, BACKBONE, BASIC, DONORS, HYDROPHOBIC_RES, RINGS, HIS_RING, residue_label
from .status import Status

COVALENT = {"C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07, "F": 0.57, "CL": 1.02, "BR": 1.20,
            "I": 1.39, "B": 0.84, "SE": 1.20}
TYPES = ["hydrogen bond (possible)", "ionic (possible salt bridge)", "hydrophobic", "aromatic (π-stacking, possible)",
         "cation–π (possible)", "metal coordination", "water-mediated", "contact"]


def _has(role: set, resname: str, atom: str) -> bool:
    return (resname, atom) in role or ("*", atom) in role


# ---------------------------------------------------------------------------
# Rings
# ---------------------------------------------------------------------------
def ligand_rings(lig: pd.DataFrame, planarity_tol: float = 0.15) -> list[np.ndarray]:
    """Centroids of planar 5- or 6-membered rings in a ligand (bonds from covalent radii)."""
    if len(lig) < 5:
        return []
    xyz = lig[["x", "y", "z"]].to_numpy()
    el = [str(e).upper() for e in lig["element"]]
    g = nx.Graph()
    g.add_nodes_from(range(len(lig)))
    for i, j in cKDTree(xyz).query_pairs(2.2):
        if np.linalg.norm(xyz[i] - xyz[j]) <= COVALENT.get(el[i], 0.8) + COVALENT.get(el[j], 0.8) + 0.45:
            g.add_edge(i, j)
    out = []
    for cyc in nx.minimum_cycle_basis(g):
        if len(cyc) not in (5, 6):
            continue
        p = xyz[list(cyc)]
        c = p.mean(0)
        _, s, vt = np.linalg.svd(p - c)
        if np.abs((p - c) @ vt[2]).max() <= planarity_tol:
            out.append(c)
    return out


def protein_ring_centroids(prot: pd.DataFrame, include_his: bool = True) -> list[tuple[pd.Series, np.ndarray]]:
    """``[(first ring atom row, centroid)]`` for F, Y, W (6-ring) and H."""
    rings = dict(RINGS, **({"HIS": HIS_RING} if include_his else {}))
    out = []
    sel = prot[prot["resname"].isin(rings)]
    for _, g in sel.groupby(["protomer", "label_asym", "resnum", "icode"], sort=False):
        rn = g["resname"].iloc[0]
        r = g[g["atom"].isin(rings[rn])]
        if len(r) == len(rings[rn]):
            out.append((r.iloc[0], r[["x", "y", "z"]].to_numpy().mean(0)))
    return out


# ---------------------------------------------------------------------------
# One ligand instance
# ---------------------------------------------------------------------------
def ligand_fingerprint(pdb: str, lig: pd.DataFrame, env: pd.DataFrame, cfg: dict, subpockets: dict[str, set[int]],
                       metals: set[str]) -> tuple[pd.DataFrame, dict]:
    """Interaction rows and fingerprint summary for one ligand instance.

    Parameters
    ----------
    lig:
        Atom table of the ligand instance.
    env:
        Atom table of everything else in the (assembly) model: protein,
        waters, ions.
    subpockets:
        ``{name: set of UniProt numbers}`` (already validated).
    metals:
        Component IDs treated as metals.
    """
    cmax, hb, ion = float(cfg["contact_max"]), float(cfg["hbond_max"]), float(cfg["ionic_max"])
    hp, arom, cpi = float(cfg["hydrophobic_max"]), float(cfg["aromatic_centroid_max"]), float(cfg["cation_pi_max"])
    mmax, wb, floor = float(cfg["metal_max"]), float(cfg["water_bridge_max"]), float(cfg["weight_floor"])
    lig = lig.reset_index(drop=True)
    prot = env[env["polymer"] & ~env["water"]].reset_index(drop=True)
    # "home" protomer = the protein protomer with most atom contacts; others are partners.
    # (The ligand's own protomer label belongs to its non-polymer chain and cannot be used.)
    own_protomer = None
    if len(prot) and "protomer" in prot:
        hits = cKDTree(prot[["x", "y", "z"]].to_numpy()).query_ball_point(lig[["x", "y", "z"]].to_numpy(), float(cfg["contact_max"]))
        counts = Counter(prot["protomer"].iloc[j] for h in hits for j in h)
        if counts:
            own_protomer = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    water = env[env["water"]].reset_index(drop=True)
    metal = env[env["resname"].str.upper().isin(metals) & ~env["polymer"]].reset_index(drop=True)
    lx = lig[["x", "y", "z"]].to_numpy()
    rows: list[dict] = []

    def prow(r: pd.Series, la: pd.Series | None, d: float, itype: str, patom: str | None = None) -> dict:
        unp = r["unp"]
        partner = own_protomer is not None and r["protomer"] != own_protomer
        return {"PDB ID": pdb, "Ligand atom": la["atom"] if la is not None else "(ring)",
                "Protein residue (UniProt)": residue_label(r["resname"], unp, r["resnum"], r["icode"]),
                "UniProt number": unp, "Author residue": f"{r['resnum']}{r['icode'] or ''}",
                "Chain": r["chain"], "Protomer": r["protomer"], "Partner protomer": partner,
                "Protein atom": patom if patom is not None else r["atom"], "Distance (Å)": round(float(d), 2),
                "Interaction type": itype,
                "Sub-pockets": ", ".join(sorted(n for n, s in subpockets.items() if pd.notna(unp) and int(unp) in s))}

    if len(prot):
        px = prot[["x", "y", "z"]].to_numpy()
        tree = cKDTree(px)
        for i, near in enumerate(tree.query_ball_point(lx, cmax)):
            la = lig.iloc[i]
            le = str(la["element"]).upper()
            for j in near:
                r = prot.iloc[j]
                d = float(np.linalg.norm(lx[i] - px[j]))
                rn, an = r["resname"], r["atom"]
                rows.append(prow(r, la, d, "contact"))
                if le in ("N", "O") and d <= hb and (_has(DONORS, rn, an) or _has(ACCEPTORS, rn, an)):
                    rows.append(prow(r, la, d, "hydrogen bond (possible)"))
                if d <= ion and ((le == "O" and (rn, an) in BASIC) or (le == "N" and (rn, an) in ACIDIC)):
                    rows.append(prow(r, la, d, "ionic (possible salt bridge)"))
                if (d <= hp and le == "C" and str(r["element"]).upper() == "C" and an not in BACKBONE
                        and rn in HYDROPHOBIC_RES):
                    rows.append(prow(r, la, d, "hydrophobic"))
        # aromatic and cation–π
        lrings = ligand_rings(lig)
        prings = protein_ring_centroids(prot)
        for lc in lrings:
            for r, pc in prings:
                d = float(np.linalg.norm(lc - pc))
                if d <= arom:
                    rows.append(prow(r, None, d, "aromatic (π-stacking, possible)", "(ring)"))
            cat = prot[((prot["resname"] == "LYS") & (prot["atom"] == "NZ")) | ((prot["resname"] == "ARG") & (prot["atom"] == "CZ"))]
            for _, r in cat.iterrows():
                d = float(np.linalg.norm(lc - r[["x", "y", "z"]].to_numpy(dtype=float)))
                if d <= cpi:
                    rows.append(prow(r, None, d, "cation–π (possible)"))
        lig_n = lig[lig["element"].str.upper() == "N"]
        for r, pc in prings:
            for _, la in lig_n.iterrows():
                d = float(np.linalg.norm(la[["x", "y", "z"]].to_numpy(dtype=float) - pc))
                if d <= cpi:
                    rows.append(prow(r, la, d, "cation–π (possible)", "(ring)"))
    metal_atoms = []
    for _, m in metal.iterrows():
        mx = m[["x", "y", "z"]].to_numpy(dtype=float)
        for i, la in lig.iterrows():
            if str(la["element"]).upper() in ("N", "O", "S"):
                d = float(np.linalg.norm(lx[i] - mx))
                if d <= mmax:
                    metal_atoms.append(f"{la['atom']}–{m['resname']}{m['resnum']}")
                    rows.append({"PDB ID": pdb, "Ligand atom": la["atom"],
                                 "Protein residue (UniProt)": f"{m['resname']}{m['resnum']}", "UniProt number": None,
                                 "Author residue": f"{m['resnum']}", "Chain": m["chain"], "Protomer": m["protomer"],
                                 "Partner protomer": False, "Protein atom": m["atom"], "Distance (Å)": round(d, 2),
                                 "Interaction type": "metal coordination", "Sub-pockets": "binuclear zinc region"
                                 if "binuclear zinc region" in subpockets else ""})
    # water-mediated
    if len(water) and len(prot):
        wx = water[["x", "y", "z"]].to_numpy()
        lpol = lig[lig["element"].str.upper().isin(["N", "O"])]
        ppol = prot[prot["element"].str.upper().isin(["N", "O"])].reset_index(drop=True)
        if len(lpol) and len(ppol):
            near_l = cKDTree(lpol[["x", "y", "z"]].to_numpy()).query_ball_point(wx, wb)
            ptree = cKDTree(ppol[["x", "y", "z"]].to_numpy())
            for k, hits in enumerate(near_l):
                if not hits:
                    continue
                for j in ptree.query_ball_point(wx[k], wb):
                    r = ppol.iloc[j]
                    d = float(np.linalg.norm(wx[k] - ppol.loc[j, ["x", "y", "z"]].to_numpy(dtype=float)))
                    rr = prow(r, lpol.iloc[hits[0]], d, "water-mediated")
                    rr["Protein atom"] = f"{r['atom']} via HOH{water.iloc[k]['resnum']}"
                    rows.append(rr)
    df = pd.DataFrame(rows)
    return df, summarise_fingerprint(df, cmax, floor, subpockets, metal_atoms)


def summarise_fingerprint(df: pd.DataFrame, cmax: float, floor: float, subpockets: dict[str, set[int]],
                          metal_atoms: list[str]) -> dict:
    """Binary, weighted, typed and sub-pocket fingerprints from interaction rows."""
    if df.empty:
        return {"residues": set(), "weighted": {}, "typed": set(), "subpockets": dict.fromkeys(subpockets, False),
                "metal_atoms": metal_atoms, "counts": {}, "partner": False}
    prot = df.dropna(subset=["UniProt number"])
    contacts = prot[prot["Interaction type"] == "contact"]
    residues = {int(u) for u in contacts["UniProt number"]}
    weighted: dict[int, float] = {}
    for u, d in zip(contacts["UniProt number"], contacts["Distance (Å)"]):
        w = 1.0 if d <= floor else max(0.0, (cmax - d) / (cmax - floor))
        weighted[int(u)] = max(weighted.get(int(u), 0.0), round(w, 3))
    typed = {f"{int(u)}:{t}" for u, t in zip(prot["UniProt number"], prot["Interaction type"]) if t != "contact"}
    typed |= {f"metal:{m.split('–')[1][:2]}" for m in metal_atoms}
    occ = {name: bool(residues & s) for name, s in subpockets.items()}
    counts = Counter(df["Interaction type"])
    return {"residues": residues, "weighted": weighted, "typed": typed, "subpockets": occ,
            "metal_atoms": metal_atoms, "counts": dict(counts), "partner": bool(prot["Partner protomer"].any())}


# ---------------------------------------------------------------------------
# Similarity and clustering
# ---------------------------------------------------------------------------
def tanimoto(a: set, b: set) -> float | None:
    """Tanimoto coefficient of two binary fingerprints given as sets of on-bits."""
    u = len(a | b)
    return len(a & b) / u if u else None


def cosine(a: dict, b: dict) -> float | None:
    """Cosine similarity of two sparse weighted vectors."""
    keys = set(a) | set(b)
    if not keys:
        return None
    va = np.array([a.get(k, 0.0) for k in keys])
    vb = np.array([b.get(k, 0.0) for k in keys])
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    return float(va @ vb / (na * nb)) if na and nb else None


def similarity_tables(fps: dict[str, dict]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Pairwise similarity table and the typed-Tanimoto matrix (for clustering)."""
    ids = list(fps)
    n = len(ids)
    T = np.eye(n)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = fps[ids[i]], fps[ids[j]]
            jr = tanimoto(a["residues"], b["residues"])
            tt = tanimoto(a["typed"], b["typed"])
            cs = cosine(a["weighted"], b["weighted"])
            so = tanimoto({k for k, v in a["subpockets"].items() if v}, {k for k, v in b["subpockets"].items() if v})
            T[i, j] = T[j, i] = tt if tt is not None else 0.0
            rows.append({"Ligand A": ids[i], "Ligand B": ids[j],
                         "Residue Jaccard": None if jr is None else round(jr, 3),
                         "Interaction Tanimoto": None if tt is None else round(tt, 3),
                         "Weighted cosine": None if cs is None else round(cs, 3),
                         "Sub-pocket Tanimoto": None if so is None else round(so, 3)})
    return pd.DataFrame(rows), T, ids


def cluster_ligands(T: np.ndarray, ids: list[str], cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hierarchical clusters at several thresholds (with diagnostics) and DBSCAN labels.

    Returns (per-ligand cluster table, diagnostics table). No fixed number of
    clusters is imposed; the chosen threshold is configurable and every
    threshold's cluster count and silhouette score are reported.
    """
    n = len(ids)
    if n < 2:
        return (pd.DataFrame({"Ligand instance": ids, "Hierarchical cluster": [1] * n, "DBSCAN cluster": [0] * n}),
                pd.DataFrame())
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    D = 1 - T
    np.fill_diagonal(D, 0)
    D = np.clip((D + D.T) / 2, 0, 1)
    Z = linkage(squareform(D, checks=False), method=cfg.get("hierarchical_method", "average"))
    diag = []
    labels_by_t = {}
    for t in cfg.get("distance_thresholds", [0.5]):
        lab = fcluster(Z, t=float(t), criterion="distance")
        labels_by_t[float(t)] = lab
        k = len(set(lab))
        sil = None
        if 1 < k < n:
            from sklearn.metrics import silhouette_score
            sil = round(float(silhouette_score(D, lab, metric="precomputed")), 3)
        diag.append({"Method": "hierarchical", "Threshold (1 − Tanimoto)": t, "Clusters": k, "Silhouette": sil,
                     "Chosen": float(t) == float(cfg.get("chosen_threshold", 0.5))})
    chosen = labels_by_t.get(float(cfg.get("chosen_threshold", 0.5)), next(iter(labels_by_t.values())))
    from sklearn.cluster import DBSCAN
    chosen_eps = float(cfg.get("dbscan_eps", 0.25))
    db = None
    for eps in sorted({float(e) for e in cfg.get("dbscan_eps_values", [])} | {chosen_eps}):
        lab = DBSCAN(eps=eps, min_samples=int(cfg.get("dbscan_min_samples", 3)), metric="precomputed").fit(D).labels_
        sizes = sorted(np.bincount(lab[lab >= 0]).tolist(), reverse=True) if (lab >= 0).any() else []
        diag.append({"Method": "DBSCAN", "Threshold (1 − Tanimoto)": eps, "Clusters": len(set(lab) - {-1}),
                     "Silhouette": None, "Chosen": eps == chosen_eps, "Noise points": int((lab == -1).sum()),
                     "Largest clusters": ", ".join(map(str, sizes[:5]))})
        if eps == chosen_eps:
            db = lab
    return (pd.DataFrame({"Ligand instance": ids, "Hierarchical cluster": chosen, "DBSCAN cluster": db}),
            pd.DataFrame(diag))


# ---------------------------------------------------------------------------
# Ligand-conditioned ranking inputs
# ---------------------------------------------------------------------------
def zinc_similarity(a_metal_atoms: list[str], b_metal_atoms: list[str]) -> float:
    """1 when both ligands coordinate zinc with the same number of atoms, scaled otherwise; 1 if neither does."""
    na = sum("ZN" in m for m in a_metal_atoms)
    nb = sum("ZN" in m for m in b_metal_atoms)
    if na == nb == 0:
        return 1.0
    return 1 - abs(na - nb) / max(na, nb)


def status_for(fp: dict) -> str:
    return Status.OK.value if fp["residues"] else Status.WARNING.value


__all__: list[Any] = ["ligand_fingerprint", "summarise_fingerprint", "ligand_rings", "tanimoto", "cosine",
                      "similarity_tables", "cluster_ligands", "zinc_similarity", "TYPES"]
