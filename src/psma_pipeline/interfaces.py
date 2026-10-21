"""Biological dimer interfaces, crystal-lattice contacts and cross-structure comparison.

Definitions (all configurable under ``interfaces`` in the config)

* **Interface residue** – target residue with any heavy atom within
  ``contact_cutoff`` (4.0 Å) of a heavy atom of the partner protomer.
* **Hydrogen bond** – donor/acceptor N/O pair ≤ ``hbond_max`` (3.5 Å).
  Distance-only (no hydrogens or angles): an upper estimate.
* **Salt bridge** – Asp/Glu carboxylate O to Lys NZ / Arg NE, NH1, NH2
  (± His ND1/NE2) ≤ ``salt_bridge_max`` (4.0 Å).
* **Hydrophobic contact** – side-chain carbons of A, V, L, I, M, F, W, P, Y, C
  ≤ ``hydrophobic_max`` (4.5 Å).
* **Aromatic contact** – ring centroids of F, Y, W (six-membered ring)[, H]
  ≤ ``aromatic_centroid_max`` (6.5 Å).
* **BSA** – SASA(A) + SASA(B) − SASA(A∪B) over protein heavy atoms
  (:mod:`psma_pipeline.sasa`; method recorded per row).

Residue sets are in **UniProt numbering**; target residues without a SIFTS
mapping are counted separately and never mixed in.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .conformations import kabsch, rmsd
from .sasa import buried_surface
from .status import Status
from .structure_atoms import Operator

log = logging.getLogger("psma_pipeline.interfaces")

THREE_TO_ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
                "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
                "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M"}
DONORS = {("*", "N"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"), ("ASN", "ND2"), ("GLN", "NE2"),
          ("HIS", "ND1"), ("HIS", "NE2"), ("LYS", "NZ"), ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"),
          ("TRP", "NE1"), ("CYS", "SG")}
ACCEPTORS = {("*", "O"), ("*", "OXT"), ("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2"),
             ("ASN", "OD1"), ("GLN", "OE1"), ("HIS", "ND1"), ("HIS", "NE2"), ("SER", "OG"),
             ("THR", "OG1"), ("TYR", "OH"), ("MET", "SD")}
ACIDIC = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
BASIC = {("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2")}
BASIC_HIS = {("HIS", "ND1"), ("HIS", "NE2")}
HYDROPHOBIC_RES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "TYR", "CYS", "MSE"}
RINGS = {"PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"], "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
         "TRP": ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"]}
HIS_RING = ["CG", "ND1", "CD2", "CE1", "NE2"]
BACKBONE = {"N", "CA", "C", "O", "OXT"}


def _has(role: set, resname: str, atom: str) -> bool:
    return (resname, atom) in role or ("*", atom) in role


def residue_label(resname: str, unp: Any, resnum: Any, icode: str = "") -> str:
    """``H377`` (UniProt) when mapped, else ``HIS377A`` (author, marked ``auth``)."""
    if pd.notna(unp):
        return f"{THREE_TO_ONE.get(resname, 'X')}{int(unp)}"
    return f"{resname}{resnum}{icode or ''}(auth)"


def _res_key(df: pd.DataFrame) -> pd.Series:
    return df["resname"] + ":" + df["resnum"].astype(str) + ":" + df["icode"].fillna("")


# ---------------------------------------------------------------------------
# Contacts and interaction typing
# ---------------------------------------------------------------------------
def contact_pairs(a: pd.DataFrame, b: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Residue–residue contacts between two protein atom tables (one row per residue pair).

    Columns: ``res_a``/``res_b`` (labels), ``unp_a``/``unp_b``, ``resname_a``/
    ``resname_b``, ``auth_a``/``auth_b``, ``min_dist`` (Å), ``atom_contacts``,
    ``hbonds``, ``salt_bridges``, ``hydrophobic``, ``aromatic`` and
    ``closest_atoms``.
    """
    cut = float(cfg.get("contact_cutoff", 4.0))
    hb, sb = float(cfg.get("hbond_max", 3.5)), float(cfg.get("salt_bridge_max", 4.0))
    hp, ar = float(cfg.get("hydrophobic_max", 4.5)), float(cfg.get("aromatic_centroid_max", 6.5))
    basic = BASIC | (BASIC_HIS if cfg.get("salt_bridge_include_his") else set())
    reach = max(cut, hb, sb, hp)
    A, B = a.reset_index(drop=True), b.reset_index(drop=True)
    if A.empty or B.empty:
        return pd.DataFrame()
    pairs = cKDTree(A[["x", "y", "z"]].to_numpy()).query_ball_tree(cKDTree(B[["x", "y", "z"]].to_numpy()), reach)
    ka, kb = _res_key(A), _res_key(B)
    acc: dict[tuple, dict] = {}
    xa, xb = A[["x", "y", "z"]].to_numpy(), B[["x", "y", "z"]].to_numpy()
    for i, js in enumerate(pairs):
        if not js:
            continue
        ra, aa = A.at[i, "resname"], A.at[i, "atom"]
        for j in js:
            d = float(np.linalg.norm(xa[i] - xb[j]))
            rb, ab = B.at[j, "resname"], B.at[j, "atom"]
            key = (ka[i], kb[j])
            rec = acc.get(key)
            if rec is None:
                rec = acc[key] = {"min_dist": 99.0, "atom_contacts": 0, "hbonds": 0, "salt_bridges": 0,
                                  "hydrophobic": 0, "i": i, "j": j, "closest_atoms": ""}
            if d <= cut:
                rec["atom_contacts"] += 1
            if d < rec["min_dist"]:
                rec["min_dist"], rec["closest_atoms"] = d, f"{aa}–{ab}"
            if d <= hb and ((_has(DONORS, ra, aa) and _has(ACCEPTORS, rb, ab))
                            or (_has(ACCEPTORS, ra, aa) and _has(DONORS, rb, ab))):
                rec["hbonds"] += 1
            if d <= sb and (((ra, aa) in ACIDIC and (rb, ab) in basic) or ((ra, aa) in basic and (rb, ab) in ACIDIC)):
                rec["salt_bridges"] += 1
            if (d <= hp and A.at[i, "element"] == "C" and B.at[j, "element"] == "C"
                    and aa not in BACKBONE and ab not in BACKBONE
                    and ra in HYDROPHOBIC_RES and rb in HYDROPHOBIC_RES):
                rec["hydrophobic"] += 1
    rows = []
    for (k1, k2), rec in acc.items():
        if rec["atom_contacts"] == 0:          # only typed partners beyond the contact cut-off
            continue
        i, j = rec.pop("i"), rec.pop("j")
        rows.append({"res_a": residue_label(A.at[i, "resname"], A.at[i, "unp"], A.at[i, "resnum"], A.at[i, "icode"]),
                     "res_b": residue_label(B.at[j, "resname"], B.at[j, "unp"], B.at[j, "resnum"], B.at[j, "icode"]),
                     "unp_a": A.at[i, "unp"], "unp_b": B.at[j, "unp"],
                     "resname_a": A.at[i, "resname"], "resname_b": B.at[j, "resname"],
                     "auth_a": f"{A.at[i, 'resnum']}{A.at[i, 'icode'] or ''}",
                     "auth_b": f"{B.at[j, 'resnum']}{B.at[j, 'icode'] or ''}", **rec})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["min_dist"] = df["min_dist"].round(2)
    df["aromatic"] = _aromatic_pairs(A, B, df, ar, bool(cfg.get("aromatic_include_his", False)))
    return df.sort_values(["unp_a", "unp_b", "res_a", "res_b"]).reset_index(drop=True)


def _ring_centroids(df: pd.DataFrame, include_his: bool) -> dict[str, np.ndarray]:
    out = {}
    rings = dict(RINGS, **({"HIS": HIS_RING} if include_his else {}))
    for (rn, num, ic), g in df[df["resname"].isin(rings)].groupby(["resname", "resnum", "icode"]):
        sel = g[g["atom"].isin(rings[rn])]
        if len(sel) == len(rings[rn]):
            out[f"{rn}:{num}:{ic or ''}"] = sel[["x", "y", "z"]].to_numpy().mean(0)
    return out


def _aromatic_pairs(A: pd.DataFrame, B: pd.DataFrame, pairs: pd.DataFrame, cutoff: float,
                    include_his: bool) -> list[int]:
    ca, cb = _ring_centroids(A, include_his), _ring_centroids(B, include_his)
    out = []
    for _, r in pairs.iterrows():
        k1 = f"{r['resname_a']}:{_split_auth(r['auth_a'])}"
        k2 = f"{r['resname_b']}:{_split_auth(r['auth_b'])}"
        x, y = ca.get(k1), cb.get(k2)
        out.append(int(x is not None and y is not None and np.linalg.norm(x - y) <= cutoff))
    return out


def _split_auth(auth: str) -> str:
    """``"100A"`` -> ``"100:A"``; ``"100"`` -> ``"100:"`` (matches ring-centroid keys)."""
    s = str(auth)
    num = s.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    return f"{num}:{s[len(num):]}"


# ---------------------------------------------------------------------------
# One interface
# ---------------------------------------------------------------------------
def residue_normalised_b(chain_atoms: pd.DataFrame) -> pd.Series:
    """Within-chain z-score of residue mean B, indexed by UniProt number (mapped residues only)."""
    b = chain_atoms["b"].to_numpy()
    mu, sd = b.mean(), b.std()
    per = chain_atoms.dropna(subset=["unp"]).groupby("unp")["b"].mean()
    return (per - mu) / sd if sd > 0 else per * np.nan


def analyse_interface(pdb: str, asm: pd.DataFrame, p1: str, p2: str, cfg: dict, outliers: dict,
                      model: int = 1, assembly: str | None = None, source: str = "biological assembly",
                      total_sasa: bool = True) -> tuple[dict, pd.DataFrame, dict]:
    """Interface row, residue-contact rows and CA coordinates for one protomer pair.

    Returns
    -------
    (row, contacts, ca)
        ``ca`` = ``{"p1": {unp: xyz}, "p2": {unp: xyz}}`` for later superposition.
    """
    prot = asm[asm["polymer"] & ~asm["water"]]
    A, B = prot[prot["protomer"] == p1], prot[prot["protomer"] == p2]
    first = lambda d, c: d[c].iloc[0] if len(d) else None  # noqa: E731
    row: dict[str, Any] = {
        "PDB ID": pdb, "Model": model, "Source": source, "Assembly": assembly,
        "Protomer 1": p1, "Protomer 2": p2,
        "Chain 1": first(A, "chain"), "Chain 2": first(B, "chain"),
        "Chain (label) 1": first(A, "label_asym"), "Chain (label) 2": first(B, "label_asym"),
        "Copy 1": first(A, "copy"), "Copy 2": first(B, "copy"), "Entity": first(A, "entity"),
    }
    warn: list[str] = []
    contacts = contact_pairs(A, B, cfg)
    if contacts.empty:
        row.update({"Atom contacts": 0, "Residue contacts": 0, "Calculation status": Status.WARNING.value,
                    "Warning": "no inter-protomer contacts: not an interface"})
        return row, contacts, {}
    set1 = set(contacts["unp_a"].dropna().astype(int))
    set2 = set(contacts["unp_b"].dropna().astype(int))
    nonmapped = int(contacts["unp_a"].isna().sum() + contacts["unp_b"].isna().sum())
    sasa_cfg = cfg.get("sasa", {})
    try:
        bsa = buried_surface(A[["x", "y", "z"]].to_numpy(), A["element"], B[["x", "y", "z"]].to_numpy(),
                             B["element"], sasa_cfg.get("method", "auto"), float(sasa_cfg.get("probe", 1.4)),
                             int(sasa_cfg.get("n_points", 100)), total=total_sasa)
    except Exception as ex:  # noqa: BLE001 - recorded, interface still reported
        bsa = None
        warn.append(f"BSA failed: {ex}")
    polar = nonpolar = None
    if bsa:
        ea = np.concatenate([A["element"].to_numpy(), B["element"].to_numpy()])
        ab = np.concatenate([bsa["atom_bsa_a"], bsa["atom_bsa_b"]])
        tot = ab.sum()
        if tot > 0:
            polar = float(ab[np.isin(ea, ["N", "O"])].sum() / tot)
            nonpolar = float(ab[ea == "C"].sum() / tot)
    zb1 = residue_normalised_b(A[A["target"]])
    zb2 = residue_normalised_b(B[B["target"]])
    zvals = [zb1.get(n) for n in set1] + [zb2.get(n) for n in set2]
    zvals = [z for z in zvals if z is not None and pd.notna(z)]
    out_hits = []
    for _, r in contacts.iterrows():
        for side, chain in (("a", row["Chain 1"]), ("b", row["Chain 2"])):
            auth = _split_auth(r[f"auth_{side}"]).split(":")
            o = outliers.get((chain, int(auth[0]), auth[1])) if auth[0].lstrip("-").isdigit() else None
            if o:
                out_hits.append(f"{r[f'res_{side}']}/{chain}: {','.join(sorted(o))}")
    out_hits = sorted(set(out_hits))
    # glycans / ligands / ions near the interface
    near = float(cfg.get("neighbourhood", 8.0))
    iface_atoms = pd.concat([A[A["unp"].isin(set1)], B[B["unp"].isin(set2)]])
    het = asm[~asm["polymer"] & ~asm["water"]]
    near_list: list[str] = []
    if len(het) and len(iface_atoms):
        tree = cKDTree(iface_atoms[["x", "y", "z"]].to_numpy())
        d, _ = tree.query(het[["x", "y", "z"]].to_numpy(), distance_upper_bound=near)
        hits = het[np.isfinite(d)]
        near_list = sorted({f"{r.resname}{r.resnum}/{r.chain}|{r.copy}" for r in hits.itertuples()})
    glycans = [x for x in near_list if x[:3] in cfg.get("_glycans", [])]
    others = [x for x in near_list if x not in glycans]
    if not set1 or not set2:
        warn.append("one side has no UniProt-mapped interface residue")
    if out_hits:
        warn.append(f"{len(out_hits)} interface residues with validation outliers")
    row.update({
        "Atom contacts": int(contacts["atom_contacts"].sum()),
        "Residue contacts": len(contacts),
        "Interface residues 1 (UniProt)": ", ".join(map(str, sorted(set1))),
        "Interface residues 2 (UniProt)": ", ".join(map(str, sorted(set2))),
        "N interface residues 1": len(set1), "N interface residues 2": len(set2),
        "Interface residue count": len(set1 | set2),
        "Non-UniProt interface residues": nonmapped,
        "H-bonds (distance-based)": int(contacts["hbonds"].sum()),
        "Salt bridges": int(contacts["salt_bridges"].sum()),
        "Hydrophobic contacts": int(contacts["hydrophobic"].sum()),
        "Aromatic contacts": int(contacts["aromatic"].sum()),
        "BSA total (Å²)": round(bsa["bsa"], 1) if bsa else None,
        "BSA protomer 1 (Å²)": round(bsa["bsa_a"], 1) if bsa else None,
        "BSA protomer 2 (Å²)": round(bsa["bsa_b"], 1) if bsa else None,
        "Dimer interface area (Å²)": round(bsa["bsa"] / 2, 1) if bsa else None,
        "SASA protomer 1 (Å²)": round(bsa["sasa_a"], 1) if bsa and np.isfinite(bsa["sasa_a"]) else None,
        "SASA protomer 2 (Å²)": round(bsa["sasa_b"], 1) if bsa and np.isfinite(bsa["sasa_b"]) else None,
        "Fraction buried 1": round(bsa["bsa_a"] / bsa["sasa_a"], 3) if bsa and np.isfinite(bsa["sasa_a"]) and bsa["sasa_a"] else None,
        "Fraction buried 2": round(bsa["bsa_b"] / bsa["sasa_b"], 3) if bsa and np.isfinite(bsa["sasa_b"]) and bsa["sasa_b"] else None,
        "Polar fraction of BSA": round(polar, 3) if polar is not None else None,
        "Non-polar fraction of BSA": round(nonpolar, 3) if nonpolar is not None else None,
        "SASA method": bsa["method"] if bsa else None,
        "Interface mean normalised B": round(float(np.mean(zvals)), 2) if zvals else None,
        "Interface max normalised B": round(float(np.max(zvals)), 2) if zvals else None,
        "Validation outliers at interface": "; ".join(out_hits),
        "Glycans near interface": ", ".join(glycans),
        "Ligands/ions near interface": ", ".join(others),
        "Interface conservation": "NOT RUN (variant data: Milestone 3)",
        "Calculation status": (Status.WARNING if warn else Status.OK).value,
        "Warning": "; ".join(warn),
    })
    ca = {}
    for key, df in (("p1", A), ("p2", B)):
        c = df[(df["atom"] == "CA") & df["unp"].notna()]
        ca[key] = {int(u): (x, y, z) for u, x, y, z in zip(c["unp"], c["x"], c["y"], c["z"])}
    contacts.insert(0, "PDB ID", pdb)
    contacts.insert(1, "Source", source)
    contacts.insert(2, "Protomer 1", p1)
    contacts.insert(3, "Protomer 2", p2)
    contacts.insert(4, "Chain 1", row["Chain 1"])
    contacts.insert(5, "Chain 2", row["Chain 2"])
    return row, contacts, ca


# ---------------------------------------------------------------------------
# Crystal-lattice contacts
# ---------------------------------------------------------------------------
def lattice_interfaces(pdb: str, atoms: pd.DataFrame, ops: list[Operator], cfg: dict,
                       min_residue_contacts: int = 5) -> tuple[list[dict], list[pd.DataFrame], dict]:
    """Contacts between each ASU target chain and every lattice image of every target chain.

    The identity operator paired with the same chain is skipped. Symmetry-
    equivalent duplicates (same unordered UniProt contact-pair set) are kept
    once. Returns (rows, contact tables, CA dict keyed by interface id).
    """
    lat = cfg.get("lattice", {})
    pre = float(lat.get("ca_prefilter", 12.0))
    prot = atoms[atoms["polymer"] & atoms["target"] & ~atoms["water"]]
    chains = list(dict.fromkeys(prot["label_asym"]))
    rows, tables, cas, seen = [], [], {}, set()
    for c1 in chains:
        A = prot[prot["label_asym"] == c1].assign(copy="1", protomer=f"{c1}|1")
        caA = A[A["atom"] == "CA"][["x", "y", "z"]].to_numpy()
        if not len(caA):
            continue
        treeA = cKDTree(caA)
        for c2 in chains:
            B0 = prot[prot["label_asym"] == c2]
            caB = B0[B0["atom"] == "CA"][["x", "y", "z"]].to_numpy()
            for op in ops:
                if op.is_identity and c1 == c2:
                    continue
                if not len(caB) or not treeA.query_ball_point(op.apply(caB), pre, return_length=True).any():
                    continue
                B = B0.copy()
                B[["x", "y", "z"]] = op.apply(B0[["x", "y", "z"]].to_numpy())
                B = B.assign(copy=op.name, protomer=f"{c2}|{op.name}")
                asm = pd.concat([A, B], ignore_index=True)
                row, con, ca = analyse_interface(pdb, asm, f"{c1}|1", f"{c2}|{op.name}", cfg, {},
                                                 source="crystal lattice", total_sasa=False)
                if row.get("Residue contacts", 0) < min_residue_contacts:
                    continue
                sig = frozenset(tuple(sorted((int(a), int(b)))) for a, b in
                                zip(con["unp_a"], con["unp_b"]) if pd.notna(a) and pd.notna(b))
                if sig in seen:
                    continue
                seen.add(sig)
                row["Lattice operator"] = op.name
                rows.append(row)
                tables.append(con)
                cas[(pdb, row["Protomer 1"], row["Protomer 2"], "lattice")] = ca
    return rows, tables, cas


# ---------------------------------------------------------------------------
# Cross-structure comparison
# ---------------------------------------------------------------------------
def _parse_set(text: Any) -> set[int]:
    return {int(x) for x in str(text or "").replace(" ", "").split(",") if x.lstrip("-").isdigit()}


def residue_set(row: dict | pd.Series) -> set[int]:
    """Union of UniProt interface residues on both protomers (homodimer interface set)."""
    return _parse_set(row.get("Interface residues 1 (UniProt)")) | _parse_set(row.get("Interface residues 2 (UniProt)"))


def pair_set(contacts: pd.DataFrame) -> set[tuple[int, int]]:
    """Unordered UniProt contact pairs (symmetric homodimer representation)."""
    c = contacts.dropna(subset=["unp_a", "unp_b"])
    return {tuple(sorted((int(a), int(b)))) for a, b in zip(c["unp_a"], c["unp_b"])}


def jaccard(a: set, b: set) -> float | None:
    """|A∩B| / |A∪B|; None when both sets are empty."""
    u = a | b
    return len(a & b) / len(u) if u else None


def canonical_sets(rows: list[dict], contacts: dict[str, set], min_fraction: float
                   ) -> tuple[set[int], set[tuple[int, int]], int]:
    """Residues / contact pairs present in at least ``min_fraction`` of the given interfaces."""
    n = len(rows)
    if n == 0:
        return set(), set(), 0
    rc = Counter(r for row in rows for r in residue_set(row))
    pc = Counter(p for row in rows for p in contacts.get(row["Interface ID"], set()))
    return ({r for r, k in rc.items() if k / n >= min_fraction},
            {p for p, k in pc.items() if k / n >= min_fraction}, n)


def rotation_angle(R: np.ndarray) -> float:
    """Rotation angle (degrees) of a 3×3 rotation matrix."""
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def interface_geometry(ca: dict, ref: dict, iface_res: set[int], min_common: int = 50) -> dict:
    """Interface RMSD and relative-orientation change versus a reference dimer.

    Superpose protomer 1 on the reference protomer 1 (all common CA), apply the
    same transform to protomer 2, then report (i) CA RMSD over the interface
    residues of both protomers and (ii) the rotation/translation needed to
    superpose the moved protomer 2 on the reference protomer 2. Both protomer
    assignments are tried (homodimer symmetry); the lower RMSD is kept.
    """
    best: dict | None = None
    for a, b in (("p1", "p2"), ("p2", "p1")):
        x1, x2, r1, r2 = ca.get(a, {}), ca.get(b, {}), ref.get("p1", {}), ref.get("p2", {})
        common = sorted(set(x1) & set(r1))
        if len(common) < min_common:
            continue
        R, pm, qm = kabsch(np.array([x1[n] for n in common]), np.array([r1[n] for n in common]))
        mv = lambda d: {n: (np.array(v) - pm) @ R.T + qm for n, v in d.items()}  # noqa: E731
        m1, m2 = mv(x1), mv(x2)
        i1 = [n for n in iface_res if n in m1 and n in r1]
        i2 = [n for n in iface_res if n in m2 and n in r2]
        if not i1 or not i2:
            continue
        P = np.array([m1[n] for n in i1] + [m2[n] for n in i2])
        Q = np.array([r1[n] for n in i1] + [r2[n] for n in i2])
        irmsd = rmsd(P, Q)
        c2 = sorted(set(m2) & set(r2))
        R2, pm2, qm2 = kabsch(np.array([m2[n] for n in c2]), np.array([r2[n] for n in c2]))
        res = {"Interface RMSD (Å)": round(irmsd, 2), "Protomer-2 rotation (°)": round(rotation_angle(R2), 1),
               "Protomer-2 translation (Å)": round(float(np.linalg.norm(qm2 - pm2)), 2),
               "Protomer assignment": f"{a}->ref p1", "Interface residues compared": len(i1) + len(i2)}
        if best is None or irmsd < best["Interface RMSD (Å)"]:
            best = res
    return best or {}


def compare_interfaces(rows: list[dict], contacts: dict[str, set], ca: dict, cfg: dict,
                       canonical_ids: list[str], reference_id: str | None
                       ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str], dict]:
    """Similarity of every interface to the canonical interface and to each other.

    Parameters
    ----------
    rows:
        Interface rows (must contain ``Interface ID``).
    contacts:
        ``{Interface ID: unordered UniProt contact-pair set}``.
    ca:
        ``{Interface ID: {"p1": {...}, "p2": {...}}}``.
    canonical_ids:
        Interfaces that define the canonical set (author-defined dimers).
    reference_id:
        Interface used for RMSD / orientation.

    Returns
    -------
    (per-interface table, pairwise table, Jaccard matrix, matrix labels, canonical info)
    """
    can_cfg = cfg.get("canonical", {})
    by_id = {r["Interface ID"]: r for r in rows}
    can_rows = [by_id[i] for i in canonical_ids if i in by_id]
    can_res, can_pairs, n_can = canonical_sets(can_rows, contacts, float(can_cfg.get("min_fraction", 0.5)))
    bsa_med = float(np.nanmedian([r.get("BSA total (Å²)") or np.nan for r in can_rows])) if can_rows else np.nan
    per = []
    for r in rows:
        iid = r["Interface ID"]
        s, p = residue_set(r), contacts.get(iid, set())
        jr, jp = jaccard(s, can_res), jaccard(p, can_pairs)
        rec = (len(s & can_res) / len(can_res)) if can_res else None
        bsa = r.get("BSA total (Å²)")
        flags = []
        if jr is not None and jr < float(can_cfg.get("noncanonical_jaccard", 0.3)):
            flags.append(f"Jaccard to canonical {jr:.2f}")
        if bsa and np.isfinite(bsa_med) and bsa < float(can_cfg.get("min_bsa_fraction", 0.5)) * bsa_med:
            flags.append(f"BSA {bsa:.0f} < {can_cfg.get('min_bsa_fraction', 0.5)}× canonical median {bsa_med:.0f}")
        geo = interface_geometry(ca.get(iid, {}), ca.get(reference_id, {}), can_res) if reference_id else {}
        per.append({"Interface ID": iid, "Canonical interface similarity": round(jr, 3) if jr is not None else None,
                    "Contact-pair similarity": round(jp, 3) if jp is not None else None,
                    "Fraction of canonical interface recovered": round(rec, 3) if rec is not None else None,
                    "Noncanonical interface flag": bool(flags), "Noncanonical reason": "; ".join(flags),
                    **geo})
    ids = [r["Interface ID"] for r in rows]
    sets = [residue_set(r) for r in rows]
    n = len(ids)
    J = np.full((n, n), np.nan)
    pair_rows = []
    for i in range(n):
        for j in range(n):
            ji = jaccard(sets[i], sets[j])
            J[i, j] = np.nan if ji is None else ji
            if i < j:
                pj = jaccard(contacts.get(ids[i], set()), contacts.get(ids[j], set()))
                pair_rows.append({"Interface A": ids[i], "Interface B": ids[j],
                                  "Residue Jaccard": None if ji is None else round(ji, 3),
                                  "Contact-pair Jaccard": None if pj is None else round(pj, 3),
                                  "Fraction of A recovered in B": round(len(sets[i] & sets[j]) / len(sets[i]), 3)
                                  if sets[i] else None})
    clusters = cluster_interfaces(J, cfg.get("clustering", {}))
    per_df = pd.DataFrame(per)
    if len(per_df):
        per_df["Interface cluster"] = clusters
        main = Counter(clusters).most_common(1)[0][0] if len(clusters) else None
        per_df["In main cluster"] = per_df["Interface cluster"] == main
    info = {"canonical residues": sorted(can_res), "canonical contact pairs": len(can_pairs),
            "interfaces defining canonical": n_can, "canonical BSA median (Å²)": None if np.isnan(bsa_med) else round(bsa_med, 1),
            "reference interface": reference_id}
    return per_df, pd.DataFrame(pair_rows), J, ids, info


def cluster_interfaces(J: np.ndarray, cfg: dict) -> list[int]:
    """Hierarchical clustering on 1 − Jaccard (no fixed cluster number; distance threshold)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    n = len(J)
    if n == 0:
        return []
    if n == 1:
        return [1]
    D = 1 - np.nan_to_num(J, nan=0.0)
    np.fill_diagonal(D, 0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method=cfg.get("method", "average"))
    return fcluster(Z, t=float(cfg.get("distance_threshold", 0.5)), criterion="distance").tolist()


__all__ = ["contact_pairs", "analyse_interface", "lattice_interfaces", "compare_interfaces",
           "cluster_interfaces", "interface_geometry", "jaccard", "residue_set", "pair_set",
           "canonical_sets", "residue_normalised_b", "rotation_angle"]
