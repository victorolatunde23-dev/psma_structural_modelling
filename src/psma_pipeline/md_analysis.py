"""Milestone 7: trajectory analysis (notebook 07). Part 1: stability by region; part 2: orientation, lid, zinc site.

Reads what notebook 06 writes in ``outputs/md_run/``: ``solute.pdb`` (protein, Zn site, glycans) and the DCD
files in ``traj/`` (one per finished segment, listed in ``progress.json``). Only numpy is needed, so the analysis
runs in Colab's own Python on a CPU runtime, alongside a production run.

The full-length dimer is a hybrid (milestone 5): residues 1–61 of each chain come from AlphaFold (membrane anchor,
mean pLDDT ~44), and residues 62–750 from the 9S7U crystal structure. Regions are defined in ``REGIONS`` (UniProt
numbering). RMSDs of flexible regions are measured after superposing on the crystal ectodomain, so they show how
those regions move *relative to* the well-determined part.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# UniProt numbering, per chain. Junction AF2 -> 9S7U at 61/62 (HybridSegments).
REGIONS = {
    "cytoplasmic tail (1–19)": (1, 19),
    "transmembrane helix (20–43)": (20, 43),
    "stalk (44–61)": (44, 61),
    "anchor, AlphaFold (1–61)": (1, 61),
    "ectodomain, crystal (62–750)": (62, 750),
    "entrance lid (541–548)": (541, 548),
}
FIT_REGION = "ectodomain, crystal (62–750)"


# ---------------------------------------------------------------------------
# DCD (CHARMM/OpenMM) reader: Fortran unformatted records, little-endian
# ---------------------------------------------------------------------------
def _record(fh) -> bytes | None:
    head = fh.read(4)
    if len(head) < 4:
        return None
    (n,) = struct.unpack("<i", head)
    data = fh.read(n)
    tail = fh.read(4)
    if len(data) < n or len(tail) < 4 or struct.unpack("<i", tail)[0] != n:
        return None                      # truncated (e.g. a segment interrupted while writing)
    return data


def read_dcd(path: str | Path, atoms: np.ndarray | None = None):
    """Yield one frame at a time as an (n, 3) float32 array in Å, for the atom indices ``atoms`` (all if None).
    Incomplete trailing frames are skipped."""
    with open(path, "rb") as fh:
        hdr = _record(fh)
        if hdr is None or hdr[:4] != b"CORD":
            return
        icntrl = struct.unpack("<20i", hdr[4:84])
        has_box = icntrl[10] != 0
        if _record(fh) is None:            # title
            return
        nat = _record(fh)
        if nat is None:
            return
        (natoms,) = struct.unpack("<i", nat)
        while True:
            if has_box and _record(fh) is None:
                return
            xyz = []
            for _ in range(3):
                r = _record(fh)
                if r is None or len(r) != 4 * natoms:
                    return
                xyz.append(np.frombuffer(r, dtype="<f4"))
            frame = np.stack(xyz, axis=1)
            yield frame if atoms is None else frame[atoms]


def write_dcd(path: str | Path, frames: np.ndarray, box: bool = True) -> None:
    """Minimal OpenMM-style DCD writer (tests only)."""
    nframes, natoms, _ = frames.shape
    with open(path, "wb") as fh:
        rec = lambda b: fh.write(struct.pack("<i", len(b)) + b + struct.pack("<i", len(b)))  # noqa: E731
        rec(b"CORD" + struct.pack("<9if", nframes, 0, 1, 0, 0, 0, 0, 0, 0, 0.01)
            + struct.pack("<10i", int(box), 0, 0, 0, 0, 0, 0, 0, 0, 24))
        rec(struct.pack("<i", 2) + b"Created by test".ljust(80) + b"x".ljust(80))
        rec(struct.pack("<i", natoms))
        for f in frames:
            if box:
                rec(struct.pack("<6d", 100.0, 0.0, 100.0, 0.0, 0.0, 100.0))
            for k in range(3):
                rec(np.ascontiguousarray(f[:, k], dtype="<f4").tobytes())


# ---------------------------------------------------------------------------
# topology and file order
# ---------------------------------------------------------------------------
def ca_table(solute_pdb: str | Path, chain_length: int = 750) -> pd.DataFrame:
    """Cα atoms of the solute in file order, with chain (A for the first ``chain_length``, then B) and UniProt
    number (position within the chain). The two chains are full length (1–750) in this model."""
    rows, idx = [], 0
    for line in Path(solute_pdb).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            if line[12:16].strip() == "CA" and line[76:78].strip() != "CA" and line[17:20].strip() != "CA":
                rows.append({"atom": idx, "resname": line[17:20].strip()})
            idx += 1
    df = pd.DataFrame(rows)
    if len(df) != 2 * chain_length:
        raise ValueError(f"expected {2 * chain_length} Cα atoms (two full-length chains), found {len(df)}")
    df["chain"] = np.where(np.arange(len(df)) < chain_length, "A", "B")
    df["uniprot"] = np.arange(len(df)) % chain_length + 1
    return df


def segment_files(md_dir: str | Path, phase: str = "prod") -> list[tuple[float, Path]]:
    """Finished segment files of one phase, in time order, as (start time in ps within the phase, path).
    Only segments listed as done in progress.json are used: a file from an interrupted segment is ignored."""
    md_dir = Path(md_dir)
    if not (md_dir / "progress.json").exists():
        parent = md_dir.parent
        seen = sorted(p.name for p in parent.iterdir()) if parent.exists() else []
        raise FileNotFoundError(
            f"{md_dir / 'progress.json'} not found. This Google account's Drive does not show the MD run: mount the "
            f"account that runs notebook 06 (or that has the shared psma_structural_modelling folder in My Drive). "
            f"Folders found in {parent}: {', '.join(seen) or 'none'}")
    done = set(json.loads((md_dir / "progress.json").read_text()).get("done", []))
    out = []
    for sid in done:
        if not sid.startswith(phase + ":"):
            continue
        f = md_dir / "traj" / f"{sid.replace(':', '_').replace(' ', '')}.dcd"
        m = re.match(r"prod:t(\d+(?:\.\d+)?)$", sid)
        if m:
            t = float(m.group(1))
        elif re.match(r"prod:production:(\d+)$", sid):                 # 1 ns segments before v0.7.33
            t = (int(sid.rsplit(":", 1)[1]) - 1) * 1000.0
        else:                                                            # equilibration: order only
            stage, seg = sid.split(":")[1:]
            num = re.findall(r"\d+(?:\.\d+)?", stage)
            t = (float(num[0]) if num else 0.0) * 1e4 + float(seg)
        if f.exists() and f.stat().st_size > 0:
            out.append((t, f))
    return sorted(out)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotation R and centres (p0, q0) superposing P onto Q: (P - p0) @ R.T + q0 ≈ Q."""
    p0, q0 = P.mean(0), Q.mean(0)
    H = (P - p0).T @ (Q - q0)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T, p0, q0


def rmsd(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))


def region_masks(ca: pd.DataFrame) -> dict[str, np.ndarray]:
    m = {}
    for name, (a, b) in REGIONS.items():
        sel = (ca["uniprot"] >= a) & (ca["uniprot"] <= b)
        m[f"{name}, both chains"] = sel.to_numpy()
        for ch in ("A", "B"):
            m[f"{name}, chain {ch}"] = (sel & (ca["chain"] == ch)).to_numpy()
    return m


def stability(md_dir: str | Path, ca: pd.DataFrame, frame_ps: float = 10.0, stride: int = 1):
    """RMSD per region over the production trajectory (reference: first production frame) and per-residue RMSF.

    - "all Cα, global fit": every Cα superposed on itself (what notebook 06 logs);
    - every region: measured after superposing the frame on the crystal ectodomain (both chains), so flexible
      regions are seen relative to the ectodomain;
    - "ectodomain, chain X (own fit)": each chain's ectodomain superposed on itself (internal change).
    Returns (per-frame table, per-residue RMSF table)."""
    idx = ca["atom"].to_numpy()
    masks = region_masks(ca)
    fit = masks[f"{FIT_REGION}, both chains"]
    own = {ch: masks[f"{FIT_REGION}, chain {ch}"] for ch in ("A", "B")}
    ref = None
    rows, fitted = [], []
    k = 0
    for t0, f in segment_files(md_dir, "prod"):
        for j, x in enumerate(read_dcd(f, idx)):
            if (k % stride) == 0:
                x = x.astype(np.float64)
                if ref is None:
                    ref = x.copy()
                R, p0, q0 = kabsch(x[fit], ref[fit])
                xf = (x - p0) @ R.T + q0
                row = {"time (ns)": (t0 + (j + 1) * frame_ps) / 1000.0,
                       "all Cα, global fit": rmsd((x - x.mean(0)) @ kabsch(x, ref)[0].T + ref.mean(0), ref)}
                for name, m in masks.items():
                    row[name] = rmsd(xf[m], ref[m])
                for ch, m in own.items():
                    R2, a0, b0 = kabsch(x[m], ref[m])
                    row[f"ectodomain, chain {ch} (own fit)"] = rmsd((x[m] - a0) @ R2.T + b0, ref[m])
                rows.append(row)
                fitted.append(xf.astype(np.float32))
            k += 1
    if not rows:
        raise FileNotFoundError(f"no finished production frames under {md_dir}")
    df = pd.DataFrame(rows)
    F = np.stack(fitted)
    rmsf = np.sqrt(np.mean(np.sum((F - F.mean(0)) ** 2, axis=2), axis=0))
    rf = ca[["chain", "uniprot", "resname"]].assign(**{"RMSF (Å)": rmsf})
    return df, rf


def summary(df: pd.DataFrame, last_fraction: float = 0.5) -> pd.DataFrame:
    """Mean / max RMSD per region over the whole run and its last part (is a region still drifting?)."""
    n = len(df)
    tail = df.iloc[int(n * (1 - last_fraction)):]
    out = []
    for c in df.columns:
        if c == "time (ns)":
            continue
        out.append({"Region": c, "mean RMSD (Å)": round(df[c].mean(), 2), "max (Å)": round(df[c].max(), 2),
                    f"mean, last {int(last_fraction * 100)}% (Å)": round(tail[c].mean(), 2),
                    "drift over the run (Å)": round(float(tail[c].mean() - df[c].iloc[: max(1, n // 10)].mean()), 2)})
    return pd.DataFrame(out)


# ===========================================================================
# Part 2: orientation in the membrane, entrance lid, zinc site (one pass over the trajectory)
# ===========================================================================
# The trajectory holds the solute only; the membrane lies in the xy plane of the box (the barostat keeps x and y
# coupled, z free), so its normal is the z axis. Coordinates are therefore used as written (not superposed) for
# the orientation, and superposed per chain for the lid.
TM_AXIS_RANGE = (22, 42)          # helical core of the transmembrane segment used for its axis (Cα)
ZN_LIGAND_CUTOFF = 2.8            # Å: N/O atoms this close to a Zn in the first frame are its ligands
ARENE_TRP = 541                   # Trp541 (arene-binding site, covered or exposed by the lid)


def atom_table(solute_pdb: str | Path, chain_length: int = 750) -> pd.DataFrame:
    """Every solute atom in file order: name, residue name, element, residue counter, chain and UniProt number
    (protein residues; the k-th residue with a Cα is UniProt k within its chain), and the starting coordinates."""
    rows, res, key0 = [], -1, None
    for line in Path(solute_pdb).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        key = (line[17:20], line[21], line[22:27])
        if key != key0:
            res, key0 = res + 1, key
        el = line[76:78].strip().upper() or line[12:16].strip()[:1].upper()
        rows.append({"atom": len(rows), "name": line[12:16].strip(), "resname": line[17:20].strip(), "element": el,
                     "res": res, "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54])})
    df = pd.DataFrame(rows)
    is_ca = (df["name"] == "CA") & (df["element"] == "C")
    prot = df.loc[is_ca, "res"].drop_duplicates().to_numpy()
    if len(prot) != 2 * chain_length:
        raise ValueError(f"expected {2 * chain_length} residues with a Cα (two full-length chains), found {len(prot)}")
    order = {r: k for k, r in enumerate(prot)}
    k = df["res"].map(order)
    df["chain"] = np.where(k.isna(), "", np.where(k < chain_length, "A", "B"))
    df["uniprot"] = (k % chain_length + 1).astype("Int64")
    return df


def _ca(at: pd.DataFrame, chain: str, a: int, b: int) -> np.ndarray:
    s = at[(at["name"] == "CA") & (at["element"] == "C") & (at["chain"] == chain) & at["uniprot"].between(a, b)]
    return s.sort_values("uniprot")["atom"].to_numpy()


def _label(at: pd.DataFrame, i: int) -> str:
    r = at.loc[i]
    return f"{r['chain']}:{r['resname']}{r['uniprot']} {r['name']}" if r["chain"] else f"{r['resname']} {r['name']} (atom {i})"


def _axis(X: np.ndarray) -> np.ndarray:
    """Principal axis of points ordered N→C, pointing from the first to the last."""
    Xc = X - X.mean(0)
    v = np.linalg.svd(Xc, full_matrices=False)[2][0]
    return v if np.dot(v, X[-1] - X[0]) >= 0 else -v


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def zinc_site(at: pd.DataFrame, x0: np.ndarray, cutoff: float = ZN_LIGAND_CUTOFF) -> pd.DataFrame:
    """Zn ions and their ligand atoms (N/O within ``cutoff`` Å in coordinates ``x0``). Each Zn is given to the
    chain whose His377 NE2 / Glu425 OE2 is nearest and named Zn1 (nearer His377, the Zn1 ligand) or Zn2."""
    zn = at.index[at["element"] == "ZN"].to_numpy()
    no = at.index[at["element"].isin(["N", "O"])].to_numpy()
    rows, names = [], {}
    for ch in ("A", "B"):
        h = at.index[(at["chain"] == ch) & (at["uniprot"] == 377) & (at["name"] == "NE2")].to_numpy()
        e = at.index[(at["chain"] == ch) & (at["uniprot"] == 425) & (at["name"].isin(["OE1", "OE2"]))].to_numpy()
        if not len(h) or not len(e):
            continue
        mine = [z for z in zn if min(np.linalg.norm(x0[z] - x0[h[0]]), np.linalg.norm(x0[z] - x0[e]).min()) < 6.0]
        mine.sort(key=lambda z: np.linalg.norm(x0[z] - x0[h[0]]))
        for k, z in enumerate(mine[:2]):
            names[z] = f"{ch}:Zn{k + 1}"
    for z in zn:
        if z not in names:
            names[z] = f"Zn (atom {z})"
        d = np.linalg.norm(x0[no] - x0[z], axis=1)
        for j in np.argsort(d):
            if d[j] > cutoff:
                break
            rows.append({"Zn": names[z], "Zn atom": int(z), "ligand": _label(at, no[j]), "ligand atom": int(no[j]),
                         "start distance (Å)": round(float(d[j]), 2)})
    zs = sorted(names, key=lambda z: names[z])
    for ch in ("A", "B"):
        pair = [z for z in zs if names[z].startswith(f"{ch}:Zn")]
        if len(pair) == 2:
            rows.append({"Zn": f"{ch}:Zn1", "Zn atom": int(pair[0]), "ligand": f"{ch}:Zn2", "ligand atom": int(pair[1]),
                         "start distance (Å)": round(float(np.linalg.norm(x0[pair[0]] - x0[pair[1]])), 2)})
    return pd.DataFrame(rows)


def reference_lid_ca(cfg: Any, pdb: str) -> dict[int, np.ndarray]:
    """Cα coordinates {UniProt number: xyz} of one chain of a reference crystal structure (the M2 lid calibration
    entries 3BHX closed / 2XEG open), mapped to UniProt with SIFTS. The first chain with the whole lid is used.
    Downloads go to the pipeline cache (so they are made once)."""
    from .cache import session_from_config
    from .pdbe import fetch_sifts, sifts_segments
    from .rcsb import load_structure
    from .structure_atoms import atoms_from_gemmi
    from .survey import http_kwargs

    json_kw, dl_kw = http_kwargs(cfg)
    sess = session_from_config(cfg)
    pdb = pdb.upper()
    st = load_structure(sess, pdb, cfg.cache_dir / "mmcif", cfg.endpoint("rcsb_cif"), **dl_kw)
    segs = sifts_segments(fetch_sifts(sess, pdb, cfg.cache_dir, cfg.endpoint("pdbe_sifts"), **json_kw), pdb, cfg.accession)
    a = atoms_from_gemmi(st, 0, segs, {s["entity_id"] for v in segs.values() for s in v})
    a = a[(a["atom"] == "CA") & a["unp"].notna() & a["polymer"]]
    lid = tuple(cfg.lid.get("range", (541, 548)))
    for _, g in a.groupby("label_asym", sort=True):
        d = {int(r.unp): np.array([r.x, r.y, r.z]) for r in g.itertuples()}
        if all(n in d for n in range(lid[0], lid[1] + 1)):
            return d
    raise ValueError(f"{pdb}: no chain with the whole lid {lid[0]}–{lid[1]} modelled")


def site_and_shape(md_dir: str | Path, at: pd.DataFrame, lid_cfg: dict | None = None,
                   references: dict[str, tuple[str, dict[int, np.ndarray]]] | None = None,
                   frame_ps: float = 10.0, stride: int = 1) -> dict[str, pd.DataFrame]:
    """One pass over the production trajectory. Returns tables keyed:

    - ``orientation``: tilt of each transmembrane helix from the membrane normal (z), the angle between the two
      helices, and the tilt and height of the ectodomain dimer above the transmembrane pair;
    - ``lid``: per chain, lid helicity (fraction of Cα i,i+3 < ``helix_ca_i3_max``, the M2 classifier's measure;
      ≥ ``helix_fraction_min`` = helical = closed), distance from the lid to the zinc pair, distance from Trp541's
      ring to the zinc pair, and the lid Cα RMSD to each reference (after superposing the chain's ectodomain,
      lid ± margin excluded, on that reference);
    - ``zinc``: every Zn–ligand distance and the Zn–Zn distance per chain;
    - ``zinc ligands``: the ligand list (picked from the first production frame).
    """
    lid_cfg = lid_cfg or {}
    c = lid_cfg.get("classifier", {})
    lid = tuple(lid_cfg.get("range", (541, 548)))
    h_max = float(c.get("helix_ca_i3_max", 6.0))
    margin = int(c.get("superposition_exclusion_margin", 2))
    references = references or {}
    tm = {ch: _ca(at, ch, *TM_AXIS_RANGE) for ch in "AB"}
    tm_all = np.concatenate([_ca(at, ch, 20, 43) for ch in "AB"])
    ecto_all = np.concatenate([_ca(at, ch, 62, 750) for ch in "AB"])
    lid_ca = {ch: _ca(at, ch, *lid) for ch in "AB"}
    trp = {ch: at.index[(at["chain"] == ch) & (at["uniprot"] == ARENE_TRP)
                        & at["name"].isin(["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"])].to_numpy() for ch in "AB"}
    ref_fit = {}
    for rname, (state, ref) in references.items():
        for ch in "AB":
            idx = _ca(at, ch, 62, 750)
            nums = at.loc[idx, "uniprot"].to_numpy()
            keep = [(i, int(n)) for i, n in zip(idx, nums) if int(n) in ref and not (lid[0] - margin <= n <= lid[1] + margin)]
            lnums = at.loc[lid_ca[ch], "uniprot"].astype(int).tolist()
            if len(keep) >= 50 and all(n in ref for n in lnums):
                ref_fit[(rname, ch)] = (np.array([i for i, _ in keep]), np.array([ref[n] for _, n in keep]),
                                        np.array([ref[n] for n in lnums]), state)
    zn_tab, side = None, None
    orient, lids, zinc = [], [], []
    k = 0
    for t0, f in segment_files(md_dir, "prod"):
        for j, x in enumerate(read_dcd(f)):
            if (k % stride) == 0:
                x = x.astype(np.float64)
                tns = (t0 + (j + 1) * frame_ps) / 1000.0
                if zn_tab is None:
                    zn_tab = zinc_site(at, x)
                    v0 = x[ecto_all].mean(0) - x[tm_all].mean(0)
                    side = 1.0 if v0[2] >= 0 else -1.0          # the side of the membrane the ectodomain is on
                    zpair = {ch: zn_tab.loc[zn_tab["Zn"].str.startswith(f"{ch}:Zn") & zn_tab["ligand"].str.startswith(f"{ch}:Zn"),
                                            ["Zn atom", "ligand atom"]].to_numpy().ravel() for ch in "AB"}
                normal = np.array([0.0, 0.0, side])
                ax = {ch: _axis(x[tm[ch]]) for ch in "AB"}
                v = x[ecto_all].mean(0) - x[tm_all].mean(0)
                orient.append({"time (ns)": tns,
                               "TM helix tilt, chain A (°)": min(_angle(ax["A"], normal), _angle(-ax["A"], normal)),
                               "TM helix tilt, chain B (°)": min(_angle(ax["B"], normal), _angle(-ax["B"], normal)),
                               "angle between TM helices (°)": _angle(ax["A"], ax["B"]),
                               "ectodomain tilt from the membrane normal (°)": _angle(v, normal),
                               "ectodomain height above the TM pair (Å)": float(np.dot(v, normal)),
                               "TM helix separation (Å)": float(np.linalg.norm(x[tm["A"]].mean(0) - x[tm["B"]].mean(0)))})
                row = {"time (ns)": tns}
                for ch in "AB":
                    L = x[lid_ca[ch]]
                    dd = np.linalg.norm(L[3:] - L[:-3], axis=1)
                    row[f"lid helicity, chain {ch}"] = float(np.mean(dd < h_max)) if len(dd) else np.nan
                    if len(zpair.get(ch, [])):
                        zc = x[zpair[ch]].mean(0)
                        row[f"lid to zinc pair, chain {ch} (Å)"] = float(np.linalg.norm(L.mean(0) - zc))
                        if len(trp[ch]):
                            row[f"Trp541 ring to zinc pair, chain {ch} (Å)"] = float(np.linalg.norm(x[trp[ch]].mean(0) - zc))
                    for (rname, cc), (idx, Q, Ql, state) in ref_fit.items():
                        if cc != ch:
                            continue
                        R, p0, q0 = kabsch(x[idx], Q)
                        row[f"lid RMSD to {rname} ({state}), chain {ch} (Å)"] = rmsd((L - p0) @ R.T + q0, Ql)
                lids.append(row)
                zr = {"time (ns)": tns}
                for r in zn_tab.itertuples(index=False):
                    zr[f"{r.Zn} – {r.ligand}"] = float(np.linalg.norm(x[r[1]] - x[r[3]]))
                zinc.append(zr)
            k += 1
    if not orient:
        raise FileNotFoundError(f"no finished production frames under {md_dir}")
    lid_df = pd.DataFrame(lids)
    hmin = float(c.get("helix_fraction_min", 0.6))
    for ch in "AB":
        col = f"lid helicity, chain {ch}"
        lid_df[f"lid state by helicity, chain {ch}"] = np.where(lid_df[col] >= hmin, "closed", "open")
        rc = [cc for cc in lid_df.columns if cc.startswith("lid RMSD to ") and cc.endswith(f"chain {ch} (Å)")]
        if len(rc) >= 2:
            best = lid_df[rc].idxmin(axis=1)
            lid_df[f"nearest reference, chain {ch}"] = best.str.extract(r"\((open|closed)\)")[0]
    return {"orientation": pd.DataFrame(orient), "lid": lid_df, "zinc": pd.DataFrame(zinc), "zinc ligands": zn_tab}


def describe(df: pd.DataFrame, last_fraction: float = 0.5) -> pd.DataFrame:
    """Mean, SD, range, and the mean at the start (first 10 %) and over the last part, for every numeric column."""
    n = len(df)
    head, tail = df.iloc[: max(1, n // 10)], df.iloc[int(n * (1 - last_fraction)):]
    out = []
    for c in df.columns:
        if c == "time (ns)" or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        out.append({"Quantity": c, "mean": round(df[c].mean(), 2), "SD": round(df[c].std(), 2),
                    "min": round(df[c].min(), 2), "max": round(df[c].max(), 2),
                    "start (first 10%)": round(head[c].mean(), 2),
                    f"last {int(last_fraction * 100)}%": round(tail[c].mean(), 2)})
    return pd.DataFrame(out)


def lid_state_summary(lid_df: pd.DataFrame) -> pd.DataFrame:
    """Share of frames with each lid state, per chain and per criterion."""
    out = []
    for c in lid_df.columns:
        if c.startswith(("lid state by helicity", "nearest reference")):
            v = lid_df[c].value_counts(normalize=True)
            out.append({"Criterion": c, "closed (% of frames)": round(100 * v.get("closed", 0.0), 1),
                        "open (% of frames)": round(100 * v.get("open", 0.0), 1)})
    return pd.DataFrame(out)


def zinc_summary(zn_df: pd.DataFrame, ligands: pd.DataFrame, warn_A: float = 0.5) -> pd.DataFrame:
    """Per Zn–ligand distance: start, mean, SD, max, and the share of frames more than ``warn_A`` Å above the start
    (a ligand leaving the zinc shows up here)."""
    out = []
    for r in ligands.itertuples(index=False):
        c = f"{r.Zn} – {r.ligand}"
        s = zn_df[c]
        out.append({"Zn": r.Zn, "ligand": r.ligand, "start (Å)": r[4], "mean (Å)": round(s.mean(), 2),
                    "SD (Å)": round(s.std(), 2), "max (Å)": round(s.max(), 2),
                    f"frames > start + {warn_A} Å (%)": round(100 * float(np.mean(s > r[4] + warn_A)), 1)})
    return pd.DataFrame(out)
