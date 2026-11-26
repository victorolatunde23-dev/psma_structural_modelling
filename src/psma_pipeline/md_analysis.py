"""Milestone 7: trajectory analysis (notebook 07). Part 1: stability by region.

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
