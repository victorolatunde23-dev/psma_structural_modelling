"""Synthetic predicted models for the Milestone 4 / membrane tests (no network, no gemmi).

Geometry of the synthetic full-length protomer (UniProt numbering, 750 residues):

* 1–19   cytoplasmic tail, below the membrane (z ≈ -22 … -30 Å)
* 20–43  transmembrane helix along +z through the membrane (z ≈ -17 … +17 Å)
* 44–56  stalk rising from the membrane to the ectodomain
* 57–750 ectodomain: a compact cloud of CA atoms centred 45 Å above the membrane

The dimer is chain A plus its image under a 180° rotation about the z axis
through (x=11, y=0): a perfect C2 dimer whose axis is the membrane normal.
"""

from __future__ import annotations

import json

import numpy as np

import fixtures as F

THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "E": "GLU", "H": "HIS", "K": "LYS", "W": "TRP",
         "Y": "TYR", "G": "GLY", "L": "LEU", "S": "SER", "T": "THR", "F": "PHE"}


def uniprot_entry_with_topology() -> dict:
    e = F.uniprot_entry()
    e["features"] = [f for f in e["features"] if f["type"] != "Topological domain"] + [
        {"type": "Topological domain", "location": {"start": {"value": 1}, "end": {"value": 19}},
         "description": "Cytoplasmic"},
        {"type": "Transmembrane", "location": {"start": {"value": 20}, "end": {"value": 43}},
         "description": "Helical; Signal-anchor for type II membrane protein"},
        {"type": "Topological domain", "location": {"start": {"value": 44}, "end": {"value": 750}},
         "description": "Extracellular"},
    ]
    return e


def protomer_ca(seed: int = 7) -> np.ndarray:
    """750 × 3 CA coordinates of the synthetic protomer."""
    rng = np.random.default_rng(seed)
    pts = []
    for u in range(1, 751):
        if u <= 19:
            pts.append([2 + 3 * np.cos(u), 3 * np.sin(u), -22 - 0.4 * u])
        elif u <= 43:
            k = u - 20
            a = np.radians(100 * k)
            pts.append([2.3 * np.cos(a), 2.3 * np.sin(a), -17.25 + 1.5 * k])
        elif u <= 56:
            k = u - 44
            pts.append([0.5 * k, 0.3 * k, 19 + 1.6 * k])
        else:
            # ectodomain: points on a jittered compact spiral (CA-CA spacing ~3.8 Å)
            k = u - 57
            r = 14 * (k / 694) ** (1 / 3) + 4
            th, ph = k * 2.39996, np.arccos(1 - 2 * (k + 0.5) / 694)
            pts.append([-6 + r * np.sin(ph) * np.cos(th), r * np.sin(ph) * np.sin(th), 45 + r * np.cos(ph)])
    xyz = np.array(pts)
    xyz[56:] += rng.normal(scale=0.3, size=(694, 3))
    return xyz


def c2_image(xyz: np.ndarray, x0: float = 11.0) -> np.ndarray:
    """180° rotation about the z axis through (x0, 0)."""
    out = xyz.copy()
    out[:, 0] = 2 * x0 - xyz[:, 0]
    out[:, 1] = -xyz[:, 1]
    return out


def plddt_profile() -> np.ndarray:
    v = np.full(750, 92.0)
    v[:19] = 35.0          # disordered cytoplasmic tail
    v[19:43] = 82.0        # TM helix
    v[43:56] = 55.0        # stalk
    v[540:548] = 74.0      # entrance lid
    return v


def pdb_text(chains: dict[str, np.ndarray], seq: str, plddt: np.ndarray, start: int = 1,
             with_cb: bool = True) -> str:
    """PDB text with CA (+ CB / one side-chain atom for non-Gly) per residue, B = pLDDT."""
    lines, n = [], 1
    for ch, xyz in chains.items():
        for i, p in enumerate(xyz):
            u = start + i
            aa = seq[u - 1]
            rn = THREE.get(aa, "ALA")
            atoms = [("CA", "C", p)]
            if with_cb and aa != "G":
                atoms.append(("CB", "C", p + np.array([0.9, 0.9, 0.6])))
            for name, el, q in atoms:
                lines.append(f"ATOM  {n:5d}  {name:<3s} {rn} {ch}{u:4d}    {q[0]:8.3f}{q[1]:8.3f}{q[2]:8.3f}"
                             f"  1.00{plddt[u - 1]:6.2f}           {el}")
                n += 1
        lines.append("TER")
    return "\n".join(lines) + "\nEND\n"


def pae_matrix(n_res: int, chain_lengths: list[int], low_tm_coupling: bool = True) -> np.ndarray:
    """Residue PAE: 1 Å within domains, 25 Å between TM/tail and ectodomain, 4 Å between chains' ectodomains."""
    tot = sum(chain_lengths)
    pae = np.full((tot, tot), 4.0)
    off = 0
    for L in chain_lengths:
        blk = np.full((L, L), 2.0)
        if low_tm_coupling:
            blk[:56, 56:] = 25.0
            blk[56:, :56] = 25.0
        pae[off:off + L, off:off + L] = blk
        off += L
    np.fill_diagonal(pae, 0.3)
    return pae


def reference_atoms(seed: int = 3):
    """Experimental-like ectodomain chain (UniProt 56–750) in a random frame, as an atom table."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    xyz = protomer_ca()[55:]
    xyz = xyz + rng.normal(scale=0.4, size=xyz.shape)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    xyz = xyz @ q.T + np.array([30.0, -12.0, 5.0])
    rows = []
    for i, p in enumerate(xyz):
        u = 56 + i
        aa = F.USEQ[u - 1]
        rows.append(("A", "A", "1", THREE.get(aa, "ALA"), u, "", i + 1, u, "CA", "C", "", *p, 20.0, 1.0,
                     True, False, True))
        if aa != "G":
            cb = p + q @ np.array([0.9, 0.9, 0.6])
            rows.append(("A", "A", "1", THREE.get(aa, "ALA"), u, "", i + 1, u, "CB", "C", "", *cb, 20.0, 1.0,
                         True, False, True))
    from psma_pipeline.structure_atoms import COLUMNS
    return pd.DataFrame(rows, columns=COLUMNS)


def write_model(tmp, name: str, chains: dict[str, np.ndarray], start: int = 1, pae: np.ndarray | None = None):
    path = tmp / f"{name}.pdb"
    path.write_text(pdb_text(chains, F.USEQ, plddt_profile(), start=start))
    pj = None
    if pae is not None:
        pj = tmp / f"{name}_pae.json"
        pj.write_text(json.dumps([{"predicted_aligned_error": pae.round(2).tolist()}]))
    return path, pj
