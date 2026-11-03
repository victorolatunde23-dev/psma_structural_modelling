"""Synthetic inputs for the Milestone 5 tests (no network, no gemmi).

* ``USEQ`` – the fixture sequence with two N-glycosylation sequons (N140, N459).
* ``af_model`` – AlphaFold2-like full-length C2 dimer (chains A/B, every residue with a
  complete heavy-atom set for its type).
* ``template_assembly`` – experimental-like dimer of residues 56–750 in a different frame
  (random rotation + 0.3 Å noise) with: a substitution at 200 (Gly for Ala), a missing
  loop 300–305, Asn257 missing OD1/ND2, two Zn per protomer coordinated by the configured
  ligands at ~2.0–2.1 Å, a bridging water, Ca and Cl ions, a ligand (removed), a bulk water,
  a core NAG on chain A Asn140 and an outer NAG (not core).
* ``donor_atoms`` – an ASU-like entry with a core NAG on Asn140 (to transplant onto chain B).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import af_fixtures as A
import fixtures as F
from psma_pipeline.structure_atoms import COLUMNS

USEQ = list(F.USEQ)
for pos, motif in ((140, "NAS"), (459, "NGT")):
    for k, aa in enumerate(motif):
        USEQ[pos - 1 + k] = aa
USEQ = "".join(USEQ)

THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "E": "GLU", "H": "HIS", "K": "LYS", "W": "TRP",
         "Y": "TYR", "G": "GLY", "S": "SER", "T": "THR"}
SIDE = {"ALA": ["CB"], "GLY": [], "SER": ["CB", "OG"], "THR": ["CB", "OG1", "CG2"],
        "ASN": ["CB", "CG", "OD1", "ND2"], "ASP": ["CB", "CG", "OD1", "OD2"],
        "GLU": ["CB", "CG", "CD", "OE1", "OE2"], "HIS": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
        "LYS": ["CB", "CG", "CD", "CE", "NZ"], "ARG": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
        "TRP": ["CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        "TYR": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"]}

# Zn site (UniProt numbers) - matches config/psma_active_site.yaml
ZN1_LIG = [(377, "HIS", "NE2"), (387, "ASP", "OD1"), (453, "ASP", "OD1")]
ZN2_LIG = [(387, "ASP", "OD2"), (425, "GLU", "OE1"), (553, "HIS", "NE2")]


def _element(name: str) -> str:
    return name[0]


def _protomer_atoms(ca: np.ndarray, rng) -> dict[int, list[tuple[str, str, np.ndarray]]]:
    """Residue -> [(atom, element, xyz)] with a simple backbone and outward side chains."""
    centre = ca[56:].mean(0)
    out = {}
    for i, p in enumerate(ca):
        u = i + 1
        rn = THREE[USEQ[i]]
        prev = ca[i - 1] if i else p - (ca[1] - ca[0])
        nxt = ca[i + 1] if i + 1 < len(ca) else p + (p - ca[i - 1])
        e1 = (prev - p) / np.linalg.norm(prev - p)
        e2 = (nxt - p) / np.linalg.norm(nxt - p)
        out_dir = p - centre if u > 56 else np.array([1.0, 0, 0])
        out_dir = out_dir / (np.linalg.norm(out_dir) + 1e-9)
        atoms = [("N", "N", p + 1.46 * e1), ("CA", "C", p.copy()), ("C", "C", p + 1.52 * e2),
                 ("O", "O", p + 1.52 * e2 + np.array([0.0, 0.0, 1.23]))]
        for k, nm in enumerate(SIDE[rn], 1):
            atoms.append((nm, _element(nm), p + out_dir * (1.53 * k) + np.array([0.3 * (k % 2), 0.0, 0.0])))
        out[u] = atoms
    # binuclear Zn site: place ligand atoms at ~2.05 Å from two Zn positions
    zn1 = ca[376] + out_dir_of(ca, 376, centre) * 6.0
    zn2 = zn1 + np.array([3.3, 0.0, 0.0])
    dirs = [np.array(v, float) / np.linalg.norm(v) for v in ([0, 1, 0], [0, -1, 0.3], [0, 0.2, -1])]
    for (u, rn, an), d in zip(ZN1_LIG, dirs):
        _set(out, u, an, zn1 + 2.05 * d)
    for (u, rn, an), d in zip(ZN2_LIG, [np.array([0, 1, 0.2]), np.array([0, -1, 0]), np.array([0.1, 0.1, -1])]):
        _set(out, u, an, zn2 + 2.08 * d / np.linalg.norm(d))
    _set(out, 377, "ND1", zn1 + np.array([0, 4.5, 0]))      # ND1 away from Zn -> HID
    _set(out, 553, "ND1", zn2 + np.array([0, 4.5, 0.5]))
    out[0] = [("ZN1", "ZN", zn1), ("ZN2", "ZN", zn2)]         # stashed for the template only
    return out


def out_dir_of(ca, i, centre):
    v = ca[i] - centre
    return v / np.linalg.norm(v)


def _set(res: dict, u: int, name: str, xyz: np.ndarray) -> None:
    res[u] = [(n, e, xyz.copy() if n == name else x) for n, e, x in res[u]]


def _rows(chain: str, resd: dict, first: int = 1, last: int = 750, b: float = 20.0, unp: bool = True, plddt=None):
    rows = []
    for u in range(first, last + 1):
        rn = THREE[USEQ[u - 1]]
        for nm, el, xyz in resd[u]:
            rows.append((chain, chain, "1", rn, u, "", u, float(u) if unp else None, nm, el, "", *xyz,
                         plddt[u - 1] if plddt is not None else b, 1.0, True, False, True))
    return rows


def af_dimer_coords():
    a = A.protomer_ca()
    return {"A": a, "B": A.c2_image(a)}


def af_model(tmp_path, cfg):
    """AlphaFold2-like AFModel via a PDB file (pLDDT in B)."""
    from psma_pipeline.alphafold import load_user_model
    from psma_pipeline.status import IssueLog
    rng = np.random.default_rng(0)
    lines, n = [], 1
    pl = A.plddt_profile()
    for ch, ca in af_dimer_coords().items():
        resd = _protomer_atoms(ca, rng)
        for u in range(1, 751):
            rn = THREE[USEQ[u - 1]]
            for nm, el, x in resd[u]:
                lines.append(f"ATOM  {n:5d} {(' ' + nm) if len(nm) < 4 else nm:<4s} {rn} {ch}{u:4d}    "
                             f"{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00{pl[u - 1]:6.2f}           {el}")
                n += 1
        lines.append("TER")
    path = tmp_path / "af2_dimer_rank_001.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")
    m = load_user_model(path, USEQ, cfg, IssueLog(), "af2", plddt_in_b=True)[0]
    m.model_id = "CF:dimer:rank_001"
    m.meta["File"] = str(path)
    return m


def random_frame(seed: int = 11):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q, np.array([25.0, -10.0, 40.0])


def template_assembly() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    q, t = random_frame()
    rows = []
    for k, (ch, ca) in enumerate(af_dimer_coords().items()):
        resd = _protomer_atoms(ca, rng)
        for u in range(56, 751):
            if 300 <= u <= 305:
                continue
            rn = "GLY" if u == 200 else THREE[USEQ[u - 1]]
            for nm, el, xyz in resd[u]:
                if u == 200 and nm not in ("N", "CA", "C", "O"):
                    continue
                if u == 257 and nm in ("OD1", "ND2"):
                    continue
                x = xyz @ q.T + t + rng.normal(scale=0.05, size=3)
                rows.append((ch, ch, "1", rn, u, "", u, float(u), nm, el, "", *x, 25.0, 1.0, True, False, True))
        zn1, zn2 = resd[0][0][2], resd[0][1][2]
        het = [("ZN", 801, "ZN", "ZN", zn1), ("ZN", 802, "ZN", "ZN", zn2),
               ("HOH", 901, "O", "O", (zn1 + zn2) / 2 + np.array([0, 0, 1.4])),          # bridging water
               ("HOH", 902, "O", "O", zn1 + np.array([0, 0, 30.0])),                     # bulk water (dropped)
               ("CA", 803, "CA", "CA", ca[600] + np.array([0, 0, 4.0])),
               ("CL", 804, "CL", "CL", ca[500] + np.array([0, 4.0, 0])),
               ("JB7", 805, "C1", "C", zn1 + np.array([4.0, 0, 0]))]                      # ligand (removed)
        if ch == "A":
            nd2 = dict((n, x) for n, e, x in resd[140])["ND2"]
            het += [("NAG", 701, "C1", "C", nd2 + np.array([1.45, 0, 0])), ("NAG", 701, "O5", "O", nd2 + np.array([2.2, 1.0, 0])),
                    ("NAG", 702, "C1", "C", nd2 + np.array([4.5, 0, 0]))]                 # outer sugar (not core)
        for comp, num, an, el, xyz in het:
            x = xyz @ q.T + t
            rows.append((ch, f"{ch}{comp}{num}", None, comp, num, "", None, None, an, el, "", *x, 30.0, 1.0,
                         False, comp == "HOH", False))
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["copy"] = "1"
    df["protomer"] = df["label_asym"] + "|1"
    df.loc[df["polymer"], "protomer"] = df.loc[df["polymer"], "chain"] + "|1"
    return df


def donor_atoms() -> pd.DataFrame:
    """Monomeric entry (frame: AF chain A) with a core NAG on Asn140."""
    rng = np.random.default_rng(9)
    resd = _protomer_atoms(af_dimer_coords()["A"], rng)
    rows = _rows("A", resd, 120, 160)
    nd2 = dict((n, x) for n, e, x in resd[140])["ND2"]
    for an, el, off in (("C1", "C", [1.45, 0, 0]), ("O5", "O", [2.2, 1.0, 0]), ("C2", "C", [2.0, -1.0, 0.3])):
        x = nd2 + np.array(off)
        rows.append(("A", "C", None, "NAG", 1, "", None, None, an, el, "", *x, 40.0, 1.0, False, False, False))
    return pd.DataFrame(rows, columns=COLUMNS)


def uniprot_entry():
    e = A.uniprot_entry_with_topology()
    e["sequence"] = {"value": USEQ}
    return e


def survey_tables():
    return {
        "Summary": pd.DataFrame({"PDB ID": ["9S7U", "DON1"], "Assembly used": ["1 (author)", "1 (author)"]}),
        "Chains": pd.DataFrame({"PDB ID": ["9S7U", "9S7U", "DON1"], "Chain (auth)": ["A", "B", "A"], "Entity": ["1", "1", "1"],
                                "UniProt range (SIFTS)": ["56-750", "56-750", "44-750"]}),
        "Glycosylation": pd.DataFrame({"PDB ID": ["9S7U", "DON1"], "Asn (UniProt)": [140, 140]}),
        "TemplateRanking": pd.DataFrame({"PDB ID": ["9S7U", "DON1"]}),
    }
