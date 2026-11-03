"""Membrane and construct context (checklist item 15, membrane part) and membrane-MD inputs.

Design: ``docs/M4_ALPHAFOLD_MEMBRANE.md`` §5–6.

* **Topology** from UniProt features and, optionally, DeepTMHMM (BioLib via the
  ``tmfold`` package), compared residue by residue.
* **Placement** of a membrane slab on every predicted model with
  ``tmfold.geometry.fit_membrane``; for homodimers the normal is additionally
  constrained to the two-fold (C2) axis, which a symmetric dimer anchored in a
  flat membrane must be parallel to.
* **QC** per model: TM-in-slab, loop-side agreement, TM tilt, TM–ectodomain
  PAE, ectodomain clearance, sequons on the wrong side, active-site height,
  C2 consistency, and an overall orientation-reliability call.
* **Construct context** of the experimental structures.
* **Membrane-ready inputs** (oriented PDB, packmol-memgen command, CHARMM-GUI
  settings) and a readiness table that is never READY in this milestone.

Labels follow DeepTMHMM: ``I`` cytoplasmic, ``O`` non-cytoplasmic, ``M``
transmembrane helix, ``B`` beta strand, ``S`` signal peptide, ``-`` unknown.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .numbering import to_ranges
from .status import IssueLog, Status

log = logging.getLogger("psma_pipeline.membrane")

MEMBRANE = {"M", "B"}


def _tmfold_geometry():
    try:
        from tmfold import geometry
        return geometry
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------
def uniprot_topology(entry: dict, length: int) -> str:
    """Per-residue labels from UniProt Topological domain / Transmembrane / Intramembrane features."""
    lab = ["-"] * length
    for f in entry.get("features", []) or []:
        t = f.get("type")
        loc = f.get("location", {})
        try:
            a, b = int(loc["start"]["value"]), int(loc["end"]["value"])
        except (KeyError, TypeError, ValueError):
            continue
        desc = str(f.get("description", "")).lower()
        if t == "Topological domain":
            c = "I" if "cytoplasm" in desc else "O"
        elif t in ("Transmembrane", "Intramembrane"):
            c = "B" if "beta" in desc else "M"
        else:
            continue
        for i in range(max(a, 1), min(b, length) + 1):
            lab[i - 1] = c
    return "".join(lab)


def deeptmhmm_topology(acc: str, useq: str, cache: Path, issues: IssueLog) -> tuple[str | None, str]:
    """DeepTMHMM labels via tmfold/BioLib (cached per accession). Returns (labels or None, note)."""
    cache_file = cache / "deeptmhmm" / f"{acc}.json"
    if cache_file.exists():
        d = json.loads(cache_file.read_text())
        if d.get("sequence") == useq:
            return d["labels"], f"cached DeepTMHMM ({d.get('type')})"
    try:
        from tmfold.sequences import Protein
        from tmfold.topology import run_deeptmhmm
    except ImportError:
        issues.add("membrane", Status.NOT_RUN, "DeepTMHMM NOT RUN: tmfold not installed (pip install -e '.[membrane]')")
        return None, "NOT RUN (tmfold not installed)"
    try:
        res = run_deeptmhmm([Protein(acc, useq)], cache / "deeptmhmm" / acc, batch_size=1, max_retries=2)
    except Exception as ex:  # noqa: BLE001 - BioLib / network
        issues.add("membrane", Status.NOT_RUN, f"DeepTMHMM NOT RUN: {ex}")
        return None, f"NOT RUN ({ex})"
    t = res.get(acc)
    if t is None or len(t.labels) != len(useq):
        issues.add("membrane", Status.NOT_RUN, "DeepTMHMM returned no usable result")
        return None, "NOT RUN (no result)"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"sequence": useq, "labels": t.labels, "type": t.type}))
    return t.labels, f"DeepTMHMM ({t.type})"


def _segments(labels: str) -> list[tuple[str, int, int]]:
    out, pos = [], 1
    for lab, grp in itertools.groupby(labels):
        n = len(list(grp))
        out.append((lab, pos, pos + n - 1))
        pos += n
    return out


def topology_table(cfg: Any, entry: dict, useq: str, issues: IssueLog, sess: Any = None,
                   deeptmhmm_labels: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-residue topology (MembraneTopology) and a source comparison (TopologyComparison)."""
    L = len(useq)
    up = uniprot_topology(entry, L)
    note_d = "provided"
    if deeptmhmm_labels is None and cfg.get("membrane.deeptmhmm.enabled", True):
        deeptmhmm_labels, note_d = deeptmhmm_topology(cfg.accession, useq, cfg.cache_dir, issues)
    elif deeptmhmm_labels is None:
        note_d = "NOT RUN (disabled in config)"
    dt = deeptmhmm_labels
    norm = lambda c: {"S": "-", "P": "I"}.get(c, c)  # noqa: E731 - signal peptide unknown; periplasm = inner side
    source = cfg.get("membrane.topology_source", "consensus")
    chosen = []
    for i in range(L):
        u = up[i]
        d = norm(dt[i]) if dt else "-"
        if source == "deeptmhmm" and dt:
            c = d
        elif source == "uniprot" or not dt:
            c = u
        else:
            c = u if u == d or d == "-" else (u if u != "-" else d)
        chosen.append(c)
    chosen_s = "".join(chosen)
    rows = [{"UniProt number": i + 1, "Residue": useq[i], "UniProt label": up[i],
             "DeepTMHMM label": dt[i] if dt else None, "Membrane label": chosen_s[i],
             "Sources agree": (norm(dt[i]) == up[i]) if dt and up[i] != "-" else None} for i in range(L)]
    topo = pd.DataFrame(rows)
    comp = []
    for name, labs, note in (("UniProt", up, "features"), ("DeepTMHMM", dt, note_d), ("Used (" + source + ")", chosen_s, "")):
        if labs is None:
            comp.append({"Source": name, "Calculation status": Status.NOT_RUN.value, "Note": note})
            continue
        segs = _segments(labs)
        tm = [(a, b) for l, a, b in segs if l in MEMBRANE]
        first = labs.lstrip("-S")[:1]
        comp.append({"Source": name, "TM segments": len(tm), "TM ranges": "; ".join(f"{a}-{b}" for a, b in tm),
                     "N-terminus": {"I": "cytoplasmic", "O": "non-cytoplasmic"}.get(first, "unknown"),
                     "Residues labelled": sum(c != "-" for c in labs), "Calculation status": Status.OK.value,
                     "Note": note})
    if dt and up.strip("-"):
        both = [(up[i], norm(dt[i])) for i in range(L) if up[i] != "-" and norm(dt[i]) != "-"]
        agree = sum(a == b for a, b in both) / len(both) if both else None
        tu = {i for i in range(L) if up[i] in MEMBRANE}
        td = {i for i in range(L) if dt[i] in MEMBRANE}
        jac = len(tu & td) / len(tu | td) if tu | td else None
        comp.append({"Source": "UniProt vs DeepTMHMM", "Residue agreement": round(agree, 3) if agree is not None else None,
                     "TM residue Jaccard": round(jac, 3) if jac is not None else None,
                     "Calculation status": (Status.OK if (jac or 0) >= 0.5 else Status.WARNING).value,
                     "Note": "" if (jac or 0) >= 0.5 else "sources disagree on the TM segment(s)"})
        if (jac or 0) < 0.5:
            issues.add("membrane", Status.WARNING, "UniProt and DeepTMHMM disagree on the TM segment(s)")
    if not any(c in MEMBRANE for c in chosen_s):
        issues.add("membrane", Status.WARNING, "no transmembrane segment in the topology: membrane placement NOT RUN")
    return topo, pd.DataFrame(comp)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def c2_axis(A: dict[int, np.ndarray], B: dict[int, np.ndarray], residues: list[int]) -> tuple[np.ndarray | None, float | None]:
    """Axis and angle (degrees) of the rotation superposing chain A onto chain B."""
    from .conformations import kabsch

    common = [u for u in residues if u in A and u in B]
    if len(common) < 20:
        return None, None
    P, Q = np.array([A[u] for u in common]), np.array([B[u] for u in common])
    R, _, _ = kabsch(P, Q)
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    w, v = np.linalg.eig(R)
    axis = np.real(v[:, np.argmin(np.abs(w - 1))])
    return axis / np.linalg.norm(axis), ang


def fit_fixed_normal(geom: Any, labels: str, ca: dict[int, np.ndarray], n: np.ndarray):
    """Membrane with a fixed normal: scan mid-plane offset and half-thickness with tmfold's score."""
    pos = np.array(sorted(ca))
    X = np.array([ca[p] for p in pos])
    lab = np.array([labels[p - 1] for p in pos])
    is_tm = np.isin(lab, list(MEMBRANE))
    side = np.array([-1 if c in ("I", "P") else 1 if c == "O" else 0 for c in lab])
    is_loop = side != 0
    c0 = X[is_tm].mean(0)
    offsets, halves = np.arange(-8, 8.01, 1.0), np.arange(12, 20.01, 1.0)
    z = ((X - c0) @ n)[None, None, :] - offsets[:, None, None]
    in_slab = np.abs(z) <= halves[None, :, None]
    tm_in = in_slab[..., is_tm].mean(-1)
    loop_in = in_slab[..., is_loop].mean(-1) if is_loop.any() else np.zeros_like(tm_in)
    # same score as tmfold.geometry.fit_membrane: loop penalty per residue relative to the TM count
    # (a large ectodomain cannot dilute it), ties resolve to the thinnest slab
    loop_pen = in_slab[..., is_loop].sum(-1) / max(int(is_tm.sum()), 1) if is_loop.any() else np.zeros_like(tm_in)
    side_ok = (np.sign(z[..., is_loop]) == side[is_loop]).mean(-1) if is_loop.any() else np.zeros((len(offsets), 1))
    score = tm_in - 0.5 * loop_pen + 0.5 * side_ok - 1e-3 * halves[None, :]
    k = np.unravel_index(np.argmax(score), score.shape)
    center = c0 + offsets[k[0]] * n
    zs = (X - center) @ n
    return geom.Membrane(n, center, float(halves[k[1]]), float(tm_in[k]), float(side_ok[k[0], 0]), float(loop_in[k]),
                         {int(p): float(v) for p, v in zip(pos, zs)})


@dataclass
class Placement:
    model_id: str
    chains: list[str]
    offsets: dict[str, int]          # chain -> position offset in the concatenated labels
    labels: str
    membrane: Any                    # tmfold.geometry.Membrane
    method: str
    tm_fit_membrane: Any = None
    c2_angle_to_normal: float | None = None
    c2_rotation: float | None = None


def _concat(m: Any, labels: str) -> tuple[str, dict[int, np.ndarray], dict[str, int]]:
    L = len(labels)
    lab, ca, offs = "", {}, {}
    for k, c in enumerate(m.target_chains):
        offs[c] = k * L
        lab += labels
        for u, xyz in m.ca(c).items():
            ca[k * L + u] = xyz
    return lab, ca, offs


def place_membrane(m: Any, labels: str, cfg: Any, issues: IssueLog) -> Placement | None:
    geom = _tmfold_geometry()
    if geom is None:
        issues.add("membrane", Status.NOT_RUN, "membrane placement NOT RUN: tmfold not installed", pdb_id=m.model_id)
        return None
    if not any(c in MEMBRANE for c in labels):
        return None
    lab, ca, offs = _concat(m, labels)
    tm_pos = [p for p in ca if lab[p - 1] in MEMBRANE]
    if len(tm_pos) < 8:
        issues.add("membrane", Status.WARNING, f"only {len(tm_pos)} TM residues modelled: placement NOT RUN",
                   pdb_id=m.model_id)
        return None
    mem = geom.fit_membrane(lab, ca)
    pl = Placement(m.model_id, m.target_chains, offs, lab, mem, "TM fit (tmfold)", tm_fit_membrane=mem)
    if m.kind == "multimer" and len(m.target_chains) >= 2:
        c1, c2 = m.target_chains[:2]
        start = int(cfg.get("membrane.ectodomain_core_start", 57))
        pl1, pl2 = m.plddt(c1), m.plddt(c2)
        conf = float(cfg.get("alphafold.confident_plddt", 70))
        core = [u for u in m.ca(c1) if u >= start and pl1.get(u, 100) >= conf and pl2.get(u, 100) >= conf]
        axis, rot = c2_axis(m.ca(c1), m.ca(c2), core)
        if axis is not None:
            if np.dot(axis, mem.normal) < 0:
                axis = -axis
            pl.c2_rotation = round(rot, 1)
            pl.c2_angle_to_normal = round(float(np.degrees(np.arccos(np.clip(np.dot(axis, mem.normal), -1, 1)))), 1)
            if pl.c2_angle_to_normal > float(cfg.get("membrane.c2_max_angle_deg", 20)):
                issues.add("membrane", Status.WARNING, f"dimer C2 axis is {pl.c2_angle_to_normal}° from the TM-fitted "
                           "normal: TM–ectodomain geometry is not consistent with a flat membrane", pdb_id=m.model_id)
            if abs(rot - 180) > 20:
                issues.add("membrane", Status.WARNING, f"chain superposition angle {rot:.0f}° is not a two-fold",
                           pdb_id=m.model_id)
            elif cfg.get("membrane.c2_constrained_placement", True):
                pl.membrane = fit_fixed_normal(geom, lab, ca, axis)
                pl.method = "C2-axis constrained (tmfold score)"
    return pl


def placement_metrics(m: Any, pl: Placement, cfg: Any, useq: str, regions: dict[str, set[int]]) -> dict:
    geom = _tmfold_geometry()
    mem = pl.membrane
    n, center, half = mem.normal, mem.center, mem.half_thickness
    start = int(cfg.get("membrane.ectodomain_core_start", 57))
    a = m.atoms[m.atoms["target"] & m.atoms["unp"].notna()]
    z_all = (a[["x", "y", "z"]].to_numpy() - center) @ n
    ecto = a["unp"].to_numpy() >= start
    ez = z_all[ecto]
    tm_res = sorted(u for u in range(1, len(useq) + 1) if pl.labels[u - 1] in MEMBRANE)
    tilts, tm_pl, tm_ecto = [], [], []
    look = m.pae_lookup()
    for c in pl.chains:
        cca = m.ca(c)
        pos = [u for u in tm_res if u in cca]
        if len(pos) >= 4 and geom is not None:
            ax = geom.segment_axis(np.array([cca[u] for u in pos]))
            tilts.append(float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(ax, n))))))))
        tm_pl += [m.plddt(c)[u] for u in pos if u in m.plddt(c)]
        if m.pae is not None:
            i = [look[(c, u)] for u in pos if (c, u) in look]
            j = [look[(c, u)] for u in range(start, len(useq) + 1) if (c, u) in look]
            if i and j:
                tm_ecto += [float(m.pae[np.ix_(i, j)].mean()), float(m.pae[np.ix_(j, i)].mean())]
    sequons = regions.get("N-glycosylation sequons", set())
    wrong = 0
    for c in pl.chains:
        cca = m.ca(c)
        for u in sequons:
            if u in cca and pl.labels[u - 1] == "O" and float((cca[u] - center) @ n) < half:
                wrong += 1
    zn = regions.get("Zn ligands", set())
    zz = [float((m.ca(c)[u] - center) @ n) for c in pl.chains for u in zn if u in m.ca(c)]
    row = {"Model ID": m.model_id, "Placement method": pl.method, "Chains placed": ", ".join(pl.chains),
           "TM in slab": round(mem.tm_in_slab, 3), "Loop side agreement": round(mem.loop_side_agreement, 3),
           "Membrane thickness (Å)": round(2 * half, 1),
           "TM helix tilt (°)": round(float(np.mean(tilts)), 1) if tilts else None,
           "TM pLDDT": round(float(np.mean(tm_pl)), 1) if tm_pl else None,
           "TM-ectodomain PAE (Å)": round(float(np.mean(tm_ecto)), 2) if tm_ecto else None,
           "Ectodomain lowest atom above outer plane (Å)": round(float(ez.min() - half), 2) if len(ez) else None,
           "Ectodomain atoms in slab": int(np.sum(np.abs(ez) <= half)) if len(ez) else None,
           "Ectodomain atoms on cytoplasmic side": int(np.sum(ez < -half)) if len(ez) else None,
           "Sequons on wrong side": wrong,
           "Active-site height above membrane (Å)": round(float(np.mean(zz) - half), 1) if zz else None,
           "C2 axis to membrane normal (°)": pl.c2_angle_to_normal,
           "C2 rotation angle (°)": pl.c2_rotation,
           "TM-fit normal to final normal (°)": round(float(np.degrees(np.arccos(np.clip(
               abs(float(np.dot(pl.tm_fit_membrane.normal, n))), -1, 1)))), 1) if pl.tm_fit_membrane is not None else None,
           "Membrane normal (x, y, z)": ", ".join(f"{v:.4f}" for v in n),
           "Membrane centre (x, y, z)": ", ".join(f"{v:.3f}" for v in center)}
    reasons_u, reasons_w = [], []
    if row["TM in slab"] < float(cfg.get("membrane.min_tm_in_slab", 0.8)):
        reasons_u.append(f"TM in slab {row['TM in slab']}")
    if (row["Ectodomain atoms in slab"] or 0) > 50:
        reasons_u.append(f"{row['Ectodomain atoms in slab']} ectodomain atoms inside the membrane")
    if wrong:
        reasons_u.append(f"{wrong} N-glycosylation sequons not above the membrane")
    pae_rel = float(cfg.get("membrane.tm_ecto_pae_reliable", 10.0))
    if row["TM-ectodomain PAE (Å)"] is not None and row["TM-ectodomain PAE (Å)"] > pae_rel:
        reasons_w.append(f"TM–ectodomain PAE {row['TM-ectodomain PAE (Å)']} Å > {pae_rel}: "
                         "ectodomain orientation on the membrane not determined by the model")
    if row["TM-ectodomain PAE (Å)"] is None:
        reasons_w.append("no PAE: TM–ectodomain coupling unknown")
    if row["TM pLDDT"] is not None and row["TM pLDDT"] < 70:
        reasons_w.append(f"TM pLDDT {row['TM pLDDT']}")
    if pl.c2_angle_to_normal is not None and pl.c2_angle_to_normal > float(cfg.get("membrane.c2_max_angle_deg", 20)):
        reasons_w.append(f"C2 axis {pl.c2_angle_to_normal}° from TM normal")
    row["Orientation reliability"] = "unsupported" if reasons_u else ("weak" if reasons_w else "supported")
    row["Reliability reasons"] = "; ".join(reasons_u + reasons_w)
    row["Calculation status"] = (Status.WARNING if reasons_u or reasons_w else Status.OK).value
    return row


# ---------------------------------------------------------------------------
# Construct context
# ---------------------------------------------------------------------------
def construct_context(chains: pd.DataFrame, topo: pd.DataFrame, length: int) -> pd.DataFrame:
    """Experimental chains against the membrane topology: what was expressed and what is modelled.

    *Construct* coverage uses the SIFTS-mapped sequence (``UniProt range (SIFTS)``), which includes
    residues that were expressed but not resolved. *Modelled* coverage subtracts the survey's
    ``Missing residues (UniProt numbering)``, so a full-length construct whose TM helix is not
    seen in the density does not count as an experimental membrane anchor.
    """
    from .conformations import construct_ranges
    from .numbering import parse_ranges

    labels = "".join(topo["Membrane label"]) if len(topo) else "-" * length
    parts = {"cytoplasmic": {i + 1 for i, c in enumerate(labels) if c == "I"},
             "transmembrane": {i + 1 for i, c in enumerate(labels) if c in MEMBRANE},
             "extracellular": {i + 1 for i, c in enumerate(labels) if c == "O"}}
    missing: dict[tuple[str, str], set[int]] = {}
    known: set[tuple[str, str]] = set()
    if len(chains) and "Missing residues (UniProt numbering)" in chains:
        for _, r in chains.iterrows():
            key = (r["PDB ID"], r["Chain (auth)"])
            txt = r.get("Missing residues (UniProt numbering)")
            if pd.notna(r.get("Observed residues")):
                known.add(key)
            try:
                missing[key] = set(parse_ranges(txt)) if pd.notna(txt) and str(txt).strip() else set()
            except ValueError:
                missing[key] = set()
    rows = []
    for (pdb, ch), (a, b) in construct_ranges(chains).items():
        span = set(range(a, b + 1))
        modelled = span - missing.get((pdb, ch), set())
        r = {"PDB ID": pdb, "Chain": ch, "UniProt range (SIFTS)": f"{a}-{b}",
             "Fraction of full length": round(len(span) / length, 3),
             "Modelled residues": len(modelled) if (pdb, ch) in known else None}
        for name, s in parts.items():
            r[f"Covers {name}"] = round(len(span & s) / len(s), 3) if s else None
            r[f"Modelled {name}"] = (round(len(modelled & s) / len(s), 3) if s else None) if (pdb, ch) in known else None
        rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        tm_built = df[df["Modelled transmembrane"].fillna(0) > 0]
        tm_expressed = df[df["Covers transmembrane"].fillna(0) > 0]
        note = {"PDB ID": "(all)", "Chain": "", "UniProt range (SIFTS)": f"{len(df) } chains",
                "Fraction of full length": round(df["Fraction of full length"].max(), 3),
                "Modelled residues": df["Modelled residues"].max()}
        for name in parts:
            note[f"Covers {name}"] = round(df[f"Covers {name}"].max(), 3)
            note[f"Modelled {name}"] = round(df[f"Modelled {name}"].max(), 3) if df[f"Modelled {name}"].notna().any() else None
        if len(tm_built):
            ids = ", ".join(sorted(set(tm_built["PDB ID"]))[:10])
            note["Note"] = (f"TM helix modelled in {len(tm_built)} chain(s) ({ids}); best coverage "
                            f"{tm_built['Modelled transmembrane'].max():.0%}")
        elif len(tm_expressed):
            ids = ", ".join(sorted(set(tm_expressed["PDB ID"]))[:10])
            note["Note"] = (f"the TM helix is expressed in {len(tm_expressed)} chain(s) ({ids}) but not modelled in any: "
                            "the membrane anchor is prediction-only")
        else:
            note["Note"] = "no experimental construct contains the TM helix: the membrane anchor is prediction-only"
        df = pd.concat([df, pd.DataFrame([note])], ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# Membrane-ready inputs and readiness
# ---------------------------------------------------------------------------
def _pdb_lines(atoms: pd.DataFrame, rot: np.ndarray, shift: np.ndarray) -> list[str]:
    xyz = (atoms[["x", "y", "z"]].to_numpy() - shift) @ rot.T
    lines = []
    for k, (r, p) in enumerate(zip(atoms.itertuples(), xyz), 1):
        name = r.atom if len(r.atom) == 4 else f" {r.atom:<3s}"
        lines.append(f"ATOM  {k % 100000:5d} {name:4s} {r.resname:>3s} {str(r.chain)[:1]}{int(r.resnum) % 10000:4d}    "
                     f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}{1.0:6.2f}{float(r.b):6.2f}          {r.element:>2s}")
    return lines


def _select_md_model(res: Any, top: pd.DataFrame) -> tuple[str | None, str]:
    """A dimer only when the confident-dimer rule says so; otherwise the best membrane-anchored monomer.

    PSMA is a functional homodimer, but an unconfirmed predicted interface is a worse MD start
    than a well-placed monomer (the dimer can be built from the canonical interface in M5).
    """
    if not len(top):
        return None, ""
    summ = res.tables["AFModels"].set_index("Model ID")
    conf = summ["Dimer confidence"] if "Dimer confidence" in summ else pd.Series(dtype=str)
    t = top[top["Ranking mode"] == "membrane_dimer"].sort_values("Rank")
    for mid in t["Model ID"]:
        if conf.get(mid) == "confident":
            return mid, "membrane_dimer ranking (dimer confidence = confident)"
    skipped = [f"{m} ({conf.get(m) or 'not assessed'})" for m in t["Model ID"]]
    t = top[top["Ranking mode"] == "membrane_anchor"].sort_values("Rank")
    for mid in t["Model ID"]:
        if summ.loc[mid, "Model kind"] == "monomer":
            why = "membrane_anchor ranking"
            if skipped:
                why += "; dimer models not used because none is confident: " + ", ".join(skipped)
            return mid, why
    for mid in t["Model ID"]:   # no monomer available: use the best dimer, flagged
        return mid, (f"membrane_anchor ranking; no monomer available, so the dimer is used although it is "
                     f"not confident ({conf.get(mid) or 'not assessed'}) - dimer not used as confirmed")
    return None, ""


def write_md_inputs(cfg: Any, res: Any, top: pd.DataFrame) -> dict:
    """Oriented model, membrane planes, packmol-memgen command and CHARMM-GUI settings for the chosen model."""
    mres = res.membrane
    mid, why = _select_md_model(res, top)
    mres.md_reason = why
    if mid is None or mid not in mres.placements:
        mres.md_choice = None
        return {}
    geom = _tmfold_geometry()
    m = next(x for x in res.models if x.model_id == mid)
    pl = mres.placements[mid]
    mem = pl.membrane
    rot, shift = mem.frame()
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", mid)
    out = cfg.outputs_dir / "md_membrane_inputs" / safe
    out.mkdir(parents=True, exist_ok=True)
    atoms = m.atoms[m.atoms["target"] & m.atoms["polymer"]]
    b = cfg.get("membrane.builder", {})
    header = [f"REMARK   1 psma_pipeline membrane-oriented model of {mid}",
              "REMARK   1 membrane normal = +z, mid-plane z = 0, cytoplasm at z < 0",
              f"REMARK   1 fitted hydrophobic thickness {2 * mem.half_thickness:.1f} A ({pl.method})",
              "REMARK   1 B-factor = pLDDT. NOT MD-READY: see MembraneMDReadiness (Zn, glycans, protonation)"]
    (out / "model_membrane_oriented.pdb").write_text("\n".join(header + _pdb_lines(atoms, rot, shift) + ["TER", "END"]) + "\n")
    dum = geom.membrane_dummy_atoms(mem, atoms[atoms["atom"] == "CA"][["x", "y", "z"]].to_numpy(), oriented=True)
    (out / "membrane_planes_dum.pdb").write_text("\n".join(dum + ["END"]) + "\n")
    cmd = (f"packmol-memgen --pdb model_membrane_oriented.pdb --preoriented --lipids {b.get('lipids', 'POPC')} "
           f"--ratio {b.get('ratio', '1')} --dist {b.get('lateral_margin_A', 15)} --dist_wat {b.get('water_margin_A', 17.5)} "
           f"--salt --salt_c {b.get('cation', 'Na+')} --saltcon {b.get('salt_molar', 0.15)} --notprotonate "
           f"--output {safe}_membrane.pdb")
    (out / "packmol_memgen_command.sh").write_text(
        "#!/bin/sh\n# Membrane-ready input, NOT MD-ready: complete Milestone 5 preparation first.\n"
        "# --notprotonate: protonation must be assigned in Milestone 5, not silently by the builder.\n"
        f"# Lipid composition is a PLACEHOLDER from config membrane.builder: {b.get('lipids')} {b.get('ratio')}\n"
        + cmd + "\n")
    lip, ratio = str(b.get("lipids", "POPC")).split(":"), str(b.get("ratio", "1")).split(":")
    charmm = {"orientation": "Use PDB orientation (membrane normal along z, already centred)",
              "upper_leaflet": dict(zip(lip, ratio)), "lower_leaflet": dict(zip(lip, ratio)),
              "water_thickness_A": b.get("water_margin_A"), "lateral_margin_A": b.get("lateral_margin_A"),
              "ions": {"cation": b.get("cation"), "anion": b.get("anion"), "concentration_M": b.get("salt_molar")},
              "fitted_hydrophobic_thickness_A": round(2 * mem.half_thickness, 1),
              "force_field": b.get("force_field_note"),
              "assumptions": ["lipid composition is a placeholder - choose for your system (e.g. asymmetric plasma membrane)",
                              "Zn2+, glycans and protonation are NOT in this model (Milestone 5)"],
              "model": mid, "placement_method": pl.method}
    (out / "charmm_gui_settings.json").write_text(json.dumps(charmm, indent=2))
    (out / "README.md").write_text(
        f"# Membrane-ready inputs for {mid}\n\nOriented with the membrane normal along +z (cytoplasm at z < 0). "
        "These files are **not MD-ready**; the blockers are listed in the `MembraneMDReadiness` sheet of the "
        "AlphaFold/membrane workbook.\n\n- `model_membrane_oriented.pdb`: protein only, B = pLDDT\n"
        "- `membrane_planes_dum.pdb`: dummy atoms on both membrane planes (visual check)\n"
        "- `packmol_memgen_command.sh`: AmberTools packmol-memgen command\n"
        "- `charmm_gui_settings.json`: settings for CHARMM-GUI Membrane Builder\n")
    mres.md_choice = mid
    return {"folder": out, "model": mid, "files": sorted(str(p) for p in out.iterdir())}


@dataclass
class MembraneResults:
    tables: dict[str, pd.DataFrame]
    model_columns: pd.DataFrame
    placements: dict[str, Placement] = field(default_factory=dict)
    md_files: dict = field(default_factory=dict)
    md_choice: str | None = None
    md_reason: str = ""
    survey: dict = field(default_factory=dict)
    useq: str = ""

    def readiness_table(self, res: Any, top: pd.DataFrame, cfg: Any) -> pd.DataFrame:
        """Blockers between the membrane-ready inputs and an MD-ready system (never READY here)."""
        rows = []
        add = lambda item, st, detail: rows.append({"Item": item, "Status": st.value, "Detail": detail})  # noqa: E731
        mid = self.md_choice
        if mid is None:
            add("model selected", Status.FAIL, "no model passed the membrane_anchor / membrane_dimer rankings")
        else:
            add("model selected", Status.OK if "not used" not in self.md_reason else Status.WARNING,
                f"{mid} ({self.md_reason}) → {self.md_files.get('folder')}")
            m = next(x for x in res.models if x.model_id == mid)
            prow = self.tables["MembranePlacement"].set_index("Model ID").loc[mid]
            rel = prow["Orientation reliability"]
            add("membrane placement", Status.PASS if rel == "supported" else Status.WARNING,
                f"{rel}: {prow['Reliability reasons'] or 'all placement checks passed'}")
            n_prot = len(m.target_chains)
            zn_atoms = m.atoms[m.atoms["resname"].astype(str).str.upper() == "ZN"]
            zn = len(zn_atoms.drop_duplicates(["chain", "resnum"]))
            need = int(cfg.active_site.get("metal_site", {}).get("expected_zn_per_protomer", 2)) * n_prot
            add("zinc ions", Status.PASS if zn == need else Status.FAIL,
                f"{zn} Zn in model, {need} required ({n_prot} protomer(s)); Zn treatment chosen in Milestone 5")
            seq_n = sorted(u for u in range(1, len(self.useq) - 1) if self.useq[u - 1] == "N"
                           and self.useq[u] != "P" and self.useq[u + 1] in "ST")
            glyc = set(cfg.ligand_classes.get("glycans", []))
            has_g = bool(m.atoms["resname"].isin(glyc).any())
            if seq_n or has_g:
                add("N-glycans", Status.PASS if has_g else Status.FAIL,
                    f"{'glycans present' if has_g else 'no glycans'}; N-X-S/T sequons at {to_ranges(seq_n) or 'none'} "
                    "(composition and strategy in Milestone 5)")
            else:
                add("N-glycans", Status.NOT_APPLICABLE, "no N-X-S/T sequons in the sequence")
            add("protonation states", Status.NOT_RUN, "not assigned (PROPKA/PDB2PQR, Milestone 5); builder run with --notprotonate")
            low = [int(u) for c in m.target_chains for u, v in m.plddt(c).items() if v < 50]
            add("low-confidence segments", Status.WARNING if low else Status.PASS,
                f"pLDDT < 50 at {to_ranges(sorted(set(low)))}: decide trim / keep / remodel" if low else "none below 50")
            add("termini", Status.NOT_RUN, "capping / charged termini not decided (Milestone 5)")
            ions = self.survey.get("Ligands", pd.DataFrame())
            if len(ions) and "Component" in ions:
                comps = sorted(set(ions["Component"].astype(str)) & set(cfg.ligand_classes.get("ions_of_interest", [])) - {"ZN"})
                if comps:
                    add("other experimental ions", Status.WARNING,
                        f"{', '.join(comps)} seen in experimental structures; decide whether to include (Milestone 5)")
        b = cfg.get("membrane.builder", {})
        add("lipid composition", Status.WARNING, f"placeholder {b.get('lipids')} {b.get('ratio')}: confirm before building")
        add("force field", Status.NOT_RUN, str(b.get("force_field_note", "choose in Milestone 6")))
        add("OVERALL", Status.FAIL, "NOT READY for MD: membrane-ready inputs only. Resolve the FAIL / NOT RUN items in "
            "Milestones 5–6 (no READY status while any critical item fails).")
        return pd.DataFrame(rows)


def run_membrane(cfg: Any, res: Any, survey: dict[str, pd.DataFrame], entry: dict, useq: str,
                 issues: IssueLog) -> MembraneResults:
    """Placement and QC for every model, construct context, and per-residue membrane depth."""
    from .alphafold import region_sets

    topo = res.tables["MembraneTopology"]
    labels = "".join(topo["Membrane label"]) if len(topo) else "-" * len(useq)
    sites = sorted(int(n) for n in survey.get("ActiveSiteDefinition", pd.DataFrame()).query("Status == 'OK'")
                   ["UniProt number"].dropna().unique()) if "ActiveSiteDefinition" in survey else []
    regions = region_sets(cfg, useq, sites)
    rows, placements = [], {}
    for m in res.models:
        pl = place_membrane(m, labels, cfg, issues)
        if pl is None:
            rows.append({"Model ID": m.model_id, "Calculation status": Status.NOT_RUN.value,
                         "Reliability reasons": "no placement (see AlphaFoldIssues)"})
            continue
        placements[m.model_id] = pl
        rows.append(placement_metrics(m, pl, cfg, useq, regions))
    place = pd.DataFrame(rows)
    rc = res.tables.get("AFResidueConfidence", pd.DataFrame())
    if len(rc) and placements:
        depth = []
        for mid, c, u in zip(rc["Model ID"], rc["Chain"], rc["UniProt number"]):
            pl = placements.get(mid)
            if pl is None or c not in pl.offsets:
                depth.append(None)
                continue
            depth.append(round(pl.membrane.z.get(pl.offsets[c] + int(u), np.nan), 2))
        rc["Membrane depth (Å)"] = depth
        rc["Membrane label"] = rc["UniProt number"].map(lambda u: labels[int(u) - 1] if 0 < int(u) <= len(labels) else None)
    cols = ["Model ID", "TM in slab", "TM helix tilt (°)", "TM-ectodomain PAE (Å)", "Ectodomain atoms in slab",
            "C2 axis to membrane normal (°)", "Orientation reliability"]
    mcols = place[[c for c in cols if c in place]] if len(place) else pd.DataFrame(columns=["Model ID"])
    tables = {"MembranePlacement": place,
              "ConstructContext": construct_context(survey.get("Chains", pd.DataFrame()), topo, len(useq))}
    return MembraneResults(tables, mcols, placements, survey=survey, useq=useq)
