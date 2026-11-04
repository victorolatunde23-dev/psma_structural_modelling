"""Model preparation (Milestone 5): hybrid model, metal site, glycans, protonation, membrane, provenance.

Design: ``docs/M5_MODEL_PREPARATION.md``.

The prepared model is a **hybrid**:

* the experimental template's biological-assembly dimer (``preparation.template``,
  default 9S7U) supplies every residue it models, its Zn²⁺ / Ca²⁺ / Cl⁻ ions, the
  waters around the Zn site and its core N-glycans;
* the Milestone 4 AlphaFold2 dimer (``preparation.anchor_model``) supplies the
  cytoplasmic tail, TM helix and stalk (in the global dimer frame, so the two TM
  helices keep their predicted arrangement), internal gaps (locally fitted on the
  flanking residues), missing side-chain atoms and the wild-type residue wherever
  the template carries a substitution.

Every atom records its source. Original files (template mmCIF, AlphaFold2 PDB) are
never modified; their hashes are checked after the run (critical check).
Atom tables use UniProt numbering for protein residues and columns
``chain, resname, resnum, atom, element, x, y, z, b, kind, source``.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .alphafold import THREE_TO_ONE, AFModel
from .cache import sha256_file
from .conformations import kabsch
from .numbering import to_ranges
from .status import IssueLog, Status

log = logging.getLogger("psma_pipeline.model_preparation")

BACKBONE = ["N", "CA", "C"]
ATOM_COLS = ["chain", "resname", "resnum", "atom", "element", "x", "y", "z", "b", "kind", "source"]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_template_assembly(cfg: Any, survey: dict[str, pd.DataFrame], pdb: str, sess: Any,
                           issues: IssueLog) -> tuple[pd.DataFrame, list[Path], str]:
    """Biological-assembly atom table of the template (gemmi), as in the Milestone 2 assessment."""
    from .pdbe import fetch_sifts, sifts_segments
    from .rcsb import load_structure
    from .structure_atoms import asu_as_assembly, assembly_generators, atoms_from_gemmi, build_assembly
    from .survey import http_kwargs

    json_kw, dl_kw = http_kwargs(cfg)
    pdb = pdb.upper()
    st = load_structure(sess, pdb, cfg.cache_dir / "mmcif", cfg.endpoint("rcsb_cif"), **dl_kw)
    segs = sifts_segments(fetch_sifts(sess, pdb, cfg.cache_dir, cfg.endpoint("pdbe_sifts"), **json_kw), pdb, cfg.accession)
    chains = survey.get("Chains", pd.DataFrame())
    ents = {str(e) for e in chains[chains["PDB ID"] == pdb]["Entity"].dropna()} if len(chains) else set()
    atoms = atoms_from_gemmi(st, 0, segs, ents)
    summ = survey.get("Summary", pd.DataFrame())
    asm_id = None
    if len(summ) and pdb in set(summ["PDB ID"]):
        txt = str(summ.set_index("PDB ID").loc[pdb].get("Assembly used") or "")
        asm_id = txt.split()[0] if txt and txt.split()[0] != "none" else None
    gens = assembly_generators(st, asm_id) if asm_id else None
    if gens is None:
        issues.add("preparation", Status.WARNING, f"assembly {asm_id} not available; deposited ASU used", pdb_id=pdb)
        asm, label = asu_as_assembly(atoms), "ASU"
    else:
        asm, label = build_assembly(atoms, gens), f"assembly {asm_id}"
    files = [cfg.cache_dir / "mmcif" / f"{pdb}.cif.gz"]
    return asm, [f for f in files if f.exists()], label


def load_anchor_model(cfg: Any, useq: str, issues: IssueLog) -> AFModel | None:
    """The Milestone 4 AlphaFold2 model named by ``preparation.anchor_model``."""
    from .alphafold import load_colabfold

    job = cfg.get("preparation.anchor_job", "dimer")
    want = cfg.get("preparation.anchor_model", f"CF:{job}:rank_001")
    folder = cfg.cache_dir / "alphafold" / "colabfold" / job
    if not folder.exists():
        issues.add("preparation", Status.FAIL, f"AlphaFold2 results not found in {folder}: run notebook 03 first")
        return None
    models = load_colabfold(folder, useq, cfg, issues, label=job, stage=False)
    m = next((x for x in models if x.model_id == want), None)
    if m is None:
        issues.add("preparation", Status.FAIL, f"anchor model {want} not among {[x.model_id for x in models]}")
    return m


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------
def _residues(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {int(u): g for u, g in df.groupby(df["unp"].astype(int))}


def _coords(res: pd.DataFrame, names: list[str]) -> np.ndarray | None:
    sel = res.drop_duplicates("atom").set_index("atom")
    if not all(n in sel.index for n in names):
        return None
    return sel.loc[names, ["x", "y", "z"]].to_numpy(float)


def _apply(R: np.ndarray, pm: np.ndarray, qm: np.ndarray, X: np.ndarray) -> np.ndarray:
    return (X - pm) @ R.T + qm


def _fit(P: list[np.ndarray], Q: list[np.ndarray]):
    P, Q = np.vstack(P), np.vstack(Q)
    R, pm, qm = kabsch(P, Q)
    return R, pm, qm, float(np.sqrt(np.mean(np.sum((_apply(R, pm, qm, P) - Q) ** 2, axis=1))))


def _cn(res_a: pd.DataFrame, res_b: pd.DataFrame) -> float | None:
    c, n = _coords(res_a, ["C"]), _coords(res_b, ["N"])
    return float(np.linalg.norm(c[0] - n[0])) if c is not None and n is not None else None


def _moved(res: pd.DataFrame, R, pm, qm) -> pd.DataFrame:
    out = res.copy()
    out[["x", "y", "z"]] = _apply(R, pm, qm, res[["x", "y", "z"]].to_numpy(float))
    return out


# ---------------------------------------------------------------------------
# Hybrid construction
# ---------------------------------------------------------------------------
@dataclass
class Hybrid:
    atoms: pd.DataFrame                       # prepared atom table (ATOM_COLS)
    residue_sources: pd.DataFrame             # chain, UniProt residue, source
    segments: pd.DataFrame                    # contiguous runs per source
    junctions: pd.DataFrame                   # peptide C–N at every graft point
    repairs: pd.DataFrame                     # side chains completed / substitutions reverted
    hetero: pd.DataFrame                      # template hetero groups kept / removed
    fit: dict = field(default_factory=dict)   # global dimer superposition
    chain_map: dict = field(default_factory=dict)   # template protomer -> (hybrid chain, AF2 chain)


def _protomer_ca(asm: pd.DataFrame) -> dict[str, dict[int, np.ndarray]]:
    from .structure_atoms import target_protomers
    out = {}
    for p in target_protomers(asm):
        s = asm[(asm["protomer"] == p) & (asm["atom"] == "CA") & asm["unp"].notna() & asm["polymer"]]
        out[p] = {int(u): np.array([x, y, z]) for u, x, y, z in zip(s["unp"], s["x"], s["y"], s["z"])}
    return out


def global_dimer_fit(asm: pd.DataFrame, af: AFModel, cfg: Any) -> dict:
    """Superpose the AlphaFold2 dimer on the template dimer (both chains at once, both chain pairings tried)."""
    start = int(cfg.get("membrane.ectodomain_core_start", 57))
    pmin = float(cfg.get("preparation.fit_plddt_min", 70))
    exp = _protomer_ca(asm)
    prots = list(exp)[:2]
    chains = af.target_chains[:2]
    if len(prots) < 2 or len(chains) < 2:
        raise ValueError(f"need two template protomers and two AlphaFold2 chains (got {len(prots)} and {len(chains)})")
    best = None
    for order in (chains, chains[::-1]):
        P, Q, n = [], [], 0
        for p, c in zip(prots, order):
            ca, pl = af.ca(c), af.plddt(c)
            core = [u for u in sorted(exp[p]) if u >= start and u in ca and pl.get(u, 100) >= pmin]
            P.append(np.array([ca[u] for u in core]))
            Q.append(np.array([exp[p][u] for u in core]))
            n += len(core)
        R, pm, qm, rmsd = _fit(P, Q)
        # A symmetric homodimer fits equally well with chains swapped; keep the direct pairing (template
        # chain names) unless the swapped one is clearly better, so the choice never depends on rounding.
        if best is None or rmsd < best["rmsd"] - 0.05:
            best = {"R": R, "pm": pm, "qm": qm, "rmsd": rmsd, "pairs": dict(zip(prots, order)), "n": n}
    return best


def build_hybrid(asm: pd.DataFrame, af: AFModel, useq: str, cfg: Any, issues: IssueLog,
                 template: str) -> Hybrid:
    """Hybrid atom table: template residues where valid, AlphaFold2 elsewhere (see module docstring)."""
    pc = cfg.get("preparation", {}) or {}
    flank = int(pc.get("local_flank", 4))
    search = int(pc.get("junction_search", 12))
    cn_max = float(pc.get("junction_max_cn", 2.0))
    fit = global_dimer_fit(asm, af, cfg)
    R, pm, qm = fit["R"], fit["pm"], fit["qm"]
    L = len(useq)
    af_label = af.model_id
    atoms_out, src_rows, junc_rows, rep_rows, seg_rows = [], [], [], [], []
    chain_map = {}
    for k, (p, c) in enumerate(fit["pairs"].items()):
        hc = "AB"[k]
        chain_map[p] = (hc, c)
        E_all = asm[(asm["protomer"] == p) & asm["polymer"] & asm["target"] & asm["unp"].notna()]
        E = _residues(E_all)
        F_raw = af.atoms[(af.atoms["chain"] == c) & af.atoms["unp"].notna() & af.atoms["polymer"]]
        F = {u: _moved(r, R, pm, qm) for u, r in _residues(F_raw).items()}
        pl = af.plddt(c)
        valid = {u for u, r in E.items() if THREE_TO_ONE.get(r["resname"].iloc[0], "X") == useq[u - 1]
                 and _coords(r, BACKBONE) is not None}
        mutated = {u for u, r in E.items() if THREE_TO_ONE.get(r["resname"].iloc[0], "X") != useq[u - 1]}
        res: dict[int, tuple[pd.DataFrame, str]] = {}
        for u in sorted(valid):
            res[u] = (E[u], f"experimental {template}")
        # substitutions: wild-type residue from AlphaFold2, fitted on the template backbone
        for u in sorted(mutated):
            if u not in F:
                continue
            Pb, Qb = _coords(F[u], BACKBONE), _coords(E[u], BACKBONE)
            if Pb is None or Qb is None:
                continue
            Rl, pml, qml, _ = _fit([Pb], [Qb])
            res[u] = (_moved(F[u], Rl, pml, qml), f"AF2 {af_label} (reverted to UniProt)")
            rep_rows.append({"Chain": hc, "UniProt number": u, "Repair": "substitution reverted to UniProt",
                             "Template residue": E[u]["resname"].iloc[0], "UniProt residue": useq[u - 1],
                             "Atoms from AF2": len(F[u])})
        # gaps (and the membrane anchor) from AlphaFold2
        gaps = [u for u in range(1, L + 1) if u not in res]
        runs = []
        for u in gaps:
            if runs and u == runs[-1][1] + 1:
                runs[-1][1] = u
            else:
                runs.append([u, u])
        for a, b in runs:
            if a == 1:   # N-terminal anchor: global dimer frame, best graft point
                best = None
                for kk in range(0, search + 1):
                    jb = b + kk
                    if jb + 1 > L or any(u not in F for u in range(a, jb + 1)):
                        break
                    if kk and jb not in res:
                        break
                    nxt = res.get(jb + 1)
                    d = _cn(F[jb], nxt[0]) if nxt is not None else None
                    if d is not None and (best is None or d < best[1] - 1e-9):
                        best = (jb, d)
                jb, d = best if best else (b, None)
                for u in range(a, jb + 1):
                    res[u] = (F[u], f"AF2 {af_label} (membrane anchor, global dimer fit)")
                junc_rows.append({"Chain": hc, "Junction": f"{jb}-{jb + 1}", "Type": "N-terminal anchor → template",
                                  "Graft shifted by (residues)": jb - b, "Peptide C–N (Å)": round(d, 3) if d else None})
                seg_rows.append({"Chain": hc, "Start": a, "End": jb, "Source": "AF2 membrane anchor (global fit)",
                                 "Fit RMSD (Å)": round(fit["rmsd"], 3), "Mean pLDDT": _mean_pl(pl, a, jb)})
                continue
            left = [u for u in range(a - flank, a) if u in res and u in F and res[u][1].startswith("experimental")]
            right = [u for u in range(b + 1, b + 1 + flank) if u in res and u in F and res[u][1].startswith("experimental")]
            P, Q = [], []
            for u in left + right:
                x, y = _coords(F[u], BACKBONE), _coords(res[u][0], BACKBONE)
                if x is not None and y is not None:
                    P.append(x)
                    Q.append(y)
            kind = "C-terminal tail" if b == L else "internal gap"
            if len(P) >= 2:
                Rl, pml, qml, frms = _fit(P, Q)
                how = f"local fit on {len(P)} flanking residues"
            else:
                Rl, pml, qml, frms = np.eye(3), np.zeros(3), np.zeros(3), None
                how = "global dimer fit (no usable flanks)"
                issues.add("preparation", Status.WARNING, f"chain {hc} {a}-{b}: no flanking residues, global fit used")
            for u in range(a, b + 1):
                if u in F:
                    res[u] = (_moved(F[u], Rl, pml, qml), f"AF2 {af_label} ({kind}, {how})")
            seg_rows.append({"Chain": hc, "Start": a, "End": b, "Source": f"AF2 {kind}",
                             "Fit RMSD (Å)": round(frms, 3) if frms is not None else None, "Mean pLDDT": _mean_pl(pl, a, b)})
            for x, y, label in ((a - 1, a, "template → AF2"), (b, b + 1, "AF2 → template")):
                if x in res and y in res and 1 <= x and y <= L:
                    d = _cn(res[x][0], res[y][0])
                    junc_rows.append({"Chain": hc, "Junction": f"{x}-{y}", "Type": f"{kind}: {label}",
                                      "Graft shifted by (residues)": 0, "Peptide C–N (Å)": round(d, 3) if d else None})
        # missing side-chain atoms of template residues, from AlphaFold2 fitted on the residue
        for u, (r, s) in list(res.items()):
            if not s.startswith("experimental") or u not in F:
                continue
            have = set(r["atom"])
            want = set(F[u]["atom"]) - {"OXT"}
            missing = sorted(want - have)
            r2 = r[(r["atom"] != "OXT") | (u == L)]
            if missing:
                common = [n for n in F[u]["atom"] if n in have and n != "OXT"]
                Pb, Qb = _coords(F[u], common), _coords(r, common)
                if Pb is not None and len(common) >= 3:
                    Rl, pml, qml, _ = _fit([Pb], [Qb])
                    add = _moved(F[u][F[u]["atom"].isin(missing)], Rl, pml, qml)
                    cols = ["resname", "atom", "element", "x", "y", "z", "b"]   # the columns used downstream
                    r2 = pd.concat([r2[cols], add[cols]], ignore_index=True)
                    res[u] = (r2, f"experimental {template} + AF2 side-chain atoms")
                    rep_rows.append({"Chain": hc, "UniProt number": u, "Repair": "missing side-chain atoms from AF2",
                                     "Template residue": r["resname"].iloc[0], "UniProt residue": useq[u - 1],
                                     "Atoms from AF2": len(missing)})
                    continue
            res[u] = (r2, s)
        # assemble the chain in UniProt order
        for u in range(1, L + 1):
            if u not in res:
                issues.add("preparation", Status.FAIL, f"chain {hc}: residue {u} could not be built")
                continue
            r, s = res[u]
            from_af = s.startswith("AF2")
            if s.endswith("+ AF2 side-chain atoms"):
                base = set(E[u]["atom"]) if u in E else set()
                bcol = [float(b) if a in base else float(pl.get(u, np.nan)) for a, b in zip(r["atom"], r["b"])]
                acol = [f"experimental {template}" if a in base else f"AF2 {af_label} (side chain)" for a in r["atom"]]
            else:
                bcol = [float(pl.get(u, np.nan))] * len(r) if from_af else r["b"].astype(float).tolist()
                acol = [s] * len(r)
            for (_, a), bb, ss in zip(r.iterrows(), bcol, acol):
                atoms_out.append((hc, THREE_TO_ONE_INV.get(useq[u - 1], a["resname"]) if s.startswith("AF2") else a["resname"],
                                  u, a["atom"], a["element"], float(a["x"]), float(a["y"]), float(a["z"]), bb, "protein", ss))
            src_rows.append({"Chain": hc, "UniProt number": u, "Residue": useq[u - 1], "Source": s,
                             "pLDDT (AF2)": round(float(pl.get(u, np.nan)), 1) if u in pl else None})
    atoms = pd.DataFrame(atoms_out, columns=ATOM_COLS)
    junc = pd.DataFrame(junc_rows)
    if len(junc):
        junc["Status"] = [Status.PASS.value if d is not None and d <= cn_max else Status.WARNING.value
                          for d in junc["Peptide C–N (Å)"]]
        for _, r in junc[junc["Status"] == Status.WARNING.value].iterrows():
            issues.add("preparation", Status.WARNING, f"chain {r['Chain']} junction {r['Junction']}: C–N "
                       f"{r['Peptide C–N (Å)']} Å > {cn_max} Å (stretched peptide bond; minimise before MD)")
    hetero, het_atoms = select_hetero(asm, atoms, chain_map, cfg, template)
    atoms = pd.concat([atoms, het_atoms], ignore_index=True)
    fit_info = {"Global fit RMSD (Å)": round(fit["rmsd"], 3), "Global fit residues": fit["n"],
                "Chain pairing (template protomer → AF2 chain)": ", ".join(f"{p}→{c}" for p, c in fit["pairs"].items())}
    return Hybrid(atoms, pd.DataFrame(src_rows), _merge_segments(pd.DataFrame(src_rows), pd.DataFrame(seg_rows)),
                  junc, pd.DataFrame(rep_rows), hetero, fit_info, chain_map)


THREE_TO_ONE_INV = {v: k for k, v in THREE_TO_ONE.items() if k not in ("MSE", "UNK")}


def _mean_pl(pl: dict, a: int, b: int) -> float | None:
    v = [pl[u] for u in range(a, b + 1) if u in pl]
    return round(float(np.mean(v)), 1) if v else None


def _merge_segments(src: pd.DataFrame, fitted: pd.DataFrame) -> pd.DataFrame:
    """Contiguous runs of residue sources per chain, with the fit details of AF2 segments."""
    rows = []
    for ch, g in src.groupby("Chain", sort=True):
        g = g.sort_values("UniProt number")
        cur = None
        for u, s in zip(g["UniProt number"], g["Source"]):
            key = s.split(" (")[0] + (" (" + s.split(" (")[1].split(",")[0] + ")" if " (" in s else "")
            if cur and cur["Source"] == key and u == cur["End"] + 1:
                cur["End"] = u
            else:
                if cur:
                    rows.append(cur)
                cur = {"Chain": ch, "Start": u, "End": u, "Source": key}
        if cur:
            rows.append(cur)
    seg = pd.DataFrame(rows)
    if len(seg):
        seg["Residues"] = seg["End"] - seg["Start"] + 1
        if len(fitted):
            f = fitted.set_index(["Chain", "Start"])
            seg["Fit RMSD (Å)"] = [f["Fit RMSD (Å)"].get((c, s)) if (c, s) in f.index else None
                                   for c, s in zip(seg["Chain"], seg["Start"])]
            seg["Mean pLDDT"] = [f["Mean pLDDT"].get((c, s)) if (c, s) in f.index else None
                                 for c, s in zip(seg["Chain"], seg["Start"])]
    return seg


def select_hetero(asm: pd.DataFrame, protein: pd.DataFrame, chain_map: dict, cfg: Any,
                  template: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Template ions, Zn-shell waters and root N-glycans to keep; everything else removed and listed."""
    pc = cfg.get("preparation", {}) or {}
    keep_ions = {str(x).upper() for x in pc.get("keep_ions", ["ZN", "CA", "CL"])}
    water_rule = pc.get("keep_waters", "zn_shell")
    wmax = float(pc.get("water_zn_max", 3.0))
    gcfg = pc.get("glycans", {}) or {}
    link = float(gcfg.get("link_max", 2.0))
    glycans = set(cfg.ligand_classes.get("glycans", []))
    het = asm[~asm["polymer"]].copy()
    het["_key"] = het["label_asym"].astype(str) + "|" + het["resnum"].astype(str) + het["icode"].astype(str) + "|" + het["copy"].astype(str)
    P = protein[["x", "y", "z"]].to_numpy(float)
    pchain = protein["chain"].to_numpy()
    nd2 = protein[(protein["resname"] == "ASN") & (protein["atom"] == "ND2")]
    ND2 = nd2[["x", "y", "z"]].to_numpy(float)
    zn_xyz = het[(het["resname"].str.upper() == "ZN")][["x", "y", "z"]].to_numpy(float) if "ZN" in keep_ions else np.zeros((0, 3))
    rows, keep_parts = [], []
    counters = {"ion": 3000, "water": 4000, "glycan": 2000}
    for key, g in het.groupby("_key", sort=False):
        comp = str(g["resname"].iloc[0]).upper()
        xyz = g[["x", "y", "z"]].to_numpy(float)
        dmin = np.linalg.norm(P[:, None, :] - xyz[None, :, :], axis=2) if len(P) else np.zeros((0, len(xyz)))
        near_chain = pchain[int(np.argmin(dmin.min(axis=1)))] if len(P) else "A"
        kind, decision, note, link_asn = None, "removed", "", None
        if comp in keep_ions:
            kind, decision = "ion", "kept"
        elif comp == "HOH":
            if water_rule == "all":
                kind, decision = "water", "kept"
            elif water_rule == "zn_shell" and len(zn_xyz) and np.linalg.norm(zn_xyz - xyz[0], axis=1).min() <= wmax:
                kind, decision, note = "water", "kept", "Zn shell"
            else:
                continue          # bulk waters are not listed one by one
        elif comp in glycans:
            c1 = g[g["atom"] == "C1"]
            if len(c1) and len(ND2):
                d = np.linalg.norm(ND2 - c1[["x", "y", "z"]].to_numpy(float)[0], axis=1)
                j = int(np.argmin(d))
                if d[j] <= link and gcfg.get("strategy", "core_glcnac") != "none":
                    kind, decision = "glycan", "kept"
                    link_asn = (nd2["chain"].iloc[j], int(nd2["resnum"].iloc[j]))
                    near_chain = link_asn[0]
                    note = f"core sugar on Asn{link_asn[1]} (C1–ND2 {d[j]:.2f} Å)"
                else:
                    note = "outer glycan residue (not the core GlcNAc)" if d[j] > link else "glycan strategy = none"
        else:
            note = "ligand / additive (apo system)" if pc.get("remove_ligands", True) else "kept"
            if not pc.get("remove_ligands", True):
                kind, decision = "ligand", "kept"
        rows.append({"Component": comp, "Template group": key, "Nearest chain": near_chain, "Decision": decision,
                     "Note": note, "Atoms": len(g),
                     "Linked Asn (UniProt)": link_asn[1] if link_asn else None})
        if decision == "kept":
            counters[kind if kind in counters else "ion"] += 1
            num = counters[kind if kind in counters else "ion"]
            part = pd.DataFrame({"chain": near_chain, "resname": comp, "resnum": num, "atom": g["atom"].values,
                                 "element": g["element"].values, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
                                 "b": g["b"].astype(float).values, "kind": kind,
                                 "source": f"experimental {template}" + (f" ({note})" if note else "")})
            if link_asn:
                part["link"] = f"{link_asn[0]}:ASN{link_asn[1]}:ND2"
            keep_parts.append(part)
    other = asm[asm["polymer"] & ~asm["target"]]
    for (chn, cp), g in other.groupby(["chain", "copy"]):
        rows.append({"Component": "polymer (non-target)", "Template group": f"{chn}|{cp}", "Nearest chain": "",
                     "Decision": "removed", "Note": "non-target polymer chain (e.g. antibody / nanobody)",
                     "Atoms": len(g), "Linked Asn (UniProt)": None})
    het_atoms = pd.concat(keep_parts, ignore_index=True) if keep_parts else pd.DataFrame(columns=ATOM_COLS)
    return pd.DataFrame(rows), het_atoms


# ---------------------------------------------------------------------------
# Glycans: core GlcNAc where observed experimentally
# ---------------------------------------------------------------------------
def sequon_observations(survey: dict[str, pd.DataFrame], useq: str) -> pd.DataFrame:
    """Per N-X-S/T sequon: PDB entries with a glycan on it, and entries whose construct covers it."""
    from .conformations import construct_ranges
    seqs = [i + 1 for i in range(len(useq) - 2) if useq[i] == "N" and useq[i + 1] != "P" and useq[i + 2] in "ST"]
    gl = survey.get("Glycosylation", pd.DataFrame())
    ch = survey.get("Chains", pd.DataFrame())
    ranges = construct_ranges(ch) if len(ch) else {}
    rows = []
    for u in seqs:
        with_g = sorted(set(gl.loc[pd.to_numeric(gl.get("Asn (UniProt)"), errors="coerce") == u, "PDB ID"])) if len(gl) else []
        cover = sorted({p for (p, _), (a, b) in ranges.items() if a <= u <= b})
        rows.append({"Asn (UniProt)": u, "Sequon": useq[u - 1:u + 2], "Entries with glycan": len(with_g),
                     "Entries covering residue": len(cover),
                     "Observed fraction": round(len(with_g) / len(cover), 3) if cover else None,
                     "Example entries": ", ".join(with_g[:6])})
    return pd.DataFrame(rows)


def load_donor_atoms(cfg: Any, survey: dict[str, pd.DataFrame], pdb: str, sess: Any) -> pd.DataFrame:
    """Deposited-model atom table of a glycan donor entry (gemmi)."""
    from .pdbe import fetch_sifts, sifts_segments
    from .rcsb import load_structure
    from .structure_atoms import atoms_from_gemmi
    from .survey import http_kwargs

    json_kw, dl_kw = http_kwargs(cfg)
    st = load_structure(sess, pdb, cfg.cache_dir / "mmcif", cfg.endpoint("rcsb_cif"), **dl_kw)
    segs = sifts_segments(fetch_sifts(sess, pdb, cfg.cache_dir, cfg.endpoint("pdbe_sifts"), **json_kw), pdb, cfg.accession)
    chains = survey.get("Chains", pd.DataFrame())
    ents = {str(e) for e in chains[chains["PDB ID"] == pdb]["Entity"].dropna()} if len(chains) else set()
    return atoms_from_gemmi(st, 0, segs, ents)


def transplant_core_glcnac(atoms: pd.DataFrame, chain: str, u: int, donor: pd.DataFrame, donor_id: str,
                           cfg: Any) -> tuple[pd.DataFrame | None, str]:
    """Copy the root sugar on Asn ``u`` of a donor entry onto the hybrid by local superposition.

    Fit: backbone N/CA/C of u-2..u+2 plus the Asn side chain. Rejected if the sugar comes
    closer than ``glycans.clash_min`` to a protein heavy atom other than Asn ``u``.
    """
    g = cfg.get("preparation.glycans", {}) or {}
    link, clash = float(g.get("link_max", 2.0)), float(g.get("clash_min", 2.0))
    glycans = set(cfg.ligand_classes.get("glycans", []))
    prot = donor[donor["polymer"] & donor["unp"].notna()]
    sugars = donor[donor["resname"].isin(glycans)]
    tgt = atoms[(atoms["kind"] == "protein") & (atoms["chain"] == chain)]
    for dch in sorted(prot["chain"].unique()):
        asn = prot[(prot["chain"] == dch) & (prot["unp"] == u) & (prot["atom"] == "ND2")]
        if asn.empty or prot[(prot["chain"] == dch) & (prot["unp"] == u)]["resname"].iloc[0] != "ASN":
            continue
        nd2 = asn[["x", "y", "z"]].to_numpy(float)[0]
        root = None
        for key, s in sugars.groupby(["chain", "resnum", "icode"]):
            c1 = s[s["atom"] == "C1"]
            if len(c1) and np.linalg.norm(c1[["x", "y", "z"]].to_numpy(float)[0] - nd2) <= link:
                root = s
                break
        if root is None:
            continue
        P, Q = [], []
        for v in range(u - 2, u + 3):
            names = BACKBONE + (["CB", "CG", "OD1", "ND2"] if v == u else [])
            dr = prot[(prot["chain"] == dch) & (prot["unp"] == v)]
            hr = tgt[tgt["resnum"] == v]
            x, y = (_coords(dr, names) if len(dr) else None), (_coords(hr, names) if len(hr) else None)
            if x is not None and y is not None:
                P.append(x)
                Q.append(y)
        if len(P) < 3:
            continue
        R, pm, qm, rmsd = _fit(P, Q)
        xyz = _apply(R, pm, qm, root[["x", "y", "z"]].to_numpy(float))
        others = atoms[(atoms["kind"] == "protein") & ~((atoms["chain"] == chain) & (atoms["resnum"] == u))]
        dmin = float(np.min(np.linalg.norm(others[["x", "y", "z"]].to_numpy(float)[:, None, :] - xyz[None], axis=2)))
        if dmin < clash:
            return None, f"{donor_id}/{dch}: sugar {dmin:.2f} Å from protein (clash)"
        part = pd.DataFrame({"chain": chain, "resname": root["resname"].values, "resnum": 0, "atom": root["atom"].values,
                             "element": root["element"].values, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
                             "b": root["b"].astype(float).values, "kind": "glycan",
                             "source": f"transplanted from {donor_id}/{dch} (local fit RMSD {rmsd:.2f} Å)",
                             "link": f"{chain}:ASN{u}:ND2"})
        return part, f"{donor_id}/{dch} (fit RMSD {rmsd:.2f} Å, closest protein atom {dmin:.2f} Å)"
    return None, f"{donor_id}: no root sugar on Asn{u}"


def apply_glycan_strategy(atoms: pd.DataFrame, obs: pd.DataFrame, donors: dict[str, pd.DataFrame],
                          cfg: Any, template: str, issues: IssueLog) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Core GlcNAc at every sequon observed glycosylated: kept from the template or transplanted."""
    g = cfg.get("preparation.glycans", {}) or {}
    strategy = g.get("strategy", "core_glcnac")
    min_obs = int(g.get("min_observed_structures", 1))
    chains = sorted(atoms.loc[atoms["kind"] == "protein", "chain"].unique())
    have = set()
    if "link" in atoms:
        for v in atoms.loc[atoms["kind"] == "glycan", "link"].dropna().unique():
            c, rest = v.split(":", 1)
            have.add((c, int(re.sub(r"\D", "", rest.split(":")[0]))))
    rows, parts = [], []
    next_num = int(atoms.loc[atoms["kind"] == "glycan", "resnum"].max()) + 1 if (atoms["kind"] == "glycan").any() else 2001
    for _, o in obs.iterrows():
        u = int(o["Asn (UniProt)"])
        for c in chains:
            row = {"Chain": c, "Asn (UniProt)": u, "Sequon": o["Sequon"], "Entries with glycan": o["Entries with glycan"],
                   "Observed fraction": o["Observed fraction"]}
            if strategy == "none":
                row.update(Decision="not glycosylated", Source="", Note="glycan strategy = none")
            elif (c, u) in have:
                row.update(Decision="core GlcNAc", Source=f"template {template}", Note="")
            elif o["Entries with glycan"] < min_obs:
                row.update(Decision="not glycosylated", Source="", Note="never observed glycosylated in the PDB")
            else:
                placed, notes = None, []
                for pdb in [p.strip() for p in str(o["Example entries"]).split(",") if p.strip()] + [
                        p for p in donors if p not in str(o["Example entries"])]:
                    if pdb not in donors:
                        continue
                    part, note = transplant_core_glcnac(pd.concat([atoms] + parts, ignore_index=True), c, u,
                                                        donors[pdb], pdb, cfg)
                    notes.append(note)
                    if part is not None:
                        part["resnum"] = next_num
                        next_num += 1
                        parts.append(part)
                        placed = note
                        break
                if placed:
                    row.update(Decision="core GlcNAc", Source="transplanted from " + placed.split(" (")[0], Note=placed)
                else:
                    row.update(Decision="not glycosylated", Source="",
                               Note="observed, but no donor placement succeeded: " + "; ".join(notes[:3]) if notes
                               else "observed, but no donor structure available")
                    issues.add("preparation", Status.WARNING, f"chain {c} Asn{u}: observed glycosylated but not placed")
            rows.append(row)
    out = pd.concat([atoms] + parts, ignore_index=True) if parts else atoms
    return out, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PDB output (heavy atoms; protonation via residue names; LINK records for N-glycans)
# ---------------------------------------------------------------------------
def _link_line(a1, r1, c1, n1, a2, r2, c2, n2, dist) -> str:
    line = [" "] * 80
    def put(col, text):  # col is 1-based
        for k, ch in enumerate(text):
            line[col - 1 + k] = ch
    put(1, "LINK")
    put(13, f"{a1:<4s}" if len(a1) == 4 else f" {a1:<3s}")
    put(18, f"{r1:>3s}"); put(22, c1); put(23, f"{n1:4d}")
    put(43, f"{a2:<4s}" if len(a2) == 4 else f" {a2:<3s}")
    put(48, f"{r2:>3s}"); put(52, c2); put(53, f"{n2:4d}")
    put(74, f"{dist:5.2f}")
    return "".join(line).rstrip()


def pdb_text(atoms: pd.DataFrame, names: dict[tuple[str, int], str] | None = None,
             remarks: list[str] | None = None, rot: np.ndarray | None = None, shift: np.ndarray | None = None) -> str:
    """PDB text: protein ATOM records (residue names from ``names`` = protonation states), HETATM for the rest."""
    names = names or {}
    xyz = atoms[["x", "y", "z"]].to_numpy(float)
    if rot is not None:
        xyz = (xyz - shift) @ rot.T
    lines = [f"REMARK   1 {r}" for r in (remarks or [])]
    links = []
    if "link" in atoms:
        for _, g in atoms[atoms["kind"] == "glycan"].groupby(["chain", "resnum"]):
            v = g["link"].dropna()
            if not len(v) or "C1" not in set(g["atom"]):
                continue
            c, rest = v.iloc[0].split(":", 1)
            n = int(re.sub(r"\D", "", rest.split(":")[0]))
            nd2 = atoms[(atoms["kind"] == "protein") & (atoms["chain"] == c) & (atoms["resnum"] == n) & (atoms["atom"] == "ND2")]
            c1 = g[g["atom"] == "C1"]
            if len(nd2):
                d = float(np.linalg.norm(nd2[["x", "y", "z"]].to_numpy(float)[0] - c1[["x", "y", "z"]].to_numpy(float)[0]))
                links.append(_link_line("C1", g["resname"].iloc[0], g["chain"].iloc[0], int(g["resnum"].iloc[0]),
                                        "ND2", names.get((c, n), "ASN"), c, n, d))
    lines += links
    order = {"protein": 0, "glycan": 1, "ion": 2, "water": 3, "ligand": 4}
    kind = atoms["kind"].to_numpy()
    chain = atoms["chain"].astype(str).to_numpy()
    resnum = atoms["resnum"].astype(int).to_numpy()
    resname = atoms["resname"].astype(str).to_numpy()
    aname = atoms["atom"].astype(str).to_numpy()
    elem = atoms["element"].astype(str).to_numpy()
    bfac = pd.to_numeric(atoms["b"], errors="coerce").fillna(0.0).to_numpy()
    idx = np.lexsort((np.arange(len(atoms)), resnum, chain, np.array([order.get(k, 9) for k in kind])))
    serial, last = 1, None
    for i in idx:
        k = kind[i]
        if k != "protein" and last == "protein":
            lines.append("TER")
        last = "protein" if k == "protein" else "het"
        rec = "ATOM  " if k == "protein" else "HETATM"
        rn = names.get((chain[i], int(resnum[i])), resname[i]) if k == "protein" else resname[i]
        nm = aname[i] if len(aname[i]) == 4 else f" {aname[i]:<3s}"
        x, y, z = xyz[i]
        lines.append(f"{rec}{serial % 100000:5d} {nm:4s} {rn:>3s} {chain[i]}{int(resnum[i]) % 10000:4d}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{bfac[i]:6.2f}          {elem[i]:>2s}")
        serial += 1
    lines += ["END"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Protonation (PROPKA at the configured pH; Zn ligands, disulfides, His tautomers, microstates)
# ---------------------------------------------------------------------------
PROPKA_LINE = re.compile(r"^\s+([A-Z][A-Z0-9+\-]{1,3})\s+(\d+)\s+(\S)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")


def run_propka(pdb_path: Path) -> tuple[pd.DataFrame | None, str]:
    """Run PROPKA 3 on a protein-only PDB; returns the summary table and the version (or None, reason)."""
    import shutil
    import subprocess
    import sys

    try:
        import propka  # noqa: F401
        version = getattr(propka, "__version__", "3")
    except ImportError:
        return None, "PROPKA not installed (pip install propka)"
    exe = shutil.which("propka3")
    cmd = [exe, pdb_path.name] if exe else [sys.executable, "-m", "propka", pdb_path.name]
    r = subprocess.run(cmd, cwd=pdb_path.parent, capture_output=True, text=True)
    pka = pdb_path.with_suffix(".pka")
    if r.returncode != 0 or not pka.exists():
        return None, f"PROPKA failed: {(r.stderr or r.stdout)[-400:]}"
    return parse_propka(pka.read_text()), f"PROPKA {version}"


def parse_propka(text: str) -> pd.DataFrame:
    rows, on = [], False
    for line in text.splitlines():
        if "SUMMARY OF THIS PREDICTION" in line:
            on = True
            continue
        if on:
            if line.startswith("---") or (rows and not line.strip()):
                break
            m = PROPKA_LINE.match(line)
            if m:
                rows.append({"Group": m.group(1), "Residue": int(m.group(2)), "Chain": m.group(3),
                             "pKa": float(m.group(4)), "Model pKa": float(m.group(5))})
    return pd.DataFrame(rows)


def _sel(atoms, chain, resnum, names):
    a = atoms[(atoms["chain"] == chain) & (atoms["resnum"] == resnum) & atoms["atom"].isin(names)]
    return a[["x", "y", "z"]].to_numpy(float), a["atom"].tolist()


def assign_protonation(atoms: pd.DataFrame, pka: pd.DataFrame | None, zn_contacts: pd.DataFrame, cfg: Any,
                       propka_note: str) -> tuple[pd.DataFrame, dict, list[dict]]:
    """Protonation table, default residue names (AMBER convention) and alternative states.

    Rules: Zn-ligand His take the tautomer that leaves the coordinating N unprotonated; Zn-ligand
    Asp/Glu are deprotonated (fixed); Cys in disulfides are CYX; otherwise PROPKA pKa vs pH
    (default pKa values when PROPKA is unavailable). Residues with |pKa − pH| ≤ window, and those
    in ``always_alternative``, get an alternative state. His tautomers of non-ligands follow the
    nearest H-bond acceptors (ND1 vs NE2).
    """
    pc = cfg.get("preparation.protonation", {}) or {}
    ph, window = float(pc.get("ph", 7.4)), float(pc.get("window", 1.0))
    hb = float(pc.get("hbond_max", 3.2))
    always = {int(re.sub(r"\D", "", c)) for c in pc.get("always_alternative", [])}
    default_pka = {"ASP": 3.8, "GLU": 4.5, "HIS": 6.5, "LYS": 10.5, "CYS": 9.0, "TYR": 10.0, "ARG": 12.5}
    prot = atoms[atoms["kind"] == "protein"]
    zl = {}
    if len(zn_contacts):
        for _, r in zn_contacts[zn_contacts["Kind"] == "protein"].iterrows():
            zl[(r["Chain"], int(re.sub(r"\D", "", r["Residue"])))] = r["Atom"]
    sg = prot[(prot["resname"] == "CYS") & (prot["atom"] == "SG")]
    SG = sg[["x", "y", "z"]].to_numpy(float)
    ss = set()
    for i in range(len(SG)):
        for j in range(i + 1, len(SG)):
            if np.linalg.norm(SG[i] - SG[j]) < 2.5:
                ss.add((sg["chain"].iat[i], int(sg["resnum"].iat[i])))
                ss.add((sg["chain"].iat[j], int(sg["resnum"].iat[j])))
    acc = atoms[(atoms["element"] == "O")]
    ACC = acc[["x", "y", "z"]].to_numpy(float)
    look = {}
    if pka is not None and len(pka):
        for r in pka.itertuples():
            look[(r.Chain, r.Residue, r.Group)] = r.pKa
    rows, names, alts = [], {}, []
    for (c, n), g in prot.groupby(["chain", "resnum"], sort=True):
        rn = g["resname"].iloc[0]
        if rn not in default_pka:
            continue
        pk = look.get((c, int(n), rn))
        src = "PROPKA" if pk is not None else "default pKa"
        if pk is None:
            pk = default_pka[rn]
        state, alt, reason = rn, None, ""
        if (c, n) in zl and rn in ("HIS", "ASP", "GLU"):
            if rn == "HIS":
                state = "HID" if zl[(c, n)] == "NE2" else "HIE"
                reason = f"Zn ligand via {zl[(c, n)]}: that N unprotonated"
            else:
                reason = "Zn ligand: deprotonated (fixed)"
            src = "fixed (Zn ligand)"
        elif rn == "CYS" and (c, n) in ss:
            state, reason, src = "CYX", "disulfide", "fixed (disulfide)"
        else:
            if rn == "ASP":
                state, other = ("ASH", "ASP") if pk > ph else ("ASP", "ASH")
            elif rn == "GLU":
                state, other = ("GLH", "GLU") if pk > ph else ("GLU", "GLH")
            elif rn == "LYS":
                state, other = ("LYS", "LYN") if pk > ph else ("LYN", "LYS")
            elif rn == "CYS":
                state, other = ("CYS", "CYM") if pk > ph else ("CYM", "CYS")
            elif rn == "HIS":
                X, nm = _sel(prot, c, n, ["ND1", "NE2"])
                near = {}
                for x, a in zip(X, nm):
                    d = np.linalg.norm(ACC - x, axis=1)
                    mask = d <= hb
                    own = (acc["chain"].to_numpy() == c) & (acc["resnum"].to_numpy() == n)
                    near[a] = int((mask & ~own).sum())
                tauto = "HID" if near.get("ND1", 0) and not near.get("NE2", 0) else "HIE"
                state, other = ("HIP", tauto) if pk > ph else (tauto, "HIP")
                reason = f"tautomer from H-bond acceptors (ND1 {near.get('ND1', 0)}, NE2 {near.get('NE2', 0)})"
            else:
                other = None
            if other and (abs(pk - ph) <= window or (n in always and rn in ("ASP", "GLU", "HIS", "LYS"))):
                alt = other
                why = "always alternative (configured)" if n in always else f"|pKa − pH| = {abs(pk - ph):.2f} ≤ {window}"
                alts.append({"Chain": c, "Residue": int(n), "Default": state, "Alternative": other,
                             "pKa": round(pk, 2), "Why": why, "_rank": abs(pk - ph) if n not in always else -1})
        names[(c, int(n))] = state
        rows.append({"Chain": c, "UniProt number": int(n), "Residue": rn, "pKa": round(pk, 2), "pKa source": src,
                     "pH": ph, "State": state, "Alternative state": alt, "Reason": reason})
    return pd.DataFrame(rows), names, sorted(alts, key=lambda a: a["_rank"])


def microstates(names: dict, alts: list[dict], max_n: int) -> list[tuple[str, dict, str]]:
    """Default microstate plus one alternative at a time (most ambiguous first), up to ``max_n``."""
    out = [("microstate_00_default", dict(names), "default")]
    for k, a in enumerate(alts[: max(0, max_n - 1)], 1):
        nm = dict(names)
        nm[(a["Chain"], a["Residue"])] = a["Alternative"]
        out.append((f"microstate_{k:02d}_{a['Chain']}{a['Residue']}_{a['Alternative']}", nm,
                    f"{a['Chain']}{a['Residue']} {a['Default']}→{a['Alternative']} ({a['Why']})"))
    return out


# ---------------------------------------------------------------------------
# Membrane placement of the hybrid
# ---------------------------------------------------------------------------
def hybrid_as_model(atoms: pd.DataFrame, name: str, useq: str) -> AFModel:
    """Wrap the hybrid protein as an AFModel so the Milestone 4 membrane code can place it."""
    from .structure_atoms import COLUMNS
    p = atoms[atoms["kind"] == "protein"]
    df = pd.DataFrame({"chain": p["chain"], "label_asym": p["chain"], "entity": "hybrid", "resname": p["resname"],
                       "resnum": p["resnum"].astype(int), "icode": "", "label_seq": p["resnum"].astype(int),
                       "unp": p["resnum"].astype(float), "atom": p["atom"], "element": p["element"],
                       "altloc": "", "x": p["x"], "y": p["y"], "z": p["z"], "b": p["b"], "occ": 1.0,
                       "polymer": True, "water": False, "target": True})[COLUMNS]
    df["copy"], df["protomer"] = "1", df["chain"] + "|1"
    chains = sorted(df["chain"].unique())
    info = [{"Chain": c, "Target": True, "Mapped residues": int(df[(df["chain"] == c) & (df["atom"] == "CA")].shape[0]),
             "UniProt range": "", "Identity to UniProt": 1.0, "Calculation status": "OK", "Warning": ""} for c in chains]
    return AFModel(name, "hybrid", "multimer" if len(chains) > 1 else "monomer", df, chain_info=info, plddt_in_b=False)


# ---------------------------------------------------------------------------
# Clashes
# ---------------------------------------------------------------------------
def heavy_atom_clashes(atoms: pd.DataFrame, cutoff: float, max_rows: int = 2000) -> pd.DataFrame:
    """Non-bonded heavy-atom pairs closer than ``cutoff`` (same and sequence-adjacent residues excluded;
    glycan C1–Asn ND2/CG excluded; ions and waters excluded). The closest ``max_rows`` pairs are listed."""
    from scipy.spatial import cKDTree

    a = atoms.reset_index(drop=True)
    X = a[["x", "y", "z"]].to_numpy(float)
    cols = ["Atom 1", "Atom 2", "Distance (Å)", "Source 1", "Source 2"]
    pairs = cKDTree(X).query_pairs(cutoff, output_type="ndarray")
    if not len(pairs):
        return pd.DataFrame(columns=cols)
    i, j = pairs[:, 0], pairs[:, 1]
    kind, ch = a["kind"].to_numpy(), a["chain"].astype(str).to_numpy()
    rn, an = a["resnum"].astype(int).to_numpy(), a["atom"].astype(str).to_numpy()
    keep = ~np.isin(kind[i], ["ion", "water"]) & ~np.isin(kind[j], ["ion", "water"])
    same_res = (ch[i] == ch[j]) & (kind[i] == kind[j]) & (rn[i] == rn[j])
    adjacent = (ch[i] == ch[j]) & (kind[i] == "protein") & (kind[j] == "protein") & (np.abs(rn[i] - rn[j]) == 1)
    link = (((kind[i] == "glycan") & (kind[j] == "protein")) | ((kind[i] == "protein") & (kind[j] == "glycan"))) & \
        (np.isin(an[i], ["C1"]) | np.isin(an[j], ["C1"])) & (np.isin(an[i], ["ND2", "CG"]) | np.isin(an[j], ["ND2", "CG"]))
    keep &= ~same_res & ~adjacent & ~link
    i, j = i[keep], j[keep]
    if not len(i):
        return pd.DataFrame(columns=cols)
    d = np.linalg.norm(X[i] - X[j], axis=1)
    o = np.argsort(d)[:max_rows]
    i, j, d = i[o], j[o], d[o]
    res = a["resname"].astype(str).to_numpy()
    src = a["source"].astype(str).to_numpy()
    lab = lambda k: [f"{c}/{r}{n}:{t}" for c, r, n, t in zip(ch[k], res[k], rn[k], an[k])]  # noqa: E731
    df = pd.DataFrame({"Atom 1": lab(i), "Atom 2": lab(j), "Distance (Å)": np.round(d, 2),
                       "Source 1": src[i], "Source 2": src[j]})
    df.attrs["total_pairs"] = int(keep.sum())
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclass
class PreparationResults:
    tables: dict[str, pd.DataFrame]
    atoms: pd.DataFrame
    names: dict
    issues: IssueLog
    files: dict = field(default_factory=dict)
    provenance: list = field(default_factory=list)
    input_files: list = field(default_factory=list)
    placement: Any = None


def _hash_all(paths: list[Path]) -> dict[str, str]:
    return {str(p): sha256_file(p) for p in paths if Path(p).exists()}


def _step(prov: list, step: str, tool: str, inputs: list, outputs: list, params: dict | None = None) -> None:
    prov.append({"step": len(prov) + 1, "name": step, "tool": tool,
                 "time_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                 "inputs": [{"path": str(p), "sha256": sha256_file(Path(p)) if Path(p).exists() else None} for p in inputs],
                 "outputs": [{"path": str(p), "sha256": sha256_file(Path(p)) if Path(p).exists() else None} for p in outputs],
                 "parameters": params or {}})


def run_preparation(cfg: Any, survey: dict[str, pd.DataFrame], alphafold: dict[str, pd.DataFrame] | None = None,
                    sess: Any = None, issues: IssueLog | None = None, template_asm: pd.DataFrame | None = None,
                    anchor: AFModel | None = None, donors: dict[str, pd.DataFrame] | None = None,
                    uniprot: dict | None = None, run_pka: bool = True,
                    template_files: list[Path] | None = None) -> PreparationResults:
    """Milestone 5: hybrid model → glycans → Zn site → protonation → membrane → files, readiness, checks.

    ``template_asm`` / ``anchor`` / ``donors`` / ``uniprot`` are injectable for tests; by default they
    are loaded from the cache (template and donors with gemmi, the anchor from notebook 03's AlphaFold2 run).
    """
    from .alphafold import region_sets
    from .cache import session_from_config
    from .membrane import place_membrane, placement_metrics, uniprot_topology
    from .metal_sites import treatment_options, treatment_status, zn_coordination
    from .survey import http_kwargs
    from .uniprot import uniprot_entry, uniprot_info

    issues = issues if issues is not None else IssueLog()
    pc = cfg.get("preparation", {}) or {}
    template = str(pc.get("template", "9S7U")).upper()
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(pc.get("name", f"hybrid_{template}")))
    out = cfg.outputs_dir / "model_preparation" / name
    (out / "work").mkdir(parents=True, exist_ok=True)
    (out / "microstates").mkdir(parents=True, exist_ok=True)
    sess = sess or session_from_config(cfg)
    json_kw, _ = http_kwargs(cfg)
    entry = uniprot if uniprot is not None else uniprot_entry(sess, cfg.accession, cfg.cache_dir, cfg.endpoint("uniprot_json"), **json_kw)
    useq = uniprot_info(entry)[0]
    prov: list = []

    # ---- inputs (originals hashed now and re-checked at the end) --------------------------------
    files: list[Path] = list(template_files or [])
    if template_asm is None:
        template_asm, tf, asm_label = load_template_assembly(cfg, survey, template, sess, issues)
        files += tf
    else:
        asm_label = "provided"
    if anchor is None:
        anchor = load_anchor_model(cfg, useq, issues)
    if anchor is None:
        raise RuntimeError("no anchor model: see issues")
    files += [Path(anchor.meta["File"])] + ([Path(anchor.meta["Scores file"])] if anchor.meta.get("Scores file") else [])
    before = _hash_all(files)

    # ---- hybrid --------------------------------------------------------------------------------
    hyb = build_hybrid(template_asm, anchor, useq, cfg, issues, template)
    _step(prov, "hybrid model", "psma_pipeline.model_preparation.build_hybrid", files, [],
          {"template": template, "template assembly": asm_label, "anchor": anchor.model_id, **hyb.fit,
           "local_flank": pc.get("local_flank"), "junction_search": pc.get("junction_search")})

    # ---- glycans -------------------------------------------------------------------------------
    obs = sequon_observations(survey, useq)
    gcfg = pc.get("glycans", {}) or {}
    atoms = hyb.atoms
    if gcfg.get("strategy", "core_glcnac") != "none":
        have = set(atoms.loc[atoms["kind"] == "glycan", "link"].dropna()) if "link" in atoms else set()
        needed = obs[(obs["Entries with glycan"] >= int(gcfg.get("min_observed_structures", 1)))]
        if donors is None:
            donors = {}
            cand = []
            for ex in needed["Example entries"]:
                cand += [p.strip() for p in str(ex).split(",") if p.strip()]
            rank = survey.get("TemplateRanking", pd.DataFrame())
            order = rank["PDB ID"].tolist() if len(rank) else []
            cand = sorted(set(cand) - {template}, key=lambda p: order.index(p) if p in order else 999)
            for pdb in cand[: int(gcfg.get("donors_max", 10))]:
                try:
                    donors[pdb] = load_donor_atoms(cfg, survey, pdb, sess)
                    files.append(cfg.cache_dir / "mmcif" / f"{pdb}.cif.gz")
                except Exception as ex:  # noqa: BLE001
                    issues.add("preparation", Status.WARNING, f"glycan donor unavailable: {ex}", pdb_id=pdb)
            before.update(_hash_all([f for f in files if str(f) not in before]))
    atoms, glyc = apply_glycan_strategy(atoms, obs, donors or {}, cfg, template, issues)
    _step(prov, "N-glycans (core GlcNAc)", "psma_pipeline.model_preparation.apply_glycan_strategy",
          [cfg.cache_dir / "mmcif" / f"{p}.cif.gz" for p in (donors or {})], [],
          {"strategy": gcfg.get("strategy"), "placed": int((glyc["Decision"] == "core GlcNAc").sum()) if len(glyc) else 0})

    # ---- Zn site ------------------------------------------------------------------------------
    zn_tab, zn_contacts = zn_coordination(atoms, cfg)
    t_status, t_detail = treatment_status(cfg)

    # ---- protonation ---------------------------------------------------------------------------
    prot_pdb = out / "work" / "hybrid_protein_for_propka.pdb"
    prot_pdb.write_text(pdb_text(atoms[atoms["kind"] == "protein"]))
    if run_pka:
        pka, pka_note = run_propka(prot_pdb)
    else:
        pka, pka_note = None, "PROPKA disabled"
    if pka is None:
        issues.add("preparation", Status.WARNING, f"{pka_note}: default pKa values used")
    ptab, names, alts = assign_protonation(atoms, pka, zn_contacts, cfg, pka_note)
    _step(prov, "protonation", pka_note if pka is not None else "default pKa values", [prot_pdb],
          [prot_pdb.with_suffix(".pka")] if prot_pdb.with_suffix(".pka").exists() else [],
          {"pH": cfg.get("preparation.protonation.ph"), "window": cfg.get("preparation.protonation.window"),
           "always_alternative": cfg.get("preparation.protonation.always_alternative")})

    # ---- clashes, membrane -----------------------------------------------------------------------
    clashes = heavy_atom_clashes(atoms, float(pc.get("clash_heavy_min", 2.2)))
    topo = (alphafold or {}).get("MembraneTopology")
    labels = "".join(topo["Membrane label"]) if topo is not None and len(topo) else uniprot_topology(entry, len(useq))
    model = hybrid_as_model(atoms, f"HYBRID:{name}", useq)
    pl = place_membrane(model, labels, cfg, issues)
    sites = sorted({int(re.sub(r"\D", "", c)) for c in cfg.active_site.get("literature_sites", {})})
    place = pd.DataFrame([placement_metrics(model, pl, cfg, useq, region_sets(cfg, useq, sites))]) if pl else pd.DataFrame()

    # ---- files ---------------------------------------------------------------------------------
    written = {}
    remarks = [f"psma_pipeline Milestone 5 hybrid: {template} ({asm_label}) + {anchor.model_id}",
               "heavy atoms only; protonation states in residue names (AMBER: HID/HIE/HIP, ASH, GLH, LYN, CYX)",
               "NOT MD-READY: see PrepReadiness (Zn treatment, parameters - Milestone 6)"]
    p = out / "hybrid_template_frame.pdb"
    p.write_text(pdb_text(atoms, names, remarks + ["coordinates in the template frame"]))
    written["template_frame"] = p
    if pl is not None:
        rot, shift = pl.membrane.frame()
        p = out / "hybrid_membrane_oriented.pdb"
        p.write_text(pdb_text(atoms, names, remarks + ["membrane normal = +z, mid-plane z = 0, cytoplasm at z < 0"],
                              rot=rot, shift=shift))
        written["membrane_oriented"] = p
        from tmfold.geometry import membrane_dummy_atoms
        dum = membrane_dummy_atoms(pl.membrane, model.atoms[model.atoms["atom"] == "CA"][["x", "y", "z"]].to_numpy(), oriented=True)
        p = out / "membrane_planes_dum.pdb"
        p.write_text("\n".join(dum + ["END"]) + "\n")
        written["membrane_planes"] = p
    ms_rows = []
    for tag, nm, desc in microstates(names, alts, int(cfg.get("preparation.protonation.max_microstates", 12))):
        p = out / "microstates" / f"{tag}.pdb"
        kw = {"rot": pl.membrane.frame()[0], "shift": pl.membrane.frame()[1]} if pl is not None else {}
        p.write_text(pdb_text(atoms, nm, remarks + [f"microstate: {desc}"], **kw))
        ms_rows.append({"Microstate": tag, "Change from default": desc, "File": str(p)})
    written["microstates"] = out / "microstates"
    _step(prov, "write prepared model", "psma_pipeline.model_preparation.pdb_text", [],
          [v for k, v in written.items() if k != "microstates"] + [Path(r["File"]) for r in ms_rows],
          {"frame": "membrane (z = normal)" if pl is not None else "template"})

    # ---- readiness and checks -------------------------------------------------------------------
    after = _hash_all(list(before))
    tables = {
        "PrepSummary": pd.DataFrame([{"Item": k, "Value": v} for k, v in {
            "Template": f"{template} ({asm_label})", "Anchor model": anchor.model_id, **hyb.fit,
            "Residues per chain": len(useq), "Chains": ", ".join(sorted(atoms.loc[atoms['kind'] == 'protein', 'chain'].unique())),
            "Zn ions": int((atoms["resname"] == "ZN").groupby([atoms["chain"], atoms["resnum"]]).any().sum()),
            "Core GlcNAc placed": int((glyc["Decision"] == "core GlcNAc").sum()) if len(glyc) else 0,
            "Zn-shell waters": int(atoms.loc[atoms["kind"] == "water", ["chain", "resnum"]].drop_duplicates().shape[0]),
            "Protonation": pka_note, "Alternative microstates": len(ms_rows) - 1,
            "Zn MD treatment": cfg.get("preparation.zn_md_treatment") or "not chosen",
            "Output folder": str(out)}.items()]),
        "HybridSegments": hyb.segments, "ResidueSources": hyb.residue_sources, "Junctions": hyb.junctions,
        "Repairs": hyb.repairs, "TemplateHetero": hyb.hetero, "Glycans": glyc, "SequonObservations": obs,
        "ZnSite": zn_tab, "ZnContacts": zn_contacts, "MetalTreatmentOptions": treatment_options(cfg),
        "Protonation": ptab, "Microstates": pd.DataFrame(ms_rows), "Clashes": clashes,
        "HybridMembrane": place,
    }
    tables["PrepReadiness"] = readiness(tables, cfg, t_status, t_detail, pka is not None, useq)
    tables["PrepChecks"] = preparation_checks(atoms, useq, before, after, zn_tab, hyb.junctions, clashes, ms_rows, prov,
                                              written, cfg)
    (out / "provenance.json").write_text(json.dumps({"originals": before, "steps": prov}, indent=1))
    written["provenance"] = out / "provenance.json"
    (out / "README.md").write_text(_readme(template, anchor.model_id, tables["PrepReadiness"]))
    return PreparationResults(tables, atoms, names, issues, written, prov, list(before), pl)


def readiness(t: dict, cfg: Any, t_status: Status, t_detail: str, propka_ok: bool, useq: str) -> pd.DataFrame:
    rows = []
    add = lambda item, st, d: rows.append({"Item": item, "Status": st.value, "Detail": d})  # noqa: E731
    src = t["ResidueSources"]
    n_ch = src["Chain"].nunique() if len(src) else 0
    complete = len(src) == n_ch * len(useq) and n_ch >= 2
    add("hybrid model complete", Status.PASS if complete else Status.FAIL,
        f"{n_ch} chains × {len(useq)} residues" + ("" if complete else " (residues missing: see PrepIssues)"))
    j = t["Junctions"]
    bad = j[j["Status"] != Status.PASS.value] if len(j) else j
    add("graft junctions", Status.PASS if not len(bad) else Status.WARNING,
        f"{len(j)} junctions; " + (", ".join(f"{r.Chain}{r.Junction}" for r in bad.itertuples()) + " stretched (minimise first)"
                                  if len(bad) else "all peptide C–N within limit"))
    c = t["Clashes"]
    n_c = int(c.attrs.get("total_pairs", len(c)))
    add("heavy-atom clashes", Status.PASS if not n_c else Status.WARNING,
        f"{n_c} pairs < {cfg.get('preparation.clash_heavy_min')} Å" + (" (resolved by minimisation in Milestone 6)" if n_c else ""))
    zn = t["ZnSite"]
    n_zn = len(zn)
    exp = int(cfg.active_site.get("metal_site", {}).get("expected_zn_per_protomer", 2)) * max(n_ch, 1)
    zst = Status.FAIL if n_zn != exp or (len(zn) and (zn["Status"] == Status.FAIL.value).any()) else \
        Status.WARNING if (zn["Status"] == Status.WARNING.value).any() else Status.PASS
    add("zinc ions placed (experimental)", zst, f"{n_zn} of {exp}; geometry: " + (", ".join(f"{r.Zn} {r.Status}" for r in zn.itertuples()) if n_zn else "none"))
    add("zinc MD treatment", t_status, t_detail)
    br = int(zn["Bridging water/hydroxide"].clip(upper=1).sum()) if len(zn) and "Bridging water/hydroxide" in zn else 0
    add("bridging water / hydroxide", Status.WARNING,
        ("present in the template; decide water vs hydroxide together with the Zn treatment" if br else
         "not in the template; add one with the Zn treatment if the chosen model needs it"))
    het = t["TemplateHetero"]
    removed = het[het["Decision"] == "removed"] if len(het) else het
    add("ligands / additives", Status.PASS, "removed (apo): " + (", ".join(sorted(set(removed["Component"]))) if len(removed) else "none"))
    g = t["Glycans"]
    placed = int((g["Decision"] == "core GlcNAc").sum()) if len(g) else 0
    missed = g[g["Note"].astype(str).str.startswith("observed")] if len(g) else g
    add("N-glycans (core GlcNAc)", Status.WARNING if len(missed) else Status.PASS,
        f"{placed} placed; {len(missed)} observed sites not placed; never-observed sites left unglycosylated "
        "(glycan parameters in Milestone 6)")
    add("protonation states", Status.PASS if propka_ok else Status.WARNING,
        ("PROPKA" if propka_ok else "default pKa values (PROPKA unavailable)") +
        f" at pH {cfg.get('preparation.protonation.ph')}; {len(t['Microstates']) - 1} alternative microstates written")
    hm = t["HybridMembrane"]
    if len(hm):
        r = hm.iloc[0]
        add("membrane placement", Status.PASS if r["Orientation reliability"] == "supported" else Status.WARNING,
            f"{r['Orientation reliability']} ({r['Placement method']}): {r['Reliability reasons'] or 'all checks passed'}")
    else:
        add("membrane placement", Status.FAIL, "not placed")
    b = cfg.get("membrane.builder", {})
    add("lipid composition", Status.WARNING, f"placeholder {b.get('lipids')} {b.get('ratio')}: confirm before building")
    add("force field and parameters", Status.NOT_RUN, "protein, Zn, glycans, ions, lipids, water: Milestone 6")
    add("OVERALL", Status.FAIL, "NOT READY for MD: prepared model only. Choose the Zn treatment, then build and "
        "parameterise the system in Milestone 6 (no READY status while any item fails).")
    return pd.DataFrame(rows)


def preparation_checks(atoms, useq, before, after, zn_tab, junctions, clashes, ms_rows, prov, written, cfg,
                       critical=("numbering_identity", "originals_unchanged", "zn_site_complete",
                                 "provenance_complete")) -> pd.DataFrame:
    rows = []
    add = lambda c, st, d, chain=None: rows.append({"Check": c, "Status": st.value, "PDB ID": None, "Chain": chain, "Detail": d})  # noqa: E731
    prot = atoms[(atoms["kind"] == "protein") & (atoms["atom"] == "CA")]
    for ch, g in prot.groupby("chain"):
        seq = "".join(THREE_TO_ONE.get(r, "X") for r in g.sort_values("resnum")["resname"])
        nums = g["resnum"].astype(int).tolist()
        ok = seq == useq and nums == list(range(1, len(useq) + 1))
        add("numbering_identity", Status.PASS if ok else Status.FAIL,
            "sequence and numbering equal UniProt" if ok else f"{sum(a != b for a, b in zip(seq, useq))} mismatches, "
            f"{len(useq) - len(seq)} missing", ch)
    changed = [p for p in before if before[p] != after.get(p)]
    add("originals_unchanged", Status.PASS if not changed else Status.FAIL,
        f"{len(before)} original files unchanged" if not changed else "modified: " + ", ".join(changed))
    exp = int(cfg.active_site.get("metal_site", {}).get("expected_zn_per_protomer", 2))
    for ch in sorted(prot["chain"].unique()):
        z = zn_tab[zn_tab["Chain"] == ch] if len(zn_tab) else zn_tab
        ok = len(z) == exp and not (z["Status"] == Status.FAIL.value).any()
        add("zn_site_complete", Status.PASS if ok else Status.FAIL,
            f"{len(z)} Zn, sites {', '.join(z['Site'].astype(str))}" if len(z) else "no Zn", ch)
    out_files = {o["path"] for s in prov for o in s["outputs"]}
    missing = [str(v) for k, v in written.items() if k not in ("microstates",) and str(v) not in out_files]
    add("provenance_complete", Status.PASS if not missing else Status.FAIL,
        f"{len(out_files)} output files recorded" if not missing else "not recorded: " + ", ".join(missing))
    bad = junctions[junctions["Status"] != Status.PASS.value] if len(junctions) else junctions
    add("junction_geometry", Status.PASS if not len(bad) else Status.WARNING, f"{len(bad)} stretched of {len(junctions)}")
    n_c = int(clashes.attrs.get("total_pairs", len(clashes)))
    add("heavy_atom_clashes", Status.PASS if not n_c else Status.WARNING, f"{n_c} pairs" + (
        f" ({len(clashes)} closest listed)" if n_c > len(clashes) else ""))
    add("microstates_written", Status.PASS if ms_rows else Status.FAIL, f"{len(ms_rows)} files (incl. default)")
    df = pd.DataFrame(rows)
    df.insert(1, "Critical", df["Check"].isin(set(critical)))
    return df


def _readme(template: str, anchor: str, ready: pd.DataFrame) -> str:
    items = "\n".join(f"- **{r.Item}**: {r.Status} — {r.Detail}" for r in ready.itertuples())
    return (f"# Prepared hybrid model ({template} ectodomain + {anchor} membrane anchor)\n\n"
            "- `hybrid_membrane_oriented.pdb`: default microstate, membrane normal along +z (cytoplasm at z < 0)\n"
            "- `hybrid_template_frame.pdb`: same model in the template's coordinate frame\n"
            "- `microstates/`: default and alternative protonation microstates (residue names only differ)\n"
            "- `membrane_planes_dum.pdb`: membrane planes as dummy atoms (visual check)\n"
            "- `provenance.json`: every step with input/output hashes\n\n"
            f"## Readiness\n\n{items}\n")


def save_preparation(cfg: Any, res: PreparationResults, tag: str = "") -> dict[str, Path]:
    """Workbook, CSV/Parquet tables and manifest for Milestone 5."""
    from .excel_output import write_tables, write_workbook
    from .manifest import build_manifest, detect_software, manifest_table, write_manifest

    software = detect_software()
    sheets = dict(res.tables)
    sheets["PrepIssues"] = res.issues.to_frame()
    sheets["Software"] = software
    manifest = build_manifest(cfg, structure_ids=[str(cfg.get("preparation.template"))],
                              input_files=[Path(p) for p in res.input_files], software=software,
                              extra={"stage": "model preparation (notebook 04)", "provenance_steps": len(res.provenance),
                                     "issue_counts": {s.value: res.issues.count(s) for s in Status}})
    sheets["Reproducibility"] = manifest_table(manifest)
    suffix = f"_{tag}" if tag else ""
    wb = cfg.output_file("preparation_workbook")
    wb = wb.with_name(wb.stem + suffix + ".xlsx")
    write_workbook(wb, sheets)
    mpath = write_manifest(manifest, cfg.outputs_dir / f"{cfg.accession}_preparation_manifest{suffix}.json")
    tdir = cfg.outputs_dir / f"preparation_tables{suffix}"
    write_tables(tdir, sheets, csv=bool(cfg.get("outputs.write_csv", True)), parquet=bool(cfg.get("outputs.write_parquet", True)))
    return {"workbook": wb, "manifest": mpath, "tables_dir": tdir, **{k: v for k, v in res.files.items()}}
