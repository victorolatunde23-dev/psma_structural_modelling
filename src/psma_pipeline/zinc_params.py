"""Bonded parameters for the binuclear Zn site with MCPB.py + PySCF (Milestone 6, part A).

Design: ``docs/M6_ZINC_PARAMETERS.md``. Workflow (all inside Colab):

1. **Site PDB** from the Milestone 5 default microstate: chain A protein, its two Zn²⁺ and the
   bridging water turned into a hydroxide residue ``OHX`` (H placed away from the Zn–Zn axis,
   user decision ``preparation.zn_bridge: hydroxide``; Glu424 is GLH in the default microstate).
   Hydrogens are added to the protein by tleap (ff14SB templates for HID/HIE/GLH/…).
2. **MCPB.py step 1** builds the small/large models and writes Gaussian inputs.
3. **QM** with PySCF through :mod:`qm_bridge` (B3LYP/6-31G*, Gaussian-equivalent settings),
   writing the ``.fchk`` / ``.log`` files MCPB.py reads (``software_version g09``).
4. **MCPB.py steps 2–4**: Seminario force constants, RESP charges, tleap input.
5. **Validation**: Zn–ligand equilibrium lengths vs the experimental site, force constants,
   RESP charges (integer residue sums), and a tleap build of the parameterised chain.

Both sites are chemically identical (symmetric dimer), so the chain A site is parameterised once
and the same residue templates and bonds are applied to chain B when the full system is built.
Original files are never modified; every step is recorded in ``provenance.json``.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cache import sha256_file
from .status import IssueLog, Status

log = logging.getLogger("psma_pipeline.zinc_params")

ZN_LIGANDS_DEFAULT = {"Zn1": ["H377", "D387", "D453"], "Zn2": ["D387", "E425", "H553"]}


# ---------------------------------------------------------------------------
# PDB helpers (keep every record: protonation names such as HID/GLH are preserved)
# ---------------------------------------------------------------------------
def read_pdb(path: str | Path) -> pd.DataFrame:
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        rows.append({"record": line[:6].strip(), "atom": line[12:16].strip(), "resname": line[17:20].strip(),
                     "chain": line[21].strip(), "resnum": int(line[22:26]), "x": float(line[30:38]),
                     "y": float(line[38:46]), "z": float(line[46:54]),
                     "element": (line[76:78].strip() or line[12:16].strip()[0]).capitalize()})
    return pd.DataFrame(rows)


def pdb_line(serial: int, record: str, atom: str, resname: str, chain: str, resnum: int, xyz, element: str) -> str:
    name = atom if len(atom) == 4 else f" {atom:<3s}"
    return (f"{record:<6s}{serial % 100000:5d} {name:4s} {resname:>3s} {chain[:1] or ' '}{resnum % 10000:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {element.upper():>2s}")


def write_pdb(df: pd.DataFrame, path: Path, ter_after_protein: bool = True) -> Path:
    lines, last_rec = [], None
    for k, r in enumerate(df.itertuples(), 1):
        if ter_after_protein and last_rec == "ATOM" and r.record == "HETATM":
            lines.append("TER")
        lines.append(pdb_line(k, r.record, r.atom, r.resname, r.chain, r.resnum, (r.x, r.y, r.z), r.element))
        last_rec = r.record
    lines += ["TER", "END"]
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Site extraction
# ---------------------------------------------------------------------------
@dataclass
class Site:
    protein: pd.DataFrame          # chain A protein heavy atoms (ATOM records)
    zinc: pd.DataFrame             # two Zn
    bridge_o: np.ndarray           # bridging O (Å)
    hydroxide_h: np.ndarray | None # placed H (Å) when the bridge is a hydroxide
    info: dict = field(default_factory=dict)


def extract_site(pdb: pd.DataFrame, chain: str = "A", bridge: str = "hydroxide", bridge_max: float = 2.6) -> Site:
    """Chain ``chain`` protein, its two Zn and the bridging water (O within ``bridge_max`` of both Zn)."""
    prot = pdb[(pdb["record"] == "ATOM") & (pdb["chain"] == chain)].copy()
    zn = pdb[(pdb["resname"] == "ZN") & (pdb["chain"] == chain)].copy()
    if len(zn) != 2:
        raise ValueError(f"chain {chain}: expected 2 Zn, found {len(zn)}")
    Z = zn[["x", "y", "z"]].to_numpy()
    w = pdb[(pdb["resname"] == "HOH") & (pdb["atom"] == "O") & (pdb["chain"] == chain)]
    W = w[["x", "y", "z"]].to_numpy()
    if not len(W):
        raise ValueError(f"chain {chain}: no water near the Zn site")
    d = np.linalg.norm(W[:, None, :] - Z[None, :, :], axis=2)
    ok = (d <= bridge_max).all(axis=1)
    if not ok.any():
        raise ValueError(f"chain {chain}: no water within {bridge_max} Å of both Zn (closest {d.max(axis=1).min():.2f} Å)")
    k = int(np.argmin(np.where(ok, d.max(axis=1), np.inf)))
    o = W[k]
    h = None
    if bridge == "hydroxide":
        mid = Z.mean(axis=0)
        v = o - mid
        if np.linalg.norm(v) < 0.3:                  # O on the Zn–Zn axis: use any perpendicular
            ax = Z[1] - Z[0]
            v = np.cross(ax, [1.0, 0.0, 0.0] if abs(ax[0]) < 0.9 * np.linalg.norm(ax) else [0.0, 1.0, 0.0])
        h = o + 0.97 * v / np.linalg.norm(v)
    info = {"Zn–Zn (Å)": round(float(np.linalg.norm(Z[0] - Z[1])), 3),
            "bridging O – Zn (Å)": ", ".join(f"{x:.3f}" for x in d[k]),
            "bridging species": bridge}
    return Site(prot, zn, o, h, info)


def site_hetero_records(site: Site, first_resnum: int) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(site.zinc.itertuples()):
        rows.append({"record": "HETATM", "atom": "ZN", "resname": "ZN", "chain": "", "resnum": first_resnum + i,
                     "x": r.x, "y": r.y, "z": r.z, "element": "Zn"})
    n = first_resnum + len(site.zinc)
    rows.append({"record": "HETATM", "atom": "O", "resname": "OHX", "chain": "", "resnum": n,
                 "x": site.bridge_o[0], "y": site.bridge_o[1], "z": site.bridge_o[2], "element": "O"})
    if site.hydroxide_h is not None:
        rows.append({"record": "HETATM", "atom": "H", "resname": "OHX", "chain": "", "resnum": n,
                     "x": site.hydroxide_h[0], "y": site.hydroxide_h[1], "z": site.hydroxide_h[2], "element": "H"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# mol2 inputs for MCPB.py (format read by pymsmt.mol.mol2io)
# ---------------------------------------------------------------------------
def zn_mol2(xyz) -> str:
    return ("@<TRIPOS>MOLECULE\nZN\n    1     0     1     0     1\nSMALL\nUSER_CHARGES\n@<TRIPOS>ATOM\n"
            f"      1 ZN      {xyz[0]:10.4f}{xyz[1]:10.4f}{xyz[2]:10.4f} Zn         1 ZN        2.000000\n"
            "@<TRIPOS>BOND\n@<TRIPOS>SUBSTRUCTURE\n     1 ZN          1 ****              0 ****  ****    0 ROOT\n")


def hydroxide_mol2(o, h) -> str:
    """OH⁻ residue with GAFF types (oh/ho); charges are refitted by MCPB.py's RESP step."""
    return ("@<TRIPOS>MOLECULE\nOHX\n    2     1     1     0     1\nSMALL\nUSER_CHARGES\n@<TRIPOS>ATOM\n"
            f"      1 O       {o[0]:10.4f}{o[1]:10.4f}{o[2]:10.4f} oh         1 OHX      -1.300000\n"
            f"      2 H       {h[0]:10.4f}{h[1]:10.4f}{h[2]:10.4f} ho         1 OHX       0.300000\n"
            "@<TRIPOS>BOND\n     1     1     2 1\n"
            "@<TRIPOS>SUBSTRUCTURE\n     1 OHX         1 ****              0 ****  ****    0 ROOT\n")


def water_mol2(o) -> str:
    return ("@<TRIPOS>MOLECULE\nWTB\n    1     0     1     0     1\nSMALL\nUSER_CHARGES\n@<TRIPOS>ATOM\n"
            f"      1 O       {o[0]:10.4f}{o[1]:10.4f}{o[2]:10.4f} ow         1 WTB      -0.834000\n"
            "@<TRIPOS>BOND\n@<TRIPOS>SUBSTRUCTURE\n     1 WTB         1 ****              0 ****  ****    0 ROOT\n")


def mcpb_input(group: str, pdb_name: str, ion_ids: list[int], cfg: Any, has_ohx: bool = True) -> str:
    m = cfg.get("md.mcpb", {}) or {}
    lines = [f"original_pdb {pdb_name}", f"group_name {group}", f"cut_off {m.get('cut_off', 2.5)}",
             "ion_ids " + " ".join(str(i) for i in ion_ids), "ion_mol2files ZN.mol2",
             f"force_field {cfg.get('md.protein_ff', 'ff14SB')}", f"water_model {cfg.get('md.water_model', 'tip3p')}",
             f"gaff {m.get('gaff', 1)}", f"ion_paraset {m.get('ion_paraset', 'cm')}",
             f"large_opt {m.get('large_opt', 0)}", f"scale_factor {m.get('scale_factor', 0.9614)}",
             f"bondfc_avg {m.get('bondfc_avg', 1)}", f"anglefc_avg {m.get('anglefc_avg', 1)}",
             "software_version g09"]
    if has_ohx:
        lines += ["naa_mol2files OHX.mol2", "frcmod_files OHX.frcmod"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Parsers for MCPB.py / tleap outputs
# ---------------------------------------------------------------------------
def parse_frcmod(text: str) -> dict[str, pd.DataFrame]:
    """BOND / ANGL / DIHE / IMPR / NONB sections of an Amber frcmod."""
    sections: dict[str, list] = {}
    cur = None
    for line in text.splitlines()[1:]:
        key = line.strip().upper()[:4]
        if key in ("MASS", "BOND", "ANGL", "DIHE", "IMPR", "NONB"):
            cur = key
            sections[cur] = []
            continue
        if not line.strip() or cur is None:
            continue
        sections[cur].append(line.rstrip())
    out = {}
    if "BOND" in sections:
        rows = []
        for l in sections["BOND"]:
            m = re.match(r"\s*(\S{1,2}\s*-\s*\S{1,2})\s+(-?[\d.]+)\s+(-?[\d.]+)\s*(.*)", l)
            if m:
                rows.append({"Bond": re.sub(r"\s+", "", m.group(1)), "k (kcal/mol/Å²)": float(m.group(2)),
                             "r0 (Å)": float(m.group(3)), "Comment": m.group(4).strip()})
        out["BOND"] = pd.DataFrame(rows)
    if "ANGL" in sections:
        rows = []
        for l in sections["ANGL"]:
            m = re.match(r"\s*(\S{1,2}\s*-\s*\S{1,2}\s*-\s*\S{1,2})\s+(-?[\d.]+)\s+(-?[\d.]+)\s*(.*)", l)
            if m:
                rows.append({"Angle": re.sub(r"\s+", "", m.group(1)), "k (kcal/mol/rad²)": float(m.group(2)),
                             "θ0 (°)": float(m.group(3)), "Comment": m.group(4).strip()})
        out["ANGL"] = pd.DataFrame(rows)
    if "NONB" in sections:
        rows = []
        for l in sections["NONB"]:
            t = l.split()
            if len(t) >= 3:
                rows.append({"Type": t[0], "R* (Å)": float(t[1]), "ε (kcal/mol)": float(t[2])})
        out["NONB"] = pd.DataFrame(rows)
    return out


def parse_mol2_charges(text: str) -> pd.DataFrame:
    rows, on = [], False
    for line in text.splitlines():
        if line.startswith("@<TRIPOS>"):
            on = line.strip() == "@<TRIPOS>ATOM"
            continue
        if on and line.strip():
            t = line.split()
            if len(t) >= 9:
                rows.append({"Atom": t[1], "Type": t[5], "Residue": t[7], "Charge": float(t[8])})
    return pd.DataFrame(rows)


def parse_tleap_log(text: str) -> dict:
    err = re.findall(r"Exiting LEaP: Errors = (\d+); Warnings = (\d+); Notes = (\d+)", text)
    charge = re.findall(r"Total (?:unperturbed )?charge:\s*(-?[\d.]+)", text)
    missing = re.findall(r"Could not find (?:bond|angle|torsion) parameter for[^\n]*", text)
    unknown = re.findall(r"(?:Unknown residue|FATAL:  Atom [^\n]*does not have a type)[^\n]*", text)
    return {"errors": int(err[-1][0]) if err else None, "warnings": int(err[-1][1]) if err else None,
            "total charge": float(charge[-1]) if charge else None, "missing parameters": missing,
            "unknown": unknown}


def site_bond_table(mcpb_pdb: pd.DataFrame, bonds: list[tuple[str, str]], site_ref: pd.DataFrame | None,
                    frcmod: dict[str, pd.DataFrame], types: dict[tuple[int, str], str]) -> pd.DataFrame:
    """Zn–ligand bonds: crystal distance, MCPB r0 and k for each bond in the tleap input."""
    idx = {(int(r.resnum), r.atom): np.array([r.x, r.y, r.z]) for r in mcpb_pdb.itertuples()}
    names = {(int(r.resnum), r.atom): r.resname for r in mcpb_pdb.itertuples()}
    bt = frcmod.get("BOND", pd.DataFrame())
    rows = []
    for a, b in bonds:
        ra, aa = a.split(".")[1:]
        rb, ab = b.split(".")[1:]
        ka, kb = (int(ra), aa), (int(rb), ab)
        if ka not in idx or kb not in idx:
            continue
        d = float(np.linalg.norm(idx[ka] - idx[kb]))
        ta, tb = types.get(ka, "?"), types.get(kb, "?")
        prm = bt[bt["Bond"].isin([f"{ta}-{tb}", f"{tb}-{ta}"])] if len(bt) else bt
        rows.append({"Atom 1": f"{names[ka]}{ka[0]}:{ka[1]}", "Atom 2": f"{names[kb]}{kb[0]}:{kb[1]}",
                     "Types": f"{ta}-{tb}", "Model distance (Å)": round(d, 3),
                     "MCPB r0 (Å)": float(prm["r0 (Å)"].iloc[0]) if len(prm) else None,
                     "MCPB k (kcal/mol/Å²)": float(prm["k (kcal/mol/Å²)"].iloc[0]) if len(prm) else None})
    df = pd.DataFrame(rows)
    if len(df):
        df["r0 − model (Å)"] = (df["MCPB r0 (Å)"] - df["Model distance (Å)"]).round(3)
    return df


# ---------------------------------------------------------------------------
# Orchestration (runs AmberTools; Colab)
# ---------------------------------------------------------------------------
def _run(cmd: list[str] | str, cwd: Path, logname: str, shell: bool = False) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, shell=shell)
    (cwd / logname).write_text((r.stdout or "") + "\n--- stderr ---\n" + (r.stderr or ""))
    return r


def ambertools_available() -> dict[str, str | None]:
    return {t: shutil.which(t) for t in ("tleap", "MCPB.py", "parmchk2", "resp", "antechamber")}


@dataclass
class ZincParamResults:
    tables: dict[str, pd.DataFrame]
    issues: IssueLog
    workdir: Path
    files: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)


def _step(prov, name, tool, inputs, outputs, params=None):
    prov.append({"step": len(prov) + 1, "name": name, "tool": tool,
                 "inputs": [{"path": str(p), "sha256": sha256_file(Path(p)) if Path(p).exists() else None} for p in inputs],
                 "outputs": [{"path": str(p), "sha256": sha256_file(Path(p)) if Path(p).exists() else None} for p in outputs],
                 "parameters": params or {}})


def run_zinc_parameters(cfg: Any, prepared_pdb: str | Path | None = None, issues: IssueLog | None = None,
                        use_gpu: bool | None = None, steps: tuple = ("site", "mcpb1", "qm", "mcpb2", "mcpb3", "mcpb4", "check")
                        ) -> ZincParamResults:
    """MCPB.py + PySCF parameterisation of the chain A binuclear Zn site (see module docstring)."""
    from . import qm_bridge as Q

    issues = issues if issues is not None else IssueLog()
    m = cfg.get("md.mcpb", {}) or {}
    group = m.get("group_name", "PSMA")
    chain = m.get("site_chain", "A")
    bridge = cfg.get("preparation.zn_bridge", "hydroxide")
    prep_dir = cfg.outputs_dir / "model_preparation" / cfg.get("preparation.name", "hybrid_9S7U_AF2")
    prepared_pdb = Path(prepared_pdb) if prepared_pdb else prep_dir / "microstates" / "microstate_00_default.pdb"
    wd = cfg.outputs_dir / "md_setup" / "zinc_mcpb"
    wd.mkdir(parents=True, exist_ok=True)
    prov: list = []
    originals = {str(prepared_pdb): sha256_file(prepared_pdb)}
    files: dict = {}
    tables: dict[str, pd.DataFrame] = {}
    tools = ambertools_available()
    tables["AmberTools"] = pd.DataFrame([{"Tool": k, "Path": v or "", "Available": bool(v)} for k, v in tools.items()])
    if not all(tools[t] for t in ("tleap", "MCPB.py", "parmchk2", "resp")):
        issues.add("zinc", Status.FAIL, "AmberTools not available: install it in Colab (notebook 05, section 0)")
        return ZincParamResults(tables, issues, wd)

    # ---- 1. site PDB ---------------------------------------------------------------------------
    pdb = read_pdb(prepared_pdb)
    site = extract_site(pdb, chain, bridge)
    noh = wd / "site_protein_noH.pdb"
    write_pdb(site.protein.assign(chain=""), noh)
    (wd / "addH.in").write_text(f"source leaprc.protein.{cfg.get('md.protein_ff', 'ff14SB')}\n"
                                f"m = loadpdb {noh.name}\nsavepdb m site_protein_H.pdb\nquit\n")
    r = _run(["tleap", "-f", "addH.in"], wd, "addH.log")
    if not (wd / "site_protein_H.pdb").exists():
        issues.add("zinc", Status.FAIL, f"tleap could not add hydrogens: see {wd / 'addH.log'}")
        return ZincParamResults(tables, issues, wd)
    prot_h = read_pdb(wd / "site_protein_H.pdb")
    prot_h = prot_h[prot_h["record"] == "ATOM"].assign(chain="")
    het = site_hetero_records(site, int(prot_h["resnum"].max()) + 1)
    full = pd.concat([prot_h, het], ignore_index=True)
    mcpb_pdb = wd / f"{group}_site.pdb"
    write_pdb(full, mcpb_pdb)
    ion_ids = [int(i) + 1 for i in full.index[full["resname"] == "ZN"]]
    (wd / "ZN.mol2").write_text(zn_mol2(site.zinc[["x", "y", "z"]].to_numpy()[0]))
    has_ohx = site.hydroxide_h is not None
    if has_ohx:
        (wd / "OHX.mol2").write_text(hydroxide_mol2(site.bridge_o, site.hydroxide_h))
        _run(["parmchk2", "-i", "OHX.mol2", "-f", "mol2", "-o", "OHX.frcmod", "-s", str(m.get("gaff", 1))], wd, "parmchk2.log")
    else:
        issues.add("zinc", Status.WARNING, "bridging water (not hydroxide): it is modelled as a TIP3P-typed residue WTB")
        (wd / "OHX.mol2").write_text(water_mol2(site.bridge_o))
    (wd / f"{group}.in").write_text(mcpb_input(group, mcpb_pdb.name, ion_ids, cfg, True))
    _step(prov, "site PDB + MCPB input", "psma_pipeline.zinc_params + tleap (hydrogens) + parmchk2", [prepared_pdb],
          [mcpb_pdb, wd / "ZN.mol2", wd / "OHX.mol2", wd / f"{group}.in"], {**site.info, "chain": chain, "ion_ids": ion_ids})
    tables["SiteGeometry"] = pd.DataFrame([{"Item": k, "Value": v} for k, v in site.info.items()])

    # ---- 2. MCPB.py step 1 -------------------------------------------------------------------------
    r = _run(["MCPB.py", "-i", f"{group}.in", "-s", "1"], wd, "mcpb_step1.log")
    coms = [wd / f"{group}_small_opt.com", wd / f"{group}_small_fc.com", wd / f"{group}_large_mk.com"]
    if r.returncode != 0 or not all(c.exists() for c in coms):
        issues.add("zinc", Status.FAIL, f"MCPB.py step 1 failed: see {wd / 'mcpb_step1.log'}")
        return ZincParamResults(tables, issues, wd, files)
    _step(prov, "MCPB.py step 1 (models + Gaussian inputs)", "MCPB.py -s 1", [wd / f"{group}.in", mcpb_pdb], coms)

    # ---- 3. QM with PySCF ------------------------------------------------------------------------------
    qm = Q.run_mcpb_qm(wd, group, use_gpu)
    sm = qm["small model"]
    if sm.get("imaginary frequencies < -50 cm-1", 0):
        issues.add("zinc", Status.WARNING, f"small model has {sm['imaginary frequencies < -50 cm-1']} imaginary "
                   "frequencies below -50 cm-1: the optimised geometry may not be a minimum")
    _step(prov, "QM (PySCF bridge)", f"PySCF {qm.get('pyscf')} ({'GPU' if qm.get('gpu') else 'CPU'})", coms,
          [wd / f"{group}_small_opt.fchk", wd / f"{group}_large_mk.log", wd / f"{group}_pyscf_summary.json"], qm)
    tables["QMSummary"] = pd.DataFrame([{"Model": k, "Item": kk, "Value": json.dumps(vv) if isinstance(vv, (dict, list)) else vv}
                                        for k, v in qm.items() if isinstance(v, dict) for kk, vv in v.items()])

    # ---- 4. MCPB.py steps 2-4 ----------------------------------------------------------------------------
    for step, label in (("2", "Seminario force constants"), ("3", "RESP charges"), ("4", "tleap input")):
        r = _run(["MCPB.py", "-i", f"{group}.in", "-s", step], wd, f"mcpb_step{step}.log")
        if r.returncode != 0:
            issues.add("zinc", Status.FAIL, f"MCPB.py step {step} ({label}) failed: see {wd / f'mcpb_step{step}.log'}")
            return ZincParamResults(tables, issues, wd, files)
    frc = wd / f"{group}_mcpbpy.frcmod"
    tleap_in = wd / f"{group}_tleap.in"
    final_pdb = wd / f"{group}_mcpbpy.pdb"
    mol2s = sorted(set(wd.glob("*.mol2")) - {wd / "ZN.mol2", wd / "OHX.mol2"})
    _step(prov, "MCPB.py steps 2-4", "MCPB.py -s 2/3/4", [wd / f"{group}_small_opt.fchk", wd / f"{group}_large_mk.log"],
          [frc, tleap_in, final_pdb] + mol2s)

    # ---- 5. validation -------------------------------------------------------------------------------------
    fr = parse_frcmod(frc.read_text()) if frc.exists() else {}
    for k, v in fr.items():
        tables[f"Frcmod{k.title()}"] = v
    ch = []
    for f in mol2s:
        c = parse_mol2_charges(f.read_text())
        if len(c):
            new = c[c["Type"].str.match(r"^(M\d|Y\d+|Z\d)$")]          # MCPB.py's new metal-site types
            ch.append({"Residue": f.stem, "Atoms": len(c), "Net charge": round(float(c["Charge"].sum()), 4),
                       "Metal-site atoms (type, RESP charge)": ", ".join(
                           f"{a} ({t}) {q:+.3f}" for a, t, q in zip(new["Atom"], new["Type"], new["Charge"]))})
    tables["SiteResidues"] = pd.DataFrame(ch)
    tl = tleap_in.read_text() if tleap_in.exists() else ""
    bonds = re.findall(r"^bond\s+(mol\.\d+\.\S+)\s+(mol\.\d+\.\S+)", tl, re.M)
    fp = read_pdb(final_pdb) if final_pdb.exists() else pd.DataFrame()
    types = {}
    for f in mol2s:
        c = parse_mol2_charges(f.read_text())
        rn = f.stem
        for r_ in fp[fp["resname"] == rn].drop_duplicates("resnum").itertuples() if len(fp) else []:
            for a, t in zip(c["Atom"], c["Type"]):
                types[(int(r_.resnum), a)] = t
    tables["ZnBonds"] = site_bond_table(fp, [b for b in bonds if any(x in b[0] + b[1] for x in (".ZN",))],
                                        None, fr, types) if len(fp) else pd.DataFrame()
    # test build
    check_in = wd / "check_build.in"
    keep = [l for l in tl.splitlines() if not re.match(r"\s*(solvate|addions|addIons|save|quit)", l, re.I)]
    check_in.write_text("\n".join(keep) + "\ncheck mol\ncharge mol\n"
                        f"saveamberparm mol {group}_siteA.prmtop {group}_siteA.inpcrd\nquit\n")
    r = _run(["tleap", "-f", check_in.name], wd, "check_build.log")
    lg = parse_tleap_log((wd / "check_build.log").read_text())
    tables["TleapCheck"] = pd.DataFrame([{"Item": k, "Value": json.dumps(v) if isinstance(v, list) else v} for k, v in lg.items()])
    files.update({"frcmod": frc, "tleap_in": tleap_in, "mcpb_pdb": final_pdb, "mol2": [str(f) for f in mol2s],
                  "prmtop": wd / f"{group}_siteA.prmtop"})
    _step(prov, "validation build (chain A)", "tleap", [tleap_in, frc] + mol2s,
          [wd / f"{group}_siteA.prmtop", wd / f"{group}_siteA.inpcrd"], lg)
    after = {p: sha256_file(Path(p)) for p in originals}
    tables["ZincChecks"] = zinc_checks(tables, lg, originals, after, qm, cfg)
    (wd / "provenance.json").write_text(json.dumps({"originals": originals, "steps": prov}, indent=1, default=str))
    files["provenance"] = wd / "provenance.json"
    return ZincParamResults(tables, issues, wd, files, {"qm": qm, "tleap": lg})


def zinc_checks(t: dict, lg: dict, before: dict, after: dict, qm: dict, cfg: Any,
                critical=("originals_unchanged", "qm_converged", "tleap_build", "site_charge_integer",
                          "zn_bonds_parameterised")) -> pd.DataFrame:
    rows = []
    add = lambda c, st, d: rows.append({"Check": c, "Status": st.value, "Detail": d})  # noqa: E731
    changed = [p for p in before if before[p] != after.get(p)]
    add("originals_unchanged", Status.FAIL if changed else Status.PASS, ", ".join(changed) or f"{len(before)} unchanged")
    sm = qm.get("small model", {})
    conv = bool(sm.get("optimisation converged"))
    add("qm_converged", Status.PASS if conv else Status.FAIL, f"small-model optimisation converged: {conv}")
    nimg = int(sm.get("imaginary frequencies < -50 cm-1", 0) or 0)
    add("qm_minimum", Status.PASS if nimg == 0 else Status.WARNING, f"{nimg} imaginary frequencies below -50 cm-1")
    ok = lg.get("errors") == 0 and not lg.get("missing parameters")
    add("tleap_build", Status.PASS if ok else Status.FAIL,
        f"errors {lg.get('errors')}, missing parameters {len(lg.get('missing parameters') or [])}")
    sr = t.get("SiteResidues", pd.DataFrame())
    tot = float(sr["Net charge"].sum()) if len(sr) else float("nan")
    ok_q = len(sr) and abs(tot - round(tot)) < 1e-3
    # RESP spreads charge over the Zn ions and ligating residues; only the whole metal site is integer
    add("site_charge_integer", Status.PASS if ok_q else Status.FAIL,
        f"{len(sr)} metal-site residues, total {tot:+.4f}" + ("" if ok_q else " (not an integer)"))
    zb = t.get("ZnBonds", pd.DataFrame())
    nb = int(zb["MCPB k (kcal/mol/Å²)"].notna().sum()) if len(zb) else 0
    add("zn_bonds_parameterised", Status.PASS if len(zb) and nb == len(zb) else Status.FAIL, f"{nb} of {len(zb)} Zn bonds have r0/k")
    if len(zb):
        dev = zb["r0 − model (Å)"].abs().max()
        add("zn_r0_vs_structure", Status.PASS if dev <= 0.15 else Status.WARNING,
            f"max |r0 − model distance| {dev:.3f} Å (QM-optimised vs the prepared model)")
    tc = lg.get("total charge")
    add("chain_total_charge_integer", Status.PASS if tc is not None and abs(tc - round(tc)) < 1e-3 else Status.FAIL,
        f"total charge of the parameterised chain {tc}")
    df = pd.DataFrame(rows)
    df.insert(1, "Critical", df["Check"].isin(set(critical)))
    return df


def save_zinc_parameters(cfg: Any, res: ZincParamResults, tag: str = "") -> dict[str, Path]:
    """Workbook (``outputs.md_workbook``), tables and manifest for the Zn parameterisation."""
    from .excel_output import write_tables, write_workbook
    from .manifest import build_manifest, detect_software, manifest_table, write_manifest

    software = detect_software()
    sheets = dict(res.tables)
    sheets["ZincIssues"] = res.issues.to_frame()
    sheets["Software"] = software
    manifest = build_manifest(cfg, structure_ids=[str(cfg.get("preparation.template"))],
                              input_files=[], software=software,
                              extra={"stage": "Zn site parameters, MCPB.py + PySCF (notebook 05 part A)",
                                     "qm": res.summary.get("qm"), "workdir": str(res.workdir),
                                     "issue_counts": {s.value: res.issues.count(s) for s in Status}})
    sheets["Reproducibility"] = manifest_table(manifest)
    suffix = f"_{tag}" if tag else ""
    wb = cfg.output_file("md_workbook")
    wb = wb.with_name(wb.stem + suffix + ".xlsx")
    write_workbook(wb, sheets)
    mpath = write_manifest(manifest, cfg.outputs_dir / f"{cfg.accession}_md_manifest{suffix}.json")
    tdir = cfg.outputs_dir / f"md_tables{suffix}"
    write_tables(tdir, sheets, csv=bool(cfg.get("outputs.write_csv", True)), parquet=bool(cfg.get("outputs.write_parquet", True)))
    return {"workbook": wb, "manifest": mpath, "tables_dir": tdir, "workdir": res.workdir}
