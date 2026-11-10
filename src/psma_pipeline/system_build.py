"""Membrane system build, minimisation and MD readiness (Milestone 6, part B).

Design: ``docs/M6_SYSTEM_BUILD.md``. Inputs: the Milestone 5 default microstate (membrane-oriented
hybrid dimer) and the part A MCPB.py outputs (residue templates, frcmod, bond list of the chain A site).

1. **Solute** (heavy atoms; tleap builds hydrogens from the templates): both chains with the MCPB.py
   residue names applied to *both* metal sites, Zn residues matched to the chain A templates by their
   ligand sets, the bridging hydroxides (with H), core GlcNAc converted to GLYCAM (``NAG`` → ``0YB``,
   ``ASN`` → ``NLN``), Ca²⁺ / Cl⁻ and the Zn-shell waters.
2. **Membrane** with packmol-memgen from the protein alone (pre-oriented, no protonation), so its
   ligand handling cannot change the solute. The solute is then superposed on the packed protein and
   every lipid / water / ion *molecule* that clashes with it is removed whole.
3. **tleap**: ff14SB + Lipid21 + GLYCAM06j + TIP3P + the MCPB.py parameters; Zn–ligand bonds and the
   renamed-residue peptide links for both sites, disulfides and N-glycan links; neutralised.
4. **OpenMM**: restrained then unrestrained minimisation; optional short restrained NVT/NPT smoke run
   (membrane barostat) on the GPU; Zn–ligand geometry, forces, energies and NaNs are checked.
5. **Readiness**: the first table in the pipeline that can report READY (no FAIL on any item).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cache import sha256_file
from .status import IssueLog, Status
from .zinc_params import _run, read_pdb

log = logging.getLogger("psma_pipeline.system_build")

PROTEIN = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO",
           "SER", "THR", "TRP", "TYR", "VAL", "HID", "HIE", "HIP", "ASH", "GLH", "LYN", "CYX", "CYM", "NLN"}
LIPID_TAILS = {"PA", "OL", "MY", "ST", "LA", "AR", "DHA", "SA"}
LIPID_HEADS = {"PC", "PE", "PS", "PGR", "PGS", "PH-", "PI", "SPM"}
STEROLS = {"CHL"}
IONS = {"Na+", "K+", "Cl-", "NA", "CL", "K"}
NAG_TO_0YB = {"C1": "C1", "C2": "C2", "C3": "C3", "C4": "C4", "C5": "C5", "C6": "C6", "O3": "O3", "O4": "O4",
              "O5": "O5", "O6": "O6", "N2": "N2", "C7": "C2N", "O7": "O2N", "C8": "CME"}


# ---------------------------------------------------------------------------
# PDB with TER-delimited molecules and CRYST1
# ---------------------------------------------------------------------------
def read_pdb_molecules(path: str | Path) -> tuple[pd.DataFrame, list[float] | None]:
    """Atoms with a molecule id. A new molecule starts at every TER *and* wherever the residue sequence implies one
    (each water / ion / sterol; a Lipid21 lipid = tail + head + tail), so a file with missing or sparse TER records
    is still split correctly."""
    rows, ter, cryst = [], 0, None
    for line in Path(path).read_text().splitlines():
        if line.startswith("CRYST1"):
            cryst = [float(line[6:15]), float(line[15:24]), float(line[24:33])]
        elif line.startswith("TER"):
            ter += 1
        elif line.startswith(("ATOM", "HETATM")):
            rows.append({"record": line[:6].strip(), "atom": line[12:16].strip(), "resname": line[17:21].strip(),
                         "chain": line[21].strip(), "resnum": int(line[22:26]), "icode": line[26].strip(),
                         "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54]),
                         "element": (line[76:78].strip() or re.sub(r"\d", "", line[12:16].strip())[:1]).capitalize(),
                         "ter": ter})
    df = pd.DataFrame(rows)
    if len(df):
        df["mol"] = molecule_ids(df)
    return df, cryst


def molecule_ids(df: pd.DataFrame) -> np.ndarray:
    key = (df["ter"].astype(str) + "|" + df["chain"] + "|" + df["resnum"].astype(str) + "|" + df["resname"]).to_numpy()
    start = np.r_[True, key[1:] != key[:-1]]
    res_first = np.flatnonzero(start)
    res_id = np.cumsum(start) - 1
    mol_of_res, mol, prev_kind, prev_ter, heads, tails = [], -1, None, None, 0, 0
    for i in res_first:
        rn, ch, t = df["resname"].iat[i], df["chain"].iat[i], df["ter"].iat[i]
        new = t != prev_ter
        if rn in PROTEIN:
            kind = ("prot", ch)
            new = new or prev_kind != kind
        elif rn in LIPID_TAILS | LIPID_HEADS:
            kind = "lipid"
            if prev_kind != "lipid" or new or (rn in LIPID_TAILS and heads and tails >= 2) or (rn in LIPID_HEADS and heads):
                new = True
            if new:
                heads = tails = 0
            heads += rn in LIPID_HEADS
            tails += rn in LIPID_TAILS
        else:
            kind, new = rn, True
        if new:
            mol += 1
        mol_of_res.append(mol)
        prev_kind, prev_ter = kind, t
    return np.asarray(mol_of_res)[res_id]


def pdb_atom_line(serial: int, record: str, atom: str, resname: str, chain: str, resnum: int, xyz, element: str) -> str:
    name = atom if len(atom) == 4 else f" {atom:<3s}"
    return (f"{record:<6s}{serial % 100000:5d} {name:4s} {resname:<4s}{chain[:1] or ' '}{resnum % 10000:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {element.upper():>2s}")


# ---------------------------------------------------------------------------
# MCPB.py site map (part A outputs) -> applicable to both chains
# ---------------------------------------------------------------------------
@dataclass
class SiteMap:
    resnames: dict[int, str]                 # protein residue number -> MCPB.py name
    zn_names: list[str]                      # MCPB.py names of the site's Zn, in site order
    zn_ligands: list[frozenset]              # protein residue numbers bonded to each Zn
    oh_name: str | None
    bonds: list[tuple]                       # ((kind, id), atom, (kind, id), atom); kind prot | zn | oh
    header: list[str]                        # addAtomTypes / loadmol2 / loadamberparams lines
    n_prot: int = 0                          # protein residues in the site PDB (tleap numbers them 1..n_prot)
    files: list[Path] = field(default_factory=list)


def site_map(zinc_dir: str | Path, group: str = "PSMA") -> SiteMap:
    zd = Path(zinc_dir)
    site = read_pdb(zd / f"{group}_site.pdb")
    final = read_pdb(zd / f"{group}_mcpbpy.pdb")
    s_res = site.drop_duplicates("resnum").set_index("resnum")["resname"]
    f_res = final.drop_duplicates("resnum").set_index("resnum")["resname"]
    n_prot = int(site.loc[site["record"] == "ATOM", "resnum"].max())
    resnames = {int(n): f_res[n] for n in s_res.index if n <= n_prot and n in f_res and f_res[n] != s_res[n]}
    zn_nums = [int(n) for n in s_res.index if s_res[n] == "ZN"]
    oh_nums = [int(n) for n in s_res.index if s_res[n] in ("OHX", "WTB")]
    tl = (zd / f"{group}_tleap.in").read_text()
    header, in_types = [], False
    for line in tl.splitlines():
        st = line.strip()
        if st.lower().startswith("addatomtypes"):
            in_types = True
        if in_types:
            header.append(line)
            if st == "}":
                in_types = False
            continue
        if re.match(r"\S+\s*=\s*loadmol2\s+\S+", st) or st.lower().startswith(("loadamberparams", "loadoff")):
            header.append(st)
    def ref(n: int):
        if n in zn_nums:
            return ("zn", zn_nums.index(n))
        if n in oh_nums:
            return ("oh", 0)
        return ("prot", n)
    bonds = []
    for a, b in re.findall(r"^\s*bond\s+mol\.(\d+\.\S+)\s+mol\.(\d+\.\S+)", tl, re.M):
        (ra, aa), (rb, ab) = a.split(".", 1), b.split(".", 1)
        bonds.append((ref(int(ra)), aa, ref(int(rb)), ab))
    ligs = []
    for k in range(len(zn_nums)):
        ligs.append(frozenset(r[1] for b in bonds for r, o in ((b[0], b[2]), (b[2], b[0]))
                              if o == ("zn", k) and r[0] == "prot"))
    files = [zd / m for m in re.findall(r"loadmol2\s+(\S+)", tl)] + \
            [zd / m for m in re.findall(r"loadamberparams\s+(\S+)", tl)]
    return SiteMap(resnames, [f_res[n] for n in zn_nums], ligs, f_res[oh_nums[0]] if oh_nums else None, bonds,
                   header, n_prot, [f for f in files if f.exists()])


def site_map_to_uniprot(smap: SiteMap, ordinals: list[int]) -> SiteMap:
    """tleap renumbered the site chain 1..n; translate to the chain's own residue numbers (UniProt), in file order."""
    if smap.n_prot and smap.n_prot != len(ordinals):
        raise ValueError(f"site PDB has {smap.n_prot} protein residues, the prepared chain {len(ordinals)}: "
                         "re-run part A on this prepared model")
    u = lambda n: int(ordinals[n - 1])  # noqa: E731
    cv = lambda r: ("prot", u(r[1])) if r[0] == "prot" else r  # noqa: E731
    return SiteMap({u(n): v for n, v in smap.resnames.items()}, smap.zn_names,
                   [frozenset(u(n) for n in s) for s in smap.zn_ligands], smap.oh_name,
                   [(cv(a), x, cv(b), y) for a, x, b, y in smap.bonds], smap.header, smap.n_prot, smap.files)


# ---------------------------------------------------------------------------
# Solute
# ---------------------------------------------------------------------------
@dataclass
class Solute:
    pdb_lines: list[str]
    residues: pd.DataFrame          # one row per residue in file order: seq, kind, chain, id, resname
    bonds: list[tuple[int, str, int, str]]
    protein_heavy: pd.DataFrame     # protein heavy atoms (for packmol-memgen and superposition)
    zn_bond_atoms: list[tuple]      # (seq1, atom1, seq2, atom2) for Zn bonds only
    glycans: pd.DataFrame
    info: dict = field(default_factory=dict)


def _zn_ligand_set(zn_xyz, prot: pd.DataFrame, cutoff: float = 2.6) -> frozenset:
    d = np.linalg.norm(prot[["x", "y", "z"]].to_numpy() - zn_xyz, axis=1)
    return frozenset(int(n) for n in prot.loc[d <= cutoff, "resnum"])


def build_solute(prepared: str | Path, smap: SiteMap, bridge: str = "hydroxide", site_chain: str = "A") -> Solute:
    """Heavy-atom solute PDB with MCPB.py names on both sites, GLYCAM core GlcNAc, ions and crystal waters."""
    from .zinc_params import extract_site

    pdb = read_pdb(prepared)
    prot = pdb[pdb["record"] == "ATOM"].copy()
    prot = prot[prot["element"] != "H"]
    chains = list(dict.fromkeys(prot["chain"]))
    smap = site_map_to_uniprot(smap, list(dict.fromkeys(prot.loc[prot["chain"] == site_chain, "resnum"])))
    het = pdb[pdb["record"] == "HETATM"].copy()
    # glycans: core NAG on Asn ND2 -> 0YB + NLN
    glyc_rows, nln = [], set()
    nag = het[het["resname"].isin(["NAG", "NDG"])]
    for (ch, num), g in nag.groupby(["chain", "resnum"], sort=False):
        c1 = g[g["atom"] == "C1"][["x", "y", "z"]].to_numpy()
        nd2 = prot[(prot["atom"] == "ND2") & (prot["resname"] == "ASN")]
        if not len(c1) or not len(nd2):
            continue
        d = np.linalg.norm(nd2[["x", "y", "z"]].to_numpy() - c1[0], axis=1)
        k = int(np.argmin(d))
        if d[k] > 2.0:
            continue
        asn = (nd2["chain"].iloc[k], int(nd2["resnum"].iloc[k]))
        nln.add(asn)
        glyc_rows.append({"chain": ch, "resnum": num, "asn_chain": asn[0], "asn": asn[1], "dist": round(float(d[k]), 3),
                          "atoms": g[g["atom"].isin(NAG_TO_0YB)].assign(atom=lambda x: x["atom"].map(NAG_TO_0YB))})
    lines, res_rows, serial, seq = [], [], 1, 0
    xyz_of = {}

    def emit(rec, rows, resname, chain, kind, rid):
        nonlocal serial, seq
        seq += 1
        for r in rows.itertuples():
            lines.append(pdb_atom_line(serial, rec, r.atom, resname, chain, seq, (r.x, r.y, r.z), r.element))
            xyz_of[(seq, r.atom)] = np.array([r.x, r.y, r.z])
            serial += 1
        res_rows.append({"seq": seq, "kind": kind, "chain": chain, "id": rid, "resname": resname, "atoms": len(rows)})
        return seq

    seq_prot = {}
    for ch in chains:
        for num, g in prot[prot["chain"] == ch].groupby("resnum", sort=False):
            rn = g["resname"].iloc[0]
            if (ch, num) in nln:
                rn = "NLN"
            elif num in smap.resnames:
                rn = smap.resnames[num]
            seq_prot[(ch, int(num))] = emit("ATOM", g, rn, ch, "protein", int(num))
        lines.append("TER")
    # metal sites
    zn_seq, oh_seq, zn_info = {}, {}, []
    for ch in chains:
        zn = het[(het["resname"] == "ZN") & (het["chain"] == ch)]
        pc = prot[prot["chain"] == ch]
        for r in zn.itertuples():
            ls = _zn_ligand_set(np.array([r.x, r.y, r.z]), pc)
            k = max(range(len(smap.zn_ligands)), key=lambda j: len(smap.zn_ligands[j] & ls)) if smap.zn_ligands else 0
            if smap.zn_ligands and not (smap.zn_ligands[k] & ls):
                raise ValueError(f"chain {ch}: Zn at {r.x:.1f},{r.y:.1f},{r.z:.1f} matches no parameterised site Zn")
            if (ch, k) in zn_seq:
                raise ValueError(f"chain {ch}: two Zn map to template {smap.zn_names[k]}")
            zn_seq[(ch, k)] = emit("HETATM", pd.DataFrame([{"atom": "ZN", "x": r.x, "y": r.y, "z": r.z, "element": "Zn"}]),
                                   smap.zn_names[k], ch, "zinc", k)
            zn_info.append({"Chain": ch, "Template": smap.zn_names[k], "Ligand residues": ", ".join(map(str, sorted(ls)))})
            lines.append("TER")
        if smap.oh_name:
            site = extract_site(pdb, ch, bridge)
            rows = [{"atom": "O", "x": site.bridge_o[0], "y": site.bridge_o[1], "z": site.bridge_o[2], "element": "O"}]
            if site.hydroxide_h is not None:
                h = site.hydroxide_h
                rows.append({"atom": "H", "x": h[0], "y": h[1], "z": h[2], "element": "H"})
            oh_seq[ch] = emit("HETATM", pd.DataFrame(rows), smap.oh_name, ch, "bridge", 0)
            lines.append("TER")
            het = het[~((het["resname"] == "HOH") & (het["chain"] == ch) &
                        (np.linalg.norm(het[["x", "y", "z"]].to_numpy() - site.bridge_o, axis=1) < 0.01))]
    # glycans
    glyc = []
    for gr in glyc_rows:
        s = emit("HETATM", gr["atoms"], "0YB", gr["chain"], "glycan", gr["resnum"])
        glyc.append({"Chain": gr["asn_chain"], "Asn (UniProt)": gr["asn"], "GLYCAM residue seq": s,
                     "C1–ND2 (Å)": gr["dist"], "Asn seq": seq_prot[(gr["asn_chain"], gr["asn"])]})
        lines.append("TER")
    # ions and crystal waters (bridging waters already converted)
    n_ion = n_wat = 0
    for r in het[het["resname"].isin(["CA", "CL", "HOH"])].itertuples():
        rn, an, el, kind = {"CA": ("CA", "CA", "Ca", "ion"), "CL": ("Cl-", "Cl-", "Cl", "ion"),
                            "HOH": ("WAT", "O", "O", "water")}[r.resname]
        if r.resname == "HOH" and r.atom != "O":
            continue
        emit("HETATM", pd.DataFrame([{"atom": an, "x": r.x, "y": r.y, "z": r.z, "element": el}]), rn, r.chain, kind, int(r.resnum))
        lines.append("TER")
        n_ion += kind == "ion"
        n_wat += kind == "water"
    # bonds: both sites, disulfides, glycan links
    bonds, zn_bonds = [], []
    for ch in chains:
        def seq_of(ref):
            if ref[0] == "prot":
                return seq_prot[(ch, ref[1])]
            if ref[0] == "zn":
                return zn_seq[(ch, ref[1])]
            return oh_seq[ch]
        for r1, a1, r2, a2 in smap.bonds:
            b = (seq_of(r1), a1, seq_of(r2), a2)
            bonds.append(b)
            if "zn" in (r1[0], r2[0]):
                zn_bonds.append(b)
    sg = prot[(prot["resname"] == "CYX") & (prot["atom"] == "SG")]
    S = sg[["x", "y", "z"]].to_numpy()
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if np.linalg.norm(S[i] - S[j]) < 2.5:
                bonds.append((seq_prot[(sg["chain"].iat[i], int(sg["resnum"].iat[i]))], "SG",
                              seq_prot[(sg["chain"].iat[j], int(sg["resnum"].iat[j]))], "SG"))
    for g in glyc:
        bonds.append((g["Asn seq"], "ND2", g["GLYCAM residue seq"], "C1"))
    return Solute(lines, pd.DataFrame(res_rows), bonds, prot, zn_bonds, pd.DataFrame(glyc),
                  {"chains": chains, "residues": seq, "Zn": len(zn_seq), "hydroxides": len(oh_seq),
                   "core GlcNAc (GLYCAM 0YB)": len(glyc), "ions (Ca, Cl)": n_ion, "crystal waters": n_wat,
                   "disulfides": sum(1 for b in bonds if b[1] == "SG"), "zn_map": zn_info,
                   "xyz": xyz_of})


def protein_only_pdb(sol: Solute, path: Path) -> Path:
    """Protein heavy atoms (original residue names), chains separated by TER, for packmol-memgen."""
    lines, n, last = [], 1, None
    for r in sol.protein_heavy.itertuples():
        if last is not None and r.chain != last:
            lines.append("TER")
        lines.append(pdb_atom_line(n, "ATOM", r.atom, r.resname, r.chain, int(r.resnum), (r.x, r.y, r.z), r.element))
        n += 1
        last = r.chain
    path.write_text("\n".join(lines + ["TER", "END"]) + "\n")
    return path


# ---------------------------------------------------------------------------
# Merge solute with the packed membrane
# ---------------------------------------------------------------------------
def merge_with_membrane(sol: Solute, packed: pd.DataFrame, cryst: list[float] | None, clash: float = 1.6
                        ) -> tuple[list[str], pd.DataFrame, dict]:
    """Superpose the solute on the packed protein, drop whole clashing molecules, return PDB lines and composition."""
    from scipy.spatial import cKDTree

    from .conformations import kabsch

    pp = packed[packed["resname"].isin(PROTEIN) & (packed["atom"] == "CA")]
    sp = sol.protein_heavy[sol.protein_heavy["atom"] == "CA"]
    if len(pp) != len(sp):
        raise ValueError(f"packed protein has {len(pp)} CA, solute {len(sp)}: cannot superpose")
    P, Qx = sp[["x", "y", "z"]].to_numpy(), pp[["x", "y", "z"]].to_numpy()
    R, pm, qm = kabsch(P, Qx)
    rmsd_fit = float(np.sqrt(np.mean(np.sum(((P - pm) @ R.T + qm - Qx) ** 2, axis=1))))
    move = lambda X: (X - pm) @ R.T + qm  # noqa: E731
    out = []
    solute_xyz = []
    for line in sol.pdb_lines:
        if line.startswith(("ATOM", "HETATM")):
            x = move(np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]))
            out.append(f"{line[:30]}{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}{line[54:]}")
            solute_xyz.append(x)
        else:
            out.append(line)
    tree = cKDTree(np.array(solute_xyz))
    env = packed[~packed["resname"].isin(PROTEIN)]
    heavy = env[env["element"] != "H"]
    d, _ = tree.query(heavy[["x", "y", "z"]].to_numpy(), k=1) if len(heavy) else (np.array([]), None)
    bad_mols = set(heavy.loc[d < clash, "mol"]) if len(heavy) else set()
    comp = []
    first_seq = int(sol.residues["seq"].max())
    seq = first_seq
    serial = sum(1 for l in out if l.startswith(("ATOM", "HETATM"))) + 1
    for mol, g in env.groupby("mol", sort=False):
        names = list(dict.fromkeys(g["resname"]))
        kind = ("lipid" if set(names) & (LIPID_TAILS | LIPID_HEADS | STEROLS) else
                "water" if names == ["WAT"] else "ion" if set(names) <= IONS else "other")
        comp.append({"mol": mol, "kind": kind, "residues": "+".join(names), "removed": mol in bad_mols})
        if mol in bad_mols:
            continue
        for (rn, num, ch), r in g.groupby(["resname", "resnum", "chain"], sort=False):
            seq += 1
            for a in r.itertuples():
                out.append(pdb_atom_line(serial, "HETATM" if kind != "lipid" else "ATOM", a.atom, rn, "", seq,
                                         (a.x, a.y, a.z), a.element))
                serial += 1
        out.append("TER")
    if cryst:
        out.insert(0, f"CRYST1{cryst[0]:9.3f}{cryst[1]:9.3f}{cryst[2]:9.3f}  90.00  90.00  90.00 P 1           1")
    out.append("END")
    comp = pd.DataFrame(comp)
    return out, comp, {"superposition RMSD (Å)": round(rmsd_fit, 3), "molecules removed": len(bad_mols)}


def composition_table(comp: pd.DataFrame, box: list[float] | None, lipids_req: str, ratio_req: str) -> pd.DataFrame:
    kept = comp[~comp["removed"]] if len(comp) else comp
    rows = []
    for kind in ("lipid", "water", "ion", "other"):
        g = kept[kept["kind"] == kind] if len(kept) else kept
        rem = int((comp["removed"] & (comp["kind"] == kind)).sum()) if len(comp) else 0
        rows.append({"Component": kind, "Molecules": len(g), "Removed (clash)": rem,
                     "Types": ", ".join(f"{k} ×{v}" for k, v in g["residues"].value_counts().items()) if len(g) else ""})
    t = pd.DataFrame(rows)
    lip = kept[kept["kind"] == "lipid"] if len(kept) else kept
    chol = int(lip["residues"].str.contains("CHL").sum()) if len(lip) else 0
    want = [float(x) for x in str(ratio_req).split(":")]
    t.attrs.update({"cholesterol fraction": round(chol / len(lip), 3) if len(lip) else None,
                    "requested cholesterol fraction": round(want[-1] / sum(want), 3) if len(want) > 1 and "CHL" in lipids_req else 0.0,
                    "box (Å)": box})
    return t


# ---------------------------------------------------------------------------
# tleap
# ---------------------------------------------------------------------------
def tleap_script(cfg: Any, smap: SiteMap, bonds: list[tuple], box: list[float] | None, pdb_name: str = "system.pdb",
                 out: str = "system") -> str:
    s = cfg.get("md.system", {}) or {}
    lines = [f"source leaprc.protein.{cfg.get('md.protein_ff', 'ff14SB')}", f"source leaprc.{s.get('lipid_ff', 'lipid21')}",
             f"source leaprc.{s.get('glycan_ff', 'GLYCAM_06j-1')}", f"source leaprc.water.{cfg.get('md.water_model', 'tip3p')}"]
    lines += smap.header
    lines.append(f"sys = loadpdb {pdb_name}")
    for r1, a1, r2, a2 in bonds:
        lines.append(f"bond sys.{r1}.{a1} sys.{r2}.{a2}")
    if box:
        lines.append(f"set sys box {{ {box[0]:.3f} {box[1]:.3f} {box[2]:.3f} }}")
    lines += ["charge sys", "addIonsRand sys Na+ 0", "addIonsRand sys Cl- 0", "charge sys", "check sys",
              f"saveamberparm sys {out}.prmtop {out}.inpcrd", f"savepdb sys {out}_tleap.pdb", "quit"]
    return "\n".join(lines) + "\n"


def parse_build_log(text: str) -> dict:
    from .zinc_params import parse_tleap_log
    d = parse_tleap_log(text)
    charges = re.findall(r"Total (?:unperturbed )?charge:\s*(-?[\d.]+)", text)
    d["charge before neutralising"] = float(charges[0]) if charges else None
    d["charge after neutralising"] = float(charges[-1]) if charges else None
    d["ions added"] = sum(int(n) for n in re.findall(r"Adding (\d+) counter ions", text))
    return d


# ---------------------------------------------------------------------------
# OpenMM minimisation and smoke run
# ---------------------------------------------------------------------------
def atom_index(tleap_pdb: str | Path) -> dict[tuple[int, str], int]:
    """(residue seq, atom name) -> 0-based atom index, from tleap's savepdb (same order as the prmtop)."""
    idx, k, seq, last = {}, 0, 0, None
    for line in Path(tleap_pdb).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        key = (line[17:21], line[21:27])
        if key != last:
            seq += 1
            last = key
        idx[(seq, line[12:16].strip())] = k
        k += 1
    return idx


def distances(xyz_nm: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.array([10.0 * np.linalg.norm(xyz_nm[i] - xyz_nm[j]) for i, j in pairs])


def pick_platform():
    import openmm as mm
    for name, props in (("CUDA", {"Precision": "mixed"}), ("OpenCL", {"Precision": "mixed"}), ("CPU", {})):
        try:
            return mm.Platform.getPlatformByName(name), props
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("no OpenMM platform available")


def run_openmm(cfg: Any, prmtop: Path, inpcrd: Path, zn_pairs: list[tuple[int, int]], out: Path,
               protein_heavy_idx: list[int]) -> tuple[pd.DataFrame, dict]:
    """Restrained + unrestrained minimisation and an optional restrained NVT/NPT smoke run."""
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u

    oc = cfg.get("md.openmm", {}) or {}
    T = float(oc.get("temperature_k", 310))
    prm, crd = app.AmberPrmtopFile(str(prmtop)), app.AmberInpcrdFile(str(inpcrd))
    system = prm.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=float(oc.get("cutoff_nm", 1.0)) * u.nanometer,
                              constraints=app.HBonds, rigidWater=True)
    k = float(oc.get("restraint_kcal", 10.0)) * 418.4          # kcal/mol/Å² -> kJ/mol/nm²
    rest = mm.CustomExternalForce("0.5*k_rest*periodicdistance(x, y, z, x0, y0, z0)^2")
    rest.addGlobalParameter("k_rest", k)
    for p in ("x0", "y0", "z0"):
        rest.addPerParticleParameter(p)
    pos = crd.getPositions(asNumpy=True).value_in_unit(u.nanometer)
    for i in protein_heavy_idx:
        rest.addParticle(i, pos[i].tolist())
    system.addForce(rest)
    platform, props = pick_platform()
    integ = mm.LangevinMiddleIntegrator(T * u.kelvin, 1 / u.picosecond, 0.002 * u.picoseconds)
    sim = app.Simulation(prm.topology, system, integ, platform, props)
    sim.context.setPositions(crd.positions)
    if crd.boxVectors is not None:
        sim.context.setPeriodicBoxVectors(*crd.boxVectors)
    rows = []

    def snap(stage):
        st = sim.context.getState(getEnergy=True, getPositions=True, getForces=True)
        e = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        f = np.linalg.norm(st.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer), axis=1)
        x = st.getPositions(asNumpy=True).value_in_unit(u.nanometer)
        d = distances(x, zn_pairs) if zn_pairs else np.array([])
        rmsd = float(np.sqrt(np.mean(np.sum((x[protein_heavy_idx] - pos[protein_heavy_idx]) ** 2, axis=1)))) * 10
        rows.append({"Stage": stage, "Potential energy (kJ/mol)": round(e, 1), "Finite": bool(np.isfinite(e)),
                     "Max force (kJ/mol/nm)": round(float(np.nanmax(f)), 1),
                     "Protein heavy-atom RMSD from start (Å)": round(rmsd, 3),
                     "Zn–ligand distances (Å)": ", ".join(f"{v:.2f}" for v in d), "_d": d,
                     "Temperature (K)": None, "Box (Å)": ", ".join(f"{10 * v:.1f}" for v in
                                                                  np.diag(st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer)))})
        return x

    snap("start")
    tol = float(oc.get("min_tolerance", 10.0)) * u.kilojoule_per_mole / u.nanometer
    mm.LocalEnergyMinimizer.minimize(sim.context, tol, 0)
    snap("minimised (restrained)")
    sim.context.setParameter("k_rest", 0.0)
    mm.LocalEnergyMinimizer.minimize(sim.context, tol, 0)
    x_min = snap("minimised (unrestrained)")
    with open(out / "system_minimised.pdb", "w") as fh:
        app.PDBFile.writeFile(prm.topology, x_min * u.nanometer, fh, keepIds=True)
    smoke = list(oc.get("smoke_ps") or [])
    if smoke and platform.getName() in ("CUDA", "OpenCL"):
        sim.context.setParameter("k_rest", k)
        nvt_steps = int(smoke[0] / 0.002)
        sim.context.setVelocitiesToTemperature(10 * u.kelvin)
        for j in range(10):                                     # heat in 10 increments to T
            integ.setTemperature((T * (j + 1) / 10) * u.kelvin)
            sim.step(max(1, nvt_steps // 10))
        snap("NVT heating (restrained)")
        rows[-1]["Temperature (K)"] = round(_temperature(sim, system), 1)
        if len(smoke) > 1:
            baro = mm.MonteCarloMembraneBarostat(1 * u.bar, 0 * u.bar * u.nanometer, T * u.kelvin,
                                                 mm.MonteCarloMembraneBarostat.XYIsotropic,
                                                 mm.MonteCarloMembraneBarostat.ZFree)
            system.addForce(baro)
            sim.context.reinitialize(preserveState=True)
            sim.step(int(smoke[1] / 0.002))
            snap("NPT membrane barostat (restrained)")
            rows[-1]["Temperature (K)"] = round(_temperature(sim, system), 1)
        st = sim.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
        with open(out / "system_smoke_state.xml", "w") as fh:
            fh.write(mm.XmlSerializer.serialize(st))
    elif smoke:
        rows.append({"Stage": "smoke run", "Finite": None, "Temperature (K)": None,
                     "Zn–ligand distances (Å)": f"NOT RUN: platform {platform.getName()} (needs a GPU)"})
    with open(out / "system.xml", "w") as fh:
        fh.write(mm.XmlSerializer.serialize(system))
    df = pd.DataFrame(rows)
    d0 = rows[0]["_d"]
    df["Max Zn–ligand change (Å)"] = [round(float(np.max(np.abs(r["_d"] - d0))), 3) if isinstance(r.get("_d"), np.ndarray)
                                       and len(r["_d"]) == len(d0) and len(d0) else None for r in rows]
    return df.drop(columns=["_d"]), {"platform": platform.getName(), "atoms": system.getNumParticles()}


def _temperature(sim, system) -> float:
    import openmm.unit as u
    st = sim.context.getState(getEnergy=True)
    dof = 3 * system.getNumParticles() - system.getNumConstraints() - 3
    ke = st.getKineticEnergy().value_in_unit(u.kilojoule_per_mole)
    return 2 * ke / (dof * 0.0083144626)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclass
class BuildResults:
    tables: dict[str, pd.DataFrame]
    issues: IssueLog
    workdir: Path
    files: dict = field(default_factory=dict)


def _step(prov, name, tool, inputs, outputs, params=None):
    prov.append({"step": len(prov) + 1, "name": name, "tool": tool,
                 "inputs": [{"path": str(p), "sha256": sha256_file(Path(p)) if Path(p).exists() else None} for p in inputs],
                 "outputs": [{"path": str(p), "sha256": sha256_file(Path(p)) if Path(p).exists() else None} for p in outputs],
                 "parameters": params or {}})


def run_system_build(cfg: Any, prepared_pdb: str | Path | None = None, zinc_dir: str | Path | None = None,
                     issues: IssueLog | None = None, run_md: bool = True) -> BuildResults:
    issues = issues if issues is not None else IssueLog()
    s = cfg.get("md.system", {}) or {}
    group = cfg.get("md.mcpb.group_name", "PSMA")
    prep_dir = cfg.outputs_dir / "model_preparation" / cfg.get("preparation.name", "hybrid_9S7U_AF2")
    prepared_pdb = Path(prepared_pdb) if prepared_pdb else prep_dir / "microstates" / "microstate_00_default.pdb"
    zinc_dir = Path(zinc_dir) if zinc_dir else cfg.outputs_dir / "md_setup" / "zinc_mcpb"
    wd = cfg.outputs_dir / "md_setup" / "system"
    wd.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}
    prov: list = []
    originals = {str(p): sha256_file(p) for p in [prepared_pdb] + sorted(zinc_dir.glob(f"{group}_*")) if p.is_file()}
    for t in ("packmol-memgen", "tleap"):
        if not shutil.which(t):
            issues.add("system", Status.FAIL, f"{t} not found: install AmberTools (notebook 05, section 0)")
            return BuildResults(tables, issues, wd)

    # ---- 1. solute -------------------------------------------------------------------------------------
    smap = site_map(zinc_dir, group)
    for f in smap.files:
        shutil.copy2(f, wd / f.name)
    sol = build_solute(prepared_pdb, smap, cfg.get("preparation.zn_bridge", "hydroxide"), cfg.get("md.mcpb.site_chain", "A"))
    tables["SoluteSummary"] = pd.DataFrame([{"Item": k, "Value": v} for k, v in sol.info.items() if k not in ("zn_map", "xyz")])
    tables["ZnTemplateMap"] = pd.DataFrame(sol.info["zn_map"])
    tables["GlycanLinks"] = sol.glycans
    prot_pdb = protein_only_pdb(sol, wd / "protein_for_packing.pdb")
    _step(prov, "solute (both metal sites, GLYCAM core GlcNAc, ions)", "psma_pipeline.system_build.build_solute",
          [prepared_pdb] + smap.files, [prot_pdb], {k: v for k, v in sol.info.items() if k not in ("zn_map", "xyz")})

    # ---- 2. membrane with packmol-memgen -------------------------------------------------------------------
    # Packing runs on the local disk (fast; Drive is slow for a 40 MB file rewritten every loop) and is copied
    # to the work folder only when packmol-memgen has finished. A marker records the settings, so a
    # half-written file from an interrupted run, or a packing made with other settings, is never reused.
    packed, marker = wd / "packed.pdb", wd / "packed.done.json"
    pack_params = {k: s.get(k, d) for k, d in (("lipids", "POPC:CHL1"), ("ratio", "7:3"), ("dist", 15.0),
                                               ("dist_wat", 17.5), ("salt_cation", "Na+"), ("salt_molar", 0.15),
                                               ("nloop", 20), ("nloop_all", 30), ("movebadrandom", True))}
    pack_params["protein_sha256"] = sha256_file(prot_pdb)
    reuse = packed.exists() and marker.exists() and json.loads(marker.read_text()) == pack_params
    if not reuse:
        import tempfile
        for f in (packed, marker):
            if f.exists():
                f.unlink()
        tmp = Path(tempfile.mkdtemp(prefix="psma_pack_"))
        shutil.copy2(prot_pdb, tmp / prot_pdb.name)
        cmd = ["packmol-memgen", "--pdb", prot_pdb.name, "--preoriented", "--notprotonate",
               "--lipids", str(pack_params["lipids"]), "--ratio", str(pack_params["ratio"]),
               "--dist", str(pack_params["dist"]), "--dist_wat", str(pack_params["dist_wat"]),
               "--salt", "--salt_c", str(pack_params["salt_cation"]), "--saltcon", str(pack_params["salt_molar"]),
               "--nloop", str(pack_params["nloop"]), "--nloop_all", str(pack_params["nloop_all"]),
               "--output", "packed.pdb", "--overwrite", "--noprogress"]
        if pack_params["movebadrandom"]:
            cmd.append("--movebadrandom")
        log.info("packing the membrane in %s (this is the slowest step of part B)", tmp)
        r = _run(cmd, tmp, "packmol_memgen.log")
        for f in tmp.glob("*.log"):
            shutil.copy2(f, wd / f.name)
        if r.returncode != 0 or not (tmp / "packed.pdb").exists():
            issues.add("system", Status.FAIL, f"packmol-memgen failed: see {wd / 'packmol_memgen.log'}")
            return BuildResults(tables, issues, wd)
        shutil.copy2(tmp / "packed.pdb", packed)
        marker.write_text(json.dumps(pack_params, indent=1))
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        log.info("reusing the finished packing %s", packed)
    pk, cryst = read_pdb_molecules(packed)
    _step(prov, "membrane packing", "packmol-memgen (protein only, pre-oriented)", [prot_pdb], [packed],
          {**pack_params, "reused": reuse})

    # ---- 3. merge and tleap -------------------------------------------------------------------------------------
    lines, comp, minfo = merge_with_membrane(sol, pk, cryst, float(s.get("clash_cutoff", 1.6)))
    sys_pdb = wd / "system.pdb"
    sys_pdb.write_text("\n".join(lines) + "\n")
    tables["Composition"] = composition_table(comp, cryst, str(s.get("lipids")), str(s.get("ratio")))
    tables["MergeSummary"] = pd.DataFrame([{"Item": k, "Value": v} for k, v in {
        **minfo, "box (Å)": cryst, **{k: v for k, v in tables["Composition"].attrs.items() if k != "box (Å)"}}.items()])
    leap = wd / "build.in"
    leap.write_text(tleap_script(cfg, smap, sol.bonds, cryst))
    _run(["tleap", "-f", leap.name], wd, "build.log")
    lg = parse_build_log((wd / "build.log").read_text())
    tables["TleapBuild"] = pd.DataFrame([{"Item": k, "Value": json.dumps(v) if isinstance(v, list) else v} for k, v in lg.items()])
    prmtop, inpcrd, tpdb = wd / "system.prmtop", wd / "system.inpcrd", wd / "system_tleap.pdb"
    _step(prov, "tleap build", "tleap", [sys_pdb, leap] + [wd / f.name for f in smap.files], [prmtop, inpcrd, tpdb], lg)
    files = {"system_pdb": sys_pdb, "prmtop": prmtop, "inpcrd": inpcrd, "tleap_in": leap}
    if lg.get("errors") != 0 or not prmtop.exists():
        issues.add("system", Status.FAIL, f"tleap build failed: see {wd / 'build.log'}")
        tables["SystemChecks"] = system_checks(tables, lg, None, originals, originals, cfg)
        return BuildResults(tables, issues, wd, files)

    # ---- 4. OpenMM -----------------------------------------------------------------------------------------------
    md = None
    if run_md:
        try:
            idx = atom_index(tpdb)
            zn_pairs = [(idx[(a, n1)], idx[(b, n2)]) for a, n1, b, n2 in sol.zn_bond_atoms
                        if (a, n1) in idx and (b, n2) in idx]
            kind = dict(zip(sol.residues["seq"], sol.residues["kind"]))
            heavy = [i for (sq, an), i in idx.items() if kind.get(sq) == "protein" and not an.startswith("H")]
            md, info = run_openmm(cfg, prmtop, inpcrd, zn_pairs, wd, heavy)
            tables["Minimisation"] = md
            tables["SystemSummary"] = pd.DataFrame([{"Item": k, "Value": v} for k, v in info.items()])
            files.update({"minimised_pdb": wd / "system_minimised.pdb", "openmm_system": wd / "system.xml"})
            _step(prov, "OpenMM minimisation and smoke run", f"OpenMM ({info['platform']})", [prmtop, inpcrd],
                  [wd / "system_minimised.pdb", wd / "system.xml"], cfg.get("md.openmm"))
        except ImportError:
            issues.add("system", Status.NOT_RUN, "OpenMM not installed: minimisation NOT RUN")
        except Exception as ex:  # noqa: BLE001
            issues.add("system", Status.FAIL, f"OpenMM minimisation failed: {ex}")
    after = {p: sha256_file(Path(p)) for p in originals}
    tables["SystemChecks"] = system_checks(tables, lg, md, originals, after, cfg)
    tables["MDReadiness"] = md_readiness(tables, cfg)
    (wd / "provenance.json").write_text(json.dumps({"originals": originals, "steps": prov}, indent=1, default=str))
    files["provenance"] = wd / "provenance.json"
    return BuildResults(tables, issues, wd, files)


def system_checks(t: dict, lg: dict, md: pd.DataFrame | None, before: dict, after: dict, cfg: Any,
                  critical=("originals_unchanged", "tleap_build", "neutral", "minimisation_finite",
                            "minimised_forces", "zn_geometry_kept")) -> pd.DataFrame:
    oc = cfg.get("md.openmm", {}) or {}
    rows = []
    add = lambda c, st, d: rows.append({"Check": c, "Status": st.value, "Detail": d})  # noqa: E731
    changed = [p for p in before if before[p] != after.get(p)]
    add("originals_unchanged", Status.FAIL if changed else Status.PASS, ", ".join(changed) or f"{len(before)} unchanged")
    ok = lg.get("errors") == 0 and not lg.get("missing parameters") and not lg.get("unknown")
    add("tleap_build", Status.PASS if ok else Status.FAIL,
        f"errors {lg.get('errors')}, missing parameters {len(lg.get('missing parameters') or [])}, unknown {len(lg.get('unknown') or [])}")
    q = lg.get("charge after neutralising")
    add("neutral", Status.PASS if q is not None and abs(q) < 1e-3 else Status.FAIL, f"net charge {q}")
    comp = t.get("Composition", pd.DataFrame())
    if len(comp):
        cf, want = comp.attrs.get("cholesterol fraction"), comp.attrs.get("requested cholesterol fraction")
        if cf is not None and want is not None:
            add("lipid_ratio", Status.PASS if abs(cf - want) <= 0.05 else Status.WARNING,
                f"cholesterol {cf:.2f} (requested {want:.2f}) after clash removal")
    if md is None:
        add("minimisation_finite", Status.NOT_RUN, "OpenMM not run (no GPU, or build only): run notebook 05 section F again on a GPU runtime; the packing is reused")
    else:
        mins = md[md["Stage"] == "minimised (unrestrained)"]
        fin = bool(md["Finite"].dropna().all())
        add("minimisation_finite", Status.PASS if fin else Status.FAIL, "all energies finite" if fin else "NaN / infinite energy")
        if len(mins):
            mf = float(mins["Max force (kJ/mol/nm)"].iloc[0])
            add("minimised_forces", Status.PASS if mf <= float(oc.get("max_force_ok", 1000)) else Status.FAIL,
                f"max force {mf:.0f} kJ/mol/nm (limit {oc.get('max_force_ok', 1000)})")
        lim = float(oc.get("zn_bond_max_change", 0.3))
        ch = md.loc[md["Stage"].str.startswith("minimised"), "Max Zn–ligand change (Å)"].dropna()
        add("zn_geometry_kept", Status.PASS if len(ch) and ch.max() <= lim else Status.FAIL,
            f"largest Zn–ligand change after minimisation {ch.max():.3f} Å (limit {lim})" if len(ch)
            else "no Zn distances measured")
        smoke = md[md["Stage"].str.contains("NVT|NPT|smoke")]
        if len(smoke):
            if smoke["Zn–ligand distances (Å)"].astype(str).str.startswith("NOT RUN").any():
                add("smoke_run", Status.NOT_RUN, "needs a GPU platform")
            else:
                T = float(cfg.get("md.openmm.temperature_k", 310))
                temps = smoke["Temperature (K)"].dropna()
                okT = len(temps) and (temps - T).abs().max() <= 15
                zc = smoke["Max Zn–ligand change (Å)"].dropna()         # thermal motion: twice the minimisation limit
                okZ = not len(zc) or zc.max() <= 2 * lim
                add("smoke_run", Status.PASS if okT and okZ and smoke["Finite"].all() else Status.WARNING,
                    f"temperatures {', '.join(f'{x:.0f}' for x in temps)} K (target {T:.0f}); largest Zn–ligand "
                    f"change {zc.max() if len(zc) else float('nan'):.2f} Å (limit {2 * lim:.1f})")
    df = pd.DataFrame(rows)
    df.insert(1, "Critical", df["Check"].isin(set(critical)))
    return df


def md_readiness(t: dict, cfg: Any) -> pd.DataFrame:
    """MD readiness: READY only if no item FAILs (assumptions are listed as WARNING)."""
    rows = []
    add = lambda item, st, d: rows.append({"Item": item, "Status": st.value, "Detail": d})  # noqa: E731
    ch = t.get("SystemChecks", pd.DataFrame()).set_index("Check") if "SystemChecks" in t else pd.DataFrame()
    for c in ("tleap_build", "neutral", "minimisation_finite", "minimised_forces", "zn_geometry_kept", "smoke_run"):
        if c in ch.index:
            st = Status(ch.loc[c, "Status"])
            if c == "smoke_run" and st == Status.NOT_RUN:      # needs a GPU; minimisation alone decides readiness
                st = Status.WARNING
            add(c.replace("_", " "), st, ch.loc[c, "Detail"])
    add("zinc model", Status.PASS, "bonded MCPB.py parameters (QM: PySCF B3LYP/6-31G*), bridging hydroxide, Glu424 GLH")
    s = cfg.get("md.system", {}) or {}
    add("lipid composition", Status.WARNING, f"ASSUMPTION {s.get('lipids')} {s.get('ratio')} (to compare with a CHARMM-GUI build)")
    add("glycans", Status.WARNING, "core GlcNAc only (GLYCAM06j), at sites observed glycosylated; full glycoforms not modelled")
    add("force fields", Status.PASS, f"{cfg.get('md.protein_ff')} / {s.get('lipid_ff')} / {s.get('glycan_ff')} / "
        f"{cfg.get('md.water_model')} + MCPB.py site")
    fails = [r["Item"] for r in rows if r["Status"] in (Status.FAIL.value, Status.NOT_RUN.value)]
    add("OVERALL", Status.FAIL if fails else Status.PASS,
        ("NOT READY: " + ", ".join(fails)) if fails else "READY for equilibration (assumptions listed as WARNING)")
    return pd.DataFrame(rows)


def save_system(cfg: Any, res: BuildResults, tag: str = "") -> dict[str, Path]:
    from .excel_output import write_tables, write_workbook
    from .manifest import build_manifest, detect_software, manifest_table, write_manifest

    software = detect_software()
    sheets = dict(res.tables)
    sheets["SystemIssues"] = res.issues.to_frame()
    sheets["Software"] = software
    manifest = build_manifest(cfg, structure_ids=[str(cfg.get("preparation.template"))], input_files=[], software=software,
                              extra={"stage": "membrane system build and minimisation (notebook 05 part B)",
                                     "workdir": str(res.workdir), "assumptions": {"lipids": cfg.get("md.system.lipids"),
                                                                                  "ratio": cfg.get("md.system.ratio")}})
    sheets["Reproducibility"] = manifest_table(manifest)
    suffix = f"_{tag}" if tag else ""
    wb = cfg.output_file("md_workbook")
    wb = wb.with_name(wb.stem + "_system" + suffix + ".xlsx")
    write_workbook(wb, sheets)
    mpath = write_manifest(manifest, cfg.outputs_dir / f"{cfg.accession}_md_system_manifest{suffix}.json")
    tdir = cfg.outputs_dir / f"md_system_tables{suffix}"
    write_tables(tdir, sheets, csv=bool(cfg.get("outputs.write_csv", True)), parquet=bool(cfg.get("outputs.write_parquet", True)))
    return {"workbook": wb, "manifest": mpath, "tables_dir": tdir, **res.files}
