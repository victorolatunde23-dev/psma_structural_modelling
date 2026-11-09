"""Milestone 6 part A: bonded Zn parameters (MCPB.py + PySCF bridge), offline.

* qm_bridge: Gaussian-input parsing, MK points, fchk / MK-log writers (round trip of values);
* zinc_params: site extraction (bridging hydroxide), mol2 / MCPB input writers, frcmod and tleap
  parsers, and the full orchestration with fake AmberTools executables (MCPB.py, tleap,
  parmchk2, resp) that write AmberTools-format files; the PySCF step is replaced by a stub.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from psma_pipeline import qm_bridge as Q
from psma_pipeline import zinc_params as ZP
from psma_pipeline.status import IssueLog, Status

ZN1, ZN2 = np.array([0.0, 0.0, 0.0]), np.array([3.3, 0.0, 0.0])
OW = np.array([1.65, 1.0, 0.0])


# ---------------------------------------------------------------------------
# qm_bridge
# ---------------------------------------------------------------------------
MK_COM = """%Chk=PSMA_large_mk.chk
%Mem=3000MB
%NProcShared=2
# B3LYP/6-31G* Integral=(Grid=UltraFine) Pop(MK,ReadRadii)
IOp(6/33=2,6/42=6)

CLR

0  1
Zn          0.000    0.000    0.000
Zn          3.300    0.000    0.000
O           1.650    1.000    0.000
H           1.650    1.970    0.000

Zn 1.395

"""


def test_parse_gaussian_inputs(tmp_path):
    p = tmp_path / "PSMA_large_mk.com"
    p.write_text(MK_COM)
    j = Q.parse_gaussian_input(p)
    assert (j.kind, j.method, j.basis, j.charge, j.mult) == ("mk", "B3LYP", "6-31G*", 0, 1)
    assert j.elements == ["Zn", "Zn", "O", "H"] and j.radii == {"Zn": 1.395} and j.density == 6.0
    opt = tmp_path / "PSMA_small_opt.com"
    opt.write_text(MK_COM.replace("Pop(MK,ReadRadii)", "Opt").replace("Zn          0.000", "Zn     -1   0.000"))
    j2 = Q.parse_gaussian_input(opt)
    assert j2.kind == "opt" and j2.frozen[0] and not j2.frozen[1]
    fc = tmp_path / "PSMA_small_fc.com"
    fc.write_text("%Chk=x\n# B3LYP/6-31G* Freq Geom=AllCheckpoint Guess=Read\nIntegral=(Grid=UltraFine) IOp(7/33=1)\n\n\n")
    assert Q.parse_gaussian_input(fc).kind == "freq"


def test_mk_points_shells_and_exclusion():
    els = ["Zn", "O", "H"]
    xyz = np.array([[0.0, 0, 0], [1.9, 0, 0], [2.4, 0.8, 0]])
    pts = Q.mk_points(els, xyz, {"Zn": 1.395}, 6.0)
    rad = np.array([1.395, 1.40, 1.20])
    d = np.linalg.norm(pts[:, None, :] - xyz[None], axis=2)
    assert (d >= 1.4 * rad[None, :] - 1e-6).all()                   # nothing inside the innermost shell
    assert (d.min(axis=1) <= 2.0 * rad.max() + 1e-6).all()
    assert 500 < len(pts) < 3000
    with pytest.raises(ValueError, match="no MK radius"):
        Q.mk_points(["Zn"], xyz[:1], {}, 6.0)


def test_fchk_and_mk_log_round_trip(tmp_path):
    """Writers reproduce MCPB.py's (pymsmt.mol.gauio) parsing rules: fixed-count blocks and columns 33-62."""
    rng = np.random.default_rng(0)
    els = ["Zn", "O", "H", "C"]
    xyz = rng.normal(scale=3, size=(4, 3)) + np.array([-60.0, 20.0, 35.0])
    H = rng.normal(size=(12, 12))
    H = H + H.T
    f = Q.write_fchk(tmp_path / "t.fchk", "t", els, xyz, H, 0, 1, -1.0, "B3LYP", "6-31G*")
    lines = f.read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if "Current cartesian coordinates" in l)
    n = int(lines[i].split()[-1])
    vals = [float(v) for l in lines[i + 1:i + 1 + -(-n // 5)] for v in l.split()]
    assert n == 12 and np.allclose(np.array(vals).reshape(4, 3) * Q.BOHR, xyz, atol=1e-6)
    i = next(k for k, l in enumerate(lines) if "Cartesian Force Constants" in l)
    n = int(lines[i].split()[-1])
    tri = [float(v) for l in lines[i + 1:i + 1 + -(-n // 5)] for v in l.split()]
    M = np.zeros((12, 12))
    k = 0
    for a in range(12):
        for b in range(a + 1):
            M[a, b] = M[b, a] = tri[k]
            k += 1
    assert n == 78 and np.allclose(M, H, rtol=1e-7, atol=1e-7)
    pts = rng.normal(size=(30, 3)) * 4 + xyz.mean(0)
    espa, espp = rng.normal(size=4), rng.normal(size=30)
    log = Q.write_mk_log(tmp_path / "t.log", els, xyz, pts, espa, espp, -1.0).read_text().splitlines()
    ac = [l for l in log if "      Atomic Center" in l]
    fc = [l for l in log if "     ESP Fit Center" in l]
    assert np.allclose([[float(l[32:42]), float(l[42:52]), float(l[52:62])] for l in ac], xyz, atol=1e-5)
    assert np.allclose([[float(l[32:42]), float(l[42:52]), float(l[52:62])] for l in fc], pts, atol=1e-5)
    b = next(k for k, l in enumerate(log) if "Electrostatic Properties (Atomic Units)" in l) + 6
    atoms = [float(l.split()[-1]) for l in log[b:] if " Atom " in l]
    fits = [float(l.split()[-1]) for l in log[b:] if " Fit " in l]
    assert np.allclose(atoms, espa, atol=1e-6) and np.allclose(fits, espp, atol=1e-6)


def test_harmonic_frequencies_units():
    """A diatomic with a known force constant: H2 (k = 0.37 Eh/bohr²) ≈ 4400 cm-1."""
    k = 0.37
    H = np.zeros((6, 6))
    H[0, 0] = H[3, 3] = k
    H[0, 3] = H[3, 0] = -k
    f = Q.harmonic_frequencies(H, ["H", "H"])
    assert abs(f.max() - 4400) < 150


# ---------------------------------------------------------------------------
# zinc_params helpers
# ---------------------------------------------------------------------------
def prepared_pdb(path: Path) -> Path:
    lines, n = [], 1
    res = [(377, "HID", [("N", -4, 4, 0), ("CA", -4, 3, 0), ("C", -3, 3, 0), ("O", -3, 2, 0), ("NE2", -2.1, 0, 0)]),
           (387, "ASP", [("N", 1, -5, 0), ("CA", 1, -4, 0), ("C", 2, -4, 0), ("O", 2, -5, 0), ("OD1", 0.2, -1.9, 0),
                         ("OD2", 3.0, -2.0, 0)]),
           (425, "GLU", [("N", 6, -4, 0), ("CA", 6, -3, 0), ("C", 7, -3, 0), ("O", 7, -4, 0), ("OE1", 4.8, -1.0, 0.8),
                         ("OE2", 5.4, 0.0, 0)]),
           (453, "ASP", [("N", -3, 0, 5), ("CA", -3, 0, 4), ("C", -2, 0, 4), ("O", -2, 1, 4), ("OD2", 0, 0, 2.0)]),
           (553, "HID", [("N", 7, 3, 0), ("CA", 7, 4, 0), ("C", 6, 4, 0), ("O", 6, 5, 0), ("NE2", 3.3, 0, -2.15)])]
    for ch, shift in (("A", 0.0), ("B", 40.0)):
        for num, rn, atoms in res:
            for a, x, y, z in atoms:
                lines.append(ZP.pdb_line(n, "ATOM", a, rn, ch, num, (x + shift, y, z), a[0]))
                n += 1
    lines.append("TER")
    for ch, shift in (("A", 0.0), ("B", 40.0)):
        for k, p in enumerate((ZN1, ZN2)):
            lines.append(ZP.pdb_line(n, "HETATM", "ZN", "ZN", ch, 3001 + k, p + [shift, 0, 0], "ZN"))
            n += 1
        lines.append(ZP.pdb_line(n, "HETATM", "O", "HOH", ch, 4001, OW + [shift, 0, 0], "O"))
        n += 1
        lines.append(ZP.pdb_line(n, "HETATM", "O", "HOH", ch, 4002, OW + [shift, 6.0, 0], "O"))   # not bridging
        n += 1
    lines.append(ZP.pdb_line(n, "HETATM", "C1", "NAG", "A", 2001, (20, 20, 20), "C"))
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def test_extract_site_hydroxide(tmp_path):
    pdb = ZP.read_pdb(prepared_pdb(tmp_path / "p.pdb"))
    site = ZP.extract_site(pdb, "A", "hydroxide")
    assert len(site.zinc) == 2 and np.allclose(site.bridge_o, OW)
    h = site.hydroxide_h
    assert abs(np.linalg.norm(h - OW) - 0.97) < 1e-6
    mid = (ZN1 + ZN2) / 2
    assert np.linalg.norm(h - mid) > np.linalg.norm(OW - mid)          # H points away from the Zn pair
    assert site.info["Zn–Zn (Å)"] == 3.3 and set(site.protein["chain"]) == {"A"}
    w = ZP.extract_site(pdb, "A", "water")
    assert w.hydroxide_h is None
    with pytest.raises(ValueError, match="no water within"):
        ZP.extract_site(pdb[~((pdb["resname"] == "HOH") & (pdb["resnum"] == 4001))], "A")


def test_mol2_and_mcpb_input(cfg):
    z = ZP.zn_mol2(ZN1)
    o = ZP.hydroxide_mol2(OW, OW + [0, 0.97, 0])
    for txt in (z, o):                                   # pymsmt reads ATOM lines up to the BOND tag
        assert "@<TRIPOS>ATOM" in txt and "@<TRIPOS>BOND" in txt
    q = sum(float(l.split()[8]) for l in o.split("@<TRIPOS>ATOM\n")[1].split("@<TRIPOS>BOND")[0].splitlines())
    assert q == pytest.approx(-1.0)
    inp = ZP.mcpb_input("PSMA", "PSMA_site.pdb", [12, 13], cfg)
    keys = {l.split()[0] for l in inp.splitlines()}
    assert {"original_pdb", "group_name", "cut_off", "ion_ids", "ion_mol2files", "force_field", "software_version",
            "naa_mol2files", "frcmod_files", "scale_factor"} <= keys
    assert "cut_off 2.5" in inp and "software_version g09" in inp and "force_field ff14SB" in inp


FRCMOD = """Remark line goes here
MASS
M1 65.38
M2 65.38
Y1 14.01         0.530
BOND
M1-Y1   62.3   2.1200     Created by Seminario method using MCPB.py
M1-Y2   75.1   1.9100     Created by Seminario method using MCPB.py
M1-Y3   70.0   2.0000     Created by Seminario method using MCPB.py
M1-Y4   90.2   1.8300     Created by Seminario method using MCPB.py
M2-Y4   60.0   2.0400     Created by Seminario method using MCPB.py
ANGL
Y1-M1-Y2   35.5   104.5     Created by Seminario method using MCPB.py

DIHE

IMPR

NONB
  M1          1.395  0.01491500  IOD set
  M2          1.395  0.01491500  IOD set
"""


def test_parse_frcmod_and_tleap_log():
    fr = ZP.parse_frcmod(FRCMOD)
    assert fr["BOND"].shape[0] == 5 and fr["BOND"].iloc[0]["Bond"] == "M1-Y1"
    assert fr["BOND"].iloc[0]["k (kcal/mol/Å²)"] == 62.3 and fr["ANGL"].iloc[0]["θ0 (°)"] == 104.5
    assert fr["NONB"].iloc[0]["R* (Å)"] == 1.395
    lg = ZP.parse_tleap_log("Checking 'mol'....\nTotal unperturbed charge:  -9.000000\n"
                            "Exiting LEaP: Errors = 0; Warnings = 2; Notes = 1.\n")
    assert lg == {"errors": 0, "warnings": 2, "total charge": -9.0, "missing parameters": [], "unknown": []}
    bad = ZP.parse_tleap_log("Could not find bond parameter for: M1 - Y9\nExiting LEaP: Errors = 1; Warnings = 0; Notes = 0.\n")
    assert bad["errors"] == 1 and len(bad["missing parameters"]) == 1


# ---------------------------------------------------------------------------
# Orchestration with fake AmberTools
# ---------------------------------------------------------------------------
FAKE_TLEAP = r'''#!{python}
import re, shutil, sys
from pathlib import Path
f = Path(sys.argv[sys.argv.index("-f") + 1]).read_text()
if "site_protein_H.pdb" in f:
    shutil.copy("site_protein_noH.pdb", "site_protein_H.pdb")
    print("Exiting LEaP: Errors = 0; Warnings = 0; Notes = 0.")
else:
    for m in re.finditer(r"saveamberparm mol (\S+) (\S+)", f):
        Path(m.group(1)).write_text("prmtop"); Path(m.group(2)).write_text("inpcrd")
    print("Checking 'mol'....\nTotal unperturbed charge:  -3.000000\nExiting LEaP: Errors = 0; Warnings = 1; Notes = 0.")
'''

FAKE_PARMCHK = r'''#!{python}
import sys
from pathlib import Path
Path(sys.argv[sys.argv.index("-o") + 1]).write_text("remark\nMASS\n\nBOND\noh-ho  563.5   0.973\n\nANGL\n\nDIHE\n\nIMPR\n\nNONB\n\n")
'''

FAKE_MCPB = r'''#!{python}
import re, sys
from pathlib import Path
inp = Path(sys.argv[sys.argv.index("-i") + 1]).read_text()
step = sys.argv[sys.argv.index("-s") + 1]
g = re.search(r"group_name (\S+)", inp).group(1)
pdb = re.search(r"original_pdb (\S+)", inp).group(1)
ions = [int(x) for x in re.search(r"ion_ids (.*)", inp).group(1).split()]
atoms = [l for l in Path(pdb).read_text().splitlines() if l.startswith(("ATOM", "HETATM"))]
assert all(atoms[i - 1][17:20].strip() == "ZN" for i in ions), "ion_ids do not point at Zn"
if step == "1":
    for n in ("small_opt", "small_fc", "large_mk"):
        Path(f"{g}_{n}.com").write_text("%Chk=x\n# B3LYP/6-31G* Opt\n\nCLR\n\n0  1\nZn 0.0 0.0 0.0\n\n")
elif step == "2":
    Path(f"{g}_mcpbpy.frcmod").write_text(Path("{frcmod}").read_text())
elif step == "3":
    def mol2(name, atoms):
        rows = "".join(f"  {k+1} {a} 0.0 0.0 0.0 {t} 1 {name} {q:.6f}\n" for k, (a, t, q) in enumerate(atoms))
        return f"@<TRIPOS>MOLECULE\n{name}\n@<TRIPOS>ATOM\n{rows}@<TRIPOS>BOND\n"
    Path("ZN1.mol2").write_text(mol2("ZN1", [("ZN", "M1", 0.60)]))
    Path("ZN2.mol2").write_text(mol2("ZN2", [("ZN", "M2", 0.40)]))
    Path("HD1.mol2").write_text(mol2("HD1", [("N", "N", -0.4157), ("NE2", "Y1", -0.3), ("CA", "CX", -0.2843)]))
    Path("AP1.mol2").write_text(mol2("AP1", [("N", "N", -0.4), ("OD1", "Y2", -0.8), ("OD2", "Y3", 0.2)]))
    Path("OH1.mol2").write_text(mol2("OH1", [("O", "Y4", -1.1), ("H", "HO", 0.1)]))
elif step == "4":
    ren = {"377": "HD1", "387": "AP1"}
    out = []
    for l in atoms:
        rn, num = l[17:20].strip(), l[22:26].strip()
        if rn == "ZN":
            rn = "ZN1" if l[6:11].strip() == str(ions[0]) else "ZN2"
        elif rn == "OHX":
            rn = "OH1"
        rn = ren.get(num, rn)
        out.append(l[:17] + f"{rn:>3s}" + l[20:])
    Path(f"{g}_mcpbpy.pdb").write_text("\n".join(out) + "\nEND\n")
    zn = [int(atoms[i - 1][22:26]) for i in ions]
    oh = int([l for l in atoms if l[17:20] == "OHX"][0][22:26])
    Path(f"{g}_tleap.in").write_text(
        "source leaprc.protein.ff14SB\nsource leaprc.water.tip3p\nloadamberparams {g}_mcpbpy.frcmod\n"
        f"mol = loadpdb {g}_mcpbpy.pdb\n"
        f"bond mol.{zn[0]}.ZN mol.377.NE2\nbond mol.{zn[0]}.ZN mol.387.OD1\nbond mol.{zn[0]}.ZN mol.{oh}.O\n"
        f"bond mol.{zn[1]}.ZN mol.{oh}.O\nbond mol.386.C mol.387.N\n"
        "solvatebox mol TIP3PBOX 10.0\naddions mol Na+ 0\nsaveamberparm mol {g}_solv.prmtop {g}_solv.inpcrd\nquit\n")
'''


@pytest.fixture
def fake_amber(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    frc = tmp_path / "frcmod.txt"
    frc.write_text(FRCMOD)
    for name, body in (("tleap", FAKE_TLEAP), ("parmchk2", FAKE_PARMCHK), ("MCPB.py", FAKE_MCPB),
                       ("resp", "#!{python}\n"), ("antechamber", "#!{python}\n")):
        p = bindir / name
        p.write_text(body.replace("{python}", sys.executable).replace("{frcmod}", str(frc)))
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return bindir


def test_run_zinc_parameters_orchestration(cfg, tmp_path, fake_amber, monkeypatch):
    calls = {}

    def fake_qm(wd, group, use_gpu=None):
        calls["qm"] = (Path(wd), group)
        for n in ("small_opt.fchk", "large_mk.log"):
            (Path(wd) / f"{group}_{n}").write_text("fake")
        s = {"group": group, "gpu": False, "pyscf": "test",
             "small model": {"atoms": 50, "optimisation converged": True, "imaginary frequencies < -50 cm-1": 0},
             "large model": {"atoms": 140, "MK points": 2500}}
        (Path(wd) / f"{group}_pyscf_summary.json").write_text(json.dumps(s))
        return s
    monkeypatch.setattr(Q, "run_mcpb_qm", fake_qm)
    pdb = prepared_pdb(tmp_path / "microstate_00_default.pdb")
    before = pdb.read_bytes()
    res = ZP.run_zinc_parameters(cfg, pdb, IssueLog())
    t = res.tables
    assert calls["qm"][1] == "PSMA"
    checks = t["ZincChecks"].set_index("Check")
    assert (t["ZincChecks"].query("Critical")["Status"] == "PASS").all(), t["ZincChecks"].to_string()
    zb = t["ZnBonds"]
    assert len(zb) == 4 and zb["MCPB k (kcal/mol/Å²)"].notna().all()
    nb = zb.set_index("Atom 2").loc["HD1377:NE2"]
    assert nb["Types"] == "M1-Y1" and nb["MCPB r0 (Å)"] == 2.12 and abs(nb["Model distance (Å)"] - 2.1) < 1e-3
    sr = t["SiteResidues"].set_index("Residue")
    assert sr.loc["OH1", "Net charge"] == pytest.approx(-1.0) and "Y4" in sr.loc["OH1", "Metal-site atoms (type, RESP charge)"]
    assert pdb.read_bytes() == before                                  # original untouched
    wd = res.workdir
    site = (wd / "PSMA_site.pdb").read_text()
    het = [l for l in site.splitlines() if l.startswith("HETATM")]
    assert [l[17:20].strip() for l in het] == ["ZN", "ZN", "OHX", "OHX"] and "NAG" not in site
    assert "solvatebox" not in (wd / "check_build.in").read_text()
    prov = json.loads((wd / "provenance.json").read_text())
    assert [s["name"] for s in prov["steps"]][:2] == ["site PDB + MCPB input", "MCPB.py step 1 (models + Gaussian inputs)"]
    from psma_pipeline.excel_output import data_dictionary
    paths = ZP.save_zinc_parameters(cfg, res, tag="test")
    assert paths["workbook"].exists()
    dd = data_dictionary({**t, "ZincIssues": res.issues.to_frame()})
    und = dd[dd["Description"] == "UNDOCUMENTED"]
    assert und.empty, und[["Sheet", "Column"]].to_string()


def test_site_charge_check():
    t = {"SiteResidues": pd.DataFrame({"Residue": ["ZN1", "OH1"], "Net charge": [0.62, -1.0]})}
    c = ZP.zinc_checks(t, {"errors": 0, "missing parameters": []}, {}, {}, {"small model": {"optimisation converged": True}},
                       None).set_index("Check")
    assert c.loc["site_charge_integer", "Status"] == "FAIL"


def test_missing_ambertools_is_fail(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(ZP, "ambertools_available", lambda: {t: None for t in ("tleap", "MCPB.py", "parmchk2", "resp")})
    issues = IssueLog()
    res = ZP.run_zinc_parameters(cfg, prepared_pdb(tmp_path / "p.pdb"), issues)
    assert issues.count(Status.FAIL) == 1 and "ZincChecks" not in res.tables


def test_opt_checkpoint_and_resume(tmp_path, monkeypatch):
    """geomeTRIC's callback saves each geometry; an interrupted run restarts from the last one."""
    import types
    com = tmp_path / "W_small_opt.com"
    com.write_text("%Chk=W\n# B3LYP/6-31G* Opt\n\nw\n\n0  1\nO 0.0 0.0 0.2\nH 0.0 0.857 -0.469\nH 0.0 -0.757 -0.469\n\n")
    job = Q.parse_gaussian_input(com)
    seen = {}

    class Mol:
        def __init__(self, xyz):
            self.xyz = np.asarray(xyz, float)

        def atom_coords(self, unit="Angstrom"):
            return self.xyz

    class MF:
        def __init__(self, mol):
            self.mol, self.e_tot = mol, -76.0

    def kernel(mf, maxsteps=200, callback=None, **kw):
        seen.setdefault("starts", []).append(mf.mol.xyz.copy())
        x = mf.mol.xyz.copy()
        for k in range(3):
            x = x + 0.01
            callback({"mol": Mol(x), "energy": -76.0 - k})
            if seen.get("crash") and k == 1:
                raise KeyboardInterrupt
        return True, Mol(x)

    mods = {"pyscf": types.ModuleType("pyscf"), "pyscf.geomopt": types.ModuleType("pyscf.geomopt"),
            "pyscf.geomopt.geometric_solver": types.ModuleType("pyscf.geomopt.geometric_solver")}
    mods["pyscf.geomopt.geometric_solver"].kernel = kernel
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.setattr(Q, "_pyscf_mol", lambda j, c=None: Mol(j.coords if c is None else c))
    monkeypatch.setattr(Q, "_scf", lambda mol, j, g: MF(mol))
    seen["crash"] = True
    with pytest.raises(KeyboardInterrupt):
        Q.run_opt(job, tmp_path, False)
    ck = np.load(tmp_path / "W_small_opt.opt_checkpoint.npz")
    assert int(ck["step"]) == 2 and np.allclose(ck["coords"], job.coords + 0.02)
    seen["crash"] = False
    xyz, e, info = Q.run_opt(job, tmp_path, False)
    assert info["resumed"] and info["steps"] == 5 and np.allclose(seen["starts"][-1], job.coords + 0.02)
    assert np.allclose(xyz, job.coords + 0.05)
