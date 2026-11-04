"""Milestone 6 part B: membrane system build, offline.

* site map: MCPB.py names / bonds read from part A outputs and translated from tleap's 1..n numbering
  to UniProt numbering, then applied to *both* chains;
* solute: Zn template matching by ligand set, hydroxides, GLYCAM core GlcNAc, bonds;
* molecule grouping of a packmol-memgen PDB with sparse TER records (Lipid21 split lipids);
* merge: superposition on the packed protein and removal of whole clashing molecules;
* orchestration with fake packmol-memgen and tleap (the fake tleap checks every ``bond`` command
  against the residue/atom it names); OpenMM is exercised only when installed.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from psma_pipeline import system_build as SB
from psma_pipeline import zinc_params as ZP
from psma_pipeline.status import IssueLog, Status
from test_zinc_params_offline import prepared_pdb

ORDER = [377, 387, 425, 453, 553]


def prepared_with_glycan(path: Path) -> Path:
    """Part A's two-chain fixture plus an Asn-linked core GlcNAc on chain B and a Ca and a Cl."""
    p = prepared_pdb(path)
    lines = [l for l in p.read_text().splitlines() if l.startswith(("ATOM", "HETATM", "TER")) and "NAG" not in l]
    asn = [("N", 50, 10, 0), ("CA", 50, 11, 0), ("C", 51, 11, 0), ("O", 51, 12, 0), ("ND2", 50, 12.5, 1.0)]
    atoms = [l for l in lines if l.startswith("ATOM")]
    n = 1000
    extra = [ZP.pdb_line(n + k, "ATOM", a, "ASN", "B", 638, (x, y, z), a[0]) for k, (a, x, y, z) in enumerate(asn)]
    nag = [("C1", 50, 12.5, 2.45), ("C2", 50, 13.5, 3.2), ("N2", 50, 14.8, 2.8), ("C7", 50, 15.6, 3.6),
           ("O7", 50, 15.4, 4.8), ("C8", 50, 16.9, 3.0), ("O5", 49, 12, 3)]
    het = [ZP.pdb_line(n + 10 + k, "HETATM", a, "NAG", "B", 1638, (x, y, z), a[0]) for k, (a, x, y, z) in enumerate(nag)]
    het += [ZP.pdb_line(n + 30, "HETATM", "CA", "CA", "A", 5001, (-20, 0, 0), "CA"),
            ZP.pdb_line(n + 31, "HETATM", "CL", "CL", "A", 5002, (-20, 5, 0), "CL")]
    rest = [l for l in lines if l.startswith("HETATM")]
    p.write_text("\n".join(atoms + extra + ["TER"] + rest + het) + "\nEND\n")
    return p


def zinc_dir(d: Path) -> Path:
    """Part A outputs as MCPB.py writes them: site numbered 1..n by tleap; ZN 6/7, OHX 8."""
    d.mkdir(parents=True, exist_ok=True)
    ren = {1: "HD1", 2: "AP1", 5: "HD2", 6: "ZN1", 7: "ZN2", 8: "OH1"}
    site, final = [], []
    names = ["HID", "ASP", "GLU", "ASP", "HID", "ZN", "ZN", "OHX"]
    for i, rn in enumerate(names, start=1):
        rec = "ATOM" if i <= 5 else "HETATM"
        for nm, rr, out in ((rn, rn, site), (rn, ren.get(i, rn), final)):
            out.append(ZP.pdb_line(i, rec, "ZN" if rn == "ZN" else "CA", rr, "", i, (0, 0, i), "C"))
    (d / "PSMA_site.pdb").write_text("\n".join(site) + "\nEND\n")
    (d / "PSMA_mcpbpy.pdb").write_text("\n".join(final) + "\nEND\n")
    for r in ("HD1", "AP1", "HD2", "ZN1", "ZN2", "OH1"):
        (d / f"{r}.mol2").write_text(f"@<TRIPOS>MOLECULE\n{r}\n")
    (d / "PSMA_mcpbpy.frcmod").write_text("frcmod\n")
    (d / "PSMA_tleap.in").write_text(
        "source leaprc.protein.ff14SB\nsource leaprc.water.tip3p\naddAtomTypes {\n"
        '        { "M1"  "Zn" "sp3" }\n        { "Y1"  "N" "sp3" }\n}\n'
        + "".join(f"{r} = loadmol2 {r}.mol2\n" for r in ("HD1", "AP1", "HD2", "ZN1", "ZN2", "OH1"))
        + "loadamberparams PSMA_mcpbpy.frcmod\nloadamberparams frcmod.ions1lm_126_tip3p\n"
          "mol = loadpdb PSMA_mcpbpy.pdb\n"
          "bond mol.6.ZN mol.1.NE2\nbond mol.6.ZN mol.2.OD1\nbond mol.6.ZN mol.4.OD2\nbond mol.6.ZN mol.8.O\n"
          "bond mol.7.ZN mol.3.OE1\nbond mol.7.ZN mol.3.OE2\nbond mol.7.ZN mol.5.NE2\nbond mol.7.ZN mol.8.O\n"
          "bond mol.4.C mol.5.N\nsolvatebox mol TIP3PBOX 10.0\nsaveamberparm mol PSMA_solv.prmtop PSMA_solv.inpcrd\nquit\n")
    return d


def test_site_map_and_uniprot_translation(tmp_path):
    sm = SB.site_map(zinc_dir(tmp_path / "z"))
    assert sm.resnames == {1: "HD1", 2: "AP1", 5: "HD2"} and sm.zn_names == ["ZN1", "ZN2"] and sm.oh_name == "OH1"
    assert sm.zn_ligands == [frozenset({1, 2, 4}), frozenset({3, 5})] and sm.n_prot == 5
    assert any(l.strip() == "}" for l in sm.header) and "ZN1 = loadmol2 ZN1.mol2" in sm.header
    assert "loadamberparams frcmod.ions1lm_126_tip3p" in sm.header and not any("loadpdb" in l for l in sm.header)
    assert len(sm.files) == 7                                    # 6 mol2 + frcmod (library frcmod not local)
    u = SB.site_map_to_uniprot(sm, ORDER)
    assert u.resnames == {377: "HD1", 387: "AP1", 553: "HD2"} and u.zn_ligands[0] == frozenset({377, 387, 453})
    assert (("prot", 453), "C", ("prot", 553), "N") in u.bonds
    with pytest.raises(ValueError, match="re-run part A"):
        SB.site_map_to_uniprot(sm, ORDER[:4])


def test_build_solute_both_sites(tmp_path):
    sm = SB.site_map(zinc_dir(tmp_path / "z"))
    sol = SB.build_solute(prepared_with_glycan(tmp_path / "p.pdb"), sm)
    r = sol.residues
    assert sol.info["Zn"] == 4 and sol.info["hydroxides"] == 2 and sol.info["core GlcNAc (GLYCAM 0YB)"] == 1
    for ch in "AB":
        got = r[(r["chain"] == ch) & (r["kind"] == "protein")].set_index("id")["resname"]
        assert got[377] == "HD1" and got[387] == "AP1" and got[553] == "HD2" and got[425] == "GLU"
        assert sorted(r[(r["chain"] == ch) & (r["kind"] == "zinc")]["resname"]) == ["ZN1", "ZN2"]
    assert r[(r["chain"] == "B") & (r["id"] == 638) & (r["kind"] == "protein")]["resname"].tolist() == ["NLN"]
    # each chain's bridging HOH became a hydroxide (O + H); the non-bridging waters stay
    assert sol.info["crystal waters"] == 2 and (r["resname"] == "OH1").sum() == 2
    assert (r.loc[r["resname"] == "OH1", "atoms"] == 2).all()
    pdb = "\n".join(sol.pdb_lines)
    assert " C2N 0YB" in pdb and " CME 0YB" in pdb and " O2N 0YB" in pdb and "NAG" not in pdb
    assert "Cl- " in pdb
    # bonds: 9 per chain from the MCPB.py list + the glycan link; all atoms exist
    seq_atoms = {}
    for l in sol.pdb_lines:
        if l.startswith(("ATOM", "HETATM")):
            seq_atoms.setdefault(int(l[22:26]), set()).add(l[12:16].strip())
    assert len(sol.bonds) == 2 * 9 + 1 and len(sol.zn_bond_atoms) == 2 * 8
    for a, x, b, y in sol.bonds:
        assert x in seq_atoms[a] and y in seq_atoms[b], (a, x, b, y)
    zmap = pd.DataFrame(sol.info["zn_map"])
    assert list(zmap["Template"]) == ["ZN1", "ZN2", "ZN1", "ZN2"]
    g = sol.glycans.iloc[0]
    assert g["Chain"] == "B" and g["Asn (UniProt)"] == 638 and g["C1–ND2 (Å)"] == pytest.approx(1.45, abs=1e-3)


def test_zn_template_mismatch_raises(tmp_path):
    sm = SB.site_map(zinc_dir(tmp_path / "z"))
    sm.zn_ligands = [frozenset({5}), frozenset({5})]
    with pytest.raises(ValueError, match="matches no parameterised"):
        SB.build_solute(prepared_with_glycan(tmp_path / "p.pdb"), sm)


def _lipid(rows, start_res, origin, chol=False):
    """Lipid21 POPC (PA + PC + OL) or a cholesterol, 3 atoms per residue."""
    parts = [("CHL", "C1")] if chol else [("PA", "C12"), ("PC", "P31"), ("OL", "C12")]
    for k, (rn, a) in enumerate(parts):
        for j in range(3):
            rows.append(("ATOM", f"{a[:2]}{j}", rn, start_res + k, origin + np.array([k * 1.6, j * 1.6, 0.0])))
    return start_res + len(parts)


def packed_lines(protein: Path, R: np.ndarray, t: np.ndarray, clash_points: list[np.ndarray]) -> str:
    out = ["CRYST1   90.000   90.000  100.000  90.00  90.00  90.00 P 1           1"]
    n = 1
    for l in protein.read_text().splitlines():
        if l.startswith("ATOM"):
            x = R @ np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]) + t
            out.append(f"{l[:30]}{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}{l[54:]}")
        elif l.startswith("TER"):
            out.append("TER")
    rows, res = [], 1
    res = _lipid(rows, res, np.array([-30.0, -30, 0]))                  # far: kept
    res = _lipid(rows, res, clash_points[0] + [0.3, 0, 0])               # clashes: removed whole
    res = _lipid(rows, res, np.array([-30.0, 30, 0]), chol=True)         # kept
    for k, p in enumerate((np.array([30.0, 30, 30]), clash_points[1] + [0.2, 0.2, 0], np.array([30.0, -30, 30]))):
        for a, d in (("O", 0), ("H1", 0.96), ("H2", -0.96)):
            rows.append(("HETATM", a, "WAT", res, p + [d, 0, 0]))
        res += 1
    rows.append(("HETATM", "Na+", "Na+", res, np.array([35.0, 0, 30])))
    rows.append(("HETATM", "Cl-", "Cl-", res + 1, np.array([-35.0, 0, 30])))
    for rec, a, rn, num, x in rows:                                     # no TER inside the membrane block
        out.append(SB.pdb_atom_line(n, rec, a, rn, "", num, x, a[0]))
        n += 1
    return "\n".join(out + ["END"]) + "\n"


def test_molecule_grouping_without_ter(tmp_path):
    R = np.eye(3)
    p = tmp_path / "prot.pdb"
    p.write_text("\n".join(ZP.pdb_line(i, "ATOM", "CA", "ALA", c, i, (i, 0, 0), "C") for i, c in ((1, "A"), (2, "B"))) + "\n")
    f = tmp_path / "packed.pdb"
    f.write_text(packed_lines(p, R, np.zeros(3), [np.array([0.0, 50, 0]), np.array([0.0, 60, 0])]))
    df, cryst = SB.read_pdb_molecules(f)
    assert cryst == [90.0, 90.0, 100.0]
    mols = df.groupby("mol", sort=False)["resname"].agg(lambda s: "+".join(dict.fromkeys(s))).tolist()
    assert mols == ["ALA", "ALA", "PA+PC+OL", "PA+PC+OL", "CHL", "WAT", "WAT", "WAT", "Na+", "Cl-"]


def test_merge_superposes_and_removes_whole_molecules(tmp_path):
    sm = SB.site_map(zinc_dir(tmp_path / "z"))
    sol = SB.build_solute(prepared_with_glycan(tmp_path / "p.pdb"), sm)
    prot = SB.protein_only_pdb(sol, tmp_path / "protein.pdb")
    th = np.radians(30)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    t = np.array([45.0, 45, 50])
    oh = np.array([1.65, 1.0, 0.0])                                    # chain A bridging O in the solute frame
    f = tmp_path / "packed.pdb"
    f.write_text(packed_lines(prot, R, t, [R @ oh + t, R @ np.array([-20.0, 0, 0]) + t]))
    pk, cryst = SB.read_pdb_molecules(f)
    lines, comp, info = SB.merge_with_membrane(sol, pk, cryst, 1.6)
    assert info["superposition RMSD (Å)"] < 1e-3 and info["molecules removed"] == 2
    removed = comp[comp["removed"]]
    assert sorted(removed["kind"]) == ["lipid", "water"] and removed["residues"].tolist()[0] == "PA+PC+OL"
    # solute moved into the packed frame: the chain A Zn1 sits at t
    zn = [l for l in lines if l.startswith("HETATM") and l[17:21].strip() == "ZN1"][0]
    assert np.allclose([float(zn[30:38]), float(zn[38:46]), float(zn[46:54])], t, atol=1e-3)
    assert lines[0].startswith("CRYST1") and lines[-1] == "END"
    tab = SB.composition_table(comp, cryst, "POPC:CHL1", "7:3").set_index("Component")
    assert tab.loc["lipid", "Molecules"] == 2 and tab.loc["lipid", "Removed (clash)"] == 1
    assert tab.attrs["cholesterol fraction"] == 0.5 and tab.attrs["requested cholesterol fraction"] == 0.3
    with pytest.raises(ValueError, match="cannot superpose"):
        SB.merge_with_membrane(sol, pk[~((pk["resname"] == "HID") & (pk["atom"] == "CA"))].iloc[1:], cryst)


def test_tleap_script(cfg, tmp_path):
    sm = SB.site_map(zinc_dir(tmp_path / "z"))
    s = SB.tleap_script(cfg, sm, [(1, "SG", 2, "SG")], [90.0, 90.0, 100.0])
    lines = s.splitlines()
    assert lines[:4] == ["source leaprc.protein.ff14SB", "source leaprc.lipid21", "source leaprc.GLYCAM_06j-1",
                         "source leaprc.water.tip3p"]
    assert "bond sys.1.SG sys.2.SG" in lines and "set sys box { 90.000 90.000 100.000 }" in lines
    assert lines.index("addIonsRand sys Na+ 0") > lines.index("sys = loadpdb system.pdb") and "solvatebox" not in s
    lg = SB.parse_build_log("Total unperturbed charge:  -5.000000\nAdding 5 counter ions to \"sys\".\n"
                            "Total unperturbed charge:   0.000000\nExiting LEaP: Errors = 0; Warnings = 3; Notes = 0.\n")
    assert lg["charge before neutralising"] == -5 and lg["charge after neutralising"] == 0 and lg["ions added"] == 5


# ---------------------------------------------------------------------------
# orchestration with fake packmol-memgen and tleap
# ---------------------------------------------------------------------------
FAKE_PACKMOL = r'''#!{python}
import sys
sys.path[:0] = {paths!r}
import numpy as np
from pathlib import Path
from test_system_build_offline import packed_lines
a = sys.argv
assert "--preoriented" in a and "--notprotonate" in a and a[a.index("--lipids") + 1] == "POPC:CHL1"
prot = Path(a[a.index("--pdb") + 1])
th = np.radians(30)
R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
t = np.array([45.0, 45, 50])
Path(a[a.index("--output") + 1]).write_text(packed_lines(prot, R, t, [R @ np.array([1.65, 1.0, 0.0]) + t,
                                                                       np.array([300.0, 0, 0])]))
'''

FAKE_TLEAP_SYS = r'''#!{python}
import re, shutil, sys
from pathlib import Path
f = Path(sys.argv[sys.argv.index("-f") + 1]).read_text()
pdb = re.search(r"sys = loadpdb (\S+)", f).group(1)
res, last, atoms = 0, None, {}
for l in Path(pdb).read_text().splitlines():
    if l.startswith("TER"):
        last = None
    elif l.startswith(("ATOM", "HETATM")):
        key = l[17:27]
        if key != last:
            res += 1
            last = key
        atoms.setdefault(res, set()).add(l[12:16].strip())
bad = [b for b in re.findall(r"bond sys\.(\d+)\.(\S+) sys\.(\d+)\.(\S+)", f)
       if b[1] not in atoms.get(int(b[0]), ()) or b[3] not in atoms.get(int(b[2]), ())]
for m in re.findall(r"loadmol2 (\S+)", f):
    assert Path(m).exists(), m
if bad:
    print("Error: bond atoms not found", bad)
    print("Exiting LEaP: Errors = 1; Warnings = 0; Notes = 0.")
    sys.exit(0)
Path("system.prmtop").write_text("prmtop"); Path("system.inpcrd").write_text("inpcrd")
shutil.copy(pdb, "system_tleap.pdb")
print("Total unperturbed charge:  -3.000000\nAdding 3 counter ions to \"sys\".\nTotal unperturbed charge:   0.000000")
print("Exiting LEaP: Errors = 0; Warnings = 2; Notes = 0.")
'''


@pytest.fixture
def fake_build_tools(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    paths = [str(Path(__file__).parent), str(Path(__file__).parents[1] / "src")]
    for name, body in (("packmol-memgen", FAKE_PACKMOL), ("tleap", FAKE_TLEAP_SYS)):
        p = bindir / name
        p.write_text(body.replace("{python}", sys.executable).replace("{paths!r}", repr(paths)))
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return bindir


def test_run_system_build_orchestration(cfg, tmp_path, fake_build_tools):
    pdb = prepared_with_glycan(tmp_path / "microstate_00_default.pdb")
    zd = zinc_dir(tmp_path / "zinc")
    before = pdb.read_bytes()
    res = SB.run_system_build(cfg, pdb, zd, IssueLog(), run_md=False)
    t = res.tables
    ch = t["SystemChecks"].set_index("Check")
    assert ch.loc["tleap_build", "Status"] == "PASS" and ch.loc["neutral", "Status"] == "PASS"
    assert ch.loc["originals_unchanged", "Status"] == "PASS" and pdb.read_bytes() == before
    assert ch.loc["minimisation_finite", "Status"] == "NOT RUN"
    rd = t["MDReadiness"].set_index("Item")
    assert rd.loc["OVERALL", "Status"] == "FAIL" and "minimisation finite" in rd.loc["OVERALL", "Detail"]
    assert rd.loc["lipid composition", "Status"] == "WARNING" and "ASSUMPTION" in rd.loc["lipid composition", "Detail"]
    ms = t["MergeSummary"].set_index("Item")["Value"]
    assert ms["molecules removed"] == 1 and ms["superposition RMSD (Å)"] < 1e-3
    wd = res.workdir
    assert (wd / "HD1.mol2").exists() and (wd / "build.in").exists() and (wd / "provenance.json").exists()
    assert "bond sys." in (wd / "build.in").read_text()
    from psma_pipeline.excel_output import data_dictionary
    paths = SB.save_system(cfg, res, tag="test")
    assert paths["workbook"].exists()
    dd = data_dictionary({**t, "SystemIssues": res.issues.to_frame()})
    und = dd[dd["Description"] == "UNDOCUMENTED"]
    assert und.empty, und[["Sheet", "Column"]].to_string()


def test_bad_bond_fails_build(cfg, tmp_path, fake_build_tools):
    zd = zinc_dir(tmp_path / "zinc")
    (zd / "PSMA_tleap.in").write_text((zd / "PSMA_tleap.in").read_text().replace("mol.1.NE2", "mol.1.XX9"))
    issues = IssueLog()
    res = SB.run_system_build(cfg, prepared_with_glycan(tmp_path / "p.pdb"), zd, issues, run_md=False)
    assert issues.count(Status.FAIL) == 1 and res.tables["SystemChecks"].set_index("Check").loc["tleap_build", "Status"] == "FAIL"


def test_missing_tools_is_fail(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(SB.shutil, "which", lambda t: None)
    issues = IssueLog()
    SB.run_system_build(cfg, prepared_with_glycan(tmp_path / "p.pdb"), zinc_dir(tmp_path / "z"), issues)
    assert issues.count(Status.FAIL) == 1


def _md_frame(change=0.05, force=300.0, finite=True):
    return pd.DataFrame([
        {"Stage": "start", "Finite": True, "Max force (kJ/mol/nm)": 1e5, "Max Zn–ligand change (Å)": 0.0,
         "Zn–ligand distances (Å)": "2.0", "Temperature (K)": None},
        {"Stage": "minimised (unrestrained)", "Finite": finite, "Max force (kJ/mol/nm)": force,
         "Max Zn–ligand change (Å)": change, "Zn–ligand distances (Å)": "2.0", "Temperature (K)": None},
        {"Stage": "NVT heating (restrained)", "Finite": True, "Max force (kJ/mol/nm)": 900.0,
         "Max Zn–ligand change (Å)": change, "Zn–ligand distances (Å)": "2.0", "Temperature (K)": 309.0}])


@pytest.mark.parametrize("md,expect", [(_md_frame(), "PASS"), (_md_frame(change=0.6), "FAIL"),
                                       (_md_frame(force=5000), "FAIL"), (_md_frame(finite=False), "FAIL")])
def test_checks_and_readiness(cfg, md, expect):
    t = {"Composition": pd.DataFrame({"Component": ["lipid"]})}
    t["Composition"].attrs.update({"cholesterol fraction": 0.31, "requested cholesterol fraction": 0.3})
    lg = {"errors": 0, "missing parameters": [], "unknown": [], "charge after neutralising": 0.0}
    t["SystemChecks"] = SB.system_checks(t, lg, md, {"a": "1"}, {"a": "1"}, cfg)
    c = t["SystemChecks"].set_index("Check")
    assert c.loc["lipid_ratio", "Status"] == "PASS" and c.loc["smoke_run", "Status"] == "PASS"
    rd = SB.md_readiness(t, cfg).set_index("Item")
    assert rd.loc["OVERALL", "Status"] == expect
    if expect == "PASS":
        assert "READY" in rd.loc["OVERALL", "Detail"]


def test_atom_index(tmp_path):
    p = tmp_path / "t.pdb"
    p.write_text("\n".join([SB.pdb_atom_line(1, "ATOM", "N", "ALA", "", 1, (0, 0, 0), "N"),
                            SB.pdb_atom_line(2, "ATOM", "CA", "ALA", "", 1, (0, 0, 0), "C"),
                            SB.pdb_atom_line(3, "HETATM", "ZN", "ZN1", "", 2, (0, 0, 0), "Zn"),
                            "TER", SB.pdb_atom_line(4, "HETATM", "O", "WAT", "", 3, (0, 0, 0), "O")]))
    assert SB.atom_index(p) == {(1, "N"): 0, (1, "CA"): 1, (2, "ZN"): 2, (3, "O"): 3}
