"""M7 part 1: DCD reading, file order and region RMSDs (no OpenMM / MDAnalysis needed)."""

import json

import numpy as np

from psma_pipeline import md_analysis as A


def _solute(tmp_path, n_chain=750):
    lines, k = [], 0
    for ch in "AB":
        for r in range(1, n_chain + 1):
            for name in ("N", "CA", "C"):
                k += 1
                lines.append(f"ATOM  {k % 100000:5d}  {name:<3s} ALA {ch}{r:4d}    {0:8.3f}{0:8.3f}{0:8.3f}  1.00  0.00           {name[0]:>2s}")
    for nm, rn, el in (("ZN", "ZN1", "ZN"), ("CA", "CA", "CA")):           # Zn and a crystal Ca2+: not Cα
        k += 1
        lines.append(f"HETATM{k:5d} {nm:<4s}{rn:>3s} A 900    {0:8.3f}{0:8.3f}{0:8.3f}  1.00  0.00          {el:>2s}")
    p = tmp_path / "solute.pdb"
    p.write_text("\n".join(lines) + "\nEND\n")
    return p, k


def test_dcd_roundtrip_and_truncation(tmp_path):
    rng = np.random.default_rng(0)
    fr = rng.normal(size=(3, 5, 3)).astype(np.float32)
    A.write_dcd(tmp_path / "a.dcd", fr)
    got = list(A.read_dcd(tmp_path / "a.dcd"))
    assert len(got) == 3 and np.allclose(got[1], fr[1])
    assert np.allclose(next(A.read_dcd(tmp_path / "a.dcd", np.array([4, 0])))[0], fr[0][4])
    b = (tmp_path / "a.dcd").read_bytes()
    (tmp_path / "t.dcd").write_bytes(b[:-20])                                  # interrupted while writing
    assert len(list(A.read_dcd(tmp_path / "t.dcd"))) == 2
    (tmp_path / "z.dcd").write_bytes(b"")
    assert list(A.read_dcd(tmp_path / "z.dcd")) == []


def test_ca_table_and_segment_order(tmp_path):
    pdb, natoms = _solute(tmp_path)
    ca = A.ca_table(pdb)
    assert len(ca) == 1500 and ca["chain"].tolist()[749:751] == ["A", "B"] and ca["uniprot"].iloc[750] == 1
    md = tmp_path / "md"
    (md / "traj").mkdir(parents=True)
    done = ["eq:6.6:20", "prod:production:1", "prod:production:2", "prod:t2000", "prod:t2250"]
    (md / "progress.json").write_text(json.dumps({"done": done}))
    for sid in done + ["prod:production:3"]:                              # 3: interrupted, not in done
        (md / "traj" / f"{sid.replace(':', '_')}.dcd").write_bytes(b"x")
    files = A.segment_files(md)
    assert [t for t, _ in files] == [0.0, 1000.0, 2000.0, 2250.0]
    assert all("production_3" not in f.name for _, f in files)


def test_stability_rigid_motion_gives_zero(tmp_path):
    pdb, natoms = _solute(tmp_path)
    ca = A.ca_table(pdb)
    rng = np.random.default_rng(1)
    base = rng.normal(scale=20, size=(natoms, 3))
    th = 0.4
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    moved = base @ R.T + 5.0
    tail = moved.copy()
    sel = ca[(ca["uniprot"] <= 19)]["atom"].to_numpy()
    tail[sel] += 3.0                                                           # tails move 3 Å relative to the rest
    md = tmp_path / "md"
    (md / "traj").mkdir(parents=True)
    (md / "progress.json").write_text(json.dumps({"done": ["prod:t0"]}))
    A.write_dcd(md / "traj" / "prod_t0.dcd", np.stack([base, moved, tail]).astype(np.float32))
    df, rf = A.stability(md, ca)
    assert len(df) == 3 and abs(df["time (ns)"].iloc[0] - 0.01) < 1e-9
    assert df.iloc[1].drop("time (ns)").max() < 1e-3                           # rigid motion: all ~0
    r = df.iloc[2]
    assert abs(r["cytoplasmic tail (1–19), both chains"] - np.sqrt(27)) < 1e-3 and r["ectodomain, crystal (62–750), both chains"] < 1e-3
    assert rf.loc[(rf["uniprot"] == 5), "RMSF (Å)"].iloc[0] > 2 and rf.loc[(rf["uniprot"] == 300), "RMSF (Å)"].max() < 1e-3
    s = A.summary(df)
    assert "drift over the run (Å)" in s.columns


# ---------------------------------------------------------------------------
# part 2: orientation, lid, zinc site
# ---------------------------------------------------------------------------
def _site_system(tmp_path):
    """Two chains with known geometry: TM helices tilted ±20° in xz, ectodomain above (+z), a helical lid, and a
    Zn pair per chain with His377 NE2, Glu425 OE1/OE2 and a hydroxide as ligands."""
    rng = np.random.default_rng(3)
    atoms, xyz = [], []

    def add(name, resn, ch, r, el, p):
        atoms.append((name, resn, ch, r, el))
        xyz.append(np.asarray(p, float))

    zc = {"A": np.array([150.0, 0, 60]), "B": np.array([-150.0, 0, 60])}
    for ch, sx in (("A", 1), ("B", -1)):
        tilt = np.radians(20 * sx)
        u = np.array([np.sin(tilt), 0, np.cos(tilt)])
        for r in range(1, 751):
            if 20 <= r <= 43:
                ca = np.array([-10.0 * sx, 0, -18]) + (r - 20) * 1.5 * u
            elif 541 <= r <= 548:
                ca = np.array([0, 30.0 * sx, 60]) + (r - 541) * np.array([1.5, 0, 0])
            elif r < 20:
                ca = np.array([-10.0 * sx, 0, -30]) + rng.normal(scale=3, size=3)
            else:
                ca = np.array([20.0 * sx, 0, 60]) + rng.normal(scale=15, size=3)
            resn = {377: "HIS", 425: "GLU", 541: "TRP"}.get(r, "ALA")
            add("N", resn, ch, r, "N", ca + [0, 0, 400])                   # N/O far from the zinc
            add("CA", resn, ch, r, "C", ca)
            add("C", resn, ch, r, "C", ca + [0, 0, 401])
            if r == 377:
                add("NE2", resn, ch, r, "N", zc[ch] + [-2.1, 0, 0])
            if r == 425:
                add("OE1", resn, ch, r, "O", zc[ch] + [3.3, 2.0, 0])
                add("OE2", resn, ch, r, "O", zc[ch] + [3.3, -2.0, 0])
            if r == 541:
                for k, nm in enumerate(("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2")):
                    add(nm, resn, ch, r, "C", zc[ch] + [0, 10 + 0.1 * k, 0])
    for ch in "AB":
        add("ZN", "ZN1", ch, 900, "ZN", zc[ch])
        add("ZN", "ZN2", ch, 901, "ZN", zc[ch] + [3.3, 0, 0])
        add("O", "OH", ch, 902, "O", zc[ch] + [1.65, 0, 1.5])
    lines = [f"ATOM  {i + 1:5d} {nm:<4s} {rn:>3s} {ch}{r:4d}    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00          {el:>2s}"
             for i, ((nm, rn, ch, r, el), p) in enumerate(zip(atoms, xyz))]
    pdb = tmp_path / "solute.pdb"
    pdb.write_text("\n".join(lines) + "\nEND\n")
    return pdb, np.array(xyz)


def test_site_and_shape_known_geometry(tmp_path):
    pdb, x0 = _site_system(tmp_path)
    at = A.atom_table(pdb)
    assert at.loc[at["element"] == "ZN", "chain"].eq("").all() and at["uniprot"].max() == 750
    lig = A.zinc_site(at, x0)
    a = lig[lig["Zn"] == "A:Zn1"]
    assert set(a["ligand"]) >= {"A:HIS377 NE2", "A:Zn2"} and a["ligand"].str.startswith("OH O (atom").any()
    assert set(lig.loc[lig["Zn"] == "A:Zn2", "ligand"]) >= {"A:GLU425 OE1", "A:GLU425 OE2"}
    x1 = x0.copy()
    lidA = A._ca(at, "A", 541, 548)
    x1[lidA] = x0[lidA[0]] + np.arange(8)[:, None] * np.array([3.8, 0, 0])   # extended lid: open
    ne2 = at.index[(at["chain"] == "A") & (at["uniprot"] == 377) & (at["name"] == "NE2")][0]
    x1[ne2] += [-1.0, 0, 0]                                                   # His377 leaves Zn1 by 1 Å
    md = tmp_path / "md"
    (md / "traj").mkdir(parents=True)
    (md / "progress.json").write_text(json.dumps({"done": ["prod:t0"]}))
    A.write_dcd(md / "traj" / "prod_t0.dcd", np.stack([x0, x1]).astype(np.float32))
    ref = {int(at.loc[i, "uniprot"]): x0[i] @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).T + 7
           for i in A._ca(at, "A", 62, 750)}                                  # chain A start, rotated and moved
    lid_cfg = {"range": (541, 548), "classifier": {"helix_ca_i3_max": 6.0, "helix_fraction_min": 0.6}}
    out = A.site_and_shape(md, at, lid_cfg, {"REF": ("closed", ref)})
    o = out["orientation"].iloc[0]
    assert abs(o["TM helix tilt, chain A (°)"] - 20) < 0.5 and abs(o["TM helix tilt, chain B (°)"] - 20) < 0.5
    assert abs(o["angle between TM helices (°)"] - 40) < 0.5
    assert o["ectodomain height above the TM pair (Å)"] > 50 and o["ectodomain tilt from the membrane normal (°)"] < 10
    lid = out["lid"]
    assert lid["lid helicity, chain A"].tolist() == [1.0, 0.0]
    assert lid["lid state by helicity, chain A"].tolist() == ["closed", "open"]
    assert lid["lid RMSD to REF (closed), chain A (Å)"].iloc[0] < 1e-3 and lid["lid RMSD to REF (closed), chain A (Å)"].iloc[1] > 5
    assert "lid RMSD to REF (closed), chain B (Å)" in lid.columns and "Trp541 ring to zinc pair, chain A (Å)" in lid.columns
    zn = A.zinc_summary(out["zinc"], out["zinc ligands"])
    h = zn[(zn["Zn"] == "A:Zn1") & (zn["ligand"] == "A:HIS377 NE2")].iloc[0]
    assert abs(h["start (Å)"] - 2.1) < 0.01 and abs(h["max (Å)"] - 3.1) < 0.01 and h["frames > start + 0.5 Å (%)"] == 50.0
    s = A.lid_state_summary(lid)
    assert s.set_index("Criterion").loc["lid state by helicity, chain A", "open (% of frames)"] == 50.0
    assert "SD" in A.describe(out["orientation"]).columns
