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
