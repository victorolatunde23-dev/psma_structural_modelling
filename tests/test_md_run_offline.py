"""Equilibration / production helpers (md_run.py) that do not need OpenMM."""

import numpy as np
import pandas as pd

from psma_pipeline import md_run as M


def test_plan_segments_cover_each_stage_exactly():
    stages = [{"name": "1 NVT", "ensemble": "NVT", "ps": 125, "dt_fs": 1, "bb": 10, "sc": 5, "lip": 2.5},
              {"name": "4 NPT", "ensemble": "NPT", "ps": 500, "dt_fs": 2, "bb": 1, "sc": 0.5, "lip": 0.5}]
    plan = M.plan_segments(stages, 25)
    assert [p["id"] for p in plan[:2]] == ["eq:1 NVT:1", "eq:1 NVT:2"]
    assert sum(p["steps"] for p in plan if p["stage"] == "1 NVT") == 125_000
    assert sum(p["steps"] for p in plan if p["stage"] == "4 NPT") == 250_000
    assert len(plan) == 5 + 20 and abs(sum(p["ps"] for p in plan) - 625) < 1e-9
    assert len({p["id"] for p in plan}) == len(plan)                       # resume keys are unique
    prod = M.plan_segments([{"name": "production", "ps": 2500, "dt_fs": 4}], 1000, "prod")
    assert [p["ps"] for p in prod] == [1000, 1000, 500]


def test_default_protocol_from_config(cfg):
    st = cfg.get("md.equilibration.stages")
    assert [s["ensemble"] for s in st] == ["NVT", "NVT", "NPT", "NPT", "NPT", "NPT"]
    assert st[-1]["bb"] == 0.1 and st[-1]["sc"] == 0.0 and abs(sum(s["ps"] for s in st) - 1875) < 1e-9


def test_classify_atoms():
    rows = [(0, "ALA", "N", "N", True), (1, "ALA", "CA", "C", True), (2, "ALA", "CB", "C", True),
            (3, "ALA", "HA", "H", True), (4, "GU1", "CA", "C", True), (5, "GU1", "OE1", "O", True),
            (6, "ZN1", "ZN", "Zn", False), (7, "OX1", "O", "O", False), (8, "OX1", "H", "H", False),
            (9, "0YB", "C1", "C", False), (10, "PC", "P31", "P", False), (11, "PA", "C12", "C", False),
            (12, "CHL", "O1", "O", False), (13, "HOH", "O", "O", False), (14, "Na+", "Na+", "Na", False),
            (15, "CA", "CA", "Ca", False)]
    g = M.classify_atoms(rows)
    assert g["bb"] == [0, 1, 4] and g["ca"] == [1, 4]
    assert g["sc"] == [2, 5, 6, 7, 9, 15]                  # side chains, Zn, hydroxide O, glycan, crystal Ca
    assert g["heads"] == [10, 12]
    assert 3 in g["solute"] and 8 in g["solute"] and 13 not in g["solute"] and 14 not in g["solute"] and 10 not in g["solute"]


def test_kabsch_and_area_per_lipid():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(20, 3))
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    assert M.kabsch_rmsd(P @ R.T + 3.0, P) < 1e-9
    apl, up, lo = M.area_per_lipid(170.0 * 170.0 / 100.0, np.r_[np.full(488, 12.0), np.full(488, 8.0)], 10.0)
    assert (up, lo) == (488, 488) and abs(apl - 170.0 * 170.0 / 488) < 1e-9


def test_equilibration_checks():
    def log(T, dev):
        return pd.DataFrame({"Stage": ["5 NPT", "6 NPT", "6 NPT"], "Potential energy (kJ/mol)": [-6e6, -6e6, -6e6],
                             "Temperature (K)": [310.2, T, T], "Box x (Å)": [171.0, 171.0, 171.2],
                             "Box y (Å)": [171.0, 171.0, 171.2], "Area per lipid (Å²)": [60.0, 60.0, 60.1],
                             "Max Zn–ligand deviation (Å)": [0.1, dev, 0.12], "Cα RMSD from start (Å)": [1.0, 1.2, 1.3]})
    import psma_pipeline
    cfg = psma_pipeline.load_config()
    c = M.equilibration_checks(log(310.4, 0.15), cfg).set_index("Check")["Status"]
    assert (c != "FAIL").all()
    c = M.equilibration_checks(log(318.0, 0.45), cfg).set_index("Check")["Status"]
    assert c["temperature"] == "FAIL" and c["zn_site"] == "FAIL"


def test_lipid_torsions_found_from_bond_graph():
    """Glycerol C2 (1 H, 2 C, 1 O in a head residue) and the oleoyl CH=CH are found without atom-name tables."""
    # PC head: C1(0)-C2(1)-C3(2); C2-O21(3); C2-H(4); C1 Hs (5,6); C3 Hs (7,8)
    # OL tail: C8(10)-C9(11)=C10(12)-C11(13); H on C9 (14), C10 (15); CH2 Hs on C8/C11 (16-19)
    atoms = [(0, "PC", "C", 0), (1, "PC", "C", 0), (2, "PC", "C", 0), (3, "PC", "O", 0), (4, "PC", "H", 0),
             (5, "PC", "H", 0), (6, "PC", "H", 0), (7, "PC", "H", 0), (8, "PC", "H", 0),
             (10, "OL", "C", 1), (11, "OL", "C", 1), (12, "OL", "C", 1), (13, "OL", "C", 1), (14, "OL", "H", 1),
             (15, "OL", "H", 1), (16, "OL", "H", 1), (17, "OL", "H", 1), (18, "OL", "H", 1), (19, "OL", "H", 1)]
    bonds = [(0, 1), (1, 2), (1, 3), (1, 4), (0, 5), (0, 6), (2, 7), (2, 8),
             (10, 11), (11, 12), (12, 13), (11, 14), (12, 15), (10, 16), (10, 17), (13, 18), (13, 19)]
    x = np.zeros((20, 3))
    x[[0, 1, 2, 3]] = [[0, 0, 0], [0.15, 0, 0], [0.2, 0.14, 0], [0.2, -0.05, 0.13]]
    x[[10, 11, 12, 13]] = [[0, 0, 1], [0.15, 0, 1], [0.22, 0.13, 1], [0.12, 0.25, 1]]
    t = M.lipid_torsions(atoms, bonds, x)
    chir = [r for r in t if r[2] == 1]
    cis = [r for r in t if r[1] == 11 and r[2] == 12]
    assert len(t) == 2 and chir and cis
    assert chir[0][:4] == (0, 2, 1, 3) and abs(chir[0][5] - np.deg2rad(2.5)) < 1e-12
    assert cis[0][:4] == (10, 11, 12, 13) and cis[0][4] == 0.0 and cis[0][5] == 0.0


def test_charmm_gui_stage_parser(tmp_path):
    d = tmp_path / "charmm_gui" / "openmm"
    d.mkdir(parents=True)
    (d / "step6.1_equilibration.inp").write_text(
        "nstep  = 125000   # number of steps\ndt     = 0.001    # ps\npcouple = no\n"
        "fc_bb  = 4184.0\nfc_sc  = 2092.0\nfc_mpos = 1046.0\nfc_mdih = 1000.0\n")
    (d / "step6.4_equilibration.inp").write_text("nstep = 250000\ndt = 0.002\npcouple = yes\nfc_bb = 418.4\n")
    st = M.charmm_gui_stages(tmp_path / "charmm_gui")
    assert [s["name"] for s in st] == ["6.1", "6.4"] and [s["ensemble"] for s in st] == ["NVT", "NPT"]
    assert abs(st[0]["ps"] - 125) < 1e-9 and st[0]["dt_fs"] == 1.0 and abs(st[1]["ps"] - 500) < 1e-9
    assert abs(st[0]["bb"] - 10) < 1e-9 and abs(st[0]["sc"] - 5) < 1e-9 and abs(st[0]["lip"] - 2.5) < 1e-9
    assert abs(st[0]["tor"] - 1000 / 4.184) < 1e-9 and st[0]["wat"] == 0.0 and st[0]["ion"] == 0.0
    assert st[1]["sc"] == 0.0
    assert M.charmm_gui_stages(tmp_path / "missing") is None


def test_find_charmm_gui_unpacks_openmm_inputs(tmp_path):
    import io
    import tarfile
    d = tmp_path / "CHARMM-GUI"
    d.mkdir()
    with tarfile.open(d / "charmm-gui.tgz", "w:gz") as tf:
        for name, text in (("charmm-gui-123/openmm/step6.1_equilibration.inp", "nstep = 10\ndt = 0.001\nfc_bb = 4184\n"),
                           ("charmm-gui-123/openmm/step5_input.psf", "big"), ("charmm-gui-123/step6.1_equilibration.inp", "x")):
            b = text.encode()
            ti = tarfile.TarInfo(name)
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
    found = M.find_charmm_gui(tmp_path)
    assert found == d and [p.name for p in (d / "openmm_inputs").iterdir()] == ["step6.1_equilibration.inp"]
    assert M.charmm_gui_stages(found)[0]["bb"] == 10.0
    (d / "openmm_inputs" / "step6.1_equilibration.inp").write_text("nstep = 10\ndt = 0.001\n")
    assert M.charmm_gui_stages(found) is None                       # unrecognised format: fall back to the config


def test_charmm_gui_lipid_keys_and_unknown_guard(tmp_path):
    """CHARMM-GUI's OpenMM inputs name the lipid restraints fc_lpos / fc_ldih (missed by v0.7.25: 0 was used)."""
    d = tmp_path / "cg" / "openmm"
    d.mkdir(parents=True)
    (d / "step6.1_equilibration.inp").write_text("nstep = 125000\ndt = 0.001\nrest = yes\nfc_bb = 4000.0\n"
                                                 "fc_sc = 2000.0\nfc_lpos = 1000.0\nfc_ldih = 1000.0\nfc_cdih = 1000.0\n")
    st = M.charmm_gui_stages(tmp_path / "cg")[0]
    assert abs(st["lip"] - 1000 / 418.4) < 1e-9 and abs(st["tor"] - 1000 / 4.184) < 1e-9
    assert "fc_lpos=1000.0" in st["fc settings read"]
    (d / "step6.1_equilibration.inp").write_text("nstep = 1\ndt = 0.001\nfc_bb = 4000\nfc_newthing = 5\n")
    assert M.charmm_gui_stages(tmp_path / "cg") is None


def test_dihedral_sign_matches_iupac_openmm():
    """v0.7.24-28 computed the mirror-image sign, so the glycerol chirality restraint pulled every POPC towards the
    wrong enantiomer (70,000 kJ/mol/nm on PC C2 at the start of the 2026-11-18 run)."""
    rng = np.random.default_rng(1)
    P = rng.normal(size=(4, 3))
    b1, b2, b3 = P[1] - P[0], P[2] - P[1], P[3] - P[2]
    iupac = np.arctan2(np.linalg.norm(b2) * b1 @ np.cross(b2, b3), np.cross(b1, b2) @ np.cross(b2, b3))
    assert abs(M.dihedrals(P, np.array([[0, 1, 2, 3]]))[0] - iupac) < 1e-12
    # a right-handed reference: +90 degrees
    Q = np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]])
    assert abs(np.degrees(M.dihedrals(Q, np.array([[0, 1, 2, 3]]))[0]) - 90.0) < 1e-9
    atoms = [(0, "PC", "C", 0), (1, "PC", "C", 0), (2, "PC", "C", 0), (3, "PC", "O", 0), (4, "PC", "H", 0),
             (5, "PC", "H", 0), (6, "PC", "H", 0), (7, "PC", "H", 0), (8, "PC", "H", 0)]
    bonds = [(0, 1), (1, 2), (1, 3), (1, 4), (0, 5), (0, 6), (2, 7), (2, 8)]
    x = rng.normal(size=(9, 3))
    t0 = M.lipid_torsions(atoms, bonds, x)[0]
    assert abs(t0[4] - M.dihedrals(x, np.array([t0[:4]]))[0]) < 1e-12      # restraint target = the built configuration


def test_equilibration_checks_mixed_stage_types():
    """Rows reloaded from log.csv had Stage 6.6 (float) while new rows had "6.6" (str): the last stage was split."""
    import psma_pipeline
    df = pd.DataFrame({"Stage": [6.5, 6.6, 6.6, "6.6"], "Potential energy (kJ/mol)": [-6e6] * 4,
                       "Temperature (K)": [310.0, 309.0, 311.0, 310.0], "Box x (Å)": [160.0] * 4, "Box y (Å)": [160.0] * 4,
                       "Area per lipid (Å²)": [52.5] * 4, "Max Zn–ligand deviation (Å)": [0.2] * 4,
                       "Cα RMSD from start (Å)": [0.5] * 4})
    c = M.equilibration_checks(df, psma_pipeline.load_config()).set_index("Check")
    assert "area_stable" in c.index and "3 segments" not in c.loc["temperature", "Detail"]


def test_production_plan_by_time_survives_segment_change():
    """Sessions on a T4 kept dying inside 1 ns segments (2026-11-25): production is now tracked by time, so the
    segment length can shrink without losing or repeating the 6 ns already done with 1 ns segments."""
    prog = {"done": ["eq:6.6:20"] + [f"prod:production:{i}" for i in range(1, 7)]}
    t0 = M.prod_done_ps(prog)
    assert t0 == 6000.0
    plan = M.production_plan(200_000.0, 250.0, t0, 4.0)
    assert plan[0]["id"] == "prod:t6000" and plan[0]["steps"] == 62_500 and plan[0]["start_ps"] == 6000.0
    assert len(plan) == (200_000 - 6000) // 250 and abs(sum(p["ps"] for p in plan) - 194_000) < 1e-6
    assert len({p["id"] for p in plan}) == len(plan)
    prog["prod_done_ps"] = 6250.0
    assert M.production_plan(200_000.0, 250.0, M.prod_done_ps(prog), 4.0)[0]["id"] == "prod:t6250"
    assert M.production_plan(1000.0, 250.0, 1000.0, 4.0) == []


STATE = '<?xml version="1.0" ?>\n<State time="{t}" type="State">\n\t<Positions>\n\t\t<Position x="1" y="2" z="3"/>\n\t</Positions>\n</State>\n'


def test_checkpoint_written_in_place_with_backup(tmp_path):
    # a rename (the old temp-file write) gives the file a new inode; on Google Drive it left an orphaned
    # 88 MB copy in the running account's My Drive at every save
    p = tmp_path / "state.xml"
    M._write_checkpoint(p, STATE.format(t=100))
    inode = p.stat().st_ino
    M._write_checkpoint(p, STATE.format(t=200))
    assert p.stat().st_ino == inode                                    # same file, overwritten in place
    assert 'time="200"' in p.read_text()
    assert 'time="100"' in M._backup_path(p).read_text()                # previous complete version kept
    assert sorted(f.name for f in tmp_path.iterdir()) == ["state.xml", "state.xml.bak"]   # no temp files left
    bak_inode = M._backup_path(p).stat().st_ino
    M._write_checkpoint(p, STATE.format(t=300))
    assert M._backup_path(p).stat().st_ino == bak_inode                 # the backup is reused, not recreated


def test_incomplete_checkpoint_falls_back_to_backup(tmp_path, caplog):
    p = tmp_path / "state.xml"
    M._write_checkpoint(p, STATE.format(t=100))
    M._write_checkpoint(p, STATE.format(t=200))
    full = p.read_text()
    p.write_text(full[: len(full) // 2])                                # session ended mid-write
    assert not M._is_complete(p) and M._is_complete(M._backup_path(p))
    assert 'time="100"' in M._read_checkpoint(p)                        # previous state, matches old progress.json
    assert "incomplete" in caplog.text
    # the next save must not copy the broken file over the good backup
    M._write_checkpoint(p, STATE.format(t=300))
    assert 'time="100"' in M._backup_path(p).read_text() and 'time="300"' in M._read_checkpoint(p)


def test_progress_json_checkpoint(tmp_path):
    p = tmp_path / "progress.json"
    M._write_checkpoint(p, '{"done": ["a"]}')
    M._write_checkpoint(p, '{"done": ["a", "b"]}')
    p.write_text('{"done": ["a", "b"')                                  # truncated JSON
    assert M._read_checkpoint(p) == '{"done": ["a"]}'
    p.unlink()                                                          # only the backup survived
    assert M._read_checkpoint(p) == '{"done": ["a"]}'
    assert not M._is_complete(tmp_path / "missing.json")
