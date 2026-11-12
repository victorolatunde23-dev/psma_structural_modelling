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
