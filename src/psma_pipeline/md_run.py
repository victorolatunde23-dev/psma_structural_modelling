"""Milestone 7a: equilibration and production MD of the built membrane system (notebook 06).

Starts from what notebook 05 left in ``outputs/md_setup/system/``:

- ``system.xml``: the OpenMM system, with the hydroxide-H LJ core and the Zn r0 correction already applied
  (they are not in the prmtop);
- ``system.prmtop``: topology only;
- ``system_minimised.pdb``: the minimised coordinates;
- ``zn_r0_correction.json``: the Zn–ligand pairs and their crystal r0 values, for monitoring.

Everything is resumable. A Colab session can end at any time; the state (positions, velocities, box) is saved to
Drive after every segment, and a re-run continues from the last finished segment. ``max_minutes`` stops cleanly
before a segment that would not finish in time.

Outputs (``outputs/md_run/``):

- ``state.xml``: latest state; ``progress.json``: finished segments;
- ``log.csv``: one row per segment (temperature, box, area per lipid, Zn distances, Cα RMSD, ns/day);
- ``solute.pdb`` + ``traj/<phase>_<segment>.dcd``: solute frames (protein, Zn site, glycans);
- ``equilibrated.pdb``: full system at the end of equilibration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .status import IssueLog, Status
from .system_build import IONS, LIPID_HEADS, LIPID_TAILS, STEROLS, distances, make_simulation

log = logging.getLogger(__name__)

WATER = {"HOH", "WAT"}
KCAL_A2_TO_KJ_NM2 = 418.4
LIPID_HEAD_ATOMS = {("PC", "P31"), ("PE", "P31"), ("PS", "P31"), ("PGR", "P31"), ("CHL", "O1")}


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def plan_segments(stages: list[dict], segment_ps: float, phase: str = "eq") -> list[dict]:
    """Split each stage into segments of at most ``segment_ps``; every segment is a resume point."""
    out = []
    for s in stages:
        dt = float(s["dt_fs"]) / 1000.0
        total = int(round(float(s["ps"]) / dt))
        per = max(1, int(round(float(segment_ps) / dt)))
        n, done = 0, 0
        while done < total:
            steps = min(per, total - done)
            n += 1
            out.append({"id": f"{phase}:{s['name']}:{n}", "phase": phase, "stage": s["name"], "segment": n,
                        "steps": steps, "dt_fs": float(s["dt_fs"]), "ensemble": s.get("ensemble", "NPT"),
                        "bb": float(s.get("bb", 0)), "sc": float(s.get("sc", 0)), "lip": float(s.get("lip", 0)),
                        "ps": steps * dt})
            done += steps
    return out


def production_stages(cfg: Any) -> list[dict]:
    p = cfg.get("md.production", {}) or {}
    return [{"name": "production", "ensemble": "NPT", "ps": float(p.get("ns", 100)) * 1000.0,
             "dt_fs": float(p.get("dt_fs", 4)), "bb": 0.0, "sc": 0.0, "lip": 0.0}]


# ---------------------------------------------------------------------------
# atoms and analysis helpers (no OpenMM needed)
# ---------------------------------------------------------------------------
def groups_from_topology(topology) -> dict[str, list[int]]:
    """Index groups from an OpenMM topology.

    bb: backbone N/CA/C/O of residues with N and CA (includes MCPB-renamed residues such as HD1, AP1, GU1);
    sc: other heavy solute atoms (side chains, Zn, hydroxide O, glycans, crystal Ca/Cl); heads: lipid head atoms
    (z restraint, area per lipid); ca: Cα; solute: everything except water, lipids and bulk Na+/Cl-."""
    rows = []
    for res in topology.residues():
        names = {a.name for a in res.atoms()}
        has_ca = "CA" in names and "N" in names
        for a in res.atoms():
            rows.append((a.index, res.name, a.name, a.element.symbol if a.element is not None else "", has_ca))
    return classify_atoms(rows)


def classify_atoms(rows: list[tuple[int, str, str, str, bool]]) -> dict[str, list[int]]:
    """rows: (index, residue name, atom name, element, residue has backbone N+CA)."""
    lipid = LIPID_TAILS | LIPID_HEADS | STEROLS
    g = {"bb": [], "sc": [], "heads": [], "ca": [], "solute": []}
    for i, rn, an, el, has_ca in rows:
        if rn in WATER:
            continue
        if rn in lipid:
            if (rn, an) in LIPID_HEAD_ATOMS:
                g["heads"].append(i)
            continue
        if rn in IONS:                 # bulk Na+/Cl- (crystal Ca2+ and its Cl- have other names; kept as solute)
            continue
        g["solute"].append(i)
        if el == "H":
            continue
        if has_ca and an in ("N", "CA", "C", "O"):
            g["bb"].append(i)
            if an == "CA":
                g["ca"].append(i)
        else:
            g["sc"].append(i)
    return g


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """RMSD of P onto Q after optimal superposition (same units as the input)."""
    p, q = P - P.mean(0), Q - Q.mean(0)
    H = p.T @ q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.sqrt(np.mean(np.sum((p @ R.T - q) ** 2, axis=1))))


def area_per_lipid(box_xy_nm2: float, head_z: np.ndarray, z_mid: float) -> tuple[float, int, int]:
    """Å² per lipid (PC + cholesterol, the mean of the two leaflets) and the leaflet counts."""
    upper = int(np.sum(head_z > z_mid))
    lower = int(np.sum(head_z <= z_mid))
    n = (upper + lower) / 2.0
    return (100.0 * box_xy_nm2 / n if n else float("nan")), upper, lower


def apply_hmr(system, topology, h_mass_amu: float, water=WATER) -> int:
    """Hydrogen mass repartitioning for solute/lipid H (water untouched): the H gets ``h_mass_amu`` and its bonded
    heavy atom loses the difference, so every molecule keeps its mass. Returns the number of H changed."""
    import openmm.unit as u
    n = 0
    for b in topology.bonds():
        a1, a2 = b[0], b[1]
        if a1.residue.name in water:
            continue
        for h, x in ((a1, a2), (a2, a1)):
            if h.element is not None and h.element.symbol == "H" and (x.element is None or x.element.symbol != "H"):
                mh = system.getParticleMass(h.index).value_in_unit(u.dalton)
                if mh <= 0 or abs(mh - h_mass_amu) < 1e-6:
                    continue
                dm = h_mass_amu - mh
                mx = system.getParticleMass(x.index).value_in_unit(u.dalton)
                if mx - dm <= 1.0:
                    continue
                system.setParticleMass(h.index, h_mass_amu * u.dalton)
                system.setParticleMass(x.index, (mx - dm) * u.dalton)
                n += 1
    return n


def strip_forces(system) -> list[str]:
    """Remove notebook 05's restraint force (global k_rest) and any barostat from a loaded system."""
    import openmm as mm
    removed = []
    for k in reversed(range(system.getNumForces())):
        f = system.getForce(k)
        is_rest = isinstance(f, mm.CustomExternalForce) and any(
            f.getGlobalParameterName(j) == "k_rest" for j in range(f.getNumGlobalParameters()))
        if is_rest or isinstance(f, (mm.MonteCarloBarostat, mm.MonteCarloMembraneBarostat,
                                     mm.MonteCarloAnisotropicBarostat)):
            removed.append(type(f).__name__)
            system.removeForce(k)
    return removed


def add_restraints(system, x0_nm: np.ndarray, groups: dict[str, list[int]]) -> None:
    """Positional restraints (backbone, side chains/metal site/glycans) and a z restraint on lipid heads, with
    global strengths kbb, ksc, klip (kJ/mol/nm²) set per stage."""
    import openmm as mm
    pos = mm.CustomExternalForce("0.5*(isbb*kbb + (1-isbb)*ksc)*periodicdistance(x, y, z, x0, y0, z0)^2")
    for n in ("kbb", "ksc"):
        pos.addGlobalParameter(n, 0.0)
    for n in ("isbb", "x0", "y0", "z0"):
        pos.addPerParticleParameter(n)
    for i in groups["bb"]:
        pos.addParticle(i, [1.0, *x0_nm[i].tolist()])
    for i in groups["sc"]:
        pos.addParticle(i, [0.0, *x0_nm[i].tolist()])
    system.addForce(pos)
    zf = mm.CustomExternalForce("0.5*klip*(z-z0)^2")
    zf.addGlobalParameter("klip", 0.0)
    zf.addPerParticleParameter("z0")
    for i in groups["heads"]:
        zf.addParticle(i, [float(x0_nm[i, 2])])
    system.addForce(zf)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _subset_topology(topology, idx: list[int]):
    import openmm.app as app
    keep = set(idx)
    top = app.Topology()
    for ch in topology.chains():
        c = None
        for res in ch.residues():
            r = None
            for a in res.atoms():
                if a.index in keep:
                    c = c or top.addChain(ch.id)
                    r = r or top.addResidue(res.name, c, res.id)
                    top.addAtom(a.name, a.element, r)
    return top


def run_md(cfg: Any, phase: str = "eq", max_minutes: float | None = None, issues: IssueLog | None = None) -> dict:
    """Run (or continue) equilibration (``phase='eq'``) or production (``'prod'``). Returns tables and status."""
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u

    issues = issues or IssueLog()
    t_start = time.time()
    sysdir = cfg.outputs_dir / "md_setup" / "system"
    out = cfg.outputs_dir / "md_run"
    (out / "traj").mkdir(parents=True, exist_ok=True)
    need = [sysdir / "system.xml", sysdir / "system.prmtop", sysdir / "system_minimised.pdb"]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        raise FileNotFoundError("run notebook 05 to the end first (MDReadiness READY); missing: " + ", ".join(missing))
    eqc = cfg.get("md.equilibration", {}) or {}
    pc = cfg.get("md.production", {}) or {}
    T = float(cfg.get("md.openmm.temperature_k", 310))
    stages = list(eqc.get("stages") or [])
    plan = plan_segments(stages, float(eqc.get("segment_ps", 25)), "eq")
    if phase == "prod":
        plan = plan_segments(production_stages(cfg), float(pc.get("segment_ps", 1000)), "prod")
    frame_ps = float((pc if phase == "prod" else eqc).get("frame_ps", 10 if phase == "prod" else 5))

    # ---- progress / resume ------------------------------------------------------------------------------
    key = f"{_sha(sysdir / 'system.xml')}:{_sha(sysdir / 'system_minimised.pdb')}"
    prog_file, state_file = out / "progress.json", out / "state.xml"
    prog = json.loads(prog_file.read_text()) if prog_file.exists() else {}
    if prog.get("key") != key:
        if prog:
            issues.add("md", Status.WARNING, "system changed since the last MD run: starting again from the minimised system")
        prog = {"key": key, "done": [], "eq_done": False}
        for f in (state_file, out / "log.csv"):
            f.unlink(missing_ok=True)
    if phase == "prod" and not prog.get("eq_done"):
        raise RuntimeError("equilibration has not finished (or did not pass): run section E first")
    todo = [s for s in plan if s["id"] not in set(prog["done"])]

    # ---- system ----------------------------------------------------------------------------------------
    prm = app.AmberPrmtopFile(str(sysdir / "system.prmtop"))
    top = prm.topology
    base = mm.XmlSerializer.deserialize((sysdir / "system.xml").read_text())
    removed = strip_forces(base)
    n_hmr = apply_hmr(base, top, float(eqc.get("hmr_mass", 3.024)))
    pdb = app.PDBFile(str(sysdir / "system_minimised.pdb"))
    x0 = np.array(pdb.getPositions(asNumpy=True).value_in_unit(u.nanometer))
    groups = groups_from_topology(top)
    if not groups["heads"]:
        issues.add("md", Status.WARNING, "no lipid head atoms (PC P31 / CHL O1) found: no lipid restraint, no area per lipid")
    add_restraints(base, x0, groups)
    box0 = prm.topology.getPeriodicBoxVectors() or app.AmberInpcrdFile(str(sysdir / "system.inpcrd")).boxVectors
    zinfo = json.loads((sysdir / "zn_r0_correction.json").read_text()) if (sysdir / "zn_r0_correction.json").exists() else {}
    zn_pairs = [tuple(p) for p in zinfo.get("pairs", [])]
    zn_ref = np.array(zinfo.get("crystal r0 (Å)", []), dtype=float)
    z_mid = float(np.mean(x0[groups["heads"], 2])) if groups["heads"] else 0.0
    ca = groups["ca"]
    solute = groups["solute"]
    if not (out / "solute.pdb").exists():
        with open(out / "solute.pdb", "w") as fh:
            app.PDBFile.writeFile(_subset_topology(top, solute), x0[solute] * u.nanometer, fh, keepIds=True)
    sub_top = _subset_topology(top, solute)
    info = {"atoms": base.getNumParticles(), "forces removed from system.xml": ", ".join(removed) or "none",
            "H with repartitioned mass": n_hmr, "backbone restrained": len(groups["bb"]),
            "other solute heavy restrained": len(groups["sc"]), "lipid heads (z restraint)": len(groups["heads"]),
            "solute atoms in trajectory": len(solute), "segments in plan": len(plan),
            "segments already done": len(plan) - len(todo)}

    rows = pd.read_csv(out / "log.csv").to_dict("records") if (out / "log.csv").exists() else []
    status, platform_name, cur_stage, sim = "complete", None, None, None
    budget = (max_minutes * 60.0) if max_minutes else None
    seg_secs: list[float] = []
    for seg in todo:
        if budget is not None and seg_secs and (time.time() - t_start) + 1.3 * np.mean(seg_secs[-3:]) > budget:
            status = "paused (time budget)"
            break
        t0 = time.time()
        if seg["stage"] != cur_stage:                                   # a new stage: a fresh context
            system = mm.XmlSerializer.clone(base)
            if seg["ensemble"] == "NPT":
                system.addForce(mm.MonteCarloMembraneBarostat(
                    float(eqc.get("pressure_bar", 1.0)) * u.bar,
                    float(eqc.get("surface_tension", 0.0)) * u.bar * u.nanometer, T * u.kelvin,
                    mm.MonteCarloMembraneBarostat.XYIsotropic, mm.MonteCarloMembraneBarostat.ZFree))
            dt = seg["dt_fs"] / 1000.0
            sim, integ, platform_name, fails = make_simulation(
                top, system, lambda: mm.LangevinMiddleIntegrator(T * u.kelvin, float(eqc.get("friction_per_ps", 1.0)) / u.picosecond,
                                                                 dt * u.picoseconds),
                x0 * u.nanometer, box0)
            if fails:
                info["platforms that failed"] = "; ".join(fails)
            if state_file.exists():
                st = mm.XmlSerializer.deserialize(state_file.read_text())
                sim.context.setPeriodicBoxVectors(*st.getPeriodicBoxVectors())
                sim.context.setPositions(st.getPositions())
                sim.context.setVelocities(st.getVelocities())
                sim.context.setTime(st.getTime())
            else:
                sim.context.setVelocitiesToTemperature(T * u.kelvin)
            for n, v in (("kbb", seg["bb"]), ("ksc", seg["sc"]), ("klip", seg["lip"])):
                sim.context.setParameter(n, v * KCAL_A2_TO_KJ_NM2)
            cur_stage = seg["stage"]
            log.info("%s: %s on %s (dt %.0f fs, restraints bb %.1f / sc %.1f / lipid %.1f kcal/mol/Å²)",
                     phase, cur_stage, platform_name, seg["dt_fs"], seg["bb"], seg["sc"], seg["lip"])
        # ---- the segment, with solute frames -----------------------------------------------------------
        fsteps = max(1, int(round(frame_ps / (seg["dt_fs"] / 1000.0))))
        dcd_path = out / "traj" / f"{seg['id'].replace(':', '_').replace(' ', '')}.dcd"
        temps = []
        with open(dcd_path, "wb") as fh:
            dcd = app.DCDFile(fh, sub_top, seg["dt_fs"] / 1000.0 * fsteps * u.picoseconds)
            done = 0
            while done < seg["steps"]:
                n = min(fsteps, seg["steps"] - done)
                sim.step(n)
                done += n
                st = sim.context.getState(getPositions=True, getEnergy=True)
                x = st.getPositions(asNumpy=True).value_in_unit(u.nanometer)
                dcd.writeModel(x[solute] * u.nanometer, periodicBoxVectors=st.getPeriodicBoxVectors())
                dof = 3 * system.getNumParticles() - system.getNumConstraints() - 3
                temps.append(2 * st.getKineticEnergy().value_in_unit(u.kilojoule_per_mole) / (dof * 0.0083144626))
        st = sim.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        e = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        x = st.getPositions(asNumpy=True).value_in_unit(u.nanometer)
        if not np.isfinite(e) or not np.isfinite(x).all():
            issues.add("md", Status.FAIL, f"{seg['id']}: energy {e} (not finite): the run stopped; the last good state is kept")
            status = "failed"
            break
        _atomic_write(state_file, mm.XmlSerializer.serialize(st))
        box = np.diag(st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer))
        apl, up, lo = area_per_lipid(box[0] * box[1], x[groups["heads"], 2], z_mid) if groups["heads"] else (np.nan, 0, 0)
        d = distances(x, zn_pairs) if zn_pairs else np.array([])
        secs = time.time() - t0
        seg_secs.append(secs)
        rows.append({"Phase": phase, "Stage": seg["stage"], "Segment": seg["segment"], "Time (ps)": round(st.getTime().value_in_unit(u.picosecond), 1),
                     "Platform": platform_name, "Potential energy (kJ/mol)": round(e, 1),
                     "Temperature (K)": round(float(np.mean(temps)), 2), "Box x (Å)": round(10 * box[0], 2),
                     "Box y (Å)": round(10 * box[1], 2), "Box z (Å)": round(10 * box[2], 2),
                     "Area per lipid (Å²)": round(apl, 2), "Lipids upper/lower": f"{up}/{lo}",
                     "Max Zn–ligand deviation (Å)": round(float(np.max(np.abs(d - zn_ref))), 3) if len(d) and len(d) == len(zn_ref) else None,
                     "Zn–ligand distances (Å)": ", ".join(f"{v:.2f}" for v in d),
                     "Cα RMSD from start (Å)": round(10 * kabsch_rmsd(x[ca], x0[ca]), 3) if ca else None,
                     "ns/day": round(seg["ps"] / 1000.0 / (secs / 86400.0), 2)})
        pd.DataFrame(rows).to_csv(out / "log.csv", index=False)
        prog["done"].append(seg["id"])
        _atomic_write(prog_file, json.dumps(prog, indent=1))
        log.info("%s done: T %.1f K, APL %.1f Å², Zn dev %s Å, %.1f ns/day", seg["id"], rows[-1]["Temperature (K)"], apl,
                 rows[-1]["Max Zn–ligand deviation (Å)"], rows[-1]["ns/day"])
    remaining = [s for s in plan if s["id"] not in set(prog["done"])]
    if remaining and status == "complete":
        status = "paused"
    df = pd.DataFrame(rows)
    tables = {"MDLog": df, "MDInfo": pd.DataFrame([{"Item": k, "Value": v} for k, v in info.items()])}
    rate = df["ns/day"].iloc[-3:].mean() if len(df) else float("nan")
    left_ns = sum(s["ps"] for s in remaining) / 1000.0
    tables["MDProgress"] = pd.DataFrame([{"Item": k, "Value": v} for k, v in {
        "phase": phase, "status": status, "segments done": len(plan) - len(remaining), "segments total": len(plan),
        "simulated time left (ns)": round(left_ns, 3), "recent speed (ns/day)": round(float(rate), 2) if rate == rate else None,
        "estimated GPU hours left": round(left_ns / rate * 24, 1) if rate and rate == rate else None}.items()])
    if phase == "eq" and not remaining and status == "complete":
        checks = equilibration_checks(df[df["Phase"] == "eq"], cfg)
        tables["EquilibrationChecks"] = checks
        ok = (checks["Status"] != "FAIL").all()
        prog["eq_done"] = bool(ok)
        _atomic_write(prog_file, json.dumps(prog, indent=1))
        st = mm.XmlSerializer.deserialize(state_file.read_text())
        with open(out / "equilibrated.pdb", "w") as fh:
            app.PDBFile.writeFile(top, st.getPositions(), fh, keepIds=True)
        if not ok:
            issues.add("md", Status.FAIL, "equilibration checks failed: see EquilibrationChecks")
    return {"tables": tables, "issues": issues, "status": status, "workdir": str(out),
            "minutes": round((time.time() - t_start) / 60, 1)}


def equilibration_checks(df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """Readiness for production, from the per-segment log of the equilibration."""
    eqc = cfg.get("md.equilibration", {}) or {}
    T = float(cfg.get("md.openmm.temperature_k", 310))
    rows = []
    add = lambda c, ok, d, warn=False: rows.append(  # noqa: E731
        {"Check": c, "Status": "PASS" if ok else ("WARNING" if warn else "FAIL"), "Detail": d})
    stages = list(dict.fromkeys(df["Stage"]))
    last = df[df["Stage"] == stages[-1]] if stages else df
    add("all_finite", df["Potential energy (kJ/mol)"].notna().all() and np.isfinite(df["Potential energy (kJ/mol)"]).all(),
        f"{len(df)} segments")
    tm = float(last["Temperature (K)"].mean()) if len(last) else float("nan")
    add("temperature", abs(tm - T) <= float(eqc.get("max_temperature_dev_k", 3.0)),
        f"mean {tm:.1f} K over the last stage (target {T:.0f} K)")
    if len(last) >= 2:
        a = (last["Box x (Å)"] * last["Box y (Å)"]).to_numpy()
        drift = abs(a[-1] - a[0]) / a[0]
        add("area_stable", drift <= float(eqc.get("max_area_drift", 0.015)),
            f"box xy area changed {100 * drift:.2f}% over the last stage", warn=True)
    apl = last["Area per lipid (Å²)"].dropna()
    if len(apl):
        add("area_per_lipid", 40 <= float(apl.iloc[-1]) <= 70, f"{float(apl.iloc[-1]):.1f} Å² (PC + cholesterol; "
            "7:3 POPC:CHL bilayers are typically ~45–55 Å²)", warn=True)
    zd = df["Max Zn–ligand deviation (Å)"].dropna()
    if len(zd):
        add("zn_site", float(zd.max()) <= float(eqc.get("max_zn_dev", 0.3)),
            f"largest Zn–ligand deviation from the crystal {float(zd.max()):.2f} Å (limit {eqc.get('max_zn_dev', 0.3)})")
    rm = df["Cα RMSD from start (Å)"].dropna()
    if len(rm):
        add("protein_rmsd", float(rm.iloc[-1]) <= 3.0, f"Cα RMSD at the end {float(rm.iloc[-1]):.2f} Å", warn=True)
    return pd.DataFrame(rows)
