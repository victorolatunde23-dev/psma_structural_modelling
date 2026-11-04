"""Run MCPB.py's Gaussian QM jobs with PySCF and write Gaussian-format results (Milestone 6).

MCPB.py (AmberTools) builds the metal-site models and writes Gaussian inputs, then reads
Gaussian results. Gaussian is not available in Colab, so this module:

1. **parses** MCPB.py's own ``.com`` files (route, charge, multiplicity, coordinates,
   frozen-atom flags, ``ReadRadii`` radii) — so MCPB.py still decides the models and method;
2. **runs** the equivalent PySCF calculation (GPU through gpu4pyscf when available):

   * ``<g>_small_opt.com`` → geometry optimisation (geomeTRIC);
   * ``<g>_small_fc.com``  → analytic Hessian at the optimised geometry;
   * ``<g>_large_mk.com``  → SCF + electrostatic potential on Merz–Kollman points;

3. **writes** the files MCPB.py reads for ``software_version g09``:

   * ``<g>_small_opt.fchk`` — ``Current cartesian coordinates`` (Bohr) and the
     lower-triangle ``Cartesian Force Constants`` (Hartree/Bohr²), 5E16.8 per line;
   * ``<g>_large_mk.log`` — ``Electrostatic Properties Using The SCF Density`` with
     ``Atomic Center`` / ``ESP Fit Center`` coordinates (Å, columns 33–62) and
     ``Electrostatic Properties (Atomic Units)`` with ``Atom`` / ``Fit`` potentials (a.u.).

Equivalence with Gaussian: B3LYP is Gaussian's variant (``b3lypg``, VWN3); ``Grid=UltraFine`` is a
(99, 590) grid; MK points are four shells at 1.4/1.6/1.8/2.0 × the MK radii with ``IOp(6/42=…)``
points per Å² (Fibonacci spheres instead of Gaussian's own point set; RESP is insensitive to this).

Every job is resumable: results are cached next to the input (``.pyscf.npz``) and reused.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("psma_pipeline.qm_bridge")

BOHR = 0.529177249          # Å per Bohr (the value pymsmt uses)
MK_RADII = {"H": 1.20, "C": 1.50, "N": 1.50, "O": 1.40, "S": 1.75, "P": 1.80, "F": 1.35, "Cl": 1.70}
MK_SHELLS = (1.4, 1.6, 1.8, 2.0)
Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17, "Ca": 20, "Mn": 25, "Fe": 26,
     "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30}
FUNCTIONALS = {"B3LYP": "b3lypg", "PBE0": "pbe0", "M06": "m06", "B97D": "b97-d", "WB97XD": "wb97x-d"}


@dataclass
class GaussianJob:
    path: Path
    route: str
    method: str
    basis: str
    charge: int
    mult: int
    elements: list[str]
    coords: np.ndarray                  # Å
    frozen: list[bool]
    radii: dict[str, float] = field(default_factory=dict)
    density: float = 6.0                # MK points per Å² (IOp(6/42=n))

    @property
    def kind(self) -> str:
        r = self.route.upper()
        if "POP(MK" in r.replace(" ", ""):
            return "mk"
        if "FREQ" in r:
            return "freq"
        if re.search(r"\bOPT\b", r):
            return "opt"
        return "sp"


def _norm_el(sym: str) -> str:
    sym = re.sub(r"[^A-Za-z]", "", sym)
    return sym[0].upper() + sym[1:].lower()


def parse_gaussian_input(path: str | Path) -> GaussianJob:
    """Parse a Gaussian ``.com`` written by MCPB.py (Link 0, route, title, charge/mult, atoms, extras)."""
    path = Path(path)
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines) and lines[i].startswith("%"):
        i += 1
    route = []
    while i < len(lines) and lines[i].strip():
        route.append(lines[i].strip())
        i += 1
    route_s = " ".join(route)
    i += 1
    while i < len(lines) and lines[i].strip():   # title
        i += 1
    i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):                           # Geom=AllCheckpoint: no molecule section
        q, m = 0, 1
        els, xyz, frozen = [], np.zeros((0, 3)), []
    else:
        q, m = (int(x) for x in lines[i].split()[:2])
        i += 1
        els, xyz, frozen = [], [], []
        while i < len(lines) and lines[i].strip():
            t = lines[i].split()
            if len(t) == 5:                       # "El  0|-1  x y z" (optimisation flag)
                els.append(_norm_el(t[0]))
                frozen.append(t[1] == "-1")
                xyz.append([float(v) for v in t[2:5]])
            elif len(t) == 4:
                els.append(_norm_el(t[0]))
                frozen.append(False)
                xyz.append([float(v) for v in t[1:4]])
            i += 1
        xyz = np.array(xyz, float).reshape(-1, 3)
    radii = {}
    for line in lines[i:]:                       # ReadRadii section: "Zn 1.395"
        t = line.split()
        if len(t) == 2 and re.fullmatch(r"[A-Za-z]{1,2}", t[0]):
            try:
                radii[_norm_el(t[0])] = float(t[1])
            except ValueError:
                pass
    mm = re.search(r"#\s*([A-Za-z0-9\-]+)/([^\s]+)", route_s)
    method, basis = (mm.group(1), mm.group(2)) if mm else ("B3LYP", "6-31G*")
    dens = re.search(r"6/42=(\d+)", route_s)
    return GaussianJob(path, route_s, method, basis, q, m, els, xyz, frozen, radii,
                       float(dens.group(1)) if dens else 6.0)


# ---------------------------------------------------------------------------
# Merz–Kollman points
# ---------------------------------------------------------------------------
def fibonacci_sphere(n: int) -> np.ndarray:
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    th = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


def mk_points(elements: list[str], coords: np.ndarray, radii: dict[str, float] | None = None,
              density: float = 6.0, shells=MK_SHELLS) -> np.ndarray:
    """MK fitting points (Å): ``density`` points/Å² on each shell, outside every other atom's shell."""
    rad = {**MK_RADII, **(radii or {})}
    missing = sorted({e for e in elements if e not in rad})
    if missing:
        raise ValueError(f"no MK radius for {missing}: add them to the ReadRadii section")
    R = np.array([rad[e] for e in elements])
    pts = []
    for s in shells:
        for a, (c, r) in enumerate(zip(coords, R)):
            n = max(1, int(round(4 * np.pi * (s * r) ** 2 * density)))
            p = c + s * r * fibonacci_sphere(n)
            d = np.linalg.norm(p[:, None, :] - coords[None, :, :], axis=2)
            inside = d < (s * R)[None, :] - 1e-6
            inside[:, a] = False
            pts.append(p[~inside.any(axis=1)])
    return np.vstack(pts)


# ---------------------------------------------------------------------------
# Writers (formats read by pymsmt.mol.gauio)
# ---------------------------------------------------------------------------
def _fchk_array(name: str, values: np.ndarray) -> list[str]:
    v = np.asarray(values, float).ravel()
    out = [f"{name:<40s}   R   N={len(v):12d}"]
    for k in range(0, len(v), 5):
        out.append("".join(f"{x:16.8E}" for x in v[k:k + 5]))
    return out


def write_fchk(path: str | Path, title: str, elements: list[str], coords_ang: np.ndarray, hessian: np.ndarray,
               charge: int, mult: int, energy: float, method: str, basis: str) -> Path:
    """Gaussian-style formatted checkpoint with the fields MCPB.py reads (and a few for other readers)."""
    n = len(elements)
    if hessian.shape != (3 * n, 3 * n):
        raise ValueError(f"Hessian shape {hessian.shape} != ({3 * n}, {3 * n})")
    tri = np.concatenate([hessian[j, : j + 1] for j in range(3 * n)])
    lines = [title[:72], f"Freq      R{method.upper():>30s}{basis.upper():>30s}",
             f"{'Number of atoms':<40s}   I     {n:12d}",
             f"{'Charge':<40s}   I     {charge:12d}",
             f"{'Multiplicity':<40s}   I     {mult:12d}",
             f"{'Atomic numbers':<40s}   I   N={n:12d}"]
    zs = [Z[e] for e in elements]
    for k in range(0, n, 6):
        lines.append("".join(f"{z:12d}" for z in zs[k:k + 6]))
    lines += _fchk_array("Current cartesian coordinates", np.asarray(coords_ang) / BOHR)
    lines.append(f"{'Total Energy':<40s}   R     {energy:22.15E}")
    lines += _fchk_array("Cartesian Force Constants", tri)
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


def write_mk_log(path: str | Path, elements: list[str], coords_ang: np.ndarray, points_ang: np.ndarray,
                 esp_atoms: np.ndarray, esp_points: np.ndarray, energy: float, note: str = "") -> Path:
    """Gaussian-style output with the two ESP sections MCPB.py parses (columns 33–62 hold x, y, z in Å)."""
    def centre(label: str, k: int, xyz) -> str:
        head = f"{label}{k:5d} is at "
        return f"{head:<32s}{xyz[0]:10.5f}{xyz[1]:10.5f}{xyz[2]:10.5f}"

    out = [" Entering Gaussian System (written by psma_pipeline.qm_bridge from a PySCF calculation)",
           f" {note}", f" SCF Done:  E(RB3LYP) =  {energy:.9f}     A.U.", "",
           " Electrostatic Properties Using The SCF Density:"]
    for k, xyz in enumerate(coords_ang, 1):
        out.append(centre("      Atomic Center", k, xyz))
    for k, xyz in enumerate(points_ang, 1):
        out.append(centre("     ESP Fit Center", k, xyz))
    out += ["", "              Electrostatic Properties (Atomic Units)", "",
            " -----------------------------------------------------------------",
            "    Center     Electric         -------- Electric Field --------",
            "               Potential          X             Y             Z",
            " -----------------------------------------------------------------"]
    for k, v in enumerate(esp_atoms, 1):
        out.append(f" {k:5d} Atom  {v:14.6f}")
    for k, v in enumerate(esp_points, len(esp_atoms) + 1):
        out.append(f" {k:5d} Fit   {v:14.6f}")
    out += [" -----------------------------------------------------------------", " Normal termination of Gaussian (PySCF bridge)"]
    Path(path).write_text("\n".join(out) + "\n")
    return Path(path)


# ---------------------------------------------------------------------------
# PySCF calculations
# ---------------------------------------------------------------------------
def _pyscf_mol(job: GaussianJob, coords: np.ndarray | None = None):
    from pyscf import gto
    xyz = job.coords if coords is None else coords
    basis = job.basis.lower().replace("(d)", "*").replace("(d,p)", "**")
    return gto.M(atom=[(e, tuple(c)) for e, c in zip(job.elements, xyz)], unit="Angstrom", basis=basis,
                 charge=job.charge, spin=job.mult - 1, verbose=3)


def _scf(mol, job: GaussianJob, use_gpu: bool, grid=(99, 590)):
    """Converged RKS/UKS; on any GPU failure (unsupported device, missing library) falls back to CPU."""
    from pyscf import dft
    xc = FUNCTIONALS.get(job.method.upper(), job.method.lower())

    def build():
        mf = dft.RKS(mol) if job.mult == 1 else dft.UKS(mol)
        mf.xc = xc
        mf.grids.atom_grid = grid
        mf.conv_tol = 1e-10
        mf.max_cycle = 200
        return mf

    if use_gpu:
        try:
            mf = build().to_gpu()
            mf.kernel()
            if mf.converged:
                return mf
            log.warning("GPU SCF did not converge; retrying on CPU")
        except Exception as ex:  # noqa: BLE001
            log.warning("GPU not used (%s); running on CPU", ex)
    mf = build()
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    return mf


def _hessian(mf):
    try:
        h = mf.Hessian().kernel()
    except Exception as ex:  # noqa: BLE001 - GPU Hessian unsupported: redo on CPU
        if not hasattr(mf, "to_cpu"):
            raise
        log.warning("GPU Hessian failed (%s); computing on CPU", ex)
        mf = mf.to_cpu()
        mf.kernel()
        h = mf.Hessian().kernel()
    return np.asarray(h.get() if hasattr(h, "get") else h)


def gpu_available() -> bool:
    try:
        import cupy  # noqa: F401
        import gpu4pyscf  # noqa: F401
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


def run_opt(job: GaussianJob, workdir: Path, use_gpu: bool, maxsteps: int = 200) -> tuple[np.ndarray, float, dict]:
    """Geometry optimisation with geomeTRIC; frozen atoms (flag -1) are fixed. Returns Å coords, energy, info."""
    from pyscf.geomopt.geometric_solver import kernel as geom_kernel
    cache = workdir / f"{job.path.stem}.pyscf.npz"
    if cache.exists():
        d = np.load(cache)
        if bool(d["converged"]):
            log.info("optimisation reused from %s", cache.name)
            return d["coords"], float(d["energy"]), {"reused": True, "converged": True}
    mol = _pyscf_mol(job)
    mf = _scf(mol, job, use_gpu)
    params = {}
    if any(job.frozen):
        cfile = workdir / f"{job.path.stem}.constraints"
        cfile.write_text("$freeze\nxyz " + ",".join(str(i + 1) for i, f in enumerate(job.frozen) if f) + "\n")
        params["constraints"] = str(cfile)
    t0 = time.time()
    conv, mol_eq = geom_kernel(mf, maxsteps=maxsteps, **params)
    xyz = mol_eq.atom_coords(unit="Angstrom")
    mf2 = _scf(mol_eq, job, use_gpu)
    np.savez(cache, coords=xyz, energy=mf2.e_tot, converged=conv)
    return xyz, float(mf2.e_tot), {"converged": bool(conv), "minutes": round((time.time() - t0) / 60, 1)}


def run_hessian(job: GaussianJob, coords: np.ndarray, workdir: Path, use_gpu: bool) -> tuple[np.ndarray, float]:
    """Analytic Hessian (Hartree/Bohr², 3N×3N) at ``coords`` (Å)."""
    cache = workdir / f"{job.path.stem}.hessian.npz"
    if cache.exists():
        d = np.load(cache)
        if d["coords"].shape == coords.shape and np.allclose(d["coords"], coords, atol=1e-6):
            log.info("Hessian reused from %s", cache.name)
            return d["hessian"], float(d["energy"])
    mol = _pyscf_mol(job, coords)
    mf = _scf(mol, job, use_gpu)
    h = _hessian(mf)
    n = mol.natm
    H = h.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    H = 0.5 * (H + H.T)
    np.savez(cache, hessian=H, energy=mf.e_tot, coords=coords)
    return H, float(mf.e_tot)


def esp_at(mf, points_ang: np.ndarray, chunk_bytes: float = 4e8) -> np.ndarray:
    """Electrostatic potential (a.u.) of the SCF density + nuclei at points (Å)."""
    if hasattr(mf, "to_cpu"):
        try:
            mf = mf.to_cpu()
        except Exception:  # noqa: BLE001
            pass
    mol = mf.mol
    dm = mf.make_rdm1()
    if dm.ndim == 3:
        dm = dm[0] + dm[1]
    pts = np.asarray(points_ang) / BOHR
    R = mol.atom_coords()
    zc = mol.atom_charges()
    nao = mol.nao
    step = max(1, int(chunk_bytes / (8 * nao * nao)))
    v_el = np.empty(len(pts))
    for k in range(0, len(pts), step):
        g = pts[k:k + step]
        ints = mol.intor("int1e_grids", grids=g)
        v_el[k:k + step] = np.einsum("pij,ij->p", ints, dm)
    d = np.linalg.norm(pts[:, None, :] - R[None, :, :], axis=2)
    with np.errstate(divide="ignore"):
        v_nuc = np.where(d > 1e-8, zc[None, :] / np.where(d > 1e-8, d, 1.0), 0.0).sum(axis=1)
    return v_nuc - v_el


def run_mk(job: GaussianJob, workdir: Path, use_gpu: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """SCF on the large model, then ESP on atoms and MK points. Returns (points, esp_atoms, esp_points, energy)."""
    cache = workdir / f"{job.path.stem}.esp.npz"
    if cache.exists():
        d = np.load(cache)
        log.info("ESP reused from %s", cache.name)
        return d["points"], d["esp_atoms"], d["esp_points"], float(d["energy"])
    mol = _pyscf_mol(job)
    mf = _scf(mol, job, use_gpu)
    pts = mk_points(job.elements, job.coords, job.radii, job.density)
    esp_p = esp_at(mf, pts)
    esp_a = esp_at(mf, job.coords)            # potential at atoms, own nucleus excluded
    np.savez(cache, points=pts, esp_atoms=esp_a, esp_points=esp_p, energy=mf.e_tot)
    return pts, esp_a, esp_p, float(mf.e_tot)


# ---------------------------------------------------------------------------
# MCPB.py workflow glue
# ---------------------------------------------------------------------------
def run_mcpb_qm(workdir: str | Path, group: str, use_gpu: bool | None = None) -> dict:
    """Run the three MCPB.py QM jobs of ``group`` in ``workdir`` and write the files MCPB.py reads.

    Returns a summary dict (also written to ``<group>_pyscf_summary.json``).
    """
    workdir = Path(workdir)
    use_gpu = gpu_available() if use_gpu is None else use_gpu
    summary: dict = {"group": group, "gpu": use_gpu}
    try:
        import pyscf
        summary["pyscf"] = pyscf.__version__
    except ImportError as ex:
        raise ImportError("PySCF is required: pip install pyscf geometric") from ex

    opt = parse_gaussian_input(workdir / f"{group}_small_opt.com")
    xyz, e_opt, info = run_opt(opt, workdir, use_gpu)
    rms_move = float(np.sqrt(np.mean(np.sum((xyz - opt.coords) ** 2, axis=1))))
    summary["small model"] = {"atoms": len(opt.elements), "charge": opt.charge, "multiplicity": opt.mult,
                              "method": f"{opt.method}/{opt.basis} (PySCF {FUNCTIONALS.get(opt.method.upper(), opt.method)})",
                              "optimisation converged": info.get("converged"), "energy (Eh)": e_opt,
                              "RMS displacement from input (Å)": round(rms_move, 3)}
    if not info.get("converged"):
        raise RuntimeError("small-model optimisation did not converge: Seminario force constants would be unreliable")
    H, e_h = run_hessian(opt, xyz, workdir, use_gpu)
    freqs = harmonic_frequencies(H, opt.elements)
    summary["small model"]["imaginary frequencies < -50 cm-1"] = int((freqs < -50).sum())
    summary["small model"]["lowest frequencies (cm-1)"] = [round(float(f), 1) for f in freqs[:8]]
    write_fchk(workdir / f"{group}_small_opt.fchk", f"{group} small model (PySCF bridge)", opt.elements, xyz, H,
               opt.charge, opt.mult, e_h, opt.method, opt.basis)
    (workdir / f"{group}_small_fc.log").write_text(
        f" Hessian computed with PySCF {summary['pyscf']} at the optimised geometry; E = {e_h:.9f}\n"
        " Normal termination of Gaussian (PySCF bridge)\n")
    mk = parse_gaussian_input(workdir / f"{group}_large_mk.com")
    pts, esp_a, esp_p, e_mk = run_mk(mk, workdir, use_gpu)
    write_mk_log(workdir / f"{group}_large_mk.log", mk.elements, mk.coords, pts, esp_a, esp_p, e_mk,
                 note=f"{mk.method}/{mk.basis} charge {mk.charge} mult {mk.mult}; {len(pts)} MK points")
    summary["large model"] = {"atoms": len(mk.elements), "charge": mk.charge, "multiplicity": mk.mult,
                              "MK points": int(len(pts)), "radii read": mk.radii, "energy (Eh)": e_mk}
    (workdir / f"{group}_pyscf_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06, "Zn": 65.38, "Ca": 40.078, "Cl": 35.45}


def _mass_weight(H: np.ndarray, elements: list[str]) -> np.ndarray:
    m = np.repeat([MASS.get(e, 12.0) for e in elements], 3)
    return H / np.sqrt(np.outer(m, m))


def harmonic_frequencies(H: np.ndarray, elements: list[str]) -> np.ndarray:
    """Harmonic frequencies (cm⁻¹, imaginary as negative) from a Cartesian Hessian in Hartree/Bohr²
    (translations and rotations are not projected out: they appear near zero)."""
    w = np.linalg.eigvalsh(_mass_weight(H, elements))
    return np.sign(w) * np.sqrt(np.abs(w)) * 5140.48


# ---------------------------------------------------------------------------
# Self-test (Colab): a fast end-to-end check before the long metal-site run
# ---------------------------------------------------------------------------
WATER_REF_CM1 = (1713, 3727, 3849)   # B3LYP/6-31G* harmonic frequencies of H2O (unscaled, literature)


def self_test(workdir: str | Path, use_gpu: bool | None = None) -> dict:
    """H2O at B3LYP/6-31G*: optimisation, Hessian, MK ESP through the bridge; frequencies compared with
    literature values and the written files read back with AmberTools' own pymsmt parsers when available."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    use_gpu = gpu_available() if use_gpu is None else use_gpu
    com = workdir / "WAT_small_opt.com"
    com.write_text("%Chk=WAT\n# B3LYP/6-31G* Opt\n\nwater\n\n0  1\nO   0.000  0.000  0.117\n"
                   "H   0.000  0.757 -0.469\nH   0.000 -0.757 -0.469\n\n")
    job = parse_gaussian_input(com)
    t0 = time.time()
    xyz, e, info = run_opt(job, workdir, use_gpu)
    H, _ = run_hessian(job, xyz, workdir, use_gpu)
    f = np.sort(harmonic_frequencies(H, job.elements))[-3:]
    write_fchk(workdir / "WAT_small_opt.fchk", "water", job.elements, xyz, H, 0, 1, e, "B3LYP", "6-31G*")
    mk = GaussianJob(com, job.route + " Pop(MK)", "B3LYP", "6-31G*", 0, 1, job.elements, xyz, [False] * 3, {}, 6.0)
    pts, ea, ep, e2 = run_mk(mk, workdir, use_gpu)
    write_mk_log(workdir / "WAT_large_mk.log", job.elements, xyz, pts, ea, ep, e2)
    res = {"gpu": use_gpu, "minutes": round((time.time() - t0) / 60, 2), "converged": info.get("converged"),
           "frequencies (cm-1)": [round(float(x), 1) for x in f],
           "max deviation from literature (cm-1)": round(float(np.max(np.abs(f - np.array(WATER_REF_CM1)))), 1),
           "MK points": int(len(pts))}
    try:
        from pymsmt.mol.gauio import get_crds_from_fchk, get_esp_from_gau, get_matrix_from_fchk
        c = np.array(get_crds_from_fchk(str(workdir / "WAT_small_opt.fchk"), 3)).reshape(3, 3) * BOHR
        M = get_matrix_from_fchk(str(workdir / "WAT_small_opt.fchk"), 9)
        get_esp_from_gau(str(workdir / "WAT_large_mk.log"), str(workdir / "WAT_large_mk.esp"))
        n_esp = int((workdir / "WAT_large_mk.esp").read_text().split()[1])
        res["pymsmt read-back"] = bool(np.allclose(c, xyz, atol=1e-5) and np.allclose(M, H, atol=1e-7)
                                       and n_esp == len(pts))
    except ImportError:
        res["pymsmt read-back"] = "NOT RUN (AmberTools not installed)"
    res["passed"] = bool(res["converged"] and res["max deviation from literature (cm-1)"] <= 25
                         and res["pymsmt read-back"] in (True, "NOT RUN (AmberTools not installed)"))
    return res
