"""Solvent-accessible surface area and buried surface area.

Method
    * ``freesasa`` – FreeSASA (C library) when installed; same radii and probe.
    * ``shrake_rupley`` – documented numpy fallback: each atom sphere of radius
      r_i + probe carries ``n_points`` quasi-uniform points (golden-section
      spiral); a point is buried if it lies inside any neighbour's expanded
      sphere; atom SASA = 4π(r_i+probe)² × exposed fraction.

    Every result records the method actually used.

Buried surface area
    BSA = SASA(A) + SASA(B) − SASA(A∪B). Only atoms within
    2·(r_max + probe) of the partner can change exposure, so the exact BSA is
    obtained by recomputing just those atoms in isolation and in the complex.

Radii (Å, element-based, Bondi-style): C 1.70, N 1.55, O 1.52, S 1.80,
P 1.80, SE 1.90; anything else 1.80. Protein heavy atoms only are passed in
by the interface code (waters, ions, glycans are excluded from BSA).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger("psma_pipeline.sasa")

RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "SE": 1.90}
DEFAULT_RADIUS = 1.80


def radii_for(elements) -> np.ndarray:
    """Radius per element symbol (upper case)."""
    return np.array([RADII.get(str(e).upper(), DEFAULT_RADIUS) for e in elements], dtype=float)


def sphere_points(n: int) -> np.ndarray:
    """``n`` quasi-uniform unit vectors (golden-section spiral)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])


def freesasa_available() -> bool:
    try:
        import freesasa  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_method(method: str) -> str:
    """``auto`` -> ``freesasa`` if installed, else ``shrake_rupley``."""
    if method == "auto":
        return "freesasa" if freesasa_available() else "shrake_rupley"
    if method == "freesasa" and not freesasa_available():
        raise ImportError("sasa.method = freesasa but FreeSASA is not installed")
    return method


def sasa_shrake_rupley(coords: np.ndarray, radii: np.ndarray, probe: float = 1.4, n_points: int = 100,
                       subset: np.ndarray | None = None) -> np.ndarray:
    """Per-atom SASA (Å²) for ``subset`` (indices; default all atoms) in the context of all atoms."""
    coords = np.asarray(coords, float)
    radii = np.asarray(radii, float)
    idx = np.arange(len(coords)) if subset is None else np.asarray(subset, int)
    if len(coords) == 0 or len(idx) == 0:
        return np.zeros(len(idx))
    unit = sphere_points(n_points)
    ext = radii + probe
    tree = cKDTree(coords)
    rmax = ext.max()
    out = np.empty(len(idx))
    neigh = tree.query_ball_point(coords[idx], ext[idx] + rmax)
    for k, (i, nb) in enumerate(zip(idx, neigh)):
        nb = np.array([j for j in nb if j != i], dtype=int)
        pts = coords[i] + ext[i] * unit
        if len(nb):
            d2 = ((pts[:, None, :] - coords[nb][None, :, :]) ** 2).sum(-1)
            exposed = ~(d2 < ext[nb][None, :] ** 2).any(1)
            frac = exposed.mean()
        else:
            frac = 1.0
        out[k] = 4 * np.pi * ext[i] ** 2 * frac
    return out


def sasa_freesasa(coords: np.ndarray, radii: np.ndarray, probe: float = 1.4,
                  n_points: int = 100) -> np.ndarray:
    """Per-atom SASA with FreeSASA (Shrake–Rupley algorithm, same parameters)."""
    import freesasa
    params = freesasa.Parameters({"algorithm": freesasa.ShrakeRupley, "probe-radius": probe,
                                  "n-points": n_points})
    res = freesasa.calcCoord(np.asarray(coords, float).ravel().tolist(), list(map(float, radii)), params)
    return np.array([res.atomArea(i) for i in range(len(coords))])


def atom_sasa(coords: np.ndarray, radii: np.ndarray, method: str, probe: float, n_points: int,
              subset: np.ndarray | None = None) -> np.ndarray:
    """Per-atom SASA using the resolved ``method`` (subset only for Shrake–Rupley)."""
    if method == "freesasa":
        full = sasa_freesasa(coords, radii, probe, n_points)
        return full if subset is None else full[np.asarray(subset, int)]
    return sasa_shrake_rupley(coords, radii, probe, n_points, subset)


def buried_surface(coords_a: np.ndarray, elem_a, coords_b: np.ndarray, elem_b, method: str = "auto",
                   probe: float = 1.4, n_points: int = 100, total: bool = True) -> dict:
    """Buried surface area between two atom sets.

    Returns
    -------
    dict with ``bsa`` (total, Å²), ``bsa_a``, ``bsa_b`` (per side),
    ``atom_bsa_a`` / ``atom_bsa_b`` (per-atom arrays), ``sasa_a`` / ``sasa_b``
    (total SASA of each isolated side, if ``total``), ``method``.

    Raises
    ------
    ValueError
        If a negative BSA results (larger than numerical noise): this is a
        failed calculation and must not be reported as a value.
    """
    method = resolve_method(method)
    A, B = np.asarray(coords_a, float), np.asarray(coords_b, float)
    ra, rb = radii_for(elem_a), radii_for(elem_b)
    if len(A) == 0 or len(B) == 0:
        raise ValueError("buried_surface needs atoms on both sides")
    reach = 2 * (max(ra.max(), rb.max()) + probe)
    cand_a = np.array([i for i, hits in enumerate(cKDTree(B).query_ball_point(A, reach)) if hits], int)
    cand_b = np.array([i for i, hits in enumerate(cKDTree(A).query_ball_point(B, reach)) if hits], int)
    AB = np.vstack([A, B])
    rab = np.concatenate([ra, rb])
    atom_a = np.zeros(len(A))
    atom_b = np.zeros(len(B))
    if method == "freesasa":
        sa, sb, sab = (atom_sasa(A, ra, method, probe, n_points), atom_sasa(B, rb, method, probe, n_points),
                       atom_sasa(AB, rab, method, probe, n_points))
        atom_a, atom_b = sa - sab[:len(A)], sb - sab[len(A):]
        tot_a, tot_b = sa.sum(), sb.sum()
    else:
        if len(cand_a):
            atom_a[cand_a] = (atom_sasa(A, ra, method, probe, n_points, cand_a)
                              - atom_sasa(AB, rab, method, probe, n_points, cand_a))
        if len(cand_b):
            atom_b[cand_b] = (atom_sasa(B, rb, method, probe, n_points, cand_b)
                              - atom_sasa(AB, rab, method, probe, n_points, cand_b + len(A)))
        tot_a = atom_sasa(A, ra, method, probe, n_points).sum() if total else np.nan
        tot_b = atom_sasa(B, rb, method, probe, n_points).sum() if total else np.nan
    bsa_a, bsa_b = float(atom_a.sum()), float(atom_b.sum())
    tol = 1e-6 * max(1.0, abs(bsa_a) + abs(bsa_b))
    if bsa_a < -tol or bsa_b < -tol:
        raise ValueError(f"negative buried surface area ({bsa_a:.2f}, {bsa_b:.2f} Å²)")
    return {"bsa": bsa_a + bsa_b, "bsa_a": bsa_a, "bsa_b": bsa_b, "atom_bsa_a": atom_a,
            "atom_bsa_b": atom_b, "sasa_a": float(tot_a), "sasa_b": float(tot_b), "method": method}
