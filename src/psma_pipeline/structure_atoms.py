"""Atom tables, biological-assembly construction and crystal-lattice images.

Everything downstream of this module works on a plain ``pandas`` atom table,
so geometry code is independent of gemmi and fully unit-testable.

Atom-table columns
    ``chain`` (author), ``label_asym``, ``entity``, ``resname``, ``resnum``
    (author number), ``icode``, ``label_seq``, ``unp`` (UniProt number via
    SIFTS; NaN outside mapped segments), ``atom``, ``element``, ``altloc``,
    ``x``, ``y``, ``z``, ``b``, ``occ``, ``polymer`` (bool), ``water`` (bool),
    ``target`` (bool: belongs to a target entity), and after assembly
    construction ``copy`` (operator name) and ``protomer``
    (``"<label_asym>|<copy>"``).

Assembly construction
    The chosen assembly's generators (sub-chains × operators) are read from
    gemmi and applied to the atom coordinates here. Unlike
    ``gemmi.make_assembly``, this keeps author chain, label_asym and UniProt
    numbering unchanged for every copy (copies differ only in ``copy``), so
    numbering cannot be lost when chains are renamed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .numbering import label_to_unp

COLUMNS = ["chain", "label_asym", "entity", "resname", "resnum", "icode", "label_seq", "unp", "atom",
           "element", "altloc", "x", "y", "z", "b", "occ", "polymer", "water", "target"]


@dataclass
class Operator:
    """A rigid transformation x' = R·x + t (Cartesian, Å)."""

    name: str
    R: np.ndarray
    t: np.ndarray

    def apply(self, xyz: np.ndarray) -> np.ndarray:
        return xyz @ self.R.T + self.t

    @property
    def is_identity(self) -> bool:
        return bool(np.allclose(self.R, np.eye(3)) and np.allclose(self.t, 0))


# ---------------------------------------------------------------------------
# gemmi extraction
# ---------------------------------------------------------------------------
def atoms_from_gemmi(st: Any, model_index: int, segs_by_asym: dict[str, list[dict]],
                     target_entities: set[str], altloc: str = "first") -> pd.DataFrame:
    """Heavy-atom table of one model.

    Hydrogens are dropped. With ``altloc="first"`` only the first alternate
    location of each atom is kept (blank or the first letter met).
    """
    import gemmi

    model = st[model_index]
    ent_of_sub: dict[str, str] = {}
    for ent in st.entities:
        for sub in ent.subchains:
            ent_of_sub[sub] = ent.name
    l2u = {sub: label_to_unp(segs) for sub, segs in segs_by_asym.items()}
    rows = []
    for ch in model:
        for res in ch:
            sub = res.subchain
            seen: set[str] = set()
            polymer = res.entity_type == gemmi.EntityType.Polymer
            water = res.is_water()
            for a in res:
                if a.element.name in ("H", "D"):
                    continue
                if altloc == "first":
                    if a.name in seen:
                        continue
                    seen.add(a.name)
                ls = res.label_seq if res.label_seq is not None else None
                rows.append((ch.name, sub, ent_of_sub.get(sub), res.name, res.seqid.num,
                             res.seqid.icode.strip(), ls,
                             l2u.get(sub, {}).get(ls) if ls is not None else None,
                             a.name, a.element.name.upper(),
                             "" if a.altloc == "\0" else a.altloc,
                             a.pos.x, a.pos.y, a.pos.z, a.b_iso, a.occ, polymer, water,
                             ent_of_sub.get(sub) in target_entities))
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["unp"] = pd.to_numeric(df["unp"], errors="coerce")
    return df


def _mat_np(m: Any) -> np.ndarray:
    """gemmi Mat33 -> 3x3 array (``tolist`` if bound, else row by row)."""
    try:
        return np.array(m.tolist(), dtype=float)
    except AttributeError:
        rows = [m.row_copy(i) for i in range(3)]
        return np.array([[r.x, r.y, r.z] for r in rows], dtype=float)


def _vec_np(v: Any) -> np.ndarray:
    try:
        return np.array(v.tolist(), dtype=float)
    except AttributeError:
        return np.array([v.x, v.y, v.z], dtype=float)


def _transform_to_np(tr: Any) -> tuple[np.ndarray, np.ndarray]:
    return _mat_np(tr.mat), _vec_np(tr.vec)


def orthogonalization_matrix(a: float, b: float, c: float, alpha: float, beta: float,
                             gamma: float) -> np.ndarray:
    """Standard PDB orthogonalisation (a along x, b in the xy plane); angles in degrees."""
    al, be, ga = np.radians([alpha, beta, gamma])
    v = np.sqrt(1 - np.cos(al) ** 2 - np.cos(be) ** 2 - np.cos(ga) ** 2
                + 2 * np.cos(al) * np.cos(be) * np.cos(ga))
    return np.array([[a, b * np.cos(ga), c * np.cos(be)],
                     [0, b * np.sin(ga), c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)],
                     [0, 0, c * v / np.sin(ga)]])


def _cell_matrices(cell: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        orth = _mat_np(cell.orth.mat)
    except AttributeError:
        orth = orthogonalization_matrix(cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)
    return orth, np.linalg.inv(orth)


def assembly_generators(st: Any, assembly_id: str) -> list[tuple[list[str], list[Operator]]] | None:
    """``[(label_asym list, [Operator, ...]), ...]`` for assembly ``assembly_id``; None if absent."""
    asm = next((a for a in st.assemblies if str(a.name).strip() == str(assembly_id).strip()), None)
    if asm is None:
        return None
    out = []
    for gen in asm.generators:
        ops = []
        for i, op in enumerate(gen.operators):
            R, t = _transform_to_np(op.transform)
            ops.append(Operator(str(op.name or i + 1), R, t))
        out.append((list(gen.subchains), ops))
    return out


def lattice_operators(st: Any, max_shift: int = 1) -> list[Operator]:
    """Cartesian operators for every space-group operation × lattice shift.

    The identity (operation 1, shift 0,0,0) is first. Names follow the
    ``<op>_<555-style shift>`` convention (e.g. ``2_565``).
    """
    sg = st.find_spacegroup()
    if sg is None or not st.cell.is_crystal():
        return []
    orth, frac = _cell_matrices(st.cell)
    ops = []
    for k, op in enumerate(sg.operations(), start=1):
        seitz = np.array(op.float_seitz(), dtype=float)
        Rf, tf = seitz[:3, :3], seitz[:3, 3]
        rng = range(-max_shift, max_shift + 1)
        for dx in rng:
            for dy in rng:
                for dz in rng:
                    shift = np.array([dx, dy, dz], dtype=float)
                    R = orth @ Rf @ frac
                    t = orth @ (tf + shift)
                    ops.append(Operator(f"{k}_{5 + dx}{5 + dy}{5 + dz}", R, t))
    ops.sort(key=lambda o: (not o.is_identity, o.name))
    return ops


# ---------------------------------------------------------------------------
# numpy-only construction
# ---------------------------------------------------------------------------
def build_assembly(atoms: pd.DataFrame,
                   generators: Iterable[tuple[list[str], list[Operator]]]) -> pd.DataFrame:
    """Apply assembly generators to an atom table.

    Every (sub-chain, operator) pair becomes one copy; identifiers are kept
    and ``copy`` / ``protomer`` columns are added.
    """
    parts = []
    for subs, ops in generators:
        sel = atoms[atoms["label_asym"].isin(subs)]
        if sel.empty:
            continue
        xyz = sel[["x", "y", "z"]].to_numpy()
        for op in ops:
            p = sel.copy()
            p[["x", "y", "z"]] = op.apply(xyz)
            p["copy"] = op.name
            parts.append(p)
    if not parts:
        return atoms.iloc[0:0].assign(copy=pd.Series(dtype=str), protomer=pd.Series(dtype=str))
    out = pd.concat(parts, ignore_index=True)
    out["protomer"] = out["label_asym"] + "|" + out["copy"].astype(str)
    return out


def asu_as_assembly(atoms: pd.DataFrame) -> pd.DataFrame:
    """The deposited model treated as one 'assembly' (copy = "1")."""
    out = atoms.copy()
    out["copy"] = "1"
    out["protomer"] = out["label_asym"] + "|1"
    return out


def target_protomers(asm_atoms: pd.DataFrame) -> list[str]:
    """Protomer IDs of target chains, sorted (author chain, then copy)."""
    t = asm_atoms[asm_atoms["target"] & asm_atoms["polymer"]]
    order = t.groupby("protomer", sort=False).agg(chain=("chain", "first"), copy=("copy", "first"))
    return order.sort_values(["chain", "copy"]).index.tolist()
