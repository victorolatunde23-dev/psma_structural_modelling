"""Small gemmi helpers shared by the coordinate analyses.

All functions import gemmi lazily so modules that use them can be imported
without gemmi installed (they then fail only when called).
"""

from __future__ import annotations

from typing import Any

from .numbering import label_to_unp


def one_letter(resname: str) -> str:
    """One-letter code of a residue name (``"X"`` if unknown)."""
    import gemmi
    try:
        return (gemmi.find_tabulated_residue(resname).one_letter_code or "X").upper()
    except Exception:  # noqa: BLE001 - unknown residue names map to X
        return "X"


def mark_contact(st: Any, query_pos: Any, mark: Any, cra: Any) -> tuple[float, bool]:
    """Distance from ``query_pos`` to a NeighborSearch hit, and whether the hit is an image.

    ``mark.pos`` is the symmetry-transformed position *wrapped into the unit
    cell*, so a plain ``query_pos.dist(mark.pos)`` can be wrong near cell
    edges. For crystals the nearest periodic image is used
    (``UnitCell.find_nearest_pbc_image(point, mark.pos, 0)``, as in the gemmi
    documentation).

    Returns
    -------
    (distance in Å, is_image)
        ``is_image`` is True when the nearest copy is not the deposited atom
        itself (another symmetry operation or a lattice translation).
    """
    direct = query_pos.dist(cra.atom.pos)
    if not st.cell.is_crystal():
        return direct, False
    d = st.cell.find_nearest_pbc_image(query_pos, mark.pos, 0).dist()
    return d, abs(direct - d) > 1e-3


def res_key(chain: Any, res: Any) -> tuple:
    """Identity of a residue within one model (author chain, number, icode, name, label asym)."""
    return (chain.name, res.seqid.num, res.seqid.icode, res.name, res.subchain)


def author_id(res: Any) -> str:
    """Author residue number with insertion code, e.g. ``"377"`` or ``"100A"``."""
    return f"{res.seqid.num}{res.seqid.icode.strip()}"


def neighbor_search(st: Any, model: Any, max_radius: float = 6.0):
    """Populated ``gemmi.NeighborSearch`` including crystal symmetry when present."""
    import gemmi
    cell = st.cell if st.cell.is_crystal() else gemmi.UnitCell()
    ns = gemmi.NeighborSearch(model, cell, max_radius)
    try:
        return ns.populate(include_h=False)
    except TypeError:          # older gemmi
        return ns.populate()


def unp_maps(ctx: dict) -> dict[str, dict[int, int]]:
    """label_asym -> {label_seq -> UniProt} for every target sub-chain of an entry."""
    coords = ctx["coords"]
    return {sub: label_to_unp(ctx["segs"].get(sub, [])) for sub in coords["target_subs"]}


def residue_label(chain_name: str, res: Any, l2u_by_sub: dict, target_subs: set,
                  sym: bool = False) -> str:
    """Label used in contact lists.

    Target residues: ``<model aa><UniProt number>/<author chain>`` (e.g. ``H377/A``).
    Other residues: ``<resname><author number>/<author chain>``. A trailing ``*``
    marks a crystal-symmetry image.
    """
    star = "*" if sym else ""
    if res.subchain in target_subs and res.label_seq in l2u_by_sub.get(res.subchain, {}):
        return f"{one_letter(res.name)}{l2u_by_sub[res.subchain][res.label_seq]}/{chain_name}{star}"
    return f"{res.name}{res.seqid.num}{res.seqid.icode.strip()}/{chain_name}{star}"


def uniprot_mismatch(res: Any, l2u_by_sub: dict, target_subs: set, useq: str) -> str | None:
    """``"A377(UniProt H)"`` if a mapped target residue differs from UniProt, else None."""
    if res.subchain not in target_subs:
        return None
    n = l2u_by_sub.get(res.subchain, {}).get(res.label_seq)
    if n is None or n > len(useq):
        return None
    aa = one_letter(res.name)
    return None if aa == useq[n - 1] else f"{aa}{n}(UniProt {useq[n - 1]})"
