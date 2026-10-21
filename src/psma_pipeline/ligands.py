"""Classification of non-polymer chemical components.

Classes (checked in this order, configured in ``config/ligand_classes.yaml``):
``ion of interest`` → ``glycan`` → ``additive / buffer`` → ``ligand of interest``.
"""

from __future__ import annotations

ION = "ion of interest"
GLYCAN = "glycan"
ADDITIVE = "additive / buffer"
LIGAND = "ligand of interest"


def classify_component(comp_id: str, classes: dict) -> str:
    """Return the class name of a PDB chemical component ID.

    Parameters
    ----------
    comp_id:
        Three-letter (or longer) PDB component ID; case-insensitive.
    classes:
        The ``ligand_classes`` configuration (``ions_of_interest``,
        ``glycans``, ``additives``).
    """
    c = comp_id.upper()
    if c in classes["ions_of_interest"]:
        return ION
    if c in classes["glycans"]:
        return GLYCAN
    if c in classes["additives"]:
        return ADDITIVE
    return LIGAND


def contact_cutoff(comp_id: str, classes: dict, contacts: dict) -> float:
    """Distance cut-off (Å) used for a component: metal, anion or ligand."""
    c = comp_id.upper()
    if c in classes["metals"]:
        return float(contacts["metal"])
    if c in classes.get("anions", ["CL"]):
        return float(contacts["anion"])
    return float(contacts["ligand"])


def is_metal(comp_id: str, classes: dict) -> bool:
    return comp_id.upper() in classes["metals"]


def is_metal_or_anion(comp_id: str, classes: dict) -> bool:
    """Components whose coordination shell is listed atom by atom."""
    c = comp_id.upper()
    return c in classes["metals"] or c in classes.get("anions", ["CL"])
