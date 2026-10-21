"""N-glycosylation sites observed in deposited models.

A sugar residue whose C1 lies within ``contacts.glycan_link`` (default 2.0 Å)
of an Asn ND2 is treated as the root of an N-linked glycan. Milestone 5 adds
full glycan composition, connectivity and preparation strategies.

Scientific rule: an observed crystallographic glycan is usually a fragment
(often one or two NAG); it is never reported as a complete biological glycoform.
"""

from __future__ import annotations

from .gemmi_utils import mark_pos, neighbor_search, unp_maps
from .ligands import GLYCAN, classify_component
from .status import Status


def find_glycosylation_sites(pdb: str, ctx: dict, classes: dict, contacts_cfg: dict) -> list[dict]:
    """One row per (root sugar, Asn) link in the analysed model.

    Returns rows with the Asn in UniProt and author numbering, the root sugar,
    the C1–ND2 distance and the root sugar mean B-factor.
    """
    import gemmi

    coords = ctx["coords"]
    st, model = coords["st"], coords["model"]
    ns = neighbor_search(st, model, float(contacts_cfg.get("neighbour_search_radius", 6.0)))
    l2u = unp_maps(ctx)
    link = float(contacts_cfg.get("glycan_link", 2.0))
    rows: list[dict] = []
    for ch in model:
        for res in ch:
            if res.is_water() or res.entity_type == gemmi.EntityType.Polymer:
                continue
            comp = res.name.upper()
            if classify_component(comp, classes) != GLYCAN:
                continue
            c1 = next((a for a in res if a.name == "C1"), None)
            if c1 is None:
                continue
            for m in ns.find_atoms(c1.pos, "\0", radius=link):
                cra = m.to_cra(model)
                if cra.residue.name == "ASN" and cra.atom.name == "ND2":
                    unp = l2u.get(cra.residue.subchain, {}).get(cra.residue.label_seq)
                    rows.append({
                        "PDB ID": pdb, "Model": coords.get("model_number", 1),
                        "Chain": cra.chain.name, "Chain (label)": cra.residue.subchain,
                        "Entity": coords["ent_of_sub"].get(cra.residue.subchain),
                        "Asn (UniProt)": unp,
                        "Asn (author)": f"{cra.residue.seqid.num}",
                        "Root sugar": f"{comp}{res.seqid.num}",
                        "Root sugar chain (label)": res.subchain,
                        "C1-ND2 (Å)": round(c1.pos.dist(mark_pos(m)), 2),
                        "Root sugar mean B": round(sum(a.b_iso for a in res) / len(res), 1),
                        "Calculation status": (Status.OK if unp is not None else Status.WARNING).value,
                        "Warning": "" if unp is not None else "Asn has no UniProt mapping (tag or unmapped segment)",
                    })
    return rows
